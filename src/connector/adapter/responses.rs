//! The Responses API (`/responses`) — this crate's primary protocol.
//!
//! Newer model families are reachable only here and reject Chat Completions
//! outright, so Responses is tried first and
//! [`chat_completions`](super::chat_completions) is the fallback for models
//! that went the other way.
//!
//! The request and response shapes differ from Chat Completions, so the subset
//! we need is modelled here and translated to and from the same
//! [`ChatRequest`] → text contract both protocols honor.

use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::protocol::{is_endpoint_absent, is_wrong_api, ProtocolError, TokenSink};
use super::sse::SseDecoder;
use super::transport::{read_body, Transport};
use crate::domain::{ChatRequest, OpenAiError};

/// Retries for the intermittently-gated Responses endpoint. Some providers
/// (GitHub Copilot) answer `403` on a fraction of requests regardless of
/// credentials — a rollout gate, not an auth failure — and the next attempt
/// succeeds.
const GATED_403_RETRIES: usize = 4;

/// Pause between those retries.
const GATED_403_BACKOFF: Duration = Duration::from_millis(400);

/// Event type carrying incremental assistant text on the SSE stream.
const TEXT_DELTA_EVENT: &str = "response.output_text.delta";

/// Output item type carrying user-facing text; `reasoning` and tool items do not.
const MESSAGE_ITEM: &str = "message";

/// Content part type holding assistant text.
const OUTPUT_TEXT_PART: &str = "output_text";

#[derive(Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    /// Structured input turns. The API also accepts a bare string, but the turn
    /// form carries the system prompt cleanly.
    input: Vec<InputItem<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    /// Structured-output constraint. The Responses API nests it under `text`,
    /// unlike Chat Completions' top-level `response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<TextConfig>,
}

#[derive(Serialize)]
struct InputItem<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct TextConfig {
    format: TextFormat,
}

#[derive(Serialize)]
struct TextFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
    strict: bool,
    schema: serde_json::Value,
}

/// Non-streaming body: `output` is a list of items; assistant text lives in
/// `message` items' `content[]` as `output_text` parts.
#[derive(Deserialize)]
struct ResponsesBody {
    #[serde(default)]
    output: Vec<OutputItem>,
}

#[derive(Deserialize)]
struct OutputItem {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    content: Vec<ContentPart>,
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

impl ResponsesBody {
    /// Concatenate the `output_text` parts of every message item, skipping
    /// reasoning and tool items.
    fn into_text(self) -> Option<String> {
        let text: String = self
            .output
            .into_iter()
            // An item with no declared type is assumed to be a message: the
            // alternative is silently discarding the only answer we got.
            .filter(|item| item.kind == MESSAGE_ITEM || item.kind.is_empty())
            .flat_map(|item| item.content)
            .filter(|part| part.kind == OUTPUT_TEXT_PART)
            .map(|part| part.text)
            .collect();
        (!text.trim().is_empty()).then_some(text)
    }
}

/// One SSE event. Only incremental text deltas matter; created, reasoning,
/// completed, and the rest are ignored.
#[derive(Deserialize)]
struct StreamEvent {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<String>,
}

/// Run `request` against the Responses API, streaming to `sink` when present.
pub(crate) async fn execute(
    transport: &Transport,
    model: &str,
    request: &ChatRequest,
    sink: TokenSink<'_>,
) -> Result<String, ProtocolError> {
    let body = ResponsesRequest {
        model,
        input: request
            .messages
            .iter()
            .map(|message| InputItem {
                role: message.role.as_str(),
                content: &message.content,
            })
            .collect(),
        stream: sink.is_some(),
        temperature: request.temperature,
        max_output_tokens: request.max_output_tokens,
        text: request.schema.as_ref().map(|schema| TextConfig {
            format: TextFormat {
                kind: "json_schema",
                name: schema.name.clone(),
                strict: schema.strict,
                schema: schema.schema.clone(),
            },
        }),
    };

    let path = &transport.routes().responses;
    let response = send_with_gate_retry(transport, path, &body).await?;

    let status = response.status();
    if !status.is_success() {
        let text = read_body(response).await;
        let error = OpenAiError::api(status.as_u16(), text.clone());

        // The model lives on the other API, or this server has no Responses
        // endpoint at all. Either way, the fallback is the answer.
        if is_wrong_api(&text) || is_endpoint_absent(status.as_u16()) {
            return Err(ProtocolError::WrongApi(error));
        }
        // A 4xx on a constrained request means the server could not honor the
        // schema; an unconstrained retry is expected to work.
        if request.schema.is_some() && status.is_client_error() {
            return Err(ProtocolError::SchemaUnsupported);
        }
        return Err(ProtocolError::fatal(error));
    }

    match sink {
        Some(tokens) => read_stream(response, tokens)
            .await
            .map_err(ProtocolError::fatal),
        None => {
            let text = response.text().await.map_err(|e| {
                ProtocolError::fatal(OpenAiError::transport(format!(
                    "failed to read Responses body: {e}"
                )))
            })?;
            let parsed: ResponsesBody = serde_json::from_str(&text).map_err(|e| {
                ProtocolError::fatal(OpenAiError::decode(format!(
                    "failed to parse Responses body: {e}"
                )))
            })?;
            parsed
                .into_text()
                .ok_or(ProtocolError::fatal(OpenAiError::EmptyResponse))
        }
    }
}

/// POST the body, retrying the intermittent gating `403` before giving up.
async fn send_with_gate_retry<B: Serialize>(
    transport: &Transport,
    path: &str,
    body: &B,
) -> Result<reqwest::Response, ProtocolError> {
    // A `loop` rather than a bounded `for`: the exit is the `return` below, so
    // there is no fall-through path needing an unreachable panic to satisfy the
    // compiler — and no way for a later edit to the bound to make one reachable.
    let mut attempt = 0;
    loop {
        let response = transport
            .post(path, body)
            .await
            .map_err(ProtocolError::fatal)?;

        let gated = response.status() == reqwest::StatusCode::FORBIDDEN;
        if !gated || attempt >= GATED_403_RETRIES {
            return Ok(response);
        }

        attempt += 1;
        debug!("Responses API returned 403 (attempt {attempt}), retrying");
        tokio::time::sleep(GATED_403_BACKOFF).await;
    }
}

/// Consume the SSE stream, forwarding text deltas as they arrive.
async fn read_stream(
    response: reqwest::Response,
    tokens: &tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<String, OpenAiError> {
    let mut bytes = response.bytes_stream();
    let mut decoder = SseDecoder::new();
    let mut full_text = String::new();

    while let Some(chunk) = bytes.next().await {
        let chunk = chunk
            .map_err(|e| OpenAiError::transport(format!("Responses stream read error: {e}")))?;
        decoder.push(&chunk);

        for payload in decoder.drain() {
            // Unparseable events are skipped: the stream carries event types we
            // deliberately do not model, and new ones appear over time.
            let Ok(event) = serde_json::from_str::<StreamEvent>(&payload) else {
                continue;
            };
            if event.kind != TEXT_DELTA_EVENT {
                continue;
            }
            if let Some(text) = event.delta {
                full_text.push_str(&text);
                // A dropped receiver just means nobody is watching the tokens;
                // the full text is still assembled and returned.
                let _ = tokens.send(text);
            }
        }
    }

    Ok(full_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenates_output_text_parts_of_message_items() {
        let json = r#"{
            "output": [
                {"type":"reasoning","content":[{"type":"reasoning_text","text":"thinking..."}]},
                {"type":"message","content":[
                    {"type":"output_text","text":"Hello"},
                    {"type":"refusal","text":"ignored"},
                    {"type":"output_text","text":", world"}
                ]}
            ]
        }"#;
        let parsed: ResponsesBody = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.into_text().as_deref(), Some("Hello, world"));
    }

    #[test]
    fn empty_or_whitespace_output_is_none() {
        let empty: ResponsesBody = serde_json::from_str(r#"{"output":[]}"#).unwrap();
        assert_eq!(empty.into_text(), None);

        let blank: ResponsesBody = serde_json::from_str(
            r#"{"output":[{"type":"message","content":[{"type":"output_text","text":"   "}]}]}"#,
        )
        .unwrap();
        assert_eq!(blank.into_text(), None);
    }

    #[test]
    fn reasoning_only_output_yields_no_text() {
        // A reply that is nothing but a reasoning trace has no answer in it.
        let parsed: ResponsesBody = serde_json::from_str(
            r#"{"output":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"hmm"}]}]}"#,
        )
        .unwrap();
        assert_eq!(parsed.into_text(), None);
    }

    #[test]
    fn untyped_items_are_treated_as_messages() {
        // Servers that omit the item type would otherwise have their only
        // answer discarded.
        let parsed: ResponsesBody = serde_json::from_str(
            r#"{"output":[{"content":[{"type":"output_text","text":"hi"}]}]}"#,
        )
        .unwrap();
        assert_eq!(parsed.into_text().as_deref(), Some("hi"));
    }

    #[test]
    fn serializes_the_schema_under_text_format() {
        // The Responses API nests structured output under `text.format`, not
        // the top-level `response_format` Chat Completions uses.
        let body = ResponsesRequest {
            model: "m",
            input: vec![InputItem {
                role: "user",
                content: "hi",
            }],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            text: Some(TextConfig {
                format: TextFormat {
                    kind: "json_schema",
                    name: "out".to_string(),
                    strict: true,
                    schema: serde_json::json!({"type": "object"}),
                },
            }),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["text"]["format"]["type"], "json_schema");
        assert_eq!(json["text"]["format"]["name"], "out");
        assert_eq!(json["text"]["format"]["strict"], true);
        // Unset sampling knobs stay off the wire entirely.
        assert!(json.get("temperature").is_none());
        assert!(json.get("max_output_tokens").is_none());
    }

    #[test]
    fn omits_text_config_when_unconstrained() {
        let body = ResponsesRequest {
            model: "m",
            input: vec![],
            stream: true,
            temperature: Some(0.0),
            max_output_tokens: Some(256),
            text: None,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("text").is_none());
        assert_eq!(json["stream"], true);
        assert_eq!(json["temperature"], 0.0);
        assert_eq!(json["max_output_tokens"], 256);
    }

    #[test]
    fn parses_text_delta_events_and_ignores_others() {
        let delta: StreamEvent =
            serde_json::from_str(r#"{"type":"response.output_text.delta","delta":"tok"}"#).unwrap();
        assert_eq!(delta.kind, TEXT_DELTA_EVENT);
        assert_eq!(delta.delta.as_deref(), Some("tok"));

        let created: StreamEvent = serde_json::from_str(r#"{"type":"response.created"}"#).unwrap();
        assert_ne!(created.kind, TEXT_DELTA_EVENT);
        assert_eq!(created.delta, None);
    }
}
