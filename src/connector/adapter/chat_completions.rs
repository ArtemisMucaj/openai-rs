//! The Chat Completions API (`/chat/completions`) — the fallback protocol.
//!
//! Tried when [`responses`](super::responses) reports that a model is served by
//! the other API, or that the server has no Responses endpoint at all. Older
//! self-hosted servers only ever speak this protocol, so the fallback is the
//! whole story for them; the client caches that outcome after the first call.

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use super::protocol::{is_endpoint_absent, is_wrong_api, ProtocolError, TokenSink};
use super::sse::{SseDecoder, DONE_SENTINEL};
use super::transport::{read_body, Transport};
use crate::domain::{ChatRequest, OpenAiError};

#[derive(Serialize)]
struct ChatCompletionsRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// `response_format: { type: "json_schema", json_schema: { … } }` — asks the
/// server to grammar-constrain output to the schema.
#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JsonSchemaSpec,
}

#[derive(Serialize)]
struct JsonSchemaSpec {
    name: String,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Deserialize)]
struct ChatCompletionsBody {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Some reasoning models route the whole answer into a separate reasoning
    /// channel and leave `content` empty. Falling back to it keeps the response
    /// from being lost.
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl ResponseMessage {
    fn into_text(self) -> Option<String> {
        self.content
            .filter(|text| !text.trim().is_empty())
            .or_else(|| {
                self.reasoning_content
                    .filter(|text| !text.trim().is_empty())
            })
    }
}

/// One chunk of a streaming response.
#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Run `request` against Chat Completions, streaming to `sink` when present.
pub(crate) async fn execute(
    transport: &Transport,
    model: &str,
    request: &ChatRequest,
    sink: TokenSink<'_>,
) -> Result<String, ProtocolError> {
    let body = ChatCompletionsRequest {
        model,
        messages: request
            .messages
            .iter()
            .map(|message| WireMessage {
                role: message.role.as_str(),
                content: &message.content,
            })
            .collect(),
        stream: sink.is_some(),
        temperature: request.temperature,
        max_tokens: request.max_output_tokens,
        response_format: request.schema.as_ref().map(|schema| ResponseFormat {
            kind: "json_schema",
            json_schema: JsonSchemaSpec {
                name: schema.name.clone(),
                strict: schema.strict,
                schema: schema.schema.clone(),
            },
        }),
    };

    let path = &transport.routes().chat_completions;
    let response = transport
        .post(path, &body)
        .await
        .map_err(ProtocolError::fatal)?;

    let status = response.status();
    if !status.is_success() {
        let text = read_body(response).await;
        let error = OpenAiError::api(status.as_u16(), text.clone());

        if is_wrong_api(&text) || is_endpoint_absent(status.as_u16()) {
            return Err(ProtocolError::WrongApi(error));
        }
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
                    "failed to read chat body: {e}"
                )))
            })?;
            let parsed: ChatCompletionsBody = serde_json::from_str(&text).map_err(|e| {
                ProtocolError::fatal(OpenAiError::decode(format!(
                    "failed to parse chat body: {e}"
                )))
            })?;
            parsed
                .choices
                .into_iter()
                .next()
                .and_then(|choice| choice.message.into_text())
                .ok_or(ProtocolError::fatal(OpenAiError::EmptyResponse))
        }
    }
}

/// Consume the SSE stream, forwarding content deltas until `[DONE]`.
async fn read_stream(
    response: reqwest::Response,
    tokens: &tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<String, OpenAiError> {
    let mut bytes = response.bytes_stream();
    let mut decoder = SseDecoder::new();
    let mut full_text = String::new();

    'outer: while let Some(chunk) = bytes.next().await {
        let chunk =
            chunk.map_err(|e| OpenAiError::transport(format!("chat stream read error: {e}")))?;
        decoder.push(&chunk);

        for payload in decoder.drain() {
            if payload == DONE_SENTINEL {
                break 'outer;
            }
            // Keep-alives and chunks we do not model are skipped rather than
            // failing a stream that is otherwise fine.
            let Ok(chunk) = serde_json::from_str::<StreamChunk>(&payload) else {
                continue;
            };
            if let Some(text) = chunk
                .choices
                .into_iter()
                .next()
                .and_then(|choice| choice.delta.content)
            {
                full_text.push_str(&text);
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
    fn prefers_content_over_the_reasoning_channel() {
        let message: ResponseMessage =
            serde_json::from_str(r#"{"content":"answer","reasoning_content":"thoughts"}"#).unwrap();
        assert_eq!(message.into_text().as_deref(), Some("answer"));
    }

    #[test]
    fn falls_back_to_the_reasoning_channel_when_content_is_blank() {
        // Reasoning models routinely leave `content` empty; treating that as an
        // empty response would discard the only answer.
        let message: ResponseMessage =
            serde_json::from_str(r#"{"content":"   ","reasoning_content":"answer"}"#).unwrap();
        assert_eq!(message.into_text().as_deref(), Some("answer"));

        let missing: ResponseMessage =
            serde_json::from_str(r#"{"reasoning_content":"answer"}"#).unwrap();
        assert_eq!(missing.into_text().as_deref(), Some("answer"));
    }

    #[test]
    fn both_channels_blank_yields_nothing() {
        let message: ResponseMessage =
            serde_json::from_str(r#"{"content":"","reasoning_content":null}"#).unwrap();
        assert_eq!(message.into_text(), None);
    }

    #[test]
    fn serializes_the_schema_as_response_format() {
        let body = ChatCompletionsRequest {
            model: "m",
            messages: vec![WireMessage {
                role: "system",
                content: "s",
            }],
            stream: false,
            temperature: Some(0.0),
            max_tokens: Some(128),
            response_format: Some(ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchemaSpec {
                    name: "out".to_string(),
                    strict: true,
                    schema: serde_json::json!({"type": "object"}),
                },
            }),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["name"], "out");
        // Chat Completions spells the output cap `max_tokens`.
        assert_eq!(json["max_tokens"], 128);
        assert_eq!(json["messages"][0]["role"], "system");
    }

    #[test]
    fn omits_unset_fields() {
        let body = ChatCompletionsRequest {
            model: "m",
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            response_format: None,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("temperature").is_none());
        assert!(json.get("max_tokens").is_none());
        assert!(json.get("response_format").is_none());
    }

    #[test]
    fn parses_stream_deltas() {
        let chunk: StreamChunk =
            serde_json::from_str(r#"{"choices":[{"delta":{"content":"tok"}}]}"#).unwrap();
        assert_eq!(
            chunk.choices.into_iter().next().unwrap().delta.content,
            Some("tok".to_string())
        );

        // The final chunk carries a role-only or empty delta.
        let closing: StreamChunk = serde_json::from_str(r#"{"choices":[{"delta":{}}]}"#).unwrap();
        assert_eq!(
            closing.choices.into_iter().next().unwrap().delta.content,
            None
        );
    }
}
