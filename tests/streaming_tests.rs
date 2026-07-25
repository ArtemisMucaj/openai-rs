//! Streaming tests: token delivery, protocol fallback mid-discovery, and the
//! guarantee that a fallback never double-delivers tokens.

use openai_rs::{ChatClient, Endpoint, OpenAiChatClient};
use tokio::sync::mpsc::unbounded_channel;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const RESPONSES_PATH: &str = "/v1/responses";
const CHAT_PATH: &str = "/v1/chat/completions";

/// An SSE body, joined with the blank-line separators a real server emits.
fn sse(events: &[&str]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

fn stream_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "text/event-stream")
}

fn client_for(server: &MockServer) -> OpenAiChatClient {
    OpenAiChatClient::new(&Endpoint::new(server.uri()), "test-model").unwrap()
}

/// Drain a token channel into a vector once the stream has finished.
fn collect(mut rx: tokio::sync::mpsc::UnboundedReceiver<String>) -> Vec<String> {
    let mut tokens = Vec::new();
    while let Ok(token) = rx.try_recv() {
        tokens.push(token);
    }
    tokens
}

#[tokio::test]
async fn streams_tokens_from_the_responses_api() {
    let server = MockServer::start().await;
    let body = sse(&[
        r#"{"type":"response.created"}"#,
        r#"{"type":"response.output_text.delta","delta":"Hello"}"#,
        r#"{"type":"response.output_text.delta","delta":", "}"#,
        r#"{"type":"response.output_text.delta","delta":"world"}"#,
        r#"{"type":"response.completed"}"#,
    ]);
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(stream_response(body))
        .mount(&server)
        .await;

    let (tx, rx) = unbounded_channel();
    let full = client_for(&server)
        .complete_stream("sys", "usr", tx)
        .await
        .unwrap();

    assert_eq!(full, "Hello, world");
    // Non-text events must not leak into the token stream.
    assert_eq!(collect(rx), vec!["Hello", ", ", "world"]);
}

#[tokio::test]
async fn streams_tokens_from_chat_completions() {
    let server = MockServer::start().await;
    let mut body = sse(&[
        r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
        r#"{"choices":[{"delta":{"content":"one"}}]}"#,
        r#"{"choices":[{"delta":{"content":" two"}}]}"#,
    ]);
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(stream_response(body))
        .mount(&server)
        .await;

    let (tx, rx) = unbounded_channel();
    let full = client_for(&server)
        .prefer(openai_rs::ApiFlavor::ChatCompletions)
        .complete_stream("sys", "usr", tx)
        .await
        .unwrap();

    assert_eq!(full, "one two");
    assert_eq!(collect(rx), vec!["one", " two"]);
}

#[tokio::test]
async fn a_streaming_fallback_delivers_each_token_exactly_once() {
    // The switch-protocols signal is detected from the response status, before
    // any byte of the stream is read — so the retry cannot replay tokens the
    // caller already saw.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {"code": "unsupported_api_for_model"}
        })))
        .mount(&server)
        .await;

    let mut body = sse(&[
        r#"{"choices":[{"delta":{"content":"a"}}]}"#,
        r#"{"choices":[{"delta":{"content":"b"}}]}"#,
    ]);
    body.push_str("data: [DONE]\n\n");
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(stream_response(body))
        .mount(&server)
        .await;

    let (tx, rx) = unbounded_channel();
    let full = client_for(&server)
        .complete_stream("sys", "usr", tx)
        .await
        .unwrap();

    assert_eq!(full, "ab");
    assert_eq!(collect(rx), vec!["a", "b"], "tokens must not be replayed");
}

#[tokio::test]
async fn stops_at_the_done_sentinel() {
    // Anything a server emits after [DONE] is not part of the answer.
    let server = MockServer::start().await;
    let mut body = sse(&[r#"{"choices":[{"delta":{"content":"kept"}}]}"#]);
    body.push_str("data: [DONE]\n\n");
    body.push_str("data: {\"choices\":[{\"delta\":{\"content\":\"dropped\"}}]}\n\n");

    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(stream_response(body))
        .mount(&server)
        .await;

    let (tx, rx) = unbounded_channel();
    let full = client_for(&server)
        .prefer(openai_rs::ApiFlavor::ChatCompletions)
        .complete_stream("sys", "usr", tx)
        .await
        .unwrap();

    assert_eq!(full, "kept");
    assert_eq!(collect(rx), vec!["kept"]);
}

#[tokio::test]
async fn a_dropped_receiver_does_not_fail_the_call() {
    // The caller may stop listening and still want the assembled text.
    let server = MockServer::start().await;
    let body = sse(&[r#"{"type":"response.output_text.delta","delta":"still assembled"}"#]);
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(stream_response(body))
        .mount(&server)
        .await;

    let (tx, rx) = unbounded_channel();
    drop(rx);

    let full = client_for(&server)
        .complete_stream("sys", "usr", tx)
        .await
        .unwrap();
    assert_eq!(full, "still assembled");
}

#[tokio::test]
async fn requests_streaming_explicitly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(stream_response(sse(&[
            r#"{"type":"response.output_text.delta","delta":"x"}"#,
        ])))
        .mount(&server)
        .await;

    let (tx, _rx) = unbounded_channel();
    client_for(&server)
        .complete_stream("sys", "usr", tx)
        .await
        .unwrap();

    let sent = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["stream"], true);
}
