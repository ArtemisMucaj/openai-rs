//! End-to-end chat tests against a mock OpenAI-compatible server.
//!
//! These exercise the behaviour that unit tests cannot reach: which endpoint is
//! actually hit, in what order, and how many times.

use openai_rs::{ApiFlavor, ChatClient, ChatRequest, Endpoint, JsonSchema, OpenAiChatClient};
use wiremock::matchers::{body_json_string, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const RESPONSES_PATH: &str = "/v1/responses";
const CHAT_PATH: &str = "/v1/chat/completions";

/// A Responses API body carrying `text`.
fn responses_body(text: &str) -> serde_json::Value {
    serde_json::json!({
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "…"}]},
            {"type": "message", "content": [{"type": "output_text", "text": text}]}
        ]
    })
}

/// A Chat Completions body carrying `text`.
fn chat_body(text: &str) -> serde_json::Value {
    serde_json::json!({ "choices": [{"message": {"content": text}}] })
}

/// The error both APIs return for a model served by the other one.
fn wrong_api_body() -> serde_json::Value {
    serde_json::json!({
        "error": {"message": "wrong endpoint for this model", "code": "unsupported_api_for_model"}
    })
}

fn client_for(server: &MockServer) -> OpenAiChatClient {
    OpenAiChatClient::new(&Endpoint::new(server.uri()), "test-model").unwrap()
}

/// How many requests the server received for `path`.
async fn hits(server: &MockServer, path: &str) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request: &&Request| request.url.path() == path)
        .count()
}

#[tokio::test]
async fn uses_the_responses_api_by_default() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body("hello")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    assert_eq!(client.complete("sys", "usr").await.unwrap(), "hello");

    // Responses is the primary path: Chat Completions must not be touched.
    assert_eq!(hits(&server, RESPONSES_PATH).await, 1);
    assert_eq!(hits(&server, CHAT_PATH).await, 0);
    assert_eq!(client.resolved_flavor(), Some(ApiFlavor::Responses));
}

#[tokio::test]
async fn sends_system_and_user_turns_as_responses_input() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .and(body_json_string(
            serde_json::json!({
                "model": "test-model",
                "input": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "usr"}
                ],
                "stream": false
            })
            .to_string(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body("ok")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    assert_eq!(client.complete("sys", "usr").await.unwrap(), "ok");
}

#[tokio::test]
async fn falls_back_to_chat_when_the_model_is_not_on_responses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(wrong_api_body()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_body("from chat")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    assert_eq!(client.complete("sys", "usr").await.unwrap(), "from chat");
    assert_eq!(client.resolved_flavor(), Some(ApiFlavor::ChatCompletions));
}

#[tokio::test]
async fn falls_back_when_the_server_has_no_responses_endpoint() {
    // Servers predating the Responses API answer 404 there. That must be read
    // as "try the other protocol", not as a fatal error.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_body("legacy server")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    assert_eq!(
        client.complete("sys", "usr").await.unwrap(),
        "legacy server"
    );
}

#[tokio::test]
async fn caches_the_discovered_protocol_across_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(wrong_api_body()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_body("ok")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    for _ in 0..3 {
        assert_eq!(client.complete("sys", "usr").await.unwrap(), "ok");
    }

    // The probe is paid once; the two later calls go straight to the protocol
    // that worked.
    assert_eq!(hits(&server, RESPONSES_PATH).await, 1);
    assert_eq!(hits(&server, CHAT_PATH).await, 3);
}

#[tokio::test]
async fn pinning_the_protocol_skips_the_probe_entirely() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_body("pinned")))
        .mount(&server)
        .await;

    let client = client_for(&server).prefer(ApiFlavor::ChatCompletions);
    assert_eq!(client.complete("sys", "usr").await.unwrap(), "pinned");
    assert_eq!(hits(&server, RESPONSES_PATH).await, 0);
}

#[tokio::test]
async fn retries_the_intermittent_gating_403() {
    // Some providers gate the Responses endpoint and reject a fraction of
    // requests regardless of credentials; the next attempt succeeds.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(403).set_body_string("gated"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body("after retry")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    assert_eq!(client.complete("sys", "usr").await.unwrap(), "after retry");
    assert_eq!(hits(&server, RESPONSES_PATH).await, 2);
    // A 403 is a gate, not a routing signal — the other protocol stays untouched.
    assert_eq!(hits(&server, CHAT_PATH).await, 0);
}

#[tokio::test]
async fn constrains_output_to_a_schema_on_responses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body(r#"{"n":1}"#)))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let schema = serde_json::json!({"type": "object", "properties": {"n": {"type": "integer"}}});
    let answer = client
        .complete_json("sys", "usr", "counter", &schema)
        .await
        .unwrap();
    assert_eq!(answer, r#"{"n":1}"#);

    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    // The Responses API nests structured output under `text.format`.
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["name"], "counter");
    assert_eq!(body["text"]["format"]["strict"], true);
}

#[tokio::test]
async fn retries_without_the_schema_when_the_server_rejects_it() {
    // A backend whose grammar cannot honor the schema returns 4xx. Falling back
    // to free-form output beats failing the call.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_string("schema not supported"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body("free-form")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let schema = serde_json::json!({"type": "object"});
    let answer = client
        .complete_json("sys", "usr", "out", &schema)
        .await
        .unwrap();
    assert_eq!(answer, "free-form");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let retry: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(
        retry.get("text").is_none(),
        "the retry must drop the constraint: {retry}"
    );
}

#[tokio::test]
async fn a_per_request_model_overrides_the_client_default() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body("ok")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let request = ChatRequest::from_prompt("sys", "usr").with_model("other-model");
    client.chat(&request).await.unwrap();

    let sent = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["model"], "other-model");
}

#[tokio::test]
async fn reports_the_error_when_both_protocols_refuse() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(wrong_api_body()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad credentials"))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let error = client.complete("sys", "usr").await.unwrap_err();
    assert_eq!(error.status(), Some(401));
    assert!(
        error.to_string().contains("bad credentials"),
        "got: {error}"
    );
}

#[tokio::test]
async fn surfaces_an_empty_completion_as_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"output": []})))
        .mount(&server)
        .await;

    let client = client_for(&server);
    assert!(matches!(
        client.complete("sys", "usr").await,
        Err(openai_rs::OpenAiError::EmptyResponse)
    ));
}

#[tokio::test]
async fn schema_requests_carry_the_strictness_flag() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RESPONSES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_body("{}")))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let request = ChatRequest::from_prompt("sys", "usr")
        .with_schema(JsonSchema::new("out", serde_json::json!({"type": "object"})).lenient());
    client.chat(&request).await.unwrap();

    let sent = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["text"]["format"]["strict"], false);
}
