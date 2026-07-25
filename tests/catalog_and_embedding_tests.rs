//! Model discovery and embedding tests against a mock server.

use std::time::Duration;

use openai_rs::{
    EmbeddingClient, EmbeddingOptions, Endpoint, ModelCatalog, OpenAiEmbeddingClient,
    OpenAiModelCatalog,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MODELS_PATH: &str = "/v1/models";
const EMBEDDINGS_PATH: &str = "/v1/embeddings";

#[tokio::test]
async fn lists_models_from_the_catalog_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {"id": "model-a", "object": "model", "owned_by": "acme"},
                {"id": "model-b", "object": "model"}
            ]
        })))
        .mount(&server)
        .await;

    let catalog = OpenAiModelCatalog::new(&Endpoint::new(server.uri())).unwrap();
    let models = catalog.list_models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "model-a");
    assert_eq!(models[0].vendor.as_deref(), Some("acme"));
    assert_eq!(models[1].id, "model-b");
    assert_eq!(models[1].vendor, None);
    // Nothing to display beyond the id, so the label is the id.
    assert_eq!(models[1].label(), "model-b");
}

#[tokio::test]
async fn surfaces_a_catalog_failure_with_its_status_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(ResponseTemplate::new(401).set_body_string("missing credentials"))
        .mount(&server)
        .await;

    let catalog = OpenAiModelCatalog::new(&Endpoint::new(server.uri())).unwrap();
    let error = catalog.list_models().await.unwrap_err();
    assert_eq!(error.status(), Some(401));
    assert!(error.to_string().contains("missing credentials"));
}

#[tokio::test]
async fn a_stalled_catalog_call_times_out() {
    // A picker populating a dropdown must not hang forever on a wedged server.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"data": []}))
                .set_delay(Duration::from_secs(30)),
        )
        .mount(&server)
        .await;

    let catalog = OpenAiModelCatalog::new(&Endpoint::new(server.uri()))
        .unwrap()
        .with_timeout(Duration::from_millis(150));

    let error = catalog.list_models().await.unwrap_err();
    assert!(error.to_string().contains("timed out"), "got: {error}");
}

#[tokio::test]
async fn embeds_and_normalises_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"embedding": [3.0, 4.0], "index": 0}]
        })))
        .mount(&server)
        .await;

    let client = OpenAiEmbeddingClient::new(&Endpoint::new(server.uri()), "embed-model").unwrap();
    let vector = client.embed("hello").await.unwrap();

    // 3,4 has length 5; normalised it is 0.6,0.8.
    assert!((vector[0] - 0.6).abs() < 1e-6);
    assert!((vector[1] - 0.8).abs() < 1e-6);
    assert_eq!(client.model(), "embed-model");
}

#[tokio::test]
async fn preserves_input_order_when_the_server_responds_out_of_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"embedding": [0.0, 1.0], "index": 1},
                {"embedding": [1.0, 0.0], "index": 0}
            ]
        })))
        .mount(&server)
        .await;

    let client = OpenAiEmbeddingClient::new(&Endpoint::new(server.uri()), "embed-model").unwrap();
    let inputs = vec!["first".to_string(), "second".to_string()];
    let vectors = client.embed_batch(&inputs).await.unwrap();

    assert_eq!(vectors[0], vec![1.0, 0.0]);
    assert_eq!(vectors[1], vec![0.0, 1.0]);
}

#[tokio::test]
async fn splits_large_inputs_into_batches() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"embedding": [1.0], "index": 0},
                {"embedding": [1.0], "index": 1}
            ]
        })))
        .mount(&server)
        .await;

    let client = OpenAiEmbeddingClient::new(&Endpoint::new(server.uri()), "embed-model")
        .unwrap()
        .with_options(EmbeddingOptions::default().with_batch_size(2));

    let inputs: Vec<String> = (0..4).map(|i| format!("input {i}")).collect();
    let vectors = client.embed_batch(&inputs).await.unwrap();

    assert_eq!(vectors.len(), 4);
    // Four inputs at two per request is two round-trips.
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn normalisation_can_be_turned_off() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"embedding": [3.0, 4.0], "index": 0}]
        })))
        .mount(&server)
        .await;

    let client = OpenAiEmbeddingClient::new(&Endpoint::new(server.uri()), "embed-model")
        .unwrap()
        .with_options(EmbeddingOptions::default().with_normalize(false));

    assert_eq!(client.embed("hello").await.unwrap(), vec![3.0, 4.0]);
}

#[tokio::test]
async fn embedding_an_empty_slice_makes_no_request() {
    let server = MockServer::start().await;
    let client = OpenAiEmbeddingClient::new(&Endpoint::new(server.uri()), "embed-model").unwrap();

    assert!(client.embed_batch(&[]).await.unwrap().is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_short_response_is_an_error_rather_than_a_silent_misalignment() {
    // Returning fewer vectors than inputs would silently pair each caller item
    // with the wrong vector, so it must fail loudly instead.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"embedding": [1.0], "index": 0}]
        })))
        .mount(&server)
        .await;

    let client = OpenAiEmbeddingClient::new(&Endpoint::new(server.uri()), "embed-model").unwrap();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let error = client.embed_batch(&inputs).await.unwrap_err();
    assert!(
        error.to_string().contains("expected 2 embeddings"),
        "got: {error}"
    );
}

#[tokio::test]
async fn sends_the_configured_credentials() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": []})))
        .mount(&server)
        .await;

    let endpoint = Endpoint::new(server.uri())
        .with_api_key("secret-key")
        .with_header("X-Custom", "value");
    OpenAiModelCatalog::new(&endpoint)
        .unwrap()
        .list_models()
        .await
        .unwrap();

    let request = &server.received_requests().await.unwrap()[0];
    assert_eq!(
        request.headers.get("authorization").unwrap(),
        "Bearer secret-key"
    );
    assert_eq!(request.headers.get("x-custom").unwrap(), "value");
}
