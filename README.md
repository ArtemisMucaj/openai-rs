# openai-rs

A Rust client for OpenAI-compatible servers — hosted OpenAI, LM Studio, vLLM, or
anything else speaking the same wire protocol.

- **Responses API first.** `/responses` is the primary path; `/chat/completions`
  is the fallback for models and servers that only support it. Whichever works
  is cached, so the probe is paid once per client rather than on every call.
- **Streaming** over SSE on both protocols.
- **Structured output** via JSON Schema, with an automatic unconstrained retry
  when a server cannot honor the constraint.
- **Model discovery** and **embeddings** behind the same configuration.
- **No configuration resolution.** No environment variables are read, no files
  are touched. The caller resolves credentials and hands over an `Endpoint`.

## Install

```toml
[dependencies]
openai-rs = { git = "https://github.com/ArtemisMucaj/openai-rs.git" }
```

## Usage

```rust
use openai_rs::{ChatClient, Endpoint, OpenAiChatClient};

let endpoint = Endpoint::new("https://api.openai.com").with_api_key("sk-…");
let client = OpenAiChatClient::new(&endpoint, "gpt-5.2")?;

let answer = client.complete("You are terse.", "Why is the sky blue?").await?;
```

### Streaming

```rust
let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
let full = client.complete_stream("You are terse.", "Explain SSE.", tx).await?;
while let Some(token) = rx.recv().await {
    print!("{token}");
}
```

### Structured output

```rust
let schema = serde_json::json!({
    "type": "object",
    "properties": { "sentiment": { "type": "string" } },
    "required": ["sentiment"]
});
let json = client
    .complete_json("Classify the input.", "I love it", "verdict", &schema)
    .await?;
```

The schema is sent as `text.format` on the Responses API and as
`response_format` on Chat Completions. If the server rejects it — some local
engines cannot grammar-constrain every schema — the call is retried
unconstrained rather than failing, so callers should still parse defensively.

### Model discovery and embeddings

```rust
use openai_rs::{EmbeddingClient, ModelCatalog, OpenAiEmbeddingClient, OpenAiModelCatalog};

for model in OpenAiModelCatalog::new(&endpoint)?.list_models().await? {
    println!("{}", model.label());
}

let embeddings = OpenAiEmbeddingClient::new(&endpoint, "text-embedding-3-small")?
    .embed_batch(&inputs)
    .await?;
```

Embeddings are batched internally (32 per request by default) and L2-normalised
so cosine similarity equals the dot product. Both are configurable through
`EmbeddingOptions`.

## Protocol selection

```
                       ┌─────────────┐
        first call ───▶│ /responses  │──── success ───▶ cached: Responses
                       └──────┬──────┘
                              │ unsupported_api_for_model
                              │ 404 / 405 / 501
                              ▼
                   ┌───────────────────┐
                   │ /chat/completions │─── success ───▶ cached: Chat
                   └───────────────────┘
```

A `403` is read as a rollout gate rather than a routing signal: some providers
reject a fraction of Responses calls regardless of credentials, so it is retried
on the same endpoint before anything else is tried.

Pin the protocol with `.prefer(ApiFlavor::ChatCompletions)` when a server is
known to speak only one — it saves the first call's failed probe.

## Reuse by other providers

A provider whose API is OpenAI-compatible but sits behind different routes and
headers describes both as data, then hands the resulting `Transport` to these
adapters. No protocol code is duplicated:

```rust
use openai_rs::{ApiRoutes, Endpoint, OpenAiChatClient, Transport};

let endpoint = Endpoint::new("https://api.example.com")
    .with_api_key(token)
    .with_routes(ApiRoutes::unversioned())
    .with_header("X-Api-Version", "2025-04-01");

let transport = Transport::new(&endpoint)?;
let client = OpenAiChatClient::with_transport(transport.clone(), "some-model");
```

[`gh-copilot-rs`](https://github.com/ArtemisMucaj/gh-copilot-rs) is built exactly
this way.

## Architecture

Ports and adapters; dependencies point inward.

| Layer | Path | Contents |
|---|---|---|
| Domain | `src/domain/` | `Endpoint`, `Message`, `ChatRequest`, `Model`, `OpenAiError`. No I/O, no async. |
| Application | `src/application/` | Port traits: `ChatClient`, `ModelCatalog`, `EmbeddingClient`. |
| Connector | `src/connector/` | `Transport` plus the HTTP adapters implementing those ports. |

See [AGENTS.md](AGENTS.md) for conventions and development workflow.

## Testing

```bash
cargo test
```

Unit tests cover wire-format parsing and policy decisions; integration tests run
against a `wiremock` server on localhost. Nothing reaches the network.

## License

MIT
