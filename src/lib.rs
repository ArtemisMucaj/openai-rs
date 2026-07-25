//! Client for OpenAI-compatible servers.
//!
//! Speaks the **Responses API** by default and falls back to **Chat
//! Completions** for models or servers that only support it, caching whichever
//! works so the probe is paid once. Also covers model discovery and embeddings.
//!
//! # Layering
//!
//! Dependencies point inward, following ports and adapters:
//!
//! - [`domain`] — pure value types ([`Endpoint`], [`Message`], [`ChatRequest`],
//!   [`Model`], [`OpenAiError`]). No I/O, no async.
//! - [`application`] — the port traits ([`ChatClient`], [`ModelCatalog`],
//!   [`EmbeddingClient`]). Depends only on the domain.
//! - [`connector`] — HTTP adapters implementing those ports.
//!
//! Configuration resolution — environment variables, config files, keychains —
//! is deliberately **out of scope**. Callers resolve credentials however they
//! like and hand over a finished [`Endpoint`].
//!
//! # Example
//!
//! ```no_run
//! use openai_rs::{ChatClient, Endpoint, OpenAiChatClient};
//!
//! # async fn run() -> Result<(), openai_rs::OpenAiError> {
//! let endpoint = Endpoint::new("https://api.openai.com").with_api_key("sk-…");
//! let client = OpenAiChatClient::new(&endpoint, "gpt-5.2")?;
//!
//! let answer = client.complete("You are terse.", "Why is the sky blue?").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```
//!
//! # Reuse by other providers
//!
//! A provider whose API is OpenAI-compatible but sits behind different routes
//! and headers builds an [`Endpoint`] describing both, then hands the resulting
//! [`Transport`] to these adapters — no protocol code is duplicated:
//!
//! ```no_run
//! use openai_rs::{ApiRoutes, Endpoint, OpenAiChatClient, Transport};
//!
//! # fn build() -> Result<(), openai_rs::OpenAiError> {
//! let endpoint = Endpoint::new("https://api.example.com")
//!     .with_api_key("token")
//!     .with_routes(ApiRoutes::unversioned())
//!     .with_header("X-Api-Version", "2025-04-01");
//!
//! let transport = Transport::new(&endpoint)?;
//! let client = OpenAiChatClient::with_transport(transport.clone(), "some-model");
//! # Ok(())
//! # }
//! ```

pub mod application;
pub mod connector;
pub mod domain;

pub use application::{ChatClient, EmbeddingClient, ModelCatalog};
pub use connector::{
    ApiFlavor, OpenAiChatClient, OpenAiEmbeddingClient, OpenAiModelCatalog, Transport,
};
pub use domain::{
    l2_normalize, ApiRoutes, ChatRequest, EmbeddingOptions, Endpoint, JsonSchema, Message, Model,
    ModelLimits, OpenAiError, Role, DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT,
};
