//! Pure value types shared by every layer.
//!
//! No I/O, no async, no dependency on the HTTP client — only `serde` and
//! `thiserror`. Anything here can be constructed and asserted on in a test
//! without a server.

pub mod embedding;
pub mod endpoint;
pub mod error;
pub mod message;
pub mod model;
pub mod request;

pub use embedding::{l2_normalize, EmbeddingOptions, DEFAULT_BATCH_SIZE};
pub use endpoint::{ApiRoutes, Endpoint, DEFAULT_TIMEOUT};
pub use error::OpenAiError;
pub use message::{Message, Role};
pub use model::{Model, ModelLimits};
pub use request::{ChatRequest, JsonSchema};
