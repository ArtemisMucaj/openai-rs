//! Concrete implementations of the application ports, over HTTP.

mod chat_completions;
mod protocol;
mod responses;
mod sse;

pub mod openai_chat_client;
pub mod openai_embedding_client;
pub mod openai_model_catalog;
pub mod transport;

pub use openai_chat_client::{ApiFlavor, OpenAiChatClient};
pub use openai_embedding_client::OpenAiEmbeddingClient;
pub use openai_model_catalog::{OpenAiModelCatalog, DEFAULT_LIST_TIMEOUT};
pub use transport::Transport;
