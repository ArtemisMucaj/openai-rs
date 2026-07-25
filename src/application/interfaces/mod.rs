//! Port traits. One file per abstraction; implementations live in the
//! connector layer.

pub mod chat_client;
pub mod embedding_client;
pub mod model_catalog;

pub use chat_client::ChatClient;
pub use embedding_client::EmbeddingClient;
pub use model_catalog::ModelCatalog;
