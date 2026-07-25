//! Adapters that implement the application ports. Depends on
//! [`crate::application`] and [`crate::domain`].

pub mod adapter;

pub use adapter::{
    ApiFlavor, OpenAiChatClient, OpenAiEmbeddingClient, OpenAiModelCatalog, Transport,
};
