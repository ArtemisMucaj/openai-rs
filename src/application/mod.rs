//! Ports the connector layer implements and consumers depend on.
//!
//! Depends only on [`crate::domain`].

pub mod interfaces;

pub use interfaces::{ChatClient, EmbeddingClient, ModelCatalog};
