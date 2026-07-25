use async_trait::async_trait;

use crate::domain::{Model, OpenAiError};

/// Port for discovering which models a provider offers.
///
/// Kept separate from [`super::ChatClient`] because not every backend can
/// enumerate its models (the Anthropic Messages API, for one, has no portable
/// discovery endpoint), and a host that only sends completions should not have
/// to care.
#[async_trait]
pub trait ModelCatalog: Send + Sync {
    /// Every model available to the configured credentials.
    async fn list_models(&self) -> Result<Vec<Model>, OpenAiError>;
}
