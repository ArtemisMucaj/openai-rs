//! [`ModelCatalog`] over `GET /models`.
//!
//! The spec only guarantees an `id` per entry, so most of the neutral
//! [`Model`] descriptor stays empty here. Providers with richer catalogs
//! implement the same port and fill in the rest.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use super::transport::Transport;
use crate::application::ModelCatalog;
use crate::domain::{Endpoint, Model, OpenAiError};

/// Wall-clock budget for a listing, so a stalled server cannot hang a UI that
/// is populating a model picker. Shorter than the completion timeout because
/// enumerating models is a fast metadata call.
pub const DEFAULT_LIST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Deserialize)]
struct ModelsBody {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

/// One catalog entry. Only `id` is required by the spec; `owned_by` is
/// widespread enough to be worth surfacing as the vendor.
#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
}

impl From<ModelEntry> for Model {
    fn from(entry: ModelEntry) -> Self {
        let model = Model::new(entry.id);
        match entry.owned_by.filter(|owner| !owner.is_empty()) {
            Some(owner) => model.with_vendor(owner),
            None => model,
        }
    }
}

pub struct OpenAiModelCatalog {
    transport: Transport,
    timeout: Duration,
}

impl OpenAiModelCatalog {
    pub fn new(endpoint: &Endpoint) -> Result<Self, OpenAiError> {
        Ok(Self::with_transport(Transport::new(endpoint)?))
    }

    /// Build on an existing transport, sharing another client's credentials and
    /// connection pool.
    pub fn with_transport(transport: Transport) -> Self {
        Self {
            transport,
            timeout: DEFAULT_LIST_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl ModelCatalog for OpenAiModelCatalog {
    async fn list_models(&self) -> Result<Vec<Model>, OpenAiError> {
        let path = &self.transport.routes().models;
        let fetch = self.transport.get_json::<ModelsBody>(path);

        let body = tokio::time::timeout(self.timeout, fetch)
            .await
            .map_err(|_| {
                OpenAiError::transport(format!(
                    "listing models timed out after {}s",
                    self.timeout.as_secs()
                ))
            })??;

        Ok(body.data.into_iter().map(Model::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_entries_to_neutral_models() {
        let body: ModelsBody = serde_json::from_str(
            r#"{"data":[
                {"id":"gpt-x","object":"model","owned_by":"openai"},
                {"id":"local-model"}
            ]}"#,
        )
        .unwrap();
        let models: Vec<Model> = body.data.into_iter().map(Model::from).collect();

        assert_eq!(models[0].id, "gpt-x");
        assert_eq!(models[0].vendor.as_deref(), Some("openai"));
        assert_eq!(models[1].id, "local-model");
        assert_eq!(models[1].vendor, None);
    }

    #[test]
    fn empty_owner_is_not_a_vendor() {
        // An empty string in the payload means "unknown", not a vendor named "".
        let body: ModelsBody =
            serde_json::from_str(r#"{"data":[{"id":"m","owned_by":""}]}"#).unwrap();
        let model = Model::from(body.data.into_iter().next().unwrap());
        assert_eq!(model.vendor, None);
    }

    #[test]
    fn missing_data_array_yields_no_models() {
        let body: ModelsBody = serde_json::from_str("{}").unwrap();
        assert!(body.data.is_empty());
    }
}
