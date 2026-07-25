//! [`EmbeddingClient`] over `POST /embeddings`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::transport::Transport;
use crate::application::EmbeddingClient;
use crate::domain::{l2_normalize, EmbeddingOptions, Endpoint, OpenAiError};

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    /// Output dimensionality, for models that support truncation. Omitted
    /// otherwise, since servers that do not understand it reject the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbeddingBody {
    #[serde(default)]
    data: Vec<EmbeddingEntry>,
}

#[derive(Deserialize)]
struct EmbeddingEntry {
    embedding: Vec<f32>,
    /// Position of the input this vector belongs to. The spec does not
    /// guarantee response ordering, so this is what restores it.
    #[serde(default)]
    index: usize,
}

pub struct OpenAiEmbeddingClient {
    transport: Transport,
    model: String,
    options: EmbeddingOptions,
}

impl OpenAiEmbeddingClient {
    pub fn new(endpoint: &Endpoint, model: impl Into<String>) -> Result<Self, OpenAiError> {
        Ok(Self::with_transport(Transport::new(endpoint)?, model))
    }

    pub fn with_transport(transport: Transport, model: impl Into<String>) -> Self {
        Self {
            transport,
            model: model.into(),
            options: EmbeddingOptions::default(),
        }
    }

    pub fn with_options(mut self, options: EmbeddingOptions) -> Self {
        self.options = options;
        self
    }

    pub fn options(&self) -> &EmbeddingOptions {
        &self.options
    }

    /// Embed one batch, small enough to send in a single request.
    async fn embed_one_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError> {
        let request = EmbeddingRequest {
            model: &self.model,
            input: inputs,
            dimensions: self.options.dimensions,
        };

        let path = &self.transport.routes().embeddings;
        let body: EmbeddingBody = self.transport.post_json(path, &request).await?;

        if body.data.len() != inputs.len() {
            return Err(OpenAiError::decode(format!(
                "expected {} embeddings, got {}",
                inputs.len(),
                body.data.len()
            )));
        }

        let mut entries = body.data;
        entries.sort_by_key(|entry| entry.index);

        // A matching count is not enough: duplicated or out-of-range indexes
        // pass the length check and then pair vectors with the wrong inputs,
        // which no caller can detect afterwards. Sorted indexes must be exactly
        // `0..n` for the mapping to be one-to-one.
        for (position, entry) in entries.iter().enumerate() {
            if entry.index != position {
                return Err(OpenAiError::decode(format!(
                    "embedding indexes do not map one-to-one onto the inputs \
                     (expected {position} at this position, got {})",
                    entry.index
                )));
            }
        }

        Ok(entries
            .into_iter()
            .map(|entry| {
                let mut vector = entry.embedding;
                if self.options.normalize {
                    l2_normalize(&mut vector);
                }
                vector
            })
            .collect())
    }
}

#[async_trait]
impl EmbeddingClient for OpenAiEmbeddingClient {
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut vectors = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(self.options.batch_size) {
            vectors.extend(self.embed_one_batch(batch).await?);
        }

        debug!(
            "embedded {} input(s) with '{}' ({} dims)",
            vectors.len(),
            self.model,
            vectors.first().map(Vec::len).unwrap_or(0)
        );
        Ok(vectors)
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_input_order_from_the_index() {
        // The spec does not promise ordered results; a shuffled response must
        // still line up with the caller's inputs.
        let body: EmbeddingBody = serde_json::from_str(
            r#"{"data":[
                {"embedding":[0.0,1.0],"index":1},
                {"embedding":[1.0,0.0],"index":0}
            ]}"#,
        )
        .unwrap();
        let mut entries = body.data;
        entries.sort_by_key(|entry| entry.index);
        assert_eq!(entries[0].embedding, vec![1.0, 0.0]);
        assert_eq!(entries[1].embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn omits_dimensions_when_unset() {
        // Servers that do not understand `dimensions` reject requests carrying it.
        let inputs = vec!["hello".to_string()];
        let request = EmbeddingRequest {
            model: "m",
            input: &inputs,
            dimensions: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert!(json.get("dimensions").is_none());

        let request = EmbeddingRequest {
            model: "m",
            input: &inputs,
            dimensions: Some(256),
        };
        assert_eq!(serde_json::to_value(&request).unwrap()["dimensions"], 256);
    }

    #[test]
    fn options_are_configurable() {
        let endpoint = Endpoint::new("http://localhost:1234");
        let client = OpenAiEmbeddingClient::new(&endpoint, "embed-model")
            .unwrap()
            .with_options(
                EmbeddingOptions::default()
                    .with_normalize(false)
                    .with_batch_size(8),
            );
        assert_eq!(client.model(), "embed-model");
        assert!(!client.options().normalize);
        assert_eq!(client.options().batch_size, 8);
    }
}
