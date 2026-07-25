use async_trait::async_trait;

use crate::domain::OpenAiError;

/// Port for turning text into vectors.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Embed several inputs, preserving input order in the returned vectors.
    ///
    /// Implementors are expected to batch internally, so callers can hand over
    /// an arbitrarily long slice without managing request sizes themselves.
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError>;

    /// Embed a single input.
    ///
    /// Defaults to a one-element [`Self::embed_batch`]; providers with a cheaper
    /// single-input path may override it.
    async fn embed(&self, input: &str) -> Result<Vec<f32>, OpenAiError> {
        let vectors = self
            .embed_batch(std::slice::from_ref(&input.to_string()))
            .await?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| OpenAiError::decode("embedding response contained no vectors"))
    }

    /// The model id used for embedding — callers persist it alongside the
    /// vectors, since vectors from different models are not comparable.
    fn model(&self) -> &str;
}
