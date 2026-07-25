use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

use crate::domain::{ChatRequest, JsonSchema, OpenAiError};

/// Port for sending a conversation to a model and receiving text back.
///
/// Implementors encapsulate transport, serialization, and provider-specific API
/// details, so consumers stay decoupled from any particular vendor or HTTP
/// client. [`Self::chat`] is the only required method; everything else has a
/// default built on top of it, so a minimal provider satisfies the whole
/// contract by implementing one function.
#[async_trait]
pub trait ChatClient: Send + Sync {
    /// Run `request` and return the assistant's text.
    async fn chat(&self, request: &ChatRequest) -> Result<String, OpenAiError>;

    /// Run `request`, forwarding each token to `tokens` as it arrives, and
    /// return the full concatenated text once the stream is exhausted.
    ///
    /// The default implementation calls [`Self::chat`] and delivers the whole
    /// response as one chunk, so providers without streaming still satisfy the
    /// contract.
    async fn chat_stream(
        &self,
        request: &ChatRequest,
        tokens: UnboundedSender<String>,
    ) -> Result<String, OpenAiError> {
        let text = self.chat(request).await?;
        // A dropped receiver is not an error: the caller stopped listening but
        // still wants the return value.
        let _ = tokens.send(text.clone());
        Ok(text)
    }

    /// Send a `system` context message followed by a `user` prompt.
    async fn complete(&self, system: &str, user: &str) -> Result<String, OpenAiError> {
        self.chat(&ChatRequest::from_prompt(system, user)).await
    }

    /// Like [`Self::complete`], but constrains the response to `schema`.
    ///
    /// `schema_name` is a short identifier for the schema; `schema` is the JSON
    /// Schema object. Providers that cannot constrain decoding fall back to
    /// free-form output, so the caller must still tolerate imperfect JSON.
    async fn complete_json(
        &self,
        system: &str,
        user: &str,
        schema_name: &str,
        schema: &serde_json::Value,
    ) -> Result<String, OpenAiError> {
        let request = ChatRequest::from_prompt(system, user)
            .with_schema(JsonSchema::new(schema_name, schema.clone()));
        self.chat(&request).await
    }

    /// Streaming counterpart of [`Self::complete`].
    async fn complete_stream(
        &self,
        system: &str,
        user: &str,
        tokens: UnboundedSender<String>,
    ) -> Result<String, OpenAiError> {
        self.chat_stream(&ChatRequest::from_prompt(system, user), tokens)
            .await
    }
}
