//! [`ChatClient`] for any OpenAI-compatible server.
//!
//! Speaks both protocols and picks between them automatically. The Responses
//! API is tried first; if the server reports that the model belongs to Chat
//! Completions — or that it has no Responses endpoint at all — the call is
//! retried there and **the outcome is cached**, so the probe is paid once per
//! client rather than on every request.

use std::sync::atomic::{AtomicU8, Ordering};

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, warn};

use super::protocol::{ProtocolError, TokenSink};
use super::transport::Transport;
use super::{chat_completions, responses};
use crate::application::ChatClient;
use crate::domain::{ChatRequest, Endpoint, OpenAiError};

/// Which protocol to speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiFlavor {
    /// `/responses` — the default.
    Responses,
    /// `/chat/completions` — for models and servers that only speak it.
    ChatCompletions,
}

impl ApiFlavor {
    /// The protocol to try after this one is rejected.
    fn other(self) -> Self {
        match self {
            ApiFlavor::Responses => ApiFlavor::ChatCompletions,
            ApiFlavor::ChatCompletions => ApiFlavor::Responses,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ApiFlavor::Responses => "responses",
            ApiFlavor::ChatCompletions => "chat completions",
        }
    }
}

// Encoding of the cached flavor. `AtomicU8` rather than a lock because the
// value is written at most twice and read on every call.
const FLAVOR_UNKNOWN: u8 = 0;
const FLAVOR_RESPONSES: u8 = 1;
const FLAVOR_CHAT: u8 = 2;

fn encode(flavor: ApiFlavor) -> u8 {
    match flavor {
        ApiFlavor::Responses => FLAVOR_RESPONSES,
        ApiFlavor::ChatCompletions => FLAVOR_CHAT,
    }
}

fn decode(raw: u8) -> Option<ApiFlavor> {
    match raw {
        FLAVOR_RESPONSES => Some(ApiFlavor::Responses),
        FLAVOR_CHAT => Some(ApiFlavor::ChatCompletions),
        _ => None,
    }
}

/// Chat client for an OpenAI-compatible endpoint.
pub struct OpenAiChatClient {
    transport: Transport,
    model: String,
    /// Protocol discovered to work for this endpoint and model, or
    /// [`FLAVOR_UNKNOWN`] until the first call settles it.
    flavor: AtomicU8,
}

impl OpenAiChatClient {
    /// Build a client for `endpoint`, sending `model` on every request.
    pub fn new(endpoint: &Endpoint, model: impl Into<String>) -> Result<Self, OpenAiError> {
        Ok(Self::with_transport(Transport::new(endpoint)?, model))
    }

    /// Build on an existing [`Transport`] — how a provider with its own
    /// credentials and protocol headers reuses this client wholesale instead of
    /// reimplementing the request and streaming logic.
    pub fn with_transport(transport: Transport, model: impl Into<String>) -> Self {
        Self {
            transport,
            model: model.into(),
            flavor: AtomicU8::new(FLAVOR_UNKNOWN),
        }
    }

    /// Pin the protocol, skipping discovery. Worth doing when the server is
    /// known to speak only one — it saves the first call's failed probe.
    pub fn prefer(self, flavor: ApiFlavor) -> Self {
        self.flavor.store(encode(flavor), Ordering::Relaxed);
        self
    }

    pub fn transport(&self) -> &Transport {
        &self.transport
    }

    /// The model sent when a request does not override it.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The protocol in use, or `None` while it is still undiscovered.
    pub fn resolved_flavor(&self) -> Option<ApiFlavor> {
        decode(self.flavor.load(Ordering::Relaxed))
    }

    /// The protocol to try first: whatever last worked, else Responses.
    fn preferred_flavor(&self) -> ApiFlavor {
        self.resolved_flavor().unwrap_or(ApiFlavor::Responses)
    }

    fn remember(&self, flavor: ApiFlavor) {
        self.flavor.store(encode(flavor), Ordering::Relaxed);
    }

    /// Send `request`, falling back across protocols and schema constraints as
    /// needed, and return the assistant's text.
    async fn dispatch(
        &self,
        request: &ChatRequest,
        sink: TokenSink<'_>,
    ) -> Result<String, OpenAiError> {
        let first = self.preferred_flavor();

        match self.attempt(first, request, sink).await {
            Ok(text) => {
                self.remember(first);
                Ok(text)
            }
            Err(ProtocolError::WrongApi(original)) => {
                let second = first.other();
                debug!(
                    "{} rejected model '{}'; retrying on {}",
                    first.name(),
                    self.model_for(request),
                    second.name()
                );
                match self.attempt(second, request, sink).await {
                    Ok(text) => {
                        self.remember(second);
                        Ok(text)
                    }
                    Err(error) => {
                        // Both protocols refused. The second error is the one
                        // returned, but the first explains half the story.
                        warn!(
                            "{} also failed after {} rejected the request: {original}",
                            second.name(),
                            first.name()
                        );
                        Err(error.into_error())
                    }
                }
            }
            Err(error) => Err(error.into_error()),
        }
    }

    /// One protocol's attempt, including the unconstrained retry when the
    /// server cannot honor the requested schema.
    async fn attempt(
        &self,
        flavor: ApiFlavor,
        request: &ChatRequest,
        sink: TokenSink<'_>,
    ) -> Result<String, ProtocolError> {
        match self.execute(flavor, request, sink).await {
            Err(ProtocolError::SchemaUnsupported) if request.schema.is_some() => {
                warn!(
                    "{} rejected the response schema; retrying without structured output",
                    flavor.name()
                );
                // Falling back to free-form output leaves the caller to parse
                // best-effort JSON — better than failing the call outright.
                self.execute(flavor, &request.unconstrained(), sink).await
            }
            other => other,
        }
    }

    async fn execute(
        &self,
        flavor: ApiFlavor,
        request: &ChatRequest,
        sink: TokenSink<'_>,
    ) -> Result<String, ProtocolError> {
        let model = self.model_for(request);
        match flavor {
            ApiFlavor::Responses => responses::execute(&self.transport, model, request, sink).await,
            ApiFlavor::ChatCompletions => {
                chat_completions::execute(&self.transport, model, request, sink).await
            }
        }
    }

    /// The per-request model override, else the client's default.
    fn model_for<'a>(&'a self, request: &'a ChatRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model)
    }
}

#[async_trait]
impl ChatClient for OpenAiChatClient {
    async fn chat(&self, request: &ChatRequest) -> Result<String, OpenAiError> {
        self.dispatch(request, None).await
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
        tokens: UnboundedSender<String>,
    ) -> Result<String, OpenAiError> {
        self.dispatch(request, Some(&tokens)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ApiRoutes;

    fn client() -> OpenAiChatClient {
        OpenAiChatClient::new(&Endpoint::new("http://localhost:1234"), "default-model").unwrap()
    }

    #[test]
    fn defaults_to_the_responses_api() {
        let client = client();
        assert_eq!(
            client.resolved_flavor(),
            None,
            "undiscovered before any call"
        );
        assert_eq!(client.preferred_flavor(), ApiFlavor::Responses);
    }

    #[test]
    fn remembers_a_discovered_flavor() {
        // The whole point of caching: after one call settles the protocol, no
        // later request re-probes.
        let client = client();
        client.remember(ApiFlavor::ChatCompletions);
        assert_eq!(client.resolved_flavor(), Some(ApiFlavor::ChatCompletions));
        assert_eq!(client.preferred_flavor(), ApiFlavor::ChatCompletions);
    }

    #[test]
    fn pinning_skips_discovery() {
        let client = client().prefer(ApiFlavor::ChatCompletions);
        assert_eq!(client.preferred_flavor(), ApiFlavor::ChatCompletions);
    }

    #[test]
    fn flavors_alternate() {
        assert_eq!(ApiFlavor::Responses.other(), ApiFlavor::ChatCompletions);
        assert_eq!(ApiFlavor::ChatCompletions.other(), ApiFlavor::Responses);
    }

    #[test]
    fn request_model_overrides_the_client_default() {
        let client = client();
        let plain = ChatRequest::from_prompt("s", "u");
        assert_eq!(client.model_for(&plain), "default-model");

        let overridden = ChatRequest::from_prompt("s", "u").with_model("other-model");
        assert_eq!(client.model_for(&overridden), "other-model");
    }

    #[test]
    fn honors_a_custom_route_layout() {
        // Providers that serve the endpoints unversioned must reach the right
        // paths without any change to this client.
        let endpoint =
            Endpoint::new("https://api.example.com").with_routes(ApiRoutes::unversioned());
        let client = OpenAiChatClient::new(&endpoint, "m").unwrap();
        assert_eq!(
            client
                .transport()
                .url(&client.transport().routes().responses),
            "https://api.example.com/responses"
        );
        assert_eq!(
            client
                .transport()
                .url(&client.transport().routes().chat_completions),
            "https://api.example.com/chat/completions"
        );
    }
}
