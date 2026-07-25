use std::time::Duration;

/// Paths of the four endpoints this crate speaks, relative to the base URL.
///
/// Hosted OpenAI and most compatible servers (LM Studio, vLLM, …) put these
/// under `/v1`; some gateways serve them unversioned at the root. Making the
/// routes data rather than constants is what lets a provider with a different
/// layout — GitHub Copilot, for one — reuse this crate unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiRoutes {
    /// The Responses API — this crate's primary path.
    pub responses: String,
    /// Chat Completions — the fallback for models not served by Responses.
    pub chat_completions: String,
    /// Model discovery.
    pub models: String,
    /// Embeddings.
    pub embeddings: String,
}

impl ApiRoutes {
    /// The `/v1`-prefixed layout used by hosted OpenAI and compatible servers.
    pub fn versioned() -> Self {
        Self {
            responses: "/v1/responses".to_string(),
            chat_completions: "/v1/chat/completions".to_string(),
            models: "/v1/models".to_string(),
            embeddings: "/v1/embeddings".to_string(),
        }
    }

    /// The unprefixed layout, where the endpoints sit at the root.
    pub fn unversioned() -> Self {
        Self {
            responses: "/responses".to_string(),
            chat_completions: "/chat/completions".to_string(),
            models: "/models".to_string(),
            embeddings: "/embeddings".to_string(),
        }
    }
}

impl Default for ApiRoutes {
    fn default() -> Self {
        Self::versioned()
    }
}

/// Default per-request timeout. Generous, because a large prompt against a
/// local model on CPU can legitimately take minutes.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Where to talk, and with what credentials.
///
/// A pure value: constructing one performs no I/O and reads no environment.
/// Resolving configuration (env vars, a config file, a keychain) is the host's
/// job; this crate only accepts the resolved result.
#[derive(Debug, Clone)]
pub struct Endpoint {
    base_url: String,
    api_key: Option<String>,
    timeout: Duration,
    headers: Vec<(String, String)>,
    routes: ApiRoutes,
}

impl Endpoint {
    /// A new endpoint at `base_url` (with or without a trailing slash; it is
    /// normalised away) using the `/v1` route layout and no credentials.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        Self {
            base_url,
            api_key: None,
            timeout: DEFAULT_TIMEOUT,
            headers: Vec::new(),
            routes: ApiRoutes::default(),
        }
    }

    /// Bearer credential sent on every request. Empty keys are treated as
    /// absent, so a host can pass through an unset config value directly.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        self.api_key = (!api_key.is_empty()).then_some(api_key);
        self
    }

    /// Bearer credential from an optional value, for the same reason.
    pub fn with_optional_api_key(self, api_key: Option<impl Into<String>>) -> Self {
        match api_key {
            Some(key) => self.with_api_key(key),
            None => self,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// An extra header sent on every request. Providers that gate access behind
    /// protocol headers (client identity, API version) declare them here rather
    /// than reaching for the HTTP client directly.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Override the route layout — see [`ApiRoutes`].
    pub fn with_routes(mut self, routes: ApiRoutes) -> Self {
        self.routes = routes;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn routes(&self) -> &ApiRoutes {
        &self.routes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_trailing_slashes_in_the_base_url() {
        assert_eq!(
            Endpoint::new("http://localhost:1234/").base_url(),
            "http://localhost:1234"
        );
        assert_eq!(
            Endpoint::new("http://localhost:1234").base_url(),
            "http://localhost:1234"
        );
    }

    #[test]
    fn empty_api_keys_are_treated_as_absent() {
        // Hosts pass unset config through verbatim; an empty string must not
        // become an `Authorization: Bearer ` header.
        assert_eq!(Endpoint::new("http://x").with_api_key("").api_key(), None);
        assert_eq!(
            Endpoint::new("http://x").with_api_key("k").api_key(),
            Some("k")
        );
        assert_eq!(
            Endpoint::new("http://x")
                .with_optional_api_key(None::<String>)
                .api_key(),
            None
        );
    }

    #[test]
    fn route_layouts_differ_only_in_the_version_prefix() {
        let versioned = ApiRoutes::versioned();
        assert_eq!(versioned.responses, "/v1/responses");
        assert_eq!(versioned.chat_completions, "/v1/chat/completions");

        let unversioned = ApiRoutes::unversioned();
        assert_eq!(unversioned.responses, "/responses");
        assert_eq!(unversioned.chat_completions, "/chat/completions");
    }

    #[test]
    fn headers_accumulate_in_order() {
        let endpoint = Endpoint::new("http://x")
            .with_header("A", "1")
            .with_header("B", "2");
        assert_eq!(
            endpoint.headers(),
            [
                ("A".to_string(), "1".to_string()),
                ("B".to_string(), "2".to_string())
            ]
        );
    }
}
