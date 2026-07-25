//! The HTTP carrier shared by every adapter in this crate.
//!
//! One [`Transport`] owns a `reqwest::Client` whose default headers already
//! carry authentication and any provider-specific protocol headers, plus the
//! base URL and route layout. Chat, model discovery, and embeddings all borrow
//! the same instance, so a provider is configured once and every capability
//! inherits it.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::warn;

use crate::domain::{ApiRoutes, Endpoint, OpenAiError};

/// Cheap to clone: `reqwest::Client` is internally reference-counted, so clones
/// share one connection pool.
#[derive(Debug, Clone)]
pub struct Transport {
    http: reqwest::Client,
    base_url: String,
    routes: ApiRoutes,
}

impl Transport {
    /// Build a transport for `endpoint`, baking its credentials and headers
    /// into the client's defaults.
    pub fn new(endpoint: &Endpoint) -> Result<Self, OpenAiError> {
        let mut headers = HeaderMap::new();

        if let Some(key) = endpoint.api_key() {
            match HeaderValue::from_str(&format!("Bearer {key}")) {
                Ok(mut value) => {
                    // Keeps the credential out of `Debug` output of the client
                    // and of any error that echoes the header map.
                    value.set_sensitive(true);
                    headers.insert(reqwest::header::AUTHORIZATION, value);
                }
                Err(e) => {
                    return Err(OpenAiError::configuration(format!(
                        "API key {} is not valid header content: {e}",
                        mask_secret(key)
                    )));
                }
            }
        }

        for (name, value) in endpoint.headers() {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                OpenAiError::configuration(format!("invalid header name '{name}': {e}"))
            })?;
            let value = HeaderValue::from_str(value).map_err(|e| {
                OpenAiError::configuration(format!("invalid value for header '{name}': {e}"))
            })?;
            headers.insert(name, value);
        }

        let http = reqwest::Client::builder()
            .timeout(endpoint.timeout())
            .default_headers(headers)
            .build()
            .map_err(|e| OpenAiError::configuration(format!("failed to build HTTP client: {e}")))?;

        Ok(Self {
            http,
            base_url: endpoint.base_url().to_string(),
            routes: endpoint.routes().clone(),
        })
    }

    /// Build a transport around an already-configured HTTP client — the escape
    /// hatch for hosts that need to share a connection pool, a proxy, or a
    /// custom TLS setup. `http` must already carry any required auth headers.
    pub fn with_http_client(
        http: reqwest::Client,
        base_url: impl Into<String>,
        routes: ApiRoutes,
    ) -> Self {
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            routes,
        }
    }

    /// The underlying client, for providers that need to call endpoints outside
    /// this crate's vocabulary with the same credentials.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn routes(&self) -> &ApiRoutes {
        &self.routes
    }

    /// Absolute URL for a route path.
    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// POST `body` to `path` and hand back the raw response without inspecting
    /// its status — for callers that must interpret failures themselves (the
    /// chat client reads the error payload to decide whether to switch APIs).
    pub async fn post<B: Serialize + ?Sized>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<reqwest::Response, OpenAiError> {
        let url = self.url(path);
        self.http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| OpenAiError::transport(format!("POST {url} failed: {e}")))
    }

    /// POST `body` to `path` and decode a successful JSON response, turning any
    /// non-success status into [`OpenAiError::Api`].
    pub async fn post_json<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, OpenAiError> {
        let response = self.post(path, body).await?;
        decode_success(response).await
    }

    /// GET `path` and decode a successful JSON response.
    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, OpenAiError> {
        let url = self.url(path);
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| OpenAiError::transport(format!("GET {url} failed: {e}")))?;
        decode_success(response).await
    }
}

/// Turn a response into either decoded JSON or an [`OpenAiError::Api`].
async fn decode_success<T: DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, OpenAiError> {
    let status = response.status();
    if !status.is_success() {
        return Err(OpenAiError::api(status.as_u16(), read_body(response).await));
    }
    let body = response
        .text()
        .await
        .map_err(|e| OpenAiError::transport(format!("failed to read response body: {e}")))?;
    serde_json::from_str(&body)
        .map_err(|e| OpenAiError::decode(format!("{e}; body was: {}", truncate(&body))))
}

/// Read an error response's body, substituting a placeholder if even that
/// fails — a failure to read the explanation must not mask the failure itself.
pub(crate) async fn read_body(response: reqwest::Response) -> String {
    match response.text().await {
        Ok(text) => text,
        Err(e) => {
            warn!("failed to read error-response body: {e}");
            format!("<failed to read body: {e}>")
        }
    }
}

/// Cap on how much of a response body is quoted back in a decode error.
const MAX_BODY_SNIPPET_CHARS: usize = 500;

/// Shorten `body` for inclusion in an error message, respecting char
/// boundaries so multi-byte text never panics the slice.
fn truncate(body: &str) -> String {
    match body.char_indices().nth(MAX_BODY_SNIPPET_CHARS) {
        Some((index, _)) => format!("{}… ({} bytes total)", &body[..index], body.len()),
        None => body.to_string(),
    }
}

/// Mask a credential for logging: first and last four characters survive.
pub(crate) fn mask_secret(secret: &str) -> String {
    let chars: Vec<char> = secret.chars().collect();
    if chars.len() <= 8 {
        return "*".repeat(chars.len());
    }
    let prefix: String = chars[..4].iter().collect();
    let suffix: String = chars[chars.len() - 4..].iter().collect();
    format!("{}{}{}", prefix, "*".repeat(chars.len() - 8), suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_all_but_the_edges() {
        assert_eq!(mask_secret("sk-abcdefghijkl"), "sk-a*******ijkl");
        // Short secrets reveal nothing at all.
        assert_eq!(mask_secret("short"), "*****");
        assert_eq!(mask_secret(""), "");
    }

    #[test]
    fn builds_urls_from_the_base_and_route() {
        let endpoint = Endpoint::new("http://localhost:1234/");
        let transport = Transport::new(&endpoint).unwrap();
        assert_eq!(
            transport.url("/v1/responses"),
            "http://localhost:1234/v1/responses"
        );
    }

    #[test]
    fn rejects_an_unusable_api_key_without_leaking_it() {
        // A newline cannot go in a header value; the error must name the
        // problem without printing the whole credential.
        let endpoint = Endpoint::new("http://x").with_api_key("bad\nkey-with-more-text");
        let error = Transport::new(&endpoint).unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, OpenAiError::Configuration(_)));
        assert!(
            !message.contains("bad\nkey-with-more-text"),
            "got: {message}"
        );
    }

    #[test]
    fn rejects_invalid_header_names() {
        let endpoint = Endpoint::new("http://x").with_header("Bad Header", "v");
        assert!(matches!(
            Transport::new(&endpoint),
            Err(OpenAiError::Configuration(_))
        ));
    }

    #[test]
    fn truncates_long_bodies_on_char_boundaries() {
        // Multi-byte characters must not be sliced mid-codepoint.
        let body = "é".repeat(MAX_BODY_SNIPPET_CHARS + 50);
        let truncated = truncate(&body);
        assert!(truncated.contains("bytes total"));
        assert!(truncated.chars().count() < body.chars().count());

        let short = "fine";
        assert_eq!(truncate(short), "fine");
    }
}
