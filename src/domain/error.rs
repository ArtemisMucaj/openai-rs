use thiserror::Error;

/// Every failure this crate can surface.
///
/// Deliberately free of `reqwest` types: the domain layer stays independent of
/// the HTTP client the connector layer happens to use, so swapping transports
/// never ripples into callers' `match` arms.
#[derive(Debug, Error)]
pub enum OpenAiError {
    /// The endpoint could not be built — a malformed base URL, an API key that
    /// is not valid header content, an unbuildable HTTP client.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// The request never produced an HTTP response: DNS failure, connection
    /// refused, TLS error, timeout, or a mid-stream read error.
    #[error("transport error: {0}")]
    Transport(String),

    /// The server answered with a non-success status. `body` is the raw
    /// response text, which for OpenAI-compatible servers carries the
    /// `{"error": {...}}` payload explaining the rejection.
    #[error("API returned {status}: {body}")]
    Api { status: u16, body: String },

    /// The response arrived but did not match the expected shape.
    #[error("failed to decode response: {0}")]
    Decode(String),

    /// The call succeeded but the model produced no usable text. Distinct from
    /// [`Self::Api`] because the transport and the request were both fine.
    #[error("the model returned an empty response")]
    EmptyResponse,
}

impl OpenAiError {
    pub fn configuration(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }

    pub fn transport(msg: impl Into<String>) -> Self {
        Self::Transport(msg.into())
    }

    pub fn decode(msg: impl Into<String>) -> Self {
        Self::Decode(msg.into())
    }

    pub fn api(status: u16, body: impl Into<String>) -> Self {
        Self::Api {
            status,
            body: body.into(),
        }
    }

    /// The HTTP status, when this error came from a server response.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Whether this is a 4xx server rejection — the class of error that means
    /// "the request was wrong" rather than "the server broke".
    pub fn is_client_error(&self) -> bool {
        matches!(self.status(), Some(s) if (400..500).contains(&s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_client_errors() {
        assert!(OpenAiError::api(400, "bad request").is_client_error());
        assert!(OpenAiError::api(403, "forbidden").is_client_error());
        assert!(!OpenAiError::api(500, "boom").is_client_error());
        assert!(!OpenAiError::transport("refused").is_client_error());
    }

    #[test]
    fn exposes_status_only_for_api_errors() {
        assert_eq!(OpenAiError::api(404, "nope").status(), Some(404));
        assert_eq!(OpenAiError::decode("garbage").status(), None);
    }
}
