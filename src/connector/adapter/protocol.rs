//! The contract between the two wire protocols and the client that arbitrates
//! between them.
//!
//! Each protocol module reports failures it cannot resolve locally, and the
//! chat client decides what to do: retry unconstrained, switch APIs, or give up.

use tokio::sync::mpsc::UnboundedSender;

use crate::domain::OpenAiError;

/// Where a streaming call sends its tokens, or `None` for a buffered call.
///
/// Both protocols implement one entry point covering streaming and
/// non-streaming, since the request bodies differ only by a `stream` flag.
pub(crate) type TokenSink<'a> = Option<&'a UnboundedSender<String>>;

/// A protocol-level failure, classified by what the caller can do about it.
#[derive(Debug)]
pub(crate) enum ProtocolError {
    /// This model is not served by this API — the other one should be tried.
    /// Carries the originating error so a failure on both can name both.
    ///
    /// Only ever returned *before* any token is emitted, which is what makes
    /// retrying a stream on the other API safe.
    WrongApi(OpenAiError),

    /// The server rejected the structured-output constraint. Retrying the same
    /// API without the schema is expected to succeed.
    SchemaUnsupported,

    /// Anything else — propagate it.
    Fatal(OpenAiError),
}

impl ProtocolError {
    /// Collapse into the error to hand the caller.
    ///
    /// The non-`Fatal` variants are signals meant to be intercepted; one
    /// arriving here means no retry path applied, so it is reported plainly
    /// rather than silently swallowed.
    pub(crate) fn into_error(self) -> OpenAiError {
        match self {
            ProtocolError::Fatal(error) => error,
            ProtocolError::WrongApi(error) => error,
            ProtocolError::SchemaUnsupported => OpenAiError::configuration(
                "the server rejected the response schema and no unconstrained retry was possible",
            ),
        }
    }

    pub(crate) fn fatal(error: OpenAiError) -> Self {
        ProtocolError::Fatal(error)
    }
}

/// Error code both APIs return when a model is reachable only via the other
/// one. GitHub Copilot emits it from each endpoint for models belonging to the
/// other, which makes it the definitive switch-APIs signal.
const WRONG_API_CODE: &str = "unsupported_api_for_model";

/// Whether an error body carries the switch-APIs signal.
pub(crate) fn is_wrong_api(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")?
                .get("code")?
                .as_str()
                .map(|code| code == WRONG_API_CODE)
        })
        .unwrap_or(false)
}

/// Whether a status means the server does not implement this endpoint at all.
///
/// Servers predating the Responses API answer `404`/`405` there. That is also
/// a switch-APIs signal, just from a server that never heard of the endpoint
/// rather than a model routed to the other one.
pub(crate) fn is_endpoint_absent(status: u16) -> bool {
    matches!(status, 404 | 405 | 501)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_the_switch_apis_signal() {
        let from_chat = r#"{"error":{"message":"model is not accessible via the /chat/completions endpoint","code":"unsupported_api_for_model"}}"#;
        let from_responses = r#"{"error":{"message":"model does not support Responses API.","code":"unsupported_api_for_model"}}"#;
        assert!(is_wrong_api(from_chat));
        assert!(is_wrong_api(from_responses));
    }

    #[test]
    fn other_failures_do_not_trigger_a_pointless_switch() {
        assert!(!is_wrong_api(
            r#"{"error":{"code":"invalid_request_error"}}"#
        ));
        assert!(!is_wrong_api(r#"{"error":{"message":"no code field"}}"#));
        assert!(!is_wrong_api("not json at all"));
        assert!(!is_wrong_api(""));
    }

    #[test]
    fn treats_missing_endpoints_as_a_switch_signal() {
        assert!(is_endpoint_absent(404));
        assert!(is_endpoint_absent(405));
        assert!(is_endpoint_absent(501));
        // A real rejection of a valid endpoint must not be mistaken for one.
        assert!(!is_endpoint_absent(400));
        assert!(!is_endpoint_absent(403));
        assert!(!is_endpoint_absent(500));
    }
}
