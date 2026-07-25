//! Server-sent-events framing shared by both streaming APIs.
//!
//! Responses and Chat Completions both stream line-oriented SSE where the
//! payload rides on `data:` lines. Bytes arrive in arbitrary chunks that split
//! lines anywhere, so the decoder buffers until a newline is seen and only then
//! yields a complete payload.

/// Sentinel that ends a Chat Completions stream.
pub(crate) const DONE_SENTINEL: &str = "[DONE]";

/// Incremental `data:` payload extractor.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    buffer: String,
}

impl SseDecoder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append a received chunk. Invalid UTF-8 is replaced rather than rejected:
    /// a chunk boundary can split a multi-byte character, and the replacement
    /// resolves once the rest arrives.
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
    }

    /// Every complete `data:` payload buffered so far, in arrival order.
    ///
    /// Non-`data` lines (`event:`, `id:`, comments, blank separators) are
    /// dropped: both APIs repeat the event type inside the JSON payload, so
    /// nothing is lost by keying off the payload alone.
    pub(crate) fn drain(&mut self) -> Vec<String> {
        let mut payloads = Vec::new();
        while let Some(newline) = self.buffer.find('\n') {
            let line = self.buffer[..newline].trim_end_matches('\r').to_string();
            self.buffer.drain(..=newline);

            let payload = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"));
            if let Some(payload) = payload {
                payloads.push(payload.trim().to_string());
            }
        }
        payloads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_data_lines_and_ignores_the_rest() {
        let mut decoder = SseDecoder::new();
        decoder.push(b"event: message\ndata: {\"a\":1}\n\n: comment\ndata: {\"a\":2}\n");
        assert_eq!(decoder.drain(), vec!["{\"a\":1}", "{\"a\":2}"]);
    }

    #[test]
    fn buffers_across_chunk_boundaries() {
        // The transport splits wherever it likes; a payload must not be emitted
        // until its terminating newline arrives.
        let mut decoder = SseDecoder::new();
        decoder.push(b"data: {\"par");
        assert!(decoder.drain().is_empty(), "incomplete line must not emit");
        decoder.push(b"tial\":true}\n");
        assert_eq!(decoder.drain(), vec!["{\"partial\":true}"]);
    }

    #[test]
    fn accepts_both_newline_conventions_and_the_spaceless_prefix() {
        let mut decoder = SseDecoder::new();
        decoder.push(b"data: crlf\r\ndata:no-space\n");
        assert_eq!(decoder.drain(), vec!["crlf", "no-space"]);
    }

    #[test]
    fn survives_a_split_multibyte_character() {
        // "é" is two bytes; splitting it across chunks must not corrupt the
        // rest of the stream.
        let mut decoder = SseDecoder::new();
        let text = "data: caf\u{e9}\n".as_bytes();
        let split = text.len() - 3;
        decoder.push(&text[..split]);
        decoder.push(&text[split..]);
        assert_eq!(decoder.drain(), vec!["caf\u{e9}"]);
    }

    #[test]
    fn drain_is_idempotent_once_empty() {
        let mut decoder = SseDecoder::new();
        decoder.push(b"data: one\n");
        assert_eq!(decoder.drain(), vec!["one"]);
        assert!(decoder.drain().is_empty());
    }
}
