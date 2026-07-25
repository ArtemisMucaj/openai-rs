//! Server-sent-events framing shared by both streaming APIs.
//!
//! Responses and Chat Completions both stream line-oriented SSE where the
//! payload rides on `data:` lines. Bytes arrive in arbitrary chunks that split
//! lines anywhere, so the decoder buffers until a newline is seen and only then
//! yields a complete payload.
//!
//! The buffer holds **raw bytes**, not text. A chunk boundary can land in the
//! middle of a multi-byte character, and decoding each chunk as it arrives
//! would replace that half-character with U+FFFD — permanently, since the
//! replacement is not undone when the rest of the character shows up. Decoding
//! is therefore deferred until a complete line is in hand.

/// Sentinel that ends a Chat Completions stream.
pub(crate) const DONE_SENTINEL: &str = "[DONE]";

/// Incremental `data:` payload extractor.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append a received chunk verbatim.
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Every complete `data:` payload buffered so far, in arrival order.
    ///
    /// Non-`data` lines (`event:`, `id:`, comments, blank separators) are
    /// dropped: both APIs repeat the event type inside the JSON payload, so
    /// nothing is lost by keying off the payload alone.
    pub(crate) fn drain(&mut self) -> Vec<String> {
        let mut payloads = Vec::new();

        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=newline).collect();
            line.pop(); // the '\n' itself
            if line.last() == Some(&b'\r') {
                line.pop();
            }

            // The line is complete, so any invalid UTF-8 in it is genuinely
            // invalid rather than an artifact of where the chunk boundary fell.
            let text = String::from_utf8_lossy(&line);
            let payload = text
                .strip_prefix("data: ")
                .or_else(|| text.strip_prefix("data:"));
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
    fn a_chunk_boundary_inside_a_character_does_not_corrupt_it() {
        // The boundary must fall *within* the two bytes of 'é' — splitting on
        // either side of it proves nothing, since both halves are then valid
        // UTF-8 on their own.
        let line = "data: café\n".as_bytes();
        let split = line.len() - 2; // between 0xC3 and 0xA9
        assert!(
            std::str::from_utf8(&line[..split]).is_err(),
            "this test is only meaningful if the first chunk is invalid UTF-8"
        );

        let mut decoder = SseDecoder::new();
        decoder.push(&line[..split]);
        assert!(decoder.drain().is_empty());
        decoder.push(&line[split..]);
        assert_eq!(decoder.drain(), vec!["café"]);
    }

    #[test]
    fn a_character_split_one_byte_at_a_time_survives() {
        // The pathological case: every byte arrives in its own chunk.
        let line = "data: 🎉 done\n".as_bytes();
        let mut decoder = SseDecoder::new();
        for byte in line {
            decoder.push(&[*byte]);
        }
        assert_eq!(decoder.drain(), vec!["🎉 done"]);
    }

    #[test]
    fn drain_is_idempotent_once_empty() {
        let mut decoder = SseDecoder::new();
        decoder.push(b"data: one\n");
        assert_eq!(decoder.drain(), vec!["one"]);
        assert!(decoder.drain().is_empty());
    }
}
