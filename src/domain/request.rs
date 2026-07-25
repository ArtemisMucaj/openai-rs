use serde::{Deserialize, Serialize};

use super::message::Message;

/// A JSON Schema the model's output must conform to.
///
/// Both supported APIs can grammar-constrain decoding to a schema — the
/// Responses API via `text.format`, Chat Completions via `response_format`. That
/// is far more robust than parsing free-form output, especially with small local
/// models. Servers that cannot honor a given schema are handled by the client,
/// which retries unconstrained rather than failing the call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Short identifier for the schema; surfaced to the server for diagnostics.
    pub name: String,
    /// Reject any output that does not match the schema exactly.
    pub strict: bool,
    /// The JSON Schema object itself.
    pub schema: serde_json::Value,
}

impl JsonSchema {
    /// A strict schema — the server must reject non-conforming output.
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            strict: true,
            schema,
        }
    }

    /// Relax the strictness flag, for servers whose strict mode is too
    /// restrictive for the schema at hand.
    pub fn lenient(mut self) -> Self {
        self.strict = false;
        self
    }
}

/// Everything needed to ask a model for one completion.
///
/// Built once and passed by reference so a retry (schema fallback, API switch)
/// never re-allocates the prompt.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatRequest {
    /// The conversation, in order.
    pub messages: Vec<Message>,
    /// Model id to use for this call, overriding the client's default. `None`
    /// uses whatever the client was constructed with.
    pub model: Option<String>,
    /// Sampling temperature. Left unset by default: reasoning models reject an
    /// explicit temperature, so callers that need determinism opt in.
    pub temperature: Option<f32>,
    /// Cap on generated tokens. Mapped to each API's own field name.
    pub max_output_tokens: Option<u32>,
    /// Structured-output constraint, when the caller needs parseable JSON.
    pub schema: Option<JsonSchema>,
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    /// The common shape: a system prompt framing a single user turn.
    pub fn from_prompt(system: &str, user: &str) -> Self {
        Self::new(vec![Message::system(system), Message::user(user)])
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn with_schema(mut self, schema: JsonSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// A copy with the structured-output constraint dropped — the retry the
    /// client makes when a server rejects the schema.
    pub(crate) fn unconstrained(&self) -> Self {
        Self {
            schema: None,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    #[test]
    fn from_prompt_builds_system_then_user() {
        let request = ChatRequest::from_prompt("be terse", "hello");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, Role::System);
        assert_eq!(request.messages[0].content, "be terse");
        assert_eq!(request.messages[1].role, Role::User);
        assert_eq!(request.messages[1].content, "hello");
    }

    #[test]
    fn defaults_leave_sampling_knobs_unset() {
        // Reasoning models reject an explicit temperature, so we must not send
        // one unless the caller asked for it.
        let request = ChatRequest::from_prompt("s", "u");
        assert_eq!(request.temperature, None);
        assert_eq!(request.max_output_tokens, None);
        assert_eq!(request.schema, None);
        assert_eq!(request.model, None);
    }

    #[test]
    fn unconstrained_drops_only_the_schema() {
        let request = ChatRequest::from_prompt("s", "u")
            .with_temperature(0.0)
            .with_model("m")
            .with_schema(JsonSchema::new("out", serde_json::json!({})));

        let relaxed = request.unconstrained();
        assert_eq!(relaxed.schema, None);
        assert_eq!(relaxed.temperature, Some(0.0));
        assert_eq!(relaxed.model.as_deref(), Some("m"));
        assert_eq!(relaxed.messages, request.messages);
    }

    #[test]
    fn schema_defaults_to_strict() {
        let schema = JsonSchema::new("out", serde_json::json!({"type": "object"}));
        assert!(schema.strict);
        assert!(!schema.lenient().strict);
    }
}
