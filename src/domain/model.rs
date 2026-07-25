use serde::{Deserialize, Serialize};

/// What a model will accept and produce.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_window_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
}

impl ModelLimits {
    /// Whether either limit is known — an all-`None` value carries no
    /// information and is better represented as an absent `limits`.
    pub fn is_empty(&self) -> bool {
        self.max_context_window_tokens.is_none() && self.max_output_tokens.is_none()
    }
}

/// A model a provider offers, described uniformly across providers.
///
/// `GET /v1/models` on a plain OpenAI-compatible server only reports an id (and
/// sometimes an owner), so most fields will be `None` there. Providers with
/// richer catalogs fill in the rest, which lets a host render one model picker
/// regardless of which backend produced the list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    /// Id to send in completion requests.
    pub id: String,
    /// Human-facing name, when the provider supplies one distinct from the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Vendor or owning organisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    /// Whether the provider flags this model as a preview.
    #[serde(default)]
    pub preview: bool,
    /// Context/output limits, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<ModelLimits>,
}

impl Model {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            vendor: None,
            preview: false,
            limits: None,
        }
    }

    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }

    pub fn with_vendor(mut self, vendor: impl Into<String>) -> Self {
        self.vendor = Some(vendor.into());
        self
    }

    pub fn with_preview(mut self, preview: bool) -> Self {
        self.preview = preview;
        self
    }

    /// Attach limits, dropping an all-`None` value so "unknown" stays absent
    /// rather than becoming an empty object.
    pub fn with_limits(mut self, limits: ModelLimits) -> Self {
        self.limits = (!limits.is_empty()).then_some(limits);
        self
    }

    /// The label to show a user: the display name when there is one, else the id.
    pub fn label(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_falls_back_to_the_id() {
        assert_eq!(Model::new("gpt-x").label(), "gpt-x");
        assert_eq!(
            Model::new("gpt-x").with_display_name("GPT X").label(),
            "GPT X"
        );
    }

    #[test]
    fn empty_limits_are_dropped() {
        // "No limits reported" and "limits reported as nothing" must not be
        // rendered differently by a picker.
        let model = Model::new("m").with_limits(ModelLimits::default());
        assert_eq!(model.limits, None);

        let model = Model::new("m").with_limits(ModelLimits {
            max_context_window_tokens: Some(128_000),
            max_output_tokens: None,
        });
        assert_eq!(
            model.limits.and_then(|l| l.max_context_window_tokens),
            Some(128_000)
        );
    }
}
