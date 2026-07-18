//! Pinned LiteLLM 1.91.2 provider, pricing, limit, and retry metadata (D16).
//!
//! Both JSON assets are checked in. The smaller runtime manifest records the
//! native verification state; the source inventory owns all 2,908 model rows.

use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{Map, Value};

const PROVIDER_MANIFEST_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../genai-inventory/provider_manifest.json"
));
const PINNED_PROVIDER_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../genai-inventory/providers.json"
));

#[derive(Debug, Deserialize)]
struct RuntimeManifest {
    providers: Vec<RuntimeProvider>,
}

#[derive(Debug, Deserialize)]
struct RuntimeProvider {
    name: String,
    adapter: String,
    capabilities: ProviderCapabilities,
    verification: String,
    unsupported: bool,
}

#[derive(Debug, Deserialize)]
struct ProviderCapabilities {
    cost: bool,
}

#[derive(Debug, Deserialize)]
struct PinnedManifest {
    models: Vec<PinnedModel>,
}

#[derive(Debug, Deserialize)]
struct PinnedModel {
    model_key: String,
    provider: String,
    limits: Map<String, Value>,
    prices: Map<String, Value>,
    tokenizer_metadata: Map<String, Value>,
}

fn runtime_manifest() -> &'static RuntimeManifest {
    static MANIFEST: OnceLock<RuntimeManifest> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        serde_json::from_str(PROVIDER_MANIFEST_JSON)
            .expect("checked-in provider_manifest.json must be valid")
    })
}

fn pinned_manifest() -> &'static PinnedManifest {
    static MANIFEST: OnceLock<PinnedManifest> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        serde_json::from_str(PINNED_PROVIDER_JSON).expect("checked-in providers.json must be valid")
    })
}

pub fn is_supported_provider(provider: &str) -> bool {
    let provider = normalize_provider(provider);
    runtime_manifest()
        .providers
        .iter()
        .any(|item| item.name == provider && !item.unsupported)
}

pub fn supported_provider_names() -> Vec<&'static str> {
    runtime_manifest()
        .providers
        .iter()
        .filter(|provider| !provider.unsupported)
        .map(|provider| provider.name.as_str())
        .collect()
}

pub fn provider_adapter_kind(provider: &str) -> Option<&'static str> {
    let provider = normalize_provider(provider);
    runtime_manifest()
        .providers
        .iter()
        .find(|item| normalize_provider(&item.name) == provider && !item.unsupported)
        .map(|item| item.adapter.as_str())
}

pub fn provider_verification(provider: &str) -> Option<&'static str> {
    let provider = normalize_provider(provider);
    runtime_manifest()
        .providers
        .iter()
        .find(|item| normalize_provider(&item.name) == provider && !item.unsupported)
        .map(|item| item.verification.as_str())
}

pub fn provider_has_cost_accounting(provider: &str) -> bool {
    let provider = normalize_provider(provider);
    runtime_manifest().providers.iter().any(|item| {
        normalize_provider(&item.name) == provider && !item.unsupported && item.capabilities.cost
    })
}

pub fn normalize_provider(provider: &str) -> &str {
    match provider {
        "amazon-bedrock" => "bedrock",
        "databricks-model-serving" => "databricks",
        "azure-openai" => "azure",
        value => value,
    }
}

pub fn default_api_base(provider: &str) -> Option<&'static str> {
    match normalize_provider(provider) {
        "openai" => Some("https://api.openai.com/v1"),
        "anthropic" => Some("https://api.anthropic.com/v1"),
        "gemini" => Some("https://generativelanguage.googleapis.com/v1beta/models"),
        "groq" => Some("https://api.groq.com/openai/v1"),
        "deepseek" => Some("https://api.deepseek.com/v1"),
        "xai" => Some("https://api.x.ai/v1"),
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "ollama" => Some("http://localhost:11434/v1"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    AuthenticationError,
    Timeout,
    RateLimitError,
    ContentPolicyViolationError,
    BadRequestError,
}

/// Match the five exception buckets consumed by LiteLLM's pinned
/// `get_num_retries_from_retry_policy` implementation.
pub fn classify_retry(status: u16, body: &Value) -> Option<RetryClass> {
    let code = body
        .pointer("/error/code")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let kind = body
        .pointer("/error/type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let message = body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match status {
        401 | 403 => Some(RetryClass::AuthenticationError),
        408 | 504 => Some(RetryClass::Timeout),
        429 => Some(RetryClass::RateLimitError),
        400 if code.contains("content_policy")
            || kind.contains("content_policy")
            || message.contains("content policy") =>
        {
            Some(RetryClass::ContentPolicyViolationError)
        }
        400 | 404 | 409 | 422 => Some(RetryClass::BadRequestError),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelAccounting {
    pub limits: Map<String, Value>,
    pub prices: Map<String, Value>,
    pub tokenizer_metadata: Map<String, Value>,
}

pub fn model_accounting(provider: &str, model: &str) -> Option<ModelAccounting> {
    let provider = normalize_provider(provider);
    let qualified = format!("{provider}/{model}");
    pinned_manifest()
        .models
        .iter()
        .find(|entry| {
            normalize_provider(&entry.provider) == provider
                && (entry.model_key == model
                    || entry.model_key == qualified
                    || entry.model_key.ends_with(&format!("/{qualified}")))
        })
        .map(|entry| ModelAccounting {
            limits: entry.limits.clone(),
            prices: entry.prices.clone(),
            tokenizer_metadata: entry.tokenizer_metadata.clone(),
        })
}

/// Token cost from the pinned table's ordinary input/output token rates.
/// Specialized image/audio/search fields remain available via
/// [`model_accounting`] and are not guessed from chat usage.
pub fn token_cost(
    provider: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    let accounting = model_accounting(provider, model)?;
    let input = accounting
        .prices
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let output = accounting
        .prices
        .get("output_cost_per_token")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    ((input != 0.0) || (output != 0.0))
        .then_some(input * input_tokens as f64 + output * output_tokens as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_full_pinned_coverage_and_no_unsupported_entries() {
        let runtime = runtime_manifest();
        assert_eq!(runtime.providers.len(), 191);
        assert!(runtime
            .providers
            .iter()
            .all(|provider| !provider.unsupported));
        assert_eq!(pinned_manifest().models.len(), 2_908);
        assert!(runtime
            .providers
            .iter()
            .all(|provider| is_supported_provider(&provider.name)));
        assert!(runtime.providers.iter().all(|provider| matches!(
            provider.verification.as_str(),
            "differential_fixture" | "hermetic_matrix"
        )));

        for provider in &runtime.providers {
            if provider.capabilities.cost {
                assert!(
                    pinned_manifest().models.iter().any(|model| {
                        normalize_provider(&model.provider) == normalize_provider(&provider.name)
                            && !model.prices.is_empty()
                    }),
                    "{} advertises cost accounting without a pinned price row",
                    provider.name
                );
            }
        }
    }

    #[test]
    fn retry_buckets_match_pinned_reference_classes() {
        assert_eq!(
            classify_retry(401, &serde_json::json!({})),
            Some(RetryClass::AuthenticationError)
        );
        assert_eq!(
            classify_retry(429, &serde_json::json!({})),
            Some(RetryClass::RateLimitError)
        );
        assert_eq!(
            classify_retry(
                400,
                &serde_json::json!({"error":{"code":"content_policy_violation"}})
            ),
            Some(RetryClass::ContentPolicyViolationError)
        );
        assert_eq!(
            classify_retry(422, &serde_json::json!({})),
            Some(RetryClass::BadRequestError)
        );
        assert_eq!(classify_retry(500, &serde_json::json!({})), None);
    }

    #[test]
    fn price_and_limit_rows_are_available_without_network_access() {
        let entry = model_accounting("openai", "gpt-4").expect("gpt-4 pinned row");
        assert!(entry.limits.contains_key("max_input_tokens"));
        assert!(entry.prices.contains_key("input_cost_per_token"));
        assert!(token_cost("openai", "gpt-4", 10, 5).is_some());
    }
}
