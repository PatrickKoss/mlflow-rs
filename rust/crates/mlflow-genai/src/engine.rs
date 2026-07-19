use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::{JobKind, ScorerPayloadError, SerializedScorer};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvalItem {
    pub inputs: Option<Value>,
    pub outputs: Option<Value>,
    pub expectations: Option<Value>,
    pub trace: Option<Value>,
    pub session: Option<Vec<Value>>,
    /// Preloaded examples are the worker/store seam used by MemoryAugmentedJudge.
    pub memory_examples: Option<Vec<MemoryExample>>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct MemoryExample {
    pub trace_id: String,
    #[serde(default)]
    pub inputs: Option<Value>,
    #[serde(default)]
    pub outputs: Option<Value>,
    #[serde(default)]
    pub expectations: Option<Value>,
    #[serde(default)]
    pub trace: Option<Value>,
    #[serde(default)]
    pub feedback: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AssessmentSource {
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

/// The Python `Feedback` fields produced by scorer execution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Feedback {
    pub name: String,
    pub value: Value,
    pub rationale: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<AssessmentSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

impl Feedback {
    pub(crate) fn code(name: &str, value: Value, rationale: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            value,
            rationale: rationale.into(),
            source: None,
            metadata: None,
            span_id: None,
            trace_id: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    ScorerPayload(#[from] ScorerPayloadError),
    #[error("invalid invoke_scorer parameters: {0}")]
    InvalidParams(String),
    #[error("job kind {0} is not implemented by this native worker phase")]
    UnsupportedJobKind(JobKind),
    #[error("scorer form is recognized but belongs to the T19.3 third-party executor")]
    UnsupportedScorer,
    #[error("builtin scorer class {0:?} is not implemented")]
    UnsupportedBuiltin(String),
    #[error("missing or invalid scorer field {0:?}")]
    InvalidScorerField(&'static str),
    #[error("judge execution requires a gateway URL")]
    MissingGatewayUrl,
    #[error("memory judge execution requires an embedding URL")]
    MissingEmbeddingUrl,
    #[error("gateway request failed: {0}")]
    Gateway(String),
    #[error("embedding request failed: {0}")]
    Embedding(String),
    #[error("gateway returned malformed completion: {0}")]
    MalformedGatewayResponse(String),
    #[error("judge tool execution failed: {0}")]
    Tool(String),
    #[error("result serialization failed: {0}")]
    Serialization(String),
}

/// Shared native scorer/judge execution surface used by workers and inline guardrails.
#[derive(Clone)]
pub struct ScorerExecutor {
    client: reqwest::Client,
}

impl ScorerExecutor {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("reqwest client configuration is static"),
        }
    }

    pub async fn execute(
        &self,
        scorer: &SerializedScorer,
        item: &EvalItem,
        gateway_url: Option<&str>,
    ) -> Result<Feedback, EngineError> {
        let mut feedback = self.execute_all(scorer, item, gateway_url, None).await?;
        if feedback.len() != 1 {
            return Err(EngineError::InvalidParams(format!(
                "scorer returned {} feedback values where one was required",
                feedback.len()
            )));
        }
        Ok(feedback.remove(0))
    }

    pub async fn execute_all(
        &self,
        scorer: &SerializedScorer,
        item: &EvalItem,
        gateway_url: Option<&str>,
        embedding_url: Option<&str>,
    ) -> Result<Vec<Feedback>, EngineError> {
        scorer.validate_for_oss_execution()?;
        match scorer {
            SerializedScorer::Builtin(payload) => {
                crate::builtins::execute(self, payload, item, gateway_url).await
            }
            SerializedScorer::Instructions(payload) => Ok(vec![
                crate::judge::execute_instructions(self, payload, item, gateway_url).await?,
            ]),
            SerializedScorer::MemoryAugmented { common, data } => {
                crate::memory::execute(self, common, data, item, gateway_url, embedding_url).await
            }
            SerializedScorer::Decorator { .. } | SerializedScorer::ThirdParty { .. } => {
                Err(EngineError::UnsupportedScorer)
            }
        }
    }

    pub(crate) fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

impl Default for ScorerExecutor {
    fn default() -> Self {
        Self::new()
    }
}
