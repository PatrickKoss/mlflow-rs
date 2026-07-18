use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const NATIVE_WORKER_PROTOCOL_VERSION: u32 = 1;

/// Closed allowlist matching `mlflow.server.jobs._ALLOWED_JOB_NAME_LIST`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    InvokeScorer,
    RunOnlineTraceScorer,
    RunOnlineSessionScorer,
    OptimizePrompts,
    InvokeIssueDetection,
    InvokeGenaiEvaluate,
}

impl std::fmt::Display for JobKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = serde_json::to_value(self).map_err(|_| std::fmt::Error)?;
        formatter.write_str(value.as_str().ok_or(std::fmt::Error)?)
    }
}

/// Versioned stdin request from the Rust job runner to a per-job worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub protocol_version: u32,
    pub job_id: String,
    pub job_kind: JobKind,
    pub params: Value,
    pub workspace: String,
    pub subject: Value,
}

/// T15.4 subset of Python's `invoke_scorer_job` parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InvokeScorerParams {
    /// JSON string exactly as persisted by scorer CRUD.
    pub serialized_scorer: String,
    #[serde(default)]
    pub inputs: Option<Value>,
    #[serde(default)]
    pub outputs: Option<Value>,
    #[serde(default)]
    pub expectations: Option<Value>,
    /// Spike injection seam; production workers use the propagated gateway URI.
    #[serde(default)]
    pub gateway_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionFailure {
    pub code: String,
    pub message: String,
}

/// The single stdout envelope emitted by the native worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WorkerResponse {
    Succeeded {
        protocol_version: u32,
        job_id: String,
        result: Value,
    },
    Failed {
        protocol_version: u32,
        job_id: String,
        error: ExecutionFailure,
    },
}

impl WorkerResponse {
    pub fn succeeded(job_id: String, result: Value) -> Self {
        Self::Succeeded {
            protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
            job_id,
            result,
        }
    }

    pub fn failed(job_id: String, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Failed {
            protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
            job_id,
            error: ExecutionFailure {
                code: code.into(),
                message: message.into(),
            },
        }
    }
}
