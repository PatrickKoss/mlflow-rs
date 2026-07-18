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

impl std::str::FromStr for JobKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "invoke_scorer" => Ok(Self::InvokeScorer),
            "run_online_trace_scorer" => Ok(Self::RunOnlineTraceScorer),
            "run_online_session_scorer" => Ok(Self::RunOnlineSessionScorer),
            "optimize_prompts" => Ok(Self::OptimizePrompts),
            "invoke_issue_detection" => Ok(Self::InvokeIssueDetection),
            "invoke_genai_evaluate" => Ok(Self::InvokeGenaiEvaluate),
            _ => Err(value.to_string()),
        }
    }
}

/// Versioned stdin request from the Rust job runner to a per-job worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub protocol_version: u32,
    pub job_id: String,
    pub job_kind: JobKind,
    pub params: Value,
    pub workspace: Option<String>,
    pub subject: Value,
}

/// Decode in the security-sensitive order required by the worker protocol.
/// Version and kind are validated before `params` can reach a dispatcher.
pub fn decode_worker_request(bytes: &[u8]) -> Result<WorkerRequest, WorkerResponse> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| {
        WorkerResponse::failed(
            "<unknown>".to_string(),
            "INVALID_REQUEST_ENVELOPE",
            error.to_string(),
        )
    })?;
    let job_id = value
        .get("job_id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string();
    let protocol_version = value
        .get("protocol_version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| {
            WorkerResponse::failed(
                job_id.clone(),
                "INVALID_REQUEST_ENVELOPE",
                "protocol_version must be an unsigned 32-bit integer",
            )
        })?;
    if protocol_version != NATIVE_WORKER_PROTOCOL_VERSION {
        return Err(WorkerResponse::failed(
            job_id,
            "UNSUPPORTED_PROTOCOL_VERSION",
            format!(
                "unsupported native worker protocol version {protocol_version}; expected {NATIVE_WORKER_PROTOCOL_VERSION}"
            ),
        ));
    }

    let job_kind = value
        .get("job_kind")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            WorkerResponse::failed(
                job_id.clone(),
                "INVALID_REQUEST_ENVELOPE",
                "job_kind must be a string",
            )
        })?;
    job_kind.parse::<JobKind>().map_err(|unknown| {
        WorkerResponse::failed(
            job_id.clone(),
            "UNKNOWN_JOB_KIND",
            format!("unknown native worker job kind {unknown:?}"),
        )
    })?;

    serde_json::from_value(value).map_err(|error| {
        WorkerResponse::failed(job_id, "INVALID_REQUEST_ENVELOPE", error.to_string())
    })
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
