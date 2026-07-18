//! Native semantic execution surfaces for MLflow GenAI.
//!
//! T15.4 intentionally implements only one deterministic builtin and the
//! instructions-judge gateway seam. The public payload, executor, protocol,
//! and subprocess-launcher types are the foundations expanded by T17/T18.

mod engine;
mod payload;
mod protocol;
mod worker;

pub use engine::{AssessmentSource, EngineError, EvalItem, Feedback, ScorerExecutor};
pub use payload::{
    BuiltinScorerPayload, InstructionsJudgePayload, ScorerPayloadError, SerializedScorer,
    SerializedScorerCommon,
};
pub use protocol::{
    ExecutionFailure, InvokeScorerParams, JobKind, WorkerRequest, WorkerResponse,
    NATIVE_WORKER_PROTOCOL_VERSION,
};
pub use worker::{WorkerLaunchError, WorkerLauncher};

/// Execute one validated native-worker request.
pub async fn execute_worker_request(request: &WorkerRequest) -> WorkerResponse {
    if request.protocol_version != NATIVE_WORKER_PROTOCOL_VERSION {
        return WorkerResponse::failed(
            request.job_id.clone(),
            "UNSUPPORTED_PROTOCOL_VERSION",
            format!(
                "unsupported native worker protocol version {}; expected {}",
                request.protocol_version, NATIVE_WORKER_PROTOCOL_VERSION
            ),
        );
    }

    let result = match request.job_kind {
        JobKind::InvokeScorer => execute_invoke_scorer(request).await,
        _ => Err(EngineError::UnsupportedJobKind(request.job_kind)),
    };

    match result {
        Ok(result) => WorkerResponse::succeeded(request.job_id.clone(), result),
        Err(error) => {
            WorkerResponse::failed(request.job_id.clone(), "ENGINE_ERROR", error.to_string())
        }
    }
}

async fn execute_invoke_scorer(request: &WorkerRequest) -> Result<serde_json::Value, EngineError> {
    let params: InvokeScorerParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    let scorer = SerializedScorer::from_json(&params.serialized_scorer)?;
    let feedback = ScorerExecutor::new()
        .execute(
            &scorer,
            &EvalItem {
                inputs: params.inputs,
                outputs: params.outputs,
                expectations: params.expectations,
            },
            params.gateway_url.as_deref(),
        )
        .await?;
    serde_json::to_value(feedback).map_err(|error| EngineError::Serialization(error.to_string()))
}
