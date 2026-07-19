//! Native semantic execution surfaces for MLflow GenAI.
//!
//! T15.4 intentionally implements only one deterministic builtin and the
//! instructions-judge gateway seam. The public payload, executor, protocol,
//! and subprocess-launcher types are the foundations expanded by T17/T18.

mod builtins;
mod engine;
mod judge;
mod memory;
mod payload;
mod protocol;
mod trace;
mod worker;

pub use engine::{
    AssessmentSource, EngineError, EvalItem, Feedback, MemoryExample, ScorerExecutor,
};
pub use payload::{
    supported_builtin_scorers, BuiltinScorerPayload, InstructionsJudgePayload, ScorerPayloadError,
    SerializedScorer, SerializedScorerCommon,
};
pub use protocol::{
    decode_worker_request, ExecutionFailure, InvokeScorerParams, JobKind, WorkerRequest,
    WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION,
};
pub use worker::{WorkerLaunchError, WorkerLauncher};

pub const MLFLOW_GENAI_WORKER_FIXTURE: &str = "MLFLOW_GENAI_WORKER_FIXTURE";

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

    let fixture_enabled = std::env::var(MLFLOW_GENAI_WORKER_FIXTURE).as_deref() == Ok("1");
    let result = if fixture_enabled {
        execute_fixture(request)
    } else {
        match request.job_kind {
            // Phase 19: replace T15.4's direct EvalItem spike with the full
            // trace-loading, assessment-writing invoke job implementation.
            JobKind::InvokeScorer => execute_invoke_scorer(request).await,
            // Phase 19: native online trace-scoring execution lands here.
            JobKind::RunOnlineTraceScorer => Err(EngineError::UnsupportedJobKind(request.job_kind)),
            // Phase 19: native online session-scoring execution lands here.
            JobKind::RunOnlineSessionScorer => {
                Err(EngineError::UnsupportedJobKind(request.job_kind))
            }
            // Phase 19: native prompt-optimization execution lands here.
            JobKind::OptimizePrompts => Err(EngineError::UnsupportedJobKind(request.job_kind)),
            // Phase 19: native issue-discovery execution lands here.
            JobKind::InvokeIssueDetection => Err(EngineError::UnsupportedJobKind(request.job_kind)),
            // Phase 19: native evaluation execution lands here.
            JobKind::InvokeGenaiEvaluate => Err(EngineError::UnsupportedJobKind(request.job_kind)),
        }
    };

    match result {
        Ok(result) => WorkerResponse::succeeded(request.job_id.clone(), result),
        Err(error) => {
            WorkerResponse::failed(request.job_id.clone(), "ENGINE_ERROR", error.to_string())
        }
    }
}

/// Deterministic, store-free Phase 17 fixture. Values use the exact result
/// shapes returned by the corresponding Python job functions and depend only
/// on the versioned request envelope.
fn execute_fixture(request: &WorkerRequest) -> Result<serde_json::Value, EngineError> {
    use serde_json::{json, Map, Value};

    let params = request.params.as_object().ok_or_else(|| {
        EngineError::InvalidParams("job params must be a JSON object".to_string())
    })?;
    let string = |name: &str| {
        params
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let array_len = |name: &str| {
        params
            .get(name)
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    };

    match request.job_kind {
        JobKind::InvokeScorer => {
            let mut results = Map::new();
            if let Some(trace_ids) = params.get("trace_ids").and_then(Value::as_array) {
                for trace_id in trace_ids.iter().filter_map(Value::as_str) {
                    results.insert(
                        trace_id.to_string(),
                        json!({"assessments": [], "failures": []}),
                    );
                }
            }
            Ok(Value::Object(results))
        }
        JobKind::RunOnlineTraceScorer | JobKind::RunOnlineSessionScorer => Ok(Value::Null),
        JobKind::OptimizePrompts => Ok(json!({
            "run_id": string("run_id"),
            "source_prompt_uri": string("prompt_uri"),
            "optimized_prompt_uri": Value::Null,
            "optimizer_name": match params.get("optimizer_type").and_then(Value::as_str) {
                Some("gepa") => "GepaPromptOptimizer",
                Some("metaprompt") => "MetaPromptOptimizer",
                Some(value) => value,
                None => "",
            },
            "initial_eval_score": Value::Null,
            "final_eval_score": Value::Null,
            "dataset_id": string("dataset_id"),
            "scorer_names": params.get("scorer_names").cloned().unwrap_or_else(|| json!([])),
        })),
        JobKind::InvokeIssueDetection => Ok(json!({
            "summary": format!(
                "fixture:{}/{}/{}",
                request.workspace.as_deref().unwrap_or(""),
                serde_json::to_string(&request.subject)
                    .map_err(|error| EngineError::Serialization(error.to_string()))?,
                string("experiment_id")
            ),
            "issues": array_len("categories"),
            "total_traces_analyzed": array_len("trace_ids"),
            "total_cost_usd": 0.0,
        })),
        JobKind::InvokeGenaiEvaluate => Ok(json!({
            "run_id": string("run_id"),
            "total_traces": array_len("trace_ids"),
            "scorer_count": array_len("serialized_scorers"),
        })),
    }
}

async fn execute_invoke_scorer(request: &WorkerRequest) -> Result<serde_json::Value, EngineError> {
    let params: InvokeScorerParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    let scorer = SerializedScorer::from_json(&params.serialized_scorer)?;
    let gateway_url = params.gateway_url.or_else(|| {
        std::env::var("MLFLOW_GATEWAY_URI")
            .ok()
            .map(|base| worker_gateway_url(&base, "/gateway/mlflow/v1/chat/completions"))
    });
    let embedding_url = params.embedding_url.or_else(|| {
        std::env::var("MLFLOW_GATEWAY_URI")
            .ok()
            .map(|base| worker_gateway_url(&base, "/gateway/openai/v1/embeddings"))
    });
    let feedback = ScorerExecutor::new()
        .execute_all(
            &scorer,
            &EvalItem {
                inputs: params.inputs,
                outputs: params.outputs,
                expectations: params.expectations,
                trace: params.trace,
                session: params.session,
                memory_examples: params.memory_examples,
            },
            gateway_url.as_deref(),
            embedding_url.as_deref(),
        )
        .await?;
    let value = if feedback.len() == 1 {
        serde_json::to_value(&feedback[0])
    } else {
        serde_json::to_value(feedback)
    };
    value.map_err(|error| EngineError::Serialization(error.to_string()))
}

fn worker_gateway_url(base: &str, path: &str) -> String {
    if base.ends_with(path) {
        base.to_string()
    } else {
        format!("{}{path}", base.trim_end_matches('/'))
    }
}
