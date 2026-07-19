use std::path::{Path, PathBuf};

use mlflow_genai::{JobKind, WorkerLauncher, WorkerRequest, NATIVE_WORKER_PROTOCOL_VERSION};
use mlflow_server::{build_app_with_state, AppState, ServerConfig};
use mlflow_store::{
    RunStatus, SpanInput, StartTraceInput, TraceTimeRange, TrackingStore, WORKSPACE_DEFAULT_NAME,
};
use mlflow_test_support::TempDb;
use serde_json::{json, Value};
use tokio::net::TcpListener;

const EXPERIMENT_ID: &str = "0";
const RESPONSE_LENGTH: &str =
    include_str!("../../mlflow-genai/tests/fixtures/builtin_response_length_scorer.json");
const INSTRUCTIONS: &str =
    include_str!("../../mlflow-genai/tests/fixtures/instructions_judge_scorer.json");

struct TestServer {
    base: String,
    store: TrackingStore,
    _database: TempDb,
    handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start() -> Self {
        let database = TempDb::new("evaluation_jobs").await;
        let store = TrackingStore::new(
            database.connect().await,
            "/tmp/mlflow-rust-evaluation-job-artifacts",
        );
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            allowed_hosts: None,
            cors_allowed_origins: None,
            ..Default::default()
        };
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = build_app_with_state(&config, AppState::new(store.clone()));
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            base: format!("http://{address}"),
            store,
            _database: database,
            handle,
        }
    }

    fn launcher(&self) -> WorkerLauncher {
        WorkerLauncher::new(worker_path())
            .clean_environment()
            .env("PATH", empty_path())
            .env("MLFLOW_TRACKING_URI", &self.base)
            .env("MLFLOW_GENAI_EVAL_MAX_RETRIES", "0")
            .env("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "0")
            .env(
                "MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS",
                "0",
            )
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

fn worker_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mlflow-genai-worker"))
}

fn empty_path() -> &'static Path {
    Path::new("/nonexistent-native-worker-path")
}

fn request(job_id: &str, job_kind: JobKind, params: Value) -> WorkerRequest {
    WorkerRequest {
        protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
        job_id: job_id.to_string(),
        job_kind,
        params,
        workspace: None,
        subject: json!({"user_id": "fake-evaluation-user"}),
    }
}

async fn seed_trace(
    store: &TrackingStore,
    trace_id: &str,
    timestamp_ms: i64,
    output: &str,
    session_id: Option<&str>,
) {
    let metadata = session_id.map_or_else(Vec::new, |session_id| {
        vec![("mlflow.trace.session".to_string(), session_id.to_string())]
    });
    store
        .start_trace(
            WORKSPACE_DEFAULT_NAME,
            &StartTraceInput {
                trace_id: trace_id.to_string(),
                experiment_id: EXPERIMENT_ID.to_string(),
                request_time: timestamp_ms,
                execution_duration: Some(1),
                state: "OK".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: None,
                tags: Vec::new(),
                trace_metadata: metadata,
                trace_metrics: Vec::new(),
            },
        )
        .await
        .unwrap();
    let content = json!({
        "trace_id": "AAAAAAAAAAAAAAAAAAAAAA==",
        "span_id": "AAAAAAAAAAE=",
        "parent_span_id": null,
        "name": "root",
        "start_time_unix_nano": timestamp_ms * 1_000_000,
        "end_time_unix_nano": timestamp_ms * 1_000_000 + 1,
        "events": [],
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {
            "mlflow.spanInputs": "{\"question\":\"What is two plus two?\"}",
            "mlflow.spanOutputs": serde_json::to_string(output).unwrap(),
            "mlflow.spanType": "\"CHAIN\""
        },
        "links": []
    })
    .to_string();
    store
        .log_spans(
            WORKSPACE_DEFAULT_NAME,
            EXPERIMENT_ID,
            &[SpanInput {
                trace_id: trace_id.to_string(),
                span_id: "0000000000000001".to_string(),
                parent_span_id: None,
                name: Some("root".to_string()),
                span_type: Some("CHAIN".to_string()),
                status: "OK".to_string(),
                start_time_unix_nano: timestamp_ms * 1_000_000,
                end_time_unix_nano: Some(timestamp_ms * 1_000_000 + 1),
                content,
                dimension_attributes: None,
            }],
            &[],
            &[TraceTimeRange {
                trace_id: trace_id.to_string(),
                min_start_ms: timestamp_ms,
                max_end_ms: Some(timestamp_ms),
                root_span_status: Some("OK".to_string()),
            }],
        )
        .await
        .unwrap();
}

fn session_scorer() -> String {
    let mut scorer: Value = serde_json::from_str(INSTRUCTIONS).unwrap();
    scorer["is_session_level_scorer"] = Value::Bool(true);
    scorer["instructions_judge_pydantic_data"]["instructions"] =
        Value::String("Judge the ordered {{ conversation }}.".to_string());
    scorer.to_string()
}

#[tokio::test]
async fn native_jobs_persist_evaluation_invoke_online_and_checkpoints() {
    let server = TestServer::start().await;
    seed_trace(
        &server.store,
        "eval-a",
        1_000,
        "two words",
        Some("session-a"),
    )
    .await;
    seed_trace(
        &server.store,
        "eval-b",
        2_000,
        "three words now",
        Some("session-a"),
    )
    .await;
    let run = server
        .store
        .create_run(
            WORKSPACE_DEFAULT_NAME,
            EXPERIMENT_ID,
            Some("fake-evaluation-user"),
            Some(1_000),
            Some("evaluation"),
            &[],
        )
        .await
        .unwrap();

    let evaluate = request(
        "evaluate",
        JobKind::InvokeGenaiEvaluate,
        json!({
            "trace_ids": ["eval-a", "eval-b"],
            "serialized_scorers": [RESPONSE_LENGTH, session_scorer()],
            "run_id": run.info.run_id,
        }),
    );
    let result = server
        .launcher()
        .env("MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING", "1")
        .run(&evaluate)
        .await
        .unwrap();
    assert_eq!(result["total_traces"], 2);
    let run = server
        .store
        .get_run(WORKSPACE_DEFAULT_NAME, run.info.run_id.as_str())
        .await
        .unwrap();
    assert_eq!(run.info.status, RunStatus::FINISHED);
    let metric = run
        .data
        .metrics
        .iter()
        .find(|metric| metric.key == "response_length/mean")
        .unwrap();
    assert_eq!(metric.value, 1.0);
    let evaluated = server
        .store
        .batch_get_traces(
            WORKSPACE_DEFAULT_NAME,
            &["eval-a".to_string(), "eval-b".to_string()],
        )
        .await
        .unwrap();
    assert_eq!(
        evaluated
            .iter()
            .map(|trace| trace.info.assessments.len())
            .sum::<usize>(),
        3
    );
    let session_error = evaluated[0]
        .info
        .assessments
        .iter()
        .find(|assessment| assessment.name == "concise_answer")
        .unwrap();
    assert!(session_error
        .error
        .as_deref()
        .unwrap()
        .contains("SCORER_ERROR"));
    assert!(session_error
        .metadata
        .as_deref()
        .unwrap()
        .contains("mlflow.trace.session"));
    let single_turn_feedback = evaluated[0]
        .info
        .assessments
        .iter()
        .find(|assessment| assessment.name == "response_length")
        .unwrap();
    let scorer_metadata: Value =
        serde_json::from_str(single_turn_feedback.metadata.as_deref().unwrap()).unwrap();
    let scorer_trace_id = scorer_metadata["mlflow.assessment.scorerTraceId"]
        .as_str()
        .unwrap();
    let scorer_trace = server
        .store
        .get_trace_info(WORKSPACE_DEFAULT_NAME, scorer_trace_id)
        .await
        .unwrap();
    assert_eq!(
        scorer_trace.tag("mlflow.trace.sourceScorer"),
        Some("response_length")
    );

    let invoke = request(
        "invoke",
        JobKind::InvokeScorer,
        json!({
            "serialized_scorer": RESPONSE_LENGTH,
            "trace_ids": ["eval-a", "eval-b"],
            "log_assessments": true,
        }),
    );
    let result = server.launcher().run(&invoke).await.unwrap();
    assert_eq!(result.as_object().unwrap().len(), 2);
    assert!(result["eval-a"]["failures"].as_array().unwrap().is_empty());

    seed_trace(&server.store, "online-trace", 3_000, "two words", None).await;
    let online_trace = request(
        "online-trace",
        JobKind::RunOnlineTraceScorer,
        json!({
            "experiment_id": EXPERIMENT_ID,
            "online_scorers": [{
                "serialized_scorer": RESPONSE_LENGTH,
                "online_config": {"sample_rate": 1.0, "filter_string": null}
            }],
            "current_time_ms": 4_000,
        }),
    );
    assert_eq!(
        server.launcher().run(&online_trace).await.unwrap(),
        Value::Null
    );
    let experiment = server
        .store
        .get_experiment(WORKSPACE_DEFAULT_NAME, EXPERIMENT_ID)
        .await
        .unwrap();
    let trace_checkpoint = experiment
        .tags
        .iter()
        .find(|tag| tag.key == "mlflow.latestOnlineScoring.trace.checkpoint")
        .and_then(|tag| tag.value.as_deref())
        .unwrap();
    assert_eq!(
        trace_checkpoint,
        "{\"timestamp_ms\": 3000, \"trace_id\": \"online-trace\"}"
    );

    seed_trace(
        &server.store,
        "online-session-a",
        5_000,
        "two words",
        Some("online-session"),
    )
    .await;
    seed_trace(
        &server.store,
        "online-session-b",
        6_000,
        "two words",
        Some("online-session"),
    )
    .await;
    let online_session = request(
        "online-session",
        JobKind::RunOnlineSessionScorer,
        json!({
            "experiment_id": EXPERIMENT_ID,
            "online_scorers": [{
                "serialized_scorer": session_scorer(),
                "online_config": {"sample_rate": 1.0, "filter_string": null}
            }],
            "current_time_ms": 7_000,
        }),
    );
    assert_eq!(
        server.launcher().run(&online_session).await.unwrap(),
        Value::Null
    );
    let experiment = server
        .store
        .get_experiment(WORKSPACE_DEFAULT_NAME, EXPERIMENT_ID)
        .await
        .unwrap();
    let session_checkpoint = experiment
        .tags
        .iter()
        .find(|tag| tag.key == "mlflow.latestOnlineScoring.session.checkpoint")
        .and_then(|tag| tag.value.as_deref())
        .unwrap();
    assert_eq!(
        session_checkpoint,
        "{\"timestamp_ms\": 6000, \"session_id\": \"online-session\"}"
    );

    let first_session_trace = server
        .store
        .get_trace_info(WORKSPACE_DEFAULT_NAME, "online-session-a")
        .await
        .unwrap();
    assert_eq!(
        first_session_trace
            .assessments
            .iter()
            .filter(|assessment| assessment.name == "concise_answer")
            .count(),
        1
    );
    seed_trace(
        &server.store,
        "online-session-c",
        8_000,
        "two words",
        Some("online-session"),
    )
    .await;
    let online_session_again = request(
        "online-session-again",
        JobKind::RunOnlineSessionScorer,
        json!({
            "experiment_id": EXPERIMENT_ID,
            "online_scorers": [{
                "serialized_scorer": session_scorer(),
                "online_config": {"sample_rate": 1.0, "filter_string": null}
            }],
            "current_time_ms": 9_000,
        }),
    );
    server.launcher().run(&online_session_again).await.unwrap();
    let first_session_trace = server
        .store
        .get_trace_info(WORKSPACE_DEFAULT_NAME, "online-session-a")
        .await
        .unwrap();
    assert_eq!(
        first_session_trace
            .assessments
            .iter()
            .filter(|assessment| assessment.name == "concise_answer")
            .count(),
        1,
        "the newly logged session assessment replaces the previous online result"
    );
}
