//! T17.4 invoke submission parity and runner-fixture integration.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_genai::{execute_worker_request, JobKind, WorkerRequest, WorkerResponse};
use mlflow_server::job_runner::{
    JobExecutionFuture, JobExecutionRequest, JobExecutionResult, JobExecutor, JobRunner,
    JobRunnerConfig,
};
use mlflow_server::native_worker::native_job_functions;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{
    Db, JobStatus, JobStore, PoolConfig, StartTraceInput, TrackingStore,
    Workspace as WorkspaceEntity, WorkspaceStore,
};
use serde_json::{json, Value};
use tower::ServiceExt;

const WS: &str = "default";
const INSTRUCTIONS_SCORER: &str =
    include_str!("../../mlflow-genai/tests/fixtures/instructions_judge_scorer.json");

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

fn auth_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("mlflow-auth")
        .join("tests")
        .join("fixtures")
        .join("basic_auth.db")
}

struct Fixture {
    _directory: tempfile::TempDir,
    tracking: TrackingStore,
    jobs: JobStore,
    workspace: String,
    experiment_id: String,
    app: axum::Router,
}

impl Fixture {
    async fn new() -> Self {
        Self::new_configured(WS, false, false).await
    }

    async fn new_in_workspace(workspace: &str, enable_workspaces: bool) -> Self {
        Self::new_configured(workspace, enable_workspaces, false).await
    }

    async fn new_authenticated() -> Self {
        Self::new_configured(WS, false, true).await
    }

    async fn new_configured(workspace: &str, enable_workspaces: bool, enable_auth: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invoke.db");
        std::fs::copy(fixture_path(), &path).unwrap();
        let uri = format!("sqlite:///{}", path.display());
        let db = Db::connect(&uri, PoolConfig::default()).await.unwrap();
        let tracking = TrackingStore::new(
            db.clone(),
            directory.path().join("artifacts").display().to_string(),
        );
        let workspace_store = WorkspaceStore::new(db.clone(), &uri);
        if enable_workspaces && workspace != WS {
            workspace_store
                .create_workspace(WorkspaceEntity::named(workspace))
                .await
                .unwrap();
        }
        let experiment_id = tracking
            .create_experiment(workspace, "invoke-http", None, &[])
            .await
            .unwrap();
        let jobs = JobStore::new(db);
        let mut state = AppState::new(tracking.clone());
        if enable_workspaces {
            state = state.with_workspace_store(workspace_store);
        }
        if enable_auth {
            let auth_path = directory.path().join("auth.db");
            std::fs::copy(auth_fixture_path(), &auth_path).unwrap();
            let auth_db = AuthDb::connect_and_verify_with(
                &format!("sqlite:///{}", auth_path.display()),
                None,
                PoolConfig::default(),
            )
            .await
            .unwrap();
            state = state.with_auth_store(AuthStore::new(auth_db));
        }
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(state.clone()),
        );
        Self {
            _directory: directory,
            tracking,
            jobs,
            workspace: workspace.to_string(),
            experiment_id,
            app,
        }
    }

    async fn post(&self, path: &str, body: Value) -> (StatusCode, Vec<u8>) {
        self.request(path, body, None).await
    }

    async fn request(
        &self,
        path: &str,
        body: Value,
        authorization: Option<&str>,
    ) -> (StatusCode, Vec<u8>) {
        self.request_with_workspace(path, body, authorization, None)
            .await
    }

    async fn request_with_workspace(
        &self,
        path: &str,
        body: Value,
        authorization: Option<&str>,
        workspace: Option<&str>,
    ) -> (StatusCode, Vec<u8>) {
        let mut request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(authorization) = authorization {
            request = request.header(header::AUTHORIZATION, authorization);
        }
        if let Some(workspace) = workspace {
            request = request.header("X-MLFLOW-WORKSPACE", workspace);
        }
        let response = self
            .app
            .clone()
            .oneshot(request.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        (status, body.to_vec())
    }
}

#[tokio::test]
async fn invoke_routes_are_authenticated_only_under_basic_auth() {
    let fixture = Fixture::new_authenticated().await;
    let cases = [
        (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "trace_ids": ["tr-a"],
                "serialized_scorers": [INSTRUCTIONS_SCORER],
            }),
        ),
        (
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "serialized_scorer": INSTRUCTIONS_SCORER,
                "trace_ids": ["tr-a"],
            }),
        ),
        (
            "/ajax-api/3.0/mlflow/issues/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "trace_ids": ["tr-a"],
                "categories": ["correctness"],
                "provider": "openai",
                "model": "gpt-5",
            }),
        ),
    ];
    for (path, body) in cases {
        let (status, _) = fixture.post(path, body.clone()).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{path}");
        let (status, response) = fixture
            .request(
                path,
                body,
                Some("Basic Ym9iX3Bia2RmMjpib2ItcGFzc3dvcmQtNDU2Nw=="),
            )
            .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{path}: {}",
            String::from_utf8_lossy(&response)
        );
    }
}

#[tokio::test]
async fn validation_errors_match_python_exactly() {
    let fixture = Fixture::new().await;
    let cases = [
        (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({}),
            StatusCode::BAD_REQUEST,
            "Missing value for required parameter 'experiment_id'. See the API docs for more information about request parameters.",
        ),
        (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({"experiment_id": fixture.experiment_id, "trace_ids": [], "serialized_scorers": [INSTRUCTIONS_SCORER]}),
            StatusCode::BAD_REQUEST,
            "Please select at least one trace to evaluate.",
        ),
        (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({"experiment_id": fixture.experiment_id, "trace_ids": ["tr-1"], "serialized_scorers": []}),
            StatusCode::BAD_REQUEST,
            "Please select at least one judge.",
        ),
        (
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json!({}),
            StatusCode::BAD_REQUEST,
            "Missing required parameter: experiment_id",
        ),
        (
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json!({"experiment_id": fixture.experiment_id, "serialized_scorer": "not-json", "trace_ids": ["tr-1"]}),
            StatusCode::BAD_REQUEST,
            "Invalid JSON in serialized scorer: Expecting value: line 1 column 1 (char 0)",
        ),
        (
            "/ajax-api/3.0/mlflow/issues/invoke",
            json!({"experiment_id": fixture.experiment_id, "trace_ids": ["tr-1"], "categories": ["correctness"], "provider": "openai"}),
            StatusCode::INTERNAL_SERVER_ERROR,
            "Either 'endpoint_name' or both 'provider' and 'model' must be provided",
        ),
    ];
    for (path, request, expected_status, expected_message) in cases {
        let (status, body) = fixture.post(path, request).await;
        assert_eq!(
            status,
            expected_status,
            "{path}: {}",
            String::from_utf8_lossy(&body)
        );
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["message"], expected_message, "{path}");
    }
}

#[tokio::test]
async fn fixed_scorer_batch_boundary_and_subject_match_python_job_rows() {
    let fixture = Fixture::new().await;
    let trace_ids = (0..101)
        .map(|index| format!("tr-{index:03}"))
        .collect::<Vec<_>>();
    let (status, body) = fixture
        .request(
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "serialized_scorer": INSTRUCTIONS_SCORER,
                "trace_ids": trace_ids,
                "log_assessments": true,
            }),
            Some("Basic YWxpY2U6cGFzc3dvcmQ="),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let response: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["jobs"].as_array().unwrap().len(), 2);
    assert_eq!(
        response["jobs"][0]["trace_ids"].as_array().unwrap().len(),
        100
    );
    assert_eq!(
        response["jobs"][1]["trace_ids"].as_array().unwrap().len(),
        1
    );
    for submitted in response["jobs"].as_array().unwrap() {
        let job = fixture
            .jobs
            .get_job(WS, submitted["job_id"].as_str().unwrap())
            .await
            .unwrap();
        let params: Value = serde_json::from_str(&job.params).unwrap();
        assert_eq!(params["username"], "alice");
        assert_eq!(params["log_assessments"], true);
        assert_eq!(params["trace_ids"], submitted["trace_ids"]);
        assert_eq!(job.workspace, fixture.workspace);
    }
}

#[tokio::test]
async fn invoke_jobs_and_runs_stay_in_the_request_workspace() {
    let fixture = Fixture::new_in_workspace("team-a", true).await;
    let (status, body) = fixture
        .request_with_workspace(
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "trace_ids": ["tr-a"],
                "serialized_scorers": [INSTRUCTIONS_SCORER],
            }),
            Some("Basic YWxpY2U6cGFzc3dvcmQ="),
            Some("team-a"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response: Value = serde_json::from_slice(&body).unwrap();
    let job_id = response["job_id"].as_str().unwrap();
    let run_id = response["run_id"].as_str().unwrap();
    let job = fixture.jobs.get_job("team-a", job_id).await.unwrap();
    assert_eq!(job.workspace, "team-a");
    let params: Value = serde_json::from_str(&job.params).unwrap();
    assert_eq!(params["username"], "alice");
    assert!(fixture.jobs.get_job(WS, job_id).await.is_err());
    assert!(fixture.tracking.get_run("team-a", run_id).await.is_ok());
    assert!(fixture.tracking.get_run(WS, run_id).await.is_err());
}

#[tokio::test]
async fn session_scorer_batches_by_session_and_orders_each_by_timestamp() {
    let fixture = Fixture::new().await;
    for (trace_id, timestamp, session) in [
        ("tr-a-late", 30, Some("session-a")),
        ("tr-b", 20, Some("session-b")),
        ("tr-no-session", 15, None),
        ("tr-a-early", 10, Some("session-a")),
    ] {
        fixture
            .tracking
            .start_trace(
                WS,
                &StartTraceInput {
                    trace_id: trace_id.to_string(),
                    experiment_id: fixture.experiment_id.clone(),
                    request_time: timestamp,
                    execution_duration: Some(1),
                    state: "OK".to_string(),
                    client_request_id: None,
                    request_preview: None,
                    response_preview: None,
                    tags: Vec::new(),
                    trace_metadata: session
                        .map(|session| {
                            vec![("mlflow.trace.session".to_string(), session.to_string())]
                        })
                        .unwrap_or_default(),
                    trace_metrics: Vec::new(),
                    assessments: Vec::new(),
                },
            )
            .await
            .unwrap();
    }
    let mut scorer: Value = serde_json::from_str(INSTRUCTIONS_SCORER).unwrap();
    scorer["is_session_level_scorer"] = json!(true);
    scorer["instructions_judge_pydantic_data"]["instructions"] =
        json!("Evaluate {{ conversation }}");
    let (status, body) = fixture
        .post(
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "serialized_scorer": scorer.to_string(),
                "trace_ids": ["tr-a-late", "tr-b", "tr-no-session", "tr-a-early"],
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["jobs"].as_array().unwrap().len(), 2);
    assert_eq!(
        response["jobs"][0]["trace_ids"],
        json!(["tr-a-early", "tr-a-late"])
    );
    assert_eq!(response["jobs"][1]["trace_ids"], json!(["tr-b"]));
}

#[tokio::test]
async fn evaluate_and_issue_precreate_python_shaped_runs_and_tags() {
    let fixture = Fixture::new().await;
    let (status, body) = fixture
        .request(
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "trace_ids": ["tr-a", "tr-b"],
                "serialized_scorers": [INSTRUCTIONS_SCORER],
            }),
            Some("Basic Ym9iOnNlY3JldA=="),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.ends_with(b"\n"));
    let response: Value = serde_json::from_slice(&body).unwrap();
    let run_id = response["run_id"].as_str().unwrap();
    let run = fixture.tracking.get_run(WS, run_id).await.unwrap();
    assert_eq!(run.info.user_id.as_deref(), Some("unknown"));
    assert_eq!(run.info.status, "RUNNING");
    assert_eq!(run.info.end_time, None);
    let tags = run
        .data
        .tags
        .iter()
        .map(|tag| (tag.key.as_str(), tag.value.as_str()))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(tags["mlflow.runType"], "genai_evaluate");
    assert_eq!(tags["mlflow.genaiEvaluate.jobId"], response["job_id"]);
    assert!(tags.contains_key("mlflow.runName"));
    assert!(!tags.contains_key("mlflow.user"));
    let job = fixture
        .jobs
        .get_job(WS, response["job_id"].as_str().unwrap())
        .await
        .unwrap();
    let params: Value = serde_json::from_str(&job.params).unwrap();
    assert_eq!(params["username"], "bob");
    assert_eq!(params["run_id"], run_id);

    let (status, body) = fixture
        .post(
            "/ajax-api/3.0/mlflow/issues/invoke",
            json!({
                "experiment_id": fixture.experiment_id,
                "trace_ids": ["tr-a", "tr-b"],
                "categories": ["correctness", "safety"],
                "provider": "openai",
                "model": "gpt-5",
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let response: Value = serde_json::from_slice(&body).unwrap();
    let run = fixture
        .tracking
        .get_run(WS, response["run_id"].as_str().unwrap())
        .await
        .unwrap();
    assert_eq!(run.info.status, "RUNNING");
    assert!(run.info.end_time.is_some());
    let tags = run
        .data
        .tags
        .iter()
        .map(|tag| (tag.key.as_str(), tag.value.as_str()))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(tags["mlflow.runType"], "issue_detection");
    assert_eq!(tags["categories"], "correctness,safety");
    assert_eq!(tags["model"], "openai:/gpt-5");
    assert_eq!(tags["total_traces"], "2");
    assert_eq!(tags["mlflow.issueDetection.jobId"], response["job_id"]);
    assert!(tags.contains_key("mlflow.user"));
    assert!(tags.contains_key("mlflow.source.name"));
    assert_eq!(tags["mlflow.source.type"], "LOCAL");
}

#[derive(Clone)]
struct InProcessFixtureExecutor;

impl JobExecutor for InProcessFixtureExecutor {
    fn execute(&self, request: JobExecutionRequest) -> JobExecutionFuture {
        Box::pin(async move {
            let job_kind = request.job_name.parse::<JobKind>().unwrap();
            let response = execute_worker_request(&WorkerRequest {
                protocol_version: mlflow_genai::NATIVE_WORKER_PROTOCOL_VERSION,
                job_id: request.job_id,
                job_kind,
                params: request.params,
                workspace: request.workspace,
                subject: request.subject,
            })
            .await;
            match response {
                WorkerResponse::Succeeded { result, .. } => JobExecutionResult::Succeeded(result),
                WorkerResponse::Failed { error, .. } => JobExecutionResult::Failed {
                    error: error.message,
                    transient: false,
                    details: None,
                },
            }
        })
    }
}

#[tokio::test]
async fn every_invoke_kind_and_prompt_optimization_reach_fixture_results_through_runner() {
    std::env::set_var(mlflow_genai::MLFLOW_GENAI_WORKER_FIXTURE, "1");
    let fixture = Fixture::new().await;
    let runner = JobRunner::new(
        fixture.jobs.clone(),
        Arc::new(InProcessFixtureExecutor),
        native_job_functions().unwrap(),
        vec![fixture.workspace.clone()],
        JobRunnerConfig {
            queue_poll_interval: Duration::from_millis(5),
            status_poll_interval: Duration::from_millis(5),
            ..Default::default()
        },
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    let submissions = [
        (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            json!({"experiment_id": fixture.experiment_id, "trace_ids": ["tr-a"], "serialized_scorers": [INSTRUCTIONS_SCORER]}),
        ),
        (
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json!({"experiment_id": fixture.experiment_id, "serialized_scorer": INSTRUCTIONS_SCORER, "trace_ids": ["tr-a"]}),
        ),
        (
            "/ajax-api/3.0/mlflow/issues/invoke",
            json!({"experiment_id": fixture.experiment_id, "trace_ids": ["tr-a"], "categories": ["correctness"], "provider": "openai", "model": "gpt-5"}),
        ),
        (
            "/api/3.0/mlflow/prompt-optimization/jobs",
            json!({
                "experiment_id": fixture.experiment_id,
                "source_prompt_uri": "prompts:/runner/1",
                "config": {"optimizer_type": "OPTIMIZER_TYPE_METAPROMPT", "scorers": []},
            }),
        ),
    ];
    let mut job_ids = Vec::new();
    for (path, request) in submissions {
        let (status, body) = fixture.post(path, request).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let body: Value = serde_json::from_slice(&body).unwrap();
        let job_id = body
            .get("job_id")
            .or_else(|| body.get("jobs").and_then(|jobs| jobs.get(0))?.get("job_id"))
            .or_else(|| body.get("job")?.get("job_id"))
            .and_then(Value::as_str)
            .unwrap();
        job_ids.push(job_id.to_string());
    }

    for job_id in job_ids {
        let job = wait_final(&fixture.jobs, &fixture.workspace, &job_id).await;
        assert_eq!(
            job.status,
            JobStatus::Succeeded,
            "{job_id}: {:?}",
            job.result
        );
        assert!(job.parsed_result().unwrap().is_some());
    }
    runner.shutdown().await;
}

async fn wait_final(jobs: &JobStore, workspace: &str, job_id: &str) -> mlflow_store::Job {
    for _ in 0..400 {
        let job = jobs.get_job(workspace, job_id).await.unwrap();
        if job.status.is_finalized() {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("job {job_id} did not finish")
}
