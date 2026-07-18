#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{extract::State, routing::post, Json, Router};
use mlflow_genai::{
    JobKind, WorkerLaunchError, WorkerLauncher, WorkerRequest, NATIVE_WORKER_PROTOCOL_VERSION,
};
use mlflow_store::{Job, JobStatus, JobStore};
use mlflow_test_support::TempDb;
use serde_json::{json, Value};
use tempfile::TempDir;

const BUILTIN_SCORER: &str =
    include_str!("../../mlflow-genai/tests/fixtures/builtin_response_length_scorer.json");
const BUILTIN_EXPECTED: &str =
    include_str!("../../mlflow-genai/tests/fixtures/builtin_response_length_expected.json");
const INSTRUCTIONS_SCORER: &str =
    include_str!("../../mlflow-genai/tests/fixtures/instructions_judge_scorer.json");
const INSTRUCTIONS_EXPECTED: &str =
    include_str!("../../mlflow-genai/tests/fixtures/instructions_judge_expected.json");
const INSTRUCTIONS_REQUEST: &str =
    include_str!("../../mlflow-genai/tests/fixtures/instructions_judge_request.json");
const SPIKE_MODE_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_MODE";

#[derive(Clone)]
struct GatewayState {
    requests: Arc<Mutex<Vec<Value>>>,
}

struct MockGateway {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    task: tokio::task::JoinHandle<()>,
}

impl MockGateway {
    async fn start() -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_completion))
            .with_state(GatewayState {
                requests: Arc::clone(&requests),
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock gateway binds");
        let address = listener.local_addr().expect("mock address");
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("mock gateway serves");
        });
        Self {
            url: format!("http://{address}/v1/chat/completions"),
            requests,
            task,
        }
    }
}

impl Drop for MockGateway {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn mock_completion(
    State(state): State<GatewayState>,
    Json(request): Json<Value>,
) -> Json<Value> {
    state.requests.lock().expect("request lock").push(request);
    Json(json!({
        "id": "mock-completion-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-judge",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"result\": \"yes\", \"rationale\": \"The response is concise.\"}"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    }))
}

#[tokio::test]
async fn both_python_oracles_execute_inside_path_isolated_worker() {
    let path = empty_path();
    assert_python_is_unreachable(path.path());
    let gateway = MockGateway::start().await;
    let launcher = isolated_launcher(path.path());
    let database = SpikeJobDatabase::new().await;

    let builtin_job = database.create_and_start("job-builtin").await;
    let builtin = launcher
        .run(&invoke_request(
            &builtin_job.job_id,
            BUILTIN_SCORER,
            json!("native Rust worker works"),
            None,
        ))
        .await
        .expect("builtin worker succeeds without Python");
    assert_eq!(builtin, fixture(BUILTIN_EXPECTED));
    database.record_success(&builtin_job.job_id, &builtin).await;
    assert_eq!(
        database.get(&builtin_job.job_id).await.status,
        JobStatus::Succeeded
    );

    let instructions_job = database.create_and_start("job-instructions").await;
    let instructions = launcher
        .run(&invoke_request(
            &instructions_job.job_id,
            INSTRUCTIONS_SCORER,
            json!("Brief answer."),
            Some(&gateway.url),
        ))
        .await
        .expect("instructions worker succeeds without Python");
    assert_eq!(instructions, fixture(INSTRUCTIONS_EXPECTED));
    database
        .record_success(&instructions_job.job_id, &instructions)
        .await;
    let instructions_job = database.get(&instructions_job.job_id).await;
    assert_eq!(instructions_job.status, JobStatus::Succeeded);
    assert_eq!(
        instructions_job.parsed_result().expect("job result JSON"),
        Some(instructions)
    );

    let requests = gateway.requests.lock().expect("request lock");
    assert_eq!(requests.as_slice(), &[fixture(INSTRUCTIONS_REQUEST)]);
}

#[tokio::test]
async fn distinct_worker_failures_are_observable_in_spike_jobs_table() {
    let path = empty_path();
    let database = SpikeJobDatabase::new().await;
    let request = invoke_request(
        "replaced-per-job",
        BUILTIN_SCORER,
        json!("native Rust worker works"),
        None,
    );

    let cases = [
        ("job-nonzero", "nonzero", "non_zero_exit", JobStatus::Failed),
        ("job-signal", "signal", "signal", JobStatus::Failed),
        (
            "job-malformed",
            "malformed",
            "malformed_output",
            JobStatus::Failed,
        ),
        (
            "job-timeout",
            "spawn-child-and-hang",
            "timeout",
            JobStatus::Timeout,
        ),
    ];

    let mut timeout_child_pid = None;
    for (job_label, mode, failure_type, expected_status) in cases {
        let job = database.create_and_start(job_label).await;
        let mut job_request = request.clone();
        job_request.job_id = job.job_id.clone();
        let error = isolated_launcher(path.path())
            .timeout(Duration::from_millis(300))
            .env(SPIKE_MODE_ENV, mode)
            .run(&job_request)
            .await
            .expect_err("fault-injected worker fails");
        if let WorkerLaunchError::Timeout { stderr, .. } = &error {
            timeout_child_pid = Some(parse_child_pid(stderr));
        }
        database.record_failure(&job.job_id, &error).await;
        let row = database.get(&job.job_id).await;
        assert_eq!(row.status, expected_status, "status for {job_label}");
        assert_eq!(
            row.status_details.as_ref().unwrap()["failure_type"],
            failure_type,
            "details for {job_label}"
        );
        if expected_status == JobStatus::Failed {
            assert!(!row.result.as_deref().unwrap_or_default().is_empty());
        } else {
            assert_eq!(row.result, None);
        }
    }

    let child_pid = timeout_child_pid.expect("timeout worker reported its child PID");
    wait_until_process_is_dead(child_pid).await;

    let failure_types: Vec<String> = database
        .all_details()
        .await
        .into_iter()
        .map(|details| details["failure_type"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        failure_types,
        ["malformed_output", "non_zero_exit", "signal", "timeout"]
    );
}

fn worker_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mlflow-genai-worker"))
}

fn empty_path() -> TempDir {
    tempfile::tempdir().expect("empty PATH directory")
}

fn isolated_launcher(path: &Path) -> WorkerLauncher {
    WorkerLauncher::new(worker_path())
        .clean_environment()
        .env("PATH", path.as_os_str())
}

fn assert_python_is_unreachable(path: &Path) {
    for executable in ["python", "python3"] {
        let error = std::process::Command::new(executable)
            .env_clear()
            .env("PATH", path)
            .status()
            .expect_err("scrubbed PATH must not resolve Python");
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}

fn invoke_request(
    job_id: &str,
    serialized_scorer: &str,
    outputs: Value,
    gateway_url: Option<&str>,
) -> WorkerRequest {
    WorkerRequest {
        protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
        job_id: job_id.to_string(),
        job_kind: JobKind::InvokeScorer,
        params: json!({
            "experiment_id": "spike-experiment",
            "serialized_scorer": serialized_scorer,
            "inputs": null,
            "outputs": outputs,
            "expectations": null,
            "gateway_url": gateway_url
        }),
        workspace: Some("spike-workspace".to_string()),
        subject: json!({"username": "spike-user"}),
    }
}

fn fixture(contents: &str) -> Value {
    serde_json::from_str(contents).expect("valid JSON fixture")
}

fn parse_child_pid(stderr: &str) -> i32 {
    stderr
        .lines()
        .find_map(|line| line.strip_prefix("child_pid="))
        .expect("child PID line")
        .parse()
        .expect("numeric child PID")
}

async fn wait_until_process_is_dead(process_id: i32) {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    for _ in 0..40 {
        if kill(Pid::from_raw(process_id), None) == Err(Errno::ESRCH) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("process-group child {process_id} survived timeout kill");
}

/// Test fixture over the production jobs store and migrated shared schema.
struct SpikeJobDatabase {
    store: JobStore,
    _database: TempDb,
}

impl SpikeJobDatabase {
    async fn new() -> Self {
        let database = TempDb::new("genai_worker_jobs").await;
        let store = JobStore::new(database.connect().await);
        Self {
            store,
            _database: database,
        }
    }

    async fn create_and_start(&self, label: &str) -> Job {
        let job = self
            .store
            .create_job(
                "spike-workspace",
                "invoke_scorer",
                &json!({"label": label}).to_string(),
                Some(0.3),
            )
            .await
            .expect("pending job inserted");
        self.store
            .start_job("spike-workspace", &job.job_id)
            .await
            .expect("job starts");
        job
    }

    async fn record_failure(&self, job_id: &str, error: &WorkerLaunchError) {
        let (status, details) = failure_record(error);
        self.store
            .update_status_details("spike-workspace", job_id, &details)
            .await
            .expect("failure details recorded");
        match status {
            JobStatus::Failed => {
                self.store
                    .fail_job("spike-workspace", job_id, &error.to_string())
                    .await
                    .expect("failure recorded");
            }
            JobStatus::Timeout => {
                self.store
                    .mark_job_timed_out("spike-workspace", job_id)
                    .await
                    .expect("timeout recorded");
            }
            other => panic!("unexpected failure status: {other}"),
        }
    }

    async fn record_success(&self, job_id: &str, result: &Value) {
        self.store
            .finish_job("spike-workspace", job_id, &result.to_string())
            .await
            .expect("success recorded");
    }

    async fn get(&self, job_id: &str) -> Job {
        self.store
            .get_job("spike-workspace", job_id)
            .await
            .expect("job row")
    }

    async fn all_details(&self) -> Vec<Value> {
        let jobs = self
            .store
            .list_jobs("spike-workspace", None, &[], None, None, None)
            .await
            .expect("job details");
        let mut details = jobs
            .into_iter()
            .map(|job| job.status_details.expect("status details JSON"))
            .collect::<Vec<_>>();
        details.sort_by_key(|value| value["failure_type"].as_str().unwrap().to_string());
        details
    }
}

fn failure_record(error: &WorkerLaunchError) -> (JobStatus, Value) {
    match error {
        WorkerLaunchError::NonZeroExit { code, .. } => (
            JobStatus::Failed,
            json!({"failure_type": "non_zero_exit", "exit_code": code}),
        ),
        WorkerLaunchError::Signal { signal, .. } => (
            JobStatus::Failed,
            json!({"failure_type": "signal", "signal": signal}),
        ),
        WorkerLaunchError::MalformedOutput { message, .. } => (
            JobStatus::Failed,
            json!({"failure_type": "malformed_output", "message": message}),
        ),
        WorkerLaunchError::Timeout { timeout, .. } => (
            JobStatus::Timeout,
            json!({"failure_type": "timeout", "timeout_ms": timeout.as_millis()}),
        ),
        other => panic!("unexpected spike failure: {other}"),
    }
}
