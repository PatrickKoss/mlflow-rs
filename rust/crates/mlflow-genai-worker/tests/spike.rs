#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{extract::State, routing::post, Json, Router};
use mlflow_genai::{
    JobKind, WorkerLaunchError, WorkerLauncher, WorkerRequest, NATIVE_WORKER_PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
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

    database.create_and_start("job-builtin").await;
    let builtin = launcher
        .run(&invoke_request(
            "job-builtin",
            BUILTIN_SCORER,
            json!("native Rust worker works"),
            None,
        ))
        .await
        .expect("builtin worker succeeds without Python");
    assert_eq!(builtin, fixture(BUILTIN_EXPECTED));
    database.record_success("job-builtin", &builtin).await;
    assert_eq!(database.get("job-builtin").await.status, 2);

    database.create_and_start("job-instructions").await;
    let instructions = launcher
        .run(&invoke_request(
            "job-instructions",
            INSTRUCTIONS_SCORER,
            json!("Brief answer."),
            Some(&gateway.url),
        ))
        .await
        .expect("instructions worker succeeds without Python");
    assert_eq!(instructions, fixture(INSTRUCTIONS_EXPECTED));
    database
        .record_success("job-instructions", &instructions)
        .await;
    let instructions_job = database.get("job-instructions").await;
    assert_eq!(instructions_job.status, 2);
    assert_eq!(
        serde_json::from_str::<Value>(&instructions_job.result).expect("job result JSON"),
        instructions
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
        ("job-nonzero", "nonzero", "non_zero_exit", 3_i64),
        ("job-signal", "signal", "signal", 3_i64),
        ("job-malformed", "malformed", "malformed_output", 3_i64),
        ("job-timeout", "spawn-child-and-hang", "timeout", 4_i64),
    ];

    let mut timeout_child_pid = None;
    for (job_id, mode, failure_type, expected_status) in cases {
        database.create_and_start(job_id).await;
        let mut job_request = request.clone();
        job_request.job_id = job_id.to_string();
        let error = isolated_launcher(path.path())
            .timeout(Duration::from_millis(300))
            .env(SPIKE_MODE_ENV, mode)
            .run(&job_request)
            .await
            .expect_err("fault-injected worker fails");
        if let WorkerLaunchError::Timeout { stderr, .. } = &error {
            timeout_child_pid = Some(parse_child_pid(stderr));
        }
        database.record_failure(job_id, &error).await;
        let row = database.get(job_id).await;
        assert_eq!(row.status, expected_status, "status for {job_id}");
        assert_eq!(
            row.status_details["failure_type"], failure_type,
            "details for {job_id}"
        );
        assert!(row.result.contains(failure_type) || !row.result.is_empty());
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
        workspace: "spike-workspace".to_string(),
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

/// Scratch-only T15.4 table adapter. T17 replaces this with `mlflow-store`.
struct SpikeJobDatabase {
    pool: SqlitePool,
    _directory: TempDir,
}

struct JobRow {
    status: i64,
    result: String,
    status_details: Value,
}

impl SpikeJobDatabase {
    async fn new() -> Self {
        let directory = tempfile::tempdir().expect("scratch jobs directory");
        let url = format!(
            "sqlite://{}?mode=rwc",
            directory.path().join("jobs.db").display()
        );
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("scratch jobs database");
        sqlx::query(
            "CREATE TABLE jobs (\
                id TEXT PRIMARY KEY, creation_time INTEGER NOT NULL, job_name TEXT NOT NULL, \
                params TEXT NOT NULL, workspace TEXT NOT NULL, timeout REAL, status INTEGER NOT NULL, \
                result TEXT, retry_count INTEGER NOT NULL, last_update_time INTEGER NOT NULL, \
                status_details TEXT\
            )",
        )
        .execute(&pool)
        .await
        .expect("scratch jobs table");
        Self {
            pool,
            _directory: directory,
        }
    }

    async fn create_and_start(&self, job_id: &str) {
        sqlx::query(
            "INSERT INTO jobs \
             (id, creation_time, job_name, params, workspace, timeout, status, result, retry_count, last_update_time, status_details) \
             VALUES (?, 1, 'invoke_scorer', '{}', 'spike-workspace', 0.3, 0, NULL, 0, 1, NULL)",
        )
        .bind(job_id)
        .execute(&self.pool)
        .await
        .expect("pending job inserted");
        let changed = sqlx::query(
            "UPDATE jobs SET status = 1, last_update_time = 2 WHERE id = ? AND status = 0",
        )
        .bind(job_id)
        .execute(&self.pool)
        .await
        .expect("job starts")
        .rows_affected();
        assert_eq!(changed, 1);
    }

    async fn record_failure(&self, job_id: &str, error: &WorkerLaunchError) {
        let (status, details) = failure_record(error);
        sqlx::query(
            "UPDATE jobs SET status = ?, result = ?, status_details = ?, last_update_time = 3 WHERE id = ?",
        )
        .bind(status)
        .bind(error.to_string())
        .bind(details.to_string())
        .bind(job_id)
        .execute(&self.pool)
        .await
        .expect("failure recorded");
    }

    async fn record_success(&self, job_id: &str, result: &Value) {
        sqlx::query(
            "UPDATE jobs SET status = 2, result = ?, status_details = '{}', last_update_time = 3 WHERE id = ? AND status = 1",
        )
        .bind(result.to_string())
        .bind(job_id)
        .execute(&self.pool)
        .await
        .expect("success recorded");
    }

    async fn get(&self, job_id: &str) -> JobRow {
        let (status, result, details): (i64, String, String) =
            sqlx::query_as("SELECT status, result, status_details FROM jobs WHERE id = ?")
                .bind(job_id)
                .fetch_one(&self.pool)
                .await
                .expect("job row");
        JobRow {
            status,
            result,
            status_details: serde_json::from_str(&details).expect("status details JSON"),
        }
    }

    async fn all_details(&self) -> Vec<Value> {
        let details: Vec<(String,)> = sqlx::query_as("SELECT status_details FROM jobs ORDER BY id")
            .fetch_all(&self.pool)
            .await
            .expect("job details");
        details
            .into_iter()
            .map(|(details,)| serde_json::from_str(&details).expect("status details JSON"))
            .collect()
    }
}

fn failure_record(error: &WorkerLaunchError) -> (i64, Value) {
    match error {
        WorkerLaunchError::NonZeroExit { code, .. } => (
            3,
            json!({"failure_type": "non_zero_exit", "exit_code": code}),
        ),
        WorkerLaunchError::Signal { signal, .. } => {
            (3, json!({"failure_type": "signal", "signal": signal}))
        }
        WorkerLaunchError::MalformedOutput { message, .. } => (
            3,
            json!({"failure_type": "malformed_output", "message": message}),
        ),
        WorkerLaunchError::Timeout { timeout, .. } => (
            4,
            json!({"failure_type": "timeout", "timeout_ms": timeout.as_millis()}),
        ),
        other => panic!("unexpected spike failure: {other}"),
    }
}
