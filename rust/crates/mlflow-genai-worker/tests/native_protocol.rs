#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use mlflow_genai::{
    WorkerLauncher, WorkerResponse, MLFLOW_GENAI_WORKER_FIXTURE, NATIVE_WORKER_PROTOCOL_VERSION,
};
use mlflow_server::job_runner::{
    JobExecutionRequest, JobExecutionResult, JobExecutor, JobFunction, JobRunner, JobRunnerConfig,
};
use mlflow_server::native_worker::NativeWorkerExecutor;
use mlflow_store::{python_json_dumps, Job, JobStatus, JobStore};
use mlflow_test_support::TempDb;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::Instant;

const WS: &str = "team-a";
const SPIKE_MODE_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_MODE";
const SPIKE_PID_FILE_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_PID_FILE";
const SPIKE_FD_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_FD";

#[tokio::test]
async fn six_kind_matrix_runs_through_db_runner_without_python() {
    let path = empty_path();
    assert_python_is_unreachable(path.path());
    let (_database, store) = store("native_worker_matrix").await;
    let executor = fixture_executor(path.path());
    let cases = fixture_cases();
    let functions = cases
        .iter()
        .map(|case| JobFunction::new(case.kind, 1))
        .collect();
    let mut jobs = Vec::new();
    for case in &cases {
        let job = store
            .create_job(WS, case.kind, &case.params.to_string(), None)
            .await
            .unwrap();
        jobs.push((job, case.expected.clone()));
    }
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(executor),
        functions,
        vec![WS.to_string()],
        runner_config(true),
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    for (job, expected) in jobs {
        let row = wait_finalized(&store, &job.job_id).await;
        assert_eq!(row.status, JobStatus::Succeeded, "{}", job.job_name);
        assert_eq!(
            row.result.as_deref(),
            Some(python_json_dumps(&expected, false).as_str()),
            "byte-compatible fixture result for {}",
            job.job_name
        );
    }
    runner.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runner_cap_is_worker_process_backpressure() {
    let path = empty_path();
    let (_database, store) = store("native_worker_backpressure").await;
    let launcher = isolated_launcher(path.path())
        .env(MLFLOW_GENAI_WORKER_FIXTURE, "1")
        .env(SPIKE_MODE_ENV, "delay");
    let executor = NativeWorkerExecutor::from_launcher(launcher);
    let mut job_ids = Vec::new();
    for index in 0..5 {
        job_ids.push(
            store
                .create_job(
                    WS,
                    "invoke_genai_evaluate",
                    &json!({
                        "run_id": format!("run-{index}"),
                        "trace_ids": [],
                        "serialized_scorers": [],
                    })
                    .to_string(),
                    None,
                )
                .await
                .unwrap()
                .job_id,
        );
    }
    let started = Instant::now();
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(executor),
        vec![JobFunction::new("invoke_genai_evaluate", 2)],
        vec![WS.to_string()],
        runner_config(true),
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    let mut peak_running = 0;
    loop {
        let rows = store
            .list_jobs(WS, Some("invoke_genai_evaluate"), &[], None, None, None)
            .await
            .unwrap();
        peak_running = peak_running.max(
            rows.iter()
                .filter(|row| row.status == JobStatus::Running)
                .count(),
        );
        if rows.iter().all(|row| row.status.is_finalized()) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(peak_running, 2);
    assert!(started.elapsed() >= Duration::from_millis(550));
    for job_id in job_ids {
        assert_eq!(
            store.get_job(WS, &job_id).await.unwrap().status,
            JobStatus::Succeeded
        );
    }
    runner.shutdown().await;
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn inherited_file_descriptors_are_closed_on_exec() {
    use std::os::fd::AsRawFd;

    let inherited = tempfile::tempfile().unwrap();
    let fd = inherited.as_raw_fd();
    // SAFETY: `fd` is owned by `inherited`; clearing CLOEXEC deliberately
    // constructs the leak condition the launcher must contain.
    let flags = unsafe { nix::libc::fcntl(fd, nix::libc::F_GETFD) };
    assert!(flags >= 0);
    assert_eq!(
        unsafe { nix::libc::fcntl(fd, nix::libc::F_SETFD, flags & !nix::libc::FD_CLOEXEC) },
        0
    );
    let result = isolated_launcher(empty_path().path())
        .env(MLFLOW_GENAI_WORKER_FIXTURE, "1")
        .env(SPIKE_MODE_ENV, "assert-fd-closed")
        .env(SPIKE_FD_ENV, fd.to_string())
        .run(&mlflow_genai::WorkerRequest {
            protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
            job_id: "fd-close".to_string(),
            job_kind: mlflow_genai::JobKind::InvokeGenaiEvaluate,
            params: json!({"run_id": "run", "trace_ids": [], "serialized_scorers": []}),
            workspace: Some(WS.to_string()),
            subject: json!("alice"),
        })
        .await
        .unwrap();
    assert_eq!(result["run_id"], "run");
}

#[tokio::test]
async fn version_and_unknown_kind_fail_before_fault_hook_with_distinct_codes() {
    let base = json!({
        "protocol_version": NATIVE_WORKER_PROTOCOL_VERSION,
        "job_id": "negative-envelope",
        "job_kind": "invoke_scorer",
        "params": "must never execute",
        "workspace": WS,
        "subject": "alice",
    });
    let mut wrong_version = base.clone();
    wrong_version["protocol_version"] = json!(NATIVE_WORKER_PROTOCOL_VERSION + 1);
    let version = raw_worker(&wrong_version, Some("nonzero")).await;
    assert_failure_code(version, "UNSUPPORTED_PROTOCOL_VERSION");

    let mut unknown_kind = base;
    unknown_kind["job_kind"] = json!("run_arbitrary_code");
    let kind = raw_worker(&unknown_kind, Some("nonzero")).await;
    assert_failure_code(kind, "UNKNOWN_JOB_KIND");

    let missing_worker = NativeWorkerExecutor::new("/definitely/missing/mlflow-genai-worker");
    let result = missing_worker
        .execute(JobExecutionRequest {
            job_id: "unknown-before-spawn".to_string(),
            job_name: "run_arbitrary_code".to_string(),
            params: json!({}),
            workspace: Some(WS.to_string()),
            subject: json!("alice"),
        })
        .await;
    let JobExecutionResult::Failed { details, .. } = result else {
        panic!("unknown runner kind must fail");
    };
    assert_eq!(details.unwrap()["failure_type"], "unknown_job_kind");
}

#[tokio::test]
async fn crash_signal_malformed_and_bounded_streams_are_persisted() {
    for (tag, mode, cap, failure_type) in [
        ("nonzero", "nonzero", 4096, "non_zero_exit"),
        ("signal", "signal", 4096, "signal"),
        ("malformed", "malformed", 4096, "malformed_output"),
        ("stdout-large", "stdout-large", 1024, "malformed_output"),
        (
            "stderr-large",
            "stderr-large-nonzero",
            1024,
            "non_zero_exit",
        ),
    ] {
        let row = run_fault(tag, mode, cap).await;
        assert_eq!(row.status, JobStatus::Failed, "{tag}");
        let details = row.status_details.unwrap();
        assert_eq!(details["failure_type"], failure_type, "{tag}");
        assert!(row.result.unwrap().starts_with("RuntimeError('"));
        if tag == "stdout-large" {
            let stdout = details["stdout"].as_str().unwrap();
            assert_eq!(stdout.len(), cap + "\n...[truncated]".len());
            assert!(stdout.ends_with("...[truncated]"));
        }
        if tag == "stderr-large" {
            let stderr = details["stderr"].as_str().unwrap();
            assert_eq!(stderr.len(), cap + "\n...[truncated]".len());
            assert!(stderr.ends_with("...[truncated]"));
        }
    }
}

#[tokio::test]
async fn timeout_and_cancel_kill_worker_process_groups_including_children() {
    let timeout_pid_file = tempfile::NamedTempFile::new().unwrap();
    let timeout_row = run_hanging_job(
        "native_worker_timeout",
        timeout_pid_file.path(),
        Some(0.08),
        false,
    )
    .await;
    assert_eq!(
        timeout_row.status,
        JobStatus::Timeout,
        "timeout row: {timeout_row:?}"
    );
    assert_eq!(
        timeout_row.status_details.as_ref().unwrap()["failure_type"],
        "timeout"
    );
    wait_until_process_is_dead(read_pid(timeout_pid_file.path())).await;

    let cancel_pid_file = tempfile::NamedTempFile::new().unwrap();
    let cancel_row =
        run_hanging_job("native_worker_cancel", cancel_pid_file.path(), None, true).await;
    assert_eq!(cancel_row.status, JobStatus::Canceled);
    wait_until_process_is_dead(read_pid(cancel_pid_file.path())).await;
}

struct FixtureCase {
    kind: &'static str,
    params: Value,
    expected: Value,
}

fn fixture_cases() -> Vec<FixtureCase> {
    vec![
        FixtureCase {
            kind: "invoke_scorer",
            params: json!({
                "experiment_id": "exp-score",
                "serialized_scorer": "{}",
                "trace_ids": ["trace-a", "trace-b"],
                "log_assessments": true,
                "username": "alice",
            }),
            expected: json!({
                "trace-a": {"assessments": [], "failures": []},
                "trace-b": {"assessments": [], "failures": []},
            }),
        },
        FixtureCase {
            kind: "run_online_trace_scorer",
            params: json!({"experiment_id": "exp-online", "online_scorers": []}),
            expected: Value::Null,
        },
        FixtureCase {
            kind: "run_online_session_scorer",
            params: json!({"experiment_id": "exp-session", "online_scorers": []}),
            expected: Value::Null,
        },
        FixtureCase {
            kind: "optimize_prompts",
            params: json!({
                "run_id": "run-opt",
                "experiment_id": "exp-opt",
                "prompt_uri": "prompts:/support/1",
                "dataset_id": "dataset-1",
                "optimizer_type": "gepa",
                "optimizer_config": {},
                "scorer_names": ["Correctness", "Safety"],
            }),
            expected: json!({
                "run_id": "run-opt",
                "source_prompt_uri": "prompts:/support/1",
                "optimized_prompt_uri": null,
                "optimizer_name": "GepaPromptOptimizer",
                "initial_eval_score": null,
                "final_eval_score": null,
                "dataset_id": "dataset-1",
                "scorer_names": ["Correctness", "Safety"],
            }),
        },
        FixtureCase {
            kind: "invoke_issue_detection",
            params: json!({
                "experiment_id": "exp-discovery",
                "trace_ids": ["trace-a", "trace-b", "trace-c"],
                "categories": ["quality", "safety"],
                "run_id": "run-discovery",
                "username": "alice",
            }),
            expected: json!({
                "summary": "fixture:team-a/\"alice\"/exp-discovery",
                "issues": 2,
                "total_traces_analyzed": 3,
                "total_cost_usd": 0.0,
            }),
        },
        FixtureCase {
            kind: "invoke_genai_evaluate",
            params: json!({
                "trace_ids": ["trace-a", "trace-b"],
                "serialized_scorers": ["scorer-a", "scorer-b", "scorer-c"],
                "run_id": "run-eval",
                "username": "alice",
            }),
            expected: json!({"run_id": "run-eval", "total_traces": 2, "scorer_count": 3}),
        },
    ]
}

async fn run_fault(tag: &str, mode: &str, cap: usize) -> Job {
    let path = empty_path();
    let (_database, store) = store(&format!("native_worker_{tag}")).await;
    let launcher = isolated_launcher(path.path())
        .max_output_bytes(cap)
        .env(SPIKE_MODE_ENV, mode);
    let executor = NativeWorkerExecutor::from_launcher(launcher);
    let job = store
        .create_job(
            WS,
            "invoke_genai_evaluate",
            r#"{"run_id":"run","trace_ids":[],"serialized_scorers":[]}"#,
            None,
        )
        .await
        .unwrap();
    let runner = start_one(&store, executor).await;
    let row = wait_finalized(&store, &job.job_id).await;
    runner.shutdown().await;
    row
}

async fn run_hanging_job(tag: &str, pid_file: &Path, timeout: Option<f64>, cancel: bool) -> Job {
    let path = empty_path();
    let (_database, store) = store(tag).await;
    let launcher = isolated_launcher(path.path())
        .timeout(Duration::from_secs(10))
        .env(SPIKE_MODE_ENV, "spawn-child-and-hang")
        .env(SPIKE_PID_FILE_ENV, pid_file);
    let executor = NativeWorkerExecutor::from_launcher(launcher);
    let job = store
        .create_job(
            WS,
            "invoke_genai_evaluate",
            r#"{"run_id":"run","trace_ids":[],"serialized_scorers":[]}"#,
            timeout,
        )
        .await
        .unwrap();
    let runner = start_one(&store, executor).await;
    wait_for_nonempty_file(pid_file).await;
    if cancel {
        store.cancel_job(WS, &job.job_id).await.unwrap();
    }
    let row = wait_finalized(&store, &job.job_id).await;
    runner.shutdown().await;
    row
}

async fn start_one(
    store: &JobStore,
    executor: NativeWorkerExecutor,
) -> mlflow_server::job_runner::JobRunnerHandle {
    JobRunner::new(
        store.clone(),
        Arc::new(executor),
        vec![JobFunction::new("invoke_genai_evaluate", 1)],
        vec![WS.to_string()],
        runner_config(true),
    )
    .start()
    .await
    .unwrap()
    .unwrap()
}

async fn raw_worker(request: &Value, spike_mode: Option<&str>) -> WorkerResponse {
    let mut command = Command::new(worker_path());
    command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(mode) = spike_mode {
        command.env(SPIKE_MODE_ENV, mode);
    }
    let mut child = command.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .await
        .unwrap();
    let output = child.wait_with_output().await.unwrap();
    assert!(output.status.success(), "fault hook ran before validation");
    serde_json::from_slice(&output.stdout).unwrap()
}

fn assert_failure_code(response: WorkerResponse, expected: &str) {
    let WorkerResponse::Failed { error, .. } = response else {
        panic!("negative envelope must fail");
    };
    assert_eq!(error.code, expected);
}

fn fixture_executor(path: &Path) -> NativeWorkerExecutor {
    NativeWorkerExecutor::from_launcher(
        isolated_launcher(path).env(MLFLOW_GENAI_WORKER_FIXTURE, "1"),
    )
    .tracking_uri("http://tracking.invalid")
    .gateway_uri("http://gateway.invalid")
    .internal_gateway_token("fixture-internal-token")
}

fn isolated_launcher(path: &Path) -> WorkerLauncher {
    WorkerLauncher::new(worker_path())
        .clean_environment()
        .env("PATH", path)
}

fn worker_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mlflow-genai-worker"))
}

fn empty_path() -> TempDir {
    tempfile::tempdir().unwrap()
}

fn assert_python_is_unreachable(path: &Path) {
    for executable in ["python", "python3"] {
        let error = std::process::Command::new(executable)
            .env_clear()
            .env("PATH", path)
            .status()
            .expect_err("production-style PATH must not resolve Python");
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}

fn runner_config(workspaces_enabled: bool) -> JobRunnerConfig {
    JobRunnerConfig {
        workspaces_enabled,
        queue_poll_interval: Duration::from_millis(5),
        status_poll_interval: Duration::from_millis(5),
        ..Default::default()
    }
}

async fn store(tag: &str) -> (TempDb, JobStore) {
    let database = TempDb::new(tag).await;
    let store = JobStore::new(database.connect().await);
    (database, store)
}

async fn wait_finalized(store: &JobStore, job_id: &str) -> Job {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let row = store.get_job(WS, job_id).await.unwrap();
        if row.status.is_finalized() {
            return row;
        }
        assert!(Instant::now() < deadline, "job {job_id} did not finalize");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn wait_for_nonempty_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if std::fs::metadata(path).is_ok_and(|metadata| metadata.len() > 0) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "worker child PID was not reported"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn read_pid(path: &Path) -> i32 {
    std::fs::read_to_string(path).unwrap().parse().unwrap()
}

async fn wait_until_process_is_dead(process_id: i32) {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    for _ in 0..100 {
        if kill(Pid::from_raw(process_id), None) == Err(Errno::ESRCH) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("process-group child {process_id} survived cancellation");
}
