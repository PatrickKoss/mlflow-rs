//! T17.1 DB-queue runner lifecycle contract.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mlflow_server::job_runner::{
    Exclusive, JobExecutionFuture, JobExecutionRequest, JobExecutionResult, JobExecutor,
    JobFunction, JobRunner, JobRunnerConfig,
};
use mlflow_store::{JobStatus, JobStore};
use mlflow_test_support::TempDb;
use serde_json::{json, Value};
use tokio::time::Instant;

const WS: &str = "default";

type TestFunction = Arc<dyn Fn(JobExecutionRequest) -> JobExecutionFuture + Send + Sync>;

#[derive(Default)]
struct TestRegistry {
    functions: HashMap<String, TestFunction>,
}

impl TestRegistry {
    fn register<F>(&mut self, name: &str, function: F)
    where
        F: Fn(JobExecutionRequest) -> JobExecutionFuture + Send + Sync + 'static,
    {
        self.functions.insert(name.to_string(), Arc::new(function));
    }
}

impl JobExecutor for TestRegistry {
    fn execute(&self, request: JobExecutionRequest) -> JobExecutionFuture {
        match self.functions.get(&request.job_name).cloned() {
            Some(function) => function(request),
            None => Box::pin(async move {
                JobExecutionResult::Failed {
                    error: format!("Invalid job name: {}", request.job_name),
                    transient: false,
                    details: None,
                }
            }),
        }
    }
}

fn config() -> JobRunnerConfig {
    JobRunnerConfig {
        queue_poll_interval: Duration::from_millis(5),
        status_poll_interval: Duration::from_millis(5),
        retry_base_delay: Duration::from_millis(30),
        retry_max_delay: Duration::from_millis(40),
        ..Default::default()
    }
}

async fn store(tag: &str) -> (TempDb, JobStore) {
    let temp = TempDb::new(tag).await;
    let store = JobStore::new(temp.connect().await);
    (temp, store)
}

async fn wait_for_status(store: &JobStore, workspace: &str, job_id: &str, status: JobStatus) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let job = store.get_job(workspace, job_id).await.unwrap();
        if job.status == status {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "job {job_id} stayed in {} instead of reaching {status}",
            job.status
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn wait_finalized(store: &JobStore, workspace: &str, job_id: &str) -> JobStatus {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let row = store.get_job(workspace, job_id).await.unwrap();
        let status = row.status;
        if status.is_finalized() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "job {job_id} ({}) stayed in {status}",
            row.job_name
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn succeeds_fails_propagates_row_context_and_creates_no_queue_files() {
    let (_temp, store) = store("runner_lifecycle").await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let mut registry = TestRegistry::default();
    let observed_for_success = Arc::clone(&observed);
    registry.register("success", move |request| {
        let observed = Arc::clone(&observed_for_success);
        Box::pin(async move {
            observed
                .lock()
                .unwrap()
                .push((request.workspace, request.subject, request.params));
            JobExecutionResult::Succeeded(json!({"answer": 7}))
        })
    });
    registry.register("failure", |_| {
        Box::pin(async {
            JobExecutionResult::Failed {
                error: "RuntimeError()".to_string(),
                transient: false,
                details: None,
            }
        })
    });

    let success = store
        .create_job(WS, "success", r#"{"x": 3, "username": "alice"}"#, None)
        .await
        .unwrap();
    let failure = store.create_job(WS, "failure", "{}", None).await.unwrap();
    let queue_dir = tempfile::tempdir().unwrap();
    // SAFETY: no other test in this binary reads the retired Huey variable.
    unsafe {
        std::env::set_var("_MLFLOW_HUEY_STORAGE_PATH", queue_dir.path());
    }
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(registry),
        vec![
            JobFunction::new("success", 1),
            JobFunction::new("failure", 1),
        ],
        vec![WS.to_string()],
        config(),
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    assert_eq!(
        wait_finalized(&store, WS, &success.job_id).await,
        JobStatus::Succeeded
    );
    assert_eq!(
        wait_finalized(&store, WS, &failure.job_id).await,
        JobStatus::Failed
    );
    let succeeded = store.get_job(WS, &success.job_id).await.unwrap();
    assert_eq!(succeeded.result.as_deref(), Some(r#"{"answer": 7}"#));
    assert_eq!(
        store
            .get_job(WS, &failure.job_id)
            .await
            .unwrap()
            .result
            .as_deref(),
        Some("RuntimeError()")
    );
    assert_eq!(
        *observed.lock().unwrap(),
        vec![(None, json!("alice"), json!({"x": 3, "username": "alice"}))]
    );
    runner.shutdown().await;

    let files = std::fs::read_dir(queue_dir.path())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(files.is_empty(), "the DB queue must create no queue files");
    // SAFETY: restore the process environment changed above.
    unsafe {
        std::env::remove_var("_MLFLOW_HUEY_STORAGE_PATH");
    }
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn exclusive_duplicate_is_canceled_and_cancel_drops_locked_execution() {
    let (_temp, store) = store("runner_exclusive_cancel").await;
    let dropped = Arc::new(AtomicBool::new(false));
    let mut registry = TestRegistry::default();
    let dropped_for_job = Arc::clone(&dropped);
    registry.register("exclusive", move |_| {
        let guard = DropFlag(Arc::clone(&dropped_for_job));
        Box::pin(async move {
            let _guard = guard;
            std::future::pending::<JobExecutionResult>().await
        })
    });
    let first = store
        .create_job(WS, "exclusive", r#"{"experiment_id":"1"}"#, None)
        .await
        .unwrap();
    let second = store
        .create_job(WS, "exclusive", r#"{"experiment_id":"1"}"#, None)
        .await
        .unwrap();
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(registry),
        vec![JobFunction::new("exclusive", 2)
            .exclusive(Exclusive::Params(vec!["experiment_id".to_string()]))],
        vec![WS.to_string()],
        config(),
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let running_id = loop {
        let first_status = store.get_job(WS, &first.job_id).await.unwrap().status;
        let second_status = store.get_job(WS, &second.job_id).await.unwrap().status;
        if first_status == JobStatus::Canceled && second_status == JobStatus::Running {
            break second.job_id.clone();
        }
        if second_status == JobStatus::Canceled && first_status == JobStatus::Running {
            break first.job_id.clone();
        }
        assert!(Instant::now() < deadline, "exclusive jobs did not settle");
        tokio::time::sleep(Duration::from_millis(5)).await;
    };

    store.cancel_job(WS, &running_id).await.unwrap();
    wait_for_status(&store, WS, &running_id, JobStatus::Canceled).await;
    let deadline = Instant::now() + Duration::from_secs(1);
    while !dropped.load(Ordering::SeqCst) {
        assert!(
            Instant::now() < deadline,
            "canceled executor was not dropped"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    runner.shutdown().await;
}

#[tokio::test]
async fn timeout_marks_timeout_and_drops_execution() {
    let (_temp, store) = store("runner_timeout").await;
    let dropped = Arc::new(AtomicBool::new(false));
    let mut registry = TestRegistry::default();
    let dropped_for_job = Arc::clone(&dropped);
    registry.register("timeout", move |_| {
        let guard = DropFlag(Arc::clone(&dropped_for_job));
        Box::pin(async move {
            let _guard = guard;
            std::future::pending::<JobExecutionResult>().await
        })
    });
    let job = store
        .create_job(WS, "timeout", "{}", Some(0.03))
        .await
        .unwrap();
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(registry),
        vec![JobFunction::new("timeout", 1)],
        vec![WS.to_string()],
        config(),
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    wait_for_status(&store, WS, &job.job_id, JobStatus::Timeout).await;
    assert_eq!(store.get_job(WS, &job.job_id).await.unwrap().result, None);
    assert!(dropped.load(Ordering::SeqCst));
    runner.shutdown().await;
}

#[tokio::test]
async fn transient_retries_use_python_counter_and_backoff_policy() {
    let (_temp, store) = store("runner_retry").await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let timestamps = Arc::new(Mutex::new(Vec::new()));
    let mut registry = TestRegistry::default();
    let attempts_for_job = Arc::clone(&attempts);
    let timestamps_for_job = Arc::clone(&timestamps);
    registry.register("eventual", move |_| {
        let attempt = attempts_for_job.fetch_add(1, Ordering::SeqCst) + 1;
        timestamps_for_job.lock().unwrap().push(Instant::now());
        Box::pin(async move {
            if attempt <= 2 {
                JobExecutionResult::Failed {
                    error: "RuntimeError('transient')".to_string(),
                    transient: true,
                    details: None,
                }
            } else {
                JobExecutionResult::Succeeded(json!(100))
            }
        })
    });
    registry.register("exhausted", |_| {
        Box::pin(async {
            JobExecutionResult::Failed {
                error: "RuntimeError('always')".to_string(),
                transient: true,
                details: None,
            }
        })
    });
    let eventual = store.create_job(WS, "eventual", "{}", None).await.unwrap();
    let exhausted = store.create_job(WS, "exhausted", "{}", None).await.unwrap();
    let mut runner_config = config();
    runner_config.max_retries = 2;
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(registry),
        vec![
            JobFunction::new("eventual", 1),
            JobFunction::new("exhausted", 1),
        ],
        vec![WS.to_string()],
        runner_config,
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    assert_eq!(
        wait_finalized(&store, WS, &eventual.job_id).await,
        JobStatus::Succeeded
    );
    assert_eq!(
        wait_finalized(&store, WS, &exhausted.job_id).await,
        JobStatus::Failed
    );
    let eventual = store.get_job(WS, &eventual.job_id).await.unwrap();
    let exhausted = store.get_job(WS, &exhausted.job_id).await.unwrap();
    assert_eq!(eventual.retry_count, 2);
    assert_eq!(eventual.result.as_deref(), Some("100"));
    assert_eq!(exhausted.retry_count, 2);
    assert_eq!(exhausted.result.as_deref(), Some("RuntimeError('always')"));

    {
        let timestamps = timestamps.lock().unwrap();
        assert_eq!(timestamps.len(), 3);
        assert!(timestamps[1].duration_since(timestamps[0]) >= Duration::from_millis(25));
        assert!(timestamps[2].duration_since(timestamps[1]) >= Duration::from_millis(35));
    }
    runner.shutdown().await;
}

#[derive(Default)]
struct ConcurrencyStats {
    active: AtomicUsize,
    peak: AtomicUsize,
}

impl ConcurrencyStats {
    fn enter(&self) {
        let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(current, Ordering::SeqCst);
    }

    fn leave(&self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn per_function_max_workers_are_independent_and_saturate() {
    let (_temp, store) = store("runner_caps").await;
    let cap_two = Arc::new(ConcurrencyStats::default());
    let cap_three = Arc::new(ConcurrencyStats::default());
    let mut registry = TestRegistry::default();
    for (name, stats) in [
        ("cap_two", Arc::clone(&cap_two)),
        ("cap_three", Arc::clone(&cap_three)),
    ] {
        registry.register(name, move |_| {
            let stats = Arc::clone(&stats);
            Box::pin(async move {
                stats.enter();
                tokio::time::sleep(Duration::from_millis(60)).await;
                stats.leave();
                JobExecutionResult::Succeeded(Value::Null)
            })
        });
    }
    let mut job_ids = Vec::new();
    for _ in 0..4 {
        job_ids.push(
            store
                .create_job(WS, "cap_two", "{}", None)
                .await
                .unwrap()
                .job_id,
        );
    }
    for _ in 0..6 {
        job_ids.push(
            store
                .create_job(WS, "cap_three", "{}", None)
                .await
                .unwrap()
                .job_id,
        );
    }
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(registry),
        vec![
            JobFunction::new("cap_two", 2),
            JobFunction::new("cap_three", 3),
        ],
        vec![WS.to_string()],
        config(),
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    for job_id in job_ids {
        assert_eq!(
            wait_finalized(&store, WS, &job_id).await,
            JobStatus::Succeeded
        );
    }
    assert_eq!(cap_two.peak.load(Ordering::SeqCst), 2);
    assert_eq!(cap_three.peak.load(Ordering::SeqCst), 3);
    runner.shutdown().await;
}

#[tokio::test]
async fn startup_recovers_running_and_pending_rows_in_each_workspace() {
    let (_temp, store) = store("runner_recovery").await;
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut registry = TestRegistry::default();
    let requests_for_job = Arc::clone(&requests);
    registry.register("recover", move |request| {
        requests_for_job
            .lock()
            .unwrap()
            .push((request.job_id.clone(), request.workspace));
        Box::pin(async { JobExecutionResult::Succeeded(Value::Null) })
    });
    let running = store
        .create_job("team-a", "recover", "{}", None)
        .await
        .unwrap();
    store.start_job("team-a", &running.job_id).await.unwrap();
    let pending = store
        .create_job("team-b", "recover", "{}", None)
        .await
        .unwrap();
    let mut runner_config = config();
    runner_config.workspaces_enabled = true;
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(registry),
        vec![JobFunction::new("recover", 1)],
        vec!["team-a".to_string(), "team-b".to_string()],
        runner_config,
    )
    .start()
    .await
    .unwrap()
    .unwrap();

    assert_eq!(
        wait_finalized(&store, "team-a", &running.job_id).await,
        JobStatus::Succeeded
    );
    assert_eq!(
        wait_finalized(&store, "team-b", &pending.job_id).await,
        JobStatus::Succeeded
    );
    {
        let requests = requests.lock().unwrap();
        assert!(requests.contains(&(running.job_id, Some("team-a".to_string()))));
        assert!(requests.contains(&(pending.job_id, Some("team-b".to_string()))));
    }
    runner.shutdown().await;
}

#[tokio::test]
async fn disabled_gate_does_not_recover_or_claim() {
    let (_temp, store) = store("runner_gate").await;
    let pending = store.create_job(WS, "disabled", "{}", None).await.unwrap();
    let running = store.create_job(WS, "disabled", "{}", None).await.unwrap();
    store.start_job(WS, &running.job_id).await.unwrap();
    let mut runner_config = config();
    runner_config.enabled = false;
    let runner = JobRunner::new(
        store.clone(),
        Arc::new(TestRegistry::default()),
        vec![JobFunction::new("disabled", 1)],
        vec![WS.to_string()],
        runner_config,
    )
    .start()
    .await
    .unwrap();
    assert!(runner.is_none());
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        store.get_job(WS, &pending.job_id).await.unwrap().status,
        JobStatus::Pending
    );
    assert_eq!(
        store.get_job(WS, &running.job_id).await.unwrap().status,
        JobStatus::Running
    );
}
