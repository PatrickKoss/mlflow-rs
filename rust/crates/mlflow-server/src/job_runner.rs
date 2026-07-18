//! D20 database-backed job runner.
//!
//! The jobs table is the only queue. Function coordinators claim through
//! [`JobStore`]'s dialect-specific atomic operations and enforce the same
//! function-local worker limits, retry policy, timeout polling, exclusive
//! cancellation, and startup recovery as Python's Huey runner.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::FutureExt;
use mlflow_error::MlflowError;
use mlflow_store::{python_json_dumps, Job, JobStatus, JobStore};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::watch;
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::Instant;

use crate::ServerConfig;

pub const MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES: &str =
    "MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES";
pub const MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY: &str =
    "MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY";
pub const MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY: &str =
    "MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY";

/// Owned request passed across the T17.2 native-worker seam.
#[derive(Debug, Clone, PartialEq)]
pub struct JobExecutionRequest {
    pub job_id: String,
    pub job_name: String,
    pub params: Value,
    /// Python propagates the persisted row workspace only when workspace mode
    /// is enabled; single-tenant workers receive no workspace context.
    pub workspace: Option<String>,
    /// Python persists the authenticated username inside supported job params
    /// (the T16.5 jobs schema has no separate subject column). The runner
    /// exposes that persisted value explicitly for T17.2's worker protocol.
    pub subject: Value,
}

/// Semantic result returned by a job implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum JobExecutionResult {
    Succeeded(Value),
    Failed { error: String, transient: bool },
}

pub type JobExecutionFuture = Pin<Box<dyn Future<Output = JobExecutionResult> + Send + 'static>>;

/// Execution boundary implemented by the native subprocess launcher in T17.2.
/// Dropping the returned future is the hard cancel/timeout signal.
pub trait JobExecutor: Send + Sync + 'static {
    fn execute(&self, request: JobExecutionRequest) -> JobExecutionFuture;
}

/// Python `@job(..., exclusive=...)` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exclusive {
    None,
    AllParams,
    Params(Vec<String>),
}

/// The runner-visible subset of Python's `JobFunctionMetadata`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobFunction {
    pub name: String,
    pub max_workers: usize,
    pub exclusive: Exclusive,
}

impl JobFunction {
    pub fn new(name: impl Into<String>, max_workers: usize) -> Self {
        Self {
            name: name.into(),
            max_workers,
            exclusive: Exclusive::None,
        }
    }

    pub fn exclusive(mut self, exclusive: Exclusive) -> Self {
        self.exclusive = exclusive;
        self
    }
}

/// Runtime policy. Production defaults mirror Python; short intervals can be
/// injected by lifecycle tests without changing the durable semantics.
#[derive(Debug, Clone)]
pub struct JobRunnerConfig {
    pub enabled: bool,
    pub workspaces_enabled: bool,
    pub max_retries: i64,
    pub retry_base_delay: Duration,
    pub retry_max_delay: Duration,
    pub queue_poll_interval: Duration,
    pub status_poll_interval: Duration,
}

impl JobRunnerConfig {
    pub fn from_server_config(config: &ServerConfig) -> Result<Self, MlflowError> {
        Ok(Self {
            enabled: config.job_execution_enabled,
            workspaces_enabled: config.enable_workspaces,
            max_retries: env_i64(MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES, 3)?,
            retry_base_delay: env_duration(MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY, 15)?,
            retry_max_delay: env_duration(MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY, 60)?,
            queue_poll_interval: Duration::from_secs(1),
            status_poll_interval: Duration::from_secs(1),
        })
    }
}

impl Default for JobRunnerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            workspaces_enabled: false,
            max_retries: 3,
            retry_base_delay: Duration::from_secs(15),
            retry_max_delay: Duration::from_secs(60),
            queue_poll_interval: Duration::from_secs(1),
            status_poll_interval: Duration::from_secs(1),
        }
    }
}

/// A configured runner. `start` performs recovery before returning a handle.
pub struct JobRunner {
    store: JobStore,
    executor: Arc<dyn JobExecutor>,
    functions: Vec<JobFunction>,
    workspaces: Vec<String>,
    config: JobRunnerConfig,
}

impl JobRunner {
    pub fn new(
        store: JobStore,
        executor: Arc<dyn JobExecutor>,
        functions: Vec<JobFunction>,
        workspaces: Vec<String>,
        config: JobRunnerConfig,
    ) -> Self {
        Self {
            store,
            executor,
            functions,
            workspaces,
            config,
        }
    }

    /// Start only when the server gate is enabled. Disabled runners do not
    /// recover or claim rows, matching Python's process-level gate.
    pub async fn start(mut self) -> Result<Option<JobRunnerHandle>, MlflowError> {
        if !self.config.enabled {
            return Ok(None);
        }
        #[cfg(windows)]
        return Err(MlflowError::internal_error(
            "MLflow job backend does not support Windows system.",
        ));

        validate_functions(&self.functions)?;
        normalize_workspaces(&mut self.workspaces);
        self.recover_unfinished().await?;

        let locks = Arc::new(ExclusiveLocks::default());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut tasks = Vec::with_capacity(self.functions.len());
        for function in self.functions {
            tasks.push(tokio::spawn(run_function(
                self.store.clone(),
                Arc::clone(&self.executor),
                function,
                self.workspaces.clone(),
                self.config.clone(),
                Arc::clone(&locks),
                shutdown_rx.clone(),
            )));
        }
        Ok(Some(JobRunnerHandle {
            shutdown_tx,
            tasks: Some(tasks),
        }))
    }

    async fn recover_unfinished(&self) -> Result<(), MlflowError> {
        let startup_time = chrono::Utc::now().timestamp_millis();
        for workspace in &self.workspaces {
            let unfinished = self
                .store
                .list_jobs(
                    workspace,
                    None,
                    &[JobStatus::Pending, JobStatus::Running],
                    None,
                    Some(startup_time),
                    None,
                )
                .await?;
            for job in unfinished {
                if job.status == JobStatus::Running {
                    self.store.reset_job(workspace, &job.job_id).await?;
                }
            }
        }
        Ok(())
    }
}

/// Owns coordinator tasks and releases all execution futures on shutdown.
pub struct JobRunnerHandle {
    shutdown_tx: watch::Sender<bool>,
    tasks: Option<Vec<JoinHandle<()>>>,
}

impl JobRunnerHandle {
    pub async fn shutdown(mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(tasks) = self.tasks.take() {
            for task in tasks {
                let _ = task.await;
            }
        }
    }
}

impl Drop for JobRunnerHandle {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(tasks) = self.tasks.take() {
            for task in tasks {
                task.abort();
            }
        }
    }
}

#[derive(Debug)]
struct JobCompletion {
    retry: Option<(String, Instant)>,
}

async fn run_function(
    store: JobStore,
    executor: Arc<dyn JobExecutor>,
    function: JobFunction,
    workspaces: Vec<String>,
    config: JobRunnerConfig,
    locks: Arc<ExclusiveLocks>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut active = JoinSet::new();
    let mut delayed = HashMap::<String, Instant>::new();
    let mut workspace_cursor = 0;

    loop {
        while let Some(completed) = active.try_join_next() {
            record_completion(completed, &mut delayed);
        }
        let now = Instant::now();
        delayed.retain(|_, due| *due > now);

        while active.len() < function.max_workers {
            let excluded = delayed.keys().cloned().collect::<Vec<_>>();
            let claim = claim_across_workspaces(
                &store,
                &workspaces,
                &mut workspace_cursor,
                &function.name,
                &excluded,
            )
            .await;
            let claimed = match claim {
                Ok(Some(job)) => job,
                Ok(None) => break,
                Err(error) => {
                    tracing::error!(job_name = %function.name, %error, "job queue claim failed");
                    break;
                }
            };
            active.spawn(execute_claimed(
                store.clone(),
                Arc::clone(&executor),
                claimed,
                function.exclusive.clone(),
                config.clone(),
                Arc::clone(&locks),
            ));
        }

        let sleep_for = delayed
            .values()
            .min()
            .map(|due| due.saturating_duration_since(Instant::now()))
            .map(|until_retry| until_retry.min(config.queue_poll_interval))
            .unwrap_or(config.queue_poll_interval);

        if active.is_empty() {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        return;
                    }
                }
                () = tokio::time::sleep(sleep_for) => {}
            }
        } else {
            tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        active.abort_all();
                        while active.join_next().await.is_some() {}
                        return;
                    }
                }
                completed = active.join_next() => {
                    if let Some(completed) = completed {
                        record_completion(completed, &mut delayed);
                    }
                }
                () = tokio::time::sleep(sleep_for) => {}
            }
        }
    }
}

fn record_completion(
    completed: Result<JobCompletion, tokio::task::JoinError>,
    delayed: &mut HashMap<String, Instant>,
) {
    match completed {
        Ok(JobCompletion {
            retry: Some((job_id, due)),
        }) => {
            delayed.insert(job_id, due);
        }
        Ok(JobCompletion { retry: None }) => {}
        Err(error) => tracing::error!(%error, "job execution task panicked"),
    }
}

async fn claim_across_workspaces(
    store: &JobStore,
    workspaces: &[String],
    cursor: &mut usize,
    job_name: &str,
    excluded: &[String],
) -> Result<Option<Job>, MlflowError> {
    for _ in 0..workspaces.len() {
        let index = *cursor % workspaces.len();
        *cursor = (*cursor + 1) % workspaces.len();
        if let Some(job) = store
            .claim_next_job_excluding(&workspaces[index], Some(job_name), excluded)
            .await?
        {
            return Ok(Some(job));
        }
    }
    Ok(None)
}

async fn execute_claimed(
    store: JobStore,
    executor: Arc<dyn JobExecutor>,
    job: Job,
    exclusive: Exclusive,
    config: JobRunnerConfig,
    locks: Arc<ExclusiveLocks>,
) -> JobCompletion {
    let params = match serde_json::from_str::<Value>(&job.params) {
        Ok(Value::Object(params)) => Value::Object(params),
        Ok(_) => {
            fail_unless_finalized(&store, &job, "Job params must be a JSON object.").await;
            return JobCompletion { retry: None };
        }
        Err(error) => {
            fail_unless_finalized(&store, &job, &error.to_string()).await;
            return JobCompletion { retry: None };
        }
    };

    let _exclusive_guard = match exclusive_key(&job, &params, &exclusive, config.workspaces_enabled)
    {
        Some(key) => match locks.acquire(key.clone()) {
            Some(guard) => Some(guard),
            None => {
                tracing::info!(job_id = %job.job_id, %key, "skipping job because exclusive lock is held");
                let outcome = finalize_with_retry(&job.job_id, "cancel_job", || {
                    store.cancel_job(&job.workspace, &job.job_id)
                })
                .await;
                if let Err(error) = outcome {
                    tracing::error!(job_id = %job.job_id, %error, "failed to cancel locked job");
                }
                return JobCompletion { retry: None };
            }
        },
        None => None,
    };

    let subject = params.get("username").cloned().unwrap_or(Value::Null);
    let request = JobExecutionRequest {
        job_id: job.job_id.clone(),
        job_name: job.job_name.clone(),
        params,
        workspace: config.workspaces_enabled.then(|| job.workspace.clone()),
        subject,
    };
    let execution = std::panic::AssertUnwindSafe(executor.execute(request)).catch_unwind();
    tokio::pin!(execution);
    let started = Instant::now();

    loop {
        tokio::select! {
            biased;
            outcome = &mut execution => {
                return finish_execution(&store, &job, outcome, &config).await;
            }
            () = tokio::time::sleep(config.status_poll_interval) => {
                match store.get_job(&job.workspace, &job.job_id).await {
                    Ok(current) if current.status == JobStatus::Canceled => {
                        return JobCompletion { retry: None };
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::error!(job_id = %job.job_id, %error, "failed to poll job status");
                        fail_unless_finalized(&store, &job, &error.message).await;
                        return JobCompletion { retry: None };
                    }
                }
                if let Some(timeout) = job.timeout {
                    if started.elapsed().as_secs_f64() >= timeout {
                        let outcome = finalize_with_retry(&job.job_id, "mark_job_timed_out", || {
                            store.mark_job_timed_out(&job.workspace, &job.job_id)
                        })
                        .await;
                        if let Err(error) = outcome {
                            tracing::error!(job_id = %job.job_id, %error, "failed to mark timed-out job");
                        }
                        return JobCompletion { retry: None };
                    }
                }
            }
        }
    }
}

async fn finish_execution(
    store: &JobStore,
    job: &Job,
    outcome: Result<JobExecutionResult, Box<dyn std::any::Any + Send>>,
    config: &JobRunnerConfig,
) -> JobCompletion {
    match outcome {
        Ok(JobExecutionResult::Succeeded(value)) => {
            let result = python_json_dumps(&value, false);
            let outcome = finalize_with_retry(&job.job_id, "finish_job", || {
                store.finish_job(&job.workspace, &job.job_id, &result)
            })
            .await;
            if let Err(error) = outcome {
                tracing::error!(job_id = %job.job_id, %error, "failed to finish job");
            }
            JobCompletion { retry: None }
        }
        Ok(JobExecutionResult::Failed {
            error,
            transient: true,
        }) => match finalize_with_retry(&job.job_id, "retry_or_fail_job", || {
            store.retry_or_fail_job(&job.workspace, &job.job_id, &error, config.max_retries)
        })
        .await
        {
            Ok(Some(retry_count)) => JobCompletion {
                retry: Some((
                    job.job_id.clone(),
                    Instant::now() + retry_delay(config, retry_count),
                )),
            },
            Ok(None) => JobCompletion { retry: None },
            Err(store_error) => {
                tracing::error!(job_id = %job.job_id, error = %store_error, "failed to retry job");
                JobCompletion { retry: None }
            }
        },
        Ok(JobExecutionResult::Failed {
            error,
            transient: false,
        }) => {
            fail_unless_finalized(store, job, &error).await;
            JobCompletion { retry: None }
        }
        Err(payload) => {
            let error = panic_message(payload);
            fail_unless_finalized(store, job, &error).await;
            JobCompletion { retry: None }
        }
    }
}

fn retry_delay(config: &JobRunnerConfig, retry_count: i64) -> Duration {
    let exponent = u32::try_from(retry_count.saturating_sub(1)).unwrap_or(u32::MAX);
    let factor = 2_u32.checked_pow(exponent).unwrap_or(u32::MAX);
    config
        .retry_base_delay
        .saturating_mul(factor)
        .min(config.retry_max_delay)
}

async fn fail_unless_finalized(store: &JobStore, job: &Job, error: &str) {
    let outcome = finalize_with_retry(&job.job_id, "fail_job", || {
        store.fail_job(&job.workspace, &job.job_id, error)
    })
    .await;
    if let Err(store_error) = outcome {
        tracing::error!(job_id = %job.job_id, error = %store_error, "failed to fail job");
    }
}

/// Retry a job-finalization write on store errors.
///
/// The runner is a background loop: unlike an HTTP handler, there is no client
/// to surface a transient DB error to, and giving up strands the row in
/// RUNNING until a server restart's recovery pass. Concretely, SQLite's WAL
/// upgrade path returns SQLITE_BUSY_SNAPSHOT immediately (uncovered by
/// `busy_timeout`) when a deferred read-then-write transaction races another
/// writer — `retry_or_fail_job` does exactly that read-then-write. Each retry
/// runs a fresh transaction, which is the required remedy; the guarded
/// `status IN (PENDING, RUNNING)` updates make re-attempts safe.
async fn finalize_with_retry<T, F, Fut>(
    job_id: &str,
    op_name: &str,
    mut op: F,
) -> Result<T, MlflowError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, MlflowError>>,
{
    let mut attempt: u32 = 0;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(error) if attempt < 4 => {
                attempt += 1;
                tracing::warn!(
                    job_id,
                    op_name,
                    attempt,
                    %error,
                    "store error during job finalization; retrying"
                );
                tokio::time::sleep(Duration::from_millis(10 * u64::from(attempt))).await;
            }
            Err(error) => return Err(error),
        }
    }
}

fn exclusive_key(
    job: &Job,
    params: &Value,
    exclusive: &Exclusive,
    workspaces_enabled: bool,
) -> Option<String> {
    let params = params.as_object()?;
    let lock_params = match exclusive {
        Exclusive::None => return None,
        Exclusive::AllParams => Value::Object(params.clone()),
        Exclusive::Params(names) => Value::Object(
            params
                .iter()
                .filter(|(name, _)| names.contains(name))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect::<Map<_, _>>(),
        ),
    };
    let job_name = if workspaces_enabled {
        format!("{}:{}", job.workspace, job.job_name)
    } else {
        job.job_name.clone()
    };
    let encoded = python_json_dumps(&lock_params, true);
    let digest = Sha256::digest(encoded.as_bytes());
    let short_hash = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Some(format!("{job_name}:{short_hash}"))
}

#[derive(Default)]
struct ExclusiveLocks {
    held: Arc<Mutex<HashSet<String>>>,
}

impl ExclusiveLocks {
    fn acquire(&self, key: String) -> Option<ExclusiveGuard> {
        let mut held = self.held.lock().unwrap_or_else(|error| error.into_inner());
        if !held.insert(key.clone()) {
            return None;
        }
        Some(ExclusiveGuard {
            held: Arc::clone(&self.held),
            key,
        })
    }
}

struct ExclusiveGuard {
    held: Arc<Mutex<HashSet<String>>>,
    key: String,
}

impl Drop for ExclusiveGuard {
    fn drop(&mut self) {
        self.held
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.key);
    }
}

fn validate_functions(functions: &[JobFunction]) -> Result<(), MlflowError> {
    let mut names = HashSet::new();
    for function in functions {
        if function.max_workers == 0 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Job function {} must configure max_workers greater than zero.",
                function.name
            )));
        }
        if !names.insert(&function.name) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Duplicate job function name: {}",
                function.name
            )));
        }
    }
    Ok(())
}

fn normalize_workspaces(workspaces: &mut Vec<String>) {
    if workspaces.is_empty() {
        workspaces.push(mlflow_store::WORKSPACE_DEFAULT_NAME.to_string());
    }
    let mut seen = HashSet::new();
    workspaces.retain(|workspace| seen.insert(workspace.clone()));
}

fn env_i64(name: &str, default: i64) -> Result<i64, MlflowError> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<i64>().map_err(|error| {
        MlflowError::invalid_parameter_value(format!(
            "Failed to convert {value:?} for {name}: {error}"
        ))
    })
}

fn env_duration(name: &str, default: u64) -> Result<Duration, MlflowError> {
    let value = env_i64(name, i64::try_from(default).unwrap_or(i64::MAX))?;
    let seconds = u64::try_from(value).map_err(|_| {
        MlflowError::invalid_parameter_value(format!("{name} must not be negative."))
    })?;
    Ok(Duration::from_secs(seconds))
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "Job executor panicked".to_string()
    }
}
