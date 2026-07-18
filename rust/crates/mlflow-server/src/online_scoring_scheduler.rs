//! Periodic online-scoring job submission.
//!
//! This module stops at the Python scheduler's submission boundary. The
//! submitted trace/session jobs remain `PENDING`; Phase 19 executes the native
//! sampler/checkpoint processors. Their deterministic primitives live here now
//! because they are part of the scheduler parity contract and differential
//! seam, but no scorer execution is started by this task.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mlflow_error::MlflowError;
use mlflow_genai::SerializedScorer;
use mlflow_store::{
    python_json_dumps, JobStore, OnlineScorer, TrackingStore, WorkspaceStore,
    WORKSPACE_DEFAULT_NAME,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::time::{Instant, MissedTickBehavior};
use tracing::{debug, info, warn};
use uuid::Uuid;

pub const ONLINE_SCORING_SCHEDULER_LOCK: &str = "online-scoring-scheduler-lock";
pub const ONLINE_TRACE_SCORER_JOB_NAME: &str = "run_online_trace_scorer";
pub const ONLINE_SESSION_SCORER_JOB_NAME: &str = "run_online_session_scorer";
pub const TRACE_CHECKPOINT_TAG: &str = "mlflow.latestOnlineScoring.trace.checkpoint";
pub const SESSION_CHECKPOINT_TAG: &str = "mlflow.latestOnlineScoring.session.checkpoint";
pub const MAX_LOOKBACK_MS: i64 = 60 * 60 * 1000;
pub const MAX_TRACES_PER_JOB: usize = 500;
pub const MAX_SESSIONS_PER_JOB: usize = 100;

const PERIOD: Duration = Duration::from_secs(60);
const LOCK_LEASE_MS: i64 = 5 * 60 * 1000;

#[derive(Debug, Clone)]
pub struct OnlineScoringScheduler {
    tracking_store: TrackingStore,
    job_store: JobStore,
    workspace_store: Option<WorkspaceStore>,
}

impl OnlineScoringScheduler {
    pub fn new(tracking_store: TrackingStore, workspace_store: Option<WorkspaceStore>) -> Self {
        let job_store = JobStore::new(tracking_store.db().clone());
        Self {
            tracking_store,
            job_store,
            workspace_store,
        }
    }

    /// Run one locked scheduling pass with an injected seed. Returning zero
    /// means either there was nothing due or another server held the DB lock.
    pub async fn run_once(&self, seed: u64) -> Result<usize, MlflowError> {
        let Some(lock) = self
            .job_store
            .try_acquire_periodic_scheduler_lock(ONLINE_SCORING_SCHEDULER_LOCK, LOCK_LEASE_MS)
            .await?
        else {
            debug!("online scoring scheduler DB lock is held; skipping pass");
            return Ok(0);
        };

        let result = self.run_unlocked(seed).await;
        let release = self.job_store.release_periodic_scheduler_lock(&lock).await;
        match (result, release) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(count), Ok(())) => Ok(count),
        }
    }

    async fn run_unlocked(&self, seed: u64) -> Result<usize, MlflowError> {
        let workspaces = match &self.workspace_store {
            Some(store) => store
                .list_workspaces()
                .await?
                .into_iter()
                .map(|workspace| workspace.name)
                .collect(),
            None => vec![WORKSPACE_DEFAULT_NAME.to_string()],
        };
        if workspaces.is_empty() {
            info!("online scoring scheduler found no workspaces; skipping");
            return Ok(0);
        }

        let mut submitted = 0;
        let mut rng = SplitMix64::new(seed);
        for workspace in workspaces {
            let active = self
                .tracking_store
                .get_active_online_scorers(&workspace)
                .await?;
            let mut groups = group_scorers(active);
            shuffle_groups(&mut groups, &mut rng);
            debug!(
                workspace,
                experiments = groups.len(),
                "grouped active online scorers"
            );

            for group in groups {
                let mut trace_scorers = Vec::new();
                let mut session_scorers = Vec::new();
                for scorer in group.scorers {
                    match SerializedScorer::from_json(&scorer.serialized_scorer) {
                        Ok(parsed) if parsed.common().is_session_level_scorer => {
                            session_scorers.push(scorer);
                        }
                        Ok(_) => trace_scorers.push(scorer),
                        Err(error) => warn!(
                            scorer = scorer.name,
                            %error,
                            "failed to load online scorer; skipping"
                        ),
                    }
                }

                if !trace_scorers.is_empty() {
                    self.submit(
                        &workspace,
                        ONLINE_TRACE_SCORER_JOB_NAME,
                        &group.experiment_id,
                        trace_scorers,
                    )
                    .await?;
                    submitted += 1;
                }
                if !session_scorers.is_empty() {
                    self.submit(
                        &workspace,
                        ONLINE_SESSION_SCORER_JOB_NAME,
                        &group.experiment_id,
                        session_scorers,
                    )
                    .await?;
                    submitted += 1;
                }
            }
        }
        Ok(submitted)
    }

    async fn submit(
        &self,
        workspace: &str,
        job_name: &str,
        experiment_id: &str,
        scorers: Vec<OnlineScorer>,
    ) -> Result<(), MlflowError> {
        let params = json!({
            "experiment_id": experiment_id,
            "online_scorers": scorers,
        });
        self.job_store
            .create_job(
                workspace,
                job_name,
                &python_json_dumps(&params, false),
                None,
            )
            .await?;
        Ok(())
    }

    /// Run at the next wall-clock minute and every minute thereafter, matching
    /// Huey's `crontab(minute="*/1")`. Missed ticks are skipped rather than
    /// replayed in a burst.
    pub async fn run_periodic(self) {
        let delay = duration_to_next_minute(SystemTime::now());
        let mut interval = tokio::time::interval_at(Instant::now() + delay, PERIOD);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let seed = entropy_seed();
            if let Err(error) = self.run_once(seed).await {
                warn!(%error, "online scoring scheduler pass failed");
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentGroup {
    pub experiment_id: String,
    pub scorers: Vec<OnlineScorer>,
}

pub fn group_and_shuffle_scorers(scorers: Vec<OnlineScorer>, seed: u64) -> Vec<ExperimentGroup> {
    let mut groups = group_scorers(scorers);
    shuffle_groups(&mut groups, &mut SplitMix64::new(seed));
    groups
}

fn group_scorers(scorers: Vec<OnlineScorer>) -> Vec<ExperimentGroup> {
    let mut groups: Vec<ExperimentGroup> = Vec::new();
    for scorer in scorers {
        let experiment_id = &scorer.online_config.experiment_id;
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.experiment_id == *experiment_id)
        {
            group.scorers.push(scorer);
        } else {
            groups.push(ExperimentGroup {
                experiment_id: experiment_id.clone(),
                scorers: vec![scorer],
            });
        }
    }
    groups
}

fn shuffle_groups(groups: &mut [ExperimentGroup], rng: &mut SplitMix64) {
    for index in (1..groups.len()).rev() {
        let selected = rng.next_bounded(index + 1);
        groups.swap(index, selected);
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_bounded(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}

fn entropy_seed() -> u64 {
    let bytes = Uuid::new_v4().into_bytes();
    u64::from_be_bytes(bytes[..8].try_into().expect("UUID prefix is eight bytes"))
}

fn duration_to_next_minute(now: SystemTime) -> Duration {
    let elapsed = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let remainder = elapsed.as_millis() % PERIOD.as_millis();
    Duration::from_millis((PERIOD.as_millis() - remainder) as u64)
}

/// A small, execution-independent scorer view used by the deterministic dense
/// sampling waterfall.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingScorer {
    pub name: String,
    pub sample_rate: f64,
}

pub fn sample_waterfall(entity_id: &str, scorers: &[SamplingScorer]) -> Vec<String> {
    let mut ordered = scorers.to_vec();
    ordered.sort_by(|left, right| {
        right
            .sample_rate
            .partial_cmp(&left.sample_rate)
            .unwrap_or(Ordering::Equal)
    });
    let mut selected = Vec::new();
    let mut previous_rate = 1.0;
    for scorer in ordered {
        let conditional_rate = if previous_rate > 0.0 {
            scorer.sample_rate / previous_rate
        } else {
            0.0
        };
        if deterministic_fraction(entity_id, &scorer.name) > conditional_rate {
            break;
        }
        previous_rate = scorer.sample_rate;
        selected.push(scorer.name);
    }
    selected
}

fn deterministic_fraction(entity_id: &str, scorer_name: &str) -> f64 {
    let digest = Sha256::digest(format!("{entity_id}:{scorer_name}").as_bytes());
    let prefix = u64::from_be_bytes(digest[..8].try_into().expect("SHA prefix is eight bytes"));
    prefix as f64 / 2_f64.powi(64)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceCheckpoint {
    pub timestamp_ms: i64,
    pub trace_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    pub timestamp_ms: i64,
    pub session_id: Option<String>,
}

pub async fn get_trace_checkpoint(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
) -> Result<Option<TraceCheckpoint>, MlflowError> {
    checkpoint_tag(store, workspace, experiment_id, TRACE_CHECKPOINT_TAG).await
}

pub async fn persist_trace_checkpoint(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
    checkpoint: &TraceCheckpoint,
) -> Result<(), MlflowError> {
    persist_checkpoint_tag(
        store,
        workspace,
        experiment_id,
        TRACE_CHECKPOINT_TAG,
        checkpoint,
    )
    .await
}

pub async fn get_session_checkpoint(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
) -> Result<Option<SessionCheckpoint>, MlflowError> {
    checkpoint_tag(store, workspace, experiment_id, SESSION_CHECKPOINT_TAG).await
}

pub async fn persist_session_checkpoint(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
    checkpoint: &SessionCheckpoint,
) -> Result<(), MlflowError> {
    persist_checkpoint_tag(
        store,
        workspace,
        experiment_id,
        SESSION_CHECKPOINT_TAG,
        checkpoint,
    )
    .await
}

async fn checkpoint_tag<T: for<'de> Deserialize<'de>>(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
    key: &str,
) -> Result<Option<T>, MlflowError> {
    let experiment = store.get_experiment(workspace, experiment_id).await?;
    Ok(experiment
        .tags
        .iter()
        .find(|tag| tag.key == key)
        .and_then(|tag| tag.value.as_deref())
        .and_then(|value| serde_json::from_str(value).ok()))
}

async fn persist_checkpoint_tag<T: Serialize>(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
    key: &str,
    checkpoint: &T,
) -> Result<(), MlflowError> {
    let value: Value = serde_json::to_value(checkpoint)
        .map_err(|error| MlflowError::internal_error(error.to_string()))?;
    store
        .set_experiment_tag(
            workspace,
            experiment_id,
            key,
            &python_json_dumps(&value, false),
        )
        .await
}

pub fn trace_time_window(now_ms: i64, checkpoint: Option<&TraceCheckpoint>) -> (i64, i64) {
    let minimum = now_ms.saturating_sub(MAX_LOOKBACK_MS);
    (
        checkpoint.map_or(minimum, |value| value.timestamp_ms.max(minimum)),
        now_ms,
    )
}

pub fn session_time_window(
    now_ms: i64,
    completion_buffer_seconds: i64,
    checkpoint: Option<&SessionCheckpoint>,
) -> (i64, i64) {
    let minimum = now_ms.saturating_sub(MAX_LOOKBACK_MS);
    let start = checkpoint.map_or(minimum, |value| value.timestamp_ms.max(minimum));
    let end = now_ms.saturating_sub(completion_buffer_seconds.max(0).saturating_mul(1000));
    (start, end)
}

pub fn cap_trace_entities(mut entities: Vec<(i64, String)>) -> Vec<(i64, String)> {
    entities.sort();
    entities.truncate(MAX_TRACES_PER_JOB);
    entities
}

pub fn cap_session_entities(mut entities: Vec<(i64, String)>) -> Vec<(i64, String)> {
    entities.sort();
    entities.truncate(MAX_SESSIONS_PER_JOB);
    entities
}

pub fn deduplicate_scorer_names(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}
