//! Server-owned trace-archival scheduling.
//!
//! Python registers a Huey task checked immediately and every minute. The task is
//! overlap-locked, reads the stale-tolerant config cache, admits a pass when
//! `interval_seconds` has elapsed since the previous admission, shuffles the
//! name-sorted workspace list, and shares one successful-archive budget across
//! those workspaces. This module preserves those boundaries while using the
//! shared SQL scheduler-lock discipline from §14.6 instead of Huey.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use mlflow_error::MlflowError;
use mlflow_store::{JobStore, TrackingStore, WorkspaceStore, WORKSPACE_DEFAULT_NAME};
use tokio::time::MissedTickBehavior;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::trace_archival::archive_traces_for_workspace_at;
use crate::{ServerConfig, SystemMonotonicClock, TraceArchivalConfigClock};

pub const TRACE_ARCHIVAL_SCHEDULER_LOCK: &str = "trace-archival-scheduler-lock";

const PERIOD: Duration = Duration::from_secs(60);
const LOCK_LEASE_MS: i64 = 5 * 60 * 1000;

#[derive(Debug, Default)]
struct SchedulerState {
    // Python initializes `_TRACE_ARCHIVAL_SCHEDULER_LAST_RUN_MONOTONIC` to
    // 0.0, rather than treating the first poll specially.
    last_run_monotonic: Duration,
}

#[derive(Clone)]
pub struct TraceArchivalScheduler {
    tracking_store: TrackingStore,
    job_store: JobStore,
    workspace_store: Option<WorkspaceStore>,
    server_config: ServerConfig,
    clock: Arc<dyn TraceArchivalConfigClock>,
    state: Arc<Mutex<SchedulerState>>,
}

impl std::fmt::Debug for TraceArchivalScheduler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TraceArchivalScheduler")
            .field("workspace_store", &self.workspace_store)
            .field("server_config", &self.server_config)
            .finish_non_exhaustive()
    }
}

impl TraceArchivalScheduler {
    pub fn new(
        tracking_store: TrackingStore,
        workspace_store: Option<WorkspaceStore>,
        server_config: ServerConfig,
    ) -> Self {
        Self::with_clock(
            tracking_store,
            workspace_store,
            server_config,
            Arc::new(SystemMonotonicClock),
        )
    }

    /// Construct a scheduler with an injected monotonic clock. Tests pair this
    /// with [`crate::TraceArchivalConfigProvider::with_clock`] and
    /// [`Self::run_once_at`] so both the config TTL and archival cutoff advance
    /// without sleeping.
    pub fn with_clock(
        tracking_store: TrackingStore,
        workspace_store: Option<WorkspaceStore>,
        server_config: ServerConfig,
        clock: Arc<dyn TraceArchivalConfigClock>,
    ) -> Self {
        let job_store = JobStore::new(tracking_store.db().clone());
        Self {
            tracking_store,
            job_store,
            workspace_store,
            server_config,
            clock,
            state: Arc::new(Mutex::new(SchedulerState::default())),
        }
    }

    /// Run one locked scheduler poll. The seed controls Python-compatible
    /// workspace shuffling; returning zero includes disabled, unconfigured,
    /// interval-gated, overlap-locked, and empty passes.
    pub async fn run_once(&self, seed: u64) -> Result<u64, MlflowError> {
        self.run_once_at(seed, chrono::Utc::now().timestamp_millis())
            .await
    }

    /// Deterministic wall-clock variant. Every workspace in an admitted pass
    /// receives the same archival cutoff, matching a frozen Python store clock.
    pub async fn run_once_at(&self, seed: u64, now_millis: i64) -> Result<u64, MlflowError> {
        let Some(lock) = self
            .job_store
            .try_acquire_periodic_scheduler_lock(TRACE_ARCHIVAL_SCHEDULER_LOCK, LOCK_LEASE_MS)
            .await?
        else {
            debug!("trace archival scheduler DB lock is held; skipping pass");
            return Ok(0);
        };

        let result = self.run_locked_at(seed, now_millis).await;
        let release = self.job_store.release_periodic_scheduler_lock(&lock).await;
        match (result, release) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(count), Ok(())) => Ok(count),
        }
    }

    async fn run_locked_at(&self, seed: u64, now_millis: i64) -> Result<u64, MlflowError> {
        let config = match self.server_config.current_trace_archival_config() {
            Ok(Some(config)) if config.enabled => config,
            Ok(_) => return Ok(0),
            Err(error) => {
                warn!(%error, "ignoring invalid trace archival scheduler configuration");
                return Ok(0);
            }
        };
        if !self.admit_interval(config.interval_seconds) {
            return Ok(0);
        }

        let mut workspaces = match &self.workspace_store {
            Some(store) => store
                .list_workspaces()
                .await?
                .into_iter()
                .map(|workspace| workspace.name)
                .collect::<Vec<_>>(),
            None => vec![WORKSPACE_DEFAULT_NAME.to_string()],
        };
        if workspaces.is_empty() {
            info!("trace archival scheduler found no workspaces; skipping");
            return Ok(0);
        }
        python_shuffle(&mut workspaces, seed);

        let started = self.clock.now();
        let mut archived_total = 0_u64;
        let mut remaining_budget = config
            .max_traces_per_pass
            .and_then(|value| usize::try_from(value).ok());
        let mut scope_count = 0_usize;
        for workspace in workspaces {
            if remaining_budget == Some(0) {
                break;
            }
            scope_count += 1;
            match archive_traces_for_workspace_at(
                &self.tracking_store,
                self.workspace_store.as_ref(),
                &workspace,
                &config,
                remaining_budget,
                now_millis,
            )
            .await
            {
                Ok(archived) => {
                    archived_total = archived_total.saturating_add(archived);
                    if let Some(remaining) = &mut remaining_budget {
                        *remaining = remaining
                            .saturating_sub(usize::try_from(archived).unwrap_or(usize::MAX));
                    }
                }
                Err(error) => warn!(workspace, %error, "trace archival scheduler scope failed"),
            }
        }

        info!(
            archived_total,
            scope_count,
            elapsed_seconds = self.clock.now().saturating_sub(started).as_secs_f64(),
            "trace archival scheduler pass completed"
        );
        Ok(archived_total)
    }

    fn admit_interval(&self, interval_seconds: u64) -> bool {
        let now = self.clock.now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if now.saturating_sub(state.last_run_monotonic) < Duration::from_secs(interval_seconds) {
            return false;
        }
        state.last_run_monotonic = now;
        true
    }

    /// Poll immediately and every minute thereafter. Huey's scheduler starts
    /// `_next_periodic` at its current monotonic time, so the every-minute cron
    /// expression is checked once on consumer startup; missed checks are not
    /// replayed in a burst. The server owns shutdown by aborting this task,
    /// just as Python terminates its dedicated Huey consumer with the server.
    pub async fn run_periodic(self) {
        let mut interval = periodic_interval();
        loop {
            interval.tick().await;
            if let Err(error) = self.run_once(entropy_seed()).await {
                warn!(%error, "trace archival scheduler pass failed");
            }
        }
    }
}

fn python_shuffle<T>(values: &mut [T], seed: u64) {
    let mut random = PythonRandom::new(seed);
    for index in (1..values.len()).rev() {
        let selected = random.randbelow(index + 1);
        values.swap(index, selected);
    }
}

/// CPython `_random.Random` MT19937 integer-seed stream plus the
/// getrandbits-based `_randbelow` used by `random.shuffle`.
#[derive(Debug, Clone)]
struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn new(seed: u64) -> Self {
        let mut key = vec![seed as u32];
        if seed > u64::from(u32::MAX) {
            key.push((seed >> 32) as u32);
        }
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_by_array(&key);
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..624 {
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1;
        let mut j = 0;
        for _ in 0..624.max(key.len()) {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn getrandbits(&mut self, bits: u32) -> u64 {
        if bits <= 32 {
            return u64::from(self.gen_u32() >> (32 - bits));
        }
        let low = u64::from(self.gen_u32());
        let remaining = bits - 32;
        let high = u64::from(self.gen_u32() >> (32 - remaining));
        low | (high << 32)
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = self.getrandbits(bits) as usize;
            if value < upper {
                return value;
            }
        }
    }
}

fn entropy_seed() -> u64 {
    let bytes = Uuid::new_v4().into_bytes();
    u64::from_be_bytes(bytes[..8].try_into().expect("UUID prefix is eight bytes"))
}

fn periodic_interval() -> tokio::time::Interval {
    let mut interval = tokio::time::interval(PERIOD);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    interval
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::{json, Value};

    use super::*;

    #[derive(Debug, Default)]
    struct ManualClock {
        millis: AtomicU64,
    }

    impl ManualClock {
        fn set(&self, millis: u64) {
            self.millis.store(millis, Ordering::SeqCst);
        }
    }

    impl TraceArchivalConfigClock for ManualClock {
        fn now(&self) -> Duration {
            Duration::from_millis(self.millis.load(Ordering::SeqCst))
        }
    }

    #[tokio::test]
    async fn interval_gate_anchors_at_admission_and_observes_config_changes() {
        let clock = Arc::new(ManualClock::default());
        let temp = mlflow_test_support::TempDb::new("trace_archival_scheduler_interval").await;
        let db = temp.connect().await;
        let scheduler = TraceArchivalScheduler::with_clock(
            TrackingStore::new(db, "file:///tmp/mlruns-unused"),
            None,
            ServerConfig::default(),
            clock.clone(),
        );

        clock.set(59_999);
        assert!(!scheduler.admit_interval(60));
        clock.set(60_000);
        assert!(scheduler.admit_interval(60));
        clock.set(70_000);
        assert!(!scheduler.admit_interval(60));
        // A shorter refreshed interval compares against the same admission
        // anchor and can make the next poll due immediately.
        assert!(scheduler.admit_interval(10));
        clock.set(79_999);
        assert!(!scheduler.admit_interval(10));
        clock.set(80_000);
        assert!(scheduler.admit_interval(10));
    }

    #[test]
    fn same_seed_scheduler_decisions_match_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let content = json!({
            "gate_cases": [
                [
                    {"configured": false, "enabled": false, "monotonic_seconds": 10.0, "interval_seconds": 300},
                    {"configured": true, "enabled": true, "monotonic_seconds": 299.0, "interval_seconds": 300},
                    {"configured": true, "enabled": true, "monotonic_seconds": 300.0, "interval_seconds": 300},
                    {"configured": true, "enabled": true, "monotonic_seconds": 599.9, "interval_seconds": 300},
                    {"configured": true, "enabled": false, "monotonic_seconds": 600.0, "interval_seconds": 300},
                    {"configured": true, "enabled": true, "monotonic_seconds": 600.0, "interval_seconds": 300}
                ],
                [
                    {"configured": true, "enabled": true, "monotonic_seconds": 100.0, "interval_seconds": 60},
                    {"configured": true, "enabled": true, "monotonic_seconds": 110.0, "interval_seconds": 10},
                    {"configured": true, "enabled": true, "monotonic_seconds": 119.999, "interval_seconds": 10},
                    {"configured": true, "enabled": true, "monotonic_seconds": 120.0, "interval_seconds": 10}
                ]
            ],
            "pass_cases": [
                {
                    "seed": 0,
                    "workspaces": ["alpha", "beta", "default", "omega"],
                    "max_traces_per_pass": 3,
                    "scopes": {
                        "alpha": {"candidates": [{"experiment_id": "a1", "trace_id": "a-old"}, {"experiment_id": "a2", "trace_id": "a-new"}]},
                        "beta": {"error": true, "candidates": [{"experiment_id": "b1", "trace_id": "b-old"}]},
                        "default": {"candidates": [{"experiment_id": "d1", "trace_id": "d-old"}, {"experiment_id": "d1", "trace_id": "d-new"}]},
                        "omega": {"candidates": [{"experiment_id": "o1", "trace_id": "o-old"}]}
                    }
                },
                {
                    "seed": 1099511627793_u64,
                    "workspaces": ["alpha", "beta", "default", "omega"],
                    "max_traces_per_pass": null,
                    "scopes": {
                        "alpha": {"candidates": [{"experiment_id": "a", "trace_id": "a"}]},
                        "beta": {"candidates": [{"experiment_id": "b", "trace_id": "b"}]},
                        "default": {"candidates": []},
                        "omega": {"error": true, "candidates": []}
                    }
                }
            ]
        });
        let output = Command::new("uv")
            .args([
                "run",
                "--frozen",
                "python",
                "rust/tools/trace_archival_scheduler_differential.py",
                "--content",
                &content.to_string(),
            ])
            .current_dir(&root)
            .output()
            .expect("run Python scheduler differential");
        assert!(
            output.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let python: Value = serde_json::from_slice(&output.stdout).unwrap();

        let mut rust_gate_traces = Vec::new();
        for polls in content["gate_cases"].as_array().unwrap() {
            let mut last_run = Duration::ZERO;
            let mut decisions = Vec::new();
            for poll in polls.as_array().unwrap() {
                let configured = poll["configured"].as_bool().unwrap();
                let enabled = poll["enabled"].as_bool().unwrap();
                if !configured || !enabled {
                    decisions.push(false);
                    continue;
                }
                let now = Duration::from_secs_f64(poll["monotonic_seconds"].as_f64().unwrap());
                let interval = Duration::from_secs(poll["interval_seconds"].as_u64().unwrap());
                let due = now.saturating_sub(last_run) >= interval;
                if due {
                    last_run = now;
                }
                decisions.push(due);
            }
            rust_gate_traces.push(decisions);
        }
        assert_eq!(python["gate_traces"], json!(rust_gate_traces));

        let mut rust_pass_traces = Vec::new();
        for case in content["pass_cases"].as_array().unwrap() {
            let mut workspaces = case["workspaces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_string())
                .collect::<Vec<_>>();
            python_shuffle(&mut workspaces, case["seed"].as_u64().unwrap());
            let mut remaining = case["max_traces_per_pass"]
                .as_u64()
                .map(|value| value as usize);
            let mut calls = Vec::new();
            let mut archived = Vec::new();
            for workspace in &workspaces {
                if remaining == Some(0) {
                    break;
                }
                calls.push(json!({"workspace": workspace, "remaining_budget": remaining}));
                let scope = &case["scopes"][workspace];
                if scope["error"].as_bool().unwrap_or(false) {
                    continue;
                }
                let candidates = scope["candidates"].as_array().unwrap();
                let take = remaining.unwrap_or(candidates.len()).min(candidates.len());
                archived.extend(candidates[..take].iter().map(|candidate| {
                    json!({
                        "workspace": workspace,
                        "experiment_id": candidate["experiment_id"],
                        "trace_id": candidate["trace_id"],
                    })
                }));
                if let Some(value) = &mut remaining {
                    *value -= take;
                }
            }
            rust_pass_traces.push(json!({
                "workspace_order": workspaces,
                "calls": calls,
                "archived": archived,
                "remaining_budget": remaining,
            }));
        }
        assert_eq!(python["pass_traces"], json!(rust_pass_traces));
    }

    #[tokio::test(start_paused = true)]
    async fn huey_cadence_has_an_immediate_tick_then_sixty_second_ticks() {
        let mut interval = periodic_interval();
        let first = interval.tick().await;
        let second = interval.tick().await;
        assert_eq!(second.duration_since(first), PERIOD);
    }
}
