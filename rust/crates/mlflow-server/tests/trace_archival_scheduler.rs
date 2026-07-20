//! T21.4 trace-archival scheduler integration tests.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use mlflow_server::trace_archival_scheduler::{
    TraceArchivalScheduler, TRACE_ARCHIVAL_SCHEDULER_LOCK,
};
use mlflow_server::{ServerConfig, TraceArchivalConfigClock, TraceArchivalConfigProvider};
use mlflow_store::{
    JobStore, SpanInput, StartTraceInput, TraceTimeRange, TrackingStore, Workspace, WorkspaceStore,
    SPANS_LOCATION_ARCHIVE_REPO, TRACE_TAG_SPANS_LOCATION,
};
use mlflow_test_support::TempDb;

const ROOT_CONTENT: &str = r#"{"trace_id":"ABEiM0RVZneImaq7zN3u/w==","span_id":"ECAwQFBgcIA=","parent_span_id":null,"name":"root","start_time_unix_nano":1,"end_time_unix_nano":2,"events":[],"status":{"code":"STATUS_CODE_OK","message":""},"attributes":{"mlflow.traceRequestId":"\"tr-00112233445566778899aabbccddeeff\"","mlflow.spanType":"\"CHAIN\""},"links":[]}"#;

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

fn write_config(
    path: &std::path::Path,
    archive_root: &std::path::Path,
    enabled: bool,
    budget: usize,
) {
    std::fs::write(
        path,
        format!(
            "trace_archival:\n  enabled: {enabled}\n  location: file://{}\n  retention: 1m\n  interval_seconds: 1\n  max_traces_per_pass: {budget}\n",
            archive_root.display()
        ),
    )
    .unwrap();
}

async fn seed_trace(store: &TrackingStore, workspace: &str, suffix: u128) -> String {
    let experiment_id = store
        .create_experiment(workspace, &format!("scheduler-{suffix}"), None, &[])
        .await
        .unwrap();
    let trace_id = format!("tr-{suffix:032x}");
    store
        .start_trace(
            workspace,
            &StartTraceInput {
                trace_id: trace_id.clone(),
                experiment_id: experiment_id.clone(),
                request_time: 0,
                execution_duration: Some(1),
                state: "OK".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: None,
                tags: Vec::new(),
                trace_metadata: Vec::new(),
                trace_metrics: Vec::new(),
                assessments: Vec::new(),
            },
        )
        .await
        .unwrap();
    store
        .log_spans(
            workspace,
            &experiment_id,
            &[SpanInput {
                trace_id: trace_id.clone(),
                span_id: format!("{suffix:016x}"),
                parent_span_id: None,
                name: Some("root".to_string()),
                span_type: Some("CHAIN".to_string()),
                status: "OK".to_string(),
                start_time_unix_nano: 1,
                end_time_unix_nano: Some(2),
                content: ROOT_CONTENT.to_string(),
                dimension_attributes: None,
            }],
            &[],
            &[TraceTimeRange {
                trace_id: trace_id.clone(),
                min_start_ms: 0,
                max_end_ms: Some(0),
                root_span_status: Some("OK".to_string()),
            }],
        )
        .await
        .unwrap();
    trace_id
}

fn scheduler_config(config_path: &std::path::Path, clock: Arc<ManualClock>) -> ServerConfig {
    ServerConfig {
        trace_archival_config: TraceArchivalConfigProvider::with_clock(
            Some(config_path.to_path_buf()),
            clock,
        ),
        ..ServerConfig::default()
    }
}

#[tokio::test]
async fn scheduler_shares_budget_in_seeded_workspace_order_and_refreshes_config() {
    let temp = TempDb::new("trace_archival_scheduler_budget").await;
    let db = temp.connect().await;
    let store = TrackingStore::new(db.clone(), "file:///tmp/mlruns-unused");
    let workspace_store = WorkspaceStore::new(db, "sqlite://scheduler-workspaces");
    workspace_store
        .create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    workspace_store
        .create_workspace(Workspace::named("team-b"))
        .await
        .unwrap();

    let default_trace = seed_trace(&store, "default", 1).await;
    let team_a_trace = seed_trace(&store, "team-a", 2).await;
    let team_b_trace = seed_trace(&store, "team-b", 3).await;
    let directory = tempfile::tempdir().unwrap();
    let archive_root = directory.path().join("archive");
    std::fs::create_dir(&archive_root).unwrap();
    let config_path = directory.path().join("config.yaml");
    write_config(&config_path, &archive_root, true, 2);

    let clock = Arc::new(ManualClock::default());
    clock.set(10_000);
    let scheduler = TraceArchivalScheduler::with_clock(
        store.clone(),
        Some(workspace_store),
        scheduler_config(&config_path, clock.clone()),
        clock.clone(),
    );

    // Python seed 17 orders the name-sorted scopes as default, team-a,
    // team-b. The two-success budget is therefore exhausted before team-b.
    assert_eq!(scheduler.run_once_at(17, 60_000).await.unwrap(), 2);
    assert_eq!(
        store
            .get_trace_info("default", &default_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
    assert_eq!(
        store
            .get_trace_info("team-a", &team_a_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
    assert_ne!(
        store
            .get_trace_info("team-b", &team_b_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );

    // Reload at the exact five-second TTL boundary. Seed 21 puts team-b first;
    // the refreshed one-trace budget must leave a new team-a trace untouched.
    let second_team_a_trace = seed_trace(&store, "team-a", 4).await;
    write_config(&config_path, &archive_root, true, 1);
    clock.set(15_000);
    assert_eq!(scheduler.run_once_at(21, 60_000).await.unwrap(), 1);
    assert_eq!(
        store
            .get_trace_info("team-b", &team_b_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
    assert_ne!(
        store
            .get_trace_info("team-a", &second_team_a_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );

    // An invalid refresh at the next TTL boundary reuses the last valid
    // one-trace config and renews its cache window.
    std::fs::write(&config_path, "trace_archival: [\n").unwrap();
    clock.set(20_000);
    assert_eq!(scheduler.run_once_at(18, 60_000).await.unwrap(), 1);
    assert_eq!(
        store
            .get_trace_info("team-a", &second_team_a_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );

    // Disabled refreshes skip without consuming a new interval anchor.
    let third_team_a_trace = seed_trace(&store, "team-a", 5).await;
    write_config(&config_path, &archive_root, false, 1);
    clock.set(25_000);
    assert_eq!(scheduler.run_once_at(21, 60_000).await.unwrap(), 0);
    assert_ne!(
        store
            .get_trace_info("team-a", &third_team_a_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
}

#[tokio::test]
async fn overlap_lock_skips_without_advancing_the_interval_gate() {
    let temp = TempDb::new("trace_archival_scheduler_overlap").await;
    let db = temp.connect().await;
    let store = TrackingStore::new(db.clone(), "file:///tmp/mlruns-unused");
    let trace_id = seed_trace(&store, "default", 10).await;
    let directory = tempfile::tempdir().unwrap();
    let archive_root = directory.path().join("archive");
    std::fs::create_dir(&archive_root).unwrap();
    let config_path = directory.path().join("config.yaml");
    write_config(&config_path, &archive_root, true, 1);
    let clock = Arc::new(ManualClock::default());
    clock.set(10_000);
    let scheduler = TraceArchivalScheduler::with_clock(
        store.clone(),
        None,
        scheduler_config(&config_path, clock.clone()),
        clock,
    );

    let jobs = JobStore::new(db);
    let held = jobs
        .try_acquire_periodic_scheduler_lock(TRACE_ARCHIVAL_SCHEDULER_LOCK, 300_000)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(scheduler.run_once_at(0, 60_000).await.unwrap(), 0);
    jobs.release_periodic_scheduler_lock(&held).await.unwrap();

    // Same monotonic instant is still due: the overlap loser never entered the
    // Python interval gate.
    assert_eq!(scheduler.run_once_at(0, 60_000).await.unwrap(), 1);
    assert_eq!(
        store
            .get_trace_info("default", &trace_id)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
}

#[tokio::test]
async fn failed_workspace_does_not_consume_budget_or_stop_later_workspaces() {
    let temp = TempDb::new("trace_archival_scheduler_isolation").await;
    let db = temp.connect().await;
    let store = TrackingStore::new(db.clone(), "file:///tmp/mlruns-unused");
    let workspace_store = WorkspaceStore::new(db, "sqlite://scheduler-workspaces");
    workspace_store
        .create_workspace(Workspace {
            name: "team-a".to_string(),
            // This is a valid file repository configuration, but writing a
            // directory below the /dev/null device fails at archive time.
            trace_archival_location: Some("file:///dev/null".to_string()),
            ..Workspace::default()
        })
        .await
        .unwrap();
    workspace_store
        .create_workspace(Workspace::named("team-b"))
        .await
        .unwrap();
    let failed_trace = seed_trace(&store, "team-a", 20).await;
    let successful_trace = seed_trace(&store, "team-b", 21).await;

    let directory = tempfile::tempdir().unwrap();
    let archive_root = directory.path().join("archive");
    std::fs::create_dir(&archive_root).unwrap();
    let config_path = directory.path().join("config.yaml");
    write_config(&config_path, &archive_root, true, 1);
    let clock = Arc::new(ManualClock::default());
    clock.set(10_000);
    let scheduler = TraceArchivalScheduler::with_clock(
        store.clone(),
        Some(workspace_store),
        scheduler_config(&config_path, clock.clone()),
        clock,
    );

    // Python seed 18 visits team-a, team-b, default. team-a's failed archive
    // returns zero, so team-b receives the still-unspent one-trace budget.
    assert_eq!(scheduler.run_once_at(18, 60_000).await.unwrap(), 1);
    assert_ne!(
        store
            .get_trace_info("team-a", &failed_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
    assert_eq!(
        store
            .get_trace_info("team-b", &successful_trace)
            .await
            .unwrap()
            .tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
}
