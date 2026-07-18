//! T17.3 online-scoring scheduler parity tests.

use mlflow_server::online_scoring_scheduler::{
    cap_session_entities, cap_trace_entities, get_session_checkpoint, get_trace_checkpoint,
    group_and_shuffle_scorers, persist_session_checkpoint, persist_trace_checkpoint,
    sample_waterfall, session_time_window, trace_time_window, OnlineScoringScheduler,
    SamplingScorer, SessionCheckpoint, TraceCheckpoint, MAX_LOOKBACK_MS,
    ONLINE_SCORING_SCHEDULER_LOCK, ONLINE_SESSION_SCORER_JOB_NAME, ONLINE_TRACE_SCORER_JOB_NAME,
};
use mlflow_store::{
    Db, JobStore, OnlineScorer, OnlineScoringConfig, TrackingStore, Workspace, WorkspaceStore,
    WORKSPACE_DEFAULT_NAME,
};
use mlflow_test_support::TempDb;
use serde_json::{json, Value};

const WS: &str = WORKSPACE_DEFAULT_NAME;
const ENDPOINT_ID: &str = "scheduler-endpoint-id";
const ENDPOINT_NAME: &str = "scheduler-endpoint";

async fn stores(tag: &str) -> (TempDb, TrackingStore, JobStore) {
    let temp = TempDb::new(tag).await;
    let db = temp.connect().await;
    seed_endpoint(&db).await;
    (
        temp,
        TrackingStore::new(db.clone(), "/tmp/mlflow-artifacts"),
        JobStore::new(db),
    )
}

fn scorer_payload(name: &str, session_level: bool) -> String {
    scorer_payload_for_endpoint(name, session_level, ENDPOINT_NAME)
}

fn scorer_payload_for_endpoint(name: &str, session_level: bool, endpoint: &str) -> String {
    json!({
        "name": name,
        "aggregations": null,
        "description": null,
        "is_session_level_scorer": session_level,
        "mlflow_version": "3.14.1.dev0",
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {
            "instructions": if session_level { "Judge {{ conversation }}" } else { "Judge {{ inputs }}" },
            "model": format!("gateway:/{endpoint}"),
            "feedback_value_type": {"title": "Result", "type": "string"},
        },
    })
    .to_string()
}

async fn add_config(
    store: &TrackingStore,
    experiment_id: &str,
    name: &str,
    session_level: bool,
    sample_rate: f64,
) -> OnlineScoringConfig {
    store
        .register_scorer(
            WS,
            experiment_id,
            name,
            &scorer_payload(name, session_level),
        )
        .await
        .unwrap();
    store
        .upsert_online_scoring_config(WS, experiment_id, name, sample_rate, None)
        .await
        .unwrap()
}

#[tokio::test]
async fn config_scan_only_returns_due_gateway_configs_at_latest_version() {
    let (_temp, store, _jobs) = stores("scheduler_due_scan").await;
    let experiment_id = store
        .create_experiment(WS, "scheduler-due-scan", None, &[])
        .await
        .unwrap();
    add_config(&store, &experiment_id, "due", false, 0.75).await;
    add_config(&store, &experiment_id, "stopped", false, 0.0).await;

    let active = store.get_active_online_scorers(WS).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].name, "due");
    assert_eq!(active[0].online_config.sample_rate, 0.75);
    assert_eq!(active[0].online_config.experiment_id, experiment_id);
    assert!(active[0]
        .serialized_scorer
        .contains("gateway:/scheduler-endpoint"));
}

#[test]
fn grouping_and_shuffle_are_deterministic_with_a_pinned_seed() {
    let scorers = ["1", "2", "1", "3", "4"]
        .into_iter()
        .enumerate()
        .map(|(index, experiment_id)| OnlineScorer {
            name: format!("s{index}"),
            serialized_scorer: "{}".to_string(),
            online_config: OnlineScoringConfig {
                online_scoring_config_id: format!("c{index}"),
                scorer_id: format!("s{index}"),
                sample_rate: 1.0,
                experiment_id: experiment_id.to_string(),
                filter_string: None,
            },
        })
        .collect::<Vec<_>>();
    let first = group_and_shuffle_scorers(scorers.clone(), 17);
    let second = group_and_shuffle_scorers(scorers.clone(), 17);
    let different = group_and_shuffle_scorers(scorers, 18);
    let ids = |groups: &[mlflow_server::online_scoring_scheduler::ExperimentGroup]| {
        groups
            .iter()
            .map(|group| group.experiment_id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&first), ids(&second));
    assert_ne!(ids(&first), ids(&different));
    let experiment_one = first
        .iter()
        .find(|group| group.experiment_id == "1")
        .unwrap();
    assert_eq!(experiment_one.scorers.len(), 2);
}

#[test]
fn sampler_matches_python_dense_waterfall() {
    let scorers = vec![
        SamplingScorer {
            name: "low".to_string(),
            sample_rate: 0.2,
        },
        SamplingScorer {
            name: "high".to_string(),
            sample_rate: 0.8,
        },
        SamplingScorer {
            name: "mid".to_string(),
            sample_rate: 0.4,
        },
    ];
    assert_eq!(
        sample_waterfall("entity-1", &scorers),
        ["high", "mid", "low"]
    );
    assert_eq!(sample_waterfall("entity-2", &scorers), ["high"]);
    assert!(sample_waterfall("entity-4", &scorers).is_empty());
}

#[tokio::test]
async fn checkpoint_tags_round_trip_with_python_json_and_time_windows() {
    let (_temp, store, _jobs) = stores("scheduler_checkpoints").await;
    let experiment_id = store
        .create_experiment(WS, "scheduler-checkpoints", None, &[])
        .await
        .unwrap();
    let trace = TraceCheckpoint {
        timestamp_ms: 123,
        trace_id: Some("tr-1".to_string()),
    };
    let session = SessionCheckpoint {
        timestamp_ms: 456,
        session_id: None,
    };
    persist_trace_checkpoint(&store, WS, &experiment_id, &trace)
        .await
        .unwrap();
    persist_session_checkpoint(&store, WS, &experiment_id, &session)
        .await
        .unwrap();
    assert_eq!(
        get_trace_checkpoint(&store, WS, &experiment_id)
            .await
            .unwrap(),
        Some(trace.clone())
    );
    assert_eq!(
        get_session_checkpoint(&store, WS, &experiment_id)
            .await
            .unwrap(),
        Some(session.clone())
    );
    let experiment = store.get_experiment(WS, &experiment_id).await.unwrap();
    let trace_tag = experiment
        .tags
        .iter()
        .find(|tag| tag.key == "mlflow.latestOnlineScoring.trace.checkpoint")
        .unwrap();
    assert_eq!(
        trace_tag.value.as_deref(),
        Some(r#"{"timestamp_ms": 123, "trace_id": "tr-1"}"#)
    );

    assert_eq!(
        trace_time_window(4_000_000, Some(&trace)),
        (400_000, 4_000_000)
    );
    let recent = TraceCheckpoint {
        timestamp_ms: 3_900_000,
        trace_id: None,
    };
    assert_eq!(
        trace_time_window(4_000_000, Some(&recent)),
        (3_900_000, 4_000_000)
    );
    assert_eq!(
        session_time_window(4_000_000, 60, None),
        (4_000_000 - MAX_LOOKBACK_MS, 3_940_000)
    );
}

#[test]
fn per_job_caps_sort_chronologically_and_use_id_tiebreakers() {
    let traces = (0..600)
        .rev()
        .map(|index| (index / 2, format!("tr-{index:04}")))
        .collect();
    let sessions = (0..150)
        .rev()
        .map(|index| (index / 2, format!("session-{index:04}")))
        .collect();
    let traces = cap_trace_entities(traces);
    let sessions = cap_session_entities(sessions);
    assert_eq!(traces.len(), 500);
    assert_eq!(sessions.len(), 100);
    assert!(traces.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(sessions.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(traces.last().unwrap().1, "tr-0499");
    assert_eq!(sessions.last().unwrap().1, "session-0099");
}

#[tokio::test]
async fn db_lock_excludes_a_second_scheduler_instance_without_double_submission() {
    let (_temp, store, jobs) = stores("scheduler_db_lock").await;
    let experiment_id = store
        .create_experiment(WS, "scheduler-db-lock", None, &[])
        .await
        .unwrap();
    add_config(&store, &experiment_id, "trace-judge", false, 1.0).await;
    let first = OnlineScoringScheduler::new(store.clone(), None);
    let second = OnlineScoringScheduler::new(store, None);
    let held = jobs
        .try_acquire_periodic_scheduler_lock(ONLINE_SCORING_SCHEDULER_LOCK, 300_000)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(second.run_once(9).await.unwrap(), 0);
    assert!(jobs
        .list_jobs(
            WS,
            Some(ONLINE_TRACE_SCORER_JOB_NAME),
            &[],
            None,
            None,
            None
        )
        .await
        .unwrap()
        .is_empty());
    jobs.release_periodic_scheduler_lock(&held).await.unwrap();

    assert_eq!(first.run_once(9).await.unwrap(), 1);
    let submitted = jobs
        .list_jobs(WS, None, &[], None, None, None)
        .await
        .unwrap();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].job_name, ONLINE_TRACE_SCORER_JOB_NAME);
    let params: Value = serde_json::from_str(&submitted[0].params).unwrap();
    assert_eq!(params["experiment_id"], experiment_id);
    assert_eq!(params["online_scorers"][0]["name"], "trace-judge");
}

#[tokio::test]
async fn scheduler_submits_one_job_per_present_scorer_kind() {
    let (_temp, store, jobs) = stores("scheduler_job_kinds").await;
    let experiment_id = store
        .create_experiment(WS, "scheduler-job-kinds", None, &[])
        .await
        .unwrap();
    add_config(&store, &experiment_id, "trace-a", false, 1.0).await;
    add_config(&store, &experiment_id, "trace-b", false, 0.5).await;
    add_config(&store, &experiment_id, "session-a", true, 1.0).await;

    let scheduler = OnlineScoringScheduler::new(store, None);
    assert_eq!(scheduler.run_once(42).await.unwrap(), 2);
    let submitted = jobs
        .list_jobs(WS, None, &[], None, None, None)
        .await
        .unwrap();
    assert_eq!(submitted.len(), 2);
    assert_eq!(submitted[0].job_name, ONLINE_TRACE_SCORER_JOB_NAME);
    assert_eq!(submitted[1].job_name, ONLINE_SESSION_SCORER_JOB_NAME);
    let trace: Value = serde_json::from_str(&submitted[0].params).unwrap();
    assert_eq!(trace["online_scorers"].as_array().unwrap().len(), 2);
    assert_eq!(
        trace["online_scorers"][0]["online_config"]["filter_string"],
        Value::Null
    );
}

#[tokio::test]
async fn scheduler_preserves_workspace_on_scan_and_submission() {
    let (_temp, store, jobs) = stores("scheduler_workspace").await;
    seed_endpoint_for_workspace(store.db(), "team-endpoint-id", "team-endpoint", "team-a").await;
    let workspace_store = WorkspaceStore::new(store.db().clone(), "sqlite:///workspace.db");
    workspace_store
        .create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    let experiment_id = store
        .create_experiment("team-a", "scheduler-team-a", None, &[])
        .await
        .unwrap();
    store
        .register_scorer(
            "team-a",
            &experiment_id,
            "team-judge",
            &scorer_payload_for_endpoint("team-judge", false, "team-endpoint"),
        )
        .await
        .unwrap();
    store
        .upsert_online_scoring_config("team-a", &experiment_id, "team-judge", 1.0, None)
        .await
        .unwrap();

    let scheduler = OnlineScoringScheduler::new(store, Some(workspace_store));
    assert_eq!(scheduler.run_once(3).await.unwrap(), 1);
    assert!(jobs
        .list_jobs(WS, None, &[], None, None, None)
        .await
        .unwrap()
        .is_empty());
    let team_jobs = jobs
        .list_jobs("team-a", None, &[], None, None, None)
        .await
        .unwrap();
    assert_eq!(team_jobs.len(), 1);
    assert_eq!(team_jobs[0].workspace, "team-a");
}

async fn seed_endpoint(db: &Db) {
    seed_endpoint_for_workspace(db, ENDPOINT_ID, ENDPOINT_NAME, WS).await;
}

async fn seed_endpoint_for_workspace(db: &Db, id: &str, name: &str, workspace: &str) {
    let sql = "INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES (?, ?, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, ?)";
    match db {
        Db::Sqlite(pool) => {
            sqlx::query(sql)
                .bind(id)
                .bind(name)
                .bind(workspace)
                .execute(pool)
                .await
                .unwrap();
        }
        Db::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES ($1, $2, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, $3)",
            )
                .bind(id)
                .bind(name)
                .bind(workspace)
                .execute(pool)
                .await
                .unwrap();
        }
        Db::MySql(pool) => {
            sqlx::query(sql)
                .bind(id)
                .bind(name)
                .bind(workspace)
                .execute(pool)
                .await
                .unwrap();
        }
    }
}
