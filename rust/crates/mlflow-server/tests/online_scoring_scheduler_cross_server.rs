//! Shared-DB differential for Python and Rust online-scoring submission.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

use mlflow_server::online_scoring_scheduler::OnlineScoringScheduler;
use mlflow_store::{Db, JobStore, StartTraceInput, TrackingStore, WORKSPACE_DEFAULT_NAME};
use mlflow_test_support::TempDb;
use serde_json::json;

const WS: &str = WORKSPACE_DEFAULT_NAME;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .to_path_buf()
}

#[tokio::test]
async fn seeded_shared_db_submits_the_same_python_and_rust_jobs() {
    let temp = TempDb::new("online_scheduler_diff").await;
    let uri = temp.uri();
    let db = temp.connect().await;
    seed_endpoint(&db).await;
    let store = TrackingStore::new(db.clone(), "/tmp/mlflow-artifacts");

    let experiment_a = store
        .create_experiment(WS, "scheduler-diff-a", None, &[])
        .await
        .unwrap();
    let experiment_b = store
        .create_experiment(WS, "scheduler-diff-b", None, &[])
        .await
        .unwrap();
    add_config(&store, &experiment_a, "trace-a", false, 1.0).await;
    add_config(&store, &experiment_a, "session-a", true, 0.5).await;
    add_config(&store, &experiment_b, "trace-b", false, 0.25).await;

    // The reference scheduler deliberately does not read this timeline. It is
    // seeded on the same DB to lock that submission/execution boundary: these
    // traces are consumed only when the PENDING jobs run in Phase 19.
    for (index, experiment_id) in [&experiment_a, &experiment_b, &experiment_a]
        .into_iter()
        .enumerate()
    {
        store
            .start_trace(
                WS,
                &StartTraceInput {
                    trace_id: format!("tr-scheduler-{index}"),
                    experiment_id: experiment_id.to_string(),
                    request_time: 1_700_000_000_000 + index as i64,
                    execution_duration: Some(10),
                    state: "OK".to_string(),
                    client_request_id: None,
                    request_preview: Some(format!("request {index}")),
                    response_preview: Some(format!("response {index}")),
                    tags: vec![],
                    trace_metadata: vec![("mlflow.trace.session".to_string(), "s1".to_string())],
                    trace_metrics: vec![],
                    assessments: vec![],
                },
            )
            .await
            .unwrap();
    }

    let python = Command::new("uv")
        .args([
            "run",
            "--frozen",
            "python",
            "-c",
            PYTHON_SCHEDULER,
            &uri,
            "2026",
        ])
        .current_dir(repo_root())
        .output()
        .expect("launch Python scheduler reference");
    assert!(
        python.status.success(),
        "Python scheduler failed:\n{}",
        String::from_utf8_lossy(&python.stderr)
    );
    let mut python_jobs: Vec<(String, String, String)> =
        serde_json::from_slice(&python.stdout).unwrap();
    python_jobs.sort();
    assert_eq!(python_jobs.len(), 3);

    delete_jobs(&db).await;
    let rust_scheduler = OnlineScoringScheduler::new(store, None);
    assert_eq!(rust_scheduler.run_once(2026).await.unwrap(), 3);
    let jobs = JobStore::new(db)
        .list_jobs(WS, None, &[], None, None, None)
        .await
        .unwrap();
    let mut rust_jobs = jobs
        .into_iter()
        .map(|job| (job.job_name, job.params, job.workspace))
        .collect::<Vec<_>>();
    rust_jobs.sort();

    assert_eq!(rust_jobs, python_jobs);
}

const PYTHON_SCHEDULER: &str = r#"
import json
import random
import sys
from unittest.mock import patch

from mlflow.genai.scorers import job as scheduler
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

uri, seed = sys.argv[1], int(sys.argv[2])
tracking = SqlAlchemyStore(uri, '/tmp/mlflow-artifacts')
jobs = SqlAlchemyJobStore(uri)

def submit(function, params, timeout=None, extra_envs=None):
    return jobs.create_job(function._job_fn_metadata.name, json.dumps(params), timeout)

random.seed(seed)
with patch.object(scheduler, '_get_tracking_store', return_value=tracking), patch.object(
    scheduler, 'submit_job', side_effect=submit
):
    scheduler.run_online_scoring_scheduler()

actual = [(job.job_name, job.params, job.workspace) for job in jobs.list_jobs()]
print(json.dumps(actual))
"#;

fn payload(name: &str, session_level: bool) -> String {
    json!({
        "name": name,
        "aggregations": null,
        "description": null,
        "is_session_level_scorer": session_level,
        "mlflow_version": "3.14.1.dev0",
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {
            "instructions": if session_level { "Judge {{ conversation }}" } else { "Judge {{ inputs }}" },
            "model": "gateway:/scheduler-diff-endpoint",
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
) {
    store
        .register_scorer(WS, experiment_id, name, &payload(name, session_level))
        .await
        .unwrap();
    store
        .upsert_online_scoring_config(WS, experiment_id, name, sample_rate, None)
        .await
        .unwrap();
}

async fn seed_endpoint(db: &Db) {
    match db {
        Db::Sqlite(pool) => {
            sqlx::query("INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES (?, ?, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, ?)")
                .bind("scheduler-diff-endpoint-id")
                .bind("scheduler-diff-endpoint")
                .bind(WS)
                .execute(pool)
                .await
                .unwrap();
        }
        Db::Postgres(pool) => {
            sqlx::query("INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES ($1, $2, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, $3)")
                .bind("scheduler-diff-endpoint-id")
                .bind("scheduler-diff-endpoint")
                .bind(WS)
                .execute(pool)
                .await
                .unwrap();
        }
        Db::MySql(pool) => {
            sqlx::query("INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES (?, ?, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, ?)")
                .bind("scheduler-diff-endpoint-id")
                .bind("scheduler-diff-endpoint")
                .bind(WS)
                .execute(pool)
                .await
                .unwrap();
        }
    }
}

async fn delete_jobs(db: &Db) {
    match db {
        Db::Sqlite(pool) => {
            sqlx::query("DELETE FROM jobs").execute(pool).await.unwrap();
        }
        Db::Postgres(pool) => {
            sqlx::query("DELETE FROM jobs").execute(pool).await.unwrap();
        }
        Db::MySql(pool) => {
            sqlx::query("DELETE FROM jobs").execute(pool).await.unwrap();
        }
    }
}
