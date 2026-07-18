//! Generic jobs store and D20 queue-claim tests.

use std::sync::Arc;

use mlflow_error::ErrorCode;
use mlflow_store::{JobStatus, JobStore};
use mlflow_test_support::TempDb;
use serde_json::json;
use tokio::sync::Barrier;

const WS: &str = "default";

async fn store(temp: &TempDb) -> JobStore {
    JobStore::new(temp.connect().await)
}

#[tokio::test]
async fn lifecycle_retries_metadata_filters_and_workspace_scoping() {
    let temp = TempDb::new("jobs_lifecycle").await;
    let store = store(&temp).await;
    let job = store
        .create_job(WS, "evaluate", r#"{"experiment_id":"1","n":2}"#, Some(3.5))
        .await
        .unwrap();
    assert_eq!(job.status, JobStatus::Pending);
    assert_eq!(job.retry_count, 0);
    assert_eq!(job.timeout, Some(3.5));
    assert_eq!(job.status_details, None);

    store.start_job(WS, &job.job_id).await.unwrap();
    assert_eq!(
        store
            .retry_or_fail_job(WS, &job.job_id, "transient", 2)
            .await
            .unwrap(),
        Some(1)
    );
    store.start_job(WS, &job.job_id).await.unwrap();
    store
        .update_status_details(
            WS,
            &job.job_id,
            &json!({"stage": "running", "progress": 25}),
        )
        .await
        .unwrap();
    store
        .update_status_details(WS, &job.job_id, &json!({"progress": 50}))
        .await
        .unwrap();
    let finished = store
        .finish_job(WS, &job.job_id, r#"{"score":1}"#)
        .await
        .unwrap();
    assert_eq!(finished.status, JobStatus::Succeeded);
    assert_eq!(finished.parsed_result().unwrap(), Some(json!({"score": 1})));
    assert_eq!(
        finished.status_details,
        Some(json!({"stage": "running", "progress": 50}))
    );

    let listed = store
        .list_jobs(
            WS,
            Some("evaluate"),
            &[JobStatus::Succeeded],
            None,
            None,
            Some(&json!({"experiment_id": "1"})),
        )
        .await
        .unwrap();
    assert_eq!(listed.len(), 1);
    assert!(store.get_job("other", &job.job_id).await.is_err());

    let scalar_params = store.create_job(WS, "scalar", "1", None).await.unwrap();
    let unfiltered = store
        .list_jobs(WS, Some("scalar"), &[], None, None, Some(&json!({})))
        .await
        .unwrap();
    assert_eq!(unfiltered, [scalar_params]);
}

#[tokio::test]
async fn exhausted_retry_preserves_python_last_update_time_semantics() {
    let temp = TempDb::new("jobs_retry_timestamp").await;
    let store = store(&temp).await;
    let job = store.create_job(WS, "retry", "{}", None).await.unwrap();
    let before = store
        .get_job(WS, &job.job_id)
        .await
        .unwrap()
        .last_update_time;
    assert_eq!(
        store
            .retry_or_fail_job(WS, &job.job_id, "transient", 0)
            .await
            .unwrap(),
        None
    );
    let failed = store.get_job(WS, &job.job_id).await.unwrap();
    assert_eq!(failed.status, JobStatus::Failed);
    assert_eq!(failed.result.as_deref(), Some("transient"));
    assert_eq!(failed.last_update_time, before);
}

#[tokio::test]
async fn finalized_jobs_reject_every_status_transition_and_retry() {
    let temp = TempDb::new("jobs_finalized").await;
    let store = store(&temp).await;
    let job = store.create_job(WS, "final", "{}", None).await.unwrap();
    store.finish_job(WS, &job.job_id, "null").await.unwrap();

    let errors = [
        store.cancel_job(WS, &job.job_id).await.unwrap_err(),
        store.reset_job(WS, &job.job_id).await.unwrap_err(),
        store.fail_job(WS, &job.job_id, "late").await.unwrap_err(),
        store.mark_job_timed_out(WS, &job.job_id).await.unwrap_err(),
        store
            .retry_or_fail_job(WS, &job.job_id, "late", 3)
            .await
            .unwrap_err(),
    ];
    for error in errors {
        assert_eq!(error.error_code, ErrorCode::InternalError);
        assert!(
            error.message.contains("already finalized"),
            "{}",
            error.message
        );
    }
    assert_eq!(
        store.get_job(WS, &job.job_id).await.unwrap().status,
        JobStatus::Succeeded
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_claim_has_exactly_one_winner() {
    const RACERS: usize = 32;
    let temp = TempDb::new("jobs_claim").await;
    let store = Arc::new(store(&temp).await);
    let job = store.create_job(WS, "claimable", "{}", None).await.unwrap();
    let barrier = Arc::new(Barrier::new(RACERS));
    let mut tasks = Vec::with_capacity(RACERS);
    for _ in 0..RACERS {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            store.claim_next_job(WS, Some("claimable")).await.unwrap()
        }));
    }
    let mut winners = Vec::new();
    for task in tasks {
        if let Some(claimed) = task.await.unwrap() {
            winners.push(claimed);
        }
    }
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0].job_id, job.job_id);
    assert_eq!(winners[0].status, JobStatus::Running);
    assert!(store
        .claim_next_job(WS, Some("claimable"))
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn delete_jobs_only_removes_finalized_rows() {
    let temp = TempDb::new("jobs_delete").await;
    let store = store(&temp).await;
    let pending = store.create_job(WS, "pending", "{}", None).await.unwrap();
    let finished = store.create_job(WS, "finished", "{}", None).await.unwrap();
    store
        .finish_job(WS, &finished.job_id, "null")
        .await
        .unwrap();
    let deleted = store.delete_jobs(WS, 0, &[]).await.unwrap();
    assert_eq!(deleted, [finished.job_id]);
    assert_eq!(
        store.get_job(WS, &pending.job_id).await.unwrap().status,
        JobStatus::Pending
    );
}
