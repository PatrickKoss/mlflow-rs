//! Behavioral integration tests for [`mlflow_store::TrackingStore`] (plan
//! T2.4/T2.5), ported from the Python store suites
//! (`tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_{experiments,runs,core}.py`).
//!
//! Each test copies the checked-in SQLite fixture (a real Alembic-migrated DB at
//! head `b7e4c1a90f23`) into a temp file and operates on it, so the committed
//! fixture is never mutated. The default artifact root is a fixed URI so
//! artifact-URI assertions are deterministic.
//!
//! Postgres/MySQL variants of the concurrency stress test are gated behind
//! `MLFLOW_RUST_TEST_PG_URI` / `MLFLOW_RUST_TEST_MYSQL_URI` (plan §6 item 8).

use std::path::{Path, PathBuf};

use mlflow_store::{Db, MetricInput, PoolConfig, TrackingStore};

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

/// Copy the fixture to a unique temp file; the returned guard removes it on drop.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_trackstore_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixture_path(), &path).expect("copy fixture");
        TempDb { path }
    }

    fn uri(&self) -> String {
        format!("sqlite:///{}", self.path.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn store(temp: &TempDb) -> TrackingStore {
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect temp fixture");
    TrackingStore::new(db, ART_ROOT)
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_experiment_defaults_artifact_location() {
    let tmp = TempDb::new("create_exp");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "exp_a", None, &[]).await.unwrap();
    let exp = s.get_experiment(WS, &id).await.unwrap();
    assert_eq!(exp.name, "exp_a");
    assert_eq!(exp.lifecycle_stage, "active");
    assert_eq!(
        exp.artifact_location.as_deref(),
        Some(format!("{ART_ROOT}/{id}").as_str())
    );
    assert!(exp.creation_time.is_some());
    assert_eq!(exp.creation_time, exp.last_update_time);
}

#[tokio::test]
async fn create_experiment_with_tags() {
    let tmp = TempDb::new("create_exp_tags");
    let s = store(&tmp).await;
    let id = s
        .create_experiment(WS, "tagged", None, &[("team", "rust"), ("env", "test")])
        .await
        .unwrap();
    let exp = s.get_experiment(WS, &id).await.unwrap();
    let mut tags: Vec<_> = exp
        .tags
        .iter()
        .map(|t| (t.key.as_str(), t.value.as_deref().unwrap_or("")))
        .collect();
    tags.sort();
    assert_eq!(tags, vec![("env", "test"), ("team", "rust")]);
}

#[tokio::test]
async fn duplicate_experiment_name_conflicts() {
    let tmp = TempDb::new("dup_exp");
    let s = store(&tmp).await;
    s.create_experiment(WS, "dupe", None, &[]).await.unwrap();
    let err = s
        .create_experiment(WS, "dupe", None, &[])
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceAlreadyExists
    );
    assert!(err.message.contains("already exists"), "{}", err.message);
}

/// The deleted-experiment name-conflict case: re-creating with the name of a
/// *deleted* experiment still conflicts (unique `(workspace, name)`).
#[tokio::test]
async fn create_conflicts_with_deleted_experiment_name() {
    let tmp = TempDb::new("dup_deleted_exp");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "ghost", None, &[]).await.unwrap();
    s.delete_experiment(WS, &id).await.unwrap();
    // get_experiment_by_name still finds the deleted one (ALL view).
    assert_eq!(
        s.get_experiment_by_name(WS, "ghost")
            .await
            .unwrap()
            .unwrap()
            .experiment_id,
        id
    );
    let err = s
        .create_experiment(WS, "ghost", None, &[])
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceAlreadyExists
    );
}

#[tokio::test]
async fn get_experiment_missing_and_invalid_id() {
    let tmp = TempDb::new("missing_exp");
    let s = store(&tmp).await;
    let err = s.get_experiment(WS, "99999").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert_eq!(err.message, "No Experiment with id=99999 exists");

    let err = s.get_experiment(WS, "not_an_int").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::InvalidParameterValue
    );
    assert!(err.message.contains("must be a valid integer"));
}

#[tokio::test]
async fn delete_restore_experiment_cascades_to_runs() {
    let tmp = TempDb::new("del_restore_exp");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "cascade", None, &[]).await.unwrap();
    let r1 = s
        .create_run(WS, &id, None, Some(1), Some("r1"), &[])
        .await
        .unwrap();
    let r2 = s
        .create_run(WS, &id, None, Some(2), Some("r2"), &[])
        .await
        .unwrap();

    s.delete_experiment(WS, &id).await.unwrap();
    assert_eq!(
        s.get_experiment(WS, &id).await.unwrap().lifecycle_stage,
        "deleted"
    );
    // Child runs are soft-deleted with a deleted_time.
    for rid in [&r1.info.run_id, &r2.info.run_id] {
        let run = s.get_run(WS, rid).await.unwrap();
        assert_eq!(run.info.lifecycle_stage, "deleted");
    }

    s.restore_experiment(WS, &id).await.unwrap();
    assert_eq!(
        s.get_experiment(WS, &id).await.unwrap().lifecycle_stage,
        "active"
    );
    for rid in [&r1.info.run_id, &r2.info.run_id] {
        assert_eq!(
            s.get_run(WS, rid).await.unwrap().info.lifecycle_stage,
            "active"
        );
    }
}

#[tokio::test]
async fn rename_experiment_requires_active() {
    let tmp = TempDb::new("rename_exp");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "old", None, &[]).await.unwrap();
    s.rename_experiment(WS, &id, "renamed").await.unwrap();
    assert_eq!(s.get_experiment(WS, &id).await.unwrap().name, "renamed");

    s.delete_experiment(WS, &id).await.unwrap();
    let err = s.rename_experiment(WS, &id, "again").await.unwrap_err();
    assert_eq!(err.error_code, mlflow_error::ErrorCode::InvalidState);
    assert_eq!(err.message, "Cannot rename a non-active experiment.");
}

#[tokio::test]
async fn set_and_delete_experiment_tag() {
    let tmp = TempDb::new("exp_tag");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "et", None, &[]).await.unwrap();
    s.set_experiment_tag(WS, &id, "k", "v1").await.unwrap();
    s.set_experiment_tag(WS, &id, "k", "v2").await.unwrap(); // upsert
    let exp = s.get_experiment(WS, &id).await.unwrap();
    assert_eq!(
        exp.tags
            .iter()
            .find(|t| t.key == "k")
            .unwrap()
            .value
            .as_deref(),
        Some("v2")
    );

    s.delete_experiment_tag(WS, &id, "k").await.unwrap();
    let err = s.delete_experiment_tag(WS, &id, "k").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert!(err.message.contains("No tag with name: k"));
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_run_syncs_run_name_tag() {
    let tmp = TempDb::new("run_name");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "rn", None, &[]).await.unwrap();

    // Explicit run_name synthesizes the mlflow.runName tag.
    let run = s
        .create_run(WS, &id, Some("alice"), Some(100), Some("my-run"), &[])
        .await
        .unwrap();
    assert_eq!(run.info.run_name, "my-run");
    assert_eq!(run.info.status, "RUNNING");
    assert_eq!(run.info.user_id.as_deref(), Some("alice"));
    assert_eq!(run.info.start_time, Some(100));
    assert_eq!(tag(&run, "mlflow.runName"), Some("my-run"));
    assert!(run
        .info
        .artifact_uri
        .as_deref()
        .unwrap()
        .ends_with("/artifacts"));

    // Name derived from the tag when no run_name is given.
    let run = s
        .create_run(
            WS,
            &id,
            None,
            Some(0),
            None,
            &[("mlflow.runName", "from-tag")],
        )
        .await
        .unwrap();
    assert_eq!(run.info.run_name, "from-tag");

    // Random name when neither is given.
    let run = s
        .create_run(WS, &id, None, Some(0), None, &[])
        .await
        .unwrap();
    assert!(!run.info.run_name.is_empty());
    assert_eq!(
        tag(&run, "mlflow.runName"),
        Some(run.info.run_name.as_str())
    );
}

#[tokio::test]
async fn create_run_conflicting_name_and_tag_errors() {
    let tmp = TempDb::new("run_conflict");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "rc", None, &[]).await.unwrap();
    let err = s
        .create_run(
            WS,
            &id,
            None,
            Some(0),
            Some("a"),
            &[("mlflow.runName", "b")],
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::InvalidParameterValue
    );
    assert_eq!(
        err.message,
        "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with different \
         values (run_name='a', mlflow.runName='b')."
    );
}

#[tokio::test]
async fn get_run_not_found() {
    let tmp = TempDb::new("run_missing");
    let s = store(&tmp).await;
    let err = s.get_run(WS, "nonexistent").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert_eq!(err.message, "Run with id=nonexistent not found");
}

#[tokio::test]
async fn update_run_info_syncs_name_and_status() {
    let tmp = TempDb::new("update_run");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "ur", None, &[]).await.unwrap();
    let run = s
        .create_run(WS, &id, None, Some(0), Some("start"), &[])
        .await
        .unwrap();
    let rid = &run.info.run_id;

    let info = s
        .update_run_info(WS, rid, Some("FINISHED"), Some(999), Some("new name"))
        .await
        .unwrap();
    assert_eq!(info.status, "FINISHED");
    assert_eq!(info.end_time, Some(999));
    assert_eq!(info.run_name, "new name");
    let run = s.get_run(WS, rid).await.unwrap();
    assert_eq!(tag(&run, "mlflow.runName"), Some("new name"));

    // set_tag on mlflow.runName also drives info.run_name.
    s.set_tag(WS, rid, "mlflow.runName", "via-tag")
        .await
        .unwrap();
    assert_eq!(s.get_run(WS, rid).await.unwrap().info.run_name, "via-tag");
}

#[tokio::test]
async fn delete_restore_run_lifecycle() {
    let tmp = TempDb::new("del_run");
    let s = store(&tmp).await;
    let id = s.create_experiment(WS, "dr", None, &[]).await.unwrap();
    let run = s
        .create_run(WS, &id, None, Some(0), Some("x"), &[])
        .await
        .unwrap();
    let rid = &run.info.run_id;

    s.delete_run(WS, rid).await.unwrap();
    assert_eq!(
        s.get_run(WS, rid).await.unwrap().info.lifecycle_stage,
        "deleted"
    );
    // Idempotent.
    s.delete_run(WS, rid).await.unwrap();

    s.restore_run(WS, rid).await.unwrap();
    assert_eq!(
        s.get_run(WS, rid).await.unwrap().info.lifecycle_stage,
        "active"
    );
    s.restore_run(WS, rid).await.unwrap(); // idempotent

    // Logging to a deleted run is rejected.
    s.delete_run(WS, rid).await.unwrap();
    let err = s.log_param(WS, rid, "k", "v").await.unwrap_err();
    assert!(
        err.message.contains("must be in the 'active' state"),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// Params & tags
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_param_immutability() {
    let tmp = TempDb::new("param_immut");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;

    s.log_param(WS, &rid, "depth", "3").await.unwrap();
    // Same value: idempotent OK.
    s.log_param(WS, &rid, "depth", "3").await.unwrap();
    // Different value: rejected with exact message.
    let err = s.log_param(WS, &rid, "depth", "5").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::InvalidParameterValue
    );
    assert_eq!(
        err.message,
        format!(
            "Changing param values is not allowed. Param with key='depth' was already logged \
             with value='3' for run ID='{rid}'. Attempted logging new value '5'."
        )
    );

    // Empty-string value is valid.
    s.log_param(WS, &rid, "empty", "").await.unwrap();
    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(param(&run, "empty"), Some(""));

    // 6000-char value OK, 6001 rejected.
    s.log_param(WS, &rid, "big", &"x".repeat(6000))
        .await
        .unwrap();
    let err = s
        .log_param(WS, &rid, "big2", &"x".repeat(6001))
        .await
        .unwrap_err();
    assert!(err.message.contains("exceeds the maximum length"));
}

#[tokio::test]
async fn set_and_delete_tag() {
    let tmp = TempDb::new("tags");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;

    s.set_tag(WS, &rid, "phase", "train").await.unwrap();
    s.set_tag(WS, &rid, "phase", "test").await.unwrap(); // upsert
    assert_eq!(param_tag(&s, &rid, "phase").await, Some("test".to_string()));

    s.delete_tag(WS, &rid, "phase").await.unwrap();
    let err = s.delete_tag(WS, &rid, "phase").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert_eq!(
        err.message,
        format!("No tag with name: phase in run with id {rid}")
    );
}

// ---------------------------------------------------------------------------
// Metrics + latest_metrics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_metric_nan_inf_storage() {
    let tmp = TempDb::new("metric_naninf");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;

    s.log_metric(WS, &rid, &m("nan", f64::NAN, 0, 0))
        .await
        .unwrap();
    s.log_metric(WS, &rid, &m("posinf", f64::INFINITY, 0, 0))
        .await
        .unwrap();
    s.log_metric(WS, &rid, &m("neginf", f64::NEG_INFINITY, 0, 0))
        .await
        .unwrap();

    let run = s.get_run(WS, &rid).await.unwrap();
    let nan = run.data.metrics.iter().find(|m| m.key == "nan").unwrap();
    assert!(nan.value.is_nan(), "NaN must round-trip as NaN");
    assert_eq!(metric_val(&run, "posinf"), Some(1.7976931348623157e308));
    assert_eq!(metric_val(&run, "neginf"), Some(-1.7976931348623157e308));
}

/// Latest-metric tie-break over out-of-order steps/timestamps/values: winner is
/// the lexicographic max of `(step, timestamp, value)`.
#[tokio::test]
async fn latest_metrics_tiebreak_out_of_order() {
    let tmp = TempDb::new("latest_tiebreak");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;

    // Logged in reversed order to prove order-independence (from the Python
    // test `..._run_data_uses_max_ts_value`).
    let points = [
        (-1i64, 800i64, 800.0),
        (-3, 900, 900.0),
        (3, 50, 20.0),
        (3, 50, 20.0), // duplicate identical — OK
        (3, 50, 10.0),
        (3, 40, 100.0),
        (0, 100, 1000.0),
    ];
    for (step, ts, val) in points {
        s.log_metric(WS, &rid, &m("k", val, ts, step))
            .await
            .unwrap();
    }

    let run = s.get_run(WS, &rid).await.unwrap();
    let latest = run.data.metrics.iter().find(|mm| mm.key == "k").unwrap();
    assert_eq!(latest.step, 3);
    assert_eq!(latest.timestamp, 50);
    assert_eq!(latest.value, 20.0);
    // Only one latest per key.
    assert_eq!(
        run.data.metrics.iter().filter(|mm| mm.key == "k").count(),
        1
    );

    // Full history contains every distinct point (the duplicate deduped by PK).
    let (hist, _) = s
        .get_metric_history(WS, &rid, "k", None, None)
        .await
        .unwrap();
    assert_eq!(hist.len(), 6);
}

#[tokio::test]
async fn metric_history_pagination() {
    let tmp = TempDb::new("metric_hist_page");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    for i in 0..10 {
        s.log_metric(WS, &rid, &m("acc", i as f64, 1000 + i, i))
            .await
            .unwrap();
    }

    // No max_results → all 10.
    let (all, tok) = s
        .get_metric_history(WS, &rid, "acc", None, None)
        .await
        .unwrap();
    assert_eq!(all.len(), 10);
    assert!(tok.is_none());

    // Paginate with page size 4: 4, 4, 2.
    let (p1, t1) = s
        .get_metric_history(WS, &rid, "acc", Some(4), None)
        .await
        .unwrap();
    assert_eq!(p1.len(), 4);
    let t1 = t1.expect("first page has a token");
    let (p2, t2) = s
        .get_metric_history(WS, &rid, "acc", Some(4), Some(&t1))
        .await
        .unwrap();
    assert_eq!(p2.len(), 4);
    let t2 = t2.expect("second page has a token");
    let (p3, t3) = s
        .get_metric_history(WS, &rid, "acc", Some(4), Some(&t2))
        .await
        .unwrap();
    assert_eq!(p3.len(), 2);
    assert!(t3.is_none());

    // Ordered by timestamp,step,value ascending; the concatenation covers 0..10.
    let mut vals: Vec<f64> = p1.iter().chain(&p2).chain(&p3).map(|mm| mm.value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, (0..10).map(|i| i as f64).collect::<Vec<_>>());

    // Invalid token errors.
    let err = s
        .get_metric_history(WS, &rid, "acc", Some(4), Some("!! not b64 !!"))
        .await
        .unwrap_err();
    assert!(err.message.contains("Invalid page token"));
}

#[tokio::test]
async fn metric_history_max_results_zero_and_over() {
    let tmp = TempDb::new("metric_hist_edge");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    for i in 0..5 {
        s.log_metric(WS, &rid, &m("acc", i as f64, 1000 + i, i))
            .await
            .unwrap();
    }
    let (p0, _) = s
        .get_metric_history(WS, &rid, "acc", Some(0), None)
        .await
        .unwrap();
    assert_eq!(p0.len(), 0);
    let (p10, tok) = s
        .get_metric_history(WS, &rid, "acc", Some(10), None)
        .await
        .unwrap();
    assert_eq!(p10.len(), 5);
    assert!(tok.is_none());
}

// ---------------------------------------------------------------------------
// log_batch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_batch_happy_path() {
    let tmp = TempDb::new("batch_ok");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    s.log_batch(
        WS,
        &rid,
        &[m("m1", 0.87, 12345, 0), m("m2", 0.49, 12345, 1)],
        &[("p1", "p1val"), ("p2", "p2val")],
        &[("t1", "t1val"), ("mlflow.runName", "my_run")],
    )
    .await
    .unwrap();
    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(param(&run, "p1"), Some("p1val"));
    assert_eq!(param(&run, "p2"), Some("p2val"));
    assert_eq!(tag(&run, "t1"), Some("t1val"));
    assert_eq!(run.info.run_name, "my_run");
    assert_eq!(metric_val(&run, "m1"), Some(0.87));
    assert_eq!(metric_val(&run, "m2"), Some(0.49));
}

#[tokio::test]
async fn log_batch_limits_rejected() {
    let tmp = TempDb::new("batch_limits");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;

    // 101 params > 100 cap.
    let params: Vec<(String, String)> = (0..101)
        .map(|i| (format!("p{i}"), "v".to_string()))
        .collect();
    let param_refs: Vec<(&str, &str)> = params
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let err = s
        .log_batch(WS, &rid, &[], &param_refs, &[])
        .await
        .unwrap_err();
    assert!(
        err.message.contains("at most 100 params. Got 101 params"),
        "{}",
        err.message
    );

    // 1001 metrics > 1000 cap.
    let metrics: Vec<MetricInput> = (0..1001).map(|i| m("k", i as f64, 0, i)).collect();
    let err = s.log_batch(WS, &rid, &metrics, &[], &[]).await.unwrap_err();
    assert!(
        err.message
            .contains("at most 1000 metrics. Got 1001 metrics"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn log_batch_param_immutability_is_atomic() {
    let tmp = TempDb::new("batch_atomic");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    s.log_param(WS, &rid, "existing", "orig").await.unwrap();

    // A batch that tries to overwrite `existing` must abort entirely — no metric
    // or tag from the same batch persists.
    let err = s
        .log_batch(
            WS,
            &rid,
            &[m("should_not_persist", 1.0, 0, 0)],
            &[("existing", "changed")],
            &[("also_not", "persist")],
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::InvalidParameterValue
    );
    assert!(err.message.contains("Changing param values is not allowed"));

    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(param(&run, "existing"), Some("orig"));
    assert!(metric_val(&run, "should_not_persist").is_none());
    assert!(tag(&run, "also_not").is_none());
}

#[tokio::test]
async fn log_batch_duplicate_param_keys_rejected() {
    let tmp = TempDb::new("batch_dupkeys");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    let err = s
        .log_batch(WS, &rid, &[], &[("k", "1"), ("k", "2")], &[])
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("Duplicate parameter keys have been submitted"),
        "{}",
        err.message
    );
    // Nothing logged.
    assert!(s.get_run(WS, &rid).await.unwrap().data.params.is_empty());
}

#[tokio::test]
async fn log_batch_duplicate_metrics_ok() {
    let tmp = TempDb::new("batch_dupmetrics");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    // Same identical metric twice in one batch — OK (deduped by PK).
    s.log_batch(WS, &rid, &[m("k", 1.0, 5, 0), m("k", 1.0, 5, 0)], &[], &[])
        .await
        .unwrap();
    let (hist, _) = s
        .get_metric_history(WS, &rid, "k", None, None)
        .await
        .unwrap();
    assert_eq!(hist.len(), 1);
}

#[tokio::test]
async fn log_batch_unchanged_and_new_params_ok() {
    let tmp = TempDb::new("batch_unchanged");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;
    s.log_param(WS, &rid, "a", "0").await.unwrap();
    // Re-log a=0 (same value) plus new b, c — allowed.
    s.log_batch(WS, &rid, &[], &[("a", "0"), ("b", "1"), ("c", "2")], &[])
        .await
        .unwrap();
    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(param(&run, "a"), Some("0"));
    assert_eq!(param(&run, "b"), Some("1"));
    assert_eq!(param(&run, "c"), Some("2"));
}

// ---------------------------------------------------------------------------
// Workspace isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workspace_isolation() {
    let tmp = TempDb::new("workspaces");
    let s = store(&tmp).await;
    // Same name in two workspaces is allowed (unique is (workspace, name)).
    let id_a = s
        .create_experiment("ws-a", "shared", None, &[])
        .await
        .unwrap();
    let id_b = s
        .create_experiment("ws-b", "shared", None, &[])
        .await
        .unwrap();
    assert_ne!(id_a, id_b);

    // Cross-workspace get is denied.
    assert!(s.get_experiment("ws-b", &id_a).await.is_err());
    assert!(s.get_experiment("ws-a", &id_a).await.is_ok());

    // Runs are scoped too.
    let run = s
        .create_run("ws-a", &id_a, None, Some(0), Some("r"), &[])
        .await
        .unwrap();
    let err = s.get_run("ws-b", &run.info.run_id).await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert_eq!(
        err.message,
        format!("Run with id={} not found", run.info.run_id)
    );
    assert!(s.get_run("ws-a", &run.info.run_id).await.is_ok());
}

// ---------------------------------------------------------------------------
// Concurrency: latest_metrics atomic upsert under parallel writers (sqlite)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_latest_metrics_is_correct() {
    let tmp = TempDb::new("concurrent");
    let s = store(&tmp).await;
    let rid = new_run(&s).await;

    // Many tasks race to log the same metric key with increasing steps. The
    // atomic upsert must leave latest = the max-step point regardless of
    // interleaving. SQLite serializes writers (busy_timeout=20s), so we assert
    // correctness rather than deadlock-freedom (that's the pg/mysql variant).
    let n = 50i64;
    let mut handles = Vec::new();
    for step in 0..n {
        let store = s.clone();
        let rid = rid.clone();
        handles.push(tokio::spawn(async move {
            // value intentionally decreases as step increases, to prove step —
            // not value — decides the winner.
            store
                .log_metric(WS, &rid, &m("race", (n - step) as f64, 1000 + step, step))
                .await
        }));
    }
    for h in handles {
        h.await.unwrap().expect("log_metric under contention");
    }

    let run = s.get_run(WS, &rid).await.unwrap();
    let latest = run.data.metrics.iter().find(|mm| mm.key == "race").unwrap();
    assert_eq!(latest.step, n - 1, "latest must be the max-step point");
    assert_eq!(latest.value, 1.0);
    let (hist, _) = s
        .get_metric_history(WS, &rid, "race", None, None)
        .await
        .unwrap();
    assert_eq!(hist.len(), n as usize);
}

// Postgres stress variant (gated). Runs many concurrent writers against the
// atomic upsert and asserts correctness + no deadlocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_latest_metrics_postgres() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_PG_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_PG_URI not set");
        return;
    };
    run_pg_mysql_stress(&uri).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_latest_metrics_mysql() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_MYSQL_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_MYSQL_URI not set");
        return;
    };
    run_pg_mysql_stress(&uri).await;
}

/// A broad behavioral smoke over pg/mysql (gated) covering the same
/// create/run/param/metric/batch flows as the sqlite tests, to catch
/// dialect-specific decode/encode issues (e.g. Postgres `experiment_id` INT4).
#[tokio::test]
async fn pg_behavioral_smoke() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_PG_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_PG_URI not set");
        return;
    };
    behavioral_smoke(&uri).await;
}

#[tokio::test]
async fn mysql_behavioral_smoke() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_MYSQL_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_MYSQL_URI not set");
        return;
    };
    behavioral_smoke(&uri).await;
}

async fn behavioral_smoke(uri: &str) {
    let db = Db::connect_and_verify(uri).await.expect("connect");
    let s = TrackingStore::new(db, ART_ROOT);
    let ws = format!("rust-smoke-{}", std::process::id());
    let name = format!("smoke-{}", uuid_like());

    let id = s
        .create_experiment(&ws, &name, None, &[("t", "v")])
        .await
        .unwrap();
    let exp = s.get_experiment(&ws, &id).await.unwrap();
    assert_eq!(exp.name, name);
    assert_eq!(
        exp.artifact_location.as_deref(),
        Some(format!("{ART_ROOT}/{id}").as_str())
    );

    // Deleted-name conflict.
    s.delete_experiment(&ws, &id).await.unwrap();
    let e = s
        .create_experiment(&ws, &name, None, &[])
        .await
        .unwrap_err();
    assert_eq!(e.error_code, mlflow_error::ErrorCode::ResourceAlreadyExists);
    s.restore_experiment(&ws, &id).await.unwrap();

    let run = s
        .create_run(&ws, &id, Some("u"), Some(1), Some("r"), &[])
        .await
        .unwrap();
    let rid = run.info.run_id.clone();
    assert_eq!(run.info.status, "RUNNING");

    s.log_param(&ws, &rid, "p", "1").await.unwrap();
    let e = s.log_param(&ws, &rid, "p", "2").await.unwrap_err();
    assert!(e.message.contains("Changing param values is not allowed"));

    // NaN/Inf + out-of-order latest.
    for (step, ts, v) in [
        (0i64, 100i64, 1.0),
        (3, 50, 20.0),
        (3, 50, 10.0),
        (3, 40, 100.0),
    ] {
        s.log_metric(&ws, &rid, &m("k", v, ts, step)).await.unwrap();
    }
    s.log_metric(&ws, &rid, &m("nan", f64::NAN, 0, 0))
        .await
        .unwrap();
    s.log_metric(&ws, &rid, &m("inf", f64::INFINITY, 0, 0))
        .await
        .unwrap();
    let run = s.get_run(&ws, &rid).await.unwrap();
    let latest = run.data.metrics.iter().find(|x| x.key == "k").unwrap();
    assert_eq!((latest.step, latest.timestamp, latest.value), (3, 50, 20.0));
    assert!(run
        .data
        .metrics
        .iter()
        .find(|x| x.key == "nan")
        .unwrap()
        .value
        .is_nan());
    assert_eq!(metric_val(&run, "inf"), Some(1.7976931348623157e308));

    // Batch atomicity.
    let e = s
        .log_batch(&ws, &rid, &[m("nope", 1.0, 0, 0)], &[("p", "changed")], &[])
        .await
        .unwrap_err();
    assert!(e.message.contains("Changing param values is not allowed"));
    assert!(metric_val(&s.get_run(&ws, &rid).await.unwrap(), "nope").is_none());

    // Metric history pagination.
    for i in 0..6 {
        s.log_metric(&ws, &rid, &m("hist", i as f64, 2000 + i, i))
            .await
            .unwrap();
    }
    let (p1, t1) = s
        .get_metric_history(&ws, &rid, "hist", Some(4), None)
        .await
        .unwrap();
    assert_eq!(p1.len(), 4);
    let (p2, t2) = s
        .get_metric_history(&ws, &rid, "hist", Some(4), t1.as_deref())
        .await
        .unwrap();
    assert_eq!(p2.len(), 2);
    assert!(t2.is_none());
}

async fn run_pg_mysql_stress(uri: &str) {
    let db = Db::connect_and_verify(uri).await.expect("connect pg/mysql");
    let s = TrackingStore::new(db, ART_ROOT);
    // Use a unique workspace so parallel CI shards don't collide.
    let ws = format!("rust-stress-{}", std::process::id());
    let id = s.create_experiment(&ws, "stress", None, &[]).await.unwrap();
    let run = s
        .create_run(&ws, &id, None, Some(0), Some("r"), &[])
        .await
        .unwrap();
    let rid = run.info.run_id;

    let n = 200i64;
    let mut handles = Vec::new();
    for step in 0..n {
        let store = s.clone();
        let rid = rid.clone();
        let ws = ws.clone();
        handles.push(tokio::spawn(async move {
            store
                .log_metric(&ws, &rid, &m("race", (n - step) as f64, 1000 + step, step))
                .await
        }));
    }
    for h in handles {
        h.await.unwrap().expect("no deadlock / error");
    }
    let run = s.get_run(&ws, &rid).await.unwrap();
    let latest = run.data.metrics.iter().find(|mm| mm.key == "race").unwrap();
    assert_eq!(latest.step, n - 1);
    assert_eq!(latest.value, 1.0);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn m(key: &str, value: f64, timestamp: i64, step: i64) -> MetricInput {
    MetricInput {
        key: key.to_string(),
        value,
        timestamp,
        step,
    }
}

async fn new_run(s: &TrackingStore) -> String {
    let id = s
        .create_experiment(WS, &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap();
    s.create_run(WS, &id, None, Some(0), Some("run"), &[])
        .await
        .unwrap()
        .info
        .run_id
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}_{}",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    )
}

fn tag<'a>(run: &'a mlflow_store::Run, key: &str) -> Option<&'a str> {
    run.data
        .tags
        .iter()
        .find(|t| t.key == key)
        .map(|t| t.value.as_str())
}

fn param<'a>(run: &'a mlflow_store::Run, key: &str) -> Option<&'a str> {
    run.data
        .params
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.as_str())
}

fn metric_val(run: &mlflow_store::Run, key: &str) -> Option<f64> {
    run.data
        .metrics
        .iter()
        .find(|mm| mm.key == key)
        .map(|mm| mm.value)
}

async fn param_tag(s: &TrackingStore, rid: &str, key: &str) -> Option<String> {
    let run = s.get_run(WS, rid).await.unwrap();
    run.data
        .tags
        .iter()
        .find(|t| t.key == key)
        .map(|t| t.value.clone())
}

// ---------------------------------------------------------------------------
// search_experiments (plan T3.1 store layer)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_experiments_active_and_all_view_types() {
    use mlflow_store::ViewType;
    let tmp = TempDb::new("search_exp_views");
    let s = store(&tmp).await;

    let a = s.create_experiment(WS, "se_a", None, &[]).await.unwrap();
    let _b = s.create_experiment(WS, "se_b", None, &[]).await.unwrap();
    s.delete_experiment(WS, &a).await.unwrap();

    // ACTIVE_ONLY excludes the deleted one.
    let page = s
        .search_experiments(WS, Some(ViewType::ActiveOnly), 100, None, &[], None)
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"se_b"));
    assert!(!names.contains(&"se_a"));

    // DELETED_ONLY includes only the deleted one.
    let page = s
        .search_experiments(WS, Some(ViewType::DeletedOnly), 100, None, &[], None)
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"se_a"));
    assert!(!names.contains(&"se_b"));

    // ALL includes both.
    let page = s
        .search_experiments(WS, Some(ViewType::All), 100, None, &[], None)
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"se_a"));
    assert!(names.contains(&"se_b"));
}

#[tokio::test]
async fn search_experiments_unspecified_view_type_returns_empty() {
    let tmp = TempDb::new("search_exp_unspecified");
    let s = store(&tmp).await;
    // None mirrors an unset proto ViewType (0) → empty stages → no rows.
    let page = s
        .search_experiments(WS, None, 100, None, &[], None)
        .await
        .unwrap();
    assert!(page.experiments.is_empty());
    assert!(page.next_page_token.is_none());
}

#[tokio::test]
async fn search_experiments_filter_by_name_and_tag() {
    use mlflow_store::ViewType;
    let tmp = TempDb::new("search_exp_filter");
    let s = store(&tmp).await;
    s.create_experiment(WS, "filter_a", None, &[("team", "rust")])
        .await
        .unwrap();
    s.create_experiment(WS, "filter_b", None, &[])
        .await
        .unwrap();

    let page = s
        .search_experiments(
            WS,
            Some(ViewType::All),
            100,
            Some("name = 'filter_a'"),
            &[],
            None,
        )
        .await
        .unwrap();
    assert_eq!(page.experiments.len(), 1);
    assert_eq!(page.experiments[0].name, "filter_a");

    // tag filter (the fixture's `rust_store_fixture` also carries team=rust).
    let page = s
        .search_experiments(
            WS,
            Some(ViewType::All),
            100,
            Some("tags.team = 'rust'"),
            &[],
            None,
        )
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"filter_a"));
    assert!(!names.contains(&"filter_b"));
}

#[tokio::test]
async fn search_experiments_order_by_name() {
    use mlflow_store::ViewType;
    let tmp = TempDb::new("search_exp_order");
    let s = store(&tmp).await;
    s.create_experiment(WS, "zzz", None, &[]).await.unwrap();
    s.create_experiment(WS, "aaa", None, &[]).await.unwrap();

    let page = s
        .search_experiments(
            WS,
            Some(ViewType::All),
            100,
            None,
            &["name ASC".to_string()],
            None,
        )
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    let aaa = names.iter().position(|n| *n == "aaa").unwrap();
    let zzz = names.iter().position(|n| *n == "zzz").unwrap();
    assert!(aaa < zzz);
}

#[tokio::test]
async fn search_experiments_pagination_token_walk() {
    use mlflow_store::ViewType;
    let tmp = TempDb::new("search_exp_page");
    let s = store(&tmp).await;
    for i in 0..5 {
        s.create_experiment(WS, &format!("page_{i}"), None, &[])
            .await
            .unwrap();
    }

    let mut seen = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let page = s
            .search_experiments(WS, Some(ViewType::All), 2, None, &[], token.as_deref())
            .await
            .unwrap();
        assert!(page.experiments.len() <= 2);
        for e in &page.experiments {
            seen.push(e.name.clone());
        }
        match page.next_page_token {
            Some(t) => token = Some(t),
            None => break,
        }
    }
    // 2 fixture experiments + 5 created = 7 total, no duplicates.
    let mut sorted = seen.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), seen.len());
    assert!(seen.len() >= 7);
}

#[tokio::test]
async fn search_experiments_rejects_bad_max_results() {
    use mlflow_store::ViewType;
    let tmp = TempDb::new("search_exp_maxres");
    let s = store(&tmp).await;

    let err = s
        .search_experiments(WS, Some(ViewType::All), 0, None, &[], None)
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::InvalidParameterValue
    );
    assert!(err.message.contains("must be a positive integer"));

    let err = s
        .search_experiments(WS, Some(ViewType::All), 99999, None, &[], None)
        .await
        .unwrap_err();
    assert!(err.message.contains("at most 50000"));
}

#[tokio::test]
async fn search_experiments_is_workspace_scoped() {
    use mlflow_store::ViewType;
    let tmp = TempDb::new("search_exp_ws");
    let s = store(&tmp).await;
    // Create an experiment in a non-default workspace; the default-workspace
    // search must not see it.
    s.create_experiment("other_ws", "isolated", None, &[])
        .await
        .unwrap();

    let page = s
        .search_experiments(WS, Some(ViewType::All), 100, None, &[], None)
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    assert!(!names.contains(&"isolated"));

    let page = s
        .search_experiments("other_ws", Some(ViewType::All), 100, None, &[], None)
        .await
        .unwrap();
    let names: Vec<&str> = page.experiments.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"isolated"));
}
