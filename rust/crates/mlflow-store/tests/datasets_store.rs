//! Behavioral integration tests for datasets / inputs / outputs and bulk metric
//! history (plan T2.7 + T2.8), ported from the Python store suites.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

use mlflow_store::{DatasetInputSpec, LoggedModelOutput, MetricInput, TrackingStore};
use mlflow_test_support::TempDb;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
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

async fn new_experiment(s: &TrackingStore) -> String {
    s.create_experiment(WS, &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap()
}

async fn new_run_in(s: &TrackingStore, exp_id: &str) -> String {
    s.create_run(WS, exp_id, None, Some(0), Some("run"), &[])
        .await
        .unwrap()
        .info
        .run_id
}

fn ds(name: &str, digest: &str) -> DatasetInputSpec {
    DatasetInputSpec {
        name: name.to_string(),
        digest: digest.to_string(),
        source_type: "local".to_string(),
        source: "path/to/data".to_string(),
        schema: Some("{\"cols\":1}".to_string()),
        profile: None,
        tags: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// log_inputs / get_run assembly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_inputs_and_get_run_assembly() {
    let tmp = TempDb::new("inputs").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;

    let mut d = ds("train", "abc123");
    d.tags = vec![
        ("mlflow.data.context".to_string(), "training".to_string()),
        ("custom".to_string(), "v".to_string()),
    ];
    s.log_inputs(WS, &rid, &[d], &[]).await.unwrap();

    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(run.inputs.dataset_inputs.len(), 1);
    let di = &run.inputs.dataset_inputs[0];
    assert_eq!(di.dataset.name, "train");
    assert_eq!(di.dataset.digest, "abc123");
    assert_eq!(di.dataset.source_type, "local");
    assert_eq!(di.dataset.source, "path/to/data");
    assert_eq!(di.dataset.schema.as_deref(), Some("{\"cols\":1}"));
    assert_eq!(di.dataset.profile, None);
    // Both input tags round-trip.
    assert_eq!(di.tags.len(), 2);
    assert!(di
        .tags
        .iter()
        .any(|t| t.key == "mlflow.data.context" && t.value == "training"));
    assert!(di.tags.iter().any(|t| t.key == "custom" && t.value == "v"));
}

#[tokio::test]
async fn log_inputs_dedup_within_call_by_name_digest() {
    let tmp = TempDb::new("dedupcall").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;

    // Two inputs with the same (name, digest): first occurrence wins, one edge.
    let mut first = ds("d", "dig");
    first.tags = vec![("k".to_string(), "first".to_string())];
    let mut second = ds("d", "dig");
    second.source = "different-source".to_string();
    second.tags = vec![("k".to_string(), "second".to_string())];

    s.log_inputs(WS, &rid, &[first, second], &[]).await.unwrap();

    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(run.inputs.dataset_inputs.len(), 1);
    let di = &run.inputs.dataset_inputs[0];
    // First occurrence's source is kept.
    assert_eq!(di.dataset.source, "path/to/data");
    assert_eq!(di.tags.len(), 1);
    assert_eq!(di.tags[0].value, "first");
}

#[tokio::test]
async fn dataset_dedup_across_runs_reuses_uuid() {
    let tmp = TempDb::new("dedupruns").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let r1 = new_run_in(&s, &exp).await;
    let r2 = new_run_in(&s, &exp).await;

    // Same dataset (name, digest) logged to two runs in the same experiment must
    // reuse the single datasets row (dedup on (experiment_id, name, digest)).
    s.log_inputs(WS, &r1, &[ds("shared", "d1")], &[])
        .await
        .unwrap();
    s.log_inputs(WS, &r2, &[ds("shared", "d1")], &[])
        .await
        .unwrap();

    // search_datasets should report exactly one summary (DISTINCT dedups it).
    let summaries = s.search_datasets(WS, &[&exp]).await.unwrap();
    let shared: Vec<_> = summaries.iter().filter(|x| x.name == "shared").collect();
    assert_eq!(shared.len(), 1, "one dataset summary despite two runs");
    assert_eq!(shared[0].digest, "d1");
    assert_eq!(shared[0].experiment_id, exp);

    // Both runs still resolve their dataset input.
    assert_eq!(
        s.get_run(WS, &r1)
            .await
            .unwrap()
            .inputs
            .dataset_inputs
            .len(),
        1
    );
    assert_eq!(
        s.get_run(WS, &r2)
            .await
            .unwrap()
            .inputs
            .dataset_inputs
            .len(),
        1
    );
}

#[tokio::test]
async fn log_inputs_idempotent_edge() {
    let tmp = TempDb::new("idempotent").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;

    // Logging the same dataset input to the same run twice must not create a
    // second edge (existing-input check on source_id/destination).
    s.log_inputs(WS, &rid, &[ds("d", "dig")], &[])
        .await
        .unwrap();
    s.log_inputs(WS, &rid, &[ds("d", "dig")], &[])
        .await
        .unwrap();

    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(run.inputs.dataset_inputs.len(), 1);
}

// ---------------------------------------------------------------------------
// model inputs / outputs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_inputs_and_outputs_roundtrip() {
    let tmp = TempDb::new("modelio").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;

    s.log_inputs(WS, &rid, &[], &["model-in-1", "model-in-2"])
        .await
        .unwrap();
    s.log_outputs(
        WS,
        &rid,
        &[
            LoggedModelOutput {
                model_id: "model-out-1".to_string(),
                step: 3,
            },
            LoggedModelOutput {
                model_id: "model-out-2".to_string(),
                step: 7,
            },
        ],
    )
    .await
    .unwrap();

    let run = s.get_run(WS, &rid).await.unwrap();
    let mut mi: Vec<_> = run
        .inputs
        .model_inputs
        .iter()
        .map(|m| m.model_id.clone())
        .collect();
    mi.sort();
    assert_eq!(mi, vec!["model-in-1", "model-in-2"]);

    let mut mo: Vec<_> = run
        .outputs
        .model_outputs
        .iter()
        .map(|m| (m.model_id.clone(), m.step))
        .collect();
    mo.sort();
    assert_eq!(
        mo,
        vec![
            ("model-out-1".to_string(), 3),
            ("model-out-2".to_string(), 7)
        ]
    );
}

#[tokio::test]
async fn model_input_upsert_is_idempotent() {
    let tmp = TempDb::new("modelupsert").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;

    // Same model-input edge twice: merge/upsert on the 4-col PK => one row.
    s.log_inputs(WS, &rid, &[], &["m"]).await.unwrap();
    s.log_inputs(WS, &rid, &[], &["m"]).await.unwrap();

    let run = s.get_run(WS, &rid).await.unwrap();
    assert_eq!(run.inputs.model_inputs.len(), 1);
}

// ---------------------------------------------------------------------------
// search_datasets
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_datasets_context_and_distinct() {
    let tmp = TempDb::new("searchds").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;

    // One dataset WITH a context tag, one WITHOUT.
    let mut with_ctx = ds("ctx", "d1");
    with_ctx.tags = vec![("mlflow.data.context".to_string(), "eval".to_string())];
    let no_ctx = ds("noctx", "d2");
    s.log_inputs(WS, &rid, &[with_ctx, no_ctx], &[])
        .await
        .unwrap();

    let summaries = s.search_datasets(WS, &[&exp]).await.unwrap();
    let ctx = summaries.iter().find(|x| x.name == "ctx").unwrap();
    assert_eq!(ctx.context.as_deref(), Some("eval"));
    // Dataset without the context tag still appears (LEFT JOIN), context = None.
    let noctx = summaries.iter().find(|x| x.name == "noctx").unwrap();
    assert_eq!(noctx.context, None);
    assert_eq!(noctx.digest, "d2");
}

#[tokio::test]
async fn search_datasets_scopes_to_workspace() {
    let tmp = TempDb::new("searchws").await;
    let s = store(&tmp).await;
    // Experiment lives in ws-a; searching from ws-b must return nothing.
    let exp = s
        .create_experiment("ws-a", &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap();
    let rid = s
        .create_run("ws-a", &exp, None, Some(0), Some("run"), &[])
        .await
        .unwrap()
        .info
        .run_id;
    s.log_inputs("ws-a", &rid, &[ds("d", "dig")], &[])
        .await
        .unwrap();

    let from_b = s.search_datasets("ws-b", &[&exp]).await.unwrap();
    assert!(from_b.is_empty(), "cross-workspace search returns nothing");
    let from_a = s.search_datasets("ws-a", &[&exp]).await.unwrap();
    assert_eq!(from_a.len(), 1);
}

// ---------------------------------------------------------------------------
// bulk metric history
// ---------------------------------------------------------------------------

fn m(key: &str, value: f64, timestamp: i64, step: i64) -> MetricInput {
    MetricInput {
        key: key.to_string(),
        value,
        timestamp,
        step,
        model_id: None,
        dataset_name: None,
        dataset_digest: None,
    }
}

#[tokio::test]
async fn get_metric_history_bulk_ordering_and_cap() {
    let tmp = TempDb::new("bulk").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let r_a = new_run_in(&s, &exp).await;
    let r_b = new_run_in(&s, &exp).await;

    // Log a few points to each run.
    for i in 0..5i64 {
        s.log_metric(WS, &r_a, &m("loss", i as f64, 100 + i, i))
            .await
            .unwrap();
        s.log_metric(WS, &r_b, &m("loss", (10 + i) as f64, 200 + i, i))
            .await
            .unwrap();
    }

    let ids = [r_a.as_str(), r_b.as_str()];
    let out = s
        .get_metric_history_bulk(WS, &ids, "loss", 25_000)
        .await
        .unwrap();
    assert_eq!(out.len(), 10);

    // Ordering: run_uuid (lexicographic) then timestamp/step/value. Assert the
    // run ids form two contiguous, lexicographically-ordered blocks.
    let mut sorted_ids = [r_a.clone(), r_b.clone()];
    sorted_ids.sort();
    let first_block: Vec<&str> = out
        .iter()
        .take_while(|x| x.run_id == sorted_ids[0])
        .map(|x| x.run_id.as_str())
        .collect();
    assert_eq!(first_block.len(), 5);
    assert!(out[5..].iter().all(|x| x.run_id == sorted_ids[1]));

    // Within a run, timestamps ascending.
    let ts: Vec<i64> = out[..5].iter().map(|x| x.metric.timestamp).collect();
    let mut sorted_ts = ts.clone();
    sorted_ts.sort();
    assert_eq!(ts, sorted_ts);

    // Cap applied globally.
    let capped = s
        .get_metric_history_bulk(WS, &ids, "loss", 3)
        .await
        .unwrap();
    assert_eq!(capped.len(), 3);
}

#[tokio::test]
async fn get_metric_history_bulk_filters_workspace() {
    let tmp = TempDb::new("bulkws").await;
    let s = store(&tmp).await;
    let exp = s
        .create_experiment("ws-a", &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap();
    let rid = s
        .create_run("ws-a", &exp, None, Some(0), Some("run"), &[])
        .await
        .unwrap()
        .info
        .run_id;
    s.log_metric("ws-a", &rid, &m("loss", 1.0, 1, 0))
        .await
        .unwrap();

    // From ws-b the run id is filtered out (not accessible) => empty.
    let out = s
        .get_metric_history_bulk("ws-b", &[rid.as_str()], "loss", 25_000)
        .await
        .unwrap();
    assert!(out.is_empty());
    // From ws-a it's visible.
    let out_a = s
        .get_metric_history_bulk("ws-a", &[rid.as_str()], "loss", 25_000)
        .await
        .unwrap();
    assert_eq!(out_a.len(), 1);
}

#[tokio::test]
async fn get_metric_history_bulk_interval_small_history_returns_all() {
    let tmp = TempDb::new("intervalsmall").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;
    for i in 0..10i64 {
        s.log_metric(WS, &rid, &m("loss", i as f64, 100 + i, i))
            .await
            .unwrap();
    }
    // Window smaller than max_results => every point returned.
    let out = s
        .get_metric_history_bulk_interval(WS, &[rid.as_str()], "loss", 2500, None, None)
        .await
        .unwrap();
    assert_eq!(out.len(), 10);
    let steps: Vec<i64> = out.iter().map(|x| x.metric.step).collect();
    assert_eq!(steps, (0..10).collect::<Vec<_>>());
}

#[tokio::test]
async fn get_metric_history_bulk_interval_empty_history() {
    let tmp = TempDb::new("intervalempty").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let rid = new_run_in(&s, &exp).await;
    let out = s
        .get_metric_history_bulk_interval(WS, &[rid.as_str()], "missing", 2500, None, None)
        .await
        .unwrap();
    assert!(out.is_empty());
}
