//! Differential replay of the metric-history interval-sampling corpus (plan
//! T2.7 AC: "identical sampled point sets vs Python on dense histories").
//!
//! `rust/tools/gen_sampling_cases.py` seeds a real Alembic-migrated SQLite DB
//! (`tests/corpus/sampling/sampling.db`) with dense metric histories and dumps,
//! for each query case, the EXACT output of the genuine Python
//! `SqlAlchemyStore.get_metric_history_bulk_interval` into `cases.json`. This
//! test copies that DB to a temp file, runs the Rust store over each case's
//! params, and asserts byte-for-byte equality of the resulting point sets.
//!
//! Regenerate the corpus with:
//!   uv run --frozen python rust/tools/gen_sampling_cases.py

use std::path::{Path, PathBuf};

use mlflow_store::{Db, MetricWithRunId, PoolConfig, TrackingStore};
use serde::Deserialize;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
        .join("sampling")
}

#[derive(Debug, Deserialize)]
struct Case {
    run_ids: Vec<String>,
    metric_key: String,
    max_results: usize,
    start_step: Option<i64>,
    end_step: Option<i64>,
    expected: Vec<ExpectedPoint>,
}

#[derive(Debug, Deserialize)]
struct ExpectedPoint {
    run_id: String,
    key: String,
    /// A JSON number, or one of the string sentinels "NaN"/"Infinity"/
    /// "-Infinity" (JSON cannot carry those values natively).
    value: serde_json::Value,
    timestamp: i64,
    step: i64,
}

impl ExpectedPoint {
    fn value_f64(&self) -> f64 {
        match &self.value {
            serde_json::Value::Number(n) => n.as_f64().unwrap(),
            serde_json::Value::String(s) => match s.as_str() {
                "NaN" => f64::NAN,
                "Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                other => panic!("unexpected value sentinel: {other}"),
            },
            other => panic!("unexpected value json: {other:?}"),
        }
    }
}

/// Copy the corpus DB to a unique temp file; removed on drop.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_sampling_{}_{}.db",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(corpus_dir().join("sampling.db"), &path).expect("copy corpus db");
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

fn point_matches(got: &MetricWithRunId, exp: &ExpectedPoint) -> bool {
    let value_eq = {
        let g = got.metric.value;
        let e = exp.value_f64();
        (g.is_nan() && e.is_nan()) || g == e
    };
    got.run_id == exp.run_id
        && got.metric.key == exp.key
        && got.metric.timestamp == exp.timestamp
        && got.metric.step == exp.step
        && value_eq
}

#[tokio::test]
async fn interval_sampling_matches_python() {
    let cases: Vec<Case> = serde_json::from_slice(
        &std::fs::read(corpus_dir().join("cases.json")).expect("read cases.json"),
    )
    .expect("parse cases.json");

    assert!(!cases.is_empty(), "corpus must contain cases");

    let tmp = TempDb::new();
    let db = Db::connect(&tmp.uri(), PoolConfig::default())
        .await
        .expect("connect corpus db");
    let store = TrackingStore::new(db, ART_ROOT);

    let mut total_points = 0usize;
    for (ci, case) in cases.iter().enumerate() {
        let run_ids: Vec<&str> = case.run_ids.iter().map(String::as_str).collect();
        let got = store
            .get_metric_history_bulk_interval(
                WS,
                &run_ids,
                &case.metric_key,
                case.max_results,
                case.start_step,
                case.end_step,
            )
            .await
            .unwrap_or_else(|e| panic!("case {ci} errored: {e:?}"));

        assert_eq!(
            got.len(),
            case.expected.len(),
            "case {ci}: point count differs (rust={}, python={})",
            got.len(),
            case.expected.len()
        );

        // Exact ordered equality: the store concatenates per-run in request
        // order with a deterministic intra-run ORDER BY, matching Python.
        for (pi, (g, e)) in got.iter().zip(case.expected.iter()).enumerate() {
            assert!(
                point_matches(g, e),
                "case {ci} point {pi} differs:\n rust  = run={} key={} val={} ts={} step={}\n python= run={} key={} val={:?} ts={} step={}",
                g.run_id,
                g.metric.key,
                g.metric.value,
                g.metric.timestamp,
                g.metric.step,
                e.run_id,
                e.key,
                e.value,
                e.timestamp,
                e.step
            );
        }
        total_points += got.len();
    }

    // Guard against an accidentally-empty corpus silently passing.
    assert!(total_points > 2500, "expected dense corpus (>2500 points)");
}
