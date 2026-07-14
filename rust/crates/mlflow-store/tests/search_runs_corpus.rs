//! Differential replay of the `search_runs` corpus (plan T2.6 AC: "ordering +
//! page boundaries identical to Python across dialects").
//!
//! `rust/tools/gen_search_runs_corpus.py` seeds a real Alembic-migrated SQLite
//! DB (`tests/corpus/search_runs/search.db`) with runs/metrics/params/tags/
//! datasets crafted to exercise NULL-metric orderings, NaN metrics, start_time
//! ties, every filter entity type, and view_type filtering. For each query case
//! it walks EVERY page of the genuine Python `SqlAlchemyStore._search_runs` and
//! dumps the ordered run_id pages into `cases.json`.
//!
//! This test copies that DB to a temp file, runs the Rust `search_runs` over the
//! same case matrix — walking pages via the Rust opaque keyset tokens (whose
//! *contents* differ from Python's offset tokens by design, plan decision D3) —
//! and asserts the ordered run_id sequence AND the per-page boundaries match
//! Python's exactly.
//!
//! Regenerate the corpus with:
//!   uv run --frozen python rust/tools/gen_search_runs_corpus.py

use std::path::{Path, PathBuf};

use mlflow_store::{Db, PoolConfig, TrackingStore, ViewType};
use serde::Deserialize;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
        .join("search_runs")
}

#[derive(Debug, Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    label: String,
    experiment_ids: Vec<String>,
    filter: String,
    order_by: Vec<String>,
    view_type: String,
    max_results: i64,
    pages: Vec<Vec<String>>,
    ordered_run_ids: Vec<String>,
}

fn view_type(s: &str) -> ViewType {
    match s {
        "ACTIVE_ONLY" => ViewType::ActiveOnly,
        "DELETED_ONLY" => ViewType::DeletedOnly,
        "ALL" => ViewType::All,
        other => panic!("unknown view_type: {other}"),
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
            "mlflow_rust_search_{}_{}.db",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(corpus_dir().join("search.db"), &path).expect("copy corpus db");
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

// The committed corpus fixture (`tests/corpus/search_runs/{search.db,cases.json}`)
// is produced by `rust/tools/gen_search_runs_corpus.py` from the genuine Python
// `SqlAlchemyStore`. Regenerate with:
//
//   uv run --frozen python rust/tools/gen_search_runs_corpus.py
#[tokio::test]
async fn search_runs_matches_python() {
    let corpus: Corpus = serde_json::from_slice(
        &std::fs::read(corpus_dir().join("cases.json")).expect("read cases.json"),
    )
    .expect("parse cases.json");

    assert!(!corpus.cases.is_empty(), "corpus must contain cases");

    let tmp = TempDb::new();
    let db = Db::connect(&tmp.uri(), PoolConfig::default())
        .await
        .expect("connect corpus db");
    let store = TrackingStore::new(db, ART_ROOT);

    let mut total_pages = 0usize;
    for case in &corpus.cases {
        let filter = if case.filter.is_empty() {
            None
        } else {
            Some(case.filter.as_str())
        };
        let vt = view_type(&case.view_type);

        // Walk every page via the Rust keyset tokens.
        let mut got_pages: Vec<Vec<String>> = Vec::new();
        let mut got_all: Vec<String> = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..1000 {
            let page = store
                .search_runs(
                    WS,
                    &case.experiment_ids,
                    filter,
                    vt,
                    Some(case.max_results),
                    &case.order_by,
                    token.as_deref(),
                )
                .await
                .unwrap_or_else(|e| panic!("case {}: search_runs errored: {e:?}", case.label));
            let ids: Vec<String> = page.runs.iter().map(|r| r.info.run_id.clone()).collect();
            got_pages.push(ids.clone());
            got_all.extend(ids);
            match page.next_page_token {
                Some(t) => token = Some(t),
                None => break,
            }
        }

        // Ordered run_id sequence must match exactly.
        assert_eq!(
            got_all, case.ordered_run_ids,
            "case {}: ordered run_id sequence differs\n rust  ={:?}\n python={:?}",
            case.label, got_all, case.ordered_run_ids
        );
        // Per-page boundaries must match exactly.
        assert_eq!(
            got_pages, case.pages,
            "case {}: page boundaries differ\n rust  ={:?}\n python={:?}",
            case.label, got_pages, case.pages
        );
        total_pages += got_pages.len();
    }

    assert!(
        total_pages >= corpus.cases.len(),
        "each case yields >=1 page"
    );
}
