//! Differential replay of the registry-search corpus (plan T7.3 AC: "generate a
//! differential corpus against the real Python store").
//!
//! `rust/tools/gen_registry_search_corpus.py` seeds a real Alembic-migrated
//! SQLite DB (`tests/corpus/search/registry.db`) — through the GENUINE Python
//! model-registry `SqlAlchemyStore` — with registered models, versions, tags,
//! aliases, and prompt-tagged rows crafted to exercise name/tag filters, MV
//! version_number/run_id-IN/source aliases, AND-of-tags, prompt in/exclusion,
//! order_by variants + tiebreaks, deleted-MV visibility, and multi-page walks.
//! For each case it walks EVERY page of the genuine Python store and records the
//! ordered result identifiers, per-page boundaries, and the offset page tokens.
//!
//! This test copies that DB to a temp file, runs the Rust
//! `search_registered_models` / `search_model_versions` over the same case
//! matrix — walking pages via the Rust offset tokens — and asserts the ordered
//! identifier sequence, per-page boundaries, AND the page-token contents match
//! Python's exactly. Registry search keeps Python's offset tokens (plan T7.3),
//! so the tokens must match byte-for-byte.
//!
//! Regenerate the corpus with:
//!   uv run python rust/tools/gen_registry_search_corpus.py

use std::path::{Path, PathBuf};

use mlflow_registry::RegistryStore;
use mlflow_store::{Db, PoolConfig};
use serde::Deserialize;

const WS: &str = "default";

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
        .join("search")
}

#[derive(Debug, Deserialize)]
struct Corpus {
    registered_models: Vec<RmCase>,
    model_versions: Vec<MvCase>,
}

#[derive(Debug, Deserialize)]
struct RmCase {
    label: String,
    filter: Option<String>,
    order_by: Vec<String>,
    max_results: i64,
    pages: Vec<Vec<String>>,
    page_tokens: Vec<Option<String>>,
    ordered_names: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MvCase {
    label: String,
    filter: Option<String>,
    order_by: Vec<String>,
    max_results: i64,
    pages: Vec<Vec<String>>,
    page_tokens: Vec<Option<String>>,
    ordered_ids: Vec<String>,
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
            "mlflow_rust_registry_corpus_{}_{}.db",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(corpus_dir().join("registry.db"), &path).expect("copy corpus db");
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

fn load_corpus() -> Corpus {
    let text = std::fs::read_to_string(corpus_dir().join("cases.json")).expect("read cases.json");
    serde_json::from_str(&text).expect("parse cases.json")
}

async fn store(temp: &TempDb) -> RegistryStore {
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect corpus db");
    RegistryStore::new(db)
}

#[tokio::test]
async fn registered_models_corpus_matches_python() {
    let corpus = load_corpus();
    let tmp = TempDb::new();
    let s = store(&tmp).await;

    for case in &corpus.registered_models {
        let mut all: Vec<String> = Vec::new();
        let mut token: Option<String> = None;
        let mut page_idx = 0usize;

        for _ in 0..1000 {
            let page = s
                .search_registered_models(
                    WS,
                    case.filter.as_deref(),
                    case.max_results,
                    &case.order_by,
                    token.as_deref(),
                )
                .await
                .unwrap_or_else(|e| panic!("[{}] search failed: {e:?}", case.label));

            let names: Vec<String> = page
                .registered_models
                .iter()
                .map(|rm| rm.name.clone())
                .collect();

            assert_eq!(
                &names, &case.pages[page_idx],
                "[{}] page {page_idx} names mismatch",
                case.label
            );
            assert_eq!(
                page.next_page_token, case.page_tokens[page_idx],
                "[{}] page {page_idx} token mismatch",
                case.label
            );

            all.extend(names);
            token = page.next_page_token;
            page_idx += 1;
            if token.is_none() {
                break;
            }
        }
        assert_eq!(
            all, case.ordered_names,
            "[{}] full ordered sequence mismatch",
            case.label
        );
        assert_eq!(
            page_idx,
            case.pages.len(),
            "[{}] page count mismatch",
            case.label
        );
    }
}

#[tokio::test]
async fn model_versions_corpus_matches_python() {
    let corpus = load_corpus();
    let tmp = TempDb::new();
    let s = store(&tmp).await;

    for case in &corpus.model_versions {
        let mut all: Vec<String> = Vec::new();
        let mut token: Option<String> = None;
        let mut page_idx = 0usize;

        for _ in 0..1000 {
            let page = s
                .search_model_versions(
                    WS,
                    case.filter.as_deref(),
                    case.max_results,
                    &case.order_by,
                    token.as_deref(),
                )
                .await
                .unwrap_or_else(|e| panic!("[{}] search failed: {e:?}", case.label));

            let ids: Vec<String> = page
                .model_versions
                .iter()
                .map(|mv| format!("{}/{}", mv.name, mv.version))
                .collect();

            assert_eq!(
                &ids, &case.pages[page_idx],
                "[{}] page {page_idx} ids mismatch",
                case.label
            );
            assert_eq!(
                page.next_page_token, case.page_tokens[page_idx],
                "[{}] page {page_idx} token mismatch",
                case.label
            );

            all.extend(ids);
            token = page.next_page_token;
            page_idx += 1;
            if token.is_none() {
                break;
            }
        }
        assert_eq!(
            all, case.ordered_ids,
            "[{}] full ordered sequence mismatch",
            case.label
        );
        assert_eq!(
            page_idx,
            case.pages.len(),
            "[{}] page count mismatch",
            case.label
        );
    }
}
