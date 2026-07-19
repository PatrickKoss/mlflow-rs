//! Shared test fixture for `mlflow-store` and `mlflow-registry` integration
//! tests (plan T2.2: run the full SQLite-based store/registry suites against
//! Postgres and MySQL, not just the ad hoc smokes).
//!
//! Every integration test file in those two crates hand-rolls an identical
//! `TempDb` struct that copies the checked-in, Alembic-migrated SQLite
//! fixture (`tests/fixtures/tracking.db`) to a temp file per test. This crate
//! generalizes that helper so the *same* test bodies can run against a live
//! Postgres/MySQL database when one is configured, while still defaulting to
//! the sqlite-copy behavior otherwise. See [`TempDb`] for the selection rule.
//!
//! **Why not per-test schema provisioning.** Rust never migrates schema
//! (`mlflow-store::db` — schema is Python/Alembic-owned); provisioning a
//! literal fresh Postgres/MySQL schema per test would mean shelling out to
//! `mlflow db upgrade` (or an equivalent) hundreds of times per run, which is
//! far too slow and reintroduces a Python dependency into the Rust test
//! binary. Instead: a single already-migrated database is provisioned once,
//! out of band (see `rust/tests/db/compose.yml` + CI), and each [`TempDb::new`]
//! call truncates every tracking/registry table before re-seeding the two
//! fixture rows the SQLite fixture carries (the `Default` experiment, created
//! by migration/init itself, plus `rust_store_fixture` + its `team=rust` tag —
//! the only fixture content any test actually asserts on). Tests in this mode
//! must run single-threaded (`--test-threads=1`) since they share one schema;
//! the sqlite path is unaffected and keeps running fully parallel.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use mlflow_registry::schema::REGISTRY_TABLES;
use mlflow_store::schema::TRACKING_TABLES;
use mlflow_store::{Db, PoolConfig};

#[cfg(unix)]
pub mod reference_server;

/// Env var selecting the live dialect to run the shared suites against.
/// Unset (the default) or any other value falls back to the SQLite fixture.
pub const DIALECT_ENV: &str = "MLFLOW_RUST_TEST_DIALECT";
/// URI env var for the live Postgres database (already Alembic-migrated).
pub const PG_URI_ENV: &str = "MLFLOW_RUST_TEST_PG_URI";
/// URI env var for the live MySQL database (already Alembic-migrated).
pub const MYSQL_URI_ENV: &str = "MLFLOW_RUST_TEST_MYSQL_URI";

/// Name of the fixture experiment the SQLite fixture carries (plan T2.1's
/// `rust/tools/make_test_db.py`), reproduced on the live dialects.
const FIXTURE_EXPERIMENT: &str = "rust_store_fixture";

fn fixture_db_path() -> PathBuf {
    // The fixture lives in mlflow-store; both crates' tests reused it as-is
    // (registry tables are physically in the same database).
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("mlflow-store")
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

/// Which backend [`TempDb`] will hand out connections against, resolved once
/// from the environment. `Sqlite` is the default and requires no env setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveTarget {
    Sqlite,
    Postgres,
    MySql,
}

fn resolve_target() -> LiveTarget {
    match std::env::var(DIALECT_ENV).ok().as_deref() {
        Some("postgres") => LiveTarget::Postgres,
        Some("mysql") => LiveTarget::MySql,
        _ => LiveTarget::Sqlite,
    }
}

/// A per-test database handle. `tag` only affects the temp-file name on
/// SQLite (kept for readability in `/tmp` during local debugging); it is
/// unused on the live dialects.
///
/// `TempDb::new` is `async` (`TempDb::new(tag).await`) so the live-dialect
/// reset can run as a normal awaited call rather than blocking the
/// `#[tokio::test]` runtime it's invoked from. SQLite construction does no
/// I/O beyond the file copy, exactly as before.
pub struct TempDb {
    target: LiveTarget,
    /// Present only for `LiveTarget::Sqlite`; the temp copy of the fixture.
    sqlite_path: Option<PathBuf>,
    /// Present only for the live dialects; the already-migrated database URI.
    live_uri: Option<String>,
}

impl TempDb {
    /// Async because the live-dialect path resets the shared schema over the
    /// network before handing back a usable handle; the SQLite path stays a
    /// cheap synchronous file copy but is `async fn` too so every call site
    /// (both crates' `tests/*.rs`) can use one signature regardless of
    /// dialect. Call as `TempDb::new(tag).await`.
    pub async fn new(tag: &str) -> Self {
        match resolve_target() {
            LiveTarget::Sqlite => Self::new_sqlite(tag),
            LiveTarget::Postgres => Self::new_live(&must_env(PG_URI_ENV)).await,
            LiveTarget::MySql => Self::new_live(&must_env(MYSQL_URI_ENV)).await,
        }
    }

    fn new_sqlite(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_test_support_{tag}_{}_{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixture_db_path(), &path).expect("copy sqlite fixture");
        TempDb {
            target: LiveTarget::Sqlite,
            sqlite_path: Some(path),
            live_uri: None,
        }
    }

    /// Reset the live database to a clean slate and re-seed the fixture
    /// content. Tests using the live dialects must run with
    /// `--test-threads=1` (see module docs): resets truncate the *entire*
    /// shared schema, so concurrent tests would race.
    async fn new_live(uri: &str) -> Self {
        let target = if uri.starts_with("postgresql") || uri.starts_with("postgres") {
            LiveTarget::Postgres
        } else {
            LiveTarget::MySql
        };
        reset_live_db(uri).await.expect("reset live database");
        TempDb {
            target,
            sqlite_path: None,
            live_uri: Some(uri.to_string()),
        }
    }

    pub fn uri(&self) -> String {
        match self.target {
            LiveTarget::Sqlite => {
                let path = self.sqlite_path.as_ref().expect("sqlite path");
                format!("sqlite:///{}", path.display())
            }
            LiveTarget::Postgres | LiveTarget::MySql => self.live_uri.clone().expect("live uri"),
        }
    }

    /// Open a [`Db`] pool against this handle's database. Every existing
    /// `store(&TempDb)` helper across the two crates does exactly this.
    pub async fn connect(&self) -> Db {
        Db::connect(&self.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture")
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        if let Some(path) = &self.sqlite_path {
            let _ = std::fs::remove_file(path);
        }
        // Live databases are left in place (already-truncated by the next
        // `TempDb::new` call); nothing owns dropping the shared schema itself.
    }
}

fn must_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "{name} must be set when {DIALECT_ENV} selects a live dialect \
             (this test binary should not be invoked with that combination missing)"
        )
    })
}

/// Truncate every tracking + registry table (children before parents, so this
/// is safe regardless of `ON DELETE CASCADE` coverage) and re-seed the fixture
/// experiment. Runs once per [`TempDb::new`] call.
async fn reset_live_db(uri: &str) -> Result<(), mlflow_store::StoreError> {
    let db = Db::connect_and_verify(uri).await?;
    let dialect = db.dialect();

    // Children-first: both TRACKING_TABLES and REGISTRY_TABLES are declared
    // parent-first, so iterate in reverse.
    for table in REGISTRY_TABLES.iter().rev() {
        delete_all(&db, dialect, table).await?;
    }
    for table in TRACKING_TABLES.iter().rev() {
        // The `Default` experiment (id 0) is (re)created by
        // `_initialize_tables`/migration, not by test seeding; deleting all
        // rows from `experiments` here is fine because we recreate it below
        // via the real store API, matching what the SQLite fixture generator
        // does (`rust/tools/make_test_db.py`).
        delete_all(&db, dialect, table).await?;
    }

    seed_fixture_rows(db).await
}

async fn delete_all(
    db: &Db,
    dialect: mlflow_store::Dialect,
    table: &str,
) -> Result<(), sqlx::Error> {
    let sql = format!("DELETE FROM {}", dialect.quote_ident(table));
    match db {
        Db::Sqlite(p) => sqlx::query(&sql).execute(p).await.map(|_| ()),
        Db::Postgres(p) => sqlx::query(&sql).execute(p).await.map(|_| ()),
        Db::MySql(p) => sqlx::query(&sql).execute(p).await.map(|_| ()),
    }
}

/// Re-create the two fixture rows the SQLite fixture carries and that a
/// handful of tests assert on: the built-in `Default` experiment (dropped by
/// the truncation above) and `rust_store_fixture` tagged `team=rust`
/// (`rust/tools/make_test_db.py`). Uses the real store API rather than raw
/// SQL so ids/timestamps are assigned exactly as they would be in production,
/// independent of dialect.
async fn seed_fixture_rows(db: Db) -> Result<(), mlflow_store::StoreError> {
    use mlflow_store::TrackingStore;

    // `create_experiment` cannot fail here (fresh, truncated tables; valid
    // names); `.expect()` turns anything unexpected into a hard test-setup
    // failure rather than threading a mismatched error type through.
    let store = TrackingStore::new(db, "s3://bucket/mlruns");
    store
        .create_experiment("default", "Default", None, &[])
        .await
        .expect("seed Default experiment");
    store
        .create_experiment("default", FIXTURE_EXPERIMENT, None, &[("team", "rust")])
        .await
        .expect("seed rust_store_fixture experiment");
    Ok(())
}
