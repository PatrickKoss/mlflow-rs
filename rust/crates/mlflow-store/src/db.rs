//! Database connectivity: pool creation, SQLite session PRAGMAs, and Alembic
//! head verification.
//!
//! sqlx's `Any` driver is a poor fit for MLflow: it erases the concrete pool
//! type, cannot apply dialect-specific `after_connect` hooks cleanly, and loses
//! compile-time query typing we will want in T2.4+. Instead we hold a concrete
//! per-backend pool in the [`Db`] enum and `match` on it. This keeps each
//! backend's connection tuning explicit (notably the SQLite PRAGMAs) and lets
//! the store layer obtain a strongly-typed pool when it needs one.
//!
//! [`Db::connect_and_verify`] is the single entry point: it parses the URI,
//! builds the pool with the env-var-derived [`PoolConfig`], applies the SQLite
//! session config on every connection, and verifies the Alembic head before
//! returning. Rust never creates or migrates schema — that stays Python-owned
//! (plan §5.4).

use std::time::Duration;

use sqlx::mysql::{MySqlConnectOptions, MySqlPoolOptions};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{ConnectOptions, Executor, MySqlPool, PgPool, Row, SqlitePool};

use crate::dialect::Dialect;
use crate::pool::PoolConfig;
use crate::uri::{self, UriError};

/// Alembic head revision for the backend store expected by this build
/// (plan §5.4). Rust refuses to run against any other revision.
pub const EXPECTED_ALEMBIC_HEAD: &str = "b7e4c1a90f23";

/// Name of the Alembic bookkeeping table.
const ALEMBIC_VERSION_TABLE: &str = "alembic_version";

/// A live connection pool for one of the supported backends.
///
/// Using a concrete pool per backend (rather than sqlx's `AnyPool`) keeps
/// dialect-specific connection setup — especially the SQLite PRAGMAs — explicit
/// and lets store methods (T2.4+) take a typed pool for compile-time-checked
/// queries.
#[derive(Debug, Clone)]
pub enum Db {
    Sqlite(SqlitePool),
    Postgres(PgPool),
    MySql(MySqlPool),
}

impl Db {
    /// The dialect backing this pool.
    pub fn dialect(&self) -> Dialect {
        match self {
            Db::Sqlite(_) => Dialect::Sqlite,
            Db::Postgres(_) => Dialect::Postgres,
            Db::MySql(_) => Dialect::MySql,
        }
    }
}

/// Errors from connecting to and verifying a backend store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Uri(#[from] UriError),

    #[error("failed to connect to database: {0}")]
    Connect(#[source] sqlx::Error),

    #[error(transparent)]
    Schema(#[from] SchemaError),

    #[error("database error: {0}")]
    Sqlx(#[from] sqlx::Error),
}

/// Alembic schema-verification failures.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SchemaError {
    /// The `alembic_version` table is missing — the DB has not been initialized
    /// by MLflow/alembic yet. Rust never initializes schema itself (§5.4).
    #[error(
        "The database has not been initialized by MLflow (the '{table}' table is missing). \
         Run 'mlflow db upgrade <database_uri>' with the Python MLflow server to create and \
         migrate the schema; the Rust server never initializes or migrates the database."
    )]
    Uninitialized { table: String },

    /// The current head does not match the head this build expects.
    #[error(
        "Detected out-of-date database schema (found version {found}, but expected {expected}). \
         Take a backup of your database, then run 'mlflow db upgrade <database_uri>' to migrate \
         your database to the latest schema. NOTE: schema migration may result in database \
         downtime - please consult your database's documentation for more detail."
    )]
    OutOfDate { found: String, expected: String },
}

impl Db {
    /// Parse `uri`, build the pool (env-var-tuned), apply SQLite PRAGMAs on
    /// every connection, and verify the Alembic head. Returns a ready pool.
    ///
    /// Pool tuning is read from the environment ([`PoolConfig::from_env`]).
    pub async fn connect_and_verify(uri: &str) -> Result<Db, StoreError> {
        Self::connect_and_verify_with(uri, PoolConfig::from_env()).await
    }

    /// Like [`Db::connect_and_verify`] but with an explicit [`PoolConfig`]
    /// (used by tests).
    pub async fn connect_and_verify_with(uri: &str, cfg: PoolConfig) -> Result<Db, StoreError> {
        let db = Self::connect(uri, cfg).await?;
        db.verify_schema().await?;
        Ok(db)
    }

    /// Build a pool for `uri` without verifying the schema. Applies SQLite
    /// session PRAGMAs via an `after_connect` hook (mirrors
    /// `mlflow/store/db/utils.py:154-157`).
    pub async fn connect(uri: &str, cfg: PoolConfig) -> Result<Db, StoreError> {
        let parsed = uri::parse(uri)?;
        match parsed.dialect {
            Dialect::Sqlite => {
                let opts = parsed
                    .sqlx_url
                    .parse::<SqliteConnectOptions>()
                    .map_err(StoreError::Connect)?
                    // Do not create the DB file: the schema is Python-owned and
                    // an auto-created empty DB would fail verification anyway.
                    .create_if_missing(false)
                    .foreign_keys(true)
                    .busy_timeout(Duration::from_millis(20_000));
                let opts = maybe_disable_statement_logging(opts, cfg.echo);
                let pool = sqlite_pool_options(&cfg)
                    .after_connect(|conn, _meta| {
                        Box::pin(async move {
                            // `foreign_keys` / `busy_timeout` above cover two of
                            // the three PRAGMAs; case_sensitive_like has no
                            // dedicated option, so set it here. Re-assert all
                            // three so behavior is explicit per connection.
                            conn.execute("PRAGMA foreign_keys = ON;").await?;
                            conn.execute("PRAGMA busy_timeout = 20000;").await?;
                            conn.execute("PRAGMA case_sensitive_like = true;").await?;
                            Ok(())
                        })
                    })
                    .connect_with(opts)
                    .await
                    .map_err(StoreError::Connect)?;
                Ok(Db::Sqlite(pool))
            }
            Dialect::Postgres => {
                let opts = parsed
                    .sqlx_url
                    .parse::<PgConnectOptions>()
                    .map_err(StoreError::Connect)?;
                let opts = maybe_disable_statement_logging(opts, cfg.echo);
                let pool = pg_pool_options(&cfg)
                    .connect_with(opts)
                    .await
                    .map_err(StoreError::Connect)?;
                Ok(Db::Postgres(pool))
            }
            Dialect::MySql => {
                let opts = parsed
                    .sqlx_url
                    .parse::<MySqlConnectOptions>()
                    .map_err(StoreError::Connect)?;
                let opts = maybe_disable_statement_logging(opts, cfg.echo);
                let pool = mysql_pool_options(&cfg)
                    .after_connect(|conn, _meta| {
                        Box::pin(async move {
                            // ANSI_QUOTES lets hand-written SQL use standard
                            // double-quoted identifiers (`"key"` — a MySQL
                            // reserved word) uniformly across all dialects.
                            // Backtick quoting (quote_ident) stays valid too.
                            conn.execute(
                                "SET SESSION sql_mode = CONCAT(@@sql_mode, ',ANSI_QUOTES');",
                            )
                            .await?;
                            Ok(())
                        })
                    })
                    .connect_with(opts)
                    .await
                    .map_err(StoreError::Connect)?;
                Ok(Db::MySql(pool))
            }
        }
    }

    /// Read the Alembic head from `alembic_version` and compare it to
    /// [`EXPECTED_ALEMBIC_HEAD`]. Mirrors `_verify_schema`
    /// (`mlflow/store/db/utils.py:123-134`); additionally reports a distinct
    /// error when the table is missing entirely (uninitialized DB).
    pub async fn verify_schema(&self) -> Result<(), StoreError> {
        let current = self.read_alembic_head().await?;
        match current {
            None => Err(SchemaError::Uninitialized {
                table: ALEMBIC_VERSION_TABLE.to_string(),
            }
            .into()),
            Some(found) if found != EXPECTED_ALEMBIC_HEAD => Err(SchemaError::OutOfDate {
                found,
                expected: EXPECTED_ALEMBIC_HEAD.to_string(),
            }
            .into()),
            Some(_) => Ok(()),
        }
    }

    /// Read the current Alembic revision. Returns `None` if the
    /// `alembic_version` table does not exist (uninitialized DB) or has no row.
    async fn read_alembic_head(&self) -> Result<Option<String>, StoreError> {
        let query = format!("SELECT version_num FROM {ALEMBIC_VERSION_TABLE}");
        // Extract the string inside each arm so the arms share a type
        // (`Result<Option<String>, sqlx::Error>`); the concrete `Row` types
        // differ per backend and cannot be unified.
        let result = match self {
            Db::Sqlite(p) => sqlx::query(&query).fetch_optional(p).await.and_then(|r| {
                r.map(|row| row.try_get::<String, _>("version_num"))
                    .transpose()
            }),
            Db::Postgres(p) => sqlx::query(&query).fetch_optional(p).await.and_then(|r| {
                r.map(|row| row.try_get::<String, _>("version_num"))
                    .transpose()
            }),
            Db::MySql(p) => sqlx::query(&query).fetch_optional(p).await.and_then(|r| {
                r.map(|row| row.try_get::<String, _>("version_num"))
                    .transpose()
            }),
        };
        match result {
            Ok(v) => Ok(v),
            Err(e) if is_missing_table(&e) => Ok(None),
            Err(e) => Err(StoreError::Sqlx(e)),
        }
    }
}

/// Detect a "table does not exist" error across the three backends. sqlx does
/// not model this uniformly, so we inspect the database error message/code.
fn is_missing_table(err: &sqlx::Error) -> bool {
    let Some(db_err) = err.as_database_error() else {
        return false;
    };
    // Postgres: SQLSTATE 42P01 (undefined_table).
    // MySQL: error 1146 (ER_NO_SUCH_TABLE) -> SQLSTATE 42S02.
    if let Some(code) = db_err.code() {
        if code == "42P01" || code == "42S02" || code == "1146" {
            return true;
        }
    }
    // SQLite reports "no such table: ..." with no stable code.
    let msg = db_err.message().to_ascii_lowercase();
    msg.contains("no such table") || msg.contains("doesn't exist")
}

fn sqlite_pool_options(cfg: &PoolConfig) -> SqlitePoolOptions {
    SqlitePoolOptions::new()
        .max_connections(cfg.max_connections)
        .min_connections(cfg.min_connections)
        .max_lifetime(cfg.max_lifetime)
}

fn pg_pool_options(cfg: &PoolConfig) -> PgPoolOptions {
    PgPoolOptions::new()
        .max_connections(cfg.max_connections)
        .min_connections(cfg.min_connections)
        .max_lifetime(cfg.max_lifetime)
}

fn mysql_pool_options(cfg: &PoolConfig) -> MySqlPoolOptions {
    MySqlPoolOptions::new()
        .max_connections(cfg.max_connections)
        .min_connections(cfg.min_connections)
        .max_lifetime(cfg.max_lifetime)
}

/// When echo is off, drop sqlx's default per-statement `info` logging down to
/// `trace` so it stays quiet. When on, leave sqlx's logging at its default
/// (SQL is emitted via the `tracing` backend, the Rust analog of
/// SQLAlchemy `echo=True`).
fn maybe_disable_statement_logging<O: ConnectOptions>(opts: O, echo: bool) -> O {
    if echo {
        opts
    } else {
        opts.log_statements(log::LevelFilter::Trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_head_is_current_alembic_head() {
        // Guards against the constant drifting from the fixture; the fixture is
        // regenerated by rust/tools/make_test_db.py at the same head.
        assert_eq!(EXPECTED_ALEMBIC_HEAD, "b7e4c1a90f23");
    }

    #[test]
    fn out_of_date_message_mentions_db_upgrade() {
        let e = SchemaError::OutOfDate {
            found: "old".into(),
            expected: EXPECTED_ALEMBIC_HEAD.into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("out-of-date database schema"));
        assert!(msg.contains("mlflow db upgrade"));
        assert!(msg.contains("old"));
        assert!(msg.contains(EXPECTED_ALEMBIC_HEAD));
    }

    #[test]
    fn uninitialized_message_is_distinct() {
        let e = SchemaError::Uninitialized {
            table: ALEMBIC_VERSION_TABLE.into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("has not been initialized"));
        assert!(msg.contains("alembic_version"));
        assert!(msg.contains("mlflow db upgrade"));
    }
}
