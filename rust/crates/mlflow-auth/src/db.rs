//! Auth-database connectivity: Alembic head verification against the
//! `alembic_version_auth` table and read-replica routing.
//!
//! The auth DB is a **separate** database from the tracking store (default
//! `sqlite:///basic_auth.db`, plan §5.3). It has its own Alembic lineage
//! (`mlflow/server/auth/db/migrations/`, head `f1a2b3c4d5e6`) recorded in a
//! version table named **`alembic_version_auth`** (not `alembic_version`) —
//! `mlflow/server/auth/db/utils.py:76` configures the migration context with
//! `version_table="alembic_version_auth"`. So we cannot reuse
//! `mlflow_store::Db::connect_and_verify` (which checks the tracking table and
//! head); [`AuthDb::connect_and_verify`] reads the auth version table instead.
//!
//! As with the tracking store, **Rust never creates or migrates the auth
//! schema** (plan §5.4). It reads the version table at startup and refuses to
//! start on a mismatch with a "run `mlflow db upgrade`" message. Python's
//! `migrate_if_needed` upgrades to head on init; the Rust server relies on that
//! having already happened.
//!
//! ## Read-replica routing (plan §5.3)
//!
//! Python's `SqlAlchemyStore.init_db(db_uri, read_db_uri=None)`
//! (`sqlalchemy_store.py:111-134`) builds a second engine when a distinct
//! `read_database_uri` is configured and routes read-only sessions to it. We
//! mirror that seam: [`AuthDb`] holds a `write` pool and an optional `read`
//! pool; read queries use [`AuthDb::reader`] (the replica if present, else the
//! writer) and writes use [`AuthDb::writer`]. When `read_db_uri == db_uri` we
//! log the same warning and disable the replica.

use mlflow_store::{Db, PoolConfig, StoreError};
use sqlx::Row;

/// Alembic head revision for the auth DB expected by this build (plan §5.3/§5.4).
/// Rust refuses to run against any other revision.
pub const EXPECTED_AUTH_ALEMBIC_HEAD: &str = "f1a2b3c4d5e6";

/// Name of the auth Alembic bookkeeping table
/// (`mlflow/server/auth/db/utils.py:76`).
pub const ALEMBIC_VERSION_AUTH_TABLE: &str = "alembic_version_auth";

/// Auth schema-verification failures. Mirrors `mlflow_store::SchemaError` but
/// names the auth version table and head so the operator sees which database is
/// stale.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AuthSchemaError {
    /// The `alembic_version_auth` table is missing — the auth DB has not been
    /// initialized by MLflow/alembic yet. Rust never initializes it (§5.4).
    #[error(
        "The auth database has not been initialized by MLflow (the '{table}' table is missing). \
         Run 'mlflow db upgrade <database_uri>' with the Python MLflow server to create and \
         migrate the auth schema; the Rust server never initializes or migrates the database."
    )]
    Uninitialized { table: String },

    /// The current auth head does not match the head this build expects.
    #[error(
        "Detected out-of-date auth database schema (found version {found}, but expected \
         {expected}). Take a backup of your database, then run 'mlflow db upgrade \
         <database_uri>' to migrate your database to the latest schema. NOTE: schema migration \
         may result in database downtime - please consult your database's documentation for \
         more detail."
    )]
    OutOfDate { found: String, expected: String },
}

/// Errors from connecting to and verifying the auth database.
#[derive(Debug, thiserror::Error)]
pub enum AuthDbError {
    /// A connection/URI/driver error from the underlying store layer.
    #[error(transparent)]
    Store(#[from] StoreError),

    /// The auth schema is missing or stale.
    #[error(transparent)]
    Schema(#[from] AuthSchemaError),

    #[error("auth database error: {0}")]
    Sqlx(#[from] sqlx::Error),
}

/// A connected auth database: a write pool plus an optional read replica pool.
///
/// Reads route to the replica when configured (mirroring Python's routing
/// session maker); writes always go to the primary.
#[derive(Debug, Clone)]
pub struct AuthDb {
    write: Db,
    read: Option<Db>,
}

impl AuthDb {
    /// Connect to the auth database (write URI + optional read-replica URI),
    /// verify the Alembic head on the **write** connection, and return a ready
    /// [`AuthDb`]. Mirrors `SqlAlchemyStore.init_db` minus the migration step
    /// (Rust never migrates).
    ///
    /// When `read_db_uri` equals `db_uri`, the replica is disabled and a warning
    /// is logged (matching `sqlalchemy_store.py:126-131`).
    pub async fn connect_and_verify(
        db_uri: &str,
        read_db_uri: Option<&str>,
    ) -> Result<AuthDb, AuthDbError> {
        Self::connect_and_verify_with(db_uri, read_db_uri, PoolConfig::from_env()).await
    }

    /// Like [`AuthDb::connect_and_verify`] with an explicit [`PoolConfig`]
    /// (used by tests).
    pub async fn connect_and_verify_with(
        db_uri: &str,
        read_db_uri: Option<&str>,
        cfg: PoolConfig,
    ) -> Result<AuthDb, AuthDbError> {
        // `Db::connect` builds the pool without running the tracking-store head
        // check (that check targets the wrong version table for the auth DB).
        let write = Db::connect(db_uri, cfg.clone()).await?;

        let read = match read_db_uri {
            Some(r) if r == db_uri => {
                tracing::warn!(
                    "read_db_uri is the same as the primary db_uri; read replica routing will \
                     not be enabled. This is likely a configuration mistake."
                );
                None
            }
            Some(r) => Some(Db::connect(r, cfg).await?),
            None => None,
        };

        let db = AuthDb { write, read };
        db.verify_schema().await?;
        Ok(db)
    }

    /// Wrap already-connected pools (used by tests that share a fixture pool).
    /// Does not verify the schema.
    pub fn from_pools(write: Db, read: Option<Db>) -> Self {
        AuthDb { write, read }
    }

    /// The write (primary) pool — used for all mutations.
    pub fn writer(&self) -> &Db {
        &self.write
    }

    /// The read pool — the replica when configured, else the primary. Read-only
    /// queries route here, mirroring Python's `read_only=True` sessions.
    pub fn reader(&self) -> &Db {
        self.read.as_ref().unwrap_or(&self.write)
    }

    /// Whether a distinct read replica is active.
    pub fn has_replica(&self) -> bool {
        self.read.is_some()
    }

    /// Verify the Alembic head recorded in `alembic_version_auth` against
    /// [`EXPECTED_AUTH_ALEMBIC_HEAD`], reading from the **write** pool.
    pub async fn verify_schema(&self) -> Result<(), AuthDbError> {
        match self.read_auth_alembic_head().await? {
            None => Err(AuthSchemaError::Uninitialized {
                table: ALEMBIC_VERSION_AUTH_TABLE.to_string(),
            }
            .into()),
            Some(found) if found != EXPECTED_AUTH_ALEMBIC_HEAD => Err(AuthSchemaError::OutOfDate {
                found,
                expected: EXPECTED_AUTH_ALEMBIC_HEAD.to_string(),
            }
            .into()),
            Some(_) => Ok(()),
        }
    }

    /// Read the current auth Alembic revision from `alembic_version_auth`.
    /// Returns `None` when the table does not exist (uninitialized) or is empty.
    async fn read_auth_alembic_head(&self) -> Result<Option<String>, AuthDbError> {
        let query = format!("SELECT version_num FROM {ALEMBIC_VERSION_AUTH_TABLE}");
        let result = match &self.write {
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
            Err(e) => Err(AuthDbError::Sqlx(e)),
        }
    }
}

/// Detect a "table does not exist" error across the three backends (same logic
/// as `mlflow_store::db::is_missing_table`, which is crate-private there).
fn is_missing_table(err: &sqlx::Error) -> bool {
    let Some(db_err) = err.as_database_error() else {
        return false;
    };
    if let Some(code) = db_err.code() {
        if code == "42P01" || code == "42S02" || code == "1146" {
            return true;
        }
    }
    let msg = db_err.message().to_ascii_lowercase();
    msg.contains("no such table") || msg.contains("doesn't exist")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_head_is_pinned() {
        // Guards against the constant drifting from the fixture; the fixture is
        // regenerated by rust/tools/make_auth_test_db.py at the same head.
        assert_eq!(EXPECTED_AUTH_ALEMBIC_HEAD, "f1a2b3c4d5e6");
        assert_eq!(ALEMBIC_VERSION_AUTH_TABLE, "alembic_version_auth");
    }

    #[test]
    fn out_of_date_message_mentions_db_upgrade() {
        let e = AuthSchemaError::OutOfDate {
            found: "old".into(),
            expected: EXPECTED_AUTH_ALEMBIC_HEAD.into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("out-of-date auth database schema"));
        assert!(msg.contains("mlflow db upgrade"));
        assert!(msg.contains("old"));
        assert!(msg.contains(EXPECTED_AUTH_ALEMBIC_HEAD));
    }

    #[test]
    fn uninitialized_message_names_auth_table() {
        let e = AuthSchemaError::Uninitialized {
            table: ALEMBIC_VERSION_AUTH_TABLE.into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("auth database has not been initialized"));
        assert!(msg.contains("alembic_version_auth"));
        assert!(msg.contains("mlflow db upgrade"));
    }
}
