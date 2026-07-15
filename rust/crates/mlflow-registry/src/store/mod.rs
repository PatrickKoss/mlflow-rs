//! The [`RegistryStore`]: registered models, model versions, tags, and
//! aliases, mirroring `mlflow/store/model_registry/sqlalchemy_store.py`.
//!
//! ## Workspace scoping (CRITICAL, plan §3.14 / §3.17)
//!
//! Every registry table has a **workspace-leading composite primary key**, and
//! every method here takes an explicit `workspace: &str`. In single-tenant mode
//! the caller passes `"default"`; when workspaces are enabled the value comes
//! from the `X-MLFLOW-WORKSPACE` header. Every query filters `workspace = ?`, so
//! a lookup in the wrong workspace yields the same `RESOURCE_DOES_NOT_EXIST`
//! ("Registered Model with name=... not found") as a genuinely missing model —
//! matching `WorkspaceAwareSqlAlchemyStore`.
//!
//! ## Rename cascade
//!
//! `rename_registered_model` renames the row in `registered_models`. The three
//! child tables (`model_versions`, `registered_model_tags`,
//! `registered_model_aliases`) carry a `(workspace, name)` foreign key declared
//! `ON UPDATE CASCADE`, and `model_version_tags` cascades transitively via
//! `model_versions`. So the DB propagates the new name to all four child tables
//! automatically **as long as FK enforcement is on** — which it is for SQLite
//! (the store sets `PRAGMA foreign_keys=ON` on every connection), Postgres, and
//! MySQL/InnoDB. Python additionally sets `model_versions.name` explicitly in
//! Python code and bumps each version's `last_updated_time`; we replicate the
//! observable result (renamed rows + bumped `last_updated_time` on every
//! version and on the model itself) by issuing the same updates inside one
//! transaction, then relying on the FK cascade for the tag/alias tables.
//!
//! ## Search (T7.3)
//!
//! The full model-version lifecycle (create with `models:/` source resolution,
//! `update`, `transition_model_version_stage`, soft-delete redaction) lives in
//! [`model_versions`] (T7.2). Registry **search** — `search_registered_models`
//! and `search_model_versions`, with the search DSL, the AND-of-tags
//! HAVING-count subquery, and the prompt-exclusion anti-join — lives in
//! [`search`] (T7.3).

mod aliases;
mod model_versions;
mod registered_models;
mod search;
mod tags;

pub use search::{ModelVersionsPage, RegisteredModelsPage};

use mlflow_error::MlflowError;
use mlflow_store::Db;

/// The model-registry store: registered models, model versions, tags, aliases.
///
/// Holds a [`Db`] pool (already connected and Alembic-verified by the caller).
#[derive(Debug, Clone)]
pub struct RegistryStore {
    db: Db,
}

impl RegistryStore {
    /// Create a store over an already-connected/verified [`Db`].
    pub fn new(db: Db) -> Self {
        Self { db }
    }

    /// The underlying database pool.
    pub fn db(&self) -> &Db {
        &self.db
    }
}

/// Now in epoch milliseconds (`get_current_time_millis`).
pub(crate) fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Map a sqlx error to an `INTERNAL_ERROR` `MlflowError`.
pub(crate) fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("database error: {e}"))
}

/// Detect a unique/primary-key violation across the three backends (used to map
/// a create/rename collision to `RESOURCE_ALREADY_EXISTS`).
pub(crate) fn is_unique_violation(e: &sqlx::Error) -> bool {
    let Some(db) = e.as_database_error() else {
        return false;
    };
    if let Some(code) = db.code() {
        // Postgres 23505; MySQL 1062 (SQLSTATE 23000).
        if code == "23505" || code == "1062" || code == "23000" {
            return true;
        }
    }
    let msg = db.message().to_ascii_lowercase();
    msg.contains("unique constraint") || msg.contains("duplicate") || msg.contains("primary key")
}
