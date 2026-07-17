//! Workspace store + table (plan T10.1), mirroring
//! `mlflow/store/workspace/sqlalchemy_store.py` and
//! `mlflow/store/workspace/abstract_store.py` byte-for-byte where observable.
//!
//! This is workspace-CRUD + config resolution only. Request-scoping middleware
//! and REST endpoints are T10.2/T10.3 and live in `mlflow-server`.
//!
//! ## What it owns
//!
//! * The `workspaces` table (columns: `name` PK `VARCHAR(63)`, `description`
//!   `TEXT`, `default_artifact_root` `TEXT`, `trace_archival_location` `TEXT`,
//!   `trace_archival_retention` `VARCHAR(32)` — see migrations
//!   `1b5f0d9ad7c1_add_workspace_columns_and_catalog.py` and
//!   `da6fb0208061_add_workspaces_trace_archival_location.py`).
//! * [`WorkspaceNameValidator`] — Kubernetes-style name rules
//!   (`WorkspaceNameValidator` in `abstract_store.py`).
//! * CRUD (`create` / `get` / `list` / `update` / `delete`), the three delete
//!   modes (`RESTRICT` / `CASCADE` / `SET_DEFAULT`), and artifact-root /
//!   trace-archival resolution with per-process TTL caches
//!   (`_artifact_root_cache` / `_trace_archival_config_cache`, `cachetools.TTLCache`
//!   with `MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_CAPACITY`=128 /
//!   `MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_TTL_SECONDS`=60).
//!
//! ## Cascade / reassignment (`delete_workspace`)
//!
//! Python iterates `_WORKSPACE_ROOT_MODELS` and calls `session.delete(obj)` /
//! `.update({workspace: "default"})`. `session.delete` fires SQLAlchemy ORM
//! relationship cascades (`cascade="all"`), which the raw-SQL Rust port
//! reproduces with explicit, FK-safe ordered `DELETE`s per root table (see
//! [`cascade`]). The SQLite pool runs with `PRAGMA foreign_keys = ON`, so a few
//! children are also covered by DB-level `ondelete=CASCADE`; the explicit
//! deletes are a superset and remain correct on all backends.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mlflow_error::{ErrorCode, MlflowError};

use super::dbutil::{Tx, Val};
use crate::db::Db;

/// The default workspace name (`DEFAULT_WORKSPACE_NAME`,
/// `mlflow/utils/workspace_utils.py`). Undeletable.
pub const DEFAULT_WORKSPACE_NAME: &str = "default";

/// The `workspaces` table name.
pub const WORKSPACES: &str = "workspaces";

/// TTL-cache capacity default (`MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_CAPACITY`).
const CACHE_CAPACITY_DEFAULT: usize = 128;
/// TTL-cache TTL default in seconds (`MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_TTL_SECONDS`).
const CACHE_TTL_SECONDS_DEFAULT: u64 = 60;

fn cache_capacity() -> usize {
    std::env::var("MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_CAPACITY")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(CACHE_CAPACITY_DEFAULT)
}

fn cache_ttl() -> Duration {
    let secs = std::env::var("MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_TTL_SECONDS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(CACHE_TTL_SECONDS_DEFAULT);
    Duration::from_secs(secs)
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

/// Minimal metadata describing a workspace (`mlflow.entities.workspace.Workspace`).
///
/// `None` leaves a field unset. For update-style calls an empty string clears an
/// existing value, while `None` keeps the current value unchanged (mirrors the
/// Python `Workspace` dataclass semantics used by `update_workspace`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Workspace {
    pub name: String,
    pub description: Option<String>,
    pub default_artifact_root: Option<String>,
    pub trace_archival_location: Option<String>,
    pub trace_archival_retention: Option<String>,
}

impl Workspace {
    /// Construct a workspace with only a name set (all other fields unset).
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
}

/// Trace archival configuration (`mlflow.entities.workspace.TraceArchivalConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TraceArchivalConfig {
    pub location: Option<String>,
    pub retention: Option<String>,
}

/// Resolved trace archival settings for a workspace
/// (`ResolvedTraceArchivalConfig` in `abstract_store.py`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTraceArchivalConfig {
    pub config: TraceArchivalConfig,
    pub append_workspace_prefix: bool,
}

/// Controls what happens to resources when a workspace is deleted
/// (`WorkspaceDeletionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkspaceDeletionMode {
    /// Reassign all resources in the workspace to the default workspace.
    SetDefault,
    /// Delete all resources in the workspace.
    Cascade,
    /// Refuse to delete the workspace if it still contains resources.
    #[default]
    Restrict,
}

impl WorkspaceDeletionMode {
    /// The wire/log string (`WorkspaceDeletionMode.value`).
    pub fn value(self) -> &'static str {
        match self {
            WorkspaceDeletionMode::SetDefault => "SET_DEFAULT",
            WorkspaceDeletionMode::Cascade => "CASCADE",
            WorkspaceDeletionMode::Restrict => "RESTRICT",
        }
    }
}

// ---------------------------------------------------------------------------
// Name validator (WorkspaceNameValidator)
// ---------------------------------------------------------------------------

/// Validator for workspace names based on Kubernetes naming conventions
/// (`WorkspaceNameValidator`).
pub struct WorkspaceNameValidator;

impl WorkspaceNameValidator {
    /// `WorkspaceNameValidator._PATTERN`.
    pub const PATTERN: &'static str = r"^(?!.*--)[a-z0-9]([-a-z0-9]*[a-z0-9])?$";
    const MIN_LENGTH: usize = 2;
    const MAX_LENGTH: usize = 63;
    /// `WorkspaceNameValidator._RESERVED`.
    pub const RESERVED: &'static [&'static str] =
        &["workspaces", "api", "ajax-api", "static-files"];

    /// Mirror `WorkspaceNameValidator.validate`. The Rust API always receives a
    /// `&str`, so the Python `"must be a string"` branch is unreachable here.
    pub fn validate(name: &str) -> Result<(), MlflowError> {
        let len = name.chars().count();
        if !(Self::MIN_LENGTH..=Self::MAX_LENGTH).contains(&len) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Workspace name '{name}' must be between {} and {} characters.",
                Self::MIN_LENGTH,
                Self::MAX_LENGTH
            )));
        }

        if !Self::matches_pattern(name) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Workspace name '{name}' must match the pattern {} \
                 (lowercase alphanumeric with optional internal hyphens).",
                Self::PATTERN
            )));
        }

        if Self::RESERVED.contains(&name) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Workspace name '{name}' is reserved and cannot be used."
            )));
        }

        Ok(())
    }

    /// `re.match(r"^(?!.*--)[a-z0-9]([-a-z0-9]*[a-z0-9])?$", name)` without a
    /// regex engine: no `--` anywhere, all chars in `[a-z0-9-]`, first and last
    /// chars alphanumeric. `re.match` anchors at the start; the pattern's `$`
    /// anchors the end, so this is a full match (min length 2 is enforced
    /// separately, matching Python's ordering).
    fn matches_pattern(name: &str) -> bool {
        if name.contains("--") {
            return false;
        }
        let bytes = name.as_bytes();
        let is_alnum = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
        let is_body = |b: u8| is_alnum(b) || b == b'-';
        match bytes {
            [] => false,
            [only] => is_alnum(*only),
            [first, mid @ .., last] => {
                is_alnum(*first) && is_alnum(*last) && mid.iter().all(|b| is_body(*b))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trace archival validation (subset of mlflow/utils/validation.py)
// ---------------------------------------------------------------------------

const MAX_TRACE_ARCHIVAL_RETENTION_LENGTH: usize = 32;

/// `_validate_trace_archival_retention_string(value)` — no `parameter_name`
/// (the path `sqlalchemy_store.py` takes for retention). Regex `^[1-9][0-9]*[mhd]$`.
fn validate_trace_archival_retention(value: &str) -> Result<String, MlflowError> {
    let trimmed = value.trim();
    let msg = "Trace archival retention must be in the form `<int><unit>`, \
               where unit is one of 'm', 'h', or 'd'.";
    if trimmed.chars().count() > MAX_TRACE_ARCHIVAL_RETENTION_LENGTH {
        return Err(MlflowError::invalid_parameter_value(
            "Trace archival duration must be at most 32 characters.",
        ));
    }
    if !retention_regex_matches(trimmed) {
        return Err(MlflowError::invalid_parameter_value(msg));
    }
    Ok(trimmed.to_string())
}

/// `re.compile(r"^[1-9][0-9]*[mhd]$").fullmatch(value)`.
fn retention_regex_matches(value: &str) -> bool {
    let bytes = value.as_bytes();
    match bytes {
        [first, rest @ .., unit] => {
            (b'1'..=b'9').contains(first)
                && rest.iter().all(u8::is_ascii_digit)
                && matches!(unit, b'm' | b'h' | b'd')
        }
        _ => false,
    }
}

/// `_validate_trace_archival_location(value, parameter_name="trace_archival_location")`.
///
/// The Databricks/DBFS "unsupported repository" branch of
/// `_validate_trace_archival_repository_support` depends on the Python artifact
/// repository registry (not yet ported to Rust), so only the URI-shape and
/// proxy-scheme checks are enforced here. See the deferral note in the report.
fn validate_trace_archival_location(value: &str) -> Result<String, MlflowError> {
    let trimmed = value.trim();
    let scheme = trimmed.split_once("://").map(|(s, _)| s).or_else(|| {
        // A URI like `dbfs:/path` has a scheme but no `//` authority; take the
        // part before the first `:` when it looks like a scheme.
        trimmed.split_once(':').map(|(s, _)| s).filter(|s| {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        })
    });
    match scheme {
        None => Err(MlflowError::invalid_parameter_value(
            "Invalid value for 'trace_archival_location'. Expected a URI string.",
        )),
        Some("mlflow-artifacts") => Err(MlflowError::invalid_parameter_value(
            "Invalid value for 'trace_archival_location'. Trace archival location cannot use \
             the proxy-only `mlflow-artifacts:` scheme.",
        )),
        Some(_) => Ok(trimmed.to_string()),
    }
}

/// `SqlAlchemyStore._validate_workspace_trace_archival_config`.
fn validate_workspace_trace_archival_config(
    location: Option<&str>,
    retention: Option<&str>,
) -> Result<(Option<String>, Option<String>), MlflowError> {
    let validated_location = match location {
        Some(loc) if !loc.is_empty() => Some(validate_trace_archival_location(loc)?),
        _ => None,
    };
    let validated_retention = match retention {
        Some(r) if !r.is_empty() => Some(validate_trace_archival_retention(r)?),
        _ => None,
    };
    Ok((validated_location, validated_retention))
}

// ---------------------------------------------------------------------------
// TTL cache (cachetools.TTLCache parity: maxsize + ttl)
// ---------------------------------------------------------------------------

struct TtlCache<V: Clone> {
    map: Mutex<HashMap<String, (Instant, V)>>,
    capacity: usize,
    ttl: Duration,
}

impl<V: Clone> TtlCache<V> {
    fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            capacity,
            ttl,
        }
    }

    fn get(&self, key: &str) -> Option<V> {
        let mut map = self.map.lock().unwrap();
        match map.get(key) {
            Some((inserted, v)) if inserted.elapsed() < self.ttl => Some(v.clone()),
            Some(_) => {
                map.remove(key);
                None
            }
            None => None,
        }
    }

    fn insert(&self, key: String, value: V) {
        let mut map = self.map.lock().unwrap();
        map.retain(|_, (inserted, _)| inserted.elapsed() < self.ttl);
        if map.len() >= self.capacity && !map.contains_key(&key) {
            // cachetools.TTLCache evicts the oldest entry when over capacity.
            if let Some(oldest) = map
                .iter()
                .min_by_key(|(_, (inserted, _))| *inserted)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest);
            }
        }
        map.insert(key, (Instant::now(), value));
    }

    fn remove(&self, key: &str) {
        self.map.lock().unwrap().remove(key);
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// SQL-backed workspace store (`mlflow.store.workspace.sqlalchemy_store.SqlAlchemyStore`).
pub struct WorkspaceStore {
    db: Db,
    workspace_uri: String,
    artifact_root_cache: TtlCache<Option<String>>,
    trace_archival_config_cache: TtlCache<(Option<String>, Option<String>)>,
}

impl std::fmt::Debug for WorkspaceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceStore")
            .field("workspace_uri", &self.workspace_uri)
            .finish_non_exhaustive()
    }
}

fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("database error: {e}"))
}

impl WorkspaceStore {
    /// Build a store over an already-connected [`Db`]. `workspace_uri` is only
    /// used to render the `mlflow gc` hint logged after a CASCADE delete.
    pub fn new(db: Db, workspace_uri: impl Into<String>) -> Self {
        let capacity = cache_capacity();
        let ttl = cache_ttl();
        Self {
            db,
            workspace_uri: workspace_uri.into(),
            artifact_root_cache: TtlCache::new(capacity, ttl),
            trace_archival_config_cache: TtlCache::new(capacity, ttl),
        }
    }

    /// The underlying database pool.
    pub fn db(&self) -> &Db {
        &self.db
    }

    fn map_row(r: &dyn super::dbutil::RowLike) -> Result<Workspace, sqlx::Error> {
        Ok(Workspace {
            name: r.get_string("name")?,
            description: r.get_opt_string("description")?,
            default_artifact_root: r.get_opt_string("default_artifact_root")?,
            trace_archival_location: r.get_opt_string("trace_archival_location")?,
            trace_archival_retention: r.get_opt_string("trace_archival_retention")?,
        })
    }

    const SELECT_COLS: &'static str =
        "name, description, default_artifact_root, trace_archival_location, \
         trace_archival_retention";

    /// `list_workspaces` — all workspaces ordered by `name` ascending.
    pub async fn list_workspaces(&self) -> Result<Vec<Workspace>, MlflowError> {
        let sql = format!(
            "SELECT {} FROM {WORKSPACES} ORDER BY name ASC",
            Self::SELECT_COLS
        );
        self.db
            .fetch_all(&sql, &[], Self::map_row)
            .await
            .map_err(internal)
    }

    /// `get_workspace` — raises `RESOURCE_DOES_NOT_EXIST` when absent.
    pub async fn get_workspace(&self, workspace_name: &str) -> Result<Workspace, MlflowError> {
        self.fetch_workspace(workspace_name)
            .await?
            .ok_or_else(|| not_found(workspace_name))
    }

    async fn fetch_workspace(&self, name: &str) -> Result<Option<Workspace>, MlflowError> {
        let dialect = self.db.dialect();
        let sql = format!(
            "SELECT {} FROM {WORKSPACES} WHERE name = {}",
            Self::SELECT_COLS,
            dialect.placeholder(1)
        );
        self.db
            .fetch_optional(&sql, &[Val::Text(name.to_string())], Self::map_row)
            .await
            .map_err(internal)
    }

    /// `get_default_workspace`.
    pub async fn get_default_workspace(&self) -> Result<Workspace, MlflowError> {
        self.get_workspace(DEFAULT_WORKSPACE_NAME).await
    }

    /// `create_workspace`. Validates the name and trace-archival config, inserts,
    /// then primes both TTL caches (only after a successful commit).
    pub async fn create_workspace(&self, workspace: Workspace) -> Result<Workspace, MlflowError> {
        WorkspaceNameValidator::validate(&workspace.name)?;
        let (loc, retention) = validate_workspace_trace_archival_config(
            workspace.trace_archival_location.as_deref(),
            workspace.trace_archival_retention.as_deref(),
        )?;
        let artifact_root = workspace
            .default_artifact_root
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let dialect = self.db.dialect();
        let sql = format!(
            "INSERT INTO {WORKSPACES} \
             (name, description, default_artifact_root, trace_archival_location, \
              trace_archival_retention) VALUES ({}, {}, {}, {}, {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
            dialect.placeholder(5),
        );
        let vals = [
            Val::Text(workspace.name.clone()),
            Val::OptText(workspace.description.clone()),
            Val::OptText(artifact_root.clone()),
            Val::OptText(loc.clone()),
            Val::OptText(retention.clone()),
        ];
        self.db.exec(&sql, &vals).await.map_err(|e| {
            if is_unique_violation(&e) {
                MlflowError::resource_already_exists(format!(
                    "Workspace '{}' already exists. Error: {e}",
                    workspace.name
                ))
            } else {
                internal(e)
            }
        })?;

        let entity = Workspace {
            name: workspace.name.clone(),
            description: workspace.description,
            default_artifact_root: artifact_root,
            trace_archival_location: loc,
            trace_archival_retention: retention,
        };
        self.prime_caches(&entity);
        Ok(entity)
    }

    /// `update_workspace`. `None` fields are left unchanged; an empty string in
    /// `default_artifact_root` / `trace_archival_location` /
    /// `trace_archival_retention` clears the column.
    pub async fn update_workspace(&self, workspace: Workspace) -> Result<Workspace, MlflowError> {
        let (loc, retention) = validate_workspace_trace_archival_config(
            workspace.trace_archival_location.as_deref(),
            workspace.trace_archival_retention.as_deref(),
        )?;

        let mut current = self
            .fetch_workspace(&workspace.name)
            .await?
            .ok_or_else(|| not_found(&workspace.name))?;

        if let Some(desc) = workspace.description {
            current.description = Some(desc);
        }
        if let Some(root) = workspace.default_artifact_root {
            current.default_artifact_root = if root.is_empty() { None } else { Some(root) };
        }
        if workspace.trace_archival_location.is_some() {
            current.trace_archival_location = loc;
        }
        if workspace.trace_archival_retention.is_some() {
            current.trace_archival_retention = retention;
        }

        let dialect = self.db.dialect();
        let sql = format!(
            "UPDATE {WORKSPACES} SET description = {}, default_artifact_root = {}, \
             trace_archival_location = {}, trace_archival_retention = {} WHERE name = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
            dialect.placeholder(5),
        );
        let vals = [
            Val::OptText(current.description.clone()),
            Val::OptText(current.default_artifact_root.clone()),
            Val::OptText(current.trace_archival_location.clone()),
            Val::OptText(current.trace_archival_retention.clone()),
            Val::Text(current.name.clone()),
        ];
        self.db.exec(&sql, &vals).await.map_err(internal)?;

        self.prime_caches(&current);
        Ok(current)
    }

    fn prime_caches(&self, entity: &Workspace) {
        self.artifact_root_cache
            .insert(entity.name.clone(), entity.default_artifact_root.clone());
        self.trace_archival_config_cache.insert(
            entity.name.clone(),
            (
                entity.trace_archival_location.clone(),
                entity.trace_archival_retention.clone(),
            ),
        );
    }

    /// `delete_workspace`. The reserved `default` workspace cannot be deleted.
    pub async fn delete_workspace(
        &self,
        workspace_name: &str,
        mode: WorkspaceDeletionMode,
    ) -> Result<(), MlflowError> {
        if workspace_name == DEFAULT_WORKSPACE_NAME {
            return Err(MlflowError::invalid_state(format!(
                "Cannot delete the reserved '{DEFAULT_WORKSPACE_NAME}' workspace"
            )));
        }

        // `_get_workspace`: existence check up front (RESOURCE_DOES_NOT_EXIST).
        if self.fetch_workspace(workspace_name).await?.is_none() {
            return Err(not_found(workspace_name));
        }

        let mut tx = self.db.begin_tx().await.map_err(internal)?;
        let dialect = self.db.dialect();

        let result = self
            .run_delete(&mut tx, dialect, workspace_name, mode)
            .await;
        match result {
            Ok(()) => {
                tx.commit().await.map_err(internal)?;
            }
            Err(e) => {
                // Drop tx (rollback) then surface the error. On unique/integrity
                // failures during SET_DEFAULT / CASCADE, Python remaps to a
                // friendly INVALID_STATE message.
                drop(tx);
                return Err(e);
            }
        }

        self.artifact_root_cache.remove(workspace_name);
        self.trace_archival_config_cache.remove(workspace_name);
        Ok(())
    }

    async fn run_delete(
        &self,
        tx: &mut Tx<'_>,
        dialect: crate::dialect::Dialect,
        workspace_name: &str,
        mode: WorkspaceDeletionMode,
    ) -> Result<(), MlflowError> {
        match mode {
            WorkspaceDeletionMode::Restrict => {
                cascade::restrict(tx, dialect, workspace_name).await?;
            }
            WorkspaceDeletionMode::Cascade => {
                cascade::cascade(tx, dialect, workspace_name).await?;
            }
            WorkspaceDeletionMode::SetDefault => {
                cascade::set_default(tx, dialect, workspace_name).await?;
            }
        }
        let del = format!(
            "DELETE FROM {WORKSPACES} WHERE name = {}",
            dialect.placeholder(1)
        );
        tx.exec(&del, &[Val::Text(workspace_name.to_string())])
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `resolve_artifact_root`. Returns `(root, append_workspace_prefix)`.
    /// A workspace override yields `(override, false)`; otherwise
    /// `(default_artifact_root, true)`. TTL-cached per workspace.
    pub async fn resolve_artifact_root(
        &self,
        default_artifact_root: Option<&str>,
        workspace_name: &str,
    ) -> Result<(Option<String>, bool), MlflowError> {
        if let Some(cached) = self.artifact_root_cache.get(workspace_name) {
            return Ok(match cached {
                Some(root) => (Some(root), false),
                None => (default_artifact_root.map(str::to_string), true),
            });
        }

        let workspace_root = self
            .fetch_workspace(workspace_name)
            .await?
            .and_then(|w| w.default_artifact_root);
        self.artifact_root_cache
            .insert(workspace_name.to_string(), workspace_root.clone());
        Ok(match workspace_root {
            Some(root) => (Some(root), false),
            None => (default_artifact_root.map(str::to_string), true),
        })
    }

    /// `resolve_trace_archival_config`. TTL-cached per workspace.
    pub async fn resolve_trace_archival_config(
        &self,
        default_trace_archival_root: &str,
        default_retention: &str,
        workspace_name: &str,
    ) -> Result<ResolvedTraceArchivalConfig, MlflowError> {
        let (workspace_root, workspace_retention) =
            match self.trace_archival_config_cache.get(workspace_name) {
                Some(v) => v,
                None => {
                    let row = self.fetch_workspace(workspace_name).await?;
                    let v = row
                        .map(|w| (w.trace_archival_location, w.trace_archival_retention))
                        .unwrap_or((None, None));
                    self.trace_archival_config_cache
                        .insert(workspace_name.to_string(), v.clone());
                    v
                }
            };

        let append_workspace_prefix = workspace_root.is_none();
        Ok(ResolvedTraceArchivalConfig {
            config: TraceArchivalConfig {
                location: Some(
                    workspace_root.unwrap_or_else(|| default_trace_archival_root.to_string()),
                ),
                retention: Some(
                    workspace_retention.unwrap_or_else(|| default_retention.to_string()),
                ),
            },
            append_workspace_prefix,
        })
    }
}

fn not_found(name: &str) -> MlflowError {
    MlflowError::new(
        format!("Workspace '{name}' not found"),
        ErrorCode::ResourceDoesNotExist,
    )
}

/// The single-tenant startup guard (plan T10.3). When the server boots with
/// workspaces **disabled**, refuse to start if any workspace-scoped root entity
/// lives outside the `default` workspace — otherwise those rows would become
/// silently unreachable (every single-tenant query filters `workspace =
/// 'default'`). Mirrors the store-construction guard exercised by
/// `test_single_tenant_startup_rejects_non_default_workspace_experiments`
/// (tracking) and `..._models` (registry) — the messages and `INVALID_STATE`
/// code are byte-matched to Python.
///
/// Each `(table, entity_label)` pair names a root table carrying a `workspace`
/// column and the plural noun for its error message ("experiments",
/// "registered models", "webhooks"). Tables that don't exist on the configured
/// backend are skipped (the count query errors → treated as "no such table" →
/// no rows). All three share the single Alembic-migrated DB pool.
pub async fn verify_single_tenant_data(
    db: &Db,
    checks: &[(&str, &str)],
) -> Result<(), MlflowError> {
    for (table, label) in checks {
        let sql = format!(
            "SELECT COUNT(*) AS n FROM {table} WHERE workspace <> {}",
            db.dialect().placeholder(1)
        );
        let count = db
            .fetch_optional(
                &sql,
                &[Val::Text(DEFAULT_WORKSPACE_NAME.to_string())],
                |r| r.get_i64("n"),
            )
            .await;
        // A missing table (backend without that feature) is not a guard
        // violation — skip it.
        if let Ok(Some(n)) = count {
            if n > 0 {
                return Err(MlflowError::new(
                    format!(
                        "Cannot disable workspaces because {label} exist outside the \
                         default workspace"
                    ),
                    ErrorCode::InvalidState,
                ));
            }
        }
    }
    Ok(())
}

/// Whether an sqlx error is a UNIQUE/PK constraint violation (SQLite code 1555 /
/// 2067, Postgres SQLSTATE 23505, MySQL 1062). Mirrors Python's `IntegrityError`
/// handling for the `workspaces_pk` primary key.
fn is_unique_violation(e: &sqlx::Error) -> bool {
    if let sqlx::Error::Database(db) = e {
        if let Some(code) = db.code() {
            return matches!(code.as_ref(), "1555" | "2067" | "23505" | "1062");
        }
        let msg = db.message().to_ascii_lowercase();
        return msg.contains("unique") || msg.contains("duplicate");
    }
    false
}

pub(crate) mod cascade {
    //! Workspace-scoped delete/reassign walking, mirroring
    //! `_WORKSPACE_ROOT_MODELS` and the SQLAlchemy ORM `cascade="all"` trees.

    use super::{internal, DEFAULT_WORKSPACE_NAME};
    use crate::dialect::Dialect;
    use crate::store::dbutil::{Tx, Val};
    use mlflow_error::MlflowError;

    include!("workspaces_cascade.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_accepts_valid_names() {
        for name in ["team-a", "ab", &"a".repeat(63), "123", "a1-b2"] {
            WorkspaceNameValidator::validate(name).expect(name);
        }
    }

    #[test]
    fn validator_rejects_invalid_names() {
        let cases: &[(&str, &str)] = &[
            ("t", "must be between"),
            ("Team-A", "must match the pattern"),
            ("team_a", "must match the pattern"),
            ("team--a", "must match the pattern"),
            ("-team", "must match the pattern"),
            ("team-", "must match the pattern"),
            ("workspaces", "is reserved"),
        ];
        for (name, fragment) in cases {
            let err = WorkspaceNameValidator::validate(name).unwrap_err();
            assert_eq!(err.error_code, ErrorCode::InvalidParameterValue, "{name}");
            assert!(err.message.contains(fragment), "{name}: {}", err.message);
        }
        // Length: 64 and 256 chars both "must be between".
        for n in [64usize, 256] {
            let long = "a".repeat(n);
            let err = WorkspaceNameValidator::validate(&long).unwrap_err();
            assert!(err.message.contains("must be between"), "{n}");
        }
    }

    #[test]
    fn validator_reserved_names_all_rejected() {
        for name in WorkspaceNameValidator::RESERVED {
            let err = WorkspaceNameValidator::validate(name).unwrap_err();
            assert!(err.message.contains("is reserved"), "{name}");
        }
    }

    #[test]
    fn retention_regex_matrix() {
        assert!(retention_regex_matches("30d"));
        assert!(retention_regex_matches("12h"));
        assert!(retention_regex_matches("1m"));
        assert!(!retention_regex_matches("0d"));
        assert!(!retention_regex_matches("30x"));
        assert!(!retention_regex_matches("d"));
        assert!(!retention_regex_matches("thirty-days"));
        assert!(!retention_regex_matches(""));
    }

    #[test]
    fn retention_error_message() {
        let err = validate_trace_archival_retention("thirty-days").unwrap_err();
        assert!(
            err.message.contains("Trace archival retention must"),
            "{}",
            err.message
        );
    }

    #[test]
    fn location_rejects_proxy_scheme() {
        let err = validate_trace_archival_location("mlflow-artifacts:/archive/team-a").unwrap_err();
        assert!(
            err.message
                .contains("proxy-only `mlflow-artifacts:` scheme"),
            "{}",
            err.message
        );
    }

    #[test]
    fn location_accepts_regular_uri() {
        assert_eq!(
            validate_trace_archival_location("s3://archive/team-a").unwrap(),
            "s3://archive/team-a"
        );
    }

    #[test]
    fn ttl_cache_expires() {
        let cache: TtlCache<i64> = TtlCache::new(4, Duration::from_millis(20));
        cache.insert("a".into(), 1);
        assert_eq!(cache.get("a"), Some(1));
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(cache.get("a"), None);
    }

    #[test]
    fn ttl_cache_capacity_evicts_oldest() {
        let cache: TtlCache<i64> = TtlCache::new(2, Duration::from_secs(60));
        cache.insert("a".into(), 1);
        cache.insert("b".into(), 2);
        cache.insert("c".into(), 3);
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), Some(2));
        assert_eq!(cache.get("c"), Some(3));
    }
}
