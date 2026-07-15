//! Model-version lifecycle, mirroring the model-version methods in
//! `mlflow/store/model_registry/sqlalchemy_store.py`.
//!
//! Covers (T7.1 core + T7.2 lifecycle):
//!
//! * [`RegistryStore::create_model_version`] — the `MAX(version)+1` insert with
//!   the contention retry loop, `models:/name/version` → `storage_location`
//!   source resolution (`runs:/` and other sources stored verbatim),
//!   `current_stage = "None"`, `status = "READY"`.
//! * [`RegistryStore::update_model_version`] — set description, bump
//!   `last_updated_time`.
//! * [`RegistryStore::transition_model_version_stage`] — canonical stage names,
//!   `archive_existing_versions` (archives all other versions in the target
//!   stage, only valid for Staging/Production), transactional over siblings.
//! * [`RegistryStore::delete_model_version`] — soft-delete →
//!   `Deleted_Internal` with source/run_id/run_link/description/status_message
//!   redaction and alias removal.
//! * [`RegistryStore::get_model_version`] / `get_model_version_download_uri` /
//!   [`RegistryStore::require_model_version`] — reads that skip
//!   `Deleted_Internal` versions.
//!
//! ## `models:/` source resolution (`sqlalchemy_store.py:1016-1035`)
//!
//! When `source` has scheme `models` and parses to `models:/name/version`, the
//! store resolves `storage_location` to that referenced version's download URI
//! (`storage_location or source`) via [`RegistryStore::get_model_version_download_uri`].
//! The proto `source` column keeps the verbatim `models:/...` string.
//!
//! The bare `models:/<model_id>` form (a logged-model id, not a name/version)
//! requires a cross-store `MlflowClient().get_logged_model()` call in Python;
//! the registry store cannot resolve it alone, so — matching the boundary of
//! this crate — such a source is stored verbatim (`storage_location = source`),
//! leaving the logged-model lookup to the caller/HTTP layer.
//!
//! `runs:/` sources are **not** specially handled by the Python registry store
//! either; they are stored verbatim.

use mlflow_error::MlflowError;
use mlflow_store::Db;

use super::registered_models::map_model_version_row;
use super::{internal, is_unique_violation, now_millis, RegistryStore};
use crate::dbutil::{DbExt, RowLike, Val};
use crate::entities::{ModelVersion, ModelVersionTag};
use crate::schema::{MODEL_VERSIONS, MODEL_VERSION_TAGS, REGISTERED_MODELS};
use crate::stages::{
    get_canonical_stage, DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS, STAGE_ARCHIVED,
    STAGE_DELETED_INTERNAL, STAGE_NONE,
};
use crate::validation;

/// `ModelVersionStatus.READY` string.
const STATUS_READY: &str = "READY";

/// Redaction sentinels written on soft-delete (`sqlalchemy_store.py:1269-1271`).
const REDACTED_SOURCE: &str = "REDACTED-SOURCE-PATH";
const REDACTED_RUN_ID: &str = "REDACTED-RUN-ID";
const REDACTED_RUN_LINK: &str = "REDACTED-RUN-LINK";

/// Number of MAX(version)+1 insert retries under contention
/// (`CREATE_MODEL_VERSION_RETRIES`).
const CREATE_MODEL_VERSION_RETRIES: usize = 3;

impl RegistryStore {
    /// `create_model_version`. Assigns `MAX(version)+1` for the model, retrying
    /// on a primary-key collision from a concurrent insert. Resolves a
    /// `models:/name/version` source to its `storage_location` (see module docs);
    /// other sources are stored verbatim.
    ///
    /// The parameter list mirrors the Python `create_model_version` signature
    /// (name/source/run_id/tags/run_link/description), hence the arg count.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_model_version(
        &self,
        workspace: &str,
        name: &str,
        source: &str,
        run_id: Option<&str>,
        tags: &[(&str, &str)],
        run_link: Option<&str>,
        description: Option<&str>,
    ) -> Result<ModelVersion, MlflowError> {
        validation::validate_model_name(name)?;
        for (k, v) in tags {
            validation::validate_model_version_tag(k, v)?;
        }
        // Resolve `models:/name/version` sources to the referenced version's
        // download URI; everything else (incl. `runs:/` and bare `models:/<id>`)
        // is stored verbatim. See module docs.
        let storage_location = self.resolve_storage_location(workspace, source).await?;

        let creation_time = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);

        let mut last_err: Option<sqlx::Error> = None;
        for _ in 0..CREATE_MODEL_VERSION_RETRIES {
            // Verify the model exists (workspace-scoped) and get the next version.
            self.require_registered_model(workspace, name).await?;
            let next_version = self.next_version(workspace, name).await?;

            let mut tx = self.db().begin_tx().await.map_err(internal)?;
            // Bump the model's last_updated_time (Python does this in the loop).
            let bump_sql = format!(
                "UPDATE {REGISTERED_MODELS} SET last_updated_time = {} \
                 WHERE workspace = {} AND name = {}",
                ph(1),
                ph(2),
                ph(3)
            );
            tx.exec(
                &bump_sql,
                &[
                    Val::Int(creation_time),
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
            )
            .await
            .map_err(internal)?;

            let insert_sql = format!(
                "INSERT INTO {MODEL_VERSIONS} \
                 (workspace, name, version, creation_time, last_updated_time, description, \
                  user_id, current_stage, source, storage_location, run_id, run_link, status, \
                  status_message) \
                 VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
                ph(4),
                ph(5),
                ph(6),
                ph(7),
                ph(8),
                ph(9),
                ph(10),
                ph(11),
                ph(12),
                ph(13),
                ph(14),
            );
            let insert_vals = vec![
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
                Val::Int(next_version),
                Val::Int(creation_time),
                Val::Int(creation_time),
                Val::OptText(description.map(str::to_string)),
                Val::OptText(None),
                Val::Text(STAGE_NONE.to_string()),
                Val::OptText(Some(source.to_string())),
                Val::OptText(Some(storage_location.to_string())),
                Val::OptText(run_id.map(str::to_string)),
                Val::OptText(run_link.map(str::to_string)),
                Val::Text(STATUS_READY.to_string()),
                Val::OptText(None),
            ];
            match tx.exec(&insert_sql, &insert_vals).await {
                Ok(_) => {}
                Err(e) if is_unique_violation(&e) => {
                    // Concurrent insert took this version — retry with a fresh MAX.
                    last_err = Some(e);
                    continue;
                }
                Err(e) => return Err(internal(e)),
            }

            // Insert version tags (dedup by key, last wins).
            let mut seen: Vec<&str> = Vec::new();
            for (k, v) in tags.iter().rev() {
                if seen.contains(k) {
                    continue;
                }
                seen.push(k);
                // `key` is a reserved word in MySQL — always quote it.
                let tag_sql = format!(
                    "INSERT INTO {MODEL_VERSION_TAGS} (workspace, name, version, {keycol}, value) \
                     VALUES ({}, {}, {}, {}, {})",
                    ph(1),
                    ph(2),
                    ph(3),
                    ph(4),
                    ph(5),
                    keycol = dialect.quote_ident("key"),
                );
                tx.exec(
                    &tag_sql,
                    &[
                        Val::Text(workspace.to_string()),
                        Val::Text(name.to_string()),
                        Val::Int(next_version),
                        Val::Text(k.to_string()),
                        Val::OptText(Some(v.to_string())),
                    ],
                )
                .await
                .map_err(internal)?;
            }
            tx.commit().await.map_err(internal)?;
            return self
                .get_model_version(workspace, name, &next_version.to_string())
                .await;
        }
        Err(internal(last_err.expect("retry loop ran at least once")))
    }

    /// `get_model_version`: the entity (with tags + aliases), skipping
    /// `Deleted_Internal`. Errors `RESOURCE_DOES_NOT_EXIST` when absent.
    pub async fn get_model_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<ModelVersion, MlflowError> {
        let row = self.require_model_version(workspace, name, version).await?;
        let tags = fetch_model_version_tags(self.db(), workspace, name, row.version).await?;
        let aliases = self.fetch_registered_model_aliases(workspace, name).await?;
        let alias_names = aliases
            .into_iter()
            .filter(|a| a.version == row.version.to_string())
            .map(|a| a.alias)
            .collect();
        Ok(row.into_entity(tags, alias_names))
    }

    /// `_get_sql_model_version_including_deleted`: fetch a version **including**
    /// soft-deleted (`Deleted_Internal`) rows. Mirrors the Python test helper
    /// used to verify redaction on delete. Errors `RESOURCE_DOES_NOT_EXIST` when
    /// no such row exists in the workspace.
    pub async fn get_model_version_including_deleted(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<ModelVersion, MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_version(version)?;
        let version_num: i64 = version.parse().expect("validated as integer");
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT workspace, name, version, creation_time, last_updated_time, description, \
             user_id, current_stage, source, run_id, status, status_message, run_link \
             FROM {MODEL_VERSIONS} WHERE workspace = {} AND name = {} AND version = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
        );
        let row = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(version_num),
                ],
                map_model_version_row,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| mv_not_found(name, version))?;
        let tags = fetch_model_version_tags(self.db(), workspace, name, row.version).await?;
        let aliases = self.fetch_registered_model_aliases(workspace, name).await?;
        let alias_names = aliases
            .into_iter()
            .filter(|a| a.version == row.version.to_string())
            .map(|a| a.alias)
            .collect();
        Ok(row.into_entity(tags, alias_names))
    }

    /// `get_model_version_download_uri`: `storage_location or source`.
    pub async fn get_model_version_download_uri(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<String, MlflowError> {
        let dialect = self.db().dialect();
        let ver = self.require_model_version(workspace, name, version).await?;
        let sql = format!(
            "SELECT storage_location, source FROM {MODEL_VERSIONS} \
             WHERE workspace = {} AND name = {} AND version = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        let uri = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(ver.version),
                ],
                |r| {
                    Ok(r.get_opt_string("storage_location")?
                        .or(r.get_opt_string("source")?))
                },
            )
            .await
            .map_err(internal)?
            .flatten();
        Ok(uri.unwrap_or_default())
    }

    /// `update_model_version`: set the description, bump `last_updated_time`.
    /// Errors `RESOURCE_DOES_NOT_EXIST` when the version is absent or soft-deleted.
    pub async fn update_model_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        description: Option<&str>,
    ) -> Result<ModelVersion, MlflowError> {
        let mv = self.require_model_version(workspace, name, version).await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {MODEL_VERSIONS} SET description = {}, last_updated_time = {} \
             WHERE workspace = {} AND name = {} AND version = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
            dialect.placeholder(5),
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::OptText(description.map(str::to_string)),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(mv.version),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_model_version(workspace, name, version).await
    }

    /// `transition_model_version_stage`: move a version to `stage` (canonical,
    /// case-insensitive). When `archive_existing_versions` is set, every *other*
    /// version currently in the target stage is moved to `Archived` in the same
    /// transaction; this is only valid when the target is an active stage
    /// (Staging/Production), else `INVALID_PARAMETER_VALUE`. Bumps
    /// `last_updated_time` on the moved version, each archived sibling, and the
    /// registered model. Mirrors `sqlalchemy_store.py:1192-1242`.
    pub async fn transition_model_version_stage(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        stage: &str,
        archive_existing_versions: bool,
    ) -> Result<ModelVersion, MlflowError> {
        let canonical = get_canonical_stage(stage)?;
        let is_active_stage = DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS.contains(&canonical);
        if archive_existing_versions && !is_active_stage {
            // Python formats the Python list repr `['Staging', 'Production']`.
            return Err(MlflowError::invalid_parameter_value(format!(
                "Model version transition cannot archive existing model versions because \
                 '{stage}' is not an Active stage. Valid stages are ['Staging', 'Production']"
            )));
        }
        // Validates the version exists (workspace-scoped, non-deleted).
        let mv = self.require_model_version(workspace, name, version).await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        if archive_existing_versions {
            let archive_sql = format!(
                "UPDATE {MODEL_VERSIONS} SET current_stage = {}, last_updated_time = {} \
                 WHERE workspace = {} AND name = {} AND version <> {} AND current_stage = {}",
                ph(1),
                ph(2),
                ph(3),
                ph(4),
                ph(5),
                ph(6),
            );
            tx.exec(
                &archive_sql,
                &[
                    Val::Text(STAGE_ARCHIVED.to_string()),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(mv.version),
                    Val::Text(canonical.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        }

        let set_stage_sql = format!(
            "UPDATE {MODEL_VERSIONS} SET current_stage = {}, last_updated_time = {} \
             WHERE workspace = {} AND name = {} AND version = {}",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
            ph(5),
        );
        tx.exec(
            &set_stage_sql,
            &[
                Val::Text(canonical.to_string()),
                Val::Int(now),
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
                Val::Int(mv.version),
            ],
        )
        .await
        .map_err(internal)?;

        let bump_rm_sql = format!(
            "UPDATE {REGISTERED_MODELS} SET last_updated_time = {} \
             WHERE workspace = {} AND name = {}",
            ph(1),
            ph(2),
            ph(3),
        );
        tx.exec(
            &bump_rm_sql,
            &[
                Val::Int(now),
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
            ],
        )
        .await
        .map_err(internal)?;

        tx.commit().await.map_err(internal)?;
        self.get_model_version(workspace, name, version).await
    }

    /// `delete_model_version`: soft-delete → `Deleted_Internal`. Redacts
    /// `source`/`run_id`/`run_link` to sentinels, nulls `description`/`user_id`/
    /// `status_message`, removes any aliases pointing at this version, and bumps
    /// `last_updated_time` on the version and the registered model — all in one
    /// transaction. Tags are intentionally kept (Python comment at `:1255`).
    /// Mirrors `sqlalchemy_store.py:1244-1273`.
    pub async fn delete_model_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<(), MlflowError> {
        let mv = self.require_model_version(workspace, name, version).await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Remove aliases pointing at this version.
        let del_alias_sql = format!(
            "DELETE FROM {aliases} WHERE workspace = {} AND name = {} AND version = {}",
            ph(1),
            ph(2),
            ph(3),
            aliases = crate::schema::REGISTERED_MODEL_ALIASES,
        );
        tx.exec(
            &del_alias_sql,
            &[
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
                Val::Int(mv.version),
            ],
        )
        .await
        .map_err(internal)?;

        // Redact + move to the internal deleted stage.
        let redact_sql = format!(
            "UPDATE {MODEL_VERSIONS} SET current_stage = {}, last_updated_time = {}, \
             description = NULL, user_id = NULL, source = {}, run_id = {}, run_link = {}, \
             status_message = NULL \
             WHERE workspace = {} AND name = {} AND version = {}",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
            ph(5),
            ph(6),
            ph(7),
            ph(8),
        );
        tx.exec(
            &redact_sql,
            &[
                Val::Text(STAGE_DELETED_INTERNAL.to_string()),
                Val::Int(now),
                Val::Text(REDACTED_SOURCE.to_string()),
                Val::Text(REDACTED_RUN_ID.to_string()),
                Val::Text(REDACTED_RUN_LINK.to_string()),
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
                Val::Int(mv.version),
            ],
        )
        .await
        .map_err(internal)?;

        // Bump the registered model's last_updated_time.
        let bump_rm_sql = format!(
            "UPDATE {REGISTERED_MODELS} SET last_updated_time = {} \
             WHERE workspace = {} AND name = {}",
            ph(1),
            ph(2),
            ph(3),
        );
        tx.exec(
            &bump_rm_sql,
            &[
                Val::Int(now),
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
            ],
        )
        .await
        .map_err(internal)?;

        tx.commit().await.map_err(internal)
    }

    /// Resolve the `storage_location` for a new model version's `source`
    /// (`sqlalchemy_store.py:1016-1035`). Only `models:/name/version` is
    /// resolved (to the referenced version's download URI); every other source
    /// — `runs:/`, plain paths, and bare `models:/<model_id>` — is stored
    /// verbatim. See the module docs for why the logged-model-id form is left
    /// to the caller.
    async fn resolve_storage_location(
        &self,
        workspace: &str,
        source: &str,
    ) -> Result<String, MlflowError> {
        let Some((ref_name, ref_version)) = parse_models_name_version(source) else {
            return Ok(source.to_string());
        };
        match self
            .get_model_version_download_uri(workspace, &ref_name, &ref_version)
            .await
        {
            Ok(uri) => Ok(uri),
            Err(e) => Err(MlflowError::invalid_parameter_value(format!(
                "Unable to fetch model from model URI source artifact location '{source}'.\
                 Error: {}",
                e.message
            ))),
        }
    }

    /// `_get_sql_model_version`: fetch the raw row (skipping `Deleted_Internal`)
    /// or error `RESOURCE_DOES_NOT_EXIST`.
    pub(crate) async fn require_model_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<ModelVersionRow, MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_version(version)?;
        let version_num: i64 = version.parse().expect("validated as integer");
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT workspace, name, version, creation_time, last_updated_time, description, \
             user_id, current_stage, source, run_id, status, status_message, run_link \
             FROM {MODEL_VERSIONS} \
             WHERE workspace = {} AND name = {} AND version = {} AND current_stage <> {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
        );
        self.db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(version_num),
                    Val::Text(STAGE_DELETED_INTERNAL.to_string()),
                ],
                map_model_version_row,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| mv_not_found(name, version))
    }

    /// `MAX(version)+1` for a model within its workspace (0 → 1 when none).
    async fn next_version(&self, workspace: &str, name: &str) -> Result<i64, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT MAX(version) AS max_version FROM {MODEL_VERSIONS} \
             WHERE workspace = {} AND name = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let max = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
                |r| r.get_opt_int("max_version"),
            )
            .await
            .map_err(internal)?
            .flatten();
        Ok(max.unwrap_or(0) + 1)
    }
}

/// A raw `model_versions` row (physical `version` as `i64`), plus its
/// [`ModelVersionRow::into_entity`] conversion to the string-versioned entity.
#[derive(Debug, Clone)]
pub(crate) struct ModelVersionRow {
    pub workspace: String,
    pub name: String,
    pub version: i64,
    pub creation_time: Option<i64>,
    pub last_updated_time: Option<i64>,
    pub description: Option<String>,
    pub user_id: Option<String>,
    pub current_stage: Option<String>,
    pub source: Option<String>,
    pub run_id: Option<String>,
    pub status: Option<String>,
    pub status_message: Option<String>,
    pub run_link: Option<String>,
}

impl ModelVersionRow {
    /// Build the entity (`version` stringified per proto), with the provided
    /// tags and alias names.
    pub(crate) fn into_entity(
        self,
        tags: Vec<ModelVersionTag>,
        aliases: Vec<String>,
    ) -> ModelVersion {
        ModelVersion {
            name: self.name,
            version: self.version.to_string(),
            creation_timestamp: self.creation_time,
            last_updated_timestamp: self.last_updated_time,
            description: self.description,
            user_id: self.user_id,
            current_stage: self.current_stage,
            source: self.source,
            run_id: self.run_id,
            status: self.status,
            status_message: self.status_message,
            tags,
            run_link: self.run_link,
            aliases,
            workspace: self.workspace,
        }
    }
}

/// Fetch a version's tags, ordered by key.
pub(crate) async fn fetch_model_version_tags(
    db: &Db,
    workspace: &str,
    name: &str,
    version: i64,
) -> Result<Vec<ModelVersionTag>, MlflowError> {
    let dialect = db.dialect();
    // `key` is a reserved word in MySQL — always quote it.
    let sql = format!(
        "SELECT {keycol}, value FROM {MODEL_VERSION_TAGS} \
         WHERE workspace = {} AND name = {} AND version = {} ORDER BY {keycol}",
        dialect.placeholder(1),
        dialect.placeholder(2),
        dialect.placeholder(3),
        keycol = dialect.quote_ident("key"),
    );
    db.fetch_all(
        &sql,
        &[
            Val::Text(workspace.to_string()),
            Val::Text(name.to_string()),
            Val::Int(version),
        ],
        |r: &dyn RowLike| {
            Ok(ModelVersionTag {
                key: r.get_string("key")?,
                value: r.get_opt_string("value")?,
            })
        },
    )
    .await
    .map_err(internal)
}

/// Parse a `models:/name/version` source into `(name, version)` when the
/// suffix is a plain integer version, mirroring the subset of `_parse_model_uri`
/// the registry store's `create_model_version` acts on. Returns `None` for any
/// other source (non-`models` scheme, alias/stage/latest suffix, or bare
/// `models:/<id>`), which is stored verbatim.
fn parse_models_name_version(source: &str) -> Option<(String, String)> {
    let rest = source.strip_prefix("models:/")?;
    // Reject alias (`name@alias`) and empty forms up front.
    if rest.contains('@') {
        return None;
    }
    let (name, suffix) = rest.split_once('/')?;
    if name.is_empty() || suffix.is_empty() || suffix.contains('/') {
        return None;
    }
    // Only a purely-numeric suffix is a version (stages/`latest` are not
    // resolvable within the registry store).
    if suffix.chars().all(|c| c.is_ascii_digit()) {
        Some((name.to_string(), suffix.to_string()))
    } else {
        None
    }
}

/// `Model Version (name={name}, version={version}) not found`.
pub(crate) fn mv_not_found(name: &str, version: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!(
        "Model Version (name={name}, version={version}) not found"
    ))
}
