//! Minimal model-version support for T7.1.
//!
//! ## Scope boundary (T7.1 vs T7.2)
//!
//! This module implements only the plain-DB parts of the model-version
//! lifecycle needed to exercise `get_latest_versions`, aliases, and the rename
//! cascade in tests:
//!
//! * [`RegistryStore::create_model_version`] — the `MAX(version)+1` insert with
//!   the contention retry loop, `storage_location = source` (no `models:/` /
//!   `runs:/` resolution), `current_stage = "None"`, `status = "READY"`. It does
//!   NOT resolve `models://` sources, transition stages, or soft-delete.
//! * [`RegistryStore::get_model_version`] / `get_model_version_download_uri` /
//!   [`RegistryStore::require_model_version`] — reads that skip
//!   `Deleted_Internal` versions.
//!
//! Everything else (source resolution, `transition_model_version_stage`,
//! `delete_model_version` soft-delete + redaction, `update_model_version`,
//! search) is **T7.2** and intentionally absent here.

use mlflow_error::MlflowError;
use mlflow_store::Db;

use super::registered_models::map_model_version_row;
use super::{internal, is_unique_violation, now_millis, RegistryStore};
use crate::dbutil::{DbExt, RowLike, Val};
use crate::entities::{ModelVersion, ModelVersionTag};
use crate::schema::{MODEL_VERSIONS, MODEL_VERSION_TAGS, REGISTERED_MODELS};
use crate::stages::{STAGE_DELETED_INTERNAL, STAGE_NONE};
use crate::validation;

/// `ModelVersionStatus.READY` string.
const STATUS_READY: &str = "READY";

/// Number of MAX(version)+1 insert retries under contention
/// (`CREATE_MODEL_VERSION_RETRIES`).
const CREATE_MODEL_VERSION_RETRIES: usize = 3;

impl RegistryStore {
    /// Minimal `create_model_version` (see module docs for the T7.1/T7.2
    /// boundary). Assigns `MAX(version)+1` for the model, retrying on a
    /// primary-key collision from a concurrent insert.
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
        // No `models:/`/`runs:/` resolution in T7.1 — storage_location = source.
        let storage_location = source;

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
                let tag_sql = format!(
                    "INSERT INTO {MODEL_VERSION_TAGS} (workspace, name, version, key, value) \
                     VALUES ({}, {}, {}, {}, {})",
                    ph(1),
                    ph(2),
                    ph(3),
                    ph(4),
                    ph(5)
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
    let sql = format!(
        "SELECT key, value FROM {MODEL_VERSION_TAGS} \
         WHERE workspace = {} AND name = {} AND version = {} ORDER BY key",
        dialect.placeholder(1),
        dialect.placeholder(2),
        dialect.placeholder(3)
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

/// `Model Version (name={name}, version={version}) not found`.
pub(crate) fn mv_not_found(name: &str, version: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!(
        "Model Version (name={name}, version={version}) not found"
    ))
}
