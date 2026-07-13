//! Registered-model operations, mirroring the registered-model methods in
//! `mlflow/store/model_registry/sqlalchemy_store.py`.

use mlflow_error::MlflowError;
use mlflow_store::dialect::UpsertSpec;

use super::model_versions::{fetch_model_version_tags, ModelVersionRow};
use super::{internal, is_unique_violation, now_millis, RegistryStore};
use crate::dbutil::{DbExt, RowLike, Val};
use crate::entities::{ModelVersion, RegisteredModel, RegisteredModelAlias, RegisteredModelTag};
use crate::schema::{
    MODEL_VERSIONS, REGISTERED_MODELS, REGISTERED_MODEL_ALIASES, REGISTERED_MODEL_TAGS,
};
use crate::stages::{get_canonical_stage, ALL_STAGES, STAGE_DELETED_INTERNAL};
use crate::validation;

impl RegistryStore {
    /// `create_registered_model`. Returns the created entity.
    pub async fn create_registered_model(
        &self,
        workspace: &str,
        name: &str,
        tags: &[(&str, &str)],
        description: Option<&str>,
    ) -> Result<RegisteredModel, MlflowError> {
        validation::validate_model_name(name)?;
        for (k, v) in tags {
            validation::validate_registered_model_tag(k, v)?;
        }
        let creation_time = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);

        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let insert_sql = format!(
            "INSERT INTO {REGISTERED_MODELS} \
             (workspace, name, creation_time, last_updated_time, description) \
             VALUES ({}, {}, {}, {}, {})",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
            ph(5),
        );
        let insert_vals = vec![
            Val::Text(workspace.to_string()),
            Val::Text(name.to_string()),
            Val::Int(creation_time),
            Val::Int(creation_time),
            Val::OptText(description.map(str::to_string)),
        ];
        if let Err(e) = tx.exec(&insert_sql, &insert_vals).await {
            if is_unique_violation(&e) {
                return Err(already_exists(name));
            }
            return Err(internal(e));
        }

        // Insert tags, deduping by key (Python builds a dict first — last wins).
        let mut seen: Vec<&str> = Vec::new();
        for (k, v) in tags.iter().rev() {
            if seen.contains(k) {
                continue;
            }
            seen.push(k);
            let sql = format!(
                "INSERT INTO {REGISTERED_MODEL_TAGS} (workspace, name, key, value) \
                 VALUES ({}, {}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
                ph(4)
            );
            tx.exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Text(k.to_string()),
                    Val::OptText(Some(v.to_string())),
                ],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)?;
        self.get_registered_model(workspace, name).await
    }

    /// `get_registered_model`: the full entity with latest_versions, tags, and
    /// aliases. Errors `RESOURCE_DOES_NOT_EXIST` when absent.
    pub async fn get_registered_model(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<RegisteredModel, MlflowError> {
        validation::validate_model_name(name)?;
        let row = self
            .fetch_registered_model_row(workspace, name)
            .await?
            .ok_or_else(|| not_found(name))?;
        let tags = self.fetch_registered_model_tags(workspace, name).await?;
        let aliases = self.fetch_registered_model_aliases(workspace, name).await?;
        let latest_versions = self.fetch_latest_versions(workspace, name).await?;
        Ok(RegisteredModel {
            name: row.name,
            creation_timestamp: row.creation_time,
            last_updated_timestamp: row.last_updated_time,
            description: row.description,
            latest_versions,
            tags,
            aliases,
            workspace: row.workspace,
        })
    }

    /// `update_registered_model`: set the description, bump `last_updated_time`.
    pub async fn update_registered_model(
        &self,
        workspace: &str,
        name: &str,
        description: Option<&str>,
    ) -> Result<RegisteredModel, MlflowError> {
        self.require_registered_model(workspace, name).await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {REGISTERED_MODELS} SET description = {}, last_updated_time = {} \
             WHERE workspace = {} AND name = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::OptText(description.map(str::to_string)),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_registered_model(workspace, name).await
    }

    /// `rename_registered_model`: rename the model and (via FK `ON UPDATE
    /// CASCADE`) all child rows; bump `last_updated_time` on the model and each
    /// of its versions. See the module docs for the cascade mechanism.
    pub async fn rename_registered_model(
        &self,
        workspace: &str,
        name: &str,
        new_name: &str,
    ) -> Result<RegisteredModel, MlflowError> {
        validation::validate_model_renaming(new_name)?;
        self.require_registered_model(workspace, name).await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);

        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Rename the registered model; the FK ON UPDATE CASCADE propagates the
        // new name to model_versions, registered_model_tags, and
        // registered_model_aliases (and transitively to model_version_tags).
        let rename_sql = format!(
            "UPDATE {REGISTERED_MODELS} SET name = {}, last_updated_time = {} \
             WHERE workspace = {} AND name = {}",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
        );
        if let Err(e) = tx
            .exec(
                &rename_sql,
                &[
                    Val::Text(new_name.to_string()),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
            )
            .await
        {
            if is_unique_violation(&e) {
                return Err(rename_conflict(new_name, &e));
            }
            return Err(internal(e));
        }

        // Bump last_updated_time on every (now-renamed) version, matching
        // Python's explicit per-version update.
        let bump_sql = format!(
            "UPDATE {MODEL_VERSIONS} SET last_updated_time = {} \
             WHERE workspace = {} AND name = {}",
            ph(1),
            ph(2),
            ph(3),
        );
        tx.exec(
            &bump_sql,
            &[
                Val::Int(now),
                Val::Text(workspace.to_string()),
                Val::Text(new_name.to_string()),
            ],
        )
        .await
        .map_err(internal)?;

        tx.commit().await.map_err(internal)?;
        self.get_registered_model(workspace, new_name).await
    }

    /// `delete_registered_model`: delete the model. The `registered_model_aliases`
    /// FK is `ON DELETE CASCADE`, so aliases drop automatically; versions and
    /// tags are removed explicitly (their FKs are not `ON DELETE CASCADE`, but
    /// the Python ORM cascade deletes them — we replicate that within one
    /// transaction).
    pub async fn delete_registered_model(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<(), MlflowError> {
        self.require_registered_model(workspace, name).await?;
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Delete children first (model_version_tags → model_versions →
        // registered_model_tags), then the model itself. Aliases are removed by
        // the ON DELETE CASCADE FK when the model row goes.
        for sql in [
            format!(
                "DELETE FROM model_version_tags WHERE workspace = {} AND name = {}",
                ph(1),
                ph(2)
            ),
            format!(
                "DELETE FROM {MODEL_VERSIONS} WHERE workspace = {} AND name = {}",
                ph(1),
                ph(2)
            ),
            format!(
                "DELETE FROM {REGISTERED_MODEL_TAGS} WHERE workspace = {} AND name = {}",
                ph(1),
                ph(2)
            ),
            format!(
                "DELETE FROM {REGISTERED_MODELS} WHERE workspace = {} AND name = {}",
                ph(1),
                ph(2)
            ),
        ] {
            tx.exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)
    }

    /// `get_latest_versions`: latest READY... (non-`Deleted_Internal`) version
    /// per stage, optionally filtered to `stages` (canonicalized,
    /// case-insensitive). When `stages` is `None`/empty, all [`ALL_STAGES`] are
    /// returned.
    pub async fn get_latest_versions(
        &self,
        workspace: &str,
        name: &str,
        stages: Option<&[&str]>,
    ) -> Result<Vec<ModelVersion>, MlflowError> {
        self.require_registered_model(workspace, name).await?;
        let latest = self.fetch_latest_versions(workspace, name).await?;
        let expected: Vec<&'static str> = match stages {
            None => ALL_STAGES.to_vec(),
            Some([]) => ALL_STAGES.to_vec(),
            Some(s) => s
                .iter()
                .map(|st| get_canonical_stage(st))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let aliases = self.fetch_registered_model_aliases(workspace, name).await?;
        let mut out: Vec<ModelVersion> = latest
            .into_iter()
            .filter(|mv| {
                mv.current_stage
                    .as_deref()
                    .is_some_and(|s| expected.contains(&s))
            })
            .collect();
        // Populate aliases for each returned version (matching Python).
        for mv in &mut out {
            mv.aliases = aliases
                .iter()
                .filter(|a| a.version == mv.version)
                .map(|a| a.alias.clone())
                .collect();
        }
        Ok(out)
    }

    // ---- internal helpers ----

    /// Fetch the raw registered-model row (no children) if it exists in the
    /// workspace.
    pub(crate) async fn fetch_registered_model_row(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Option<RegisteredModelRow>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT workspace, name, creation_time, last_updated_time, description \
             FROM {REGISTERED_MODELS} WHERE workspace = {} AND name = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        self.db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
                |r| {
                    Ok(RegisteredModelRow {
                        workspace: r.get_string("workspace")?,
                        name: r.get_string("name")?,
                        creation_time: r.get_opt_i64("creation_time")?,
                        last_updated_time: r.get_opt_i64("last_updated_time")?,
                        description: r.get_opt_string("description")?,
                    })
                },
            )
            .await
            .map_err(internal)
    }

    /// `_get_registered_model`: fetch or error `RESOURCE_DOES_NOT_EXIST`.
    pub(crate) async fn require_registered_model(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<RegisteredModelRow, MlflowError> {
        validation::validate_model_name(name)?;
        self.fetch_registered_model_row(workspace, name)
            .await?
            .ok_or_else(|| not_found(name))
    }

    async fn fetch_registered_model_tags(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Vec<RegisteredModelTag>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT key, value FROM {REGISTERED_MODEL_TAGS} \
             WHERE workspace = {} AND name = {} ORDER BY key",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        self.db()
            .fetch_all(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
                |r| {
                    Ok(RegisteredModelTag {
                        key: r.get_string("key")?,
                        value: r.get_opt_string("value")?,
                    })
                },
            )
            .await
            .map_err(internal)
    }

    pub(crate) async fn fetch_registered_model_aliases(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Vec<RegisteredModelAlias>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT alias, version FROM {REGISTERED_MODEL_ALIASES} \
             WHERE workspace = {} AND name = {} ORDER BY alias",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        self.db()
            .fetch_all(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
                |r| {
                    Ok(RegisteredModelAlias {
                        alias: r.get_string("alias")?,
                        version: r.get_int("version")?.to_string(),
                    })
                },
            )
            .await
            .map_err(internal)
    }

    /// Compute the latest (highest-version) non-`Deleted_Internal` model
    /// version per stage for one model. Mirrors `_get_latest_versions_for_models`
    /// (ROW_NUMBER window over `(workspace, name, current_stage)` ordered by
    /// `version DESC`, keep `rn = 1`).
    pub(crate) async fn fetch_latest_versions(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Vec<ModelVersion>, MlflowError> {
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let sql = format!(
            "SELECT workspace, name, version, creation_time, last_updated_time, description, \
             user_id, current_stage, source, run_id, status, status_message, run_link \
             FROM ( \
                SELECT mv.*, ROW_NUMBER() OVER ( \
                    PARTITION BY workspace, name, current_stage ORDER BY version DESC \
                ) AS rn \
                FROM {MODEL_VERSIONS} mv \
                WHERE workspace = {} AND name = {} AND current_stage <> {} \
             ) ranked \
             WHERE rn = 1",
            ph(1),
            ph(2),
            ph(3),
        );
        let mut rows: Vec<ModelVersionRow> = self
            .db()
            .fetch_all(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Text(STAGE_DELETED_INTERNAL.to_string()),
                ],
                map_model_version_row,
            )
            .await
            .map_err(internal)?;
        // Deterministic order (version ascending) for stable entity output.
        rows.sort_by_key(|r| r.version);
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let tags = fetch_model_version_tags(self.db(), workspace, name, row.version).await?;
            out.push(row.into_entity(tags, Vec::new()));
        }
        Ok(out)
    }
}

/// Raw `registered_models` row (no children).
#[derive(Debug, Clone)]
pub(crate) struct RegisteredModelRow {
    pub workspace: String,
    pub name: String,
    pub creation_time: Option<i64>,
    pub last_updated_time: Option<i64>,
    pub description: Option<String>,
}

/// Row mapper for a full `model_versions` row (shared with `model_versions`).
pub(crate) fn map_model_version_row(r: &dyn RowLike) -> Result<ModelVersionRow, sqlx::Error> {
    Ok(ModelVersionRow {
        workspace: r.get_string("workspace")?,
        name: r.get_string("name")?,
        version: r.get_int("version")?,
        creation_time: r.get_opt_i64("creation_time")?,
        last_updated_time: r.get_opt_i64("last_updated_time")?,
        description: r.get_opt_string("description")?,
        user_id: r.get_opt_string("user_id")?,
        current_stage: r.get_opt_string("current_stage")?,
        source: r.get_opt_string("source")?,
        run_id: r.get_opt_string("run_id")?,
        status: r.get_opt_string("status")?,
        status_message: r.get_opt_string("status_message")?,
        run_link: r.get_opt_string("run_link")?,
    })
}

/// `handle_resource_already_exist_error` (non-prompt path):
/// `Registered Model (name={name}) already exists`.
fn already_exists(name: &str) -> MlflowError {
    MlflowError::resource_already_exists(format!("Registered Model (name={name}) already exists"))
}

/// The rename-collision error mirrors Python's
/// `Registered Model (name={new_name}) already exists. Error: {e}`.
fn rename_conflict(new_name: &str, e: &sqlx::Error) -> MlflowError {
    MlflowError::resource_already_exists(format!(
        "Registered Model (name={new_name}) already exists. Error: {e}"
    ))
}

/// `Registered Model with name={name} not found`.
pub(crate) fn not_found(name: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("Registered Model with name={name} not found"))
}

/// Upsert spec for `registered_model_tags` (PK `(workspace, key, name)`).
pub(crate) fn registered_model_tag_upsert() -> UpsertSpec<'static> {
    UpsertSpec {
        table: REGISTERED_MODEL_TAGS,
        columns: &["workspace", "name", "key", "value"],
        pk_columns: &["workspace", "key", "name"],
        update_columns: &["value"],
    }
}
