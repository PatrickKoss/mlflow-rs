//! Param and tag operations, mirroring `log_param`, `set_tag`, `delete_tag`
//! (and their batch helpers) in `sqlalchemy_store.py`.

use mlflow_error::MlflowError;

use super::dbutil::{Tx, Val};
use super::experiments::internal;
use super::runs::{check_run_active, sync_run_name_tag};
use super::validation;
use super::{TrackingStore, MLFLOW_RUN_NAME};

impl TrackingStore {
    /// `log_param`. Immutable: re-logging the same key with the same value is a
    /// no-op; a different value raises `INVALID_PARAMETER_VALUE` with Python's
    /// exact message. Requires the run to be ACTIVE.
    pub async fn log_param(
        &self,
        workspace: &str,
        run_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_param(key, value)?;
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;

        // Check for an existing value first (Python relies on the DB unique
        // constraint + rollback, but a pre-read gives identical observable
        // behavior and avoids depending on backend-specific error text).
        if let Some(existing) = self.existing_param_value(run_id, key).await? {
            if existing == value {
                return Ok(()); // idempotent
            }
            return Err(MlflowError::invalid_parameter_value(format!(
                "Changing param values is not allowed. Param with key='{key}' was already logged \
                 with value='{existing}' for run ID='{run_id}'. Attempted logging new value \
                 '{value}'."
            )));
        }

        let dialect = self.db().dialect();
        let sql = format!(
            "INSERT INTO params (\"key\", value, run_uuid) VALUES ({}, {}, {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        // On a race, the unique constraint fires; re-check for the mutability
        // message rather than surfacing a raw DB error.
        if let Err(e) = self
            .db()
            .exec(
                &sql,
                &[
                    Val::Text(key.to_string()),
                    Val::Text(value.to_string()),
                    Val::Text(run_id.to_string()),
                ],
            )
            .await
        {
            if super::experiments::is_unique_violation(&e) {
                if let Some(existing) = self.existing_param_value(run_id, key).await? {
                    if existing == value {
                        return Ok(());
                    }
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "Changing param values is not allowed. Param with key='{key}' was already \
                         logged with value='{existing}' for run ID='{run_id}'. Attempted logging \
                         new value '{value}'."
                    )));
                }
            }
            return Err(internal(e));
        }
        Ok(())
    }

    /// `set_tag` (upsert). When the key is `mlflow.runName`, this also updates
    /// the run's `name` column (both-directions sync). Requires ACTIVE run.
    pub async fn set_tag(
        &self,
        workspace: &str,
        run_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_tag(key, value, None)?;
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        if key == MLFLOW_RUN_NAME {
            // Update run.name and the tag together (mirrors update_run_info).
            let sql = format!(
                "UPDATE runs SET name = {} WHERE run_uuid = {}",
                dialect.placeholder(1),
                dialect.placeholder(2)
            );
            tx.exec(
                &sql,
                &[Val::Text(value.to_string()), Val::Text(run_id.to_string())],
            )
            .await
            .map_err(internal)?;
            sync_run_name_tag(&mut tx, dialect, run_id, value).await?;
        } else {
            upsert_tag(&mut tx, dialect, run_id, key, value).await?;
        }
        tx.commit().await.map_err(internal)
    }

    /// `delete_tag`. Errors `RESOURCE_DOES_NOT_EXIST` when the tag is absent.
    /// Requires the run to be ACTIVE.
    pub async fn delete_tag(
        &self,
        workspace: &str,
        run_id: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM tags WHERE run_uuid = {} AND \"key\" = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let affected = self
            .db()
            .exec(
                &sql,
                &[Val::Text(run_id.to_string()), Val::Text(key.to_string())],
            )
            .await
            .map_err(internal)?;
        if affected == 0 {
            return Err(MlflowError::resource_does_not_exist(format!(
                "No tag with name: {key} in run with id {run_id}"
            )));
        }
        Ok(())
    }

    async fn existing_param_value(
        &self,
        run_id: &str,
        key: &str,
    ) -> Result<Option<String>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT value FROM params WHERE run_uuid = {} AND \"key\" = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        self.db()
            .fetch_optional(
                &sql,
                &[Val::Text(run_id.to_string()), Val::Text(key.to_string())],
                |r| r.get_string("value"),
            )
            .await
            .map_err(internal)
    }
}

/// Upsert one non-runName tag inside a transaction.
pub(crate) async fn upsert_tag(
    tx: &mut Tx<'_>,
    dialect: crate::dialect::Dialect,
    run_id: &str,
    key: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: "tags",
        columns: &["key", "value", "run_uuid"],
        pk_columns: &["key", "run_uuid"],
        update_columns: &["value"],
        ..Default::default()
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(key.to_string()),
            Val::Text(value.to_string()),
            Val::Text(run_id.to_string()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}
