//! Registered-model alias operations, mirroring the alias methods in
//! `mlflow/store/model_registry/sqlalchemy_store.py`.
//!
//! Semantics:
//! * `set_registered_model_alias` validates the alias name (char rules +
//!   reserved: `latest`, `v<N>`), then requires the **target model version** to
//!   exist (workspace-scoped, non-`Deleted_Internal`), then upserts the alias
//!   (an existing alias is repointed — overwrite).
//! * `delete_registered_model_alias` requires the registered model to exist,
//!   then deletes the alias if present (missing alias = no-op).
//! * `get_model_version_by_alias` special-cases the reserved `latest` alias
//!   (returns the first latest version), otherwise requires the model to exist
//!   and resolves the alias, erroring `INVALID_PARAMETER_VALUE` "Registered
//!   model alias {alias} not found." when the alias does not exist.

use mlflow_error::MlflowError;
use mlflow_store::dialect::UpsertSpec;

use super::{internal, RegistryStore};
use crate::dbutil::{DbExt, Val};
use crate::entities::ModelVersion;
use crate::schema::REGISTERED_MODEL_ALIASES;
use crate::validation;

/// `_REGISTERED_MODEL_ALIAS_LATEST`.
const REGISTERED_MODEL_ALIAS_LATEST: &str = "latest";

impl RegistryStore {
    /// `set_registered_model_alias`.
    pub async fn set_registered_model_alias(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
        version: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_alias_name(alias)?;
        validation::validate_model_alias_name_reserved(alias)?;
        validation::validate_model_version(version)?;
        // The target model version must exist (workspace-scoped).
        let mv = self.require_model_version(workspace, name, version).await?;
        let dialect = self.db().dialect();
        let spec = UpsertSpec {
            table: REGISTERED_MODEL_ALIASES,
            columns: &["workspace", "name", "alias", "version"],
            pk_columns: &["workspace", "name", "alias"],
            update_columns: &["version"],
            ..Default::default()
        };
        let sql = dialect.upsert(&spec);
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Text(alias.to_string()),
                    Val::Int(mv.version),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `delete_registered_model_alias`: no-op if the alias is absent.
    pub async fn delete_registered_model_alias(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_alias_name(alias)?;
        self.require_registered_model(workspace, name).await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM {REGISTERED_MODEL_ALIASES} \
             WHERE workspace = {} AND name = {} AND alias = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Text(alias.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `get_model_version_by_alias`.
    pub async fn get_model_version_by_alias(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
    ) -> Result<ModelVersion, MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_alias_name(alias)?;

        if alias.to_lowercase() == REGISTERED_MODEL_ALIAS_LATEST {
            let latest = self.get_latest_versions(workspace, name, None).await?;
            return latest.into_iter().next().ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Latest version not found for model {name}."
                ))
            });
        }

        // Registered model must exist (workspace-scoped).
        self.require_registered_model(workspace, name).await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT version FROM {REGISTERED_MODEL_ALIASES} \
             WHERE workspace = {} AND name = {} AND alias = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        let version = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Text(alias.to_string()),
                ],
                |r| crate::dbutil::RowLike::get_int(r, "version"),
            )
            .await
            .map_err(internal)?;
        match version {
            Some(v) => {
                self.get_model_version(workspace, name, &v.to_string())
                    .await
            }
            None => Err(MlflowError::invalid_parameter_value(format!(
                "Registered model alias {alias} not found."
            ))),
        }
    }
}
