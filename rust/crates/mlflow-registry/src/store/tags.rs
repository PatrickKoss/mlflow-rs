//! Registered-model and model-version tag operations, mirroring the tag methods
//! in `mlflow/store/model_registry/sqlalchemy_store.py`.
//!
//! Semantics (both tag kinds):
//! * set = **upsert** (`session.merge`): overwrites the value for an existing
//!   key, after checking the parent (model / model version) exists.
//! * delete = check parent exists, then delete the tag **if present**;
//!   deleting a missing key is a silent no-op (no error).

use mlflow_error::MlflowError;
use mlflow_store::dialect::UpsertSpec;

use super::registered_models::registered_model_tag_upsert;
use super::{internal, RegistryStore};
use crate::dbutil::{DbExt, Val};
use crate::schema::{MODEL_VERSION_TAGS, REGISTERED_MODEL_TAGS};
use crate::validation;

impl RegistryStore {
    /// `set_registered_model_tag` (upsert on `(workspace, key, name)`).
    pub async fn set_registered_model_tag(
        &self,
        workspace: &str,
        name: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_registered_model_tag(key, value)?;
        self.require_registered_model(workspace, name).await?;
        let dialect = self.db().dialect();
        let sql = dialect.upsert(&registered_model_tag_upsert());
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Text(key.to_string()),
                    Val::OptText(Some(value.to_string())),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `delete_registered_model_tag`: no-op if the tag is absent.
    pub async fn delete_registered_model_tag(
        &self,
        workspace: &str,
        name: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_tag_key(key)?;
        self.require_registered_model(workspace, name).await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM {REGISTERED_MODEL_TAGS} WHERE workspace = {} AND name = {} AND key = {}",
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
                    Val::Text(key.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `set_model_version_tag` (upsert on `(workspace, key, name, version)`).
    pub async fn set_model_version_tag(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_version(version)?;
        validation::validate_model_version_tag(key, value)?;
        let mv = self.require_model_version(workspace, name, version).await?;
        let dialect = self.db().dialect();
        let spec = UpsertSpec {
            table: MODEL_VERSION_TAGS,
            columns: &["workspace", "name", "version", "key", "value"],
            pk_columns: &["workspace", "key", "name", "version"],
            update_columns: &["value"],
        };
        let sql = dialect.upsert(&spec);
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(mv.version),
                    Val::Text(key.to_string()),
                    Val::OptText(Some(value.to_string())),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `delete_model_version_tag`: no-op if the tag is absent.
    pub async fn delete_model_version_tag(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_model_name(name)?;
        validation::validate_model_version(version)?;
        validation::validate_tag_key(key)?;
        let mv = self.require_model_version(workspace, name, version).await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM {MODEL_VERSION_TAGS} \
             WHERE workspace = {} AND name = {} AND version = {} AND key = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4)
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::Int(mv.version),
                    Val::Text(key.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }
}
