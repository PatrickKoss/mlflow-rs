//! Experiment operations, mirroring the experiment methods in
//! `mlflow/store/tracking/sqlalchemy_store.py`.
//!
//! Workspace scoping (plan §3.17): every query filters
//! `experiments.workspace = ?`. The `(workspace, name)` unique constraint
//! (`uq_experiments_workspace_name`) means a create colliding with *any*
//! experiment of that name in the workspace — active OR deleted — yields
//! `RESOURCE_ALREADY_EXISTS` (the deleted-name-conflict case).

use mlflow_error::{ErrorCode, MlflowError};

use super::dbutil::{Tx, Val};
use super::entities::{Experiment, ExperimentTag, LifecycleStage};
use super::uri_util::append_to_uri_path;
use super::validation;
use super::TrackingStore;
use crate::schema::runs::{EXPERIMENTS, EXPERIMENT_TAGS};

/// View-type filter for lifecycle stages (`ViewType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewType {
    ActiveOnly,
    DeletedOnly,
    All,
}

impl ViewType {
    fn stages(self) -> &'static [&'static str] {
        match self {
            ViewType::ActiveOnly => &[LifecycleStage::ACTIVE],
            ViewType::DeletedOnly => &[LifecycleStage::DELETED],
            ViewType::All => &[LifecycleStage::ACTIVE, LifecycleStage::DELETED],
        }
    }
}

impl TrackingStore {
    /// `create_experiment`. Returns the new experiment id (stringified).
    pub async fn create_experiment(
        &self,
        workspace: &str,
        name: &str,
        artifact_location: Option<&str>,
        tags: &[(&str, &str)],
    ) -> Result<String, MlflowError> {
        validation::validate_experiment_name(name)?;
        for (k, v) in tags {
            validation::validate_experiment_tag(k, v)?;
        }
        let creation_time = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);

        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Insert with NULL artifact_location first so we can derive the default
        // location from the generated id (mirrors Python's double-flush).
        let insert_sql = format!(
            "INSERT INTO {EXPERIMENTS} \
             (name, workspace, artifact_location, lifecycle_stage, creation_time, last_update_time) \
             VALUES ({}, {}, {}, {}, {}, {})",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
            ph(5),
            ph(6),
        );
        let insert_vals = vec![
            Val::Text(name.to_string()),
            Val::Text(workspace.to_string()),
            Val::OptText(artifact_location.map(str::to_string)),
            Val::Text(LifecycleStage::ACTIVE.to_string()),
            Val::Int(creation_time),
            Val::Int(creation_time),
        ];
        if let Err(e) = tx.exec(&insert_sql, &insert_vals).await {
            return Err(map_insert_err(e, name));
        }

        // Recover the generated id (workspace-scoped by name — unique).
        let id = self
            .experiment_id_by_name_tx(&mut tx, workspace, name)
            .await?
            .ok_or_else(|| internal_msg("experiment insert did not produce a row"))?;

        if artifact_location.is_none() {
            let loc = self.default_experiment_artifact_location(id);
            let sql = format!(
                "UPDATE {EXPERIMENTS} SET artifact_location = {} WHERE experiment_id = {}",
                ph(1),
                ph(2)
            );
            tx.exec(&sql, &[Val::Text(loc), Val::Int(id)])
                .await
                .map_err(internal)?;
        }

        for (k, v) in tags {
            let sql = format!(
                "INSERT INTO {EXPERIMENT_TAGS} (\"key\", value, experiment_id) VALUES ({}, {}, {})",
                ph(1),
                ph(2),
                ph(3)
            );
            tx.exec(
                &sql,
                &[
                    Val::Text(k.to_string()),
                    Val::Text(v.to_string()),
                    Val::Int(id),
                ],
            )
            .await
            .map_err(internal)?;
        }

        tx.commit().await.map_err(internal)?;
        Ok(id.to_string())
    }

    /// `get_experiment` (view type ALL).
    pub async fn get_experiment(
        &self,
        workspace: &str,
        experiment_id: &str,
    ) -> Result<Experiment, MlflowError> {
        let id = parse_experiment_id(experiment_id)?;
        self.fetch_experiment(workspace, id, ViewType::All)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!("No Experiment with id={id} exists"))
            })
    }

    /// `get_experiment_by_name` (view ALL). Returns `None` when absent.
    pub async fn get_experiment_by_name(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Option<Experiment>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT experiment_id FROM {EXPERIMENTS} \
             WHERE name = {} AND workspace = {} AND lifecycle_stage IN ('active', 'deleted')",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let id = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(name.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |r| r.get_int("experiment_id"),
            )
            .await
            .map_err(internal)?;
        match id {
            None => Ok(None),
            Some(id) => self.fetch_experiment(workspace, id, ViewType::All).await,
        }
    }

    /// `delete_experiment`: mark ACTIVE experiment DELETED and cascade the
    /// soft-delete to child runs (setting each run's `deleted_time`).
    pub async fn delete_experiment(
        &self,
        workspace: &str,
        experiment_id: &str,
    ) -> Result<(), MlflowError> {
        let id = parse_experiment_id(experiment_id)?;
        self.require_experiment(workspace, id, ViewType::ActiveOnly)
            .await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        tx.exec(
            &format!(
                "UPDATE {EXPERIMENTS} SET lifecycle_stage = 'deleted', last_update_time = {} \
                 WHERE experiment_id = {}",
                ph(1),
                ph(2)
            ),
            &[Val::Int(now), Val::Int(id)],
        )
        .await
        .map_err(internal)?;
        tx.exec(
            &format!(
                "UPDATE runs SET lifecycle_stage = 'deleted', deleted_time = {} \
                 WHERE experiment_id = {}",
                ph(1),
                ph(2)
            ),
            &[Val::Int(now), Val::Int(id)],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)
    }

    /// `restore_experiment`: mark DELETED experiment ACTIVE and restore child
    /// runs (clearing `deleted_time`).
    pub async fn restore_experiment(
        &self,
        workspace: &str,
        experiment_id: &str,
    ) -> Result<(), MlflowError> {
        let id = parse_experiment_id(experiment_id)?;
        self.require_experiment(workspace, id, ViewType::DeletedOnly)
            .await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        tx.exec(
            &format!(
                "UPDATE {EXPERIMENTS} SET lifecycle_stage = 'active', last_update_time = {} \
                 WHERE experiment_id = {}",
                ph(1),
                ph(2)
            ),
            &[Val::Int(now), Val::Int(id)],
        )
        .await
        .map_err(internal)?;
        tx.exec(
            &format!(
                "UPDATE runs SET lifecycle_stage = 'active', deleted_time = NULL \
                 WHERE experiment_id = {}",
                ph(1)
            ),
            &[Val::Int(id)],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)
    }

    /// `rename_experiment`. Requires the experiment to be ACTIVE.
    pub async fn rename_experiment(
        &self,
        workspace: &str,
        experiment_id: &str,
        new_name: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_experiment_name(new_name)?;
        let id = parse_experiment_id(experiment_id)?;
        let exp = self
            .require_experiment(workspace, id, ViewType::All)
            .await?;
        if exp.lifecycle_stage != LifecycleStage::ACTIVE {
            return Err(MlflowError::new(
                "Cannot rename a non-active experiment.",
                ErrorCode::InvalidState,
            ));
        }
        let now = now_millis();
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {EXPERIMENTS} SET name = {}, last_update_time = {} WHERE experiment_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        self.db()
            .exec(
                &sql,
                &[Val::Text(new_name.to_string()), Val::Int(now), Val::Int(id)],
            )
            .await
            .map_err(|e| map_insert_err(e, new_name))?;
        Ok(())
    }

    /// `set_experiment_tag` (upsert). Requires the experiment to be ACTIVE.
    pub async fn set_experiment_tag(
        &self,
        workspace: &str,
        experiment_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        validation::validate_experiment_tag(key, value)?;
        let id = parse_experiment_id(experiment_id)?;
        let exp = self
            .require_experiment(workspace, id, ViewType::All)
            .await?;
        require_active_experiment(&exp)?;
        let dialect = self.db().dialect();
        let spec = crate::dialect::UpsertSpec {
            table: EXPERIMENT_TAGS,
            columns: &["key", "value", "experiment_id"],
            pk_columns: &["key", "experiment_id"],
            update_columns: &["value"],
            ..Default::default()
        };
        let sql = dialect.upsert(&spec);
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(key.to_string()),
                    Val::Text(value.to_string()),
                    Val::Int(id),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `delete_experiment_tag`. Errors `RESOURCE_DOES_NOT_EXIST` when absent.
    /// Requires the experiment to be ACTIVE.
    pub async fn delete_experiment_tag(
        &self,
        workspace: &str,
        experiment_id: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        let id = parse_experiment_id(experiment_id)?;
        let exp = self
            .require_experiment(workspace, id, ViewType::All)
            .await?;
        require_active_experiment(&exp)?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM {EXPERIMENT_TAGS} WHERE experiment_id = {} AND \"key\" = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let affected = self
            .db()
            .exec(&sql, &[Val::Int(id), Val::Text(key.to_string())])
            .await
            .map_err(internal)?;
        if affected == 0 {
            return Err(MlflowError::resource_does_not_exist(format!(
                "No tag with name: {key} in experiment with id {experiment_id}"
            )));
        }
        Ok(())
    }

    // ---- internal helpers ----

    fn default_experiment_artifact_location(&self, experiment_id: i64) -> String {
        append_to_uri_path(self.artifact_root_uri(), &[&experiment_id.to_string()])
    }

    /// Fetch the experiment (with tags) if it exists in the workspace and its
    /// lifecycle stage matches `view`.
    pub(crate) async fn fetch_experiment(
        &self,
        workspace: &str,
        experiment_id: i64,
        view: ViewType,
    ) -> Result<Option<Experiment>, MlflowError> {
        let dialect = self.db().dialect();
        let stages = stages_in_list(view.stages());
        let sql = format!(
            "SELECT experiment_id, name, artifact_location, lifecycle_stage, \
             creation_time, last_update_time FROM {EXPERIMENTS} \
             WHERE experiment_id = {} AND workspace = {} AND lifecycle_stage IN ({stages})",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let exp = self
            .db()
            .fetch_optional(
                &sql,
                &[Val::Int(experiment_id), Val::Text(workspace.to_string())],
                |r| {
                    Ok(Experiment {
                        experiment_id: r.get_int("experiment_id")?.to_string(),
                        name: r.get_string("name")?,
                        artifact_location: r.get_opt_string("artifact_location")?,
                        lifecycle_stage: r.get_string("lifecycle_stage")?,
                        creation_time: r.get_opt_i64("creation_time")?,
                        last_update_time: r.get_opt_i64("last_update_time")?,
                        tags: Vec::new(),
                    })
                },
            )
            .await
            .map_err(internal)?;
        let Some(mut exp) = exp else {
            return Ok(None);
        };
        exp.tags = self.fetch_experiment_tags(experiment_id).await?;
        Ok(Some(exp))
    }

    async fn fetch_experiment_tags(
        &self,
        experiment_id: i64,
    ) -> Result<Vec<ExperimentTag>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT \"key\", value FROM {EXPERIMENT_TAGS} WHERE experiment_id = {} ORDER BY \"key\"",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Int(experiment_id)], |r| {
                Ok(ExperimentTag {
                    key: r.get_string("key")?,
                    value: r.get_opt_string("value")?,
                })
            })
            .await
            .map_err(internal)
    }

    async fn require_experiment(
        &self,
        workspace: &str,
        experiment_id: i64,
        view: ViewType,
    ) -> Result<Experiment, MlflowError> {
        self.fetch_experiment(workspace, experiment_id, view)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "No Experiment with id={experiment_id} exists"
                ))
            })
    }

    async fn experiment_id_by_name_tx(
        &self,
        tx: &mut Tx<'_>,
        workspace: &str,
        name: &str,
    ) -> Result<Option<i64>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT experiment_id FROM {EXPERIMENTS} WHERE name = {} AND workspace = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let rows = tx
            .fetch_all(
                &sql,
                &[
                    Val::Text(name.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |r| r.get_int("experiment_id"),
            )
            .await
            .map_err(internal)?;
        Ok(rows.into_iter().next())
    }
}

/// `_check_experiment_is_active`.
fn require_active_experiment(exp: &Experiment) -> Result<(), MlflowError> {
    if exp.lifecycle_stage != LifecycleStage::ACTIVE {
        return Err(MlflowError::invalid_parameter_value(format!(
            "The experiment {} must be in the 'active' state. Current state is {}.",
            exp.experiment_id, exp.lifecycle_stage
        )));
    }
    Ok(())
}

fn stages_in_list(stages: &[&str]) -> String {
    stages
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `_get_experiment` integer parse.
pub(crate) fn parse_experiment_id(experiment_id: &str) -> Result<i64, MlflowError> {
    experiment_id.parse::<i64>().map_err(|_| {
        MlflowError::invalid_parameter_value(format!(
            "Invalid experiment ID '{experiment_id}'. Experiment ID must be a valid integer."
        ))
    })
}

fn already_exists(name: &str) -> MlflowError {
    MlflowError::resource_already_exists(format!(
        "Experiment(name={name}) already exists. Error: (workspace, name) must be unique"
    ))
}

/// Now in epoch milliseconds (`get_current_time_millis`).
pub(crate) fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub(crate) fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("database error: {e}"))
}

pub(crate) fn internal_msg(msg: &str) -> MlflowError {
    MlflowError::internal_error(msg.to_string())
}

/// Map an insert error to RESOURCE_ALREADY_EXISTS on unique violation.
fn map_insert_err(e: sqlx::Error, name: &str) -> MlflowError {
    if is_unique_violation(&e) {
        already_exists(name)
    } else {
        internal(e)
    }
}

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
    msg.contains("unique constraint") || msg.contains("duplicate")
}
