//! Run operations, mirroring the run methods in
//! `mlflow/store/tracking/sqlalchemy_store.py`.
//!
//! Workspace scoping: a run is reachable only when its experiment is in the
//! active workspace. [`TrackingStore::resolve_run_row`] enforces this with a
//! semi-join and returns the Python-identical "Run with id=... not found"
//! error otherwise (mirrors `WorkspaceAwareSqlAlchemyStore._validate_run_accessible`).

use mlflow_error::MlflowError;
use uuid::Uuid;

use super::dbutil::{RowLike, Tx, Val};
use super::entities::{LifecycleStage, Param, Run, RunData, RunInfo, RunStatus, RunTag};
use super::experiments::{internal, now_millis, parse_experiment_id, ViewType};
use super::names::generate_random_name;
use super::uri_util::append_to_uri_path;
use super::{TrackingStore, ARTIFACTS_FOLDER_NAME, MLFLOW_RUN_NAME};
use crate::schema::runs::RUNS;

/// The physical row of `runs` we read for entity assembly.
pub(crate) struct RunRow {
    pub(crate) run_uuid: String,
    pub(crate) name: Option<String>,
    pub(crate) user_id: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) start_time: Option<i64>,
    pub(crate) end_time: Option<i64>,
    pub(crate) lifecycle_stage: String,
    pub(crate) artifact_uri: Option<String>,
    pub(crate) experiment_id: i64,
}

impl RunRow {
    pub(crate) fn from_row(r: &dyn RowLike) -> Result<Self, sqlx::Error> {
        Ok(RunRow {
            run_uuid: r.get_string("run_uuid")?,
            name: r.get_opt_string("name")?,
            user_id: r.get_opt_string("user_id")?,
            status: r.get_opt_string("status")?,
            start_time: r.get_opt_i64("start_time")?,
            end_time: r.get_opt_i64("end_time")?,
            lifecycle_stage: r.get_string("lifecycle_stage")?,
            artifact_uri: r.get_opt_string("artifact_uri")?,
            experiment_id: r.get_int("experiment_id")?,
        })
    }

    const SELECT_COLS: &'static str =
        "run_uuid, name, user_id, status, start_time, end_time, lifecycle_stage, \
         artifact_uri, experiment_id";
}

impl TrackingStore {
    /// `create_run`. The experiment must exist in the workspace and be ACTIVE.
    ///
    /// `tags` are the caller-supplied run tags. Run-name resolution mirrors
    /// Python exactly: an explicit `run_name` and an `mlflow.runName` tag must
    /// agree; the effective name is `run_name` else the tag else a random name;
    /// an `mlflow.runName` tag is synthesized if absent.
    pub async fn create_run(
        &self,
        workspace: &str,
        experiment_id: &str,
        user_id: Option<&str>,
        start_time: Option<i64>,
        run_name: Option<&str>,
        tags: &[(&str, &str)],
    ) -> Result<Run, MlflowError> {
        let exp_id = parse_experiment_id(experiment_id)?;
        let experiment = self
            .fetch_experiment(workspace, exp_id, ViewType::All)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "No Experiment with id={exp_id} exists"
                ))
            })?;
        if experiment.lifecycle_stage != LifecycleStage::ACTIVE {
            return Err(MlflowError::invalid_parameter_value(format!(
                "The experiment {} must be in the 'active' state. Current state is {}.",
                experiment.experiment_id, experiment.lifecycle_stage
            )));
        }

        let run_id = Uuid::new_v4().simple().to_string();
        let artifact_uri = append_to_uri_path(
            experiment.artifact_location.as_deref().unwrap_or(""),
            &[&run_id, ARTIFACTS_FOLDER_NAME],
        );

        // Resolve run name and sync the mlflow.runName tag (Python semantics).
        let mut tag_pairs: Vec<(String, String)> = tags
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let run_name_tag = tag_pairs
            .iter()
            .find(|(k, _)| k == MLFLOW_RUN_NAME)
            .map(|(_, v)| v.clone());
        let run_name_arg = run_name.filter(|s| !s.is_empty());
        if let (Some(arg), Some(tag)) = (run_name_arg, run_name_tag.as_deref()) {
            if arg != tag {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with \
                     different values (run_name='{arg}', mlflow.runName='{tag}')."
                )));
            }
        }
        let effective_name = run_name_arg
            .map(str::to_string)
            .or_else(|| run_name_tag.clone())
            .unwrap_or_else(generate_random_name);
        if run_name_tag.is_none() {
            tag_pairs.push((MLFLOW_RUN_NAME.to_string(), effective_name.clone()));
        }

        let start = start_time;
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        let insert_run = format!(
            "INSERT INTO {RUNS} \
             (run_uuid, name, source_type, source_name, entry_point_name, user_id, status, \
              start_time, end_time, deleted_time, source_version, lifecycle_stage, artifact_uri, \
              experiment_id) \
             VALUES ({}, {}, 'UNKNOWN', '', '', {}, {}, {}, NULL, NULL, '', {}, {}, {})",
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
            &insert_run,
            &[
                Val::Text(run_id.clone()),
                Val::Text(effective_name.clone()),
                Val::OptText(user_id.map(str::to_string)),
                Val::Text(RunStatus::RUNNING.to_string()),
                Val::OptInt(start),
                Val::Text(LifecycleStage::ACTIVE.to_string()),
                Val::Text(artifact_uri.clone()),
                Val::Int(experiment.experiment_id.parse::<i64>().unwrap_or(exp_id)),
            ],
        )
        .await
        .map_err(internal)?;

        for (k, v) in &tag_pairs {
            insert_tag_tx(&mut tx, dialect, &run_id, k, v).await?;
        }

        tx.commit().await.map_err(internal)?;

        self.get_run(workspace, &run_id).await
    }

    /// `get_run`: assemble the full [`Run`] entity (info + latest metrics,
    /// params, tags, dataset/model inputs, model outputs). Workspace-scoped.
    pub async fn get_run(&self, workspace: &str, run_id: &str) -> Result<Run, MlflowError> {
        let row = self.resolve_run_row(workspace, run_id).await?;
        let info = self.run_info_from_row(&row);
        let data = self.load_run_data(run_id).await?;
        let inputs = self.load_run_inputs(run_id).await?;
        let outputs = self.load_run_outputs(run_id).await?;
        Ok(Run {
            info,
            data,
            inputs,
            outputs,
        })
    }

    /// `update_run_info`: update status/end_time/run_name (syncing the
    /// `mlflow.runName` tag). Requires the run to be ACTIVE. Returns the updated
    /// [`RunInfo`].
    pub async fn update_run_info(
        &self,
        workspace: &str,
        run_id: &str,
        status: Option<&str>,
        end_time: Option<i64>,
        run_name: Option<&str>,
    ) -> Result<RunInfo, MlflowError> {
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;
        if let Some(s) = status {
            if !RunStatus::is_valid(s) {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid run status: {s}."
                )));
            }
        }
        let name = run_name.filter(|s| !s.is_empty());
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Build a dynamic UPDATE for the provided fields.
        let mut sets = Vec::new();
        let mut vals = Vec::new();
        let mut idx = 1usize;
        if let Some(s) = status {
            sets.push(format!("status = {}", ph(idx)));
            vals.push(Val::Text(s.to_string()));
            idx += 1;
        }
        if let Some(e) = end_time {
            sets.push(format!("end_time = {}", ph(idx)));
            vals.push(Val::Int(e));
            idx += 1;
        }
        if let Some(n) = name {
            sets.push(format!("name = {}", ph(idx)));
            vals.push(Val::Text(n.to_string()));
            idx += 1;
        }
        if !sets.is_empty() {
            let sql = format!(
                "UPDATE {RUNS} SET {} WHERE run_uuid = {}",
                sets.join(", "),
                ph(idx)
            );
            vals.push(Val::Text(run_id.to_string()));
            tx.exec(&sql, &vals).await.map_err(internal)?;
        }

        if let Some(n) = name {
            sync_run_name_tag(&mut tx, dialect, run_id, n).await?;
        }

        tx.commit().await.map_err(internal)?;

        let updated = self.resolve_run_row(workspace, run_id).await?;
        Ok(self.run_info_from_row(&updated))
    }

    /// `delete_run`: soft-delete (lifecycle DELETED + `deleted_time`).
    /// Idempotent. Not gated on active state (matches Python).
    pub async fn delete_run(&self, workspace: &str, run_id: &str) -> Result<(), MlflowError> {
        self.resolve_run_row(workspace, run_id).await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {RUNS} SET lifecycle_stage = 'deleted', deleted_time = {} WHERE run_uuid = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        self.db()
            .exec(&sql, &[Val::Int(now), Val::Text(run_id.to_string())])
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `restore_run`: lifecycle ACTIVE + clear `deleted_time`. Idempotent.
    pub async fn restore_run(&self, workspace: &str, run_id: &str) -> Result<(), MlflowError> {
        self.resolve_run_row(workspace, run_id).await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {RUNS} SET lifecycle_stage = 'active', deleted_time = NULL WHERE run_uuid = {}",
            dialect.placeholder(1)
        );
        self.db()
            .exec(&sql, &[Val::Text(run_id.to_string())])
            .await
            .map_err(internal)?;
        Ok(())
    }

    // ---- internal helpers ----

    /// Fetch the `runs` row, scoped to the workspace via a semi-join to
    /// `experiments`. Errors `RESOURCE_DOES_NOT_EXIST` "Run with id=... not
    /// found" when the run is missing or belongs to another workspace.
    pub(crate) async fn resolve_run_row(
        &self,
        workspace: &str,
        run_id: &str,
    ) -> Result<RunRow, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT {cols} FROM {RUNS} r \
             WHERE r.run_uuid = {} AND r.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
            cols = RunRow::SELECT_COLS,
        );
        self.db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(run_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                RunRow::from_row,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!("Run with id={run_id} not found"))
            })
    }

    pub(crate) fn run_info_from_row(&self, row: &RunRow) -> RunInfo {
        RunInfo {
            run_id: row.run_uuid.clone(),
            run_name: row.name.clone().unwrap_or_default(),
            experiment_id: row.experiment_id.to_string(),
            user_id: row.user_id.clone(),
            status: row
                .status
                .clone()
                .unwrap_or_else(|| RunStatus::RUNNING.to_string()),
            start_time: row.start_time,
            end_time: row.end_time,
            lifecycle_stage: row.lifecycle_stage.clone(),
            artifact_uri: row.artifact_uri.clone(),
        }
    }

    /// Load latest metrics, params, and tags for a run (RunData). If the run
    /// name is empty in `runs.name`, Python fills it from the `mlflow.runName`
    /// tag; we surface the tag here and the info assembly already reads
    /// `runs.name`, so callers may reconcile if needed.
    async fn load_run_data(&self, run_id: &str) -> Result<RunData, MlflowError> {
        let dialect = self.db().dialect();
        let ph1 = dialect.placeholder(1);

        let params = self
            .db()
            .fetch_all(
                &format!("SELECT key, value FROM params WHERE run_uuid = {ph1} ORDER BY key"),
                &[Val::Text(run_id.to_string())],
                |r| {
                    Ok(Param {
                        key: r.get_string("key")?,
                        value: r.get_string("value")?,
                    })
                },
            )
            .await
            .map_err(internal)?;

        let tags = self
            .db()
            .fetch_all(
                &format!("SELECT key, value FROM tags WHERE run_uuid = {ph1} ORDER BY key"),
                &[Val::Text(run_id.to_string())],
                |r| {
                    Ok(RunTag {
                        key: r.get_string("key")?,
                        value: r.get_opt_string("value")?.unwrap_or_default(),
                    })
                },
            )
            .await
            .map_err(internal)?;

        let metrics = self.load_latest_metrics(run_id).await?;

        Ok(RunData {
            metrics,
            params,
            tags,
        })
    }
}

/// `_check_run_is_active`.
pub(crate) fn check_run_active(row: &RunRow) -> Result<(), MlflowError> {
    if row.lifecycle_stage != LifecycleStage::ACTIVE {
        return Err(MlflowError::invalid_parameter_value(format!(
            "The run {} must be in the 'active' state. Current state is {}.",
            row.run_uuid, row.lifecycle_stage
        )));
    }
    Ok(())
}

/// Insert one tag row (used at run creation).
async fn insert_tag_tx(
    tx: &mut Tx<'_>,
    dialect: crate::dialect::Dialect,
    run_id: &str,
    key: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let sql = format!(
        "INSERT INTO tags (key, value, run_uuid) VALUES ({}, {}, {})",
        dialect.placeholder(1),
        dialect.placeholder(2),
        dialect.placeholder(3)
    );
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

/// Upsert the `mlflow.runName` tag to `value` (used by `update_run_info` and
/// `set_tag`). Mirrors Python's "create if missing, else update value".
pub(crate) async fn sync_run_name_tag(
    tx: &mut Tx<'_>,
    dialect: crate::dialect::Dialect,
    run_id: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: "tags",
        columns: &["key", "value", "run_uuid"],
        pk_columns: &["key", "run_uuid"],
        update_columns: &["value"],
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(MLFLOW_RUN_NAME.to_string()),
            Val::Text(value.to_string()),
            Val::Text(run_id.to_string()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}
