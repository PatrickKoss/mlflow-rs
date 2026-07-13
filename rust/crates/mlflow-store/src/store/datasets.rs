//! Datasets, inputs, and outputs (plan T2.8), mirroring `log_inputs`,
//! `_log_inputs_impl`, `log_outputs`, `_search_datasets`, `_get_run_inputs`,
//! `_get_model_inputs`, and `_get_model_outputs` in
//! `mlflow/store/tracking/sqlalchemy_store.py`.
//!
//! ## The `inputs` table is a typed edge list
//!
//! All of datasets→run, run→model-input, and run→model-output are edges in the
//! single `inputs` table (`SqlInput`), distinguished by the
//! `(source_type, destination_type)` pair — NOT the `entity_associations` table:
//!
//! | edge | source_type | source_id | destination_type | destination_id | step |
//! |---|---|---|---|---|---|
//! | dataset → run   | `DATASET`    | dataset_uuid | `RUN`          | run_id   | 0 (default) |
//! | run → model in  | `RUN_INPUT`  | run_id       | `MODEL_INPUT`  | model_id | 0 (default) |
//! | run → model out | `RUN_OUTPUT` | run_id       | `MODEL_OUTPUT` | model_id | model.step  |
//!
//! The `inputs` PK is `(source_type, source_id, destination_type, destination_id)`,
//! so a repeated edge dedups on that tuple.
//!
//! ## Dataset dedup (`_log_inputs_impl`)
//!
//! Datasets dedup by `(experiment_id, name, digest)` — the `datasets` PK. Before
//! inserting, Python looks up existing rows for the run's experiment matching any
//! of the incoming `(name, digest)` pairs and reuses their `dataset_uuid`. The
//! incoming `dataset_inputs` list is itself deduped by `(name, digest)`, keeping
//! the first occurrence. All UUIDs are `uuid.uuid4().hex` (32-char lowercase hex,
//! no dashes) — `Uuid::new_v4().simple()` in Rust.

use std::collections::HashMap;

use mlflow_error::MlflowError;
use uuid::Uuid;

use super::dbutil::{RowLike, Tx, Val};
use super::entities::{
    Dataset, DatasetInput, DatasetSummary, InputTag, LoggedModelInput, LoggedModelOutput,
    RunInputs, RunOutputs,
};
use super::experiments::{internal, parse_experiment_id};
use super::runs::check_run_active;
use super::validation;
use super::TrackingStore;
use crate::dialect::Dialect;

/// `mlflow.data.context` — the input tag that carries a dataset's context in
/// `search_datasets` (`MLFLOW_DATASET_CONTEXT`).
const MLFLOW_DATASET_CONTEXT: &str = "mlflow.data.context";

/// `SqlAlchemyStore._search_datasets`: cap on summaries returned
/// (`MAX_DATASET_SUMMARIES_RESULTS`).
pub const MAX_DATASET_SUMMARIES_RESULTS: usize = 1000;

/// A dataset to log as an input to a run (`DatasetInput` argument to
/// `log_inputs`).
#[derive(Debug, Clone)]
pub struct DatasetInputSpec {
    pub name: String,
    pub digest: String,
    pub source_type: String,
    pub source: String,
    pub schema: Option<String>,
    pub profile: Option<String>,
    pub tags: Vec<(String, String)>,
}

fn new_uuid() -> String {
    Uuid::new_v4().simple().to_string()
}

impl TrackingStore {
    /// `log_inputs`: attach datasets (and model-input references) to a run.
    /// Workspace-scoped; requires the run to be ACTIVE. Mirrors
    /// `_log_inputs_impl` dedup semantics exactly.
    pub async fn log_inputs(
        &self,
        workspace: &str,
        run_id: &str,
        datasets: &[DatasetInputSpec],
        model_inputs: &[&str],
    ) -> Result<(), MlflowError> {
        // Validate datasets/tags (mirrors `_validate_dataset_inputs`).
        for d in datasets {
            validation::validate_dataset(
                &d.name,
                &d.digest,
                &d.source,
                d.schema.as_deref(),
                d.profile.as_deref(),
            )?;
            for (k, v) in &d.tags {
                validation::validate_input_tag(k, v)?;
            }
        }

        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;
        let experiment_id = row.experiment_id;

        // Dedup incoming dataset inputs by (name, digest), first occurrence wins.
        let mut seen: Vec<(String, String)> = Vec::new();
        let mut deduped: Vec<&DatasetInputSpec> = Vec::new();
        for d in datasets {
            let key = (d.name.clone(), d.digest.clone());
            if !seen.contains(&key) {
                seen.push(key);
                deduped.push(d);
            }
        }

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Resolve existing dataset_uuids for (experiment_id, name, digest).
        let mut dataset_uuids: HashMap<(String, String), String> = HashMap::new();
        for d in &deduped {
            if let Some(uuid) = self
                .existing_dataset_uuid(&mut tx, dialect, experiment_id, &d.name, &d.digest)
                .await?
            {
                dataset_uuids.insert((d.name.clone(), d.digest.clone()), uuid);
            }
        }

        // Insert new datasets.
        for d in &deduped {
            let key = (d.name.clone(), d.digest.clone());
            if let std::collections::hash_map::Entry::Vacant(slot) = dataset_uuids.entry(key) {
                let ds_uuid = new_uuid();
                slot.insert(ds_uuid.clone());
                insert_dataset(&mut tx, dialect, &ds_uuid, experiment_id, d).await?;
            }
        }

        // Resolve existing dataset→run input edges (source_type=DATASET,
        // destination=RUN/run_id) so we don't re-insert.
        let mut existing_edges: Vec<String> = Vec::new();
        for ds_uuid in dataset_uuids.values() {
            if self
                .dataset_input_exists(&mut tx, dialect, ds_uuid, run_id)
                .await?
            {
                existing_edges.push(ds_uuid.clone());
            }
        }

        // Insert new input edges + their input_tags.
        for d in &deduped {
            let ds_uuid = dataset_uuids
                .get(&(d.name.clone(), d.digest.clone()))
                .expect("dataset uuid resolved above");
            if existing_edges.contains(ds_uuid) {
                continue;
            }
            let input_uuid = new_uuid();
            insert_input(
                &mut tx,
                dialect,
                &input_uuid,
                "DATASET",
                ds_uuid,
                "RUN",
                run_id,
                0,
            )
            .await?;
            for (k, v) in &d.tags {
                insert_input_tag(&mut tx, dialect, &input_uuid, k, v).await?;
            }
        }

        // Model inputs (run → model-input edges) use merge/upsert semantics.
        for model_id in model_inputs {
            let input_uuid = new_uuid();
            upsert_input(
                &mut tx,
                dialect,
                &input_uuid,
                "RUN_INPUT",
                run_id,
                "MODEL_INPUT",
                model_id,
                0,
            )
            .await?;
        }

        tx.commit().await.map_err(internal)
    }

    /// `log_outputs`: record run → model-output edges. Workspace-scoped;
    /// requires the run to be ACTIVE. Each output carries its `step`.
    pub async fn log_outputs(
        &self,
        workspace: &str,
        run_id: &str,
        models: &[LoggedModelOutput],
    ) -> Result<(), MlflowError> {
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for model in models {
            let input_uuid = new_uuid();
            insert_input(
                &mut tx,
                dialect,
                &input_uuid,
                "RUN_OUTPUT",
                run_id,
                "MODEL_OUTPUT",
                &model.model_id,
                model.step,
            )
            .await?;
        }
        tx.commit().await.map_err(internal)
    }

    /// `_search_datasets`: return dataset summaries for the given experiments,
    /// scoped to `workspace`. DISTINCT over `(experiment_id, name, digest,
    /// context)`; context is the `mlflow.data.context` input tag (left-joined,
    /// so datasets without it appear with `context = None`). Capped at
    /// [`MAX_DATASET_SUMMARIES_RESULTS`].
    pub async fn search_datasets(
        &self,
        workspace: &str,
        experiment_ids: &[&str],
    ) -> Result<Vec<DatasetSummary>, MlflowError> {
        let mut ids: Vec<i64> = Vec::with_capacity(experiment_ids.len());
        for e in experiment_ids {
            ids.push(parse_experiment_id(e)?);
        }
        // Scope to the workspace (mirrors `_filter_experiment_ids`): keep only
        // experiment ids that live in this workspace.
        let accessible = self.filter_experiment_ids(workspace, &ids).await?;
        if accessible.is_empty() {
            return Ok(Vec::new());
        }

        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut vals: Vec<Val> = vec![Val::Text(MLFLOW_DATASET_CONTEXT.to_string())];
        let placeholders: Vec<String> = accessible
            .iter()
            .enumerate()
            .map(|(i, id)| {
                vals.push(Val::Int(*id));
                ph(i + 2)
            })
            .collect();

        let sql = format!(
            "SELECT DISTINCT d.experiment_id AS experiment_id, d.name AS name, \
             d.digest AS digest, it.value AS context \
             FROM datasets d \
             JOIN inputs i ON i.source_id = d.dataset_uuid \
             LEFT JOIN input_tags it ON it.input_uuid = i.input_uuid AND it.name = {} \
             WHERE d.experiment_id IN ({}) \
             LIMIT {}",
            ph(1),
            placeholders.join(", "),
            MAX_DATASET_SUMMARIES_RESULTS,
        );

        self.db()
            .fetch_all(&sql, &vals, |r| {
                Ok(DatasetSummary {
                    experiment_id: r.get_int("experiment_id")?.to_string(),
                    name: r.get_string("name")?,
                    digest: r.get_string("digest")?,
                    context: r.get_opt_string("context")?,
                })
            })
            .await
            .map_err(internal)
    }

    /// Load `run.inputs` (`RunInputs`): dataset inputs (with tags) + model
    /// inputs. Used by `get_run`.
    pub(crate) async fn load_run_inputs(&self, run_id: &str) -> Result<RunInputs, MlflowError> {
        let dataset_inputs = self.load_dataset_inputs(run_id).await?;
        let model_inputs = self.load_model_inputs(run_id).await?;
        Ok(RunInputs {
            dataset_inputs,
            model_inputs,
        })
    }

    /// Load `run.outputs` (`RunOutputs`): model outputs. Used by `get_run`.
    pub(crate) async fn load_run_outputs(&self, run_id: &str) -> Result<RunOutputs, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT destination_id, step FROM inputs \
             WHERE source_type = 'RUN_OUTPUT' AND source_id = {} \
             AND destination_type = 'MODEL_OUTPUT'",
            dialect.placeholder(1)
        );
        let model_outputs = self
            .db()
            .fetch_all(&sql, &[Val::Text(run_id.to_string())], |r| {
                Ok(LoggedModelOutput {
                    model_id: r.get_string("destination_id")?,
                    step: r.get_i64("step")?,
                })
            })
            .await
            .map_err(internal)?;
        Ok(RunOutputs { model_outputs })
    }

    /// `_get_run_inputs` for a single run: dataset inputs grouped by dataset,
    /// each with its input tags. Order mirrors Python's first-seen grouping.
    async fn load_dataset_inputs(&self, run_id: &str) -> Result<Vec<DatasetInput>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT d.dataset_uuid AS dataset_uuid, d.name AS name, d.digest AS digest, \
             d.dataset_source_type AS dataset_source_type, d.dataset_source AS dataset_source, \
             d.dataset_schema AS dataset_schema, d.dataset_profile AS dataset_profile, \
             it.name AS tag_name, it.value AS tag_value \
             FROM inputs i \
             JOIN datasets d ON i.source_id = d.dataset_uuid \
             LEFT JOIN input_tags it ON it.input_uuid = i.input_uuid \
             WHERE i.destination_type = 'RUN' AND i.destination_id = {}",
            dialect.placeholder(1)
        );
        let rows = self
            .db()
            .fetch_all(
                &sql,
                &[Val::Text(run_id.to_string())],
                DatasetInputRow::from_row,
            )
            .await
            .map_err(internal)?;

        // Group by dataset_uuid, first sighting builds the DatasetInput; each
        // non-null tag row appends. Preserve first-seen order.
        let mut order: Vec<String> = Vec::new();
        let mut by_uuid: HashMap<String, DatasetInput> = HashMap::new();
        for row in rows {
            let di = by_uuid.entry(row.dataset_uuid.clone()).or_insert_with(|| {
                order.push(row.dataset_uuid.clone());
                DatasetInput {
                    dataset: Dataset {
                        name: row.name.clone(),
                        digest: row.digest.clone(),
                        source_type: row.dataset_source_type.clone(),
                        source: row.dataset_source.clone(),
                        schema: row.dataset_schema.clone(),
                        profile: row.dataset_profile.clone(),
                    },
                    tags: Vec::new(),
                }
            });
            if let (Some(k), Some(v)) = (row.tag_name, row.tag_value) {
                di.tags.push(InputTag { key: k, value: v });
            }
        }
        Ok(order
            .into_iter()
            .map(|u| by_uuid.remove(&u).unwrap())
            .collect())
    }

    /// `_get_model_inputs` for a run.
    async fn load_model_inputs(&self, run_id: &str) -> Result<Vec<LoggedModelInput>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT destination_id FROM inputs \
             WHERE source_type = 'RUN_INPUT' AND source_id = {} \
             AND destination_type = 'MODEL_INPUT'",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Text(run_id.to_string())], |r| {
                Ok(LoggedModelInput {
                    model_id: r.get_string("destination_id")?,
                })
            })
            .await
            .map_err(internal)
    }

    // ---- internal helpers ----

    async fn existing_dataset_uuid(
        &self,
        tx: &mut Tx<'_>,
        dialect: Dialect,
        experiment_id: i64,
        name: &str,
        digest: &str,
    ) -> Result<Option<String>, MlflowError> {
        let sql = format!(
            "SELECT dataset_uuid FROM datasets \
             WHERE experiment_id = {} AND name = {} AND digest = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        let rows = tx
            .fetch_all(
                &sql,
                &[
                    Val::Int(experiment_id),
                    Val::Text(name.to_string()),
                    Val::Text(digest.to_string()),
                ],
                |r| r.get_string("dataset_uuid"),
            )
            .await
            .map_err(internal)?;
        Ok(rows.into_iter().next())
    }

    async fn dataset_input_exists(
        &self,
        tx: &mut Tx<'_>,
        dialect: Dialect,
        dataset_uuid: &str,
        run_id: &str,
    ) -> Result<bool, MlflowError> {
        let sql = format!(
            "SELECT input_uuid FROM inputs \
             WHERE source_type = 'DATASET' AND source_id = {} \
             AND destination_type = 'RUN' AND destination_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let rows = tx
            .fetch_all(
                &sql,
                &[
                    Val::Text(dataset_uuid.to_string()),
                    Val::Text(run_id.to_string()),
                ],
                |r| r.get_string("input_uuid"),
            )
            .await
            .map_err(internal)?;
        Ok(!rows.is_empty())
    }

    /// `_filter_experiment_ids`: keep only experiment ids that exist in the
    /// workspace. Returns them in the input order.
    async fn filter_experiment_ids(
        &self,
        workspace: &str,
        ids: &[i64],
    ) -> Result<Vec<i64>, MlflowError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut vals: Vec<Val> = Vec::with_capacity(ids.len() + 1);
        let placeholders: Vec<String> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                vals.push(Val::Int(*id));
                ph(i + 1)
            })
            .collect();
        vals.push(Val::Text(workspace.to_string()));
        let sql = format!(
            "SELECT experiment_id FROM experiments \
             WHERE experiment_id IN ({}) AND workspace = {}",
            placeholders.join(", "),
            ph(ids.len() + 1)
        );
        let found: Vec<i64> = self
            .db()
            .fetch_all(&sql, &vals, |r| r.get_int("experiment_id"))
            .await
            .map_err(internal)?;
        // Preserve the caller's order (Python returns DB order, but callers only
        // use membership; keeping input order is deterministic and stable).
        Ok(ids
            .iter()
            .copied()
            .filter(|id| found.contains(id))
            .collect())
    }
}

/// A joined dataset-input row (dataset columns + optional tag columns).
struct DatasetInputRow {
    dataset_uuid: String,
    name: String,
    digest: String,
    dataset_source_type: String,
    dataset_source: String,
    dataset_schema: Option<String>,
    dataset_profile: Option<String>,
    tag_name: Option<String>,
    tag_value: Option<String>,
}

impl DatasetInputRow {
    fn from_row(r: &dyn RowLike) -> Result<Self, sqlx::Error> {
        Ok(DatasetInputRow {
            dataset_uuid: r.get_string("dataset_uuid")?,
            name: r.get_string("name")?,
            digest: r.get_string("digest")?,
            dataset_source_type: r.get_string("dataset_source_type")?,
            dataset_source: r.get_string("dataset_source")?,
            dataset_schema: r.get_opt_string("dataset_schema")?,
            dataset_profile: r.get_opt_string("dataset_profile")?,
            tag_name: r.get_opt_string("tag_name")?,
            tag_value: r.get_opt_string("tag_value")?,
        })
    }
}

async fn insert_dataset(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    dataset_uuid: &str,
    experiment_id: i64,
    d: &DatasetInputSpec,
) -> Result<(), MlflowError> {
    let ph = |i| dialect.placeholder(i);
    let sql = format!(
        "INSERT INTO datasets \
         (dataset_uuid, experiment_id, name, digest, dataset_source_type, dataset_source, \
          dataset_schema, dataset_profile) \
         VALUES ({}, {}, {}, {}, {}, {}, {}, {})",
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
        &sql,
        &[
            Val::Text(dataset_uuid.to_string()),
            Val::Int(experiment_id),
            Val::Text(d.name.clone()),
            Val::Text(d.digest.clone()),
            Val::Text(d.source_type.clone()),
            Val::Text(d.source.clone()),
            Val::OptText(d.schema.clone()),
            Val::OptText(d.profile.clone()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn insert_input(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    input_uuid: &str,
    source_type: &str,
    source_id: &str,
    destination_type: &str,
    destination_id: &str,
    step: i64,
) -> Result<(), MlflowError> {
    let ph = |i| dialect.placeholder(i);
    let sql = format!(
        "INSERT INTO inputs \
         (input_uuid, source_type, source_id, destination_type, destination_id, step) \
         VALUES ({}, {}, {}, {}, {}, {})",
        ph(1),
        ph(2),
        ph(3),
        ph(4),
        ph(5),
        ph(6),
    );
    tx.exec(
        &sql,
        &[
            Val::Text(input_uuid.to_string()),
            Val::Text(source_type.to_string()),
            Val::Text(source_id.to_string()),
            Val::Text(destination_type.to_string()),
            Val::Text(destination_id.to_string()),
            Val::Int(step),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

/// Upsert an `inputs` edge on the 4-col PK (mirrors Python's `session.merge`
/// for model inputs). A repeated edge is a no-op.
#[allow(clippy::too_many_arguments)]
async fn upsert_input(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    input_uuid: &str,
    source_type: &str,
    source_id: &str,
    destination_type: &str,
    destination_id: &str,
    step: i64,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: "inputs",
        columns: &[
            "input_uuid",
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
            "step",
        ],
        pk_columns: &[
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
        ],
        update_columns: &["input_uuid", "step"],
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(input_uuid.to_string()),
            Val::Text(source_type.to_string()),
            Val::Text(source_id.to_string()),
            Val::Text(destination_type.to_string()),
            Val::Text(destination_id.to_string()),
            Val::Int(step),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

async fn insert_input_tag(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    input_uuid: &str,
    name: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let ph = |i| dialect.placeholder(i);
    let sql = format!(
        "INSERT INTO input_tags (input_uuid, name, value) VALUES ({}, {}, {})",
        ph(1),
        ph(2),
        ph(3)
    );
    tx.exec(
        &sql,
        &[
            Val::Text(input_uuid.to_string()),
            Val::Text(name.to_string()),
            Val::Text(value.to_string()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}
