//! Experiments, runs, params, tags, datasets, inputs, and entity associations.
//!
//! Mirrors `SqlExperiment`, `SqlRun`, `SqlExperimentTag`, `SqlTag`, `SqlParam`,
//! `SqlDataset`, `SqlInput`, `SqlInputTag`, and `SqlEntityAssociation` in
//! `mlflow/store/tracking/dbmodels/models.py`.

use sqlx::FromRow;

pub const EXPERIMENTS: &str = "experiments";
pub const EXPERIMENT_TAGS: &str = "experiment_tags";
pub const RUNS: &str = "runs";
pub const PARAMS: &str = "params";
pub const TAGS: &str = "tags";
pub const DATASETS: &str = "datasets";
pub const INPUTS: &str = "inputs";
pub const INPUT_TAGS: &str = "input_tags";
pub const ENTITY_ASSOCIATIONS: &str = "entity_associations";

/// Row of the `experiments` table (`SqlExperiment`).
///
/// `experiment_id` is a DB `Integer` (auto-increment PK). `workspace` defaults
/// to `'default'` at the DB level (plan §3.17).
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Experiment {
    pub experiment_id: i64,
    pub name: String,
    pub workspace: String,
    pub artifact_location: Option<String>,
    pub lifecycle_stage: Option<String>,
    pub creation_time: Option<i64>,
    pub last_update_time: Option<i64>,
}

/// Row of the `runs` table (`SqlRun`). `run_uuid` is the PK.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Run {
    pub run_uuid: String,
    pub name: Option<String>,
    pub source_type: Option<String>,
    pub source_name: Option<String>,
    pub entry_point_name: Option<String>,
    pub user_id: Option<String>,
    pub status: Option<String>,
    pub start_time: Option<i64>,
    pub end_time: Option<i64>,
    pub deleted_time: Option<i64>,
    pub source_version: Option<String>,
    pub lifecycle_stage: Option<String>,
    pub artifact_uri: Option<String>,
    pub experiment_id: Option<i64>,
}

/// Row of the `experiment_tags` table (`SqlExperimentTag`). PK `(key, experiment_id)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct ExperimentTag {
    pub key: String,
    pub value: Option<String>,
    pub experiment_id: i64,
}

/// Row of the `tags` table (`SqlTag`). PK `(key, run_uuid)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Tag {
    pub key: String,
    pub value: Option<String>,
    pub run_uuid: String,
}

/// Row of the `params` table (`SqlParam`). PK `(key, run_uuid)`. `value` non-null.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Param {
    pub key: String,
    pub value: String,
    pub run_uuid: String,
}

/// Row of the `datasets` table (`SqlDataset`). PK `(experiment_id, name, digest)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Dataset {
    pub dataset_uuid: String,
    pub experiment_id: i64,
    pub name: String,
    pub digest: String,
    pub dataset_source_type: String,
    pub dataset_source: String,
    pub dataset_schema: Option<String>,
    pub dataset_profile: Option<String>,
}

/// Row of the `inputs` table (`SqlInput`).
///
/// PK `(source_type, source_id, destination_type, destination_id)`. `step`
/// non-null (server default `0`).
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Input {
    pub input_uuid: String,
    pub source_type: String,
    pub source_id: String,
    pub destination_type: String,
    pub destination_id: String,
    pub step: i64,
}

/// Row of the `input_tags` table (`SqlInputTag`). PK `(input_uuid, name)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct InputTag {
    pub input_uuid: String,
    pub name: String,
    pub value: String,
}

/// Row of the `entity_associations` table (`SqlEntityAssociation`).
///
/// PK `(source_type, source_id, destination_type, destination_id)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct EntityAssociation {
    pub association_id: String,
    pub source_type: String,
    pub source_id: String,
    pub destination_type: String,
    pub destination_id: String,
    pub created_time: Option<i64>,
}
