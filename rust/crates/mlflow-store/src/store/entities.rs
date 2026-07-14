//! Owned entity types returned by [`crate::store::TrackingStore`].
//!
//! These mirror the corresponding `mlflow.entities.*` classes closely enough for
//! the HTTP layer to map them onto protos, without pulling a proto dependency
//! into the store crate. Field names and semantics follow `to_mlflow_entity`
//! in `mlflow/store/tracking/dbmodels/models.py`.

/// Lifecycle stage for experiments and runs (`LifecycleStage`).
pub mod lifecycle_stage {
    pub const ACTIVE: &str = "active";
    pub const DELETED: &str = "deleted";
}

/// Re-export as a namespaced marker type for ergonomic references.
pub struct LifecycleStage;

impl LifecycleStage {
    pub const ACTIVE: &'static str = lifecycle_stage::ACTIVE;
    pub const DELETED: &'static str = lifecycle_stage::DELETED;
}

/// Run status strings (`RunStatus`). MLflow persists the enum *name*.
pub struct RunStatus;

impl RunStatus {
    pub const RUNNING: &'static str = "RUNNING";
    pub const SCHEDULED: &'static str = "SCHEDULED";
    pub const FINISHED: &'static str = "FINISHED";
    pub const FAILED: &'static str = "FAILED";
    pub const KILLED: &'static str = "KILLED";

    /// The set of valid status strings accepted on writes.
    pub fn is_valid(s: &str) -> bool {
        matches!(
            s,
            Self::RUNNING | Self::SCHEDULED | Self::FINISHED | Self::FAILED | Self::KILLED
        )
    }
}

/// An experiment (`mlflow.entities.Experiment`). `experiment_id` is stringified
/// at the entity boundary, matching MLflow.
#[derive(Debug, Clone, PartialEq)]
pub struct Experiment {
    pub experiment_id: String,
    pub name: String,
    pub artifact_location: Option<String>,
    pub lifecycle_stage: String,
    pub creation_time: Option<i64>,
    pub last_update_time: Option<i64>,
    pub tags: Vec<ExperimentTag>,
}

/// An experiment tag.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentTag {
    pub key: String,
    pub value: Option<String>,
}

/// A run tag (`RunTag`).
#[derive(Debug, Clone, PartialEq)]
pub struct RunTag {
    pub key: String,
    pub value: String,
}

/// A run param (`Param`).
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub key: String,
    pub value: String,
}

/// A metric point (`Metric`). `value` is the sanitized value stored in the DB
/// (NaN → `f64::NAN` on read; ±Inf clamped to ±max f64), `step`/`timestamp` as
/// logged.
#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    pub key: String,
    pub value: f64,
    pub timestamp: i64,
    pub step: i64,
}

/// Run info (`RunInfo`).
#[derive(Debug, Clone, PartialEq)]
pub struct RunInfo {
    pub run_id: String,
    pub run_name: String,
    pub experiment_id: String,
    pub user_id: Option<String>,
    pub status: String,
    pub start_time: Option<i64>,
    pub end_time: Option<i64>,
    pub lifecycle_stage: String,
    pub artifact_uri: Option<String>,
}

/// Run data (`RunData`): latest metrics, params, tags.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunData {
    pub metrics: Vec<Metric>,
    pub params: Vec<Param>,
    pub tags: Vec<RunTag>,
}

/// A run (`mlflow.entities.Run`).
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub info: RunInfo,
    pub data: RunData,
    /// `run.inputs` (`RunInputs`): dataset inputs + model inputs.
    pub inputs: RunInputs,
    /// `run.outputs` (`RunOutputs`): model outputs.
    pub outputs: RunOutputs,
}

/// A dataset (`mlflow.entities.Dataset`). Field names mirror the entity, not the
/// DB column names: `source_type`/`source` map from `dataset_source_type`/
/// `dataset_source`, and `schema`/`profile` from `dataset_schema`/
/// `dataset_profile`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub name: String,
    pub digest: String,
    pub source_type: String,
    pub source: String,
    pub schema: Option<String>,
    pub profile: Option<String>,
}

/// An input tag (`mlflow.entities.InputTag`).
#[derive(Debug, Clone, PartialEq)]
pub struct InputTag {
    pub key: String,
    pub value: String,
}

/// A dataset input (`mlflow.entities.DatasetInput`): a dataset plus its tags.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetInput {
    pub dataset: Dataset,
    pub tags: Vec<InputTag>,
}

/// A model input reference (`mlflow.entities.LoggedModelInput`, proto
/// `ModelInput`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedModelInput {
    pub model_id: String,
}

/// A model output reference (`mlflow.entities.LoggedModelOutput`, proto
/// `ModelOutput`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedModelOutput {
    pub model_id: String,
    pub step: i64,
}

/// `run.inputs` (`mlflow.entities.RunInputs`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunInputs {
    pub dataset_inputs: Vec<DatasetInput>,
    pub model_inputs: Vec<LoggedModelInput>,
}

/// `run.outputs` (`mlflow.entities.RunOutputs`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunOutputs {
    pub model_outputs: Vec<LoggedModelOutput>,
}

/// A dataset summary (`SqlAlchemyStore._DatasetSummary`), returned by
/// `search_datasets`. `experiment_id` is stringified at the entity boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetSummary {
    pub experiment_id: String,
    pub name: String,
    pub digest: String,
    pub context: Option<String>,
}

/// A metric point paired with the run it belongs to
/// (`SqlAlchemyStore.MetricWithRunId`), returned by the bulk metric-history
/// APIs.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricWithRunId {
    pub run_id: String,
    pub metric: Metric,
}

/// The source of an assessment (`mlflow.entities.AssessmentSource`).
/// `source_id` defaults to `"default"` at the entity layer in Python; the
/// store treats it as an already-resolved, optional string.
#[derive(Debug, Clone, PartialEq)]
pub struct AssessmentSource {
    pub source_type: String,
    pub source_id: Option<String>,
}

/// `mlflow.entities.AssessmentError` (feedback-only). Stored as JSON in the
/// `assessments.error` column (`AssessmentError.to_dictionary`); field order
/// here matches Python's dict for readability, though JSON object key order
/// is not semantically significant.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssessmentError {
    pub error_code: String,
    pub error_message: Option<String>,
    pub stack_trace: Option<String>,
}

/// The discriminated payload of an assessment: exactly one of expectation,
/// feedback, or issue (mirrors `Assessment.__post_init__`'s "exactly one of"
/// invariant and `SqlAssessments.assessment_type`).
#[derive(Debug, Clone, PartialEq)]
pub enum AssessmentValue {
    /// `mlflow.entities.ExpectationValue`. `value` is the raw JSON
    /// representation of `ExpectationValue.value` (`json.dumps(value)` in
    /// `SqlAssessments.from_mlflow_entity`).
    Expectation { value_json: String },
    /// `mlflow.entities.FeedbackValue`. `value_json` is
    /// `json.dumps(FeedbackValue.value)`; `error` is the optional
    /// `AssessmentError`.
    Feedback {
        value_json: String,
        error: Option<AssessmentError>,
    },
    /// `mlflow.entities.IssueReferenceValue`. Stored as
    /// `json.dumps({"issue_name": ...})`.
    Issue { issue_name: String },
}

/// An assessment (`mlflow.entities.Assessment`/`Feedback`/`Expectation`),
/// mirroring `SqlAssessments.to_mlflow_entity`.
#[derive(Debug, Clone, PartialEq)]
pub struct Assessment {
    pub assessment_id: String,
    pub trace_id: String,
    pub name: String,
    pub value: AssessmentValue,
    pub source: AssessmentSource,
    pub run_id: Option<String>,
    pub span_id: Option<String>,
    pub rationale: Option<String>,
    pub metadata: Option<std::collections::BTreeMap<String, String>>,
    pub create_time_ms: i64,
    pub last_update_time_ms: i64,
    /// The assessment_id this one overrides/supersedes, if any.
    pub overrides: Option<String>,
    /// Whether the assessment is still in effect (false once overridden).
    pub valid: bool,
}
