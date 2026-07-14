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

// ===========================================================================
// Tracing V3 entities (plan §3.6, T2.10/T2.11)
// ===========================================================================

/// The reserved trace-tag key that stores a trace's artifact location
/// (`MLFLOW_ARTIFACT_LOCATION`, `mlflow/utils/mlflow_tags.py`).
pub const MLFLOW_ARTIFACT_LOCATION: &str = "mlflow.artifactLocation";

/// Trace-metadata key that records the source run (`TraceMetadataKey.SOURCE_RUN`).
pub const TRACE_METADATA_SOURCE_RUN: &str = "mlflow.sourceRun";

/// Trace-metadata key set by `start_trace` to signal that authoritative
/// trace-level values were written (`TraceMetadataKey.TRACE_INFO_FINALIZED`).
pub const TRACE_METADATA_INFO_FINALIZED: &str = "mlflow.trace.infoFinalized";

/// Trace-tag key that records where span payloads live
/// (`TraceTagKey.SPANS_LOCATION`).
pub const TRACE_TAG_SPANS_LOCATION: &str = "mlflow.trace.spansLocation";

/// The `SpansLocation.TRACKING_STORE` value written by `log_spans`.
pub const SPANS_LOCATION_TRACKING_STORE: &str = "TRACKING_STORE";

/// `TraceState` string values (persisted verbatim in `trace_info.status`).
pub struct TraceState;

impl TraceState {
    pub const STATE_UNSPECIFIED: &'static str = "STATE_UNSPECIFIED";
    pub const IN_PROGRESS: &'static str = "IN_PROGRESS";
    pub const OK: &'static str = "OK";
    pub const ERROR: &'static str = "ERROR";
}

/// A trace assessment (`mlflow.entities.Assessment`), carried on a
/// [`TraceInfo`]. Fields mirror `SqlAssessments.to_mlflow_entity`; JSON-typed
/// payloads (`value`, `error`, `rationale`, `metadata`) stay as raw strings so
/// the store crate need not depend on the assessment proto (Phase T2.12 owns
/// full assessment semantics).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceAssessment {
    pub assessment_id: String,
    pub trace_id: String,
    pub name: String,
    pub assessment_type: String,
    pub value: String,
    pub error: Option<String>,
    pub created_timestamp: i64,
    pub last_updated_timestamp: i64,
    pub source_type: String,
    pub source_id: Option<String>,
    pub run_id: Option<String>,
    pub span_id: Option<String>,
    pub rationale: Option<String>,
    pub overrides: Option<String>,
    pub valid: bool,
    pub metadata: Option<String>,
}

/// Trace info (`mlflow.entities.TraceInfo`, V3), as returned by the store.
///
/// `experiment_id` is the stringified experiment id (matching
/// `TraceLocation.from_experiment_id`). `tags` and `trace_metadata` are ordered
/// by key (Python builds them from ORM relationships; we sort for determinism).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceInfo {
    pub trace_id: String,
    pub experiment_id: String,
    pub request_time: i64,
    pub execution_duration: Option<i64>,
    pub state: String,
    pub client_request_id: Option<String>,
    pub request_preview: Option<String>,
    pub response_preview: Option<String>,
    pub tags: Vec<(String, Option<String>)>,
    pub trace_metadata: Vec<(String, Option<String>)>,
    pub assessments: Vec<TraceAssessment>,
}

impl TraceInfo {
    /// Look up a tag value by key.
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.as_deref())
    }

    /// Look up a metadata value by key.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.trace_metadata
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.as_deref())
    }
}

/// A stored span (`mlflow.entities` span row → `SqlSpan`).
///
/// `duration_ns` is the read-only generated column (NULL for in-progress
/// spans). `content` is the JSON payload; an empty string means the payload was
/// cleared (archival), which reads treat as "no span" (plan T2.11).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredSpan {
    pub trace_id: String,
    pub experiment_id: i64,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: Option<String>,
    pub span_type: Option<String>,
    pub status: String,
    pub start_time_unix_nano: i64,
    pub end_time_unix_nano: Option<i64>,
    pub duration_ns: Option<i64>,
    pub content: String,
    pub dimension_attributes: Option<String>,
}

/// A full trace: its [`TraceInfo`] plus the DB-backed spans (payloads only, no
/// OTel reconstruction — that is a serialization concern for the HTTP layer).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceWithSpans {
    pub info: TraceInfo,
    pub spans: Vec<StoredSpan>,
}
