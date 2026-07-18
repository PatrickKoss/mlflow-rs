//! The tracking [`TrackingStore`]: experiments, runs, params, tags, and
//! metrics operations (plan T2.4 + T2.5), mirroring
//! `mlflow/store/tracking/sqlalchemy_store.py` semantics exactly.
//!
//! ## Workspace scoping (CRITICAL, plan §3.17)
//!
//! Every method takes an explicit `workspace: &str`. In single-tenant mode the
//! caller passes `"default"`; when workspaces are enabled the value comes from
//! the `X-MLFLOW-WORKSPACE` request header. Scoping is anchored on the
//! `experiments.workspace` column (mirrors `WorkspaceAwareSqlAlchemyStore`):
//!
//! * experiment queries filter `experiments.workspace = ?` directly;
//! * run / param / tag / metric queries reach a run only if its experiment is in
//!   the workspace (a semi-join `runs.experiment_id IN (SELECT experiment_id
//!   FROM experiments WHERE workspace = ?)`), exactly like the Python mixin's
//!   `_get_query(SqlRun)` join.
//!
//! This means a run lookup in the wrong workspace yields the same
//! `RESOURCE_DOES_NOT_EXIST` "Run with id=... not found" as a genuinely missing
//! run — matching `WorkspaceAwareSqlAlchemyStore._validate_run_accessible`.
//!
//! ## Entity model
//!
//! The store returns lightweight owned entities defined in [`entities`] rather
//! than the proto types — the HTTP layer (Phase 3) maps them to protos. This
//! keeps the store crate free of a proto dependency and mirrors MLflow's own
//! `Experiment` / `Run` / `Metric` entities.

mod assessments;
mod datasets;
mod dbutil;
mod entities;
mod evaluation_datasets;
mod experiments;
mod issues;
mod jobs;
mod label_schemas;
mod logged_models;
mod metrics;
mod metrics_bulk;
mod names;
mod names_data;
mod params_tags;
mod record_logged_model;
mod runs;
mod scorers;
mod search;
mod search_experiments;
mod spans;
mod trace_correlation;
mod traces;
mod traces_analytics;
mod traces_search;
mod uri_util;
mod validation;
mod workspaces;

pub use assessments::{AssessmentUpdate, FeedbackUpdate, NewAssessment};
pub use datasets::{DatasetInputSpec, MAX_DATASET_SUMMARIES_RESULTS};
pub use entities::{
    Assessment, AssessmentError, AssessmentSource, AssessmentValue, Dataset, DatasetInput,
    DatasetSummary, Experiment, ExperimentTag, InputTag, LifecycleStage, LoggedModelInput,
    LoggedModelOutput, Metric, MetricWithRunId, Param, Run, RunData, RunInfo, RunInputs,
    RunOutputs, RunStatus, RunTag, StoredSpan, TraceAssessment, TraceInfo, TraceState,
    TraceWithSpans, MLFLOW_ARTIFACT_LOCATION, SPANS_LOCATION_ARCHIVE_REPO,
    SPANS_LOCATION_ARTIFACT_REPO, SPANS_LOCATION_TRACKING_STORE, TRACE_TAG_SPANS_LOCATION,
};
pub use evaluation_datasets::{
    python_json_dumps, EvaluationDataset, EvaluationDatasetsPage, EvaluationRecord,
    EvaluationRecordsPage, UpsertEvaluationRecordsResult,
};
pub use experiments::{ViewType, WorkspaceArtifactRoot};
pub use issues::{Issue, IssueUpdate, IssuesPage};
pub use jobs::{Job, JobStatus, JobStore};
pub use label_schemas::{
    LabelSchema, LabelSchemaInput, LabelSchemaType, LabelSchemaUpdate, LabelSchemasPage,
    DEFAULT_LABEL_SCHEMA_INSTRUCTION, DEFAULT_LABEL_SCHEMA_NAME,
};
pub use logged_models::{
    logged_models_page_token, logged_models_token_offset, DatasetFilter, LoggedModel,
    LoggedModelKv, LoggedModelMetric, LoggedModelMetricInput, LoggedModelOrderByInput,
    LoggedModelStatus, LoggedModelsPage, SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT,
};
pub use metrics::{MetricInput, GET_METRIC_HISTORY_MAX_RESULTS};
pub use metrics_bulk::{MAX_RESULTS_PER_RUN, MAX_RUNS_GET_METRIC_HISTORY_BULK};
pub use scorers::{OnlineScoringConfig, ScorerVersion};
pub use search::{RunsPage, SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD};
pub use search_experiments::ExperimentsPage;
pub use spans::{SpanInput, SpanMetricInput, TraceTimeRange};
pub use traces::{StartTraceInput, MAX_TRACE_LINKS_PER_REQUEST};
pub use traces_analytics::{
    MetricAggregation, MetricDataPoint, MetricViewType, TraceFilterCorrelationResult,
    MAX_RESULTS_QUERY_TRACE_METRICS,
};
pub use traces_search::{TracesPage, SEARCH_TRACES_DEFAULT_MAX_RESULTS};
pub use workspaces::{
    verify_single_tenant_data, ResolvedTraceArchivalConfig, TraceArchivalConfig, Workspace,
    WorkspaceDeletionMode, WorkspaceNameValidator, WorkspaceStore, DEFAULT_WORKSPACE_NAME,
    WORKSPACES,
};

use crate::db::Db;

/// The reserved tag key that mirrors a run's name (`mlflow.runName`).
pub(crate) const MLFLOW_RUN_NAME: &str = "mlflow.runName";

/// Subdirectory appended to a run's artifact URI (`ARTIFACTS_FOLDER_NAME`).
pub(crate) const ARTIFACTS_FOLDER_NAME: &str = "artifacts";

/// The tracking store: experiments, runs, params, tags, and metrics.
///
/// Holds a [`Db`] pool and the resolved artifact-root URI (the server's
/// `--default-artifact-root`, already run through `resolve_uri_if_local` by the
/// caller — the store treats it as an opaque, already-resolved URI, exactly as
/// `SqlAlchemyStore.artifact_root_uri` is).
#[derive(Debug, Clone)]
pub struct TrackingStore {
    db: Db,
    artifact_root_uri: String,
}

impl TrackingStore {
    /// Create a store over an already-connected/verified [`Db`] and a resolved
    /// artifact-root URI.
    pub fn new(db: Db, artifact_root_uri: impl Into<String>) -> Self {
        Self {
            db,
            artifact_root_uri: artifact_root_uri.into(),
        }
    }

    /// The underlying database pool.
    pub fn db(&self) -> &Db {
        &self.db
    }

    /// The default artifact root URI.
    pub fn artifact_root_uri(&self) -> &str {
        &self.artifact_root_uri
    }
}
