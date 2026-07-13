//! Logged-model tables: `logged_models`, `logged_model_params`,
//! `logged_model_tags`, `logged_model_metrics`.
//!
//! Mirrors `SqlLoggedModel`, `SqlLoggedModelParam`, `SqlLoggedModelTag`, and
//! `SqlLoggedModelMetric` (`mlflow/store/tracking/dbmodels/models.py`).

use sqlx::FromRow;

pub const LOGGED_MODELS: &str = "logged_models";
pub const LOGGED_MODEL_PARAMS: &str = "logged_model_params";
pub const LOGGED_MODEL_TAGS: &str = "logged_model_tags";
pub const LOGGED_MODEL_METRICS: &str = "logged_model_metrics";

/// Row of the `logged_models` table (`SqlLoggedModel`). PK `model_id`.
///
/// `status` is a DB `Integer` (`LoggedModelStatus` int code).
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct LoggedModel {
    pub model_id: String,
    pub experiment_id: i64,
    pub name: String,
    pub artifact_location: String,
    pub creation_timestamp_ms: i64,
    pub last_updated_timestamp_ms: i64,
    pub status: i64,
    pub lifecycle_stage: Option<String>,
    pub model_type: Option<String>,
    pub source_run_id: Option<String>,
    pub status_message: Option<String>,
}

/// Row of the `logged_model_params` table (`SqlLoggedModelParam`).
///
/// PK `(model_id, param_key)`. `param_value` is `Text`, non-null.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct LoggedModelParam {
    pub model_id: String,
    pub experiment_id: i64,
    pub param_key: String,
    pub param_value: String,
}

/// Row of the `logged_model_tags` table (`SqlLoggedModelTag`).
///
/// PK `(model_id, tag_key)`. `tag_value` is `Text`, non-null.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct LoggedModelTag {
    pub model_id: String,
    pub experiment_id: i64,
    pub tag_key: String,
    pub tag_value: String,
}

/// Row of the `logged_model_metrics` table (`SqlLoggedModelMetric`).
///
/// 5-column PK `(model_id, metric_name, metric_timestamp_ms, metric_step,
/// run_id)` (plan §5.1). `metric_value` is nullable.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct LoggedModelMetric {
    pub model_id: String,
    pub metric_name: String,
    pub metric_timestamp_ms: i64,
    pub metric_step: i64,
    pub metric_value: Option<f64>,
    pub experiment_id: i64,
    pub run_id: String,
    pub dataset_uuid: Option<String>,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}
