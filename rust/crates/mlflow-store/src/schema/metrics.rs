//! Metric tables: `metrics` and `latest_metrics`.
//!
//! Mirrors `SqlMetric` and `SqlLatestMetric`
//! (`mlflow/store/tracking/dbmodels/models.py`).

use sqlx::FromRow;

pub const METRICS: &str = "metrics";
pub const LATEST_METRICS: &str = "latest_metrics";

/// Row of the `metrics` table (`SqlMetric`).
///
/// Wide 6-column PK `(key, timestamp, step, run_uuid, value, is_nan)` acts as a
/// dedup key (plan §5.1). `value` is non-null; NaN is represented by
/// `is_nan = true` with `value` set to `0.0`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Metric {
    pub key: String,
    pub value: f64,
    pub timestamp: Option<i64>,
    pub step: i64,
    pub is_nan: bool,
    pub run_uuid: String,
}

/// Row of the `latest_metrics` table (`SqlLatestMetric`). PK `(key, run_uuid)`.
///
/// Maintained via an atomic upsert that keeps the row with the greatest
/// `(step, timestamp, value)` (plan §5.2 Q5).
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct LatestMetric {
    pub key: String,
    pub value: f64,
    pub timestamp: Option<i64>,
    pub step: i64,
    pub is_nan: bool,
    pub run_uuid: String,
}
