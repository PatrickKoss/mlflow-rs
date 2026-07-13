//! Rust structs mirroring the MLflow backend-store tables (plan §5.1).
//!
//! Source of truth: `mlflow/store/tracking/dbmodels/models.py`. Each struct
//! mirrors the SQLAlchemy column names, types, and nullability exactly, using
//! the type mapping:
//!
//! | SQLAlchemy | Rust |
//! |---|---|
//! | `Integer` (experiment_id, status) | `i64` |
//! | `BigInteger` | `i64` |
//! | `String(N)` / `Text` / `UnicodeText` / `LONGTEXT` | `String` |
//! | `Float(53)` | `f64` |
//! | `Boolean` | `bool` |
//! | `JSON` (`dimension_attributes`) | `Option<String>` (raw JSON text) |
//! | nullable column | wrapped in `Option<_>` |
//!
//! These are plain data structs (`sqlx::FromRow` where the store will read
//! whole rows) plus per-table column-name constants — no ORM. Relationships,
//! cascades, and `to_mlflow_entity` conversions live in the store layer
//! (T2.4+), not here.
//!
//! Note on `experiment_id`: SQLAlchemy models it as `Integer` and MLflow
//! entities carry it as a *string*; the store converts at the entity boundary.
//! Here it is `i64` to match the physical column.

pub mod logged_models;
pub mod metrics;
pub mod runs;
pub mod traces;

/// All tracking/tracing table names owned by the Rust store (plan §5.1).
///
/// Kept as a single source for startup checks and diagnostics.
pub const TRACKING_TABLES: &[&str] = &[
    runs::EXPERIMENTS,
    runs::EXPERIMENT_TAGS,
    runs::RUNS,
    runs::PARAMS,
    runs::TAGS,
    metrics::METRICS,
    metrics::LATEST_METRICS,
    runs::DATASETS,
    runs::INPUTS,
    runs::INPUT_TAGS,
    logged_models::LOGGED_MODELS,
    logged_models::LOGGED_MODEL_PARAMS,
    logged_models::LOGGED_MODEL_TAGS,
    logged_models::LOGGED_MODEL_METRICS,
    traces::TRACE_INFO,
    traces::TRACE_TAGS,
    traces::TRACE_REQUEST_METADATA,
    traces::TRACE_METRICS,
    traces::SPANS,
    traces::SPAN_METRICS,
    traces::ASSESSMENTS,
    runs::ENTITY_ASSOCIATIONS,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_table_count() {
        // 22 tables per plan §5.1.
        assert_eq!(TRACKING_TABLES.len(), 22);
    }

    #[test]
    fn tracking_tables_unique() {
        let mut seen = std::collections::HashSet::new();
        for t in TRACKING_TABLES {
            assert!(seen.insert(*t), "duplicate table name: {t}");
        }
    }
}
