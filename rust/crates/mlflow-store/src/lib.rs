//! `mlflow-store`: the tracking/tracing backend store.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§5.1, Phase 2), this crate owns
//! experiments, runs, params, tags, metrics/`latest_metrics`, datasets,
//! inputs/outputs, logged models, and tracing V3 (`trace_info`, spans,
//! assessments, entity associations) against SQLite/PostgreSQL/MySQL via
//! `sqlx`. It is responsible for the wire-invisible query improvements
//! called out in §5.2 (keyset pagination, semi-joins, atomic
//! `latest_metrics` upserts, single-transaction `log_batch`) while matching
//! the observable behavior of `mlflow/store/tracking/sqlalchemy_store.py`
//! exactly, including workspace-scoped variants (§3.17).
//!
//! This module currently implements the T2.1/T2.2 foundation:
//!
//! * [`uri`] — SQLAlchemy URI parsing (with `+driver` suffixes) to `sqlx`.
//! * [`dialect`] — per-dialect SQL forms (upserts, LIKE case-sensitivity,
//!   quoting, capabilities).
//! * [`pool`] — pool tuning mapped from MLflow env vars.
//! * [`db`] — the [`db::Db`] pool enum, SQLite session PRAGMAs, and Alembic
//!   head verification ([`db::Db::connect_and_verify`]).
//! * [`schema`] — plain data structs mirroring every §5.1 tracking table.
//!
//! Integration guidance for the store methods to come (T2.4+): take a
//! [`db::Db`] and, when a query differs per backend, ask [`db::Db::dialect`]
//! for a [`dialect::Dialect`] and build SQL through its helpers (e.g.
//! [`dialect::Dialect::upsert`], [`dialect::Dialect::case_sensitive_like`]).
//! Bind values positionally in the order the helper emitted its placeholders.

pub mod db;
pub mod dialect;
pub mod pool;
pub mod schema;
pub mod store;
pub mod uri;

pub use db::{Db, SchemaError, StoreError, EXPECTED_ALEMBIC_HEAD};
pub use dialect::{Dialect, UpsertSpec};
pub use pool::PoolConfig;
pub use store::{
    Dataset, DatasetInput, DatasetInputSpec, DatasetSummary, Experiment, ExperimentTag, InputTag,
    LoggedModelInput, LoggedModelOutput, Metric, MetricInput, MetricWithRunId, Param, Run, RunData,
    RunInfo, RunInputs, RunOutputs, RunStatus, RunTag, RunsPage, TrackingStore, ViewType,
    GET_METRIC_HISTORY_MAX_RESULTS, MAX_DATASET_SUMMARIES_RESULTS, MAX_RESULTS_PER_RUN,
    MAX_RUNS_GET_METRIC_HISTORY_BULK, SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD,
};
pub use uri::{ParsedUri, UriError};
