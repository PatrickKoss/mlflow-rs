//! Logged models: CRUD, the finalize state machine, tags/params, and
//! `search_logged_models` (plan T2.9), mirroring the logged-model methods in
//! `mlflow/store/tracking/sqlalchemy_store.py`.
//!
//! ## ID / artifact-location generation (`create_logged_model`)
//!
//! `model_id = "m-" + uuid.uuid4().hex` (32-char lowercase hex, no dashes —
//! `Uuid::new_v4().simple()` in Rust). The artifact location is
//! `<experiment.artifact_location>/models/<model_id>/artifacts`
//! (`MODELS_FOLDER_NAME` + `ARTIFACTS_FOLDER_NAME`). The model name defaults to
//! a random name (same generator as runs) when absent.
//!
//! ## Finalize (`finalize_logged_model`)
//!
//! Python has **no state-machine guard**: it unconditionally sets `status` and
//! `last_updated_timestamp_ms`, regardless of the model's current status (a
//! finalized model can be "re-finalized" to a different status with no error).
//! The only failure mode is "model not found". We port that exactly rather
//! than inventing a PENDING-only restriction.
//!
//! ## `search_logged_models` filter semantics
//!
//! The SqlAlchemyStore filter parser is **not** the sqlparse-grammar
//! `SearchLoggedModelsUtils` class (that one belongs to `FileStore`) — it is
//! the standalone, simpler `mlflow.utils.search_logged_model_utils.parse_filter_string`,
//! ported in `mlflow-search` as
//! [`mlflow_search::parse::logged_models_filter_sqlalchemy`]. See that
//! module's docs for the two-parsers explanation and the faithfully-preserved
//! Python quirks (dotted `attributes.<numeric-alias>` skipping alias
//! resolution; the `validate_op` error message always naming `string_ops`).
//!
//! Non-attribute filters (metric/param/tag) are applied as `EXISTS` semi-joins
//! against the child tables rather than literal SQLAlchemy `.join(subquery)`
//! calls. Python's join-based approach fans out one `logged_models` row per
//! matching child row and relies on SQLAlchemy 2.0's automatic ORM-entity
//! result deduplication to collapse them back down — but that dedup happens
//! **after** the `OFFSET`/`LIMIT` window is applied to the raw (fanned-out)
//! rows, which can silently drop models from a page when one model has
//! multiple matching child rows (verified against a live SqlAlchemyStore: a
//! `metrics.acc > 0` filter matching 3 datasets per model, with
//! `max_results=2`, returns 1 model and **no continuation token** even though
//! 3 distinct models match). This is an ORM-internals-dependent, effectively
//! non-deterministic bug, not a documented contract — we intentionally do not
//! reproduce it; the EXISTS form returns the correct distinct-model set and
//! paginates correctly. Documented as a deviation in the T2.9 report.
//!
//! ## Dataset-scoped metric ordering (`order_by: [{"field_name": "metrics.X",
//! "dataset_name": ..., "dataset_digest": ...}]`)
//!
//! Ported verbatim from `_apply_order_by_search_logged_models`: for each
//! metric order-by clause, rank `logged_model_metrics` rows per
//! `(model_id, metric_name)` by `(timestamp DESC, step DESC)` (optionally
//! restricted to a `(dataset_name, dataset_digest)` pair first), keep rank 1,
//! outer-join that onto the model list, and sort by that value with NULLs
//! last (a `CASE WHEN value IS NULL THEN 1 ELSE 0 END` column ahead of the
//! value, exactly like the runs order-by NULLS-LAST emulation). A
//! `creation_timestamp_ms DESC` tiebreak is appended whenever the caller
//! didn't already order by creation timestamp.
//!
//! ## Pagination (`SearchLoggedModelsPaginationToken`)
//!
//! Plain base64(JSON) offset token, ported byte-for-byte from
//! `mlflow.utils.search_utils.SearchLoggedModelsPaginationToken`: JSON object
//! `{"experiment_ids", "filter_string", "order_by", "offset"}` (Python
//! `dataclasses.asdict` field order), validated against the *current*
//! request's `experiment_ids`/`filter_string`/`order_by` on decode. The
//! default `max_results` is **100** (`SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT`
//! in `mlflow/store/tracking/__init__.py`), and — matching the real Python
//! source exactly — there is **no** enforced maximum; the plan's "default
//! 50, max 50" text does not match `sqlalchemy_store.py:3433` or its handler,
//! so we follow the actual Python constant/behavior instead (documented
//! deviation from the plan text, not from Python).

use base64::Engine;
use mlflow_error::MlflowError;
use mlflow_search::{SqlaComparison, SqlaEntityType, SqlaValue};
use uuid::Uuid;

use super::dbutil::{RowLike, Val};
use super::entities::LifecycleStage;
use super::experiments::{internal, parse_experiment_id, ViewType};
use super::names::generate_random_name;
use super::uri_util::append_to_uri_path;
use super::validation;
use super::{TrackingStore, ARTIFACTS_FOLDER_NAME};
use crate::dialect::Dialect;
use crate::schema::logged_models::{
    LOGGED_MODELS, LOGGED_MODEL_METRICS, LOGGED_MODEL_PARAMS, LOGGED_MODEL_TAGS,
};

/// `MODELS_FOLDER_NAME` (`SqlAlchemyStore.MODELS_FOLDER_NAME`).
const MODELS_FOLDER_NAME: &str = "models";

/// `SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT` (`mlflow/store/tracking/__init__.py`).
pub const SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT: usize = 100;

/// `LoggedModelStatus` (`mlflow.entities.LoggedModelStatus`): the DB-persisted
/// int code plus the wire string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoggedModelStatus {
    Unspecified,
    Pending,
    Ready,
    Failed,
}

impl LoggedModelStatus {
    pub fn from_int(v: i64) -> Result<Self, MlflowError> {
        match v {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::Pending),
            2 => Ok(Self::Ready),
            3 => Ok(Self::Failed),
            other => Err(MlflowError::invalid_parameter_value(format!(
                "Unknown model status: {other}"
            ))),
        }
    }

    pub fn to_int(self) -> i64 {
        match self {
            Self::Unspecified => 0,
            Self::Pending => 1,
            Self::Ready => 2,
            Self::Failed => 3,
        }
    }
}

/// A logged-model tag or param (`LoggedModelTag`/`LoggedModelParameter`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedModelKv {
    pub key: String,
    pub value: String,
}

/// A logged-model metric point, including its `(run_id, dataset_name,
/// dataset_digest)` provenance (`mlflow.entities.Metric` as attached to a
/// `LoggedModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedModelMetric {
    pub key: String,
    /// The stored value. Unlike run metrics (`metrics`/`latest_metrics`, which
    /// have a companion `is_nan` flag), `logged_model_metrics` has no such
    /// flag: `_log_model_metrics` sanitizes NaN to a plain `0.0` and discards
    /// the "was NaN" bit entirely (`_, value = self.sanitize_metric_value(...)`
    /// — the boolean is thrown away). So a NaN metric is permanently
    /// indistinguishable from a logged `0.0`; we do not attempt to restore it.
    /// `None` only if the column is somehow SQL NULL (no current writer
    /// produces this, but the schema allows it).
    pub value: Option<f64>,
    pub timestamp: i64,
    pub step: i64,
    pub run_id: String,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// One metric to log via [`TrackingStore::log_logged_model_metrics`].
#[derive(Debug, Clone)]
pub struct LoggedModelMetricInput {
    pub key: String,
    pub value: f64,
    pub timestamp: i64,
    pub step: i64,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// A logged model (`mlflow.entities.LoggedModel`), with its params/tags/metrics
/// inlined exactly like `SqlLoggedModel.to_mlflow_entity` (all metric rows —
/// not deduplicated to "latest per key").
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedModel {
    pub model_id: String,
    pub experiment_id: String,
    pub name: String,
    pub artifact_location: String,
    pub creation_timestamp: i64,
    pub last_updated_timestamp: i64,
    pub status: i64,
    pub model_type: Option<String>,
    pub source_run_id: Option<String>,
    pub status_message: Option<String>,
    pub tags: Vec<LoggedModelKv>,
    pub params: Vec<LoggedModelKv>,
    pub metrics: Vec<LoggedModelMetric>,
}

/// One `datasets` clause of `search_logged_models` (`DatasetFilter`).
#[derive(Debug, Clone)]
pub struct DatasetFilter {
    pub dataset_name: String,
    pub dataset_digest: Option<String>,
}

/// One `order_by` clause of `search_logged_models`.
#[derive(Debug, Clone)]
pub struct LoggedModelOrderByInput {
    pub field_name: String,
    pub ascending: bool,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// A page of logged models plus the opaque continuation token.
#[derive(Debug)]
pub struct LoggedModelsPage {
    pub models: Vec<LoggedModel>,
    pub next_page_token: Option<String>,
}

impl TrackingStore {
    /// `create_logged_model`. Requires the experiment to exist (any workspace
    /// visibility rule — resolved via [`TrackingStore::fetch_experiment`]) and
    /// be ACTIVE.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_logged_model(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: Option<&str>,
        source_run_id: Option<&str>,
        tags: &[LoggedModelKv],
        params: &[LoggedModelKv],
        model_type: Option<&str>,
    ) -> Result<LoggedModel, MlflowError> {
        validate_logged_model_name(name)?;
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

        for p in params {
            validation::validate_param(&p.key, &p.value)?;
        }
        for t in tags {
            validation::validate_tag(&t.key, &t.value, None)?;
        }

        let model_id = format!("m-{}", Uuid::new_v4().simple());
        let artifact_location = append_to_uri_path(
            experiment.artifact_location.as_deref().unwrap_or(""),
            &[MODELS_FOLDER_NAME, &model_id, ARTIFACTS_FOLDER_NAME],
        );
        let name = match name {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => generate_random_name(),
        };
        let now = super::experiments::now_millis();

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        let ph = |i| dialect.placeholder(i);
        let sql = format!(
            "INSERT INTO {LOGGED_MODELS} \
             (model_id, experiment_id, name, artifact_location, creation_timestamp_ms, \
              last_updated_timestamp_ms, model_type, status, lifecycle_stage, source_run_id) \
             VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
            ph(5),
            ph(6),
            ph(7),
            ph(8),
            ph(9),
            ph(10),
        );
        tx.exec(
            &sql,
            &[
                Val::Text(model_id.clone()),
                Val::Int(exp_id),
                Val::Text(name.clone()),
                Val::Text(artifact_location.clone()),
                Val::Int(now),
                Val::Int(now),
                Val::OptText(model_type.map(str::to_string)),
                Val::Int(LoggedModelStatus::Pending.to_int()),
                Val::Text(LifecycleStage::ACTIVE.to_string()),
                Val::OptText(source_run_id.map(str::to_string)),
            ],
        )
        .await
        .map_err(internal)?;

        for p in params {
            insert_param_tx(&mut tx, dialect, &model_id, exp_id, &p.key, &p.value).await?;
        }
        for t in tags {
            insert_tag_tx(&mut tx, dialect, &model_id, exp_id, &t.key, &t.value).await?;
        }

        tx.commit().await.map_err(internal)?;

        self.get_logged_model(workspace, &model_id, false).await
    }

    /// `log_logged_model_params`. Requires the model to exist (workspace- and
    /// deletion-agnostic, matching `_get_logged_model_record` — no
    /// `lifecycle_stage` filter there).
    pub async fn log_logged_model_params(
        &self,
        workspace: &str,
        model_id: &str,
        params: &[LoggedModelKv],
    ) -> Result<(), MlflowError> {
        for p in params {
            validation::validate_param(&p.key, &p.value)?;
        }
        let row = self
            .resolve_logged_model_row(workspace, model_id, true)
            .await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for p in params {
            insert_param_tx(
                &mut tx,
                dialect,
                model_id,
                row.experiment_id,
                &p.key,
                &p.value,
            )
            .await?;
        }
        tx.commit().await.map_err(internal)
    }

    /// `_log_model_metrics` (sqlalchemy_store.py:1288), standalone entry point:
    /// opens its own transaction and delegates to
    /// [`TrackingStore::log_model_metrics_tx`]. Used directly by tests that
    /// don't go through `log_batch`/`log_metric`.
    pub async fn log_logged_model_metrics(
        &self,
        model_id: &str,
        experiment_id: i64,
        run_id: &str,
        dataset_uuid: Option<&str>,
        metrics: &[LoggedModelMetricInput],
    ) -> Result<(), MlflowError> {
        if metrics.is_empty() {
            return Ok(());
        }
        let workspace = self.workspace_of_experiment(experiment_id).await?;
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let owned: Vec<super::metrics::MetricInput> = metrics
            .iter()
            .map(|m| super::metrics::MetricInput {
                key: m.key.clone(),
                value: m.value,
                timestamp: m.timestamp,
                step: m.step,
                model_id: Some(model_id.to_string()),
                dataset_name: m.dataset_name.clone(),
                dataset_digest: m.dataset_digest.clone(),
            })
            .collect();
        self.log_model_metrics_tx(
            &mut tx,
            &workspace,
            experiment_id,
            run_id,
            dataset_uuid,
            &owned,
        )
        .await?;
        tx.commit().await.map_err(internal)
    }

    /// `_log_model_metrics` (sqlalchemy_store.py:1288): the production writer,
    /// callable from within an already-open transaction so `log_batch`/
    /// `log_metric` (`metrics.rs`) can fold it into their single transaction
    /// (Q6 spirit — see `metrics.rs`'s `log_batch` doc comment for why this
    /// deviates from Python's separate-session behavior here).
    ///
    /// `metrics` is the **whole** metrics list handed to the caller (not
    /// pre-filtered to one model_id) — exactly Python's shape: `Metric`
    /// carries its own `model_id` per element (`SqlLoggedModelMetric(
    /// model_id=metric.model_id, ...)`), so one call can legitimately write
    /// rows for several different models at once (e.g. a `log_batch` request
    /// mixing metrics for model A and model B). Preserving this shape (rather
    /// than pre-grouping by model_id before calling in) matters for two
    /// subtle bits of parity: `is_single_metric = len(metrics) == 1` is
    /// computed over the *whole* list Python was handed (true even if only
    /// one of several metrics carries a model_id), and the `metrics[idx]`
    /// validation-error path index is the position in that *original* list,
    /// not in a per-model subset.
    ///
    /// Steps, mirroring Python exactly except where noted:
    /// 1. Skip elements with no `model_id` (Python's `if metric.model_id is
    ///    None: continue`).
    /// 2. Dedup by full `(key, value, timestamp, step, model_id, dataset_name,
    ///    dataset_digest)` tuple, keeping the first occurrence — mirrors
    ///    Python's `Metric.__eq__`/`__hash__`, which include every field.
    /// 3. `_validate_metric` per element (same validator run metrics use).
    /// 4. `sanitize_metric_value` (NaN → 0.0, the "was NaN" bit is discarded —
    ///    `logged_model_metrics` has no `is_nan` column, see
    ///    [`LoggedModelMetric::value`]'s doc).
    /// 5. Insert with **conflict-target = the table's 5-column PK**
    ///    `(model_id, metric_name, metric_timestamp_ms, metric_step, run_id)`,
    ///    `DO NOTHING` on conflict. This is a deliberate simplification of
    ///    Python's IntegrityError-catch-and-retry loop: Python attempts one
    ///    `add_all` + `commit` for the whole batch, and on a PK violation
    ///    rolls back, re-queries existing rows for the run, and re-inserts
    ///    only the rows that are not already present — the net *observable*
    ///    effect is "a row whose PK already exists is silently kept as-is;
    ///    every other row is inserted", which is exactly what
    ///    `ON CONFLICT ... DO NOTHING` gives us in one statement, without
    ///    reproducing the retry machinery.
    ///
    /// ## Deviation: model_id existence
    ///
    /// Python has **no explicit check** that `model_id` refers to an existing
    /// `LoggedModel` before this insert — it relies entirely on the `logged_model_metrics.model_id`
    /// foreign key to `logged_models.model_id` (`ondelete=CASCADE`) raising at
    /// commit time. On SQLite (`PRAGMA foreign_keys = ON`, `mlflow/store/db/utils.py:155`)
    /// that FK violation is an `IntegrityError`, which Python's `except
    /// sqlalchemy.exc.IntegrityError` handler here (written only for the
    /// PK-duplicate case) misclassifies as "duplicate metrics to filter out",
    /// retries the insert, and the retry raises the *same* `IntegrityError`
    /// again — this time uncaught by this method, propagating up through
    /// `ManagedSessionMaker`'s generic `except sqlalchemy.exc.SQLAlchemyError`
    /// handler as `MlflowException(error_code=BAD_REQUEST)` wrapping a raw
    /// "FOREIGN KEY constraint failed" message. That's an accident of
    /// unrelated exception-handling code, not a designed error contract (no
    /// test in the Python suite exercises this path, and the exact wording is
    /// DB-driver-specific). We instead validate explicitly, workspace-scoped,
    /// and raise the same clean `RESOURCE_DOES_NOT_EXIST` "Logged model with
    /// ID '...' not found." used everywhere else in this module — matching
    /// Python's *intent* (invalid model_id must fail) with a real error
    /// message instead of Python's incidental one. We check every distinct
    /// model_id referenced by the sanitized batch (still one query per
    /// distinct id, but at most as many as distinct models in the request).
    pub(crate) async fn log_model_metrics_tx(
        &self,
        tx: &mut super::dbutil::Tx<'_>,
        workspace: &str,
        experiment_id: i64,
        run_id: &str,
        dataset_uuid: Option<&str>,
        metrics: &[super::metrics::MetricInput],
    ) -> Result<(), MlflowError> {
        let is_single_metric = metrics.len() == 1;
        let mut seen: Vec<&super::metrics::MetricInput> = Vec::new();
        let mut sanitized: Vec<(&super::metrics::MetricInput, f64)> = Vec::new();
        for (idx, m) in metrics.iter().enumerate() {
            if m.model_id.is_none() {
                continue;
            }
            if seen.iter().any(|s| model_metric_eq(s, m)) {
                continue;
            }
            seen.push(m);

            let path = (!is_single_metric).then(|| format!("metrics[{idx}]"));
            validation::validate_metric(&m.key, m.value, m.timestamp, m.step, path.as_deref())?;
            let (_, value) = super::metrics::sanitize_metric_value(m.value);
            sanitized.push((m, value));
        }
        if sanitized.is_empty() {
            return Ok(());
        }

        // Every distinct model_id referenced must exist (workspace-scoped) —
        // see the deviation note above for why this is an explicit check
        // rather than a Python-style FK-violation fallthrough.
        let mut checked: Vec<&str> = Vec::new();
        for (m, _) in &sanitized {
            let model_id = m.model_id.as_deref().expect("filtered above");
            if !checked.contains(&model_id) {
                self.check_model_exists_tx(tx, workspace, model_id).await?;
                checked.push(model_id);
            }
        }

        let dialect = self.db().dialect();
        let sql = model_metric_insert_sql(dialect);
        for (m, value) in sanitized {
            let model_id = m.model_id.as_deref().expect("filtered above");
            let binds = [
                Val::Text(model_id.to_string()),
                Val::Text(m.key.clone()),
                Val::Int(m.timestamp),
                Val::Int(m.step),
                Val::Float(value),
                Val::Int(experiment_id),
                Val::Text(run_id.to_string()),
                Val::OptText(dataset_uuid.map(str::to_string)),
                Val::OptText(m.dataset_name.clone()),
                Val::OptText(m.dataset_digest.clone()),
            ];
            tx.exec(&sql, &binds).await.map_err(internal)?;
        }
        Ok(())
    }

    /// Existence check for `model_id`, workspace-scoped, inside `tx` (no
    /// `Tx::fetch_optional` exists — an empty `fetch_all` is equivalent).
    async fn check_model_exists_tx(
        &self,
        tx: &mut super::dbutil::Tx<'_>,
        workspace: &str,
        model_id: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT 1 AS one FROM {LOGGED_MODELS} m \
             WHERE m.model_id = {} AND m.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        let rows = tx
            .fetch_all(
                &sql,
                &[
                    Val::Text(model_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |r| r.get_i64("one"),
            )
            .await
            .map_err(internal)?;
        if rows.is_empty() {
            return Err(not_found(model_id));
        }
        Ok(())
    }

    /// The workspace owning `experiment_id` — used by
    /// [`TrackingStore::log_logged_model_metrics`], whose signature (matching
    /// its existing test-facing shape) takes an `experiment_id` rather than a
    /// `workspace`, unlike every other store method.
    async fn workspace_of_experiment(&self, experiment_id: i64) -> Result<String, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT workspace FROM experiments WHERE experiment_id = {}",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_optional(&sql, &[Val::Int(experiment_id)], |r| {
                r.get_string("workspace")
            })
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "No Experiment with id={experiment_id} exists"
                ))
            })
    }

    /// `get_logged_model`. `allow_deleted` mirrors the Python flag: when
    /// `false`, a soft-deleted model behaves as not-found.
    pub async fn get_logged_model(
        &self,
        workspace: &str,
        model_id: &str,
        allow_deleted: bool,
    ) -> Result<LoggedModel, MlflowError> {
        let row = self
            .resolve_logged_model_row(workspace, model_id, allow_deleted)
            .await?;
        self.assemble_logged_model(row).await
    }

    /// `delete_logged_model`: soft-delete (lifecycle DELETED +
    /// `last_updated_timestamp_ms`). Requires the model to currently be
    /// resolvable (not already excluded by lifecycle — matches
    /// `_get_logged_model_record`, which has no lifecycle filter, so deleting
    /// an already-deleted model is a harmless idempotent no-op-ish update).
    pub async fn delete_logged_model(
        &self,
        workspace: &str,
        model_id: &str,
    ) -> Result<(), MlflowError> {
        self.resolve_logged_model_row(workspace, model_id, true)
            .await?;
        let now = super::experiments::now_millis();
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {LOGGED_MODELS} SET lifecycle_stage = {}, last_updated_timestamp_ms = {} \
             WHERE model_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(LifecycleStage::DELETED.to_string()),
                    Val::Int(now),
                    Val::Text(model_id.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `finalize_logged_model`. No state-machine guard (see module docs):
    /// unconditionally sets `status` + `last_updated_timestamp_ms`.
    pub async fn finalize_logged_model(
        &self,
        workspace: &str,
        model_id: &str,
        status: LoggedModelStatus,
    ) -> Result<LoggedModel, MlflowError> {
        self.resolve_logged_model_row(workspace, model_id, true)
            .await?;
        let now = super::experiments::now_millis();
        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {LOGGED_MODELS} SET status = {}, last_updated_timestamp_ms = {} \
             WHERE model_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::Int(status.to_int()),
                    Val::Int(now),
                    Val::Text(model_id.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_logged_model(workspace, model_id, true).await
    }

    /// `set_logged_model_tags` (upsert per tag).
    pub async fn set_logged_model_tags(
        &self,
        workspace: &str,
        model_id: &str,
        tags: &[LoggedModelKv],
    ) -> Result<(), MlflowError> {
        for t in tags {
            validation::validate_tag(&t.key, &t.value, None)?;
        }
        let row = self
            .resolve_logged_model_row(workspace, model_id, true)
            .await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for t in tags {
            upsert_tag_tx(
                &mut tx,
                dialect,
                model_id,
                row.experiment_id,
                &t.key,
                &t.value,
            )
            .await?;
        }
        tx.commit().await.map_err(internal)
    }

    /// `delete_logged_model_tag`. Errors `RESOURCE_DOES_NOT_EXIST` when the
    /// tag key is absent on the model.
    pub async fn delete_logged_model_tag(
        &self,
        workspace: &str,
        model_id: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        self.resolve_logged_model_row(workspace, model_id, true)
            .await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM {LOGGED_MODEL_TAGS} WHERE model_id = {} AND tag_key = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        let affected = self
            .db()
            .exec(
                &sql,
                &[Val::Text(model_id.to_string()), Val::Text(key.to_string())],
            )
            .await
            .map_err(internal)?;
        if affected == 0 {
            return Err(MlflowError::resource_does_not_exist(format!(
                "No tag with key {} found for model with ID {}.",
                py_repr(key),
                py_repr(model_id),
            )));
        }
        Ok(())
    }

    /// `search_logged_models`. See module docs for the filter/order-by/token
    /// semantics and the documented pagination-fanout deviation.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_logged_models(
        &self,
        workspace: &str,
        experiment_ids: &[String],
        filter_string: Option<&str>,
        datasets: &[DatasetFilter],
        max_results: Option<usize>,
        order_by: &[LoggedModelOrderByInput],
        page_token: Option<&str>,
    ) -> Result<LoggedModelsPage, MlflowError> {
        if datasets.iter().any(|d| d.dataset_name.is_empty()) {
            return Err(MlflowError::invalid_parameter_value(
                "`dataset_name` in the `datasets` clause must be specified.",
            ));
        }

        let offset = match page_token {
            Some(t) => {
                let token = PaginationToken::decode(t)?;
                token.validate(experiment_ids, filter_string, order_by)?;
                token.offset
            }
            None => 0,
        };
        let max_results = max_results.unwrap_or(SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT);

        let comparisons = mlflow_search::parse::logged_models_filter_sqlalchemy(filter_string)
            .map_err(search_err)?;

        let dialect = self.db().dialect();
        let mut ph = PlaceholderGen::new(dialect);
        // Bind values must be pushed in the exact textual order their `?`
        // placeholders appear in the final SQL string (positional binding).
        // The rendered statement is `SELECT ... FROM logged_models lm
        // <order_join> WHERE <wheres> ORDER BY <order_sql> LIMIT ? OFFSET ?`,
        // so the JOIN clause's placeholders (built by `build_order_by`) come
        // textually *before* the WHERE clause's — bind them first.
        let mut binds: Vec<Val> = Vec::new();

        let (order_sql, order_join, mut order_binds) = build_order_by(dialect, order_by, &mut ph)?;
        binds.append(&mut order_binds);

        let mut exp_ids: Vec<i64> = Vec::with_capacity(experiment_ids.len());
        for e in experiment_ids {
            exp_ids.push(parse_experiment_id(e)?);
        }

        let mut wheres: Vec<String> = vec![format!(
            "lifecycle_stage != {}",
            push_text(&mut ph, &mut binds, LifecycleStage::DELETED)
        )];
        if exp_ids.is_empty() {
            wheres.push("1 = 0".to_string());
        } else {
            let phs: Vec<String> = exp_ids
                .iter()
                .map(|id| {
                    binds.push(Val::Int(*id));
                    ph.next()
                })
                .collect();
            wheres.push(format!("experiment_id IN ({})", phs.join(", ")));
            // Workspace scoping (plan §3.17): an experiment_id that exists but
            // belongs to another workspace must not leak its models, mirroring
            // every other workspace-scoped query's semi-join against
            // `experiments`.
            let ws_ph = push_text(&mut ph, &mut binds, workspace);
            wheres.push(format!(
                "experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = {ws_ph})"
            ));
        }

        let mut has_metric_filter = false;
        for c in &comparisons {
            if c.entity_type == SqlaEntityType::Metric {
                has_metric_filter = true;
            }
            wheres.push(build_filter_predicate(
                dialect, c, datasets, &mut ph, &mut binds,
            )?);
        }

        // Dataset filters with no metric filter: require *any* metric on one
        // of the named datasets (`_apply_filter_string_datasets_search_logged_models`).
        if !datasets.is_empty() && !has_metric_filter {
            wheres.push(build_any_metric_on_datasets_predicate(
                dialect, datasets, &mut ph, &mut binds,
            ));
        }

        let limit_ph = ph.next();
        binds.push(Val::Int((max_results + 1) as i64));
        let offset_ph = ph.next();
        binds.push(Val::Int(offset as i64));

        // No `SELECT DISTINCT`: the metric order-by JOIN is against a
        // rank-1-per-model subquery, so it does not fan out rows in the
        // normal case. (A tied rank — two metric rows with identical
        // (timestamp, step) — can still produce more than one row per model,
        // exactly like Python's `RANK()`-based subquery with no dedup either;
        // we collapse that below rather than leaning on `DISTINCT`, which
        // Postgres rejects here since `ORDER BY` references a joined column
        // outside the select list.)
        let sql = format!(
            "SELECT lm.model_id AS model_id FROM {LOGGED_MODELS} lm {order_join} \
             WHERE {} ORDER BY {order_sql} LIMIT {limit_ph} OFFSET {offset_ph}",
            wheres.join(" AND "),
        );

        let raw_ids: Vec<String> = self
            .db()
            .fetch_all(&sql, &binds, |r| r.get_string("model_id"))
            .await
            .map_err(internal)?;
        let mut seen = std::collections::HashSet::new();
        let ids: Vec<String> = raw_ids
            .into_iter()
            .filter(|id| seen.insert(id.clone()))
            .collect();

        let (page_ids, next_token) = if ids.len() > max_results {
            let kept = &ids[..max_results];
            let token = PaginationToken {
                offset: offset + max_results,
                experiment_ids: experiment_ids.to_vec(),
                filter_string: filter_string.map(str::to_string),
                order_by: order_by.to_vec(),
            }
            .encode();
            (kept, Some(token))
        } else {
            (&ids[..], None)
        };

        let mut models = Vec::with_capacity(page_ids.len());
        for id in page_ids {
            let row = self.resolve_logged_model_row(workspace, id, true).await?;
            models.push(self.assemble_logged_model(row).await?);
        }

        Ok(LoggedModelsPage {
            models,
            next_page_token: next_token,
        })
    }

    // ---- internal helpers ----

    /// Fetch the `logged_models` row, scoped to the workspace via a semi-join
    /// to `experiments`. Errors `RESOURCE_DOES_NOT_EXIST` "Logged model with
    /// ID '...' not found." when missing/out-of-workspace (matches
    /// `_raise_model_not_found`). When `allow_deleted` is `false`, a
    /// soft-deleted model is treated as not found (matches `get_logged_model`'s
    /// `allow_deleted` flag — the other mutating methods always pass `true`
    /// here since `_get_logged_model_record` has no lifecycle filter).
    async fn resolve_logged_model_row(
        &self,
        workspace: &str,
        model_id: &str,
        allow_deleted: bool,
    ) -> Result<LoggedModelRow, MlflowError> {
        let dialect = self.db().dialect();
        let mut sql = format!(
            "SELECT {cols} FROM {LOGGED_MODELS} m \
             WHERE m.model_id = {} AND m.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
            cols = LoggedModelRow::SELECT_COLS,
        );
        if !allow_deleted {
            sql.push_str(&format!(
                " AND m.lifecycle_stage != '{}'",
                LifecycleStage::DELETED
            ));
        }
        self.db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(model_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                LoggedModelRow::from_row,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| not_found(model_id))
    }

    async fn assemble_logged_model(&self, row: LoggedModelRow) -> Result<LoggedModel, MlflowError> {
        let params = self.load_logged_model_params(&row.model_id).await?;
        let tags = self.load_logged_model_tags(&row.model_id).await?;
        let metrics = self.load_logged_model_metrics(&row.model_id).await?;
        Ok(LoggedModel {
            model_id: row.model_id,
            experiment_id: row.experiment_id.to_string(),
            name: row.name,
            artifact_location: row.artifact_location,
            creation_timestamp: row.creation_timestamp_ms,
            last_updated_timestamp: row.last_updated_timestamp_ms,
            status: row.status,
            model_type: row.model_type,
            source_run_id: row.source_run_id,
            status_message: row.status_message,
            tags,
            params,
            metrics,
        })
    }

    async fn load_logged_model_params(
        &self,
        model_id: &str,
    ) -> Result<Vec<LoggedModelKv>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT param_key, param_value FROM {LOGGED_MODEL_PARAMS} WHERE model_id = {} \
             ORDER BY param_key",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Text(model_id.to_string())], |r| {
                Ok(LoggedModelKv {
                    key: r.get_string("param_key")?,
                    value: r.get_string("param_value")?,
                })
            })
            .await
            .map_err(internal)
    }

    async fn load_logged_model_tags(
        &self,
        model_id: &str,
    ) -> Result<Vec<LoggedModelKv>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT tag_key, tag_value FROM {LOGGED_MODEL_TAGS} WHERE model_id = {} ORDER BY tag_key",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Text(model_id.to_string())], |r| {
                Ok(LoggedModelKv {
                    key: r.get_string("tag_key")?,
                    value: r.get_string("tag_value")?,
                })
            })
            .await
            .map_err(internal)
    }

    /// All metric rows for the model (no "latest per key" dedup — matches
    /// `SqlLoggedModel.to_mlflow_entity`, which inlines the raw
    /// `metrics` relationship).
    async fn load_logged_model_metrics(
        &self,
        model_id: &str,
    ) -> Result<Vec<LoggedModelMetric>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT metric_name, metric_value, metric_timestamp_ms, metric_step, run_id, \
             dataset_name, dataset_digest FROM {LOGGED_MODEL_METRICS} WHERE model_id = {} \
             ORDER BY metric_name, metric_timestamp_ms, metric_step, run_id",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Text(model_id.to_string())], |r| {
                Ok(LoggedModelMetric {
                    key: r.get_string("metric_name")?,
                    value: r.get_opt_f64("metric_value")?,
                    timestamp: r.get_i64("metric_timestamp_ms")?,
                    step: r.get_i64("metric_step")?,
                    run_id: r.get_string("run_id")?,
                    dataset_name: r.get_opt_string("dataset_name")?,
                    dataset_digest: r.get_opt_string("dataset_digest")?,
                })
            })
            .await
            .map_err(internal)
    }
}

/// The physical row of `logged_models` read for entity assembly / mutation.
struct LoggedModelRow {
    model_id: String,
    experiment_id: i64,
    name: String,
    artifact_location: String,
    creation_timestamp_ms: i64,
    last_updated_timestamp_ms: i64,
    status: i64,
    model_type: Option<String>,
    source_run_id: Option<String>,
    status_message: Option<String>,
}

impl LoggedModelRow {
    const SELECT_COLS: &'static str = "m.model_id, m.experiment_id, m.name, m.artifact_location, \
         m.creation_timestamp_ms, m.last_updated_timestamp_ms, m.status, m.model_type, \
         m.source_run_id, m.status_message";

    fn from_row(r: &dyn RowLike) -> Result<Self, sqlx::Error> {
        Ok(Self {
            model_id: r.get_string("model_id")?,
            experiment_id: r.get_int("experiment_id")?,
            name: r.get_string("name")?,
            artifact_location: r.get_string("artifact_location")?,
            creation_timestamp_ms: r.get_i64("creation_timestamp_ms")?,
            last_updated_timestamp_ms: r.get_i64("last_updated_timestamp_ms")?,
            status: r.get_i64("status")?,
            model_type: r.get_opt_string("model_type")?,
            source_run_id: r.get_opt_string("source_run_id")?,
            status_message: r.get_opt_string("status_message")?,
        })
    }
}

/// `_raise_model_not_found`.
fn not_found(model_id: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("Logged model with ID '{model_id}' not found."))
}

/// `_validate_logged_model_name`.
fn validate_logged_model_name(name: Option<&str>) -> Result<(), MlflowError> {
    let Some(name) = name else {
        return Ok(());
    };
    const BAD_CHARS: &[char] = &['/', ':', '.', '%', '"', '\''];
    if name.is_empty() || name.chars().any(|c| BAD_CHARS.contains(&c)) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid model name ({}) provided. Model name must be a non-empty string and \
             cannot contain the following characters: ('/', ':', '.', '%', '\"', \"'\")",
            py_repr(name),
        )));
    }
    Ok(())
}

/// Python `repr()` of a plain string: single-quoted, switching to double
/// quotes only when the string contains a single quote but no double quote
/// (matches CPython's quote-choice heuristic for the simple ASCII identifiers
/// — model ids, tag/param keys — these error messages embed).
fn py_repr(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let (quote, escape) = if has_single && !has_double {
        ('"', '"')
    } else {
        ('\'', '\'')
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == escape => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

async fn insert_param_tx(
    tx: &mut super::dbutil::Tx<'_>,
    dialect: Dialect,
    model_id: &str,
    experiment_id: i64,
    key: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let ph = |i| dialect.placeholder(i);
    let sql = format!(
        "INSERT INTO {LOGGED_MODEL_PARAMS} (model_id, experiment_id, param_key, param_value) \
         VALUES ({}, {}, {}, {})",
        ph(1),
        ph(2),
        ph(3),
        ph(4),
    );
    tx.exec(
        &sql,
        &[
            Val::Text(model_id.to_string()),
            Val::Int(experiment_id),
            Val::Text(key.to_string()),
            Val::Text(value.to_string()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

async fn insert_tag_tx(
    tx: &mut super::dbutil::Tx<'_>,
    dialect: Dialect,
    model_id: &str,
    experiment_id: i64,
    key: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let ph = |i| dialect.placeholder(i);
    let sql = format!(
        "INSERT INTO {LOGGED_MODEL_TAGS} (model_id, experiment_id, tag_key, tag_value) \
         VALUES ({}, {}, {}, {})",
        ph(1),
        ph(2),
        ph(3),
        ph(4),
    );
    tx.exec(
        &sql,
        &[
            Val::Text(model_id.to_string()),
            Val::Int(experiment_id),
            Val::Text(key.to_string()),
            Val::Text(value.to_string()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

/// Upsert one logged-model tag (`session.merge(SqlLoggedModelTag(...))`).
async fn upsert_tag_tx(
    tx: &mut super::dbutil::Tx<'_>,
    dialect: Dialect,
    model_id: &str,
    experiment_id: i64,
    key: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: LOGGED_MODEL_TAGS,
        columns: &["model_id", "experiment_id", "tag_key", "tag_value"],
        pk_columns: &["model_id", "tag_key"],
        update_columns: &["tag_value"],
        ..Default::default()
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(model_id.to_string()),
            Val::Int(experiment_id),
            Val::Text(key.to_string()),
            Val::Text(value.to_string()),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

/// `Metric.__eq__` restricted to the fields that vary across elements of one
/// `log_model_metrics_tx` call (`key`, `value`, `timestamp`, `step`,
/// `model_id`, `dataset_name`, `dataset_digest`) — `run_id` is constant for
/// the whole call (it's a `log_model_metrics_tx` parameter, not a per-metric
/// field here), matching Python's per-call dedup `seen` set. `model_id` is
/// included because one call can carry metrics for several distinct models
/// (see the `log_model_metrics_tx` doc comment), so two metrics with the same
/// key/value/timestamp/step/dataset but different `model_id` are NOT
/// duplicates of each other. Plain `==` on `value` (not `to_bits()`) matches
/// Python's dict/tuple equality: `NaN != NaN`, so two NaN metrics are never
/// deduped as "the same" here either, exactly like Python's `set`-based dedup.
fn model_metric_eq(a: &super::metrics::MetricInput, b: &super::metrics::MetricInput) -> bool {
    a.key == b.key
        && a.value == b.value
        && a.timestamp == b.timestamp
        && a.step == b.step
        && a.model_id == b.model_id
        && a.dataset_name == b.dataset_name
        && a.dataset_digest == b.dataset_digest
}

/// `INSERT INTO logged_model_metrics (...) VALUES (...) ON CONFLICT/DUPLICATE
/// DO NOTHING`, conflict target = the table's 5-column PK `(model_id,
/// metric_name, metric_timestamp_ms, metric_step, run_id)`. Bind order: model_id,
/// metric_name, metric_timestamp_ms, metric_step, metric_value, experiment_id,
/// run_id, dataset_uuid, dataset_name, dataset_digest.
fn model_metric_insert_sql(dialect: Dialect) -> String {
    let spec = crate::dialect::UpsertSpec {
        table: LOGGED_MODEL_METRICS,
        columns: &[
            "model_id",
            "metric_name",
            "metric_timestamp_ms",
            "metric_step",
            "metric_value",
            "experiment_id",
            "run_id",
            "dataset_uuid",
            "dataset_name",
            "dataset_digest",
        ],
        pk_columns: &[
            "model_id",
            "metric_name",
            "metric_timestamp_ms",
            "metric_step",
            "run_id",
        ],
        update_columns: &[],
    };
    dialect.upsert(&spec)
}

// ============================================================================
// search_logged_models: filter -> SQL
// ============================================================================

fn search_err(e: mlflow_search::SearchError) -> MlflowError {
    use mlflow_error::ErrorCode;
    let code = match e.error_code {
        mlflow_search::ErrorCode::InvalidParameterValue => ErrorCode::InvalidParameterValue,
        // `PythonValueError` (bare `float()` ValueError) and the internal-error
        // dead-branch both surface as an uncaught-exception 500, matching
        // Python (neither is wrapped as an `MlflowException` there).
        _ => ErrorCode::InternalError,
    };
    MlflowError::new(e.message, code)
}

/// Positional placeholder generator (mirrors `search.rs`'s).
struct PlaceholderGen {
    dialect: Dialect,
    idx: usize,
}

impl PlaceholderGen {
    fn new(dialect: Dialect) -> Self {
        Self { dialect, idx: 0 }
    }
    fn next(&mut self) -> String {
        self.idx += 1;
        self.dialect.placeholder(self.idx)
    }
    fn next_index(&mut self) -> usize {
        self.idx += 1;
        self.idx
    }
    fn reserve_like(&mut self) -> usize {
        let first = self.next_index();
        if let Dialect::MySql = self.dialect {
            self.idx += 1;
        }
        first
    }
}

fn push_text(ph: &mut PlaceholderGen, binds: &mut Vec<Val>, s: &str) -> String {
    binds.push(Val::Text(s.to_string()));
    ph.next()
}

/// Build one filter comparison as a WHERE-clause fragment (attribute
/// predicates directly on `lm.<col>`; metric/param/tag as `EXISTS` semi-joins
/// against the child tables — see module docs for why EXISTS instead of
/// Python's literal `JOIN`).
fn build_filter_predicate(
    dialect: Dialect,
    c: &SqlaComparison,
    datasets: &[DatasetFilter],
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    match c.entity_type {
        SqlaEntityType::Attribute => {
            let col = attr_column(&c.key)?;
            value_predicate(
                dialect,
                &format!("lm.{col}"),
                &c.comparator,
                &c.value,
                ph,
                binds,
            )
        }
        SqlaEntityType::Metric => {
            let key_ph = push_text(ph, binds, &c.key);
            let num = match &c.value {
                SqlaValue::Num(n) => *n,
                _ => {
                    return Err(MlflowError::internal_error(
                        "metric comparison value must be numeric".to_string(),
                    ))
                }
            };
            let val_pred =
                value_predicate_num(dialect, "e.metric_value", &c.comparator, num, ph, binds);
            let mut dataset_pred = String::new();
            if !datasets.is_empty() {
                let clauses: Vec<String> = datasets
                    .iter()
                    .map(|d| dataset_match_clause(ph, binds, d))
                    .collect();
                dataset_pred = format!(" AND ({})", clauses.join(" OR "));
            }
            Ok(format!(
                "EXISTS (SELECT 1 FROM {LOGGED_MODEL_METRICS} e WHERE e.model_id = lm.model_id \
                 AND e.metric_name = {key_ph} AND {val_pred}{dataset_pred})"
            ))
        }
        SqlaEntityType::Param => {
            let key_ph = push_text(ph, binds, &c.key);
            let val_pred =
                value_predicate(dialect, "e.param_value", &c.comparator, &c.value, ph, binds)?;
            Ok(format!(
                "EXISTS (SELECT 1 FROM {LOGGED_MODEL_PARAMS} e WHERE e.model_id = lm.model_id \
                 AND e.param_key = {key_ph} AND {val_pred})"
            ))
        }
        SqlaEntityType::Tag => {
            let key_ph = push_text(ph, binds, &c.key);
            let val_pred =
                value_predicate(dialect, "e.tag_value", &c.comparator, &c.value, ph, binds)?;
            Ok(format!(
                "EXISTS (SELECT 1 FROM {LOGGED_MODEL_TAGS} e WHERE e.model_id = lm.model_id \
                 AND e.tag_key = {key_ph} AND {val_pred})"
            ))
        }
    }
}

/// `attributes.<key>` -> the physical `logged_models` column. Faithfully
/// leaves the dotted-form alias bug in place: callers pass whatever key the
/// parser produced (e.g. `attributes.creation_timestamp`, un-resolved), and
/// an unrecognized key here mirrors Python's uncaught `AttributeError` (bare
/// `getattr(SqlLoggedModel, comp.entity.key)`, no `try/except`) as an
/// internal (500) error rather than a 400.
fn attr_column(key: &str) -> Result<&'static str, MlflowError> {
    match key {
        "model_id" => Ok("model_id"),
        "name" => Ok("name"),
        "model_type" => Ok("model_type"),
        "status" => Ok("status"),
        "source_run_id" => Ok("source_run_id"),
        "creation_timestamp_ms" => Ok("creation_timestamp_ms"),
        "last_updated_timestamp_ms" => Ok("last_updated_timestamp_ms"),
        other => Err(MlflowError::internal_error(format!(
            "'SqlLoggedModel' object has no attribute '{other}'"
        ))),
    }
}

/// One `(dataset_name[, dataset_digest])` match clause against the metric
/// EXISTS subquery's `e` alias.
fn dataset_match_clause(
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
    d: &DatasetFilter,
) -> String {
    let name_ph = push_text(ph, binds, &d.dataset_name);
    let mut clause = format!("e.dataset_name = {name_ph}");
    if let Some(digest) = &d.dataset_digest {
        let digest_ph = push_text(ph, binds, digest);
        clause = format!("({clause} AND e.dataset_digest = {digest_ph})");
    }
    clause
}

/// "has any metric on one of the datasets" (no metric filter in the string,
/// but `datasets` was specified).
fn build_any_metric_on_datasets_predicate(
    _dialect: Dialect,
    datasets: &[DatasetFilter],
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> String {
    let clauses: Vec<String> = datasets
        .iter()
        .map(|d| dataset_match_clause(ph, binds, d))
        .collect();
    format!(
        "EXISTS (SELECT 1 FROM {LOGGED_MODEL_METRICS} e WHERE e.model_id = lm.model_id AND ({}))",
        clauses.join(" OR ")
    )
}

/// Render a comparison predicate on a text column for
/// `=,!=,<,<=,>,>=,LIKE,ILIKE,IN,NOT IN`, matching `get_sql_comparison_func`.
fn value_predicate(
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value: &SqlaValue,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    match comparator {
        "LIKE" => {
            let idx = ph.reserve_like();
            let s = as_str(value)?;
            binds.push(Val::Text(s.clone()));
            if let Dialect::MySql = dialect {
                binds.push(Val::Text(s));
            }
            Ok(dialect.case_sensitive_like(column, idx))
        }
        "ILIKE" => {
            let idx = ph.next_index();
            binds.push(Val::Text(as_str(value)?));
            Ok(dialect.case_insensitive_like(column, idx))
        }
        "IN" | "NOT IN" => {
            let items = as_list(value)?;
            if items.is_empty() {
                return Ok(if comparator == "IN" {
                    "1 = 0".to_string()
                } else {
                    "1 = 1".to_string()
                });
            }
            let phs: Vec<String> = items
                .iter()
                .map(|it| {
                    binds.push(Val::Text(it.clone()));
                    ph.next()
                })
                .collect();
            let op = if comparator == "IN" { "IN" } else { "NOT IN" };
            Ok(format!("{column} {op} ({})", phs.join(", ")))
        }
        "=" | "!=" | "<" | "<=" | ">" | ">=" => {
            let p = ph.next();
            binds.push(Val::Text(as_str(value)?));
            Ok(format!("{column} {comparator} {p}"))
        }
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid comparator '{other}'"
        ))),
    }
}

fn value_predicate_num(
    _dialect: Dialect,
    column: &str,
    comparator: &str,
    value: f64,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> String {
    let p = ph.next();
    binds.push(Val::Float(value));
    format!("{column} {comparator} {p}")
}

fn as_str(value: &SqlaValue) -> Result<String, MlflowError> {
    match value {
        SqlaValue::Str(s) => Ok(s.clone()),
        SqlaValue::Num(n) => Ok(n.to_string()),
        SqlaValue::Tuple(_) => Err(MlflowError::invalid_parameter_value(
            "Expected a string value".to_string(),
        )),
    }
}

fn as_list(value: &SqlaValue) -> Result<Vec<String>, MlflowError> {
    match value {
        SqlaValue::Tuple(items) => Ok(items.clone()),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a list value for IN/NOT IN".to_string(),
        )),
    }
}

// ============================================================================
// search_logged_models: order_by -> SQL
// ============================================================================

/// Build the `ORDER BY` clause, any extra `JOIN`s it needs (the ranked-metric
/// subqueries), and their bind values. Mirrors
/// `_apply_order_by_search_logged_models` exactly, including the
/// NULLS-LAST `CASE` emulation and the `creation_timestamp_ms DESC` tiebreak.
fn build_order_by(
    dialect: Dialect,
    order_by: &[LoggedModelOrderByInput],
    ph: &mut PlaceholderGen,
) -> Result<(String, String, Vec<Val>), MlflowError> {
    let mut order_cols: Vec<String> = Vec::new();
    let mut joins: Vec<String> = Vec::new();
    let mut binds: Vec<Val> = Vec::new();
    let mut has_creation_timestamp = false;

    for (i, ob) in order_by.iter().enumerate() {
        let field_name = resolve_order_alias(&ob.field_name);
        if !field_name.contains('.') {
            let col = attr_column(&field_name)?;
            if col == "creation_timestamp_ms" {
                has_creation_timestamp = true;
            }
            order_cols.push(format!(
                "(CASE WHEN lm.{col} IS NULL THEN 1 ELSE 0 END) ASC"
            ));
            order_cols.push(format!(
                "lm.{col} {}",
                if ob.ascending { "ASC" } else { "DESC" }
            ));
            continue;
        }

        let (_, metric_name) = field_name
            .split_once('.')
            .expect("contains '.' checked above");
        let alias = format!("oj_{i}");
        // Bind order must match the rendered SQL text order: `WHERE
        // sub.metric_name = ? AND sub.dataset_name = ? AND sub.dataset_digest
        // = ?` (see `rank_subquery`) — metric_name first, then dataset_name,
        // then dataset_digest.
        let name_ph = push_text(ph, &mut binds, metric_name);
        let mut dataset_filter = String::new();
        if let Some(name) = &ob.dataset_name {
            let name_ph = push_text(ph, &mut binds, name);
            dataset_filter.push_str(&format!(" AND sub.dataset_name = {name_ph}"));
        }
        if let Some(digest) = &ob.dataset_digest {
            let digest_ph = push_text(ph, &mut binds, digest);
            dataset_filter.push_str(&format!(" AND sub.dataset_digest = {digest_ph}"));
        }
        // Rank per (model_id) at metric_name, ordered by (timestamp DESC, step
        // DESC), restricted to the dataset filter first (mirrors the Python
        // subquery's `.filter(metric_name == ..., *dataset_filter)` applied
        // *before* the window function, then keep rank 1).
        let ranked = rank_subquery(dialect, &name_ph, &dataset_filter);
        joins.push(format!(
            "LEFT JOIN ({ranked}) {alias} ON {alias}.model_id = lm.model_id"
        ));
        order_cols.push(format!(
            "(CASE WHEN {alias}.metric_value IS NULL THEN 1 ELSE 0 END) ASC"
        ));
        order_cols.push(format!(
            "{alias}.metric_value {}",
            if ob.ascending { "ASC" } else { "DESC" }
        ));
    }

    if !has_creation_timestamp {
        order_cols.push("lm.creation_timestamp_ms DESC".to_string());
    }

    Ok((order_cols.join(", "), joins.join(" "), binds))
}

/// `SqlLoggedModel.ALIASES` applied to an order-by field name (only the
/// non-dotted form is aliased, matching `_apply_order_by_search_logged_models`
/// which does `SqlLoggedModel.ALIASES.get(field_name, field_name)` only in the
/// `"." not in field_name` branch).
fn resolve_order_alias(field_name: &str) -> String {
    if field_name.contains('.') {
        return field_name.to_string();
    }
    match field_name {
        "creation_time" | "creation_timestamp" => "creation_timestamp_ms".to_string(),
        "last_updated_timestamp" => "last_updated_timestamp_ms".to_string(),
        other => other.to_string(),
    }
}

/// The per-model ranked-metric subquery used by a `metrics.<name>` order-by
/// clause: `RANK() OVER (PARTITION BY model_id ORDER BY timestamp DESC, step
/// DESC)`, keep rank 1. Uses window functions, supported by SQLite (>= 3.25),
/// Postgres, and MySQL (>= 8.0) — the three dialects this store targets.
fn rank_subquery(_dialect: Dialect, name_ph: &str, dataset_filter: &str) -> String {
    format!(
        "SELECT model_id, metric_value FROM (\
           SELECT model_id, metric_value, \
             RANK() OVER (PARTITION BY model_id ORDER BY metric_timestamp_ms DESC, metric_step DESC) \
             AS rnk \
           FROM {LOGGED_MODEL_METRICS} sub WHERE sub.metric_name = {name_ph}{dataset_filter}\
         ) ranked WHERE ranked.rnk = 1"
    )
}

// ============================================================================
// SearchLoggedModelsPaginationToken
// ============================================================================

/// Byte-for-byte port of `SearchLoggedModelsPaginationToken`: base64(JSON)
/// `{"experiment_ids", "filter_string", "order_by", "offset"}` (Python
/// `dataclasses.asdict` always emits all four keys — no field is omitted).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PaginationToken {
    experiment_ids: Vec<String>,
    filter_string: Option<String>,
    order_by: Vec<LoggedModelOrderByInput>,
    offset: usize,
}

// serde needs (De)Serialize on LoggedModelOrderByInput; derive it alongside.
impl serde::Serialize for LoggedModelOrderByInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("field_name", &self.field_name)?;
        map.serialize_entry("ascending", &self.ascending)?;
        if let Some(n) = &self.dataset_name {
            map.serialize_entry("dataset_name", n)?;
        }
        if let Some(d) = &self.dataset_digest {
            map.serialize_entry("dataset_digest", d)?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for LoggedModelOrderByInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw {
            field_name: String,
            #[serde(default = "default_true")]
            ascending: bool,
            dataset_name: Option<String>,
            dataset_digest: Option<String>,
        }
        fn default_true() -> bool {
            true
        }
        let raw = Raw::deserialize(deserializer)?;
        Ok(LoggedModelOrderByInput {
            field_name: raw.field_name,
            ascending: raw.ascending,
            dataset_name: raw.dataset_name,
            dataset_digest: raw.dataset_digest,
        })
    }
}

impl PartialEq for LoggedModelOrderByInput {
    fn eq(&self, other: &Self) -> bool {
        self.field_name == other.field_name
            && self.ascending == other.ascending
            && self.dataset_name == other.dataset_name
            && self.dataset_digest == other.dataset_digest
    }
}

impl PaginationToken {
    fn encode(&self) -> String {
        let json = serde_json::to_string(self).expect("token serialization cannot fail");
        base64::engine::general_purpose::STANDARD.encode(json)
    }

    /// `SearchLoggedModelsPaginationToken.decode`.
    fn decode(token: &str) -> Result<Self, MlflowError> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(token.as_bytes())
            .map_err(|_| invalid_token(token))?;
        let text = String::from_utf8(bytes).map_err(|_| invalid_token(token))?;
        serde_json::from_str(&text).map_err(|_| invalid_token(token))
    }

    /// `SearchLoggedModelsPaginationToken.validate`.
    fn validate(
        &self,
        experiment_ids: &[String],
        filter_string: Option<&str>,
        order_by: &[LoggedModelOrderByInput],
    ) -> Result<(), MlflowError> {
        if self.experiment_ids != experiment_ids {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Experiment IDs in the page token do not match the requested experiment IDs. \
                 Expected: {:?}. Found: {:?}",
                experiment_ids, self.experiment_ids
            )));
        }
        if self.filter_string.as_deref() != filter_string {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Filter string in the page token does not match the requested filter string. \
                 Expected: {:?}. Found: {:?}",
                filter_string, self.filter_string
            )));
        }
        if self.order_by.as_slice() != order_by {
            return Err(MlflowError::invalid_parameter_value(
                "Order by in the page token does not match the requested order by.".to_string(),
            ));
        }
        Ok(())
    }
}

fn invalid_token(token: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!("Invalid page token: {token}."))
}
