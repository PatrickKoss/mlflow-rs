//! Tracing V3 store operations (plan T2.10): `start_trace`, `get_trace_info`,
//! batch-get, `delete_traces`, trace-tag CRUD, and `link_traces_to_run` —
//! mirroring the trace methods of `mlflow/store/tracking/sqlalchemy_store.py`.
//!
//! ## Workspace scoping (CRITICAL, plan §3.17)
//!
//! Traces are reachable only through their experiment: every trace query joins
//! `trace_info.experiment_id IN (SELECT experiment_id FROM experiments WHERE
//! workspace = ?)`, exactly like the run semi-join in [`super::runs`]. This
//! mirrors `WorkspaceAwareSqlAlchemyStore._trace_query`.
//!
//! ## Write-ordering discipline (plan §4 item 11, commit `4c5548c39`)
//!
//! `start_trace` and `log_spans` race on the same trace's child tables. To keep
//! Postgres from deadlocking, both writers emit `trace_request_metadata` /
//! `trace_metrics` rows in **sorted key order** and trace ids in sorted order,
//! and both wrap the operation in [`TrackingStore::run_with_deadlock_retry`]
//! (2 retries, exponential backoff with jitter). Deadlocks surface as
//! `sqlx` database errors whose message contains "deadlock"; anything else
//! propagates immediately.

use std::time::Duration;

use mlflow_error::{ErrorCode, MlflowError};
use uuid::Uuid;

use super::dbutil::{Tx, Val};
use super::entities::{
    TraceAssessment, TraceInfo, TraceWithSpans, MLFLOW_ARTIFACT_LOCATION,
    TRACE_METADATA_INFO_FINALIZED,
};
use super::experiments::{internal, parse_experiment_id, ViewType};
use super::spans::load_spans_for_traces;
use super::{TrackingStore, ARTIFACTS_FOLDER_NAME};
use crate::dialect::Dialect;
use crate::schema::traces::{
    ASSESSMENTS, TRACE_INFO, TRACE_METRICS, TRACE_REQUEST_METADATA, TRACE_TAGS,
};

/// `MAX_TRACE_LINKS_PER_REQUEST` (`mlflow/store/tracking/__init__.py:30`).
pub const MAX_TRACE_LINKS_PER_REQUEST: usize = 100;

/// `_TRACE_WRITE_MAX_DEADLOCK_RETRIES` (`sqlalchemy_store.py:275`).
pub(crate) const TRACE_WRITE_MAX_DEADLOCK_RETRIES: u32 = 2;

/// Subdirectory under an experiment's artifact location that isolates trace
/// artifacts (`SqlAlchemyStore.TRACE_FOLDER_NAME`).
const TRACE_FOLDER_NAME: &str = "traces";

/// `entity_associations.source_type` / `destination_type` values
/// (`EntityAssociationType`).
pub(crate) const ENTITY_TYPE_TRACE: &str = "trace";
pub(crate) const ENTITY_TYPE_RUN: &str = "run";

/// A trace to create via [`TrackingStore::start_trace`] (the store-level shape
/// of a V3 `TraceInfo` write; the HTTP layer maps the proto onto this).
#[derive(Debug, Clone)]
pub struct StartTraceInput {
    pub trace_id: String,
    pub experiment_id: String,
    pub request_time: i64,
    pub execution_duration: Option<i64>,
    pub state: String,
    pub client_request_id: Option<String>,
    pub request_preview: Option<String>,
    pub response_preview: Option<String>,
    /// Caller-supplied tags (the artifact-location tag is added by the store).
    pub tags: Vec<(String, String)>,
    /// Caller-supplied metadata. `TRACE_INFO_FINALIZED` is added by the store.
    pub trace_metadata: Vec<(String, String)>,
    /// Token-usage-derived metrics (key → value), already parsed by the caller.
    pub trace_metrics: Vec<(String, f64)>,
}

impl TrackingStore {
    /// `start_trace` (V3). Creates the `trace_info` row plus tags, request
    /// metadata, and trace metrics, then returns the assembled [`TraceInfo`].
    ///
    /// Wrapped in the deadlock-retry discipline so a concurrent `log_spans`
    /// racing to create the same trace does not drop the write (plan §4/11).
    pub async fn start_trace(
        &self,
        workspace: &str,
        input: &StartTraceInput,
    ) -> Result<TraceInfo, MlflowError> {
        self.run_with_deadlock_retry(|| self.start_trace_once(workspace, input))
            .await
    }

    async fn start_trace_once(
        &self,
        workspace: &str,
        input: &StartTraceInput,
    ) -> Result<TraceInfo, MlflowError> {
        let exp_id = parse_experiment_id(&input.experiment_id)?;
        let experiment = self
            .fetch_experiment(workspace, exp_id, ViewType::All)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "No Experiment with id={exp_id} exists"
                ))
            })?;
        if experiment.lifecycle_stage != super::entities::LifecycleStage::ACTIVE {
            return Err(MlflowError::invalid_parameter_value(format!(
                "The experiment {} must be in the 'active' state. Current state is {}.",
                experiment.experiment_id, experiment.lifecycle_stage
            )));
        }

        let trace_id = &input.trace_id;
        let dialect = self.db().dialect();

        // Build tags: caller tags + artifact-location tag.
        let artifact_uri = append_trace_artifact_location(
            experiment.artifact_location.as_deref().unwrap_or(""),
            trace_id,
        );
        let mut tags: Vec<(String, String)> = input.tags.clone();
        tags.push((MLFLOW_ARTIFACT_LOCATION.to_string(), artifact_uri));

        // Metadata: caller metadata + TRACE_INFO_FINALIZED. Sorted for the
        // deadlock-avoidance lock ordering (§4/11).
        let mut metadata: Vec<(String, String)> = input.trace_metadata.clone();
        metadata.push((
            TRACE_METADATA_INFO_FINALIZED.to_string(),
            "true".to_string(),
        ));
        metadata.sort_by(|a, b| a.0.cmp(&b.0));

        let mut metrics: Vec<(String, f64)> = input.trace_metrics.clone();
        metrics.sort_by(|a, b| a.0.cmp(&b.0));

        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // The trace may already exist (log_spans race). Upsert the trace_info
        // row and child rows additively (Python's IntegrityError merge path):
        // start_trace holds authoritative top-level values, so we overwrite them
        // and upsert every child row rather than failing on conflict.
        upsert_trace_info(&mut tx, dialect, input, exp_id).await?;

        for (k, v) in &tags {
            upsert_trace_child(
                &mut tx,
                dialect,
                TRACE_TAGS,
                trace_id,
                k,
                Val::Text(v.clone()),
            )
            .await?;
        }
        for (k, v) in &metadata {
            upsert_trace_child(
                &mut tx,
                dialect,
                TRACE_REQUEST_METADATA,
                trace_id,
                k,
                Val::Text(v.clone()),
            )
            .await?;
        }
        for (k, v) in &metrics {
            upsert_trace_metric(&mut tx, dialect, trace_id, k, *v).await?;
        }

        tx.commit().await.map_err(map_db_err)?;

        self.get_trace_info(workspace, trace_id).await
    }

    /// `get_trace_info`: fetch a trace's [`TraceInfo`] (tags, metadata,
    /// assessments; no spans). Workspace-scoped.
    pub async fn get_trace_info(
        &self,
        workspace: &str,
        trace_id: &str,
    ) -> Result<TraceInfo, MlflowError> {
        self.fetch_trace_info(workspace, trace_id)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Trace with ID '{trace_id}' not found."
                ))
            })
    }

    /// `batch_get_trace_infos`: fetch trace infos for `trace_ids`, preserving
    /// the requested order and silently skipping ids not in the workspace
    /// (mirrors Python's `IN` + `order_by(case)`).
    pub async fn batch_get_trace_infos(
        &self,
        workspace: &str,
        trace_ids: &[String],
    ) -> Result<Vec<TraceInfo>, MlflowError> {
        if trace_ids.is_empty() {
            return Ok(vec![]);
        }
        let infos = self.fetch_trace_infos_in(workspace, trace_ids).await?;
        Ok(order_by_requested(infos, trace_ids, |i| &i.trace_id))
    }

    /// `batch_get_traces`: full traces (info + DB-backed spans), preserving
    /// requested order. Spans are loaded only for traces whose `SPANS_LOCATION`
    /// tag is `TRACKING_STORE`; cleared payloads (`content=""`) are skipped
    /// (plan T2.11). Unlike Python, this store returns every requested trace's
    /// spans as-is (it does not enforce "fully exported"; that retry lives in
    /// the HTTP layer).
    pub async fn batch_get_traces(
        &self,
        workspace: &str,
        trace_ids: &[String],
    ) -> Result<Vec<TraceWithSpans>, MlflowError> {
        if trace_ids.is_empty() {
            return Ok(vec![]);
        }
        let infos = self.batch_get_trace_infos(workspace, trace_ids).await?;
        let tracking_store_ids: Vec<String> = infos
            .iter()
            .filter(|i| {
                i.tag(super::entities::TRACE_TAG_SPANS_LOCATION)
                    == Some(super::entities::SPANS_LOCATION_TRACKING_STORE)
            })
            .map(|i| i.trace_id.clone())
            .collect();
        let mut spans_by_trace = load_spans_for_traces(self, &tracking_store_ids).await?;
        Ok(infos
            .into_iter()
            .map(|info| {
                let spans = spans_by_trace.remove(&info.trace_id).unwrap_or_default();
                TraceWithSpans { info, spans }
            })
            .collect())
    }

    /// `set_trace_tag`: upsert a trace tag. Errors if the trace is not in the
    /// workspace.
    pub async fn set_trace_tag(
        &self,
        workspace: &str,
        trace_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        self.validate_trace_accessible(workspace, trace_id).await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        upsert_trace_child(
            &mut tx,
            dialect,
            TRACE_TAGS,
            trace_id,
            key,
            Val::Text(value.to_string()),
        )
        .await?;
        tx.commit().await.map_err(internal)?;
        Ok(())
    }

    /// `delete_trace_tag`: delete a trace tag. Errors `RESOURCE_DOES_NOT_EXIST`
    /// if no such tag exists (matches Python's `deleted == 0` check).
    pub async fn delete_trace_tag(
        &self,
        workspace: &str,
        trace_id: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        self.validate_trace_accessible(workspace, trace_id).await?;
        let dialect = self.db().dialect();
        let sql = format!(
            "DELETE FROM {TRACE_TAGS} WHERE request_id = {} AND \"key\" = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        let deleted = self
            .db()
            .exec(
                &sql,
                &[Val::Text(trace_id.to_string()), Val::Text(key.to_string())],
            )
            .await
            .map_err(internal)?;
        if deleted == 0 {
            return Err(MlflowError::resource_does_not_exist(format!(
                "No trace tag with key '{key}' for trace with ID '{trace_id}'"
            )));
        }
        Ok(())
    }

    /// `link_traces_to_run`: create `entity_associations` (trace → run) for each
    /// trace id, deduplicating against existing links. At most
    /// `MAX_TRACE_LINKS_PER_REQUEST` ids; empty input is a no-op.
    pub async fn link_traces_to_run(
        &self,
        workspace: &str,
        trace_ids: &[String],
        run_id: &str,
    ) -> Result<(), MlflowError> {
        if trace_ids.is_empty() {
            return Ok(());
        }
        if trace_ids.len() > MAX_TRACE_LINKS_PER_REQUEST {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Cannot link more than {MAX_TRACE_LINKS_PER_REQUEST} traces to a run in a \
                 single request. Provided {} traces.",
                trace_ids.len()
            )));
        }
        // Validate the run is reachable in the workspace (Python's
        // `_validate_run_accessible`).
        self.resolve_run_row(workspace, run_id).await?;

        let dialect = self.db().dialect();
        // Existing (trace → run) links to dedup against.
        let existing = self.existing_trace_run_links(trace_ids, run_id).await?;

        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for trace_id in trace_ids {
            if existing.contains(trace_id) {
                continue;
            }
            let sql = format!(
                "INSERT INTO entity_associations \
                 (association_id, source_type, source_id, destination_type, destination_id) \
                 VALUES ({}, {}, {}, {}, {})",
                dialect.placeholder(1),
                dialect.placeholder(2),
                dialect.placeholder(3),
                dialect.placeholder(4),
                dialect.placeholder(5),
            );
            tx.exec(
                &sql,
                &[
                    Val::Text(Uuid::new_v4().simple().to_string()),
                    Val::Text(ENTITY_TYPE_TRACE.to_string()),
                    Val::Text(trace_id.clone()),
                    Val::Text(ENTITY_TYPE_RUN.to_string()),
                    Val::Text(run_id.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)?;
        Ok(())
    }

    /// `delete_traces` (public entry): validate the argument combination
    /// (`HasField` semantics for `max_timestamp_millis` vs `trace_ids`) then
    /// delegate to the DB-backed deletion. Returns the number of traces deleted.
    ///
    /// `max_timestamp_millis == Some(0)` is a *set* zero (delete traces at/before
    /// epoch 0), distinct from `None` (unset) — this is the plan §4/10
    /// `HasField` edge, carried as `Option<i64>` here.
    pub async fn delete_traces(
        &self,
        workspace: &str,
        experiment_id: &str,
        max_timestamp_millis: Option<i64>,
        max_traces: Option<i64>,
        trace_ids: Option<&[String]>,
    ) -> Result<u64, MlflowError> {
        let has_trace_ids = trace_ids.map(|t| !t.is_empty()).unwrap_or(false);
        if max_timestamp_millis.is_none() && !has_trace_ids {
            return Err(MlflowError::invalid_parameter_value(
                "Either `max_timestamp_millis` or `trace_ids` must be specified.",
            ));
        }
        if max_timestamp_millis.is_some() && has_trace_ids {
            return Err(MlflowError::invalid_parameter_value(
                "Only one of `max_timestamp_millis` and `trace_ids` can be specified.",
            ));
        }
        if has_trace_ids && max_traces.is_some() {
            return Err(MlflowError::invalid_parameter_value(
                "`max_traces` can't be specified if `trace_ids` is specified.",
            ));
        }
        if let Some(mt) = max_traces {
            if mt <= 0 {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "`max_traces` must be a positive integer, received {mt}."
                )));
            }
        }
        self.delete_traces_impl(
            workspace,
            experiment_id,
            max_timestamp_millis,
            max_traces,
            trace_ids,
        )
        .await
    }

    /// `_delete_traces` (DB-backed subset). Selects trace ids matching the
    /// filters in deterministic `(timestamp_ms, request_id)` order (deadlock
    /// discipline), then deletes them; FK `ON DELETE CASCADE` removes tags,
    /// metadata, metrics, spans, span_metrics, and assessments.
    ///
    /// Parity note: Python's `_delete_traces` deletes only `trace_info` (relying
    /// on FK cascade) and explicitly clears `review_queue_items`; it leaves
    /// `entity_associations` orphaned (that table has no FK to `trace_info`). We
    /// match Python exactly — associations are NOT removed here. Review-queue
    /// items are a genai-only table (Phase 12) not owned by this store.
    ///
    /// Archive-backed traces (`SPANS_LOCATION == ARCHIVE_REPO`) require object
    /// -store payload cleanup and are handled in Phase 4; this DB-backed path
    /// covers the tracking-store case.
    async fn delete_traces_impl(
        &self,
        workspace: &str,
        experiment_id: &str,
        max_timestamp_millis: Option<i64>,
        max_traces: Option<i64>,
        trace_ids: Option<&[String]>,
    ) -> Result<u64, MlflowError> {
        let exp_id = parse_experiment_id(experiment_id)?;
        let dialect = self.db().dialect();

        // Build the selection query, workspace-scoped via the experiment
        // semi-join.
        let mut binds: Vec<Val> = Vec::new();
        let mut ph = 0usize;
        let mut next = |binds: &mut Vec<Val>, v: Val| {
            ph += 1;
            binds.push(v);
            dialect.placeholder(ph)
        };

        let exp_ph = next(&mut binds, Val::Int(exp_id));
        let mut wheres = vec![format!("ti.experiment_id = {exp_ph}")];
        // Workspace semi-join.
        let ws_ph = next(&mut binds, Val::Text(workspace.to_string()));
        wheres.push(format!(
            "ti.experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = {ws_ph})"
        ));
        if let Some(ts) = max_timestamp_millis {
            let p = next(&mut binds, Val::Int(ts));
            wheres.push(format!("ti.timestamp_ms <= {p}"));
        }
        if let Some(ids) = trace_ids {
            if ids.is_empty() {
                return Ok(0);
            }
            let phs: Vec<String> = ids
                .iter()
                .map(|id| next(&mut binds, Val::Text(id.clone())))
                .collect();
            wheres.push(format!("ti.request_id IN ({})", phs.join(", ")));
        }

        let mut sql = format!(
            "SELECT ti.request_id FROM {TRACE_INFO} ti WHERE {} \
             ORDER BY ti.timestamp_ms, ti.request_id",
            wheres.join(" AND ")
        );
        if let Some(mt) = max_traces {
            sql.push_str(&format!(" LIMIT {mt}"));
        }

        let selected: Vec<String> = self
            .db()
            .fetch_all(&sql, &binds, |r| r.get_string("request_id"))
            .await
            .map_err(internal)?;
        if selected.is_empty() {
            return Ok(0);
        }

        // Delete the trace_info rows; FK cascades handle the child tables
        // (tags, metadata, metrics, spans, span_metrics, assessments). Matches
        // Python: entity_associations are intentionally left orphaned.
        let (del_sql, del_binds) = in_delete(dialect, TRACE_INFO, "request_id", &selected);
        let deleted = self
            .db()
            .exec(&del_sql, &del_binds)
            .await
            .map_err(internal)?;
        Ok(deleted)
    }

    // ---- internal helpers ----

    /// Validate a trace is reachable in the workspace, erroring
    /// `RESOURCE_DOES_NOT_EXIST` otherwise (Python's `_validate_trace_accessible`
    /// + the workspace semi-join).
    pub(crate) async fn validate_trace_accessible(
        &self,
        workspace: &str,
        trace_id: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT 1 AS present FROM {TRACE_INFO} ti WHERE ti.request_id = {} \
             AND ti.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        let found = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(trace_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |r| r.get_i64("present"),
            )
            .await
            .map_err(internal)?;
        if found.is_none() {
            return Err(MlflowError::resource_does_not_exist(format!(
                "Trace with ID '{trace_id}' not found."
            )));
        }
        Ok(())
    }

    /// Fetch one trace's [`TraceInfo`] (or `None`), workspace-scoped.
    pub(crate) async fn fetch_trace_info(
        &self,
        workspace: &str,
        trace_id: &str,
    ) -> Result<Option<TraceInfo>, MlflowError> {
        let ids = [trace_id.to_string()];
        Ok(self.fetch_trace_infos_in(workspace, &ids).await?.pop())
    }

    /// Fetch trace infos for `trace_ids` (unordered), workspace-scoped, with
    /// tags/metadata/assessments loaded via batched `IN` queries.
    async fn fetch_trace_infos_in(
        &self,
        workspace: &str,
        trace_ids: &[String],
    ) -> Result<Vec<TraceInfo>, MlflowError> {
        let dialect = self.db().dialect();
        let mut binds: Vec<Val> = Vec::with_capacity(trace_ids.len() + 1);
        let phs: Vec<String> = trace_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                binds.push(Val::Text(id.clone()));
                dialect.placeholder(i + 1)
            })
            .collect();
        binds.push(Val::Text(workspace.to_string()));
        let sql = format!(
            "SELECT ti.request_id, ti.experiment_id, ti.timestamp_ms, ti.execution_time_ms, \
             ti.status, ti.client_request_id, ti.request_preview, ti.response_preview \
             FROM {TRACE_INFO} ti WHERE ti.request_id IN ({}) AND ti.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            phs.join(", "),
            dialect.placeholder(trace_ids.len() + 1),
        );
        let mut infos: Vec<TraceInfo> = self
            .db()
            .fetch_all(&sql, &binds, |r| {
                Ok(TraceInfo {
                    trace_id: r.get_string("request_id")?,
                    experiment_id: r.get_int("experiment_id")?.to_string(),
                    request_time: r.get_i64("timestamp_ms")?,
                    execution_duration: r.get_opt_i64("execution_time_ms")?,
                    state: r.get_string("status")?,
                    client_request_id: r.get_opt_string("client_request_id")?,
                    request_preview: r.get_opt_string("request_preview")?,
                    response_preview: r.get_opt_string("response_preview")?,
                    tags: Vec::new(),
                    trace_metadata: Vec::new(),
                    assessments: Vec::new(),
                })
            })
            .await
            .map_err(internal)?;
        if infos.is_empty() {
            return Ok(vec![]);
        }

        let present_ids: Vec<String> = infos.iter().map(|i| i.trace_id.clone()).collect();
        let tags = self.fetch_kv_in(TRACE_TAGS, &present_ids).await?;
        let metadata = self
            .fetch_kv_in(TRACE_REQUEST_METADATA, &present_ids)
            .await?;
        let assessments = self.fetch_assessments_in(&present_ids).await?;

        for info in &mut infos {
            info.tags = tags
                .iter()
                .filter(|(rid, _, _)| rid == &info.trace_id)
                .map(|(_, k, v)| (k.clone(), v.clone()))
                .collect();
            info.trace_metadata = metadata
                .iter()
                .filter(|(rid, _, _)| rid == &info.trace_id)
                .map(|(_, k, v)| (k.clone(), v.clone()))
                .collect();
            info.assessments = assessments
                .iter()
                .filter(|a| a.trace_id == info.trace_id)
                .cloned()
                .collect();
        }
        Ok(infos)
    }

    /// Fetch `(request_id, key, value)` rows for a KV child table, ordered by
    /// `(request_id, key)` for determinism.
    async fn fetch_kv_in(
        &self,
        table: &str,
        trace_ids: &[String],
    ) -> Result<Vec<(String, String, Option<String>)>, MlflowError> {
        let dialect = self.db().dialect();
        let mut binds = Vec::with_capacity(trace_ids.len());
        let phs: Vec<String> = trace_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                binds.push(Val::Text(id.clone()));
                dialect.placeholder(i + 1)
            })
            .collect();
        let sql = format!(
            "SELECT request_id, \"key\", value FROM {table} WHERE request_id IN ({}) \
             ORDER BY request_id, \"key\"",
            phs.join(", ")
        );
        self.db()
            .fetch_all(&sql, &binds, |r| {
                Ok((
                    r.get_string("request_id")?,
                    r.get_string("key")?,
                    r.get_opt_string("value")?,
                ))
            })
            .await
            .map_err(internal)
    }

    /// Fetch assessments for `trace_ids`, ordered by `(trace_id,
    /// created_timestamp)`.
    async fn fetch_assessments_in(
        &self,
        trace_ids: &[String],
    ) -> Result<Vec<TraceAssessment>, MlflowError> {
        let dialect = self.db().dialect();
        let mut binds = Vec::with_capacity(trace_ids.len());
        let phs: Vec<String> = trace_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                binds.push(Val::Text(id.clone()));
                dialect.placeholder(i + 1)
            })
            .collect();
        let sql = format!(
            "SELECT assessment_id, trace_id, name, assessment_type, value, error, \
             created_timestamp, last_updated_timestamp, source_type, source_id, run_id, \
             span_id, rationale, overrides, valid, assessment_metadata \
             FROM {ASSESSMENTS} WHERE trace_id IN ({}) \
             ORDER BY trace_id, created_timestamp, assessment_id",
            phs.join(", ")
        );
        self.db()
            .fetch_all(&sql, &binds, |r| {
                Ok(TraceAssessment {
                    assessment_id: r.get_string("assessment_id")?,
                    trace_id: r.get_string("trace_id")?,
                    name: r.get_string("name")?,
                    assessment_type: r.get_string("assessment_type")?,
                    value: r.get_string("value")?,
                    error: r.get_opt_string("error")?,
                    created_timestamp: r.get_i64("created_timestamp")?,
                    last_updated_timestamp: r.get_i64("last_updated_timestamp")?,
                    source_type: r.get_string("source_type")?,
                    source_id: r.get_opt_string("source_id")?,
                    run_id: r.get_opt_string("run_id")?,
                    span_id: r.get_opt_string("span_id")?,
                    rationale: r.get_opt_string("rationale")?,
                    overrides: r.get_opt_string("overrides")?,
                    valid: r.get_bool("valid")?,
                    metadata: r.get_opt_string("assessment_metadata")?,
                })
            })
            .await
            .map_err(internal)
    }

    /// Existing (trace → run) association source ids, for link dedup.
    async fn existing_trace_run_links(
        &self,
        trace_ids: &[String],
        run_id: &str,
    ) -> Result<std::collections::HashSet<String>, MlflowError> {
        let dialect = self.db().dialect();
        let mut binds: Vec<Val> = Vec::with_capacity(trace_ids.len() + 3);
        binds.push(Val::Text(ENTITY_TYPE_TRACE.to_string()));
        binds.push(Val::Text(ENTITY_TYPE_RUN.to_string()));
        binds.push(Val::Text(run_id.to_string()));
        let phs: Vec<String> = trace_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                binds.push(Val::Text(id.clone()));
                dialect.placeholder(i + 4)
            })
            .collect();
        let sql = format!(
            "SELECT source_id FROM entity_associations \
             WHERE source_type = {} AND destination_type = {} AND destination_id = {} \
             AND source_id IN ({})",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            phs.join(", "),
        );
        let rows: Vec<String> = self
            .db()
            .fetch_all(&sql, &binds, |r| r.get_string("source_id"))
            .await
            .map_err(internal)?;
        Ok(rows.into_iter().collect())
    }

    /// Run a trace-write closure, retrying on DB deadlocks
    /// (`_run_with_deadlock_retry`, `sqlalchemy_store.py:3469`). Retries up to
    /// [`TRACE_WRITE_MAX_DEADLOCK_RETRIES`] times with exponential backoff +
    /// jitter; only errors whose message contains "deadlock" are retried.
    pub(crate) async fn run_with_deadlock_retry<T, F, Fut>(
        &self,
        mut op: F,
    ) -> Result<T, MlflowError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, MlflowError>>,
    {
        let mut attempt: u32 = 0;
        loop {
            match op().await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let is_deadlock = e.message.to_lowercase().contains("deadlock");
                    if !is_deadlock || attempt >= TRACE_WRITE_MAX_DEADLOCK_RETRIES {
                        return Err(e);
                    }
                    // Exponential backoff with jitter (matches Python's
                    // `(2**attempt) - 1 + uniform(0,1)` in seconds), scaled down
                    // so tests stay fast while keeping the shape.
                    let base = (1u64 << attempt).saturating_sub(1);
                    let jitter = pseudo_jitter_ms();
                    tokio::time::sleep(Duration::from_millis(base * 100 + jitter)).await;
                    attempt += 1;
                }
            }
        }
    }
}

/// Map a `sqlx` error to an `MlflowError`, tagging deadlocks so the retry
/// wrapper recognizes them (mirrors Python surfacing DeadlockDetected as
/// `TEMPORARILY_UNAVAILABLE` with a "deadlock" message).
fn map_db_err(e: sqlx::Error) -> MlflowError {
    let msg = e.to_string();
    if msg.to_lowercase().contains("deadlock") {
        MlflowError::new(msg, ErrorCode::TemporarilyUnavailable)
    } else {
        internal(e)
    }
}

/// Crate-visible alias so the spans module shares the deadlock-tagging mapper.
pub(crate) fn map_db_err_pub(e: sqlx::Error) -> MlflowError {
    map_db_err(e)
}

/// Build the trace artifact-location URI
/// (`experiment_artifact/traces/<trace_id>/artifacts`).
fn append_trace_artifact_location(experiment_artifact: &str, trace_id: &str) -> String {
    super::uri_util::append_to_uri_path(
        experiment_artifact,
        &[TRACE_FOLDER_NAME, trace_id, ARTIFACTS_FOLDER_NAME],
    )
}

/// Upsert the `trace_info` row. On conflict (a concurrent `log_spans` created
/// the trace) `start_trace` holds authoritative top-level values and overwrites
/// them — except `request_preview`/`response_preview`, which are preserved when
/// the incoming value is NULL, mirroring Python's
/// `if trace_info.request_preview is not None:` guard on the merge path. This is
/// expressed as `COALESCE(excluded.<col>, <col>)` so a None input never clears a
/// preview that `log_spans` backfilled.
async fn upsert_trace_info(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    input: &StartTraceInput,
    exp_id: i64,
) -> Result<(), MlflowError> {
    let cols = [
        "request_id",
        "experiment_id",
        "timestamp_ms",
        "execution_time_ms",
        "status",
        "client_request_id",
        "request_preview",
        "response_preview",
    ];
    let placeholders: Vec<String> = (1..=cols.len()).map(|i| dialect.placeholder(i)).collect();
    let quoted: Vec<String> = cols.iter().map(|c| dialect.quote_ident(c)).collect();
    let insert = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        dialect.quote_ident(TRACE_INFO),
        quoted.join(", "),
        placeholders.join(", ")
    );
    // Overwrite columns on conflict; previews use COALESCE to preserve existing.
    let overwrite = [
        "experiment_id",
        "timestamp_ms",
        "execution_time_ms",
        "status",
        "client_request_id",
    ];
    let sql = match dialect {
        Dialect::Sqlite | Dialect::Postgres => {
            let mut sets: Vec<String> = overwrite
                .iter()
                .map(|c| {
                    let q = dialect.quote_ident(c);
                    format!("{q} = excluded.{q}")
                })
                .collect();
            for c in ["request_preview", "response_preview"] {
                let q = dialect.quote_ident(c);
                sets.push(format!("{q} = COALESCE(excluded.{q}, {q})"));
            }
            format!(
                "{insert} ON CONFLICT ({}) DO UPDATE SET {}",
                dialect.quote_ident("request_id"),
                sets.join(", ")
            )
        }
        Dialect::MySql => {
            let mut sets: Vec<String> = overwrite
                .iter()
                .map(|c| {
                    let q = dialect.quote_ident(c);
                    format!("{q} = VALUES({q})")
                })
                .collect();
            for c in ["request_preview", "response_preview"] {
                let q = dialect.quote_ident(c);
                sets.push(format!("{q} = COALESCE(VALUES({q}), {q})"));
            }
            format!("{insert} ON DUPLICATE KEY UPDATE {}", sets.join(", "))
        }
    };
    tx.exec(
        &sql,
        &[
            Val::Text(input.trace_id.clone()),
            Val::Int(exp_id),
            Val::Int(input.request_time),
            Val::OptInt(input.execution_duration),
            Val::Text(input.state.clone()),
            Val::OptText(input.client_request_id.clone()),
            Val::OptText(input.request_preview.clone()),
            Val::OptText(input.response_preview.clone()),
        ],
    )
    .await
    .map_err(map_db_err)?;
    Ok(())
}

/// Upsert a `(request_id, key, value)` row into a trace KV child table.
async fn upsert_trace_child(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    table: &str,
    trace_id: &str,
    key: &str,
    value: Val,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table,
        columns: &["request_id", "key", "value"],
        pk_columns: &["request_id", "key"],
        update_columns: &["value"],
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(trace_id.to_string()),
            Val::Text(key.to_string()),
            value,
        ],
    )
    .await
    .map_err(map_db_err)?;
    Ok(())
}

/// Upsert a `trace_metrics` row.
async fn upsert_trace_metric(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_id: &str,
    key: &str,
    value: f64,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: TRACE_METRICS,
        columns: &["request_id", "key", "value"],
        pk_columns: &["request_id", "key"],
        update_columns: &["value"],
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(trace_id.to_string()),
            Val::Text(key.to_string()),
            Val::Float(value),
        ],
    )
    .await
    .map_err(map_db_err)?;
    Ok(())
}

/// Build a `DELETE ... WHERE col IN (...)` with positional binds.
fn in_delete(dialect: Dialect, table: &str, col: &str, ids: &[String]) -> (String, Vec<Val>) {
    let mut binds = Vec::with_capacity(ids.len());
    let phs: Vec<String> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            binds.push(Val::Text(id.clone()));
            dialect.placeholder(i + 1)
        })
        .collect();
    (
        format!("DELETE FROM {table} WHERE {col} IN ({})", phs.join(", ")),
        binds,
    )
}

/// Reorder `items` to match the order of `requested`, skipping items whose key
/// is absent (Python's `order_by(case(...))` over an `IN` fetch).
fn order_by_requested<T, F>(items: Vec<T>, requested: &[String], key: F) -> Vec<T>
where
    F: Fn(&T) -> &String,
{
    let mut by_id: std::collections::HashMap<String, T> =
        items.into_iter().map(|i| (key(&i).clone(), i)).collect();
    requested.iter().filter_map(|id| by_id.remove(id)).collect()
}

/// Cheap non-cryptographic jitter in `[0, 100)` ms derived from the wall clock,
/// so retries don't thundering-herd. (The magnitude is small; correctness does
/// not depend on it, only deadlock-recovery latency does.)
fn pseudo_jitter_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| (d.subsec_nanos() as u64) % 100)
        .unwrap_or(0)
}
