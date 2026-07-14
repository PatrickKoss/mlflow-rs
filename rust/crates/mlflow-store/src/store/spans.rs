//! Span store operations (plan T2.11): `log_spans` bulk upsert, trace
//! time-range updates, `span_metrics` maintenance, and the lazy span-content
//! reads that back `batch_get_traces` — mirroring `log_spans` /
//! `_load_tracking_store_span_snapshots` in
//! `mlflow/store/tracking/sqlalchemy_store.py`.
//!
//! ## Store-layer contract (deviation from Python, documented)
//!
//! Python's `log_spans` receives fully-hydrated `Span` entities and performs
//! OTLP-specific work — `translate_span_when_storing`, token-usage/cost
//! aggregation, session/user-id extraction, resource-attribute → tag mapping,
//! and preview backfill — all of which is *serialization*, not SQL. In the Rust
//! split that translation belongs to the HTTP/OTLP layer (Phase 3). This module
//! takes already-prepared [`SpanInput`] rows plus a precomputed
//! [`TraceTimeRange`] per trace and performs the identical **database**
//! semantics: bulk span/metric upsert, atomic min-start / max-end trace time
//! update (skipped when `start_trace` finalized the trace), span-derived trace
//! status, and the `SPANS_LOCATION = TRACKING_STORE` tag. Token-usage/cost trace
//! metadata & metrics are written by the caller through the same sorted-key
//! upsert path as `start_trace`.
//!
//! ## Lazy content reads (commit `d5dce6e8f`)
//!
//! Reads that only need [`TraceInfo`] never select `spans.content`.
//! [`load_spans_for_traces`] is the sole span read path; it selects the full
//! span row (callers need it) but is invoked only for `TRACKING_STORE` traces,
//! and it skips rows whose `content == ""` (cleared payload), exactly like
//! `_load_tracking_store_span_snapshots`.

use std::collections::{BTreeSet, HashMap};

use mlflow_error::MlflowError;

use super::dbutil::{Tx, Val};
use super::entities::{
    StoredSpan, TraceState, SPANS_LOCATION_TRACKING_STORE, TRACE_TAG_SPANS_LOCATION,
};
use super::experiments::{internal, parse_experiment_id};
use super::TrackingStore;
use crate::dialect::Dialect;
use crate::schema::traces::{SPANS, SPAN_METRICS, TRACE_INFO, TRACE_TAGS};

/// A prepared span row to upsert (the store-level shape; the OTLP→row
/// translation is a Phase-3 concern). `duration_ns` is never set — it is a
/// generated column.
#[derive(Debug, Clone)]
pub struct SpanInput {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: Option<String>,
    pub span_type: Option<String>,
    pub status: String,
    pub start_time_unix_nano: i64,
    pub end_time_unix_nano: Option<i64>,
    /// Span JSON payload. An empty string means "cleared" (archival); reads skip
    /// such rows.
    pub content: String,
    pub dimension_attributes: Option<String>,
}

/// A prepared span-metric row (`span_metrics`).
#[derive(Debug, Clone)]
pub struct SpanMetricInput {
    pub trace_id: String,
    pub span_id: String,
    pub key: String,
    pub value: f64,
}

/// Precomputed per-trace time range + inferred status for a `log_spans` batch,
/// mirroring the pure-Python aggregate `_TraceAggregate`.
#[derive(Debug, Clone)]
pub struct TraceTimeRange {
    pub trace_id: String,
    /// `min(start_time_ns) // 1_000_000` across the batch's spans (ms).
    pub min_start_ms: i64,
    /// `max(end_time_ns) // 1_000_000` if any span has an end time (ms).
    pub max_end_ms: Option<i64>,
    /// Root-span-derived trace status (`OK`/`ERROR`), or `None` if no root span.
    pub root_span_status: Option<String>,
}

impl TrackingStore {
    /// `log_spans` — bulk-upsert `spans`/`span_metrics` for one experiment, then
    /// apply per-trace time-range/status updates and mark spans as stored in the
    /// tracking store. Missing traces are auto-created (Python's Phase 2).
    ///
    /// Wrapped in the deadlock-retry discipline (plan §4/11): trace ids are
    /// processed in sorted order and metadata/metric keys are sorted upstream so
    /// concurrent `start_trace`/`log_spans` transactions acquire PK-index locks
    /// in a consistent order.
    pub async fn log_spans(
        &self,
        workspace: &str,
        experiment_id: &str,
        spans: &[SpanInput],
        span_metrics: &[SpanMetricInput],
        time_ranges: &[TraceTimeRange],
    ) -> Result<(), MlflowError> {
        if spans.is_empty() {
            return Ok(());
        }
        self.run_with_deadlock_retry(|| {
            self.log_spans_once(workspace, experiment_id, spans, span_metrics, time_ranges)
        })
        .await
    }

    async fn log_spans_once(
        &self,
        workspace: &str,
        experiment_id: &str,
        spans: &[SpanInput],
        span_metrics: &[SpanMetricInput],
        time_ranges: &[TraceTimeRange],
    ) -> Result<(), MlflowError> {
        let exp_id = parse_experiment_id(experiment_id)?;
        let dialect = self.db().dialect();

        // Distinct trace ids in the batch (sorted for deterministic lock order).
        let trace_ids: BTreeSet<String> = spans.iter().map(|s| s.trace_id.clone()).collect();
        let ranges: HashMap<&str, &TraceTimeRange> = time_ranges
            .iter()
            .map(|r| (r.trace_id.as_str(), r))
            .collect();

        // Verify the experiment is in the workspace (mirrors get_experiment in
        // Python's Phase 2). A missing/foreign experiment errors like Python.
        self.fetch_experiment(workspace, exp_id, super::experiments::ViewType::All)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "No Experiment with id={exp_id} exists"
                ))
            })?;

        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Phase 1: which traces already exist, and are they finalized?
        let existing = fetch_existing_trace_status(&mut tx, dialect, &trace_ids).await?;

        // Phase 2: create trace_info rows for missing traces from the aggregate.
        for trace_id in &trace_ids {
            if existing.contains_key(trace_id) {
                continue;
            }
            let range = ranges.get(trace_id.as_str()).ok_or_else(|| {
                MlflowError::invalid_parameter_value(format!(
                    "Missing time range for trace '{trace_id}'"
                ))
            })?;
            create_trace_info_from_spans(&mut tx, dialect, trace_id, exp_id, range).await?;
        }

        // Phase 3: bulk upsert spans + span metrics (batched, dialect upsert).
        for span in spans {
            upsert_span(&mut tx, dialect, span, exp_id).await?;
        }
        for metric in span_metrics {
            upsert_span_metric(&mut tx, dialect, metric).await?;
        }

        // Phase 5: per-trace time-range / status updates, sorted trace ids.
        for trace_id in &trace_ids {
            let range = match ranges.get(trace_id.as_str()) {
                Some(r) => r,
                None => continue,
            };
            let finalized = existing.get(trace_id).map(|s| s.finalized).unwrap_or(false);
            apply_trace_time_range(&mut tx, dialect, trace_id, range, finalized).await?;

            // Mark span payloads as stored in the tracking store.
            upsert_spans_location_tag(&mut tx, dialect, trace_id).await?;
        }

        tx.commit().await.map_err(super::traces::map_db_err_pub)?;
        Ok(())
    }
}

/// Existing-trace status read for `log_spans` (whether the trace exists, and
/// whether `start_trace` has finalized its authoritative values).
struct TraceStatusRow {
    finalized: bool,
}

/// Batch-read which of `trace_ids` already exist and whether each carries the
/// `TRACE_INFO_FINALIZED` metadata flag.
async fn fetch_existing_trace_status(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_ids: &BTreeSet<String>,
) -> Result<HashMap<String, TraceStatusRow>, MlflowError> {
    if trace_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let ids: Vec<&String> = trace_ids.iter().collect();
    let mut binds: Vec<Val> = Vec::with_capacity(ids.len() + 1);
    // The finalized-flag placeholder appears first in the SQL text (in the
    // correlated subquery), so it must be bound first — positional placeholders
    // (`?`) bind in SQL-text order on SQLite/MySQL.
    binds.push(Val::Text(
        super::entities::TRACE_METADATA_INFO_FINALIZED.to_string(),
    ));
    let flag_ph = dialect.placeholder(1);
    let phs: Vec<String> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            binds.push(Val::Text((*id).clone()));
            dialect.placeholder(i + 2)
        })
        .collect();
    let sql = format!(
        "SELECT ti.request_id, \
         (SELECT COUNT(*) FROM trace_request_metadata m \
          WHERE m.request_id = ti.request_id AND m.\"key\" = {flag_ph}) AS finalized \
         FROM {TRACE_INFO} ti WHERE ti.request_id IN ({})",
        phs.join(", ")
    );
    let rows = tx
        .fetch_all(&sql, &binds, |r| {
            Ok((r.get_string("request_id")?, r.get_i64("finalized")? > 0))
        })
        .await
        .map_err(internal)?;
    Ok(rows
        .into_iter()
        .map(|(id, finalized)| (id, TraceStatusRow { finalized }))
        .collect())
}

/// Create a `trace_info` row for a trace observed only via its spans (Python's
/// Phase 2 `SqlTraceInfo(...)`): start = min-start, duration = max-end − min
/// -start (or NULL), status = root-span status or `IN_PROGRESS`, plus the
/// artifact-location tag.
async fn create_trace_info_from_spans(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_id: &str,
    exp_id: i64,
    range: &TraceTimeRange,
) -> Result<(), MlflowError> {
    let execution_time_ms = range.max_end_ms.map(|e| e - range.min_start_ms);
    let status = range
        .root_span_status
        .clone()
        .unwrap_or_else(|| TraceState::IN_PROGRESS.to_string());

    // Insert-or-ignore: a concurrent start_trace may have created it since our
    // existence read. ON CONFLICT DO NOTHING mirrors Python's IntegrityError
    // rollback-and-refetch loop (we simply keep the existing row).
    let spec = crate::dialect::UpsertSpec {
        table: TRACE_INFO,
        columns: &[
            "request_id",
            "experiment_id",
            "timestamp_ms",
            "execution_time_ms",
            "status",
        ],
        pk_columns: &["request_id"],
        update_columns: &[],
        ..Default::default()
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(trace_id.to_string()),
            Val::Int(exp_id),
            Val::Int(range.min_start_ms),
            Val::OptInt(execution_time_ms),
            Val::Text(status),
        ],
    )
    .await
    .map_err(super::traces::map_db_err_pub)?;
    Ok(())
}

/// Bulk-upsert one span row. On conflict every non-PK, non-generated column is
/// overwritten (`_bulk_upsert` with the computed `duration_ns` excluded).
async fn upsert_span(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    span: &SpanInput,
    exp_id: i64,
) -> Result<(), MlflowError> {
    // `type` is a keyword-ish column; the upsert helper quotes identifiers.
    let spec = crate::dialect::UpsertSpec {
        table: SPANS,
        columns: &[
            "trace_id",
            "experiment_id",
            "span_id",
            "parent_span_id",
            "name",
            "type",
            "status",
            "start_time_unix_nano",
            "end_time_unix_nano",
            "content",
            "dimension_attributes",
        ],
        pk_columns: &["trace_id", "span_id"],
        update_columns: &[
            "experiment_id",
            "parent_span_id",
            "name",
            "type",
            "status",
            "start_time_unix_nano",
            "end_time_unix_nano",
            "content",
            "dimension_attributes",
        ],
        // `dimension_attributes` is a `json`-typed column (models.py:2027);
        // Postgres refuses a bare text bind for it.
        json_columns: &["dimension_attributes"],
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(span.trace_id.clone()),
            Val::Int(exp_id),
            Val::Text(span.span_id.clone()),
            Val::OptText(span.parent_span_id.clone()),
            Val::OptText(span.name.clone()),
            Val::OptText(span.span_type.clone()),
            Val::Text(span.status.clone()),
            Val::Int(span.start_time_unix_nano),
            Val::OptInt(span.end_time_unix_nano),
            Val::Text(span.content.clone()),
            Val::OptText(span.dimension_attributes.clone()),
        ],
    )
    .await
    .map_err(super::traces::map_db_err_pub)?;
    Ok(())
}

/// Bulk-upsert one `span_metrics` row.
async fn upsert_span_metric(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    metric: &SpanMetricInput,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: SPAN_METRICS,
        columns: &["trace_id", "span_id", "key", "value"],
        pk_columns: &["trace_id", "span_id", "key"],
        update_columns: &["value"],
        ..Default::default()
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(metric.trace_id.clone()),
            Val::Text(metric.span_id.clone()),
            Val::Text(metric.key.clone()),
            Val::Float(metric.value),
        ],
    )
    .await
    .map_err(super::traces::map_db_err_pub)?;
    Ok(())
}

/// Apply the atomic min-start / max-end trace time-range update plus the
/// span-derived status transition. Skipped entirely when `start_trace` finalized
/// the trace (`TRACE_INFO_FINALIZED`), matching Python.
///
/// * `timestamp_ms := min(existing, min_start_ms)`
/// * `execution_time_ms := max(existing_end, max_end_ms) - new_timestamp` (only
///   when `max_end_ms` is present)
/// * `status := root_span_status` only when currently `IN_PROGRESS`/unspecified
async fn apply_trace_time_range(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_id: &str,
    range: &TraceTimeRange,
    finalized: bool,
) -> Result<(), MlflowError> {
    let mut sets: Vec<String> = Vec::new();
    let mut binds: Vec<Val> = Vec::new();
    let mut ph = 0usize;
    // Emit one placeholder AND push its bind. Every `?` occurrence needs its own
    // bind because SQLite/MySQL placeholders are positional (a repeated `?`
    // consumes the *next* bind, not the same one); Postgres `$N` would tolerate
    // reuse, but binding per-occurrence is correct on all three dialects.
    let mut next = |binds: &mut Vec<Val>, v: Val| {
        ph += 1;
        binds.push(v);
        dialect.placeholder(ph)
    };

    if !finalized {
        // min(existing, incoming) via CASE — two placeholders, same value.
        let a = next(&mut binds, Val::Int(range.min_start_ms));
        let b = next(&mut binds, Val::Int(range.min_start_ms));
        sets.push(format!(
            "timestamp_ms = CASE WHEN timestamp_ms > {a} THEN {b} ELSE timestamp_ms END"
        ));

        if let Some(max_end) = range.max_end_ms {
            // new_end = max(existing_end, incoming_end); duration = new_end - new_start.
            let end_a = next(&mut binds, Val::Int(max_end));
            let end_b = next(&mut binds, Val::Int(max_end));
            let start_a = next(&mut binds, Val::Int(range.min_start_ms));
            let start_b = next(&mut binds, Val::Int(range.min_start_ms));
            sets.push(format!(
                "execution_time_ms = (CASE WHEN (timestamp_ms + execution_time_ms) > {end_a} \
                 THEN (timestamp_ms + execution_time_ms) ELSE {end_b} END) - \
                 (CASE WHEN timestamp_ms > {start_a} THEN {start_b} ELSE timestamp_ms END)"
            ));
        }
    }

    // Status: only when currently IN_PROGRESS/STATE_UNSPECIFIED and a root status exists.
    if let Some(status) = &range.root_span_status {
        let in_progress = next(&mut binds, Val::Text(TraceState::IN_PROGRESS.to_string()));
        let unspecified = next(
            &mut binds,
            Val::Text(TraceState::STATE_UNSPECIFIED.to_string()),
        );
        let status_p = next(&mut binds, Val::Text(status.clone()));
        sets.push(format!(
            "status = CASE WHEN status IN ({in_progress}, {unspecified}) THEN {status_p} \
             ELSE status END"
        ));
    }

    if sets.is_empty() {
        return Ok(());
    }
    let id_p = next(&mut binds, Val::Text(trace_id.to_string()));
    let sql = format!(
        "UPDATE {TRACE_INFO} SET {} WHERE request_id = {id_p}",
        sets.join(", ")
    );
    tx.exec(&sql, &binds)
        .await
        .map_err(super::traces::map_db_err_pub)?;
    Ok(())
}

/// Upsert the `SPANS_LOCATION = TRACKING_STORE` tag for a trace.
async fn upsert_spans_location_tag(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_id: &str,
) -> Result<(), MlflowError> {
    let spec = crate::dialect::UpsertSpec {
        table: TRACE_TAGS,
        columns: &["request_id", "key", "value"],
        pk_columns: &["request_id", "key"],
        update_columns: &["value"],
        ..Default::default()
    };
    let sql = dialect.upsert(&spec);
    tx.exec(
        &sql,
        &[
            Val::Text(trace_id.to_string()),
            Val::Text(TRACE_TAG_SPANS_LOCATION.to_string()),
            Val::Text(SPANS_LOCATION_TRACKING_STORE.to_string()),
        ],
    )
    .await
    .map_err(super::traces::map_db_err_pub)?;
    Ok(())
}

/// Load DB-backed spans for the given trace ids, grouped per trace, ordered
/// root-first then by `start_time_unix_nano` (mirrors
/// `_get_spans_with_trace_info`'s sort). Rows with `content == ""` (cleared
/// payloads) are skipped — commit `d5dce6e8f` / plan T2.11.
pub(crate) async fn load_spans_for_traces(
    store: &TrackingStore,
    trace_ids: &[String],
) -> Result<HashMap<String, Vec<StoredSpan>>, MlflowError> {
    if trace_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let dialect = store.db().dialect();
    let mut binds: Vec<Val> = Vec::with_capacity(trace_ids.len());
    let phs: Vec<String> = trace_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            binds.push(Val::Text(id.clone()));
            dialect.placeholder(i + 1)
        })
        .collect();
    let sql = format!(
        "SELECT trace_id, experiment_id, span_id, parent_span_id, name, \"type\" AS span_type, \
         status, start_time_unix_nano, end_time_unix_nano, duration_ns, content, \
         dimension_attributes FROM {SPANS} WHERE trace_id IN ({}) \
         ORDER BY trace_id, (CASE WHEN parent_span_id IS NULL THEN 0 ELSE 1 END), \
         start_time_unix_nano, span_id",
        phs.join(", ")
    );
    let rows: Vec<StoredSpan> = store
        .db()
        .fetch_all(&sql, &binds, |r| {
            Ok(StoredSpan {
                trace_id: r.get_string("trace_id")?,
                experiment_id: r.get_int("experiment_id")?,
                span_id: r.get_string("span_id")?,
                parent_span_id: r.get_opt_string("parent_span_id")?,
                name: r.get_opt_string("name")?,
                span_type: r.get_opt_string("span_type")?,
                status: r.get_string("status")?,
                start_time_unix_nano: r.get_i64("start_time_unix_nano")?,
                end_time_unix_nano: r.get_opt_i64("end_time_unix_nano")?,
                duration_ns: r.get_opt_i64("duration_ns")?,
                content: r.get_string("content")?,
                dimension_attributes: r.get_opt_string("dimension_attributes")?,
            })
        })
        .await
        .map_err(internal)?;

    let mut map: HashMap<String, Vec<StoredSpan>> = HashMap::new();
    for span in rows {
        // Cleared payload (archival) — skip, matching Python.
        if span.content.is_empty() {
            continue;
        }
        map.entry(span.trace_id.clone()).or_default().push(span);
    }
    Ok(map)
}
