//! Trace analytics store methods (plan T4.1, §3.6):
//! `calculate_trace_filter_correlation` (NPMI) and `query_trace_metrics`
//! (aggregations over traces/spans/assessments) — mirroring
//! `SqlAlchemyStore.calculate_trace_filter_correlation`
//! (`sqlalchemy_store.py:4814`) and `query_trace_metrics` (`:4139` →
//! `mlflow/store/tracking/utils/sql_trace_metrics_utils.py`).
//!
//! Both reuse the trace-search filter machinery in [`super::traces_search`] so
//! filter semantics are byte-identical with `search_traces`.
//!
//! ## Scope (documented deviations)
//!
//! `query_trace_metrics` implements the **core** aggregations Python supports
//! and defers the advanced surface. See [`TrackingStore::query_trace_metrics`]
//! for the precise supported/deferred matrix.

use std::collections::BTreeMap;

use mlflow_error::MlflowError;

use super::dbutil::Val;
use super::experiments::internal;
use super::trace_correlation::{calculate_npmi_from_counts, NpmiResult, TraceCorrelationCounts};
use super::traces_search::{build_trace_filter_wheres, Ph};
use super::TrackingStore;
use crate::dialect::Dialect;
use crate::schema::traces::{ASSESSMENTS, SPANS, SPAN_METRICS, TRACE_INFO, TRACE_TAGS};

/// `MAX_RESULTS_QUERY_TRACE_METRICS` (`sqlalchemy_store` default of 1000).
pub const MAX_RESULTS_QUERY_TRACE_METRICS: i64 = 1000;

/// The result of [`TrackingStore::calculate_trace_filter_correlation`], mapped
/// onto the `CalculateTraceFilterCorrelation.Response` proto by the HTTP layer
/// (`TraceFilterCorrelationResult`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceFilterCorrelationResult {
    pub npmi: f64,
    pub npmi_smoothed: f64,
    pub filter1_count: i64,
    pub filter2_count: i64,
    pub joint_count: i64,
    pub total_count: i64,
}

/// The view over which [`TrackingStore::query_trace_metrics`] aggregates
/// (`MetricViewType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricViewType {
    Traces,
    Spans,
    Assessments,
}

impl MetricViewType {
    /// The wire string used in error messages (`view_type.value`).
    pub fn as_str(self) -> &'static str {
        match self {
            MetricViewType::Traces => "TRACES",
            MetricViewType::Spans => "SPANS",
            MetricViewType::Assessments => "ASSESSMENTS",
        }
    }
}

/// One aggregation to apply, plus its output label
/// (`str(MetricAggregation)`: `"COUNT"`, `"SUM"`, `"AVG"`, `"P95.0"`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricAggregation {
    Count,
    Sum,
    Avg,
    /// Percentile with the requested value (0-100); `label()` renders `P<v>`.
    Percentile(f64),
}

impl MetricAggregation {
    /// `str(agg)` — the key used in the output `values` map.
    pub fn label(self) -> String {
        match self {
            MetricAggregation::Count => "COUNT".to_string(),
            MetricAggregation::Sum => "SUM".to_string(),
            MetricAggregation::Avg => "AVG".to_string(),
            // Python renders the float repr, e.g. 95 -> "P95.0".
            MetricAggregation::Percentile(v) => format!("P{}", format_percentile(v)),
        }
    }

    /// The aggregation-type name for validation error messages.
    fn type_name(self) -> &'static str {
        match self {
            MetricAggregation::Count => "COUNT",
            MetricAggregation::Sum => "SUM",
            MetricAggregation::Avg => "AVG",
            MetricAggregation::Percentile(_) => "PERCENTILE",
        }
    }
}

/// Format a percentile value like Python's `float` repr for the `P{value}`
/// label: integral values get a trailing `.0` (`95` -> `95.0`), fractional
/// values print their decimal form (`99.9` -> `99.9`).
fn format_percentile(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.1}")
    } else {
        // `{}` on f64 already trims trailing zeros and matches Python repr for
        // the common cases (99.9, 99.5, ...).
        format!("{v}")
    }
}

/// One aggregated output row (`MetricDataPoint`): the metric name, the
/// dimension key→value map, and the aggregation label→value map.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDataPoint {
    pub metric_name: String,
    pub dimensions: BTreeMap<String, String>,
    pub values: BTreeMap<String, f64>,
}

impl TrackingStore {
    /// `calculate_trace_filter_correlation` (`sqlalchemy_store.py:4814`).
    ///
    /// Computes NPMI between two trace filters over the workspace-scoped
    /// experiments. When `base_filter` is set, both filters are ANDed onto it
    /// (`"{base} and {f}"`) and the universe (`total_count`) is the set of
    /// traces matching `base_filter`; otherwise the universe is all traces in
    /// the experiments. The four counts are gathered in a single query using
    /// `LEFT JOIN` subqueries (MSSQL-safe, matching Python).
    pub async fn calculate_trace_filter_correlation(
        &self,
        workspace: &str,
        experiment_ids: &[String],
        filter_string1: &str,
        filter_string2: &str,
        base_filter: Option<&str>,
    ) -> Result<TraceFilterCorrelationResult, MlflowError> {
        let dialect = self.db().dialect();
        let exp_ids = self
            .filter_trace_experiment_ids(workspace, experiment_ids)
            .await?;
        if exp_ids.is_empty() {
            let counts = TraceCorrelationCounts {
                total_count: 0,
                filter1_count: 0,
                filter2_count: 0,
                joint_count: 0,
            };
            return Ok(to_correlation_result(counts));
        }

        let filter1_combined = combine_filter(base_filter, filter_string1);
        let filter2_combined = combine_filter(base_filter, filter_string2);

        let mut binds: Vec<Val> = Vec::new();
        let mut ph = Ph::new(dialect);

        // Build subqueries in the same textual order they appear in the final
        // SQL (base, then f1, then f2) so positional `?` / `$N` placeholders
        // bind in order. `base` is either the base-filter subquery or all
        // traces in the experiments.
        let base_filter_ref = base_filter.filter(|s| !s.is_empty());
        let base_sub =
            self.filter_subquery_sql(dialect, &exp_ids, base_filter_ref, &mut ph, &mut binds)?;
        let base_from = format!("({base_sub}) base");
        let base_request_id = "base.request_id".to_string();

        let f1_sub = self.filter_subquery_sql(
            dialect,
            &exp_ids,
            Some(&filter1_combined),
            &mut ph,
            &mut binds,
        )?;
        let f2_sub = self.filter_subquery_sql(
            dialect,
            &exp_ids,
            Some(&filter2_combined),
            &mut ph,
            &mut binds,
        )?;

        let sql = format!(
            "SELECT \
             COUNT({base_request_id}) AS total, \
             COUNT(f1.request_id) AS filter1, \
             COUNT(f2.request_id) AS filter2, \
             COUNT(CASE WHEN f1.request_id IS NOT NULL AND f2.request_id IS NOT NULL \
             THEN {base_request_id} ELSE NULL END) AS joint \
             FROM {base_from} \
             LEFT JOIN ({f1_sub}) f1 ON {base_request_id} = f1.request_id \
             LEFT JOIN ({f2_sub}) f2 ON {base_request_id} = f2.request_id"
        );

        let counts = self
            .db()
            .fetch_optional(&sql, &binds, |r| {
                Ok(TraceCorrelationCounts {
                    total_count: r.get_opt_i64("total")?.unwrap_or(0),
                    filter1_count: r.get_opt_i64("filter1")?.unwrap_or(0),
                    filter2_count: r.get_opt_i64("filter2")?.unwrap_or(0),
                    joint_count: r.get_opt_i64("joint")?.unwrap_or(0),
                })
            })
            .await
            .map_err(internal)?
            .unwrap_or(TraceCorrelationCounts {
                total_count: 0,
                filter1_count: 0,
                filter2_count: 0,
                joint_count: 0,
            });

        Ok(to_correlation_result(counts))
    }

    /// Build a `SELECT request_id FROM trace_info ti WHERE <exp> [AND <filter>]`
    /// subquery, workspace already resolved to `exp_ids`. Shares the filter
    /// clauses with `search_traces`.
    fn filter_subquery_sql(
        &self,
        dialect: Dialect,
        exp_ids: &[i64],
        filter: Option<&str>,
        ph: &mut Ph,
        binds: &mut Vec<Val>,
    ) -> Result<String, MlflowError> {
        let exp_phs: Vec<String> = exp_ids
            .iter()
            .map(|id| ph.next(binds, Val::Int(*id)))
            .collect();
        let mut wheres = vec![format!("ti.experiment_id IN ({})", exp_phs.join(", "))];
        wheres.extend(build_trace_filter_wheres(dialect, filter, ph, binds)?);
        Ok(format!(
            "SELECT ti.request_id FROM {TRACE_INFO} ti WHERE {}",
            wheres.join(" AND ")
        ))
    }

    /// `query_trace_metrics` (`sqlalchemy_store.py:4139` +
    /// `sql_trace_metrics_utils.query_metrics`).
    ///
    /// ## Supported (this port)
    ///
    /// * **Views**: `TRACES`, `SPANS`, `ASSESSMENTS` — base joins per view.
    /// * **Metrics**: TRACES `trace_count`/`latency`/token metrics
    ///   (`input_tokens`, `output_tokens`, `total_tokens`,
    ///   `cache_read_input_tokens`, `cache_creation_input_tokens`),
    ///   `session_count`; SPANS `span_count`/`latency`/cost metrics
    ///   (`input_cost`/`output_cost`/`total_cost`); ASSESSMENTS
    ///   `assessment_count`/`assessment_value`.
    /// * **Aggregations**: `COUNT`, `SUM`, `AVG` (plain SQL, dialect-uniform).
    /// * **Dimensions**: all documented dimensions per view (`trace_status`,
    ///   `trace_name`, `span_name`/`span_type`/`span_status`, span model
    ///   name/provider JSON extraction, `assessment_name`/`assessment_value`).
    /// * **Filters**: `filters` list via the shared trace-filter machinery
    ///   (trace status/tag/metadata; span/assessment via view joins).
    /// * Experiment-id + `start_time_ms`/`end_time_ms` bounds on the trace rows.
    /// * Output row dropping (any NULL dimension → skip row; empty values →
    ///   skip row); value keys = aggregation labels.
    ///
    /// ## Deferred (documented; see final task report)
    ///
    /// * **PERCENTILE** aggregation — Python has four dialect-specific
    ///   implementations (Postgres ordered-set, SQLite UDF, MSSQL/MySQL window
    ///   subqueries). Requesting a percentile returns `INVALID_PARAMETER_VALUE`
    ///   ("percentile aggregation is not yet supported by this server").
    /// * **`time_interval_seconds` time bucketing** — floor arithmetic +
    ///   ISO-8601 bucket labels. Requesting it returns `INVALID_PARAMETER_VALUE`.
    /// * Real pagination (`page_token`) — Python itself never implemented it
    ///   (always returns `next_page_token=None`), so this matches Python.
    #[allow(clippy::too_many_arguments)]
    pub async fn query_trace_metrics(
        &self,
        workspace: &str,
        experiment_ids: &[String],
        view_type: MetricViewType,
        metric_name: &str,
        aggregations: &[MetricAggregation],
        dimensions: &[String],
        filters: &[String],
        time_interval_seconds: Option<i64>,
        start_time_ms: Option<i64>,
        end_time_ms: Option<i64>,
        max_results: i64,
    ) -> Result<Vec<MetricDataPoint>, MlflowError> {
        validate_query_trace_metrics_params(view_type, metric_name, aggregations, dimensions)?;

        if time_interval_seconds.is_some() && (start_time_ms.is_none() || end_time_ms.is_none()) {
            return Err(MlflowError::invalid_parameter_value(
                "start_time_ms and end_time_ms are required if time_interval_seconds is set",
            ));
        }
        if time_interval_seconds.is_some() {
            return Err(MlflowError::invalid_parameter_value(
                "time_interval_seconds bucketing is not yet supported by this server",
            ));
        }
        if aggregations
            .iter()
            .any(|a| matches!(a, MetricAggregation::Percentile(_)))
        {
            return Err(MlflowError::invalid_parameter_value(
                "PERCENTILE aggregation is not yet supported by this server",
            ));
        }

        let dialect = self.db().dialect();
        let exp_ids = self
            .filter_trace_experiment_ids(workspace, experiment_ids)
            .await?;
        if exp_ids.is_empty() {
            return Ok(vec![]);
        }

        let plan = build_metrics_plan(dialect, view_type, metric_name, dimensions)?;

        let mut binds: Vec<Val> = Vec::new();
        let mut ph = Ph::new(dialect);

        // FROM + view join.
        let mut sql = String::from("SELECT ");

        // SELECT columns: dimensions (labeled) then aggregations (labeled).
        let mut select_cols: Vec<String> = Vec::new();
        for d in &plan.dimensions {
            select_cols.push(format!("{} AS {}", d.expr, d.label));
        }
        for agg in aggregations {
            let expr = aggregation_sql(*agg, &plan.agg_column);
            // The `values` proto map is double-typed; COUNT returns an integer
            // on most backends, so cast every aggregation to float for a uniform
            // f64 decode (and to match the double wire type).
            select_cols.push(format!(
                "CAST({} AS {}) AS {}",
                expr,
                float_type(dialect),
                agg.label_sql_alias()
            ));
        }
        sql.push_str(&select_cols.join(", "));
        sql.push_str(&format!(" FROM {TRACE_INFO} ti"));
        sql.push_str(&plan.view_join);
        for j in &plan.extra_joins {
            sql.push(' ');
            sql.push_str(j);
        }
        for d in &plan.dimensions {
            if let Some(j) = &d.join {
                sql.push(' ');
                sql.push_str(j);
            }
        }

        // WHERE: experiment ids + time bounds + filters.
        let exp_phs: Vec<String> = exp_ids
            .iter()
            .map(|id| ph.next(&mut binds, Val::Int(*id)))
            .collect();
        let mut wheres = vec![format!("ti.experiment_id IN ({})", exp_phs.join(", "))];
        if let Some(st) = start_time_ms {
            let p = ph.next(&mut binds, Val::Int(st));
            wheres.push(format!("ti.timestamp_ms >= {p}"));
        }
        if let Some(et) = end_time_ms {
            let p = ph.next(&mut binds, Val::Int(et));
            wheres.push(format!("ti.timestamp_ms <= {p}"));
        }
        for filter in filters {
            wheres.extend(build_trace_filter_wheres(
                dialect,
                Some(filter),
                &mut ph,
                &mut binds,
            )?);
        }
        sql.push_str(" WHERE ");
        sql.push_str(&wheres.join(" AND "));

        // GROUP BY / ORDER BY on the underlying (unlabeled) dimension exprs.
        if !plan.dimensions.is_empty() {
            let group: Vec<String> = plan.dimensions.iter().map(|d| d.expr.clone()).collect();
            sql.push_str(&format!(" GROUP BY {}", group.join(", ")));
            sql.push_str(&format!(" ORDER BY {}", group.join(", ")));
        }
        sql.push_str(&format!(" LIMIT {max_results}"));

        // Fetch rows: each row = [dimension values..., aggregation values...].
        let dim_labels: Vec<String> = plan.dimensions.iter().map(|d| d.label.clone()).collect();
        let agg_specs: Vec<(String, String)> = aggregations
            .iter()
            .map(|a| (a.label_sql_alias(), a.label()))
            .collect();

        let rows = self
            .db()
            .fetch_all(&sql, &binds, |r| {
                let mut dims: BTreeMap<String, Option<String>> = BTreeMap::new();
                for label in &dim_labels {
                    // Dimension values may be NULL; keep as Option to drop the row.
                    dims.insert(label.clone(), r.get_opt_string(label)?);
                }
                let mut values: BTreeMap<String, Option<f64>> = BTreeMap::new();
                for (alias, out_key) in &agg_specs {
                    values.insert(out_key.clone(), r.get_opt_f64(alias)?);
                }
                Ok((dims, values))
            })
            .await
            .map_err(internal)?;

        // Row post-processing (convert_results_to_metric_data_points):
        // drop rows with any NULL dimension; drop NULL aggregation values;
        // drop rows whose values map ends up empty.
        let mut out: Vec<MetricDataPoint> = Vec::new();
        for (dims, values) in rows {
            let mut resolved_dims: BTreeMap<String, String> = BTreeMap::new();
            let mut skip = false;
            for (k, v) in dims {
                match v {
                    Some(val) => {
                        resolved_dims.insert(k, val);
                    }
                    None => {
                        skip = true;
                        break;
                    }
                }
            }
            if skip {
                continue;
            }
            let resolved_values: BTreeMap<String, f64> = values
                .into_iter()
                .filter_map(|(k, v)| v.map(|val| (k, val)))
                .collect();
            if resolved_values.is_empty() {
                continue;
            }
            out.push(MetricDataPoint {
                metric_name: metric_name.to_string(),
                dimensions: resolved_dims,
                values: resolved_values,
            });
        }
        Ok(out)
    }
}

impl MetricAggregation {
    /// A SQL-safe alias for the aggregation column (labels like `P95.0` are not
    /// valid bare identifiers), keyed back to the output label after fetch.
    fn label_sql_alias(self) -> String {
        match self {
            MetricAggregation::Count => "agg_count".to_string(),
            MetricAggregation::Sum => "agg_sum".to_string(),
            MetricAggregation::Avg => "agg_avg".to_string(),
            MetricAggregation::Percentile(v) => {
                format!("agg_p{}", format_percentile(v).replace('.', "_"))
            }
        }
    }
}

fn combine_filter(base_filter: Option<&str>, f: &str) -> String {
    match base_filter {
        Some(bf) if !bf.is_empty() => format!("{bf} and {f}"),
        _ => f.to_string(),
    }
}

fn to_correlation_result(counts: TraceCorrelationCounts) -> TraceFilterCorrelationResult {
    let NpmiResult {
        npmi,
        npmi_smoothed,
    } = calculate_npmi_from_counts(counts);
    TraceFilterCorrelationResult {
        npmi,
        npmi_smoothed,
        filter1_count: counts.filter1_count,
        filter2_count: counts.filter2_count,
        joint_count: counts.joint_count,
        total_count: counts.total_count,
    }
}

/// SQL for one aggregation over `column` (COUNT/SUM/AVG only; PERCENTILE is
/// rejected earlier).
fn aggregation_sql(agg: MetricAggregation, column: &AggColumn) -> String {
    match agg {
        MetricAggregation::Count => match column {
            AggColumn::CountDistinct(c) => format!("COUNT(DISTINCT {c})"),
            AggColumn::Plain(c) => format!("COUNT({c})"),
        },
        MetricAggregation::Sum => format!("SUM({})", column.expr()),
        MetricAggregation::Avg => format!("AVG({})", column.expr()),
        // Rejected before reaching here.
        MetricAggregation::Percentile(_) => format!("AVG({})", column.expr()),
    }
}

/// The column/expression the aggregation runs over.
enum AggColumn {
    Plain(String),
    /// `session_count` aggregates `COUNT(DISTINCT trace_metadata.value)`.
    CountDistinct(String),
}

impl AggColumn {
    fn expr(&self) -> &str {
        match self {
            AggColumn::Plain(c) | AggColumn::CountDistinct(c) => c,
        }
    }
}

/// One resolved dimension: its labeled SELECT expr and an optional extra join.
struct DimensionCol {
    expr: String,
    label: String,
    join: Option<String>,
}

/// The resolved query plan for a `query_trace_metrics` request.
struct MetricsPlan {
    view_join: String,
    extra_joins: Vec<String>,
    agg_column: AggColumn,
    dimensions: Vec<DimensionCol>,
}

const TRACE_NAME_TAG: &str = "mlflow.traceName";
const SESSION_METADATA_KEY: &str = "mlflow.trace.session";
const SPAN_MODEL_NAME_KEY: &str = "mlflow.llm.model";
const SPAN_MODEL_PROVIDER_KEY: &str = "mlflow.llm.provider";

/// Token-usage TRACES metric names (`token_usage_keys`).
const TOKEN_USAGE_KEYS: &[&str] = &[
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
];

/// Cost SPANS metric names (`cost_keys`).
const COST_KEYS: &[&str] = &["input_cost", "output_cost", "total_cost"];

fn build_metrics_plan(
    dialect: Dialect,
    view_type: MetricViewType,
    metric_name: &str,
    dimensions: &[String],
) -> Result<MetricsPlan, MlflowError> {
    let (view_join, mut extra_joins, agg_column) = match view_type {
        MetricViewType::Traces => {
            let (extra, col) = traces_agg_column(dialect, metric_name);
            (String::new(), extra, col)
        }
        MetricViewType::Spans => {
            let (extra, col) = spans_agg_column(metric_name);
            (
                format!(" JOIN {SPANS} s ON s.trace_id = ti.request_id"),
                extra,
                col,
            )
        }
        MetricViewType::Assessments => (
            format!(
                " JOIN {ASSESSMENTS} a ON a.trace_id = ti.request_id AND a.valid = {}",
                true_lit(dialect)
            ),
            Vec::new(),
            assessments_agg_column(metric_name),
        ),
    };

    let mut dims: Vec<DimensionCol> = Vec::new();
    for d in dimensions {
        dims.push(resolve_dimension(dialect, view_type, d, &mut extra_joins)?);
    }

    Ok(MetricsPlan {
        view_join,
        extra_joins,
        agg_column,
        dimensions: dims,
    })
}

fn traces_agg_column(dialect: Dialect, metric_name: &str) -> (Vec<String>, AggColumn) {
    match metric_name {
        "trace_count" => (Vec::new(), AggColumn::Plain("ti.request_id".to_string())),
        "session_count" => {
            let join = format!(
                "JOIN trace_request_metadata sess ON sess.request_id = ti.request_id \
                 AND sess.\"key\" = '{SESSION_METADATA_KEY}'"
            );
            (
                vec![join],
                AggColumn::CountDistinct("sess.value".to_string()),
            )
        }
        "latency" => (
            Vec::new(),
            AggColumn::Plain("ti.execution_time_ms".to_string()),
        ),
        name if TOKEN_USAGE_KEYS.contains(&name) => {
            let join = format!(
                "JOIN trace_metrics tm ON tm.request_id = ti.request_id \
                 AND tm.\"key\" = '{}'",
                sql_str_literal(name)
            );
            let _ = dialect;
            (vec![join], AggColumn::Plain("tm.value".to_string()))
        }
        // Unknown names are rejected by validation before reaching here.
        _ => (Vec::new(), AggColumn::Plain("ti.request_id".to_string())),
    }
}

fn spans_agg_column(metric_name: &str) -> (Vec<String>, AggColumn) {
    match metric_name {
        "span_count" => (Vec::new(), AggColumn::Plain("s.span_id".to_string())),
        "latency" => (
            Vec::new(),
            // (end - start) // 1_000_000 (ns -> ms floor). Both nullable; NULL
            // end (in-progress) yields NULL, dropped by AVG.
            AggColumn::Plain(
                "((s.end_time_unix_nano - s.start_time_unix_nano) / 1000000)".to_string(),
            ),
        ),
        name if COST_KEYS.contains(&name) => {
            let join = format!(
                "JOIN {SPAN_METRICS} sm ON sm.trace_id = s.trace_id AND sm.span_id = s.span_id \
                 AND sm.\"key\" = '{}'",
                sql_str_literal(name)
            );
            (vec![join], AggColumn::Plain("sm.value".to_string()))
        }
        _ => (Vec::new(), AggColumn::Plain("s.span_id".to_string())),
    }
}

fn assessments_agg_column(metric_name: &str) -> AggColumn {
    match metric_name {
        "assessment_count" => AggColumn::Plain("a.assessment_id".to_string()),
        // assessment_value → numeric CASE over the JSON `value` column.
        "assessment_value" => AggColumn::Plain(assessment_numeric_value_case("a.value")),
        _ => AggColumn::Plain("a.assessment_id".to_string()),
    }
}

/// `_get_assessment_numeric_value_column`: map the JSON-encoded assessment
/// `value` to a numeric (yes/true→1, no/false→0, null/string/array/object→NULL,
/// numeric→CAST). Uses `substr(value, 1, 1)` for the leading-char check.
fn assessment_numeric_value_case(col: &str) -> String {
    format!(
        "(CASE \
         WHEN {col} IN ('true', '\"yes\"') THEN 1.0 \
         WHEN {col} IN ('false', '\"no\"') THEN 0.0 \
         WHEN {col} = 'null' THEN NULL \
         WHEN substr({col}, 1, 1) IN ('\"', '[', '{{') THEN NULL \
         ELSE CAST({col} AS FLOAT) END)"
    )
}

fn resolve_dimension(
    dialect: Dialect,
    view_type: MetricViewType,
    dimension: &str,
    _extra_joins: &mut [String],
) -> Result<DimensionCol, MlflowError> {
    let dim = match (view_type, dimension) {
        (MetricViewType::Traces, "trace_status") => DimensionCol {
            expr: "ti.status".to_string(),
            label: "trace_status".to_string(),
            join: None,
        },
        (MetricViewType::Traces, "trace_name") => DimensionCol {
            expr: "dim_trace_name.value".to_string(),
            label: "trace_name".to_string(),
            join: Some(format!(
                "JOIN {TRACE_TAGS} dim_trace_name ON dim_trace_name.request_id = ti.request_id \
                 AND dim_trace_name.\"key\" = '{TRACE_NAME_TAG}'"
            )),
        },
        (MetricViewType::Spans, "span_name") => DimensionCol {
            expr: "s.name".to_string(),
            label: "span_name".to_string(),
            join: None,
        },
        (MetricViewType::Spans, "span_type") => DimensionCol {
            expr: "s.\"type\"".to_string(),
            label: "span_type".to_string(),
            join: None,
        },
        (MetricViewType::Spans, "span_status") => DimensionCol {
            expr: "s.status".to_string(),
            label: "span_status".to_string(),
            join: None,
        },
        (MetricViewType::Spans, "span_model_name") => DimensionCol {
            expr: json_dimension_expr(dialect, SPAN_MODEL_NAME_KEY),
            label: "span_model_name".to_string(),
            join: None,
        },
        (MetricViewType::Spans, "span_model_provider") => DimensionCol {
            expr: json_dimension_expr(dialect, SPAN_MODEL_PROVIDER_KEY),
            label: "span_model_provider".to_string(),
            join: None,
        },
        (MetricViewType::Assessments, "assessment_name") => DimensionCol {
            expr: "a.name".to_string(),
            label: "assessment_name".to_string(),
            join: None,
        },
        (MetricViewType::Assessments, "assessment_value") => DimensionCol {
            expr: "a.value".to_string(),
            label: "assessment_value".to_string(),
            join: None,
        },
        // Validation already rejects unsupported dimensions.
        _ => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Unsupported dimension `{dimension}` with view type {}",
                view_type.as_str()
            )))
        }
    };
    Ok(dim)
}

/// `_get_json_dimension_column`: extract a text value from the
/// `spans.dimension_attributes` JSON column, per dialect.
fn json_dimension_expr(dialect: Dialect, json_key: &str) -> String {
    match dialect {
        Dialect::Postgres => {
            format!("(s.dimension_attributes ->> '{json_key}')")
        }
        // SQLite / MySQL: JSON_EXTRACT with the `$.key` path, unquoted for text.
        Dialect::Sqlite => {
            format!("json_extract(s.dimension_attributes, '$.\"{json_key}\"')")
        }
        Dialect::MySql => {
            format!("JSON_UNQUOTE(JSON_EXTRACT(s.dimension_attributes, '$.\"{json_key}\"'))")
        }
    }
}

/// The floating-point cast target type per dialect.
fn float_type(dialect: Dialect) -> &'static str {
    match dialect {
        // SQLite's `REAL`, Postgres `DOUBLE PRECISION`, MySQL `DECIMAL`
        // (MySQL rejects `CAST(x AS FLOAT/DOUBLE)` pre-8.0.17; DECIMAL is
        // universally accepted and decodes to f64).
        Dialect::Sqlite => "REAL",
        Dialect::Postgres => "DOUBLE PRECISION",
        Dialect::MySql => "DECIMAL(38, 10)",
    }
}

fn true_lit(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Postgres => "TRUE",
        Dialect::Sqlite | Dialect::MySql => "1",
    }
}

/// Escape a single-quoted SQL string literal (the metric-name join keys are
/// validated against a fixed allow-list, so this only guards against `'`).
fn sql_str_literal(s: &str) -> String {
    s.replace('\'', "''")
}

// ===========================================================================
// validate_query_trace_metrics_params
// ===========================================================================

fn validate_query_trace_metrics_params(
    view_type: MetricViewType,
    metric_name: &str,
    aggregations: &[MetricAggregation],
    dimensions: &[String],
) -> Result<(), MlflowError> {
    let config = view_config(view_type);
    let metric = config.iter().find(|(name, _, _)| *name == metric_name);
    let Some((_, allowed_aggs, allowed_dims)) = metric else {
        let names: Vec<&str> = config.iter().map(|(n, _, _)| *n).collect();
        return Err(MlflowError::invalid_parameter_value(format!(
            "metric_name must be one of {names:?}, got '{metric_name}'"
        )));
    };

    // Invalid aggregation types.
    let invalid_aggs: Vec<&str> = aggregations
        .iter()
        .map(|a| a.type_name())
        .filter(|t| !allowed_aggs.contains(t))
        .collect();
    if !invalid_aggs.is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Found invalid aggregation_type(s): {invalid_aggs:?}. Supported aggregation types: {allowed_aggs:?}"
        )));
    }

    // Invalid dimensions.
    let invalid_dims: Vec<&String> = dimensions
        .iter()
        .filter(|d| !allowed_dims.contains(&d.as_str()))
        .collect();
    if !invalid_dims.is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Found invalid dimension(s): {invalid_dims:?}. Supported dimensions: {allowed_dims:?}"
        )));
    }
    Ok(())
}

/// (metric_name, allowed_aggregation_type_names, allowed_dimensions) per view.
type MetricConfig = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);

fn view_config(view_type: MetricViewType) -> &'static [MetricConfig] {
    const COUNT: &[&str] = &["COUNT"];
    const NUMERIC: &[&str] = &["AVG", "PERCENTILE"];
    const SUM_NUMERIC: &[&str] = &["SUM", "AVG", "PERCENTILE"];
    match view_type {
        MetricViewType::Traces => &[
            ("trace_count", COUNT, &["trace_name", "trace_status"]),
            ("session_count", COUNT, &[]),
            ("latency", NUMERIC, &["trace_name"]),
            ("input_tokens", SUM_NUMERIC, &["trace_name"]),
            ("output_tokens", SUM_NUMERIC, &["trace_name"]),
            ("total_tokens", SUM_NUMERIC, &["trace_name"]),
            ("cache_read_input_tokens", SUM_NUMERIC, &["trace_name"]),
            ("cache_creation_input_tokens", SUM_NUMERIC, &["trace_name"]),
        ],
        MetricViewType::Spans => &[
            (
                "span_count",
                COUNT,
                &[
                    "span_name",
                    "span_type",
                    "span_status",
                    "span_model_name",
                    "span_model_provider",
                ],
            ),
            ("latency", NUMERIC, &["span_name", "span_status"]),
            (
                "input_cost",
                SUM_NUMERIC,
                &["span_model_name", "span_model_provider"],
            ),
            (
                "output_cost",
                SUM_NUMERIC,
                &["span_model_name", "span_model_provider"],
            ),
            (
                "total_cost",
                SUM_NUMERIC,
                &["span_model_name", "span_model_provider"],
            ),
        ],
        MetricViewType::Assessments => &[
            (
                "assessment_count",
                COUNT,
                &["assessment_name", "assessment_value"],
            ),
            ("assessment_value", NUMERIC, &["assessment_name"]),
        ],
    }
}
