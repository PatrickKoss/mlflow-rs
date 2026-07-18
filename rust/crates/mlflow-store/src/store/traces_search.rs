//! `search_traces` (plan T2.10) — mirrors `SqlAlchemyStore.search_traces`
//! (`sqlalchemy_store.py:3755`), `_get_filter_clauses_for_search_traces`
//! (`:9390`), `_get_orderby_clauses_for_search_traces` (`:9312`), and
//! `_apply_trace_filter_clauses` (`:3687`).
//!
//! ## Query strategy
//!
//! Python inner-joins one subquery per tag/metadata/span filter and outer-joins
//! per order-by key. Because each joined right side is unique on `request_id`,
//! no `DISTINCT` is needed. We reproduce the observable result set with
//! **correlated `EXISTS` semi-joins** for filters (plan §5.2 Q2/Q3 — no
//! fan-out) and `LEFT JOIN` subqueries for order-by keys. Filters, ordering, and
//! the tiebreak (`timestamp_ms DESC, request_id ASC`) match Python byte-for-byte
//! on the page contents.
//!
//! ## Special filter cases (mirroring Python)
//!
//! * **run_id** (`request_metadata.mlflow.sourceRun = <run>`) → OR of an
//!   `entity_associations` link (trace→run) and a `SOURCE_RUN` metadata match.
//! * **span** (`span.name/type/status`, `span.attributes.*`, `span.content`
//!   a.k.a. `trace.text`) → EXISTS over `spans`; multiple span predicates must
//!   match the *same* span (combined into one EXISTS).
//! * **assessment** (`feedback.*`/`expectation.*`) → EXISTS over `assessments`
//!   (name + type + `valid`), value compared against the JSON `value` column.
//!   (Session-scoped assessment coverage is a genai concern deferred to
//!   Phase 12; direct matches — the common case — are covered here.)
//! * **tag / metadata** (incl. `IS NULL`/`IS NOT NULL`) → EXISTS/NOT EXISTS.
//!
//! ## Pagination
//!
//! Python uses OFFSET pagination behind an opaque base64(JSON `{"offset": N}`)
//! token; the page contents are offset-defined. We keep that exact scheme so
//! tokens and page boundaries match (the keyset rewrite from `search_runs` would
//! change page contents under concurrent writes, so trace search stays on offset
//! for parity; §5.2 Q1 revisits this in a later phase).

use mlflow_error::MlflowError;
use mlflow_search::{Comparison, OrderBy, Value};

use super::dbutil::Val;
use super::entities::{TraceInfo, TRACE_METADATA_SOURCE_RUN};
use super::experiments::internal;
use super::traces::{ENTITY_TYPE_RUN, ENTITY_TYPE_TRACE};
use super::TrackingStore;
use crate::dialect::Dialect;
use crate::schema::traces::TRACE_INFO;

/// `SEARCH_TRACES_DEFAULT_MAX_RESULTS` (`mlflow/store/tracking/__init__.py:23`).
pub const SEARCH_TRACES_DEFAULT_MAX_RESULTS: i64 = 100;

/// `SEARCH_MAX_RESULTS_THRESHOLD` (`mlflow/store/tracking/__init__.py:21`).
const SEARCH_MAX_RESULTS_THRESHOLD: i64 = 50000;

/// A page of trace infos plus the optional next-page token.
#[derive(Debug)]
pub struct TracesPage {
    pub trace_infos: Vec<TraceInfo>,
    pub next_page_token: Option<String>,
}

impl TrackingStore {
    /// `search_traces` over `experiment_ids` (locations), filtered/ordered per
    /// the trace DSL, with offset pagination. Workspace scoping filters the
    /// requested experiment ids to those in the active workspace.
    pub async fn search_traces(
        &self,
        workspace: &str,
        experiment_ids: &[String],
        filter: Option<&str>,
        max_results: i64,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<TracesPage, MlflowError> {
        validate_max_results(max_results)?;

        let filters = parse_filter(filter)?;
        let order_cols = build_order_cols(order_by)?;
        let dialect = self.db().dialect();

        // Workspace + numeric experiment-id filtering (`_filter_experiment_ids`).
        let exp_ids = self
            .filter_trace_experiment_ids(workspace, experiment_ids)
            .await?;
        if exp_ids.is_empty() {
            return Ok(TracesPage {
                trace_infos: vec![],
                next_page_token: None,
            });
        }

        let offset = match page_token {
            Some(t) => parse_offset_token(t)?,
            None => 0,
        };

        let query = build_search_sql(
            dialect,
            &exp_ids,
            &filters,
            &order_cols,
            max_results,
            offset,
        )?;
        let trace_ids: Vec<String> = self
            .db()
            .fetch_all(&query.sql, &query.binds, |r| r.get_string("request_id"))
            .await
            .map_err(internal)?;

        let next_page_token = if trace_ids.len() as i64 == max_results {
            Some(encode_offset_token(offset + max_results))
        } else {
            None
        };

        // Assemble ordered TraceInfos (batched tag/metadata/assessment loads),
        // preserving the SQL result order.
        let trace_infos = self.load_trace_infos_ordered(workspace, &trace_ids).await?;

        Ok(TracesPage {
            trace_infos,
            next_page_token,
        })
    }

    /// Filter `experiment_ids` (stringified) to those present in `workspace`,
    /// preserving order and de-duplicating. Non-integer ids raise the same
    /// `INVALID_PARAMETER_VALUE` Python's `int(e)` would
    /// (`_filter_experiment_ids`).
    pub(crate) async fn filter_trace_experiment_ids(
        &self,
        workspace: &str,
        experiment_ids: &[String],
    ) -> Result<Vec<i64>, MlflowError> {
        let mut wanted: Vec<i64> = Vec::with_capacity(experiment_ids.len());
        for e in experiment_ids {
            let id = e.parse::<i64>().map_err(|_| {
                MlflowError::invalid_parameter_value(format!(
                    "invalid literal for int() with base 10: '{e}'"
                ))
            })?;
            if !wanted.contains(&id) {
                wanted.push(id);
            }
        }
        if wanted.is_empty() {
            return Ok(vec![]);
        }
        let dialect = self.db().dialect();
        let mut binds: Vec<Val> = Vec::with_capacity(wanted.len() + 1);
        let placeholders: Vec<String> = wanted
            .iter()
            .enumerate()
            .map(|(i, id)| {
                binds.push(Val::Int(*id));
                dialect.placeholder(i + 1)
            })
            .collect();
        binds.push(Val::Text(workspace.to_string()));
        let sql = format!(
            "SELECT experiment_id FROM experiments WHERE experiment_id IN ({}) AND workspace = {}",
            placeholders.join(", "),
            dialect.placeholder(wanted.len() + 1)
        );
        let present: Vec<i64> = self
            .db()
            .fetch_all(&sql, &binds, |r| r.get_int("experiment_id"))
            .await
            .map_err(internal)?;
        Ok(wanted
            .into_iter()
            .filter(|id| present.contains(id))
            .collect())
    }

    /// Load `TraceInfo`s for `trace_ids` preserving that order (the search
    /// result order).
    async fn load_trace_infos_ordered(
        &self,
        workspace: &str,
        trace_ids: &[String],
    ) -> Result<Vec<TraceInfo>, MlflowError> {
        if trace_ids.is_empty() {
            return Ok(vec![]);
        }
        // batch_get_trace_infos already preserves the requested order.
        self.batch_get_trace_infos(workspace, trace_ids).await
    }
}

// ===========================================================================
// max_results / filter / order-by parsing
// ===========================================================================

fn validate_max_results(max_results: i64) -> Result<(), MlflowError> {
    if max_results < 1 {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. It must be a \
             positive integer"
        )));
    }
    if max_results > SEARCH_MAX_RESULTS_THRESHOLD {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. It must be at \
             most {SEARCH_MAX_RESULTS_THRESHOLD}"
        )));
    }
    Ok(())
}

fn parse_filter(filter: Option<&str>) -> Result<Vec<Comparison>, MlflowError> {
    let s = filter.unwrap_or("");
    mlflow_search::parse::traces_filter(s).map_err(search_err)
}

fn search_err(e: mlflow_search::SearchError) -> MlflowError {
    use mlflow_error::ErrorCode;
    let code = match e.error_code {
        mlflow_search::ErrorCode::InvalidParameterValue => ErrorCode::InvalidParameterValue,
        _ => ErrorCode::InternalError,
    };
    MlflowError::new(e.message, code)
}

/// One resolved order-by column.
#[derive(Debug, Clone)]
struct OrderCol {
    /// SQL value expression (already safe/qualified).
    expr: String,
    ascending: bool,
    /// LEFT JOIN needed for a tag/metadata key (alias, table, key).
    join: Option<(String, &'static str, String)>,
}

/// Build the ordered [`OrderCol`] list, mirroring
/// `_get_orderby_clauses_for_search_traces`: per clause a NULL-last `CASE`
/// (implicit, we emit `expr IS NULL` as a leading sort term) then the value,
/// followed by the `(timestamp_ms DESC, request_id ASC)` tiebreak (appended
/// only when not already present).
fn build_order_cols(order_by: &[String]) -> Result<Vec<OrderCol>, MlflowError> {
    let mut cols: Vec<OrderCol> = Vec::new();
    let mut observed: Vec<(String, String)> = Vec::new();
    let mut join_idx = 0usize;
    let mut timestamp_seen = false;
    let mut request_id_seen = false;

    for clause in order_by {
        let ob: OrderBy = mlflow_search::parse::traces_order_by(clause).map_err(search_err)?;
        let dedup = (ob.entity_type.clone(), ob.key.clone());
        if observed.contains(&dedup) {
            return Err(MlflowError::new(
                format!("`order_by` contains duplicate fields: {order_by:?}"),
                mlflow_error::ErrorCode::InvalidParameterValue,
            ));
        }
        observed.push(dedup);

        match ob.entity_type.as_str() {
            "attribute" => {
                if ob.key == "timestamp_ms" {
                    timestamp_seen = true;
                }
                if ob.key == "request_id" {
                    request_id_seen = true;
                }
                let col = attr_order_column(&ob.key)?;
                cols.push(OrderCol {
                    expr: format!("ti.{col}"),
                    ascending: ob.ascending,
                    join: None,
                });
            }
            "tag" | "request_metadata" => {
                let table = if ob.entity_type == "tag" {
                    "trace_tags"
                } else {
                    "trace_request_metadata"
                };
                let alias = format!("oj_{join_idx}");
                join_idx += 1;
                cols.push(OrderCol {
                    expr: format!("{alias}.value"),
                    ascending: ob.ascending,
                    join: Some((alias, table, ob.key.clone())),
                });
            }
            other => {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid identifier type '{other}'"
                )));
            }
        }
    }

    if !timestamp_seen {
        cols.push(OrderCol {
            expr: "ti.timestamp_ms".to_string(),
            ascending: false,
            join: None,
        });
    }
    if !request_id_seen {
        cols.push(OrderCol {
            expr: "ti.request_id".to_string(),
            ascending: true,
            join: None,
        });
    }
    Ok(cols)
}

/// `getattr(SqlTraceInfo, key)` order columns.
fn attr_order_column(key: &str) -> Result<&'static str, MlflowError> {
    Ok(match key {
        "timestamp_ms" => "timestamp_ms",
        "execution_time_ms" => "execution_time_ms",
        "status" => "status",
        "request_id" => "request_id",
        "experiment_id" => "experiment_id",
        // The trace order-by DSL remaps `name`/`run_id` to tag/metadata before
        // reaching here, so an unmapped one is a genuine attribute.
        "client_request_id" => "client_request_id",
        other => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid order by key '{other}' specified."
            )))
        }
    })
}

// ===========================================================================
// SQL generation
// ===========================================================================

#[derive(Debug)]
struct Query {
    sql: String,
    binds: Vec<Val>,
}

pub(crate) struct Ph {
    dialect: Dialect,
    idx: usize,
}

impl Ph {
    pub(crate) fn new(dialect: Dialect) -> Self {
        Self { dialect, idx: 0 }
    }
    pub(crate) fn next(&mut self, binds: &mut Vec<Val>, v: Val) -> String {
        self.idx += 1;
        binds.push(v);
        self.dialect.placeholder(self.idx)
    }
    /// A LIKE placeholder pair helper (MySQL binds twice).
    fn like(&mut self, binds: &mut Vec<Val>, s: String) -> usize {
        self.idx += 1;
        let first = self.idx;
        binds.push(Val::Text(s.clone()));
        if let Dialect::MySql = self.dialect {
            self.idx += 1;
            binds.push(Val::Text(s));
        }
        first
    }
}

/// Build the trace-search `SELECT request_id`: order-by LEFT JOINs, EXISTS
/// filters, experiment-id predicate, `ORDER BY`, `LIMIT`, `OFFSET`.
fn build_search_sql(
    dialect: Dialect,
    exp_ids: &[i64],
    filters: &[Comparison],
    order_cols: &[OrderCol],
    max_results: i64,
    offset: i64,
) -> Result<Query, MlflowError> {
    let mut binds: Vec<Val> = Vec::new();
    let mut ph = Ph::new(dialect);

    let mut sql = format!("SELECT ti.request_id FROM {} ti", TRACE_INFO);

    // Order-by joins.
    for c in order_cols {
        if let Some((alias, table, key)) = &c.join {
            let key_ph = ph.next(&mut binds, Val::Text(key.clone()));
            sql.push_str(&format!(
                " LEFT JOIN (SELECT request_id, value FROM {table} WHERE \"key\" = {key_ph}) \
                 {alias} ON {alias}.request_id = ti.request_id"
            ));
        }
    }

    // WHERE.
    let mut wheres: Vec<String> = Vec::new();
    let exp_phs: Vec<String> = exp_ids
        .iter()
        .map(|id| ph.next(&mut binds, Val::Int(*id)))
        .collect();
    wheres.push(format!("ti.experiment_id IN ({})", exp_phs.join(", ")));

    // Filters.
    let mut span_conditions: Vec<String> = Vec::new();
    for f in filters {
        if let Some(pred) =
            build_filter_predicate(dialect, f, &mut ph, &mut binds, &mut span_conditions)?
        {
            wheres.push(pred);
        }
    }
    // Combined span EXISTS (all span predicates must match the same span).
    if !span_conditions.is_empty() {
        wheres.push(format!(
            "EXISTS (SELECT 1 FROM spans s WHERE s.trace_id = ti.request_id AND {})",
            span_conditions.join(" AND ")
        ));
    }

    sql.push_str(" WHERE ");
    sql.push_str(&wheres.join(" AND "));

    // ORDER BY: for each col, NULL-rank ascending then the value.
    sql.push_str(" ORDER BY ");
    let mut terms: Vec<String> = Vec::new();
    for c in order_cols {
        // NULL-last: `(expr IS NULL) ASC` then `expr <dir>`.
        terms.push(format!(
            "(CASE WHEN {} IS NULL THEN 1 ELSE 0 END) ASC",
            c.expr
        ));
        terms.push(format!(
            "{} {}",
            c.expr,
            if c.ascending { "ASC" } else { "DESC" }
        ));
    }
    sql.push_str(&terms.join(", "));

    sql.push_str(&format!(" LIMIT {max_results}"));
    if offset > 0 {
        sql.push_str(&format!(" OFFSET {offset}"));
    }

    Ok(Query { sql, binds })
}

/// Build the WHERE predicates for a trace-search `filter` string against the
/// `trace_info ti` alias, reusing the full search filter machinery
/// (attribute/tag/metadata/span/assessment/run_id). Used by
/// `calculate_trace_filter_correlation` (and any caller needing a "traces
/// matching this filter" subquery) so the correlation counts share
/// byte-identical filter semantics with `search_traces`.
///
/// Appends to `binds`/`ph`; an empty/absent filter yields no predicates.
pub(crate) fn build_trace_filter_wheres(
    dialect: Dialect,
    filter: Option<&str>,
    ph: &mut Ph,
    binds: &mut Vec<Val>,
) -> Result<Vec<String>, MlflowError> {
    let filters = parse_filter(filter)?;
    let mut wheres: Vec<String> = Vec::new();
    let mut span_conditions: Vec<String> = Vec::new();
    for f in &filters {
        if let Some(pred) = build_filter_predicate(dialect, f, ph, binds, &mut span_conditions)? {
            wheres.push(pred);
        }
    }
    if !span_conditions.is_empty() {
        wheres.push(format!(
            "EXISTS (SELECT 1 FROM spans s WHERE s.trace_id = ti.request_id AND {})",
            span_conditions.join(" AND ")
        ));
    }
    Ok(wheres)
}

/// Build one filter comparison into a WHERE predicate. Span predicates are
/// accumulated into `span_conditions` (combined by the caller) and return
/// `None`. `run_id` is expanded inline to the link-OR-metadata OR.
fn build_filter_predicate(
    dialect: Dialect,
    f: &Comparison,
    ph: &mut Ph,
    binds: &mut Vec<Val>,
    span_conditions: &mut Vec<String>,
) -> Result<Option<String>, MlflowError> {
    let comparator = f.comparator.to_uppercase();
    match f.entity_type.as_str() {
        "attribute" => {
            let numeric = is_numeric_attr(&f.key);
            validate_attr_comparator(&comparator, numeric)?;
            // end_time(_ms) is a computed column: timestamp_ms + coalesce(exec,0).
            let target = if f.key == "end_time_ms" || f.key == "end_time" {
                "(ti.timestamp_ms + COALESCE(ti.execution_time_ms, 0))".to_string()
            } else {
                format!("ti.{}", attr_filter_column(&f.key)?)
            };
            Ok(Some(value_predicate(
                dialect,
                &target,
                &comparator,
                &f.value,
                numeric,
                ph,
                binds,
            )?))
        }
        "tag" | "request_metadata" => {
            validate_kv_comparator(&comparator)?;
            let table = if f.entity_type == "tag" {
                "trace_tags"
            } else {
                "trace_request_metadata"
            };
            // run_id special-case: request_metadata SOURCE_RUN "=" → link OR metadata.
            if f.entity_type == "request_metadata"
                && f.key == TRACE_METADATA_SOURCE_RUN
                && comparator == "="
            {
                return Ok(Some(run_id_predicate(&f.value, ph, binds)?));
            }
            if comparator == "IS NULL" || comparator == "IS NOT NULL" {
                let key_ph = ph.next(binds, Val::Text(f.key.clone()));
                let inner = format!(
                    "SELECT 1 FROM {table} e WHERE e.request_id = ti.request_id AND e.\"key\" = {key_ph}"
                );
                return Ok(Some(if comparator == "IS NULL" {
                    format!("NOT EXISTS ({inner})")
                } else {
                    format!("EXISTS ({inner})")
                }));
            }
            let key_ph = ph.next(binds, Val::Text(f.key.clone()));
            let val_pred =
                value_predicate(dialect, "e.value", &comparator, &f.value, false, ph, binds)?;
            Ok(Some(format!(
                "EXISTS (SELECT 1 FROM {table} e WHERE e.request_id = ti.request_id \
                 AND e.\"key\" = {key_ph} AND {val_pred})"
            )))
        }
        "span" => {
            span_conditions.push(build_span_condition(
                dialect,
                &f.key,
                &comparator,
                &f.value,
                ph,
                binds,
            )?);
            Ok(None)
        }
        "feedback" | "expectation" => Ok(Some(build_assessment_predicate(
            dialect,
            &f.entity_type,
            &f.key,
            &comparator,
            &f.value,
            ph,
            binds,
        )?)),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid search expression type '{other}'"
        ))),
    }
}

/// The run_id OR predicate: linked via entity_associations OR SOURCE_RUN metadata.
fn run_id_predicate(
    value: &Value,
    ph: &mut Ph,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    let run_id = as_str(value)?;
    let link_src = ph.next(binds, Val::Text(ENTITY_TYPE_TRACE.to_string()));
    let link_dst = ph.next(binds, Val::Text(ENTITY_TYPE_RUN.to_string()));
    let link_run = ph.next(binds, Val::Text(run_id.clone()));
    let linked = format!(
        "EXISTS (SELECT 1 FROM entity_associations ea WHERE ea.source_id = ti.request_id \
         AND ea.source_type = {link_src} AND ea.destination_type = {link_dst} \
         AND ea.destination_id = {link_run})"
    );
    let meta_key = ph.next(binds, Val::Text(TRACE_METADATA_SOURCE_RUN.to_string()));
    let meta_val = ph.next(binds, Val::Text(run_id));
    let meta = format!(
        "EXISTS (SELECT 1 FROM trace_request_metadata m WHERE m.request_id = ti.request_id \
         AND m.\"key\" = {meta_key} AND m.value = {meta_val})"
    );
    Ok(format!("({linked} OR {meta})"))
}

/// One span predicate against the `spans s` alias (name/type/status,
/// attributes.*, or content/text).
fn build_span_condition(
    dialect: Dialect,
    key: &str,
    comparator: &str,
    value: &Value,
    ph: &mut Ph,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    if let Some(attr) = key.strip_prefix("attributes.") {
        // Python trims identifier backticks after removing the `attributes.`
        // prefix, so dotted keys such as
        // `span.attributes.`gen_ai.request.model`` address the same extracted
        // key as their unquoted form.
        let attr = attr.trim_matches('`');
        let val = as_str(value)?;
        let pattern = format!("%\"{attr}\"{val}%");
        if comparator == "RLIKE" {
            // Python applies the regular expression to serialized JSON and
            // wraps anchors around the JSON-encoded string value.
            let mut transformed = val.as_str();
            let search_prefix = if let Some(rest) = transformed.strip_prefix('^') {
                transformed = rest;
                "\"\\\\\""
            } else {
                "\"\\\\\".*"
            };
            let search_suffix = if let Some(rest) = transformed.strip_suffix('$') {
                transformed = rest;
                "\\\\\""
            } else {
                ""
            };
            let regex = format!("\"{attr}\": {search_prefix}{transformed}{search_suffix}");
            let regex_ph = ph.next(binds, Val::Text(regex));
            return Ok(match dialect {
                Dialect::Sqlite => format!("s.content REGEXP {regex_ph}"),
                Dialect::Postgres => format!("s.content ~ {regex_ph}"),
                Dialect::MySql => {
                    format!("CAST(s.content AS BINARY) REGEXP BINARY {regex_ph}")
                }
            });
        }
        // ILIKE makes the attribute name itself case-insensitive in Python's
        // content scan. Attribute names containing SQL wildcard/escape syntax
        // have similarly loose semantics. Keep the content predicate for those
        // cases; the common case-sensitive LIKE path uses the extraction table.
        let can_use_extraction = comparator == "LIKE"
            && attr.chars().count() <= 250
            && !attr.chars().any(|c| matches!(c, '%' | '_' | '"' | '\\'));
        if can_use_extraction {
            let key_ph = ph.next(binds, Val::Text(attr.to_string()));
            let content_like = {
                let idx = ph.like(binds, pattern);
                dialect.case_sensitive_like("s.content", idx)
            };
            // Keep the original content predicate as a residual after the
            // indexed key lookup. This preserves Python's loose suffix match:
            // a value in a later attribute can satisfy the pattern after the
            // requested key. The table narrows the scan to spans that actually
            // carry the top-level key without normalizing JSON escaping.
            return Ok(format!(
                "EXISTS (SELECT 1 FROM span_attributes sa WHERE sa.trace_id = s.trace_id \
                 AND sa.span_id = s.span_id AND sa.\"key\" = {key_ph} AND \
                 {content_like})",
            ));
        }
        return match comparator {
            "LIKE" => {
                let idx = ph.like(binds, pattern);
                Ok(dialect.case_sensitive_like("s.content", idx))
            }
            "ILIKE" => {
                let p = ph.next(binds, Val::Text(pattern));
                let _ = p;
                let idx = ph.idx;
                Ok(dialect.case_insensitive_like("s.content", idx))
            }
            other => Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator '{other}' for span attribute"
            ))),
        };
    }
    let column = match key {
        "name" => "s.name",
        "type" => "s.\"type\"",
        "status" => "s.status",
        "content" => "s.content",
        other => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid span attribute '{other}'."
            )))
        }
    };
    value_predicate(dialect, column, comparator, value, false, ph, binds)
}

/// Assessment (feedback/expectation) EXISTS predicate over `assessments`.
fn build_assessment_predicate(
    dialect: Dialect,
    assessment_type: &str,
    name: &str,
    comparator: &str,
    value: &Value,
    ph: &mut Ph,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    let type_ph = ph.next(binds, Val::Text(assessment_type.to_string()));
    let name_ph = ph.next(binds, Val::Text(name.to_string()));
    let base = format!(
        "a.trace_id = ti.request_id AND a.assessment_type = {type_ph} AND a.name = {name_ph} \
         AND a.valid = {}",
        true_literal(dialect)
    );
    if comparator == "IS NULL" {
        return Ok(format!(
            "NOT EXISTS (SELECT 1 FROM assessments a WHERE {base})"
        ));
    }
    if comparator == "IS NOT NULL" {
        return Ok(format!("EXISTS (SELECT 1 FROM assessments a WHERE {base})"));
    }
    // Value comparison against the JSON `value` column. Assessment values are
    // JSON-encoded; numeric comparators compare the raw text (parity with
    // Python's JSON comparison which operates on the stored text).
    let numeric = matches!(comparator, ">" | ">=" | "<" | "<=");
    let val_pred = value_predicate(dialect, "a.value", comparator, value, numeric, ph, binds)?;
    Ok(format!(
        "EXISTS (SELECT 1 FROM assessments a WHERE {base} AND {val_pred})"
    ))
}

fn true_literal(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Postgres => "TRUE",
        // SQLite/MySQL store booleans as 1/0.
        Dialect::Sqlite | Dialect::MySql => "1",
    }
}

/// Render a comparison predicate on a column, handling `=,!=,<,<=,>,>=,LIKE,
/// ILIKE,IN,NOT IN`.
#[allow(clippy::too_many_arguments)]
fn value_predicate(
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value: &Value,
    numeric: bool,
    ph: &mut Ph,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    match comparator {
        "LIKE" => {
            let idx = ph.like(binds, as_str(value)?);
            Ok(dialect.case_sensitive_like(column, idx))
        }
        "ILIKE" => {
            let p = ph.next(binds, Val::Text(as_str(value)?));
            let _ = p;
            Ok(dialect.case_insensitive_like(column, ph.idx))
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
                .map(|it| ph.next(binds, Val::Text(it.clone())))
                .collect();
            let op = if comparator == "IN" { "IN" } else { "NOT IN" };
            Ok(format!("{column} {op} ({})", phs.join(", ")))
        }
        "=" | "!=" | "<" | "<=" | ">" | ">=" => {
            let p = if numeric {
                ph.next(binds, Val::Int(as_i64(value)?))
            } else {
                ph.next(binds, Val::Text(as_str(value)?))
            };
            Ok(format!("{column} {comparator} {p}"))
        }
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid comparator '{other}'"
        ))),
    }
}

/// `SearchTraceUtils.VALID_STRING_ATTRIBUTE_COMPARATORS` (string attributes) and
/// `VALID_NUMERIC_ATTRIBUTE_COMPARATORS` (numeric attributes).
fn validate_attr_comparator(comparator: &str, numeric: bool) -> Result<(), MlflowError> {
    let valid: &[&str] = if numeric {
        &[">", ">=", "!=", "=", "<", "<="]
    } else {
        &["!=", "=", "IN", "NOT IN", "LIKE", "ILIKE", "RLIKE"]
    };
    if !valid.contains(&comparator) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid comparator '{comparator}' not one of '{valid:?}'"
        )));
    }
    Ok(())
}

/// `SearchTraceUtils.VALID_TAG_COMPARATORS` / `VALID_METADATA_COMPARATORS`
/// (identical sets).
fn validate_kv_comparator(comparator: &str) -> Result<(), MlflowError> {
    const VALID: &[&str] = &[
        "!=",
        "=",
        "LIKE",
        "ILIKE",
        "RLIKE",
        "IS NULL",
        "IS NOT NULL",
    ];
    if !VALID.contains(&comparator) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid comparator '{comparator}' not one of '{VALID:?}'"
        )));
    }
    Ok(())
}

fn attr_filter_column(key: &str) -> Result<&'static str, MlflowError> {
    Ok(match key {
        "timestamp_ms" => "timestamp_ms",
        "execution_time_ms" => "execution_time_ms",
        "status" => "status",
        "request_id" => "request_id",
        "client_request_id" => "client_request_id",
        "experiment_id" => "experiment_id",
        other => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid attribute key '{other}' specified."
            )))
        }
    })
}

fn is_numeric_attr(key: &str) -> bool {
    matches!(
        key,
        "timestamp_ms" | "execution_time_ms" | "end_time_ms" | "end_time" | "timestamp"
    )
}

fn as_str(value: &Value) -> Result<String, MlflowError> {
    match value {
        Value::Str(s) => Ok(s.clone()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(f.to_string()),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a string value",
        )),
    }
}

fn as_i64(value: &Value) -> Result<i64, MlflowError> {
    match value {
        Value::Int(i) => Ok(*i),
        Value::Float(f) => Ok(*f as i64),
        Value::Str(s) => s
            .parse::<i64>()
            .map_err(|_| MlflowError::invalid_parameter_value("Expected a numeric value")),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a numeric value",
        )),
    }
}

fn as_list(value: &Value) -> Result<Vec<String>, MlflowError> {
    match value {
        Value::List(items) => Ok(items.clone()),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a list value for IN/NOT IN",
        )),
    }
}

// ===========================================================================
// Offset page token (base64(JSON {"offset": N})) — Python parity
// ===========================================================================

/// Parse the opaque offset token. The payload is the minimal JSON
/// `{"offset": N}` (base64-encoded), matching Python's
/// `SearchTraceUtils.create_page_token`. We parse it by hand to avoid pulling
/// `serde_json` into the crate's runtime dependencies.
fn parse_offset_token(token: &str) -> Result<i64, MlflowError> {
    let bad = || MlflowError::invalid_parameter_value("Invalid page token");
    let bytes = base64_decode(token).ok_or_else(bad)?;
    let text = String::from_utf8(bytes).map_err(|_| bad())?;
    // Expect `{"offset": <digits>}` (with arbitrary interior whitespace).
    let inner = text.trim();
    let inner = inner
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(bad)?;
    let (k, v) = inner.split_once(':').ok_or_else(bad)?;
    if k.trim().trim_matches('"') != "offset" {
        return Err(bad());
    }
    let offset: i64 = v.trim().parse().map_err(|_| bad())?;
    if offset < 0 {
        return Err(bad());
    }
    Ok(offset)
}

fn encode_offset_token(offset: i64) -> String {
    base64_encode(format!("{{\"offset\": {offset}}}").as_bytes())
}

// Standard-alphabet base64 (same implementation as `search.rs`, kept local so
// trace search does not depend on the run-search module's wiring).
fn base64_encode(input: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(A[((n >> 18) & 63) as usize] as char);
        out.push(A[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            A[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            A[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some(u32::from(c - b'A')),
            b'a'..=b'z' => Some(u32::from(c - b'a' + 26)),
            b'0'..=b'9' => Some(u32::from(c - b'0' + 52)),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let cleaned: Vec<u8> = input.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::new();
    for chunk in cleaned.chunks(4) {
        if chunk.len() == 1 {
            return None;
        }
        let mut n = 0u32;
        let mut bits = 0;
        for &c in chunk {
            n = (n << 6) | val(c)?;
            bits += 6;
        }
        let bytes = bits / 8;
        n <<= (4 - chunk.len()) as u32 * 6;
        let full = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&full[..bytes]);
    }
    Some(out)
}
