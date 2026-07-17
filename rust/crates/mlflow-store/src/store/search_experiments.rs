//! `search_experiments` — the store-level experiment search (plan T3.1 wiring,
//! mirroring `SqlAlchemyStore._search_experiments`
//! (`sqlalchemy_store.py:591-641`), `_get_search_experiments_filter_clauses`
//! (`:9240`) and `_get_search_experiments_order_by_clauses` (`:9294`)).
//!
//! Ported semantics (must match Python byte-for-byte through the HTTP layer):
//!
//! * **Filter** — `attribute` clauses (`name` string; `creation_time`/
//!   `last_update_time` numeric) become direct predicates on `experiments`;
//!   `tag` clauses become correlated `EXISTS` subqueries against
//!   `experiment_tags` (Python joins a per-tag subquery, but an EXISTS
//!   semi-join is observationally identical and avoids row fan-out —
//!   consistent with the §5.2 Q2/Q3 improvements applied in `search_runs`).
//!   `tags IS NULL`/`IS NOT NULL` map to `NOT EXISTS`/`EXISTS`.
//! * **Comparator validation** — Python validates comparators at
//!   filter-application time (not parse time): string attributes accept
//!   `=,!=,LIKE,ILIKE`; numeric attributes accept `=,!=,<,<=,>,>=`; tags accept
//!   `=,!=,LIKE,ILIKE` plus the `IS NULL`/`IS NOT NULL` forms. The verbatim
//!   error messages match `_get_search_experiments_filter_clauses`.
//! * **Order by** — defaults to `creation_time DESC, experiment_id ASC`; a
//!   trailing `experiment_id ASC` tiebreak is appended unless the user already
//!   ordered by `experiment_id`. Only `attribute` order-by is valid.
//! * **Pagination** — offset-based `base64(json {"offset": N})` tokens,
//!   identical to Python's `SearchExperimentsUtils.create_page_token` /
//!   `parse_start_offset_from_page_token` (§4 item 7 permits keeping the offset
//!   token here; experiment search has no keyset story to preserve and the
//!   token stays opaque). Over-fetch `max_results + 1` to detect a next page.
//! * **max_results** — validated `[1, 50000]` (`allow_null=False`), so an unset
//!   proto value (0) is rejected exactly as Python's handler pass-through does.

use mlflow_error::MlflowError;
use mlflow_search::{parse_start_offset_from_page_token, Comparison, OrderBy, Value};

use super::dbutil::Val;
use super::entities::{Experiment, ExperimentTag};
use super::experiments::{internal, ViewType};
use super::search::SEARCH_MAX_RESULTS_THRESHOLD;
use super::TrackingStore;
use crate::dialect::Dialect;
use crate::schema::runs::{EXPERIMENTS, EXPERIMENT_TAGS};

/// A page of experiments plus the optional next-page token.
#[derive(Debug)]
pub struct ExperimentsPage {
    pub experiments: Vec<Experiment>,
    pub next_page_token: Option<String>,
}

/// Valid comparators per identifier kind, matching
/// `_get_search_experiments_filter_clauses`.
const VALID_STRING_ATTR_COMPARATORS: &[&str] = &["=", "!=", "LIKE", "ILIKE"];
const VALID_NUMERIC_ATTR_COMPARATORS: &[&str] = &["=", "!=", "<", "<=", ">", ">="];
const VALID_TAG_COMPARATORS: &[&str] = &["=", "!=", "LIKE", "ILIKE"];

const NUMERIC_ATTRIBUTES: &[&str] = &["creation_time", "last_update_time"];

impl TrackingStore {
    /// `search_experiments`. `view_type` selects the lifecycle stages; `None`
    /// mirrors an *unspecified* proto `ViewType` (value `0`), which
    /// `LifecycleStage.view_type_to_stages` maps to an **empty** stage set —
    /// i.e. no experiments match (the raw-request edge case; real clients always
    /// send `ACTIVE_ONLY`). `max_results` is the raw (non-null) value from the
    /// request (validated `[1, 50000]`); `filter` and `order_by` are raw strings
    /// parsed via `mlflow-search`; `page_token` is the offset token.
    pub async fn search_experiments(
        &self,
        workspace: &str,
        view_type: Option<ViewType>,
        max_results: i64,
        filter: Option<&str>,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<ExperimentsPage, MlflowError> {
        validate_max_results(max_results)?;

        let parsed =
            mlflow_search::parse::experiments_filter(filter.unwrap_or("")).map_err(search_err)?;
        let order_cols = build_order_by(order_by)?;
        let offset = parse_start_offset_from_page_token(page_token).map_err(search_err)?;

        let dialect = self.db().dialect();
        // Unspecified proto view type → empty stages → no rows (Python parity).
        let stages: &[&str] = match view_type {
            Some(vt) => vt.stages(),
            None => &[],
        };
        if stages.is_empty() {
            return Ok(ExperimentsPage {
                experiments: vec![],
                next_page_token: None,
            });
        }

        // Build the WHERE clause: workspace + lifecycle + filter predicates.
        let mut binds: Vec<Val> = Vec::new();
        let mut ph = PlaceholderGen::new(dialect);
        let mut where_parts: Vec<String> = Vec::new();

        where_parts.push(format!(
            "e.workspace = {}",
            ph.next_bind(&mut binds, Val::Text(workspace.to_string()))
        ));

        let stage_phs: Vec<String> = stages
            .iter()
            .map(|s| ph.next_bind(&mut binds, Val::Text((*s).to_string())))
            .collect();
        where_parts.push(format!("e.lifecycle_stage IN ({})", stage_phs.join(", ")));

        for f in &parsed {
            where_parts.push(build_filter_predicate(dialect, f, &mut ph, &mut binds)?);
        }

        let order_sql = order_cols
            .iter()
            .map(|c| {
                format!(
                    "e.{} {}",
                    c.column,
                    if c.ascending { "ASC" } else { "DESC" }
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        // Over-fetch max_results + 1 with the requested offset (mirrors
        // Python's `.offset(offset).limit(max_results + 1)`).
        let limit_ph = ph.next_bind(&mut binds, Val::Int(max_results + 1));
        let offset_ph = ph.next_bind(&mut binds, Val::Int(offset));
        let sql = format!(
            "SELECT e.experiment_id, e.name, e.artifact_location, e.lifecycle_stage, \
             e.creation_time, e.last_update_time FROM {EXPERIMENTS} e \
             WHERE {} ORDER BY {order_sql} LIMIT {limit_ph} OFFSET {offset_ph}",
            where_parts.join(" AND "),
        );

        let mut rows: Vec<Experiment> = self
            .db()
            .fetch_all(&sql, &binds, |r| {
                Ok(Experiment {
                    experiment_id: r.get_int("experiment_id")?.to_string(),
                    name: r.get_string("name")?,
                    artifact_location: r.get_opt_string("artifact_location")?,
                    lifecycle_stage: r.get_string("lifecycle_stage")?,
                    creation_time: r.get_opt_i64("creation_time")?,
                    last_update_time: r.get_opt_i64("last_update_time")?,
                    tags: Vec::new(),
                })
            })
            .await
            .map_err(internal)?;

        // `compute_next_token`: a next token exists iff we fetched max+1 rows.
        let next_page_token = if rows.len() as i64 == max_results + 1 {
            Some(create_page_token(offset + max_results))
        } else {
            None
        };
        rows.truncate(max_results as usize);

        // Eager-load tags per experiment (Python eager-loads via ORM options).
        for exp in &mut rows {
            exp.tags = self.load_experiment_tags(&exp.experiment_id).await?;
        }

        Ok(ExperimentsPage {
            experiments: rows,
            next_page_token,
        })
    }

    async fn load_experiment_tags(
        &self,
        experiment_id: &str,
    ) -> Result<Vec<ExperimentTag>, MlflowError> {
        let id: i64 = experiment_id
            .parse()
            .map_err(|_| internal(sqlx::Error::Protocol("invalid experiment id".into())))?;
        let dialect = self.db().dialect();
        // `key` is unquoted here in an unqualified column-list position, which
        // MySQL parses as the reserved word `KEY` rather than an identifier
        // (syntax error). Every other `key` column reference in this crate
        // quotes it (`"key"`, relying on the session-level ANSI_QUOTES set
        // for MySQL in `db.rs`); this one was missed (plan T2.2 dialect bug).
        let sql = format!(
            "SELECT \"key\", value FROM {EXPERIMENT_TAGS} WHERE experiment_id = {} ORDER BY \"key\"",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Int(id)], |r| {
                Ok(ExperimentTag {
                    key: r.get_string("key")?,
                    value: r.get_opt_string("value")?,
                })
            })
            .await
            .map_err(internal)
    }
}

/// An order-by column resolved to an `experiments` attribute column.
struct OrderCol {
    column: String,
    ascending: bool,
}

/// `_get_search_experiments_order_by_clauses`: default
/// `creation_time DESC, experiment_id ASC`, only attribute keys allowed, and a
/// trailing `experiment_id ASC` tiebreak unless the user already ordered by it.
fn build_order_by(order_by: &[String]) -> Result<Vec<OrderCol>, MlflowError> {
    let clauses: Vec<OrderBy> = if order_by.is_empty() {
        vec![
            parse_order_by_clause("creation_time DESC")?,
            parse_order_by_clause("experiment_id ASC")?,
        ]
    } else {
        order_by
            .iter()
            .map(|s| parse_order_by_clause(s))
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut cols: Vec<OrderCol> = Vec::with_capacity(clauses.len() + 1);
    let mut has_experiment_id = false;
    for c in clauses {
        if c.entity_type != "attribute" {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid order_by entity: {}",
                c.entity_type
            )));
        }
        if c.key == "experiment_id" {
            has_experiment_id = true;
        }
        cols.push(OrderCol {
            column: c.key,
            ascending: c.ascending,
        });
    }
    if !has_experiment_id {
        cols.push(OrderCol {
            column: "experiment_id".to_string(),
            ascending: false,
        });
    }
    Ok(cols)
}

fn parse_order_by_clause(s: &str) -> Result<OrderBy, MlflowError> {
    mlflow_search::parse::experiments_order_by(s).map_err(search_err)
}

/// Build a single filter predicate for the WHERE clause. `attribute` clauses
/// become direct predicates; `tag` clauses become `EXISTS`/`NOT EXISTS`
/// subqueries over `experiment_tags`.
fn build_filter_predicate(
    dialect: Dialect,
    f: &Comparison,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    match f.entity_type.as_str() {
        "attribute" => build_attribute_predicate(dialect, f, ph, binds),
        "tag" => build_tag_predicate(dialect, f, ph, binds),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid token type: {other}"
        ))),
    }
}

fn build_attribute_predicate(
    dialect: Dialect,
    f: &Comparison,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    let comparator = f.comparator.to_uppercase();
    let is_numeric = NUMERIC_ATTRIBUTES.contains(&f.key.as_str());
    if is_numeric {
        if !VALID_NUMERIC_ATTR_COMPARATORS.contains(&comparator.as_str()) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator for numeric attribute: {}",
                f.comparator
            )));
        }
        let n = numeric_value(&f.value)?;
        let p = ph.next_bind(binds, Val::Int(n));
        Ok(format!("e.{} {comparator} {p}", f.key))
    } else {
        // String attribute (`name`).
        if !VALID_STRING_ATTR_COMPARATORS.contains(&comparator.as_str()) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator for string attribute: {}",
                f.comparator
            )));
        }
        Ok(string_predicate(
            dialect,
            &format!("e.{}", f.key),
            &comparator,
            &as_str(&f.value)?,
            ph,
            binds,
        ))
    }
}

fn build_tag_predicate(
    dialect: Dialect,
    f: &Comparison,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    let comparator = f.comparator.to_uppercase();
    if comparator == "IS NULL" || comparator == "IS NOT NULL" {
        let key_ph = ph.next_bind(binds, Val::Text(f.key.clone()));
        let exists = format!(
            "EXISTS (SELECT 1 FROM {EXPERIMENT_TAGS} t \
             WHERE t.experiment_id = e.experiment_id AND t.key = {key_ph})"
        );
        return Ok(if comparator == "IS NULL" {
            format!("NOT {exists}")
        } else {
            exists
        });
    }
    if !VALID_TAG_COMPARATORS.contains(&comparator.as_str()) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid comparator for tag: {}",
            f.comparator
        )));
    }
    let key_ph = ph.next_bind(binds, Val::Text(f.key.clone()));
    let val_pred = string_predicate(
        dialect,
        "t.value",
        &comparator,
        &as_str(&f.value)?,
        ph,
        binds,
    );
    Ok(format!(
        "EXISTS (SELECT 1 FROM {EXPERIMENT_TAGS} t \
         WHERE t.experiment_id = e.experiment_id AND t.key = {key_ph} AND {val_pred})"
    ))
}

/// Render a `=,!=,LIKE,ILIKE` predicate on a string column, matching
/// `get_sql_comparison_func` (LIKE/ILIKE case semantics per dialect).
fn string_predicate(
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value: &str,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> String {
    match comparator {
        "LIKE" => {
            let idx = ph.reserve_like(dialect, binds, value);
            dialect.case_sensitive_like(column, idx)
        }
        "ILIKE" => {
            let idx = ph.next_index(binds, Val::Text(value.to_string()));
            dialect.case_insensitive_like(column, idx)
        }
        _ => {
            let p = ph.next_bind(binds, Val::Text(value.to_string()));
            format!("{column} {comparator} {p}")
        }
    }
}

fn as_str(value: &Value) -> Result<String, MlflowError> {
    match value {
        Value::Str(s) => Ok(s.clone()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(f.to_string()),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Expected a string value, got {other:?}"
        ))),
    }
}

fn numeric_value(value: &Value) -> Result<i64, MlflowError> {
    match value {
        Value::Int(i) => Ok(*i),
        Value::Str(s) => s.parse::<i64>().or_else(|_| {
            s.parse::<f64>().map(|f| f as i64).map_err(|_| {
                MlflowError::invalid_parameter_value(format!("Expected a numeric value, got '{s}'"))
            })
        }),
        Value::Float(f) => Ok(*f as i64),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Expected a numeric value, got {other:?}"
        ))),
    }
}

/// `_validate_max_results_param(max_results)` with `allow_null=False`.
fn validate_max_results(max_results: i64) -> Result<(), MlflowError> {
    if max_results < 1 {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. It must be \
             a positive integer"
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

/// `SearchExperimentsUtils.create_page_token`: `base64(json.dumps({"offset": N}))`.
/// `json.dumps` default separators render `{"offset": N}` (space after colon).
fn create_page_token(offset: i64) -> String {
    use base64::Engine;
    let json = format!("{{\"offset\": {offset}}}");
    base64::engine::general_purpose::STANDARD.encode(json.as_bytes())
}

fn search_err(e: mlflow_search::SearchError) -> MlflowError {
    use mlflow_error::ErrorCode;
    let code = match e.error_code {
        mlflow_search::ErrorCode::InvalidParameterValue => ErrorCode::InvalidParameterValue,
        _ => ErrorCode::InternalError,
    };
    MlflowError::new(e.message, code)
}

/// A positional placeholder generator that appends binds in lockstep so the
/// placeholder index and the `binds` vector never drift. Carries the dialect so
/// it can render backend-appropriate placeholder strings (`?` / `$N`).
struct PlaceholderGen {
    dialect: Dialect,
    next: usize,
}

impl PlaceholderGen {
    fn new(dialect: Dialect) -> Self {
        Self { dialect, next: 1 }
    }

    fn next_index_only(&mut self) -> usize {
        let idx = self.next;
        self.next += 1;
        idx
    }

    /// Push a bind and return its rendered placeholder string.
    fn next_bind(&mut self, binds: &mut Vec<Val>, v: Val) -> String {
        binds.push(v);
        self.dialect.placeholder(self.next_index_only())
    }

    /// Push a bind and return its 1-based placeholder index (for `LIKE` helpers
    /// that build their own placeholder string).
    fn next_index(&mut self, binds: &mut Vec<Val>, v: Val) -> usize {
        binds.push(v);
        self.next_index_only()
    }

    /// Reserve placeholder slot(s) for a case-sensitive LIKE. MySQL needs the
    /// pattern bound twice (`col LIKE ? AND BINARY col LIKE ?`).
    fn reserve_like(&mut self, dialect: Dialect, binds: &mut Vec<Val>, value: &str) -> usize {
        binds.push(Val::Text(value.to_string()));
        let idx = self.next_index_only();
        if let Dialect::MySql = dialect {
            binds.push(Val::Text(value.to_string()));
            self.next_index_only();
        }
        idx
    }
}
