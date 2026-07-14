//! `search_runs` — the store-level run search (plan T2.6), mirroring
//! `SqlAlchemyStore._search_runs` (`sqlalchemy_store.py:2006`),
//! `_get_sqlalchemy_filter_clauses` (`:9054`) and `_get_orderby_clauses`
//! (`:9158`) — but with the plan's §5.2 query improvements:
//!
//! * **Q2/Q3 — EXISTS semi-joins** instead of `SELECT DISTINCT` over N filter
//!   joins. Each filter comparison becomes a correlated `EXISTS`/`NOT EXISTS`
//!   subquery (metrics against `latest_metrics`, params/tags against their
//!   tables, attributes as direct predicates, datasets via `inputs`). No
//!   row fan-out, so no `DISTINCT` is needed.
//! * **Q1 — keyset pagination** behind an opaque base64(JSON) token, instead of
//!   `OFFSET`. The cursor is the full ordered key tuple (every order-by value +
//!   the `start_time`/`run_uuid` tiebreak), with per-key null flags so the
//!   NULLS-LAST emulation round-trips.
//! * **Q8 — batched eager loading**: params/latest-metrics/tags/inputs/outputs
//!   for the whole page are read with `IN (...)` queries, not per run.
//!
//! ## Ordering parity (must match Python byte-for-byte)
//!
//! `_get_orderby_clauses` emits, per user order-by clause, a null-rank `CASE`
//! (always ASC) followed by the value (ASC/DESC). For metrics the rank is
//! `is_nan => 1, value NULL => 2, else 0`; for everything else `value NULL => 1
//! else 0`. It then appends `start_time DESC` (unless the user already ordered
//! by `start_time`) and finally `run_uuid ASC`, both *raw* (DB-default NULLS
//! placement). We reproduce that exact column sequence in [`SortCol`], and both
//! the `ORDER BY` string and the keyset predicate are generated from that one
//! list so they can never drift.
//!
//! ## NULLS-LAST emulation per dialect
//!
//! Because the null-rank `CASE` precedes each value column, NaN/NULL rows always
//! sort last regardless of ASC/DESC — this is dialect-independent (works on
//! MySQL, which lacks `NULLS LAST`). The only DB-default-NULLS column is the raw
//! `start_time DESC` tiebreak, which Python emits with no explicit `NULLS`
//! clause: on SQLite/MySQL `DESC` places NULLs **last** (NULL ranks smallest),
//! while Postgres `DESC` places NULLs **first** (NULL ranks largest). We encode
//! that per-dialect placement into both the `ORDER BY` (explicit `NULLS
//! FIRST/LAST` on SQLite/Postgres; bare on MySQL, whose default matches) and the
//! keyset comparison, so page boundaries stay identical. See [`apply_dialect_nulls`].

use std::collections::HashMap;

use mlflow_error::MlflowError;
use mlflow_search::{Comparison, OrderBy, Value};

use super::dbutil::{RowLike, Val};
use super::entities::{LifecycleStage, Metric, Param, Run, RunData, RunTag};
use super::experiments::{internal, ViewType};
use super::runs::RunRow;
use super::{TrackingStore, MLFLOW_RUN_NAME};
use crate::dialect::Dialect;

/// `SEARCH_MAX_RESULTS_THRESHOLD` (`mlflow/store/tracking/__init__.py:21`).
pub const SEARCH_MAX_RESULTS_THRESHOLD: i64 = 50000;

/// `SEARCH_MAX_RESULTS_DEFAULT` (`mlflow/store/tracking/__init__.py:20`). Applied
/// by the *handler* layer (Phase 3), not the store — `_search_runs` itself
/// treats `max_results = None` as "no limit".
pub const SEARCH_MAX_RESULTS_DEFAULT: i64 = 1000;

/// The `mlflow.datasets.context` input-tag name (`MLFLOW_DATASET_CONTEXT`).
const MLFLOW_DATASET_CONTEXT: &str = "mlflow.data.context";

// Per-entity valid comparator sets, mirroring `SearchUtils`
// (`search_utils.py:178-183`). Python validates these at *filter-application*
// time (`is_metric`/`is_param`/… inside `_get_sqlalchemy_filter_clauses`), NOT
// at parse time — the `mlflow-search` parser accepts e.g. `params.p > 'x'`, so
// the store must reject it here with the same message/code. The set order below
// matches Python's `set` repr rendered by [`py_set`] (sorted), used to build the
// verbatim error strings (including Python's occasional missing closing quote).
const VALID_METRIC_COMPARATORS: &[&str] = &[">", ">=", "!=", "=", "<", "<="];
const VALID_PARAM_COMPARATORS: &[&str] = &["!=", "=", "LIKE", "ILIKE", "IS NULL", "IS NOT NULL"];
const VALID_TAG_COMPARATORS: &[&str] = &["!=", "=", "LIKE", "ILIKE", "IS NULL", "IS NOT NULL"];
const VALID_STRING_ATTRIBUTE_COMPARATORS: &[&str] = &["!=", "=", "LIKE", "ILIKE", "IN", "NOT IN"];
const VALID_DATASET_COMPARATORS: &[&str] = &["!=", "=", "LIKE", "ILIKE", "IN", "NOT IN"];

/// A page of runs plus the optional next-page token.
#[derive(Debug)]
pub struct RunsPage {
    pub runs: Vec<Run>,
    pub next_page_token: Option<String>,
}

impl TrackingStore {
    /// `search_runs` — see the module docs.
    ///
    /// `experiment_ids` are stringified experiment ids (as they arrive on the
    /// wire); `filter` is the raw filter string (parsed via `mlflow-search`);
    /// `order_by` is the list of raw order-by strings. `max_results = None`
    /// means "no limit" (Python's `allow_null=True`); otherwise it is validated
    /// against `[1, 50000]`. Workspace scoping filters `experiment_ids` down to
    /// those in the active workspace (mirrors
    /// `WorkspaceAwareSqlAlchemyStore._filter_experiment_ids`).
    #[allow(clippy::too_many_arguments)]
    pub async fn search_runs(
        &self,
        workspace: &str,
        experiment_ids: &[String],
        filter: Option<&str>,
        run_view_type: ViewType,
        max_results: Option<i64>,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<RunsPage, MlflowError> {
        validate_max_results(max_results)?;

        let parsed_filters = parse_filter(filter)?;
        let order_cols = build_order_cols(order_by)?;

        let dialect = self.db().dialect();

        // Workspace + numeric experiment-id filtering (mirrors
        // `_filter_experiment_ids`). Non-numeric ids raise the same error the
        // Python `int(e)` conversion would.
        let exp_ids = self
            .filter_experiment_ids_for_workspace(workspace, experiment_ids)
            .await?;

        let stages = view_type_stages(run_view_type);

        // Empty experiment set (either none requested or none in the workspace)
        // -> no runs, no next page. Matches Python's `IN ()` producing nothing.
        if exp_ids.is_empty() {
            return Ok(RunsPage {
                runs: vec![],
                next_page_token: None,
            });
        }

        let cursor = match page_token {
            Some(t) => Some(Cursor::decode(t, order_cols.len())?),
            None => None,
        };

        let query = build_search_sql(
            dialect,
            &exp_ids,
            stages,
            &parsed_filters,
            &order_cols,
            cursor.as_ref(),
            max_results,
        )?;

        let kinds: Vec<ColKind> = order_cols.iter().map(|c| c.kind).collect();
        let rows: Vec<SearchRow> = self
            .db()
            .fetch_all(&query.sql, &query.binds, |r| SearchRow::from_row(r, &kinds))
            .await
            .map_err(internal)?;

        // Keyset over-fetch: request max+1, keep max, emit a token from the last
        // kept row's key tuple.
        let (page_rows, next_token) = match max_results {
            Some(mr) if rows.len() as i64 == mr + 1 => {
                let kept = &rows[..mr as usize];
                let last = kept.last().expect("mr>=1 so kept is non-empty");
                (kept, Some(Cursor::encode(&last.keys)))
            }
            _ => (&rows[..], None),
        };

        let run_ids: Vec<String> = page_rows.iter().map(|r| r.run.run_uuid.clone()).collect();
        let runs = self.assemble_runs(page_rows, &run_ids).await?;

        Ok(RunsPage {
            runs,
            next_page_token: next_token,
        })
    }

    /// Filter `experiment_ids` (stringified) to those present in `workspace`,
    /// preserving order and de-duplicating. Non-integer ids raise the same
    /// `INVALID_PARAMETER_VALUE` Python's `int(e)` would.
    async fn filter_experiment_ids_for_workspace(
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
            "SELECT experiment_id FROM experiments \
             WHERE experiment_id IN ({}) AND workspace = {}",
            placeholders.join(", "),
            dialect.placeholder(wanted.len() + 1)
        );
        let present: Vec<i64> = self
            .db()
            .fetch_all(&sql, &binds, |r| r.get_int("experiment_id"))
            .await
            .map_err(internal)?;

        // Preserve requested order, keep only those present.
        Ok(wanted
            .into_iter()
            .filter(|id| present.contains(id))
            .collect())
    }

    /// Assemble full [`Run`] entities for the page: batched eager loading (Q8)
    /// of params, latest metrics, tags, dataset/model inputs, and model outputs.
    async fn assemble_runs(
        &self,
        rows: &[SearchRow],
        run_ids: &[String],
    ) -> Result<Vec<Run>, MlflowError> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let mut params = self.load_params_bulk(run_ids).await?;
        let mut metrics = self.load_latest_metrics_bulk(run_ids).await?;
        let mut tags = self.load_tags_bulk(run_ids).await?;
        let mut inputs = self.load_run_inputs_bulk(run_ids).await?;
        let mut outputs = self.load_run_outputs_bulk(run_ids).await?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = &row.run.run_uuid;
            let mut info = self.run_info_from_row(&row.run);
            let run_tags = tags.remove(id).unwrap_or_default();
            // Python fills an empty `runs.name` from the mlflow.runName tag.
            if info.run_name.is_empty() {
                if let Some(t) = run_tags.iter().find(|t| t.key == MLFLOW_RUN_NAME) {
                    info.run_name = t.value.clone();
                }
            }
            out.push(Run {
                info,
                data: RunData {
                    metrics: metrics.remove(id).unwrap_or_default(),
                    params: params.remove(id).unwrap_or_default(),
                    tags: run_tags,
                },
                inputs: inputs.remove(id).unwrap_or_default(),
                outputs: outputs.remove(id).unwrap_or_default(),
            });
        }
        Ok(out)
    }
}

/// Return the lifecycle stages for a view type (`view_type_to_stages`).
fn view_type_stages(vt: ViewType) -> &'static [&'static str] {
    match vt {
        ViewType::ActiveOnly => &[LifecycleStage::ACTIVE],
        ViewType::DeletedOnly => &[LifecycleStage::DELETED],
        ViewType::All => &[LifecycleStage::ACTIVE, LifecycleStage::DELETED],
    }
}

/// `_validate_max_results_param(max_results, allow_null=True)`.
fn validate_max_results(max_results: Option<i64>) -> Result<(), MlflowError> {
    if let Some(mr) = max_results {
        if mr < 1 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {mr} for parameter 'max_results' supplied. It must be \
                 a positive integer"
            )));
        }
        if mr > SEARCH_MAX_RESULTS_THRESHOLD {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {mr} for parameter 'max_results' supplied. It must be at \
                 most {SEARCH_MAX_RESULTS_THRESHOLD}"
            )));
        }
    }
    Ok(())
}

/// Parse the filter string via `mlflow-search`, mapping parse errors to
/// `MlflowError` with the verbatim Python message + code.
fn parse_filter(filter: Option<&str>) -> Result<Vec<Comparison>, MlflowError> {
    let s = filter.unwrap_or("");
    mlflow_search::parse::runs_filter(s).map_err(search_err)
}

fn search_err(e: mlflow_search::SearchError) -> MlflowError {
    use mlflow_error::ErrorCode;
    let code = match e.error_code {
        mlflow_search::ErrorCode::InvalidParameterValue => ErrorCode::InvalidParameterValue,
        // The remaining variants mark Python dead-ends / uncaught ValueErrors;
        // both surface as 500 (INTERNAL_ERROR), matching Python.
        _ => ErrorCode::InternalError,
    };
    MlflowError::new(e.message, code)
}

// ===========================================================================
// Order-by column model
// ===========================================================================

/// The kind of value an order/keyset column carries (drives casting + JSON).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColKind {
    /// Non-null integer rank column (the NULLS-LAST `CASE`). Always ASC.
    Rank,
    /// Numeric value (metric value, or a numeric attribute like start_time).
    Num,
    /// String value (param/tag value, or a string attribute).
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dir {
    Asc,
    Desc,
}

/// How NULLs are placed for this column, so the keyset comparison matches the
/// emitted `ORDER BY`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Nulls {
    /// Column is non-nullable (rank cols, run_uuid) — no NULL handling.
    NonNull,
    /// NULLs sort first (SQLite/Postgres `DESC` default on the raw start_time).
    First,
    /// NULLs sort last (MySQL `DESC` default; also value columns whose NULLs are
    /// already separated by a preceding rank column, so treated as last).
    Last,
}

/// One `ORDER BY` column, and the source of truth for the keyset predicate.
#[derive(Debug, Clone)]
struct SortCol {
    /// The SQL value expression (already qualified/safe).
    expr: String,
    dir: Dir,
    nulls: Nulls,
    kind: ColKind,
}

/// A user order-by target resolved against the run schema.
enum OrderTarget {
    /// A run attribute column (already the SqlRun column name).
    Attr { col: &'static str, num: bool },
    /// A joined entity value (`latest_metrics`/`params`/`tags`).
    Joined { entity: JoinEntity, key: String },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum JoinEntity {
    Metric,
    Param,
    Tag,
}

impl JoinEntity {
    fn table(self) -> &'static str {
        match self {
            JoinEntity::Metric => "latest_metrics",
            JoinEntity::Param => "params",
            JoinEntity::Tag => "tags",
        }
    }
}

/// Build the ordered [`SortCol`] list from the user order-by list, appended with
/// the `start_time DESC` (if absent) and `run_uuid ASC` tiebreaks — exactly
/// `_get_orderby_clauses`.
fn build_order_cols(order_by: &[String]) -> Result<Vec<SortCol>, MlflowError> {
    let mut cols: Vec<SortCol> = Vec::new();
    let mut observed: Vec<(String, String)> = Vec::new();
    let mut join_idx = 0usize;
    let mut start_time_specified = false;

    for clause in order_by {
        let ob: OrderBy = mlflow_search::parse::runs_order_by(clause).map_err(search_err)?;
        let key = translate_key_alias(&ob.key);
        let target = resolve_order_target(&ob.entity_type, &key)?;

        // Duplicate-field detection uses (key_type, key) with the resolved key.
        let dedup_key = (ob.entity_type.clone(), key.clone());
        if observed.contains(&dedup_key) {
            return Err(MlflowError::new(
                format!("`order_by` contains duplicate fields: {order_by:?}"),
                mlflow_error::ErrorCode::InternalError,
            ));
        }
        observed.push(dedup_key);

        if ob.entity_type == "attribute" && key == "start_time" {
            start_time_specified = true;
        }

        let dir = if ob.ascending { Dir::Asc } else { Dir::Desc };

        match target {
            OrderTarget::Attr { col, num } => {
                // Null-rank CASE (value NULL => 1 else 0), ASC.
                cols.push(SortCol {
                    expr: format!("(CASE WHEN r.{col} IS NULL THEN 1 ELSE 0 END)"),
                    dir: Dir::Asc,
                    nulls: Nulls::NonNull,
                    kind: ColKind::Rank,
                });
                cols.push(SortCol {
                    expr: format!("r.{col}"),
                    dir,
                    nulls: Nulls::Last,
                    kind: if num { ColKind::Num } else { ColKind::Text },
                });
            }
            OrderTarget::Joined { entity, key: k } => {
                let alias = format!("oj_{join_idx}");
                join_idx += 1;
                let val = format!("{alias}.value");
                let rank_expr = if entity == JoinEntity::Metric {
                    // metric: is_nan => 1, value NULL => 2, else 0.
                    format!(
                        "(CASE WHEN {alias}.is_nan = 1 THEN 1 \
                         WHEN {val} IS NULL THEN 2 ELSE 0 END)"
                    )
                } else {
                    format!("(CASE WHEN {val} IS NULL THEN 1 ELSE 0 END)")
                };
                cols.push(SortCol {
                    expr: rank_expr,
                    dir: Dir::Asc,
                    nulls: Nulls::NonNull,
                    kind: ColKind::Rank,
                });
                cols.push(SortCol {
                    expr: val,
                    dir,
                    nulls: Nulls::Last,
                    kind: if entity == JoinEntity::Metric {
                        ColKind::Num
                    } else {
                        ColKind::Text
                    },
                });
                // Remember the join so the SQL builder can emit it. We stash the
                // entity+key on the SortCol expr via a side table below; simpler:
                // rebuild joins from cols is hard, so record here.
                JOIN_REGISTRY.with(|reg| {
                    reg.borrow_mut().push(JoinSpec {
                        alias,
                        table: entity.table(),
                        key: k,
                        is_metric: entity == JoinEntity::Metric,
                    })
                });
            }
        }
    }

    if !start_time_specified {
        // Raw `start_time DESC` — DB-default NULLS placement (set per dialect in
        // the SQL builder / keyset comparison).
        cols.push(SortCol {
            expr: "r.start_time".to_string(),
            dir: Dir::Desc,
            nulls: Nulls::First, // overwritten per-dialect in the builder
            kind: ColKind::Num,
        });
    }
    // run_uuid ASC, non-null.
    cols.push(SortCol {
        expr: "r.run_uuid".to_string(),
        dir: Dir::Asc,
        nulls: Nulls::NonNull,
        kind: ColKind::Text,
    });

    Ok(cols)
}

// The order-by joins collected while building `SortCol`s. Using a thread-local
// keeps `build_order_cols` a pure function of its inputs while still letting the
// SQL builder pick up the joins in order; it is drained per `search_runs` call.
#[derive(Debug, Clone)]
struct JoinSpec {
    alias: String,
    table: &'static str,
    key: String,
    is_metric: bool,
}

thread_local! {
    static JOIN_REGISTRY: std::cell::RefCell<Vec<JoinSpec>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `_get_identifier`-resolved order target validation. The parser already
/// classified the entity type; here we map to a column/entity or reject.
fn resolve_order_target(entity_type: &str, key: &str) -> Result<OrderTarget, MlflowError> {
    match entity_type {
        "attribute" => {
            let col = attr_column(key)?;
            Ok(OrderTarget::Attr {
                col,
                num: is_numeric_attr(key),
            })
        }
        "metric" => Ok(OrderTarget::Joined {
            entity: JoinEntity::Metric,
            key: key.to_string(),
        }),
        "parameter" => Ok(OrderTarget::Joined {
            entity: JoinEntity::Param,
            key: key.to_string(),
        }),
        "tag" => Ok(OrderTarget::Joined {
            entity: JoinEntity::Tag,
            key: key.to_string(),
        }),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid identifier type '{other}'"
        ))),
    }
}

/// `SqlRun.get_attribute_name` + column set. Returns the physical `runs` column.
fn attr_column(key: &str) -> Result<&'static str, MlflowError> {
    Ok(match key {
        "run_name" => "name",
        "run_id" => "run_uuid",
        "status" => "status",
        "user_id" => "user_id",
        "start_time" => "start_time",
        "end_time" => "end_time",
        "artifact_uri" => "artifact_uri",
        other => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid attribute key '{other}' specified."
            )))
        }
    })
}

fn is_numeric_attr(key: &str) -> bool {
    matches!(key, "start_time" | "end_time")
}

/// `SearchUtils.translate_key_alias`.
fn translate_key_alias(key: &str) -> String {
    match key {
        "created" | "Created" => "start_time".to_string(),
        "run name" | "Run name" | "Run Name" => "run_name".to_string(),
        other => other.to_string(),
    }
}

// ===========================================================================
// SQL generation
// ===========================================================================

/// A generated statement plus its positional binds.
#[derive(Debug)]
struct Query {
    sql: String,
    binds: Vec<Val>,
}

/// Build the full `search_runs` `SELECT`: run columns + order/keyset columns,
/// order-by joins, EXISTS filters, workspace/lifecycle/experiment predicates,
/// keyset predicate, `ORDER BY`, and `LIMIT max+1`.
fn build_search_sql(
    dialect: Dialect,
    exp_ids: &[i64],
    stages: &[&str],
    filters: &[Comparison],
    order_cols: &[SortCol],
    cursor: Option<&Cursor>,
    max_results: Option<i64>,
) -> Result<Query, MlflowError> {
    // Drain the joins collected by `build_order_cols` for this call.
    let joins: Vec<JoinSpec> = JOIN_REGISTRY.with(|reg| std::mem::take(&mut *reg.borrow_mut()));

    // Fix up the raw start_time DESC NULLS placement per dialect.
    let order_cols = apply_dialect_nulls(dialect, order_cols);

    let mut binds: Vec<Val> = Vec::new();
    let mut ph = PlaceholderGen::new(dialect);

    // SELECT list: run columns aliased under `r`, then each order-col value expr.
    let mut select = format!("SELECT {}", RunRow::select_cols_prefixed("r"));
    for (i, c) in order_cols.iter().enumerate() {
        select.push_str(&format!(", ({}) AS k{i}", c.expr));
    }

    let mut sql = format!("{select} FROM runs r");

    // Order-by joins (LEFT JOIN a per-key subquery), matching Python's outerjoin.
    for j in &joins {
        let key_ph = ph.next();
        binds.push(Val::Text(j.key.clone()));
        let cols = if j.is_metric {
            "run_uuid, value, is_nan"
        } else {
            "run_uuid, value"
        };
        sql.push_str(&format!(
            " LEFT JOIN (SELECT {cols} FROM {tbl} WHERE key = {key_ph}) {alias} \
             ON {alias}.run_uuid = r.run_uuid",
            tbl = j.table,
            alias = j.alias,
        ));
    }

    // WHERE.
    let mut wheres: Vec<String> = Vec::new();

    // experiment_id IN (...)
    let exp_phs: Vec<String> = exp_ids
        .iter()
        .map(|id| {
            binds.push(Val::Int(*id));
            ph.next()
        })
        .collect();
    wheres.push(format!("r.experiment_id IN ({})", exp_phs.join(", ")));

    // lifecycle_stage IN (...)
    let stage_phs: Vec<String> = stages
        .iter()
        .map(|s| {
            binds.push(Val::Text((*s).to_string()));
            ph.next()
        })
        .collect();
    wheres.push(format!("r.lifecycle_stage IN ({})", stage_phs.join(", ")));

    // Filter clauses -> EXISTS semi-joins / direct attribute predicates.
    for f in filters {
        wheres.push(build_filter_predicate(dialect, f, &mut ph, &mut binds)?);
    }

    // Keyset predicate.
    if let Some(cur) = cursor {
        wheres.push(build_keyset_predicate(
            &order_cols,
            cur,
            &mut ph,
            &mut binds,
        ));
    }

    sql.push_str(" WHERE ");
    sql.push_str(&wheres.join(" AND "));

    // ORDER BY.
    sql.push_str(" ORDER BY ");
    let order_terms: Vec<String> = order_cols
        .iter()
        .enumerate()
        .map(|(i, c)| order_term(dialect, i, c))
        .collect();
    sql.push_str(&order_terms.join(", "));

    // LIMIT max+1 for over-fetch (keyset paging).
    if let Some(mr) = max_results {
        sql.push_str(&format!(" LIMIT {}", mr + 1));
    }

    Ok(Query { sql, binds })
}

/// Overwrite the raw `start_time DESC` tiebreak's NULLS placement to match the
/// dialect's *default* null ordering, since Python emits `SqlRun.start_time.desc()`
/// with no explicit `NULLS` clause and relies on that default.
///
/// The three backends split on how they rank NULLs:
///
/// * **SQLite / MySQL** — NULL sorts as the *smallest* value, so `DESC` (largest
///   first) places NULLs **LAST**. (Verified: `ORDER BY x DESC` yields
///   `3,2,1,NULL` on SQLite; MySQL matches.)
/// * **Postgres** — NULL sorts as the *largest* value, so `DESC` places NULLs
///   **FIRST**.
///
/// The [`order_term`] renderer then emits an explicit `NULLS FIRST/LAST` on
/// SQLite/Postgres (reproducing the default deterministically) and a bare `DESC`
/// on MySQL (whose default already matches), and the keyset predicate reads the
/// same `Nulls` flag — so ORDER BY and page boundaries never drift. Value/rank
/// columns are untouched (rank cols separate NULLs before the value is compared).
fn apply_dialect_nulls(dialect: Dialect, cols: &[SortCol]) -> Vec<SortCol> {
    cols.iter()
        .map(|c| {
            let mut c = c.clone();
            if c.expr == "r.start_time" && c.kind == ColKind::Num && c.dir == Dir::Desc {
                c.nulls = match dialect {
                    Dialect::Sqlite | Dialect::MySql => Nulls::Last,
                    Dialect::Postgres => Nulls::First,
                };
            }
            c
        })
        .collect()
}

/// Render one `ORDER BY` term. For columns with an explicit NULLS placement we
/// emit `NULLS FIRST/LAST` on SQLite/Postgres; on MySQL (no NULLS syntax) the
/// placement matches the default, so a bare `dir` suffices.
fn order_term(dialect: Dialect, idx: usize, c: &SortCol) -> String {
    let dir = match c.dir {
        Dir::Asc => "ASC",
        Dir::Desc => "DESC",
    };
    // Reference the SELECT alias `kN` so the (possibly complex) expr is written
    // once; all supported dialects allow ORDER BY on a select alias.
    let base = format!("k{idx} {dir}");
    match (dialect, c.nulls) {
        (Dialect::MySql, _) | (_, Nulls::NonNull) => base,
        (Dialect::Sqlite | Dialect::Postgres, Nulls::First) => format!("k{idx} {dir} NULLS FIRST"),
        (Dialect::Sqlite | Dialect::Postgres, Nulls::Last) => format!("k{idx} {dir} NULLS LAST"),
    }
}

/// Render a Rust comparator set the way Python renders `str(set)` — a `{...}`
/// blob — but with the elements **sorted**. Python's set-iteration order is
/// hash-randomized per interpreter run, so its raw repr is non-deterministic;
/// the corpus generator normalizes those blobs to sorted form (see
/// `gen_search_corpus.py::_normalize`), and we match that here so error messages
/// are reproducible and comparable. The message text otherwise reproduces
/// Python's `_get_sqlalchemy_filter_clauses` validators verbatim — including
/// Python's occasional missing closing quote (`is_metric`/`is_tag`/dataset/
/// numeric-attribute).
fn py_set_sorted(items: &[&str]) -> String {
    let mut sorted: Vec<&str> = items.to_vec();
    sorted.sort_unstable();
    let inner: Vec<String> = sorted.iter().map(|s| format!("'{s}'")).collect();
    format!("{{{}}}", inner.join(", "))
}

/// Validate a comparator against the entity's valid-comparator set, raising the
/// same `INVALID_PARAMETER_VALUE` message Python's `is_metric`/`is_param`/
/// `is_tag`/`is_string_attribute`/`is_numeric_attribute`/`is_dataset` raise.
///
/// Python validates these at filter-application time (not parse time), so the
/// `mlflow-search` parser lets e.g. `params.p > 'x'` through and this is where
/// it is rejected. `msg_set` is the set named in the message (numeric-attribute
/// deliberately references the STRING set, mirroring the Python typo).
fn validate_comparator(
    comparator: &str,
    valid: &[&str],
    msg_set: &[&str],
    trailing_quote: bool,
) -> Result<(), MlflowError> {
    if valid.contains(&comparator) {
        return Ok(());
    }
    let close = if trailing_quote { "'" } else { "" };
    Err(MlflowError::invalid_parameter_value(format!(
        "Invalid comparator '{comparator}' not one of '{}{close}",
        py_set_sorted(msg_set)
    )))
}

/// Build one filter comparison as an SQL predicate (Q2/Q3: EXISTS semi-joins or
/// direct attribute predicates), mirroring `_get_sqlalchemy_filter_clauses`.
///
/// Comparator validation ordering mirrors Python: the attribute branch is
/// selected by `is_string_attribute`/`is_numeric_attribute` (which validate the
/// STRING/NUMERIC comparator sets); the metric/param/tag/dataset branches by the
/// matching `is_*` validators. Each validator raises before any SQL is emitted.
fn build_filter_predicate(
    dialect: Dialect,
    f: &Comparison,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    let key = translate_key_alias(&f.key);
    let comparator = f.comparator.to_uppercase();

    match f.entity_type.as_str() {
        "attribute" => {
            // `is_numeric_attribute` (start_time/end_time) validates the NUMERIC
            // set; `is_string_attribute` validates the STRING set. Python's
            // numeric-attribute message references the STRING set (a source typo)
            // and omits the closing quote.
            if is_numeric_attr(&key) {
                validate_comparator(
                    &comparator,
                    VALID_METRIC_COMPARATORS,
                    VALID_STRING_ATTRIBUTE_COMPARATORS,
                    false,
                )?;
            } else {
                validate_comparator(
                    &comparator,
                    VALID_STRING_ATTRIBUTE_COMPARATORS,
                    VALID_STRING_ATTRIBUTE_COMPARATORS,
                    true,
                )?;
            }
            if key == "run_name" {
                // attributes.run_name -> tags.`mlflow.runName` EXISTS.
                let mut inner = format!(
                    "SELECT 1 FROM tags t WHERE t.run_uuid = r.run_uuid AND t.key = {}",
                    push_text(ph, binds, MLFLOW_RUN_NAME)
                );
                inner.push_str(" AND ");
                inner.push_str(&value_predicate(
                    dialect,
                    "t.value",
                    &comparator,
                    &f.value,
                    ph,
                    binds,
                )?);
                Ok(format!("EXISTS ({inner})"))
            } else {
                let col = attr_column(&key)?;
                let target = format!("r.{col}");
                if is_numeric_attr(&key) {
                    // Numeric attributes (start_time/end_time) bind numerically so
                    // the comparison works across dialects (Postgres rejects a
                    // text bind against a bigint column; SQLite/MySQL coerce). The
                    // valid comparator set here is `=,!=,<,<=,>,>=` — no LIKE/IN —
                    // so a scalar numeric predicate covers every case.
                    let num = numeric_attr_value(&f.value)?;
                    Ok(value_predicate_num(
                        dialect,
                        &target,
                        &comparator,
                        num,
                        ph,
                        binds,
                    ))
                } else {
                    value_predicate(dialect, &target, &comparator, &f.value, ph, binds)
                }
            }
        }
        "metric" | "parameter" | "tag" => {
            let (table, valid, trailing_quote): (&str, &[&str], bool) = match f.entity_type.as_str()
            {
                // Python's metric/tag messages omit the closing quote; param's keeps it.
                "metric" => ("latest_metrics", VALID_METRIC_COMPARATORS, false),
                "parameter" => ("params", VALID_PARAM_COMPARATORS, true),
                _ => ("tags", VALID_TAG_COMPARATORS, false),
            };
            validate_comparator(&comparator, valid, valid, trailing_quote)?;
            if comparator == "IS NULL" || comparator == "IS NOT NULL" {
                let inner = format!(
                    "SELECT 1 FROM {table} e WHERE e.run_uuid = r.run_uuid AND e.key = {}",
                    push_text(ph, binds, &key)
                );
                return Ok(if comparator == "IS NULL" {
                    format!("NOT EXISTS ({inner})")
                } else {
                    format!("EXISTS ({inner})")
                });
            }
            let key_ph = push_text(ph, binds, &key);
            let val_pred = if f.entity_type == "metric" {
                // metric values compare numerically (Python `value = float(value)`).
                let num = metric_filter_value(&f.value)?;
                value_predicate_num(dialect, "e.value", &comparator, num, ph, binds)
            } else {
                value_predicate(dialect, "e.value", &comparator, &f.value, ph, binds)?
            };
            Ok(format!(
                "EXISTS (SELECT 1 FROM {table} e WHERE e.run_uuid = r.run_uuid \
                 AND e.key = {key_ph} AND {val_pred})"
            ))
        }
        "dataset" => {
            // Python's dataset message omits the closing quote.
            validate_comparator(
                &comparator,
                VALID_DATASET_COMPARATORS,
                VALID_DATASET_COMPARATORS,
                false,
            )?;
            build_dataset_predicate(dialect, &key, &comparator, &f.value, ph, binds)
        }
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid search expression type '{other}'"
        ))),
    }
}

/// Dataset filter (`name`/`digest`/`context`) as an EXISTS over inputs+datasets.
fn build_dataset_predicate(
    dialect: Dialect,
    key: &str,
    comparator: &str,
    value: &Value,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    if key == "context" {
        let pred = value_predicate(dialect, "it.value", comparator, value, ph, binds)?;
        Ok(format!(
            "EXISTS (SELECT 1 FROM inputs i \
             JOIN datasets d ON i.source_id = d.dataset_uuid \
             JOIN input_tags it ON it.input_uuid = i.input_uuid \
             WHERE i.destination_id = r.run_uuid AND i.destination_type = 'RUN' \
             AND it.name = {} AND {pred})",
            push_text(ph, binds, MLFLOW_DATASET_CONTEXT),
        ))
    } else {
        let col = match key {
            "name" => "d.name",
            "digest" => "d.digest",
            other => {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid dataset key '{other}'."
                )))
            }
        };
        let pred = value_predicate(dialect, col, comparator, value, ph, binds)?;
        Ok(format!(
            "EXISTS (SELECT 1 FROM inputs i \
             JOIN datasets d ON i.source_id = d.dataset_uuid \
             WHERE i.destination_id = r.run_uuid AND i.destination_type = 'RUN' AND {pred})"
        ))
    }
}

/// Render a comparison predicate on a string column for `=,!=,<,<=,>,>=,LIKE,
/// ILIKE,IN,NOT IN`, matching `get_sql_comparison_func`.
fn value_predicate(
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value: &Value,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> Result<String, MlflowError> {
    match comparator {
        "LIKE" => {
            let idx = ph.reserve_like(dialect);
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
                // `col IN ()` is invalid SQL and always false; `NOT IN ()` true.
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

/// Numeric comparison predicate (metric filter values).
fn value_predicate_num(
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value: f64,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> String {
    let _ = dialect;
    let p = ph.next();
    binds.push(Val::Float(value));
    format!("{column} {comparator} {p}")
}

/// Resolve a numeric-attribute filter value (`start_time`/`end_time`) to `f64`.
/// The parser hands these to us as a `Value::Str` holding the raw numeric token
/// (having already rejected non-numeric tokens), so a parse here is defensive.
fn numeric_attr_value(value: &Value) -> Result<f64, MlflowError> {
    match value {
        Value::Str(s) => s.parse::<f64>().map_err(|_| {
            MlflowError::invalid_parameter_value(format!(
                "could not convert string to float: '{s}'"
            ))
        }),
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a numeric value for a numeric attribute".to_string(),
        )),
    }
}

fn metric_filter_value(value: &Value) -> Result<f64, MlflowError> {
    match value {
        Value::Str(s) => s.parse::<f64>().map_err(|_| {
            MlflowError::invalid_parameter_value(format!(
                "could not convert string to float: '{s}'"
            ))
        }),
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a numeric value for metric filter".to_string(),
        )),
    }
}

fn as_str(value: &Value) -> Result<String, MlflowError> {
    match value {
        Value::Str(s) => Ok(s.clone()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(f.to_string()),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a string value".to_string(),
        )),
    }
}

fn as_list(value: &Value) -> Result<Vec<String>, MlflowError> {
    match value {
        Value::List(items) => Ok(items.clone()),
        _ => Err(MlflowError::invalid_parameter_value(
            "Expected a list value for IN/NOT IN".to_string(),
        )),
    }
}

fn push_text(ph: &mut PlaceholderGen, binds: &mut Vec<Val>, s: &str) -> String {
    binds.push(Val::Text(s.to_string()));
    ph.next()
}

/// Positional placeholder generator (`$N` on Postgres, `?` elsewhere).
struct PlaceholderGen {
    dialect: Dialect,
    idx: usize,
}

impl PlaceholderGen {
    fn new(dialect: Dialect) -> Self {
        Self { dialect, idx: 0 }
    }
    /// Consume one placeholder slot, return its rendered string.
    fn next(&mut self) -> String {
        self.idx += 1;
        self.dialect.placeholder(self.idx)
    }
    /// Consume one slot, return its 1-based index (for the LIKE helpers, which
    /// render the placeholder themselves).
    fn next_index(&mut self) -> usize {
        self.idx += 1;
        self.idx
    }
    /// Reserve slot(s) for a case-sensitive LIKE (2 placeholders on MySQL, 1
    /// elsewhere) and return the first index.
    fn reserve_like(&mut self, dialect: Dialect) -> usize {
        let first = self.next_index();
        if let Dialect::MySql = dialect {
            self.idx += 1; // second placeholder consumed by the BINARY clause
        }
        first
    }
}

// ===========================================================================
// Keyset cursor
// ===========================================================================

/// One cursor cell = the value of an order column for the boundary row.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Null,
    Int(i64),
    Num(f64),
    Text(String),
}

/// The decoded keyset cursor: one [`Cell`] per order column.
#[derive(Debug, Clone)]
struct Cursor {
    cells: Vec<Cell>,
}

impl Cursor {
    /// Encode the boundary row's key tuple into an opaque base64(JSON) token.
    fn encode(keys: &[Cell]) -> String {
        let arr: Vec<serde_json::Value> = keys
            .iter()
            .map(|c| match c {
                Cell::Null => serde_json::Value::Null,
                Cell::Int(i) => serde_json::json!({"i": i}),
                Cell::Num(f) => {
                    if f.is_finite() {
                        serde_json::json!({ "f": f })
                    } else if f.is_nan() {
                        serde_json::json!({"f": "NaN"})
                    } else if *f > 0.0 {
                        serde_json::json!({"f": "Infinity"})
                    } else {
                        serde_json::json!({"f": "-Infinity"})
                    }
                }
                Cell::Text(s) => serde_json::json!({ "s": s }),
            })
            .collect();
        let payload = serde_json::json!({"v": 2, "k": arr});
        base64_encode(serde_json::to_string(&payload).unwrap().as_bytes())
    }

    /// Decode + validate an opaque token. Any malformation raises the Python
    /// "Invalid page token" family of errors.
    fn decode(token: &str, expected_len: usize) -> Result<Self, MlflowError> {
        let bad = || MlflowError::invalid_parameter_value("Invalid page token".to_string());
        let bytes = base64_decode(token).ok_or_else(bad)?;
        let text = String::from_utf8(bytes).map_err(|_| bad())?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|_| bad())?;
        let arr = json.get("k").and_then(|v| v.as_array()).ok_or_else(bad)?;
        if arr.len() != expected_len {
            return Err(bad());
        }
        let mut cells = Vec::with_capacity(arr.len());
        for v in arr {
            cells.push(match v {
                serde_json::Value::Null => Cell::Null,
                serde_json::Value::Object(o) => {
                    if let Some(i) = o.get("i").and_then(|x| x.as_i64()) {
                        Cell::Int(i)
                    } else if let Some(f) = o.get("f") {
                        Cell::Num(parse_num_cell(f).ok_or_else(bad)?)
                    } else if let Some(s) = o.get("s").and_then(|x| x.as_str()) {
                        Cell::Text(s.to_string())
                    } else {
                        return Err(bad());
                    }
                }
                _ => return Err(bad()),
            });
        }
        Ok(Cursor { cells })
    }
}

fn parse_num_cell(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

/// Build the keyset predicate "row is strictly after the cursor" as an expanded
/// lexicographic OR-chain over the ordered columns. Row-value comparison can't
/// mix ASC/DESC or emulate NULLS placement, so we expand it explicitly.
fn build_keyset_predicate(
    cols: &[SortCol],
    cursor: &Cursor,
    ph: &mut PlaceholderGen,
    binds: &mut Vec<Val>,
) -> String {
    let n = cols.len();
    let mut or_terms: Vec<String> = Vec::new();
    for j in 0..n {
        let mut and_terms: Vec<String> = Vec::new();
        for (col, cell) in cols.iter().take(j).zip(cursor.cells.iter()) {
            and_terms.push(eq_term(&col.expr, cell, ph, binds));
        }
        and_terms.push(after_term(&cols[j], &cursor.cells[j], ph, binds));
        or_terms.push(format!("({})", and_terms.join(" AND ")));
    }
    format!("({})", or_terms.join(" OR "))
}

/// Equality within the keyset prefix, NULL-aware. A NULL cursor cell matches
/// only NULL rows; a non-null cell matches only equal non-null rows.
fn eq_term(expr: &str, cell: &Cell, ph: &mut PlaceholderGen, binds: &mut Vec<Val>) -> String {
    match cell {
        Cell::Null => format!("{expr} IS NULL"),
        _ => {
            let p = bind_cell(cell, ph, binds);
            format!("{expr} = {p}")
        }
    }
}

/// "Row strictly after cursor at this column", honoring direction + NULLS.
fn after_term(col: &SortCol, cell: &Cell, ph: &mut PlaceholderGen, binds: &mut Vec<Val>) -> String {
    let expr = &col.expr;
    match col.nulls {
        Nulls::NonNull => {
            // Rank cols and run_uuid are never null.
            let p = bind_cell(cell, ph, binds);
            match col.dir {
                Dir::Asc => format!("{expr} > {p}"),
                Dir::Desc => format!("{expr} < {p}"),
            }
        }
        Nulls::Last => match cell {
            // Cursor is NULL and NULLs are last => nothing sorts after it.
            Cell::Null => "1 = 0".to_string(),
            _ => {
                let p = bind_cell(cell, ph, binds);
                match col.dir {
                    // non-null rows greater, plus NULL rows (they come after).
                    Dir::Asc => format!("({expr} > {p} OR {expr} IS NULL)"),
                    Dir::Desc => format!("({expr} < {p} OR {expr} IS NULL)"),
                }
            }
        },
        Nulls::First => match cell {
            // Cursor is NULL and NULLs are first => any non-null row is after.
            Cell::Null => format!("{expr} IS NOT NULL"),
            _ => {
                let p = bind_cell(cell, ph, binds);
                match col.dir {
                    // NULLs already passed; only non-null rows compare.
                    Dir::Asc => format!("{expr} > {p}"),
                    Dir::Desc => format!("{expr} < {p}"),
                }
            }
        },
    }
}

fn bind_cell(cell: &Cell, ph: &mut PlaceholderGen, binds: &mut Vec<Val>) -> String {
    match cell {
        Cell::Null => unreachable!("bind_cell called on NULL cell"),
        Cell::Int(i) => binds.push(Val::Int(*i)),
        Cell::Num(f) => binds.push(Val::Float(*f)),
        Cell::Text(s) => binds.push(Val::Text(s.clone())),
    }
    ph.next()
}

// ===========================================================================
// Row decoding
// ===========================================================================

/// A search result row: the run columns + the typed key cells from the SELECT.
struct SearchRow {
    run: RunRow,
    keys: Vec<Cell>,
}

impl SearchRow {
    /// Decode the run columns plus each `kN` key column by its declared
    /// [`ColKind`] so cursor cells round-trip exactly (a numeric-looking param
    /// value must stay `Text`, not become `Num`).
    fn from_row(r: &dyn RowLike, kinds: &[ColKind]) -> Result<Self, sqlx::Error> {
        let run = RunRow::from_row(r)?;
        let mut keys = Vec::with_capacity(kinds.len());
        for (i, kind) in kinds.iter().enumerate() {
            let col = format!("k{i}");
            keys.push(read_cell(r, &col, *kind)?);
        }
        Ok(SearchRow { run, keys })
    }
}

/// Read a `kN` column into a [`Cell`] using its declared kind.
fn read_cell(r: &dyn RowLike, col: &str, kind: ColKind) -> Result<Cell, sqlx::Error> {
    Ok(match kind {
        // Rank columns are the CASE output: always a non-null small integer.
        ColKind::Rank => Cell::Int(r.get_i64(col)?),
        ColKind::Num => match r.get_opt_i64(col) {
            Ok(Some(i)) => Cell::Num(i as f64),
            _ => match r.get_f64(col) {
                Ok(f) => Cell::Num(f),
                Err(_) => Cell::Null,
            },
        },
        ColKind::Text => match r.get_opt_string(col)? {
            Some(s) => Cell::Text(s),
            None => Cell::Null,
        },
    })
}

// ===========================================================================
// Batched eager loaders (Q8)
// ===========================================================================

impl TrackingStore {
    async fn load_params_bulk(
        &self,
        run_ids: &[String],
    ) -> Result<HashMap<String, Vec<Param>>, MlflowError> {
        let (sql, binds) = in_query(
            self.db().dialect(),
            "SELECT run_uuid, key, value FROM params WHERE run_uuid IN",
            " ORDER BY run_uuid, key",
            run_ids,
        );
        let rows = self
            .db()
            .fetch_all(&sql, &binds, |r| {
                Ok((
                    r.get_string("run_uuid")?,
                    Param {
                        key: r.get_string("key")?,
                        value: r.get_string("value")?,
                    },
                ))
            })
            .await
            .map_err(internal)?;
        Ok(group_by(rows))
    }

    async fn load_tags_bulk(
        &self,
        run_ids: &[String],
    ) -> Result<HashMap<String, Vec<RunTag>>, MlflowError> {
        let (sql, binds) = in_query(
            self.db().dialect(),
            "SELECT run_uuid, key, value FROM tags WHERE run_uuid IN",
            " ORDER BY run_uuid, key",
            run_ids,
        );
        let rows = self
            .db()
            .fetch_all(&sql, &binds, |r| {
                Ok((
                    r.get_string("run_uuid")?,
                    RunTag {
                        key: r.get_string("key")?,
                        value: r.get_opt_string("value")?.unwrap_or_default(),
                    },
                ))
            })
            .await
            .map_err(internal)?;
        Ok(group_by(rows))
    }

    async fn load_latest_metrics_bulk(
        &self,
        run_ids: &[String],
    ) -> Result<HashMap<String, Vec<Metric>>, MlflowError> {
        let (sql, binds) = in_query(
            self.db().dialect(),
            "SELECT run_uuid, key, value, timestamp, step, is_nan FROM latest_metrics \
             WHERE run_uuid IN",
            " ORDER BY run_uuid, key",
            run_ids,
        );
        let rows = self
            .db()
            .fetch_all(&sql, &binds, |r| {
                let is_nan = r.get_bool("is_nan")?;
                let stored = r.get_f64("value")?;
                Ok((
                    r.get_string("run_uuid")?,
                    Metric {
                        key: r.get_string("key")?,
                        value: if is_nan { f64::NAN } else { stored },
                        timestamp: r.get_opt_i64("timestamp")?.unwrap_or(0),
                        step: r.get_i64("step")?,
                    },
                ))
            })
            .await
            .map_err(internal)?;
        Ok(group_by(rows))
    }
}

/// Group `(run_id, item)` rows into a per-run vector, preserving row order.
fn group_by<T>(rows: Vec<(String, T)>) -> HashMap<String, Vec<T>> {
    let mut map: HashMap<String, Vec<T>> = HashMap::new();
    for (id, item) in rows {
        map.entry(id).or_default().push(item);
    }
    map
}

/// Build a `... IN (ph, ph, ...) <suffix>` query with positional binds.
fn in_query(
    dialect: Dialect,
    prefix: &str,
    suffix: &str,
    run_ids: &[String],
) -> (String, Vec<Val>) {
    let mut binds = Vec::with_capacity(run_ids.len());
    let phs: Vec<String> = run_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            binds.push(Val::Text(id.clone()));
            dialect.placeholder(i + 1)
        })
        .collect();
    (format!("{prefix} ({}){suffix}", phs.join(", ")), binds)
}

impl RunRow {
    /// The run SELECT columns, each qualified by `alias`.
    pub(crate) fn select_cols_prefixed(alias: &str) -> String {
        [
            "run_uuid",
            "name",
            "user_id",
            "status",
            "start_time",
            "end_time",
            "lifecycle_stage",
            "artifact_uri",
            "experiment_id",
        ]
        .iter()
        .map(|c| format!("{alias}.{c}"))
        .collect::<Vec<_>>()
        .join(", ")
    }
}

// ===========================================================================
// base64 (standard alphabet, tolerant decode) — same as metrics.rs, kept local
// ===========================================================================

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

// ===========================================================================
// Unit tests (dialect-independent string/logic parity — no DB required).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Drain the thread-local join registry so a prior test's leftover joins
    /// never leak into the next `build_order_cols` call.
    fn drain_joins() {
        JOIN_REGISTRY.with(|r| r.borrow_mut().clear());
    }

    #[test]
    fn max_results_validation_matches_python() {
        assert!(validate_max_results(None).is_ok());
        assert!(validate_max_results(Some(1)).is_ok());
        assert!(validate_max_results(Some(SEARCH_MAX_RESULTS_THRESHOLD)).is_ok());
        let e = validate_max_results(Some(0)).unwrap_err();
        assert!(
            e.message.contains("must be a positive integer"),
            "{}",
            e.message
        );
        let e = validate_max_results(Some(SEARCH_MAX_RESULTS_THRESHOLD + 1)).unwrap_err();
        assert!(e.message.contains("at most 50000"), "{}", e.message);
    }

    #[test]
    fn view_type_stages_map() {
        assert_eq!(view_type_stages(ViewType::ActiveOnly), &["active"]);
        assert_eq!(view_type_stages(ViewType::DeletedOnly), &["deleted"]);
        assert_eq!(view_type_stages(ViewType::All), &["active", "deleted"]);
    }

    #[test]
    fn default_order_appends_start_time_then_run_uuid() {
        drain_joins();
        let cols = build_order_cols(&[]).unwrap();
        // Just the two tiebreaks: start_time DESC, run_uuid ASC.
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].expr, "r.start_time");
        assert_eq!(cols[0].dir, Dir::Desc);
        assert_eq!(cols[1].expr, "r.run_uuid");
        assert_eq!(cols[1].dir, Dir::Asc);
        drain_joins();
    }

    #[test]
    fn start_time_order_by_suppresses_extra_tiebreak() {
        drain_joins();
        let cols = build_order_cols(&["attribute.start_time ASC".to_string()]).unwrap();
        // rank CASE + value + run_uuid  (no second start_time appended).
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].kind, ColKind::Rank);
        assert_eq!(cols[1].expr, "r.start_time");
        assert_eq!(cols[1].dir, Dir::Asc);
        assert_eq!(cols[2].expr, "r.run_uuid");
        drain_joins();
    }

    #[test]
    fn metric_order_by_emits_three_way_rank() {
        drain_joins();
        let cols = build_order_cols(&["metrics.acc DESC".to_string()]).unwrap();
        // rank CASE (is_nan=>1, null=>2) + value DESC + start_time DESC + run_uuid.
        assert_eq!(cols.len(), 4);
        assert_eq!(cols[0].kind, ColKind::Rank);
        assert!(cols[0].expr.contains("is_nan = 1"));
        assert!(cols[0].expr.contains("IS NULL THEN 2"));
        assert_eq!(cols[1].kind, ColKind::Num);
        assert_eq!(cols[1].dir, Dir::Desc);
        // One order-by join was registered.
        let n = JOIN_REGISTRY.with(|r| r.borrow().len());
        assert_eq!(n, 1);
        drain_joins();
    }

    #[test]
    fn duplicate_order_by_fields_rejected() {
        drain_joins();
        let e = build_order_cols(&["metrics.acc".to_string(), "metrics.acc DESC".to_string()])
            .unwrap_err();
        assert!(e.message.contains("duplicate fields"), "{}", e.message);
        drain_joins();
    }

    #[test]
    fn order_by_created_alias_is_start_time() {
        drain_joins();
        // `created` translates to start_time, so the extra start_time tiebreak
        // is suppressed (same key as the user clause).
        let cols = build_order_cols(&["created ASC".to_string()]).unwrap();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[1].expr, "r.start_time");
        drain_joins();
    }

    #[test]
    fn dialect_null_placement_for_start_time_desc() {
        let cols = vec![SortCol {
            expr: "r.start_time".to_string(),
            dir: Dir::Desc,
            nulls: Nulls::First,
            kind: ColKind::Num,
        }];
        // SQLite/MySQL DESC => NULLs last; Postgres DESC => NULLs first.
        assert_eq!(
            apply_dialect_nulls(Dialect::Sqlite, &cols)[0].nulls,
            Nulls::Last
        );
        assert_eq!(
            apply_dialect_nulls(Dialect::MySql, &cols)[0].nulls,
            Nulls::Last
        );
        assert_eq!(
            apply_dialect_nulls(Dialect::Postgres, &cols)[0].nulls,
            Nulls::First
        );
    }

    #[test]
    fn order_term_rendering_per_dialect() {
        let last = SortCol {
            expr: "r.start_time".to_string(),
            dir: Dir::Desc,
            nulls: Nulls::Last,
            kind: ColKind::Num,
        };
        assert_eq!(order_term(Dialect::Sqlite, 3, &last), "k3 DESC NULLS LAST");
        assert_eq!(
            order_term(Dialect::Postgres, 3, &last),
            "k3 DESC NULLS LAST"
        );
        // MySQL has no NULLS syntax; its default already matches.
        assert_eq!(order_term(Dialect::MySql, 3, &last), "k3 DESC");
        let nonnull = SortCol {
            expr: "r.run_uuid".to_string(),
            dir: Dir::Asc,
            nulls: Nulls::NonNull,
            kind: ColKind::Text,
        };
        assert_eq!(order_term(Dialect::Postgres, 5, &nonnull), "k5 ASC");
    }

    #[test]
    fn comparator_validation_messages() {
        // param `>` is invalid (params only support =,!=,LIKE,ILIKE,IS [NOT] NULL).
        let f = Comparison {
            entity_type: "parameter".to_string(),
            key: "p".to_string(),
            comparator: ">".to_string(),
            value: Value::Str("x".to_string()),
        };
        let mut ph = PlaceholderGen::new(Dialect::Sqlite);
        let mut binds = Vec::new();
        let e = build_filter_predicate(Dialect::Sqlite, &f, &mut ph, &mut binds).unwrap_err();
        assert!(
            e.message
                .starts_with("Invalid comparator '>' not one of '{"),
            "{}",
            e.message
        );
        // param message keeps the closing quote.
        assert!(e.message.ends_with("}'"), "{}", e.message);

        // metric LIKE is invalid and the metric message OMITS the closing quote.
        let f = Comparison {
            entity_type: "metric".to_string(),
            key: "m".to_string(),
            comparator: "LIKE".to_string(),
            value: Value::Str("1".to_string()),
        };
        let mut ph = PlaceholderGen::new(Dialect::Sqlite);
        let mut binds = Vec::new();
        let e = build_filter_predicate(Dialect::Sqlite, &f, &mut ph, &mut binds).unwrap_err();
        assert!(
            e.message.starts_with("Invalid comparator 'LIKE'"),
            "{}",
            e.message
        );
        assert!(e.message.ends_with('}'), "no trailing quote: {}", e.message);
    }

    #[test]
    fn py_set_sorted_is_sorted() {
        assert_eq!(py_set_sorted(&[">", "=", "<"]), "{'<', '=', '>'}");
    }

    #[test]
    fn cursor_round_trip_all_cell_kinds() {
        let keys = vec![
            Cell::Int(2),
            Cell::Num(0.94),
            Cell::Text("run-abc".to_string()),
            Cell::Null,
            Cell::Num(f64::NAN),
            Cell::Num(f64::INFINITY),
            Cell::Num(f64::NEG_INFINITY),
        ];
        let token = Cursor::encode(&keys);
        let decoded = Cursor::decode(&token, keys.len()).unwrap();
        assert_eq!(decoded.cells.len(), keys.len());
        assert_eq!(decoded.cells[0], Cell::Int(2));
        assert_eq!(decoded.cells[1], Cell::Num(0.94));
        assert_eq!(decoded.cells[2], Cell::Text("run-abc".to_string()));
        assert_eq!(decoded.cells[3], Cell::Null);
        match decoded.cells[4] {
            Cell::Num(f) => assert!(f.is_nan()),
            ref other => panic!("expected NaN, got {other:?}"),
        }
        assert_eq!(decoded.cells[5], Cell::Num(f64::INFINITY));
        assert_eq!(decoded.cells[6], Cell::Num(f64::NEG_INFINITY));
    }

    #[test]
    fn cursor_decode_rejects_wrong_length_and_garbage() {
        let token = Cursor::encode(&[Cell::Int(1)]);
        assert!(Cursor::decode(&token, 2).is_err());
        assert!(Cursor::decode("!!!not-base64!!!", 1).is_err());
        assert!(Cursor::decode("", 1).is_err());
    }

    #[test]
    fn attr_column_mapping_matches_python() {
        assert_eq!(attr_column("run_name").unwrap(), "name");
        assert_eq!(attr_column("run_id").unwrap(), "run_uuid");
        assert_eq!(attr_column("status").unwrap(), "status");
        assert_eq!(attr_column("start_time").unwrap(), "start_time");
        assert!(attr_column("experiment_id").is_err());
    }
}
