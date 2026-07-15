//! `search_registered_models` + `search_model_versions` on the
//! [`RegistryStore`] (plan T7.3), mirroring
//! `mlflow/store/model_registry/sqlalchemy_store.py`:
//!
//! * `search_registered_models` (`:525-587`),
//!   `_get_search_registered_model_filter_query` (`:589-663`),
//!   `_parse_search_registered_models_order_by` (`:834-868`).
//! * `search_model_versions` (`:1309-1378`),
//!   `_get_search_model_versions_filter_clauses` (`:665-774`),
//!   `_parse_search_model_versions_order_by` (`:1380-1425`).
//! * `_update_query_to_exclude_prompts` (`:776-819`) and `_is_querying_prompt`
//!   (`:821-832`) — the prompt anti-join, shared by both searches.
//!
//! ## Ported semantics (must match Python page-for-page)
//!
//! * **Filter → SQL.** Attribute clauses become direct predicates on the main
//!   table (`registered_models` / `model_versions`), with the same
//!   comparator-validation and the same `source_path→source` /
//!   `version_number→version` column aliases Python applies. The value of a
//!   `name = 'x'` filter is **not** auto-wrapped in `%x%`; the store passes the
//!   parsed literal straight through (the `%x%` wrapping mentioned in the proto
//!   doc lives in the client, not the store). `run_id IN (...)` becomes a
//!   plain `IN` list.
//! * **AND-of-tags via HAVING-count subquery.** Tag predicates are grouped per
//!   key; the per-key groups are OR-ed, grouped by `(workspace, name[, version])`
//!   and required (`HAVING COUNT(*) = <#distinct keys>`) to match every key —
//!   the exact strategy of `_get_search_..._filter_*`. The subquery is joined
//!   back to the main table (an inner join, i.e. a semi-join since the subquery
//!   yields one row per matching entity).
//! * **Prompt exclusion (anti-join).** By default (`_is_querying_prompt` false)
//!   a `LEFT JOIN` to the set of prompt-tagged entities plus an
//!   `IS NULL` filter excludes all rows tagged `mlflow.prompt.is_prompt='true'`.
//!   Python pops `IS_PROMPT_TAG_KEY` out of `tag_filters` inside
//!   `_update_query_to_exclude_prompts` **before** the HAVING-count subquery is
//!   built, so a prompt-tag predicate that does *not* trigger the
//!   `_is_querying_prompt` bypass (e.g. `LIKE`) is dropped from the tag count —
//!   replicated here. A prompt tag whose predicate is `= 'true'` or
//!   `!= 'false'` flips `_is_querying_prompt` and disables the anti-join
//!   entirely (so prompts are returned).
//! * **Order by + tiebreak.** RM default order is `name ASC`; the only extra key
//!   is a timestamp (`creation_timestamp`/`last_updated_timestamp` →
//!   `last_updated_time`), and `name ASC` is appended as a tiebreak unless the
//!   user ordered by `name`. MV default order is
//!   `last_updated_timestamp DESC, name ASC, version DESC`; tiebreaks `name ASC`
//!   then `version DESC` are appended unless already present.
//! * **Deleted MVs.** `search_model_versions` filters
//!   `current_stage != 'Deleted_Internal'`; `search_registered_models` has no
//!   such filter (registered models have no soft-delete stage).
//! * **Aliases.** `search_model_versions` returns entities **without** aliases
//!   populated (Python's `to_mlflow_entity` does not attach them on this path);
//!   `search_registered_models` returns each model with its aliases and latest
//!   versions, matching `get_registered_model`.
//! * **Pagination.** Offset-token contract (`base64(json {"offset": N})`), with
//!   an over-fetch of `max_results + 1` to detect the next page — Python's
//!   `_compute_next_token`. (Registry search keeps offset tokens, unlike
//!   tracking's keyset tokens — plan T7.3.)
//! * **max_results thresholds** (store-level, `model_registry/__init__.py`):
//!   RM default 100 / threshold 1000; MV default 10000 / threshold 200000. The
//!   store validates the threshold (and, for MV, `>= 1`) exactly as Python's
//!   store does; the *default* is a handler concern (T7.4) — the store methods
//!   take an explicit `max_results`.

use mlflow_error::MlflowError;
use mlflow_search::{parse_start_offset_from_page_token, Comparison, OrderBy, Value};
use mlflow_store::dialect::Dialect;

use super::registered_models::map_model_version_row;
use super::{internal, RegistryStore};
use crate::dbutil::{DbExt, Val};
use crate::entities::{ModelVersion, RegisteredModel};
use crate::schema::{MODEL_VERSIONS, MODEL_VERSION_TAGS, REGISTERED_MODELS, REGISTERED_MODEL_TAGS};
use crate::stages::STAGE_DELETED_INTERNAL;

/// `mlflow.prompt.is_prompt` (`mlflow/prompt/constants.py:4`).
const IS_PROMPT_TAG_KEY: &str = "mlflow.prompt.is_prompt";

/// Store-level max_results bounds (`model_registry/__init__.py:4-10`).
const SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD: i64 = 1000;
const SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD: i64 = 200_000;

const VALID_RM_ATTR_COMPARATORS: &[&str] = &["=", "!=", "LIKE", "ILIKE"];
const VALID_TAG_COMPARATORS: &[&str] = &["=", "!=", "LIKE", "ILIKE"];
const VALID_MV_STRING_COMPARATORS: &[&str] = &["=", "!=", "LIKE", "ILIKE", "IN"];
const VALID_MV_NUMERIC_COMPARATORS: &[&str] = &[">", ">=", "!=", "=", "<", "<="];

const MV_VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] =
    &["name", "version_number", "run_id", "source_path"];
const MV_NUMERIC_ATTRIBUTES: &[&str] = &[
    "version_number",
    "creation_timestamp",
    "last_updated_timestamp",
];

/// A page of registered models plus the optional next-page token.
#[derive(Debug)]
pub struct RegisteredModelsPage {
    pub registered_models: Vec<RegisteredModel>,
    pub next_page_token: Option<String>,
}

/// A page of model versions plus the optional next-page token.
#[derive(Debug)]
pub struct ModelVersionsPage {
    pub model_versions: Vec<ModelVersion>,
    pub next_page_token: Option<String>,
}

impl RegistryStore {
    /// `search_registered_models`. `max_results` is the resolved value (the
    /// default 100 is applied by the caller/handler); it is validated against
    /// the store threshold 1000. `filter`/`order_by`/`page_token` are raw.
    pub async fn search_registered_models(
        &self,
        workspace: &str,
        filter: Option<&str>,
        max_results: i64,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<RegisteredModelsPage, MlflowError> {
        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value for request parameter max_results. It must be at most \
                 {SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}"
            )));
        }

        let parsed = mlflow_search::parse::registered_models_filter(filter.unwrap_or(""))
            .map_err(search_err)?;
        let order_cols = build_rm_order_by(order_by)?;
        let offset = parse_start_offset_from_page_token(page_token).map_err(search_err)?;

        let dialect = self.db().dialect();
        let mut b = QueryBuilder::new(dialect);
        let ff = build_filter_from(&mut b, workspace, &parsed, &TableSpec::registered_models())?;
        let where_sql = ff.where_clause();
        let from = ff.from;

        let order_sql = order_cols
            .iter()
            .map(|c| {
                format!(
                    "rm.{} {}",
                    c.column,
                    if c.ascending { "ASC" } else { "DESC" }
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let limit_ph = b.bind(Val::Int(max_results + 1));
        let sql = if page_token.is_some() {
            let offset_ph = b.bind(Val::Int(offset));
            format!(
                "SELECT rm.workspace, rm.name, rm.creation_time, rm.last_updated_time, \
                 rm.description FROM {from}{where_sql} ORDER BY {order_sql} \
                 LIMIT {limit_ph} OFFSET {offset_ph}"
            )
        } else {
            format!(
                "SELECT rm.workspace, rm.name, rm.creation_time, rm.last_updated_time, \
                 rm.description FROM {from}{where_sql} ORDER BY {order_sql} LIMIT {limit_ph}"
            )
        };

        let names: Vec<(String, String)> = self
            .db()
            .fetch_all(&sql, b.binds(), |r| {
                Ok((r.get_string("workspace")?, r.get_string("name")?))
            })
            .await
            .map_err(internal)?;

        let next_page_token =
            compute_next_token(names.len() as i64, max_results + 1, offset, max_results);
        let kept = names.into_iter().take(max_results as usize);

        let mut out = Vec::new();
        for (ws, name) in kept {
            // Re-fetch each model's full entity (tags/aliases/latest versions),
            // mirroring `to_mlflow_entity(preloaded_latest_versions=...)`.
            out.push(self.get_registered_model(&ws, &name).await?);
        }
        Ok(RegisteredModelsPage {
            registered_models: out,
            next_page_token,
        })
    }

    /// `search_model_versions`. `max_results` is validated as a positive integer
    /// `<=` the store threshold 200000 (the default 10000 is a handler concern).
    /// Excludes `Deleted_Internal` versions. Returned entities carry **no**
    /// aliases (Python parity on this path).
    pub async fn search_model_versions(
        &self,
        workspace: &str,
        filter: Option<&str>,
        max_results: i64,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<ModelVersionsPage, MlflowError> {
        if max_results < 1 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value for max_results. It must be a positive integer, but got {max_results}"
            )));
        }
        if max_results > SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value for request parameter max_results. It must be at most \
                 {SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD}, but got value {max_results}"
            )));
        }

        let parsed = mlflow_search::parse::model_versions_filter(filter.unwrap_or(""))
            .map_err(search_err)?;
        let default_order = [
            "last_updated_timestamp DESC".to_string(),
            "name ASC".to_string(),
            "version_number DESC".to_string(),
        ];
        let order_source: &[String] = if order_by.is_empty() {
            &default_order
        } else {
            order_by
        };
        let order_cols = build_mv_order_by(order_source)?;
        let offset = parse_start_offset_from_page_token(page_token).map_err(search_err)?;

        let dialect = self.db().dialect();
        let mut b = QueryBuilder::new(dialect);
        let mut ff = build_filter_from(&mut b, workspace, &parsed, &TableSpec::model_versions())?;

        // `current_stage != 'Deleted_Internal'` (`sqlalchemy_store.py:1367`).
        let stage_ph = b.bind(Val::Text(STAGE_DELETED_INTERNAL.to_string()));
        ff.where_preds
            .push(format!("mv.current_stage <> {stage_ph}"));
        let where_sql = ff.where_clause();
        let from = ff.from;

        let order_sql = order_cols
            .iter()
            .map(|c| {
                format!(
                    "mv.{} {}",
                    c.column,
                    if c.ascending { "ASC" } else { "DESC" }
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let limit_ph = b.bind(Val::Int(max_results + 1));
        let sql = if page_token.is_some() {
            let offset_ph = b.bind(Val::Int(offset));
            format!(
                "SELECT mv.workspace, mv.name, mv.version, mv.creation_time, \
                 mv.last_updated_time, mv.description, mv.user_id, mv.current_stage, mv.source, \
                 mv.run_id, mv.status, mv.status_message, mv.run_link \
                 FROM {from}{where_sql} ORDER BY {order_sql} \
                 LIMIT {limit_ph} OFFSET {offset_ph}"
            )
        } else {
            format!(
                "SELECT mv.workspace, mv.name, mv.version, mv.creation_time, \
                 mv.last_updated_time, mv.description, mv.user_id, mv.current_stage, mv.source, \
                 mv.run_id, mv.status, mv.status_message, mv.run_link \
                 FROM {from}{where_sql} ORDER BY {order_sql} LIMIT {limit_ph}"
            )
        };

        let rows = self
            .db()
            .fetch_all(&sql, b.binds(), map_model_version_row)
            .await
            .map_err(internal)?;

        let next_page_token =
            compute_next_token(rows.len() as i64, max_results + 1, offset, max_results);

        let mut out = Vec::new();
        for row in rows.into_iter().take(max_results as usize) {
            let tags = super::model_versions::fetch_model_version_tags(
                self.db(),
                &row.workspace,
                &row.name,
                row.version,
            )
            .await?;
            // Aliases are intentionally NOT populated here (Python parity).
            out.push(row.into_entity(tags, Vec::new()));
        }
        Ok(ModelVersionsPage {
            model_versions: out,
            next_page_token,
        })
    }
}

/// Describes the main entity table and its tag table for filter/anti-join
/// building. `group_cols` are the entity key columns (`(workspace, name)` for
/// registered models, `(workspace, name, version)` for model versions).
struct TableSpec {
    main_table: &'static str,
    main_alias: &'static str,
    tag_table: &'static str,
    /// Entity key columns present on both the main table and the tag table.
    key_cols: &'static [&'static str],
    /// Whether this is the model-versions table (drives attribute handling).
    is_model_versions: bool,
}

impl TableSpec {
    fn registered_models() -> Self {
        Self {
            main_table: REGISTERED_MODELS,
            main_alias: "rm",
            tag_table: REGISTERED_MODEL_TAGS,
            key_cols: &["workspace", "name"],
            is_model_versions: false,
        }
    }

    fn model_versions() -> Self {
        Self {
            main_table: MODEL_VERSIONS,
            main_alias: "mv",
            tag_table: MODEL_VERSION_TAGS,
            key_cols: &["workspace", "name", "version"],
            is_model_versions: true,
        }
    }
}

/// A parsed tag predicate accumulated per key (mirroring Python's
/// `tag_filters[key]` list: one `key = ?` plus one value clause per filter on
/// that key). The value predicates are built lazily (in
/// [`append_tag_having_join`]) so their binds are pushed in SQL-text order.
struct TagKeyGroup {
    key: String,
    /// `(comparator, value)` per filter clause on this key.
    clauses: Vec<(String, String)>,
}

/// The result of building the filtered `FROM`: the `FROM` fragment (main table
/// wrapped as a subquery, plus prompt/tag joins) and any WHERE predicates that
/// must live in the outer `WHERE` (the prompt anti-join's `IS NULL`, plus the
/// caller-appended MV deleted-stage clause).
struct FilterFrom {
    from: String,
    where_preds: Vec<String>,
}

impl FilterFrom {
    /// Render the outer WHERE (` WHERE a AND b`), or empty if none.
    fn where_clause(&self) -> String {
        if self.where_preds.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", self.where_preds.join(" AND "))
        }
    }
}

/// Build the filtered `FROM` (main table as a subquery carrying its attribute
/// predicates, plus the prompt anti-join and tag HAVING-count join). Attribute
/// predicates live inside the main subquery; the prompt anti-join contributes an
/// `IS NULL` to the outer WHERE (`_update_query_to_exclude_prompts` uses a LEFT
/// JOIN + `.filter(prompts.c.name.is_(None))`).
fn build_filter_from(
    b: &mut QueryBuilder,
    workspace: &str,
    parsed: &[Comparison],
    spec: &TableSpec,
) -> Result<FilterFrom, MlflowError> {
    let dialect = b.dialect;
    let alias = spec.main_alias;

    // Collect attribute predicates and per-key tag groups.
    let mut attribute_preds: Vec<String> = Vec::new();
    // Preserve first-seen key order (Python dict insertion order) for stable SQL.
    let mut tag_groups: Vec<TagKeyGroup> = Vec::new();

    for f in parsed {
        match f.entity_type.as_str() {
            "attribute" => {
                // Attribute predicates live INSIDE the main-table subquery
                // `(SELECT * FROM <table> WHERE ...)`, where columns are
                // unqualified — so no alias prefix.
                attribute_preds.push(build_attribute_pred(b, dialect, f, spec)?);
            }
            "tag" => {
                let comparator = f.comparator.to_uppercase();
                if !VALID_TAG_COMPARATORS.contains(&comparator.as_str()) {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "Invalid comparator for tag: {}",
                        f.comparator
                    )));
                }
                // Defer building the value predicate (and its bind) until the tag
                // HAVING-count join is assembled, so the bind order matches the
                // placeholder text order (the value predicate lands late in the
                // SQL, inside the `tagf` subquery).
                let value = as_str(&f.value)?;
                match tag_groups.iter_mut().find(|g| g.key == f.key) {
                    Some(g) => g.clauses.push((comparator, value)),
                    None => tag_groups.push(TagKeyGroup {
                        key: f.key.clone(),
                        clauses: vec![(comparator, value)],
                    }),
                }
            }
            other => {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid token type: {other}"
                )));
            }
        }
    }

    // Workspace clause on the main table (always) — bare column, inside the
    // subquery.
    let ws_ph = b.bind(Val::Text(workspace.to_string()));
    attribute_preds.push(format!("workspace = {ws_ph}"));

    // The main table, filtered by attribute predicates, becomes a subquery so
    // the returned fragment is a single `FROM`-able expression that the caller
    // can further constrain (MV's deleted clause) and order.
    let main_where = attribute_preds.join(" AND ");
    let mut from = format!(
        "(SELECT * FROM {main} WHERE {main_where}) {alias}",
        main = spec.main_table
    );

    let querying_prompt = is_querying_prompt(parsed);
    let mut where_preds: Vec<String> = Vec::new();

    // Prompt exclusion pops the prompt tag key out of the tag groups FIRST
    // (`_update_query_to_exclude_prompts`), then LEFT JOINs the prompt set and
    // filters on `IS NULL` in the outer WHERE.
    if !querying_prompt {
        tag_groups.retain(|g| g.key != IS_PROMPT_TAG_KEY);
        let (joined, is_null) = append_prompt_anti_join(b, dialect, from, spec, workspace);
        from = joined;
        where_preds.push(is_null);
    }

    // AND-of-tags: OR the per-key groups, GROUP BY key cols, HAVING count ==
    // number of distinct keys, then join back to the main table.
    if !tag_groups.is_empty() {
        from = append_tag_having_join(b, dialect, from, spec, workspace, &tag_groups);
    }

    Ok(FilterFrom { from, where_preds })
}

/// Build a single attribute predicate on the (unqualified) main-table column,
/// applying the per-search comparator validation and column aliases. The
/// predicate lands inside the main-table subquery, so columns are unqualified.
fn build_attribute_pred(
    b: &mut QueryBuilder,
    dialect: Dialect,
    f: &Comparison,
    spec: &TableSpec,
) -> Result<String, MlflowError> {
    let comparator = f.comparator.to_uppercase();
    if spec.is_model_versions {
        if !MV_VALID_SEARCH_ATTRIBUTE_KEYS.contains(&f.key.as_str()) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid attribute name: {}",
                f.key
            )));
        }
        let is_numeric = MV_NUMERIC_ATTRIBUTES.contains(&f.key.as_str());
        if is_numeric {
            if !VALID_MV_NUMERIC_COMPARATORS.contains(&comparator.as_str()) {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid comparator for attribute {}: {}",
                    f.key, f.comparator
                )));
            }
        } else if !VALID_MV_STRING_COMPARATORS.contains(&comparator.as_str())
            || (comparator == "IN" && f.key != "run_id")
        {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator for attribute: {}",
                f.comparator
            )));
        }
        // Column aliases: `source_path`→`source`, `version_number`→`version`.
        let column = match f.key.as_str() {
            "source_path" => "source",
            "version_number" => "version",
            other => other,
        };
        if comparator == "IN" {
            return in_predicate(b, column, &f.value);
        }
        if is_numeric {
            let n = numeric_value(&f.value)?;
            let p = b.bind(Val::Int(n));
            return Ok(format!("{column} {comparator} {p}"));
        }
        Ok(string_predicate(
            b,
            dialect,
            column,
            &comparator,
            &as_str(&f.value)?,
        ))
    } else {
        // Registered models: only `name`, string comparators.
        if f.key != "name" {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid attribute name: {}",
                f.key
            )));
        }
        if !VALID_RM_ATTR_COMPARATORS.contains(&comparator.as_str()) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator for attribute: {}",
                f.comparator
            )));
        }
        Ok(string_predicate(
            b,
            dialect,
            "name",
            &comparator,
            &as_str(&f.value)?,
        ))
    }
}

/// `_update_query_to_exclude_prompts`: LEFT JOIN the set of prompt-tagged
/// entities (workspace, name) and require the join to miss (`IS NULL` in the
/// outer WHERE). Returns `(from_with_join, is_null_where_pred)`.
///
/// The prompt subquery groups by `(workspace, name)` only — even for model
/// versions — because a prompt tag lives on the *registered model* / *name*
/// (`_update_query_to_exclude_prompts` groups the tag table by
/// `(workspace, name)` for both entity kinds), so any version of a prompt-tagged
/// name is excluded.
fn append_prompt_anti_join(
    b: &mut QueryBuilder,
    dialect: Dialect,
    from: String,
    spec: &TableSpec,
    workspace: &str,
) -> (String, String) {
    let alias = spec.main_alias;
    let key_ph = b.bind(Val::Text(IS_PROMPT_TAG_KEY.to_string()));
    // Python uses the dialect `=` comparison func on key/value; `=` is a plain
    // equality on all backends (no LIKE binary trickery), so bind directly.
    let v_ph = b.bind(Val::Text("true".to_string()));
    let ws_ph = b.bind(Val::Text(workspace.to_string()));
    let join_on = format!("{alias}.workspace = prompts.workspace AND {alias}.name = prompts.name");
    // `key` is reserved in MySQL — quote it.
    let keycol = dialect.quote_ident("key");
    let subq = format!(
        "(SELECT pt.workspace, pt.name FROM {tag} pt WHERE pt.{keycol} = {key_ph} \
         AND pt.value = {v_ph} AND pt.workspace = {ws_ph} GROUP BY pt.workspace, pt.name) prompts",
        tag = spec.tag_table
    );
    let from = format!("{from} LEFT JOIN {subq} ON {join_on}");
    (from, "prompts.name IS NULL".to_string())
}

/// AND-of-tags HAVING-count subquery joined back to the main table.
fn append_tag_having_join(
    b: &mut QueryBuilder,
    dialect: Dialect,
    from: String,
    spec: &TableSpec,
    workspace: &str,
    tag_groups: &[TagKeyGroup],
) -> String {
    let alias = spec.main_alias;
    let keycol = dialect.quote_ident("key");

    // Each key group: `(t.key = ? AND t.workspace = ? AND <val> [AND <val>...])`,
    // all OR-ed across keys. Binds are pushed in strict SQL-text order (key,
    // workspace, then each value predicate) so positional (`?`) placeholders
    // line up with their values on SQLite/MySQL.
    let mut group_clauses: Vec<String> = Vec::new();
    for g in tag_groups {
        let key_ph = b.bind(Val::Text(g.key.clone()));
        let ws_ph = b.bind(Val::Text(workspace.to_string()));
        // Python's per-key list is AND-ed: `[key = ?, *workspace_clauses, val,
        // ...]` joined by `and_`. Multiple clauses on the same key each append
        // their own value predicate, so all values are AND-ed together.
        let ands = g
            .clauses
            .iter()
            .map(|(cmp, val)| string_predicate(b, dialect, "t.value", cmp, val))
            .collect::<Vec<_>>()
            .join(" AND ");
        group_clauses.push(format!(
            "(t.{keycol} = {key_ph} AND t.workspace = {ws_ph} AND {ands})"
        ));
    }
    let or_clause = group_clauses.join(" OR ");
    let group_cols = spec.key_cols.join(", ");
    let n_keys = tag_groups.len();
    let join_on = spec
        .key_cols
        .iter()
        .map(|c| format!("{alias}.{c} = tagf.{c}"))
        .collect::<Vec<_>>()
        .join(" AND ");
    let select_cols = spec
        .key_cols
        .iter()
        .map(|c| format!("t.{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    let subq = format!(
        "(SELECT {select_cols} FROM {tag} t WHERE {or_clause} GROUP BY {group_cols} \
         HAVING COUNT(1) = {n_keys}) tagf",
        tag = spec.tag_table
    );
    format!("{from} JOIN {subq} ON {join_on}")
}

/// Build an `IN (...)` predicate over a list value.
fn in_predicate(b: &mut QueryBuilder, column: &str, value: &Value) -> Result<String, MlflowError> {
    let items = match value {
        Value::List(items) => items.clone(),
        other => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Expected a list value for IN, got {other:?}"
            )))
        }
    };
    if items.is_empty() {
        // Empty IN — matches nothing (SQL `IN ()` is invalid; emit a false).
        return Ok("1 = 0".to_string());
    }
    let phs: Vec<String> = items.into_iter().map(|s| b.bind(Val::Text(s))).collect();
    Ok(format!("{column} IN ({})", phs.join(", ")))
}

/// `=,!=,LIKE,ILIKE` predicate on a string column, matching
/// `get_sql_comparison_func` (LIKE/ILIKE case semantics per dialect).
fn string_predicate(
    b: &mut QueryBuilder,
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value: &str,
) -> String {
    match comparator {
        "LIKE" => {
            let idx = b.reserve_like(value);
            dialect.case_sensitive_like(column, idx)
        }
        "ILIKE" => {
            let idx = b.bind_index(Val::Text(value.to_string()));
            dialect.case_insensitive_like(column, idx)
        }
        _ => {
            let p = b.bind(Val::Text(value.to_string()));
            format!("{column} {comparator} {p}")
        }
    }
}

/// `_is_querying_prompt` (`sqlalchemy_store.py:821-832`).
fn is_querying_prompt(parsed: &[Comparison]) -> bool {
    for f in parsed {
        if f.entity_type != "tag" || f.key != IS_PROMPT_TAG_KEY {
            continue;
        }
        let value = as_str(&f.value).unwrap_or_default().to_lowercase();
        return (f.comparator.eq_ignore_ascii_case("=") && value == "true")
            || (f.comparator.eq_ignore_ascii_case("!=") && value == "false");
    }
    false
}

/// An order-by column resolved to a physical column on the main table.
struct OrderCol {
    column: String,
    ascending: bool,
}

/// `_parse_search_registered_models_order_by` (`sqlalchemy_store.py:834-868`):
/// key `name` maps to the `name` column; a **timestamp** key (`timestamp` or
/// `last_updated_timestamp`, per `SearchUtils.VALID_TIMESTAMP_ORDER_BY_KEYS`)
/// maps to `last_updated_time`; a trailing `name ASC` tiebreak is appended
/// unless the user ordered by `name`. Duplicate order-by *columns* are rejected
/// (Python dedups on `field.key`, so `timestamp` and `last_updated_timestamp`
/// both resolving to `last_updated_time` collide). NB: `creation_timestamp` is
/// **not** a valid store order-by key — the parser rejects it before we map.
fn build_rm_order_by(order_by: &[String]) -> Result<Vec<OrderCol>, MlflowError> {
    let mut cols: Vec<OrderCol> = Vec::new();
    let mut observed: Vec<&'static str> = Vec::new();
    for clause in order_by {
        let (token, ascending) =
            mlflow_search::parse::registered_models_order_by_store(clause).map_err(search_err)?;
        let column: &'static str = if token == "name" {
            "name"
        } else {
            // `timestamp` / `last_updated_timestamp` → last_updated_time.
            "last_updated_time"
        };
        if observed.contains(&column) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "`order_by` contains duplicate fields: {order_by:?}"
            )));
        }
        observed.push(column);
        cols.push(OrderCol {
            column: column.to_string(),
            ascending,
        });
    }
    if !observed.contains(&"name") {
        cols.push(OrderCol {
            column: "name".to_string(),
            ascending: true,
        });
    }
    Ok(cols)
}

/// `_parse_search_model_versions_order_by`: keys `name`/`version_number`/
/// `creation_timestamp`/`last_updated_timestamp`, with `name ASC` then
/// `version DESC` tiebreaks. Duplicate order-by fields rejected.
fn build_mv_order_by(order_by: &[String]) -> Result<Vec<OrderCol>, MlflowError> {
    let mut cols: Vec<OrderCol> = Vec::new();
    let mut observed: Vec<&'static str> = Vec::new();
    for clause in order_by {
        let parsed = mlflow_search::parse::model_versions_order_by(clause).map_err(search_err)?;
        let column = mv_order_column(&parsed)?;
        if observed.contains(&column) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "`order_by` contains duplicate fields: {order_by:?}"
            )));
        }
        observed.push(column);
        cols.push(OrderCol {
            column: column.to_string(),
            ascending: parsed.ascending,
        });
    }
    if !observed.contains(&"name") {
        cols.push(OrderCol {
            column: "name".to_string(),
            ascending: true,
        });
    }
    if !observed.contains(&"version") {
        cols.push(OrderCol {
            column: "version".to_string(),
            ascending: false,
        });
    }
    Ok(cols)
}

fn mv_order_column(o: &OrderBy) -> Result<&'static str, MlflowError> {
    // The parser already validated the key against VALID_ORDER_BY_ATTRIBUTE_KEYS.
    match o.key.as_str() {
        "name" => Ok("name"),
        "version_number" => Ok("version"),
        "creation_timestamp" => Ok("creation_time"),
        "last_updated_timestamp" => Ok("last_updated_time"),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid order by key '{other}' specified. Valid keys are \
             {{'name', 'version_number', 'creation_timestamp', 'last_updated_timestamp'}}"
        ))),
    }
}

/// `_compute_next_token`: a next token exists iff we fetched exactly
/// `max_results_for_query` rows.
fn compute_next_token(
    current_size: i64,
    max_results_for_query: i64,
    offset: i64,
    max_results: i64,
) -> Option<String> {
    if current_size == max_results_for_query {
        Some(create_page_token(offset + max_results))
    } else {
        None
    }
}

/// `SearchUtils.create_page_token`: `base64(json.dumps({"offset": N}))`.
fn create_page_token(offset: i64) -> String {
    use base64::Engine;
    let json = format!("{{\"offset\": {offset}}}");
    base64::engine::general_purpose::STANDARD.encode(json.as_bytes())
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
        Value::Float(f) => Ok(*f as i64),
        Value::Str(s) => s.parse::<i64>().map_err(|_| {
            MlflowError::invalid_parameter_value(format!("Expected a numeric value, got '{s}'"))
        }),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Expected a numeric value, got {other:?}"
        ))),
    }
}

fn search_err(e: mlflow_search::SearchError) -> MlflowError {
    use mlflow_error::ErrorCode;
    let code = match e.error_code {
        mlflow_search::ErrorCode::InvalidParameterValue => ErrorCode::InvalidParameterValue,
        _ => ErrorCode::InternalError,
    };
    MlflowError::new(e.message, code)
}

/// A positional placeholder + bind accumulator (same lockstep contract as the
/// tracking store's `PlaceholderGen`): every `bind` pushes a value and returns
/// its rendered placeholder, so the placeholder index and the `binds` vector
/// never drift. Carries the dialect for backend-appropriate placeholders and
/// LIKE rendering.
struct QueryBuilder {
    dialect: Dialect,
    binds: Vec<Val>,
    next: usize,
}

impl QueryBuilder {
    fn new(dialect: Dialect) -> Self {
        Self {
            dialect,
            binds: Vec::new(),
            next: 1,
        }
    }

    fn binds(&self) -> &[Val] {
        &self.binds
    }

    fn next_index(&mut self) -> usize {
        let idx = self.next;
        self.next += 1;
        idx
    }

    /// Push a bind and return its rendered placeholder string.
    fn bind(&mut self, v: Val) -> String {
        self.binds.push(v);
        self.dialect.placeholder(self.next_index())
    }

    /// Push a bind and return its 1-based placeholder index (for LIKE helpers).
    fn bind_index(&mut self, v: Val) -> usize {
        self.binds.push(v);
        self.next_index()
    }

    /// Reserve slot(s) for a case-sensitive LIKE. MySQL binds twice
    /// (`col LIKE ? AND BINARY col LIKE ?`).
    fn reserve_like(&mut self, value: &str) -> usize {
        self.binds.push(Val::Text(value.to_string()));
        let idx = self.next_index();
        if let Dialect::MySql = self.dialect {
            self.binds.push(Val::Text(value.to_string()));
            self.next_index();
        }
        idx
    }
}
