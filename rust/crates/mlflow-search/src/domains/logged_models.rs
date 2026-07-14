//! Logged models domain — ports `SearchLoggedModelsUtils` (the `search_utils.py`
//! class used by the **FileStore**) and `search_logged_model_utils`
//! (`parse_filter_string`, the much simpler hand-rolled parser the
//! **SqlAlchemyStore** actually calls from `_apply_filter_string_datasets_search_logged_models`).
//!
//! These are two genuinely different Python parsers that both happen to serve
//! "logged models filter strings":
//!
//! * [`parse_search_filter`] / [`OrderByInput`] port `SearchLoggedModelsUtils`
//!   (subclasses `SearchUtils`, the sqlparse-grammar base shared with runs) —
//!   used by `FileStore.search_logged_models`.
//! * [`parse_filter_string_sqlalchemy`] ports the standalone
//!   `mlflow.utils.search_logged_model_utils.parse_filter_string` — used by
//!   `SqlAlchemyStore._apply_filter_string_datasets_search_logged_models`, the
//!   store this crate's `mlflow-store` T2.9 module targets. It shares
//!   `_join_in_comparison_tokens` (via [`crate::common::process_statement`])
//!   with `SearchUtils`, but everything downstream — value coercion, the
//!   entity/operator validation, and error messages — is its own, simpler
//!   logic (single comparator set, `float()`-or-bust numeric values, no `IS
//!   NULL` support since the joined 2-token form fails its 3-token arity
//!   check). The order_by parser (`parse_order_by`/[`OrderByInput`] below) is
//!   observably identical between the two Python classes, so
//!   `SqlAlchemyStore._apply_order_by_search_logged_models` reuses it as-is.

use crate::ast::Comparison;
use crate::common::{process_statement, CmpToken};
use crate::domains::runs::{parse_search_filter_with, BaseConfig};
use crate::domains::shared::parse_filter_statement;
use crate::error::{Result, SearchError};
use crate::literal_eval::{literal_eval, PyLit};
use serde::Serialize;

const NUMERIC_ATTRIBUTES: &[&str] = &[
    "creation_timestamp",
    "creation_time",
    "last_updated_timestamp",
    "last_updated_time",
];
const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &[
    "name",
    "model_id",
    "model_type",
    "status",
    "source_run_id",
    "creation_timestamp",
    "creation_time",
    "last_updated_timestamp",
    "last_updated_time",
];
// VALID_ORDER_BY_ATTRIBUTE_KEYS == VALID_SEARCH_ATTRIBUTE_KEYS
const VALID_ORDER_BY_ATTRIBUTE_KEYS: &[&str] = VALID_SEARCH_ATTRIBUTE_KEYS;

const CONFIG: BaseConfig = BaseConfig {
    search_attribute_keys: VALID_SEARCH_ATTRIBUTE_KEYS,
    numeric_attributes: NUMERIC_ATTRIBUTES,
    any_attribute_lists: true,
};

pub fn parse_search_filter(filter_string: &str) -> Result<Vec<Comparison>> {
    parse_search_filter_with(filter_string, &CONFIG)
}

/// The order_by input for logged models (a JSON object).
#[derive(Debug, Clone)]
pub struct OrderByInput {
    pub field_name: Option<String>,
    pub ascending: Option<AscendingValue>,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// `ascending` may legitimately be a bool, or an invalid non-bool JSON value
/// (which yields a typed error message echoing the Python `type()`).
#[derive(Debug, Clone)]
pub enum AscendingValue {
    Bool(bool),
    /// A non-boolean value with its Python `type()` name (e.g. `str`, `int`).
    Other(&'static str),
}

/// The parsed order_by, mirroring `SearchLoggedModelsUtils.OrderBy`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoggedModelOrderBy {
    pub field_name: String,
    pub ascending: bool,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// `parse_order_by_for_logged_models`.
pub fn parse_order_by(order_by: &OrderByInput) -> Result<LoggedModelOrderBy> {
    let mut field_name = order_by.field_name.clone().ok_or_else(|| {
        SearchError::invalid_parameter_value(
            "`field_name` in the `order_by` clause must be specified.",
        )
    })?;

    if field_name.contains('.') {
        let entity = field_name.split_once('.').unwrap().0;
        if entity != "metrics" {
            return Err(SearchError::invalid_parameter_value(format!(
                "Invalid order by field name: {entity}, only `metrics.<name>` is allowed."
            )));
        }
    } else {
        field_name = field_name.trim().to_string();
        if !VALID_ORDER_BY_ATTRIBUTE_KEYS.contains(&field_name.as_str()) {
            return Err(SearchError::invalid_parameter_value(format!(
                "Invalid order by field name: {field_name}."
            )));
        }
    }

    let ascending = match &order_by.ascending {
        None => true,
        Some(AscendingValue::Bool(b)) => *b,
        Some(AscendingValue::Other(type_name)) => {
            return Err(SearchError::invalid_parameter_value(format!(
                "Value of `ascending` in the `order_by` clause must be a boolean, got \
                 <class '{type_name}'> for field {field_name}."
            )));
        }
    };

    let dataset_name = order_by.dataset_name.clone();
    let dataset_digest = order_by.dataset_digest.clone();
    if dataset_digest.is_some() && dataset_name.is_none() {
        return Err(SearchError::invalid_parameter_value(
            "`dataset_digest` can only be specified if `dataset_name` is also specified.",
        ));
    }

    // aliases = {"creation_time": "creation_timestamp"}
    let field_name = if field_name == "creation_time" {
        "creation_timestamp".to_string()
    } else {
        field_name
    };

    Ok(LoggedModelOrderBy {
        field_name,
        ascending,
        dataset_name,
        dataset_digest,
    })
}

// ============================================================================
// `search_logged_model_utils.parse_filter_string` (the SqlAlchemyStore parser)
// ============================================================================

/// `EntityType` (`search_logged_model_utils.py`): the entity a comparison's
/// left-hand side refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlaEntityType {
    Attribute,
    Metric,
    Param,
    Tag,
}

impl SqlaEntityType {
    /// `EntityType.from_str`. `s` is the raw dotted-prefix token (e.g.
    /// `"metrics"`), already lowercased by the identifier regex (`[a-z]+`).
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "attributes" => Ok(Self::Attribute),
            "metrics" => Ok(Self::Metric),
            "params" => Ok(Self::Param),
            "tags" => Ok(Self::Tag),
            other => Err(SearchError::invalid_parameter_value(format!(
                "Invalid entity type: {}. Expected one of ['attributes', 'metrics', 'params', \
                 'tags'].",
                crate::literal_eval::py_repr_str(other)
            ))),
        }
    }

    /// The `EntityType.value` used in `Entity.__repr__` (`{type.value}.{key}`).
    fn repr_value(self) -> &'static str {
        match self {
            Self::Attribute => "attributes",
            Self::Metric => "metrics",
            Self::Param => "params",
            Self::Tag => "tags",
        }
    }
}

/// A parsed comparison value for the SqlAlchemyStore logged-model filter
/// parser (`search_logged_model_utils.Comparison.value`, a Python `str |
/// float`, plus the tuple shape `ast.literal_eval` can produce for a
/// parenthesised non-numeric value).
#[derive(Debug, Clone, PartialEq)]
pub enum SqlaValue {
    /// A numeric entity's value (`float(value)`).
    Num(f64),
    /// A plain (non-parenthesised) string value, quotes stripped.
    Str(String),
    /// A parenthesised value: `ast.literal_eval` result, always coerced to a
    /// tuple (a bare string result is wrapped as a 1-tuple).
    Tuple(Vec<String>),
}

/// A parsed comparison (`search_logged_model_utils.Comparison`).
#[derive(Debug, Clone, PartialEq)]
pub struct SqlaComparison {
    pub entity_type: SqlaEntityType,
    pub key: String,
    pub comparator: String,
    pub value: SqlaValue,
}

/// `ALIASES` (`SqlLoggedModel.ALIASES`): non-dotted attribute key aliases,
/// resolved to the physical column name.
fn resolve_attribute_alias(key: &str) -> &str {
    match key {
        "creation_time" | "creation_timestamp" => "creation_timestamp_ms",
        "last_updated_timestamp" => "last_updated_timestamp_ms",
        other => other,
    }
}

/// `SqlLoggedModel.is_numeric(s)`: alias-resolves `s` itself (independent of
/// whether the caller already alias-resolved it — Python re-resolves every
/// call), then checks membership in the two timestamp columns. This is why
/// `attributes.creation_timestamp` (dotted form, key left un-resolved by
/// [`parse_entity`]) is still detected as numeric even though the stored key
/// differs from the physical column name.
fn is_numeric_attribute_key(key: &str) -> bool {
    matches!(
        resolve_attribute_alias(key),
        "creation_timestamp_ms" | "last_updated_timestamp_ms"
    )
}

/// `Entity.is_numeric`: a metric, or a numeric attribute.
fn entity_is_numeric(entity_type: SqlaEntityType, key: &str) -> bool {
    match entity_type {
        SqlaEntityType::Metric => true,
        SqlaEntityType::Attribute => is_numeric_attribute_key(key),
        _ => false,
    }
}

const NUMERIC_OPS: &[&str] = &["<", "<=", ">", ">=", "=", "!="];
const STRING_OPS: &[&str] = &["=", "!=", "LIKE", "ILIKE", "IN", "NOT IN"];

/// `Entity.validate_op`. Note the Python source has a genuine bug reproduced
/// here verbatim: the error message always renders `string_ops` in the
/// "Expected one of ..." text, even when the entity is numeric and the
/// actually-accepted set was `numeric_ops`.
fn validate_op(entity_type: SqlaEntityType, key: &str, op: &str) -> Result<()> {
    let numeric = entity_is_numeric(entity_type, key);
    let ops: &[&str] = if numeric { NUMERIC_OPS } else { STRING_OPS };
    if !ops.contains(&op) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid comparison operator for {}.{}: {}. Expected one of {}.",
            entity_type.repr_value(),
            key,
            crate::literal_eval::py_repr_str(op),
            py_tuple_repr(STRING_OPS),
        )));
    }
    Ok(())
}

/// Python `repr()` of a tuple of string literals, e.g. `('=', '!=', ...)`.
fn py_tuple_repr(items: &[&str]) -> String {
    let inner: Vec<String> = items
        .iter()
        .map(|s| crate::literal_eval::py_repr_str(s))
        .collect();
    format!("({})", inner.join(", "))
}

/// `Entity.from_str`: split a dotted identifier `<entity_type>.<key>`, or fall
/// back to a bare attribute key (alias-resolved, backtick-stripped).
fn parse_entity(identifier: &str) -> Result<(SqlaEntityType, String)> {
    // `IDENTIFIER_RE = re.compile(r"^([a-z]+)\.(.+)$")`: lowercase-only prefix,
    // a literal dot, then any (non-empty) remainder.
    if let Some(dot) = identifier.find('.') {
        let (prefix, rest) = identifier.split_at(dot);
        let rest = &rest[1..]; // drop the dot
        if !rest.is_empty() && !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_lowercase())
        {
            let entity_type = SqlaEntityType::from_str(prefix)?;
            return Ok((entity_type, trim_backticks_all(rest)));
        }
    }
    // No match: bare attribute key, alias-resolved on the *whole* identifier
    // string (matching `SqlLoggedModel.ALIASES.get(s, s)`), then backtick-trimmed.
    let resolved = resolve_attribute_alias(identifier);
    Ok((SqlaEntityType::Attribute, trim_backticks_all(resolved)))
}

/// `str.strip("`")`: strip *all* leading/trailing backtick characters (not
/// just one), matching Python's `strip` semantics for a multi-char charset.
fn trim_backticks_all(s: &str) -> String {
    s.trim_matches('`').to_string()
}

/// `str.strip("'")`: strip all leading/trailing single-quote characters.
fn strip_single_quotes(s: &str) -> String {
    s.trim_matches('\'').to_string()
}

/// Python `float(s)`: trims surrounding whitespace (Rust's `f64::from_str`
/// does not), then parses. Accepts the same `inf`/`infinity`/`nan` spellings
/// (case-insensitive, optional sign) as Rust's parser, matching CPython.
fn python_float(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

/// `search_logged_model_utils.parse_filter_string`: the SqlAlchemyStore's
/// logged-model filter parser. Ported field-for-field from the Python source,
/// including its dead (overwritten) intermediate assignment and its
/// `validate_op` message bug (see [`validate_op`]).
pub fn parse_filter_string_sqlalchemy(filter_string: Option<&str>) -> Result<Vec<SqlaComparison>> {
    let Some(filter_string) = filter_string.filter(|s| !s.is_empty()) else {
        return Ok(Vec::new());
    };

    let Some(input) = parse_filter_statement(filter_string) else {
        return Ok(Vec::new());
    };
    if input.multiple_statements() {
        // `len(parsed) != 1` in Python maps to sqlparse yielding >1 statement;
        // our lexer always yields one, so this only fires on an embedded `;`.
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid filter string: {}. Expected a single SQL expression.",
            crate::literal_eval::py_repr_str(filter_string)
        )));
    }

    let clauses = process_statement(&input.statement, false, true)
        .map_err(|_| invalid_clause_fallback(filter_string))?;

    let mut out = Vec::with_capacity(clauses.len());
    for tokens in &clauses {
        out.push(parse_one_comparison(tokens)?);
    }
    Ok(out)
}

/// `process_statement`'s "Invalid clause(s)" error uses `SearchUtils`'
/// wording; the logged-model parser instead raises its own "Expected a list
/// of comparisons separated by 'AND'" message for a non-AND, non-Comparison
/// top-level token. Since both parsers reject the same inputs (anything that
/// isn't a Comparison or an AND keyword), we only need to re-word the error.
fn invalid_clause_fallback(filter_string: &str) -> SearchError {
    SearchError::invalid_parameter_value(format!(
        "Invalid filter string: {}. Expected a list of comparisons separated by 'AND' \
         (e.g. 'metrics.loss > 0.1 AND params.lr = 0.01').",
        crate::literal_eval::py_repr_str(filter_string)
    ))
}

fn parse_one_comparison(tokens: &[CmpToken]) -> Result<SqlaComparison> {
    if tokens.len() != 3 {
        // `str(stmt)` on the (possibly synthesized) sqlparse Comparison /
        // TokenList: raw source-text concatenation, no separators added.
        let rendered: String = tokens.iter().map(CmpToken::value).collect();
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid comparison: {rendered}. Expected a comparison with 3 tokens."
        )));
    }
    let identifier = tokens[0].value();
    let op = tokens[1].value();
    let raw_value = tokens[2].value();

    let (entity_type, key) = parse_entity(identifier.trim())?;
    validate_op(entity_type, &key, &op)?;

    let numeric = entity_is_numeric(entity_type, &key);
    let value = if numeric {
        // Python: bare `float(value)` on the *raw, unstripped* token text — a
        // quoted numeric-entity value (e.g. `metrics.loss = 'v'`) fails with
        // the quotes still embedded in both the parse attempt and the message.
        let parsed: f64 = python_float(&raw_value).ok_or_else(|| {
            SearchError::python_value_error(format!(
                "could not convert string to float: {}",
                crate::literal_eval::py_repr_str(&raw_value)
            ))
        })?;
        SqlaValue::Num(parsed)
    } else if raw_value.starts_with('(') && raw_value.ends_with(')') {
        // `ast.literal_eval(value)`, then `(value,) if isinstance(value, str)
        // else value`. Unlike `SearchLoggedModelsUtils`' FileStore parser,
        // this one does **no** element-type or emptiness validation — a bare
        // string result is wrapped as a 1-tuple, a non-string element (e.g.
        // `('a', 5)`) is kept as-is, and `()` stays an empty tuple. The result
        // is later handed straight to SQLAlchemy's `column.in_(value)` with no
        // client-side check, so a mixed-type tuple only fails (if at all) at
        // the database layer on a strictly-typed backend — out of scope here;
        // we render every element via its natural string form.
        match literal_eval(&raw_value)? {
            PyLit::Str(s) => SqlaValue::Tuple(vec![s]),
            PyLit::Tuple(items) => SqlaValue::Tuple(items.iter().map(pylit_to_string).collect()),
            other => SqlaValue::Tuple(vec![pylit_to_string(&other)]),
        }
    } else {
        SqlaValue::Str(strip_single_quotes(&raw_value))
    };

    Ok(SqlaComparison {
        entity_type,
        key,
        comparator: op,
        value,
    })
}

/// Render a [`PyLit`] via its natural string form (not `repr()`) for a tuple
/// element: `Str("a")` -> `"a"`, `Int(5)` -> `"5"`, nested tuples flattened to
/// their `repr()` (never produced by this grammar, kept only for totality).
fn pylit_to_string(v: &PyLit) -> String {
    match v {
        PyLit::Str(s) => s.clone(),
        PyLit::Int(i) => i.to_string(),
        PyLit::Float(f) => f.to_string(),
        PyLit::Tuple(_) => v.repr(),
    }
}

#[cfg(test)]
mod sqla_filter_tests {
    use super::*;

    fn parse(s: &str) -> Result<Vec<SqlaComparison>> {
        parse_filter_string_sqlalchemy(Some(s))
    }

    #[test]
    fn empty_and_none() {
        assert_eq!(parse_filter_string_sqlalchemy(None).unwrap(), vec![]);
        assert_eq!(parse_filter_string_sqlalchemy(Some("")).unwrap(), vec![]);
    }

    #[test]
    fn bare_attribute_string() {
        let got = parse("name = 'foo'").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].entity_type, SqlaEntityType::Attribute);
        assert_eq!(got[0].key, "name");
        assert_eq!(got[0].comparator, "=");
        assert_eq!(got[0].value, SqlaValue::Str("foo".to_string()));
    }

    #[test]
    fn dotted_metric_numeric() {
        let got = parse("metrics.loss > 0.1").unwrap();
        assert_eq!(got[0].entity_type, SqlaEntityType::Metric);
        assert_eq!(got[0].key, "loss");
        assert_eq!(got[0].value, SqlaValue::Num(0.1));
    }

    #[test]
    fn bare_attribute_alias_resolves_to_column() {
        // Bare (non-dotted) alias resolves to the physical column name.
        let got = parse("creation_timestamp > 100").unwrap();
        assert_eq!(got[0].key, "creation_timestamp_ms");
        assert_eq!(got[0].value, SqlaValue::Num(100.0));
    }

    #[test]
    fn dotted_attribute_alias_does_not_resolve() {
        // Faithful Python quirk: the dotted `attributes.` form skips alias
        // resolution, so the key stays the pre-alias name even though the
        // entity is still detected as numeric (`SqlLoggedModel.is_numeric`
        // itself alias-resolves internally). Downstream `getattr` on this key
        // would 500 in Python; the store layer must reproduce that, not "fix"
        // it by alias-resolving here.
        let got = parse("attributes.creation_timestamp > 100").unwrap();
        assert_eq!(got[0].key, "creation_timestamp");
        assert_eq!(got[0].value, SqlaValue::Num(100.0));
    }

    #[test]
    fn tags_and_params() {
        let got = parse("tags.k = 'v'").unwrap();
        assert_eq!(got[0].entity_type, SqlaEntityType::Tag);
        assert_eq!(got[0].key, "k");

        let got = parse("params.p != 'x'").unwrap();
        assert_eq!(got[0].entity_type, SqlaEntityType::Param);
        assert_eq!(got[0].key, "p");
    }

    #[test]
    fn in_list_multi_and_single() {
        let got = parse("name IN ('a', 'b')").unwrap();
        assert_eq!(
            got[0].value,
            SqlaValue::Tuple(vec!["a".to_string(), "b".to_string()])
        );

        let got = parse("name IN ('a')").unwrap();
        assert_eq!(got[0].value, SqlaValue::Tuple(vec!["a".to_string()]));
    }

    #[test]
    fn in_list_empty_and_mixed_types_no_validation() {
        // Python performs no element-type or emptiness validation here.
        let got = parse("name IN ()").unwrap();
        assert_eq!(got[0].value, SqlaValue::Tuple(vec![]));

        let got = parse("name IN ('a', 5)").unwrap();
        assert_eq!(
            got[0].value,
            SqlaValue::Tuple(vec!["a".to_string(), "5".to_string()])
        );
    }

    #[test]
    fn invalid_entity_type() {
        let err = parse("bogus.k = 'v'").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid entity type: 'bogus'. Expected one of ['attributes', 'metrics', 'params', \
             'tags']."
        );
    }

    #[test]
    fn invalid_operator_for_string_entity() {
        let err = parse("status > 'v'").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid comparison operator for attributes.status: '>'. Expected one of \
             ('=', '!=', 'LIKE', 'ILIKE', 'IN', 'NOT IN')."
        );
    }

    #[test]
    fn invalid_operator_for_numeric_entity_still_quotes_string_ops() {
        // The Python bug: even on the numeric branch, the message always
        // lists `string_ops`, not the `numeric_ops` that were actually
        // enforced.
        let err = parse("metrics.loss LIKE '0.1'").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid comparison operator for metrics.loss: 'LIKE'. Expected one of \
             ('=', '!=', 'LIKE', 'ILIKE', 'IN', 'NOT IN')."
        );
    }

    #[test]
    fn non_numeric_value_for_metric_is_python_value_error() {
        let err = parse("metrics.loss = 'v'").unwrap_err();
        assert_eq!(err.error_code, crate::error::ErrorCode::PythonValueError);
        assert_eq!(err.message, "could not convert string to float: \"'v'\"");
    }

    #[test]
    fn or_is_rejected() {
        let err = parse("name = 'a' OR name = 'b'").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid filter string: \"name = 'a' OR name = 'b'\". Expected a list of \
             comparisons separated by 'AND' (e.g. 'metrics.loss > 0.1 AND params.lr = 0.01')."
        );
    }

    #[test]
    fn multiple_statements_rejected() {
        let err = parse("name = 'a'; name = 'b'").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid filter string: \"name = 'a'; name = 'b'\". Expected a single SQL \
             expression."
        );
    }

    #[test]
    fn is_null_rejected_as_arity_error() {
        let err = parse("name IS NULL").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid comparison: nameIS NULL. Expected a comparison with 3 tokens."
        );
    }

    #[test]
    fn and_joined_clauses() {
        let got = parse("name = 'a' AND metrics.loss > 0.1").unwrap();
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn backtick_quoted_tag_key() {
        let got = parse("tags.`weird key` = 'v'").unwrap();
        assert_eq!(got[0].key, "weird key");
    }
}
