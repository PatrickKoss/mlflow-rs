//! Runs domain — ports `SearchUtils` (the base class) for `search_runs`.

use crate::ast::{Comparison, OrderBy, Value};
use crate::common::{process_statement, strip_quotes, trim_backticks, CmpToken};
use crate::domains::shared::{parse_filter_statement, parse_order_by_string};
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

// Identifier canonical names.
const METRIC: &str = "metric";
const PARAM: &str = "parameter";
const TAG: &str = "tag";
const ATTRIBUTE: &str = "attribute";
const DATASET: &str = "dataset";

const IDENTIFIERS: &[&str] = &[METRIC, PARAM, TAG, ATTRIBUTE, DATASET];

const ALTERNATE_METRIC: &[&str] = &["metrics"];
const ALTERNATE_PARAM: &[&str] = &["parameters", "param", "params"];
const ALTERNATE_TAG: &[&str] = &["tags"];
const ALTERNATE_ATTRIBUTE: &[&str] = &["attr", "attributes", "run"];
const ALTERNATE_DATASET: &[&str] = &["datasets"];

const DATASET_ATTRIBUTES: &[&str] = &["name", "digest", "context"];

// VALID_SEARCH_ATTRIBUTE_KEYS for runs (sorted for stable error text).
pub(crate) const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &[
    "Created",
    "Run Name",
    "Run name",
    "artifact_uri",
    "created",
    "end_time",
    "run name",
    "run_id",
    "run_name",
    "start_time",
    "status",
    "user_id",
];

const VALID_ORDER_BY_ATTRIBUTE_KEYS: &[&str] = &[
    "Created",
    "artifact_uri",
    "created",
    "end_time",
    "run_id",
    "run_name",
    "start_time",
    "status",
    "user_id",
];

// NUMERIC_ATTRIBUTES = _BUILTIN_NUMERIC_ATTRIBUTES | _ALTERNATE_NUMERIC_ATTRIBUTES
const NUMERIC_ATTRIBUTES: &[&str] = &["start_time", "end_time", "created", "Created"];

/// Per-domain knobs for the shared base parser (`SearchUtils`), so
/// `SearchLoggedModelsUtils` (which subclasses `SearchUtils` and only changes
/// the attribute key sets and `validate_list_supported`) can reuse it.
pub(crate) struct BaseConfig {
    pub search_attribute_keys: &'static [&'static str],
    pub numeric_attributes: &'static [&'static str],
    /// `false` restricts `IN`/list attributes to `run_id` (base `SearchUtils`);
    /// `true` allows any attribute (`SearchLoggedModelsUtils` override).
    pub any_attribute_lists: bool,
}

const RUNS_CONFIG: BaseConfig = BaseConfig {
    search_attribute_keys: VALID_SEARCH_ATTRIBUTE_KEYS,
    numeric_attributes: NUMERIC_ATTRIBUTES,
    any_attribute_lists: false,
};

/// Public: `SearchUtils.parse_search_filter`.
pub fn parse_search_filter(filter_string: &str) -> Result<Vec<Comparison>> {
    parse_search_filter_with(filter_string, &RUNS_CONFIG)
}

/// Shared base `parse_search_filter`, parameterized by [`BaseConfig`].
pub(crate) fn parse_search_filter_with(
    filter_string: &str,
    config: &BaseConfig,
) -> Result<Vec<Comparison>> {
    let Some(input) = parse_filter_statement(filter_string) else {
        return Ok(vec![]);
    };
    if let Some(err) = input.multiple_expression_error() {
        return Err(err);
    }
    let clauses = process_statement(&input.statement, false, true)?;
    clauses
        .iter()
        .map(|tokens| get_comparison(tokens, config))
        .collect()
}

/// `SearchUtils.parse_order_by_for_search_runs`.
pub fn parse_order_by(order_by: &str) -> Result<OrderBy> {
    let (token_value, is_ascending) = parse_order_by_string(order_by)?;
    let ident = get_identifier(token_value.trim(), VALID_ORDER_BY_ATTRIBUTE_KEYS)?;
    Ok(OrderBy {
        entity_type: ident.entity_type,
        key: ident.key,
        ascending: is_ascending,
    })
}

/// Result of `_get_identifier`: `{type, key}`.
pub(crate) struct Ident {
    pub entity_type: String,
    pub key: String,
}

/// `_valid_entity_type`: canonicalize an entity-type token (with backticks
/// already possibly present) or raise "Invalid entity type".
fn valid_entity_type(entity_type: &str) -> Result<String> {
    let et = trim_backticks(entity_type);
    let all_valid = |s: &str| {
        IDENTIFIERS.contains(&s)
            || ALTERNATE_METRIC.contains(&s)
            || ALTERNATE_PARAM.contains(&s)
            || ALTERNATE_TAG.contains(&s)
            || ALTERNATE_ATTRIBUTE.contains(&s)
            || ALTERNATE_DATASET.contains(&s)
    };
    if !all_valid(&et) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid entity type '{et}'. Valid values are {}",
            py_list(IDENTIFIERS)
        )));
    }
    Ok(if ALTERNATE_PARAM.contains(&et.as_str()) {
        PARAM.to_string()
    } else if ALTERNATE_METRIC.contains(&et.as_str()) {
        METRIC.to_string()
    } else if ALTERNATE_TAG.contains(&et.as_str()) {
        TAG.to_string()
    } else if ALTERNATE_ATTRIBUTE.contains(&et.as_str()) {
        ATTRIBUTE.to_string()
    } else if ALTERNATE_DATASET.contains(&et.as_str()) {
        DATASET.to_string()
    } else {
        et
    })
}

/// `_get_identifier(identifier, valid_attributes)`.
pub(crate) fn get_identifier(identifier: &str, valid_attributes: &[&str]) -> Result<Ident> {
    // tokens = identifier.split(".", 1)
    let (entity_type, key) = match identifier.split_once('.') {
        None => (ATTRIBUTE.to_string(), identifier.to_string()),
        Some((e, k)) => (e.to_string(), k.to_string()),
    };
    // Note: Python catches ValueError from unpacking only when split yields >2,
    // which cannot happen with maxsplit=1; the invalid-identifier branch is only
    // reachable via that (unreachable) path, so we skip it.
    let ident_type = valid_entity_type(&entity_type)?;
    let key = trim_backticks(&strip_quotes(&key, false)?);

    if ident_type == ATTRIBUTE && !valid_attributes.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid attribute key '{key}' specified. Valid keys are '{}'",
            py_set(valid_attributes)
        )));
    }
    if ident_type == DATASET && !DATASET_ATTRIBUTES.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid dataset key '{key}' specified. Valid keys are '{}'",
            py_set(DATASET_ATTRIBUTES)
        )));
    }
    Ok(Ident {
        entity_type: ident_type,
        key,
    })
}

/// `validate_list_supported`: base restricts to `run_id`; the logged-models
/// override (`any_attribute_lists`) allows any attribute.
fn validate_list_supported(key: &str, config: &BaseConfig) -> Result<()> {
    if config.any_attribute_lists {
        return Ok(());
    }
    if key != "run_id" {
        return Err(SearchError::invalid_parameter_value(
            "Only the 'run_id' attribute supports comparison with a list of quoted string values.",
        ));
    }
    Ok(())
}

/// `_get_value(identifier_type, key, token)`.
fn get_value(
    identifier_type: &str,
    key: &str,
    token: &CmpToken,
    config: &BaseConfig,
) -> Result<Value> {
    let ttype = token.ttype();
    let is_string_or_ident =
        matches!(ttype, Some(TokenKind::StringSingle)) || token.is_identifier();
    let is_numeric = matches!(ttype, Some(TokenKind::Integer) | Some(TokenKind::Float));

    match identifier_type {
        METRIC => {
            if !is_numeric {
                return Err(SearchError::invalid_parameter_value(format!(
                    "Expected numeric value type for metric. Found {}",
                    token.value()
                )));
            }
            Ok(Value::Str(token.value()))
        }
        PARAM | TAG => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value for {identifier_type} (e.g. 'my-value'). Got value {}",
                    token.value()
                )))
            }
        }
        ATTRIBUTE => {
            if config.numeric_attributes.contains(&key) {
                if !is_numeric {
                    return Err(SearchError::invalid_parameter_value(format!(
                        "Expected numeric value type for numeric attribute: {key}. Found {}",
                        token.value()
                    )));
                }
                Ok(Value::Str(token.value()))
            } else if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if token.is_parenthesis() {
                validate_list_supported(key, config)?;
                Ok(Value::List(parse_run_ids(token)?))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value for attributes. Got value {}",
                    token.value()
                )))
            }
        }
        DATASET => {
            if DATASET_ATTRIBUTES.contains(&key) && is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if token.is_parenthesis() {
                if !matches!(key, "name" | "digest" | "context") {
                    return Err(SearchError::invalid_parameter_value(
                        "Only the dataset 'name' and 'digest' supports comparison with a list of \
                         quoted string values.",
                    ));
                }
                Ok(Value::List(parse_run_ids(token)?))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value for dataset attributes. Got value {}",
                    token.value()
                )))
            }
        }
        _ => Err(SearchError::internal_error(format!(
            "Invalid identifier type. Expected one of {}.",
            py_list(&[METRIC, PARAM])
        ))),
    }
}

/// `_parse_run_ids`: parse the parenthesis list and drop non-lowercase entries.
pub(crate) fn parse_run_ids(token: &CmpToken) -> Result<Vec<String>> {
    let list = parse_list_from_token(token)?;
    Ok(list.into_iter().filter(|r| is_lower(r)).collect())
}

/// `_parse_list_from_sql_token` + `_check_valid_identifier_list`.
pub(crate) fn parse_list_from_token(token: &CmpToken) -> Result<Vec<String>> {
    use crate::literal_eval::{literal_eval, PyLit};
    let text = token.value();
    let parsed = literal_eval(&text)?;
    let tup = match &parsed {
        PyLit::Tuple(items) => items.clone(),
        other => vec![other.clone()],
    };
    // _check_valid_identifier_list
    if tup.is_empty() {
        return Err(SearchError::invalid_parameter_value(
            "While parsing a list in the query, expected a non-empty list of string values, \
             but got empty list",
        ));
    }
    if !tup.iter().all(|x| matches!(x, PyLit::Str(_))) {
        return Err(SearchError::invalid_parameter_value(format!(
            "While parsing a list in the query, expected string value, punctuation, or \
             whitespace, but got different type in list: {}",
            parsed.repr()
        )));
    }
    Ok(tup
        .into_iter()
        .map(|x| match x {
            PyLit::Str(s) => s,
            _ => unreachable!(),
        })
        .collect())
}

/// Python `str.islower()`: at least one cased char and no uppercase char.
pub(crate) fn is_lower(s: &str) -> bool {
    let mut has_cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            return false;
        }
        if c.is_lowercase() {
            has_cased = true;
        }
    }
    has_cased
}

/// `_validate_comparison(stripped_tokens)`: shared 2-/3-token structural checks.
pub(crate) fn validate_comparison(tokens: &[CmpToken]) -> Result<()> {
    let base = "Invalid comparison clause";
    if tokens.len() == 2 {
        let comparator = tokens[1].value().to_uppercase();
        if comparator == "IS NULL" || comparator == "IS NOT NULL" {
            if !tokens[0].is_identifier() {
                return Err(SearchError::invalid_parameter_value(format!(
                    "{base}. Expected 'Identifier' found '{}'",
                    tokens[0].value()
                )));
            }
            return Ok(());
        }
    }
    if tokens.len() != 3 {
        return Err(SearchError::invalid_parameter_value(format!(
            "{base}. Expected 3 tokens found {}",
            tokens.len()
        )));
    }
    if !tokens[0].is_identifier() {
        return Err(SearchError::invalid_parameter_value(format!(
            "{base}. Expected 'Identifier' found '{}'",
            tokens[0].value()
        )));
    }
    // Python's tokens[1]/tokens[2] `not isinstance(Token)` checks are always
    // false for real sqlparse tokens (everything is a Token), so those two
    // branches never raise. We replicate that (no-op) behavior.
    Ok(())
}

/// `SearchUtils._get_comparison`.
fn get_comparison(tokens: &[CmpToken], config: &BaseConfig) -> Result<Comparison> {
    validate_comparison(tokens)?;

    if tokens.len() == 2 {
        let comparator = tokens[1].value().to_uppercase();
        let ident = get_identifier(&tokens[0].value(), config.search_attribute_keys)?;
        if ident.entity_type != TAG && ident.entity_type != PARAM {
            return Err(SearchError::invalid_parameter_value(format!(
                "IS NULL / IS NOT NULL is only supported for tags and params, not for '{}' '{}'",
                ident.entity_type, ident.key
            )));
        }
        return Ok(Comparison {
            entity_type: ident.entity_type,
            key: ident.key,
            comparator,
            value: Value::Null,
        });
    }

    let ident = get_identifier(&tokens[0].value(), config.search_attribute_keys)?;
    let comparator = tokens[1].value(); // NB: runs preserve original case
    let value = get_value(&ident.entity_type, &ident.key, &tokens[2], config)?;
    Ok(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator,
        value,
    })
}

/// Python `repr(list_of_str)` for `_IDENTIFIERS`: `['metric', 'parameter', ...]`.
pub(crate) fn py_list(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

/// Python `repr(set)` — normalized to sorted order to match the corpus
/// generator's `_normalize` (Python set iteration order is unstable).
pub(crate) fn py_set(items: &[&str]) -> String {
    let mut sorted: Vec<&&str> = items.iter().collect();
    sorted.sort();
    let inner: Vec<String> = sorted.iter().map(|s| format!("'{s}'")).collect();
    format!("{{{}}}", inner.join(", "))
}
