//! Registered models domain — ports `SearchModelUtils`.
//!
//! Entity types: attribute (name only) + tag/tags. Comparators are uppercased.
//! order_by keys: name / creation_timestamp / last_updated_timestamp, parsed
//! via `SearchExperimentsUtils._get_identifier`.

use crate::ast::{Comparison, OrderBy, Value};
use crate::common::{process_statement, strip_quotes, trim_backticks, CmpToken};
use crate::domains::experiments::get_identifier as experiments_get_identifier;
use crate::domains::runs::{py_set, validate_comparison};
use crate::domains::shared::{parse_filter_statement, parse_order_by_string};
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

const ATTRIBUTE: &str = "attribute";
const TAG: &str = "tag";

const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &["name"];
const VALID_ORDER_BY_KEYS: &[&str] = &["name", "creation_timestamp", "last_updated_timestamp"];

pub fn parse_search_filter(filter_string: &str) -> Result<Vec<Comparison>> {
    let Some(input) = parse_filter_statement(filter_string) else {
        return Ok(vec![]);
    };
    if let Some(err) = input.multiple_expression_error() {
        return Err(err);
    }
    let clauses = process_statement(&input.statement, false, false)?;
    clauses
        .iter()
        .map(|tokens| get_comparison(tokens))
        .collect()
}

pub fn parse_order_by(order_by: &str) -> Result<OrderBy> {
    let (token_value, is_ascending) = parse_order_by_string(order_by)?;
    let ident = experiments_get_identifier(token_value.trim(), VALID_ORDER_BY_KEYS)?;
    Ok(OrderBy {
        entity_type: ident.entity_type,
        key: ident.key,
        ascending: is_ascending,
    })
}

/// The order-by keys the **store** accepts (`SearchUtils`-bound
/// `VALID_ORDER_BY_KEYS_REGISTERED_MODELS`, `search_utils.py:232` — NOT the
/// `SearchModelUtils`-bound set at `:1270` used by [`parse_order_by`]/client-side
/// `sort`). The store's `_parse_search_registered_models_order_by`
/// (`sqlalchemy_store.py:846`) calls `SearchUtils.parse_order_by_for_search_registered_models`,
/// so `timestamp`/`last_updated_timestamp` are valid and `creation_timestamp` is
/// **not**.
const STORE_VALID_ORDER_BY_KEYS: &[&str] = &["timestamp", "last_updated_timestamp", "name"];

/// `SearchUtils.parse_order_by_for_search_registered_models` (the **store**
/// variant, `search_utils.py:843-853`): returns the raw `(token_value,
/// ascending)` after validating the (stripped) token against the store's key set
/// (`name`/`timestamp`/`last_updated_timestamp`). Unlike [`parse_order_by`] this
/// does **not** go through `_get_identifier`; it is a plain membership check, so
/// the caller (the registry store) maps `timestamp`/`last_updated_timestamp` to
/// the `last_updated_time` column itself.
pub fn parse_order_by_store(order_by: &str) -> Result<(String, bool)> {
    let (token_value, is_ascending) = parse_order_by_string(order_by)?;
    let token_value = token_value.trim().to_string();
    if !STORE_VALID_ORDER_BY_KEYS.contains(&token_value.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid order by key '{token_value}' specified. Valid keys are '{}'",
            py_set(&["name", "timestamp"])
        )));
    }
    Ok((token_value, is_ascending))
}

/// `_get_model_search_identifier` (shared shape with model_versions).
pub(crate) struct Ident {
    pub entity_type: String,
    pub key: String,
}

pub(crate) fn get_model_search_identifier(
    identifier: &str,
    valid_attributes: &[&str],
) -> Result<Ident> {
    let (ident_type, key) = match identifier.split_once('.') {
        None => (ATTRIBUTE.to_string(), identifier.to_string()),
        Some((entity_type, key)) => {
            let valid = ["attribute", "tag", "tags"];
            if !valid.contains(&entity_type) {
                return Err(SearchError::invalid_parameter_value(format!(
                    "Invalid entity type '{entity_type}'. Valid entity types are ('attribute', 'tag', 'tags')"
                )));
            }
            let canonical = if entity_type == "tag" || entity_type == "tags" {
                TAG.to_string()
            } else {
                ATTRIBUTE.to_string()
            };
            (canonical, key.to_string())
        }
    };
    // NB: attribute-key validation happens BEFORE backtick/quote trimming in
    // `_get_model_search_identifier` (unlike the base `_get_identifier`).
    if ident_type == ATTRIBUTE && !valid_attributes.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid attribute key '{key}' specified. Valid keys are '{}'",
            py_set(valid_attributes)
        )));
    }
    let key = trim_backticks(&strip_quotes(&key, false)?);
    Ok(Ident {
        entity_type: ident_type,
        key,
    })
}

/// `SearchModelUtils._get_value`: tag or attribute (no lists, no numbers).
fn get_value(identifier_type: &str, key: &str, token: &CmpToken) -> Result<Value> {
    let ttype = token.ttype();
    let is_string_or_ident =
        matches!(ttype, Some(TokenKind::StringSingle)) || token.is_identifier();
    match identifier_type {
        TAG => {
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
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if token.is_parenthesis() {
                if key != "run_id" {
                    return Err(SearchError::invalid_parameter_value(
                        "Only the 'run_id' attribute supports comparison with a list of quoted \
                         string values.",
                    ));
                }
                Ok(Value::List(crate::domains::runs::parse_run_ids(token)?))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value or a list of quoted string values for \
                     attributes. Got value {}",
                    token.value()
                )))
            }
        }
        _ => Err(SearchError::invalid_parameter_value(
            "Invalid identifier type. Expected one of ['attribute', 'tag'].",
        )),
    }
}

fn get_comparison(tokens: &[CmpToken]) -> Result<Comparison> {
    validate_comparison(tokens)?;
    // SearchModelUtils._get_comparison does `left, comparator, right =
    // stripped_comparison`. When the base validator accepts a 2-token
    // `IS NULL`/`IS NOT NULL` (identifier + keyword), the unpack raises an
    // uncaught Python `ValueError` — reproduced here for corpus parity.
    if tokens.len() == 2 {
        return Err(SearchError::python_value_error(
            "not enough values to unpack (expected 3, got 2)",
        ));
    }
    let ident = get_model_search_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
    let comparator = tokens[1].value().to_uppercase();
    let value = get_value(&ident.entity_type, &ident.key, &tokens[2])?;
    Ok(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator,
        value,
    })
}
