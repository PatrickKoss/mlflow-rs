//! Model versions domain — ports `SearchModelVersionUtils`.
//!
//! Entity types: attribute (name, version_number, run_id, source_path) +
//! tag/tags. Supports `IN` for run_id and numeric comparison for
//! version_number/creation_timestamp/last_updated_timestamp (values typed as
//! JSON numbers). Comparators uppercased.

use crate::ast::{Comparison, OrderBy, Value};
use crate::common::{process_statement, strip_quotes, trim_backticks, CmpToken};
use crate::domains::experiments::get_identifier as experiments_get_identifier;
use crate::domains::runs::{parse_run_ids, py_set, validate_comparison};
use crate::domains::shared::{parse_filter_statement, parse_order_by_string};
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

const ATTRIBUTE: &str = "attribute";
const TAG: &str = "tag";

const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &["name", "version_number", "run_id", "source_path"];
const VALID_ORDER_BY_ATTRIBUTE_KEYS: &[&str] = &[
    "name",
    "version_number",
    "creation_timestamp",
    "last_updated_timestamp",
];
const NUMERIC_ATTRIBUTES: &[&str] = &[
    "version_number",
    "creation_timestamp",
    "last_updated_timestamp",
];

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
    let ident = experiments_get_identifier(token_value.trim(), VALID_ORDER_BY_ATTRIBUTE_KEYS)?;
    Ok(OrderBy {
        entity_type: ident.entity_type,
        key: ident.key,
        ascending: is_ascending,
    })
}

struct Ident {
    entity_type: String,
    key: String,
}

/// `_get_model_version_search_identifier`.
fn get_model_version_search_identifier(
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

/// `SearchModelVersionUtils._get_value`.
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
                Ok(Value::List(parse_run_ids(token)?))
            } else if matches!(ttype, Some(TokenKind::Integer) | Some(TokenKind::Float)) {
                if !NUMERIC_ATTRIBUTES.contains(&key) {
                    return Err(SearchError::invalid_parameter_value(format!(
                        "Only the '{}' attributes support comparison with numeric values.",
                        py_set(NUMERIC_ATTRIBUTES)
                    )));
                }
                Ok(parse_number(token)?)
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

fn parse_number(token: &CmpToken) -> Result<Value> {
    match token.ttype() {
        Some(TokenKind::Integer) => token
            .value()
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| SearchError::invalid_parameter_value("invalid integer")),
        Some(TokenKind::Float) => token
            .value()
            .parse::<f64>()
            .map(Value::Float)
            .map_err(|_| SearchError::invalid_parameter_value("invalid float")),
        _ => unreachable!(),
    }
}

fn get_comparison(tokens: &[CmpToken]) -> Result<Comparison> {
    validate_comparison(tokens)?;
    // `left, comparator, right = stripped_comparison` raises an uncaught Python
    // `ValueError` on an accepted 2-token IS NULL/IS NOT NULL — reproduced for
    // corpus parity.
    if tokens.len() == 2 {
        return Err(SearchError::python_value_error(
            "not enough values to unpack (expected 3, got 2)",
        ));
    }
    let ident =
        get_model_version_search_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
    let comparator = tokens[1].value().to_uppercase();
    let value = get_value(&ident.entity_type, &ident.key, &tokens[2])?;
    Ok(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator,
        value,
    })
}
