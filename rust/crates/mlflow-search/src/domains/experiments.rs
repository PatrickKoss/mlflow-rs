//! Experiments domain — ports `SearchExperimentsUtils`.
//!
//! Differences from runs: only `attribute`/`tag`/`tags` entity types; a
//! different (tags-only) IS NULL rule; and its own `_get_identifier` with a
//! distinct "Valid entity types are (...)" error.

use crate::ast::{Comparison, OrderBy, Value};
use crate::common::{process_statement, strip_quotes, trim_backticks, CmpToken};
use crate::domains::runs::{py_set, validate_comparison};
use crate::domains::shared::{parse_filter_statement, parse_order_by_string};
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

const ATTRIBUTE: &str = "attribute";
const TAG: &str = "tag";

const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &["name", "creation_time", "last_update_time"];
const VALID_ORDER_BY_ATTRIBUTE_KEYS: &[&str] =
    &["name", "experiment_id", "creation_time", "last_update_time"];
const NUMERIC_ATTRIBUTES: &[&str] = &["creation_time", "last_update_time"];

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
    let ident = get_identifier(token_value.trim(), VALID_ORDER_BY_ATTRIBUTE_KEYS)?;
    Ok(OrderBy {
        entity_type: ident.entity_type,
        key: ident.key,
        ascending: is_ascending,
    })
}

pub(crate) struct Ident {
    pub entity_type: String,
    pub key: String,
}

/// `SearchExperimentsUtils._get_identifier` (also reused by model-registry
/// order_by parsing, which is why it is `pub(crate)`).
pub(crate) fn get_identifier(identifier: &str, valid_attributes: &[&str]) -> Result<Ident> {
    let (ident_type, key) = match identifier.split_once('.') {
        None => (ATTRIBUTE.to_string(), identifier.to_string()),
        Some((entity_type, key)) => {
            let valid = ["attribute", "tag", "tags"];
            if !valid.contains(&entity_type) {
                return Err(SearchError::invalid_parameter_value(format!(
                    "Invalid entity type '{entity_type}'. Valid entity types are ('attribute', 'tag', 'tags')"
                )));
            }
            // _valid_entity_type maps tag/tags→tag, attribute→attribute.
            let canonical = if entity_type == "tags" || entity_type == "tag" {
                TAG.to_string()
            } else {
                ATTRIBUTE.to_string()
            };
            (canonical, key.to_string())
        }
    };
    let key = trim_backticks(&strip_quotes(&key, false)?);
    if ident_type == ATTRIBUTE && !valid_attributes.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid attribute key '{key}' specified. Valid keys are '{}'",
            py_set(valid_attributes)
        )));
    }
    Ok(Ident {
        entity_type: ident_type,
        key,
    })
}

fn get_value(identifier_type: &str, key: &str, token: &CmpToken) -> Result<Value> {
    // Experiments reuse SearchUtils._get_value, restricted to attribute/tag.
    let ttype = token.ttype();
    let is_string_or_ident =
        matches!(ttype, Some(TokenKind::StringSingle)) || token.is_identifier();
    let is_numeric = matches!(ttype, Some(TokenKind::Integer) | Some(TokenKind::Float));
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
            if NUMERIC_ATTRIBUTES.contains(&key) {
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
                // Base SearchUtils._get_value: only run_id may compare to a
                // list — but experiments have no run_id attribute, so this
                // always raises the "Only the 'run_id' attribute ..." error.
                if key != "run_id" {
                    return Err(SearchError::invalid_parameter_value(
                        "Only the 'run_id' attribute supports comparison with a list of quoted \
                         string values.",
                    ));
                }
                Ok(Value::List(crate::domains::runs::parse_run_ids(token)?))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value for attributes. Got value {}",
                    token.value()
                )))
            }
        }
        _ => Err(SearchError::internal_error(
            "Invalid identifier type. Expected one of ['metric', 'parameter'].",
        )),
    }
}

fn get_comparison(tokens: &[CmpToken]) -> Result<Comparison> {
    validate_comparison_experiments(tokens)?;

    if tokens.len() == 2 {
        let comparator = tokens[1].value().to_uppercase();
        let ident = get_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
        if ident.entity_type != TAG {
            return Err(SearchError::invalid_parameter_value(format!(
                "IS NULL / IS NOT NULL is only supported for tags, not for attribute '{}'",
                ident.key
            )));
        }
        return Ok(Comparison {
            entity_type: ident.entity_type,
            key: ident.key,
            comparator,
            value: Value::Null,
        });
    }

    let ident = get_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
    let comparator = tokens[1].value(); // experiments preserve case
    let value = get_value(&ident.entity_type, &ident.key, &tokens[2])?;
    Ok(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator,
        value,
    })
}

/// `SearchExperimentsUtils._validate_comparison`: like the base, but with a
/// tags-only 2-token IS NULL message that omits the base "Invalid comparison
/// clause" prefix wording ("Invalid comparison clause. Expected 'Identifier'
/// found ...").
fn validate_comparison_experiments(tokens: &[CmpToken]) -> Result<()> {
    if tokens.len() == 2 {
        let comparator = tokens[1].value().to_uppercase();
        if comparator == "IS NULL" || comparator == "IS NOT NULL" {
            if !tokens[0].is_identifier() {
                return Err(SearchError::invalid_parameter_value(format!(
                    "Invalid comparison clause. Expected 'Identifier' found '{}'",
                    tokens[0].value()
                )));
            }
            return Ok(());
        }
    }
    validate_comparison(tokens)
}
