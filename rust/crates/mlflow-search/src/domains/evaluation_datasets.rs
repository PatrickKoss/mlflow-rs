//! Evaluation-dataset domain — ports `SearchEvaluationDatasetsUtils`.

use crate::ast::{Comparison, OrderBy, Value};
use crate::common::{process_statement, strip_quotes, CmpToken};
use crate::domains::runs::{py_set, validate_comparison};
use crate::domains::shared::{parse_filter_statement, parse_order_by_string};
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

const ATTRIBUTE: &str = "attribute";
const TAG: &str = "tag";

const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &[
    "name",
    "created_time",
    "last_update_time",
    "created_by",
    "last_updated_by",
];
const VALID_ORDER_BY_ATTRIBUTE_KEYS: &[&str] = &["name", "created_time", "last_update_time"];
const NUMERIC_ATTRIBUTES: &[&str] = &["created_time", "last_update_time"];

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

struct Ident {
    entity_type: String,
    key: String,
}

fn get_identifier(identifier: &str, valid_attributes: &[&str]) -> Result<Ident> {
    let (entity_type, key) = match identifier.split_once('.') {
        None => (ATTRIBUTE, identifier),
        Some(("tags", key)) => (TAG, key),
        Some((invalid, _)) => {
            return Err(SearchError::invalid_parameter_value(format!(
                "Invalid identifier token '{invalid}' specified"
            )));
        }
    };
    // Unlike several older SearchUtils domains, the Python dataset utility
    // does not trim quoting from identifier keys. Backticks are therefore
    // observable (and invalid for attributes) and remain part of tag keys.
    let key = key.to_string();
    if entity_type == ATTRIBUTE && !valid_attributes.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid attribute key '{key}' specified. Valid keys are: {}",
            py_set(valid_attributes)
        )));
    }
    Ok(Ident {
        entity_type: entity_type.to_string(),
        key,
    })
}

fn get_value(identifier_type: &str, key: &str, token: &CmpToken) -> Result<Value> {
    let token_type = token.ttype();
    let is_string_or_ident =
        matches!(token_type, Some(TokenKind::StringSingle)) || token.is_identifier();
    let is_numeric = matches!(
        token_type,
        Some(TokenKind::Integer) | Some(TokenKind::Float)
    );

    match identifier_type {
        TAG => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value for tag (e.g. 'my-value'). Got value {}",
                    token.value()
                )))
            }
        }
        ATTRIBUTE if NUMERIC_ATTRIBUTES.contains(&key) => {
            if is_numeric {
                Ok(Value::Str(token.value()))
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected numeric value type for numeric attribute: {key}. Found {}",
                    token.value()
                )))
            }
        }
        ATTRIBUTE if is_string_or_ident => Ok(Value::Str(strip_quotes(&token.value(), true)?)),
        ATTRIBUTE => Err(SearchError::invalid_parameter_value(format!(
            "Expected a quoted string value for attributes. Got value {}",
            token.value()
        ))),
        _ => Err(SearchError::internal_error(
            "Invalid identifier type. Expected one of ['metric', 'parameter'].",
        )),
    }
}

fn get_comparison(tokens: &[CmpToken]) -> Result<Comparison> {
    validate_comparison(tokens)?;
    let ident = get_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
    let comparator = tokens[1].value();
    let value = get_value(&ident.entity_type, &ident.key, &tokens[2])?;
    Ok(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_attributes_tags_and_ordering() {
        let parsed = parse_search_filter(
            "name LIKE 'eval_%' AND tags.priority != 'low' AND created_time >= 12",
        )
        .unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].entity_type, "attribute");
        assert_eq!(parsed[0].key, "name");
        assert_eq!(parsed[1].entity_type, "tag");
        assert_eq!(parsed[1].key, "priority");
        assert_eq!(parsed[2].value, Value::Str("12".to_string()));

        assert_eq!(
            parse_order_by("last_update_time DESC").unwrap(),
            OrderBy {
                entity_type: "attribute".to_string(),
                key: "last_update_time".to_string(),
                ascending: false,
            }
        );
    }

    #[test]
    fn rejects_non_python_identifiers_and_types() {
        assert_eq!(
            parse_search_filter("attribute.name = 'x'")
                .unwrap_err()
                .message,
            "Invalid identifier token 'attribute' specified"
        );
        assert!(parse_search_filter("created_time = 'later'")
            .unwrap_err()
            .message
            .contains("Expected numeric value type"));
        assert!(parse_order_by("created_by")
            .unwrap_err()
            .message
            .contains("Invalid attribute key"));
        assert!(parse_search_filter("`name` = 'x'")
            .unwrap_err()
            .message
            .contains("Invalid attribute key '`name`'"));
        assert_eq!(
            parse_search_filter("tags.`a.b` = 'x'").unwrap()[0].key,
            "`a.b`"
        );
    }
}
