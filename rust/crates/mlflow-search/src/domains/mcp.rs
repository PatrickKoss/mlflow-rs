//! MCP registry search domains (`SearchMCP*Utils`).

use crate::ast::{Comparison, Value};
use crate::common::{process_statement, strip_quotes, trim_backticks, CmpToken};
use crate::domains::runs::parse_list_from_token;
use crate::domains::shared::parse_filter_statement;
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

const ATTRIBUTE: &str = "attribute";
const TAG: &str = "tag";

#[derive(Clone, Copy)]
enum Domain {
    Server,
    Version,
    Endpoint,
}

pub fn parse_server_filter(filter: &str) -> Result<Vec<Comparison>> {
    parse(filter, Domain::Server)
}

pub fn parse_version_filter(filter: &str) -> Result<Vec<Comparison>> {
    parse(filter, Domain::Version)
}

pub fn parse_endpoint_filter(filter: &str) -> Result<Vec<Comparison>> {
    parse(filter, Domain::Endpoint)
}

fn parse(filter: &str, domain: Domain) -> Result<Vec<Comparison>> {
    let Some(input) = parse_filter_statement(filter) else {
        return Ok(Vec::new());
    };
    if let Some(error) = input.multiple_expression_error() {
        return Err(error);
    }
    process_statement(&input.statement, false, true)?
        .iter()
        .map(|tokens| comparison(tokens, domain))
        .collect()
}

fn comparison(tokens: &[CmpToken], domain: Domain) -> Result<Comparison> {
    if tokens.len() != 3 || !tokens[0].is_identifier() {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid comparison: expected 3 tokens, got {}",
            tokens.len()
        )));
    }
    let raw_identifier = tokens[0].value();
    let (entity_type, raw_key) = match raw_identifier.split_once('.') {
        Some(("tag" | "tags", key)) => (TAG, key),
        Some(("attribute" | "attributes" | "attr", key)) => (ATTRIBUTE, key),
        Some((kind, _)) => {
            return Err(SearchError::invalid_parameter_value(format!(
                "Invalid entity type '{kind}'. Valid values are ['metric', 'parameter', 'tag', 'attribute', 'dataset']"
            )))
        }
        None => (ATTRIBUTE, raw_identifier.as_str()),
    };
    let key = trim_backticks(&strip_quotes(raw_key, false)?);
    let valid = match domain {
        Domain::Server => &[
            "name",
            "display_name",
            "status",
            "has_access_endpoints",
            "created_at",
            "last_updated_at",
        ][..],
        Domain::Version => &["name", "version", "status", "created_at", "last_updated_at"][..],
        Domain::Endpoint => &[
            "status",
            "server_name",
            "transport_type",
            "created_at",
            "last_updated_at",
        ][..],
    };
    if entity_type == ATTRIBUTE && !valid.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid attribute key '{key}' specified. Valid keys are '{}'",
            py_set(valid)
        )));
    }
    if entity_type == TAG && !matches!(domain, Domain::Server) {
        return Err(SearchError::invalid_parameter_value(
            "Invalid filter type 'tag'.".to_string(),
        ));
    }

    let comparator = tokens[1].value().to_uppercase();
    let allowed = ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN"];
    if !allowed.contains(&comparator.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid comparator '{comparator}' for {entity_type} '{key}'."
        )));
    }

    let numeric = matches!(key.as_str(), "created_at" | "last_updated_at");
    let token = &tokens[2];
    let value = if token.is_parenthesis() {
        if key != "status" {
            let message = match domain {
                Domain::Server => format!(
                    "Only 'status' supports IN comparisons for MCP servers, got '{key}'."
                ),
                Domain::Version => format!(
                    "Only 'status' supports IN comparisons for MCP server versions, got '{key}'."
                ),
                Domain::Endpoint => {
                    "Only the 'run_id' attribute supports comparison with a list of quoted string values."
                        .to_string()
                }
            };
            return Err(SearchError::invalid_parameter_value(message));
        }
        Value::List(parse_list_from_token(token)?)
    } else if numeric {
        if !matches!(token.ttype(), Some(TokenKind::Integer | TokenKind::Float)) {
            return Err(SearchError::invalid_parameter_value(format!(
                "Expected numeric value type for numeric attribute: {key}. Found {}",
                token.value()
            )));
        }
        Value::Str(token.value())
    } else {
        Value::Str(strip_quotes(&token.value(), true)?)
    };
    Ok(Comparison {
        entity_type: entity_type.to_string(),
        key,
        comparator,
        value,
    })
}

fn py_set(values: &[&str]) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_server_attributes_tags_and_status_lists() {
        let values = parse_server_filter(
            "name LIKE '%example%' AND tags.env = 'prod' AND status IN ('active', 'draft')",
        )
        .unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[1].entity_type, "tag");
        assert_eq!(
            values[2].value,
            Value::List(vec!["active".into(), "draft".into()])
        );
    }
}
