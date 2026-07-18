//! Issue-search domain — ports `SearchIssuesUtils`.

use crate::ast::{Comparison, Value};
use crate::common::{process_statement, strip_quotes, CmpToken};
use crate::domains::shared::parse_filter_statement;
use crate::error::{Result, SearchError};

const ATTRIBUTE: &str = "attribute";
const VALID_KEYS: &[&str] = &["status", "source_run_id"];
const VALID_COMPARATORS: &[&str] = &["=", "!="];

pub fn parse_search_filter(filter_string: &str) -> Result<Vec<Comparison>> {
    let Some(input) = parse_filter_statement(filter_string) else {
        return Ok(vec![]);
    };
    if let Some(error) = input.multiple_expression_error() {
        return Err(error);
    }
    process_statement(&input.statement, false, false)?
        .iter()
        .map(|tokens| get_comparison(tokens))
        .collect()
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

fn get_comparison(tokens: &[CmpToken]) -> Result<Comparison> {
    if tokens.len() != 3 {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid comparison: expected 3 tokens, got {}",
            tokens.len()
        )));
    }
    if !tokens[0].is_identifier() {
        return Err(SearchError::invalid_parameter_value(
            "Invalid comparison: left side must be an identifier".to_string(),
        ));
    }

    let key = strip_quotes(&tokens[0].value(), false)?.trim().to_string();
    if !VALID_KEYS.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid filter field '{key}'. Supported fields: {}",
            py_set(VALID_KEYS)
        )));
    }
    let comparator = tokens[1].value().to_uppercase();
    if !VALID_COMPARATORS.contains(&comparator.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid comparator '{comparator}'. Supported comparators: {}",
            py_set(VALID_COMPARATORS)
        )));
    }
    let value = strip_quotes(&tokens[2].value(), false)?.trim().to_string();
    Ok(Comparison {
        entity_type: ATTRIBUTE.to_string(),
        key,
        comparator,
        value: Value::Str(value),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_issue_filters() {
        let filters =
            parse_search_filter("status = 'resolved' AND source_run_id != 'run-1'").unwrap();
        assert_eq!(filters.len(), 2);
        assert_eq!(filters[0].key, "status");
        assert_eq!(filters[0].value, Value::Str("resolved".to_string()));
        assert_eq!(filters[1].comparator, "!=");
    }

    #[test]
    fn rejects_unsupported_fields_and_comparators() {
        assert!(parse_search_filter("severity = 'high'")
            .unwrap_err()
            .message
            .contains("Invalid filter field"));
        assert!(parse_search_filter("status LIKE 'resolved'")
            .unwrap_err()
            .message
            .contains("Invalid comparator"));
    }
}
