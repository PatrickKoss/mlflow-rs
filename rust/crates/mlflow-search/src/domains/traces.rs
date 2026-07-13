//! Traces domain — ports `SearchTraceUtils`.
//!
//! Adds request_metadata / span / feedback / expectation / issue entity types,
//! the `timestamp` builtin comparison path, numeric-attribute typing, IN lists
//! for a fixed key set, assessment numeric/string value validation, and the
//! post-parse key remapping (`_replace_key_to_tag_or_metadata`).

use crate::ast::{Comparison, OrderBy, Value};
use crate::common::{process_statement, strip_quotes, trim_backticks, CmpToken};
use crate::domains::runs::{parse_list_from_token, py_set, validate_comparison};
use crate::domains::shared::{parse_filter_statement, parse_order_by_string};
use crate::error::{Result, SearchError};
use crate::token::TokenKind;

const TAG: &str = "tag";
const ATTRIBUTE: &str = "attribute";
const REQUEST_METADATA: &str = "request_metadata";
const SPAN: &str = "span";
const FEEDBACK: &str = "feedback";
const EXPECTATION: &str = "expectation";
const ISSUE: &str = "issue";

const VALID_SEARCH_ATTRIBUTE_KEYS: &[&str] = &[
    "request_id",
    "timestamp",
    "timestamp_ms",
    "execution_time",
    "execution_time_ms",
    "end_time",
    "end_time_ms",
    "status",
    "client_request_id",
    "name",
    "run_id",
    "prompt",
    "text",
];
const VALID_ORDER_BY_ATTRIBUTE_KEYS: &[&str] = &[
    "experiment_id",
    "timestamp",
    "timestamp_ms",
    "execution_time",
    "execution_time_ms",
    "end_time",
    "end_time_ms",
    "status",
    "request_id",
    "name",
    "run_id",
];
const NUMERIC_ATTRIBUTES: &[&str] = &[
    "timestamp_ms",
    "timestamp",
    "execution_time_ms",
    "execution_time",
    "end_time_ms",
    "end_time",
];

const SUPPORT_IN_COMPARISON_ATTRIBUTE_KEYS: &[&str] = &[
    "name",
    "status",
    "request_id",
    "run_id",
    "client_request_id",
];

const SUPPORTED_SPAN_ATTRIBUTES: &[&str] = &["name", "type", "status"];
const SPAN_CONTENT_KEY: &str = "content";
const VALID_SPAN_ATTRIBUTE_COMPARATORS: &[&str] =
    &["!=", "=", "IN", "NOT IN", "LIKE", "ILIKE", "RLIKE"];
const VALID_SPAN_CONTENT_COMPARATORS: &[&str] = &["LIKE", "ILIKE"];
const NUMERIC_ASSESSMENT_COMPARATORS: &[&str] = &[">", ">=", "<", "<="];
const VALID_ASSESSMENT_COMPARATORS: &[&str] = &[
    "!=",
    "=",
    ">",
    ">=",
    "<",
    "<=",
    "LIKE",
    "ILIKE",
    "RLIKE",
    "IS NULL",
    "IS NOT NULL",
];
const SUPPORTED_ISSUE_ATTRIBUTES: &[&str] = &["id"];
const VALID_ISSUE_COMPARATORS: &[&str] = &["="];

// _VALID_IDENTIFIERS (base + alternates).
const VALID_IDENTIFIERS: &[&str] = &[
    "tag",
    "request_metadata",
    "attribute",
    "span",
    "feedback",
    "expectation",
    "issue",
    "tags",
    "attributes",
    "trace",
    "metadata",
];

// Key remapping constants (resolved from mlflow.tracing.constant).
const TRACE_NAME_TAG: &str = "mlflow.traceName";
const LINKED_PROMPTS_TAG: &str = "mlflow.linkedPrompts";
const SOURCE_RUN_METADATA: &str = "mlflow.sourceRun";

pub fn parse_search_filter(filter_string: &str) -> Result<Vec<Comparison>> {
    let parsed = parse_search_filter_raw(filter_string)?;
    Ok(parsed
        .into_iter()
        .map(replace_key_to_tag_or_metadata)
        .collect())
}

fn parse_search_filter_raw(filter_string: &str) -> Result<Vec<Comparison>> {
    let Some(input) = parse_filter_statement(filter_string) else {
        return Ok(vec![]);
    };
    if let Some(err) = input.multiple_expression_error() {
        return Err(err);
    }
    let clauses = process_statement(&input.statement, true, true)?;
    clauses
        .iter()
        .map(|tokens| get_comparison(tokens))
        .collect()
}

pub fn parse_order_by(order_by: &str) -> Result<OrderBy> {
    let (token_value, is_ascending) = parse_order_by_string(order_by)?;
    let ident = get_identifier(token_value.trim(), VALID_ORDER_BY_ATTRIBUTE_KEYS)?;
    // parse_order_by_for_search_traces applies _replace_key_to_tag_or_metadata.
    let comp = replace_key_to_tag_or_metadata(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator: String::new(),
        value: Value::Null,
    });
    Ok(OrderBy {
        entity_type: comp.entity_type,
        key: comp.key,
        ascending: is_ascending,
    })
}

struct Ident {
    entity_type: String,
    key: String,
}

/// `_valid_entity_type` for traces (uses _VALID_IDENTIFIERS + _ALTERNATE_IDENTIFIERS).
fn valid_entity_type(entity_type: &str) -> Result<String> {
    let et = trim_backticks(entity_type);
    if !VALID_IDENTIFIERS.contains(&et.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid entity type '{et}'. Valid values are {}",
            py_set(VALID_IDENTIFIERS)
        )));
    }
    Ok(match et.as_str() {
        "tags" => TAG.to_string(),
        "attributes" => ATTRIBUTE.to_string(),
        "trace" => ATTRIBUTE.to_string(),
        "metadata" => REQUEST_METADATA.to_string(),
        other => other.to_string(),
    })
}

/// Base `_get_identifier` (traces inherit SearchUtils._get_identifier but with
/// their own _valid_entity_type). Note `identifier.split('.', 1)` → for
/// `span.attributes.model` the key retains the rest (`attributes.model`).
fn get_identifier(identifier: &str, valid_attributes: &[&str]) -> Result<Ident> {
    let (entity_type, key) = match identifier.split_once('.') {
        None => (ATTRIBUTE.to_string(), identifier.to_string()),
        Some((e, k)) => (e.to_string(), k.to_string()),
    };
    let ident_type = valid_entity_type(&entity_type)?;
    let key = trim_backticks(&strip_quotes(&key, false)?);
    if ident_type == ATTRIBUTE && !valid_attributes.contains(&key.as_str()) {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid attribute key '{key}' specified. Valid keys are '{}'",
            py_set(valid_attributes)
        )));
    }
    // Note: the base _get_identifier also has a DATASET branch, but traces
    // never resolve to DATASET, so it is unreachable here.
    Ok(Ident {
        entity_type: ident_type,
        key,
    })
}

/// `SearchTraceUtils._get_value`.
fn get_value(identifier_type: &str, key: &str, token: &CmpToken) -> Result<Value> {
    let ttype = token.ttype();
    let is_string_or_ident =
        matches!(ttype, Some(TokenKind::StringSingle)) || token.is_identifier();
    let is_numeric = matches!(ttype, Some(TokenKind::Integer) | Some(TokenKind::Float));
    match identifier_type {
        TAG => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if token.is_parenthesis() {
                Ok(Value::List(parse_list_from_token(token)?))
            } else {
                Err(quoted_string_err(identifier_type, token))
            }
        }
        ATTRIBUTE => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if token.is_parenthesis() {
                if !SUPPORT_IN_COMPARISON_ATTRIBUTE_KEYS.contains(&key) {
                    return Err(SearchError::invalid_parameter_value(format!(
                        "Only attributes in {} supports comparison with a list of quoted string values.",
                        py_set(SUPPORT_IN_COMPARISON_ATTRIBUTE_KEYS)
                    )));
                }
                Ok(Value::List(parse_list_from_token(token)?))
            } else if is_numeric {
                if !NUMERIC_ATTRIBUTES.contains(&key) {
                    return Err(SearchError::invalid_parameter_value(format!(
                        "Only the '{}' attributes support comparison with numeric values.",
                        py_set(NUMERIC_ATTRIBUTES)
                    )));
                }
                number_value(token)
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value or a list of quoted string values for \
                     attributes. Got value {}",
                    token.value()
                )))
            }
        }
        REQUEST_METADATA => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else {
                Err(quoted_string_err(identifier_type, token))
            }
        }
        SPAN => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if token.is_parenthesis() {
                Ok(Value::List(parse_list_from_token(token)?))
            } else {
                Err(quoted_string_err(identifier_type, token))
            }
        }
        FEEDBACK | EXPECTATION => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else if is_numeric {
                number_value(token)
            } else {
                Err(SearchError::invalid_parameter_value(format!(
                    "Expected a quoted string value or numeric value for {identifier_type} \
                     (e.g. 'my-value' or 0.8). Got value {}",
                    token.value()
                )))
            }
        }
        ISSUE => {
            if is_string_or_ident {
                Ok(Value::Str(strip_quotes(&token.value(), true)?))
            } else {
                Err(quoted_string_err(identifier_type, token))
            }
        }
        _ => Err(SearchError::invalid_parameter_value(format!(
            "Invalid identifier type: {identifier_type}. Expected one of {}.",
            py_set(VALID_IDENTIFIERS)
        ))),
    }
}

fn quoted_string_err(identifier_type: &str, token: &CmpToken) -> SearchError {
    SearchError::invalid_parameter_value(format!(
        "Expected a quoted string value for {identifier_type} (e.g. 'my-value'). Got value {}",
        token.value()
    ))
}

fn number_value(token: &CmpToken) -> Result<Value> {
    match token.ttype() {
        Some(TokenKind::Integer) => Ok(Value::Int(token.value().parse().unwrap())),
        Some(TokenKind::Float) => Ok(Value::Float(token.value().parse().unwrap())),
        _ => unreachable!(),
    }
}

/// `SearchTraceUtils._validate_comparison`.
fn validate_comparison_traces(tokens: &[CmpToken]) -> Result<()> {
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
    // timestamp / timestamp_ms as a non-Identifier first token is allowed.
    if tokens.len() == 3
        && !tokens[0].is_identifier()
        && matches!(tokens[0].ttype(), Some(TokenKind::NameBuiltin))
        && matches!(
            tokens[0].value().as_str(),
            "timestamp" | "timestamp_ms" | "TIMESTAMP" | "TIMESTAMP_MS"
        )
    {
        return Ok(());
    }
    validate_comparison(tokens)
}

fn get_comparison(tokens: &[CmpToken]) -> Result<Comparison> {
    validate_comparison_traces(tokens)?;

    if tokens.len() == 2 {
        let comparator = tokens[1].value().to_uppercase();
        let ident = get_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
        return Ok(Comparison {
            entity_type: ident.entity_type,
            key: ident.key,
            comparator,
            value: Value::Null,
        });
    }

    let ident = get_identifier(&tokens[0].value(), VALID_SEARCH_ATTRIBUTE_KEYS)?;
    let comparator = tokens[1].value(); // traces preserve case
    validate_assessment_comparison_value(&ident.entity_type, &comparator, &tokens[2])?;
    let value = get_value(&ident.entity_type, &ident.key, &tokens[2])?;

    if ident.entity_type == SPAN {
        is_span(&ident.entity_type, &ident.key, &comparator)?;
    }

    Ok(Comparison {
        entity_type: ident.entity_type,
        key: ident.key,
        comparator,
        value,
    })
}

/// `_validate_assessment_comparison_value`.
fn validate_assessment_comparison_value(
    identifier_type: &str,
    comparator: &str,
    token: &CmpToken,
) -> Result<()> {
    if identifier_type != FEEDBACK && identifier_type != EXPECTATION {
        return Ok(());
    }
    let cmp_upper = comparator.to_uppercase();
    let is_numeric_cmp = NUMERIC_ASSESSMENT_COMPARATORS.contains(&cmp_upper.as_str());
    let msg = if is_numeric_cmp {
        format!(
            "Expected a numeric value for {identifier_type} when using comparator '{comparator}'. \
             Got value {}",
            token.value()
        )
    } else {
        format!(
            "Expected a quoted string value for {identifier_type} (e.g. 'my-value'). Got value {}",
            token.value()
        )
    };
    let ttype = token.ttype();
    let is_string_or_ident =
        matches!(ttype, Some(TokenKind::StringSingle)) || token.is_identifier();
    let is_numeric = matches!(ttype, Some(TokenKind::Integer) | Some(TokenKind::Float));
    if is_string_or_ident {
        if is_numeric_cmp {
            return Err(SearchError::invalid_parameter_value(msg));
        }
    } else if is_numeric {
        if !is_numeric_cmp {
            return Err(SearchError::invalid_parameter_value(msg));
        }
    } else {
        return Err(SearchError::invalid_parameter_value(msg));
    }
    Ok(())
}

/// `is_span` (validation side-effect only; returns Ok on success).
fn is_span(_key_type: &str, key_name: &str, comparator: &str) -> Result<()> {
    let cmp = comparator.to_uppercase();
    if let Some(attr_name) = key_name.strip_prefix("attributes.") {
        if attr_name.is_empty() {
            return Err(SearchError::invalid_parameter_value(
                "Span attribute name cannot be empty after 'attributes.'",
            ));
        }
        if !VALID_SPAN_ATTRIBUTE_COMPARATORS.contains(&cmp.as_str()) {
            return Err(span_cmp_err(
                key_name,
                comparator,
                VALID_SPAN_ATTRIBUTE_COMPARATORS,
            ));
        }
    } else if SUPPORTED_SPAN_ATTRIBUTES.contains(&key_name) {
        if !VALID_SPAN_ATTRIBUTE_COMPARATORS.contains(&cmp.as_str()) {
            return Err(span_cmp_err(
                key_name,
                comparator,
                VALID_SPAN_ATTRIBUTE_COMPARATORS,
            ));
        }
    } else if key_name == SPAN_CONTENT_KEY {
        if !VALID_SPAN_CONTENT_COMPARATORS.contains(&cmp.as_str()) {
            return Err(span_cmp_err(
                key_name,
                comparator,
                VALID_SPAN_CONTENT_COMPARATORS,
            ));
        }
    } else {
        let mut attrs: Vec<&str> = SUPPORTED_SPAN_ATTRIBUTES.to_vec();
        attrs.sort();
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid span attribute '{key_name}'. Supported attributes: {}, attributes.<attribute_name>.",
            attrs.join(", ")
        )));
    }
    Ok(())
}

fn span_cmp_err(key_name: &str, comparator: &str, valid: &[&str]) -> SearchError {
    SearchError::invalid_parameter_value(format!(
        "span.{key_name} comparator '{comparator}' not one of '{}'",
        py_set(valid)
    ))
}

/// `_replace_key_to_tag_or_metadata`.
fn replace_key_to_tag_or_metadata(mut parsed: Comparison) -> Comparison {
    if parsed.entity_type == SPAN {
        return parsed;
    }
    let key = parsed.key.to_lowercase();
    match key.as_str() {
        "name" => {
            parsed.entity_type = TAG.to_string();
            parsed.key = TRACE_NAME_TAG.to_string();
        }
        "prompt" => {
            parsed.entity_type = TAG.to_string();
            parsed.key = LINKED_PROMPTS_TAG.to_string();
        }
        "run_id" => {
            parsed.entity_type = REQUEST_METADATA.to_string();
            parsed.key = SOURCE_RUN_METADATA.to_string();
        }
        "text" => {
            parsed.entity_type = SPAN.to_string();
            parsed.key = SPAN_CONTENT_KEY.to_string();
        }
        "timestamp" => parsed.key = "timestamp_ms".to_string(),
        "execution_time" => parsed.key = "execution_time_ms".to_string(),
        "end_time" => parsed.key = "end_time_ms".to_string(),
        _ => {}
    }
    parsed
}

// The issue/assessment comparator-set constants are referenced only by the
// validation performed during in-memory filtering, which is out of scope for
// pure parsing; keep them wired for completeness.
#[allow(dead_code)]
const _UNUSED: (&[&str], &[&str], &[&str]) = (
    VALID_ASSESSMENT_COMPARATORS,
    SUPPORTED_ISSUE_ATTRIBUTES,
    VALID_ISSUE_COMPARATORS,
);
