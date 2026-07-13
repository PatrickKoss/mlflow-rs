//! Logged models domain — ports `SearchLoggedModelsUtils`.
//!
//! The filter parser subclasses `SearchUtils` (base runs logic), changing only
//! the valid attribute-key / numeric-attribute sets and allowing any attribute
//! to compare against a list. The order_by parser is entirely different: it
//! consumes a **dict** (`{"field_name", "ascending", "dataset_name",
//! "dataset_digest"}`) rather than an SQL string.

use crate::ast::Comparison;
use crate::domains::runs::{parse_search_filter_with, BaseConfig};
use crate::error::{Result, SearchError};
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
