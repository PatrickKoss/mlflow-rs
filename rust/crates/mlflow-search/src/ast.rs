//! Parsed filter/order_by AST.
//!
//! Mirrors the Python parsers' return shapes so the same values serialize
//! identically for the parity corpus:
//!
//! - `parse_search_filter` returns a list of dicts
//!   `{"type", "key", "comparator", "value"}` (Python joins clauses with AND;
//!   there is no OR — an `OR` token is rejected as an invalid clause).
//! - `value` is a raw string for runs/experiments/registered-models, but a
//!   JSON number for trace/model-version numeric attributes, `null` for
//!   `IS NULL` / `IS NOT NULL`, and a list for `IN` / `NOT IN`.
//! - order_by parsers return `(type, key, ascending)`.

use serde::Serialize;

/// A parsed comparison value. Serializes to the same JSON as the Python value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `None` (from `IS NULL` / `IS NOT NULL`).
    Null,
    /// A raw string (quotes already trimmed).
    Str(String),
    /// An integer (trace/model-version numeric attributes).
    Int(i64),
    /// A float (trace/model-version numeric attributes).
    Float(f64),
    /// A list of strings (`IN` / `NOT IN`; run_id lowercase-filtered).
    List(Vec<String>),
}

impl Value {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::Int(i) => serde_json::Value::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::List(items) => serde_json::Value::Array(
                items
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        }
    }
}

/// A single parsed comparison clause, serializing to Python's dict shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub entity_type: String,
    pub key: String,
    pub comparator: String,
    pub value: Value,
}

impl Serialize for Comparison {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        // Match Python dict key order for readability; corpus compare is
        // order-independent (both sides parse to serde_json::Value maps).
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("type", &self.entity_type)?;
        map.serialize_entry("key", &self.key)?;
        map.serialize_entry("comparator", &self.comparator)?;
        map.serialize_entry("value", &self.value.to_json())?;
        map.end()
    }
}

/// A parsed order_by clause: `(type, key, ascending)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OrderBy {
    pub entity_type: String,
    pub key: String,
    pub ascending: bool,
}
