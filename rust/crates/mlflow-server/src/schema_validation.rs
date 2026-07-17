//! Handler-level schema validation, byte-matched to Python's
//! `_validate_request_json_with_schema` + the `_assert_*` family
//! (`mlflow/server/handlers.py:820-1005`).
//!
//! ## Why this exists (the lenient-parse mechanism, T12.5)
//!
//! Python's `_get_request_message` runs `parse_dict` inside a swallowing
//! try/except, records whether it succeeded (`proto_parsing_succeeded`), then
//! validates the **raw request JSON** against a per-endpoint schema. The
//! observable consequences (verified with the T12.4 differential harness):
//!
//! 1. A codec failure is swallowed. If schema validation on the raw JSON then
//!    fails, the client sees the *schema* error (e.g. the `_assert_string`
//!    "Invalid value 123 for parameter 'name' supplied: ..." message), never a
//!    codec/parse error.
//! 2. If schema validation passes despite the codec failure, the handler runs on
//!    the **partially-parsed** message (see [`mlflow_proto::lenient_from_mlflow_json`]).
//!
//! `proto_parsing_succeeded` gates the type validators: when proto parsing
//! succeeded, Python assumes the field types were already correct and *skips* the
//! `_TYPE_VALIDATORS` (`_assert_string`/`_assert_intlike`/`_assert_bool`/
//! `_assert_floatlike`/`_assert_array`/`_assert_item_type_string`), running only
//! the non-type checks (`_assert_required`, custom closures). When proto parsing
//! failed, the type validators DO run — that is how a type-mismatched field
//! surfaces its schema error instead of a raw codec error
//! (`_validate_param_against_schema`, `handlers.py:922-924`).
//!
//! This module operates on the raw `serde_json::Value` body exactly as Python
//! operates on `request_json`.

use mlflow_error::{ErrorCode, MlflowError};
use serde_json::Value as Json;

/// One schema validator, mirroring an `_assert_*` function. `Required` and the
/// type validators are distinguished because Python treats them differently under
/// `proto_parsing_succeeded` (type validators are skipped when parsing succeeded)
/// and produces different default messages on failure.
#[derive(Clone, Copy)]
pub enum Validator {
    /// `_assert_required`: value present and not the empty string.
    Required,
    /// `_assert_string`: `isinstance(x, str)`.
    String,
    /// `_assert_intlike`: `int(x)` coerces (ints, int-strings, floats, bools).
    IntLike,
    /// `_assert_bool`: `isinstance(x, bool)`.
    Bool,
    /// `_assert_floatlike`: `float(x)` coerces.
    FloatLike,
    /// `_assert_array`: `isinstance(x, list)`.
    Array,
    /// `_assert_item_type_string`: every element is a string.
    ItemTypeString,
    /// An endpoint-specific closure (e.g. log-batch's
    /// `_assert_metrics_fields_present`). Runs regardless of
    /// `proto_parsing_succeeded` (it is not a type validator) and owns its own
    /// error message, so it returns a fully-formed [`MlflowError`] on failure.
    Custom(fn(&Json) -> Result<(), MlflowError>),
}

impl Validator {
    /// Whether this is one of Python's `_TYPE_VALIDATORS`, which are skipped when
    /// `proto_parsing_succeeded` is true (`handlers.py:894-901,923-924`).
    fn is_type_validator(self) -> bool {
        matches!(
            self,
            Validator::String
                | Validator::IntLike
                | Validator::Bool
                | Validator::FloatLike
                | Validator::Array
                | Validator::ItemTypeString
        )
    }

    /// Run the assertion; `Ok(())` on pass, `Err(())` on an `_assert_*` failure
    /// (the caller composes the message). Mirrors the Python semantics exactly,
    /// including bool-is-int and numeric-string coercion.
    fn check(self, value: &Json) -> Result<(), ()> {
        match self {
            Validator::Required => {
                // `assert x is not None` and `assert x != ""`.
                match value {
                    Json::Null => Err(()),
                    Json::String(s) if s.is_empty() => Err(()),
                    _ => Ok(()),
                }
            }
            // `isinstance(x, str)`.
            Validator::String => match value {
                Json::String(_) => Ok(()),
                _ => Err(()),
            },
            // `isinstance(x, bool)`.
            Validator::Bool => match value {
                Json::Bool(_) => Ok(()),
                _ => Err(()),
            },
            // `x = int(x)` (may coerce), then `isinstance(x, int)`. `int()`
            // succeeds for Python ints, bools, floats, and numeric strings; a
            // non-numeric string keeps `x` as-is and the isinstance fails.
            Validator::IntLike => {
                if int_coercible(value) {
                    Ok(())
                } else {
                    Err(())
                }
            }
            // `x = float(x)` (may coerce), then `isinstance(x, float)`.
            Validator::FloatLike => {
                if float_coercible(value) {
                    Ok(())
                } else {
                    Err(())
                }
            }
            // `isinstance(x, list)`.
            Validator::Array => match value {
                Json::Array(_) => Ok(()),
                _ => Err(()),
            },
            // `all(isinstance(item, str) for item in x)`.
            Validator::ItemTypeString => match value {
                Json::Array(items) => {
                    if items.iter().all(|i| matches!(i, Json::String(_))) {
                        Ok(())
                    } else {
                        Err(())
                    }
                }
                // Iterating a non-list would raise in Python; treat as failure.
                _ => Err(()),
            },
            // `Custom` owns its own error and is dispatched separately in
            // `validate_param`; it never reaches here.
            Validator::Custom(_) => Ok(()),
        }
    }
}

/// `int(x)` would succeed and yield an int: Python ints, bools, floats (truncate),
/// and strings parseable as an integer literal.
fn int_coercible(value: &Json) -> bool {
    match value {
        Json::Bool(_) => true,
        Json::Number(n) => n.is_i64() || n.is_u64() || n.is_f64(),
        // Python `int("5")` ok, `int("5.0")` raises ValueError -> then
        // isinstance("5.0", int) is False, so only integer-literal strings pass.
        Json::String(s) => s.trim().parse::<i64>().is_ok() || s.trim().parse::<u64>().is_ok(),
        _ => false,
    }
}

/// `float(x)` would succeed: Python numbers, bools, and float-parseable strings.
fn float_coercible(value: &Json) -> bool {
    match value {
        Json::Bool(_) | Json::Number(_) => true,
        Json::String(s) => {
            let t = s.trim();
            t.parse::<f64>().is_ok()
                || matches!(
                    t.to_ascii_lowercase().as_str(),
                    "inf" | "-inf" | "+inf" | "infinity" | "-infinity" | "+infinity" | "nan"
                )
        }
        _ => false,
    }
}

/// Python `type(x).__name__` for the JSON value (`str`/`int`/`float`/`bool`/
/// `list`/`dict`/`NoneType`), used in the `invalid_value` "Hint" message.
fn python_type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Number(n) => {
            // A JSON number that is integral (no fraction/exponent) came from a
            // Python int; otherwise a float. serde_json exposes this via i64/u64.
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Json::String(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))` for the value, used
/// verbatim in `invalid_value`. Objects have sorted keys and no whitespace.
fn compact_json(value: &Json) -> String {
    match value {
        // serde_json's compact writer already matches `separators=(",",":")`;
        // for objects we must sort keys to mirror `sort_keys=True`.
        Json::Object(_) | Json::Array(_) => {
            let sorted = sort_keys(value);
            serde_json::to_string(&sorted).unwrap_or_default()
        }
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

/// Recursively rebuild a value with object keys sorted (`sort_keys=True`).
fn sort_keys(value: &Json) -> Json {
    match value {
        Json::Object(map) => {
            let mut entries: Vec<(&String, &Json)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = serde_json::Map::new();
            for (k, v) in entries {
                out.insert(k.clone(), sort_keys(v));
            }
            Json::Object(out)
        }
        Json::Array(items) => Json::Array(items.iter().map(sort_keys).collect()),
        other => other.clone(),
    }
}

/// Compose the `invalid_value` message for a failed type validator, matching
/// `handlers.py:934-936` + `validation.py:113-122`:
/// `Invalid value <json> for parameter '<param>' supplied:  Hint: Value was of
/// type '<type>'.` (note the double space before "Hint": the composed
/// hint-string itself has a leading space and `invalid_value` inserts `: `).
fn invalid_value_message(param: &str, value: &Json) -> String {
    let formatted = compact_json(value);
    let ty = python_type_name(value);
    format!(
        "Invalid value {formatted} for parameter '{param}' supplied:  Hint: Value was of type '{ty}'."
    )
}

/// `missing_value(path)` (`validation.py:125-126`).
fn missing_value_message(param: &str) -> String {
    format!("Missing value for required parameter '{param}'.")
}

/// A `_assert_required(x, path=...)` failure as an [`MlflowError`], with the
/// `path`-based "Missing value for required parameter '<path>'." message and the
/// trailing " See the API docs ..." suffix — for endpoint-specific
/// [`Validator::Custom`] closures (e.g. log-batch's per-element presence checks).
pub fn missing_required_error(path: &str) -> MlflowError {
    MlflowError::new(
        format!(
            "{} See the API docs for more information about request parameters.",
            missing_value_message(path)
        ),
        ErrorCode::InvalidParameterValue,
    )
}

/// One schema entry: a parameter name and its ordered list of validators.
pub struct SchemaEntry {
    pub param: &'static str,
    pub validators: &'static [Validator],
}

/// `_validate_request_json_with_schema(request_json, schema, proto_parsing_succeeded)`.
///
/// `body` is the raw request JSON (an object; a non-object body validates as if
/// empty, since `.get(key)` on a non-dict would only ever be absent). For each
/// schema entry whose key is present OR whose validators include `Required`,
/// runs `_validate_param_against_schema` on the value (or `None` when absent).
/// The `run_id`/`run_uuid` fallback (`handlers.py:998-999`) is applied.
pub fn validate_request_json_with_schema(
    body: &Json,
    schema: &[SchemaEntry],
    proto_parsing_succeeded: bool,
) -> Result<(), MlflowError> {
    let obj = body.as_object();
    for entry in schema {
        let present = obj.map(|o| o.contains_key(entry.param)).unwrap_or(false);
        let has_required = entry
            .validators
            .iter()
            .any(|v| matches!(v, Validator::Required));
        if !present && !has_required {
            continue;
        }

        let value: Json = obj
            .and_then(|o| o.get(entry.param))
            .cloned()
            .or_else(|| {
                // `run_id` falls back to `run_uuid` when absent.
                if entry.param == "run_id" {
                    obj.and_then(|o| o.get("run_uuid")).cloned()
                } else {
                    None
                }
            })
            .unwrap_or(Json::Null);

        validate_param(
            entry.param,
            &value,
            entry.validators,
            proto_parsing_succeeded,
        )?;
    }
    Ok(())
}

/// `_validate_param_against_schema` (`handlers.py:904-944`).
fn validate_param(
    param: &str,
    value: &Json,
    validators: &[Validator],
    proto_parsing_succeeded: bool,
) -> Result<(), MlflowError> {
    for v in validators {
        // Type validators are skipped when proto parsing succeeded.
        if v.is_type_validator() && proto_parsing_succeeded {
            continue;
        }
        // Custom validators carry their own error message (with the exact
        // `metrics[i].key`-style path), so they short-circuit here.
        if let Validator::Custom(f) = v {
            f(value)?;
            continue;
        }
        if v.check(value).is_ok() {
            continue;
        }
        // Failure: compose the message exactly as Python does. `_assert_required`
        // yields the "Missing value" message; every other (type) validator yields
        // the "Invalid value ... Hint" message.
        let base = match v {
            Validator::Required => missing_value_message(param),
            _ => invalid_value_message(param, value),
        };
        return Err(MlflowError::new(
            format!("{base} See the API docs for more information about request parameters."),
            ErrorCode::InvalidParameterValue,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(param: &'static str, validators: &'static [Validator]) -> SchemaEntry {
        SchemaEntry { param, validators }
    }

    #[test]
    fn create_experiment_int_name_matches_python() {
        // Python (verified live):
        //   h._validate_request_json_with_schema({'name':123}, {'name':[Required,String],...}, False)
        //   -> "Invalid value 123 for parameter 'name' supplied:  Hint: Value was
        //       of type 'int'. See the API docs ..."
        let body: Json = serde_json::json!({"name": 123});
        let schema = [
            s("name", &[Validator::Required, Validator::String]),
            s("artifact_location", &[Validator::String]),
            s("tags", &[Validator::Array]),
        ];
        let err = validate_request_json_with_schema(&body, &schema, false).unwrap_err();
        assert_eq!(
            err.message,
            "Invalid value 123 for parameter 'name' supplied:  Hint: Value was of type 'int'. \
             See the API docs for more information about request parameters."
        );
        assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    }

    #[test]
    fn missing_required_model_id_matches_python() {
        // finalize/params body omits model_id; proto parse SUCCEEDS (unknown
        // fields ignored) so type validators skip, but Required still runs.
        let body: Json = serde_json::json!({"model": {"status": "LOGGED_MODEL_READY"}});
        let schema = [
            s("model_id", &[Validator::String, Validator::Required]),
            s("status", &[Validator::IntLike, Validator::Required]),
        ];
        let err = validate_request_json_with_schema(&body, &schema, true).unwrap_err();
        assert_eq!(
            err.message,
            "Missing value for required parameter 'model_id'. \
             See the API docs for more information about request parameters."
        );
    }

    #[test]
    fn type_validators_skipped_when_proto_succeeded() {
        // name is an int but proto "succeeded" -> String validator skipped, only
        // Required runs (and passes, since 123 is present and non-empty).
        let body: Json = serde_json::json!({"name": 123});
        let schema = [s("name", &[Validator::Required, Validator::String])];
        assert!(validate_request_json_with_schema(&body, &schema, true).is_ok());
    }

    #[test]
    fn intlike_accepts_bool_int_and_numeric_string() {
        assert!(Validator::IntLike.check(&serde_json::json!(true)).is_ok());
        assert!(Validator::IntLike.check(&serde_json::json!(5)).is_ok());
        assert!(Validator::IntLike.check(&serde_json::json!("5")).is_ok());
        assert!(Validator::IntLike.check(&serde_json::json!(5.0)).is_ok());
        assert!(Validator::IntLike.check(&serde_json::json!("abc")).is_err());
    }

    #[test]
    fn string_rejects_bool() {
        assert!(Validator::String.check(&serde_json::json!(true)).is_err());
        assert!(Validator::String.check(&serde_json::json!("x")).is_ok());
    }

    #[test]
    fn bool_rejects_int() {
        assert!(Validator::Bool.check(&serde_json::json!(1)).is_err());
        assert!(Validator::Bool.check(&serde_json::json!(true)).is_ok());
    }

    #[test]
    fn invalid_value_string_type_name_and_quoting() {
        // status='abc' -> Invalid value "abc" ... type 'str'
        let body: Json = serde_json::json!({"status": "abc"});
        let schema = [s("status", &[Validator::IntLike, Validator::Required])];
        let err = validate_request_json_with_schema(&body, &schema, false).unwrap_err();
        assert_eq!(
            err.message,
            "Invalid value \"abc\" for parameter 'status' supplied:  Hint: Value was of type \
             'str'. See the API docs for more information about request parameters."
        );
    }

    #[test]
    fn run_uuid_fallback_for_run_id() {
        let body: Json = serde_json::json!({"run_uuid": "r-123"});
        let schema = [s("run_id", &[Validator::String, Validator::Required])];
        // proto_parsing_succeeded=false so String runs; run_uuid value is a
        // string -> passes, Required passes.
        assert!(validate_request_json_with_schema(&body, &schema, false).is_ok());
    }
}
