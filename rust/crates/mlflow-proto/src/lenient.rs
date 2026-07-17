//! Lenient, ParseDict-parity JSON parsing (T12.5).
//!
//! Python's `_get_request_message` (`mlflow/server/handlers.py:1041-1049`) parses
//! the request body via google's `ParseDict(..., ignore_unknown_fields=True)`
//! **inside a swallowing try/except**: a `ParseError` sets
//! `proto_parsing_succeeded = False` but is otherwise ignored, and the handler
//! then proceeds with whatever the message ended up holding. `ParseDict` is *not*
//! transactional — google's `_Parser` mutates the message field-by-field as it
//! walks the JSON dict, so every field it managed to set *before* the failing one
//! stays set, and everything at/after the failure point is left at its default.
//!
//! This module reproduces that observable partial-parse contract on top of the
//! strict MLflow codec (`super::json`). The strict codec (prost-reflect's serde
//! deserializer) is all-or-nothing, so we can't just reuse it; instead we walk
//! the `serde_json::Value` field-by-field, delegating each individual field's
//! coercion to the strict codec (so int64-as-string, enum-by-name, timestamps,
//! base64, camelCase names, etc. all still behave exactly like `parse_dict`), and
//! stop at the first field that fails — after descending into message/repeated
//! fields to apply their own leading, successfully-parsed sub-fields.
//!
//! ## Derived ParseDict contract (verified against live Python — see
//! `tests/lenient_parity.rs` for the exact `uv run python` commands + outputs)
//!
//! * Top-level fields are applied in **JSON dict insertion order** (not proto
//!   field number). The first field whose value cannot be parsed aborts the walk;
//!   no later field is applied.
//! * Within a singular **message** field whose value fails, the parser descends
//!   and applies that sub-message's own leading fields (same rule, recursively).
//! * Within a **repeated** field, elements are applied in array order; the first
//!   failing element is descended into (for message elements) and its partial
//!   result appended, then the walk of that field — and everything after it —
//!   stops.
//! * Unknown fields are ignored (`ignore_unknown_fields=True`).

use prost_reflect::{
    DescriptorPool, DynamicMessage, FieldDescriptor, Kind, MessageDescriptor, ReflectMessage, Value,
};
use serde_json::Value as Json;

use crate::json::JsonCodecError;

/// Outcome of a lenient parse: the (possibly partial) message plus whether the
/// strict `parse_dict` would have succeeded. `proto_parsing_succeeded` gates
/// Python's schema type-validation (`_validate_param_against_schema` skips the
/// `_TYPE_VALIDATORS` when proto parsing succeeded).
pub struct LenientParse {
    pub message: DynamicMessage,
    pub proto_parsing_succeeded: bool,
}

fn descriptor_pool() -> &'static DescriptorPool {
    crate::json::descriptor_pool()
}

/// Parse `json` into a `DynamicMessage` of `type_name`, mirroring `parse_dict`'s
/// partial-parse-on-failure behavior.
///
/// First tries the strict codec; on success returns the fully-parsed message with
/// `proto_parsing_succeeded = true`. On a strict failure, re-walks the JSON
/// field-by-field applying the ParseDict contract and returns the partial message
/// with `proto_parsing_succeeded = false`. A malformed-JSON error (the body isn't
/// valid JSON at all) still surfaces as an error — `parse_dict` is only reached
/// after `get_json` produced a dict, so a non-parseable body is a different
/// failure mode (`_get_normalized_request_json`), reported as `ParseJson`.
pub fn lenient_from_mlflow_json(
    json: &str,
    type_name: &str,
) -> Result<LenientParse, JsonCodecError> {
    // Strict attempt first: the overwhelmingly common path (well-formed request).
    if let Ok(message) = crate::json::dynamic_from_mlflow_json(json, type_name) {
        return Ok(LenientParse {
            message,
            proto_parsing_succeeded: true,
        });
    }

    // The body must still be a JSON *value* for `parse_dict` to have run at all;
    // a syntax error is the `_get_normalized_request_json` failure mode.
    let value: Json = serde_json::from_str(json)?;

    let desc = descriptor_pool()
        .get_message_by_name(type_name)
        .ok_or_else(|| JsonCodecError::UnknownMessageType(type_name.to_string()))?;
    let mut message = DynamicMessage::new(desc);

    // `parse_dict` requires the top-level JSON to be an object; a non-object
    // top-level (e.g. a bare int) is a ParseError with nothing applied.
    if let Json::Object(_) = &value {
        partial_parse_message(&value, &mut message);
    }

    Ok(LenientParse {
        message,
        proto_parsing_succeeded: false,
    })
}

/// Concrete-message form of [`lenient_from_mlflow_json`], transcoding the
/// (possibly partial) `DynamicMessage` into `M`. Returns the concrete message
/// alongside `proto_parsing_succeeded`.
pub fn lenient_message_from_mlflow_json<M: prost::Message + Default>(
    json: &str,
    type_name: &str,
) -> Result<(M, bool), JsonCodecError> {
    let LenientParse {
        message,
        proto_parsing_succeeded,
    } = lenient_from_mlflow_json(json, type_name)?;
    Ok((message.transcode_to::<M>()?, proto_parsing_succeeded))
}

/// Apply the fields of a JSON object onto `message` in insertion order, stopping
/// at (and after descending into) the first field that fails to parse. Returns
/// `true` if every field parsed, `false` on the first failure.
fn partial_parse_message(value: &Json, message: &mut DynamicMessage) -> bool {
    let Json::Object(obj) = value else {
        return false;
    };
    let desc = message.descriptor();
    for (key, field_json) in obj {
        // `ignore_unknown_fields=True`: prost-reflect matches json_name first,
        // then the raw field name — mirror that lookup order.
        let Some(field) = desc
            .get_field_by_json_name(key)
            .or_else(|| desc.get_field_by_name(key))
        else {
            continue;
        };

        if apply_field(&field, field_json, message) {
            continue;
        }
        return false;
    }
    true
}

/// Try to apply a single field's JSON value onto `message`. Returns `true` on
/// full success. On failure, applies whatever leading sub-structure parses
/// (descending into message / repeated-message values) and returns `false`.
fn apply_field(field: &FieldDescriptor, field_json: &Json, message: &mut DynamicMessage) -> bool {
    // Fast path: let the strict codec coerce this one field. We build a
    // single-field object `{name: value}` and deserialize into a fresh message
    // of the same type, then move the parsed value over. This reuses the codec's
    // full type handling (int64-as-string, enum-by-name, Timestamp, base64, …).
    let parent_desc = message.descriptor();
    if let Some(parsed) = try_strict_single_field(field, field_json, &parent_desc) {
        message.set_field(field, parsed);
        return true;
    }

    // Strict single-field parse failed. Descend to salvage leading sub-fields,
    // matching ParseDict's non-transactional mutation.
    if field.is_list() {
        if let (Kind::Message(elem_desc), Json::Array(items)) = (field.kind(), field_json) {
            let mut list: Vec<Value> = Vec::new();
            for item in items {
                if let Some(Value::List(mut v)) =
                    try_strict_single_field(field, &Json::Array(vec![item.clone()]), &parent_desc)
                {
                    // A wholly-valid element: pull the single element back out.
                    if let Some(first) = v.pop() {
                        list.push(first);
                        continue;
                    }
                }
                // First failing element: descend into it (if a message/object),
                // append the partial, then stop this field entirely.
                if let Json::Object(_) = item {
                    let mut sub = DynamicMessage::new(elem_desc.clone());
                    partial_parse_message(item, &mut sub);
                    list.push(Value::Message(sub));
                }
                break;
            }
            if !list.is_empty() {
                message.set_field(field, Value::List(list));
            }
        }
        return false;
    }

    if let (Kind::Message(sub_desc), Json::Object(_)) = (field.kind(), field_json) {
        let mut sub = DynamicMessage::new(sub_desc.clone());
        let full = partial_parse_message(field_json, &mut sub);
        // Even a fully-empty partial message is set — ParseDict leaves the field
        // "present" once it starts descending (matches the observed
        // `trace_info` retaining trace_id/trace_location while request_time is
        // left default, and E4's empty tag element being appended).
        message.set_field(field, Value::Message(sub));
        return full;
    }

    // Scalar (or type-mismatched) failure: nothing applied, stop here.
    false
}

/// Deserialize the single-field object `{field: value}` into a fresh message of
/// `parent_desc` via the strict codec, returning the parsed field value on
/// success. Uses the field's proto `name` as the key (the codec accepts both the
/// snake_case name and json_name).
fn try_strict_single_field(
    field: &FieldDescriptor,
    value: &Json,
    parent_desc: &MessageDescriptor,
) -> Option<Value> {
    let mut obj = serde_json::Map::with_capacity(1);
    obj.insert(field.name().to_string(), value.clone());
    let single = Json::Object(obj).to_string();

    let options = prost_reflect::DeserializeOptions::new().deny_unknown_fields(false);
    let mut de = serde_json::Deserializer::from_str(&single);
    let parsed =
        DynamicMessage::deserialize_with_options(parent_desc.clone(), &mut de, &options).ok()?;
    de.end().ok()?;

    // The field may be absent from `parsed` if `value` was JSON null and the
    // field has no presence — treat "field not populated" as success with the
    // default (get_field returns the default Cow), which set_field re-applies.
    Some(parsed.get_field(field).into_owned())
}
