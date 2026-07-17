//! MLflow-compatible protobuf-JSON codec (T1.3).
//!
//! This module produces JSON that is **byte-identical** to Python's
//! `mlflow.utils.proto_json_utils.message_to_json` and accepts JSON the same way
//! `parse_dict` does. It is the wire contract for every proto-backed endpoint
//! (§4 of `RUST_TRACKING_SERVER_PLAN.md`).
//!
//! ## The rules we replicate (verified empirically against Python)
//!
//! * **snake_case field names** (`preserving_proto_field_name=True`).
//! * **Field order = ascending field number** — exactly what Google's
//!   `MessageToJson` emits (it iterates fields by number, not declaration
//!   order). `prost-reflect` stores fields in a `BTreeMap<u32, _>`, so iterating
//!   its `DynamicMessage` fields already yields field-number order.
//! * **proto2 presence semantics**: a field is emitted iff it is *present* on
//!   the wire (`HasField`). An `optional` field explicitly set to its default
//!   (e.g. `start_time = 0`) IS emitted; an unset `optional` field is omitted.
//!   Unset repeated/map fields (empty) are omitted.
//! * **int64/uint64/fixed64/… as JSON numbers** — MLflow un-does protobuf-JSON's
//!   string encoding (`_mark_int64_fields`). Values `> 2^53` are still emitted as
//!   full-precision integers.
//! * **int64 map *keys* stay strings** (JSON has no non-string keys); int64 map
//!   *values* are numbers.
//! * **enums by name** (`"FINISHED"`), falling back to the number if the value
//!   is not in the enum.
//! * **bytes → standard base64**.
//! * **well-known types**: `Timestamp` → RFC3339 (`"...Z"`), `Duration` →
//!   `"1.500s"`, mirroring `MessageToJson`.
//! * **pretty-printed, `indent=2`**, no trailing newline, `ensure_ascii=True`
//!   (non-ASCII escaped as `\uXXXX`, astral chars as surrogate pairs), and
//!   Python `repr`-style float formatting (`1e+20`, `1.0`, `-0.0`).
//!
//! ## Map key ordering — the one intentional deviation
//!
//! Google's `MessageToJson` iterates protobuf map fields in *hash order*, which
//! is randomized per process (verified: the same map serializes in different key
//! orders across Python runs, even with a fixed `PYTHONHASHSEED`). That output is
//! therefore not reproducible on either side. We instead emit **map keys sorted
//! lexicographically by their JSON-string form**, which is deterministic and what
//! `gen_goldens.py` normalizes the Python goldens to. Every non-map part of the
//! output is byte-identical to Python.

use std::sync::OnceLock;

use base64::Engine;
use prost_reflect::{DescriptorPool, DynamicMessage, MapKey, ReflectMessage, Value};

/// Errors from the MLflow JSON codec.
#[derive(Debug, thiserror::Error)]
pub enum JsonCodecError {
    #[error("unknown protobuf message type: {0}")]
    UnknownMessageType(String),
    #[error("failed to encode prost message: {0}")]
    Encode(#[from] prost::EncodeError),
    #[error("failed to transcode message via descriptor pool: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error("failed to parse JSON: {0}")]
    ParseJson(#[from] serde_json::Error),
    /// A GET query parameter for a `bool` proto field was not `true`/`false`.
    /// Mirrors Python's `_get_request_message` boolean validation
    /// (`mlflow/server/handlers.py:1031-1036`); the caller maps this to
    /// `INVALID_PARAMETER_VALUE`.
    #[error("Invalid boolean value: {0}, must be 'true' or 'false'.")]
    InvalidBoolQueryValue(String),
}

/// The runtime descriptor pool, rebuilt from the extension-preserving
/// `FileDescriptorSet` embedded at build time. Lazily initialized once.
pub(crate) fn descriptor_pool() -> &'static DescriptorPool {
    static POOL: OnceLock<DescriptorPool> = OnceLock::new();
    POOL.get_or_init(|| {
        static FDS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/file_descriptor_set.bin"));
        DescriptorPool::decode(FDS).expect("failed to decode embedded FileDescriptorSet")
    })
}

/// Transcode a concrete prost message into a `DynamicMessage` by round-tripping
/// through the wire bytes. Avoids requiring `ReflectMessage`/`Name` impls on the
/// generated types; the caller supplies the fully-qualified proto type name
/// (e.g. `"mlflow.Run"`).
fn to_dynamic<M: prost::Message>(
    message: &M,
    type_name: &str,
) -> Result<DynamicMessage, JsonCodecError> {
    let desc = descriptor_pool()
        .get_message_by_name(type_name)
        .ok_or_else(|| JsonCodecError::UnknownMessageType(type_name.to_string()))?;
    let bytes = message.encode_to_vec();
    Ok(DynamicMessage::decode(desc, bytes.as_slice())?)
}

/// Serialize a prost message to MLflow-compatible pretty JSON.
///
/// `type_name` is the fully-qualified protobuf message name (e.g. `"mlflow.Run"`,
/// `"mlflow.TraceInfoV3"`). Output matches Python `message_to_json` byte-for-byte
/// (modulo the documented map-key ordering).
pub fn to_mlflow_json<M: prost::Message>(
    message: &M,
    type_name: &str,
) -> Result<String, JsonCodecError> {
    let dynamic = to_dynamic(message, type_name)?;
    let value = message_to_json_value(&dynamic);
    let mut out = String::new();
    write_pretty(&mut out, &value, 0);
    Ok(out)
}

/// Parse MLflow-compatible JSON into a `DynamicMessage`, ignoring unknown fields
/// (`ParseDict(..., ignore_unknown_fields=True)` semantics).
///
/// Accepts int64 as number or string, enums by name or number, bytes as base64,
/// and both snake_case and camelCase field names (`prost-reflect`'s deserializer
/// matches on `json_name` then `name`). This is the reflection-level entry point;
/// [`from_mlflow_json`] wraps it to yield a concrete prost message.
pub fn dynamic_from_mlflow_json(
    json: &str,
    type_name: &str,
) -> Result<DynamicMessage, JsonCodecError> {
    let desc = descriptor_pool()
        .get_message_by_name(type_name)
        .ok_or_else(|| JsonCodecError::UnknownMessageType(type_name.to_string()))?;
    let options = prost_reflect::DeserializeOptions::new().deny_unknown_fields(false);
    let mut de = serde_json::Deserializer::from_str(json);
    let dynamic = DynamicMessage::deserialize_with_options(desc, &mut de, &options)?;
    de.end()?;
    Ok(dynamic)
}

/// Parse MLflow-compatible JSON into a concrete prost message, ignoring unknown
/// fields. See [`dynamic_from_mlflow_json`] for the accepted-input details.
pub fn from_mlflow_json<M: prost::Message + Default>(
    json: &str,
    type_name: &str,
) -> Result<M, JsonCodecError> {
    Ok(dynamic_from_mlflow_json(json, type_name)?.transcode_to::<M>()?)
}

/// Parse GET-request query parameters into a concrete prost message (T3.5),
/// mirroring the GET branch of Python's `_get_request_message`
/// (`mlflow/server/handlers.py:1008-1049`).
///
/// Only fields declared on the message are consulted (unknown query params are
/// ignored, matching `ignore_unknown_fields=True`). Per Python:
///
/// * a **repeated** field collects *all* query values for its name into a JSON
///   array — even a single occurrence becomes a one-element list, because the
///   query parser has no type information and protobuf requires repeated fields
///   to be lists;
/// * a **bool** field is validated to be `true`/`false` (case-insensitively) and
///   converted to a JSON boolean, else [`JsonCodecError::InvalidBoolQueryValue`];
/// * every other scalar field takes the (last) query value as a JSON string —
///   the codec's deserializer then coerces it (int64/enum/…), accepting both the
///   numeric and, for enums, the name form.
///
/// `pairs` is the ordered list of `(name, value)` query parameters (repeated
/// names appear multiple times), matching `werkzeug`'s `MultiDict` iteration.
pub fn from_query_pairs<M: prost::Message + Default>(
    pairs: &[(String, String)],
    type_name: &str,
) -> Result<M, JsonCodecError> {
    Ok(dynamic_from_query_pairs(pairs, type_name)?.transcode_to::<M>()?)
}

/// Reflection-level form of [`from_query_pairs`], yielding a `DynamicMessage`.
pub fn dynamic_from_query_pairs(
    pairs: &[(String, String)],
    type_name: &str,
) -> Result<DynamicMessage, JsonCodecError> {
    use prost_reflect::Kind;

    let desc = descriptor_pool()
        .get_message_by_name(type_name)
        .ok_or_else(|| JsonCodecError::UnknownMessageType(type_name.to_string()))?;

    let mut obj = serde_json::Map::new();
    for field in desc.fields() {
        let name = field.name();
        // `getlist(name)` — every value for this query key, in order.
        let values: Vec<&str> = pairs
            .iter()
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
            .collect();
        if values.is_empty() {
            continue;
        }

        if field.is_list() {
            obj.insert(
                name.to_string(),
                serde_json::Value::Array(
                    values
                        .iter()
                        .map(|v| serde_json::Value::String((*v).to_string()))
                        .collect(),
                ),
            );
        } else {
            // Scalar: Python's `flask_request.args.get(name)` returns the FIRST
            // value for a repeated key.
            let value = values[0];
            let json_value = if matches!(field.kind(), Kind::Bool) {
                let lowered = value.to_ascii_lowercase();
                match lowered.as_str() {
                    "true" => serde_json::Value::Bool(true),
                    "false" => serde_json::Value::Bool(false),
                    _ => return Err(JsonCodecError::InvalidBoolQueryValue(value.to_string())),
                }
            } else {
                serde_json::Value::String(value.to_string())
            };
            obj.insert(name.to_string(), json_value);
        }
    }

    let json = serde_json::Value::Object(obj).to_string();
    dynamic_from_mlflow_json(&json, type_name)
}

// ---------------------------------------------------------------------------
// Ordered JSON intermediate + walker
// ---------------------------------------------------------------------------

/// An ordered JSON value. We keep our own tree (rather than `serde_json::Value`)
/// so object key order is exactly the insertion order we control, `Int64`/`Uint64`
/// survive as full-precision integers, and floats format the Python way.
enum JsonValue {
    Null,
    Bool(bool),
    /// A signed 64-bit integer (int64/sint64/sfixed64/int32/…), rendered as a
    /// JSON number.
    Int(i64),
    /// An unsigned 64-bit integer (uint64/fixed64/uint32), rendered as a JSON
    /// number.
    Uint(u64),
    /// A 64-bit float, rendered with Python `repr` semantics.
    Double(f64),
    /// A 32-bit float, rendered with Python `repr` semantics (widened to f64).
    Float(f32),
    /// A raw, already-formatted JSON string token — including its surrounding
    /// quotes and any needed escaping. Used for well-known-type strings that
    /// carry their own formatting (Timestamp/Duration) and plain strings.
    Str(String),
    Array(Vec<JsonValue>),
    /// Insertion-ordered object. Keys are already the final JSON key strings.
    Object(Vec<(String, JsonValue)>),
}

/// Walk a `DynamicMessage` into our ordered JSON tree, applying every MLflow rule.
fn message_to_json_value(message: &DynamicMessage) -> JsonValue {
    // Well-known types serialize to their scalar JSON form, not an object.
    if let Some(v) = well_known_to_json(message) {
        return v;
    }

    let mut entries: Vec<(String, JsonValue)> = Vec::new();
    // `fields()` iterates only *present* fields (proto2 HasField semantics) in
    // ascending field-number order (BTreeMap key order).
    for field in message.fields() {
        // `fields()` yields (FieldDescriptor, &Value) for set fields.
        let (fd, value) = field;
        let name = fd.name().to_string();
        entries.push((name, value_to_json(value, &fd.kind())));
    }
    JsonValue::Object(entries)
}

/// Convert a well-known-type message to its JSON scalar, or `None` if `message`
/// is not a well-known type we special-case.
fn well_known_to_json(message: &DynamicMessage) -> Option<JsonValue> {
    match message.descriptor().full_name() {
        "google.protobuf.Timestamp" => {
            let ts: prost_types::Timestamp = message.transcode_to().ok()?;
            Some(JsonValue::Str(quote_json_string(&format_timestamp(&ts))))
        }
        "google.protobuf.Duration" => {
            let d: prost_types::Duration = message.transcode_to().ok()?;
            Some(JsonValue::Str(quote_json_string(&format_duration(&d))))
        }
        // `google.protobuf.Value`/`Struct`/`ListValue` (used by
        // `assessments.Feedback`/`Expectation.value`, T4.4): these collapse to
        // their bare JSON representation — a `Value` with its `kind` oneof
        // unset serializes as `null` (verified against Python's
        // `MessageToJson(Value())` == `"null"`, matching `ParseDict(None,
        // Value())`'s round-trip for a feedback/expectation value of `None`),
        // a `struct_value`/`list_value` recurses into an object/array, and the
        // scalar kinds recurse through `value_to_json` on the one present
        // field. `Struct` (a bare `{string: Value}` map, when it appears
        // outside a `Value.struct_value`) and `ListValue` (a bare `[Value]`)
        // are handled the same way for completeness, though no MLflow proto
        // currently embeds them directly.
        "google.protobuf.Value" => Some(value_message_to_json(message)),
        "google.protobuf.Struct" => Some(struct_message_to_json(message)),
        "google.protobuf.ListValue" => Some(list_value_message_to_json(message)),
        _ => None,
    }
}

/// `google.protobuf.Value` -> JSON: the one present field of the `kind` oneof
/// (field numbers 1-6: `null_value`, `number_value`, `string_value`,
/// `bool_value`, `struct_value`, `list_value`), or `null` when unset. A set
/// `null_value` already renders as `JsonValue::Null` via `value_to_json`'s
/// `google.protobuf.NullValue` special case, same as the unset fallback.
fn value_message_to_json(message: &DynamicMessage) -> JsonValue {
    match message.fields().next() {
        Some((fd, value)) => value_to_json(value, &fd.kind()),
        None => JsonValue::Null,
    }
}

/// `google.protobuf.Struct` -> JSON object: the `fields` map (`string ->
/// Value`), sorted the same way ordinary proto maps are (see the module docs
/// on map-key ordering).
fn struct_message_to_json(message: &DynamicMessage) -> JsonValue {
    match message.fields().next() {
        Some((fd, value)) => value_to_json(value, &fd.kind()),
        None => JsonValue::Object(Vec::new()),
    }
}

/// `google.protobuf.ListValue` -> JSON array: the `values` repeated `Value`.
fn list_value_message_to_json(message: &DynamicMessage) -> JsonValue {
    match message.fields().next() {
        Some((fd, value)) => value_to_json(value, &fd.kind()),
        None => JsonValue::Array(Vec::new()),
    }
}

/// Convert a single `Value` (with its field `Kind`) to JSON.
fn value_to_json(value: &Value, kind: &prost_reflect::Kind) -> JsonValue {
    use prost_reflect::Kind;
    match value {
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::I32(v) => JsonValue::Int(*v as i64),
        Value::I64(v) => JsonValue::Int(*v),
        Value::U32(v) => JsonValue::Uint(*v as u64),
        Value::U64(v) => JsonValue::Uint(*v),
        Value::F32(v) => JsonValue::Float(*v),
        Value::F64(v) => JsonValue::Double(*v),
        Value::String(s) => JsonValue::Str(quote_json_string(s)),
        Value::Bytes(b) => JsonValue::Str(quote_json_string(
            &base64::engine::general_purpose::STANDARD.encode(b),
        )),
        Value::EnumNumber(number) => match kind {
            Kind::Enum(enum_ty) => match enum_ty.get_value(*number) {
                // NullValue (used by google.protobuf.Value) serializes to null.
                _ if enum_ty.full_name() == "google.protobuf.NullValue" => JsonValue::Null,
                Some(v) => JsonValue::Str(quote_json_string(v.name())),
                None => JsonValue::Int(*number as i64),
            },
            _ => JsonValue::Int(*number as i64),
        },
        Value::Message(m) => message_to_json_value(m),
        Value::List(values) => {
            JsonValue::Array(values.iter().map(|v| value_to_json(v, kind)).collect())
        }
        Value::Map(map) => {
            let value_kind = match kind {
                Kind::Message(m) if m.is_map_entry() => m.map_entry_value_field().kind(),
                other => other.clone(),
            };
            // Sort by the JSON-string form of the key for deterministic output
            // (see module docs on the map-ordering deviation).
            let mut items: Vec<(String, &Value)> =
                map.iter().map(|(k, v)| (map_key_to_string(k), v)).collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            JsonValue::Object(
                items
                    .into_iter()
                    .map(|(k, v)| (k, value_to_json(v, &value_kind)))
                    .collect(),
            )
        }
    }
}

/// Render a protobuf map key to its JSON object-key string (always a string;
/// int64/bool keys are stringified, matching JSON and Python).
fn map_key_to_string(key: &MapKey) -> String {
    match key {
        MapKey::Bool(b) => b.to_string(),
        MapKey::I32(v) => v.to_string(),
        MapKey::I64(v) => v.to_string(),
        MapKey::U32(v) => v.to_string(),
        MapKey::U64(v) => v.to_string(),
        MapKey::String(s) => s.clone(),
    }
}

// ---------------------------------------------------------------------------
// Well-known type string formatting (matches protobuf's MessageToJson)
// ---------------------------------------------------------------------------

/// Format a `Timestamp` as RFC3339 with a `Z` suffix, matching
/// `Timestamp.ToJsonString()` (0/3/6/9 fractional digits).
fn format_timestamp(ts: &prost_types::Timestamp) -> String {
    // Normalize nanos into [0, 1e9) as protobuf does.
    let mut seconds = ts.seconds;
    let mut nanos = ts.nanos;
    if nanos < 0 {
        seconds -= 1;
        nanos += 1_000_000_000;
    }
    let datetime = chrono_like_from_unix(seconds);
    let frac = format_fraction(nanos as u32);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}Z",
        datetime.year,
        datetime.month,
        datetime.day,
        datetime.hour,
        datetime.minute,
        datetime.second,
        frac,
    )
}

/// Format a `Duration` like protobuf's `Duration.ToJsonString()` — a decimal
/// number of seconds with an `s` suffix, e.g. `"1.500s"`, `"3s"`, `"-0.010s"`.
fn format_duration(d: &prost_types::Duration) -> String {
    let seconds = d.seconds;
    let nanos = d.nanos;
    let negative = seconds < 0 || nanos < 0;
    let abs_seconds = seconds.unsigned_abs();
    let abs_nanos = nanos.unsigned_abs();
    let sign = if negative { "-" } else { "" };
    let frac = format_fraction(abs_nanos);
    format!("{sign}{abs_seconds}{frac}s")
}

/// Render a nanosecond fraction the way protobuf does: empty when zero, else a
/// leading `.` and 3/6/9 digits (trailing zero groups trimmed to the coarsest
/// non-zero precision).
fn format_fraction(nanos: u32) -> String {
    if nanos == 0 {
        return String::new();
    }
    if nanos % 1_000_000 == 0 {
        format!(".{:03}", nanos / 1_000_000)
    } else if nanos % 1_000 == 0 {
        format!(".{:06}", nanos / 1_000)
    } else {
        format!(".{nanos:09}")
    }
}

/// Minimal civil-time breakdown of a Unix timestamp (seconds), UTC. Uses the
/// standard days-from-civil algorithm; avoids pulling `chrono` into this crate.
struct CivilTime {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

fn chrono_like_from_unix(unix_seconds: i64) -> CivilTime {
    let days = unix_seconds.div_euclid(86_400);
    let secs_of_day = unix_seconds.rem_euclid(86_400);
    let hour = (secs_of_day / 3600) as u32;
    let minute = ((secs_of_day % 3600) / 60) as u32;
    let second = (secs_of_day % 60) as u32;

    // Howard Hinnant's civil_from_days algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    CivilTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
    }
}

// ---------------------------------------------------------------------------
// Pretty-printer matching Python json.dumps(indent=2)
// ---------------------------------------------------------------------------

/// Serialize our JSON tree with Python `json.dumps(..., indent=2)` formatting:
/// two-space indent per level, `": "` after keys, `","` (no trailing space)
/// between items on their own lines, `{}`/`[]` for empties, no trailing newline.
fn write_pretty(out: &mut String, value: &JsonValue, indent: usize) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Int(v) => out.push_str(&v.to_string()),
        JsonValue::Uint(v) => out.push_str(&v.to_string()),
        JsonValue::Double(v) => out.push_str(&python_float_repr(*v)),
        JsonValue::Float(v) => out.push_str(&python_float_repr(*v as f64)),
        JsonValue::Str(s) => out.push_str(s),
        JsonValue::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            let child_indent = indent + 1;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                push_indent(out, child_indent);
                write_pretty(out, item, child_indent);
            }
            out.push('\n');
            push_indent(out, indent);
            out.push(']');
        }
        JsonValue::Object(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            let child_indent = indent + 1;
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                push_indent(out, child_indent);
                out.push_str(&quote_json_string(key));
                out.push_str(": ");
                write_pretty(out, val, child_indent);
            }
            out.push('\n');
            push_indent(out, indent);
            out.push('}');
        }
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent * 2 {
        out.push(' ');
    }
}

/// Quote and escape a string exactly like Python `json.dumps` with the default
/// `ensure_ascii=True`: control chars use short escapes where defined, other
/// control chars and all non-ASCII use `\uXXXX` (astral scalar values become a
/// UTF-16 surrogate pair). Forward slash is NOT escaped.
///
/// Exposed (not just used internally) so hand-rolled JSON endpoints — e.g. the
/// ajax-only `get-history-bulk` (`handlers.py:2112`, a Flask `dict` return via
/// `jsonify`, not proto-serialized) — can reproduce Python's exact string
/// escaping without duplicating this logic.
pub fn quote_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if c.is_ascii() => out.push(c),
            c => {
                // Non-ASCII: emit \uXXXX, using surrogate pairs for astral chars.
                let cp = c as u32;
                if cp > 0xFFFF {
                    let v = cp - 0x10000;
                    let high = 0xD800 + (v >> 10);
                    let low = 0xDC00 + (v & 0x3FF);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                } else {
                    out.push_str(&format!("\\u{cp:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

/// Format an `f64` exactly like CPython's `repr()` / `json.dumps` (shortest
/// round-trip, `1e+20`/`1.0`/`-0.0`/`Infinity` forms).
///
/// Exposed for the same reason as [`quote_json_string`]: hand-rolled JSON
/// endpoints need Python's `allow_nan=True` float formatting (`NaN`/
/// `Infinity`/`-Infinity` literals), which `serde_json` does not reproduce
/// (it silently maps non-finite floats to `null`).
///
/// NB: Rust's shortest-float algorithm (ryu/grisu) and CPython's dtoa can pick a
/// different *final digit* in rare tie-break cases (~0.01% of arbitrary bit
/// patterns); both still round-trip to the same `f64`. Real metric values are
/// exceedingly unlikely to hit this. Documented as a wire risk in the plan.
pub fn python_float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }

    let neg = x < 0.0;
    let ax = x.abs();
    // Rust's `{:e}` gives the shortest round-trip form "d[.ddd]e{exp}".
    let sci = format!("{ax:e}");
    let (mant, exp_str) = sci.split_once('e').expect("scientific form has 'e'");
    let exp: i32 = exp_str.parse().expect("valid exponent");
    let (int_part, frac_part) = match mant.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mant, ""),
    };
    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(int_part);
    digits.push_str(frac_part);
    let ndigits = digits.len() as i32;

    // CPython 'r' format switches to exponential when exp < -4 or exp >= 16.
    let use_exp = !(-4..16).contains(&exp);
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if use_exp {
        out.push_str(&digits[..1]);
        if ndigits > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp >= 0 { '+' } else { '-' });
        let ae = exp.unsigned_abs();
        if ae < 10 {
            out.push('0');
        }
        out.push_str(&ae.to_string());
    } else if exp >= 0 {
        let intdigits = (exp + 1) as usize;
        if (ndigits as usize) <= intdigits {
            out.push_str(&digits);
            for _ in 0..(intdigits - ndigits as usize) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..intdigits]);
            out.push('.');
            out.push_str(&digits[intdigits..]);
        }
    } else {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        out.push_str(&digits);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python_examples() {
        // Values cross-checked against Python `json.dumps` in gen_goldens probes.
        let cases = [
            (1.0_f64, "1.0"),
            (0.1, "0.1"),
            (1e20, "1e+20"),
            (1e-7, "1e-07"),
            (1e-5, "1e-05"),
            (1e16, "1e+16"),
            (1e17, "1e+17"),
            (123456789012345.6, "123456789012345.6"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (2.71892818284591, "2.71892818284591"),
            (1e100, "1e+100"),
            (1e-300, "1e-300"),
            (2.5, "2.5"),
            (100.0, "100.0"),
            (0.0001, "0.0001"),
            (1234567.0, "1234567.0"),
        ];
        for (v, expected) in cases {
            assert_eq!(python_float_repr(v), expected, "float {v}");
        }
        assert_eq!(python_float_repr(f64::INFINITY), "Infinity");
        assert_eq!(python_float_repr(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn string_escaping_matches_python_ensure_ascii() {
        // héllo 世界 😀 with quotes/backslash/control chars.
        let input = "h\u{e9}llo \u{4e16}\u{754c} \u{1f600} \"q\" \\b / \n\t";
        assert_eq!(
            quote_json_string(input),
            "\"h\\u00e9llo \\u4e16\\u754c \\ud83d\\ude00 \\\"q\\\" \\\\b / \\n\\t\""
        );
    }

    #[test]
    fn duration_and_timestamp_formatting() {
        assert_eq!(
            format_duration(&prost_types::Duration {
                seconds: 1,
                nanos: 500_000_000
            }),
            "1.500s"
        );
        assert_eq!(
            format_duration(&prost_types::Duration {
                seconds: 3,
                nanos: 0
            }),
            "3s"
        );
        // 1700000000123 ms => 2023-11-14T22:13:20.123Z
        assert_eq!(
            format_timestamp(&prost_types::Timestamp {
                seconds: 1_700_000_000,
                nanos: 123_000_000
            }),
            "2023-11-14T22:13:20.123Z"
        );
    }
}
