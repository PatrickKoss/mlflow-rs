//! OTLP JSON body parsing (`/v1/traces` with `Content-Type: application/json`).
//!
//! Mirrors `mlflow/server/otel_api.py:69-88, 148-154`: OTLP/JSON encodes
//! `trace_id`/`span_id`/`parent_span_id` as **lowercase hex strings**
//! (<https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding>), but the
//! canonical protobuf JSON mapping Python's `google.protobuf.json_format.Parse`
//! implements expects **base64** for `bytes` fields (proto3 spec). Python
//! papers over this with [`_convert_otlp_json_ids_to_base64`] before calling
//! `Parse(..., ignore_unknown_fields=True)`.
//!
//! Rather than pull in a full protobuf-JSON library, this module hand-rolls a
//! `serde_json::Value` → [`ExportTraceServiceRequest`] decoder for the small,
//! fixed OTLP trace schema (`ResourceSpans`/`ScopeSpans`/`Span`/`KeyValue`/
//! `AnyValue`/`Status`/`Event`/`Link`). Both `camelCase` (the protobuf-JSON
//! default) and `snake_case` field names are accepted, matching
//! `ignore_unknown_fields=True` + protobuf-JSON's documented tolerance for
//! either casing on parse. Unknown fields are ignored, matching Python.
//!
//! Hex trace/span IDs are decoded directly to bytes (no intermediate
//! base64 round-trip needed in Rust, since we control the decode ourselves).

use base64::Engine;
use mlflow_proto::opentelemetry::proto::collector::trace::v1::ExportTraceServiceRequest;
use mlflow_proto::opentelemetry::proto::common::v1::{
    any_value, AnyValue, ArrayValue, InstrumentationScope, KeyValue, KeyValueList,
};
use mlflow_proto::opentelemetry::proto::resource::v1::Resource;
use mlflow_proto::opentelemetry::proto::trace::v1::{
    span, ResourceSpans, ScopeSpans, Span, Status,
};
use serde_json::Value;

/// Parse error for OTLP JSON bodies. Every variant maps to Python's blanket
/// `"Invalid OpenTelemetry format"` 400 (`otel_api.py:166-170`); the message is
/// kept here only for diagnostics/tests, not surfaced to clients.
#[derive(Debug, thiserror::Error)]
pub enum OtlpJsonError {
    #[error("malformed JSON: {0}")]
    Malformed(#[from] serde_json::Error),
    #[error("invalid id encoding for field '{0}': {1}")]
    InvalidId(&'static str, String),
    #[error("expected a JSON object for {0}")]
    NotAnObject(&'static str),
}

/// Parse an OTLP/JSON request body into an [`ExportTraceServiceRequest`].
pub fn parse_otlp_json(body: &[u8]) -> Result<ExportTraceServiceRequest, OtlpJsonError> {
    let value: Value = serde_json::from_slice(body)?;
    let obj = value
        .as_object()
        .ok_or(OtlpJsonError::NotAnObject("ExportTraceServiceRequest"))?;
    let resource_spans = match get(obj, &["resourceSpans", "resource_spans"]) {
        Some(Value::Array(items)) => items
            .iter()
            .map(parse_resource_spans)
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    Ok(ExportTraceServiceRequest { resource_spans })
}

/// Look up a key trying multiple aliases (camelCase then snake_case), matching
/// protobuf-JSON's tolerance for either casing on parse.
fn get<'a>(obj: &'a serde_json::Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| obj.get(*k))
}

fn as_obj<'a>(
    v: &'a Value,
    ctx: &'static str,
) -> Result<&'a serde_json::Map<String, Value>, OtlpJsonError> {
    v.as_object().ok_or(OtlpJsonError::NotAnObject(ctx))
}

fn parse_resource_spans(v: &Value) -> Result<ResourceSpans, OtlpJsonError> {
    let obj = as_obj(v, "ResourceSpans")?;
    let resource = match get(obj, &["resource"]) {
        Some(r) => Some(parse_resource(r)?),
        None => None,
    };
    let scope_spans = match get(obj, &["scopeSpans", "scope_spans"]) {
        Some(Value::Array(items)) => items
            .iter()
            .map(parse_scope_spans)
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    let schema_url = string_field(obj, &["schemaUrl", "schema_url"]);
    Ok(ResourceSpans {
        resource,
        scope_spans,
        schema_url,
    })
}

fn parse_resource(v: &Value) -> Result<Resource, OtlpJsonError> {
    let obj = as_obj(v, "Resource")?;
    let attributes = parse_key_values(get(obj, &["attributes"]))?;
    Ok(Resource {
        attributes,
        dropped_attributes_count: u32_field(
            obj,
            &["droppedAttributesCount", "dropped_attributes_count"],
        ),
        entity_refs: Vec::new(),
    })
}

fn parse_scope_spans(v: &Value) -> Result<ScopeSpans, OtlpJsonError> {
    let obj = as_obj(v, "ScopeSpans")?;
    let scope = match get(obj, &["scope"]) {
        Some(s) => Some(parse_instrumentation_scope(s)?),
        None => None,
    };
    let spans = match get(obj, &["spans"]) {
        Some(Value::Array(items)) => items
            .iter()
            .map(parse_span)
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    let schema_url = string_field(obj, &["schemaUrl", "schema_url"]);
    Ok(ScopeSpans {
        scope,
        spans,
        schema_url,
    })
}

fn parse_instrumentation_scope(v: &Value) -> Result<InstrumentationScope, OtlpJsonError> {
    let obj = as_obj(v, "InstrumentationScope")?;
    Ok(InstrumentationScope {
        name: string_field(obj, &["name"]),
        version: string_field(obj, &["version"]),
        attributes: parse_key_values(get(obj, &["attributes"]))?,
        dropped_attributes_count: u32_field(
            obj,
            &["droppedAttributesCount", "dropped_attributes_count"],
        ),
    })
}

fn parse_span(v: &Value) -> Result<Span, OtlpJsonError> {
    let obj = as_obj(v, "Span")?;
    let trace_id = hex_id_field(obj, &["traceId", "trace_id"], "traceId")?;
    let span_id = hex_id_field(obj, &["spanId", "span_id"], "spanId")?;
    let parent_span_id = hex_id_field(obj, &["parentSpanId", "parent_span_id"], "parentSpanId")?;

    let status = match get(obj, &["status"]) {
        Some(s) => Some(parse_status(s)?),
        None => None,
    };
    let events = match get(obj, &["events"]) {
        Some(Value::Array(items)) => items
            .iter()
            .map(parse_event)
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    let links = match get(obj, &["links"]) {
        Some(Value::Array(items)) => items
            .iter()
            .map(parse_link)
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };

    Ok(Span {
        trace_id,
        span_id,
        trace_state: string_field(obj, &["traceState", "trace_state"]),
        parent_span_id,
        flags: u32_field(obj, &["flags"]),
        name: string_field(obj, &["name"]),
        kind: i32_field(obj, &["kind"]),
        start_time_unix_nano: u64_field(obj, &["startTimeUnixNano", "start_time_unix_nano"]),
        end_time_unix_nano: u64_field(obj, &["endTimeUnixNano", "end_time_unix_nano"]),
        attributes: parse_key_values(get(obj, &["attributes"]))?,
        dropped_attributes_count: u32_field(
            obj,
            &["droppedAttributesCount", "dropped_attributes_count"],
        ),
        events,
        dropped_events_count: u32_field(obj, &["droppedEventsCount", "dropped_events_count"]),
        links,
        dropped_links_count: u32_field(obj, &["droppedLinksCount", "dropped_links_count"]),
        status,
    })
}

fn parse_status(v: &Value) -> Result<Status, OtlpJsonError> {
    let obj = as_obj(v, "Status")?;
    let code = match get(obj, &["code"]) {
        Some(Value::String(s)) => match s.as_str() {
            "STATUS_CODE_OK" => 1,
            "STATUS_CODE_ERROR" => 2,
            _ => 0,
        },
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0) as i32,
        _ => 0,
    };
    Ok(Status {
        message: string_field(obj, &["message"]),
        code,
    })
}

fn parse_event(v: &Value) -> Result<span::Event, OtlpJsonError> {
    let obj = as_obj(v, "Span.Event")?;
    Ok(span::Event {
        time_unix_nano: u64_field(obj, &["timeUnixNano", "time_unix_nano"]),
        name: string_field(obj, &["name"]),
        attributes: parse_key_values(get(obj, &["attributes"]))?,
        dropped_attributes_count: u32_field(
            obj,
            &["droppedAttributesCount", "dropped_attributes_count"],
        ),
    })
}

fn parse_link(v: &Value) -> Result<span::Link, OtlpJsonError> {
    let obj = as_obj(v, "Span.Link")?;
    let trace_id = hex_id_field(obj, &["traceId", "trace_id"], "traceId")?;
    let span_id = hex_id_field(obj, &["spanId", "span_id"], "spanId")?;
    Ok(span::Link {
        trace_id,
        span_id,
        trace_state: string_field(obj, &["traceState", "trace_state"]),
        attributes: parse_key_values(get(obj, &["attributes"]))?,
        dropped_attributes_count: u32_field(
            obj,
            &["droppedAttributesCount", "dropped_attributes_count"],
        ),
        flags: u32_field(obj, &["flags"]),
    })
}

fn parse_key_values(v: Option<&Value>) -> Result<Vec<KeyValue>, OtlpJsonError> {
    let Some(Value::Array(items)) = v else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            let obj = as_obj(item, "KeyValue")?;
            let key = string_field(obj, &["key"]);
            let value = match get(obj, &["value"]) {
                Some(v) => Some(parse_any_value(v)?),
                None => None,
            };
            Ok(KeyValue { key, value })
        })
        .collect()
}

fn parse_any_value(v: &Value) -> Result<AnyValue, OtlpJsonError> {
    let obj = as_obj(v, "AnyValue")?;
    let value = if let Some(Value::String(s)) = get(obj, &["stringValue", "string_value"]) {
        Some(any_value::Value::StringValue(s.clone()))
    } else if let Some(Value::Bool(b)) = get(obj, &["boolValue", "bool_value"]) {
        Some(any_value::Value::BoolValue(*b))
    } else if let Some(v) = get(obj, &["intValue", "int_value"]) {
        Some(any_value::Value::IntValue(json_number_to_i64(v)))
    } else if let Some(v) = get(obj, &["doubleValue", "double_value"]) {
        Some(any_value::Value::DoubleValue(v.as_f64().unwrap_or(0.0)))
    } else if let Some(Value::Array(items)) =
        get(obj, &["arrayValue", "array_value"]).and_then(|a| {
            as_obj(a, "ArrayValue")
                .ok()
                .and_then(|o| get(o, &["values"]))
        })
    {
        let values = items
            .iter()
            .map(parse_any_value)
            .collect::<Result<Vec<_>, _>>()?;
        Some(any_value::Value::ArrayValue(ArrayValue { values }))
    } else if let Some(kv) = get(obj, &["kvlistValue", "kvlist_value"]) {
        let kv_obj = as_obj(kv, "KeyValueList")?;
        let values = parse_key_values(get(kv_obj, &["values"]))?;
        Some(any_value::Value::KvlistValue(KeyValueList { values }))
    } else if let Some(Value::String(s)) = get(obj, &["bytesValue", "bytes_value"]) {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| OtlpJsonError::InvalidId("bytesValue", e.to_string()))?;
        Some(any_value::Value::BytesValue(decoded))
    } else {
        None
    };
    Ok(AnyValue { value })
}

/// Decode a hex-encoded OTLP/JSON id field (`traceId`/`spanId`/`parentSpanId`)
/// into raw bytes. An absent/empty field decodes to an empty `Vec` (matching
/// protobuf-JSON's empty-string-for-unset-bytes convention).
fn hex_id_field(
    obj: &serde_json::Map<String, Value>,
    keys: &[&str],
    field: &'static str,
) -> Result<Vec<u8>, OtlpJsonError> {
    match get(obj, keys) {
        Some(Value::String(s)) if !s.is_empty() => {
            hex::decode(s).map_err(|e| OtlpJsonError::InvalidId(field, e.to_string()))
        }
        _ => Ok(Vec::new()),
    }
}

fn string_field(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> String {
    match get(obj, keys) {
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

fn u32_field(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> u32 {
    match get(obj, keys) {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as u32,
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

fn i32_field(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> i32 {
    match get(obj, keys) {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0) as i32,
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

fn u64_field(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> u64 {
    match get(obj, keys) {
        // OTLP/JSON encodes fixed64/uint64 as strings to avoid JS precision
        // loss, but also accept a bare number for lenient/hand-built payloads.
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0),
        _ => 0,
    }
}

fn json_number_to_i64(v: &Value) -> i64 {
    match v {
        Value::String(s) => s.parse().unwrap_or(0),
        Value::Number(n) => n.as_i64().unwrap_or(0),
        _ => 0,
    }
}

/// Minimal hex decode (avoids pulling in a `hex` crate dependency for four call
/// sites).
mod hex {
    pub fn decode(s: &str) -> Result<Vec<u8>, String> {
        if !s.len().is_multiple_of(2) {
            return Err(format!("odd-length hex string: {s:?}"));
        }
        let mut out = Vec::with_capacity(s.len() / 2);
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let hi = nibble(bytes[i])?;
            let lo = nibble(bytes[i + 1])?;
            out.push((hi << 4) | lo);
            i += 2;
        }
        Ok(out)
    }

    fn nibble(b: u8) -> Result<u8, String> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(format!("invalid hex digit: {}", b as char)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_export_request() {
        let body = br#"{
            "resourceSpans": [{
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "claude-code"}}]},
                "scopeSpans": [{
                    "spans": [{
                        "traceId": "0102030405060708090a0b0c0d0e0f10",
                        "spanId": "0102030405060708",
                        "name": "root",
                        "startTimeUnixNano": "1000000000",
                        "endTimeUnixNano": "2000000000",
                        "status": {"code": "STATUS_CODE_OK"}
                    }]
                }]
            }]
        }"#;
        let req = parse_otlp_json(body).unwrap();
        assert_eq!(req.resource_spans.len(), 1);
        let span = &req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(
            span.trace_id,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(span.span_id, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(span.start_time_unix_nano, 1_000_000_000);
        assert_eq!(span.end_time_unix_nano, 2_000_000_000);
        assert_eq!(span.status.as_ref().unwrap().code, 1);
    }

    #[test]
    fn empty_resource_spans_is_ok() {
        let req = parse_otlp_json(b"{}").unwrap();
        assert!(req.resource_spans.is_empty());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_otlp_json(b"not json").is_err());
    }

    #[test]
    fn accepts_snake_case_field_names() {
        let body = br#"{
            "resource_spans": [{
                "scope_spans": [{
                    "spans": [{
                        "trace_id": "0102030405060708090a0b0c0d0e0f10",
                        "span_id": "0102030405060708",
                        "name": "root",
                        "start_time_unix_nano": "1000000000"
                    }]
                }]
            }]
        }"#;
        let req = parse_otlp_json(body).unwrap();
        assert_eq!(req.resource_spans[0].scope_spans[0].spans[0].name, "root");
    }
}
