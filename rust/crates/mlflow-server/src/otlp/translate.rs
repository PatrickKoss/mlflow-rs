//! OTLP proto span → MLflow [`SpanInput`] row translation (plan T4.3, §3.8).
//!
//! This is the "OTLP→row translation" the T2.11 store layer explicitly
//! deferred to Phase 3 (`mlflow-store/src/store/spans.rs` module docs). It
//! mirrors:
//!
//! * `Span.from_otel_proto` (`mlflow/entities/span.py:458-560`) — builds the
//!   MLflow span's identity (`trace_id`/`span_id`/`parent_id` derived from the
//!   OTel ids), status, and attributes (every attribute value individually
//!   JSON-dumped via `dump_span_attribute_value`, matching OTel's
//!   `set_attribute` behavior).
//! * `Span.to_dict()` (`span.py:306-337`) — the span's `content` JSON blob:
//!   `trace_id`/`span_id`/`parent_span_id` inside this blob are
//!   **base64-encoded big-endian bytes** (16/8 bytes respectively), NOT the
//!   `"tr-<hex>"` string used as the DB row's `trace_id` column/FK; `status`
//!   is `{"code": "STATUS_CODE_OK"|"STATUS_CODE_ERROR"|"STATUS_CODE_UNSET",
//!   "message": ...}` (the OTel proto enum *name*, distinct from the plain
//!   `"OK"/"ERROR"/"UNSET"` written to the `spans.status` DB column).
//! * `translate_span_when_storing`'s `sanitize_attributes` step
//!   (`mlflow/tracing/otel/translation/__init__.py:56-74, 461-486`) — undoes
//!   the double-JSON-encoding that results from `from_otel_proto` dumping each
//!   attribute value and `to_dict()` then embedding that already-JSON string
//!   into the span dict, which gets JSON-encoded *again* at the top level.
//! * `_get_trace_status_from_root_span` (`sqlalchemy_store.py:5491-5508`) and
//!   the per-trace time-range aggregate (`sqlalchemy_store.py:5007-5011`) —
//!   [`TraceTimeRange`] (already defined by the T2.11 store layer; this module
//!   only computes it).
//!
//! ## Deviation: OTEL-schema attribute translators are out of scope
//!
//! Python's `translate_span_when_storing` (beyond `sanitize_attributes`) also
//! runs a battery of per-vendor OTEL-schema translators (OpenInference,
//! Traceloop, GenAI semconv, Google ADK, Vercel AI, LiveKit, Laminar,
//! Langfuse, Spring AI, VoltAgent, Gemini CLI —
//! `mlflow/tracing/otel/translation/*.py`) that *infer* MLflow-specific
//! attributes (`mlflow.spanInputs`/`mlflow.spanOutputs`/`mlflow.spanType`,
//! token usage, model name/provider, and LLM cost) from vendor-specific
//! attribute conventions when the client didn't set the MLflow attribute
//! directly. That inference is a large, independent subsystem (11 translator
//! modules, a cost-rate table, chat-format normalization) with no wire-format
//! or store-schema implications — it only affects *which values are already
//! present* in `attributes` before serialization. It is not part of the
//! `/v1/traces` HTTP contract (status codes, headers, persistence semantics)
//! this task covers, and, like the token-usage/cost/session/user-id
//! aggregation the T2.11 store layer deferred, is left for a follow-up phase.
//! Spans translated by this module still persist correctly and are queryable;
//! they just won't have vendor-inferred `mlflow.span*`/model/cost attributes
//! auto-filled in when the client used a non-MLflow OTEL convention.

use std::collections::BTreeMap;

use base64::Engine;
use mlflow_proto::opentelemetry::proto::collector::trace::v1::ExportTraceServiceRequest;
#[cfg(test)]
use mlflow_proto::opentelemetry::proto::common::v1::KeyValue;
use mlflow_proto::opentelemetry::proto::common::v1::{any_value, AnyValue};
use mlflow_proto::opentelemetry::proto::resource::v1::Resource;
use mlflow_proto::opentelemetry::proto::trace::v1::{span, Span as OtelSpan};
use mlflow_store::{SpanInput, SpanMetricInput, TraceTimeRange};
use serde_json::{Map, Value};

/// `SpanAttributeKey.REQUEST_ID` (`mlflow/tracing/constant.py:100`).
const ATTR_REQUEST_ID: &str = "mlflow.traceRequestId";
/// `SpanAttributeKey.MODEL` / `MODEL_PROVIDER` (`constant.py:118-119`) — read
/// into `spans.dimension_attributes`, matching `sqlalchemy_store.py:5047-5054`.
const ATTR_MODEL: &str = "mlflow.llm.model";
const ATTR_MODEL_PROVIDER: &str = "mlflow.llm.provider";
/// `SpanAttributeKey.LLM_COST` (`constant.py:116`) — read into `span_metrics`.
const ATTR_LLM_COST: &str = "mlflow.llm.cost";
/// `TRACE_REQUEST_ID_PREFIX` (`mlflow/tracing/constant.py:184`).
const TRACE_REQUEST_ID_PREFIX: &str = "tr-";

/// One fully-translated OTel span, ready to become a [`SpanInput`] /
/// [`SpanMetricInput`] plus contribute to its trace's [`TraceTimeRange`].
pub struct TranslatedSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: Option<String>,
    /// DB `spans.status` value: `"OK"|"ERROR"|"UNSET"`.
    pub status: &'static str,
    pub start_time_unix_nano: i64,
    pub end_time_unix_nano: Option<i64>,
    pub content: String,
    pub dimension_attributes: Option<String>,
    pub metrics: Vec<SpanMetricInput>,
    pub is_root: bool,
}

/// Error translating a single OTel proto span — mirrors the `except Exception`
/// catch-all in `otel_api.py:204-208` that maps ANY span-conversion failure to
/// a 422. We only need the fact that it failed, not why (Python discards the
/// underlying exception in the response too), but keep a message for tests.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct SpanConversionError(pub String);

/// Translate every span across every `resource_spans`/`scope_spans` entry.
/// Returns spans in encounter order (matching Python's nested-loop iteration),
/// plus the set of `service.name` resource attribute values seen (restricted
/// to the known-CLI allowlist, mirroring `_KNOWN_SERVICE_NAMES`).
pub fn translate_request(
    request: &ExportTraceServiceRequest,
) -> Result<(Vec<TranslatedSpan>, Vec<String>), SpanConversionError> {
    let mut spans = Vec::new();
    let mut service_names: Vec<String> = Vec::new();

    for resource_span in &request.resource_spans {
        let resource_service_name = resource_span
            .resource
            .as_ref()
            .and_then(resource_service_name);
        if let Some(name) = &resource_service_name {
            if !service_names.contains(name) {
                service_names.push(name.clone());
            }
        }

        for scope_span in &resource_span.scope_spans {
            for otel_span in &scope_span.spans {
                // Python's message never includes the underlying exception
                // text (`otel_api.py:204-208`: bare `except Exception:` with a
                // fixed `detail`) — match that exactly for byte-parity. The
                // `SpanConversionError`'s own `Display` (used by tests/logs)
                // still carries `e` for diagnostics.
                let translated = translate_span(
                    otel_span,
                    resource_span.resource.as_ref(),
                    resource_service_name.as_deref(),
                )
                .map_err(|_| {
                    SpanConversionError(
                        "Cannot convert OpenTelemetry span to MLflow span".to_string(),
                    )
                })?;
                spans.push(translated);
            }
        }
    }

    Ok((spans, service_names))
}

/// `_KNOWN_SERVICE_NAMES` (`mlflow/server/otel_api.py:50-61`).
const KNOWN_SERVICE_NAMES: &[&str] = &[
    "claude-code",
    "codex_cli_rs",
    "codex_vscode",
    "gemini-cli",
    "qwen-code",
];

/// Extract `resource.attributes["service.name"]`, restricted to the allowlist
/// (`otel_api.py:176-184`).
fn resource_service_name(resource: &Resource) -> Option<String> {
    resource.attributes.iter().find_map(|attr| {
        if attr.key != "service.name" {
            return None;
        }
        let value = decode_any_value(attr.value.as_ref())?;
        let s = match &value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        KNOWN_SERVICE_NAMES.contains(&s.as_str()).then_some(s)
    })
}

/// `Span.from_otel_proto` + `Span.to_dict()` + `translate_span_when_storing`'s
/// `sanitize_attributes` step, fused into one row-building pass.
fn translate_span(
    otel_span: &OtelSpan,
    resource: Option<&Resource>,
    resource_service_name: Option<&str>,
) -> Result<TranslatedSpan, String> {
    if otel_span.trace_id.is_empty() {
        return Err("trace_id is required but was empty".to_string());
    }
    if otel_span.span_id.is_empty() {
        return Err("span_id is required but was empty".to_string());
    }

    let trace_id_hex = hex_lower(&otel_span.trace_id);
    let span_id_hex = hex_lower(&otel_span.span_id);
    let mlflow_trace_id = format!("{TRACE_REQUEST_ID_PREFIX}{trace_id_hex}");
    let parent_span_id_hex =
        (!otel_span.parent_span_id.is_empty()).then(|| hex_lower(&otel_span.parent_span_id));

    // `status.code` proto enum -> SpanStatusCode DB value (span.py:488-493):
    // OK/ERROR map directly, everything else (including absent `status`) is
    // UNSET. `from_otel_proto` does NOT go through `SpanStatus::from_otel_proto_status`.
    let status_proto_code = otel_span.status.as_ref().map(|s| s.code).unwrap_or(0);
    let db_status = match status_proto_code {
        1 => "OK",
        2 => "ERROR",
        _ => "UNSET",
    };
    let otel_proto_status_name = match status_proto_code {
        1 => "STATUS_CODE_OK",
        2 => "STATUS_CODE_ERROR",
        _ => "STATUS_CODE_UNSET",
    };
    let status_message = otel_span
        .status
        .as_ref()
        .map(|s| s.message.clone())
        .unwrap_or_default();

    // `serialized_attributes` (span.py:495-498): every OTel attribute value,
    // JSON-dumped individually via `dump_span_attribute_value`.
    let mut serialized_attributes: BTreeMap<String, String> = BTreeMap::new();
    for kv in &otel_span.attributes {
        let value = decode_any_value(kv.value.as_ref()).unwrap_or(Value::Null);
        serialized_attributes.insert(kv.key.clone(), dump_span_attribute_value(&value));
    }
    // `preserve_request_id=False` (server ingest never trusts a client-sent
    // request id): always overwrite with the server-derived trace id
    // (span.py:499-515).
    serialized_attributes.insert(
        ATTR_REQUEST_ID.to_string(),
        dump_span_attribute_value(&Value::String(mlflow_trace_id.clone())),
    );

    // `sanitize_attributes` (translation/__init__.py:461-486): strip one layer
    // of double-encoding introduced by dumping an already-JSON-string value.
    let attributes = sanitize_attributes(serialized_attributes);

    let name = (!otel_span.name.is_empty()).then(|| otel_span.name.clone());
    let start_time_unix_nano = otel_span.start_time_unix_nano as i64;
    let end_time_unix_nano =
        (otel_span.end_time_unix_nano != 0).then_some(otel_span.end_time_unix_nano as i64);

    // `Span.to_dict()` (span.py:306-337): base64(big-endian bytes) ids,
    // OTel-proto-enum-name status, raw (un-dumped) event attributes, resource
    // ignored (resource is carried only via the `service.name` root-span
    // propagation below, matching Python — `to_dict()` never serializes the
    // OTel SDK `Resource` at all).
    let content = build_content_json(
        &otel_span.trace_id,
        &otel_span.span_id,
        (!otel_span.parent_span_id.is_empty()).then_some(&otel_span.parent_span_id),
        &name,
        start_time_unix_nano,
        otel_span.end_time_unix_nano,
        otel_proto_status_name,
        &status_message,
        &otel_span.events,
        &otel_span.links,
        &attributes,
    );

    // Root-span service.name propagation (`otel_api.py:196-201`): only applied
    // to root spans, and only affects the DB-visible attribute (which we've
    // already captured in `content`/`attributes` above via a second pass,
    // since it must land in the *stored* attributes dict, not just resource).
    let is_root = parent_span_id_hex.is_none();
    let content = if is_root {
        if let Some(service_name) = resource_service_name {
            inject_service_name_attribute(&content, service_name)
        } else {
            content
        }
    } else {
        content
    };
    let _ = resource; // Resource is consulted only for `service.name` above.

    // `dimension_attributes` (sqlalchemy_store.py:5046-5054): MODEL / MODEL_PROVIDER,
    // each unwrapped one JSON layer via `_try_parse_json_string`.
    let mut dimension_attributes = Map::new();
    for key in [ATTR_MODEL, ATTR_MODEL_PROVIDER] {
        if let Some(raw) = attributes.get(key) {
            dimension_attributes.insert(key.to_string(), try_parse_json_string(raw));
        }
    }
    let dimension_attributes =
        (!dimension_attributes.is_empty()).then(|| Value::Object(dimension_attributes).to_string());

    // `span_metrics` from LLM cost (sqlalchemy_store.py:5071-5079): the cost
    // attribute is a JSON object of cost-component -> float.
    let mut metrics = Vec::new();
    if let Some(raw_cost) = attributes.get(ATTR_LLM_COST) {
        if let Ok(Value::Object(cost_obj)) = serde_json::from_str::<Value>(raw_cost) {
            for (cost_key, cost_value) in cost_obj {
                if let Some(f) = cost_value.as_f64() {
                    metrics.push(SpanMetricInput {
                        trace_id: mlflow_trace_id.clone(),
                        span_id: span_id_hex.clone(),
                        key: cost_key,
                        value: f,
                    });
                }
            }
        }
    }

    Ok(TranslatedSpan {
        trace_id: mlflow_trace_id,
        span_id: span_id_hex,
        parent_span_id: parent_span_id_hex,
        name,
        status: db_status,
        start_time_unix_nano,
        end_time_unix_nano,
        content,
        dimension_attributes,
        metrics,
        is_root,
    })
}

/// `_try_parse_json_string` (`sqlalchemy_store.py:9751-9756`): unwrap one JSON
/// layer only if the parsed result is itself a string; otherwise keep the raw
/// text.
fn try_parse_json_string(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::String(s)) => Value::String(s),
        _ => Value::String(raw.to_string()),
    }
}

/// `dump_span_attribute_value` (`mlflow/tracing/utils/__init__.py:125-142`).
/// OTLP `AnyValue`s decode only to JSON-primitive-compatible types (string,
/// bool, int, float, bytes, array, object), so the `TraceJSONEncoder` custom
/// hooks (pydantic/dataclass/unsafe-`__str__`) never trigger for this path —
/// standard `serde_json` serialization is byte-equivalent. The one exception
/// is raw `bytes` (OTLP `bytesValue`), which Python's encoder falls through to
/// `str(obj)` for (a bytes `repr()`); see [`format_bytes_repr`].
fn dump_span_attribute_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Object(map) if map.contains_key("__mlflow_bytes__") => {
            let bytes = map["__mlflow_bytes__"].as_str().unwrap_or_default();
            serde_json::to_string(&format_bytes_repr(bytes)).expect("string always serializes")
        }
        other => serde_json::to_string(other).expect("JSON value always serializes"),
    }
}

/// Python `str(b'...')` repr for a bytes value, used as the `dump_span_attribute_value`
/// fallback for OTLP `bytesValue` attributes (raw bytes are not natively
/// JSON-serializable, so `TraceJSONEncoder` falls back to `str(obj)` — see
/// module docs). `bytes_b64` is the value's standard-base64 encoding (how we
/// carry raw bytes through this function's `Value` representation).
fn format_bytes_repr(bytes_b64: &str) -> String {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(bytes_b64)
        .unwrap_or_default();
    let mut out = String::from("b'");
    for byte in raw {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push('\'');
    out
}

/// Decode an OTel proto `AnyValue` to a `serde_json::Value`
/// (`_decode_otel_proto_anyvalue`, `mlflow/tracing/utils/otlp.py:216-236`).
/// Raw bytes are represented as `{"__mlflow_bytes__": "<base64>"}`, an internal
/// sentinel [`dump_span_attribute_value`] recognizes — never serialized as-is.
fn decode_any_value(value: Option<&AnyValue>) -> Option<Value> {
    let value = value?.value.as_ref()?;
    Some(match value {
        any_value::Value::StringValue(s) => Value::String(s.clone()),
        any_value::Value::BoolValue(b) => Value::Bool(*b),
        any_value::Value::IntValue(i) => Value::Number((*i).into()),
        any_value::Value::DoubleValue(d) => serde_json::Number::from_f64(*d)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        any_value::Value::BytesValue(b) => {
            let mut m = Map::new();
            m.insert(
                "__mlflow_bytes__".to_string(),
                Value::String(base64::engine::general_purpose::STANDARD.encode(b)),
            );
            Value::Object(m)
        }
        any_value::Value::ArrayValue(arr) => Value::Array(
            arr.values
                .iter()
                .map(|v| decode_any_value(Some(v)).unwrap_or(Value::Null))
                .collect(),
        ),
        any_value::Value::KvlistValue(kv) => {
            let mut m = Map::new();
            for entry in &kv.values {
                m.insert(
                    entry.key.clone(),
                    decode_any_value(entry.value.as_ref()).unwrap_or(Value::Null),
                );
            }
            Value::Object(m)
        }
    })
}

/// `sanitize_attributes` (`mlflow/tracing/otel/translation/__init__.py:461-486`):
/// every attribute value here is already a JSON-encoded string (from
/// `dump_span_attribute_value` above). If that string, when parsed, is ITSELF
/// a JSON string that parses again into a str/dict/list, strip one encoding
/// layer (double-encoding artifact of OTLP attribute values that were already
/// JSON text before this server dumped them again). Primitives (int/bool) are
/// deliberately excluded to avoid e.g. misreading `"1"` as an int.
fn sanitize_attributes(attributes: BTreeMap<String, String>) -> BTreeMap<String, String> {
    attributes
        .into_iter()
        .map(|(key, value)| {
            let Ok(Value::String(once)) = serde_json::from_str::<Value>(&value) else {
                return (key, value);
            };
            match serde_json::from_str::<Value>(&once) {
                Ok(v @ (Value::String(_) | Value::Object(_) | Value::Array(_))) => {
                    (key, serde_json::to_string(&v).unwrap_or(once))
                }
                _ => (key, value),
            }
        })
        .collect()
}

/// Inject/overwrite `service.name` into an already-built `content` JSON blob's
/// `attributes` map (`otel_api.py:198-201`: `dump_span_attribute_value(name)`
/// on the *serialized* attributes, applied only to root spans after
/// `sanitize_attributes` in Python's actual flow — this mirrors doing it as a
/// post-processing step for simplicity, which is observably identical since
/// JSON object key insertion order does not affect either side's stored
/// value).
fn inject_service_name_attribute(content: &str, service_name: &str) -> String {
    let mut value: Value = serde_json::from_str(content).expect("content is valid JSON");
    if let Value::Object(root) = &mut value {
        if let Some(Value::Object(attrs)) = root.get_mut("attributes") {
            attrs.insert(
                "service.name".to_string(),
                Value::String(dump_span_attribute_value(&Value::String(
                    service_name.to_string(),
                ))),
            );
        }
    }
    value.to_string()
}

/// `Span.to_dict()` (`mlflow/entities/span.py:306-337`), producing the exact
/// `content` JSON blob (`json.dumps(span_dict, cls=TraceJSONEncoder)` —
/// `ensure_ascii` defaults to `True` at this call site, unlike
/// `dump_span_attribute_value`'s per-attribute dumps, so this top-level
/// serialization escapes non-ASCII as `\uXXXX`).
#[allow(clippy::too_many_arguments)]
fn build_content_json(
    trace_id_bytes: &[u8],
    span_id_bytes: &[u8],
    parent_span_id_bytes: Option<&[u8]>,
    name: &Option<String>,
    start_time_unix_nano: i64,
    end_time_unix_nano: u64,
    status_code_name: &str,
    status_message: &str,
    events: &[span::Event],
    links: &[span::Link],
    attributes: &BTreeMap<String, String>,
) -> String {
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut root = Map::new();
    root.insert(
        "trace_id".to_string(),
        Value::String(b64.encode(trace_id_bytes)),
    );
    root.insert(
        "span_id".to_string(),
        Value::String(b64.encode(span_id_bytes)),
    );
    root.insert(
        "parent_span_id".to_string(),
        match parent_span_id_bytes {
            Some(bytes) => Value::String(b64.encode(bytes)),
            None => Value::Null,
        },
    );
    root.insert(
        "name".to_string(),
        name.clone().map(Value::String).unwrap_or(Value::Null),
    );
    root.insert(
        "start_time_unix_nano".to_string(),
        Value::Number(start_time_unix_nano.into()),
    );
    root.insert(
        "end_time_unix_nano".to_string(),
        if end_time_unix_nano == 0 {
            Value::Null
        } else {
            Value::Number(end_time_unix_nano.into())
        },
    );
    root.insert(
        "events".to_string(),
        Value::Array(
            events
                .iter()
                .map(|e| {
                    let mut m = Map::new();
                    m.insert("name".to_string(), Value::String(e.name.clone()));
                    m.insert(
                        "time_unix_nano".to_string(),
                        Value::Number(e.time_unix_nano.into()),
                    );
                    let mut attrs = Map::new();
                    for kv in &e.attributes {
                        attrs.insert(
                            kv.key.clone(),
                            decode_any_value(kv.value.as_ref()).unwrap_or(Value::Null),
                        );
                    }
                    m.insert("attributes".to_string(), Value::Object(attrs));
                    Value::Object(m)
                })
                .collect(),
        ),
    );
    let mut status = Map::new();
    status.insert(
        "code".to_string(),
        Value::String(status_code_name.to_string()),
    );
    status.insert(
        "message".to_string(),
        Value::String(status_message.to_string()),
    );
    root.insert("status".to_string(), Value::Object(status));
    // `to_dict()`'s `attributes` field is `dict(self._span.attributes)` — the
    // OTel SDK attribute values, which are the (sanitized) JSON-*encoded
    // strings* `serialized_attributes` built above, copied verbatim, NOT
    // re-decoded. The outer `json.dumps(span_dict, ...)` then JSON-string-encodes
    // each of those strings again, so e.g. an attribute whose Python value was
    // `"custom-value"` appears in the final blob as `"\"custom-value\""` (a
    // JSON string containing literal quote characters) unless
    // `sanitize_attributes` already stripped a double-encoding layer.
    root.insert(
        "attributes".to_string(),
        Value::Object(
            attributes
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect(),
        ),
    );
    root.insert(
        "links".to_string(),
        Value::Array(links.iter().map(link_to_dict).collect()),
    );
    to_ascii_json(&Value::Object(root))
}

/// `Link.to_dict()` (`mlflow/entities/link.py:30-35`) for an OTLP proto link
/// (`Link.from_otel_proto`, `link.py:45-64`): `trace_id` becomes `"tr-<hex>"`,
/// `span_id` becomes the 16-hex-char form, and `bytes`-typed link attributes
/// are base64-encoded (unlike span attributes, which are JSON-string-dumped).
fn link_to_dict(link: &span::Link) -> Value {
    let mut m = Map::new();
    m.insert(
        "trace_id".to_string(),
        Value::String(format!(
            "{TRACE_REQUEST_ID_PREFIX}{}",
            hex_lower(&link.trace_id)
        )),
    );
    m.insert(
        "span_id".to_string(),
        Value::String(hex_lower(&link.span_id)),
    );
    let mut attrs = Map::new();
    for kv in &link.attributes {
        let decoded = decode_any_value(kv.value.as_ref()).unwrap_or(Value::Null);
        let value = match decoded {
            Value::Object(ref obj) if obj.contains_key("__mlflow_bytes__") => {
                Value::String(obj["__mlflow_bytes__"].as_str().unwrap_or("").to_string())
            }
            other => other,
        };
        attrs.insert(kv.key.clone(), value);
    }
    m.insert(
        "attributes".to_string(),
        if attrs.is_empty() {
            Value::Null
        } else {
            Value::Object(attrs)
        },
    );
    Value::Object(m)
}

/// Lowercase hex encoding of raw id bytes (`format_trace_id`/`format_span_id`:
/// zero-padded, lowercase, no `0x` prefix).
fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Serialize with `ensure_ascii=True` semantics (Python's `json.dumps`
/// default, used for the top-level `content` blob): escape everything outside
/// printable ASCII as `\uXXXX` (astral-plane code points as UTF-16 surrogate
/// pairs), matching `mlflow_proto::json`'s existing ASCII-escaping helper
/// contract (kept local here since that module's escaping is proto-JSON
/// specific and not exported).
fn to_ascii_json(value: &Value) -> String {
    let compact = value.to_string();
    let mut out = String::with_capacity(compact.len());
    let mut in_string = false;
    let mut escaped = false;
    for c in compact.chars() {
        if in_string {
            if escaped {
                out.push(c);
                escaped = false;
                continue;
            }
            match c {
                '\\' => {
                    out.push(c);
                    escaped = true;
                }
                '"' => {
                    out.push(c);
                    in_string = false;
                }
                c if (c as u32) > 0x7f => escape_ascii(c, &mut out),
                c => out.push(c),
            }
        } else if c == '"' {
            out.push(c);
            in_string = true;
        } else {
            out.push(c);
        }
    }
    out
}

fn escape_ascii(c: char, out: &mut String) {
    let code = c as u32;
    if code > 0xffff {
        // Encode as a UTF-16 surrogate pair, matching `ensure_ascii=True`.
        let v = code - 0x10000;
        let high = 0xd800 + (v >> 10);
        let low = 0xdc00 + (v & 0x3ff);
        out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
    } else {
        out.push_str(&format!("\\u{code:04x}"));
    }
}

/// Precompute the per-trace [`TraceTimeRange`] aggregate for a batch of
/// already-translated spans belonging to the same trace
/// (`sqlalchemy_store.py:5007-5011`, `_get_trace_status_from_root_span`).
pub fn compute_time_range(trace_id: &str, spans: &[&TranslatedSpan]) -> TraceTimeRange {
    let min_start_ms = spans
        .iter()
        .map(|s| s.start_time_unix_nano / 1_000_000)
        .min()
        .unwrap_or(0);
    let max_end_ms = spans
        .iter()
        .filter_map(|s| s.end_time_unix_nano)
        .max()
        .map(|ns| ns / 1_000_000);
    // `_get_trace_status_from_root_span`: first root span found wins; UNSET
    // maps to OK, same as OK (sqlalchemy_store.py:5491-5508).
    let root_span_status = spans.iter().find(|s| s.is_root).map(|s| {
        if s.status == "ERROR" {
            "ERROR".to_string()
        } else {
            "OK".to_string()
        }
    });
    TraceTimeRange {
        trace_id: trace_id.to_string(),
        min_start_ms,
        max_end_ms,
        root_span_status,
    }
}

/// Convert a [`TranslatedSpan`] into the store's [`SpanInput`] row shape.
pub fn to_span_input(span: &TranslatedSpan) -> SpanInput {
    SpanInput {
        trace_id: span.trace_id.clone(),
        span_id: span.span_id.clone(),
        parent_span_id: span.parent_span_id.clone(),
        name: span.name.clone(),
        // Span type is not populated by this task's translation scope (the
        // OTEL-schema translators that would infer `mlflow.spanType` from
        // vendor conventions are deferred — see module docs).
        span_type: None,
        status: span.status.to_string(),
        start_time_unix_nano: span.start_time_unix_nano,
        end_time_unix_nano: span.end_time_unix_nano,
        content: span.content.clone(),
        dimension_attributes: span.dimension_attributes.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlflow_proto::opentelemetry::proto::trace::v1::{ResourceSpans, ScopeSpans, Status};

    fn span_with(trace_id: Vec<u8>, span_id: Vec<u8>, name: &str) -> OtelSpan {
        OtelSpan {
            trace_id,
            span_id,
            name: name.to_string(),
            start_time_unix_nano: 1_000_000_000,
            end_time_unix_nano: 2_000_000_000,
            ..Default::default()
        }
    }

    #[test]
    fn root_span_translates_trace_id_prefix_and_status() {
        let otel_span = OtelSpan {
            status: Some(Status {
                code: 1,
                message: String::new(),
            }),
            ..span_with(vec![1; 16], vec![2; 8], "root")
        };
        let translated = translate_span(&otel_span, None, None).unwrap();
        assert_eq!(translated.trace_id, "tr-01010101010101010101010101010101");
        assert_eq!(translated.span_id, "0202020202020202");
        assert_eq!(translated.status, "OK");
        assert!(translated.parent_span_id.is_none());
        assert!(translated.is_root);
    }

    #[test]
    fn missing_status_defaults_to_unset() {
        let otel_span = span_with(vec![1; 16], vec![2; 8], "root");
        let translated = translate_span(&otel_span, None, None).unwrap();
        assert_eq!(translated.status, "UNSET");
    }

    #[test]
    fn error_status_maps_to_error() {
        let otel_span = OtelSpan {
            status: Some(Status {
                code: 2,
                message: "boom".to_string(),
            }),
            ..span_with(vec![1; 16], vec![2; 8], "root")
        };
        let translated = translate_span(&otel_span, None, None).unwrap();
        assert_eq!(translated.status, "ERROR");
        let content: Value = serde_json::from_str(&translated.content).unwrap();
        assert_eq!(content["status"]["code"], "STATUS_CODE_ERROR");
        assert_eq!(content["status"]["message"], "boom");
    }

    #[test]
    fn content_encodes_ids_as_base64() {
        let otel_span = span_with(vec![1; 16], vec![2; 8], "root");
        let translated = translate_span(&otel_span, None, None).unwrap();
        let content: Value = serde_json::from_str(&translated.content).unwrap();
        assert_eq!(
            content["trace_id"],
            base64::engine::general_purpose::STANDARD.encode([1u8; 16])
        );
        assert_eq!(
            content["span_id"],
            base64::engine::general_purpose::STANDARD.encode([2u8; 8])
        );
        assert!(content["parent_span_id"].is_null());
    }

    #[test]
    fn child_span_has_parent_and_is_not_root() {
        let otel_span = OtelSpan {
            parent_span_id: vec![9; 8],
            ..span_with(vec![1; 16], vec![2; 8], "child")
        };
        let translated = translate_span(&otel_span, None, None).unwrap();
        assert_eq!(
            translated.parent_span_id.as_deref(),
            Some("0909090909090909")
        );
        assert!(!translated.is_root);
    }

    #[test]
    fn empty_trace_id_is_rejected() {
        let otel_span = span_with(vec![], vec![2; 8], "root");
        assert!(translate_span(&otel_span, None, None).is_err());
    }

    #[test]
    fn empty_span_id_is_rejected() {
        let otel_span = span_with(vec![1; 16], vec![], "root");
        assert!(translate_span(&otel_span, None, None).is_err());
    }

    #[test]
    fn request_id_attribute_is_always_server_derived() {
        let otel_span = OtelSpan {
            attributes: vec![KeyValue {
                key: ATTR_REQUEST_ID.to_string(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::StringValue(
                        "tr-clienttrusted".to_string(),
                    )),
                }),
            }],
            ..span_with(vec![1; 16], vec![2; 8], "root")
        };
        let translated = translate_span(&otel_span, None, None).unwrap();
        let content: Value = serde_json::from_str(&translated.content).unwrap();
        // Attribute values in `content.attributes` are the JSON-*encoded*
        // string produced by `dump_span_attribute_value` (see
        // `build_content_json`'s doc comment) — a plain string value is
        // therefore stored double-quoted.
        let request_id_attr = content["attributes"][ATTR_REQUEST_ID].as_str().unwrap();
        assert_eq!(request_id_attr, "\"tr-01010101010101010101010101010101\"");
    }

    #[test]
    fn sanitize_attributes_strips_double_encoding() {
        let mut attrs = BTreeMap::new();
        // Double-encoded: `dump_span_attribute_value("hello")` twice.
        attrs.insert("k".to_string(), "\"\\\"hello\\\"\"".to_string());
        let sanitized = sanitize_attributes(attrs);
        assert_eq!(sanitized["k"], "\"hello\"");
    }

    #[test]
    fn sanitize_attributes_leaves_single_encoded_primitives_alone() {
        let mut attrs = BTreeMap::new();
        attrs.insert("k".to_string(), "1".to_string());
        let sanitized = sanitize_attributes(attrs);
        assert_eq!(sanitized["k"], "1");
    }

    #[test]
    fn dimension_attributes_extracted_from_model_and_provider() {
        let otel_span = OtelSpan {
            attributes: vec![
                KeyValue {
                    key: ATTR_MODEL.to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("\"gpt-4\"".to_string())),
                    }),
                },
                KeyValue {
                    key: ATTR_MODEL_PROVIDER.to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("\"openai\"".to_string())),
                    }),
                },
            ],
            ..span_with(vec![1; 16], vec![2; 8], "root")
        };
        let translated = translate_span(&otel_span, None, None).unwrap();
        let dims: Value =
            serde_json::from_str(translated.dimension_attributes.as_ref().unwrap()).unwrap();
        assert_eq!(dims[ATTR_MODEL], "gpt-4");
        assert_eq!(dims[ATTR_MODEL_PROVIDER], "openai");
    }

    #[test]
    fn service_name_from_resource_propagates_to_root_span_only() {
        let resource = Resource {
            attributes: vec![KeyValue {
                key: "service.name".to_string(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::StringValue("claude-code".to_string())),
                }),
            }],
            ..Default::default()
        };
        let request = ExportTraceServiceRequest {
            resource_spans: vec![ResourceSpans {
                resource: Some(resource),
                scope_spans: vec![ScopeSpans {
                    scope: None,
                    spans: vec![
                        span_with(vec![1; 16], vec![2; 8], "root"),
                        OtelSpan {
                            parent_span_id: vec![2; 8],
                            ..span_with(vec![1; 16], vec![3; 8], "child")
                        },
                    ],
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        };
        let (spans, service_names) = translate_request(&request).unwrap();
        assert_eq!(service_names, vec!["claude-code".to_string()]);
        let root_content: Value = serde_json::from_str(&spans[0].content).unwrap();
        assert_eq!(
            root_content["attributes"]["service.name"],
            "\"claude-code\""
        );
        let child_content: Value = serde_json::from_str(&spans[1].content).unwrap();
        assert!(child_content["attributes"].get("service.name").is_none());
    }

    #[test]
    fn unknown_service_name_is_not_recorded() {
        let resource = Resource {
            attributes: vec![KeyValue {
                key: "service.name".to_string(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::StringValue("some-random-app".to_string())),
                }),
            }],
            ..Default::default()
        };
        let request = ExportTraceServiceRequest {
            resource_spans: vec![ResourceSpans {
                resource: Some(resource),
                scope_spans: vec![ScopeSpans {
                    scope: None,
                    spans: vec![span_with(vec![1; 16], vec![2; 8], "root")],
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        };
        let (spans, service_names) = translate_request(&request).unwrap();
        assert!(service_names.is_empty());
        let content: Value = serde_json::from_str(&spans[0].content).unwrap();
        assert!(content["attributes"].get("service.name").is_none());
    }

    #[test]
    fn compute_time_range_uses_root_span_status_and_min_max() {
        let s1 = TranslatedSpan {
            trace_id: "tr-x".into(),
            span_id: "a".into(),
            parent_span_id: None,
            name: None,
            status: "OK",
            start_time_unix_nano: 5_000_000,
            end_time_unix_nano: Some(10_000_000),
            content: "{}".into(),
            dimension_attributes: None,
            metrics: vec![],
            is_root: true,
        };
        let s2 = TranslatedSpan {
            trace_id: "tr-x".into(),
            span_id: "b".into(),
            parent_span_id: Some("a".into()),
            name: None,
            status: "ERROR",
            start_time_unix_nano: 1_000_000,
            end_time_unix_nano: Some(20_000_000),
            content: "{}".into(),
            dimension_attributes: None,
            metrics: vec![],
            is_root: false,
        };
        let range = compute_time_range("tr-x", &[&s1, &s2]);
        assert_eq!(range.min_start_ms, 1);
        assert_eq!(range.max_end_ms, Some(20));
        // Root span (s1) is OK, so trace status is OK even though a child errored.
        assert_eq!(range.root_span_status.as_deref(), Some("OK"));
    }

    #[test]
    fn non_ascii_attribute_is_escaped_in_content() {
        let otel_span = OtelSpan {
            attributes: vec![KeyValue {
                key: "note".to_string(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::StringValue("héllo".to_string())),
                }),
            }],
            ..span_with(vec![1; 16], vec![2; 8], "root")
        };
        let translated = translate_span(&otel_span, None, None).unwrap();
        assert!(translated.content.contains("\\u00e9"));
        assert!(!translated.content.contains('é'));
    }
}
