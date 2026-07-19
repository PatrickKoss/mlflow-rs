//! Cross-language golden and differential tests for T21.3 `traces.pb`.

use std::path::{Path, PathBuf};
use std::process::Command;

use mlflow_proto::opentelemetry::proto::common::v1::{any_value, AnyValue, ArrayValue, KeyValue};
use mlflow_proto::opentelemetry::proto::resource::v1::Resource;
use mlflow_server::trace_archival::{
    decode_traces_pb, encode_traces_pb, stored_spans_to_traces_pb, TraceArchive,
    TRACE_ARCHIVAL_FILENAME,
};
use mlflow_store::StoredSpan;
use serde::Deserialize;
use serde_json::Value;

const PYTHON_DB_GOLDEN: &[u8] = include_bytes!("fixtures/trace_archival/python_db_traces.pb");
const PYTHON_RESOURCE_GOLDEN: &[u8] =
    include_bytes!("fixtures/trace_archival/python_resource_traces.pb");
const MANIFEST: &str = include_str!("fixtures/trace_archival/manifest.json");

#[derive(Deserialize)]
struct Manifest {
    expected_order: Vec<String>,
    resource_attributes: Vec<(String, Value)>,
    stored_spans: Vec<FixtureStoredSpan>,
}

#[derive(Deserialize)]
struct FixtureStoredSpan {
    trace_id: String,
    experiment_id: i64,
    span_id: String,
    parent_span_id: Option<String>,
    name: Option<String>,
    span_type: Option<String>,
    status: String,
    start_time_unix_nano: i64,
    end_time_unix_nano: Option<i64>,
    duration_ns: Option<i64>,
    content: String,
    dimension_attributes: Option<String>,
}

impl From<FixtureStoredSpan> for StoredSpan {
    fn from(span: FixtureStoredSpan) -> Self {
        Self {
            trace_id: span.trace_id,
            experiment_id: span.experiment_id,
            span_id: span.span_id,
            parent_span_id: span.parent_span_id,
            name: span.name,
            span_type: span.span_type,
            status: span.status,
            start_time_unix_nano: span.start_time_unix_nano,
            end_time_unix_nano: span.end_time_unix_nano,
            duration_ns: span.duration_ns,
            content: span.content,
            dimension_attributes: span.dimension_attributes,
        }
    }
}

fn manifest() -> Manifest {
    serde_json::from_str(MANIFEST).unwrap()
}

#[test]
fn rust_reads_python_goldens_and_round_trips_losslessly() {
    let manifest = manifest();
    let db_archive = decode_traces_pb(PYTHON_DB_GOLDEN).unwrap();
    assert_eq!(
        db_archive
            .spans
            .iter()
            .map(|span| span.name.as_str())
            .collect::<Vec<_>>(),
        manifest
            .expected_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    assert!(db_archive.resource.attributes.is_empty());
    assert_eq!(encode_traces_pb(&db_archive).unwrap(), PYTHON_DB_GOLDEN);

    let resource_archive = decode_traces_pb(PYTHON_RESOURCE_GOLDEN).unwrap();
    assert_eq!(resource_archive.resource.attributes.len(), 5);
    assert_eq!(resource_archive.resource.attributes[0].key, "service.name");
    assert_eq!(
        string_value(&resource_archive.resource.attributes[0]),
        "archive-fixture"
    );
    assert_eq!(resource_archive.spans[0].events.len(), 1);
    assert_eq!(resource_archive.spans[0].links.len(), 1);
    assert_eq!(resource_archive.spans[0].status.as_ref().unwrap().code, 2);
    assert_eq!(
        encode_traces_pb(&resource_archive).unwrap(),
        PYTHON_RESOURCE_GOLDEN
    );
}

#[test]
fn rust_stored_entity_writer_is_byte_identical_to_python() {
    let stored_spans: Vec<StoredSpan> = manifest()
        .stored_spans
        .into_iter()
        .map(StoredSpan::from)
        .collect();
    let rust_bytes = stored_spans_to_traces_pb(&stored_spans).unwrap();
    assert_eq!(rust_bytes, PYTHON_DB_GOLDEN);

    let archive = decode_traces_pb(&rust_bytes).unwrap();
    assert_eq!(archive.spans[0].name, "root");
    let root = &archive.spans[0];
    assert_eq!(root.events[0].name, "exception");
    assert_eq!(root.links[0].attributes.len(), 3);
    assert_eq!(root.status.as_ref().unwrap().message, "root failed");
}

#[test]
fn rust_resource_writer_is_byte_identical_to_python() {
    let manifest = manifest();
    let mut archive = decode_traces_pb(PYTHON_DB_GOLDEN).unwrap();
    archive.resource = Resource {
        attributes: manifest
            .resource_attributes
            .iter()
            .map(|(key, value)| KeyValue {
                key: key.clone(),
                value: Some(any_value(value)),
            })
            .collect(),
        ..Default::default()
    };
    assert_eq!(
        encode_traces_pb(&TraceArchive {
            resource: archive.resource,
            spans: archive.spans,
        })
        .unwrap(),
        PYTHON_RESOURCE_GOLDEN
    );
}

#[test]
fn python_reads_rust_bytes_and_round_trips_them() {
    let stored_spans: Vec<StoredSpan> = manifest()
        .stored_spans
        .into_iter()
        .map(StoredSpan::from)
        .collect();
    let rust_bytes = stored_spans_to_traces_pb(&stored_spans).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let rust_payload = directory.path().join(TRACE_ARCHIVAL_FILENAME);
    std::fs::write(&rust_payload, rust_bytes).unwrap();

    let root = repo_root();
    let output =
        Command::new("uv")
            .args([
                "run",
                "--frozen",
                "python",
                "rust/tools/trace_archival_cross_language.py",
            ])
            .arg(&rust_payload)
            .arg(root.join(
                "rust/crates/mlflow-server/tests/fixtures/trace_archival/python_db_traces.pb",
            ))
            .current_dir(&root)
            .output()
            .expect("run Python trace-archival differential");
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["python_read_rust"], true);
    assert_eq!(result["python_round_trip_byte_equal"], true);
    assert_eq!(result["span_count"], 3);
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn string_value(key_value: &KeyValue) -> &str {
    match key_value
        .value
        .as_ref()
        .and_then(|value| value.value.as_ref())
    {
        Some(any_value::Value::StringValue(value)) => value,
        other => panic!("expected string AnyValue, got {other:?}"),
    }
}

fn any_value(value: &Value) -> AnyValue {
    let value = match value {
        Value::Bool(value) => Some(any_value::Value::BoolValue(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(any_value::Value::IntValue)
            .or_else(|| value.as_f64().map(any_value::Value::DoubleValue)),
        Value::String(value) => Some(any_value::Value::StringValue(value.clone())),
        Value::Array(items) => Some(any_value::Value::ArrayValue(ArrayValue {
            values: items.iter().map(any_value).collect(),
        })),
        other => panic!("unsupported fixture resource value: {other}"),
    };
    AnyValue { value }
}
