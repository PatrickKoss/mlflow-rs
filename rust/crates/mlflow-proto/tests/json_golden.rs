//! Golden tests for the MLflow-compatible JSON codec (T1.3).
//!
//! Fixtures are produced by `rust/tools/gen_goldens.py`
//! (`uv run --frozen python rust/tools/gen_goldens.py`). For each golden `<name>`
//! there is a `<name>.pb` (protobuf wire bytes) and a `<name>.json`
//! (`message_to_json` output, map keys sorted). `manifest.json` maps each name to
//! its fully-qualified protobuf type name.
//!
//! Each golden is exercised three ways:
//!   1. **serialize**: decode the `.pb` and assert the Rust codec produces bytes
//!      identical to `<name>.json`.
//!   2. **round-trip**: parse `<name>.json` back into a message and assert it
//!      re-serializes to the same JSON.
//!   3. **unknown-field tolerance**: inject an extra field into the JSON and
//!      assert parsing still succeeds and ignores it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlflow_proto::{dynamic_from_mlflow_json, to_mlflow_json};
use prost_reflect::{DescriptorPool, DynamicMessage};
use serde_json::Value;

fn goldens_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
}

/// The descriptor pool, rebuilt from the same embedded FDS the crate uses. The
/// test needs it to decode `.pb` bytes into a `DynamicMessage` (the codec's
/// public API takes concrete prost messages, but here we work generically).
fn pool() -> DescriptorPool {
    // Re-decode from the crate's build output; kept in sync via OUT_DIR.
    static FDS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/file_descriptor_set.bin"));
    DescriptorPool::decode(FDS).expect("decode FDS")
}

fn manifest() -> BTreeMap<String, String> {
    let raw = std::fs::read_to_string(goldens_dir().join("manifest.json")).expect("read manifest");
    serde_json::from_str(&raw).expect("parse manifest")
}

/// A `DynamicMessage` implements `prost::Message`, so we can drive the public
/// codec API with it directly — exactly the transcode path a concrete generated
/// type would take.
fn load_pb(name: &str, type_name: &str) -> DynamicMessage {
    let desc = pool()
        .get_message_by_name(type_name)
        .unwrap_or_else(|| panic!("unknown type {type_name}"));
    let bytes = std::fs::read(goldens_dir().join(format!("{name}.pb"))).expect("read pb");
    DynamicMessage::decode(desc, bytes.as_slice()).expect("decode pb")
}

fn load_json(name: &str) -> String {
    std::fs::read_to_string(goldens_dir().join(format!("{name}.json"))).expect("read json")
}

#[test]
fn serialize_matches_python_goldens() {
    for (name, type_name) in manifest() {
        let message = load_pb(&name, &type_name);
        let got = to_mlflow_json(&message, &type_name)
            .unwrap_or_else(|e| panic!("serialize {name}: {e}"));
        let expected = load_json(&name);
        assert_eq!(got, expected, "byte mismatch for golden {name}");
    }
}

#[test]
fn round_trip_parses_back_to_equal_message() {
    for (name, type_name) in manifest() {
        let json = load_json(&name);
        // Parse into a fresh DynamicMessage, then re-serialize; must match.
        let parsed = dynamic_from_mlflow_json(&json, &type_name)
            .unwrap_or_else(|e| panic!("parse {name}: {e}"));
        let reserialized = to_mlflow_json(&parsed, &type_name)
            .unwrap_or_else(|e| panic!("reserialize {name}: {e}"));
        assert_eq!(reserialized, json, "round-trip mismatch for golden {name}");
    }
}

#[test]
fn unknown_fields_are_ignored_on_parse() {
    for (name, type_name) in manifest() {
        let json = load_json(&name);
        let mut value: Value = serde_json::from_str(&json).expect("parse json value");
        // Inject an unknown top-level field. If the golden isn't an object
        // (it always is for these messages), skip.
        if let Value::Object(map) = &mut value {
            map.insert(
                "definitely_not_a_real_field_xyz".to_string(),
                Value::String("ignored".to_string()),
            );
            map.insert(
                "another_unknown".to_string(),
                serde_json::json!({"nested": [1, 2, 3]}),
            );
        }
        let injected = serde_json::to_string(&value).expect("reserialize");
        let parsed = dynamic_from_mlflow_json(&injected, &type_name)
            .unwrap_or_else(|e| panic!("parse-with-unknown {name}: {e}"));
        // The unknown fields must not have leaked into the message; re-serializing
        // yields the original golden.
        let reserialized = to_mlflow_json(&parsed, &type_name)
            .unwrap_or_else(|e| panic!("reserialize {name}: {e}"));
        assert_eq!(
            reserialized,
            load_json(&name),
            "unknown-field tolerance mismatch for golden {name}"
        );
    }
}

#[test]
fn int64_accepted_as_number_or_string() {
    // A run whose start_time is provided as a string parses the same as a number.
    let as_string = r#"{"info": {"run_id": "x", "start_time": "9007199254740993"}}"#;
    let as_number = r#"{"info": {"run_id": "x", "start_time": 9007199254740993}}"#;
    let m1 = dynamic_from_mlflow_json(as_string, "mlflow.Run").expect("string int64");
    let m2 = dynamic_from_mlflow_json(as_number, "mlflow.Run").expect("number int64");
    let j1 = to_mlflow_json(&m1, "mlflow.Run").unwrap();
    let j2 = to_mlflow_json(&m2, "mlflow.Run").unwrap();
    assert_eq!(j1, j2);
    assert!(j1.contains("9007199254740993"));
}

#[test]
fn enum_accepted_as_name_or_number() {
    // RunStatus FINISHED == 3.
    let by_name = r#"{"info": {"run_id": "x", "status": "FINISHED"}}"#;
    let by_number = r#"{"info": {"run_id": "x", "status": 3}}"#;
    let m1 = dynamic_from_mlflow_json(by_name, "mlflow.Run").expect("enum name");
    let m2 = dynamic_from_mlflow_json(by_number, "mlflow.Run").expect("enum number");
    assert_eq!(
        to_mlflow_json(&m1, "mlflow.Run").unwrap(),
        to_mlflow_json(&m2, "mlflow.Run").unwrap()
    );
    assert!(to_mlflow_json(&m1, "mlflow.Run")
        .unwrap()
        .contains("\"FINISHED\""));
}
