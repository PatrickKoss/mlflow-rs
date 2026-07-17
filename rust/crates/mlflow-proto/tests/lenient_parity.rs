//! ParseDict partial-parse parity tests (T12.5).
//!
//! Every expectation below is derived from a live Python experiment against
//! `mlflow.utils.proto_json_utils.parse_dict` (google's `ParseDict`, which
//! `_get_request_message` runs inside a swallowing try/except). The exact
//! `uv run --frozen python -c '...'` command and its output are recorded in each
//! test so the Rust contract is auditable against the source of truth rather than
//! guessed.

use mlflow_proto::{lenient_from_mlflow_json, to_mlflow_json};
use prost_reflect::Value;

/// Serialize the parsed dynamic message back to MLflow JSON for comparison.
fn parse_and_dump(json: &str, type_name: &str) -> (String, bool) {
    let parsed = lenient_from_mlflow_json(json, type_name).expect("lenient parse");
    let dumped = to_mlflow_json(&parsed.message, type_name).expect("dump");
    (dumped, parsed.proto_parsing_succeeded)
}

fn field_str(json: &str, type_name: &str, field: &str) -> String {
    let parsed = lenient_from_mlflow_json(json, type_name).expect("parse");
    match parsed.message.get_field_by_name(field) {
        Some(v) => match v.as_ref() {
            Value::String(s) => s.clone(),
            other => format!("{other:?}"),
        },
        None => String::new(),
    }
}

// EXPERIMENT E1 (log-batch, bool tag value swallowed):
//   uv run --frozen python -c "
//   from mlflow.utils.proto_json_utils import parse_dict
//   from mlflow.protos.service_pb2 import LogBatch
//   msg = LogBatch()
//   req = {'run_id':'abc123','metrics':[{'key':'m1','value':1.5,'timestamp':100,'step':2}],
//          'params':[{'key':'p1','value':'pv'}],'tags':[{'key':'batch_tag','value':True}]}
//   try: parse_dict(req,msg)
//   except Exception as e: print('EXC', e)
//   print(msg.run_id, [(m.key,m.value,m.timestamp,m.step) for m in msg.metrics],
//         [(p.key,p.value) for p in msg.params], [(t.key,t.value) for t in msg.tags])"
// Output:
//   EXC 'Failed to parse tags field: Failed to parse value field: expected string ...'
//   abc123 [('m1', 1.5, 100, 2)] [('p1', 'pv')] [('batch_tag', '')]
#[test]
fn log_batch_bool_tag_value_partial_parse() {
    let json = r#"{"run_id":"abc123",
        "metrics":[{"key":"m1","value":1.5,"timestamp":100,"step":2}],
        "params":[{"key":"p1","value":"pv"}],
        "tags":[{"key":"batch_tag","value":true}]}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.LogBatch");
    assert!(!ok, "codec failure must be recorded");
    let v: serde_json::Value = serde_json::from_str(&dumped).unwrap();
    // run_id, metrics, params all applied (they precede tags in insertion order).
    assert_eq!(v["run_id"], "abc123");
    assert_eq!(v["metrics"][0]["key"], "m1");
    assert_eq!(v["metrics"][0]["value"], 1.5);
    assert_eq!(v["metrics"][0]["timestamp"], 100);
    assert_eq!(v["metrics"][0]["step"], 2);
    assert_eq!(v["params"][0]["key"], "p1");
    assert_eq!(v["params"][0]["value"], "pv");
    // The failing tag element keeps its leading `key`; `value` stays default
    // (empty string, hence omitted by proto2 presence in the JSON dump).
    assert_eq!(v["tags"][0]["key"], "batch_tag");
    assert!(v["tags"][0].get("value").is_none());
}

// EXPERIMENT E2 (bad field first aborts everything):
//   req = {'run_id':123,'metrics':[{'key':'m1','value':1.5,'timestamp':100}],
//          'params':[{'key':'p1','value':'x'}]}
// Output: EXC 'Failed to parse run_id field ...'; run_id='' metrics=0 params=0
#[test]
fn log_batch_bad_first_field_aborts_all() {
    let json = r#"{"run_id":123,
        "metrics":[{"key":"m1","value":1.5,"timestamp":100}],
        "params":[{"key":"p1","value":"x"}]}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.LogBatch");
    assert!(!ok);
    let v: serde_json::Value = serde_json::from_str(&dumped).unwrap();
    assert!(v.get("run_id").is_none(), "run_id must be unset");
    assert!(v.get("metrics").is_none(), "metrics after failure unset");
    assert!(v.get("params").is_none(), "params after failure unset");
}

// EXPERIMENT E3 (repeated element ordering: 2nd of 3 fails):
//   req = {'run_id':'r','tags':[{'key':'t1','value':'ok'},
//          {'key':'t2','value':True},{'key':'t3','value':'after'}]}
// Output: EXC 'Failed to parse tags field ...'; tags=[('t1','ok'),('t2','')]
#[test]
fn log_batch_repeated_element_stops_at_first_failure() {
    let json = r#"{"run_id":"r","tags":[
        {"key":"t1","value":"ok"},
        {"key":"t2","value":true},
        {"key":"t3","value":"after"}]}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.LogBatch");
    assert!(!ok);
    let v: serde_json::Value = serde_json::from_str(&dumped).unwrap();
    assert_eq!(v["run_id"], "r");
    assert_eq!(
        v["tags"].as_array().unwrap().len(),
        2,
        "t1 + partial t2 only"
    );
    assert_eq!(v["tags"][0]["key"], "t1");
    assert_eq!(v["tags"][0]["value"], "ok");
    assert_eq!(v["tags"][1]["key"], "t2");
    assert!(v["tags"][1].get("value").is_none());
}

// EXPERIMENT E4 (within-object field order: value before key, value fails):
//   req = {'run_id':'r','tags':[{'value':True,'key':'t1'}]}
// Output: EXC 'Failed to parse tags field ...'; tags=[('','')]
#[test]
fn log_batch_within_object_field_order_respected() {
    let json = r#"{"run_id":"r","tags":[{"value":true,"key":"t1"}]}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.LogBatch");
    assert!(!ok);
    let v: serde_json::Value = serde_json::from_str(&dumped).unwrap();
    assert_eq!(v["run_id"], "r");
    // value comes first and fails, so key (after it) is never applied: empty tag.
    assert_eq!(v["tags"].as_array().unwrap().len(), 1);
    assert!(v["tags"][0].get("key").is_none());
    assert!(v["tags"][0].get("value").is_none());
}

// EXPERIMENT E5 (createExperiment int name; name is first -> whole msg empty):
//   req = {'name':123,'artifact_location':'/tmp/x','tags':[{'key':'k','value':'v'}]}
// Output: EXC 'Failed to parse name field ...'; name='' artifact='' tags=[]
#[test]
fn create_experiment_int_name_leaves_message_empty() {
    let json = r#"{"name":123,"artifact_location":"/tmp/x","tags":[{"key":"k","value":"v"}]}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.CreateExperiment");
    assert!(!ok);
    assert_eq!(dumped, "{}", "name-first failure aborts everything");
    assert_eq!(field_str(json, "mlflow.CreateExperiment", "name"), "");
}

// EXPERIMENT E6 (numeric strings coerce; proto parse SUCCEEDS):
//   req = {'run_id':'r','metrics':[{'key':'m','value':'1.5','timestamp':'100','step':'2'}]}
// Output: no exception; metrics=[('m',1.5,100,2)]
#[test]
fn log_batch_numeric_string_coercion_succeeds() {
    let json =
        r#"{"run_id":"r","metrics":[{"key":"m","value":"1.5","timestamp":"100","step":"2"}]}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.LogBatch");
    assert!(ok, "numeric strings are valid proto3 JSON, parse succeeds");
    let v: serde_json::Value = serde_json::from_str(&dumped).unwrap();
    assert_eq!(v["metrics"][0]["value"], 1.5);
    assert_eq!(v["metrics"][0]["timestamp"], 100);
    assert_eq!(v["metrics"][0]["step"], 2);
}

// EXPERIMENT TRACE (start_trace_v3, string request_time not RFC3339):
//   req = {'trace':{'trace_info':{'trace_id':'tr-compliance-1',
//          'trace_location':{'type':'MLFLOW_EXPERIMENT','mlflow_experiment':{'experiment_id':'123'}},
//          'request_time':'1000','execution_duration_ms':'5','state':'OK'}}}
// Output: EXC 'Failed to parse trace field: ... request_time field: ... missing valid timezone offset';
//   trace_id='tr-compliance-1' request_time.seconds=0 state=0 location.type=1
#[test]
fn start_trace_v3_bad_timestamp_partial_parse() {
    let json = r#"{"trace":{"trace_info":{
        "trace_id":"tr-compliance-1",
        "trace_location":{"type":"MLFLOW_EXPERIMENT","mlflow_experiment":{"experiment_id":"123"}},
        "request_time":"1000",
        "execution_duration_ms":"5",
        "state":"OK"}}}"#;
    let (dumped, ok) = parse_and_dump(json, "mlflow.StartTraceV3");
    assert!(!ok);
    let v: serde_json::Value = serde_json::from_str(&dumped).unwrap();
    let ti = &v["trace"]["trace_info"];
    // trace_id + trace_location precede request_time -> applied.
    assert_eq!(ti["trace_id"], "tr-compliance-1");
    assert_eq!(ti["trace_location"]["type"], "MLFLOW_EXPERIMENT");
    assert_eq!(
        ti["trace_location"]["mlflow_experiment"]["experiment_id"],
        "123"
    );
    // request_time failed -> unset; state + execution_duration after it -> unset.
    assert!(ti.get("request_time").is_none(), "failing field unset");
    // state=STATE_UNSPECIFIED (0) is proto2-default and omitted from the dump.
    assert!(ti.get("state").is_none() || ti["state"] == "STATE_UNSPECIFIED");
    assert!(ti.get("execution_duration").is_none());
}

// Sanity: a fully-valid body parses strictly and reports success.
#[test]
fn valid_body_parses_strictly() {
    let json = r#"{"name":"exp_alpha","artifact_location":"/tmp/a"}"#;
    let parsed = lenient_from_mlflow_json(json, "mlflow.CreateExperiment").unwrap();
    assert!(parsed.proto_parsing_succeeded);
    assert_eq!(
        parsed
            .message
            .get_field_by_name("name")
            .unwrap()
            .as_str()
            .unwrap(),
        "exp_alpha"
    );
}
