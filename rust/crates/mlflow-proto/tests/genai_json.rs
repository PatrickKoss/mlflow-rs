//! MLflow-JSON round-trip coverage for the Part II entity protos (T15.2).
//!
//! All five files use proto2. The expected strings were produced with Python's
//! `mlflow.utils.proto_json_utils.message_to_json`; map keys use the Rust
//! codec's documented deterministic sort instead of Python's hash order.

use std::collections::HashMap;

use mlflow_proto::mlflow::{
    self, datasets, issues,
    label_schemas::{self, label_schema_input},
    review_queues,
};
use mlflow_proto::{from_mlflow_json, to_mlflow_json};

fn assert_round_trip<M>(message: &M, type_name: &str, expected: &str)
where
    M: prost::Message + Default,
{
    let json = to_mlflow_json(message, type_name).expect("serialize MLflow JSON");
    assert_eq!(json, expected);

    let parsed: M = from_mlflow_json(&json, type_name).expect("parse MLflow JSON");
    assert_eq!(
        to_mlflow_json(&parsed, type_name).expect("re-serialize MLflow JSON"),
        expected
    );
}

#[test]
fn datasets_proto_json_round_trip_preserves_presence_and_json_strings() {
    let message = datasets::Dataset {
        dataset_id: Some("d-1".to_string()),
        tags: Some(r#"{"team":"genai"}"#.to_string()),
        created_time: Some(0),
        experiment_ids: vec!["2".to_string(), "1".to_string()],
        ..Default::default()
    };

    assert_round_trip(
        &message,
        "mlflow.datasets.Dataset",
        r#"{
  "dataset_id": "d-1",
  "tags": "{\"team\":\"genai\"}",
  "created_time": 0,
  "experiment_ids": [
    "2",
    "1"
  ]
}"#,
    );
}

#[test]
fn issues_proto_json_round_trip_keeps_int64_numeric() {
    let message = issues::Issue {
        issue_id: Some("i-1".to_string()),
        root_causes: vec!["timeout".to_string(), "retry".to_string()],
        created_timestamp: Some(9_007_199_254_740_993),
        trace_count: Some(0),
        ..Default::default()
    };

    assert_round_trip(
        &message,
        "mlflow.issues.Issue",
        r#"{
  "issue_id": "i-1",
  "root_causes": [
    "timeout",
    "retry"
  ],
  "created_timestamp": 9007199254740993,
  "trace_count": 0
}"#,
    );
}

#[test]
fn label_schemas_proto_json_round_trip_handles_enum_and_oneof() {
    let message = label_schemas::LabelSchema {
        schema_id: Some("ls-1".to_string()),
        r#type: Some(label_schemas::LabelSchemaType::Expectation as i32),
        enable_comment: Some(false),
        input: Some(label_schemas::LabelSchemaInput {
            input: Some(label_schema_input::Input::Categorical(
                label_schemas::InputCategorical {
                    options: vec!["good".to_string(), "bad".to_string()],
                    multi_select: Some(false),
                },
            )),
        }),
        created_at: Some(0),
        ..Default::default()
    };

    assert_round_trip(
        &message,
        "mlflow.label_schemas.LabelSchema",
        r#"{
  "schema_id": "ls-1",
  "type": "EXPECTATION",
  "enable_comment": false,
  "input": {
    "categorical": {
      "options": [
        "good",
        "bad"
      ],
      "multi_select": false
    }
  },
  "created_at": 0
}"#,
    );
}

#[test]
fn review_queues_proto_json_round_trip_keeps_repeated_order() {
    let message = review_queues::ReviewQueue {
        queue_id: Some("rq-1".to_string()),
        queue_type: Some(review_queues::ReviewQueueType::Custom as i32),
        creation_time_ms: Some(0),
        users: vec!["alice".to_string(), "bob".to_string()],
        schema_ids: vec!["ls-1".to_string()],
        ..Default::default()
    };

    assert_round_trip(
        &message,
        "mlflow.review_queues.ReviewQueue",
        r#"{
  "queue_id": "rq-1",
  "queue_type": "CUSTOM",
  "creation_time_ms": 0,
  "users": [
    "alice",
    "bob"
  ],
  "schema_ids": [
    "ls-1"
  ]
}"#,
    );
}

#[test]
fn prompt_optimization_proto_json_round_trip_sorts_maps() {
    let message = mlflow::PromptOptimizationJob {
        job_id: Some("j-1".to_string()),
        config: Some(mlflow::PromptOptimizationJobConfig {
            optimizer_type: Some(mlflow::OptimizerType::Gepa as i32),
            dataset_id: Some("d-1".to_string()),
            scorers: vec!["Correctness".to_string()],
            ..Default::default()
        }),
        creation_timestamp_ms: Some(9_007_199_254_740_993),
        initial_eval_scores: HashMap::from([("zeta".to_string(), 0.5), ("alpha".to_string(), 1.0)]),
        ..Default::default()
    };

    assert_round_trip(
        &message,
        "mlflow.PromptOptimizationJob",
        r#"{
  "job_id": "j-1",
  "config": {
    "optimizer_type": "OPTIMIZER_TYPE_GEPA",
    "dataset_id": "d-1",
    "scorers": [
      "Correctness"
    ]
  },
  "creation_timestamp_ms": 9007199254740993,
  "initial_eval_scores": {
    "alpha": 1.0,
    "zeta": 0.5
  }
}"#,
    );
}
