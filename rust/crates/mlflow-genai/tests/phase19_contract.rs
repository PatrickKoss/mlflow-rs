use std::collections::BTreeSet;

use mlflow_genai::{supported_builtin_scorers, SerializedScorer};
use serde_json::{json, Value};

#[test]
fn scorer_inventory_has_complete_phase19_partition() {
    let inventory: Value =
        serde_json::from_str(include_str!("../../../genai-inventory/scorers.json")).unwrap();
    let manifest = inventory["builtin_scorers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    let native = supported_builtin_scorers()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(native, manifest);
    assert_eq!(native.len(), 24);
    assert_eq!(inventory["serialized_judges"].as_array().unwrap().len(), 2);
    assert_eq!(inventory["rejected_payloads"].as_array().unwrap().len(), 1);
    let third_party = inventory["third_party_metrics"].as_array().unwrap();
    assert_eq!(third_party.len(), 112);
    assert_eq!(
        third_party
            .iter()
            .filter(|entry| entry["family"] == "deepeval")
            .count(),
        44
    );
    assert_eq!(
        third_party
            .iter()
            .filter(|entry| entry["family"] == "ragas")
            .count(),
        37
    );
    assert_eq!(
        third_party
            .iter()
            .filter(|entry| entry["family"] == "trulens")
            .count(),
        25
    );
    assert_eq!(
        third_party
            .iter()
            .filter(|entry| entry["family"] == "phoenix")
            .count(),
        6
    );
}

#[test]
fn all_five_serialized_representations_parse_and_round_trip() {
    let cases = [
        json!({
            "name": "length",
            "builtin_scorer_class": "ResponseLength",
            "builtin_scorer_pydantic_data": {"max_length": 10},
            "future_field": {"retained": true}
        }),
        json!({
            "name": "code",
            "call_source": "return True",
            "call_signature": "(outputs)",
            "original_func_name": "code"
        }),
        json!({
            "name": "judge",
            "instructions_judge_pydantic_data": {
                "instructions": "Evaluate {{ outputs }}.",
                "model": "openai:/fake-chat",
                "feedback_value_type": {"type": "string"}
            }
        }),
        json!({
            "name": "memory",
            "memory_augmented_judge_data": {
                "base_judge": {
                    "name": "judge",
                    "instructions_judge_pydantic_data": {
                        "instructions": "Evaluate {{ outputs }}.",
                        "model": "openai:/fake-chat",
                        "feedback_value_type": {"type": "string"}
                    }
                },
                "episodic_trace_ids": [],
                "semantic_memory": [],
                "retrieval_k": 2,
                "embedding_model": "openai:/fake-embedding",
                "embedding_dim": 3
            }
        }),
        json!({
            "name": "metric",
            "third_party_scorer_data": {
                "module": "mlflow.genai.scorers.ragas",
                "class": "Faithfulness",
                "metric_name": "Faithfulness"
            }
        }),
    ];
    for value in cases {
        let encoded = value.to_string();
        let parsed = SerializedScorer::from_json(&encoded).unwrap();
        assert_eq!(parsed.to_json_value(), value);
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.to_json().unwrap()).unwrap(),
            value
        );
    }
}

#[test]
fn execution_rejections_match_python_messages() {
    let decorator = SerializedScorer::from_value(json!({
        "name": "x",
        "call_source": "return True",
        "call_signature": "(outputs)",
        "original_func_name": "x"
    }))
    .unwrap();
    assert_eq!(
        decorator.validate_for_oss_execution().unwrap_err().to_string(),
        "Custom scorer registration (using @scorer decorator) is not supported outside of Databricks tracking environments due to security concerns. Custom scorers require arbitrary code execution during deserialization.\n\nTo use custom scorers:\n1. Configure MLflow to use a Databricks tracking URI, or\n2. Manage your custom scorer code in a source code repository (e.g., GitHub) and import it directly, or\n3. Use built-in scorers or make_judge() scorers instead.\nRegistered scorer code:\n\n\nfrom mlflow.genai import scorer\n\n@scorer\ndef x(outputs):\n    return True\n"
    );

    let incomplete = SerializedScorer::from_value(json!({
        "name": "x",
        "call_source": "return True"
    }))
    .unwrap();
    assert_eq!(
        incomplete.validate_for_oss_execution().unwrap_err().to_string(),
        "Failed to load scorer 'x'. The scorer is serialized in an unknown format that cannot be deserialized. Please make sure you are using a compatible MLflow version or recreate the scorer. Scorer was created with MLflow version: 3.14.1.dev0, serialization version: 1, current MLflow version: 3.14.1.dev0."
    );
}
