use mlflow_genai::SerializedScorer;

const BUILTIN: &str = include_str!("fixtures/builtin_response_length_scorer.json");
const INSTRUCTIONS: &str = include_str!("fixtures/instructions_judge_scorer.json");

#[test]
fn parses_python_builtin_payload() {
    let scorer = SerializedScorer::from_json(BUILTIN).expect("Python fixture parses");
    let SerializedScorer::Builtin(payload) = scorer else {
        panic!("expected builtin payload");
    };
    assert_eq!(payload.class_name, "ResponseLength");
    assert_eq!(payload.common.name, "response_length");
    assert_eq!(payload.common.serialization_version, 1);
    assert_eq!(payload.pydantic_data["unit"], "words");
    assert_eq!(payload.pydantic_data["min_length"], 2);
    assert_eq!(payload.pydantic_data["max_length"], 4);
}

#[test]
fn parses_python_instructions_payload() {
    let scorer = SerializedScorer::from_json(INSTRUCTIONS).expect("Python fixture parses");
    let SerializedScorer::Instructions(payload) = scorer else {
        panic!("expected instructions payload");
    };
    assert_eq!(payload.common.name, "concise_answer");
    assert_eq!(payload.pydantic_data["model"], "openai:/mock-judge");
}

#[test]
fn rejects_multiple_representations() {
    let payload = serde_json::json!({
        "name": "ambiguous",
        "builtin_scorer_class": "ResponseLength",
        "builtin_scorer_pydantic_data": {},
        "instructions_judge_pydantic_data": {}
    });
    let error = SerializedScorer::from_json(&payload.to_string()).unwrap_err();
    assert!(error.to_string().contains("exactly one"));
}
