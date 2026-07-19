use std::io::Read;

use mlflow_genai::SerializedScorer;
use serde_json::{json, Value};

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&input).unwrap();
    let output = cases
        .into_iter()
        .map(|value| match SerializedScorer::from_value(value.clone()) {
            Ok(scorer) => match scorer.validate_for_oss_execution() {
                Ok(()) => json!({"ok": true, "roundtrip": scorer.to_json_value()}),
                Err(error) => json!({
                    "ok": false,
                    "error": format!("Failed to validate scorer: {error}"),
                    "error_class": "INVALID_PARAMETER_VALUE",
                    "status": 400,
                }),
            },
            Err(error) => json!({
                "ok": false,
                "error": format!("Failed to validate scorer: {error}"),
                "error_class": "INVALID_PARAMETER_VALUE",
                "status": 400,
            }),
        })
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string(&output).unwrap());
}
