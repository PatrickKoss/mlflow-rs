use std::collections::BTreeMap;
use std::io::{self, Read};

use mlflow_genai::{
    compute_aggregated_metrics, parse_rate_limit, standardize_scorer_value, AssessmentSource,
    CanonicalAssessment, NamedScorer, SerializedScorer,
};
use serde_json::{json, Value};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let corpus: Value = serde_json::from_str(&input).unwrap();

    let rates = corpus["rates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|raw| {
            let parsed = parse_rate_limit(raw.as_str()).unwrap();
            json!({
                "requests_per_second": parsed.requests_per_second,
                "adaptive": parsed.adaptive,
            })
        })
        .collect::<Vec<_>>();

    let standardized = corpus["standard_values"]
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .map(|value| {
            standardize_scorer_value("seeded", value)
                .unwrap()
                .into_iter()
                .map(|assessment| {
                    json!({
                        "name": assessment.name,
                        "value": assessment.value,
                        "source_type": assessment.source.source_type,
                        "source_id": assessment.source.source_id,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let scorer = NamedScorer {
        scorer: SerializedScorer::from_value(json!({
            "name": "quality",
            "aggregations": ["min", "max", "mean", "median", "variance", "p90"],
            "builtin_scorer_class": "ResponseLength",
            "builtin_scorer_pydantic_data": {"max_length": 100}
        }))
        .unwrap(),
        gateway_url: None,
        embedding_url: None,
    };
    let assessments = corpus["aggregate_values"]
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .map(|value| CanonicalAssessment {
            name: "quality".to_string(),
            value: Some(value),
            rationale: None,
            source: AssessmentSource {
                source_type: "CODE".to_string(),
                source_id: Some("quality".to_string()),
            },
            metadata: BTreeMap::new(),
            span_id: None,
            error: None,
            create_time_ms: 0,
            last_update_time_ms: 0,
        })
        .collect::<Vec<_>>();
    let metrics = compute_aggregated_metrics(&assessments, &[scorer]);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "rates": rates,
            "standardized": standardized,
            "metrics": metrics,
        }))
        .unwrap()
    );
}
