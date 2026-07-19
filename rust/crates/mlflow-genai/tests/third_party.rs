use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use mlflow_genai::{
    supported_third_party_metrics, EvalItem, ScorerExecutor, SerializedScorer, ThirdPartyFamily,
};
use serde_json::{json, Value};

#[derive(Clone)]
struct Script {
    requests: Arc<Mutex<Vec<Value>>>,
    responses: Arc<Mutex<VecDeque<Value>>>,
}

async fn scripted(State(script): State<Script>, Json(request): Json<Value>) -> Json<Value> {
    script.requests.lock().unwrap().push(request);
    Json(script.responses.lock().unwrap().pop_front().unwrap())
}

async fn server(contents: &[&str]) -> (String, Script) {
    let responses = contents
        .iter()
        .map(|content| json!({"choices":[{"message":{"content":content}}]}))
        .collect::<VecDeque<_>>();
    let script = Script {
        requests: Arc::new(Mutex::new(Vec::new())),
        responses: Arc::new(Mutex::new(responses)),
    };
    let app = Router::new()
        .route("/", post(scripted))
        .with_state(script.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}/"), script)
}

fn payload(family: &str, metric: &str, model: Value, kwargs: Value) -> SerializedScorer {
    SerializedScorer::from_value(json!({
        "name": metric,
        "third_party_scorer_data": {
            "module": format!("mlflow.genai.scorers.{family}"),
            "class": match family {
                "deepeval" => "DeepEvalScorer",
                "ragas" => "RagasScorer",
                "trulens" => "TruLensScorer",
                _ => metric,
            },
            "metric_name": metric,
            "model": model,
            "kwargs": kwargs,
        }
    }))
    .unwrap()
}

#[test]
fn native_registry_covers_the_pinned_manifest() {
    let manifest: Value =
        serde_json::from_str(include_str!("../../../genai-inventory/scorers.json")).unwrap();
    let mut expected = manifest["third_party_metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            (
                entry["family"].as_str().unwrap().to_string(),
                entry["metric"].as_str().unwrap().to_string(),
                entry["execution"] == "deterministic",
            )
        })
        .collect::<Vec<_>>();
    let mut actual = supported_third_party_metrics()
        .into_iter()
        .map(|entry| {
            (
                match entry.family {
                    ThirdPartyFamily::DeepEval => "deepeval",
                    ThirdPartyFamily::Ragas => "ragas",
                    ThirdPartyFamily::TruLens => "trulens",
                    ThirdPartyFamily::Phoenix => "phoenix",
                }
                .to_string(),
                entry.name.to_string(),
                entry.deterministic,
            )
        })
        .collect::<Vec<_>>();
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 112);
}

#[tokio::test]
async fn deterministic_feedback_matches_the_pinned_golden_cases() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/third_party_golden.json")).unwrap();
    let executor = ScorerExecutor::new();
    for case in corpus["deterministic_cases"].as_array().unwrap() {
        let family = case["family"].as_str().unwrap();
        let metric = case["metric"].as_str().unwrap();
        let scorer = payload(
            family,
            metric,
            case.get("model")
                .cloned()
                .unwrap_or_else(|| json!("openai:/fake-t19-3")),
            case["kwargs"].clone(),
        );
        let result = executor
            .execute(
                &scorer,
                &EvalItem {
                    inputs: (!case["inputs"].is_null()).then(|| case["inputs"].clone()),
                    outputs: (!case["outputs"].is_null()).then(|| case["outputs"].clone()),
                    expectations: (!case["expectations"].is_null())
                        .then(|| case["expectations"].clone()),
                    trace: case.get("trace").filter(|value| !value.is_null()).cloned(),
                    ..EvalItem::default()
                },
                None,
            )
            .await
            .unwrap_or_else(|error| panic!("{family}/{metric}: {error}"));
        let expected = &case["feedback"];
        if let (Some(actual), Some(expected)) = (result.value.as_f64(), expected["value"].as_f64())
        {
            assert!(
                (actual - expected).abs() <= f64::EPSILON,
                "{family}/{metric}: {actual} != {expected}"
            );
        } else {
            assert_eq!(result.value, expected["value"], "{family}/{metric}");
        }
        if let Some(rationale) = expected["rationale"].as_str() {
            assert_eq!(result.rationale, rationale, "{family}/{metric}");
        }
        let source = result.source.as_ref().unwrap();
        assert_eq!(
            source.source_type, expected["source_type"],
            "{family}/{metric}"
        );
        assert_eq!(source.source_id.as_deref(), expected["source_id"].as_str());
        let metadata = result.metadata.as_ref().unwrap();
        for (key, value) in expected["metadata"].as_object().unwrap() {
            let actual = metadata.get(key).unwrap();
            if let (Some(actual), Some(expected)) = (actual.as_f64(), value.as_f64()) {
                assert_eq!(actual, expected, "{family}/{metric}/{key}");
            } else {
                assert_eq!(actual, value, "{family}/{metric}/{key}");
            }
        }
    }
}

#[test]
fn phoenix_d23_rejects_every_manifest_metric_and_wire_spelling() {
    let equivalents = BTreeMap::from([
        ("Hallucination", "Faithfulness"),
        ("QA", "Correctness"),
        ("Relevance", "RelevanceToQuery"),
        ("SQL", "custom instructions judge"),
        ("Summarization", "custom instructions judge"),
        ("Toxicity", "Safety"),
    ]);
    for (metric, equivalent) in equivalents {
        for (module, class, metric_name) in [
            ("mlflow.genai.scorers.phoenix", metric, metric),
            ("mlflow.genai.scorers.phoenix.scorers", metric, metric),
            ("", metric, metric),
            ("", "PhoenixScorer", metric),
        ] {
            let scorer = SerializedScorer::from_value(json!({
                "name": metric,
                "third_party_scorer_data": {
                    "module": module,
                    "class": class,
                    "metric_name": metric_name,
                }
            }))
            .unwrap();
            let error = scorer.validate_for_oss_execution().unwrap_err().to_string();
            assert!(error.contains("Elastic-2.0"), "{metric}: {error}");
            assert!(error.contains(equivalent), "{metric}: {error}");
        }
    }
}

#[tokio::test]
async fn dynamic_metric_errors_match_the_pinned_families() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/third_party_golden.json")).unwrap();
    let executor = ScorerExecutor::new();
    for family in ["deepeval", "ragas", "trulens"] {
        let scorer = payload(
            family,
            "DefinitelyMissingMetric",
            json!("openai:/fake-t19-3"),
            json!({}),
        );
        let error = executor
            .execute(&scorer, &EvalItem::default(), None)
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(error, corpus["dynamic_errors"][family]["message"]);
    }
}

#[tokio::test]
async fn scripted_family_adapter_requests_and_feedback_are_diff_clean() {
    let (url, script) = server(&[
        r#"{"score":0.75,"reason":"deep scripted"}"#,
        r#"{"score":0.75,"reason":"ragas scripted"}"#,
        r#"{"score":2,"criteria":"coherent","supporting_evidence":"structured"}"#,
    ])
    .await;
    let executor = ScorerExecutor::new();
    let item = EvalItem {
        inputs: Some(json!("reference input")),
        outputs: Some(json!("reference output")),
        expectations: Some(
            json!({"expected_output":"reference output","context":"reference context"}),
        ),
        ..EvalItem::default()
    };
    let deep = executor
        .execute(
            &payload(
                "deepeval",
                "AnswerRelevancy",
                json!("openai:/fake-t19-3"),
                json!({"threshold":0.7}),
            ),
            &item,
            Some(&url),
        )
        .await
        .unwrap();
    let ragas = executor
        .execute(
            &payload(
                "ragas",
                "Faithfulness",
                json!("openai:/fake-t19-3"),
                json!({"threshold":0.7}),
            ),
            &item,
            Some(&url),
        )
        .await
        .unwrap();
    let trulens = executor
        .execute(
            &payload(
                "trulens",
                "Coherence",
                json!("openai:/fake-t19-3"),
                json!({"threshold":0.5}),
            ),
            &item,
            Some(&url),
        )
        .await
        .unwrap();
    assert_eq!(deep.value, "yes");
    assert_eq!(deep.rationale, "deep scripted");
    assert_eq!(ragas.value, "yes");
    assert_eq!(ragas.rationale, "ragas scripted");
    assert_eq!(trulens.value, "yes");
    assert_eq!(
        trulens.rationale,
        "reason: Criteria: coherent\nSupporting Evidence: structured"
    );

    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/third_party_golden.json")).unwrap();
    for (index, family) in ["deepeval", "ragas"].into_iter().enumerate() {
        let oracle = corpus["adapter_transcripts"][family]["request"]["messages"][0]["content"]
            .as_str()
            .unwrap();
        let suffix = oracle.strip_prefix("REFERENCE PROMPT").unwrap();
        assert!(
            requests[index]["messages"][0]["content"]
                .as_str()
                .unwrap()
                .ends_with(suffix),
            "{family} adapter suffix differs"
        );
        assert_eq!(requests[index]["model"], "fake-t19-3");
        assert!(requests[index].get("response_format").is_none());
    }
    assert_eq!(requests[2]["messages"][0]["role"], "system");
    assert_eq!(requests[2]["messages"][1]["role"], "user");
    assert_eq!(requests[2]["temperature"], 0.0);
    assert_eq!(
        requests[2]["response_format"]["json_schema"]["name"],
        "ChainOfThoughtResponse"
    );
}
