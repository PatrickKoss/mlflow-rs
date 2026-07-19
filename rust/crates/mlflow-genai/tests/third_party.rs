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

async fn server(responses: Vec<Value>) -> (String, Script) {
    let responses = responses.into_iter().collect::<VecDeque<_>>();
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

fn scripted_responses(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .map(|call| {
            if call["kind"] == "embedding" {
                json!({"data":[{"embedding":[1.0,0.0]},{"embedding":[1.0,0.0]}]})
            } else {
                json!({"choices":[{"message":{"content":call["response"]}}]})
            }
        })
        .collect()
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
async fn every_pinned_workflow_request_and_feedback_is_diff_clean() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/third_party_golden.json")).unwrap();
    let executor = ScorerExecutor::new();
    let trace = corpus["workflow_case"]["trace"].clone();
    let item = EvalItem {
        inputs: Some(json!("reference input")),
        outputs: Some(json!("reference output")),
        expectations: Some(corpus["workflow_case"]["expectations"].clone()),
        trace: Some(trace.clone()),
        session: Some(vec![trace]),
        ..EvalItem::default()
    };
    for workflow in corpus["workflow_transcripts"].as_array().unwrap() {
        let family = workflow["family"].as_str().unwrap();
        let metric = workflow["metric"].as_str().unwrap();
        let scorer = payload(
            family,
            metric,
            json!("openai:/fake-t19-3"),
            workflow["kwargs"].clone(),
        );
        if workflow["status"] == "pinned-error" {
            let error = executor
                .execute_all(&scorer, &item, None, None)
                .await
                .unwrap_err()
                .to_string();
            assert_eq!(error, workflow["error"]["message"], "{family}/{metric}");
            continue;
        }

        let calls = workflow["calls"].as_array().unwrap();
        let (url, script) = server(scripted_responses(calls)).await;
        let result = executor
            .execute_all(&scorer, &item, Some(&url), Some(&url))
            .await
            .unwrap_or_else(|error| panic!("{family}/{metric}: {error}"));
        assert_eq!(result.len(), 1, "{family}/{metric}");
        let result = &result[0];
        let expected = &workflow["feedback"];
        assert_eq!(result.value, expected["value"], "{family}/{metric}/value");
        assert_eq!(
            result.rationale,
            expected["rationale"].as_str().unwrap_or_default(),
            "{family}/{metric}/rationale"
        );
        assert_eq!(
            result.source.as_ref().unwrap().source_type,
            expected["source_type"],
            "{family}/{metric}/source_type"
        );
        assert_eq!(
            result.source.as_ref().unwrap().source_id.as_deref(),
            expected["source_id"].as_str(),
            "{family}/{metric}/source_id"
        );
        assert_eq!(
            serde_json::to_value(result.metadata.as_ref().unwrap()).unwrap(),
            expected["metadata"],
            "{family}/{metric}/metadata"
        );
        assert_eq!(
            script.requests.lock().unwrap().as_slice(),
            calls
                .iter()
                .map(|call| call["request"].clone())
                .collect::<Vec<_>>(),
            "{family}/{metric}/ordered requests"
        );
    }
}

#[tokio::test]
async fn every_pinned_parser_matches_its_malformed_transcript() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/third_party_golden.json")).unwrap();
    let executor = ScorerExecutor::new();
    let trace = corpus["workflow_case"]["trace"].clone();
    let item = EvalItem {
        inputs: Some(json!("reference input")),
        outputs: Some(json!("reference output")),
        expectations: Some(corpus["workflow_case"]["expectations"].clone()),
        trace: Some(trace.clone()),
        session: Some(vec![trace]),
        ..EvalItem::default()
    };
    for workflow in corpus["workflow_transcripts"].as_array().unwrap() {
        let Some(malformed) = workflow.get("malformed") else {
            continue;
        };
        let family = workflow["family"].as_str().unwrap();
        let metric = workflow["metric"].as_str().unwrap();
        let scorer = payload(
            family,
            metric,
            json!("openai:/fake-t19-3"),
            workflow["kwargs"].clone(),
        );
        let calls = malformed["calls"].as_array().unwrap();
        let (url, script) = server(scripted_responses(calls)).await;
        let result = executor
            .execute_all(&scorer, &item, Some(&url), Some(&url))
            .await;
        if let Some(message) = malformed["error"]["message"].as_str() {
            assert_eq!(
                result.unwrap_err().to_string(),
                message,
                "{family}/{metric}/malformed error"
            );
        } else {
            let result =
                result.unwrap_or_else(|error| panic!("{family}/{metric}/malformed: {error}"));
            assert_eq!(result[0].value, malformed["feedback"]["value"]);
            assert_eq!(
                result[0].rationale,
                malformed["feedback"]["rationale"]
                    .as_str()
                    .unwrap_or_default()
            );
        }
        assert_eq!(
            script.requests.lock().unwrap().as_slice(),
            calls
                .iter()
                .map(|call| call["request"].clone())
                .collect::<Vec<_>>(),
            "{family}/{metric}/malformed ordered requests"
        );
    }
}
