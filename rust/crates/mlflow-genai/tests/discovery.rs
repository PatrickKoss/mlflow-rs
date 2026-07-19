use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Router;
use mlflow_genai::{
    execute_worker_request, JobKind, WorkerRequest, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION,
};
use serde_json::{json, Value};

#[derive(Clone)]
struct Script {
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    completions: Arc<Mutex<VecDeque<Value>>>,
}

fn completion(content: impl Into<String>) -> Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": content.into()}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "response_cost": 0.1,
    })
}

async fn scripted(State(script): State<Script>, request: Request) -> Response {
    let path = request.uri().path().to_string();
    let body = to_bytes(request.into_body(), 1024 * 1024).await.unwrap();
    let value = if path == "/ajax-api/2.0/mlflow/upload-artifact" {
        Value::String(String::from_utf8(body.to_vec()).unwrap())
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };
    script.calls.lock().unwrap().push((path.clone(), value));
    if path == "/gateway/mlflow/v1/chat/completions" {
        return axum::Json(script.completions.lock().unwrap().pop_front().unwrap()).into_response();
    }
    match path.as_str() {
        "/api/3.0/mlflow/traces/batchGet" => axum::Json(json!({
            "traces": [{
                "trace_info": {
                    "trace_id": "tr-1",
                    "trace_location": {"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": "0"}},
                    "request_time": "2026-01-01T00:00:00Z",
                    "execution_duration": "1.5s",
                    "trace_metadata": {}, "tags": {}, "assessments": []
                },
                "spans": [{
                    "span_id": "root", "name": "agent",
                    "attributes": [
                        {"key": "mlflow.spanInputs", "value": {"string_value": "{\"question\":\"capital?\"}"}},
                        {"key": "mlflow.spanOutputs", "value": {"string_value": "\"London\""}},
                        {"key": "mlflow.spanType", "value": {"string_value": "\"CHAIN\""}}
                    ],
                    "status": {"code": "STATUS_CODE_OK"}
                }]
            }]
        }))
        .into_response(),
        "/api/3.0/mlflow/issues" => axum::Json(json!({
            "issue": {
                "issue_id": "iss-1", "experiment_id": "0", "name": "Incorrect capital answer",
                "description": "The response names the wrong capital.", "status": "pending",
                "severity": "high", "root_causes": ["agent response generation"],
                "source_run_id": "run-1", "categories": ["correctness"]
            }
        }))
        .into_response(),
        _ => (StatusCode::OK, axum::Json(json!({"assessment": {}}))).into_response(),
    }
}

async fn server() -> (String, Script) {
    let script = Script {
        calls: Arc::new(Mutex::new(Vec::new())),
        completions: Arc::new(Mutex::new(VecDeque::from([
            completion(
                r#"{"result":{"passed":"false","categories":"correctness"},"rationale":"[wrong capital] London is not France's capital."}"#,
            ),
            completion(
                r#"{"result":{"passed":"false","categories":"correctness"},"rationale":"[wrong capital] London is not France's capital."}"#,
            ),
            completion("returned the wrong capital for France"),
            completion(
                r#"{"name":"Issue: Incorrect capital answer","description":"The response names the wrong capital.","root_cause":"agent response generation","example_indices":[0],"severity":"high","categories":["correctness"],"category_rationale":"correctness: London is not France's capital."}"#,
            ),
            completion(
                "The user asked for a capital, but the agent returned London instead of Paris.",
            ),
        ]))),
    };
    let app = Router::new().fallback(scripted).with_state(script.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), script)
}

#[tokio::test]
async fn scripted_issue_discovery_persists_python_contract() {
    let golden: Value =
        serde_json::from_str(include_str!("fixtures/issue_discovery_golden.json")).unwrap();
    let expected = &golden["e2e"];
    let (url, script) = server().await;
    let request = WorkerRequest {
        protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
        job_id: "job-1".to_string(),
        job_kind: JobKind::InvokeIssueDetection,
        params: json!({
            "experiment_id": "0", "trace_ids": ["tr-1"], "categories": ["correctness"],
            "run_id": "run-1", "model": "openai:/fake-chat",
            "tracking_url": url, "gateway_url": url,
        }),
        workspace: None,
        subject: Value::Null,
    };
    let response = execute_worker_request(&request).await;
    let WorkerResponse::Succeeded {
        result,
        status_details,
        ..
    } = response
    else {
        panic!("discovery failed: {response:?}");
    };
    assert_eq!(result, expected["result"]);
    assert_eq!(status_details.as_deref(), Some(&expected["status_details"]));

    let calls = script.calls.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .filter(|(path, _)| path == "/gateway/mlflow/v1/chat/completions")
            .count(),
        expected["model_calls"].as_u64().unwrap() as usize
    );
    let issue = calls
        .iter()
        .find(|(path, _)| path == "/api/3.0/mlflow/issues")
        .unwrap();
    assert_eq!(issue.1["severity"], "high");
    assert_eq!(issue.1["categories"], json!(["correctness"]));
    assert_eq!(issue.1["root_causes"], json!(["agent response generation"]));
    assert_eq!(issue.1["source_run_id"], "run-1");

    let issue_assessment = calls.iter().find(|(path, body)| {
        path == "/api/3.0/mlflow/traces/tr-1/assessments"
            && body.pointer("/assessment/issue").is_some()
    });
    assert_eq!(
        issue_assessment
            .unwrap()
            .1
            .pointer("/assessment/assessment_name"),
        Some(&json!("iss-1"))
    );
    let affected_trace_ids = calls
        .iter()
        .filter(|(path, body)| {
            path.ends_with("/assessments") && body.pointer("/assessment/issue").is_some()
        })
        .filter_map(|(path, _)| path.split('/').nth_back(1))
        .collect::<Vec<_>>();
    assert_eq!(json!(affected_trace_ids), expected["affected_trace_ids"]);
    assert_eq!(
        calls
            .iter()
            .filter(|(path, _)| path.ends_with("/assessments"))
            .count(),
        expected["assessment_count"].as_u64().unwrap() as usize
    );
    assert_eq!(
        calls
            .iter()
            .filter(|(path, _)| path == "/ajax-api/2.0/mlflow/upload-artifact")
            .count(),
        expected["artifact_count"].as_u64().unwrap() as usize
    );
    let artifacts = calls
        .iter()
        .filter(|(path, _)| path == "/ajax-api/2.0/mlflow/upload-artifact")
        .map(|(_, body)| body.as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        artifacts[0],
        expected["result"]["summary"].as_str().unwrap()
    );
    assert_eq!(
        serde_json::from_str::<Value>(artifacts[1]).unwrap(),
        json!([{
            "issue_id": "iss-1", "name": "Incorrect capital answer",
            "description": "The response names the wrong capital.",
            "root_causes": ["agent response generation"], "severity": "high",
            "status": "pending",
        }])
    );
    let mut metadata = serde_json::from_str::<Value>(artifacts[2]).unwrap();
    metadata.as_object_mut().unwrap().remove("elapsed_seconds");
    assert_eq!(
        metadata,
        json!({
            "total_traces_analyzed": 1, "num_issues": 1, "model": "openai:/fake-chat",
            "scorer_names": ["_issue_discovery_judge"], "triage_run_id": "run-1",
            "max_issues": 20, "experiment_id": "0", "filter_string": null,
            "input_tokens": 8, "output_tokens": 4, "total_tokens": 12, "cost_usd": 0.4,
        })
    );
    assert!(calls.iter().any(|(path, body)| {
        path == "/api/2.0/mlflow/runs/set-tag"
            && body == &json!({"run_id": "run-1", "key": "total_cost_usd", "value": "0.4"})
    }));
    assert!(calls.iter().any(|(path, body)| {
        path == "/api/2.0/mlflow/runs/update"
            && body["run_id"] == "run-1"
            && body["status"] == "FINISHED"
            && body["end_time"].as_i64().is_some()
    }));
    let finished_at = calls
        .iter()
        .position(|(path, body)| {
            path == "/api/2.0/mlflow/runs/update" && body["status"] == "FINISHED"
        })
        .unwrap();
    let cost_tag_at = calls
        .iter()
        .position(|(path, body)| {
            path == "/api/2.0/mlflow/runs/set-tag" && body["key"] == "total_cost_usd"
        })
        .unwrap();
    assert!(finished_at < cost_tag_at);
    assert!(script.completions.lock().unwrap().is_empty());
}
