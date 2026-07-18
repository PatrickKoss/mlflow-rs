use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, EndpointModelConfig, PoolConfig, TrackingStore, WORKSPACE_DEFAULT_NAME};
use serde_json::{json, Value};
use tokio::net::TcpListener;

#[derive(Debug, Clone)]
struct RecordedCall {
    path: String,
    headers: HeaderMap,
    body: Value,
}

#[derive(Clone)]
struct MockState {
    calls: Arc<Mutex<Vec<RecordedCall>>>,
}

struct Fixture {
    _directory: tempfile::TempDir,
    store: TrackingStore,
    server_base: String,
    mock_base: String,
    calls: Arc<Mutex<Vec<RecordedCall>>>,
    inbound: Arc<Mutex<Vec<RecordedCall>>>,
    target_model_id: String,
    judge_pass_endpoint: String,
    judge_violation_endpoint: String,
    sanitizer_endpoint_id: String,
    experiment_id: String,
}

impl Fixture {
    async fn new() -> Self {
        let mock_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mock_base = format!("http://{}", mock_listener.local_addr().unwrap());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mock_state = MockState {
            calls: calls.clone(),
        };
        tokio::spawn(async move {
            axum::serve(
                mock_listener,
                Router::new()
                    .fallback(post(mock_endpoint))
                    .with_state(mock_state),
            )
            .await
            .unwrap();
        });

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("guardrails.db");
        std::fs::copy(fixture_path(), &path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let store =
            TrackingStore::new(db, directory.path().join("artifacts").display().to_string());
        let secret = store
            .create_gateway_secret(
                WORKSPACE_DEFAULT_NAME,
                "obvious-fake-guardrail-secret",
                &HashMap::from([(
                    "api_key".to_string(),
                    "obvious-fake-guardrail-key".to_string(),
                )]),
                Some("openai"),
                &HashMap::from([
                    ("auth_mode".to_string(), "api_key".to_string()),
                    ("api_base".to_string(), format!("{mock_base}/v1")),
                ]),
                Some("guardrail-test"),
            )
            .await
            .unwrap();
        let target_model_id = create_model(&store, &secret.secret_id, "target-model").await;
        let pass_model = create_model(&store, &secret.secret_id, "judge-pass-model").await;
        let violation_model =
            create_model(&store, &secret.secret_id, "judge-violation-model").await;
        let sanitizer_model = create_model(&store, &secret.secret_id, "sanitizer-model").await;
        let judge_pass_endpoint = "guardrail-judge-pass".to_string();
        create_endpoint(&store, &judge_pass_endpoint, &pass_model).await;
        let judge_violation_endpoint = "guardrail-judge-violation".to_string();
        create_endpoint(&store, &judge_violation_endpoint, &violation_model).await;
        let sanitizer = create_endpoint(&store, "sanitizer", &sanitizer_model).await;
        let experiment_id = store
            .create_experiment(WORKSPACE_DEFAULT_NAME, "guardrail-runtime-tests", None, &[])
            .await
            .unwrap();

        let inbound = Arc::new(Mutex::new(Vec::new()));
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(AppState::new(store.clone())),
        )
        .layer(middleware::from_fn_with_state(
            inbound.clone(),
            record_inbound,
        ));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        Self {
            _directory: directory,
            store,
            server_base,
            mock_base,
            calls,
            inbound,
            target_model_id,
            judge_pass_endpoint,
            judge_violation_endpoint,
            sanitizer_endpoint_id: sanitizer,
            experiment_id,
        }
    }

    async fn matrix_endpoint(&self, stage: &str, action: &str, outcome: &str) -> String {
        let endpoint_name = format!(
            "matrix-{}-{}-{outcome}",
            stage.to_ascii_lowercase(),
            action.to_ascii_lowercase()
        );
        let endpoint_id = create_endpoint(&self.store, &endpoint_name, &self.target_model_id).await;
        let judge_endpoint = if outcome == "pass" {
            &self.judge_pass_endpoint
        } else {
            &self.judge_violation_endpoint
        };
        self.add_guardrail(
            &endpoint_id,
            &endpoint_name,
            stage,
            action,
            judge_endpoint,
            Some(1),
        )
        .await;
        endpoint_name
    }

    #[allow(clippy::too_many_arguments)]
    async fn add_guardrail(
        &self,
        endpoint_id: &str,
        name: &str,
        stage: &str,
        action: &str,
        judge_endpoint: &str,
        order: Option<i64>,
    ) {
        let scorer = self
            .store
            .register_scorer(
                WORKSPACE_DEFAULT_NAME,
                &self.experiment_id,
                &format!("scorer-{name}"),
                &instructions_scorer(name, stage, judge_endpoint).to_string(),
            )
            .await
            .unwrap();
        let action_endpoint =
            (action == "SANITIZATION").then_some(self.sanitizer_endpoint_id.as_str());
        let guardrail = self
            .store
            .create_gateway_guardrail(
                WORKSPACE_DEFAULT_NAME,
                name,
                &scorer.scorer_id,
                scorer.scorer_version,
                stage,
                action,
                action_endpoint,
                Some("guardrail-test"),
            )
            .await
            .unwrap();
        self.store
            .add_guardrail_to_endpoint(
                WORKSPACE_DEFAULT_NAME,
                endpoint_id,
                &guardrail.guardrail_id,
                order,
                Some("guardrail-test"),
            )
            .await
            .unwrap();
    }

    async fn post(&self, endpoint: &str, stream: bool) -> (StatusCode, String) {
        let response = reqwest::Client::new()
            .post(format!(
                "{}/gateway/{endpoint}/mlflow/invocations",
                self.server_base
            ))
            .header(
                header::AUTHORIZATION,
                "Bearer obvious-fake-client-authorization",
            )
            .json(&json!({
                "messages": [{"role":"user", "content":"unsafe input"}],
                "stream": stream,
            }))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.text().await.unwrap();
        (status, body)
    }

    fn take_calls(&self) -> Vec<RecordedCall> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

async fn create_model(store: &TrackingStore, secret_id: &str, model: &str) -> String {
    store
        .create_gateway_model_definition(
            WORKSPACE_DEFAULT_NAME,
            &format!("definition-{model}"),
            secret_id,
            "openai",
            model,
            Some("guardrail-test"),
        )
        .await
        .unwrap()
        .model_definition_id
}

async fn create_endpoint(store: &TrackingStore, name: &str, model_id: &str) -> String {
    store
        .create_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            name,
            &[EndpointModelConfig {
                model_definition_id: model_id.to_string(),
                linkage_type: "PRIMARY".to_string(),
                weight: 1.0,
                fallback_order: None,
            }],
            Some("guardrail-test"),
            None,
            None,
            None,
            false,
        )
        .await
        .unwrap()
        .endpoint_id
}

fn instructions_scorer(name: &str, stage: &str, endpoint: &str) -> Value {
    let field = if stage == "BEFORE" {
        "inputs"
    } else {
        "outputs"
    };
    json!({
        "name": name,
        "aggregations": null,
        "description": null,
        "is_session_level_scorer": false,
        "mlflow_version": "obvious-fake-test-version",
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {
            "instructions": format!("Evaluate {{{{ {field} }}}}."),
            "model": format!("gateway:/{endpoint}"),
            "feedback_value_type": {"title":"Result", "type":"string"},
        },
    })
}

async fn record_inbound(
    State(records): State<Arc<Mutex<Vec<RecordedCall>>>>,
    request: Request,
    next: Next,
) -> Response {
    if !request.uri().path().contains("/gateway/sanitizer/") {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap();
    records.lock().unwrap().push(RecordedCall {
        path: parts.uri.path().to_string(),
        headers: parts.headers.clone(),
        body,
    });
    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

async fn mock_endpoint(State(state): State<MockState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = body.collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let path = parts.uri.path().to_string();
    state.calls.lock().unwrap().push(RecordedCall {
        path: path.clone(),
        headers: parts.headers.clone(),
        body: body.clone(),
    });

    if path == "/judge/pass" || path == "/judge/violation" {
        let passing = path.ends_with("pass");
        return json_response(json!({
            "result": if passing { "yes" } else { "no" },
            "rationale": if passing { "allowed" } else { "fixture violation" },
        }));
    }
    if path.contains("/gateway/sanitizer/") {
        return sanitizer_response(&body);
    }

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model == "judge-pass-model" || model == "judge-violation-model" {
        let passing = model == "judge-pass-model";
        let content = json!({
            "result": if passing { "yes" } else { "no" },
            "rationale": if passing { "allowed" } else { "fixture violation" },
        })
        .to_string();
        return openai_response(model, &content);
    }
    if model == "sanitizer-model" {
        return sanitizer_response(&body);
    }
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        return Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(
                "data: {\"id\":\"stream-id\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"target-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fixture answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
            ))
            .unwrap();
    }
    openai_response("target-model", "fixture answer")
}

fn sanitizer_response(body: &Value) -> Response {
    let prompt = body
        .pointer("/messages/0/content")
        .and_then(Value::as_str)
        .unwrap();
    let (_, payload) = prompt.rsplit_once("Input payload:\n").unwrap();
    let mut payload: Value = serde_json::from_str(payload).unwrap();
    let request_payload = payload.get("messages").is_some();
    let pointer = if request_payload {
        "/messages/0/content"
    } else {
        "/choices/0/message/content"
    };
    if let Some(content) = payload.pointer_mut(pointer) {
        *content = Value::String(
            if request_payload {
                "cleaned input"
            } else {
                "cleaned output"
            }
            .to_string(),
        );
    }
    openai_response("sanitizer-model", &payload.to_string())
}

fn openai_response(model: &str, content: &str) -> Response {
    json_response(json!({
        "id":"openai-fixture-id",
        "object":"chat.completion",
        "created":7,
        "model":model,
        "choices":[{
            "index":0,
            "message":{"role":"assistant", "content":content},
            "finish_reason":"stop",
        }],
        "usage":{"prompt_tokens":2, "completion_tokens":3, "total_tokens":5},
    }))
}

fn json_response(value: Value) -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}

fn python_oracle(mock_base: &str) -> Value {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repository root");
    let output = Command::new("uv")
        .args([
            "run",
            "--frozen",
            "python",
            "rust/tools/guardrail_oracle.py",
        ])
        .env("MLFLOW_GUARDRAIL_MOCK_URL", mock_base)
        .current_dir(repository)
        .output()
        .expect("run Python guardrail oracle");
    assert!(
        output.status.success(),
        "Python guardrail oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn guardrail_matrix_is_byte_identical_to_python() {
    let fixture = Fixture::new().await;
    let oracle = python_oracle(&fixture.mock_base);
    let python_sanitization_payloads = fixture
        .take_calls()
        .into_iter()
        .filter(|call| call.path == "/gateway/sanitizer/mlflow/invocations")
        .map(|call| call.body)
        .collect::<Vec<_>>();
    for stage in ["BEFORE", "AFTER"] {
        for action in ["VALIDATION", "SANITIZATION"] {
            for outcome in ["pass", "violation"] {
                let endpoint = fixture.matrix_endpoint(stage, action, outcome).await;
                let (status, body) = fixture.post(&endpoint, false).await;
                let key = format!("{stage}-{action}-{outcome}");
                assert_eq!(u64::from(status.as_u16()), oracle[&key]["status"]);
                assert_eq!(body, oracle[&key]["body"].as_str().unwrap(), "{key}");
            }
        }
    }
    let rust_sanitization_payloads = fixture
        .inbound
        .lock()
        .unwrap()
        .iter()
        .map(|call| call.body.clone())
        .collect::<Vec<_>>();
    assert_eq!(rust_sanitization_payloads, python_sanitization_payloads);
}

#[tokio::test(flavor = "multi_thread")]
async fn ordering_chaining_short_circuit_and_sanitization_bypass_are_pinned() {
    let fixture = Fixture::new().await;
    let endpoint = create_endpoint(&fixture.store, "ordered", &fixture.target_model_id).await;
    fixture
        .add_guardrail(
            &endpoint,
            "order-1-pass",
            "BEFORE",
            "VALIDATION",
            &fixture.judge_pass_endpoint,
            Some(1),
        )
        .await;
    fixture
        .add_guardrail(
            &endpoint,
            "order-2-sanitize",
            "BEFORE",
            "SANITIZATION",
            &fixture.judge_violation_endpoint,
            Some(2),
        )
        .await;
    fixture
        .add_guardrail(
            &endpoint,
            "order-3-pass",
            "BEFORE",
            "VALIDATION",
            &fixture.judge_pass_endpoint,
            Some(3),
        )
        .await;

    let (status, _) = fixture.post("ordered", false).await;
    assert_eq!(status, StatusCode::OK);
    let calls = fixture.take_calls();
    let models = calls
        .iter()
        .filter_map(|call| call.body.get("model").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        models,
        [
            "judge-pass-model",
            "judge-violation-model",
            "sanitizer-model",
            "judge-pass-model",
            "target-model",
        ]
    );
    let third_judge = calls
        .iter()
        .filter(|call| call.body["model"] == "judge-pass-model")
        .nth(1)
        .unwrap();
    assert!(third_judge.body["messages"][1]["content"]
        .as_str()
        .unwrap()
        .contains("cleaned input"));
    let target = calls
        .iter()
        .find(|call| call.body["model"] == "target-model")
        .unwrap();
    assert_eq!(target.body["messages"][0]["content"], "cleaned input");

    {
        let inbound = fixture.inbound.lock().unwrap();
        let sanitization = inbound
            .iter()
            .find(|call| call.path == "/gateway/sanitizer/mlflow/invocations")
            .unwrap();
        assert_eq!(sanitization.headers["x-mlflow-guardrail-bypass"], "1");
        assert_eq!(
            sanitization.headers[header::AUTHORIZATION],
            "Bearer obvious-fake-client-authorization"
        );
    }

    let blocked = create_endpoint(&fixture.store, "short-circuit", &fixture.target_model_id).await;
    fixture
        .add_guardrail(
            &blocked,
            "short-1-block",
            "BEFORE",
            "VALIDATION",
            &fixture.judge_violation_endpoint,
            Some(1),
        )
        .await;
    fixture
        .add_guardrail(
            &blocked,
            "short-2-never",
            "BEFORE",
            "VALIDATION",
            &fixture.judge_pass_endpoint,
            Some(2),
        )
        .await;
    fixture.take_calls();
    let (status, body) = fixture.post("short-circuit", false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body,
        "{\"detail\":\"Guardrail 'short-1-block' blocked: fixture violation\"}"
    );
    assert_eq!(
        fixture
            .take_calls()
            .iter()
            .filter_map(|call| call.body.get("model").and_then(Value::as_str))
            .collect::<Vec<_>>(),
        ["judge-violation-model"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_run_before_guardrails_and_skip_after_guardrails() {
    let fixture = Fixture::new().await;
    let before = fixture
        .matrix_endpoint("BEFORE", "VALIDATION", "violation")
        .await;
    let (status, body) = fixture.post(&before, true).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        format!(
            "data: {{\"error\": {{\"message\": \"Guardrail '{before}' blocked: fixture violation\", \"type\": \"GuardrailViolation\"}}}}\n\n"
        )
    );
    assert_eq!(
        fixture
            .take_calls()
            .iter()
            .filter_map(|call| call.body.get("model").and_then(Value::as_str))
            .collect::<Vec<_>>(),
        ["judge-violation-model"]
    );

    let after = fixture
        .matrix_endpoint("AFTER", "VALIDATION", "violation")
        .await;
    let (status, body) = fixture.post(&after, true).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("fixture answer"));
    let calls = fixture.take_calls();
    assert!(calls
        .iter()
        .any(|call| call.body["model"] == "target-model"));
    assert!(!calls
        .iter()
        .any(|call| call.body["model"] == "judge-violation-model"));
}
