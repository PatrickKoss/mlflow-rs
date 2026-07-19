//! T20.1 Assistant route, session-store, localhost-gate, and SSE coverage.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, Request};
use axum::http::{header, Method, Response, StatusCode};
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};
use futures::{FutureExt, StreamExt};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::assistant::{
    AssistantEvent, AssistantProvider, AssistantProviderError, AssistantProviderRequest,
    AssistantRuntime,
};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tower::ServiceExt;

const PREFIX: &str = "/ajax-api/3.0/mlflow/assistant";
type ModelCalls = Arc<Mutex<Vec<(Option<String>, Option<String>)>>>;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

#[derive(Debug)]
struct ScriptedProvider {
    requests: Arc<Mutex<Vec<AssistantProviderRequest>>>,
    model_calls: ModelCalls,
}

impl AssistantProvider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }

    fn resolve_skills_path(&self, base_directory: &Path) -> PathBuf {
        base_directory.join(".scripted/skills")
    }

    fn check_connection(
        &self,
        _config: Option<Value>,
    ) -> BoxFuture<'static, Result<(), AssistantProviderError>> {
        async { Ok(()) }.boxed()
    }

    fn list_models(
        &self,
        base_url: Option<String>,
        api_key: Option<String>,
        _config: Option<Value>,
    ) -> BoxFuture<'static, Result<Vec<String>, AssistantProviderError>> {
        self.model_calls.lock().unwrap().push((base_url, api_key));
        async { Ok(vec!["fixture-a".to_string(), "fixture-b".to_string()]) }.boxed()
    }

    fn stream(&self, request: AssistantProviderRequest) -> BoxStream<'static, AssistantEvent> {
        self.requests.lock().unwrap().push(request);
        stream::iter(vec![
            AssistantEvent::new(
                "message",
                json!({"message": {"role": "assistant", "content": "hello"}}),
            ),
            AssistantEvent::new(
                "stream_event",
                json!({"event": {"type": "content_delta", "delta": {"text": "!"}}}),
            ),
            AssistantEvent::new(
                "permission_request",
                json!({"request_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "pwd"}}),
            ),
            AssistantEvent::new("interrupted", json!({"message": "Assistant was interrupted"})),
            AssistantEvent::new("error", json!({"error": "scripted error"})),
            AssistantEvent::new(
                "done",
                json!({"result": null, "session_id": "provider-session-1"}),
            ),
        ])
        .boxed()
    }
}

#[derive(Debug)]
struct FailingProvider {
    name: &'static str,
    health: AssistantProviderError,
    models: AssistantProviderError,
}

impl AssistantProvider for FailingProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn resolve_skills_path(&self, base_directory: &Path) -> PathBuf {
        base_directory.join(".fixture/skills")
    }

    fn check_connection(
        &self,
        _config: Option<Value>,
    ) -> BoxFuture<'static, Result<(), AssistantProviderError>> {
        let error = self.health.clone();
        async move { Err(error) }.boxed()
    }

    fn list_models(
        &self,
        _base_url: Option<String>,
        _api_key: Option<String>,
        _config: Option<Value>,
    ) -> BoxFuture<'static, Result<Vec<String>, AssistantProviderError>> {
        let error = self.models.clone();
        async move { Err(error) }.boxed()
    }

    fn stream(&self, _request: AssistantProviderRequest) -> BoxStream<'static, AssistantEvent> {
        stream::empty().boxed()
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    app: axum::Router,
    runtime: AssistantRuntime,
    requests: Arc<Mutex<Vec<AssistantProviderRequest>>>,
    model_calls: ModelCalls,
}

impl Fixture {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let db_path = directory.path().join("assistant.db");
        std::fs::copy(fixture_path(), &db_path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", db_path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let store =
            TrackingStore::new(db, directory.path().join("artifacts").display().to_string());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model_calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(ScriptedProvider {
            requests: requests.clone(),
            model_calls: model_calls.clone(),
        });
        let skills = directory.path().join("bundled-skills");
        std::fs::create_dir_all(skills.join("alpha")).unwrap();
        std::fs::write(skills.join("alpha/SKILL.md"), "---\nname: alpha\n---\n").unwrap();
        let home = directory.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let providers: Vec<Arc<dyn AssistantProvider>> = vec![
            provider,
            Arc::new(FailingProvider {
                name: "not_implemented",
                health: AssistantProviderError::NotImplemented("fixture no probe".to_string()),
                models: AssistantProviderError::NotImplemented(String::new()),
            }),
            Arc::new(FailingProvider {
                name: "cli_missing",
                health: AssistantProviderError::CliNotInstalled("fixture cli missing".to_string()),
                models: AssistantProviderError::CliNotInstalled("fixture cli missing".to_string()),
            }),
            Arc::new(FailingProvider {
                name: "auth_missing",
                health: AssistantProviderError::NotAuthenticated(
                    "fixture auth missing".to_string(),
                ),
                models: AssistantProviderError::NotConfigured("fixture models missing".to_string()),
            }),
        ];
        let runtime = AssistantRuntime::new(
            directory.path().join("sessions"),
            home.join(".mlflow/assistant/config.json"),
            skills,
            home,
            providers,
        );
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(AppState::new(store).with_assistant_runtime(runtime.clone())),
        );
        Self {
            _directory: directory,
            app,
            runtime,
            requests,
            model_calls,
        }
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        peer: IpAddr,
        extra_headers: &[(&str, &str)],
    ) -> Response<Body> {
        let mut builder = Request::builder()
            .method(method)
            .uri(path)
            .header(header::HOST, "localhost:5000");
        for (name, value) in extra_headers {
            builder = builder.header(*name, *value);
        }
        let body = match body {
            Some(body) => {
                builder = builder.header(header::CONTENT_TYPE, "application/json");
                Body::from(serde_json::to_vec(&body).unwrap())
            }
            None => Body::empty(),
        };
        let mut request = builder.body(body).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::new(peer, 4242)));
        self.app.clone().oneshot(request).await.unwrap()
    }

    async fn local(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, axum::http::HeaderMap, Bytes) {
        collect(
            self.request(method, path, body, IpAddr::V4(Ipv4Addr::LOCALHOST), &[])
                .await,
        )
        .await
    }

    async fn select_provider(&self) {
        let (status, _, _) = self
            .local(
                Method::PUT,
                &format!("{PREFIX}/config"),
                Some(json!({"providers": {"scripted": {"selected": true}}})),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
    }
}

async fn collect(response: Response<Body>) -> (StatusCode, axum::http::HeaderMap, Bytes) {
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, headers, body)
}

fn json_body(body: &[u8]) -> Value {
    serde_json::from_slice(body).unwrap()
}

#[tokio::test]
async fn localhost_gate_covers_all_nine_routes_and_accepts_ipv6_loopback() {
    let fixture = Fixture::new().await;
    let routes = [
        (Method::POST, format!("{PREFIX}/message")),
        (Method::GET, format!("{PREFIX}/sessions/id/stream")),
        (Method::PATCH, format!("{PREFIX}/sessions/id")),
        (Method::POST, format!("{PREFIX}/sessions/id/permission")),
        (Method::GET, format!("{PREFIX}/providers/id/health")),
        (Method::GET, format!("{PREFIX}/config")),
        (Method::PUT, format!("{PREFIX}/config")),
        (Method::POST, format!("{PREFIX}/skills/install")),
        (Method::GET, format!("{PREFIX}/providers/id/models")),
    ];
    for (method, path) in routes {
        let (status, _, body) = collect(
            fixture
                .request(
                    method,
                    &path,
                    None,
                    IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10)),
                    &[],
                )
                .await,
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
        assert_eq!(
            body,
            r#"{"detail":"Assistant API is only accessible from the same host where the MLflow server is running."}"#
        );
    }

    let (status, _, body) = collect(
        fixture
            .request(
                Method::GET,
                &format!("{PREFIX}/config"),
                None,
                IpAddr::V6(Ipv6Addr::LOCALHOST),
                &[],
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, r#"{"providers":{},"projects":{}}"#);
}

#[tokio::test]
async fn config_health_models_and_skills_routes_match_python_shapes() {
    let fixture = Fixture::new().await;
    let (status, _, body) = fixture
        .local(Method::GET, &format!("{PREFIX}/config"), None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, r#"{"providers":{},"projects":{}}"#);

    let project = fixture._directory.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    let update = json!({
        "providers": {"scripted": {"model": "fixture", "selected": true, "permissions": {"full_access": true}}},
        "projects": {"7": {"location": project}},
    });
    let (status, _, body) = fixture
        .local(Method::PUT, &format!("{PREFIX}/config"), Some(update))
        .await;
    assert_eq!(status, StatusCode::OK);
    let config = json_body(&body);
    assert_eq!(config["providers"]["scripted"]["model"], "fixture");
    assert_eq!(config["providers"]["scripted"]["selected"], true);
    assert_eq!(
        config["providers"]["scripted"]["permissions"],
        json!({"allow_edit_files": true, "allow_read_docs": true, "full_access": true})
    );
    assert_eq!(config["projects"]["7"]["type"], "local");

    let (status, _, body) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/providers/scripted/health"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, r#"{"status":"ok"}"#);
    let (status, _, body) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/providers/missing/health"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"detail":"Provider 'missing' not found"}"#);
    for (provider, status, detail) in [
        (
            "not_implemented",
            StatusCode::NOT_IMPLEMENTED,
            "fixture no probe",
        ),
        (
            "cli_missing",
            StatusCode::PRECONDITION_FAILED,
            "fixture cli missing",
        ),
        (
            "auth_missing",
            StatusCode::UNAUTHORIZED,
            "fixture auth missing",
        ),
    ] {
        let (actual, _, body) = fixture
            .local(
                Method::GET,
                &format!("{PREFIX}/providers/{provider}/health"),
                None,
            )
            .await;
        assert_eq!(actual, status);
        assert_eq!(json_body(&body), json!({"detail": detail}));
    }

    let response = fixture
        .request(
            Method::GET,
            &format!("{PREFIX}/providers/scripted/models?base_url=http%3A%2F%2Ffixture"),
            None,
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            &[("x-api-key", "obvious-fake-model-key")],
        )
        .await;
    let (status, _, body) = collect(response).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, r#"{"models":["fixture-a","fixture-b"]}"#);
    assert_eq!(
        fixture.model_calls.lock().unwrap().as_slice(),
        &[(
            Some("http://fixture".to_string()),
            Some("obvious-fake-model-key".to_string())
        )]
    );
    let (status, _, body) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/providers/not_implemented/models"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        json_body(&body),
        json!({"detail": "Model listing is not supported for provider 'not_implemented'"})
    );
    let (status, _, body) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/providers/auth_missing/models"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        json_body(&body),
        json!({"detail": "fixture models missing"})
    );

    let (status, _, body) = fixture
        .local(
            Method::POST,
            &format!("{PREFIX}/skills/install"),
            Some(json!({"type": "project", "experiment_id": "7"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let installed = json_body(&body);
    assert_eq!(installed["installed_skills"], json!(["alpha"]));
    assert_eq!(
        installed["skills_directory"],
        project.join(".scripted/skills").to_string_lossy().as_ref()
    );
    assert!(project.join(".scripted/skills/alpha/SKILL.md").is_file());
}

#[tokio::test]
async fn message_stream_permission_and_cancel_lifecycle_is_persistent() {
    let fixture = Fixture::new().await;
    fixture.select_provider().await;
    let (status, _, body) = fixture
        .local(
            Method::POST,
            &format!("{PREFIX}/message"),
            Some(json!({"message": "hello", "context": {"traceId": "tr-1"}})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let sent = json_body(&body);
    let session_id = sent["session_id"].as_str().unwrap();
    assert_eq!(
        sent["stream_url"],
        format!("{PREFIX}/sessions/{session_id}/stream")
    );
    let session_file = fixture
        .runtime
        .sessions()
        .root()
        .join(format!("{session_id}.json"));
    let stored = std::fs::read_to_string(&session_file).unwrap();
    assert_eq!(
        stored,
        format!(
            "{{\"context\": {{\"traceId\": \"tr-1\"}}, \"messages\": [{{\"role\": \"user\", \"content\": \"hello\"}}], \"pending_message\": {{\"role\": \"user\", \"content\": \"hello\"}}, \"provider_session_id\": null, \"working_dir\": null, \"pending_tool_decisions\": {{}}}}"
        )
    );

    let (status, headers, stream) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/sessions/{session_id}/stream"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers[header::CONTENT_TYPE],
        "text/event-stream; charset=utf-8"
    );
    assert_eq!(headers[header::CACHE_CONTROL], "no-cache");
    assert_eq!(headers["x-accel-buffering"], "no");
    assert_eq!(
        stream,
        concat!(
            "event: message\n",
            "data: {\"message\": {\"role\": \"assistant\", \"content\": \"hello\"}}\n\n",
            "event: stream_event\n",
            "data: {\"event\": {\"type\": \"content_delta\", \"delta\": {\"text\": \"!\"}}}\n\n",
            "event: permission_request\n",
            "data: {\"request_id\": \"tool-1\", \"tool_name\": \"Bash\", \"tool_input\": {\"command\": \"pwd\"}}\n\n",
            "event: interrupted\n",
            "data: {\"message\": \"Assistant was interrupted\"}\n\n",
            "event: error\n",
            "data: {\"error\": \"scripted error\"}\n\n",
            "event: done\n",
            "data: {\"result\": null, \"session_id\": \"provider-session-1\"}\n\n",
        )
    );
    {
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].prompt, "hello");
        assert_eq!(requests[0].tracking_uri, "http://localhost:5000");
        assert_eq!(requests[0].context["traceId"], "tr-1");
    }
    let session = fixture
        .runtime
        .sessions()
        .load(session_id)
        .unwrap()
        .unwrap();
    assert_eq!(
        session.provider_session_id.as_deref(),
        Some("provider-session-1")
    );
    assert!(session.pending_message.is_none());

    let (status, _, body) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/sessions/{session_id}/stream"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body, r#"{"detail":"No pending message to process"}"#);

    let (status, _, body) = fixture
        .local(
            Method::POST,
            &format!("{PREFIX}/sessions/{session_id}/permission"),
            Some(json!({"request_id": "tool-1", "decision": "allow"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json_body(&body)["stream_url"],
        format!("{PREFIX}/sessions/{session_id}/stream")
    );
    let (status, _, _) = fixture
        .local(
            Method::GET,
            &format!("{PREFIX}/sessions/{session_id}/stream"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    {
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].prompt, "");
        assert_eq!(
            requests[1].context["tool_decisions"],
            json!({"tool-1": "allow"})
        );
    }

    let mut child = Command::new("sleep").arg("30").spawn().unwrap();
    fixture
        .runtime
        .sessions()
        .save_process_pid(session_id, child.id() as i32)
        .unwrap();
    let (status, _, body) = fixture
        .local(
            Method::PATCH,
            &format!("{PREFIX}/sessions/{session_id}"),
            Some(json!({"status": "cancelled"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        r#"{"message":"Session cancelled and process terminated"}"#
    );
    child.wait().unwrap();
    assert!(!fixture
        .runtime
        .sessions()
        .root()
        .join(format!("{session_id}.process.json"))
        .exists());
}

#[tokio::test]
async fn fastapi_validation_and_session_errors_are_exact() {
    let fixture = Fixture::new().await;
    let cases = [
        (
            Method::POST,
            format!("{PREFIX}/message"),
            Some(json!({})),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail":[{"type":"missing","loc":["body","message"],"msg":"Field required","input":{}}]}),
        ),
        (
            Method::PATCH,
            format!("{PREFIX}/sessions/nope"),
            Some(json!({"status":"other"})),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail":[{"type":"literal_error","loc":["body","status"],"msg":"Input should be 'cancelled'","input":"other","ctx":{"expected":"'cancelled'"}}]}),
        ),
        (
            Method::POST,
            format!("{PREFIX}/sessions/nope/permission"),
            Some(json!({"request_id":"x","decision":"allow"})),
            StatusCode::BAD_REQUEST,
            json!({"detail":"Invalid session ID format"}),
        ),
        (
            Method::GET,
            format!("{PREFIX}/sessions/nope/stream"),
            None,
            StatusCode::NOT_FOUND,
            json!({"detail":"Session not found"}),
        ),
        (
            Method::POST,
            format!("{PREFIX}/skills/install"),
            Some(json!({"type":"custom"})),
            StatusCode::PRECONDITION_FAILED,
            json!({"detail":"No assistant provider is configured or available."}),
        ),
    ];
    for (method, path, request, expected_status, expected_body) in cases {
        let (status, _, body) = fixture.local(method, &path, request).await;
        assert_eq!(
            status,
            expected_status,
            "{path}: {}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(json_body(&body), expected_body, "{path}");
    }
}
