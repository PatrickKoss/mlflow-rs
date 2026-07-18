//! HTTP parity for the four ajax discovery routes and legacy gateway bridge.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn tracking_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

fn auth_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates directory")
        .join("mlflow-auth/tests/fixtures/basic_auth.db")
}

struct TempDb(PathBuf);

impl TempDb {
    fn new(tag: &str) -> Self {
        Self::from_source(tag, &tracking_fixture_path())
    }

    fn from_source(tag: &str, source: &Path) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_gateway_discovery_{tag}_{}_{}.db",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(source, &path).expect("copy DB fixture");
        Self(path)
    }

    fn uri(&self) -> String {
        format!("sqlite:///{}", self.0.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    _db: TempDb,
    _auth_db: Option<TempDb>,
}

impl TestServer {
    async fn start(tag: &str, populated_gateway: bool) -> Self {
        Self::start_with_modes(tag, populated_gateway, false, false).await
    }

    async fn start_with_modes(
        tag: &str,
        populated_gateway: bool,
        auth_enabled: bool,
        workspaces_enabled: bool,
    ) -> Self {
        let db_file = TempDb::new(tag);
        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect tracking fixture");
        let store = TrackingStore::new(db.clone(), ART_ROOT);
        if populated_gateway {
            let secret = store
                .create_gateway_secret(
                    "default",
                    "obvious-fake-discovery-secret",
                    &HashMap::from([(
                        "api_key".to_string(),
                        "obvious-fake-discovery-value".to_string(),
                    )]),
                    Some("openai"),
                    &HashMap::new(),
                    Some("test-user"),
                )
                .await
                .expect("seed fake gateway secret");
            store
                .create_gateway_model_definition(
                    "default",
                    "fake-discovery-model-definition",
                    &secret.secret_id,
                    "openai",
                    "fake-discovery-model",
                    Some("test-user"),
                )
                .await
                .expect("seed fake model definition");
        }

        let mut state = AppState::new(store);
        if workspaces_enabled {
            state = state.with_workspace_store(WorkspaceStore::new(db, db_file.uri()));
        }
        let auth_db = if auth_enabled {
            let auth_db = TempDb::from_source(&format!("{tag}_auth"), &auth_fixture_path());
            let db = AuthDb::connect_and_verify_with(&auth_db.uri(), None, PoolConfig::default())
                .await
                .expect("connect auth fixture");
            state = state.with_auth_store(AuthStore::new(db));
            Some(auth_db)
        } else {
            None
        };

        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            serve_artifacts: true,
            x_frame_options: "SAMEORIGIN".to_string(),
            ..Default::default()
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(&config, recorder, Some(state));
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let address = listener.local_addr().expect("listener address");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve test app");
        });
        Self {
            base: format!("http://{address}"),
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
            _db: db_file,
            _auth_db: auth_db,
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

struct HttpResult {
    status: StatusCode,
    content_type: Option<String>,
    headers: hyper::HeaderMap,
    body: String,
}

async fn request(
    server: &TestServer,
    method: Method,
    path: &str,
    body: &str,
    headers: &[(&str, &str)],
) -> HttpResult {
    let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
    let mut builder = Request::builder()
        .method(method)
        .uri(format!("{}{}", server.base, path));
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let response = client
        .request(
            builder
                .body(Full::new(Bytes::copy_from_slice(body.as_bytes())))
                .expect("request"),
        )
        .await
        .expect("HTTP request");
    let status = response.status();
    let headers = response.headers().clone();
    let content_type = headers
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    HttpResult {
        status,
        content_type,
        headers,
        body: String::from_utf8(bytes.to_vec()).expect("UTF-8 response"),
    }
}

async fn get(server: &TestServer, path: &str) -> HttpResult {
    request(server, Method::GET, path, "", &[]).await
}

fn sha256(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn set_env(name: &str, value: Option<&str>) {
    // SAFETY: every test in this binary that mutates process environment holds
    // ENV_LOCK, and no spawned task mutates it.
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

#[tokio::test]
async fn discovery_json_matches_python_with_empty_and_populated_gateway_stores() {
    let _env = ENV_LOCK.lock().await;
    set_env("MLFLOW_MODEL_CATALOG_URI", Some(""));
    set_env("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", None);
    set_env("MLFLOW_CRYPTO_KEK_PASSPHRASE", None);

    for populated in [false, true] {
        let server =
            TestServer::start(if populated { "populated" } else { "empty" }, populated).await;
        let cases = [
            (
                "/ajax-api/3.0/mlflow/gateway/supported-providers",
                1045,
                "6d587556c256ec1f329523a661d4a1793168cd9000caac3b4e9affaf0fef0b13",
            ),
            (
                "/ajax-api/3.0/mlflow/gateway/supported-models?provider=openai",
                40659,
                "df682c8f907d32c553dd3044730a3c9c0e5c4323adf35b0c72e610842130bac1",
            ),
            (
                "/ajax-api/3.0/mlflow/gateway/supported-models",
                1020938,
                "595d94ecffd1b7ffba2c1cbbf4ffce0bafb9bb4f56e0a3cfac97f84f12b024a8",
            ),
            (
                "/ajax-api/3.0/mlflow/gateway/provider-config?provider=openai",
                328,
                "4b82bd5af08109b7a5d716174a1f19478b4b13a2c84f357732cb2208e900c61d",
            ),
            (
                "/ajax-api/3.0/mlflow/gateway/secrets/config",
                59,
                "d7d3577cde36ce55951e86380339668d381c490e351854a82806a62b089edba2",
            ),
        ];
        for (path, python_len, python_sha256) in cases {
            let result = get(&server, path).await;
            assert_eq!(result.status, StatusCode::OK, "{path}: {}", result.body);
            assert_eq!(result.content_type.as_deref(), Some("application/json"));
            assert_eq!(result.body.len(), python_len, "{path}");
            assert_eq!(sha256(&result.body), python_sha256, "{path}");
        }
    }

    set_env("MLFLOW_MODEL_CATALOG_URI", None);
}

#[tokio::test]
async fn discovery_empty_and_validation_quirks_match_python() {
    let _env = ENV_LOCK.lock().await;
    set_env("MLFLOW_MODEL_CATALOG_URI", Some(""));
    let server = TestServer::start("discovery_edges", false).await;

    let models = get(
        &server,
        "/ajax-api/3.0/mlflow/gateway/supported-models?provider=does-not-exist",
    )
    .await;
    assert_eq!(models.status, StatusCode::OK);
    assert_eq!(models.body, "{\"models\":[]}\n");

    let missing = get(&server, "/ajax-api/3.0/mlflow/gateway/provider-config").await;
    assert_eq!(missing.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        missing.body,
        r#"{"error_code": "INVALID_PARAMETER_VALUE", "message": "Provider parameter is required", "sqlstate": "KAM00", "error_class": "INVALID_PARAMETER_VALUE"}"#
    );

    set_env("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", Some("anthropic"));
    let providers = get(&server, "/ajax-api/3.0/mlflow/gateway/supported-providers").await;
    assert_eq!(providers.body, "{\"providers\":[\"anthropic\"]}\n");
    let blocked = get(
        &server,
        "/ajax-api/3.0/mlflow/gateway/provider-config?provider=openai",
    )
    .await;
    assert_eq!(blocked.status, StatusCode::BAD_REQUEST);
    assert!(blocked.body.contains("Provider 'openai' is not allowed"));

    set_env("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", None);
    set_env("MLFLOW_MODEL_CATALOG_URI", None);
}

#[tokio::test]
async fn discovery_routes_are_authenticated_only_and_workspace_resolved() {
    let _env = ENV_LOCK.lock().await;
    set_env("MLFLOW_MODEL_CATALOG_URI", Some(""));

    let auth_server = TestServer::start_with_modes("discovery_auth", false, true, false).await;
    let unauthenticated = get(
        &auth_server,
        "/ajax-api/3.0/mlflow/gateway/supported-providers",
    )
    .await;
    assert_eq!(unauthenticated.status, StatusCode::UNAUTHORIZED);
    let credentials =
        base64::engine::general_purpose::STANDARD.encode("bob_pbkdf2:bob-password-4567");
    let authorization = format!("Basic {credentials}");
    let authenticated = request(
        &auth_server,
        Method::GET,
        "/ajax-api/3.0/mlflow/gateway/supported-providers",
        "",
        &[("authorization", &authorization)],
    )
    .await;
    assert_eq!(authenticated.status, StatusCode::OK);

    let workspace_server =
        TestServer::start_with_modes("discovery_workspace", false, false, true).await;
    let missing_workspace = request(
        &workspace_server,
        Method::GET,
        "/ajax-api/3.0/mlflow/gateway/secrets/config",
        "",
        &[("x-mlflow-workspace", "does-not-exist")],
    )
    .await;
    assert_eq!(missing_workspace.status, StatusCode::NOT_FOUND);
    let default_workspace = request(
        &workspace_server,
        Method::GET,
        "/ajax-api/3.0/mlflow/gateway/secrets/config",
        "",
        &[("x-mlflow-workspace", "default")],
    )
    .await;
    assert_eq!(default_workspace.status, StatusCode::OK);
    assert_eq!(
        default_workspace.body,
        "{\"secrets_available\":true,\"using_default_passphrase\":true}\n"
    );

    set_env("MLFLOW_MODEL_CATALOG_URI", None);
}

#[tokio::test]
async fn proxy_unset_target_bypasses_parsing_and_validation() {
    let _env = ENV_LOCK.lock().await;
    set_env("MLFLOW_DEPLOYMENTS_TARGET", None);
    let server = TestServer::start("proxy_empty", false).await;

    let get_result = get(
        &server,
        "/ajax-api/2.0/mlflow/gateway-proxy?gateway_path=invalid",
    )
    .await;
    assert_eq!(get_result.status, StatusCode::OK);
    assert_eq!(get_result.body, "{\"endpoints\":[]}\n");

    let post_result = request(
        &server,
        Method::POST,
        "/ajax-api/2.0/mlflow/gateway-proxy",
        "not-json",
        &[],
    )
    .await;
    assert_eq!(post_result.status, StatusCode::OK);
    assert_eq!(post_result.body, "{\"endpoints\":[]}\n");
}

#[tokio::test]
async fn proxy_validation_matches_python_before_connecting() {
    let _env = ENV_LOCK.lock().await;
    set_env("MLFLOW_DEPLOYMENTS_TARGET", Some("http://127.0.0.1:1"));
    let server = TestServer::start("proxy_validation", false).await;

    for (path, message) in [
        (
            "/ajax-api/2.0/mlflow/gateway-proxy",
            "Deployments proxy request must specify a gateway_path.",
        ),
        (
            "/ajax-api/2.0/mlflow/gateway-proxy?gateway_path=nope",
            "Invalid gateway_path: nope for method: GET",
        ),
    ] {
        let result = get(&server, path).await;
        assert_eq!(result.status, StatusCode::BAD_REQUEST);
        assert!(result.body.contains(message), "{}", result.body);
    }

    for (body, message) in [
        (
            r#"{}"#,
            "Deployments proxy request must specify a gateway_path.",
        ),
        (
            r#"{"gateway_path":"gateway/a/b/invocations"}"#,
            "Invalid gateway_path: gateway/a/b/invocations for method: POST",
        ),
    ] {
        let result = request(
            &server,
            Method::POST,
            "/ajax-api/2.0/mlflow/gateway-proxy",
            body,
            &[("content-type", "application/json")],
        )
        .await;
        assert_eq!(result.status, StatusCode::BAD_REQUEST);
        assert!(result.body.contains(message), "{}", result.body);
    }

    let unsupported = request(
        &server,
        Method::POST,
        "/ajax-api/2.0/mlflow/gateway-proxy",
        "{}",
        &[],
    )
    .await;
    assert_eq!(unsupported.status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert!(unsupported
        .body
        .contains("Did not attempt to load JSON data"));

    set_env("MLFLOW_DEPLOYMENTS_TARGET", None);
}

#[derive(Debug)]
struct ForwardedRequest {
    method: Method,
    path: String,
    headers: hyper::HeaderMap,
    body: String,
}

async fn stub_handler(
    request: Request<Incoming>,
    tx: tokio::sync::mpsc::UnboundedSender<ForwardedRequest>,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let (parts, body) = request.into_parts();
    let bytes = body.collect().await.expect("stub request body").to_bytes();
    tx.send(ForwardedRequest {
        method: parts.method,
        path: parts.uri.path().to_string(),
        headers: parts.headers,
        body: String::from_utf8(bytes.to_vec()).expect("stub UTF-8 body"),
    })
    .expect("record forwarded request");
    let response = if parts.uri.path().contains("failure") {
        Response::builder()
            .status(StatusCode::IM_A_TEAPOT)
            .header("content-type", "text/plain")
            .body(Full::new(Bytes::from_static(b"stub says no")))
            .expect("stub error response")
    } else {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("x-stub", "not-forwarded")
            .body(Full::new(Bytes::from_static(
                br#"{"z":1,"a":{"nested":true}}"#,
            )))
            .expect("stub response")
    };
    Ok(response)
}

#[tokio::test]
async fn proxy_forwards_only_method_path_and_json_data_and_maps_responses() {
    let _env = ENV_LOCK.lock().await;
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind stub");
    let address = listener.local_addr().expect("stub address");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let stub = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        TokioIo::new(stream),
                        service_fn(move |request| stub_handler(request, tx.clone())),
                    )
                    .await;
            });
        }
    });
    set_env(
        "MLFLOW_DEPLOYMENTS_TARGET",
        Some(&format!("http://{address}/root")),
    );
    let server = TestServer::start("proxy_forward", false).await;

    let get_result = request(
        &server,
        Method::GET,
        "/ajax-api/2.0/mlflow/gateway-proxy?gateway_path=api%2F2.0%2Fendpoints&json_data=%7B%22x%22%3A1%7D",
        "",
        &[("authorization", "Bearer obvious-fake-token"), ("x-test", "private")],
    )
    .await;
    assert_eq!(get_result.status, StatusCode::OK);
    assert_eq!(get_result.body, "{\"a\":{\"nested\":true},\"z\":1}\n");
    assert!(!get_result.headers.contains_key("x-stub"));
    let forwarded_get = rx.recv().await.expect("forwarded GET");
    assert_eq!(forwarded_get.method, Method::GET);
    assert_eq!(forwarded_get.path, "/root/api/2.0/endpoints");
    assert_eq!(forwarded_get.body, r#""{\"x\":1}""#);
    assert!(!forwarded_get.headers.contains_key("authorization"));
    assert!(!forwarded_get.headers.contains_key("x-test"));

    let post_result = request(
        &server,
        Method::POST,
        "/ajax-api/2.0/mlflow/gateway-proxy",
        r#"{"gateway_path":"gateway/demo/invocations","json_data":{"messages":["hi"],"temperature":0.5},"ignored":"x"}"#,
        &[("content-type", "application/json"), ("x-test", "private")],
    )
    .await;
    assert_eq!(post_result.status, StatusCode::OK);
    assert_eq!(post_result.body, "{\"a\":{\"nested\":true},\"z\":1}\n");
    let forwarded_post = rx.recv().await.expect("forwarded POST");
    assert_eq!(forwarded_post.method, Method::POST);
    assert_eq!(forwarded_post.path, "/root/gateway/demo/invocations");
    assert_eq!(
        forwarded_post.body,
        r#"{"messages": ["hi"], "temperature": 0.5}"#
    );
    assert_eq!(
        forwarded_post
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert!(!forwarded_post.headers.contains_key("x-test"));

    let failure = request(
        &server,
        Method::POST,
        "/ajax-api/2.0/mlflow/gateway-proxy",
        r#"{"gateway_path":"gateway/failure/invocations"}"#,
        &[("content-type", "application/json")],
    )
    .await;
    assert_eq!(failure.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        failure.body,
        r#"{"error_code": "INTERNAL_ERROR", "message": "Deployments proxy request failed with error code 418. Error message: stub says no", "sqlstate": "XXM00", "error_class": "CLIENT_INTERNAL_ERROR"}"#
    );
    let _ = rx.recv().await.expect("forwarded failure");

    set_env("MLFLOW_DEPLOYMENTS_TARGET", None);
    stub.abort();
}
