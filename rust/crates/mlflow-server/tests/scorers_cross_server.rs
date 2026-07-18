//! Shared-database scorer serialization interoperability between Python and Rust.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use http_body_util::{BodyExt, Empty, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WORKSPACE_DEFAULT_NAME};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const BACKEND_ENV: &str = "_MLFLOW_SERVER_FILE_STORE";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .to_path_buf()
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct PythonServer {
    child: Child,
    base: String,
}

impl PythonServer {
    async fn start(uri: &str) -> Self {
        let port = free_port();
        let child = Command::new("uv")
            .args([
                "run",
                "--frozen",
                "python",
                "-m",
                "uvicorn",
                "mlflow.server.fastapi_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--log-level",
                "error",
            ])
            .current_dir(repo_root())
            .env(BACKEND_ENV, uri)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("launch Python server through uv");
        wait_for_port(port).await;
        Self {
            child,
            base: format!("http://127.0.0.1:{port}"),
        }
    }
}

impl Drop for PythonServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn wait_for_port(port: u16) {
    for _ in 0..100 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("Python server did not listen on port {port}");
}

struct HttpResponse {
    status: StatusCode,
    bytes: Bytes,
}

impl HttpResponse {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.bytes).unwrap()
    }
}

async fn get(url: &str) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let response = client
        .request(
            Request::builder()
                .method(Method::GET)
                .uri(url)
                .body(Empty::<Bytes>::new())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    HttpResponse { status, bytes }
}

async fn post_json(url: &str, value: &Value) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let response = client
        .request(
            Request::builder()
                .method(Method::POST)
                .uri(url)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(serde_json::to_vec(value).unwrap())))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    HttpResponse { status, bytes }
}

async fn register(base: &str, experiment_id: &str, name: &str, payload: &str) {
    let response = post_json(
        &format!("{base}/api/3.0/mlflow/scorers/register"),
        &json!({
            "experiment_id": experiment_id,
            "name": name,
            "serialized_scorer": payload,
        }),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK, "{}", response.json());
}

async fn assert_reads_match(rust_base: &str, python_base: &str, experiment_id: &str, name: &str) {
    let path = format!("/api/3.0/mlflow/scorers/get?experiment_id={experiment_id}&name={name}");
    let rust = get(&format!("{rust_base}{path}")).await;
    let python = get(&format!("{python_base}{path}")).await;
    assert_eq!(rust.status, StatusCode::OK, "{}", rust.json());
    assert_eq!(python.status, StatusCode::OK, "{}", python.json());
    assert_eq!(rust.bytes, python.bytes);
}

#[tokio::test]
async fn scorer_payloads_written_by_either_server_read_byte_identically() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("shared.db");
    std::fs::copy(fixture_path(), &db_path).unwrap();
    let uri = format!("sqlite:///{}", db_path.display());
    let db = Db::connect(&uri, PoolConfig::default()).await.unwrap();
    let store = TrackingStore::new(db, directory.path().join("artifacts").display().to_string());
    let experiment_id = store
        .create_experiment(WORKSPACE_DEFAULT_NAME, "scorer-cross-server", None, &[])
        .await
        .unwrap();

    let recorder = PrometheusBuilder::new().build_recorder().handle();
    let app = build_app_with_recorder(
        &ServerConfig {
            disable_security_middleware: true,
            ..Default::default()
        },
        recorder,
        Some(AppState::new(store)),
    );
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let rust_address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let rust_server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });
    let rust_base = format!("http://{rust_address}");
    let python = PythonServer::start(&uri).await;

    let decorator_request = json!({
        "experiment_id": experiment_id,
        "name": "decorator",
        "serialized_scorer": json!({
            "name": "decorator",
            "call_source": "def decorator(): pass",
        })
        .to_string(),
    });
    let rust_error = post_json(
        &format!("{rust_base}/api/3.0/mlflow/scorers/register"),
        &decorator_request,
    )
    .await;
    let python_error = post_json(
        &format!("{}/api/3.0/mlflow/scorers/register", python.base),
        &decorator_request,
    )
    .await;
    assert_eq!(rust_error.status, StatusCode::BAD_REQUEST);
    assert_eq!(python_error.status, StatusCode::BAD_REQUEST);
    assert_eq!(rust_error.bytes, python_error.bytes);

    register(
        &rust_base,
        &experiment_id,
        "rust-written",
        r#"{ "name": "rust-side", "description": "café", "unknown": [1, true] }"#,
    )
    .await;
    assert_reads_match(&rust_base, &python.base, &experiment_id, "rust-written").await;

    register(
        &python.base,
        &experiment_id,
        "python-written",
        r#"{"name":"python-side","description":"line\nbreak","unknown":{"b":2,"a":1}}"#,
    )
    .await;
    assert_reads_match(&rust_base, &python.base, &experiment_id, "python-written").await;

    let _ = shutdown_tx.send(());
    rust_server.await.unwrap();
}
