//! Shared-DB Python/Rust job-store and endpoint interoperability.

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
use mlflow_store::{Db, JobStatus, JobStore, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const BACKEND_ENV: &str = "_MLFLOW_SERVER_FILE_STORE";
const WS: &str = "default";

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
    async fn start(uri: &str, huey_storage_path: &Path) -> Self {
        let port = free_port();
        let test_jobs_path = repo_root().join("tests/server/jobs");
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
            .env("MLFLOW_TRACKING_URI", format!("http://127.0.0.1:{port}"))
            .env("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
            .env("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "2")
            .env(
                "MLFLOW_RUN_CONTEXT",
                r#"{"mlflow.user":"cross-server","mlflow.source.name":"cross-server","mlflow.source.type":"LOCAL"}"#,
            )
            .env(
                "_MLFLOW_SUPPORTED_JOB_FUNCTION_LIST",
                "test_endpoint.simple_job_fun",
            )
            .env("_MLFLOW_ALLOWED_JOB_NAME_LIST", "simple_job_fun")
            .env("_MLFLOW_HUEY_STORAGE_PATH", huey_storage_path)
            .env("PYTHONPATH", test_jobs_path)
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

async fn request(method: Method, url: &str) -> (StatusCode, Value) {
    let (status, body) = request_bytes(method, url).await;
    (status, serde_json::from_slice(&body).unwrap())
}

async fn request_bytes(method: Method, url: &str) -> (StatusCode, Vec<u8>) {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let response = client
        .request(
            Request::builder()
                .method(method)
                .uri(url)
                .body(Empty::<Bytes>::new())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, body.to_vec())
}

async fn post_json(url: &str, value: &Value) -> (StatusCode, Value) {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let body = serde_json::to_vec(value).unwrap();
    let response = client
        .request(
            Request::builder()
                .method(Method::POST)
                .uri(url)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(body)))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn python_and_rust_create_read_and_cancel_each_others_rows() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("shared.db");
    std::fs::copy(fixture_path(), &db_path).unwrap();
    let uri = format!("sqlite:///{}", db_path.display());
    let db = Db::connect(&uri, PoolConfig::default()).await.unwrap();
    let jobs = JobStore::new(db.clone());

    let recorder = PrometheusBuilder::new().build_recorder().handle();
    let app = build_app_with_recorder(
        &ServerConfig {
            disable_security_middleware: true,
            ..Default::default()
        },
        recorder,
        Some(AppState::new(TrackingStore::new(
            db,
            directory.path().join("artifacts").display().to_string(),
        ))),
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
    let python = PythonServer::start(&uri, directory.path()).await;

    let (status, body) = post_json(
        &format!("{}/ajax-api/3.0/jobs/", python.base),
        &serde_json::json!({
            "job_name": "simple_job_fun",
            "params": {"x": 3, "y": 4},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let python_job_id = body["job_id"].as_str().unwrap().to_string();
    let (status, body) = request(
        Method::GET,
        &format!("{rust_base}/ajax-api/3.0/mlflow/jobs/{python_job_id}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "PENDING");
    let (status, _) = request(
        Method::PATCH,
        &format!("{rust_base}/ajax-api/3.0/mlflow/jobs/cancel/{python_job_id}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = request(
        Method::GET,
        &format!("{}/ajax-api/3.0/mlflow/jobs/{python_job_id}", python.base),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "CANCELED");

    let rust_job = jobs
        .create_job(WS, "rust_job", r#"{"label":"rust-to-python"}"#, None)
        .await
        .unwrap();
    let (status, body) = request(
        Method::GET,
        &format!(
            "{}/ajax-api/3.0/mlflow/jobs/{}",
            python.base, rust_job.job_id
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "PENDING");
    let (status, body) = request(
        Method::PATCH,
        &format!(
            "{}/ajax-api/3.0/mlflow/jobs/cancel/{}",
            python.base, rust_job.job_id
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "CANCELED");
    assert_eq!(
        jobs.get_job(WS, &rust_job.job_id).await.unwrap().status,
        JobStatus::Canceled
    );

    shutdown_tx.send(()).unwrap();
    rust_server.await.unwrap();
}

#[tokio::test]
async fn prompt_optimization_rows_created_by_either_server_have_identical_get_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("shared-prompt-optimization.db");
    std::fs::copy(fixture_path(), &db_path).unwrap();
    let uri = format!("sqlite:///{}", db_path.display());
    let db = Db::connect(&uri, PoolConfig::default()).await.unwrap();
    let tracking = TrackingStore::new(
        db.clone(),
        directory.path().join("artifacts").display().to_string(),
    );
    let experiment_id = tracking
        .create_experiment(WS, "cross-server-prompt-optimization", None, &[])
        .await
        .unwrap();
    let jobs = JobStore::new(db);

    let recorder = PrometheusBuilder::new().build_recorder().handle();
    let app = build_app_with_recorder(
        &ServerConfig {
            disable_security_middleware: true,
            ..Default::default()
        },
        recorder,
        Some(AppState::new(tracking)),
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
    let python = PythonServer::start(&uri, directory.path()).await;
    let request = serde_json::json!({
        "experiment_id": experiment_id,
        "source_prompt_uri": "prompts:/cross-server/1",
        "config": {
            "optimizer_type": "OPTIMIZER_TYPE_METAPROMPT",
            "scorers": [],
            "optimizer_config_json": "{\"reflection_model\": \"openai:/gpt-5\"}"
        }
    });

    let (status, rust_created) = post_json(
        &format!("{rust_base}/api/3.0/mlflow/prompt-optimization/jobs"),
        &request,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(rust_created["job"]["state"]["status"], "JOB_STATUS_PENDING");
    let rust_job_id = rust_created["job"]["job_id"].as_str().unwrap();

    let (status, python_created) = post_json(
        &format!("{}/api/3.0/mlflow/prompt-optimization/jobs", python.base),
        &request,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        python_created["job"]["state"]["status"],
        "JOB_STATUS_PENDING"
    );
    let python_job_id = python_created["job"]["job_id"].as_str().unwrap();

    for job_id in [rust_job_id, python_job_id] {
        let rust_url = format!("{rust_base}/ajax-api/3.0/mlflow/prompt-optimization/jobs/{job_id}");
        let python_url = format!(
            "{}/ajax-api/3.0/mlflow/prompt-optimization/jobs/{job_id}",
            python.base
        );
        let (rust_status, rust_body) = request_bytes(Method::GET, &rust_url).await;
        let (python_status, python_body) = request_bytes(Method::GET, &python_url).await;
        assert_eq!(rust_status, StatusCode::OK);
        assert_eq!(python_status, StatusCode::OK);
        assert_eq!(rust_body, python_body);

        let row = jobs.get_job(WS, job_id).await.unwrap();
        assert_eq!(row.job_name, "optimize_prompts");
        assert_eq!(row.status, JobStatus::Pending);
        let params: Value = serde_json::from_str(&row.params).unwrap();
        assert_eq!(params["experiment_id"], experiment_id);
        assert_eq!(params["optimizer_type"], "metaprompt");
        assert_eq!(
            params["optimizer_config"]["reflection_model"],
            "openai:/gpt-5"
        );
    }

    shutdown_tx.send(()).unwrap();
    rust_server.await.unwrap();
}

#[tokio::test]
async fn invoke_jobs_and_precreated_runs_are_byte_readable_by_either_server() {
    std::env::set_var(
        "MLFLOW_RUN_CONTEXT",
        r#"{"mlflow.user":"cross-server","mlflow.source.name":"cross-server","mlflow.source.type":"LOCAL"}"#,
    );
    std::env::set_var("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "2");
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("shared-invoke.db");
    std::fs::copy(fixture_path(), &db_path).unwrap();
    let uri = format!("sqlite:///{}", db_path.display());
    let db = Db::connect(&uri, PoolConfig::default()).await.unwrap();
    let tracking = TrackingStore::new(
        db.clone(),
        directory.path().join("artifacts").display().to_string(),
    );
    let experiment_id = tracking
        .create_experiment(WS, "cross-server-invoke", None, &[])
        .await
        .unwrap();
    let jobs = JobStore::new(db);
    let recorder = PrometheusBuilder::new().build_recorder().handle();
    let app = build_app_with_recorder(
        &ServerConfig {
            disable_security_middleware: true,
            ..Default::default()
        },
        recorder,
        Some(AppState::new(tracking)),
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
    let python = PythonServer::start(&uri, directory.path()).await;
    let scorer = include_str!("../../mlflow-genai/tests/fixtures/instructions_judge_scorer.json");

    for base in [&rust_base, &python.base] {
        let (status, evaluate) = post_json(
            &format!("{base}/ajax-api/3.0/mlflow/genai/evaluate/invoke"),
            &json!({
                "experiment_id": experiment_id,
                "trace_ids": ["tr-a", "tr-b"],
                "serialized_scorers": [scorer],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let evaluate_job = jobs
            .get_job(WS, evaluate["job_id"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(evaluate_job.job_name, "invoke_genai_evaluate");
        assert_eq!(evaluate_job.status, JobStatus::Pending);
        let params: Value = serde_json::from_str(&evaluate_job.params).unwrap();
        assert_eq!(params["trace_ids"], json!(["tr-a", "tr-b"]));
        assert_eq!(params["serialized_scorers"], json!([scorer]));
        cross_read_run_bytes(
            &rust_base,
            &python.base,
            evaluate["run_id"].as_str().unwrap(),
        )
        .await;

        let (status, scorer_response) = post_json(
            &format!("{base}/ajax-api/3.0/mlflow/scorer/invoke"),
            &json!({
                "experiment_id": experiment_id,
                "serialized_scorer": scorer,
                "trace_ids": ["tr-a", "tr-b", "tr-c"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(scorer_response["jobs"].as_array().unwrap().len(), 2);
        for submitted in scorer_response["jobs"].as_array().unwrap() {
            let job = jobs
                .get_job(WS, submitted["job_id"].as_str().unwrap())
                .await
                .unwrap();
            assert_eq!(job.job_name, "invoke_scorer");
            assert_eq!(job.status, JobStatus::Pending);
        }

        let (status, issues) = post_json(
            &format!("{base}/ajax-api/3.0/mlflow/issues/invoke"),
            &json!({
                "experiment_id": experiment_id,
                "trace_ids": ["tr-a", "tr-b"],
                "categories": ["correctness", "safety"],
                "provider": "openai",
                "model": "gpt-5",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let issues_job = jobs
            .get_job(WS, issues["job_id"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(issues_job.job_name, "invoke_issue_detection");
        assert_eq!(issues_job.status, JobStatus::Pending);
        cross_read_run_bytes(&rust_base, &python.base, issues["run_id"].as_str().unwrap()).await;
    }

    shutdown_tx.send(()).unwrap();
    rust_server.await.unwrap();
}

async fn cross_read_run_bytes(rust_base: &str, python_base: &str, run_id: &str) {
    let tail = format!("/api/2.0/mlflow/runs/get?run_id={run_id}");
    let (rust_status, rust_body) = request_bytes(Method::GET, &format!("{rust_base}{tail}")).await;
    let (python_status, python_body) =
        request_bytes(Method::GET, &format!("{python_base}{tail}")).await;
    assert_eq!(rust_status, StatusCode::OK);
    assert_eq!(python_status, StatusCode::OK);
    let mut rust: Value = serde_json::from_slice(&rust_body).unwrap();
    let mut python: Value = serde_json::from_slice(&python_body).unwrap();
    canonicalize_tag_arrays(&mut rust);
    canonicalize_tag_arrays(&mut python);
    assert_eq!(rust, python);
}

fn canonicalize_tag_arrays(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                canonicalize_tag_arrays(value);
                if key == "tags" {
                    if let Some(tags) = value.as_array_mut() {
                        tags.sort_by_key(|tag| tag["key"].as_str().unwrap_or("").to_string());
                    }
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                canonicalize_tag_arrays(value);
            }
        }
        _ => {}
    }
}
