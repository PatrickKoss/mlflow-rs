//! HTTP parity tests for scorer registration validation and online configs.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::scorers::DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WORKSPACE_DEFAULT_NAME};
use serde_json::{json, Value};
use tokio::net::TcpListener;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct TestServer {
    base: String,
    experiment_id: String,
    _directory: tempfile::TempDir,
    _server: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start(tag: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let db_path = directory.path().join(format!("{tag}.db"));
        std::fs::copy(fixture_path(), &db_path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", db_path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let Db::Sqlite(pool) = &db else {
            unreachable!("HTTP tests use SQLite")
        };
        sqlx::query(
            "INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES (?, ?, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, ?)",
        )
        .bind("test-endpoint-id")
        .bind("test-endpoint")
        .bind(WORKSPACE_DEFAULT_NAME)
        .execute(pool)
        .await
        .unwrap();
        let store =
            TrackingStore::new(db, directory.path().join("artifacts").display().to_string());
        let experiment_id = store
            .create_experiment(
                WORKSPACE_DEFAULT_NAME,
                &format!("scorers-http-{tag}"),
                None,
                &[],
            )
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
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            base: format!("http://{address}"),
            experiment_id,
            _directory: directory,
            _server: server,
        }
    }

    async fn register(&self, name: &str, serialized_scorer: Value) -> HttpResponse {
        send(
            &self.base,
            Method::POST,
            "/api/3.0/mlflow/scorers/register",
            Some(json!({
                "experiment_id": self.experiment_id,
                "name": name,
                "serialized_scorer": serialized_scorer.to_string(),
            })),
        )
        .await
    }
}

struct HttpResponse {
    status: StatusCode,
    body: Value,
}

async fn send(base: &str, method: Method, path: &str, body: Option<Value>) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let mut request = Request::builder()
        .method(method)
        .uri(format!("{base}{path}"));
    let body = match body {
        Some(body) => {
            request = request.header("content-type", "application/json");
            Bytes::from(body.to_string())
        }
        None => Bytes::new(),
    };
    let response = client
        .request(request.body(Full::new(body)).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    HttpResponse {
        status,
        body: serde_json::from_slice(&bytes).unwrap(),
    }
}

#[tokio::test]
async fn decorator_registration_error_is_byte_identical_to_python() {
    let server = TestServer::start("decorator").await;
    let response = server
        .register(
            "custom",
            json!({
                "name": "custom",
                "call_source": "def custom(): pass",
            }),
        )
        .await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.body["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        response.body["message"].as_str().unwrap().as_bytes(),
        DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR.as_bytes()
    );
}

#[tokio::test]
async fn all_six_phoenix_metrics_are_rejected_at_registration() {
    let server = TestServer::start("phoenix").await;
    for metric in [
        "Hallucination",
        "QA",
        "Relevance",
        "SQL",
        "Summarization",
        "Toxicity",
    ] {
        let response = server
            .register(
                metric,
                json!({
                    "name": metric,
                    "third_party_scorer_data": {"metric_name": metric},
                }),
            )
            .await;
        assert_eq!(response.status, StatusCode::BAD_REQUEST, "{metric}");
        assert_eq!(response.body["error_code"], "INVALID_PARAMETER_VALUE");
        let message = response.body["message"].as_str().unwrap();
        assert!(message.contains(&format!("Phoenix scorer metric '{metric}'")));
        assert!(message.contains("Elastic-2.0"));
        assert!(message.contains("Faithfulness"));
        assert!(message.contains("custom instructions judge"));
    }
}

#[tokio::test]
async fn online_config_put_and_get_round_trip() {
    let server = TestServer::start("online-config").await;
    let registered = server
        .register(
            "online-judge",
            json!({
                "instructions_judge_pydantic_data": {
                    "model": "gateway:/test-endpoint",
                    "instructions": "Judge {{ outputs }}",
                },
            }),
        )
        .await;
    assert_eq!(registered.status, StatusCode::OK, "{:?}", registered.body);
    let scorer_id = registered.body["scorer_id"].as_str().unwrap();

    let updated = send(
        &server.base,
        Method::PUT,
        "/api/3.0/mlflow/scorers/online-config",
        Some(json!({
            "experiment_id": server.experiment_id,
            "name": "online-judge",
            "sample_rate": 0.75,
            "filter_string": "status = 'OK'",
        })),
    )
    .await;
    assert_eq!(updated.status, StatusCode::OK, "{:?}", updated.body);
    assert_eq!(updated.body["config"]["scorer_id"], scorer_id);
    assert_eq!(updated.body["config"]["sample_rate"], 0.75);
    assert_eq!(updated.body["config"]["filter_string"], "status = 'OK'");

    let fetched = send(
        &server.base,
        Method::GET,
        &format!(
            "/ajax-api/3.0/mlflow/scorers/online-configs?scorer_ids=missing&scorer_ids={scorer_id}"
        ),
        None,
    )
    .await;
    assert_eq!(fetched.status, StatusCode::OK, "{:?}", fetched.body);
    assert_eq!(fetched.body["configs"].as_array().unwrap().len(), 1);
    assert_eq!(fetched.body["configs"][0], updated.body["config"]);
}
