//! T20.4 promptlab + adjacent demo route HTTP and cross-language coverage.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{header, Method, Response, StatusCode};
use axum::routing::post;
use axum::Router;
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, EndpointModelConfig, PoolConfig, TrackingStore, WORKSPACE_DEFAULT_NAME};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tower::ServiceExt;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tracking.db")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repository root")
        .to_path_buf()
}

struct Fixture {
    _directory: tempfile::TempDir,
    artifact_root: PathBuf,
    app: Router,
    store: TrackingStore,
}

impl Fixture {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let db_path = directory.path().join("tracking.db");
        std::fs::copy(fixture_path(), &db_path).unwrap();
        let artifact_root = directory.path().join("artifacts");
        let db = Db::connect(
            &format!("sqlite:///{}", db_path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let store = TrackingStore::new(db, artifact_root.display().to_string());
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(AppState::new(store.clone())),
        );
        Self {
            _directory: directory,
            artifact_root,
            app,
            store,
        }
    }

    async fn request(&self, path: &str, body: &str) -> (StatusCode, String) {
        request(self.app.clone(), path, body).await
    }
}

async fn request(app: Router, path: &str, body: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn promptlab_body(experiment_id: &str) -> String {
    json!({
        "experiment_id": experiment_id,
        "run_name": "promptlab-cross-language",
        "prompt_template": "Write about {{ thing }}.",
        "prompt_parameters": [{"key":"thing","value":"books"}],
        "model_route": "openai-endpoint",
        "model_parameters": [
            {"key":"temperature","value":"0.1"},
            {"key":"max_tokens","value":"10"}
        ],
        "model_input": "Write about books.",
        "model_output": "gateway:Write about books.",
        "model_output_parameters": [{"key":"latency","value":"100"}],
        "mlflow_version": "1.0.0",
        "user_id": "cross-language",
        "start_time": 123456
    })
    .to_string()
}

#[tokio::test]
async fn promptlab_validation_run_data_and_artifact_layout() {
    let fixture = Fixture::new().await;
    let (status, body) = fixture
        .request("/ajax-api/2.0/mlflow/runs/create-promptlab-run", "{}")
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["message"],
        "CreatePromptlabRun request must specify experiment_id."
    );

    let experiment_id = fixture
        .store
        .create_experiment(WORKSPACE_DEFAULT_NAME, "promptlab-http", None, &[])
        .await
        .unwrap();
    let (status, body) = fixture
        .request(
            "/ajax-api/2.0/mlflow/runs/create-promptlab-run",
            &promptlab_body(&experiment_id),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).unwrap();
    let run = &response["run"];
    assert_eq!(run["info"]["status"], "FINISHED");
    assert_eq!(run["info"]["start_time"], 123456);
    // Empty repeated proto fields are omitted by MLflow's JSON codec.
    assert!(run["data"]["metrics"].is_null());
    let params = run["data"]["params"].as_array().unwrap();
    for expected in [
        json!({"key":"max_tokens","value":"10"}),
        json!({"key":"model_route","value":"openai-endpoint"}),
        json!({"key":"prompt_template","value":"Write about {{ thing }}."}),
        json!({"key":"temperature","value":"0.1"}),
    ] {
        assert!(params.contains(&expected), "missing {expected}: {params:?}");
    }
    let tags = run["data"]["tags"].as_array().unwrap();
    assert!(tags.contains(&json!({
        "key":"mlflow.loggedArtifacts",
        "value":"[{\"path\": \"eval_results_table.json\", \"type\": \"table\"}]"
    })));
    assert!(tags.contains(&json!({
        "key":"mlflow.runSourceType","value":"PROMPT_ENGINEERING"
    })));
    assert!(tags
        .iter()
        .any(|tag| tag["key"] == "mlflow.log-model.history"));

    let run_id = run["info"]["run_id"].as_str().unwrap();
    let root = fixture
        .artifact_root
        .join(&experiment_id)
        .join(run_id)
        .join("artifacts");
    let files = [
        "eval_results_table.json",
        "model/MLmodel",
        "model/conda.yaml",
        "model/input_example.json",
        "model/parameters.yaml",
        "model/python_env.yaml",
        "model/requirements.txt",
        "model/serving_input_example.json",
    ];
    for file in files {
        assert!(root.join(file).is_file(), "missing {file}");
    }
    assert_eq!(
        std::fs::read_to_string(root.join("eval_results_table.json")).unwrap(),
        "{\"columns\": [\"thing\", \"prompt\", \"output\", \"latency\"], \"data\": [[\"books\", \"Write about books.\", \"gateway:Write about books.\", \"100\"]]}"
    );
}

#[tokio::test]
async fn demo_routes_match_flask_bytes_selection_idempotence_and_delete() {
    let fixture = Fixture::new().await;
    let path = "/ajax-api/3.0/mlflow/demo/generate";
    let (status, first) = fixture
        .request(path, r#"{"features":["traces","prompts","unknown"]}"#)
        .await;
    assert_eq!(status, StatusCode::OK);
    let first_json: Value = serde_json::from_str(&first).unwrap();
    assert_eq!(first_json["status"], "created");
    assert_eq!(
        first_json["features_generated"],
        json!(["prompts", "traces"])
    );
    assert!(first.ends_with('\n'));

    let (_, second) = fixture
        .request(path, r#"{"features":["traces","prompts"]}"#)
        .await;
    let id = first_json["experiment_id"].as_str().unwrap();
    assert_eq!(
        second,
        format!(
            "{{\"experiment_id\":\"{id}\",\"features_generated\":[],\"navigation_url\":\"/experiments/{id}\",\"status\":\"exists\"}}\n"
        )
    );

    let (_, mixed) = fixture
        .request(path, r#"{"features":["traces","evaluation"]}"#)
        .await;
    assert_eq!(
        mixed,
        format!(
            "{{\"experiment_id\":\"{id}\",\"features_generated\":[\"evaluation\"],\"navigation_url\":\"/experiments/{id}\",\"status\":\"created\"}}\n"
        )
    );

    let (_, deleted) = fixture
        .request("/ajax-api/3.0/mlflow/demo/delete", "")
        .await;
    assert_eq!(
        deleted,
        "{\"features_deleted\":[\"prompts\",\"traces\",\"evaluation\"],\"status\":\"deleted\"}\n"
    );
    let (_, deleted_again) = fixture
        .request("/ajax-api/3.0/mlflow/demo/delete", "")
        .await;
    assert_eq!(
        deleted_again,
        "{\"features_deleted\":[],\"status\":\"deleted\"}\n"
    );
}

#[tokio::test]
async fn python_loads_rust_model_and_both_predict_through_scripted_rust_gateway() {
    let fixture = Fixture::new().await;
    let provider_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_base = format!("http://{}", provider_listener.local_addr().unwrap());
    let provider_task = tokio::spawn(async move {
        axum::serve(
            provider_listener,
            Router::new().fallback(post(mock_provider)),
        )
        .await
        .unwrap();
    });
    seed_gateway(&fixture.store, &provider_base).await;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_base = format!("http://{}", listener.local_addr().unwrap());
    let server_app = fixture.app.clone();
    let server_task = tokio::spawn(async move {
        axum::serve(listener, server_app).await.unwrap();
    });

    let experiment_id = fixture
        .store
        .create_experiment(
            WORKSPACE_DEFAULT_NAME,
            "promptlab-cross-language",
            None,
            &[],
        )
        .await
        .unwrap();
    let response = reqwest::Client::new()
        .post(format!(
            "{server_base}/ajax-api/2.0/mlflow/runs/create-promptlab-run"
        ))
        .header(header::CONTENT_TYPE.as_str(), "application/json")
        .body(promptlab_body(&experiment_id))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response: Value = response.json().await.unwrap();
    let run_id = response["run"]["info"]["run_id"].as_str().unwrap();
    let rust_model = fixture
        .artifact_root
        .join(&experiment_id)
        .join(run_id)
        .join("artifacts/model");

    let oracle_gateway = server_base.clone();
    let output = tokio::task::spawn_blocking(move || {
        Command::new("uv")
            .args([
                "run",
                "--frozen",
                "python",
                "rust/tools/promptlab_cross_language.py",
            ])
            .arg(&rust_model)
            .arg(&oracle_gateway)
            .current_dir(repo_root())
            .output()
            .expect("run promptlab cross-language oracle")
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let differential: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        differential["rust_prediction"],
        json!(["gateway:Write about books.", "gateway:Write about coffee."])
    );
    assert_eq!(
        differential["rust_prediction"],
        differential["python_prediction"]
    );

    server_task.abort();
    provider_task.abort();
}

async fn seed_gateway(store: &TrackingStore, provider_base: &str) {
    let secret = store
        .create_gateway_secret(
            WORKSPACE_DEFAULT_NAME,
            "obvious-fake-promptlab-secret",
            &HashMap::from([(
                "api_key".to_string(),
                "obvious-fake-promptlab-key".to_string(),
            )]),
            Some("openai"),
            &HashMap::from([("api_base".to_string(), format!("{provider_base}/v1"))]),
            Some("promptlab-cross-language"),
        )
        .await
        .unwrap();
    let model = store
        .create_gateway_model_definition(
            WORKSPACE_DEFAULT_NAME,
            "promptlab-openai-definition",
            &secret.secret_id,
            "openai",
            "scripted-promptlab-model",
            Some("promptlab-cross-language"),
        )
        .await
        .unwrap();
    store
        .create_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            "openai-endpoint",
            &[EndpointModelConfig {
                model_definition_id: model.model_definition_id,
                linkage_type: "PRIMARY".to_string(),
                weight: 1.0,
                fallback_order: None,
            }],
            Some("promptlab-cross-language"),
            None,
            None,
            None,
            false,
        )
        .await
        .unwrap();
}

async fn mock_provider(request: Request) -> Response<Body> {
    let body: Value =
        serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let prompt = body["messages"][0]["content"].as_str().unwrap();
    let response = json!({
        "id":"scripted-promptlab-id",
        "object":"chat.completion",
        "created":7,
        "model":"scripted-promptlab-model",
        "choices":[{
            "index":0,
            "message":{"role":"assistant","content":format!("gateway:{prompt}")},
            "finish_reason":"stop"
        }],
        "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(response.to_string()))
        .unwrap()
}
