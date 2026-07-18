//! Prompt-optimization CRUD and queued-submission lifecycle tests.

use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, JobStatus, JobStore, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tower::ServiceExt;

const WS: &str = "default";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct Fixture {
    _directory: tempfile::TempDir,
    tracking: TrackingStore,
    jobs: JobStore,
    experiment_id: String,
    app: axum::Router,
}

impl Fixture {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prompt-optimization.db");
        std::fs::copy(fixture_path(), &path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let tracking = TrackingStore::new(
            db.clone(),
            directory.path().join("artifacts").display().to_string(),
        );
        let experiment_id = tracking
            .create_experiment(WS, "prompt-optimization-http", None, &[])
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
            Some(AppState::new(tracking.clone())),
        );
        Self {
            _directory: directory,
            tracking,
            jobs,
            experiment_id,
            app,
        }
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut request = Request::builder().method(method).uri(path);
        let body = match body {
            Some(body) => {
                request = request.header(header::CONTENT_TYPE, "application/json");
                Body::from(body.to_string())
            }
            None => Body::empty(),
        };
        let response = self
            .app
            .clone()
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }
}

#[tokio::test]
async fn create_stays_queued_and_crud_rebuilds_from_the_shared_job_and_run() {
    let fixture = Fixture::new().await;
    let (status, created) = fixture
        .request(
            Method::POST,
            "/api/3.0/mlflow/prompt-optimization/jobs",
            Some(json!({
                "experiment_id": fixture.experiment_id,
                "source_prompt_uri": "prompts:/support-bot/7",
                "config": {
                    "optimizer_type": "OPTIMIZER_TYPE_GEPA",
                    "scorers": ["Correctness", "Safety"],
                    "optimizer_config_json": "{\"reflection_model\": \"openai:/gpt-5\", \"max_metric_calls\": 200}"
                },
                "tags": [{"key": "env", "value": "test"}]
            })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(created["job"]["state"]["status"], "JOB_STATUS_PENDING");
    assert_eq!(
        created["job"]["tags"],
        json!([{"key": "env", "value": "test"}])
    );

    let job_id = created["job"]["job_id"].as_str().unwrap();
    let run_id = created["job"]["run_id"].as_str().unwrap();
    let queued = fixture.jobs.get_job(WS, job_id).await.unwrap();
    assert_eq!(queued.status, JobStatus::Pending);
    assert_eq!(queued.job_name, "optimize_prompts");
    assert_eq!(
        queued.params,
        format!(
            "{{\"run_id\": \"{run_id}\", \"experiment_id\": \"{}\", \
             \"prompt_uri\": \"prompts:/support-bot/7\", \"dataset_id\": \"\", \
             \"optimizer_type\": \"gepa\", \"optimizer_config\": \
             {{\"reflection_model\": \"openai:/gpt-5\", \"max_metric_calls\": 200}}, \
             \"scorer_names\": [\"Correctness\", \"Safety\"]}}",
            fixture.experiment_id
        )
    );
    let run = fixture.tracking.get_run(WS, run_id).await.unwrap();
    let params: std::collections::HashMap<_, _> = run
        .data
        .params
        .iter()
        .map(|param| (param.key.as_str(), param.value.as_str()))
        .collect();
    assert_eq!(params["optimizer_type"], "gepa");
    assert_eq!(params["scorer_names"], "[\"Correctness\", \"Safety\"]");

    let (status, fetched) = fixture
        .request(
            Method::GET,
            &format!("/ajax-api/3.0/mlflow/prompt-optimization/jobs/{job_id}"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        fetched["job"]["config"]["optimizer_type"],
        "OPTIMIZER_TYPE_GEPA"
    );
    assert_eq!(
        fetched["job"]["config"]["optimizer_config_json"],
        "{\"reflection_model\": \"openai:/gpt-5\", \"max_metric_calls\": 200}"
    );
    // Tags exist only on the immediate create response; Python does not persist
    // them and therefore omits them when rebuilding later responses.
    assert!(fetched["job"].get("tags").is_none());

    let (status, searched) = fixture
        .request(
            Method::GET,
            &format!(
                "/api/3.0/mlflow/prompt-optimization/jobs/search?experiment_id={}",
                fixture.experiment_id
            ),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(searched["jobs"].as_array().unwrap().len(), 1);

    let (status, canceled) = fixture
        .request(
            Method::POST,
            &format!("/api/3.0/mlflow/prompt-optimization/jobs/{job_id}/cancel"),
            Some(json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["job"]["state"]["status"], "JOB_STATUS_CANCELED");
    assert_eq!(
        fixture
            .tracking
            .get_run(WS, run_id)
            .await
            .unwrap()
            .info
            .status,
        "KILLED"
    );

    let (status, deleted) = fixture
        .request(
            Method::DELETE,
            &format!("/ajax-api/3.0/mlflow/prompt-optimization/jobs/{job_id}"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted, json!({}));
    assert!(fixture.jobs.get_job(WS, job_id).await.is_err());
    assert_eq!(
        fixture
            .tracking
            .get_run(WS, run_id)
            .await
            .unwrap()
            .info
            .lifecycle_stage,
        "deleted"
    );
}
