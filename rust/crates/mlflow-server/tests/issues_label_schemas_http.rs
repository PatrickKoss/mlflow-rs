use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use base64::Engine;
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tower::ServiceExt;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

fn auth_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("mlflow-auth")
        .join("tests")
        .join("fixtures")
        .join("basic_auth.db")
}

struct Fixture {
    path: PathBuf,
    auth_path: Option<PathBuf>,
    app: axum::Router,
}

impl Fixture {
    async fn new(tag: &str) -> Self {
        Self::new_with_auth(tag, false).await
    }

    async fn new_with_auth(tag: &str, with_auth: bool) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_issues_labels_{tag}_{}_{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixture_path(), &path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let store = TrackingStore::new(db, "s3://bucket/mlruns");
        let mut state = AppState::new(store);
        let auth_path = if with_auth {
            let auth_path = path.with_extension("auth.db");
            std::fs::copy(auth_fixture_path(), &auth_path).unwrap();
            let auth_db = AuthDb::connect_and_verify_with(
                &format!("sqlite:///{}", auth_path.display()),
                None,
                PoolConfig::default(),
            )
            .await
            .unwrap();
            state = state.with_auth_store(AuthStore::new(auth_db));
            Some(auth_path)
        } else {
            None
        };
        let config = ServerConfig {
            disable_security_middleware: true,
            ..Default::default()
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(&config, recorder, Some(state));
        Self {
            path,
            auth_path,
            app,
        }
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        self.request_with_auth(method, path, body, None).await
    }

    async fn request_with_auth(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        authorization: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut request = Request::builder().method(method).uri(path);
        if let Some(authorization) = authorization {
            request = request.header(header::AUTHORIZATION, authorization);
        }
        let bytes = if let Some(body) = body {
            request = request.header(header::CONTENT_TYPE, "application/json");
            Body::from(body.to_string())
        } else {
            Body::empty()
        };
        let response = self
            .app
            .clone()
            .oneshot(request.body(bytes).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        if let Some(path) = &self.auth_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn basic(username: &str, password: &str) -> String {
    let encoded =
        base64::engine::general_purpose::STANDARD.encode(format!("{username}:{password}"));
    format!("Basic {encoded}")
}

#[tokio::test]
async fn issue_crud_round_trip_works_on_both_prefixes() {
    for prefix in ["/api/3.0", "/ajax-api/3.0"] {
        let fixture = Fixture::new(if prefix.starts_with("/ajax") {
            "issue-ajax"
        } else {
            "issue-api"
        })
        .await;
        let (status, created) = fixture
            .request(
                Method::POST,
                &format!("{prefix}/mlflow/issues"),
                Some(json!({
                    "experiment_id": "0",
                    "name": "HTTP issue",
                    "description": "round trip",
                    "severity": "medium",
                    "root_causes": ["cause"],
                    "categories": ["quality"]
                })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{created}");
        let issue_id = created["issue"]["issue_id"].as_str().unwrap();
        assert_eq!(created["issue"]["status"], "pending");

        let (status, fetched) = fixture
            .request(
                Method::GET,
                &format!("{prefix}/mlflow/issues/{issue_id}"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(fetched["issue"]["root_causes"], json!(["cause"]));

        let (status, updated) = fixture
            .request(
                Method::PATCH,
                &format!("{prefix}/mlflow/issues/{issue_id}"),
                Some(json!({"name": "updated", "status": "resolved"})),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["issue"]["name"], "updated");

        let (status, searched) = fixture
            .request(
                Method::POST,
                &format!("{prefix}/mlflow/issues/search"),
                Some(json!({"experiment_id": "0", "include_trace_count": true})),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(searched["issues"][0]["trace_count"], 0);
    }
}

#[tokio::test]
async fn label_schema_crud_errors_and_presence_round_trip() {
    let fixture = Fixture::new("label-schema-http").await;
    let prefix = "/api/3.0/mlflow/label-schemas";
    let body = json!({
        "experiment_id": "0",
        "name": "HTTP correctness",
        "type": "FEEDBACK",
        "input": {
            "pass_fail": {"positive_label": "Correct", "negative_label": "Incorrect"}
        },
        "enable_comment": false
    });
    let (status, created) = fixture
        .request(
            Method::POST,
            &format!("{prefix}/create"),
            Some(body.clone()),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    let schema_id = created["label_schema"]["schema_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(created["label_schema"]["enable_comment"], false);

    let (status, duplicate) = fixture
        .request(Method::POST, &format!("{prefix}/create"), Some(body))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(duplicate["error_code"], "RESOURCE_ALREADY_EXISTS");
    assert_eq!(
        duplicate["message"],
        "Label schema with name 'HTTP correctness' already exists for experiment '0'."
    );

    let (status, fetched) = fixture
        .request(
            Method::GET,
            &format!("{prefix}/get?schema_id={schema_id}"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["label_schema"]["type"], "FEEDBACK");

    let (status, immutable) = fixture
        .request(
            Method::PATCH,
            &format!("{prefix}/update"),
            Some(json!({
                "schema_id": schema_id,
                "input": {"numeric": {"min_value": 0.0, "max_value": 1.0}}
            })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        immutable["message"],
        "A label schema's input type cannot be changed after creation (existing: \
         InputPassFail, got: InputNumeric)."
    );

    let (status, updated) = fixture
        .request(
            Method::PATCH,
            &format!("{prefix}/update"),
            Some(json!({
                "schema_id": schema_id,
                "name": "HTTP correctness updated",
                "enable_comment": true
            })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["label_schema"]["enable_comment"], true);

    let (status, listed) = fixture
        .request(Method::GET, &format!("{prefix}/list?experiment_id=0"), None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed["label_schemas"].as_array().unwrap().len(), 2);
    assert_eq!(listed["next_page_token"], "");

    let (status, deleted) = fixture
        .request(
            Method::DELETE,
            &format!("{prefix}/delete"),
            Some(json!({"schema_id": schema_id})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted, json!({}));
}

#[tokio::test]
async fn issues_are_authenticated_only_while_label_schemas_inherit_experiment_permissions() {
    let fixture = Fixture::new_with_auth("auth-contract", true).await;
    let bob = basic("bob_pbkdf2", "bob-password-4567");
    let alice = basic("alice_scrypt", "alice-password-123");

    let (status, _) = fixture
        .request(
            Method::POST,
            "/api/3.0/mlflow/issues",
            Some(json!({
                "experiment_id": "0",
                "name": "unauthenticated",
                "description": "must fail"
            })),
        )
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // D21: no per-experiment validator runs for issues, so any authenticated
    // caller reaches the store.
    let (status, _) = fixture
        .request_with_auth(
            Method::POST,
            "/api/3.0/mlflow/issues",
            Some(json!({
                "experiment_id": "0",
                "name": "authenticated",
                "description": "allowed"
            })),
            Some(&bob),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    let create_body = json!({
        "experiment_id": "0",
        "name": "Auth schema",
        "type": "FEEDBACK",
        "input": {"text": {}}
    });
    let (status, _) = fixture
        .request_with_auth(
            Method::POST,
            "/api/3.0/mlflow/label-schemas/create",
            Some(create_body.clone()),
            Some(&bob),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let (status, created) = fixture
        .request_with_auth(
            Method::POST,
            "/api/3.0/mlflow/label-schemas/create",
            Some(create_body),
            Some(&alice),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let schema_id = created["label_schema"]["schema_id"].as_str().unwrap();

    // Bob's EDIT permission includes READ but not MANAGE.
    let (status, _) = fixture
        .request_with_auth(
            Method::GET,
            &format!("/api/3.0/mlflow/label-schemas/get?schema_id={schema_id}"),
            None,
            Some(&bob),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = fixture
        .request_with_auth(
            Method::PATCH,
            "/api/3.0/mlflow/label-schemas/update",
            Some(json!({"schema_id": schema_id, "name": "hijacked"})),
            Some(&bob),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}
