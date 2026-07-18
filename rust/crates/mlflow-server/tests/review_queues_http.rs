use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use base64::Engine;
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{
    Db, LabelSchemaInput, LabelSchemaType, PoolConfig, ReviewItemType, ReviewQueueType,
    StartTraceInput, TrackingStore,
};
use serde_json::{json, Value};
use tower::ServiceExt;

const WS: &str = "default";
const ALICE: (&str, &str) = ("alice_scrypt", "alice-password-123");
const BOB: (&str, &str) = ("bob_pbkdf2", "bob-password-4567");

fn tracking_fixture_path() -> PathBuf {
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
    tracking_path: PathBuf,
    auth_path: Option<PathBuf>,
    app: axum::Router,
    tracking: TrackingStore,
    auth: Option<AuthStore>,
}

impl Fixture {
    async fn new(tag: &str) -> Self {
        Self::build(tag, false).await
    }

    async fn with_auth(tag: &str) -> Self {
        Self::build(tag, true).await
    }

    async fn build(tag: &str, with_auth: bool) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tracking_path = std::env::temp_dir().join(format!(
            "mlflow_rust_review_queues_{tag}_{}_{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tracking_path);
        std::fs::copy(tracking_fixture_path(), &tracking_path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", tracking_path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let tracking = TrackingStore::new(db, "s3://bucket/mlruns");
        let mut state = AppState::new(tracking.clone());

        let (auth_path, auth) = if with_auth {
            let path = tracking_path.with_extension("auth.db");
            std::fs::copy(auth_fixture_path(), &path).unwrap();
            let db = AuthDb::connect_and_verify_with(
                &format!("sqlite:///{}", path.display()),
                None,
                PoolConfig::default(),
            )
            .await
            .unwrap();
            let auth = AuthStore::new(db);
            state = state.with_auth_store(auth.clone());
            (Some(path), Some(auth))
        } else {
            (None, None)
        };

        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(state),
        );
        Self {
            tracking_path,
            auth_path,
            app,
            tracking,
            auth,
        }
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        self.request_as(method, path, body, None).await
    }

    async fn request_as(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        credentials: Option<(&str, &str)>,
    ) -> (StatusCode, Value) {
        let mut request = Request::builder().method(method).uri(path);
        if let Some((username, password)) = credentials {
            let value =
                base64::engine::general_purpose::STANDARD.encode(format!("{username}:{password}"));
            request = request.header(header::AUTHORIZATION, format!("Basic {value}"));
        }
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
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }

    async fn create_trace(&self, experiment_id: &str, trace_id: &str) {
        self.tracking
            .start_trace(
                WS,
                &StartTraceInput {
                    trace_id: trace_id.to_string(),
                    experiment_id: experiment_id.to_string(),
                    request_time: 1,
                    execution_duration: None,
                    state: "OK".to_string(),
                    client_request_id: None,
                    request_preview: None,
                    response_preview: None,
                    tags: vec![],
                    trace_metadata: vec![],
                    trace_metrics: vec![],
                },
            )
            .await
            .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.tracking_path);
        if let Some(path) = &self.auth_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[tokio::test]
async fn all_review_queue_rpcs_round_trip_on_both_prefixes() {
    for (tag, prefix) in [("api", "/api/3.0"), ("ajax", "/ajax-api/3.0")] {
        let fixture = Fixture::new(tag).await;
        let trace_id = format!("tr-review-{tag}");
        fixture.create_trace("0", &trace_id).await;
        let schema = fixture
            .tracking
            .create_label_schema(
                WS,
                "0",
                &format!("Review question {tag}"),
                LabelSchemaType::Feedback,
                &LabelSchemaInput::Text { max_length: None },
                None,
                true,
            )
            .await
            .unwrap();
        let base = format!("{prefix}/mlflow/review-queues");

        let (status, created) = fixture
            .request(
                Method::POST,
                &format!("{base}/create"),
                Some(json!({
                    "experiment_id": "0",
                    "name": "CaseQueue",
                    "queue_type": "CUSTOM",
                    "users": [" Reviewer ", "reviewer"],
                    "schema_ids": [schema.schema_id]
                })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{created}");
        let queue = &created["review_queue"];
        let queue_id = queue["queue_id"].as_str().unwrap();
        assert_eq!(queue["name"], "CaseQueue");
        assert_eq!(queue["queue_type"], "CUSTOM");
        assert_eq!(queue["users"], json!(["reviewer"]));

        let (status, duplicate) = fixture
            .request(
                Method::POST,
                &format!("{base}/create"),
                Some(json!({
                    "experiment_id": "0",
                    "name": "casequeue",
                    "queue_type": "CUSTOM"
                })),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{duplicate}");
        assert_eq!(duplicate["error_code"], "RESOURCE_ALREADY_EXISTS");

        let other_experiment_id = fixture
            .tracking
            .create_experiment(WS, &format!("Review queue name scope {tag}"), None, &[])
            .await
            .unwrap();
        let (status, same_name_elsewhere) = fixture
            .request(
                Method::POST,
                &format!("{base}/create"),
                Some(json!({
                    "experiment_id": other_experiment_id,
                    "name": "CASEQUEUE",
                    "queue_type": "CUSTOM"
                })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{same_name_elsewhere}");

        let (status, fetched) = fixture
            .request(
                Method::GET,
                &format!("{base}/get?queue_id={queue_id}"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{fetched}");
        assert_eq!(fetched["review_queue"]["queue_id"], queue_id);

        let (status, by_name) = fixture
            .request(
                Method::GET,
                &format!("{base}/get-by-name?experiment_id=0&name=CASEQUEUE"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{by_name}");
        assert_eq!(by_name["review_queue"]["queue_id"], queue_id);

        let (status, listed) = fixture
            .request(
                Method::GET,
                &format!("{base}/list?experiment_id=0&user=REVIEWER"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{listed}");
        assert_eq!(listed["review_queues"].as_array().unwrap().len(), 1);

        let (status, updated) = fixture
            .request(
                Method::POST,
                &format!("{base}/update"),
                Some(json!({
                    "queue_id": queue_id,
                    "update_users": true,
                    "users": ["second"],
                    "name": "Renamed Queue"
                })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{updated}");
        assert_eq!(updated["review_queue"]["users"], json!(["second"]));

        let (status, added) = fixture
            .request(
                Method::POST,
                &format!("{base}/items/add"),
                Some(json!({"queue_id": queue_id, "item_ids": [trace_id]})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{added}");
        assert_eq!(added["items"][0]["status"], "PENDING");
        assert_eq!(added["items"][0]["item_type"], "TRACE");

        let (status, items) = fixture
            .request(
                Method::GET,
                &format!("{base}/items/list?queue_id={queue_id}&status=PENDING"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{items}");
        assert_eq!(items["items"].as_array().unwrap().len(), 1);

        let (status, completed) = fixture
            .request(
                Method::POST,
                &format!("{base}/items/set-status"),
                Some(json!({
                    "queue_id": queue_id,
                    "item_id": trace_id,
                    "status": "COMPLETE",
                    "completed_by": " reviewer "
                })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{completed}");
        assert_eq!(completed["item"]["status"], "COMPLETE");
        assert_eq!(completed["item"]["completed_by"], "reviewer");

        let (status, removed) = fixture
            .request(
                Method::POST,
                &format!("{base}/items/remove"),
                Some(json!({"queue_id": queue_id, "item_ids": [trace_id]})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{removed}");
        assert_eq!(removed, json!({}));

        let (status, user_queue) = fixture
            .request(
                Method::POST,
                &format!("{base}/get-or-create-user"),
                Some(json!({"experiment_id": "0", "user": " User@Example.COM "})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{user_queue}");
        assert_eq!(user_queue["review_queue"]["name"], "user@example.com");
        assert_eq!(
            user_queue["review_queue"]["users"],
            json!(["user@example.com"])
        );
        assert!(user_queue["review_queue"]["schema_ids"].is_null());

        let (status, deleted) = fixture
            .request(
                Method::POST,
                &format!("{base}/delete"),
                Some(json!({"queue_id": queue_id})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{deleted}");
        assert_eq!(deleted, json!({}));
    }
}

#[tokio::test]
async fn soft_references_and_user_queue_schema_resolution_match_python() {
    let fixture = Fixture::new("soft-reference").await;
    fixture.create_trace("0", "tr-soft-reference").await;
    let schema = fixture
        .tracking
        .create_label_schema(
            WS,
            "0",
            "Soft reference question",
            LabelSchemaType::Feedback,
            &LabelSchemaInput::Text { max_length: None },
            None,
            true,
        )
        .await
        .unwrap();
    let custom = fixture
        .tracking
        .create_review_queue(
            WS,
            "0",
            "Soft refs",
            ReviewQueueType::Custom,
            None,
            &[],
            std::slice::from_ref(&schema.schema_id),
        )
        .await
        .unwrap();
    fixture
        .tracking
        .add_items_to_review_queue(
            WS,
            &custom.queue_id,
            &["tr-soft-reference".to_string()],
            ReviewItemType::Trace,
        )
        .await
        .unwrap();

    fixture
        .tracking
        .delete_label_schema(WS, &schema.schema_id)
        .await
        .unwrap();
    let literal = fixture
        .tracking
        .get_review_queue(WS, &custom.queue_id)
        .await
        .unwrap();
    assert_eq!(
        literal.schema_ids.as_slice(),
        std::slice::from_ref(&schema.schema_id)
    );
    let live = fixture
        .tracking
        .resolve_review_queue_schema_ids(WS, &custom.queue_id)
        .await
        .unwrap();
    assert!(!live.contains(&schema.schema_id));

    let user = fixture
        .tracking
        .get_or_create_user_queue(WS, "0", "Reviewer@Example.COM")
        .await
        .unwrap();
    assert!(user.schema_ids.is_empty());
    let inherited_before = fixture
        .tracking
        .resolve_review_queue_schema_ids(WS, &user.queue_id)
        .await
        .unwrap();
    let later_schema = fixture
        .tracking
        .create_label_schema(
            WS,
            "0",
            "Question created later",
            LabelSchemaType::Expectation,
            &LabelSchemaInput::Text {
                max_length: Some(100),
            },
            None,
            false,
        )
        .await
        .unwrap();
    let inherited_after = fixture
        .tracking
        .resolve_review_queue_schema_ids(WS, &user.queue_id)
        .await
        .unwrap();
    assert!(!inherited_before.contains(&later_schema.schema_id));
    assert!(inherited_after.contains(&later_schema.schema_id));
    assert!(fixture
        .tracking
        .get_review_queue(WS, &user.queue_id)
        .await
        .unwrap()
        .schema_ids
        .is_empty());

    fixture
        .tracking
        .delete_traces(
            WS,
            "0",
            None,
            None,
            Some(&["tr-soft-reference".to_string()]),
        )
        .await
        .unwrap();
    let items = fixture
        .tracking
        .list_review_queue_items(WS, &custom.queue_id, None, None, None)
        .await
        .unwrap();
    assert!(items.items.is_empty());
    assert_eq!(
        fixture
            .tracking
            .get_review_queue(WS, &custom.queue_id)
            .await
            .unwrap()
            .queue_id,
        custom.queue_id
    );
}

#[tokio::test]
async fn review_queue_operations_remain_workspace_scoped() {
    let fixture = Fixture::new("workspace").await;
    let experiment_id = fixture
        .tracking
        .create_experiment("review-workspace", "Workspace queue experiment", None, &[])
        .await
        .unwrap();
    let queue = fixture
        .tracking
        .create_review_queue(
            "review-workspace",
            &experiment_id,
            "Workspace queue",
            ReviewQueueType::Custom,
            Some("owner"),
            &[],
            &[],
        )
        .await
        .unwrap();

    let error = fixture
        .tracking
        .get_review_queue(WS, &queue.queue_id)
        .await
        .unwrap_err();
    assert_eq!(
        error.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    fixture
        .tracking
        .delete_review_queue(WS, &queue.queue_id)
        .await
        .unwrap();
    assert_eq!(
        fixture
            .tracking
            .get_review_queue("review-workspace", &queue.queue_id)
            .await
            .unwrap()
            .name,
        "Workspace queue"
    );
}

#[tokio::test]
async fn review_queue_auth_ports_owner_member_manager_and_integrity_rules() {
    let fixture = Fixture::with_auth("auth").await;
    let auth = fixture.auth.as_ref().unwrap();
    auth.create_user("queue_reader", "queue-reader-password", false)
        .await
        .unwrap();
    auth.grant_user_permission("queue_reader", "experiment", "0", "READ", WS)
        .await
        .unwrap();

    let base = "/api/3.0/mlflow/review-queues";
    let (status, created) = fixture
        .request_as(
            Method::POST,
            &format!("{base}/create"),
            Some(json!({
                "experiment_id": "0",
                "name": "Owned by Bob",
                "queue_type": "CUSTOM",
                "users": ["queue_reader"]
            })),
            Some(BOB),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(created["review_queue"]["created_by"], BOB.0);
    let queue_id = created["review_queue"]["queue_id"].as_str().unwrap();

    let (status, member_view) = fixture
        .request_as(
            Method::GET,
            &format!("{base}/get?queue_id={queue_id}"),
            None,
            Some(("queue_reader", "queue-reader-password")),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{member_view}");

    let (status, denied_update) = fixture
        .request_as(
            Method::POST,
            &format!("{base}/update"),
            Some(json!({"queue_id": queue_id, "name": "not allowed"})),
            Some(("queue_reader", "queue-reader-password")),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{denied_update}");

    let (status, owner_update) = fixture
        .request_as(
            Method::POST,
            &format!("{base}/update"),
            Some(json!({"queue_id": queue_id, "name": "Owner renamed"})),
            Some(BOB),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{owner_update}");

    let (status, shadowed) = fixture
        .request_as(
            Method::POST,
            &format!("{base}/create"),
            Some(json!({
                "experiment_id": "0",
                "name": BOB.0,
                "queue_type": "CUSTOM"
            })),
            Some(ALICE),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{shadowed}");

    let (status, denied_delete) = fixture
        .request_as(
            Method::POST,
            &format!("{base}/delete"),
            Some(json!({"queue_id": queue_id})),
            Some(("queue_reader", "queue-reader-password")),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{denied_delete}");

    let (status, deleted) = fixture
        .request_as(
            Method::POST,
            &format!("{base}/delete"),
            Some(json!({"queue_id": queue_id})),
            Some(BOB),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
}
