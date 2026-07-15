//! Behavioral tests for [`mlflow_webhooks::WebhookStore`] (plan T8.1), ported
//! from the Python webhook store suite
//! (`tests/store/model_registry/test_sqlalchemy_store.py` webhook cases).
//!
//! Each test copies the checked-in Alembic-migrated SQLite fixture (shared with
//! `mlflow-store`, already containing the `webhooks`/`webhook_events` tables) to
//! a temp file so the fixture is never mutated.

use std::path::{Path, PathBuf};

use mlflow_error::ErrorCode;
use mlflow_store::{Db, PoolConfig};
use mlflow_webhooks::{
    SecretCipher, Webhook, WebhookAction, WebhookEntity, WebhookEvent, WebhookStatus, WebhookStore,
};

const WS: &str = "default";
const OTHER_WS: &str = "team-a";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("mlflow-store")
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_webhookstore_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixture_path(), &path).expect("copy fixture");
        TempDb { path }
    }

    fn uri(&self) -> String {
        format!("sqlite:///{}", self.path.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn store(tag: &str) -> (WebhookStore, TempDb) {
    let db_file = TempDb::new(tag);
    let db = Db::connect(&db_file.uri(), PoolConfig::default())
        .await
        .expect("connect temp fixture");
    // Pin a fixed cipher so secrets round-trip deterministically within a test.
    let cipher = SecretCipher::from_key("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=").unwrap();
    (WebhookStore::with_cipher(db, cipher), db_file)
}

fn ev(entity: WebhookEntity, action: WebhookAction) -> WebhookEvent {
    WebhookEvent::new(entity, action)
}

#[tokio::test]
async fn create_and_get_round_trips_all_fields() {
    let (store, _db) = store("create_get").await;
    let events = vec![
        ev(WebhookEntity::RegisteredModel, WebhookAction::Created),
        ev(WebhookEntity::ModelVersionTag, WebhookAction::Set),
    ];
    let created = store
        .create_webhook(
            WS,
            "my-hook",
            "https://example.com/hook",
            &events,
            Some("a description"),
            Some("s3cr3t"),
            Some(WebhookStatus::Disabled),
        )
        .await
        .expect("create");

    assert_eq!(created.name, "my-hook");
    assert_eq!(created.url, "https://example.com/hook");
    assert_eq!(created.description.as_deref(), Some("a description"));
    assert_eq!(created.status, WebhookStatus::Disabled);
    assert_eq!(created.secret.as_deref(), Some("s3cr3t"));
    assert_eq!(created.workspace, WS);
    assert!(created.creation_timestamp.is_some());
    assert_eq!(created.creation_timestamp, created.last_updated_timestamp);
    assert_eq!(created.events.len(), 2);

    let got = store.get_webhook(WS, &created.webhook_id).await.unwrap();
    assert_eq!(got.webhook_id, created.webhook_id);
    assert_eq!(got.secret.as_deref(), Some("s3cr3t"));
    assert_events_eq(&got, &events);
}

fn assert_events_eq(webhook: &Webhook, expected: &[WebhookEvent]) {
    let mut got: Vec<WebhookEvent> = webhook.events.clone();
    let mut want: Vec<WebhookEvent> = expected.to_vec();
    got.sort_by_key(|e| (e.entity.as_db_str(), e.action.as_db_str()));
    want.sort_by_key(|e| (e.entity.as_db_str(), e.action.as_db_str()));
    assert_eq!(got, want);
}

#[tokio::test]
async fn create_defaults_status_active_and_null_secret() {
    let (store, _db) = store("defaults").await;
    let created = store
        .create_webhook(
            WS,
            "hook2",
            "https://example.com/h",
            &[ev(WebhookEntity::ModelVersion, WebhookAction::Created)],
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(created.status, WebhookStatus::Active);
    assert_eq!(created.secret, None);
    assert_eq!(created.description, None);
}

#[tokio::test]
async fn get_missing_is_resource_does_not_exist() {
    let (store, _db) = store("get_missing").await;
    let err = store.get_webhook(WS, "does-not-exist").await.unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert!(err.message.contains("does-not-exist"));
}

#[tokio::test]
async fn workspace_isolation_hides_other_workspace_webhook() {
    let (store, _db) = store("ws_iso").await;
    let created = store
        .create_webhook(
            OTHER_WS,
            "team-hook",
            "https://example.com/h",
            &[ev(WebhookEntity::RegisteredModel, WebhookAction::Created)],
            None,
            None,
            None,
        )
        .await
        .unwrap();

    // Not visible from the default workspace.
    let err = store
        .get_webhook(WS, &created.webhook_id)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);

    // Visible from its own workspace.
    let got = store
        .get_webhook(OTHER_WS, &created.webhook_id)
        .await
        .unwrap();
    assert_eq!(got.workspace, OTHER_WS);

    // Listing the default workspace excludes it.
    let page = store.list_webhooks(WS, None, None).await.unwrap();
    assert!(page
        .webhooks
        .iter()
        .all(|w| w.webhook_id != created.webhook_id));
}

#[tokio::test]
async fn update_partial_fields_and_replaces_events() {
    let (store, _db) = store("update").await;
    let created = store
        .create_webhook(
            WS,
            "hook",
            "https://example.com/h",
            &[ev(WebhookEntity::RegisteredModel, WebhookAction::Created)],
            None,
            None,
            None,
        )
        .await
        .unwrap();

    let new_events = vec![
        ev(WebhookEntity::ModelVersionAlias, WebhookAction::Created),
        ev(WebhookEntity::ModelVersionAlias, WebhookAction::Deleted),
    ];
    let updated = store
        .update_webhook(
            WS,
            &created.webhook_id,
            Some("renamed"),
            Some("new desc"),
            None,
            Some(&new_events),
            Some("newsecret"),
            Some(WebhookStatus::Disabled),
        )
        .await
        .unwrap();

    assert_eq!(updated.name, "renamed");
    assert_eq!(updated.description.as_deref(), Some("new desc"));
    assert_eq!(updated.url, "https://example.com/h"); // untouched
    assert_eq!(updated.status, WebhookStatus::Disabled);
    assert_eq!(updated.secret.as_deref(), Some("newsecret"));
    assert_events_eq(&updated, &new_events);
    assert!(updated.last_updated_timestamp >= created.last_updated_timestamp);
}

#[tokio::test]
async fn update_missing_is_resource_does_not_exist() {
    let (store, _db) = store("update_missing").await;
    let err = store
        .update_webhook(WS, "nope", Some("x"), None, None, None, None, None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
}

#[tokio::test]
async fn soft_delete_hides_from_get_and_list() {
    let (store, _db) = store("soft_delete").await;
    let created = store
        .create_webhook(
            WS,
            "hook",
            "https://example.com/h",
            &[ev(WebhookEntity::RegisteredModel, WebhookAction::Created)],
            None,
            None,
            None,
        )
        .await
        .unwrap();

    store.delete_webhook(WS, &created.webhook_id).await.unwrap();

    let err = store
        .get_webhook(WS, &created.webhook_id)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);

    let page = store.list_webhooks(WS, None, None).await.unwrap();
    assert!(page
        .webhooks
        .iter()
        .all(|w| w.webhook_id != created.webhook_id));

    // Deleting again is a not-found (the row is already soft-deleted).
    let err = store
        .delete_webhook(WS, &created.webhook_id)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
}

#[tokio::test]
async fn list_paginates_with_offset_token() {
    let (store, _db) = store("list_page").await;
    // Create 3 webhooks; creation_timestamp DESC ordering means newest first.
    let mut ids = Vec::new();
    for i in 0..3 {
        // Ensure distinct creation timestamps for a stable order.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let w = store
            .create_webhook(
                WS,
                &format!("hook-{i}"),
                "https://example.com/h",
                &[ev(WebhookEntity::RegisteredModel, WebhookAction::Created)],
                None,
                None,
                None,
            )
            .await
            .unwrap();
        ids.push(w.webhook_id);
    }

    let page1 = store.list_webhooks(WS, Some(2), None).await.unwrap();
    assert_eq!(page1.webhooks.len(), 2);
    let token = page1.next_page_token.expect("should have next page");

    let page2 = store
        .list_webhooks(WS, Some(2), Some(&token))
        .await
        .unwrap();
    assert!(!page2.webhooks.is_empty());

    // No duplicates across pages.
    let mut seen: Vec<&str> = page1
        .webhooks
        .iter()
        .map(|w| w.webhook_id.as_str())
        .collect();
    for w in &page2.webhooks {
        assert!(!seen.contains(&w.webhook_id.as_str()));
        seen.push(&w.webhook_id);
    }
}

#[tokio::test]
async fn list_max_results_out_of_range_is_error() {
    let (store, _db) = store("list_range").await;
    assert_eq!(
        store
            .list_webhooks(WS, Some(0), None)
            .await
            .unwrap_err()
            .error_code,
        ErrorCode::InvalidParameterValue
    );
    assert_eq!(
        store
            .list_webhooks(WS, Some(1001), None)
            .await
            .unwrap_err()
            .error_code,
        ErrorCode::InvalidParameterValue
    );
}

#[tokio::test]
async fn create_invalid_event_combination_rejected() {
    let (store, _db) = store("bad_event").await;
    // registered_model only supports `created`; `deleted` is invalid.
    let err = store
        .create_webhook(
            WS,
            "hook",
            "https://example.com/h",
            &[ev(WebhookEntity::RegisteredModel, WebhookAction::Deleted)],
            None,
            None,
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err.message.contains("Invalid action 'deleted'"));
}

#[tokio::test]
async fn list_by_event_filters_subscription() {
    let (store, _db) = store("by_event").await;
    let a = store
        .create_webhook(
            WS,
            "hook-a",
            "https://example.com/h",
            &[ev(WebhookEntity::RegisteredModel, WebhookAction::Created)],
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let _b = store
        .create_webhook(
            WS,
            "hook-b",
            "https://example.com/h",
            &[ev(WebhookEntity::ModelVersion, WebhookAction::Created)],
            None,
            None,
            None,
        )
        .await
        .unwrap();

    let page = store
        .list_webhooks_by_event(
            WS,
            ev(WebhookEntity::RegisteredModel, WebhookAction::Created),
            None,
            None,
        )
        .await
        .unwrap();
    assert!(page.webhooks.iter().any(|w| w.webhook_id == a.webhook_id));
    assert!(page.webhooks.iter().all(|w| w.name != "hook-b"));
}
