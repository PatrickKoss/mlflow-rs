//! Webhook request assembly (the wrapped payload + signed headers) shared by
//! the `/test` path and the async delivery engine, plus the `/test` entry point
//! itself. Ports `_send_webhook_request` + `test_webhook`
//! (`mlflow/webhooks/delivery.py:142-357`).
//!
//! ## What [`build_signed_request`] does (matching `_send_webhook_request`)
//!
//! 1. Wrap the event's `data` payload as
//!    `{entity, action, timestamp, data[, workspace]}`, JSON-encoded (field
//!    order preserved — `serde_json` has `preserve_order` on workspace-wide).
//! 2. Generate a uuid4 `delivery_id` and a unix-second `timestamp`.
//! 3. Emit `X-MLflow-Delivery-Id`, `X-MLflow-Timestamp`, and — when the webhook
//!    has a secret — `X-MLflow-Signature: v1,<b64>` over
//!    `"{delivery_id}.{timestamp}.{payload}"`.
//!
//! The result is a [`SignedRequest`]; both callers hand it to
//! [`crate::http_send::send_with_ssrf_guard`], so the test path and the async
//! engine share the exact same signing + connect-time SSRF behavior.
//!
//! ## `/test`
//!
//! [`test_webhook`] picks the event (caller's, else the webhook's first),
//! builds the example payload, sends one request, and returns a
//! [`WebhookTestResult`] — never `Err`, mirroring Python's `try/except` that
//! wraps the whole send.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::entities::{Webhook, WebhookEvent, WebhookTestResult};
use crate::http_send::{send_with_ssrf_guard, Resolver, SendConfig, SignedRequest, SystemResolver};
use crate::payloads::example_payload_for_event;
use crate::signing::{
    generate_hmac_signature, WEBHOOK_DELIVERY_ID_HEADER, WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
};
use crate::validation::validate_webhook_url;

/// `MLFLOW_ENABLE_WORKSPACES` — when set, the payload includes `workspace`.
const ENABLE_WORKSPACES_ENV: &str = "MLFLOW_ENABLE_WORKSPACES";

/// Assemble the wrapped, signed webhook request for `event`/`payload` — the
/// shared core of `_send_webhook_request` (`delivery.py:142`) used by both the
/// `/test` path and the async delivery engine. `payload` is the event's `data`
/// object. Returns an error string (used as the test result's `error_message`)
/// only when the payload can't be serialized, which shouldn't happen.
pub(crate) fn build_signed_request(
    webhook: &Webhook,
    event: WebhookEvent,
    payload: &Value,
    workspace: &str,
) -> Result<SignedRequest, String> {
    // Wrapped payload: {entity, action, timestamp, data[, workspace]}.
    let mut wrapped: Map<String, Value> = Map::new();
    wrapped.insert(
        "entity".into(),
        Value::String(event.entity.as_db_str().to_string()),
    );
    wrapped.insert(
        "action".into(),
        Value::String(event.action.as_db_str().to_string()),
    );
    wrapped.insert("timestamp".into(), Value::String(iso8601_now_utc()));
    wrapped.insert("data".into(), payload.clone());
    if workspaces_enabled() {
        wrapped.insert("workspace".into(), Value::String(workspace.to_string()));
    }
    let payload_json = Value::Object(wrapped).to_string();

    let delivery_id = new_uuid();
    let unix_timestamp = chrono::Utc::now().timestamp().to_string();

    let mut headers: Vec<(&'static str, String)> = vec![
        (WEBHOOK_DELIVERY_ID_HEADER, delivery_id.clone()),
        (WEBHOOK_TIMESTAMP_HEADER, unix_timestamp.clone()),
    ];
    if let Some(secret) = webhook.secret.as_deref().filter(|s| !s.is_empty()) {
        let signature =
            generate_hmac_signature(secret, &delivery_id, &unix_timestamp, &payload_json);
        headers.push((WEBHOOK_SIGNATURE_HEADER, signature));
    }

    Ok(SignedRequest {
        url: webhook.url.clone(),
        body: payload_json,
        headers,
    })
}

/// `test_webhook(webhook, event)` (`delivery.py:329`). Never returns `Err`: a
/// failure is reported inside the [`WebhookTestResult`], exactly like Python's
/// `try/except` that wraps the whole send.
pub async fn test_webhook(webhook: &Webhook, event: Option<WebhookEvent>) -> WebhookTestResult {
    test_webhook_with_resolver(webhook, event, Arc::new(SystemResolver)).await
}

/// [`test_webhook`] with an explicit resolver seam (for tests). The `/test` path
/// re-validates the URL (`_validate_webhook_url`) and delivers through the same
/// connect-time SSRF-guarded sender the async engine uses.
pub async fn test_webhook_with_resolver(
    webhook: &Webhook,
    event: Option<WebhookEvent>,
    resolver: Arc<dyn Resolver>,
) -> WebhookTestResult {
    // `test_event = event or webhook.events[0]`. An empty event list can't
    // happen for a stored webhook (create validates non-empty), but guard it.
    let test_event = match event.or_else(|| webhook.events.first().copied()) {
        Some(ev) => ev,
        None => return failure("Failed to test webhook: webhook has no events".to_string()),
    };

    // `_validate_webhook_url(webhook.url)` — re-validate + resolve-time SSRF gate.
    if let Err(e) = validate_webhook_url(&webhook.url) {
        return failure(format!("Failed to test webhook: {}", e.message));
    }

    let data = match example_payload_for_event(test_event) {
        Some(d) => d,
        None => {
            return failure(format!(
                "Failed to test webhook: Unknown event type: {}.{}",
                test_event.entity.as_db_str(),
                test_event.action.as_db_str()
            ));
        }
    };

    let signed = match build_signed_request(webhook, test_event, &data, &webhook.workspace) {
        Ok(s) => s,
        Err(msg) => return failure(format!("Failed to test webhook: {msg}")),
    };

    match send_with_ssrf_guard(&signed, test_send_config(), resolver).await {
        Ok(resp) => WebhookTestResult {
            success: resp.status < 400,
            response_status: Some(i32::from(resp.status)),
            response_body: Some(resp.body),
            error_message: None,
        },
        Err(e) => failure(format!("Failed to test webhook: {e}")),
    }
}

/// The `/test` path uses the request timeout but **no retries** (Python's
/// `test_webhook` reports the first response as-is), unlike the async engine.
fn test_send_config() -> SendConfig {
    SendConfig {
        max_retries: 0,
        ..SendConfig::from_env()
    }
}

fn failure(error_message: String) -> WebhookTestResult {
    WebhookTestResult {
        success: false,
        response_status: None,
        response_body: None,
        error_message: Some(error_message),
    }
}

fn workspaces_enabled() -> bool {
    matches!(
        std::env::var(ENABLE_WORKSPACES_ENV).ok().as_deref(),
        Some("true" | "True" | "TRUE" | "1")
    )
}

/// `datetime.now(timezone.utc).isoformat()` — RFC3339 with microseconds and
/// `+00:00` offset (Python renders the UTC offset, not `Z`).
fn iso8601_now_utc() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.6f+00:00")
        .to_string()
}

/// `str(uuid.uuid4())`.
fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}
