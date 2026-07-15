//! The `/test` endpoint's single real webhook delivery, porting the
//! test-delivery slice of `mlflow/webhooks/delivery.py`
//! (`_send_webhook_request` + `test_webhook`, `delivery.py:142-357`).
//!
//! Only the *test* path lives here. The full fire-and-forget async delivery
//! engine (thread pool, TTL cache by event, retries on `[429,500,502,503,504]`
//! with backoff, and the connect-time `SSRFProtectedHTTPAdapter`) is T8.3.
//!
//! ## What this does (matching Python's `test_webhook`)
//!
//! 1. Pick the event: the caller's event, else the webhook's first event.
//! 2. Build the example payload for that event
//!    (`get_example_payload_for_event`) wrapped as
//!    `{entity, action, timestamp, data[, workspace]}`, JSON-encoded.
//! 3. Generate a uuid4 `delivery_id` and a unix-second `timestamp`.
//! 4. Send `POST <url>` with `Content-Type: application/json`,
//!    `X-MLflow-Delivery-Id`, `X-MLflow-Timestamp`, and — when the webhook has a
//!    secret — `X-MLflow-Signature: v1,<b64>` over
//!    `"{delivery_id}.{timestamp}.{payload}"`.
//! 5. Return `WebhookTestResult{ success: status < 400, response_status,
//!    response_body }`; any error yields
//!    `WebhookTestResult{ success:false, error_message:"Failed to test webhook: <repr>" }`.
//!
//! ## SSRF on the test path
//!
//! Python's `_send_webhook_request` calls `_validate_webhook_url(webhook.url)`
//! (resolve + reject non-public IPs) *and* routes through the connect-time
//! SSRF adapter. We apply [`crate::validation::validate_webhook_url`] before the
//! request (the same resolve-and-check), which conservatively rejects a URL
//! resolving to a private IP unless `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` is set.
//! Tests that target a local listener set that flag, matching how a dev server
//! must to deliver to localhost.

use std::time::Duration;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::header::{CONTENT_TYPE, HOST};
use hyper::Request;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::{Map, Value};

use crate::entities::WebhookTestResult;
use crate::entities::{Webhook, WebhookEvent};
use crate::payloads::example_payload_for_event;
use crate::signing::{
    generate_hmac_signature, WEBHOOK_DELIVERY_ID_HEADER, WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
};
use crate::validation::validate_webhook_url;

/// `MLFLOW_WEBHOOK_REQUEST_TIMEOUT` (default 30s,
/// `mlflow/environment_variables.py:1373`).
const REQUEST_TIMEOUT_ENV: &str = "MLFLOW_WEBHOOK_REQUEST_TIMEOUT";
/// `MLFLOW_ENABLE_WORKSPACES` — when set, the payload includes `workspace`.
const ENABLE_WORKSPACES_ENV: &str = "MLFLOW_ENABLE_WORKSPACES";

/// `test_webhook(webhook, event)` (`delivery.py:329`). Never returns `Err`: a
/// failure is reported inside the [`WebhookTestResult`], exactly like Python's
/// `try/except` that wraps the whole send.
pub async fn test_webhook(webhook: &Webhook, event: Option<WebhookEvent>) -> WebhookTestResult {
    // `test_event = event or webhook.events[0]`. An empty event list can't
    // happen for a stored webhook (create validates non-empty), but guard it.
    let test_event = match event.or_else(|| webhook.events.first().copied()) {
        Some(ev) => ev,
        None => {
            return failure("Failed to test webhook: webhook has no events".to_string());
        }
    };
    match send_test_request(webhook, test_event).await {
        Ok((status, body)) => WebhookTestResult {
            success: status < 400,
            response_status: Some(i32::from(status)),
            response_body: Some(body),
            error_message: None,
        },
        Err(msg) => failure(format!("Failed to test webhook: {msg}")),
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

/// `_send_webhook_request` for the test path: build the wrapped payload, sign,
/// and POST. Returns `(status_code, response_body)` or an error string that
/// becomes the test result's `error_message`.
async fn send_test_request(
    webhook: &Webhook,
    event: WebhookEvent,
) -> Result<(u16, String), String> {
    // `_validate_webhook_url(webhook.url)` — re-validate + SSRF resolve gate.
    validate_webhook_url(&webhook.url).map_err(|e| e.message.clone())?;

    let data = example_payload_for_event(event).ok_or_else(|| {
        format!(
            "Unknown event type: {}.{}",
            event.entity.as_db_str(),
            event.action.as_db_str()
        )
    })?;

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
    wrapped.insert("data".into(), data);
    if workspaces_enabled() {
        wrapped.insert("workspace".into(), Value::String(webhook.workspace.clone()));
    }
    let payload_json = Value::Object(wrapped).to_string();

    let delivery_id = new_uuid();
    let unix_timestamp = (chrono::Utc::now().timestamp()).to_string();

    let authority = parse_authority(&webhook.url)?;
    let mut builder = Request::builder()
        .method("POST")
        .uri(&webhook.url)
        .header(CONTENT_TYPE, "application/json")
        .header(HOST, authority)
        .header(WEBHOOK_DELIVERY_ID_HEADER, &delivery_id)
        .header(WEBHOOK_TIMESTAMP_HEADER, &unix_timestamp);
    if let Some(secret) = webhook.secret.as_deref().filter(|s| !s.is_empty()) {
        let signature =
            generate_hmac_signature(secret, &delivery_id, &unix_timestamp, &payload_json);
        builder = builder.header(WEBHOOK_SIGNATURE_HEADER, signature);
    }

    let request = builder
        .body(Full::new(Bytes::from(payload_json)))
        .map_err(|e| format!("failed to build request: {e}"))?;

    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let timeout = request_timeout();

    let response = tokio::time::timeout(timeout, client.request(request))
        .await
        .map_err(|_| format!("request timed out after {}s", timeout.as_secs()))?
        .map_err(|e| format!("{e}"))?;

    let status = response.status().as_u16();
    let body_bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|e| format!("failed to read response body: {e}"))?
        .to_bytes();
    let body = String::from_utf8_lossy(&body_bytes).into_owned();
    Ok((status, body))
}

fn workspaces_enabled() -> bool {
    matches!(
        std::env::var(ENABLE_WORKSPACES_ENV).ok().as_deref(),
        Some("true" | "True" | "TRUE" | "1")
    )
}

fn request_timeout() -> Duration {
    let secs = std::env::var(REQUEST_TIMEOUT_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    Duration::from_secs(secs)
}

/// `datetime.now(timezone.utc).isoformat()` — RFC3339 with microseconds and
/// `+00:00` offset (Python renders the UTC offset, not `Z`).
fn iso8601_now_utc() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.6f+00:00")
        .to_string()
}

/// Extract the `host[:port]` authority from a URL for the `Host` header.
/// hyper's low-level client does not auto-populate `Host` from an absolute URI.
fn parse_authority(url: &str) -> Result<String, String> {
    let uri: hyper::Uri = url
        .parse()
        .map_err(|e| format!("invalid webhook URL: {e}"))?;
    let host = uri
        .host()
        .ok_or_else(|| "webhook URL has no host".to_string())?;
    Ok(match uri.port_u16() {
        Some(p) => format!("{host}:{p}"),
        None => host.to_string(),
    })
}

/// `str(uuid.uuid4())` (see [`crate::store`] for the same minimal generator).
fn new_uuid() -> String {
    use base64::Engine;
    let key = fernet::Fernet::generate_key();
    let raw = base64::engine::general_purpose::URL_SAFE
        .decode(key.as_bytes())
        .unwrap_or_else(|_| vec![0u8; 32]);
    let mut bytes = [0u8; 16];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = raw.get(i).copied().unwrap_or(0);
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let h = |b: u8| format!("{b:02x}");
    format!(
        "{}{}{}{}-{}{}-{}{}-{}{}-{}{}{}{}{}{}",
        h(bytes[0]),
        h(bytes[1]),
        h(bytes[2]),
        h(bytes[3]),
        h(bytes[4]),
        h(bytes[5]),
        h(bytes[6]),
        h(bytes[7]),
        h(bytes[8]),
        h(bytes[9]),
        h(bytes[10]),
        h(bytes[11]),
        h(bytes[12]),
        h(bytes[13]),
        h(bytes[14]),
        h(bytes[15]),
    )
}
