//! Webhook REST endpoints (plan T8.2, §4.16): the 6 `WebhookService` routes
//! `POST/GET /mlflow/webhooks`, `GET/PATCH/DELETE /mlflow/webhooks/{webhook_id}`,
//! `POST /mlflow/webhooks/{webhook_id}/test`.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`
//! (`_create_webhook`..`_test_webhook`, `handlers.py:3367-3473`): parse the
//! request proto (via [`crate::proto_http`], using the shared path-param overlay
//! for `{webhook_id}`), run the same required-field / enum validation, call the
//! workspace-scoped [`mlflow_webhooks::WebhookStore`], then serialize the
//! response proto.
//!
//! ## Secrets are never returned
//!
//! Python's `Webhook.to_proto` (`mlflow/entities/webhook.py:357`) has no
//! `secret` field, so no create/get/list/update response ever echoes a secret.
//! [`to_proto_webhook`] mirrors that: the entity's decrypted `secret` is dropped.
//!
//! ## `/test` fires a real request
//!
//! `_test_webhook` -> `test_webhook` sends one real HTTP POST to the webhook URL
//! with the example payload, the `X-MLflow-*` headers, and (when a secret is
//! set) the `v1,<b64>` HMAC signature, returning a `WebhookTestResult`. That
//! lives in [`mlflow_webhooks::delivery`]; the full async delivery engine is
//! T8.3.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_webhooks::{
    Webhook, WebhookAction, WebhookEntity, WebhookEvent, WebhookStatus, WebhookTestResult,
};

use crate::proto_http::{parse_request, parse_request_with_path_params, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `_create_webhook` (`handlers.py:3367`), path: `POST /mlflow/webhooks`.
pub async fn create_webhook(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateWebhook = parse_request(&parts, &body, "mlflow.CreateWebhook")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let url = require_non_empty(req.url.as_deref(), "url")?;
    if req.events.is_empty() {
        return Err(missing_param("events"));
    }
    let events = events_from_proto(&req.events)?;
    let status = status_from_proto(req.status)?;

    let webhook = state
        .webhook_store()?
        .create_webhook(
            workspace.name(),
            name,
            url,
            &events,
            req.description.as_deref().filter(|s| !s.is_empty()),
            req.secret.as_deref().filter(|s| !s.is_empty()),
            status,
        )
        .await?;

    let resp = pb::create_webhook::Response {
        webhook: Some(to_proto_webhook(webhook)),
    };
    proto_response(&resp, "mlflow.CreateWebhook.Response")
}

/// `_list_webhooks` (`handlers.py:3394`), path: `GET /mlflow/webhooks`.
pub async fn list_webhooks(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::ListWebhooks = parse_request(&parts, &body, "mlflow.ListWebhooks")?;
    let page = state
        .webhook_store()?
        .list_webhooks(
            workspace.name(),
            req.max_results,
            req.page_token.as_deref().filter(|s| !s.is_empty()),
        )
        .await?;

    let resp = pb::list_webhooks::Response {
        webhooks: page.webhooks.into_iter().map(to_proto_webhook).collect(),
        next_page_token: page.next_page_token,
    };
    proto_response(&resp, "mlflow.ListWebhooks.Response")
}

/// `_get_webhook` (`handlers.py:3415`), path: `GET /mlflow/webhooks/{webhook_id}`.
pub async fn get_webhook(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let req: pb::GetWebhook = parse_request_with_path_params(
        &parts,
        &Bytes::new(),
        "mlflow.GetWebhook",
        &path_param_pairs(&path_params, &["webhook_id"]),
    )?;
    let webhook_id = require_non_empty(req.webhook_id.as_deref(), "webhook_id")?;

    let webhook = state
        .webhook_store()?
        .get_webhook(workspace.name(), webhook_id)
        .await?;
    let resp = pb::get_webhook::Response {
        webhook: Some(to_proto_webhook(webhook)),
    };
    proto_response(&resp, "mlflow.GetWebhook.Response")
}

/// `_update_webhook` (`handlers.py:3421`), path: `PATCH /mlflow/webhooks/{webhook_id}`.
pub async fn update_webhook(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateWebhook = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.UpdateWebhook",
        &path_param_pairs(&path_params, &["webhook_id"]),
    )?;
    let webhook_id = require_non_empty(req.webhook_id.as_deref(), "webhook_id")?;

    // Python passes `... or None` for every optional field: an empty string is
    // treated as "not provided". `events` is provided only when non-empty.
    let events_vec;
    let events: Option<&[WebhookEvent]> = if req.events.is_empty() {
        None
    } else {
        events_vec = events_from_proto(&req.events)?;
        Some(&events_vec)
    };
    let status = status_from_proto(req.status)?;

    let webhook = state
        .webhook_store()?
        .update_webhook(
            workspace.name(),
            webhook_id,
            req.name.as_deref().filter(|s| !s.is_empty()),
            req.description.as_deref().filter(|s| !s.is_empty()),
            req.url.as_deref().filter(|s| !s.is_empty()),
            events,
            req.secret.as_deref().filter(|s| !s.is_empty()),
            status,
        )
        .await?;

    let resp = pb::update_webhook::Response {
        webhook: Some(to_proto_webhook(webhook)),
    };
    proto_response(&resp, "mlflow.UpdateWebhook.Response")
}

/// `_delete_webhook` (`handlers.py:3452`), path: `DELETE /mlflow/webhooks/{webhook_id}`.
pub async fn delete_webhook(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteWebhook = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.DeleteWebhook",
        &path_param_pairs(&path_params, &["webhook_id"]),
    )?;
    let webhook_id = require_non_empty(req.webhook_id.as_deref(), "webhook_id")?;

    state
        .webhook_store()?
        .delete_webhook(workspace.name(), webhook_id)
        .await?;
    proto_response(
        &pb::delete_webhook::Response {},
        "mlflow.DeleteWebhook.Response",
    )
}

/// `_test_webhook` (`handlers.py:3460`), path: `POST /mlflow/webhooks/{webhook_id}/test`.
pub async fn test_webhook(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::TestWebhook = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.TestWebhook",
        &path_param_pairs(&path_params, &["webhook_id"]),
    )?;
    let webhook_id = require_non_empty(req.webhook_id.as_deref(), "webhook_id")?;

    // `event = WebhookEvent.from_proto(...) if request_message.HasField("event") else None`.
    let event = match req.event {
        Some(ev) => Some(event_from_proto(&ev)?),
        None => None,
    };

    let store = state.webhook_store()?;
    let webhook = store.get_webhook(workspace.name(), webhook_id).await?;
    let result = mlflow_webhooks::delivery::test_webhook(&webhook, event).await;

    let resp = pb::test_webhook::Response {
        result: Some(to_proto_test_result(result)),
    };
    proto_response(&resp, "mlflow.TestWebhook.Response")
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// `Webhook.to_proto` (`mlflow/entities/webhook.py:357`). Note: **no secret**.
/// `description` is only set when non-empty (Python guards `if self.description`).
fn to_proto_webhook(webhook: Webhook) -> pb::Webhook {
    pb::Webhook {
        webhook_id: Some(webhook.webhook_id),
        name: Some(webhook.name),
        description: webhook.description.filter(|d| !d.is_empty()),
        url: Some(webhook.url),
        events: webhook.events.iter().map(to_proto_event).collect(),
        status: Some(webhook.status.to_proto_i32()),
        creation_timestamp: webhook.creation_timestamp,
        last_updated_timestamp: webhook.last_updated_timestamp,
    }
}

fn to_proto_event(event: &WebhookEvent) -> pb::WebhookEvent {
    pb::WebhookEvent {
        entity: Some(event.entity.to_proto_i32()),
        action: Some(event.action.to_proto_i32()),
    }
}

/// `WebhookTestResult.to_proto` (`mlflow/entities/webhook.py:436`).
fn to_proto_test_result(result: WebhookTestResult) -> pb::WebhookTestResult {
    pb::WebhookTestResult {
        success: Some(result.success),
        response_status: result.response_status,
        response_body: result.response_body,
        error_message: result.error_message,
    }
}

/// `[WebhookEvent.from_proto(e) for e in events]` (`handlers.py:3383`).
fn events_from_proto(events: &[pb::WebhookEvent]) -> Result<Vec<WebhookEvent>, MlflowError> {
    events.iter().map(event_from_proto).collect()
}

/// `WebhookEvent.from_proto` (`mlflow/entities/webhook.py:202`), including the
/// `ENTITY_UNSPECIFIED`/`ACTION_UNSPECIFIED` and unknown-value rejection that
/// Python surfaces as a `ValueError` -> invalid-parameter error, plus the
/// entity/action combination validity check.
fn event_from_proto(event: &pb::WebhookEvent) -> Result<WebhookEvent, MlflowError> {
    let entity_i = event.entity.unwrap_or(0);
    let action_i = event.action.unwrap_or(0);
    let entity = WebhookEntity::from_proto_i32(entity_i).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "Unknown or unspecified webhook entity: {entity_i}"
        ))
    })?;
    let action = WebhookAction::from_proto_i32(action_i).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "Unknown or unspecified webhook action: {action_i}"
        ))
    })?;
    mlflow_webhooks::validation::validate_event_combination(entity, action)?;
    Ok(WebhookEvent::new(entity, action))
}

/// `WebhookStatus.from_proto(status) if status else None` (`handlers.py:3386`):
/// proto2 leaves an unset enum at 0 (falsy), so 0 -> `None` (store defaults to
/// ACTIVE). A set ACTIVE(1)/DISABLED(2) maps through; any other value errors.
fn status_from_proto(status: Option<i32>) -> Result<Option<WebhookStatus>, MlflowError> {
    match status.unwrap_or(0) {
        0 => Ok(None),
        1 => Ok(Some(WebhookStatus::Active)),
        2 => Ok(Some(WebhookStatus::Disabled)),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Unknown webhook status: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers (mirrors of the shared logged-model helpers)
// ---------------------------------------------------------------------------

fn require_non_empty<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(missing_param(param)),
    }
}

fn missing_param(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. \
         See the API docs for more information about request parameters."
    ))
}

fn path_param_pairs(
    path_params: &HashMap<String, String>,
    names: &[&'static str],
) -> Vec<(&'static str, String)> {
    names
        .iter()
        .filter_map(|name| path_params.get(*name).map(|v| (*name, v.clone())))
        .collect()
}
