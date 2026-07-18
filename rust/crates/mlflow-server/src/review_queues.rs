//! Review-queue RPC handlers (T16.4).

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow::review_queues as pb;
use mlflow_store::{
    ReviewItemType, ReviewQueue, ReviewQueueItem, ReviewQueueType, ReviewQueueUpdate, ReviewStatus,
};

use crate::auth_middleware::AuthContext;
use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

pub async fn create_review_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let username = authenticated_username(&parts);
    let req: pb::CreateReviewQueue =
        parse_request(&parts, &body, "mlflow.review_queues.CreateReviewQueue")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let queue_type = queue_type(req.queue_type)?;
    let queue = state
        .tracking_store()
        .create_review_queue(
            workspace.name(),
            experiment_id,
            name,
            queue_type,
            username.as_deref(),
            &req.users,
            &req.schema_ids,
        )
        .await?;
    proto_response(
        &pb::create_review_queue::Response {
            review_queue: Some(to_proto_queue(queue)),
        },
        "mlflow.review_queues.CreateReviewQueue.Response",
    )
}

pub async fn get_or_create_user_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetOrCreateUserQueue =
        parse_request(&parts, &body, "mlflow.review_queues.GetOrCreateUserQueue")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let user = required(req.user.as_deref(), "user")?;
    let queue = state
        .tracking_store()
        .get_or_create_user_queue(workspace.name(), experiment_id, user)
        .await?;
    proto_response(
        &pb::get_or_create_user_queue::Response {
            review_queue: Some(to_proto_queue(queue)),
        },
        "mlflow.review_queues.GetOrCreateUserQueue.Response",
    )
}

pub async fn get_review_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetReviewQueue =
        parse_request(&parts, &body, "mlflow.review_queues.GetReviewQueue")?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    let queue = state
        .tracking_store()
        .get_review_queue(workspace.name(), queue_id)
        .await?;
    proto_response(
        &pb::get_review_queue::Response {
            review_queue: Some(to_proto_queue(queue)),
        },
        "mlflow.review_queues.GetReviewQueue.Response",
    )
}

pub async fn get_review_queue_by_name(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetReviewQueueByName =
        parse_request(&parts, &body, "mlflow.review_queues.GetReviewQueueByName")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let queue = state
        .tracking_store()
        .get_review_queue_by_name(workspace.name(), experiment_id, name)
        .await?;
    proto_response(
        &pb::get_review_queue_by_name::Response {
            review_queue: Some(to_proto_queue(queue)),
        },
        "mlflow.review_queues.GetReviewQueueByName.Response",
    )
}

pub async fn list_review_queues(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::ListReviewQueues =
        parse_request(&parts, &body, "mlflow.review_queues.ListReviewQueues")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let page = state
        .tracking_store()
        .list_review_queues(
            workspace.name(),
            experiment_id,
            req.user.as_deref(),
            req.item_id.as_deref(),
            req.max_results,
            req.page_token.as_deref(),
        )
        .await?;
    proto_response(
        &pb::list_review_queues::Response {
            review_queues: page.queues.into_iter().map(to_proto_queue).collect(),
            next_page_token: Some(page.next_page_token.unwrap_or_default()),
        },
        "mlflow.review_queues.ListReviewQueues.Response",
    )
}

pub async fn update_review_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateReviewQueue =
        parse_request(&parts, &body, "mlflow.review_queues.UpdateReviewQueue")?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    let queue = state
        .tracking_store()
        .update_review_queue(
            workspace.name(),
            queue_id,
            ReviewQueueUpdate {
                users: req
                    .update_users
                    .unwrap_or(false)
                    .then_some(req.users.as_slice()),
                schema_ids: req
                    .update_schema_ids
                    .unwrap_or(false)
                    .then_some(req.schema_ids.as_slice()),
                name: req.name.as_deref(),
                new_owner: req.new_owner.as_deref(),
            },
        )
        .await?;
    proto_response(
        &pb::update_review_queue::Response {
            review_queue: Some(to_proto_queue(queue)),
        },
        "mlflow.review_queues.UpdateReviewQueue.Response",
    )
}

pub async fn delete_review_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteReviewQueue =
        parse_request(&parts, &body, "mlflow.review_queues.DeleteReviewQueue")?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    state
        .tracking_store()
        .delete_review_queue(workspace.name(), queue_id)
        .await?;
    proto_response(
        &pb::delete_review_queue::Response {},
        "mlflow.review_queues.DeleteReviewQueue.Response",
    )
}

pub async fn add_items_to_review_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::AddItemsToReviewQueue =
        parse_request(&parts, &body, "mlflow.review_queues.AddItemsToReviewQueue")?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    let item_ids = normalize_item_ids(&req.item_ids)?;
    let item_type = item_type(req.item_type)?;
    let queue = state
        .tracking_store()
        .get_review_queue(workspace.name(), queue_id)
        .await?;
    let trace_infos = state
        .tracking_store()
        .batch_get_trace_infos(workspace.name(), &item_ids)
        .await?;
    let experiment_by_trace = trace_infos
        .into_iter()
        .map(|info| (info.trace_id, info.experiment_id))
        .collect::<HashMap<_, _>>();
    let invalid = item_ids
        .iter()
        .filter(|item_id| experiment_by_trace.get(*item_id) != Some(&queue.experiment_id))
        .cloned()
        .collect::<Vec<_>>();
    if !invalid.is_empty() {
        return Err(MlflowError::resource_does_not_exist(format!(
            "Cannot attach trace(s) {} to review queue '{queue_id}': they do not exist in \
             experiment '{}'.",
            python_string_list(&invalid),
            queue.experiment_id
        )));
    }
    let items = state
        .tracking_store()
        .add_items_to_review_queue(workspace.name(), queue_id, &item_ids, item_type)
        .await?;
    proto_response(
        &pb::add_items_to_review_queue::Response {
            items: items.into_iter().map(to_proto_item).collect(),
        },
        "mlflow.review_queues.AddItemsToReviewQueue.Response",
    )
}

pub async fn remove_items_from_review_queue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::RemoveItemsFromReviewQueue = parse_request(
        &parts,
        &body,
        "mlflow.review_queues.RemoveItemsFromReviewQueue",
    )?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    state
        .tracking_store()
        .remove_items_from_review_queue(workspace.name(), queue_id, &req.item_ids)
        .await?;
    proto_response(
        &pb::remove_items_from_review_queue::Response {},
        "mlflow.review_queues.RemoveItemsFromReviewQueue.Response",
    )
}

pub async fn list_review_queue_items(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::ListReviewQueueItems =
        parse_request(&parts, &body, "mlflow.review_queues.ListReviewQueueItems")?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    let status = optional_status(req.status)?;
    let page = state
        .tracking_store()
        .list_review_queue_items(
            workspace.name(),
            queue_id,
            status,
            req.max_results,
            req.page_token.as_deref(),
        )
        .await?;
    proto_response(
        &pb::list_review_queue_items::Response {
            items: page.items.into_iter().map(to_proto_item).collect(),
            next_page_token: Some(page.next_page_token.unwrap_or_default()),
        },
        "mlflow.review_queues.ListReviewQueueItems.Response",
    )
}

pub async fn set_review_queue_item_status(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let username = authenticated_username(&parts);
    let req: pb::SetReviewQueueItemStatus = parse_request(
        &parts,
        &body,
        "mlflow.review_queues.SetReviewQueueItemStatus",
    )?;
    let queue_id = required(req.queue_id.as_deref(), "queue_id")?;
    let item_id = required(req.item_id.as_deref(), "item_id")?;
    let status = required_status(req.status)?;
    let completed_by = match (username.as_deref(), status) {
        (Some(_), ReviewStatus::Pending) => None,
        (Some(username), _) => Some(username),
        (None, _) => req.completed_by.as_deref(),
    };
    let item = state
        .tracking_store()
        .set_review_queue_item_status(workspace.name(), queue_id, item_id, status, completed_by)
        .await?;
    proto_response(
        &pb::set_review_queue_item_status::Response {
            item: Some(to_proto_item(item)),
        },
        "mlflow.review_queues.SetReviewQueueItemStatus.Response",
    )
}

fn authenticated_username(parts: &Parts) -> Option<String> {
    parts
        .extensions
        .get::<AuthContext>()
        .map(|auth| auth.username.clone())
}

fn queue_type(value: Option<i32>) -> Result<ReviewQueueType, MlflowError> {
    match value.and_then(|value| pb::ReviewQueueType::try_from(value).ok()) {
        Some(pb::ReviewQueueType::User) => Ok(ReviewQueueType::User),
        Some(pb::ReviewQueueType::Custom) => Ok(ReviewQueueType::Custom),
        _ => Err(MlflowError::invalid_parameter_value(format!(
            "`queue_type` must be one of USER or CUSTOM; got proto enum value {}.",
            value.unwrap_or_default()
        ))),
    }
}

fn item_type(value: Option<i32>) -> Result<ReviewItemType, MlflowError> {
    match value {
        None | Some(0) | Some(1) => Ok(ReviewItemType::Trace),
        Some(value) => Err(MlflowError::invalid_parameter_value(format!(
            "`item_type` must be TRACE; got proto enum value {value}."
        ))),
    }
}

fn optional_status(value: Option<i32>) -> Result<Option<ReviewStatus>, MlflowError> {
    match value {
        None | Some(0) => Ok(None),
        Some(1) => Ok(Some(ReviewStatus::Pending)),
        Some(2) => Ok(Some(ReviewStatus::Complete)),
        Some(3) => Ok(Some(ReviewStatus::Declined)),
        Some(value) => Err(MlflowError::invalid_parameter_value(format!(
            "`status` must be one of PENDING, COMPLETE, or DECLINED; got proto enum value {value}."
        ))),
    }
}

fn required_status(value: Option<i32>) -> Result<ReviewStatus, MlflowError> {
    match optional_status(value)? {
        Some(status) => Ok(status),
        None => Err(MlflowError::invalid_parameter_value(format!(
            "`status` must be one of PENDING, COMPLETE, or DECLINED; got proto enum value {}.",
            value.unwrap_or_default()
        ))),
    }
}

fn to_proto_queue(queue: ReviewQueue) -> pb::ReviewQueue {
    let queue_type = match queue.queue_type {
        ReviewQueueType::User => pb::ReviewQueueType::User,
        ReviewQueueType::Custom => pb::ReviewQueueType::Custom,
    };
    pb::ReviewQueue {
        queue_id: Some(queue.queue_id),
        experiment_id: Some(queue.experiment_id),
        name: Some(queue.name),
        queue_type: Some(queue_type as i32),
        created_by: queue.created_by,
        creation_time_ms: Some(queue.creation_time_ms),
        last_update_time_ms: Some(queue.last_update_time_ms),
        users: queue.users,
        schema_ids: queue.schema_ids,
    }
}

fn to_proto_item(item: ReviewQueueItem) -> pb::ReviewQueueItem {
    let status = match item.status {
        ReviewStatus::Pending => pb::ReviewStatus::Pending,
        ReviewStatus::Complete => pb::ReviewStatus::Complete,
        ReviewStatus::Declined => pb::ReviewStatus::Declined,
    };
    pb::ReviewQueueItem {
        queue_id: Some(item.queue_id),
        item_type: Some(pb::ReviewItemType::Trace as i32),
        item_id: Some(item.item_id),
        status: Some(status as i32),
        completed_by: item.completed_by,
        completed_time_ms: item.completed_time_ms,
        creation_time_ms: Some(item.creation_time_ms),
        last_update_time_ms: Some(item.last_update_time_ms),
    }
}

fn normalize_item_ids(item_ids: &[String]) -> Result<Vec<String>, MlflowError> {
    if item_ids.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "`item_ids` must be a non-empty list; got [].",
        ));
    }
    let mut output = Vec::new();
    for raw in item_ids {
        let item_id = raw.trim().to_string();
        if item_id.is_empty() {
            return Err(MlflowError::invalid_parameter_value(
                "`item_id` must be a non-empty string; got ''.",
            ));
        }
        let length = item_id.chars().count();
        if length > 50 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "`item_id` must be at most 50 characters; got {length}."
            )));
        }
        if !output.contains(&item_id) {
            output.push(item_id);
        }
    }
    Ok(output)
}

fn required<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    value
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing_required(param))
}

fn missing_required(param: &str) -> MlflowError {
    MlflowError::new(
        format!(
            "Missing value for required parameter '{param}'. See the API docs for more \
             information about request parameters."
        ),
        ErrorCode::InvalidParameterValue,
    )
}

fn python_string_list(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
