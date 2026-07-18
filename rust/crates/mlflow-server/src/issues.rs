//! Issue RPC handlers (T16.3).

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow::issues as pb;
use mlflow_store::{Issue, IssueUpdate};

use crate::proto_http::{parse_request, parse_request_with_path_params, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

// AUTH GAP: issues (D21) — Python registers no per-resource validators for
// these four RPCs. The shared middleware still requires authentication.

pub async fn create_issue(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateIssue = parse_request(&parts, &body, "mlflow.issues.CreateIssue")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let description = required(req.description.as_deref(), "description")?;
    let status = req.status.as_deref().unwrap_or("pending");
    validate_status(status)?;
    if let Some(severity) = req.severity.as_deref() {
        validate_severity(severity)?;
    }
    let issue = state
        .tracking_store()
        .create_issue(
            workspace.name(),
            experiment_id,
            name,
            description,
            status,
            req.severity.as_deref(),
            &req.root_causes,
            req.source_run_id
                .as_deref()
                .filter(|value| !value.is_empty()),
            &req.categories,
            req.created_by.as_deref().filter(|value| !value.is_empty()),
        )
        .await?;
    proto_response(
        &pb::create_issue::Response {
            issue: Some(to_proto(issue)),
        },
        "mlflow.issues.CreateIssue.Response",
    )
}

pub async fn update_issue(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(issue_id): Path<String>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateIssue = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.issues.UpdateIssue",
        &[("issue_id", issue_id.clone())],
    )?;
    if let Some(status) = req.status.as_deref() {
        validate_status(status)?;
    }
    if let Some(severity) = req.severity.as_deref() {
        validate_severity(severity)?;
    }
    let issue = state
        .tracking_store()
        .update_issue(
            workspace.name(),
            &issue_id,
            IssueUpdate {
                name: req.name.as_deref().filter(|value| !value.is_empty()),
                description: req.description.as_deref().filter(|value| !value.is_empty()),
                status: req.status.as_deref(),
                severity: req.severity.as_deref(),
            },
        )
        .await?;
    proto_response(
        &pb::update_issue::Response {
            issue: Some(to_proto(issue)),
        },
        "mlflow.issues.UpdateIssue.Response",
    )
}

pub async fn get_issue(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(issue_id): Path<String>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let _: pb::GetIssue = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.issues.GetIssue",
        &[("issue_id", issue_id.clone())],
    )?;
    let issue = state
        .tracking_store()
        .get_issue(workspace.name(), &issue_id)
        .await?;
    proto_response(
        &pb::get_issue::Response {
            issue: Some(to_proto(issue)),
        },
        "mlflow.issues.GetIssue.Response",
    )
}

pub async fn search_issues(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchIssues = parse_request(&parts, &body, "mlflow.issues.SearchIssues")?;
    let page = state
        .tracking_store()
        .search_issues(
            workspace.name(),
            req.experiment_id
                .as_deref()
                .filter(|value| !value.is_empty()),
            req.filter_string
                .as_deref()
                .filter(|value| !value.is_empty()),
            req.max_results,
            req.page_token.as_deref().filter(|value| !value.is_empty()),
            req.include_trace_count.unwrap_or(false),
        )
        .await?;
    proto_response(
        &pb::search_issues::Response {
            issues: page.issues.into_iter().map(to_proto).collect(),
            next_page_token: Some(page.next_page_token.unwrap_or_default()),
        },
        "mlflow.issues.SearchIssues.Response",
    )
}

fn to_proto(issue: Issue) -> pb::Issue {
    pb::Issue {
        issue_id: Some(issue.issue_id),
        experiment_id: Some(issue.experiment_id),
        name: Some(issue.name),
        description: Some(issue.description),
        status: Some(issue.status),
        severity: issue.severity,
        root_causes: issue.root_causes,
        source_run_id: issue.source_run_id,
        created_timestamp: Some(issue.created_timestamp),
        last_updated_timestamp: Some(issue.last_updated_timestamp),
        created_by: issue.created_by,
        categories: issue.categories,
        trace_count: issue.trace_count,
    }
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

fn validate_status(status: &str) -> Result<(), MlflowError> {
    if ["pending", "rejected", "resolved"].contains(&status) {
        Ok(())
    } else {
        Err(MlflowError::invalid_parameter_value(format!(
            "'{status}' is not a valid IssueStatus"
        )))
    }
}

fn validate_severity(severity: &str) -> Result<(), MlflowError> {
    if ["not_an_issue", "low", "medium", "high"].contains(&severity) {
        Ok(())
    } else {
        Err(MlflowError::invalid_parameter_value(format!(
            "'{severity}' is not a valid IssueSeverity"
        )))
    }
}
