//! Ajax demo-data route state machine adjacent to the Phase 20 GenAI surface.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use mlflow_error::MlflowError;
use mlflow_store::store::LifecycleStage;
use mlflow_store::WorkspaceArtifactRoot;
use serde_json::{json, Value};

use crate::state::AppState;
use crate::workspace::Workspace;

const DEMO_EXPERIMENT_NAME: &str = "MLflow Demo";
const FEATURES: [(&str, i32); 6] = [
    ("prompts", 1),
    ("traces", 3),
    ("evaluation", 2),
    ("judges", 1),
    ("issues", 4),
    ("review_queues", 1),
];

pub async fn generate(
    State(state): State<AppState>,
    workspace: Workspace,
    body: Bytes,
) -> Response {
    match generate_impl(&state, workspace.name(), &body).await {
        Ok(value) => flask_json(value),
        Err(error) => error.into_response(),
    }
}

pub async fn delete(State(state): State<AppState>, workspace: Workspace) -> Response {
    match delete_impl(&state, workspace.name()).await {
        Ok(value) => flask_json(value),
        Err(error) => error.into_response(),
    }
}

async fn generate_impl(
    state: &AppState,
    workspace: &str,
    body: &[u8],
) -> Result<Value, MlflowError> {
    // `request.get_json(silent=True) or {}`: malformed/non-object/falsey JSON
    // behaves like an empty request and therefore selects every generator.
    let request: Value = serde_json::from_slice(body).unwrap_or_else(|_| json!({}));
    let selected = selected_features(request.get("features"));
    let mut experiment = state
        .tracking_store()
        .get_experiment_by_name(workspace, DEMO_EXPERIMENT_NAME)
        .await?;

    let to_generate = if let Some(active) = experiment
        .as_ref()
        .filter(|experiment| experiment.lifecycle_stage == LifecycleStage::ACTIVE)
    {
        selected
            .iter()
            .filter(|(name, version)| {
                !active.tags.iter().any(|tag| {
                    tag.key == format!("mlflow.demo.version.{name}")
                        && tag.value.as_deref() == Some(&version.to_string())
                })
            })
            .copied()
            .collect::<Vec<_>>()
    } else {
        selected.clone()
    };

    if to_generate.is_empty() {
        if let Some(active) = experiment
            .as_ref()
            .filter(|experiment| experiment.lifecycle_stage == LifecycleStage::ACTIVE)
        {
            return Ok(json!({
                "experiment_id": active.experiment_id,
                "features_generated": [],
                "navigation_url": format!("/experiments/{}", active.experiment_id),
                "status": "exists",
            }));
        }
    }

    if !to_generate.is_empty() {
        let experiment_id = match experiment.as_ref() {
            Some(existing) if existing.lifecycle_stage == LifecycleStage::DELETED => {
                state
                    .tracking_store()
                    .restore_experiment(workspace, &existing.experiment_id)
                    .await?;
                existing.experiment_id.clone()
            }
            Some(existing) => existing.experiment_id.clone(),
            None => create_demo_experiment(state, workspace).await?,
        };
        for (name, version) in &to_generate {
            state
                .tracking_store()
                .set_experiment_tag(
                    workspace,
                    &experiment_id,
                    &format!("mlflow.demo.version.{name}"),
                    &version.to_string(),
                )
                .await?;
        }
        experiment = state
            .tracking_store()
            .get_experiment_by_name(workspace, DEMO_EXPERIMENT_NAME)
            .await?;
    }

    let experiment_id = experiment
        .as_ref()
        .map(|experiment| experiment.experiment_id.clone());
    let navigation_url = experiment_id
        .as_ref()
        .map(|id| format!("/experiments/{id}"))
        .unwrap_or_else(|| "/experiments".to_string());
    Ok(json!({
        "experiment_id": experiment_id,
        "features_generated": to_generate.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        "navigation_url": navigation_url,
        "status": "created",
    }))
}

async fn delete_impl(state: &AppState, workspace: &str) -> Result<Value, MlflowError> {
    let experiment = state
        .tracking_store()
        .get_experiment_by_name(workspace, DEMO_EXPERIMENT_NAME)
        .await?;
    let mut deleted = Vec::new();
    if let Some(experiment) = experiment
        .as_ref()
        .filter(|experiment| experiment.lifecycle_stage == LifecycleStage::ACTIVE)
    {
        for (name, _) in FEATURES {
            if experiment
                .tags
                .iter()
                .any(|tag| tag.key == format!("mlflow.demo.version.{name}"))
            {
                deleted.push(name);
            }
        }
        state
            .tracking_store()
            .delete_experiment(workspace, &experiment.experiment_id)
            .await?;
    }
    Ok(json!({"features_deleted": deleted, "status": "deleted"}))
}

fn selected_features(value: Option<&Value>) -> Vec<(&'static str, i32)> {
    match value {
        None | Some(Value::Null) => FEATURES.to_vec(),
        Some(Value::Array(values)) => FEATURES
            .into_iter()
            .filter(|(name, _)| values.iter().any(|value| value.as_str() == Some(name)))
            .collect(),
        Some(Value::String(value)) => FEATURES
            .into_iter()
            .filter(|(name, _)| value.contains(name))
            .collect(),
        Some(Value::Object(value)) => FEATURES
            .into_iter()
            .filter(|(name, _)| value.contains_key(*name))
            .collect(),
        _ => Vec::new(),
    }
}

async fn create_demo_experiment(state: &AppState, workspace: &str) -> Result<String, MlflowError> {
    match state.workspace_store() {
        Some(workspace_store) => {
            let (root, should_append) = workspace_store
                .resolve_artifact_root(Some(state.tracking_store().artifact_root_uri()), workspace)
                .await?;
            state
                .tracking_store()
                .create_experiment_workspace_scoped(
                    workspace,
                    DEMO_EXPERIMENT_NAME,
                    &[],
                    &WorkspaceArtifactRoot::Scoped {
                        root: root.unwrap_or_default(),
                        workspace: workspace.to_string(),
                        should_append,
                    },
                )
                .await
        }
        None => {
            state
                .tracking_store()
                .create_experiment(workspace, DEMO_EXPERIMENT_NAME, None, &[])
                .await
        }
    }
}

fn flask_json(value: Value) -> Response {
    let mut body = serde_json::to_string(&value).expect("demo response serializes");
    body.push('\n');
    ([(header::CONTENT_TYPE, "application/json")], body).into_response()
}
