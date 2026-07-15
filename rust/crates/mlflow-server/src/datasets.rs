//! `search-datasets` endpoint (plan T3.4, §3.4): `_search_datasets_handler` /
//! `search_datasets_impl` in `mlflow/server/handlers.py`.
//!
//! ## Route quirks reproduced verbatim
//!
//! The proto path (`service.proto:705`) is `"mlflow/experiments/search-datasets"`
//! — **no leading slash** — which Python's f-string concatenation
//! (`f"/api/2.0{path}"`) turns into the literal, slash-missing route
//! `/api/2.0mlflow/experiments/search-datasets` (and the `/ajax-api/` twin). The
//! `mlflow-proto` route table already reproduces this (T1.2); we register
//! whatever it yields, unmodified.
//!
//! Separately, `mlflow/server/__init__.py:135` hand-registers a second,
//! correctly-slashed ajax route for the same handler:
//! `/ajax-api/2.0/mlflow/experiments/search-datasets`. Both paths are wired to
//! [`search_datasets`] here.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;

use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `search_datasets_impl` (`handlers.py:2265`): at least one, at most 20
/// `experiment_ids`.
const MAX_EXPERIMENT_IDS_PER_REQUEST: usize = 20;

/// `_search_datasets_handler` / `search_datasets_impl` (`handlers.py:2253-2292`).
pub async fn search_datasets(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchDatasets = parse_request(&parts, &body, "mlflow.SearchDatasets")?;

    if req.experiment_ids.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "SearchDatasets request must specify at least one experiment_id.".to_string(),
        ));
    }
    if req.experiment_ids.len() > MAX_EXPERIMENT_IDS_PER_REQUEST {
        return Err(MlflowError::new(
            format!(
                "SearchDatasets request cannot specify more than {MAX_EXPERIMENT_IDS_PER_REQUEST} \
                 experiment_ids. Received {} experiment_ids.",
                req.experiment_ids.len()
            ),
            ErrorCode::InvalidParameterValue,
        ));
    }

    let experiment_ids: Vec<&str> = req.experiment_ids.iter().map(String::as_str).collect();
    let summaries = state
        .tracking_store()
        .search_datasets(workspace.name(), &experiment_ids)
        .await?;

    let resp = pb::search_datasets::Response {
        dataset_summaries: summaries.into_iter().map(to_proto_summary).collect(),
    };
    proto_response(&resp, "mlflow.SearchDatasets.Response")
}

fn to_proto_summary(s: mlflow_store::DatasetSummary) -> pb::DatasetSummary {
    pb::DatasetSummary {
        experiment_id: Some(s.experiment_id),
        name: Some(s.name),
        digest: Some(s.digest),
        context: s.context,
    }
}
