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
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_proto::mlflow::datasets as dataset_pb;
use mlflow_store::{EvaluationDataset, EvaluationRecord};
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::proto_http::{parse_request, parse_request_with_path_params, proto_response};
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

// Evaluation datasets (T16.1). Dataset metadata and records deliberately use
// JSON *strings* inside their proto fields; do not replace these with nested
// JSON values. AUTH GAP: datasets (D21) — Python registers no per-resource
// validator, so the shared auth middleware enforces authentication only.

pub async fn create_evaluation_dataset(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateDataset = parse_request(&parts, &body, "mlflow.CreateDataset")?;
    let name = required(req.name.as_deref(), "name")?;
    let tags = parse_json_object(
        req.tags
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("{}"),
        "tags",
    )?;
    let dataset = state
        .tracking_store()
        .create_evaluation_dataset(workspace.name(), name, &tags, &req.experiment_ids)
        .await?;
    proto_response(
        &pb::create_dataset::Response {
            dataset: Some(to_proto_evaluation_dataset(dataset)),
        },
        "mlflow.CreateDataset.Response",
    )
}

pub async fn get_evaluation_dataset(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let _: pb::GetDataset = parse_with_dataset_id(&parts, &body, "mlflow.GetDataset", dataset_id)?;
    let dataset = state
        .tracking_store()
        .get_evaluation_dataset(workspace.name(), dataset_id)
        .await?;
    proto_response(
        &pb::get_dataset::Response {
            dataset: Some(to_proto_evaluation_dataset(dataset)),
            next_page_token: None,
        },
        "mlflow.GetDataset.Response",
    )
}

pub async fn delete_evaluation_dataset(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let _: pb::DeleteDataset =
        parse_with_dataset_id(&parts, &body, "mlflow.DeleteDataset", dataset_id)?;
    state
        .tracking_store()
        .delete_evaluation_dataset(workspace.name(), dataset_id)
        .await?;
    proto_response(
        &pb::delete_dataset::Response {},
        "mlflow.DeleteDataset.Response",
    )
}

pub async fn search_evaluation_datasets(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchEvaluationDatasets =
        parse_request(&parts, &body, "mlflow.SearchEvaluationDatasets")?;
    let page = state
        .tracking_store()
        .search_evaluation_datasets(
            workspace.name(),
            &req.experiment_ids,
            req.filter_string
                .as_deref()
                .filter(|value| !value.is_empty()),
            req.max_results.unwrap_or(1000),
            &req.order_by,
            req.page_token.as_deref().filter(|value| !value.is_empty()),
        )
        .await?;
    proto_response(
        &pb::search_evaluation_datasets::Response {
            datasets: page
                .datasets
                .into_iter()
                .map(to_proto_evaluation_dataset)
                .collect(),
            next_page_token: page.next_page_token,
        },
        "mlflow.SearchEvaluationDatasets.Response",
    )
}

pub async fn set_evaluation_dataset_tags(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let req: pb::SetDatasetTags =
        parse_with_dataset_id(&parts, &body, "mlflow.SetDatasetTags", dataset_id)?;
    let tags = parse_json_object(required(req.tags.as_deref(), "tags")?, "tags")?;
    state
        .tracking_store()
        .set_evaluation_dataset_tags(workspace.name(), dataset_id, &tags)
        .await?;
    // Python's handler does not populate the proto's optional `dataset` field.
    proto_response(
        &pb::set_dataset_tags::Response { dataset: None },
        "mlflow.SetDatasetTags.Response",
    )
}

pub async fn delete_evaluation_dataset_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let key = path_value(&path, "key")?;
    let _: pb::DeleteDatasetTag = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.DeleteDatasetTag",
        &[
            ("dataset_id", dataset_id.to_string()),
            ("key", key.to_string()),
        ],
    )?;
    state
        .tracking_store()
        .delete_evaluation_dataset_tag(workspace.name(), dataset_id, key)
        .await?;
    proto_response(
        &pb::delete_dataset_tag::Response {},
        "mlflow.DeleteDatasetTag.Response",
    )
}

pub async fn upsert_evaluation_dataset_records(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let req: pb::UpsertDatasetRecords =
        parse_with_dataset_id(&parts, &body, "mlflow.UpsertDatasetRecords", dataset_id)?;
    let records_text = required(req.records.as_deref(), "records")?;
    let records: Vec<Value> = serde_json::from_str(records_text).map_err(|error| {
        MlflowError::invalid_parameter_value(format!("Invalid JSON in records: {error}"))
    })?;
    let result = state
        .tracking_store()
        .upsert_evaluation_records(workspace.name(), dataset_id, &records)
        .await?;
    proto_response(
        &pb::upsert_dataset_records::Response {
            inserted_count: Some(result.inserted),
            updated_count: Some(result.updated),
        },
        "mlflow.UpsertDatasetRecords.Response",
    )
}

pub async fn get_evaluation_dataset_experiment_ids(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let _: pb::GetDatasetExperimentIds =
        parse_with_dataset_id(&parts, &body, "mlflow.GetDatasetExperimentIds", dataset_id)?;
    let experiment_ids = state
        .tracking_store()
        .get_evaluation_dataset_experiment_ids(workspace.name(), dataset_id)
        .await?;
    proto_response(
        &pb::get_dataset_experiment_ids::Response { experiment_ids },
        "mlflow.GetDatasetExperimentIds.Response",
    )
}

pub async fn get_evaluation_dataset_records(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let req: pb::GetDatasetRecords =
        parse_with_dataset_id(&parts, &body, "mlflow.GetDatasetRecords", dataset_id)?;
    let page = state
        .tracking_store()
        .load_evaluation_records(
            workspace.name(),
            dataset_id,
            req.max_results.filter(|value| *value != 0).unwrap_or(1000),
            req.page_token.as_deref().filter(|value| !value.is_empty()),
        )
        .await?;
    let records = Value::Array(page.records.into_iter().map(record_to_dict).collect());
    proto_response(
        &pb::get_dataset_records::Response {
            records: Some(mlflow_store::python_json_dumps(&records, false)),
            next_page_token: page.next_page_token,
        },
        "mlflow.GetDatasetRecords.Response",
    )
}

pub async fn delete_evaluation_dataset_records(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let req: pb::DeleteDatasetRecords =
        parse_with_dataset_id(&parts, &body, "mlflow.DeleteDatasetRecords", dataset_id)?;
    let deleted = state
        .tracking_store()
        .delete_evaluation_records(workspace.name(), dataset_id, &req.dataset_record_ids)
        .await?;
    proto_response(
        &pb::delete_dataset_records::Response {
            deleted_count: Some(deleted),
        },
        "mlflow.DeleteDatasetRecords.Response",
    )
}

pub async fn add_evaluation_dataset_to_experiments(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let req: pb::AddDatasetToExperiments =
        parse_with_dataset_id(&parts, &body, "mlflow.AddDatasetToExperiments", dataset_id)?;
    let dataset = state
        .tracking_store()
        .add_evaluation_dataset_to_experiments(workspace.name(), dataset_id, &req.experiment_ids)
        .await?;
    proto_response(
        &pb::add_dataset_to_experiments::Response {
            dataset: Some(to_proto_evaluation_dataset(dataset)),
        },
        "mlflow.AddDatasetToExperiments.Response",
    )
}

pub async fn remove_evaluation_dataset_from_experiments(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let dataset_id = path_value(&path, "dataset_id")?;
    let req: pb::RemoveDatasetFromExperiments = parse_with_dataset_id(
        &parts,
        &body,
        "mlflow.RemoveDatasetFromExperiments",
        dataset_id,
    )?;
    let dataset = state
        .tracking_store()
        .remove_evaluation_dataset_from_experiments(
            workspace.name(),
            dataset_id,
            &req.experiment_ids,
        )
        .await?;
    proto_response(
        &pb::remove_dataset_from_experiments::Response {
            dataset: Some(to_proto_evaluation_dataset(dataset)),
        },
        "mlflow.RemoveDatasetFromExperiments.Response",
    )
}

fn parse_with_dataset_id<M: prost::Message + Default>(
    parts: &Parts,
    body: &Bytes,
    type_name: &str,
    dataset_id: &str,
) -> Result<M, MlflowError> {
    parse_request_with_path_params(
        parts,
        body,
        type_name,
        &[("dataset_id", dataset_id.to_string())],
    )
}

fn path_value<'a>(path: &'a HashMap<String, String>, key: &str) -> Result<&'a str, MlflowError> {
    path.get(key).map(String::as_str).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!("Missing required path parameter '{key}'"))
    })
}

fn required<'a>(value: Option<&'a str>, name: &str) -> Result<&'a str, MlflowError> {
    value.filter(|value| !value.is_empty()).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!("Missing required parameter '{name}'"))
    })
}

fn parse_json_object(text: &str, field: &str) -> Result<Map<String, Value>, MlflowError> {
    serde_json::from_str::<Value>(text)
        .map_err(|error| {
            MlflowError::invalid_parameter_value(format!("Invalid JSON in {field}: {error}"))
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("{field} must be a JSON object"))
        })
}

fn to_proto_evaluation_dataset(dataset: EvaluationDataset) -> dataset_pb::Dataset {
    dataset_pb::Dataset {
        dataset_id: Some(dataset.dataset_id),
        name: Some(dataset.name),
        tags: Some(mlflow_store::python_json_dumps(
            &Value::Object(dataset.tags),
            false,
        )),
        schema: dataset.schema,
        profile: dataset.profile,
        digest: dataset.digest,
        created_time: dataset.created_time,
        last_update_time: dataset.last_update_time,
        created_by: dataset.created_by,
        last_updated_by: dataset.last_updated_by,
        experiment_ids: dataset.experiment_ids.unwrap_or_default(),
    }
}

fn record_to_dict(record: EvaluationRecord) -> Value {
    let mut value = Map::new();
    value.insert(
        "dataset_record_id".to_string(),
        Value::String(record.dataset_record_id),
    );
    value.insert("dataset_id".to_string(), Value::String(record.dataset_id));
    value.insert("inputs".to_string(), record.inputs);
    if let Some(expectations) = record.expectations {
        value.insert("expectations".to_string(), expectations);
    }
    value.insert(
        "tags".to_string(),
        record.tags.unwrap_or_else(|| Value::Object(Map::new())),
    );
    if let Some(source) = record.source {
        value.insert("source".to_string(), source);
    }
    if let Some(source_id) = record.source_id {
        value.insert("source_id".to_string(), Value::String(source_id));
    }
    if let Some(source_type) = record.source_type {
        value.insert("source_type".to_string(), Value::String(source_type));
    }
    value.insert(
        "created_time".to_string(),
        Value::Number(record.created_time.unwrap_or_default().into()),
    );
    value.insert(
        "last_update_time".to_string(),
        Value::Number(record.last_update_time.unwrap_or_default().into()),
    );
    if let Some(created_by) = record.created_by {
        value.insert("created_by".to_string(), Value::String(created_by));
    }
    if let Some(last_updated_by) = record.last_updated_by {
        value.insert(
            "last_updated_by".to_string(),
            Value::String(last_updated_by),
        );
    }
    if let Some(outputs) = record.outputs {
        value.insert("outputs".to_string(), outputs);
    }
    Value::Object(value)
}

#[cfg(test)]
mod evaluation_dataset_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_json_matches_python_protobuf_field_order() {
        let record = EvaluationRecord {
            dataset_record_id: "record".to_string(),
            dataset_id: "dataset".to_string(),
            inputs: json!({"x": 1}),
            outputs: Some(json!({"y": 2})),
            expectations: Some(json!({"expected": {"value": 3}})),
            tags: Some(json!({"tag": "value"})),
            source: None,
            source_id: Some("trace".to_string()),
            source_type: Some("TRACE".to_string()),
            created_time: Some(1),
            last_update_time: Some(2),
            created_by: Some("creator".to_string()),
            last_updated_by: Some("updater".to_string()),
        };

        assert_eq!(
            mlflow_store::python_json_dumps(&json!([record_to_dict(record)]), false),
            r#"[{"dataset_record_id": "record", "dataset_id": "dataset", "inputs": {"x": 1}, "expectations": {"expected": {"value": 3}}, "tags": {"tag": "value"}, "source_id": "trace", "source_type": "TRACE", "created_time": 1, "last_update_time": 2, "created_by": "creator", "last_updated_by": "updater", "outputs": {"y": 2}}]"#
        );
    }
}
