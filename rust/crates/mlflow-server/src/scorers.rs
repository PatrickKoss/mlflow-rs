//! Registered scorer CRUD and online-scoring configuration routes.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use mlflow_error::MlflowError;
use mlflow_genai::{ScorerPayloadError, SerializedScorer};
use mlflow_proto::mlflow as pb;
use mlflow_store::{python_json_dumps, OnlineScoringConfig, ScorerVersion};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::proto_http::{parse_query_pairs, parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

pub const DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR: &str =
    "Custom scorer registration (using @scorer decorator) is not supported outside of Databricks \
     tracking environments due to security concerns. Custom scorers require arbitrary code \
     execution during deserialization.\n\nTo use custom scorers:\n1. Configure MLflow to use a \
     Databricks tracking URI, or\n2. Manage your custom scorer code in a source code repository \
     (e.g., GitHub) and import it directly, or\n3. Use built-in scorers or make_judge() scorers \
     instead.";

pub async fn register_scorer(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::RegisterScorer = parse_request(&parts, &body, "mlflow.RegisterScorer")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let serialized_scorer = required(req.serialized_scorer.as_deref(), "serialized_scorer")?;
    let serialized_data: Value = serde_json::from_str(serialized_scorer).map_err(|_| {
        MlflowError::invalid_parameter_value("serialized_scorer must be valid JSON")
    })?;
    if serialized_data
        .get("call_source")
        .is_some_and(|value| !value.is_null())
    {
        return Err(MlflowError::invalid_parameter_value(
            DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR,
        ));
    }
    if let Err(ScorerPayloadError::PhoenixLicense { metric }) =
        SerializedScorer::from_json(serialized_scorer)
    {
        return Err(MlflowError::invalid_parameter_value(
            ScorerPayloadError::PhoenixLicense { metric }.to_string(),
        ));
    }
    let scorer = state
        .tracking_store()
        .register_scorer(workspace.name(), experiment_id, name, serialized_scorer)
        .await?;
    proto_response(
        &pb::register_scorer::Response {
            version: Some(scorer.scorer_version),
            scorer_id: Some(scorer.scorer_id),
            experiment_id: Some(scorer.experiment_id),
            name: Some(scorer.scorer_name),
            serialized_scorer: Some(scorer.serialized_scorer),
            creation_time: scorer.creation_time,
        },
        "mlflow.RegisterScorer.Response",
    )
}

pub async fn list_scorers(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::ListScorers = parse_request(&parts, &body, "mlflow.ListScorers")?;
    let scorers = state
        .tracking_store()
        .list_scorers(workspace.name(), req.experiment_id.as_deref())
        .await?;
    proto_response(
        &pb::list_scorers::Response {
            scorers: scorers.into_iter().map(to_proto_scorer).collect(),
        },
        "mlflow.ListScorers.Response",
    )
}

pub async fn list_scorer_versions(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::ListScorerVersions = parse_request(&parts, &body, "mlflow.ListScorerVersions")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let scorers = state
        .tracking_store()
        .list_scorer_versions(workspace.name(), experiment_id, name)
        .await?;
    proto_response(
        &pb::list_scorer_versions::Response {
            scorers: scorers.into_iter().map(to_proto_scorer).collect(),
        },
        "mlflow.ListScorerVersions.Response",
    )
}

pub async fn get_scorer(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetScorer = parse_request(&parts, &body, "mlflow.GetScorer")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let scorer = state
        .tracking_store()
        .get_scorer(workspace.name(), experiment_id, name, req.version)
        .await?;
    proto_response(
        &pb::get_scorer::Response {
            scorer: Some(to_proto_scorer(scorer)),
        },
        "mlflow.GetScorer.Response",
    )
}

pub async fn delete_scorer(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteScorer = parse_request(&parts, &body, "mlflow.DeleteScorer")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    state
        .tracking_store()
        .delete_scorer(workspace.name(), experiment_id, name, req.version)
        .await?;
    proto_response(
        &pb::delete_scorer::Response {},
        "mlflow.DeleteScorer.Response",
    )
}

pub async fn get_online_scoring_configs(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let scorer_ids = parts
        .uri
        .query()
        .map(parse_query_pairs)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(key, value)| (key == "scorer_ids").then_some(value))
        .collect::<Vec<_>>();
    if scorer_ids.is_empty() {
        return Err(missing("scorer_ids"));
    }
    let configs = state
        .tracking_store()
        .get_online_scoring_configs(workspace.name(), &scorer_ids)
        .await?;
    Ok(json_response(json_object_response(
        "configs",
        Value::Array(configs.iter().map(config_json).collect()),
    )))
}

#[derive(Deserialize)]
struct UpsertOnlineConfigRequest {
    experiment_id: Option<Value>,
    name: Option<Value>,
    sample_rate: Option<Value>,
    #[serde(default)]
    filter_string: Option<Value>,
}

pub async fn upsert_online_scoring_config(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    validate_json_content_type(&parts)?;
    let request: UpsertOnlineConfigRequest = serde_json::from_slice(&body).map_err(|error| {
        MlflowError::invalid_parameter_value(format!("failed to parse JSON: {error}"))
    })?;
    let experiment_id = required_json_string(request.experiment_id.as_ref(), "experiment_id")?;
    let name = required_json_string(request.name.as_ref(), "name")?;
    let sample_value = request
        .sample_rate
        .as_ref()
        .ok_or_else(|| missing("sample_rate"))?;
    let sample_rate = json_float(sample_value).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "sample_rate must be a number, got {}",
            python_type_name(sample_value)
        ))
    })?;
    let filter_string = match request.filter_string.as_ref() {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        Some(value) => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {} for parameter 'filter_string' supplied: Value was of type '{}'. Expected type 'str' or None.",
                python_repr(value),
                python_type_name(value)
            )))
        }
    };
    let config = state
        .tracking_store()
        .upsert_online_scoring_config(
            workspace.name(),
            experiment_id,
            name,
            sample_rate,
            filter_string,
        )
        .await?;
    Ok(json_response(json_object_response(
        "config",
        config_json(&config),
    )))
}

fn to_proto_scorer(scorer: ScorerVersion) -> pb::Scorer {
    pb::Scorer {
        experiment_id: scorer.experiment_id.parse().ok(),
        scorer_name: Some(scorer.scorer_name),
        scorer_version: Some(scorer.scorer_version),
        serialized_scorer: Some(scorer.serialized_scorer),
        creation_time: scorer.creation_time,
        scorer_id: Some(scorer.scorer_id),
    }
}

fn config_json(config: &OnlineScoringConfig) -> Value {
    let mut value = Map::new();
    value.insert(
        "online_scoring_config_id".to_string(),
        Value::String(config.online_scoring_config_id.clone()),
    );
    value.insert(
        "scorer_id".to_string(),
        Value::String(config.scorer_id.clone()),
    );
    value.insert("sample_rate".to_string(), Value::from(config.sample_rate));
    value.insert(
        "experiment_id".to_string(),
        Value::String(config.experiment_id.clone()),
    );
    if let Some(filter_string) = &config.filter_string {
        value.insert(
            "filter_string".to_string(),
            Value::String(filter_string.clone()),
        );
    }
    Value::Object(value)
}

fn json_object_response(key: &str, value: Value) -> String {
    let mut object = Map::new();
    object.insert(key.to_string(), value);
    python_json_dumps(&Value::Object(object), false)
}

fn json_response(body: String) -> Response {
    ([("content-type", "application/json")], body).into_response()
}

fn required<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    value
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing(param))
}

fn required_json_string<'a>(value: Option<&'a Value>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(Value::String(value)) if !value.is_empty() => Ok(value),
        None | Some(Value::Null) | Some(Value::String(_)) => Err(missing(param)),
        Some(value) => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {} for parameter '{param}' supplied:  Hint: Value was of type '{}'. See the API docs for more information about request parameters.",
            serde_json::to_string(value).unwrap_or_default(),
            python_type_name(value)
        ))),
    }
}

fn missing(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. See the API docs for more information about request parameters."
    ))
}

fn validate_json_content_type(parts: &Parts) -> Result<(), MlflowError> {
    let valid = parts
        .headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.split(';').next() == Some("application/json"));
    if valid {
        Ok(())
    } else {
        Err(MlflowError::invalid_parameter_value(
            "Content-Type must be 'application/json'.",
        ))
    }
}

fn json_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_i64() || number.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::String(value) => format!("'{value}'"),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_string(),
        Value::Null => "None".to_string(),
        _ => value.to_string(),
    }
}
