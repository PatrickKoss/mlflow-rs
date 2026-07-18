//! UI-only GenAI invoke submission routes (plan T17.4, §12.2-§12.4).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use mlflow_error::MlflowError;
use mlflow_genai::{ScorerPayloadError, SerializedScorer};
use mlflow_store::{python_json_dumps, RunStatus};
use serde_json::{json, Map, Value};

use crate::auth_middleware::AuthContext;
use crate::schema_validation::{validate_request_json_with_schema, SchemaEntry, Validator};
use crate::state::AppState;
use crate::workspace::Workspace;

const EVALUATE_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "experiment_id",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "trace_ids",
        validators: &[Validator::Required, Validator::Array],
    },
    SchemaEntry {
        param: "serialized_scorers",
        validators: &[Validator::Required, Validator::Array],
    },
];

const ISSUE_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "experiment_id",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "trace_ids",
        validators: &[Validator::Required, Validator::Array],
    },
    SchemaEntry {
        param: "categories",
        validators: &[Validator::Required, Validator::Array],
    },
    SchemaEntry {
        param: "provider",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "model",
        validators: &[Validator::String],
    },
    SchemaEntry {
        param: "secret_id",
        validators: &[Validator::String],
    },
    SchemaEntry {
        param: "endpoint_name",
        validators: &[Validator::String],
    },
];

pub async fn invoke_genai_evaluate(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let request = validated_json(&parts, &body, EVALUATE_SCHEMA)?;
    let object = request.as_object().expect("validated request is an object");
    let experiment_id = object["experiment_id"].as_str().unwrap();
    let trace_ids = object["trace_ids"].as_array().unwrap();
    let serialized_scorers = object["serialized_scorers"].as_array().unwrap();
    if trace_ids.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Please select at least one trace to evaluate.",
        ));
    }
    if serialized_scorers.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Please select at least one judge.",
        ));
    }

    let run = state
        .tracking_store()
        .create_run(
            workspace.name(),
            experiment_id,
            Some("unknown"),
            Some(chrono::Utc::now().timestamp_millis()),
            None,
            &[("mlflow.runType", "genai_evaluate")],
        )
        .await?;
    let run_id = run.info.run_id;
    let username = request_username(&parts);
    let params = ordered_object([
        ("trace_ids", Value::Array(trace_ids.clone())),
        (
            "serialized_scorers",
            Value::Array(serialized_scorers.clone()),
        ),
        ("run_id", Value::String(run_id.clone())),
        ("username", username.map_or(Value::Null, Value::String)),
    ]);
    let job = match create_job(&state, workspace.name(), "invoke_genai_evaluate", params).await {
        Ok(job) => job,
        Err(error) => {
            let _ = state
                .tracking_store()
                .update_run_info(
                    workspace.name(),
                    &run_id,
                    Some(RunStatus::FAILED),
                    Some(chrono::Utc::now().timestamp_millis()),
                    None,
                )
                .await;
            return Err(error);
        }
    };
    state
        .tracking_store()
        .set_tag(
            workspace.name(),
            &run_id,
            "mlflow.genaiEvaluate.jobId",
            &job.job_id,
        )
        .await?;
    flask_json(json!({"job_id": job.job_id, "run_id": run_id}))
}

pub async fn invoke_scorer(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let request = request_json(&parts, &body)?;
    let object = request
        .as_object()
        .ok_or_else(|| MlflowError::internal_error("'list' object has no attribute 'get'"))?;
    let experiment_id = object.get("experiment_id").cloned().unwrap_or(Value::Null);
    let serialized_scorer = object
        .get("serialized_scorer")
        .cloned()
        .unwrap_or(Value::Null);
    let trace_ids = object
        .get("trace_ids")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let log_assessments = object
        .get("log_assessments")
        .cloned()
        .unwrap_or(Value::Bool(false));

    if !json_truthy(&experiment_id) {
        return Err(MlflowError::invalid_parameter_value(
            "Missing required parameter: experiment_id",
        ));
    }
    if !json_truthy(&serialized_scorer) {
        return Err(MlflowError::invalid_parameter_value(
            "Missing required parameter: serialized_scorer",
        ));
    }
    if !json_truthy(&trace_ids) {
        return Err(MlflowError::invalid_parameter_value(
            "Please select at least one trace to evaluate.",
        ));
    }
    let serialized_scorer = serialized_scorer.as_str().ok_or_else(|| {
        MlflowError::internal_error("the JSON object must be str, bytes or bytearray, not dict")
    })?;
    let scorer = validate_serialized_scorer(serialized_scorer)?;
    let trace_ids = trace_ids
        .as_array()
        .ok_or_else(|| MlflowError::internal_error("scorer trace_ids must be an array"))?;
    let trace_ids = trace_ids
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| MlflowError::internal_error("scorer trace_ids must contain strings"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let batches = if scorer.common().is_session_level_scorer {
        session_batches(&state, workspace.name(), &trace_ids).await?
    } else {
        fixed_batches(&trace_ids)?
    };

    let username = request_username(&parts);
    let mut jobs = Vec::with_capacity(batches.len());
    for batch in batches {
        let params = ordered_object([
            ("experiment_id", experiment_id.clone()),
            (
                "serialized_scorer",
                Value::String(serialized_scorer.to_string()),
            ),
            (
                "trace_ids",
                Value::Array(batch.iter().cloned().map(Value::String).collect()),
            ),
            ("log_assessments", log_assessments.clone()),
            (
                "username",
                username.clone().map_or(Value::Null, Value::String),
            ),
        ]);
        let job = create_job(&state, workspace.name(), "invoke_scorer", params).await?;
        jobs.push(json!({"job_id": job.job_id, "trace_ids": batch}));
    }
    flask_json(json!({"jobs": jobs}))
}

pub async fn invoke_issue_detection(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let request = validated_json(&parts, &body, ISSUE_SCHEMA)?;
    let object = request.as_object().expect("validated request is an object");
    let experiment_id = object["experiment_id"].as_str().unwrap();
    let trace_ids = object["trace_ids"].as_array().unwrap();
    let categories = object["categories"].as_array().unwrap();
    let provider = object["provider"].as_str().unwrap();
    let model = object.get("model").and_then(Value::as_str);
    let endpoint_name = object.get("endpoint_name").and_then(Value::as_str);
    if endpoint_name.is_none_or(str::is_empty)
        && (provider.is_empty() || model.is_none_or(str::is_empty))
    {
        return Err(MlflowError::internal_error(
            "Either 'endpoint_name' or both 'provider' and 'model' must be provided",
        ));
    }
    let categories = categories
        .iter()
        .map(|value| {
            value.as_str().ok_or_else(|| {
                MlflowError::internal_error("sequence item 0: expected str instance")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let model_name = endpoint_name.filter(|value| !value.is_empty()).map_or_else(
        || format!("{provider}:/{}", model.unwrap()),
        |name| format!("gateway:/{name}"),
    );
    let start_time = chrono::Utc::now().timestamp_millis();
    let context_tags = issue_run_context_tags();
    let user_id = context_tags
        .iter()
        .find_map(|(key, value)| (key == "mlflow.user").then_some(value.clone()))
        .unwrap_or_else(|| "unknown".to_string());
    let total_traces = trace_ids.len().to_string();
    let categories_tag = categories.join(",");
    let mut tags = context_tags;
    tags.extend([
        ("mlflow.runType".to_string(), "issue_detection".to_string()),
        ("categories".to_string(), categories_tag),
        ("model".to_string(), model_name.clone()),
        ("total_traces".to_string(), total_traces),
    ]);
    if let Some(endpoint_name) = endpoint_name.filter(|value| !value.is_empty()) {
        tags.push(("endpoint_name".to_string(), endpoint_name.to_string()));
    }
    let tag_refs = tags
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    let run = state
        .tracking_store()
        .create_run(
            workspace.name(),
            experiment_id,
            Some(&user_id),
            Some(start_time),
            None,
            &tag_refs,
        )
        .await?;
    let run_id = run.info.run_id;
    let params = ordered_object([
        ("experiment_id", Value::String(experiment_id.to_string())),
        ("trace_ids", Value::Array(trace_ids.clone())),
        (
            "categories",
            Value::Array(categories.iter().map(|value| json!(value)).collect()),
        ),
        ("run_id", Value::String(run_id.clone())),
        ("model", Value::String(model_name)),
    ]);
    // Provider-secret decryption belongs to T18.1. The secret identifier is
    // deliberately not persisted in Python's durable job row; fixture-mode
    // T17 execution therefore has the same DB contract with or without it.
    let job = create_job(&state, workspace.name(), "invoke_issue_detection", params).await?;
    state
        .tracking_store()
        .set_tag(
            workspace.name(),
            &run_id,
            "mlflow.issueDetection.jobId",
            &job.job_id,
        )
        .await?;
    state
        .tracking_store()
        .update_run_info(
            workspace.name(),
            &run_id,
            Some(RunStatus::RUNNING),
            Some(chrono::Utc::now().timestamp_millis()),
            None,
        )
        .await?;
    flask_json(json!({"job_id": job.job_id, "run_id": run_id}))
}

async fn create_job(
    state: &AppState,
    workspace: &str,
    name: &str,
    params: Value,
) -> Result<mlflow_store::Job, MlflowError> {
    state
        .job_store()
        .create_job(workspace, name, &python_json_dumps(&params, false), None)
        .await
}

async fn session_batches(
    state: &AppState,
    workspace: &str,
    trace_ids: &[String],
) -> Result<Vec<Vec<String>>, MlflowError> {
    let infos = state
        .tracking_store()
        .batch_get_trace_infos(workspace, trace_ids)
        .await?;
    let mut groups = Vec::<(String, Vec<(String, i64)>)>::new();
    for info in infos {
        let Some(session_id) = info.metadata("mlflow.trace.session") else {
            continue;
        };
        let index = groups
            .iter()
            .position(|(existing, _)| existing == session_id)
            .unwrap_or_else(|| {
                groups.push((session_id.to_string(), Vec::new()));
                groups.len() - 1
            });
        groups[index].1.push((info.trace_id, info.request_time));
    }
    Ok(groups
        .into_iter()
        .map(|(_, mut traces)| {
            traces.sort_by_key(|(_, timestamp)| {
                if *timestamp == 0 {
                    i64::MAX
                } else {
                    *timestamp
                }
            });
            traces.into_iter().map(|(trace_id, _)| trace_id).collect()
        })
        .collect())
}

fn fixed_batches(trace_ids: &[String]) -> Result<Vec<Vec<String>>, MlflowError> {
    let size = match std::env::var("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE") {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            MlflowError::internal_error(format!(
                "invalid literal for int() with base 10: {value:?}: {error}"
            ))
        })?,
        Err(_) => 100,
    };
    if size == 0 {
        return Err(MlflowError::internal_error(
            "range() arg 3 must not be zero",
        ));
    }
    Ok(trace_ids.chunks(size).map(<[String]>::to_vec).collect())
}

fn validate_serialized_scorer(value: &str) -> Result<SerializedScorer, MlflowError> {
    SerializedScorer::from_json(value).map_err(|error| {
        let message = match error {
            ScorerPayloadError::Json(error) => format!(
                "Invalid JSON in serialized scorer: {}",
                python_json_error(value, &error)
            ),
            ScorerPayloadError::RepresentationCount(0) => {
                let parsed: Value = serde_json::from_str(value).unwrap_or(Value::Null);
                if parsed.as_object().is_some_and(|object| !object.contains_key("name")) {
                    "Failed to validate scorer: Failed to parse serialized scorer data: SerializedScorer.__init__() missing 1 required positional argument: 'name'".to_string()
                } else {
                    format!(
                        "Failed to validate scorer: Failed to load scorer '{}'. The scorer is serialized in an unknown format that cannot be deserialized.",
                        parsed.get("name").and_then(Value::as_str).unwrap_or("")
                    )
                }
            }
            other => format!("Failed to validate scorer: {other}"),
        };
        MlflowError::invalid_parameter_value(message)
    })
}

fn validated_json(
    parts: &Parts,
    body: &Bytes,
    schema: &[SchemaEntry],
) -> Result<Value, MlflowError> {
    let value = request_json(parts, body)?;
    validate_request_json_with_schema(&value, schema, false)?;
    Ok(value)
}

fn request_json(parts: &Parts, body: &Bytes) -> Result<Value, MlflowError> {
    validate_content_type(parts)?;
    let parsed = serde_json::from_slice::<Value>(body).unwrap_or(Value::Null);
    let parsed = match parsed {
        Value::String(encoded) => serde_json::from_str(&encoded)
            .map_err(|error| MlflowError::internal_error(error.to_string()))?,
        value => value,
    };
    Ok(if parsed.is_null() { json!({}) } else { parsed })
}

fn validate_content_type(parts: &Parts) -> Result<(), MlflowError> {
    let Some(content_type) = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(MlflowError::invalid_parameter_value(
            "Bad Request. Content-Type header is missing.",
        ));
    };
    if content_type.split(';').next() != Some("application/json") {
        return Err(MlflowError::invalid_parameter_value(
            "Bad Request. Content-Type must be one of ['application/json'].",
        ));
    }
    Ok(())
}

fn request_username(parts: &Parts) -> Option<String> {
    if let Some(auth) = parts.extensions.get::<AuthContext>() {
        return Some(auth.username.clone());
    }
    let encoded = parts
        .headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Basic ")?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    decoded
        .split_once(':')
        .map(|(username, _)| username.to_string())
}

fn issue_run_context_tags() -> Vec<(String, String)> {
    let user = std::env::var("MLFLOW_TRACKING_USERNAME")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(current_user);
    let source = std::env::args()
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "<console>".to_string());
    let mut tags = vec![
        ("mlflow.user".to_string(), user),
        ("mlflow.source.name".to_string(), source),
        ("mlflow.source.type".to_string(), "LOCAL".to_string()),
    ];
    if let Ok(context) = std::env::var("MLFLOW_RUN_CONTEXT") {
        if let Ok(Value::Object(context)) = serde_json::from_str(&context) {
            for (key, value) in context {
                if let Some(value) = value.as_str() {
                    if let Some((_, existing)) = tags.iter_mut().find(|(tag, _)| tag == &key) {
                        *existing = value.to_string();
                    } else {
                        tags.push((key, value.to_string()));
                    }
                }
            }
        }
    }
    tags
}

fn current_user() -> String {
    ["LOGNAME", "USER", "LNAME", "USERNAME"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "unknown".to_string())
}

fn ordered_object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    let mut object = Map::new();
    for (key, value) in entries {
        object.insert(key.to_string(), value);
    }
    Value::Object(object)
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_json_error(input: &str, error: &serde_json::Error) -> String {
    let line = error.line();
    let column = error.column();
    let character = input
        .split_inclusive('\n')
        .take(line.saturating_sub(1))
        .map(str::len)
        .sum::<usize>()
        + column.saturating_sub(1);
    if error.to_string().starts_with("expected value")
        || error.to_string().starts_with("expected ident")
    {
        let (line, column, character) = if error.to_string().starts_with("expected ident") {
            (1, 1, 0)
        } else {
            (line, column, character)
        };
        format!("Expecting value: line {line} column {column} (char {character})")
    } else {
        error.to_string()
    }
}

fn flask_json(value: Value) -> Result<Response, MlflowError> {
    let mut body = serde_json::to_string(&value)
        .map_err(|error| MlflowError::internal_error(error.to_string()))?;
    body.push('\n');
    Ok(([(header::CONTENT_TYPE, "application/json")], body).into_response())
}
