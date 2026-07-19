//! Promptlab run creation and static pyfunc artifact writer (plan §12.11, D19).

use std::collections::BTreeMap;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, request::Parts, StatusCode};
use axum::response::{IntoResponse, Response};
use chrono::Utc;
use futures::{stream, StreamExt};
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_store::{python_json_dumps, RunStatus};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::proto_http::{proto_response, validate_content_type};
use crate::runs::to_proto_run;
use crate::state::AppState;
use crate::workspace::Workspace;

const LOGGED_ARTIFACTS_TAG: &str = "mlflow.loggedArtifacts";
const RUN_SOURCE_TYPE_TAG: &str = "mlflow.runSourceType";
const PROMPTLAB_SOURCE_TYPE: &str = "PROMPT_ENGINEERING";
const LOADER_MODULE: &str = "mlflow.prompt.promptlab_model";

#[derive(Clone, Debug)]
struct KeyValue {
    key: String,
    value: String,
}

#[derive(Debug)]
struct PromptlabRequest {
    experiment_id: String,
    run_name: Option<String>,
    tags: Vec<KeyValue>,
    prompt_template: String,
    prompt_parameters: Vec<KeyValue>,
    model_route: String,
    model_parameters: Vec<KeyValue>,
    model_input: Value,
    model_output_parameters: Vec<KeyValue>,
    model_output: Value,
    user_id: String,
    start_time: i64,
}

pub async fn create_promptlab_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Response {
    if let Err(error) = validate_content_type(&parts) {
        return error.into_response();
    }
    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => return flask_bad_request(),
    };
    let request = match parse_request(&value) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    match create_impl(&state, workspace.name(), request).await {
        Ok(run) => proto_response(
            &pb::create_run::Response {
                run: Some(to_proto_run(run)),
            },
            "mlflow.CreateRun.Response",
        )
        .unwrap_or_else(IntoResponse::into_response),
        Err(error) => error.into_response(),
    }
}

fn parse_request(value: &Value) -> Result<PromptlabRequest, MlflowError> {
    let object = value.as_object().ok_or_else(|| {
        MlflowError::internal_error("CreatePromptlabRun request body must be a JSON object.")
    })?;
    let experiment_id = required_string(object, "experiment_id")?;
    let prompt_template = required_string(object, "prompt_template")?;
    let prompt_parameters = required_params(object, "prompt_parameters")?;
    let model_route = required_string(object, "model_route")?;
    let model_input = required(object, "model_input")?.clone();
    // The handler requires this field for client-version compatibility but the
    // Python writer pins the running server version, not the supplied value.
    required(object, "mlflow_version")?;

    let run_name = object
        .get("run_name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let tags = optional_params(object, "tags")?;
    let model_parameters = optional_params(object, "model_parameters")?;
    let model_output_parameters = optional_params(object, "model_output_parameters")?;
    let model_output = object.get("model_output").cloned().unwrap_or(Value::Null);
    let user_id = object
        .get("user_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let start_time = object
        .get("start_time")
        .and_then(Value::as_i64)
        .unwrap_or_else(now_millis);

    Ok(PromptlabRequest {
        experiment_id,
        run_name,
        tags,
        prompt_template,
        prompt_parameters,
        model_route,
        model_parameters,
        model_input,
        model_output_parameters,
        model_output,
        user_id,
        start_time,
    })
}

fn required<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a Value, MlflowError> {
    match object.get(name).filter(|value| python_truthy(value)) {
        Some(value) => Ok(value),
        None => Err(MlflowError::invalid_parameter_value(format!(
            "CreatePromptlabRun request must specify {name}."
        ))),
    }
}

fn required_string(object: &Map<String, Value>, name: &str) -> Result<String, MlflowError> {
    let value = required(object, name)?;
    value.as_str().map(str::to_string).ok_or_else(|| {
        MlflowError::internal_error(format!(
            "CreatePromptlabRun request {name} must be a string."
        ))
    })
}

fn required_params(object: &Map<String, Value>, name: &str) -> Result<Vec<KeyValue>, MlflowError> {
    required(object, name)?;
    optional_params(object, name)
}

fn optional_params(object: &Map<String, Value>, name: &str) -> Result<Vec<KeyValue>, MlflowError> {
    let Some(value) = object.get(name) else {
        return Ok(Vec::new());
    };
    let values = value.as_array().ok_or_else(|| {
        MlflowError::internal_error(format!(
            "CreatePromptlabRun request {name} must be an array."
        ))
    })?;
    values
        .iter()
        .map(|value| {
            let item = value.as_object().ok_or_else(|| {
                MlflowError::internal_error(format!(
                    "CreatePromptlabRun request {name} entries must be objects."
                ))
            })?;
            Ok(KeyValue {
                key: python_string(item.get("key").unwrap_or(&Value::Null)),
                value: python_string(item.get("value").unwrap_or(&Value::Null)),
            })
        })
        .collect()
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        // UI parameters are scalar. Preserve Python-like readable values for
        // unexpected containers rather than silently dropping them.
        Value::Array(_) | Value::Object(_) => python_json_dumps(value, false),
    }
}

async fn create_impl(
    state: &AppState,
    workspace: &str,
    request: PromptlabRequest,
) -> Result<mlflow_store::Run, MlflowError> {
    let input_tags: Vec<(&str, &str)> = request
        .tags
        .iter()
        .map(|tag| (tag.key.as_str(), tag.value.as_str()))
        .collect();
    let run = state
        .tracking_store()
        .create_run(
            workspace,
            &request.experiment_id,
            Some(&request.user_id),
            Some(request.start_time),
            request.run_name.as_deref(),
            &input_tags,
        )
        .await?;
    let run_id = run.info.run_id.clone();

    let result = populate_run(state, workspace, &run, &request).await;
    let status = if result.is_ok() {
        RunStatus::FINISHED
    } else {
        RunStatus::FAILED
    };
    // Python deliberately swallows every post-create failure and returns the
    // failed run; preserve that contract after updating the terminal state.
    state
        .tracking_store()
        .update_run_info(
            workspace,
            &run_id,
            Some(status),
            Some(now_millis()),
            request.run_name.as_deref(),
        )
        .await?;
    state.tracking_store().get_run(workspace, &run_id).await
}

async fn populate_run(
    state: &AppState,
    workspace: &str,
    run: &mlflow_store::Run,
    request: &PromptlabRequest,
) -> Result<(), MlflowError> {
    let mut params: Vec<(&str, &str)> = request
        .model_parameters
        .iter()
        .map(|param| (param.key.as_str(), param.value.as_str()))
        .collect();
    params.push(("model_route", &request.model_route));
    params.push(("prompt_template", &request.prompt_template));
    let logged_artifacts = "[{\"path\": \"eval_results_table.json\", \"type\": \"table\"}]";
    state
        .tracking_store()
        .log_batch(
            workspace,
            &run.info.run_id,
            &[],
            &params,
            &[
                (LOGGED_ARTIFACTS_TAG, logged_artifacts),
                (RUN_SOURCE_TYPE_TAG, PROMPTLAB_SOURCE_TYPE),
            ],
        )
        .await?;

    let utc_time_created = Utc::now().format("%Y-%m-%d %H:%M:%S.%6f").to_string();
    let model_uuid = Uuid::new_v4().simple().to_string();
    let history_model = json!({
        "run_id": run.info.run_id,
        "artifact_path": "model",
        "utc_time_created": utc_time_created,
        "model_uuid": model_uuid,
        "flavors": {},
    });
    state
        .tracking_store()
        .record_logged_model(workspace, &run.info.run_id, &history_model)
        .await?;

    let artifact_uri = run.info.artifact_uri.as_deref().unwrap_or("");
    for (path, contents) in
        artifact_files(&run.info.run_id, &utc_time_created, &model_uuid, request)?
    {
        let resolved = state.resolve_artifact(artifact_uri, &path)?;
        resolved
            .repo
            .put(
                &resolved.path,
                stream::once(async move { Ok(Bytes::from(contents)) }).boxed(),
            )
            .await?;
    }
    Ok(())
}

fn artifact_files(
    run_id: &str,
    utc_time_created: &str,
    model_uuid: &str,
    request: &PromptlabRequest,
) -> Result<Vec<(String, String)>, MlflowError> {
    let prompt_values: Vec<Value> = request
        .prompt_parameters
        .iter()
        .map(|param| Value::String(param.value.clone()))
        .collect();
    let input_example = json!({"inputs": prompt_values});
    let serving_example = json!({"inputs": input_example});

    let mut eval_columns: Vec<Value> = request
        .prompt_parameters
        .iter()
        .map(|param| Value::String(param.key.clone()))
        .collect();
    eval_columns.extend([
        Value::String("prompt".into()),
        Value::String("output".into()),
    ]);
    eval_columns.extend(
        request
            .model_output_parameters
            .iter()
            .map(|param| Value::String(param.key.clone())),
    );
    let mut eval_data: Vec<Value> = request
        .prompt_parameters
        .iter()
        .map(|param| Value::String(param.value.clone()))
        .collect();
    eval_data.extend([request.model_input.clone(), request.model_output.clone()]);
    eval_data.extend(
        request
            .model_output_parameters
            .iter()
            .map(|param| Value::String(param.value.clone())),
    );
    let eval_results = json!({"columns": eval_columns, "data": [eval_data]});

    Ok(vec![
        (
            "model/MLmodel".into(),
            mlmodel_yaml(run_id, utc_time_created, model_uuid, request)?,
        ),
        ("model/conda.yaml".into(), conda_yaml()),
        (
            "model/input_example.json".into(),
            python_json_dumps(&input_example, false),
        ),
        ("model/parameters.yaml".into(), parameters_yaml(request)?),
        ("model/python_env.yaml".into(), python_env_yaml()),
        (
            "model/requirements.txt".into(),
            format!("mlflow[gateway]=={}", crate::routes::MLFLOW_VERSION),
        ),
        (
            "model/serving_input_example.json".into(),
            serde_json::to_string_pretty(&serving_example)
                .map_err(|error| MlflowError::internal_error(error.to_string()))?,
        ),
        (
            "eval_results_table.json".into(),
            python_json_dumps(&eval_results, false),
        ),
    ])
}

fn mlmodel_yaml(
    run_id: &str,
    utc_time_created: &str,
    model_uuid: &str,
    request: &PromptlabRequest,
) -> Result<String, MlflowError> {
    let input_signature: Vec<Value> = request
        .prompt_parameters
        .iter()
        .map(|param| json!({"type":"string","name":param.key,"required":true}))
        .collect();
    let output_signature = vec![json!({"type":"string","name":"output","required":true})];
    let python = promptlab_python_version();
    let model = json!({
        "artifact_path": "model",
        "flavors": {
            "python_function": {
                "env": {"conda":"conda.yaml","virtualenv":"python_env.yaml"},
                "loader_module": LOADER_MODULE,
                "parameters_path": "parameters.yaml",
                "python_version": python,
            }
        },
        "is_signature_from_type_hint": false,
        "mlflow_version": crate::routes::MLFLOW_VERSION,
        "model_id": Value::Null,
        "model_uuid": model_uuid,
        "prompts": Value::Null,
        "run_id": run_id,
        "saved_input_example_info": {
            "artifact_path":"input_example.json",
            "serving_input_path":"serving_input_example.json",
            "type":"json_object"
        },
        "signature": {
            "inputs": python_json_dumps(&Value::Array(input_signature), false),
            "outputs": python_json_dumps(&Value::Array(output_signature), false),
            "params": Value::Null,
        },
        "type_hint_from_example": false,
        "utc_time_created": utc_time_created,
    });
    yaml(&model)
}

fn parameters_yaml(request: &PromptlabRequest) -> Result<String, MlflowError> {
    let model_parameters: BTreeMap<&str, &str> = request
        .model_parameters
        .iter()
        .map(|param| (param.key.as_str(), param.value.as_str()))
        .collect();
    let prompt_parameters: BTreeMap<&str, &str> = request
        .prompt_parameters
        .iter()
        .map(|param| (param.key.as_str(), param.value.as_str()))
        .collect();
    let value = json!({
        "model_parameters": model_parameters,
        "model_route": request.model_route,
        "prompt_parameters": prompt_parameters,
        "prompt_template": request.prompt_template,
    });
    yaml(&value)
}

fn yaml(value: &Value) -> Result<String, MlflowError> {
    serde_yaml::to_string(value).map_err(|error| MlflowError::internal_error(error.to_string()))
}

// Promptlab's Python writer captures its server interpreter/build tools. The
// Rust distribution targets the repository's pinned Python 3.10 environment;
// these values are artifact metadata only and are not executed by Rust.
fn promptlab_python_version() -> &'static str {
    "3.10.18"
}

fn conda_yaml() -> String {
    format!(
        "channels:\n- conda-forge\ndependencies:\n- python={}\n- pip<=26.1.2\n- pip:\n  - mlflow[gateway]=={}\nname: mlflow-env\n",
        promptlab_python_version(),
        crate::routes::MLFLOW_VERSION
    )
}

fn python_env_yaml() -> String {
    format!(
        "python: {}\nbuild_dependencies:\n- pip==26.1.2\n- setuptools==81.0.0\n- wheel==0.47.0\ndependencies:\n- -r requirements.txt\n",
        promptlab_python_version()
    )
}

fn now_millis() -> i64 {
    Utc::now().timestamp_millis()
}

fn flask_bad_request() -> Response {
    const BODY: &str = "<!doctype html>\n<html lang=en>\n<title>400 Bad Request</title>\n<h1>Bad Request</h1>\n<p>The browser (or proxy) sent a request that this server could not understand.</p>\n";
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(BODY))
        .expect("valid response")
}
