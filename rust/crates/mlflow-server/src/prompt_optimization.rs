//! Prompt-optimization job CRUD (plan T16.6, §12.7).
//!
//! Python has no prompt-optimization table. The public proto is rebuilt from
//! the generic `jobs` row, while its MLflow run stores the submitted config as
//! params and the evaluation scores as metrics. Submission only creates those
//! two records here; the Phase 17 runner claims the generic job row and sends
//! it through the native worker protocol.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_store::{python_json_dumps, DatasetInputSpec, Job, JobStatus, RunStatus};
use serde_json::{json, Map, Value};

use crate::proto_http::{parse_request_lenient, proto_response};
use crate::schema_validation::{SchemaEntry, Validator};
use crate::state::AppState;
use crate::workspace::Workspace;

const OPTIMIZE_PROMPTS_JOB_NAME: &str = "optimize_prompts";

const CREATE_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "experiment_id",
        validators: &[Validator::String],
    },
    SchemaEntry {
        param: "source_prompt_uri",
        validators: &[Validator::String, Validator::Required],
    },
    SchemaEntry {
        param: "config",
        validators: &[Validator::Required],
    },
    SchemaEntry {
        param: "tags",
        validators: &[Validator::Array],
    },
];

const SEARCH_SCHEMA: &[SchemaEntry] = &[SchemaEntry {
    param: "experiment_id",
    validators: &[Validator::Required, Validator::String],
}];

pub async fn create_prompt_optimization_job(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreatePromptOptimizationJob = parse_request_lenient(
        &parts,
        &body,
        "mlflow.CreatePromptOptimizationJob",
        CREATE_SCHEMA,
    )?;

    let prompt_uri = req.source_prompt_uri.as_deref().unwrap_or("");
    if prompt_uri.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "source_prompt_uri is required for optimization job",
        ));
    }
    let config = req.config.as_ref().ok_or_else(|| {
        MlflowError::invalid_parameter_value("config is required for optimization job")
    })?;
    let optimizer_type = optimizer_type_name(config.optimizer_type.unwrap_or(0))?;
    let experiment_id = req.experiment_id.as_deref().unwrap_or("").trim();
    if experiment_id.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "experiment_id is required for optimization job",
        ));
    }

    // Python parses this before creating the run, so malformed JSON must not
    // leave an unused run behind. Keep the parsed JSON value in the job params:
    // the field is a JSON *string* on the request/run, but nested JSON here.
    let optimizer_config = match config
        .optimizer_config_json
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        Some(value) => Some(serde_json::from_str::<Value>(value).map_err(|error| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid JSON in optimizer_config_json: {}",
                python_json_decode_error(value, &error)
            ))
        })?),
        None => None,
    };
    let dataset_id = config.dataset_id.as_deref().unwrap_or("");
    let start_time = chrono::Utc::now().timestamp_millis();
    let (prompt_name, prompt_version) = parse_prompt_uri(prompt_uri);
    let run_name =
        format!("optimize_prompt_{optimizer_type}_{prompt_name}_{prompt_version}_{start_time}");
    let user = current_user();
    let run = state
        .tracking_store()
        .create_run(
            workspace.name(),
            experiment_id,
            Some(&user),
            Some(start_time),
            Some(&run_name),
            &[],
        )
        .await?;
    let run_id = run.info.run_id;

    let scorer_names = Value::Array(
        config
            .scorers
            .iter()
            .map(|name| Value::String(name.clone()))
            .collect(),
    );
    let scorer_names_json = python_json_dumps(&scorer_names, false);
    let mut run_params = vec![
        ("source_prompt_uri", prompt_uri),
        ("optimizer_type", optimizer_type),
        ("dataset_id", dataset_id),
        ("scorer_names", scorer_names_json.as_str()),
    ];
    if let Some(value) = config
        .optimizer_config_json
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        run_params.push(("optimizer_config_json", value));
    }
    state
        .tracking_store()
        .log_batch(workspace.name(), &run_id, &[], &run_params, &[])
        .await?;

    if !dataset_id.is_empty() {
        let dataset = state
            .tracking_store()
            .get_evaluation_dataset(workspace.name(), dataset_id)
            .await?;
        let input = DatasetInputSpec {
            name: dataset.name,
            digest: dataset.digest.unwrap_or_default(),
            source_type: "mlflow_evaluation_dataset".to_string(),
            source: python_json_dumps(&json!({"dataset_id": dataset_id}), false),
            schema: dataset.schema,
            profile: dataset.profile,
            tags: vec![(
                "mlflow.data.context".to_string(),
                "optimization".to_string(),
            )],
        };
        state
            .tracking_store()
            .log_inputs(workspace.name(), &run_id, &[input], &[])
            .await?;
    }

    let params = job_params(
        &run_id,
        experiment_id,
        prompt_uri,
        dataset_id,
        optimizer_type,
        optimizer_config,
        &config.scorers,
    );
    let serialized_params = python_json_dumps(&params, false);
    let job = state
        .job_store()
        .create_job(
            workspace.name(),
            OPTIMIZE_PROMPTS_JOB_NAME,
            &serialized_params,
            None,
        )
        .await?;
    // The D20 database row is the only queue. The Phase 17 runner claims it and
    // dispatches `optimize_prompts` through the native worker; no Huey side
    // queue exists in the Rust server.

    let response_job = pb::PromptOptimizationJob {
        job_id: Some(job.job_id),
        run_id: Some(run_id),
        state: Some(pb::JobState {
            status: Some(pb::JobStatus::Pending as i32),
            error_message: None,
            metadata: HashMap::new(),
        }),
        experiment_id: Some(experiment_id.to_string()),
        source_prompt_uri: Some(prompt_uri.to_string()),
        optimized_prompt_uri: None,
        config: req.config,
        creation_timestamp_ms: Some(job.creation_time),
        completion_timestamp_ms: None,
        tags: req
            .tags
            .into_iter()
            .map(|tag| pb::PromptOptimizationJobTag {
                key: Some(tag.key.unwrap_or_default()),
                value: Some(tag.value.unwrap_or_default()),
            })
            .collect(),
        initial_eval_scores: HashMap::new(),
        final_eval_scores: HashMap::new(),
    };
    proto_response(
        &pb::create_prompt_optimization_job::Response {
            job: Some(response_job),
        },
        "mlflow.CreatePromptOptimizationJob.Response",
    )
}

pub async fn get_prompt_optimization_job(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(job_id): Path<String>,
) -> Result<Response, MlflowError> {
    let job = state.job_store().get_job(workspace.name(), &job_id).await?;
    let mut optimization_job = build_prompt_optimization_job(&job)?;

    if let Some(run_id) = optimization_job.run_id.as_deref() {
        if let Ok(run) = state
            .tracking_store()
            .get_run(workspace.name(), run_id)
            .await
        {
            let mut total_metric_calls = None;
            for metric in run.data.metrics {
                match metric.key.split_once('.') {
                    None if metric.key == "initial_eval_score" => {
                        optimization_job
                            .initial_eval_scores
                            .insert("aggregate".to_string(), metric.value);
                    }
                    None if metric.key == "final_eval_score" => {
                        optimization_job
                            .final_eval_scores
                            .insert("aggregate".to_string(), metric.value);
                    }
                    Some(("initial_eval_score", scorer)) => {
                        optimization_job
                            .initial_eval_scores
                            .insert(scorer.to_string(), metric.value);
                    }
                    Some(("final_eval_score", scorer)) => {
                        optimization_job
                            .final_eval_scores
                            .insert(scorer.to_string(), metric.value);
                    }
                    None if metric.key == "total_metric_calls" => {
                        total_metric_calls = Some(metric.value);
                    }
                    _ => {}
                }
            }
            if let (Some(total), Some(maximum)) =
                (total_metric_calls, max_metric_calls(&job.params))
            {
                if maximum != 0.0 {
                    let progress = ((total / maximum).min(1.0) * 100.0).round_ties_even() / 100.0;
                    optimization_job
                        .state
                        .get_or_insert_with(empty_job_state)
                        .metadata
                        .insert(
                            "progress".to_string(),
                            mlflow_proto::python_float_repr(progress),
                        );
                }
            }
        }
    }

    proto_response(
        &pb::get_prompt_optimization_job::Response {
            job: Some(optimization_job),
        },
        "mlflow.GetPromptOptimizationJob.Response",
    )
}

pub async fn search_prompt_optimization_jobs(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchPromptOptimizationJobs = parse_request_lenient(
        &parts,
        &body,
        "mlflow.SearchPromptOptimizationJobs",
        SEARCH_SCHEMA,
    )?;
    let experiment_id = req.experiment_id.as_deref().unwrap_or("");
    let jobs = state
        .job_store()
        .list_jobs(
            workspace.name(),
            Some(OPTIMIZE_PROMPTS_JOB_NAME),
            &[],
            None,
            None,
            Some(&json!({"experiment_id": experiment_id})),
        )
        .await?;
    let jobs = jobs
        .iter()
        .map(build_prompt_optimization_job)
        .collect::<Result<Vec<_>, _>>()?;
    proto_response(
        &pb::search_prompt_optimization_jobs::Response { jobs },
        "mlflow.SearchPromptOptimizationJobs.Response",
    )
}

pub async fn cancel_prompt_optimization_job(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(job_id): Path<String>,
) -> Result<Response, MlflowError> {
    let job = state
        .job_store()
        .cancel_job(workspace.name(), &job_id)
        .await?;
    let mut optimization_job = build_prompt_optimization_job(&job)?;
    optimization_job
        .state
        .get_or_insert_with(empty_job_state)
        .status = Some(pb::JobStatus::Canceled as i32);

    if let Some(run_id) = optimization_job.run_id.as_deref() {
        let _ = state
            .tracking_store()
            .update_run_info(
                workspace.name(),
                run_id,
                Some(RunStatus::KILLED),
                Some(chrono::Utc::now().timestamp_millis()),
                None,
            )
            .await;
    }

    proto_response(
        &pb::cancel_prompt_optimization_job::Response {
            job: Some(optimization_job),
        },
        "mlflow.CancelPromptOptimizationJob.Response",
    )
}

pub async fn delete_prompt_optimization_job(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(job_id): Path<String>,
) -> Result<Response, MlflowError> {
    let job = state.job_store().get_job(workspace.name(), &job_id).await?;
    let run_id = build_prompt_optimization_job(&job)?.run_id;
    state
        .job_store()
        .delete_jobs(workspace.name(), 0, std::slice::from_ref(&job_id))
        .await?;
    if let Some(run_id) = run_id {
        if state
            .tracking_store()
            .get_run(workspace.name(), &run_id)
            .await
            .is_ok()
        {
            state
                .tracking_store()
                .delete_run(workspace.name(), &run_id)
                .await?;
        }
    }
    proto_response(
        &pb::delete_prompt_optimization_job::Response {},
        "mlflow.DeletePromptOptimizationJob.Response",
    )
}

fn build_prompt_optimization_job(job: &Job) -> Result<pb::PromptOptimizationJob, MlflowError> {
    let params: Value = serde_json::from_str(&job.params)
        .map_err(|error| MlflowError::internal_error(error.to_string()))?;
    let params = params.as_object().ok_or_else(|| {
        MlflowError::internal_error("Prompt optimization job params must be a JSON object")
    })?;

    let mut config = pb::PromptOptimizationJobConfig::default();
    let mut config_present = false;
    if let Some(optimizer_type) = params
        .get("optimizer_type")
        .and_then(Value::as_str)
        .and_then(optimizer_type_proto)
    {
        config.optimizer_type = Some(optimizer_type);
        config_present = true;
    }
    if let Some(dataset_id) = params
        .get("dataset_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        config.dataset_id = Some(dataset_id.to_string());
        config_present = true;
    }
    if let Some(mut scorers) = params.get("scorer_names").cloned() {
        if let Value::String(encoded) = &scorers {
            if let Ok(decoded) = serde_json::from_str(encoded) {
                scorers = decoded;
            }
        }
        if let Value::Array(values) = scorers {
            let names: Vec<String> = values
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            if !names.is_empty() {
                config.scorers = names;
                config_present = true;
            }
        }
    }
    if let Some(optimizer_config) = params
        .get("optimizer_config")
        .filter(|value| json_truthy(value))
    {
        match optimizer_config {
            Value::Object(_) => {
                config.optimizer_config_json = Some(python_json_dumps(optimizer_config, false));
                config_present = true;
            }
            Value::String(value) => {
                config.optimizer_config_json = Some(value.clone());
                config_present = true;
            }
            _ => {}
        }
    }

    let mut optimization_job = pb::PromptOptimizationJob {
        job_id: Some(job.job_id.clone()),
        run_id: params
            .get("run_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        state: Some(pb::JobState {
            status: Some(job_status_proto(job.status)),
            error_message: None,
            metadata: HashMap::new(),
        }),
        experiment_id: params
            .get("experiment_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        source_prompt_uri: params
            .get("prompt_uri")
            .and_then(Value::as_str)
            .map(str::to_string),
        optimized_prompt_uri: None,
        config: config_present.then_some(config),
        creation_timestamp_ms: Some(job.creation_time),
        completion_timestamp_ms: None,
        tags: Vec::new(),
        initial_eval_scores: HashMap::new(),
        final_eval_scores: HashMap::new(),
    };

    if job.status == JobStatus::Succeeded {
        if let Some(uri) = job
            .parsed_result()?
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|result| result.get("optimized_prompt_uri"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            optimization_job.optimized_prompt_uri = Some(uri.to_string());
        }
    } else if job.status == JobStatus::Failed {
        if let Some(error) = job.result.as_deref().filter(|value| !value.is_empty()) {
            optimization_job
                .state
                .get_or_insert_with(empty_job_state)
                .error_message = Some(error.to_string());
        }
    }
    Ok(optimization_job)
}

fn job_params(
    run_id: &str,
    experiment_id: &str,
    prompt_uri: &str,
    dataset_id: &str,
    optimizer_type: &str,
    optimizer_config: Option<Value>,
    scorer_names: &[String],
) -> Value {
    let mut params = Map::new();
    params.insert("run_id".to_string(), Value::String(run_id.to_string()));
    params.insert(
        "experiment_id".to_string(),
        Value::String(experiment_id.to_string()),
    );
    params.insert(
        "prompt_uri".to_string(),
        Value::String(prompt_uri.to_string()),
    );
    params.insert(
        "dataset_id".to_string(),
        Value::String(dataset_id.to_string()),
    );
    params.insert(
        "optimizer_type".to_string(),
        Value::String(optimizer_type.to_string()),
    );
    params.insert(
        "optimizer_config".to_string(),
        optimizer_config.unwrap_or(Value::Null),
    );
    params.insert(
        "scorer_names".to_string(),
        Value::Array(
            scorer_names
                .iter()
                .map(|name| Value::String(name.clone()))
                .collect(),
        ),
    );
    Value::Object(params)
}

fn optimizer_type_name(value: i32) -> Result<&'static str, MlflowError> {
    match pb::OptimizerType::try_from(value) {
        Ok(pb::OptimizerType::Gepa) => Ok("gepa"),
        Ok(pb::OptimizerType::Metaprompt) => Ok("metaprompt"),
        Ok(pb::OptimizerType::Unspecified) => Err(MlflowError::invalid_parameter_value(
            "optimizer_type is required. Supported types: ['OPTIMIZER_TYPE_GEPA', \
             'OPTIMIZER_TYPE_METAPROMPT']",
        )),
        Err(_) => Err(MlflowError::invalid_parameter_value(format!(
            "Unsupported optimizer_type value: {value}. Supported types: \
             ['OPTIMIZER_TYPE_GEPA', 'OPTIMIZER_TYPE_METAPROMPT']"
        ))),
    }
}

fn optimizer_type_proto(value: &str) -> Option<i32> {
    match value {
        "gepa" => Some(pb::OptimizerType::Gepa as i32),
        "metaprompt" => Some(pb::OptimizerType::Metaprompt as i32),
        _ => None,
    }
}

fn job_status_proto(status: JobStatus) -> i32 {
    match status {
        JobStatus::Pending => pb::JobStatus::Pending as i32,
        JobStatus::Running => pb::JobStatus::InProgress as i32,
        JobStatus::Succeeded => pb::JobStatus::Completed as i32,
        JobStatus::Failed | JobStatus::Timeout => pb::JobStatus::Failed as i32,
        JobStatus::Canceled => pb::JobStatus::Canceled as i32,
    }
}

fn empty_job_state() -> pb::JobState {
    pb::JobState {
        status: None,
        error_message: None,
        metadata: HashMap::new(),
    }
}

fn max_metric_calls(params: &str) -> Option<f64> {
    serde_json::from_str::<Value>(params)
        .ok()?
        .get("optimizer_config")?
        .get("max_metric_calls")?
        .as_f64()
        .filter(|value| *value != 0.0)
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn parse_prompt_uri(prompt_uri: &str) -> (&str, &str) {
    let Some(rest) = prompt_uri.strip_prefix("prompts:/") else {
        return ("", "");
    };
    let mut parts = rest.split('/');
    match (parts.next(), parts.next()) {
        (Some(name), Some(version)) => (name, version),
        _ => ("", ""),
    }
}

fn current_user() -> String {
    ["LOGNAME", "USER", "LNAME", "USERNAME"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "unknown".to_string())
}

fn python_json_decode_error(input: &str, error: &serde_json::Error) -> String {
    let serde_message = error.to_string();
    if serde_message.starts_with("expected value") {
        let line = error.line();
        let column = error.column();
        let character = input
            .split_inclusive('\n')
            .take(line.saturating_sub(1))
            .map(|part| part.chars().count())
            .sum::<usize>()
            + column.saturating_sub(1);
        return format!("Expecting value: line {line} column {column} (char {character})");
    }
    serde_message
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(status: JobStatus, params: Value, result: Option<&str>) -> Job {
        Job {
            job_id: "job-1".to_string(),
            creation_time: 123,
            job_name: OPTIMIZE_PROMPTS_JOB_NAME.to_string(),
            params: python_json_dumps(&params, false),
            workspace: "default".to_string(),
            timeout: None,
            status,
            result: result.map(str::to_string),
            retry_count: 0,
            last_update_time: 123,
            status_details: None,
        }
    }

    #[test]
    fn rebuilds_config_from_native_and_json_string_job_params() {
        let native = job(
            JobStatus::Pending,
            json!({
                "experiment_id": "7",
                "prompt_uri": "prompts:/source/3",
                "run_id": "run-1",
                "optimizer_type": "gepa",
                "dataset_id": "d-1",
                "scorer_names": ["Correctness", "Safety"],
                "optimizer_config": {"reflection_model": "openai:/gpt-5", "max_metric_calls": 8}
            }),
            None,
        );
        let rebuilt = build_prompt_optimization_job(&native).unwrap();
        let config = rebuilt.config.unwrap();
        assert_eq!(config.optimizer_type, Some(pb::OptimizerType::Gepa as i32));
        assert_eq!(config.scorers, ["Correctness", "Safety"]);
        assert_eq!(
            config.optimizer_config_json.as_deref(),
            Some("{\"reflection_model\": \"openai:/gpt-5\", \"max_metric_calls\": 8}")
        );

        let encoded = job(
            JobStatus::Pending,
            json!({
                "optimizer_type": "metaprompt",
                "scorer_names": "[\"Correctness\"]",
                "optimizer_config": "{\"guidelines\": \"brief\"}"
            }),
            None,
        );
        let config = build_prompt_optimization_job(&encoded)
            .unwrap()
            .config
            .unwrap();
        assert_eq!(config.scorers, ["Correctness"]);
        assert_eq!(
            config.optimizer_config_json.as_deref(),
            Some("{\"guidelines\": \"brief\"}")
        );
    }

    #[test]
    fn optimized_prompt_uri_only_comes_from_a_success_result() {
        let params = json!({"optimizer_type": "gepa"});
        let succeeded = job(
            JobStatus::Succeeded,
            params.clone(),
            Some(r#"{"optimized_prompt_uri":"prompts:/optimized/2"}"#),
        );
        assert_eq!(
            build_prompt_optimization_job(&succeeded)
                .unwrap()
                .optimized_prompt_uri
                .as_deref(),
            Some("prompts:/optimized/2")
        );

        let failed = job(
            JobStatus::Failed,
            params,
            Some(r#"{"optimized_prompt_uri":"prompts:/ignored/2"}"#),
        );
        let rebuilt = build_prompt_optimization_job(&failed).unwrap();
        assert_eq!(rebuilt.optimized_prompt_uri, None);
        assert_eq!(
            rebuilt.state.unwrap().error_message.as_deref(),
            Some(r#"{"optimized_prompt_uri":"prompts:/ignored/2"}"#)
        );
    }
}
