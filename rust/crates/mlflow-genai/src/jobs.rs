use std::sync::Arc;

use futures::stream::{self, StreamExt};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::online::{run_online_session_job, run_online_trace_job, OnlineJobParams};
use crate::store::{assessment_dictionary, set_scorer_trace_metadata, TraceRecord, TrackingClient};
use crate::{
    compute_aggregated_metrics, EngineError, EvaluationConfig, EvaluationEngine, NamedScorer,
    RateConfig, SerializedScorer, WorkerRequest,
};

const TRACE_SESSION: &str = "mlflow.trace.session";

#[derive(Debug, Deserialize)]
struct EvaluateParams {
    trace_ids: Vec<String>,
    serialized_scorers: Vec<String>,
    run_id: String,
}

#[derive(Debug, Deserialize)]
struct InvokeParams {
    serialized_scorer: String,
    trace_ids: Vec<String>,
    #[serde(default = "default_true")]
    log_assessments: bool,
}

const fn default_true() -> bool {
    true
}

pub(crate) async fn execute_evaluate(request: &WorkerRequest) -> Result<Value, EngineError> {
    let params: EvaluateParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    let client = TrackingClient::from_request(request)?;
    let setup = async {
        client
            .link_traces_to_run(&params.trace_ids, &params.run_id)
            .await?;
        let traces = client.fetch_traces(&params.trace_ids).await?;
        ensure_all_traces(&params.trace_ids, &traces)?;
        let traces = ordered_traces(&params.trace_ids, traces);
        let scorers = named_scorers(&params.serialized_scorers)?;
        Ok::<_, EngineError>((traces, scorers))
    }
    .await;
    let (traces, scorers) = match setup {
        Ok(setup) => setup,
        Err(error) => {
            let _ = client.terminate_run(&params.run_id, "FAILED").await;
            return Err(error);
        }
    };
    let experiment_id = traces
        .first()
        .map(|trace| trace.experiment_id.clone())
        .unwrap_or_default();
    let (single, session) = scorers
        .iter()
        .cloned()
        .partition::<Vec<_>, _>(|scorer| !scorer.scorer.common().is_session_level_scorer);
    let config = EvaluationConfig::from_env(scorers.len())?;
    let engine = EvaluationEngine::new(config.clone())?;
    let items = traces
        .iter()
        .map(|trace| trace.eval_item.clone())
        .collect::<Vec<_>>();
    let single_results = engine.score_items(items, Arc::new(single.clone())).await;
    let mut all_assessments = Vec::new();
    for (trace, mut result) in traces.iter().zip(single_results) {
        add_evaluator_traces(
            &client,
            &experiment_id,
            Some(&params.run_id),
            trace,
            &single,
            &mut result.assessments,
            config.enable_scorer_tracing,
        )
        .await;
        for assessment in &result.assessments {
            // Python warns and continues on assessment write failures.
            let _ = client
                .log_assessment(trace, assessment, Some(&params.run_id))
                .await;
        }
        all_assessments.extend(result.assessments);
    }

    let session_groups = session_groups(&traces);
    let session_engine = engine.with_scorer_workers(session.len());
    let session_results = stream::iter(session_groups.into_iter())
        .map(|(session_id, grouped)| {
            let engine = session_engine.clone();
            let session = session.clone();
            async move {
                let first = grouped
                    .iter()
                    .min_by_key(|trace| trace.timestamp_ms)
                    .expect("session group is non-empty")
                    .clone();
                let mut item = first.eval_item.clone();
                item.session = Some(
                    grouped
                        .iter()
                        .filter_map(|trace| trace.eval_item.trace.clone())
                        .collect(),
                );
                let mut result = engine.score_item(&item, &session).await;
                for assessment in &mut result.assessments {
                    assessment
                        .metadata
                        .insert(TRACE_SESSION.to_string(), Value::String(session_id.clone()));
                }
                (first, result)
            }
        })
        .buffer_unordered(config.row_workers)
        .collect::<Vec<_>>()
        .await;
    for (trace, result) in session_results {
        for assessment in &result.assessments {
            let _ = client
                .log_assessment(&trace, assessment, Some(&params.run_id))
                .await;
        }
        all_assessments.extend(result.assessments);
    }

    let metrics = compute_aggregated_metrics(&all_assessments, &scorers);
    if let Err(error) = client.log_metrics(&params.run_id, &metrics).await {
        let _ = client.terminate_run(&params.run_id, "FAILED").await;
        return Err(error);
    }
    client.terminate_run(&params.run_id, "FINISHED").await?;
    Ok(json!({
        "run_id": params.run_id,
        "total_traces": params.trace_ids.len(),
        "scorer_count": scorers.len(),
    }))
}

pub(crate) async fn execute_invoke(request: &WorkerRequest) -> Result<Value, EngineError> {
    let params: InvokeParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    let client = TrackingClient::from_request(request)?;
    let traces = client.fetch_traces(&params.trace_ids).await?;
    ensure_all_traces(&params.trace_ids, &traces)?;
    let traces = ordered_traces(&params.trace_ids, traces);
    let scorer = named_scorers(&[params.serialized_scorer])?
        .into_iter()
        .next()
        .expect("one scorer was supplied");
    if scorer.scorer.common().is_session_level_scorer {
        execute_session_invoke(&client, &traces, &scorer, params.log_assessments).await
    } else {
        execute_single_invoke(&client, traces, scorer, params.log_assessments).await
    }
}

async fn execute_single_invoke(
    client: &TrackingClient,
    traces: Vec<TraceRecord>,
    scorer: NamedScorer,
    log_assessments: bool,
) -> Result<Value, EngineError> {
    let workers = env_workers("MLFLOW_GENAI_EVAL_MAX_WORKERS", 10)?;
    let config = EvaluationConfig {
        row_workers: workers,
        scorer_workers: 1,
        max_retries: 0,
        scorer_rate: RateConfig {
            requests_per_second: None,
            adaptive: false,
        },
        enable_scorer_tracing: scorer_tracing(),
    };
    let engine = EvaluationEngine::new(config.clone())?;
    let scorer = vec![scorer];
    let results = stream::iter(traces.into_iter())
        .map(|trace| {
            let engine = engine.clone();
            let scorer = scorer.clone();
            async move {
                let result = engine.score_item(&trace.eval_item, &scorer).await;
                (trace, result)
            }
        })
        .buffer_unordered(workers)
        .collect::<Vec<_>>()
        .await;
    let mut output = Map::new();
    for (trace, mut result) in results {
        if config.enable_scorer_tracing {
            // Invoke jobs have no run context, but scorer traces live beside
            // the input trace. The experiment ID is not required for normal
            // execution and defaults to the single-tenant experiment in tests.
            add_evaluator_traces(
                client,
                &trace.experiment_id,
                None,
                &trace,
                &scorer,
                &mut result.assessments,
                true,
            )
            .await;
        }
        let mut failures = Vec::new();
        for assessment in &result.assessments {
            if let Some(error) = &assessment.error {
                failures.push(json!({
                    "error_code": error.error_code,
                    "error_message": error.error_message,
                }));
            }
        }
        let mut dictionaries = result
            .assessments
            .iter()
            .map(assessment_dictionary)
            .collect::<Vec<_>>();
        if log_assessments {
            for (assessment, dictionary) in result.assessments.iter().zip(&mut dictionaries) {
                match client.log_assessment(&trace, assessment, None).await {
                    Ok(_) => {
                        dictionary["trace_id"] = Value::String(trace.trace_id.clone());
                        if let Some(span_id) = &trace.root_span_id {
                            dictionary["span_id"] = Value::String(span_id.clone());
                        }
                    }
                    Err(error) => {
                        failures = vec![json!({
                            "error_code": "EngineError",
                            "error_message": error.to_string(),
                        })];
                        dictionaries.clear();
                        break;
                    }
                }
            }
        }
        output.insert(
            trace.trace_id,
            json!({"assessments": dictionaries, "failures": failures}),
        );
    }
    Ok(Value::Object(output))
}

async fn execute_session_invoke(
    client: &TrackingClient,
    traces: &[TraceRecord],
    scorer: &NamedScorer,
    log_assessments: bool,
) -> Result<Value, EngineError> {
    let first = traces
        .iter()
        .min_by_key(|trace| trace.timestamp_ms)
        .expect("invoke trace list is non-empty");
    let session_id = first.session_id().ok_or_else(|| {
        EngineError::InvalidParams(format!(
            "Session-level scorer requires traces with session metadata. Trace {} is missing 'mlflow.trace.session' in its metadata.",
            first.trace_id
        ))
    })?;
    let mut item = first.eval_item.clone();
    item.session = Some(
        traces
            .iter()
            .filter_map(|trace| trace.eval_item.trace.clone())
            .collect(),
    );
    let config = EvaluationConfig {
        row_workers: 1,
        scorer_workers: 1,
        max_retries: 0,
        scorer_rate: RateConfig {
            requests_per_second: None,
            adaptive: false,
        },
        enable_scorer_tracing: scorer_tracing(),
    };
    let engine = EvaluationEngine::new(config)?;
    let mut result = engine.score_item(&item, std::slice::from_ref(scorer)).await;
    for assessment in &mut result.assessments {
        assessment.metadata.insert(
            TRACE_SESSION.to_string(),
            Value::String(session_id.to_string()),
        );
    }
    let mut failures = result
        .assessments
        .iter()
        .filter_map(|assessment| assessment.error.as_ref())
        .map(|error| json!({"error_code": error.error_code, "error_message": error.error_message}))
        .collect::<Vec<_>>();
    let mut dictionaries = result
        .assessments
        .iter()
        .map(assessment_dictionary)
        .collect::<Vec<_>>();
    if log_assessments {
        for (assessment, dictionary) in result.assessments.iter().zip(&mut dictionaries) {
            match client.log_assessment(first, assessment, None).await {
                Ok(_) => {
                    dictionary["trace_id"] = Value::String(first.trace_id.clone());
                    if let Some(span_id) = &first.root_span_id {
                        dictionary["span_id"] = Value::String(span_id.clone());
                    }
                }
                Err(error) => {
                    failures = vec![json!({
                        "error_code": "EngineError",
                        "error_message": error.to_string(),
                    })];
                    dictionaries.clear();
                    break;
                }
            }
        }
    }
    Ok(json!({
        first.trace_id.clone(): {"assessments": dictionaries, "failures": failures}
    }))
}

pub(crate) async fn execute_online_trace(request: &WorkerRequest) -> Result<Value, EngineError> {
    let params: OnlineJobParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    run_online_trace_job(request, params).await
}

pub(crate) async fn execute_online_session(request: &WorkerRequest) -> Result<Value, EngineError> {
    let params: OnlineJobParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    run_online_session_job(request, params).await
}

fn named_scorers(serialized: &[String]) -> Result<Vec<NamedScorer>, EngineError> {
    let gateway_url = std::env::var("MLFLOW_GATEWAY_URI")
        .ok()
        .map(|base| worker_gateway_url(&base, "/gateway/mlflow/v1/chat/completions"));
    let embedding_url = std::env::var("MLFLOW_GATEWAY_URI")
        .ok()
        .map(|base| worker_gateway_url(&base, "/gateway/openai/v1/embeddings"));
    serialized
        .iter()
        .map(|value| {
            Ok(NamedScorer {
                scorer: SerializedScorer::from_json(value)?,
                gateway_url: gateway_url.clone(),
                embedding_url: embedding_url.clone(),
            })
        })
        .collect()
}

fn ensure_all_traces(requested: &[String], traces: &[TraceRecord]) -> Result<(), EngineError> {
    let found = traces
        .iter()
        .map(|trace| trace.trace_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    let missing = requested
        .iter()
        .filter(|trace_id| !found.contains(trace_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(EngineError::Store(format!("Traces not found: {missing:?}")))
    }
}

fn ordered_traces(requested: &[String], traces: Vec<TraceRecord>) -> Vec<TraceRecord> {
    let mut traces = traces
        .into_iter()
        .map(|trace| (trace.trace_id.clone(), trace))
        .collect::<std::collections::HashMap<_, _>>();
    requested
        .iter()
        .filter_map(|trace_id| traces.remove(trace_id))
        .collect()
}

fn session_groups(traces: &[TraceRecord]) -> Vec<(String, Vec<TraceRecord>)> {
    let mut groups: Vec<(String, Vec<TraceRecord>)> = Vec::new();
    for trace in traces {
        let Some(session_id) = trace.session_id() else {
            continue;
        };
        if let Some((_, grouped)) = groups.iter_mut().find(|(id, _)| id == session_id) {
            grouped.push(trace.clone());
        } else {
            groups.push((session_id.to_string(), vec![trace.clone()]));
        }
    }
    groups
}

async fn add_evaluator_traces(
    client: &TrackingClient,
    experiment_id: &str,
    run_id: Option<&str>,
    _input_trace: &TraceRecord,
    scorers: &[NamedScorer],
    assessments: &mut [crate::CanonicalAssessment],
    enabled: bool,
) {
    if !enabled {
        return;
    }
    for scorer in scorers {
        let Ok(trace_id) = client
            .create_evaluator_trace(experiment_id, run_id, scorer.name())
            .await
        else {
            continue;
        };
        let mut matched = false;
        for assessment in assessments.iter_mut().filter(|assessment| {
            assessment.name == scorer.name()
                || assessment.name.rsplit('/').next() == Some(scorer.name())
        }) {
            assessment.metadata.insert(
                "mlflow.assessment.scorerTraceId".to_string(),
                Value::String(trace_id.clone()),
            );
            matched = true;
        }
        if !matched && scorers.len() == 1 {
            set_scorer_trace_metadata(assessments, &trace_id);
        }
    }
}

fn scorer_tracing() -> bool {
    std::env::var("MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING")
        .is_ok_and(|value| value.eq_ignore_ascii_case("true") || value == "1")
}

fn env_workers(name: &str, default: usize) -> Result<usize, EngineError> {
    let value = std::env::var(name)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?
        .unwrap_or(default);
    if value == 0 {
        return Err(EngineError::InvalidParams(format!(
            "{name} must be greater than zero."
        )));
    }
    Ok(value)
}

fn worker_gateway_url(base: &str, path: &str) -> String {
    if base.ends_with(path) {
        base.to_string()
    } else {
        format!("{}{path}", base.trim_end_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(id: &str, timestamp: i64, session: Option<&str>) -> TraceRecord {
        TraceRecord {
            trace_id: id.to_string(),
            experiment_id: "0".to_string(),
            timestamp_ms: timestamp,
            metadata: session
                .map(|session| {
                    std::collections::BTreeMap::from([(
                        TRACE_SESSION.to_string(),
                        session.to_string(),
                    )])
                })
                .unwrap_or_default(),
            assessments: Vec::new(),
            root_span_id: None,
            eval_item: Default::default(),
        }
    }

    #[test]
    fn session_group_order_is_first_submission_order_and_turn_order_is_preserved() {
        let traces = vec![
            trace("b2", 20, Some("b")),
            trace("a2", 20, Some("a")),
            trace("b1", 10, Some("b")),
            trace("none", 1, None),
        ];
        let groups = session_groups(&traces);
        assert_eq!(groups[0].0, "b");
        assert_eq!(groups[1].0, "a");
        assert_eq!(
            groups[0]
                .1
                .iter()
                .map(|trace| trace.trace_id.as_str())
                .collect::<Vec<_>>(),
            vec!["b2", "b1"]
        );
    }
}
