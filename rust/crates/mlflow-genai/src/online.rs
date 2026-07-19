use futures::stream::{self, StreamExt};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

use crate::store::{
    trace_info_id, trace_info_metadata, trace_info_timestamp, TraceRecord, TrackingClient,
};
use crate::{
    EngineError, EvaluationConfig, EvaluationEngine, NamedScorer, RateConfig, SerializedScorer,
    WorkerRequest,
};

pub(crate) const TRACE_CHECKPOINT_TAG: &str = "mlflow.latestOnlineScoring.trace.checkpoint";
pub(crate) const SESSION_CHECKPOINT_TAG: &str = "mlflow.latestOnlineScoring.session.checkpoint";
const ONLINE_SESSION_ID: &str = "mlflow.assessment.onlineScoringSessionId";
const TRACE_SESSION: &str = "mlflow.trace.session";
const MAX_LOOKBACK_MS: i64 = 60 * 60 * 1000;
const MAX_TRACES_PER_JOB: usize = 500;
const MAX_SESSIONS_PER_JOB: usize = 100;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OnlineJobParams {
    pub experiment_id: String,
    pub online_scorers: Vec<OnlineScorerPayload>,
    #[serde(default)]
    pub current_time_ms: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OnlineScorerPayload {
    pub serialized_scorer: String,
    pub online_config: OnlineConfigPayload,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OnlineConfigPayload {
    pub sample_rate: f64,
    #[serde(default)]
    pub filter_string: Option<String>,
}

#[derive(Debug, Clone)]
struct ConfiguredScorer {
    named: NamedScorer,
    rate: f64,
    filter: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct TraceCheckpoint {
    timestamp_ms: i64,
    trace_id: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct SessionCheckpoint {
    timestamp_ms: i64,
    session_id: Option<String>,
}

pub(crate) async fn run_online_trace_job(
    request: &WorkerRequest,
    params: OnlineJobParams,
) -> Result<Value, EngineError> {
    let client = TrackingClient::from_request(request)?;
    let scorers = configured_scorers(&params.online_scorers, false)?;
    let now = params
        .current_time_ms
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
    let checkpoint = read_trace_checkpoint(&client, &params.experiment_id).await?;
    let minimum = checkpoint.as_ref().map_or(now - MAX_LOOKBACK_MS, |value| {
        value.timestamp_ms.max(now - MAX_LOOKBACK_MS)
    });
    let base_filter = "metadata.mlflow.sourceRun IS NULL";
    let mut tasks: BTreeMap<String, (i64, Vec<ConfiguredScorer>)> = BTreeMap::new();

    for (filter, grouped) in group_by_filter(&scorers) {
        let filter = filter.as_deref().map_or_else(
            || base_filter.to_string(),
            |filter| format!("{base_filter} AND {filter}"),
        );
        let time_filter =
            format!("trace.timestamp_ms >= {minimum} AND trace.timestamp_ms <= {now} AND {filter}");
        let mut infos = client
            .search_trace_infos(
                &params.experiment_id,
                Some(&time_filter),
                &["timestamp_ms ASC", "request_id ASC"],
            )
            .await?;
        infos.truncate(MAX_TRACES_PER_JOB);
        for info in infos {
            let Some(trace_id) = trace_info_id(&info).map(str::to_string) else {
                continue;
            };
            let timestamp = trace_info_timestamp(&info);
            if checkpoint.as_ref().is_some_and(|checkpoint| {
                checkpoint.trace_id.as_ref().is_some_and(|boundary| {
                    timestamp == checkpoint.timestamp_ms && trace_id <= *boundary
                })
            }) {
                continue;
            }
            let selected = dense_sample(&trace_id, &grouped);
            if selected.is_empty() {
                continue;
            }
            let task = tasks.entry(trace_id).or_insert((timestamp, Vec::new()));
            for scorer in selected {
                if !task
                    .1
                    .iter()
                    .any(|existing| existing.named.name() == scorer.named.name())
                {
                    task.1.push(scorer.clone());
                }
            }
        }
    }

    let mut ordered = tasks.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| (left.1 .0, &left.0).cmp(&(right.1 .0, &right.0)));
    ordered.truncate(MAX_TRACES_PER_JOB);
    if ordered.is_empty() {
        persist_trace_checkpoint(
            &client,
            &params.experiment_id,
            &TraceCheckpoint {
                timestamp_ms: now,
                trace_id: None,
            },
        )
        .await?;
        return Ok(Value::Null);
    }

    let trace_ids = ordered.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    let traces = client.fetch_traces(&trace_ids).await?;
    let trace_map = traces
        .iter()
        .cloned()
        .map(|trace| (trace.trace_id.clone(), trace))
        .collect::<HashMap<_, _>>();
    let entity_workers = online_workers()?;
    let results = stream::iter(ordered.into_iter())
        .map(|(trace_id, (_, scorers))| {
            let client = client.clone();
            let trace = trace_map.get(&trace_id).cloned();
            let experiment_id = params.experiment_id.clone();
            async move {
                let Some(trace) = trace else {
                    return Ok::<(), EngineError>(());
                };
                score_and_log(&client, &experiment_id, None, &trace, &scorers, None)
                    .await
                    .map(|_| ())
            }
        })
        .buffer_unordered(entity_workers)
        .collect::<Vec<_>>()
        .await;
    for result in results {
        // Python records the individual task failure and advances the batch;
        // one trace must not prevent the checkpoint from moving forward.
        let _ = result;
    }

    let checkpoint = traces
        .iter()
        .max_by(|left, right| {
            (left.timestamp_ms, &left.trace_id).cmp(&(right.timestamp_ms, &right.trace_id))
        })
        .map_or(
            TraceCheckpoint {
                timestamp_ms: now,
                trace_id: None,
            },
            |trace| TraceCheckpoint {
                timestamp_ms: trace.timestamp_ms,
                trace_id: Some(trace.trace_id.clone()),
            },
        );
    persist_trace_checkpoint(&client, &params.experiment_id, &checkpoint).await?;
    Ok(Value::Null)
}

pub(crate) async fn run_online_session_job(
    request: &WorkerRequest,
    params: OnlineJobParams,
) -> Result<Value, EngineError> {
    let client = TrackingClient::from_request(request)?;
    let scorers = configured_scorers(&params.online_scorers, true)?;
    if params.online_scorers.is_empty() {
        return Ok(Value::Null);
    }
    let now = params
        .current_time_ms
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
    let buffer_seconds =
        std::env::var("MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS")
            .ok()
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(300)
            .max(0);
    let maximum = now - buffer_seconds * 1000;
    let checkpoint = read_session_checkpoint(&client, &params.experiment_id).await?;
    let minimum = checkpoint.as_ref().map_or(now - MAX_LOOKBACK_MS, |value| {
        value.timestamp_ms.max(now - MAX_LOOKBACK_MS)
    });
    let mut tasks: BTreeMap<String, (i64, i64, Vec<ConfiguredScorer>)> = BTreeMap::new();

    for (filter, grouped) in group_by_filter(&scorers) {
        let candidates = completed_sessions(
            &client,
            &params.experiment_id,
            minimum,
            maximum,
            filter.as_deref(),
        )
        .await?;
        for (session_id, first, last) in candidates {
            if checkpoint.as_ref().is_some_and(|checkpoint| {
                checkpoint.session_id.as_ref().is_some_and(|boundary| {
                    last == checkpoint.timestamp_ms && session_id <= *boundary
                })
            }) {
                continue;
            }
            let selected = dense_sample(&session_id, &grouped);
            if selected.is_empty() {
                continue;
            }
            let task = tasks.entry(session_id).or_insert((first, last, Vec::new()));
            for scorer in selected {
                if !task
                    .2
                    .iter()
                    .any(|existing| existing.named.name() == scorer.named.name())
                {
                    task.2.push(scorer.clone());
                }
            }
        }
    }

    let mut ordered = tasks.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| (left.1 .1, &left.0).cmp(&(right.1 .1, &right.0)));
    ordered.truncate(MAX_SESSIONS_PER_JOB);
    if ordered.is_empty() {
        persist_session_checkpoint(
            &client,
            &params.experiment_id,
            &SessionCheckpoint {
                timestamp_ms: maximum,
                session_id: None,
            },
        )
        .await?;
        return Ok(Value::Null);
    }

    let entity_workers = online_workers()?;
    let results = stream::iter(ordered.iter().cloned())
        .map(|(session_id, (_, _, scorers))| {
            let client = client.clone();
            let experiment_id = params.experiment_id.clone();
            async move { score_session(&client, &experiment_id, &session_id, &scorers).await }
        })
        .buffer_unordered(entity_workers)
        .collect::<Vec<_>>()
        .await;
    for result in results {
        // Python logs and continues when one session fails.
        let _ = result;
    }

    let (session_id, (_, timestamp_ms, _)) = ordered.last().expect("ordered is non-empty");
    persist_session_checkpoint(
        &client,
        &params.experiment_id,
        &SessionCheckpoint {
            timestamp_ms: *timestamp_ms,
            session_id: Some(session_id.clone()),
        },
    )
    .await?;
    Ok(Value::Null)
}

async fn score_session(
    client: &TrackingClient,
    experiment_id: &str,
    session_id: &str,
    scorers: &[ConfiguredScorer],
) -> Result<(), EngineError> {
    let escaped = session_id.replace('\'', "\\'");
    let filter =
        format!("metadata.mlflow.sourceRun IS NULL AND metadata.`{TRACE_SESSION}` = '{escaped}'");
    let infos = client
        .search_trace_infos(
            experiment_id,
            Some(&filter),
            &["timestamp_ms ASC", "request_id ASC"],
        )
        .await?;
    let ids = infos
        .iter()
        .filter_map(trace_info_id)
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut traces = client.fetch_traces(&ids).await?;
    traces.sort_by(|left, right| {
        (left.timestamp_ms, &left.trace_id).cmp(&(right.timestamp_ms, &right.trace_id))
    });
    let Some(first) = traces.first().cloned() else {
        return Ok(());
    };
    let session = traces
        .iter()
        .filter_map(|trace| trace.eval_item.trace.clone())
        .collect::<Vec<_>>();
    let mut trace = first;
    trace.eval_item.session = Some(session);
    let mut assessments = score_and_log(
        client,
        experiment_id,
        None,
        &trace,
        scorers,
        Some(session_id),
    )
    .await?;
    let names = assessments
        .iter()
        .map(|assessment| assessment.name.clone())
        .collect::<std::collections::HashSet<_>>();
    for old in &trace.assessments {
        let old_session = old
            .pointer(&format!("/metadata/{ONLINE_SESSION_ID}"))
            .and_then(Value::as_str);
        let old_name = old.get("assessment_name").and_then(Value::as_str);
        let old_id = old.get("assessment_id").and_then(Value::as_str);
        if old_session == Some(session_id) && old_name.is_some_and(|name| names.contains(name)) {
            if let Some(old_id) = old_id {
                client.delete_assessment(&trace.trace_id, old_id).await?;
            }
        }
    }
    assessments.clear();
    Ok(())
}

async fn score_and_log(
    client: &TrackingClient,
    experiment_id: &str,
    run_id: Option<&str>,
    trace: &TraceRecord,
    scorers: &[ConfiguredScorer],
    online_session_id: Option<&str>,
) -> Result<Vec<crate::CanonicalAssessment>, EngineError> {
    let named = scorers
        .iter()
        .map(|scorer| scorer.named.clone())
        .collect::<Vec<_>>();
    let config = EvaluationConfig {
        row_workers: 1,
        scorer_workers: named.len().clamp(1, 10),
        max_retries: 0,
        scorer_rate: RateConfig {
            requests_per_second: None,
            adaptive: false,
        },
        enable_scorer_tracing: online_session_id.is_none()
            && std::env::var("MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING")
                .is_ok_and(|value| value.eq_ignore_ascii_case("true") || value == "1"),
    };
    let engine = EvaluationEngine::new(config.clone())?;
    let mut result = engine.score_item(&trace.eval_item, &named).await;
    if let Some(session_id) = online_session_id {
        for assessment in &mut result.assessments {
            assessment.metadata.insert(
                TRACE_SESSION.to_string(),
                Value::String(session_id.to_string()),
            );
            assessment.metadata.insert(
                ONLINE_SESSION_ID.to_string(),
                Value::String(session_id.to_string()),
            );
        }
    }
    if config.enable_scorer_tracing {
        for scorer in &named {
            let scorer_trace = client
                .create_evaluator_trace(experiment_id, run_id, scorer.name())
                .await?;
            let matching = result
                .assessments
                .iter_mut()
                .filter(|assessment| assessment.name == scorer.name())
                .collect::<Vec<_>>();
            for assessment in matching {
                assessment.metadata.insert(
                    "mlflow.assessment.scorerTraceId".to_string(),
                    Value::String(scorer_trace.clone()),
                );
            }
        }
    }
    for assessment in &result.assessments {
        client.log_assessment(trace, assessment, run_id).await?;
    }
    Ok(result.assessments)
}

fn configured_scorers(
    values: &[OnlineScorerPayload],
    session_level: bool,
) -> Result<Vec<ConfiguredScorer>, EngineError> {
    let gateway_url = std::env::var("MLFLOW_GATEWAY_URI")
        .ok()
        .map(|base| worker_gateway_url(&base, "/gateway/mlflow/v1/chat/completions"));
    let embedding_url = std::env::var("MLFLOW_GATEWAY_URI")
        .ok()
        .map(|base| worker_gateway_url(&base, "/gateway/openai/v1/embeddings"));
    let mut scorers = Vec::new();
    for value in values {
        let scorer = match SerializedScorer::from_json(&value.serialized_scorer) {
            Ok(scorer) if scorer.common().is_session_level_scorer == session_level => scorer,
            Ok(_) | Err(_) => continue,
        };
        scorers.push(ConfiguredScorer {
            named: NamedScorer {
                scorer,
                gateway_url: gateway_url.clone(),
                embedding_url: embedding_url.clone(),
            },
            rate: value.online_config.sample_rate,
            filter: value.online_config.filter_string.clone(),
        });
    }
    Ok(scorers)
}

fn group_by_filter(scorers: &[ConfiguredScorer]) -> Vec<(Option<String>, Vec<ConfiguredScorer>)> {
    let mut groups: Vec<(Option<String>, Vec<ConfiguredScorer>)> = Vec::new();
    for scorer in scorers {
        if let Some((_, values)) = groups
            .iter_mut()
            .find(|(filter, _)| filter == &scorer.filter)
        {
            values.push(scorer.clone());
        } else {
            groups.push((scorer.filter.clone(), vec![scorer.clone()]));
        }
    }
    groups
}

fn dense_sample<'a>(entity_id: &str, scorers: &'a [ConfiguredScorer]) -> Vec<&'a ConfiguredScorer> {
    let mut sorted = scorers.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| right.rate.total_cmp(&left.rate));
    let mut selected = Vec::new();
    let mut previous = 1.0;
    for scorer in sorted {
        let conditional = if previous > 0.0 {
            scorer.rate / previous
        } else {
            0.0
        };
        let digest = Sha256::digest(format!("{entity_id}:{}", scorer.named.name()).as_bytes());
        let hash = digest
            .iter()
            .fold(0.0, |value, byte| value * 256.0 + f64::from(*byte))
            / 2_f64.powi(256);
        if hash > conditional {
            break;
        }
        selected.push(scorer);
        previous = scorer.rate;
    }
    selected
}

async fn completed_sessions(
    client: &TrackingClient,
    experiment_id: &str,
    minimum: i64,
    maximum: i64,
    first_trace_filter: Option<&str>,
) -> Result<Vec<(String, i64, i64)>, EngineError> {
    let recent_filter =
        format!("trace.timestamp_ms >= {minimum} AND metadata.mlflow.sourceRun IS NULL");
    let recent = client
        .search_trace_infos(
            experiment_id,
            Some(&recent_filter),
            &["timestamp_ms ASC", "request_id ASC"],
        )
        .await?;
    let mut candidates = recent
        .iter()
        .filter_map(|info| trace_info_metadata(info, TRACE_SESSION))
        .map(str::to_string)
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.dedup();
    let mut sessions = Vec::new();
    for session_id in candidates {
        let escaped = session_id.replace('\'', "\\'");
        let session_filter = format!(
            "metadata.mlflow.sourceRun IS NULL AND metadata.`{TRACE_SESSION}` = '{escaped}'"
        );
        let all = client
            .search_trace_infos(
                experiment_id,
                Some(&session_filter),
                &["timestamp_ms ASC", "request_id ASC"],
            )
            .await?;
        let Some(first) = all.first() else {
            continue;
        };
        let Some(last) = all.last() else {
            continue;
        };
        let first_timestamp = trace_info_timestamp(first);
        let last_timestamp = trace_info_timestamp(last);
        if last_timestamp < minimum || last_timestamp > maximum {
            continue;
        }
        if let Some(filter) = first_trace_filter {
            let filtered = client
                .search_trace_infos(
                    experiment_id,
                    Some(&format!("{session_filter} AND {filter}")),
                    &["timestamp_ms ASC", "request_id ASC"],
                )
                .await?;
            if filtered.first().and_then(trace_info_id) != trace_info_id(first) {
                continue;
            }
        }
        sessions.push((session_id, first_timestamp, last_timestamp));
    }
    sessions.sort_by(|left, right| (left.2, &left.0).cmp(&(right.2, &right.0)));
    sessions.truncate(MAX_SESSIONS_PER_JOB);
    Ok(sessions)
}

async fn read_trace_checkpoint(
    client: &TrackingClient,
    experiment_id: &str,
) -> Result<Option<TraceCheckpoint>, EngineError> {
    Ok(experiment_tag(client, experiment_id, TRACE_CHECKPOINT_TAG)
        .await?
        .and_then(|value| serde_json::from_str(&value).ok()))
}

async fn read_session_checkpoint(
    client: &TrackingClient,
    experiment_id: &str,
) -> Result<Option<SessionCheckpoint>, EngineError> {
    Ok(
        experiment_tag(client, experiment_id, SESSION_CHECKPOINT_TAG)
            .await?
            .and_then(|value| serde_json::from_str(&value).ok()),
    )
}

async fn experiment_tag(
    client: &TrackingClient,
    experiment_id: &str,
    key: &str,
) -> Result<Option<String>, EngineError> {
    let response = client.get_experiment(experiment_id).await?;
    let tags = response.pointer("/experiment/tags");
    if let Some(tags) = tags.and_then(Value::as_array) {
        return Ok(tags.iter().find_map(|tag| {
            (tag.get("key").and_then(Value::as_str) == Some(key))
                .then(|| tag.get("value").and_then(Value::as_str).map(str::to_string))
                .flatten()
        }));
    }
    Ok(tags
        .and_then(Value::as_object)
        .and_then(|tags| tags.get(key))
        .and_then(Value::as_str)
        .map(str::to_string))
}

async fn persist_trace_checkpoint(
    client: &TrackingClient,
    experiment_id: &str,
    checkpoint: &TraceCheckpoint,
) -> Result<(), EngineError> {
    client
        .set_experiment_tag(
            experiment_id,
            TRACE_CHECKPOINT_TAG,
            &format!(
                "{{\"timestamp_ms\": {}, \"trace_id\": {}}}",
                checkpoint.timestamp_ms,
                serde_json::to_string(&checkpoint.trace_id)
                    .map_err(|error| EngineError::Serialization(error.to_string()))?
            ),
        )
        .await
}

async fn persist_session_checkpoint(
    client: &TrackingClient,
    experiment_id: &str,
    checkpoint: &SessionCheckpoint,
) -> Result<(), EngineError> {
    client
        .set_experiment_tag(
            experiment_id,
            SESSION_CHECKPOINT_TAG,
            &format!(
                "{{\"timestamp_ms\": {}, \"session_id\": {}}}",
                checkpoint.timestamp_ms,
                serde_json::to_string(&checkpoint.session_id)
                    .map_err(|error| EngineError::Serialization(error.to_string()))?
            ),
        )
        .await
}

fn online_workers() -> Result<usize, EngineError> {
    let value = std::env::var("MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?
        .unwrap_or(10);
    if value == 0 {
        return Err(EngineError::InvalidParams(
            "MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS must be greater than zero.".to_string(),
        ));
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
    use serde_json::json;

    fn configured(name: &str, rate: f64) -> ConfiguredScorer {
        ConfiguredScorer {
            named: NamedScorer {
                scorer: SerializedScorer::from_value(json!({
                    "name": name,
                    "builtin_scorer_class": "ResponseLength",
                    "builtin_scorer_pydantic_data": {"max_length": 10}
                }))
                .unwrap(),
                gateway_url: None,
                embedding_url: None,
            },
            rate,
            filter: None,
        }
    }

    #[test]
    fn dense_sampling_is_stable_and_a_waterfall() {
        let scorers = vec![configured("high", 1.0), configured("low", 0.0)];
        let selected = dense_sample("trace-1", &scorers);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].named.name(), "high");
        assert_eq!(
            dense_sample("trace-1", &scorers)
                .iter()
                .map(|scorer| scorer.named.name())
                .collect::<Vec<_>>(),
            vec!["high"]
        );
    }

    #[test]
    fn checkpoint_json_matches_python_asdict_spacing_independently() {
        let checkpoint = TraceCheckpoint {
            timestamp_ms: 123,
            trace_id: Some("tr-1".to_string()),
        };
        assert_eq!(
            serde_json::to_value(&checkpoint).unwrap(),
            json!({"timestamp_ms": 123, "trace_id": "tr-1"})
        );
    }
}
