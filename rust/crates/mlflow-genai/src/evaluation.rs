use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::{AssessmentSource, EngineError, EvalItem, Feedback, ScorerExecutor, SerializedScorer};

pub const AUTO_INITIAL_RPS: f64 = 10.0;
const AIMD_UPPER_MULTIPLIER: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateConfig {
    pub requests_per_second: Option<f64>,
    pub adaptive: bool,
}

pub fn parse_rate_limit(raw: Option<&str>) -> Result<RateConfig, EngineError> {
    let Some(raw) = raw else {
        return Ok(RateConfig {
            requests_per_second: None,
            adaptive: false,
        });
    };
    if raw.trim().eq_ignore_ascii_case("auto") {
        return Ok(RateConfig {
            requests_per_second: Some(AUTO_INITIAL_RPS),
            adaptive: true,
        });
    }
    let rate = raw
        .parse::<f64>()
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    Ok(RateConfig {
        requests_per_second: (rate > 0.0).then_some(rate),
        adaptive: false,
    })
}

fn pool_size(rps: Option<f64>, multiplier: f64) -> usize {
    let Some(rps) = rps else {
        return 10;
    };
    ((rps * multiplier * 2.0) as usize).clamp(10, 500)
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationConfig {
    pub row_workers: usize,
    pub scorer_workers: usize,
    pub max_retries: usize,
    pub scorer_rate: RateConfig,
    pub enable_scorer_tracing: bool,
}

impl EvaluationConfig {
    pub fn from_env(num_scorers: usize) -> Result<Self, EngineError> {
        let predict = parse_rate_limit(Some(
            std::env::var("MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT")
                .as_deref()
                .unwrap_or("auto"),
        ))?;
        let scorer_rate = match std::env::var("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT") {
            Ok(raw) => parse_rate_limit(Some(&raw))?,
            Err(_) => RateConfig {
                requests_per_second: predict.requests_per_second.map(|rate| {
                    if num_scorers == 0 {
                        rate
                    } else {
                        rate * num_scorers as f64
                    }
                }),
                adaptive: predict.adaptive,
            },
        };
        let row_workers =
            env_positive_usize("MLFLOW_GENAI_EVAL_MAX_WORKERS")?.unwrap_or_else(|| {
                pool_size(
                    scorer_rate.requests_per_second,
                    if predict.adaptive {
                        AIMD_UPPER_MULTIPLIER
                    } else {
                        1.0
                    },
                )
            });
        let scorer_workers = env_positive_usize("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS")?
            .unwrap_or(10)
            .min(num_scorers.max(1));
        let max_retries = env_usize("MLFLOW_GENAI_EVAL_MAX_RETRIES")?.unwrap_or(3);
        let enable_scorer_tracing = std::env::var("MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING")
            .is_ok_and(|value| value.eq_ignore_ascii_case("true") || value == "1");
        Ok(Self {
            row_workers,
            scorer_workers,
            max_retries,
            scorer_rate,
            enable_scorer_tracing,
        })
    }
}

fn env_usize(name: &str) -> Result<Option<usize>, EngineError> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| EngineError::InvalidParams(format!("{name}: {error}")))
        })
        .transpose()
}

fn env_positive_usize(name: &str) -> Result<Option<usize>, EngineError> {
    let value = env_usize(name)?;
    if value == Some(0) {
        return Err(EngineError::InvalidParams(format!(
            "{name} must be greater than zero."
        )));
    }
    Ok(value)
}

#[derive(Debug)]
struct LimiterState {
    rps: f64,
    max_tokens: f64,
    tokens: f64,
    last_refill: Instant,
    last_throttle: Option<Instant>,
}

#[derive(Debug, Clone)]
pub struct RateLimiter {
    state: Option<Arc<Mutex<LimiterState>>>,
    initial_rps: f64,
    adaptive: bool,
    max_rps: f64,
}

impl RateLimiter {
    pub fn new(config: RateConfig) -> Result<Self, EngineError> {
        let Some(rps) = config.requests_per_second else {
            return Ok(Self {
                state: None,
                initial_rps: 0.0,
                adaptive: false,
                max_rps: 0.0,
            });
        };
        if rps <= 0.0 || !rps.is_finite() {
            return Err(EngineError::InvalidParams(
                "requests_per_second must be positive".to_string(),
            ));
        }
        Ok(Self {
            state: Some(Arc::new(Mutex::new(LimiterState {
                rps,
                max_tokens: rps.max(1.0),
                tokens: rps,
                last_refill: Instant::now(),
                last_throttle: None,
            }))),
            initial_rps: rps,
            adaptive: config.adaptive,
            max_rps: AIMD_UPPER_MULTIPLIER * rps,
        })
    }

    pub async fn acquire(&self) {
        let Some(state) = &self.state else {
            return;
        };
        loop {
            let wait = {
                let mut state = state.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_refill).as_secs_f64();
                state.tokens = (state.tokens + elapsed * state.rps).min(state.max_tokens);
                state.last_refill = now;
                if state.tokens >= 1.0 - 1e-9 {
                    state.tokens -= 1.0;
                    None
                } else {
                    Some(Duration::from_secs_f64((1.0 - state.tokens) / state.rps))
                }
            };
            match wait {
                Some(wait) => tokio::time::sleep(wait).await,
                None => return,
            }
        }
    }

    pub async fn report_throttle(&self) {
        if !self.adaptive {
            return;
        }
        let Some(state) = &self.state else {
            return;
        };
        let mut state = state.lock().await;
        let now = Instant::now();
        if state
            .last_throttle
            .is_some_and(|last| now.duration_since(last) < Duration::from_secs(5))
        {
            return;
        }
        state.last_throttle = Some(now);
        state.rps = (state.rps * 0.5).max(1.0);
        state.max_tokens = state.rps;
        state.tokens = state.tokens.min(state.max_tokens);
    }

    pub async fn report_success(&self) {
        if !self.adaptive {
            return;
        }
        let Some(state) = &self.state else {
            return;
        };
        let mut state = state.lock().await;
        state.rps = (state.rps + 1.0 / state.rps).min(self.max_rps);
        state.max_tokens = state.rps;
    }

    pub async fn current_rps(&self) -> Option<f64> {
        match &self.state {
            Some(state) => Some(state.lock().await.rps),
            None => None,
        }
    }

    pub fn initial_rps(&self) -> Option<f64> {
        (self.initial_rps > 0.0).then_some(self.initial_rps)
    }
}

pub fn is_rate_limit_error(error: &EngineError) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("429") || message.contains("rate limit")
}

async fn call_with_retry<F, Fut, T>(
    mut call: F,
    limiter: &RateLimiter,
    max_retries: usize,
) -> Result<T, EngineError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, EngineError>>,
{
    for attempt in 0..=max_retries {
        limiter.acquire().await;
        match call().await {
            Ok(result) => {
                limiter.report_success().await;
                return Ok(result);
            }
            Err(error) if is_rate_limit_error(&error) && attempt < max_retries => {
                limiter.report_throttle().await;
                tokio::time::sleep(Duration::from_secs((1_u64 << attempt.min(6)).min(60))).await;
            }
            Err(error) => {
                if is_rate_limit_error(&error) {
                    limiter.report_throttle().await;
                }
                return Err(error);
            }
        }
    }
    unreachable!("retry loop always returns")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScorerAssessmentError {
    pub error_code: String,
    pub error_message: String,
    pub stack_trace: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalAssessment {
    pub name: String,
    pub value: Option<Value>,
    pub rationale: Option<String>,
    pub source: AssessmentSource,
    pub metadata: BTreeMap<String, Value>,
    pub span_id: Option<String>,
    pub error: Option<ScorerAssessmentError>,
    pub create_time_ms: i64,
    pub last_update_time_ms: i64,
}

impl CanonicalAssessment {
    fn from_feedback(feedback: Feedback) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            name: feedback.name,
            value: Some(feedback.value),
            rationale: Some(feedback.rationale),
            source: feedback.source.unwrap_or(AssessmentSource {
                source_type: "CODE".to_string(),
                source_id: Some("default".to_string()),
            }),
            metadata: feedback.metadata.unwrap_or_default(),
            span_id: feedback.span_id,
            error: None,
            create_time_ms: now,
            last_update_time_ms: now,
        }
    }

    fn scorer_error(name: &str, error: &EngineError) -> Self {
        let message = error.to_string();
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            name: name.to_string(),
            value: None,
            rationale: None,
            source: AssessmentSource {
                source_type: "CODE".to_string(),
                source_id: Some(name.to_string()),
            },
            metadata: BTreeMap::new(),
            span_id: None,
            error: Some(ScorerAssessmentError {
                error_code: "SCORER_ERROR".to_string(),
                error_message: message.clone(),
                stack_trace: format!("Traceback (most recent call last):\n{message}"),
            }),
            create_time_ms: now,
            last_update_time_ms: now,
        }
    }
}

/// Compatibility standardization for raw decorator/third-party return values.
/// Native T19.1 executors already return `Feedback`; this seam remains public
/// for T19.3 scorer families and mirrors Python's primitive/list behavior.
pub fn standardize_scorer_value(
    scorer_name: &str,
    value: Value,
) -> Result<Vec<CanonicalAssessment>, EngineError> {
    if value.is_null() {
        return Ok(Vec::new());
    }
    if value.is_boolean() || value.is_number() || value.is_string() {
        return Ok(vec![primitive_assessment(scorer_name, value)]);
    }
    if let Value::Array(values) = value {
        return Ok(values
            .into_iter()
            .map(|value| primitive_assessment(scorer_name, value))
            .collect());
    }
    // Python accepts every `Collection`; iterating a dictionary produces its
    // keys, and each non-Feedback item becomes one feedback value.
    if let Value::Object(values) = value {
        return Ok(values
            .into_iter()
            .map(|(value, _)| primitive_assessment(scorer_name, Value::String(value)))
            .collect());
    }
    Err(EngineError::InvalidParams(format!(
        "Got unsupported result from scorer '{scorer_name}'. Expected the metric value to be a number, or a boolean, or a string, or an Feedback, or a list of Feedbacks. Got {value}."
    )))
}

fn primitive_assessment(scorer_name: &str, value: Value) -> CanonicalAssessment {
    let now = chrono::Utc::now().timestamp_millis();
    CanonicalAssessment {
        name: scorer_name.to_string(),
        value: Some(value),
        rationale: None,
        source: AssessmentSource {
            source_type: "CODE".to_string(),
            source_id: Some(scorer_name.to_string()),
        },
        metadata: BTreeMap::new(),
        span_id: None,
        error: None,
        create_time_ms: now,
        last_update_time_ms: now,
    }
}

#[derive(Debug, Clone)]
pub struct NamedScorer {
    pub scorer: SerializedScorer,
    pub gateway_url: Option<String>,
    pub embedding_url: Option<String>,
}

impl NamedScorer {
    pub fn name(&self) -> &str {
        &self.scorer.common().name
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredItem {
    pub assessments: Vec<CanonicalAssessment>,
    pub failures: BTreeMap<String, usize>,
}

#[derive(Clone)]
pub struct EvaluationEngine {
    executor: ScorerExecutor,
    config: EvaluationConfig,
    limiter: RateLimiter,
}

impl EvaluationEngine {
    pub fn new(config: EvaluationConfig) -> Result<Self, EngineError> {
        let limiter = RateLimiter::new(config.scorer_rate)?;
        Ok(Self {
            executor: ScorerExecutor::new(),
            config,
            limiter,
        })
    }

    pub fn config(&self) -> &EvaluationConfig {
        &self.config
    }

    /// Session-level Python evaluation uses one worker per session scorer,
    /// while retaining the shared rate limiter used by single-turn scoring.
    pub fn with_scorer_workers(&self, scorer_workers: usize) -> Self {
        let mut engine = self.clone();
        engine.config.scorer_workers = scorer_workers.max(1);
        engine
    }

    pub async fn score_item(&self, item: &EvalItem, scorers: &[NamedScorer]) -> ScoredItem {
        let mut results = stream::iter(scorers.iter().cloned().enumerate())
            .map(|(index, scorer)| {
                let executor = self.executor.clone();
                let limiter = self.limiter.clone();
                let item = item.clone();
                let max_retries = self.config.max_retries;
                async move {
                    let result = call_with_retry(
                        || {
                            executor.execute_all(
                                &scorer.scorer,
                                &item,
                                scorer.gateway_url.as_deref(),
                                scorer.embedding_url.as_deref(),
                            )
                        },
                        &limiter,
                        max_retries,
                    )
                    .await;
                    (index, scorer.name().to_string(), result)
                }
            })
            .buffer_unordered(self.config.scorer_workers)
            .collect::<Vec<_>>()
            .await;
        results.sort_by_key(|(index, _, _)| *index);

        let mut assessments = Vec::new();
        let mut failures = BTreeMap::new();
        for (_, name, result) in results {
            match result {
                Ok(feedbacks) => assessments.extend(
                    feedbacks
                        .into_iter()
                        .map(CanonicalAssessment::from_feedback),
                ),
                Err(error) => {
                    *failures.entry(name.clone()).or_default() += 1;
                    assessments.push(CanonicalAssessment::scorer_error(&name, &error));
                }
            }
        }
        ScoredItem {
            assessments,
            failures,
        }
    }

    pub async fn score_items(
        &self,
        items: Vec<EvalItem>,
        scorers: Arc<Vec<NamedScorer>>,
    ) -> Vec<ScoredItem> {
        let mut results = stream::iter(items.into_iter().enumerate())
            .map(|(index, item)| {
                let engine = self.clone();
                let scorers = scorers.clone();
                async move { (index, engine.score_item(&item, &scorers).await) }
            })
            .buffer_unordered(self.config.row_workers)
            .collect::<Vec<_>>()
            .await;
        results.sort_by_key(|(index, _)| *index);
        results.into_iter().map(|(_, result)| result).collect()
    }
}

pub fn compute_aggregated_metrics(
    assessments: &[CanonicalAssessment],
    scorers: &[NamedScorer],
) -> BTreeMap<String, f64> {
    let aggregations = scorers
        .iter()
        .map(|scorer| {
            (
                scorer.name().to_string(),
                scorer
                    .scorer
                    .common()
                    .aggregations
                    .clone()
                    .unwrap_or_else(|| vec!["mean".to_string()]),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for assessment in assessments {
        if let Some(value) = assessment.value.as_ref().and_then(assessment_float) {
            values
                .entry(assessment.name.clone())
                .or_default()
                .push(value);
        }
    }
    let mut metrics = BTreeMap::new();
    for (name, mut values) in values {
        if values.is_empty() {
            continue;
        }
        values.sort_by(f64::total_cmp);
        let scorer_name = name
            .split_once('/')
            .map_or(name.as_str(), |(_, suffix)| suffix);
        let requested = aggregations
            .get(scorer_name)
            .cloned()
            .unwrap_or_else(|| vec!["mean".to_string()]);
        for aggregation in requested {
            let value = match aggregation.as_str() {
                "min" => values[0],
                "max" => *values.last().expect("values is non-empty"),
                "mean" => values.iter().sum::<f64>() / values.len() as f64,
                "median" => percentile(&values, 0.5),
                "variance" => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f64>()
                        / values.len() as f64
                }
                "p90" => percentile(&values, 0.9),
                _ => continue,
            };
            metrics.insert(format!("{name}/{aggregation}"), value);
        }
    }
    metrics
}

fn assessment_float(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(value) => Some(u8::from(*value) as f64),
        Value::Number(value) => value.as_f64(),
        Value::String(value) if value.eq_ignore_ascii_case("yes") => Some(1.0),
        Value::String(value) if value.eq_ignore_ascii_case("no") => Some(0.0),
        _ => None,
    }
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let index = (values.len() - 1) as f64 * quantile;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    values[lower] + (values[upper] - values[lower]) * index.fract()
}

pub fn scorer_error_shape(name: &str, message: &str) -> Value {
    json!({
        "name": name,
        "source": {"source_type": "CODE", "source_id": name},
        "feedback": {"error": {
            "error_code": "SCORER_ERROR",
            "error_message": message,
            "stack_trace": format!("Traceback (most recent call last):\n{message}"),
        }}
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::extract::State;
    use axum::routing::post;
    use axum::{Json, Router};

    use super::*;

    #[derive(Clone)]
    struct ConcurrencyState {
        active: Arc<AtomicUsize>,
        maximum: Arc<AtomicUsize>,
    }

    async fn delayed_judge(
        State(state): State<ConcurrencyState>,
        Json(_request): Json<Value>,
    ) -> Json<Value> {
        let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
        state.maximum.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(25)).await;
        state.active.fetch_sub(1, Ordering::SeqCst);
        Json(json!({
            "choices": [{"message": {"role": "assistant", "content":
                "{\"result\":\"yes\",\"rationale\":\"scripted\"}"}}]
        }))
    }

    async fn concurrency_server() -> (String, ConcurrencyState) {
        let state = ConcurrencyState {
            active: Arc::new(AtomicUsize::new(0)),
            maximum: Arc::new(AtomicUsize::new(0)),
        };
        let app = Router::new()
            .route("/", post(delayed_judge))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}/"), state)
    }

    fn instruction_scorer(name: &str, gateway_url: &str) -> NamedScorer {
        NamedScorer {
            scorer: SerializedScorer::from_value(json!({
                "name": name,
                "instructions_judge_pydantic_data": {
                    "instructions": "Judge {{ outputs }}.",
                    "model": "openai:/fake-model",
                    "feedback_value_type": {"type": "string"}
                }
            }))
            .unwrap(),
            gateway_url: Some(gateway_url.to_string()),
            embedding_url: None,
        }
    }

    #[test]
    fn rate_config_and_pool_rules_match_python() {
        assert_eq!(
            parse_rate_limit(Some(" auto ")).unwrap(),
            RateConfig {
                requests_per_second: Some(10.0),
                adaptive: true
            }
        );
        assert_eq!(
            parse_rate_limit(Some("0")).unwrap().requests_per_second,
            None
        );
        assert_eq!(pool_size(None, 2.0), 10);
        assert_eq!(pool_size(Some(200.0), 2.0), 500);
    }

    #[test]
    fn standardization_edges_match_primitive_and_list_contract() {
        assert!(standardize_scorer_value("s", Value::Null)
            .unwrap()
            .is_empty());
        assert_eq!(
            standardize_scorer_value("s", json!(true)).unwrap()[0].value,
            Some(json!(true))
        );
        let list = standardize_scorer_value("s", json!([1, ["yes", "no"], false])).unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[1].value, Some(json!(["yes", "no"])));
        assert_eq!(
            standardize_scorer_value("s", json!({"bad": true})).unwrap()[0].value,
            Some(json!("bad"))
        );
    }

    #[test]
    fn aggregates_use_numpy_population_and_linear_percentile_semantics() {
        let scorer = NamedScorer {
            scorer: SerializedScorer::from_value(json!({
                "name": "quality",
                "aggregations": ["mean", "min", "max", "median", "variance", "p90"],
                "builtin_scorer_class": "ResponseLength",
                "builtin_scorer_pydantic_data": {"max_length": 10}
            }))
            .unwrap(),
            gateway_url: None,
            embedding_url: None,
        };
        let assessments = [0.7, 0.5, 0.6]
            .into_iter()
            .map(|value| CanonicalAssessment {
                name: "quality".to_string(),
                value: Some(json!(value)),
                rationale: None,
                source: AssessmentSource {
                    source_type: "CODE".to_string(),
                    source_id: Some("quality".to_string()),
                },
                metadata: BTreeMap::new(),
                span_id: None,
                error: None,
                create_time_ms: 0,
                last_update_time_ms: 0,
            })
            .collect::<Vec<_>>();
        let metrics = compute_aggregated_metrics(&assessments, &[scorer]);
        assert!((metrics["quality/variance"] - 0.006666666666666665).abs() < 1e-12);
        assert!((metrics["quality/p90"] - 0.68).abs() < 1e-12);
    }

    #[tokio::test]
    async fn adaptive_limiter_halves_and_recovers() {
        let limiter = RateLimiter::new(RateConfig {
            requests_per_second: Some(10.0),
            adaptive: true,
        })
        .unwrap();
        limiter.report_throttle().await;
        assert_eq!(limiter.current_rps().await, Some(5.0));
        limiter.report_success().await;
        assert_eq!(limiter.current_rps().await, Some(5.2));
    }

    #[tokio::test]
    async fn scorer_concurrency_never_exceeds_configured_bound() {
        let (gateway, state) = concurrency_server().await;
        let engine = EvaluationEngine::new(EvaluationConfig {
            row_workers: 1,
            scorer_workers: 2,
            max_retries: 0,
            scorer_rate: RateConfig {
                requests_per_second: None,
                adaptive: false,
            },
            enable_scorer_tracing: false,
        })
        .unwrap();
        let scorers = (0..5)
            .map(|index| instruction_scorer(&format!("judge-{index}"), &gateway))
            .collect::<Vec<_>>();
        let item = EvalItem {
            outputs: Some(json!("answer")),
            ..Default::default()
        };
        let result = engine.score_item(&item, &scorers).await;
        assert_eq!(result.assessments.len(), 5);
        assert_eq!(state.maximum.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_policy_retries_only_rate_limits_and_counts_retries() {
        let limiter = RateLimiter::new(RateConfig {
            requests_per_second: None,
            adaptive: false,
        })
        .unwrap();
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = call_with_retry(
            || {
                let attempts = attempts.clone();
                async move {
                    if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        Err(EngineError::Gateway("HTTP 429".to_string()))
                    } else {
                        Ok("success")
                    }
                }
            },
            &limiter,
            1,
        )
        .await;
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);

        let attempts = Arc::new(AtomicUsize::new(0));
        let result = call_with_retry(
            || {
                let attempts = attempts.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>(EngineError::Gateway("HTTP 500".to_string()))
                }
            },
            &limiter,
            3,
        )
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retry_classification_is_429_only() {
        assert!(is_rate_limit_error(&EngineError::Gateway(
            "HTTP 429 Too Many Requests".to_string()
        )));
        assert!(is_rate_limit_error(&EngineError::Gateway(
            "rate limit exceeded".to_string()
        )));
        assert!(!is_rate_limit_error(&EngineError::Gateway(
            "HTTP 500".to_string()
        )));
    }
}
