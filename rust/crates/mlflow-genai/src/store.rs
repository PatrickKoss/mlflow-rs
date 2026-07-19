use std::collections::BTreeMap;

use chrono::{DateTime, SecondsFormat, Utc};
use reqwest::{Method, RequestBuilder};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::{CanonicalAssessment, EngineError, EvalItem, ScorerAssessmentError, WorkerRequest};

const WORKSPACE_HEADER: &str = "X-MLFLOW-WORKSPACE";
const SOURCE_RUN_ID: &str = "mlflow.assessment.sourceRunId";
const SCORER_TRACE_ID: &str = "mlflow.assessment.scorerTraceId";
const SOURCE_SCORER_NAME: &str = "mlflow.trace.sourceScorer";

#[derive(Debug, Clone)]
pub(crate) struct TraceRecord {
    pub trace_id: String,
    pub experiment_id: String,
    pub timestamp_ms: i64,
    pub metadata: BTreeMap<String, String>,
    pub assessments: Vec<Value>,
    pub root_span_id: Option<String>,
    pub eval_item: EvalItem,
}

impl TraceRecord {
    pub fn session_id(&self) -> Option<&str> {
        self.metadata
            .get("mlflow.trace.session")
            .map(String::as_str)
    }
}

#[derive(Clone)]
pub(crate) struct TrackingClient {
    base: String,
    client: reqwest::Client,
    workspace: Option<String>,
    username: Option<String>,
    password: Option<String>,
}

impl TrackingClient {
    pub fn from_request(request: &WorkerRequest) -> Result<Self, EngineError> {
        let base = std::env::var("MLFLOW_TRACKING_URI").map_err(|_| {
            EngineError::Store("MLFLOW_TRACKING_URI is required for native job execution".into())
        })?;
        Ok(Self {
            base: base.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .map_err(|error| EngineError::Store(error.to_string()))?,
            workspace: request.workspace.clone(),
            username: std::env::var("MLFLOW_TRACKING_USERNAME").ok(),
            password: std::env::var("MLFLOW_TRACKING_PASSWORD").ok(),
        })
    }

    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        if let Some(workspace) = &self.workspace {
            request = request.header(WORKSPACE_HEADER, workspace);
        }
        if let Some(username) = &self.username {
            request = request.basic_auth(username, self.password.as_ref());
        }
        request
    }

    async fn send_json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, EngineError> {
        let mut request = self.request(method, path);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request
            .send()
            .await
            .map_err(|error| EngineError::Store(error.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| EngineError::Store(error.to_string()))?;
        if !status.is_success() {
            return Err(EngineError::Store(format!(
                "HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }
        if bytes.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(&bytes).map_err(|error| EngineError::Store(error.to_string()))
    }

    pub async fn fetch_traces(
        &self,
        trace_ids: &[String],
    ) -> Result<Vec<TraceRecord>, EngineError> {
        if trace_ids.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({"trace_ids": trace_ids});
        let response = self
            .send_json(Method::GET, "/api/3.0/mlflow/traces/batchGet", Some(&body))
            .await?;
        response
            .get("traces")
            .and_then(Value::as_array)
            .ok_or_else(|| EngineError::Store("batchGet response omitted traces".to_string()))?
            .iter()
            .map(trace_record)
            .collect()
    }

    pub async fn search_trace_infos(
        &self,
        experiment_id: &str,
        filter: Option<&str>,
        order_by: &[&str],
    ) -> Result<Vec<Value>, EngineError> {
        let mut all = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let mut body = json!({
                "locations": [{
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id}
                }],
                "max_results": 500,
                "order_by": order_by,
            });
            if let Some(filter) = filter {
                body["filter"] = Value::String(filter.to_string());
            }
            if let Some(token) = &page_token {
                body["page_token"] = Value::String(token.clone());
            }
            let response = self
                .send_json(Method::POST, "/api/3.0/mlflow/traces/search", Some(&body))
                .await?;
            let page = response
                .get("traces")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            all.extend(page);
            page_token = response
                .get("next_page_token")
                .and_then(Value::as_str)
                .map(str::to_string);
            if page_token.is_none() {
                break;
            }
        }
        Ok(all)
    }

    pub async fn link_traces_to_run(
        &self,
        trace_ids: &[String],
        run_id: &str,
    ) -> Result<(), EngineError> {
        for chunk in trace_ids.chunks(100) {
            self.send_json(
                Method::POST,
                "/api/2.0/mlflow/traces/link-to-run",
                Some(&json!({"trace_ids": chunk, "run_id": run_id})),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn log_assessment(
        &self,
        trace: &TraceRecord,
        assessment: &CanonicalAssessment,
        run_id: Option<&str>,
    ) -> Result<Value, EngineError> {
        let body = assessment_wire(trace, assessment, run_id);
        let response = self
            .send_json(
                Method::POST,
                &format!("/api/3.0/mlflow/traces/{}/assessments", trace.trace_id),
                Some(&json!({"assessment": body})),
            )
            .await?;
        Ok(response.get("assessment").cloned().unwrap_or(Value::Null))
    }

    pub async fn delete_assessment(
        &self,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<(), EngineError> {
        self.send_json(
            Method::DELETE,
            &format!("/api/3.0/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
            None,
        )
        .await?;
        Ok(())
    }

    pub async fn set_experiment_tag(
        &self,
        experiment_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), EngineError> {
        self.send_json(
            Method::POST,
            "/api/2.0/mlflow/experiments/set-experiment-tag",
            Some(&json!({"experiment_id": experiment_id, "key": key, "value": value})),
        )
        .await?;
        Ok(())
    }

    pub async fn get_experiment(&self, experiment_id: &str) -> Result<Value, EngineError> {
        self.send_json(
            Method::GET,
            "/api/2.0/mlflow/experiments/get",
            Some(&json!({"experiment_id": experiment_id})),
        )
        .await
    }

    pub async fn log_metrics(
        &self,
        run_id: &str,
        metrics: &BTreeMap<String, f64>,
    ) -> Result<(), EngineError> {
        let timestamp = Utc::now().timestamp_millis();
        let metrics = metrics
            .iter()
            .map(|(key, value)| {
                json!({
                    "key": key,
                    "value": value,
                    "timestamp": timestamp,
                    "step": 0,
                })
            })
            .collect::<Vec<_>>();
        self.send_json(
            Method::POST,
            "/api/2.0/mlflow/runs/log-batch",
            Some(&json!({"run_id": run_id, "metrics": metrics, "params": [], "tags": []})),
        )
        .await?;
        Ok(())
    }

    pub async fn terminate_run(&self, run_id: &str, status: &str) -> Result<(), EngineError> {
        self.send_json(
            Method::POST,
            "/api/2.0/mlflow/runs/update",
            Some(&json!({
                "run_id": run_id,
                "status": status,
                "end_time": Utc::now().timestamp_millis(),
            })),
        )
        .await?;
        Ok(())
    }

    pub async fn create_evaluator_trace(
        &self,
        experiment_id: &str,
        run_id: Option<&str>,
        scorer_name: &str,
    ) -> Result<String, EngineError> {
        let trace_id = format!("tr-{}", Uuid::new_v4().simple());
        let mut metadata = Map::new();
        metadata.insert(
            "mlflow.traceName".to_string(),
            Value::String(scorer_name.to_string()),
        );
        if let Some(run_id) = run_id {
            metadata.insert(
                "mlflow.sourceRun".to_string(),
                Value::String(run_id.to_string()),
            );
        }
        self.send_json(
            Method::POST,
            "/api/3.0/mlflow/traces",
            Some(&json!({"trace": {"trace_info": {
                "trace_id": trace_id,
                "trace_location": {
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id}
                },
                "request_time": Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true),
                "execution_duration": "0s",
                "state": "OK",
                "trace_metadata": metadata,
                "tags": {SOURCE_SCORER_NAME: scorer_name},
            }}})),
        )
        .await?;
        if let Some(run_id) = run_id {
            self.link_traces_to_run(std::slice::from_ref(&trace_id), run_id)
                .await?;
        }
        Ok(trace_id)
    }
}

fn assessment_wire(
    trace: &TraceRecord,
    assessment: &CanonicalAssessment,
    run_id: Option<&str>,
) -> Value {
    let mut metadata = assessment
        .metadata
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(python_str(value))))
        .collect::<Map<_, _>>();
    if let Some(run_id) = run_id {
        metadata.insert(SOURCE_RUN_ID.to_string(), Value::String(run_id.to_string()));
    }
    let feedback = match &assessment.error {
        Some(error) => json!({"error": assessment_error_wire(error)}),
        None => json!({"value": assessment.value.clone().unwrap_or(Value::Null)}),
    };
    let mut wire = json!({
        "assessment_name": assessment.name,
        "trace_id": trace.trace_id,
        "source": {
            "source_type": assessment.source.source_type,
            "source_id": assessment.source.source_id.clone().unwrap_or_else(|| "default".to_string()),
        },
        "feedback": feedback,
        "metadata": metadata,
        "create_time": assessment_time(assessment.create_time_ms),
        "last_update_time": assessment_time(assessment.last_update_time_ms),
        "valid": true,
    });
    if let Some(rationale) = &assessment.rationale {
        wire["rationale"] = Value::String(rationale.clone());
    }
    if let Some(span_id) = assessment.span_id.as_ref().or(trace.root_span_id.as_ref()) {
        wire["span_id"] = Value::String(span_id.clone());
    }
    wire
}

fn assessment_time(timestamp_ms: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(timestamp_ms)
        .expect("assessment timestamp is in the supported datetime range")
        .to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn assessment_error_wire(error: &ScorerAssessmentError) -> Value {
    json!({
        "error_code": error.error_code,
        "error_message": error.error_message,
        "stack_trace": error.stack_trace,
    })
}

fn python_str(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{key:?}: {}", python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::String(value) => format!("{value:?}"),
        _ => python_str(value),
    }
}

fn trace_record(trace: &Value) -> Result<TraceRecord, EngineError> {
    let info = trace
        .get("trace_info")
        .and_then(Value::as_object)
        .ok_or_else(|| EngineError::Store("trace omitted trace_info".to_string()))?;
    let trace_id = info
        .get("trace_id")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::Store("trace omitted trace_id".to_string()))?
        .to_string();
    let timestamp_ms = info
        .get("request_time")
        .and_then(Value::as_str)
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.timestamp_millis())
        .unwrap_or_default();
    let experiment_id = info
        .get("trace_location")
        .and_then(|location| location.pointer("/mlflow_experiment/experiment_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let metadata = string_map(info.get("trace_metadata"));
    let tags = string_map(info.get("tags"));
    let assessments = info
        .get("assessments")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let spans = trace
        .get("spans")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let converted_spans = spans.iter().map(convert_span).collect::<Vec<_>>();
    let root = converted_spans.iter().find(|span| {
        span.get("parent_span_id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    });
    let root_span_id = root
        .and_then(|span| span.get("span_id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let inputs = root
        .and_then(|span| span.pointer("/attributes/mlflow.spanInputs"))
        .cloned();
    let outputs = root
        .and_then(|span| span.pointer("/attributes/mlflow.spanOutputs"))
        .cloned();
    let expectations = assessments
        .iter()
        .filter_map(|assessment| {
            let name = assessment.get("assessment_name")?.as_str()?.to_string();
            let value = assessment.pointer("/expectation/value")?.clone();
            Some((name, value))
        })
        .collect::<Map<_, _>>();
    let canonical_trace = json!({
        "info": {
            "trace_id": trace_id,
            "trace_location": info.get("trace_location").cloned().unwrap_or(Value::Null),
            "request_time": info.get("request_time").cloned().unwrap_or(Value::Null),
            "timestamp_ms": timestamp_ms,
            "trace_metadata": metadata,
            "tags": tags,
            "assessments": assessments,
        },
        "data": {"spans": converted_spans},
    });
    Ok(TraceRecord {
        trace_id,
        experiment_id,
        timestamp_ms,
        metadata,
        assessments,
        root_span_id,
        eval_item: EvalItem {
            inputs,
            outputs,
            expectations: Some(Value::Object(expectations)),
            trace: Some(canonical_trace),
            session: None,
            memory_examples: None,
        },
    })
}

fn string_map(value: Option<&Value>) -> BTreeMap<String, String> {
    value
        .and_then(Value::as_object)
        .map(|values| {
            values
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn convert_span(span: &Value) -> Value {
    let mut result = span.as_object().cloned().unwrap_or_default();
    let attributes = span
        .get("attributes")
        .and_then(Value::as_array)
        .map(|attributes| {
            attributes
                .iter()
                .filter_map(|attribute| {
                    let key = attribute.get("key")?.as_str()?.to_string();
                    let value = attribute.get("value").map(any_value)?;
                    Some((key, value))
                })
                .collect::<Map<_, _>>()
        })
        .or_else(|| span.get("attributes").and_then(Value::as_object).cloned())
        .unwrap_or_default();
    result.insert("attributes".to_string(), Value::Object(attributes));
    Value::Object(result)
}

fn any_value(value: &Value) -> Value {
    if let Some(value) = value.get("string_value").and_then(Value::as_str) {
        return serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
    }
    for key in ["bool_value", "int_value", "double_value"] {
        if let Some(value) = value.get(key) {
            return value.clone();
        }
    }
    Value::Null
}

pub(crate) fn trace_info_id(info: &Value) -> Option<&str> {
    info.get("trace_id").and_then(Value::as_str)
}

pub(crate) fn trace_info_timestamp(info: &Value) -> i64 {
    info.get("request_time")
        .and_then(Value::as_str)
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.timestamp_millis())
        .unwrap_or_default()
}

pub(crate) fn trace_info_metadata<'a>(info: &'a Value, key: &str) -> Option<&'a str> {
    info.pointer(&format!("/trace_metadata/{key}"))
        .and_then(Value::as_str)
}

pub(crate) fn assessment_dictionary(assessment: &CanonicalAssessment) -> Value {
    let mut value = json!({
        "assessment_name": assessment.name,
        "trace_id": "",
        "source": {
            "source_type": assessment.source.source_type,
            "source_id": assessment.source.source_id.clone().unwrap_or_else(|| "default".to_string()),
        },
        "metadata": assessment.metadata.iter().map(|(key, value)| (key.clone(), Value::String(python_str(value)))).collect::<Map<_, _>>(),
        "create_time": assessment_time(assessment.create_time_ms),
        "last_update_time": assessment_time(assessment.last_update_time_ms),
        "valid": true,
    });
    if let Some(feedback) = &assessment.value {
        value["feedback"] = json!({"value": feedback});
    } else if let Some(error) = &assessment.error {
        value["feedback"] = json!({"error": assessment_error_wire(error)});
    }
    if let Some(rationale) = &assessment.rationale {
        value["rationale"] = Value::String(rationale.clone());
    }
    value
}

pub(crate) fn set_scorer_trace_metadata(assessments: &mut [CanonicalAssessment], trace_id: &str) {
    for assessment in assessments {
        assessment.metadata.insert(
            SCORER_TRACE_ID.to_string(),
            Value::String(trace_id.to_string()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otel_trace_is_projected_to_executor_shape() {
        let trace = json!({
            "trace_info": {
                "trace_id": "tr-1",
                "request_time": "2026-01-01T00:00:00Z",
                "trace_metadata": {"mlflow.trace.session": "s-1"},
                "tags": {},
                "assessments": [{"assessment_name": "expected", "expectation": {"value": "yes"}}]
            },
            "spans": [{
                "span_id": "root",
                "name": "agent",
                "attributes": [
                    {"key": "mlflow.spanInputs", "value": {"string_value": "{\"q\":\"x\"}"}},
                    {"key": "mlflow.spanOutputs", "value": {"string_value": "\"answer\""}}
                ]
            }]
        });
        let record = trace_record(&trace).unwrap();
        assert_eq!(record.session_id(), Some("s-1"));
        assert_eq!(record.eval_item.inputs, Some(json!({"q": "x"})));
        assert_eq!(record.eval_item.outputs, Some(json!("answer")));
        assert_eq!(
            record.eval_item.expectations,
            Some(json!({"expected": "yes"}))
        );
        assert_eq!(record.root_span_id.as_deref(), Some("root"));
    }

    #[test]
    fn metadata_uses_python_string_conversion() {
        assert_eq!(python_str(&json!(true)), "True");
        assert_eq!(python_str(&json!(["a", 2])), "[\"a\", 2]");
    }
}
