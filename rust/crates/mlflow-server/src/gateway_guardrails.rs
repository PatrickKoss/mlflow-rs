use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use mlflow_genai::{EvalItem, Feedback, ScorerExecutor, SerializedScorer};
use mlflow_store::{GatewayGuardrail, ResolvedGatewayEndpointConfig};
use serde_json::{json, Value};

use crate::state::AppState;

pub const SANITIZE_BYPASS_HEADER: &str = "x-mlflow-guardrail-bypass";
const MAX_RATIONALE_LEN: usize = 500;
const SANITIZE_SYSTEM_PROMPT: &str = "You are a content sanitizer. You will receive a JSON payload and an issue description.\nFix the issue by modifying the content using the following rules:\n- Replace content that cannot be safely rephrased (e.g. sensitive data, PII, credentials)\n  with [REDACTED].\n- Rewrite content that can be made acceptable (e.g. soften hostile tone, remove bias,\n  generalize specifics).\nPreserve the payload structure and overall intent. Do not add new fields or change the schema.\nReturn ONLY a valid JSON object with the same schema as the input payload.\n\nIssue: {rationale}\n\nInput payload:\n{payload_json}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailPayloadSchema {
    ChatRequest,
    ChatResponse,
}

impl GuardrailPayloadSchema {
    fn value(self) -> Value {
        let raw = match self {
            Self::ChatRequest => include_str!("gateway_guardrail_chat_request_schema.json"),
            Self::ChatResponse => include_str!("gateway_guardrail_chat_response_schema.json"),
        };
        serde_json::from_str(raw).expect("checked-in guardrail schema is valid JSON")
    }
}

#[derive(Debug, Clone)]
struct RunnableGuardrail {
    entity: GatewayGuardrail,
    scorer: SerializedScorer,
}

#[derive(Clone)]
pub struct LoadedGuardrails {
    guardrails: Vec<RunnableGuardrail>,
    executor: ScorerExecutor,
    server_base_url: Option<String>,
    authorization: Option<HeaderValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailExecutionError {
    pub status: StatusCode,
    pub detail: String,
    pub stream_type: &'static str,
}

impl GuardrailExecutionError {
    fn violation(name: &str, rationale: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: format!("Guardrail '{name}' blocked: {rationale}"),
            stream_type: "GuardrailViolation",
        }
    }

    fn internal(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
            stream_type: "MlflowException",
        }
    }
}

impl LoadedGuardrails {
    pub fn empty() -> Self {
        Self {
            guardrails: Vec::new(),
            executor: ScorerExecutor::new(),
            server_base_url: None,
            authorization: None,
        }
    }

    pub async fn before(
        &self,
        mut payload: Value,
        schema: Option<GuardrailPayloadSchema>,
    ) -> Result<Value, GuardrailExecutionError> {
        for guardrail in &self.guardrails {
            if guardrail.entity.stage == "BEFORE" {
                let feedback = self.execute(guardrail, Some(payload.clone()), None).await?;
                payload = self.enforce(guardrail, payload, &feedback, schema).await?;
            }
        }
        Ok(payload)
    }

    pub async fn after(
        &self,
        request: &Value,
        mut response: Value,
        schema: Option<GuardrailPayloadSchema>,
    ) -> Result<Value, GuardrailExecutionError> {
        for guardrail in &self.guardrails {
            if guardrail.entity.stage == "AFTER" {
                let feedback = self
                    .execute(guardrail, Some(request.clone()), Some(response.clone()))
                    .await?;
                response = self.enforce(guardrail, response, &feedback, schema).await?;
            }
        }
        Ok(response)
    }

    async fn execute(
        &self,
        guardrail: &RunnableGuardrail,
        inputs: Option<Value>,
        outputs: Option<Value>,
    ) -> Result<Feedback, GuardrailExecutionError> {
        let gateway_url = self
            .server_base_url
            .as_ref()
            .map(|base| format!("{base}/gateway/mlflow/v1/chat/completions"));
        self.executor
            .execute(
                &guardrail.scorer,
                &EvalItem {
                    inputs,
                    outputs,
                    expectations: None,
                    ..EvalItem::default()
                },
                gateway_url.as_deref(),
            )
            .await
            .map_err(|error| GuardrailExecutionError::internal(error.to_string()))
    }

    async fn enforce(
        &self,
        guardrail: &RunnableGuardrail,
        payload: Value,
        feedback: &Feedback,
        schema: Option<GuardrailPayloadSchema>,
    ) -> Result<Value, GuardrailExecutionError> {
        if is_passing(&feedback.value)? {
            return Ok(payload);
        }
        let rationale = rationale(feedback);
        if guardrail.entity.action == "VALIDATION" {
            return Err(GuardrailExecutionError::violation(
                &guardrail.entity.name,
                &rationale,
            ));
        }
        self.sanitize(guardrail, payload, &rationale, schema).await
    }

    async fn sanitize(
        &self,
        guardrail: &RunnableGuardrail,
        payload: Value,
        rationale: &str,
        schema: Option<GuardrailPayloadSchema>,
    ) -> Result<Value, GuardrailExecutionError> {
        let Some(action_endpoint) = guardrail.entity.action_endpoint_name.as_deref() else {
            return Err(GuardrailExecutionError::violation(
                &guardrail.entity.name,
                "Sanitization requires an action_llm_url but none was configured.",
            ));
        };
        let Some(base_url) = self.server_base_url.as_deref() else {
            return Err(GuardrailExecutionError::violation(
                &guardrail.entity.name,
                "Sanitization requires an action_llm_url but none was configured.",
            ));
        };
        let prompt = SANITIZE_SYSTEM_PROMPT
            .replace("{rationale}", rationale)
            .replace("{payload_json}", &python_json_dumps_pretty(&payload));
        let mut body = json!({
            "messages": [{"role": "user", "content": prompt}],
        });
        if let Some(schema) = schema {
            body.as_object_mut().expect("JSON object").insert(
                "response_format".to_string(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sanitized_payload",
                        "strict": false,
                        "schema": schema.value(),
                    },
                }),
            );
        }

        let mut request = reqwest::Client::new()
            .post(format!(
                "{base_url}/gateway/{action_endpoint}/mlflow/invocations"
            ))
            .header(header::CONTENT_TYPE, "application/json")
            .header(SANITIZE_BYPASS_HEADER, "1")
            .json(&body);
        if let Some(authorization) = &self.authorization {
            request = request.header(header::AUTHORIZATION, authorization);
        }
        let response = request.send().await.map_err(|error| {
            GuardrailExecutionError::internal(format!("Sanitization request failed: {error}"))
        })?;
        let status = response.status();
        let response: Value = response.json().await.map_err(|error| {
            GuardrailExecutionError::internal(format!("Sanitization request failed: {error}"))
        })?;
        if !status.is_success() {
            let detail = response
                .pointer("/error/message")
                .or_else(|| response.get("detail"))
                .cloned()
                .unwrap_or(response);
            return Err(GuardrailExecutionError::violation(
                &guardrail.entity.name,
                &format!("Sanitization request failed: {}", display_detail(&detail)),
            ));
        }
        let content = response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GuardrailExecutionError::violation(
                    &guardrail.entity.name,
                    "Sanitization LLM response is missing 'choices[0].message.content'.",
                )
            })?;
        serde_json::from_str(content).map_err(|_| {
            GuardrailExecutionError::violation(
                &guardrail.entity.name,
                "Sanitization LLM returned invalid JSON.",
            )
        })
    }
}

pub async fn load_guardrails(
    state: &AppState,
    workspace: &str,
    endpoint: &ResolvedGatewayEndpointConfig,
    headers: &HeaderMap,
) -> LoadedGuardrails {
    let bypass = headers
        .get(SANITIZE_BYPASS_HEADER)
        .is_some_and(|value| value == "1");
    let mut guardrails = Vec::new();
    if !bypass {
        match state
            .tracking_store()
            .list_endpoint_guardrail_configs(workspace, &endpoint.endpoint_id)
            .await
        {
            Ok(configs) => {
                for config in configs {
                    let Some(mut entity) = config.guardrail else {
                        continue;
                    };
                    let loaded = async {
                        entity.scorer = state
                            .tracking_store()
                            .resolve_endpoint_in_scorer(workspace, &entity.scorer)
                            .await?;
                        let scorer = SerializedScorer::from_json(&entity.scorer.serialized_scorer)
                            .map_err(|error| {
                                mlflow_error::MlflowError::internal_error(error.to_string())
                            })?;
                        Ok::<_, mlflow_error::MlflowError>(RunnableGuardrail { entity, scorer })
                    }
                    .await;
                    match loaded {
                        Ok(guardrail) => guardrails.push(guardrail),
                        Err(error) => tracing::warn!(
                            guardrail_id = %config.guardrail_id,
                            error = %error,
                            "Failed to load guardrail, skipping"
                        ),
                    }
                }
            }
            Err(error) => tracing::warn!(
                endpoint_id = %endpoint.endpoint_id,
                error = %error,
                "Failed to load endpoint guardrails, skipping"
            ),
        }
    }
    LoadedGuardrails {
        guardrails,
        executor: ScorerExecutor::new(),
        server_base_url: request_base_url(headers),
        authorization: headers.get(header::AUTHORIZATION).cloned(),
    }
}

fn request_base_url(headers: &HeaderMap) -> Option<String> {
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("http");
    let host = headers
        .get("x-forwarded-host")
        .or_else(|| headers.get(header::HOST))?
        .to_str()
        .ok()?
        .split(',')
        .next()?
        .trim();
    (!host.is_empty()).then(|| format!("{scheme}://{host}"))
}

fn is_passing(value: &Value) -> Result<bool, GuardrailExecutionError> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::String(value) => Ok(value.trim().eq_ignore_ascii_case("yes")),
        value => Err(GuardrailExecutionError::internal(format!(
            "Scorer or Feedback returned an unexpected value type '{}'; expected bool or str.",
            json_type_name(value)
        ))),
    }
}

fn rationale(feedback: &Feedback) -> String {
    let raw = if feedback.rationale.is_empty() {
        match &feedback.value {
            Value::Bool(true) => "True".to_string(),
            Value::Bool(false) => "False".to_string(),
            Value::String(value) => value.clone(),
            value => value.to_string(),
        }
    } else {
        feedback.rationale.clone()
    };
    raw.chars().take(MAX_RATIONALE_LEN).collect()
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn display_detail(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

fn python_json_dumps_pretty(value: &Value) -> String {
    fn write(out: &mut String, value: &Value, depth: usize) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => out.push_str(&value.to_string()),
            Value::String(value) => out.push_str(&mlflow_store::python_json_dumps(
                &Value::String(value.clone()),
                false,
            )),
            Value::Array(values) => {
                if values.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (index, value) in values.iter().enumerate() {
                    out.push_str(&" ".repeat((depth + 1) * 2));
                    write(out, value, depth + 1);
                    if index + 1 < values.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&" ".repeat(depth * 2));
                out.push(']');
            }
            Value::Object(values) => {
                if values.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (index, (key, value)) in values.iter().enumerate() {
                    out.push_str(&" ".repeat((depth + 1) * 2));
                    out.push_str(&mlflow_store::python_json_dumps(
                        &Value::String(key.clone()),
                        false,
                    ));
                    out.push_str(": ");
                    write(out, value, depth + 1);
                    if index + 1 < values.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&" ".repeat(depth * 2));
                out.push('}');
            }
        }
    }

    let mut output = String::new();
    write(&mut output, value, 0);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlflow_store::ScorerVersion;

    #[test]
    fn pretty_json_matches_python_indentation_and_ascii_escaping() {
        let value = json!({"messages":[{"role":"user","content":"héllo"}],"n":1});
        assert_eq!(
            python_json_dumps_pretty(&value),
            "{\n  \"messages\": [\n    {\n      \"role\": \"user\",\n      \"content\": \"h\\u00e9llo\"\n    }\n  ],\n  \"n\": 1\n}"
        );
    }

    #[test]
    fn pass_values_and_rationale_match_python_guardrail_rules() {
        assert!(is_passing(&json!(true)).unwrap());
        assert!(is_passing(&json!(" YES ")).unwrap());
        assert!(!is_passing(&json!("no")).unwrap());
        let feedback = Feedback {
            name: "fixture".to_string(),
            value: json!(false),
            rationale: "x".repeat(510),
            source: None,
            metadata: None,
            span_id: None,
            trace_id: None,
        };
        assert_eq!(rationale(&feedback).len(), 500);
    }

    fn runnable(name: &str, stage: &str, scorer: Value) -> RunnableGuardrail {
        let serialized_scorer = scorer.to_string();
        RunnableGuardrail {
            entity: GatewayGuardrail {
                guardrail_id: format!("gr-{name}"),
                name: name.to_string(),
                scorer: ScorerVersion {
                    experiment_id: "0".to_string(),
                    scorer_name: name.to_string(),
                    scorer_version: 1,
                    serialized_scorer: serialized_scorer.clone(),
                    creation_time: Some(0),
                    scorer_id: format!("s-{name}"),
                },
                stage: stage.to_string(),
                action: "VALIDATION".to_string(),
                action_endpoint_name: None,
                created_by: None,
                created_at: 0,
                last_updated_by: None,
                last_updated_at: 0,
                workspace: "default".to_string(),
            },
            scorer: SerializedScorer::from_json(&serialized_scorer).unwrap(),
        }
    }

    fn response_length(name: &str, max_length: usize) -> Value {
        json!({
            "name": name,
            "builtin_scorer_class": "ResponseLength",
            "builtin_scorer_pydantic_data": {
                "unit": "words",
                "min_length": null,
                "max_length": max_length,
            },
        })
    }

    #[tokio::test]
    async fn after_guardrails_preserve_order_and_stop_at_first_violation() {
        let unsupported = json!({
            "name": "must-not-run",
            "call_source": "def must_not_run(): pass",
        });
        let guardrails = LoadedGuardrails {
            guardrails: vec![
                runnable("first-pass", "AFTER", response_length("first-pass", 3)),
                runnable("second-block", "AFTER", response_length("second-block", 1)),
                runnable("must-not-run", "AFTER", unsupported),
            ],
            executor: ScorerExecutor::new(),
            server_base_url: None,
            authorization: None,
        };
        let error = guardrails
            .after(&json!({}), json!("two words"), None)
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(
            error.detail,
            "Guardrail 'second-block' blocked: Output length (2 words) exceeds the maximum (1 words)"
        );
    }

    #[tokio::test]
    async fn orchestration_runs_only_the_matching_stage() {
        let guardrails = LoadedGuardrails {
            guardrails: vec![runnable(
                "before-only",
                "BEFORE",
                json!({
                    "name": "before-only",
                    "call_source": "def before_only(): pass",
                }),
            )],
            executor: ScorerExecutor::new(),
            server_base_url: None,
            authorization: None,
        };
        assert_eq!(
            guardrails
                .after(&json!({}), json!("unchanged"), None)
                .await
                .unwrap(),
            json!("unchanged")
        );
    }
}
