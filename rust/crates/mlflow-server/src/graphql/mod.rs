//! `/graphql` (GET + POST) — the MLflow GraphQL endpoint (plan T6.1/T6.2,
//! §3.12), a faithful port of `mlflow/server/handlers.py::_graphql` and the
//! graphene schema it executes.
//!
//! ## Approach: `graphql-parser` + hand-written resolution
//!
//! The MLflow GraphQL schema is small and **closed** (7 operations plus the two
//! `test`/`testMutation` echo fields — no runtime schema changes). Rather than
//! pull in a full GraphQL server crate (`async-graphql`/`juniper`) whose
//! spec-compliant execution and error shapes would *differ* from graphene's
//! OSS behavior, we parse the query into an AST with the small, stable
//! `graphql-parser` crate and resolve it by hand against the fixed schema. This
//! is the most parity-faithful route because two graphene-isms are load-bearing
//! and awkward to reproduce through a general engine:
//!
//! 1. **Error shape.** `_graphql` returns `{"errors": [e.message for e in
//!    result.errors]}` — a list of bare message *strings*, not spec error
//!    objects with `locations`/`path`. A resolver exception becomes exactly one
//!    such string and that root field is `null` in `data`.
//! 2. **`apiError` is a data field, not a GraphQL error.** The response types
//!    carry an `apiError` union member, but the OSS impls never populate it, so
//!    graphene serializes it as `null`; store errors surface as `errors`
//!    strings. (Databricks backends populate `apiError`; OSS does not.)
//!
//! ## Request handling (both methods)
//!
//! `_graphql` reads the request via `_get_request_json()` =
//! `get_json(force=True, silent=True)` after `_validate_content_type` — and
//! content-type validation only applies to POST/PUT. So **GET and POST both
//! parse the JSON body** (GET with a JSON body, content-type unchecked); there
//! is no query-string parsing. We reproduce that exactly.
//!
//! The extracted `query` is parsed, run through the query-safety / no-batching
//! guard ([`no_batching::check_query_safety`]), then executed. A batched request
//! (a JSON *array* body) has no `query`/`variables`/`operationName` keys and is
//! rejected — Python would raise an uncaught `AttributeError` (list has no
//! `.get`) → 500; we return a clean GraphQL error body instead, a deliberate,
//! documented hardening that is still a rejection.

pub mod executor;
pub mod no_batching;
pub mod resolvers;
pub mod schema;
pub mod value;

use axum::body::Bytes;
use axum::http::header;
use axum::http::request::Parts;
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use graphql_parser::query::parse_query;

use crate::state::AppState;
use crate::workspace::Workspace;
use executor::{error_only_body, execute, parse_variables, GraphQlRequest};
use value::GqlVal;

/// The `/graphql` handler (GET + POST). Always responds `200 application/json`
/// with a `{"data", "errors"}` body — GraphQL transports errors in-band, so even
/// a malformed query is a 200 with a populated `errors` array (matching
/// graphene/`jsonify`).
pub async fn graphql(
    axum::extract::State(state): axum::extract::State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Response {
    // `_validate_content_type(["application/json"])` — POST/PUT only.
    if let Err(err) = validate_content_type(&parts) {
        // Content-type errors are raised *before* GraphQL execution in Python
        // (`_get_request_json`), so they surface as a normal MlflowException
        // JSON error response, not an in-band GraphQL `errors` array.
        return err.into_response();
    }

    let request = match extract_request(&body) {
        Ok(req) => req,
        Err(body) => return json_ok(body),
    };

    // Parse + query-safety guard.
    let doc = match parse_query::<String>(&request.query) {
        Ok(doc) => doc,
        Err(e) => {
            // graphene surfaces a parse failure as a single error string
            // (the `GraphQLSyntaxError` message). We use `graphql-parser`'s
            // message; the shape (`{"data": null, "errors": [msg]}`) matches.
            return json_ok(error_only_body(&e.to_string()));
        }
    };
    if let Err(msg) = no_batching::check_query_safety(&doc) {
        return json_ok(error_only_body(&msg));
    }

    // Execute: resolve each selected root field against the stores.
    let body = execute(&doc, &request, |field_name, input| {
        let state = state.clone();
        let workspace = workspace.clone();
        async move { resolve_root(&state, &workspace, &field_name, input).await }
    })
    .await;

    json_ok(body)
}

/// Dispatch a root field to its resolver, mapping any [`MlflowError`] into the
/// bare message string graphene would surface (`str(exception)`).
async fn resolve_root(
    state: &AppState,
    workspace: &Workspace,
    field_name: &str,
    input: serde_json::Map<String, serde_json::Value>,
) -> Result<GqlVal, String> {
    let result = match field_name {
        "mlflowGetExperiment" => resolvers::get_experiment(state, workspace, &input).await,
        "mlflowGetRun" => resolvers::get_run(state, workspace, &input).await,
        "mlflowGetMetricHistoryBulkInterval" => {
            resolvers::get_metric_history_bulk_interval(state, workspace, &input).await
        }
        "mlflowListArtifacts" => resolvers::list_artifacts(state, workspace, &input).await,
        "mlflowSearchModelVersions" => {
            resolvers::search_model_versions(state, workspace, &input).await
        }
        "mlflowSearchRuns" => resolvers::search_runs(state, workspace, &input).await,
        "mlflowSearchDatasets" => resolvers::search_datasets(state, workspace, &input).await,
        "test" => return Ok(resolvers::test_echo("Test", &input)),
        "testMutation" => return Ok(resolvers::test_echo("TestMutation", &input)),
        // An unknown root field would be a validation error in graphene
        // ("Cannot query field ..."). Real UI/test queries never hit this.
        other => {
            return Err(format!(
                "Cannot query field \"{other}\" on type \"Query\"."
            ))
        }
    };
    result.map_err(|e| e.message)
}

/// Extract `{query, variables, operationName}` from the JSON body. Returns
/// `Err(body)` carrying a ready `{"data": null, "errors": [...]}` string for the
/// cases that can't yield a runnable query (batched array, non-object body,
/// missing `query`).
fn extract_request(body: &Bytes) -> Result<GraphQlRequest, String> {
    let text = std::str::from_utf8(body).unwrap_or("");
    let json: serde_json::Value = if text.trim().is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_str(text).unwrap_or(serde_json::Value::Null)
    };

    let obj = match &json {
        serde_json::Value::Object(obj) => obj,
        serde_json::Value::Array(_) => {
            return Err(error_only_body(
                "Batched GraphQL requests are not supported.",
            ));
        }
        _ => {
            return Err(error_only_body("Must provide query string."));
        }
    };

    let Some(query) = obj.get("query").and_then(|v| v.as_str()) else {
        return Err(error_only_body("Must provide query string."));
    };

    Ok(GraphQlRequest {
        query: query.to_string(),
        variables: parse_variables(obj.get("variables")),
        operation_name: obj
            .get("operationName")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    })
}

/// `_validate_content_type(request, ["application/json"])` — POST/PUT only.
fn validate_content_type(parts: &Parts) -> Result<(), mlflow_error::MlflowError> {
    if parts.method != Method::POST && parts.method != Method::PUT {
        return Ok(());
    }
    let Some(content_type) = parts.headers.get(header::CONTENT_TYPE) else {
        return Err(mlflow_error::MlflowError::invalid_parameter_value(
            "Bad Request. Content-Type header is missing.".to_string(),
        ));
    };
    let value = content_type.to_str().unwrap_or("");
    let base = value.split(';').next().unwrap_or("").trim();
    if base != "application/json" {
        return Err(mlflow_error::MlflowError::invalid_parameter_value(
            "Bad Request. Content-Type must be one of ['application/json'].".to_string(),
        ));
    }
    Ok(())
}

/// A `200 application/json` response for a GraphQL body. Flask's `jsonify`
/// appends a trailing newline; graphene results go through the same provider, so
/// the byte-exact body ends in `\n`.
fn json_ok(mut body: String) -> Response {
    body.push('\n');
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .expect("valid response")
}
