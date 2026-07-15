//! `mlflow-server`: the Rust MLflow tracking server library.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§2, Phase 1 T1.5), this crate wires
//! together `mlflow-proto`, `mlflow-store`, `mlflow-registry`,
//! `mlflow-auth`, `mlflow-search`, `mlflow-artifacts`, and `mlflow-webhooks`
//! into an axum HTTP application serving the tracking, tracing, model
//! registry, webhooks, auth/RBAC, and workspaces API with byte-compatible
//! JSON wire behavior against the Python MLflow server. It sits behind
//! nginx, which routes everything except genai endpoints here (§2.2).
//!
//! `main.rs` is intentionally thin: all app construction lives here in
//! [`build_app`] so tests (and later tasks) can compose/exercise the
//! `Router` without booting a real listener.

pub mod artifacts;
pub mod assessments;
pub mod config;
pub mod datasets;
pub mod experiments;
pub mod logged_models;
pub mod metric_history;
pub mod metrics;
pub mod otlp;
pub mod proto_http;
pub mod routes;
pub mod runs;
pub mod state;
pub mod trace_artifact;
pub mod traces;
pub mod traces_v2;
pub mod webhooks;
pub mod workspace;

use axum::extract::MatchedPath;
use axum::http::Request;
use axum::middleware;
use axum::routing::{get, MethodRouter};
use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::trace::TraceLayer;
use tracing::info_span;

pub use config::{Cli, ServerConfig, StaticPrefixError};
pub use state::AppState;

/// Builds the full application `Router` (ops endpoints only — no store).
/// Retained for the ops/skeleton tests that don't need a backend store.
///
/// Request logging and metrics middleware are applied, and everything is nested
/// under `config.static_prefix` when set (mirroring `_add_static_prefix`,
/// `mlflow/server/handlers.py:6731-6734`, which prepends the prefix to every
/// registered route).
pub fn build_app(config: &ServerConfig) -> Router {
    let metrics_handle = metrics::install_recorder();
    build_app_with_recorder(config, metrics_handle, None)
}

/// Builds the full application `Router` with a backend store, registering every
/// proto-backed endpoint implemented so far (Phase 3: experiments + runs) in
/// addition to the ops endpoints. `main` uses this; tests inject a store over a
/// temp DB.
pub fn build_app_with_state(config: &ServerConfig, state: AppState) -> Router {
    let metrics_handle = metrics::install_recorder();
    build_app_with_recorder(config, metrics_handle, Some(state))
}

/// Same as the builders above, but takes an already-installed
/// [`PrometheusHandle`] instead of installing the global recorder. Exists so
/// tests can build multiple `Router`s in the same process without hitting
/// "recorder already installed" panics from `metrics-exporter-prometheus`.
///
/// When `state` is `Some`, the proto-backed endpoints (experiments + runs) are
/// registered on both `/api/2.0/...` and `/ajax-api/2.0/...` (driven by the
/// `mlflow-proto` route table) honoring the static prefix. When `None`, only
/// the ops endpoints are served.
pub fn build_app_with_recorder(
    config: &ServerConfig,
    metrics_handle: PrometheusHandle,
    state: Option<AppState>,
) -> Router {
    let mut api = Router::new()
        .route("/health", get(routes::health))
        .route("/version", get(routes::version))
        .route(
            "/metrics",
            get(move || routes::metrics(metrics_handle.clone())),
        );

    if let Some(state) = state {
        api = api.merge(register_proto_routes(state));
    }

    let api = api
        .layer(middleware::from_fn(metrics::track_metrics))
        .layer(
            TraceLayer::new_for_http().make_span_with(|request: &Request<_>| {
                let path = request
                    .extensions()
                    .get::<MatchedPath>()
                    .map(MatchedPath::as_str)
                    .unwrap_or_else(|| request.uri().path());
                info_span!(
                    "http_request",
                    method = %request.method(),
                    path,
                )
            }),
        );

    match &config.static_prefix {
        Some(prefix) => Router::new().nest(prefix, api),
        None => api,
    }
}

/// Build a `Router` of the implemented proto-backed endpoints, registered on
/// both URL prefixes, driving path/method from the `mlflow-proto` route table
/// (not hand-written paths) so later endpoints slot in by extending
/// [`handler_for`]. The static prefix is applied by the app-level `nest`, so we
/// register the bare `/api/…` + `/ajax-api/…` paths here (passing an empty
/// prefix to `expand`). `with_state` erases the state type so the result merges
/// into the ops router.
///
/// Route-table paths use Flask's `<param>` path-parameter syntax (T1.2); axum
/// (matchit) uses `{param}` instead, so [`to_axum_path`] converts before
/// registering.
///
/// A few routes are hand-registered alongside the route-table-driven ones
/// (same pre-static-prefix router, so `_add_static_prefix` nesting still
/// applies): the correctly-slashed `search-datasets` ajax route
/// (`mlflow/server/__init__.py:135` — the route table only produces the
/// leading-slash-missing form, §3.4 quirk) and the ajax-only,
/// non-proto-backed `get-history-bulk` (plan T3.3).
fn register_proto_routes(state: AppState) -> Router {
    use axum::routing::get;

    let mut router: Router<AppState> = Router::new();
    for spec in mlflow_proto::ROUTE_TABLE {
        let Some(handler) = handler_for(spec.service, spec.method, spec.http_method) else {
            continue;
        };
        for route in spec.expand("") {
            router = router.route(&to_axum_path(&route.path), handler.clone());
        }
    }
    router = router.route(
        "/ajax-api/2.0/mlflow/experiments/search-datasets",
        axum::routing::post(datasets::search_datasets),
    );
    router = router.route(
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        get(metric_history::get_metric_history_bulk),
    );
    // `get-trace-artifact` (plan T4.5, §3.10) — ajax-only, served under both
    // the 2.0 and 3.0 ajax prefixes (`mlflow/server/__init__.py:159-161`);
    // not proto-route-table-driven (plain `request_id`/`path` query params).
    router = router.route(
        "/ajax-api/2.0/mlflow/get-trace-artifact",
        get(trace_artifact::get_trace_artifact),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/get-trace-artifact",
        get(trace_artifact::get_trace_artifact),
    );
    // OTLP trace ingestion (plan T4.3, §3.8) — not a proto-route-table
    // endpoint (its own wire protocol, `mlflow/server/otel_api.py`), so it is
    // hand-registered here like the routes above. `OTLP_TRACES_PATH` is
    // `/v1/traces` (`mlflow/tracing/utils/otlp.py:20`); the static-prefix
    // nesting in `build_app_with_recorder` still applies to it.
    router = router.route("/v1/traces", axum::routing::post(otlp::export_traces));
    // Artifact plane (plan T5.1/T5.3/T5.4, §3.11). `/get-artifact` and
    // `/model-versions/get-artifact` are served at the root
    // (`mlflow/server/__init__.py:111,117`), NOT under an api/ajax prefix.
    router = router.route("/get-artifact", get(artifacts::get_artifact));
    router = router.route(
        "/model-versions/get-artifact",
        get(artifacts::get_model_version_artifact),
    );
    // ajax-only `upload-artifact` (`__init__.py:151`) + logged-model artifact
    // file download (`__init__.py:166`).
    router = router.route(
        "/ajax-api/2.0/mlflow/upload-artifact",
        axum::routing::post(artifacts::upload_artifact),
    );
    router = router.route(
        "/ajax-api/2.0/mlflow/logged-models/{model_id}/artifacts/files",
        get(artifacts::get_logged_model_artifact),
    );
    router.with_state(state)
}

/// Convert a Flask-style path to axum/matchit syntax:
///  * `<param>` → `{param}` (a single path segment);
///  * `<path:param>` → `{*param}` (Flask's `path` converter matches slashes, so
///    it becomes an axum wildcard capture — used by the `MlflowArtifactsService`
///    routes' `<path:artifact_path>`).
///
/// Non-parameterized paths pass through unchanged.
fn to_axum_path(path: &str) -> String {
    // Do the `path:` wildcard rewrite first so the generic `<`/`>` swap below
    // doesn't need to know about the converter prefix.
    path.replace("<path:", "{*")
        .replace('<', "{")
        .replace('>', "}")
}

/// Map a `(service, method, http_method)` route-table entry to its axum
/// handler. Returns `None` for endpoints not yet implemented (they fall through
/// to the 404 `_not_implemented` form). Extend this as later phases land.
fn handler_for(service: &str, method: &str, http_method: &str) -> Option<MethodRouter<AppState>> {
    use axum::routing::{delete, get, patch, post, put};
    // `MlflowArtifactsService` (plan T5.2, §3.11) is a distinct proto service —
    // its 8 endpoints live under `/(api|ajax-api)/2.0/mlflow-artifacts/...` and
    // route to the artifact-proxy handlers (gated by `--serve-artifacts` inside
    // each handler). Their `<path:artifact_path>` segments become axum wildcards
    // via `to_axum_path`.
    if service == "MlflowArtifactsService" {
        return Some(match (method, http_method) {
            ("downloadArtifact", "GET") => get(artifacts::proxy_download),
            ("uploadArtifact", "PUT") => put(artifacts::proxy_upload),
            ("listArtifacts", "GET") => get(artifacts::proxy_list),
            ("deleteArtifact", "DELETE") => delete(artifacts::proxy_delete),
            ("createMultipartUpload", "POST") => post(artifacts::proxy_create_multipart),
            ("completeMultipartUpload", "POST") => post(artifacts::proxy_complete_multipart),
            ("abortMultipartUpload", "POST") => post(artifacts::proxy_abort_multipart),
            ("getPresignedDownloadUrl", "GET") => get(artifacts::proxy_presigned_download),
            _ => return None,
        });
    }
    // `WebhookService` (plan T8.2, §4.16) is a distinct proto service — its 6
    // endpoints live under `/(api|ajax-api)/2.0/mlflow/webhooks[/{webhook_id}]`
    // and route to the webhook CRUD + test handlers. The `{webhook_id}` segment
    // becomes an axum path param via `to_axum_path`, overlaid onto the request
    // proto by `parse_request_with_path_params` (same mechanism as
    // logged-models).
    if service == "WebhookService" {
        return Some(match (method, http_method) {
            ("createWebhook", "POST") => post(webhooks::create_webhook),
            ("listWebhooks", "GET") => get(webhooks::list_webhooks),
            ("getWebhook", "GET") => get(webhooks::get_webhook),
            ("updateWebhook", "PATCH") => patch(webhooks::update_webhook),
            ("deleteWebhook", "DELETE") => delete(webhooks::delete_webhook),
            ("testWebhook", "POST") => post(webhooks::test_webhook),
            _ => return None,
        });
    }
    if service != "MlflowService" {
        return None;
    }
    Some(match (method, http_method) {
        ("listLoggedModelArtifacts", "GET") => get(artifacts::list_logged_model_artifacts),
        ("createExperiment", "POST") => post(experiments::create_experiment),
        ("getExperiment", "GET") => get(experiments::get_experiment),
        ("getExperimentByName", "GET") => get(experiments::get_experiment_by_name),
        ("searchExperiments", "POST") => post(experiments::search_experiments),
        ("searchExperiments", "GET") => get(experiments::search_experiments),
        ("deleteExperiment", "POST") => post(experiments::delete_experiment),
        ("restoreExperiment", "POST") => post(experiments::restore_experiment),
        ("updateExperiment", "POST") => post(experiments::update_experiment),
        ("setExperimentTag", "POST") => post(experiments::set_experiment_tag),
        ("deleteExperimentTag", "POST") => post(experiments::delete_experiment_tag),
        ("searchDatasets", "POST") => post(datasets::search_datasets),
        ("createLoggedModel", "POST") => post(logged_models::create_logged_model),
        ("finalizeLoggedModel", "PATCH") => patch(logged_models::finalize_logged_model),
        ("getLoggedModel", "GET") => get(logged_models::get_logged_model),
        ("deleteLoggedModel", "DELETE") => delete(logged_models::delete_logged_model),
        ("searchLoggedModels", "POST") => post(logged_models::search_logged_models),
        ("setLoggedModelTags", "PATCH") => patch(logged_models::set_logged_model_tags),
        ("deleteLoggedModelTag", "DELETE") => delete(logged_models::delete_logged_model_tag),
        ("LogLoggedModelParams", "POST") => post(logged_models::log_logged_model_params),
        ("createRun", "POST") => post(runs::create_run),
        ("updateRun", "POST") => post(runs::update_run),
        ("deleteRun", "POST") => post(runs::delete_run),
        ("restoreRun", "POST") => post(runs::restore_run),
        ("getRun", "GET") => get(runs::get_run),
        ("searchRuns", "POST") => post(runs::search_runs),
        ("logMetric", "POST") => post(runs::log_metric),
        ("logParam", "POST") => post(runs::log_param),
        ("setTag", "POST") => post(runs::set_tag),
        ("deleteTag", "POST") => post(runs::delete_tag),
        ("logBatch", "POST") => post(runs::log_batch),
        ("logModel", "POST") => post(runs::log_model),
        ("logInputs", "POST") => post(runs::log_inputs),
        ("logOutputs", "POST") => post(runs::log_outputs),
        ("getMetricHistory", "GET") => get(metric_history::get_metric_history),
        ("getMetricHistoryBulkInterval", "GET") => {
            get(metric_history::get_metric_history_bulk_interval)
        }
        ("createAssessment", "POST") => post(assessments::create_assessment),
        ("GetAssessment", "GET") => get(assessments::get_assessment),
        ("updateAssessment", "PATCH") => patch(assessments::update_assessment),
        ("deleteAssessment", "DELETE") => delete(assessments::delete_assessment),
        // Tracing V3 (T4.1, §3.6).
        ("startTraceV3", "POST") => post(traces::start_trace_v3),
        ("getTraceInfoV3", "GET") => get(traces::get_trace_info_v3),
        ("getTrace", "GET") => get(traces::get_trace),
        ("batchGetTraces", "GET") => get(traces::batch_get_traces),
        ("batchGetTraceInfos", "POST") => post(traces::batch_get_trace_infos),
        ("searchTracesV3", "POST") => post(traces::search_traces_v3),
        ("deleteTracesV3", "POST") => post(traces::delete_traces_v3),
        ("setTraceTagV3", "PATCH") => patch(traces::set_trace_tag_v3),
        ("deleteTraceTagV3", "DELETE") => delete(traces::delete_trace_tag_v3),
        ("linkTracesToRun", "POST") => post(traces::link_traces_to_run),
        ("linkPromptsToTrace", "POST") => post(traces::link_prompts_to_trace),
        ("calculateTraceFilterCorrelation", "POST") => {
            post(traces::calculate_trace_filter_correlation)
        }
        ("queryTraceMetrics", "POST") => post(traces::query_trace_metrics),
        // Tracing V2 (T4.2, §3.7) — deprecated adapters, registered only at
        // the `/api/2.0` prefix (`since.major = 2`), so they never collide
        // with their V3 twins above despite sharing tail paths.
        ("startTrace", "POST") => post(traces_v2::start_trace),
        ("endTrace", "PATCH") => patch(traces_v2::end_trace),
        ("getTraceInfo", "GET") => get(traces_v2::get_trace_info),
        ("searchTraces", "GET") => get(traces_v2::search_traces),
        ("deleteTraces", "POST") => post(traces_v2::delete_traces),
        ("setTraceTag", "PATCH") => patch(traces_v2::set_trace_tag),
        ("deleteTraceTag", "DELETE") => delete(traces_v2::delete_trace_tag),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use metrics_exporter_prometheus::PrometheusBuilder;
    use tower::ServiceExt;

    fn test_config(static_prefix: Option<&str>) -> ServerConfig {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            static_prefix: static_prefix.map(str::to_string),
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_destination: None,
        }
    }

    fn test_app(static_prefix: Option<&str>) -> Router {
        let handle = PrometheusBuilder::new().build_recorder().handle();
        build_app_with_recorder(&test_config(static_prefix), handle, None)
    }

    async fn body_string(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok_without_prefix() {
        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/html; charset=utf-8"
        );
        assert_eq!(body_string(response).await, "OK");
    }

    #[tokio::test]
    async fn version_matches_mlflow_version_py() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(std::path::Path::parent)
            .and_then(std::path::Path::parent)
            .unwrap();
        let version_py = std::fs::read_to_string(repo_root.join("mlflow/version.py")).unwrap();
        let expected = version_py
            .lines()
            .find_map(|line| {
                let rest = line.trim().strip_prefix("VERSION")?.trim_start();
                let rest = rest.strip_prefix('=')?.trim_start();
                let rest = rest.strip_prefix('"')?;
                let end = rest.find('"')?;
                Some(rest[..end].to_string())
            })
            .expect("VERSION line in mlflow/version.py");

        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/version")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_string(response).await, expected);
    }

    #[tokio::test]
    async fn metrics_returns_prometheus_exposition_format() {
        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        // No assertions on specific metric names: the `metrics` crate's
        // recorder is per-Router here (fresh handle per test), so the body
        // may be empty until requests are recorded. We only care that the
        // endpoint responds successfully with the expected content type.
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/plain; version=0.0.4"
        );
    }

    #[tokio::test]
    async fn routes_are_nested_under_static_prefix() {
        let app = test_app(Some("/mlflow"));

        let prefixed = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/mlflow/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(prefixed.status(), StatusCode::OK);

        let unprefixed = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unprefixed.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_path_returns_404() {
        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_path_returns_404_with_prefix() {
        let response = test_app(Some("/mlflow"))
            .oneshot(
                HttpRequest::builder()
                    .uri("/mlflow/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn to_axum_path_converts_flask_param_syntax() {
        assert_eq!(
            to_axum_path("/mlflow/logged-models/<model_id>"),
            "/mlflow/logged-models/{model_id}"
        );
        assert_eq!(
            to_axum_path("/mlflow/logged-models/<model_id>/tags/<tag_key>"),
            "/mlflow/logged-models/{model_id}/tags/{tag_key}"
        );
        assert_eq!(
            to_axum_path("/mlflow/experiments/create"),
            "/mlflow/experiments/create"
        );
    }
}
