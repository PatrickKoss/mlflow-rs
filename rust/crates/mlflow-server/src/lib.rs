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

pub mod config;
pub mod metrics;
pub mod routes;

use axum::extract::MatchedPath;
use axum::http::Request;
use axum::middleware;
use axum::routing::get;
use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::trace::TraceLayer;
use tracing::info_span;

pub use config::{Cli, ServerConfig, StaticPrefixError};

/// Builds the full application `Router`, including request logging and
/// metrics middleware, nested under `config.static_prefix` when set
/// (mirroring `_add_static_prefix`, `mlflow/server/handlers.py:6731-6734`,
/// which prepends the prefix to every registered route).
pub fn build_app(config: &ServerConfig) -> Router {
    let metrics_handle = metrics::install_recorder();
    build_app_with_recorder(config, metrics_handle)
}

/// Same as [`build_app`], but takes an already-installed
/// [`PrometheusHandle`] instead of installing the global recorder. Exists so
/// tests can build multiple `Router`s in the same process without hitting
/// "recorder already installed" panics from `metrics-exporter-prometheus`.
pub fn build_app_with_recorder(config: &ServerConfig, metrics_handle: PrometheusHandle) -> Router {
    let api = Router::new()
        .route("/health", get(routes::health))
        .route("/version", get(routes::version))
        .route(
            "/metrics",
            get(move || routes::metrics(metrics_handle.clone())),
        )
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
        }
    }

    fn test_app(static_prefix: Option<&str>) -> Router {
        let handle = PrometheusBuilder::new().build_recorder().handle();
        build_app_with_recorder(&test_config(static_prefix), handle)
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
}
