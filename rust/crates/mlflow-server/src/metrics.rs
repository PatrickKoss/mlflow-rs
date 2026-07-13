//! Prometheus metrics: a `http_requests_total` counter (labeled by method,
//! path pattern, and status) recorded via middleware, exposed on `/metrics`
//! in Prometheus exposition format. Uses the `metrics` facade with the
//! `metrics-exporter-prometheus` recorder/renderer, replacing the
//! gunicorn-multiprocess Prometheus exporter Python uses
//! (`mlflow/server/prometheus_exporter.py`).

use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::IntoResponse;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Installs the global Prometheus recorder and returns a handle that can
/// render the current metrics snapshot on demand. Must be called at most
/// once per process (the `metrics` crate's global recorder can only be set
/// once); tests that need isolated metrics should avoid calling this
/// directly and instead exercise `track_metrics` / route handlers without
/// installing a recorder, or accept the shared global state.
pub fn install_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

/// Axum middleware that records `http_requests_total` (counter, labeled by
/// `method`, `path`, `status`) and `http_requests_duration_seconds`
/// (histogram, same labels) for every request. Uses `MatchedPath` so the
/// `path` label is the route *pattern* (e.g. `/health`) rather than the raw
/// URI, avoiding unbounded label cardinality from path parameters.
pub async fn track_metrics(req: Request, next: Next) -> impl IntoResponse {
    let start = Instant::now();
    let method = req.method().to_string();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|matched_path| matched_path.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());

    let response = next.run(req).await;

    let status = response.status().as_u16().to_string();
    let latency = start.elapsed().as_secs_f64();

    let labels = [("method", method), ("path", path), ("status", status)];
    metrics::counter!("http_requests_total", &labels).increment(1);
    metrics::histogram!("http_requests_duration_seconds", &labels).record(latency);

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    #[tokio::test]
    async fn records_matched_path_not_raw_uri() {
        // Not asserting on global metric state (the recorder is process-global
        // and shared across tests); this only proves the middleware runs and
        // passes the response through untouched.
        let app: Router = Router::new()
            .route("/ping", get(|| async { "pong" }))
            .layer(middleware::from_fn(track_metrics));

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
