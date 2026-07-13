//! Route handlers for the endpoints implemented so far: `/health`,
//! `/version`, `/metrics`. Behavior mirrors `mlflow/server/__init__.py`'s
//! Flask handlers exactly (see doc comments on each handler).

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::PrometheusHandle;

/// The MLflow version, parsed at build time from `mlflow/version.py` (see
/// `build.rs`) so the binary needs no runtime file access.
pub const MLFLOW_VERSION: &str = env!("MLFLOW_VERSION");

/// `GET /health`. Mirrors `mlflow/server/__init__.py:99-101`:
/// `return "OK", 200`. Flask serializes a bare string return with
/// `Content-Type: text/html; charset=utf-8`, so this handler matches that
/// content type rather than defaulting to `text/plain`.
pub async fn health() -> impl IntoResponse {
    text_ok("OK")
}

/// `GET /version`. Mirrors `mlflow/server/__init__.py:105-107`:
/// `return VERSION, 200`, i.e. the plain version string of the running
/// MLflow release.
pub async fn version() -> impl IntoResponse {
    text_ok(MLFLOW_VERSION)
}

/// Builds a 200 response with Flask's default content type for a bare
/// string handler return (`text/html; charset=utf-8`).
fn text_ok(body: &'static str) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(body))
        .expect("static response is always valid")
}

/// `GET /metrics`. Renders the current Prometheus exposition-format
/// snapshot. There's no single Python equivalent line to mirror here (the
/// gunicorn multiprocess Prometheus exporter wires this up dynamically,
/// `mlflow/server/prometheus_exporter.py`); this matches the conventional
/// Prometheus content type instead.
pub async fn metrics(handle: PrometheusHandle) -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(handle.render()))
        .expect("static response is always valid")
}

// `MLFLOW_VERSION` is exercised end-to-end (against the actual
// `mlflow/version.py` VERSION line) by `version_matches_mlflow_version_py`
// in `lib.rs`'s test module, which builds the app and asserts on the
// `/version` response body.
