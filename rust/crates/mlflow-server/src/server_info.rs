//! `GET /(api|ajax-api)/3.0/mlflow/server-info` (plan T11.5, D5).
//!
//! Hand-registered exactly like `/graphql` (`lib.rs` `register_proto_routes`) —
//! this is **not** a proto `ROUTE_TABLE` route. Mirrors `_get_server_info`
//! (`mlflow/server/handlers.py:6586-6616`), registered on both URL prefixes via
//! `_get_paths("/mlflow/server-info", version=3)`
//! (`mlflow/server/handlers.py:6797-6802`), `["GET"]` only.
//!
//! Response shape (`jsonify({...})`, `handlers.py:6612-6616`) — exactly three
//! fields, no more:
//!
//! ```json
//! {"store_type": "SqlStore" | "FileStore" | null, "workspaces_enabled": bool, "trace_archival_enabled": bool}
//! ```
//!
//! `store_type` is `"FileStore"` / `"SqlStore"` / `None` depending on
//! `isinstance(store, FileStore | SqlAlchemyStore)`. The Rust server has no
//! `FileStore` backend (`TrackingStore` is always a SQL-backed store), so this
//! always reports `"SqlStore"` — this handler is only reachable when a
//! `TrackingStore` is wired into `AppState` (`register_proto_routes` only
//! registers this router when `state: Some(..)`), so there is no `None` case
//! in practice for this server.
//!
//! `workspaces_enabled` mirrors `MLFLOW_ENABLE_WORKSPACES.get()`: whether the
//! server was started with workspace support, i.e.
//! [`AppState::workspace_store`] is `Some`.
//!
//! `trace_archival_enabled` mirrors
//! `trace_archival_config is not None and trace_archival_config.enabled and
//! _store_supports_trace_archival(store)` (`handlers.py:6600-6604`). Trace
//! archival (`MLFLOW_TRACE_ARCHIVAL_CONFIG` + YAML config file,
//! `mlflow/tracing/trace_archival_config.py`) is not implemented in the Rust
//! server at all (no config-file loading, no archival job), so this always
//! reports `false` — the byte-parity default for every deployment that
//! doesn't configure trace archival, which is the only case this server
//! supports today.
//!
//! There is deliberately **no** `auth_enabled` field: Python's `_get_server_info`
//! does not emit one, and the UI's `ServerInfoResponse` interface
//! (`mlflow/server/js/src/experiment-tracking/hooks/useServerInfo.tsx:9-13`)
//! only declares `store_type` / `workspaces_enabled` / `trace_archival_enabled`.
//! The plan's D5 line ("auth, workspaces...") describes deployment-consistency
//! intent, not a wire field — auth-gated UI reads a different signal (the
//! authenticated-user/whoami surface), not `server-info`.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::state::AppState;

/// `_get_server_info` (`handlers.py:6586-6616`).
pub async fn server_info(State(state): State<AppState>) -> impl IntoResponse {
    let payload = json!({
        "store_type": "SqlStore",
        "workspaces_enabled": state.workspace_store().is_some(),
        "trace_archival_enabled": false,
    });
    json_ok(&payload)
}

/// `jsonify(...)`: `200` with `Content-Type: application/json` and a compact
/// body (Flask's `jsonify` does not pretty-print by default).
fn json_ok(value: &serde_json::Value) -> Response {
    let body = serde_json::to_string(value).expect("JSON value serializes");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .expect("valid response")
}
