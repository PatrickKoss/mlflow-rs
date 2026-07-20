//! axum `Router` for the `MlflowArtifactsService` HTTP surface (T5.2 / §3.11).
//!
//! Routes are built **relative** — the server mounts this under both
//! `/api/2.0` and `/ajax-api/2.0` (and any `MLFLOW_STATIC_PREFIX`) in Phase 5,
//! exactly like `RouteSpec::expand`. The concrete paths here come from
//! `mlflow_artifacts.proto`'s `MlflowArtifactsService` endpoints:
//!
//! | Method | Path | Handler |
//! |---|---|---|
//! | GET    | `/mlflow-artifacts/artifacts/{*path}` | download (stream) |
//! | PUT    | `/mlflow-artifacts/artifacts/{*path}` | upload (stream)   |
//! | DELETE | `/mlflow-artifacts/artifacts/{*path}` | delete            |
//! | GET    | `/mlflow-artifacts/artifacts`         | list (`?path=`)   |
//! | POST   | `/mlflow-artifacts/mpu/create/{*path}`   | create MPU     |
//! | POST   | `/mlflow-artifacts/mpu/complete/{*path}` | complete MPU   |
//! | POST   | `/mlflow-artifacts/mpu/abort/{*path}`    | abort MPU      |
//! | GET    | `/mlflow-artifacts/presigned/{*path}`    | presigned URL  |
//!
//! ## What is implemented here vs deferred to Phase 5
//!
//! Implemented: the full local-FS surface (download/upload/list/delete +
//! multipart, which returns `NOT_IMPLEMENTED` for local like Python). JSON
//! responses use `mlflow-proto`'s codec; errors use `mlflow-error`.
//!
//! Deferred to Phase 5 server wiring (NOT this crate's job):
//!  * the `--serve-artifacts` gate (`_disable_unless_serve_artifacts` → 503) —
//!    the server decides whether to mount this router at all / wrap it;
//!  * workspace prefixing (`_get_workspace_scoped_repo_path_if_enabled`);
//!  * `/get-artifact` & `/model-versions/get-artifact` (they need the tracking /
//!    registry store to resolve a run/model to a repo — see
//!    [`crate::get_artifact`] for the store-agnostic streaming core they reuse);
//!  * presigned URLs (local FS has none → `NOT_IMPLEMENTED`).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::stream::StreamExt;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow::artifacts as pb;
use serde::Deserialize;

use crate::get_artifact::send_artifact_response;
use crate::path_safety::validate_path_is_safe;
use crate::repo::{ArtifactRepo, MultipartUploadPart};

/// Shared handler state: the resolved proxied-artifacts [`ArtifactRepo`]
/// (`--artifacts-destination`).
#[derive(Clone)]
pub struct ArtifactsState {
    repo: Arc<dyn ArtifactRepo>,
}

impl ArtifactsState {
    pub fn new(repo: Arc<dyn ArtifactRepo>) -> Self {
        Self { repo }
    }
}

/// Build the `MlflowArtifactsService` router rooted at a local artifact
/// directory. Convenience for the common single-local-store deployment; the
/// server can also build [`ArtifactsState`] from any [`ArtifactRepo`] and call
/// [`artifacts_router_with_state`].
pub fn artifacts_router(repo_root: &std::path::Path) -> Result<Router, MlflowError> {
    let repo = crate::repo::local_repo(repo_root)?;
    Ok(artifacts_router_with_state(ArtifactsState::new(Arc::new(
        repo,
    ))))
}

/// Build the router over an already-constructed [`ArtifactsState`].
pub fn artifacts_router_with_state(state: ArtifactsState) -> Router {
    Router::new()
        .route("/mlflow-artifacts/artifacts", get(list_artifacts))
        .route(
            "/mlflow-artifacts/artifacts/{*artifact_path}",
            get(download_artifact)
                .put(upload_artifact)
                .delete(delete_artifact),
        )
        .route(
            "/mlflow-artifacts/mpu/create/{*artifact_path}",
            post(create_multipart_upload),
        )
        .route(
            "/mlflow-artifacts/mpu/complete/{*artifact_path}",
            post(complete_multipart_upload),
        )
        .route(
            "/mlflow-artifacts/mpu/abort/{*artifact_path}",
            post(abort_multipart_upload),
        )
        .route(
            "/mlflow-artifacts/presigned/{*artifact_path}",
            get(get_presigned_download_url),
        )
        .with_state(state)
}

/// Render a proto-shaped response as MLflow pretty JSON, mirroring Python's
/// `Response(mimetype="application/json"); set_data(message_to_json(msg))`.
fn json_response<M: prost::Message>(msg: &M, type_name: &str) -> Response {
    match mlflow_proto::to_mlflow_json(msg, type_name) {
        Ok(body) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response(),
        Err(e) => MlflowError::internal_error(format!("Failed to encode response JSON: {e}"))
            .into_response(),
    }
}

// --- download -------------------------------------------------------------

async fn download_artifact(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
) -> Response {
    // `send_artifact_response` validates the path and streams the file with the
    // right headers; it returns an already-rendered error response on failure.
    send_artifact_response(state.repo.as_ref(), &artifact_path).await
}

// --- upload ---------------------------------------------------------------

async fn upload_artifact(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
    body: Body,
) -> Response {
    let result = async {
        let safe = validate_path_is_safe(&artifact_path)?;
        // Stream the request body straight into the repo — no buffering.
        let stream = body
            .into_data_stream()
            .map(|chunk| {
                chunk.map_err(|e| MlflowError::internal_error(format!("Upload read error: {e}")))
            })
            .boxed();
        state.repo.put(&safe, stream).await?;
        Ok::<_, MlflowError>(())
    }
    .await;

    match result {
        Ok(()) => json_response(
            &pb::upload_artifact::Response {},
            "mlflow.artifacts.UploadArtifact.Response",
        ),
        Err(e) => e.into_response(),
    }
}

// --- list -----------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ListQuery {
    path: Option<String>,
}

async fn list_artifacts(
    State(state): State<ArtifactsState>,
    Query(query): Query<ListQuery>,
) -> Response {
    let result = async {
        // Mirrors: validate the path iff present (`HasField("path")`).
        let validated = match query.path.as_deref() {
            Some(p) => Some(validate_path_is_safe(p)?),
            None => None,
        };
        let files = state.repo.list(validated.as_deref()).await?;
        // Python reduces each returned path to its basename before building the
        // response `FileInfo` (`posixpath.basename(file_info.path)`).
        let proto_files = files
            .into_iter()
            .map(|f| pb::FileInfo {
                path: Some(basename(&f.path).to_string()),
                is_dir: Some(f.is_dir),
                file_size: f.file_size,
            })
            .collect();
        Ok::<_, MlflowError>(pb::list_artifacts::Response { files: proto_files })
    }
    .await;

    match result {
        Ok(resp) => json_response(&resp, "mlflow.artifacts.ListArtifacts.Response"),
        Err(e) => e.into_response(),
    }
}

fn basename(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((_, name)) => name,
        None => path,
    }
}

// --- delete ---------------------------------------------------------------

async fn delete_artifact(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
) -> Response {
    let result = async {
        let safe = validate_path_is_safe(&artifact_path)?;
        state.repo.delete(&safe).await?;
        Ok::<_, MlflowError>(())
    }
    .await;

    match result {
        Ok(()) => json_response(
            &pb::delete_artifact::Response {},
            "mlflow.artifacts.DeleteArtifact.Response",
        ),
        Err(e) => e.into_response(),
    }
}

// --- multipart ------------------------------------------------------------

async fn create_multipart_upload(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
    body: Body,
) -> Response {
    let result = async {
        let artifact_path = validate_path_is_safe(&artifact_path)?;
        let req: pb::CreateMultipartUpload =
            parse_json_body(body, "mlflow.artifacts.CreateMultipartUpload").await?;
        let path = crate::multipart_upload_path(&req.path.unwrap_or_default(), &artifact_path);
        let num_parts = req.num_parts.unwrap_or_default();
        let res = state.repo.create_multipart_upload(&path, num_parts).await?;
        Ok::<_, MlflowError>(pb::create_multipart_upload::Response {
            upload_id: Some(res.upload_id),
            credentials: res
                .credentials
                .into_iter()
                .map(|c| pb::MultipartUploadCredential {
                    url: Some(c.url),
                    part_number: Some(c.part_number),
                    headers: c.headers.into_iter().collect(),
                })
                .collect(),
        })
    }
    .await;

    match result {
        Ok(resp) => json_response(&resp, "mlflow.artifacts.CreateMultipartUpload.Response"),
        Err(e) => e.into_response(),
    }
}

async fn complete_multipart_upload(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
    body: Body,
) -> Response {
    let result = async {
        let artifact_path = validate_path_is_safe(&artifact_path)?;
        let req: pb::CompleteMultipartUpload =
            parse_json_body(body, "mlflow.artifacts.CompleteMultipartUpload").await?;
        let path = crate::multipart_upload_path(&req.path.unwrap_or_default(), &artifact_path);
        let upload_id = req.upload_id.unwrap_or_default();
        let parts: Vec<MultipartUploadPart> = req
            .parts
            .into_iter()
            .map(|p| MultipartUploadPart {
                part_number: p.part_number.unwrap_or_default(),
                etag: p.etag.unwrap_or_default(),
                url: p.url.unwrap_or_default(),
            })
            .collect();
        state
            .repo
            .complete_multipart_upload(&path, &upload_id, &parts)
            .await?;
        Ok::<_, MlflowError>(())
    }
    .await;

    match result {
        Ok(()) => json_response(
            &pb::complete_multipart_upload::Response {},
            "mlflow.artifacts.CompleteMultipartUpload.Response",
        ),
        Err(e) => e.into_response(),
    }
}

async fn abort_multipart_upload(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
    body: Body,
) -> Response {
    let result = async {
        let artifact_path = validate_path_is_safe(&artifact_path)?;
        let req: pb::AbortMultipartUpload =
            parse_json_body(body, "mlflow.artifacts.AbortMultipartUpload").await?;
        let path = crate::multipart_upload_path(&req.path.unwrap_or_default(), &artifact_path);
        let upload_id = req.upload_id.unwrap_or_default();
        state.repo.abort_multipart_upload(&path, &upload_id).await?;
        Ok::<_, MlflowError>(())
    }
    .await;

    match result {
        Ok(()) => json_response(
            &pb::abort_multipart_upload::Response {},
            "mlflow.artifacts.AbortMultipartUpload.Response",
        ),
        Err(e) => e.into_response(),
    }
}

async fn get_presigned_download_url(
    State(state): State<ArtifactsState>,
    Path(artifact_path): Path<String>,
) -> Response {
    let result = async {
        let path = validate_path_is_safe(&artifact_path)?;
        let ttl = crate::presigned_download_ttl_seconds()?;
        state.repo.get_download_presigned_url(&path, ttl).await
    }
    .await;
    match result {
        Ok(result) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::json!({
                "url": result.url,
                "headers": result.headers.into_iter().collect::<std::collections::BTreeMap<_, _>>(),
                "file_size": result.file_size,
            })
            .to_string(),
        )
            .into_response(),
        Err(e) => e.into_response(),
    }
}

/// Read the request body fully (control-message bodies are small) and parse it
/// with the MLflow JSON codec (unknown-field tolerant), mirroring
/// `_get_request_message`. Unlike artifact bytes, these proto control messages
/// are bounded, so buffering them is fine.
async fn parse_json_body<M: prost::Message + Default>(
    body: Body,
    type_name: &str,
) -> Result<M, MlflowError> {
    let bytes = axum::body::to_bytes(body, MAX_CONTROL_BODY_BYTES)
        .await
        .map_err(|e| MlflowError::invalid_parameter_value(format!("Failed to read body: {e}")))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| MlflowError::invalid_parameter_value("Request body is not valid UTF-8"))?;
    mlflow_proto::from_mlflow_json(text, type_name)
        .map_err(|e| MlflowError::invalid_parameter_value(format!("Malformed request: {e}")))
}

/// Cap for control-message request bodies (multipart create/complete/abort).
/// Artifact bytes never go through here — they stream via `upload_artifact`.
const MAX_CONTROL_BODY_BYTES: usize = 16 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn router(dir: &std::path::Path) -> Router {
        artifacts_router(dir).unwrap()
    }

    async fn body_string(resp: Response) -> String {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn upload_list_download_delete_roundtrip() {
        let dir = TempDir::new().unwrap();
        let app = router(dir.path());

        // Upload.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::PUT)
                    .uri("/mlflow-artifacts/artifacts/exp/run/data.txt")
                    .body(Body::from("payload-bytes"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "{}");

        // List the parent dir.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/mlflow-artifacts/artifacts?path=exp/run")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let listing = body_string(resp).await;
        assert!(listing.contains("\"path\": \"data.txt\""), "{listing}");
        assert!(listing.contains("\"is_dir\": false"), "{listing}");
        assert!(listing.contains("\"file_size\": 13"), "{listing}");

        // Download.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/mlflow-artifacts/artifacts/exp/run/data.txt")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()["content-type"], "text/plain");
        assert_eq!(body_string(resp).await, "payload-bytes");

        // Delete.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/mlflow-artifacts/artifacts/exp/run/data.txt")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Gone.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/mlflow-artifacts/artifacts/exp/run/data.txt")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_string(resp).await;
        assert!(body.contains("RESOURCE_DOES_NOT_EXIST"), "{body}");
    }

    #[tokio::test]
    async fn traversal_download_is_400_with_python_error() {
        let dir = TempDir::new().unwrap();
        let app = router(dir.path());
        // `%2e%2e` -> `..` after decode; axum matches the `{*path}` capture.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/mlflow-artifacts/artifacts/%2e%2e/secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_string(resp).await;
        assert!(body.contains("INVALID_PARAMETER_VALUE"), "{body}");
        assert!(body.contains("Invalid path"), "{body}");
    }

    #[tokio::test]
    async fn multipart_create_is_not_implemented() {
        let dir = TempDir::new().unwrap();
        let app = router(dir.path());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/mlflow-artifacts/mpu/create/big.bin")
                    .body(Body::from(r#"{"path": "big.bin", "num_parts": 3}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        // NOT_IMPLEMENTED maps to HTTP 501.
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = body_string(resp).await;
        assert!(body.contains("NOT_IMPLEMENTED"), "{body}");
    }

    #[tokio::test]
    async fn list_empty_dir_returns_empty_files() {
        let dir = TempDir::new().unwrap();
        let app = router(dir.path());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/mlflow-artifacts/artifacts")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Empty repeated field is omitted → `{}`.
        assert_eq!(body_string(resp).await, "{}");
    }
}
