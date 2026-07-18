//! Store-agnostic core of the `/get-artifact` download handler
//! (`handlers.py::get_artifact_handler` → `_send_artifact` →
//! `_create_artifact_file_response` / `_response_with_file_attachment_headers`).
//!
//! The run-artifact-URI resolution (looking up the run's `artifact_uri` in the
//! tracking store, the `mlflow-artifacts:` proxying, workspace prefixes) is
//! Phase 5 wiring done at the server level. What lives here is the reusable
//! part: given an already-resolved [`ArtifactRepo`] and a `path` query param,
//! validate the path and stream the file back with the exact headers Python
//! sends.

use axum::body::Body;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use mlflow_error::MlflowError;

use crate::mime::{content_disposition_attachment, guess_mime_type};
use crate::path_safety::validate_path_is_safe;
use crate::repo::ArtifactRepo;

/// Stream the artifact at `path` from `repo` as an attachment.
///
/// Mirrors `_send_artifact` / `_create_artifact_file_response`:
///  * `path` is run through [`validate_path_is_safe`] first (traversal → 400).
///  * content-type is guessed from the artifact name (`_guess_mime_type`).
///  * `Content-Disposition: attachment; filename=...` (always an attachment, to
///    prevent the browser rendering artifacts on our origin — the XSS guard in
///    the Python comment).
///  * `X-Content-Type-Options: nosniff`.
///  * a directory path → 400 `INVALID_PARAMETER_VALUE`
///    (`"Artifact path refers to a directory, not a file"`).
///
/// The body streams from the backend — no full-file buffering.
pub async fn send_artifact(repo: &dyn ArtifactRepo, path: &str) -> Result<Response, MlflowError> {
    let safe_path = validate_path_is_safe(path)?;

    // Mirror `_create_artifact_file_response`, which explicitly rejects a
    // directory with a 400 (`"...refers to a directory, not a file"`). Backends
    // differ in how a `get` on a directory fails (LocalFileSystem yields
    // `NotFound`, others an EISDIR-flavored generic error), so on any get
    // failure we disambiguate: if the path lists as a non-empty directory it's
    // the 400 case, otherwise the original error (a genuine
    // `RESOURCE_DOES_NOT_EXIST`) stands.
    let download = match repo.get(&safe_path).await {
        Ok(d) => d,
        Err(e) if is_directory_get_error(&e) || is_missing(&e) => {
            if path_is_directory(repo, &safe_path).await {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Artifact path refers to a directory, not a file: '{safe_path}'"
                )));
            }
            return Err(e);
        }
        Err(e) => return Err(e),
    };

    let mime = guess_mime_type(&safe_path);
    let filename = safe_path.rsplit('/').next().unwrap_or(&safe_path);
    let content_disposition = content_disposition_attachment(filename);

    let body = Body::from_stream(download.stream.map_err_into_axum());

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .map_err(|e| MlflowError::internal_error(format!("Failed to build response: {e}")))?;

    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&mime)
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&content_disposition)
            .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from(download.size.max(0) as u64),
    );
    headers.insert(
        "X-Content-Type-Options",
        HeaderValue::from_static("nosniff"),
    );
    Ok(response)
}

/// Convenience wrapper that turns any `MlflowError` into its wire response, so a
/// server handler can call `send_artifact(...).await.into_response_or_error()`.
pub async fn send_artifact_response(repo: &dyn ArtifactRepo, path: &str) -> Response {
    match send_artifact(repo, path).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

/// LocalFileSystem `get` on a directory surfaces a `NotFound`-or-generic error
/// with a filesystem message. We can't cheaply distinguish "missing" from "is a
/// directory" across every backend, so we treat the "Is a directory" generic
/// message as the directory case (Python's explicit 400). A true missing file
/// stays `RESOURCE_DOES_NOT_EXIST` (from the repo's `get`).
fn is_directory_get_error(e: &MlflowError) -> bool {
    e.message.contains("Is a directory") || e.message.contains("is a directory")
}

/// True when `e` is the repo's `RESOURCE_DOES_NOT_EXIST` (a `get` miss).
fn is_missing(e: &MlflowError) -> bool {
    e.error_code == mlflow_error::ErrorCode::ResourceDoesNotExist
}

/// A path is a directory iff listing it yields at least one entry (files and
/// dirs only exist implicitly via their contents in an object store).
async fn path_is_directory(repo: &dyn ArtifactRepo, path: &str) -> bool {
    repo.list(Some(path))
        .await
        .map(|entries| !entries.is_empty())
        .unwrap_or(false)
}

// --- stream adapter -------------------------------------------------------

use futures::stream::StreamExt;

/// Adapt a `Stream<Item = Result<Bytes, MlflowError>>` into the
/// `Result<Bytes, std::io::Error>` shape `axum::body::Body::from_stream` wants,
/// preserving the error message.
trait StreamMapErrExt: futures::Stream + Sized {
    fn map_err_into_axum(self) -> futures::stream::Map<Self, fn(Self::Item) -> AxumChunk>;
}

type AxumChunk = Result<bytes::Bytes, std::io::Error>;

impl<S> StreamMapErrExt for S
where
    S: futures::Stream<Item = Result<bytes::Bytes, MlflowError>> + Send + 'static,
{
    fn map_err_into_axum(self) -> futures::stream::Map<Self, fn(Self::Item) -> AxumChunk> {
        self.map(convert_chunk as fn(Result<bytes::Bytes, MlflowError>) -> AxumChunk)
    }
}

fn convert_chunk(item: Result<bytes::Bytes, MlflowError>) -> AxumChunk {
    item.map_err(|e| std::io::Error::other(e.message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::local_repo;
    use bytes::Bytes;
    use futures::stream;
    use http_body_util::BodyExt;
    use tempfile::TempDir;

    fn body_from(
        b: &'static [u8],
    ) -> futures::stream::BoxStream<'static, Result<Bytes, MlflowError>> {
        stream::once(async move { Ok(Bytes::from_static(b)) }).boxed()
    }

    #[tokio::test]
    async fn streams_file_with_headers() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        repo.put("logs/output.txt", body_from(b"hello"))
            .await
            .unwrap();

        let resp = send_artifact(&repo, "logs/output.txt").await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[header::CONTENT_TYPE], "text/plain");
        assert_eq!(
            resp.headers()[header::CONTENT_DISPOSITION],
            "attachment; filename=output.txt"
        );
        assert_eq!(resp.headers()["x-content-type-options"], "nosniff");
        assert_eq!(resp.headers()[header::CONTENT_LENGTH], "5");

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"hello");
    }

    #[tokio::test]
    async fn traversal_path_is_rejected() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        let err = send_artifact(&repo, "../etc/passwd").await.err().unwrap();
        assert_eq!(
            err.error_code,
            mlflow_error::ErrorCode::InvalidParameterValue
        );
        assert_eq!(err.message, "Invalid path");
    }

    #[tokio::test]
    async fn missing_file_is_resource_does_not_exist() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        let err = send_artifact(&repo, "nope.bin").await.err().unwrap();
        assert_eq!(
            err.error_code,
            mlflow_error::ErrorCode::ResourceDoesNotExist
        );
    }

    #[tokio::test]
    async fn directory_path_is_400() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        repo.put("d/inner.txt", body_from(b"x")).await.unwrap();
        let err = send_artifact(&repo, "d").await.err().unwrap();
        assert_eq!(
            err.error_code,
            mlflow_error::ErrorCode::InvalidParameterValue
        );
        assert!(err.message.contains("refers to a directory"));
    }
}
