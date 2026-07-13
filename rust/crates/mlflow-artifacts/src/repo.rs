//! The [`ArtifactRepo`] abstraction over the `object_store` crate.
//!
//! Mirrors the semantics of `mlflow/store/artifact/local_artifact_repo.py` and
//! `artifact_repo.py` that the server actually depends on: streaming `get`, a
//! streaming `put` from an arbitrary byte-stream (so a 5 GB upload never
//! buffers — the plan's AC), non-recursive `list` with Python's
//! `is_dir`/`file_size` semantics and `path`-sorted ordering, recursive
//! `delete`, and multipart create/complete/abort shaped like
//! `MlflowArtifactsService` (the local backend returns `NOT_IMPLEMENTED`,
//! matching `LocalArtifactRepository` — which does NOT implement
//! `MultipartUploadMixin`, so `_validate_support_multipart_upload` raises).
//!
//! ## Why `object_store`
//!
//! One API for local FS / S3 / GCS / Azure. v1 wires only the local `fs`
//! backend (no cloud SDK deps). [`factory::repo_from_uri`] is the single
//! resolution point where cloud schemes get plugged in later (feature-gated
//! `object_store` backends), so Phase 5 just fills in the `match` arms.
//!
//! ## Path model
//!
//! Every method takes a *repo-relative* artifact path (already run through
//! [`crate::path_safety::validate_path_is_safe`] by the caller). We convert it
//! to an `object_store::path::Path` (which normalizes to `/`-delimited,
//! rejects `..`), rooted at the repo's artifact root. `object_store`'s `Path`
//! is itself traversal-safe, giving defense in depth on top of our validator.

use std::sync::Arc;

use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use mlflow_error::MlflowError;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjPath;
use object_store::{GetOptions, ObjectStore, PutMultipartOptions, PutPayload};

/// One entry of a non-recursive artifact listing. Field semantics match
/// `mlflow.entities.FileInfo` / the `FileInfo` proto: `is_dir` true for
/// directories, `file_size` present only for files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactFileInfo {
    /// Repo-relative path (posix, `/`-delimited). For the proxy list endpoint
    /// the handler later reduces this to a basename (matching Python).
    pub path: String,
    pub is_dir: bool,
    /// Byte size, `None` for directories (matches proto `file_size` being unset
    /// for dirs).
    pub file_size: Option<i64>,
}

/// A streaming download: the total size (for `Content-Length`) plus a stream of
/// byte chunks. The stream is lazy — bytes are read from the backend on demand,
/// so the whole file never lands in memory.
pub struct ArtifactDownload {
    pub size: i64,
    pub stream: BoxStream<'static, Result<Bytes, MlflowError>>,
}

/// The artifact repository port. Object-safe so the router can hold a
/// `Arc<dyn ArtifactRepo>` chosen at startup by [`factory::repo_from_uri`].
#[async_trait::async_trait]
pub trait ArtifactRepo: Send + Sync {
    /// Stream a single file at `path`. Errors `RESOURCE_DOES_NOT_EXIST` if
    /// absent (matching `LocalArtifactRepository._download_file`), or
    /// `INVALID_PARAMETER_VALUE` if `path` denotes a directory.
    async fn get(&self, path: &str) -> Result<ArtifactDownload, MlflowError>;

    /// Stream-upload the bytes from `body` to `path`, creating parent
    /// "directories" as needed. Must not buffer the whole body.
    async fn put(
        &self,
        path: &str,
        body: BoxStream<'static, Result<Bytes, MlflowError>>,
    ) -> Result<(), MlflowError>;

    /// Non-recursive listing of `path` (a directory). Returns entries sorted by
    /// `path`, mirroring `LocalArtifactRepository.list_artifacts`. A missing or
    /// non-directory `path` yields an empty list (Python returns `[]`). `None`
    /// lists the repo root.
    async fn list(&self, path: Option<&str>) -> Result<Vec<ArtifactFileInfo>, MlflowError>;

    /// Recursively delete `path` (a file or directory subtree). Deleting an
    /// absent path is a no-op, matching `LocalArtifactRepository.delete_artifacts`.
    async fn delete(&self, path: &str) -> Result<(), MlflowError>;

    /// Multipart upload — shaped like the `MlflowArtifactsService` proto. The
    /// local backend returns `NOT_IMPLEMENTED` (parity with Python, whose
    /// `LocalArtifactRepository` lacks `MultipartUploadMixin`). Cloud backends
    /// will override.
    async fn create_multipart_upload(
        &self,
        _path: &str,
        _num_parts: i64,
    ) -> Result<CreateMultipartUploadResult, MlflowError> {
        Err(multipart_not_supported())
    }

    async fn complete_multipart_upload(
        &self,
        _path: &str,
        _upload_id: &str,
        _parts: &[MultipartUploadPart],
    ) -> Result<(), MlflowError> {
        Err(multipart_not_supported())
    }

    async fn abort_multipart_upload(
        &self,
        _path: &str,
        _upload_id: &str,
    ) -> Result<(), MlflowError> {
        Err(multipart_not_supported())
    }
}

/// Result of `create_multipart_upload`, shaped like the proto response.
#[derive(Debug, Clone)]
pub struct CreateMultipartUploadResult {
    pub upload_id: String,
    pub credentials: Vec<MultipartUploadCredential>,
}

#[derive(Debug, Clone)]
pub struct MultipartUploadCredential {
    pub url: String,
    pub part_number: i64,
    pub headers: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct MultipartUploadPart {
    pub part_number: i64,
    pub etag: String,
    pub url: String,
}

/// Mirrors `_UnsupportedMultipartUploadException` (`handlers.py`): the message
/// and `NOT_IMPLEMENTED` code the local backend surfaces.
fn multipart_not_supported() -> MlflowError {
    MlflowError::not_implemented("Multipart upload is not supported for the current artifact store")
}

/// An [`ArtifactRepo`] backed by any `object_store` implementation, rooted at a
/// prefix within that store.
pub struct ObjectStoreRepo {
    store: Arc<dyn ObjectStore>,
    /// Prefix within the store that is this repo's artifact root (empty for the
    /// store root). All method paths are joined under this.
    root: ObjPath,
}

impl ObjectStoreRepo {
    pub fn new(store: Arc<dyn ObjectStore>, root: ObjPath) -> Self {
        Self { store, root }
    }

    /// Join a repo-relative path onto the root, producing a full store path.
    /// `object_store::Path::from` normalizes and rejects `..`, so this is
    /// traversal-safe (defense in depth on top of `validate_path_is_safe`).
    fn full_path(&self, rel: &str) -> ObjPath {
        let mut p = self.root.clone();
        for part in rel.split('/').filter(|s| !s.is_empty()) {
            p = p.join(part);
        }
        p
    }
}

#[async_trait::async_trait]
impl ArtifactRepo for ObjectStoreRepo {
    async fn get(&self, path: &str) -> Result<ArtifactDownload, MlflowError> {
        let full = self.full_path(path);
        let result = match self.store.get_opts(&full, GetOptions::default()).await {
            Ok(r) => r,
            Err(object_store::Error::NotFound { .. }) => {
                return Err(MlflowError::resource_does_not_exist(format!(
                    "No such artifact: '{path}'"
                )));
            }
            Err(e) => return Err(store_error(e)),
        };
        let size = result.meta.size as i64;
        // `into_stream` yields backend chunks lazily — no full-file buffering.
        let stream = result
            .into_stream()
            .map(|chunk| chunk.map_err(store_error))
            .boxed();
        Ok(ArtifactDownload { size, stream })
    }

    async fn put(
        &self,
        path: &str,
        body: BoxStream<'static, Result<Bytes, MlflowError>>,
    ) -> Result<(), MlflowError> {
        let full = self.full_path(path);
        // `put_multipart` streams parts to the backend as the body arrives; we
        // pump the request-body stream chunk-by-chunk into it, so peak memory
        // is bounded by one buffered part, not the whole upload.
        let mut upload = self
            .store
            .put_multipart_opts(&full, PutMultipartOptions::default())
            .await
            .map_err(store_error)?;
        let mut body = body;
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            upload
                .put_part(PutPayload::from_bytes(chunk))
                .await
                .map_err(store_error)?;
        }
        upload.complete().await.map_err(store_error)?;
        Ok(())
    }

    async fn list(&self, path: Option<&str>) -> Result<Vec<ArtifactFileInfo>, MlflowError> {
        let prefix = match path {
            Some(p) if !p.is_empty() => self.full_path(p),
            _ => self.root.clone(),
        };
        // Non-recursive: `list_with_delimiter` gives immediate children —
        // `objects` (files) + `common_prefixes` (subdirectories).
        let listing = match self.store.list_with_delimiter(Some(&prefix)).await {
            Ok(l) => l,
            // A path that isn't a directory → empty list (Python returns []).
            Err(object_store::Error::NotFound { .. }) => return Ok(Vec::new()),
            Err(e) => return Err(store_error(e)),
        };

        let mut infos = Vec::new();
        for obj in listing.objects {
            let rel = self.strip_root(&obj.location);
            // Skip in-flight temp uploads (parity with the `_TEMP_ARTIFACT_PREFIX`
            // filter in `LocalArtifactRepository.list_artifacts`).
            if basename(&rel).starts_with(TEMP_ARTIFACT_PREFIX) {
                continue;
            }
            infos.push(ArtifactFileInfo {
                path: rel,
                is_dir: false,
                file_size: Some(obj.size as i64),
            });
        }
        for dir in listing.common_prefixes {
            let rel = self.strip_root(&dir);
            infos.push(ArtifactFileInfo {
                path: rel,
                is_dir: true,
                file_size: None,
            });
        }
        // `sorted(infos, key=lambda f: f.path)`.
        infos.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(infos)
    }

    async fn delete(&self, path: &str) -> Result<(), MlflowError> {
        let full = self.full_path(path);
        // List the subtree under `full` (the directory case).
        let mut locations = Vec::new();
        let mut stream = self.store.list(Some(&full));
        while let Some(meta) = stream.next().await {
            locations.push(meta.map_err(store_error)?.location);
        }
        // If the subtree is empty, `full` is either a single file or absent —
        // add it as a direct target. When it's a non-empty prefix we only
        // delete the listed leaf objects (LocalFileSystem prunes the now-empty
        // directories itself; cloud stores have no real directories).
        if locations.is_empty() {
            locations.push(full);
        }
        // `delete_stream` deletes concurrently; both `NotFound` (absent path —
        // Python's delete is a no-op) and an "is a directory" generic error
        // (when `full` was a directory that had no leaf children of its own)
        // are benign.
        let to_delete = futures::stream::iter(locations.into_iter().map(Ok)).boxed();
        let mut results = self.store.delete_stream(to_delete);
        while let Some(res) = results.next().await {
            match res {
                Ok(_) | Err(object_store::Error::NotFound { .. }) => {}
                Err(e) if is_directory_error(&e) => {}
                Err(e) => return Err(store_error(e)),
            }
        }
        Ok(())
    }
}

impl ObjectStoreRepo {
    /// Strip the repo root prefix from a full store location, yielding the
    /// repo-relative path.
    fn strip_root(&self, loc: &ObjPath) -> String {
        let full = loc.as_ref();
        let root = self.root.as_ref();
        if root.is_empty() {
            full.to_string()
        } else if let Some(rest) = full.strip_prefix(root) {
            rest.trim_start_matches('/').to_string()
        } else {
            full.to_string()
        }
    }
}

const TEMP_ARTIFACT_PREFIX: &str = ".artifact.uploading.";

fn basename(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((_, name)) => name,
        None => path,
    }
}

/// Map an `object_store` error to an `MlflowError`. Not-found is handled at call
/// sites (it carries endpoint-specific messages); everything else is an internal
/// error, matching how Flask would surface an unexpected repo exception (500).
fn store_error(e: object_store::Error) -> MlflowError {
    MlflowError::internal_error(format!("Artifact store error: {e}"))
}

/// True when `e` is a LocalFileSystem "Is a directory" (EISDIR) generic error —
/// benign during subtree deletion (the directory entry itself is pruned once
/// its contents are gone). `object_store` surfaces this as a `Generic` error, so
/// we match on the message.
fn is_directory_error(e: &object_store::Error) -> bool {
    e.to_string().contains("Is a directory")
}

/// Construct the local-filesystem [`LocalFileSystem`] store rooted at
/// `artifact_dir` (which must exist). Used by [`factory::repo_from_uri`] and by
/// the router factory.
pub fn local_repo(artifact_dir: &std::path::Path) -> Result<ObjectStoreRepo, MlflowError> {
    let store = LocalFileSystem::new_with_prefix(artifact_dir).map_err(|e| {
        MlflowError::internal_error(format!(
            "Failed to open local artifact store at {}: {e}",
            artifact_dir.display()
        ))
    })?;
    Ok(ObjectStoreRepo::new(Arc::new(store), ObjPath::default()))
}

/// URI → repo resolution. This is the single seam where cloud backends are
/// added later (Phase 5). Today only local paths / `file:` URIs resolve; cloud
/// schemes return a `NOT_IMPLEMENTED` describing what feature to enable.
pub mod factory {
    use super::*;

    /// Resolve an artifact-store URI to an [`ArtifactRepo`]. Mirrors
    /// `get_artifact_repository`'s dispatch on scheme, but only for the schemes
    /// the Rust server owns in v1.
    pub fn repo_from_uri(uri: &str) -> Result<Arc<dyn ArtifactRepo>, MlflowError> {
        let scheme = uri.split_once("://").map(|(s, _)| s);
        match scheme {
            // Local filesystem: bare path or `file://` URI.
            None | Some("file") => {
                let path = local_path_from_uri(uri);
                Ok(Arc::new(local_repo(std::path::Path::new(&path))?))
            }
            // Cloud schemes are structurally supported (feature-gated
            // `object_store` backends) but not wired for v1.
            Some(other @ ("s3" | "gs" | "gcs" | "wasbs" | "abfss" | "az" | "azure")) => {
                Err(MlflowError::not_implemented(format!(
                    "Artifact scheme '{other}' is not yet enabled in the Rust server; \
                     rebuild with the corresponding object_store feature"
                )))
            }
            Some(other) => Err(MlflowError::invalid_parameter_value(format!(
                "Unsupported artifact URI scheme: '{other}'"
            ))),
        }
    }

    /// Extract a local filesystem path from a bare path or a `file://` URI.
    fn local_path_from_uri(uri: &str) -> String {
        if let Some(rest) = uri.strip_prefix("file://") {
            // `file:///abs` → `/abs`; `file://host/abs` keeps host in netloc,
            // which we don't support locally — take the path after the netloc.
            match rest.find('/') {
                Some(0) => rest.to_string(),
                Some(i) => rest[i..].to_string(),
                None => rest.to_string(),
            }
        } else if let Some(rest) = uri.strip_prefix("file:") {
            rest.to_string()
        } else {
            uri.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use tempfile::TempDir;

    fn body_from(bytes: &'static [u8]) -> BoxStream<'static, Result<Bytes, MlflowError>> {
        stream::once(async move { Ok(Bytes::from_static(bytes)) }).boxed()
    }

    async fn collect(download: ArtifactDownload) -> Vec<u8> {
        let mut out = Vec::new();
        let mut s = download.stream;
        while let Some(chunk) = s.next().await {
            out.extend_from_slice(&chunk.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn put_get_roundtrip() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        repo.put("a/b/c.txt", body_from(b"hello world"))
            .await
            .unwrap();
        let dl = repo.get("a/b/c.txt").await.unwrap();
        assert_eq!(dl.size, 11);
        assert_eq!(collect(dl).await, b"hello world");
    }

    #[tokio::test]
    async fn get_missing_is_resource_does_not_exist() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        let err = repo.get("nope.txt").await.err().unwrap();
        assert_eq!(
            err.error_code,
            mlflow_error::ErrorCode::ResourceDoesNotExist
        );
        assert!(err.message.contains("No such artifact"));
    }

    #[tokio::test]
    async fn list_non_recursive_with_sizes_and_dirs() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        repo.put("root.txt", body_from(b"abc")).await.unwrap();
        repo.put("sub/nested.txt", body_from(b"xyz")).await.unwrap();
        repo.put("sub/deep/leaf.txt", body_from(b"1"))
            .await
            .unwrap();

        // Root listing: one file + one dir.
        let mut top = repo.list(None).await.unwrap();
        top.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].path, "root.txt");
        assert!(!top[0].is_dir);
        assert_eq!(top[0].file_size, Some(3));
        assert_eq!(top[1].path, "sub");
        assert!(top[1].is_dir);
        assert_eq!(top[1].file_size, None);

        // Nested listing: one file + one dir, non-recursive.
        let sub = repo.list(Some("sub")).await.unwrap();
        assert_eq!(sub.len(), 2);
        assert_eq!(sub[0].path, "sub/deep");
        assert!(sub[0].is_dir);
        assert_eq!(sub[1].path, "sub/nested.txt");
        assert_eq!(sub[1].file_size, Some(3));
    }

    #[tokio::test]
    async fn list_empty_or_missing_dir() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        assert!(repo.list(None).await.unwrap().is_empty());
        assert!(repo.list(Some("does/not/exist")).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_file_and_subtree() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        repo.put("keep.txt", body_from(b"k")).await.unwrap();
        repo.put("gone/a.txt", body_from(b"a")).await.unwrap();
        repo.put("gone/b/c.txt", body_from(b"c")).await.unwrap();

        // Delete a single file.
        repo.put("solo.txt", body_from(b"s")).await.unwrap();
        repo.delete("solo.txt").await.unwrap();
        assert!(repo.get("solo.txt").await.is_err());

        // Delete a subtree recursively.
        repo.delete("gone").await.unwrap();
        assert!(repo.get("gone/a.txt").await.is_err());
        assert!(repo.get("gone/b/c.txt").await.is_err());

        // Untouched sibling survives.
        assert_eq!(collect(repo.get("keep.txt").await.unwrap()).await, b"k");

        // Deleting an absent path is a no-op.
        repo.delete("never-existed").await.unwrap();
    }

    #[tokio::test]
    async fn multipart_is_not_implemented_locally() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        let err = repo.create_multipart_upload("x", 3).await.unwrap_err();
        assert_eq!(err.error_code, mlflow_error::ErrorCode::NotImplemented);
        let err = repo
            .complete_multipart_upload("x", "id", &[])
            .await
            .unwrap_err();
        assert_eq!(err.error_code, mlflow_error::ErrorCode::NotImplemented);
        let err = repo.abort_multipart_upload("x", "id").await.unwrap_err();
        assert_eq!(err.error_code, mlflow_error::ErrorCode::NotImplemented);
    }

    #[tokio::test]
    async fn temp_upload_files_are_hidden_from_listing() {
        let dir = TempDir::new().unwrap();
        let repo = local_repo(dir.path()).unwrap();
        repo.put(".artifact.uploading.tmp123", body_from(b"partial"))
            .await
            .unwrap();
        repo.put("real.txt", body_from(b"done")).await.unwrap();
        let listed = repo.list(None).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].path, "real.txt");
    }
}
