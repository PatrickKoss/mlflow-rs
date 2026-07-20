//! `mlflow-artifacts`: artifact storage proxy and streaming.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§3.11, Phase 5 T5.1/T5.2), this crate
//! implements the artifact plane of the Rust MLflow server:
//!
//! * [`path_safety::validate_path_is_safe`] — an exact port of
//!   `mlflow.utils.uri.validate_path_is_safe`, the path-traversal guard every
//!   artifact endpoint runs first.
//! * [`repo::ArtifactRepo`] — a streaming repository abstraction over the
//!   `object_store` crate (local FS and S3 are wired in the stock server;
//!   GCS/Azure remain feature-gated seams).
//! * [`get_artifact::send_artifact`] — the store-agnostic streaming core of the
//!   `/get-artifact` download handler (the run/model → repo resolution is Phase
//!   5 server wiring).
//! * [`router::artifacts_router`] — the `MlflowArtifactsService` HTTP surface
//!   (download/upload/list/delete + multipart), built relative to the
//!   `/api/2.0` + `/ajax-api/2.0` prefixes the server adds.
//!
//! Unlike the Python WSGI bridge (`fastapi_app.py:41`, which buffers whole
//! bodies), uploads and downloads stream in both directions: a 5 GB upload
//! flows chunk-by-chunk from the request body into `object_store`'s multipart
//! `put`, and downloads stream out of the backend — peak memory is bounded by a
//! single chunk/part, not the payload size (T5.1 AC).

pub mod get_artifact;
pub mod mime;
pub mod path_safety;
pub mod repo;
pub mod router;
#[cfg(feature = "aws")]
mod s3;

pub use get_artifact::{send_artifact, send_artifact_response};
pub use path_safety::validate_path_is_safe;
pub use repo::{
    factory, local_repo, multipart_upload_path, presigned_download_ttl_seconds, ArtifactDownload,
    ArtifactFileInfo, ArtifactRepo, ObjectStoreRepo, PresignedDownloadResult,
};
pub use router::{artifacts_router, artifacts_router_with_state, ArtifactsState};
