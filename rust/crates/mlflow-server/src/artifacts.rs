//! Artifact HTTP surface (plan T5.1-T5.3, §3.11).
//!
//! This module wires the artifact-plane handlers to [`AppState`], reusing the
//! store-agnostic streaming core in the `mlflow-artifacts` crate
//! ([`mlflow_artifacts::send_artifact_response`], [`ArtifactRepo`],
//! [`mlflow_artifacts::validate_path_is_safe`]). It ports:
//!
//! * **T5.1** `GET /get-artifact?run_id=&path=` — `get_artifact_handler`
//!   (`handlers.py:1519`): resolve the run's `artifact_uri`, stream the file.
//! * **T5.2** the `MlflowArtifactsService` proxy (download/upload/list/delete +
//!   multipart create/complete/abort + presigned) under
//!   `/(api|ajax-api)/2.0/mlflow-artifacts/...` (`handlers.py:3536-3878`), gated
//!   by `--serve-artifacts` (`_disable_unless_serve_artifacts`). Streams both
//!   directions — no whole-body buffering (the Python WSGI-bridge defect the
//!   plan calls out).
//! * **T5.3** ajax `POST /ajax-api/2.0/mlflow/upload-artifact`
//!   (`upload_artifact_handler`, `handlers.py:2408`), `listLoggedModelArtifacts`
//!   (`_list_logged_model_artifacts`, `handlers.py:5403`; proto-route-table), and
//!   the ajax-only logged-model artifact file download
//!   (`get_logged_model_artifact_handler`, `handlers.py:5214`).
//!
//! Multipart + presigned-URL endpoints go through the repo trait, whose local-FS
//! backend returns `NOT_IMPLEMENTED` (parity with `LocalArtifactRepository`,
//! which lacks the multipart/presigned mixins).
//!
//! * **T5.4** `GET /model-versions/get-artifact?name=&version=&path=`
//!   (`get_model_version_artifact_handler`, `handlers.py:3033`): resolve
//!   `storage_location or source` via the [`mlflow_registry::RegistryStore`],
//!   then stream the file through the same proxied/direct resolution seam as
//!   T5.1. `models:/name/version`-sourced versions already carry a resolved
//!   `storage_location` (the registry store's `create_model_version`
//!   resolves that at write time, per `mlflow-registry`'s docs), so no extra
//!   indirection is needed here.

use std::collections::HashMap;

use axum::body::{Body, Bytes};
use axum::extract::{MatchedPath, Path, State};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::stream::StreamExt;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_proto::mlflow::artifacts as art_pb;

use crate::proto_http::{parse_query_pairs, parse_request_with_path_params, proto_response};
use crate::state::{proxied_run_artifact_destination_path, AppState};
use crate::workspace::Workspace;

/// Cap for the ajax `upload-artifact` body (`10 * 1024 * 1024`,
/// `handlers.py:2424`).
const MAX_UPLOAD_ARTIFACT_BYTES: usize = 10 * 1024 * 1024;

/// Cap for the proxy multipart control-message bodies (create/complete/abort).
/// Artifact bytes never pass through here — they stream via `proxy_upload`.
const MAX_CONTROL_BODY_BYTES: usize = 16 * 1024 * 1024;

// ===========================================================================
// T5.1 — GET /get-artifact
// ===========================================================================

/// `get_artifact_handler` (`handlers.py:1519`), served at the root
/// `/get-artifact` (`mlflow/server/__init__.py:111`). Plain `run_id`/`run_uuid`
/// + `path` query params (not a proto message).
pub async fn get_artifact(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
) -> Response {
    let result = async {
        let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
        // `run_id or run_uuid` — `request.args["path"]` raises KeyError (→ 400)
        // when missing, which `catch_mlflow_exception` does NOT wrap; Flask
        // returns its own 400 BadRequest page. We surface a BAD_REQUEST MlflowError
        // for the missing-`path` case (observably a 400 with a JSON body — a
        // documented, benign deviation from Flask's HTML 400).
        let run_id = query_param(&pairs, "run_id").or_else(|| query_param(&pairs, "run_uuid"));
        let Some(path) = query_param(&pairs, "path") else {
            return Err(MlflowError::new(
                "Request must specify a 'path' query parameter.",
                ErrorCode::BadRequest,
            ));
        };
        let safe_path = mlflow_artifacts::validate_path_is_safe(&path)?;
        let Some(run_id) = run_id.filter(|s| !s.is_empty()) else {
            return Err(MlflowError::new(
                "Request must specify a 'run_id' query parameter.",
                ErrorCode::BadRequest,
            ));
        };

        let run = state
            .tracking_store()
            .get_run(workspace.name(), &run_id)
            .await?;
        let artifact_uri = run.info.artifact_uri.unwrap_or_default();
        let resolved = state.resolve_artifact(&artifact_uri, &safe_path)?;
        Ok::<_, MlflowError>((resolved.repo, resolved.path))
    }
    .await;

    match result {
        Ok((repo, path)) => mlflow_artifacts::send_artifact_response(repo.as_ref(), &path).await,
        Err(e) => e.into_response(),
    }
}

// ===========================================================================
// T5.4 — GET /model-versions/get-artifact
// ===========================================================================

/// `get_model_version_artifact_handler` (`handlers.py:3033`), served at the
/// root `/model-versions/get-artifact` (`mlflow/server/__init__.py:117`) —
/// no ajax alias, matching Python. Plain `name`/`version`/`path` query
/// params (not a proto message).
pub async fn get_model_version_artifact(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
) -> Response {
    let result = async {
        let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
        // `request.args["path"]` raises KeyError (→ 400) when missing, which
        // `catch_mlflow_exception` does NOT wrap; same documented, benign
        // deviation as `get_artifact`'s missing-`path` case (see its doc
        // comment above).
        let Some(path) = query_param(&pairs, "path") else {
            return Err(MlflowError::new(
                "Request must specify a 'path' query parameter.",
                ErrorCode::BadRequest,
            ));
        };
        let safe_path = mlflow_artifacts::validate_path_is_safe(&path)?;
        // `request.args.get("name")` — `None` when absent. The registry
        // store's `_validate_model_name` (invoked inside
        // `get_model_version_download_uri`) raises the exact same "Missing
        // value for required parameter" `INVALID_PARAMETER_VALUE` error for
        // `None` as for `""`, so passing through the empty string reproduces
        // Python's error byte-for-byte without a separate required-param
        // guard here.
        let name = query_param(&pairs, "name").unwrap_or_default();
        // `request.args.get("version")` — `None` when absent. UNLIKE `name`,
        // `_validate_model_version` (`mlflow/utils/validation.py:684`) has no
        // explicit `is None` check: it calls `int(model_version)` inside a
        // `try/except ValueError`, and `int(None)` raises `TypeError`, NOT
        // `ValueError` — so the exception is NOT caught there. It propagates
        // out of the SQLAlchemy store's `ManagedSessionMaker` context
        // manager, whose blanket `except Exception as e: raise
        // MlflowException(message=e, error_code=INTERNAL_ERROR) from e`
        // (`mlflow/store/db/utils.py:188`) wraps it into a 500
        // `INTERNAL_ERROR`, distinct from the 400 `INVALID_PARAMETER_VALUE`
        // a present-but-non-numeric `version` (e.g. `"abc"`, or `""` for
        // `version=` with an empty value) gets from the caught `ValueError`
        // path. Verified against the real Python handler. We special-case
        // the missing-query-param case to reproduce this exact asymmetry;
        // `validate_model_version` inside the registry store handles the
        // present-but-invalid cases identically to Python.
        let Some(version) = query_param(&pairs, "version") else {
            return Err(MlflowError::internal_error(
                "int() argument must be a string, a bytes-like object or a real number, not \
                 'NoneType'",
            ));
        };

        let registry = state.registry_store()?;
        let artifact_uri = registry
            .get_model_version_download_uri(workspace.name(), &name, &version)
            .await?;
        let resolved = state.resolve_artifact(&artifact_uri, &safe_path)?;
        Ok::<_, MlflowError>((resolved.repo, resolved.path))
    }
    .await;

    match result {
        Ok((repo, path)) => mlflow_artifacts::send_artifact_response(repo.as_ref(), &path).await,
        Err(e) => e.into_response(),
    }
}

// ===========================================================================
// T5.3 — ajax POST /ajax-api/2.0/mlflow/upload-artifact
// ===========================================================================

/// `upload_artifact_handler` (`handlers.py:2408`): `run_uuid` + `path` query
/// params, raw request body (max 10 MB), written under the run's artifacts.
pub async fn upload_artifact(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Response {
    let result = async {
        let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
        let run_uuid = query_param(&pairs, "run_uuid").filter(|s| !s.is_empty());
        let Some(run_uuid) = run_uuid else {
            return Err(MlflowError::invalid_parameter_value(
                "Request must specify run_uuid.",
            ));
        };
        let path = query_param(&pairs, "path").filter(|s| !s.is_empty());
        let Some(path) = path else {
            return Err(MlflowError::invalid_parameter_value(
                "Request must specify path.",
            ));
        };
        let safe_path = mlflow_artifacts::validate_path_is_safe(&path)?;

        if body.len() > MAX_UPLOAD_ARTIFACT_BYTES {
            return Err(MlflowError::invalid_parameter_value(
                "Artifact size is too large. Max size is 10MB.",
            ));
        }
        if body.is_empty() {
            return Err(MlflowError::invalid_parameter_value(
                "Request must specify data.",
            ));
        }

        let run = state
            .tracking_store()
            .get_run(workspace.name(), &run_uuid)
            .await?;
        let artifact_uri = run.info.artifact_uri.clone().unwrap_or_default();

        // Python writes the file at `<run artifact root>/<path>` (the run's
        // artifact repo joins `dirname` and logs `basename`); resolving the
        // artifact URI against the full `path` yields the same destination for
        // both the direct and proxied cases.
        let resolved = state.resolve_artifact(&artifact_uri, &safe_path)?;
        let stream = futures::stream::once(async move { Ok(body) }).boxed();
        resolved.repo.put(&resolved.path, stream).await?;
        Ok::<_, MlflowError>(())
    }
    .await;

    match result {
        // Python returns `Response(mimetype="application/json")` with an empty
        // body (no proto message) — a 200 with an empty JSON-typed body.
        Ok(()) => Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::empty())
            .expect("valid response"),
        Err(e) => e.into_response(),
    }
}

// ===========================================================================
// T5.3 — ajax logged-model artifact file download + listLoggedModelArtifacts
// ===========================================================================

/// `get_logged_model_artifact_handler` (`handlers.py:5214`), served at
/// `/ajax-api/2.0/mlflow/logged-models/{model_id}/artifacts/files`
/// (`mlflow/server/__init__.py:166`). `artifact_file_path` query param.
pub async fn get_logged_model_artifact(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
) -> Response {
    let result = async {
        let model_id = path_param(&path_params, "model_id")?;
        let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
        let artifact_file_path =
            query_param(&pairs, "artifact_file_path").filter(|s| !s.is_empty());
        let Some(artifact_file_path) = artifact_file_path else {
            return Err(MlflowError::new(
                "Request must include the \"artifact_file_path\" query parameter.",
                ErrorCode::BadRequest,
            ));
        };
        let safe_path = mlflow_artifacts::validate_path_is_safe(&artifact_file_path)?;

        let model = state
            .tracking_store()
            .get_logged_model(workspace.name(), &model_id, false)
            .await?;
        let resolved = state.resolve_artifact(&model.artifact_location, &safe_path)?;
        Ok::<_, MlflowError>((resolved.repo, resolved.path))
    }
    .await;

    match result {
        Ok((repo, path)) => mlflow_artifacts::send_artifact_response(repo.as_ref(), &path).await,
        Err(e) => e.into_response(),
    }
}

/// `_list_logged_model_artifacts` (`handlers.py:5403`) — proto-route-table GET
/// `/mlflow/logged-models/{model_id}/artifacts/directories`.
pub async fn list_logged_model_artifacts(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let model_id = path_param(&path_params, "model_id")?;
    let req: pb::ListLoggedModelArtifacts = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.ListLoggedModelArtifacts",
        &[("model_id", model_id.clone())],
    )?;
    let dir_path = req
        .artifact_directory_path
        .as_deref()
        .filter(|_| req.artifact_directory_path.is_some());
    let validated = match dir_path {
        Some(p) => Some(mlflow_artifacts::validate_path_is_safe(p)?),
        None => None,
    };

    let model = state
        .tracking_store()
        .get_logged_model(workspace.name(), &model_id, false)
        .await?;

    let files = list_artifacts_at(&state, &model.artifact_location, validated.as_deref()).await?;
    let resp = pb::list_logged_model_artifacts::Response {
        root_uri: Some(model.artifact_location),
        files,
        next_page_token: None,
    };
    proto_response(&resp, "mlflow.ListLoggedModelArtifacts.Response")
}

/// List artifacts under an artifact root at `relative_path`, mirroring the
/// direct vs proxied branch in `_list_artifacts_for_proxied_run_artifact_root` /
/// `list_artifacts`. Returns proto `FileInfo`s.
///
/// * Direct (non-proxied): `get_artifact_repository(root).list_artifacts(path)`
///   — full run-relative paths.
/// * Proxied + servable: list from the `--artifacts-destination` repo at the
///   resolved destination path, then rewrite each entry's path to its basename
///   re-joined under `relative_path` (Python's
///   `posixpath.join(relative_path, basename)`).
pub(crate) async fn list_artifacts_at(
    state: &AppState,
    artifact_root: &str,
    relative_path: Option<&str>,
) -> Result<Vec<pb::FileInfo>, MlflowError> {
    if state.is_servable_proxied_run_artifact_root(artifact_root) {
        let repo = state.proxied_artifacts_repo()?;
        let dest = proxied_run_artifact_destination_path(artifact_root, relative_path)?;
        let entries = repo.list(Some(&dest)).await?;
        Ok(entries
            .into_iter()
            .map(|f| {
                let base = basename(&f.path);
                let run_relative = match relative_path {
                    Some(rel) if !rel.is_empty() => format!("{}/{base}", rel.trim_end_matches('/')),
                    _ => base.to_string(),
                };
                pb::FileInfo {
                    path: Some(run_relative),
                    is_dir: Some(f.is_dir),
                    file_size: f.file_size,
                }
            })
            .collect())
    } else {
        let repo = mlflow_artifacts::factory::repo_from_uri(artifact_root)?;
        let entries = repo.list(relative_path).await?;
        Ok(entries
            .into_iter()
            .map(|f| pb::FileInfo {
                path: Some(f.path),
                is_dir: Some(f.is_dir),
                file_size: f.file_size,
            })
            .collect())
    }
}

// ===========================================================================
// T5.2 — MlflowArtifactsService proxy (gated by --serve-artifacts)
// ===========================================================================

/// `_disable_unless_serve_artifacts` (`handlers.py:1186`): when `--serve-artifacts`
/// is off, return the exact 503 body Python sends, naming the matched route.
fn disabled_response(parts: &Parts) -> Response {
    let rule = parts
        .extensions
        .get::<MatchedPath>()
        .map(|m| m.as_str())
        .unwrap_or_else(|| parts.uri.path());
    let body = format!(
        "Endpoint: {rule} disabled due to the mlflow server running with \
         `--no-serve-artifacts`. To enable artifacts server functionality, run \
         `mlflow server` with `--serve-artifacts`"
    );
    (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
}

/// GET `/mlflow-artifacts/artifacts/{*artifact_path}` — `_download_artifact`
/// (`handlers.py:3538`). Streams the file from the proxy repo.
pub async fn proxy_download(
    State(state): State<AppState>,
    Path(artifact_path): Path<String>,
    parts: Parts,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let repo = match state.proxied_artifacts_repo() {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    // `send_artifact_response` validates the path and streams with headers.
    mlflow_artifacts::send_artifact_response(repo.as_ref(), &artifact_path).await
}

/// PUT `/mlflow-artifacts/artifacts/{*artifact_path}` — `_upload_artifact`
/// (`handlers.py:3573`). Streams the request body into the proxy repo (no
/// buffering).
pub async fn proxy_upload(
    State(state): State<AppState>,
    Path(artifact_path): Path<String>,
    parts: Parts,
    body: Body,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        let repo = state.proxied_artifacts_repo()?;
        let safe = mlflow_artifacts::validate_path_is_safe(&artifact_path)?;
        let stream = body
            .into_data_stream()
            .map(|chunk| {
                chunk.map_err(|e| MlflowError::internal_error(format!("Upload read error: {e}")))
            })
            .boxed();
        repo.put(&safe, stream).await?;
        Ok::<_, MlflowError>(())
    }
    .await;
    match result {
        Ok(()) => proto_json(
            &art_pb::upload_artifact::Response {},
            "mlflow.artifacts.UploadArtifact.Response",
        ),
        Err(e) => e.into_response(),
    }
}

/// DELETE `/mlflow-artifacts/artifacts/{*artifact_path}` —
/// `_delete_artifact_mlflow_artifacts` (`handlers.py:3621`).
pub async fn proxy_delete(
    State(state): State<AppState>,
    Path(artifact_path): Path<String>,
    parts: Parts,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        let repo = state.proxied_artifacts_repo()?;
        let safe = mlflow_artifacts::validate_path_is_safe(&artifact_path)?;
        repo.delete(&safe).await?;
        Ok::<_, MlflowError>(())
    }
    .await;
    match result {
        Ok(()) => proto_json(
            &art_pb::delete_artifact::Response {},
            "mlflow.artifacts.DeleteArtifact.Response",
        ),
        Err(e) => e.into_response(),
    }
}

/// GET `/mlflow-artifacts/artifacts?path=` — `_list_artifacts_mlflow_artifacts`
/// (`handlers.py:3598`). Each returned `FileInfo` path is reduced to its
/// basename (`posixpath.basename`).
pub async fn proxy_list(State(state): State<AppState>, parts: Parts) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        let repo = state.proxied_artifacts_repo()?;
        let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
        // `HasField("path")`: validate iff present.
        let validated = match query_param(&pairs, "path") {
            Some(p) => Some(mlflow_artifacts::validate_path_is_safe(&p)?),
            None => None,
        };
        let files = repo.list(validated.as_deref()).await?;
        let proto_files = files
            .into_iter()
            .map(|f| pb::FileInfo {
                path: Some(basename(&f.path).to_string()),
                is_dir: Some(f.is_dir),
                file_size: f.file_size,
            })
            .collect();
        Ok::<_, MlflowError>(pb::list_artifacts::Response {
            files: proto_files,
            root_uri: None,
            next_page_token: None,
        })
    }
    .await;
    match result {
        Ok(resp) => proto_json(&resp, "mlflow.ListArtifacts.Response"),
        Err(e) => e.into_response(),
    }
}

/// POST `/mlflow-artifacts/mpu/create/{*artifact_path}` —
/// `_create_multipart_upload_artifact` (`handlers.py:3749`). Local FS is not a
/// `MultipartUploadMixin`, so this returns `NOT_IMPLEMENTED`.
pub async fn proxy_create_multipart(
    State(state): State<AppState>,
    Path(_artifact_path): Path<String>,
    parts: Parts,
    body: Bytes,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        let repo = state.proxied_artifacts_repo()?;
        mlflow_artifacts::validate_path_is_safe(&_artifact_path)?;
        let req: art_pb::CreateMultipartUpload =
            parse_control_body(&body, "mlflow.artifacts.CreateMultipartUpload")?;
        let path = req.path.unwrap_or_default();
        let num_parts = req.num_parts.unwrap_or_default();
        let res = repo.create_multipart_upload(&path, num_parts).await?;
        Ok::<_, MlflowError>(art_pb::create_multipart_upload::Response {
            upload_id: Some(res.upload_id),
            credentials: res
                .credentials
                .into_iter()
                .map(|c| art_pb::MultipartUploadCredential {
                    url: Some(c.url),
                    part_number: Some(c.part_number),
                    headers: c.headers.into_iter().collect(),
                })
                .collect(),
        })
    }
    .await;
    match result {
        Ok(resp) => proto_json(&resp, "mlflow.artifacts.CreateMultipartUpload.Response"),
        Err(e) => e.into_response(),
    }
}

/// POST `/mlflow-artifacts/mpu/complete/{*artifact_path}` —
/// `_complete_multipart_upload_artifact` (`handlers.py:3783`).
pub async fn proxy_complete_multipart(
    State(state): State<AppState>,
    Path(_artifact_path): Path<String>,
    parts: Parts,
    body: Bytes,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        let repo = state.proxied_artifacts_repo()?;
        mlflow_artifacts::validate_path_is_safe(&_artifact_path)?;
        let req: art_pb::CompleteMultipartUpload =
            parse_control_body(&body, "mlflow.artifacts.CompleteMultipartUpload")?;
        let path = req.path.unwrap_or_default();
        let upload_id = req.upload_id.unwrap_or_default();
        let parts: Vec<mlflow_artifacts::repo::MultipartUploadPart> = req
            .parts
            .into_iter()
            .map(|p| mlflow_artifacts::repo::MultipartUploadPart {
                part_number: p.part_number.unwrap_or_default(),
                etag: p.etag.unwrap_or_default(),
                url: p.url.unwrap_or_default(),
            })
            .collect();
        repo.complete_multipart_upload(&path, &upload_id, &parts)
            .await?;
        Ok::<_, MlflowError>(())
    }
    .await;
    match result {
        Ok(()) => proto_json(
            &art_pb::complete_multipart_upload::Response {},
            "mlflow.artifacts.CompleteMultipartUpload.Response",
        ),
        Err(e) => e.into_response(),
    }
}

/// POST `/mlflow-artifacts/mpu/abort/{*artifact_path}` —
/// `_abort_multipart_upload_artifact` (`handlers.py:3817`).
pub async fn proxy_abort_multipart(
    State(state): State<AppState>,
    Path(_artifact_path): Path<String>,
    parts: Parts,
    body: Bytes,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        let repo = state.proxied_artifacts_repo()?;
        mlflow_artifacts::validate_path_is_safe(&_artifact_path)?;
        let req: art_pb::AbortMultipartUpload =
            parse_control_body(&body, "mlflow.artifacts.AbortMultipartUpload")?;
        let path = req.path.unwrap_or_default();
        let upload_id = req.upload_id.unwrap_or_default();
        repo.abort_multipart_upload(&path, &upload_id).await?;
        Ok::<_, MlflowError>(())
    }
    .await;
    match result {
        Ok(()) => proto_json(
            &art_pb::abort_multipart_upload::Response {},
            "mlflow.artifacts.AbortMultipartUpload.Response",
        ),
        Err(e) => e.into_response(),
    }
}

/// GET `/mlflow-artifacts/presigned/{*artifact_path}` —
/// `_get_presigned_download_url` (`handlers.py:3848`). Local FS has no presigned
/// URLs (`_validate_support_multipart_download` → NOT_IMPLEMENTED).
pub async fn proxy_presigned_download(
    State(state): State<AppState>,
    Path(artifact_path): Path<String>,
    parts: Parts,
) -> Response {
    if !state.serve_artifacts() {
        return disabled_response(&parts);
    }
    let result = async {
        // Python resolves the repo first (`_get_artifact_repo_mlflow_artifacts`),
        // then `_validate_support_multipart_download` raises NOT_IMPLEMENTED for
        // local FS. Validate the path for parity with the other handlers.
        state.proxied_artifacts_repo()?;
        mlflow_artifacts::validate_path_is_safe(&artifact_path)?;
        Err::<(), _>(MlflowError::not_implemented(
            "Presigned download URLs are not supported for the current artifact store",
        ))
    }
    .await;
    match result {
        Ok(()) => unreachable!(),
        Err(e) => e.into_response(),
    }
}

// ===========================================================================
// helpers
// ===========================================================================

fn query_param(pairs: &[(String, String)], name: &str) -> Option<String> {
    pairs
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
}

fn path_param(params: &HashMap<String, String>, name: &str) -> Result<String, MlflowError> {
    params
        .get(name)
        .cloned()
        .ok_or_else(|| MlflowError::internal_error(format!("Missing path parameter '{name}'.")))
}

fn basename(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((_, name)) => name,
        None => path,
    }
}

/// Render a proto message as MLflow pretty JSON (`Response(...); set_data(
/// message_to_json(...))`).
fn proto_json<M: prost::Message>(msg: &M, type_name: &str) -> Response {
    match proto_response(msg, type_name) {
        Ok(resp) => resp,
        Err(e) => e.into_response(),
    }
}

/// Parse a bounded control-message body with the MLflow JSON codec
/// (unknown-field tolerant), mirroring `_get_request_message`.
fn parse_control_body<M: prost::Message + Default>(
    body: &Bytes,
    type_name: &str,
) -> Result<M, MlflowError> {
    if body.len() > MAX_CONTROL_BODY_BYTES {
        return Err(MlflowError::invalid_parameter_value(
            "Request body is too large.",
        ));
    }
    let text = std::str::from_utf8(body)
        .map_err(|_| MlflowError::invalid_parameter_value("Request body is not valid UTF-8"))?;
    let json = if text.trim().is_empty() { "{}" } else { text };
    mlflow_proto::from_mlflow_json(json, type_name)
        .map_err(|e| MlflowError::invalid_parameter_value(format!("Malformed request: {e}")))
}
