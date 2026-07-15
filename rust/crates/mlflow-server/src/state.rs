//! Shared application state for the HTTP layer.
//!
//! `AppState` is the single value threaded through every axum handler via
//! `State`. For Phase 3 it carries the tracking [`TrackingStore`]; later phases
//! (runs, metrics, traces, registry, auth, webhooks) add their own stores here,
//! so this is the extension point for the whole server.
//!
//! Phase 5 (T5.1-T5.3) adds the artifact plane: the `--serve-artifacts` flag and
//! the resolved `--artifacts-destination` proxy [`ArtifactRepo`], plus the
//! run/logged-model artifact-URI → repo resolution that mirrors
//! `handlers.py`'s `_is_servable_proxied_run_artifact_root` /
//! `_get_proxied_run_artifact_destination_path` /
//! `_get_artifact_repo_mlflow_artifacts` seam.
//!
//! T5.4 adds the [`mlflow_registry::RegistryStore`] handle so
//! `/model-versions/get-artifact` can resolve `storage_location or source`
//! (`_get_model_registry_store()`, `handlers.py:674`) alongside the same
//! artifact-resolution seam.
//!
//! The state is cheap to clone (`TrackingStore`/`RegistryStore` hold
//! `Arc`-backed pools, the proxy repo is an `Arc`), which is what axum's
//! `State` extractor requires.

use std::sync::Arc;

use mlflow_artifacts::ArtifactRepo;
use mlflow_error::MlflowError;
use mlflow_registry::RegistryStore;
use mlflow_store::TrackingStore;
use mlflow_webhooks::WebhookStore;

/// A resolved artifact repository plus the repo-relative path to operate on —
/// the output of resolving a run's / logged model's artifact URI against the
/// server's proxy configuration. Mirrors the `(artifact_repo, artifact_path)`
/// pair the Python handlers compute before calling `_send_artifact`.
pub struct ResolvedArtifact {
    pub repo: Arc<dyn ArtifactRepo>,
    pub path: String,
}

/// Application state shared across all HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    tracking_store: TrackingStore,
    /// The model-registry store, sharing the same backing `Db` pool as
    /// `tracking_store` (both stores are thin query layers over the same
    /// Alembic-migrated database). `None` in the ops-only / no-backend-store
    /// configuration ([`AppState::new`]) used by tests that don't touch the
    /// registry.
    registry_store: Option<RegistryStore>,
    /// The webhook store (T8.1/T8.2), sharing the tracking DB pool. `None` for
    /// backends that don't support webhooks (e.g. a future file store); the
    /// webhook handlers return a `not-implemented`-style error when absent.
    webhook_store: Option<WebhookStore>,
    /// `_is_serving_proxied_artifacts()` — whether `--serve-artifacts` is on.
    serve_artifacts: bool,
    /// The lazily-shared `--artifacts-destination` proxy repo, built once at
    /// startup from `artifacts_destination` (parity with the memoized
    /// `_artifact_repo` global in `_get_artifact_repo_mlflow_artifacts`). `None`
    /// when no destination is configured.
    proxied_artifacts_repo: Option<Arc<dyn ArtifactRepo>>,
    /// The raw `--artifacts-destination` URI (for error messages / diagnostics).
    artifacts_destination: Option<String>,
}

impl AppState {
    /// Build the state from an already-constructed [`TrackingStore`], with the
    /// artifact proxy disabled, no destination, and no registry store. Tests
    /// that don't exercise artifacts/the registry use this; `main`/`build_app`
    /// use [`AppState::with_artifacts`] (+ [`AppState::with_registry`] for the
    /// registry-store handle).
    pub fn new(tracking_store: TrackingStore) -> Self {
        Self::build(tracking_store, None, None, false, None, None)
    }

    /// Attach a [`WebhookStore`] to this state (T8.1/T8.2). Additive builder so
    /// existing constructors stay valid; returns a new `AppState` sharing the
    /// same inner fields plus the webhook store.
    pub fn with_webhook_store(self, webhook_store: WebhookStore) -> Self {
        let inner = &self.inner;
        Self {
            inner: Arc::new(AppStateInner {
                tracking_store: inner.tracking_store.clone(),
                registry_store: inner.registry_store.clone(),
                webhook_store: Some(webhook_store),
                serve_artifacts: inner.serve_artifacts,
                proxied_artifacts_repo: inner.proxied_artifacts_repo.clone(),
                artifacts_destination: inner.artifacts_destination.clone(),
            }),
        }
    }

    /// Build the state with the artifact proxy configuration. `serve_artifacts`
    /// mirrors `--serve-artifacts`; `proxied_artifacts_repo` is the resolved
    /// `--artifacts-destination` repo (already constructed once so it's shared,
    /// like Python's memoized `_artifact_repo`).
    pub fn with_artifacts(
        tracking_store: TrackingStore,
        serve_artifacts: bool,
        proxied_artifacts_repo: Option<Arc<dyn ArtifactRepo>>,
        artifacts_destination: Option<String>,
    ) -> Self {
        Self::build(
            tracking_store,
            None,
            None,
            serve_artifacts,
            proxied_artifacts_repo,
            artifacts_destination,
        )
    }

    /// Same as [`AppState::with_artifacts`], additionally wiring the
    /// model-registry store (T5.4: `/model-versions/get-artifact` needs
    /// `_get_model_registry_store()`).
    pub fn with_registry(
        tracking_store: TrackingStore,
        registry_store: RegistryStore,
        serve_artifacts: bool,
        proxied_artifacts_repo: Option<Arc<dyn ArtifactRepo>>,
        artifacts_destination: Option<String>,
    ) -> Self {
        Self::build(
            tracking_store,
            Some(registry_store),
            None,
            serve_artifacts,
            proxied_artifacts_repo,
            artifacts_destination,
        )
    }

    fn build(
        tracking_store: TrackingStore,
        registry_store: Option<RegistryStore>,
        webhook_store: Option<WebhookStore>,
        serve_artifacts: bool,
        proxied_artifacts_repo: Option<Arc<dyn ArtifactRepo>>,
        artifacts_destination: Option<String>,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                tracking_store,
                registry_store,
                webhook_store,
                serve_artifacts,
                proxied_artifacts_repo,
                artifacts_destination,
            }),
        }
    }

    /// The tracking store (experiments, runs, metrics, traces, …).
    pub fn tracking_store(&self) -> &TrackingStore {
        &self.inner.tracking_store
    }

    /// The model-registry store (`_get_model_registry_store()`,
    /// `handlers.py:674`). Errors `INTERNAL_ERROR` when this server instance
    /// wasn't wired with a registry store (mirrors the shape of
    /// [`AppState::proxied_artifacts_repo`]'s misconfiguration error — this
    /// server has no dedicated `--registry-store-uri` flag yet, so it is
    /// always available whenever a backend store is configured; the `None`
    /// case only arises for ops-only [`AppState::new`] test builders).
    pub fn registry_store(&self) -> Result<&RegistryStore, MlflowError> {
        self.inner.registry_store.as_ref().ok_or_else(|| {
            MlflowError::internal_error(
                "The MLflow server is not configured with a model registry store.",
            )
        })
    }

    /// The webhook store, or a `RESOURCE_DOES_NOT_EXIST`-shaped error when the
    /// backend does not support webhooks (`None`). Mirrors how the Python
    /// handlers assume the model-registry store implements the webhook APIs.
    pub fn webhook_store(&self) -> Result<&WebhookStore, MlflowError> {
        self.inner.webhook_store.as_ref().ok_or_else(|| {
            MlflowError::not_implemented(
                "Webhooks are not supported by the configured backend store.".to_string(),
            )
        })
    }

    /// `_is_serving_proxied_artifacts()` — whether the `mlflow-artifacts` proxy
    /// surface is enabled (`--serve-artifacts`).
    pub fn serve_artifacts(&self) -> bool {
        self.inner.serve_artifacts
    }

    /// The configured `--artifacts-destination` URI, if any.
    pub fn artifacts_destination(&self) -> Option<&str> {
        self.inner.artifacts_destination.as_deref()
    }

    /// The `--artifacts-destination` proxy repo, or an error mirroring Python's
    /// `os.environ[ARTIFACTS_DESTINATION_ENV_VAR]` `KeyError` when the server
    /// serves proxied artifacts without a destination configured (a
    /// misconfiguration → 500).
    pub fn proxied_artifacts_repo(&self) -> Result<Arc<dyn ArtifactRepo>, MlflowError> {
        self.inner.proxied_artifacts_repo.clone().ok_or_else(|| {
            let dest = self.artifacts_destination().unwrap_or("<unset>");
            MlflowError::internal_error(format!(
                "The MLflow server is serving proxied artifacts but no usable \
                 --artifacts-destination is configured (destination: {dest})."
            ))
        })
    }

    /// `_is_servable_proxied_run_artifact_root(run_artifact_root)`
    /// (`handlers.py:574`): the artifact root uses a proxied scheme
    /// (`http`/`https`/`mlflow-artifacts`) AND this server serves proxied
    /// artifacts.
    pub fn is_servable_proxied_run_artifact_root(&self, artifact_root: &str) -> bool {
        is_proxied_scheme(artifact_root) && self.serve_artifacts()
    }

    /// Resolve an artifact URI + a repo-relative path into a concrete repo and
    /// path, mirroring the branch shared by `get_artifact_handler`,
    /// `upload_artifact_handler`, and `get_logged_model_artifact_handler`:
    ///
    /// * proxied + servable → the `--artifacts-destination` repo, with the path
    ///   rewritten to `_get_proxied_run_artifact_destination_path(root, path)`;
    /// * otherwise → `get_artifact_repository(artifact_uri)` with `path`
    ///   unchanged.
    ///
    /// Workspace prefixing (`_get_workspace_scoped_repo_path_if_enabled`) is a
    /// no-op while workspaces are disabled (the default this server ships), so it
    /// is elided here; it slots in at this seam when the workspaces phase lands.
    pub fn resolve_artifact(
        &self,
        artifact_uri: &str,
        relative_path: &str,
    ) -> Result<ResolvedArtifact, MlflowError> {
        if self.is_servable_proxied_run_artifact_root(artifact_uri) {
            let repo = self.proxied_artifacts_repo()?;
            let path = proxied_run_artifact_destination_path(artifact_uri, Some(relative_path))?;
            Ok(ResolvedArtifact { repo, path })
        } else {
            let repo = mlflow_artifacts::factory::repo_from_uri(artifact_uri)?;
            Ok(ResolvedArtifact {
                repo,
                path: relative_path.to_string(),
            })
        }
    }
}

/// `urlparse(uri).scheme in ["http", "https", "mlflow-artifacts"]`.
fn is_proxied_scheme(uri: &str) -> bool {
    matches!(
        scheme_of(uri).as_deref(),
        Some("http" | "https" | "mlflow-artifacts")
    )
}

/// The URI scheme (lowercased), or `None` for a bare path.
fn scheme_of(uri: &str) -> Option<String> {
    let (scheme, _) = uri.split_once("://")?;
    Some(scheme.to_ascii_lowercase())
}

/// `_get_proxied_run_artifact_destination_path(proxied_artifact_root,
/// relative_path)` (`handlers.py:616`): resolve a proxied artifact root to the
/// storage-relative path within the `--artifacts-destination`.
///
/// * `mlflow-artifacts://<netloc>/path` → the path component, leading `/`
///   stripped.
/// * `http(s)://.../api/2.0/mlflow-artifacts/artifacts/<rest>` → `<rest>`,
///   leading `/` stripped (the fixed route anchor Python splits on).
///
/// then `posixpath.join(root_path, relative_path)` when `relative_path` is set.
pub(crate) fn proxied_run_artifact_destination_path(
    proxied_artifact_root: &str,
    relative_path: Option<&str>,
) -> Result<String, MlflowError> {
    let scheme = scheme_of(proxied_artifact_root);
    let root_path = match scheme.as_deref() {
        Some("mlflow-artifacts") => {
            // `mlflow-artifacts://<netloc>/path/to/artifact` — everything after
            // the netloc (Python reads `parsed.path`), leading slash stripped.
            let after_scheme = proxied_artifact_root
                .split_once("://")
                .map(|(_, rest)| rest)
                .unwrap_or("");
            let path = match after_scheme.split_once('/') {
                Some((_netloc, path)) => path,
                None => "",
            };
            path.trim_start_matches('/').to_string()
        }
        Some("http") | Some("https") => {
            const ANCHOR: &str = "/api/2.0/mlflow-artifacts/artifacts/";
            match proxied_artifact_root.split_once(ANCHOR) {
                Some((_, rest)) => rest.trim_start_matches('/').to_string(),
                None => {
                    return Err(MlflowError::internal_error(format!(
                        "Cannot resolve proxied artifact root '{proxied_artifact_root}': \
                         missing '{ANCHOR}' route anchor."
                    )));
                }
            }
        }
        _ => {
            return Err(MlflowError::internal_error(format!(
                "Cannot resolve non-proxied artifact root '{proxied_artifact_root}'."
            )));
        }
    };

    Ok(match relative_path {
        Some(rel) => posix_join(&root_path, rel),
        None => root_path,
    })
}

/// `posixpath.join(a, b)`: if `b` is absolute it replaces `a`; an empty `a`
/// yields `b`; otherwise `a` + `/` + `b` (collapsing a trailing slash on `a`).
fn posix_join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        return b.to_string();
    }
    if a.is_empty() {
        return b.to_string();
    }
    format!("{}/{}", a.trim_end_matches('/'), b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxied_scheme_detection() {
        assert!(is_proxied_scheme("mlflow-artifacts://host/exp/run"));
        assert!(is_proxied_scheme(
            "http://host/api/2.0/mlflow-artifacts/artifacts/x"
        ));
        assert!(is_proxied_scheme(
            "https://host/api/2.0/mlflow-artifacts/artifacts/x"
        ));
        assert!(!is_proxied_scheme("s3://bucket/prefix"));
        assert!(!is_proxied_scheme("/local/path"));
        assert!(!is_proxied_scheme("file:///local/path"));
    }

    #[test]
    fn mlflow_artifacts_uri_destination_path() {
        let p = proxied_run_artifact_destination_path(
            "mlflow-artifacts://host/1/abc/artifacts",
            Some("model/data.txt"),
        )
        .unwrap();
        assert_eq!(p, "1/abc/artifacts/model/data.txt");

        // No relative path → just the root path.
        let p =
            proxied_run_artifact_destination_path("mlflow-artifacts://host/1/abc/artifacts", None)
                .unwrap();
        assert_eq!(p, "1/abc/artifacts");
    }

    #[test]
    fn http_uri_destination_path_splits_on_anchor() {
        let p = proxied_run_artifact_destination_path(
            "http://host:5000/api/2.0/mlflow-artifacts/artifacts/1/abc/artifacts",
            Some("f.txt"),
        )
        .unwrap();
        assert_eq!(p, "1/abc/artifacts/f.txt");
    }

    #[test]
    fn http_uri_without_anchor_is_internal_error() {
        let err = proxied_run_artifact_destination_path("http://host/no/anchor/here", Some("f"))
            .unwrap_err();
        assert_eq!(err.error_code, mlflow_error::ErrorCode::InternalError);
    }
}
