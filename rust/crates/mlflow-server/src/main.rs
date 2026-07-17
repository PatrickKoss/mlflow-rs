//! `mlflow-server`: the Rust MLflow tracking server binary.
//!
//! Thin entry point: parses CLI args, initializes tracing, builds the
//! `Router` via [`mlflow_server::build_app`], and serves it with graceful
//! shutdown on SIGINT/SIGTERM. All actual app wiring lives in `lib.rs` so it
//! can be exercised directly in tests.

use clap::Parser;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app, build_app_with_state, AppState, Cli, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use mlflow_webhooks::{WebhookDispatcher, WebhookStore};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

/// `DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH` (`mlflow/store/tracking/__init__.py:12`).
const DEFAULT_ARTIFACT_ROOT: &str = "./mlruns";

/// Python's enable signal for the basic-auth app: the `MLFLOW_AUTH_CONFIG_PATH`
/// env var (set by `mlflow server --app-name basic-auth`). We opt into the auth
/// API surface when it is present.
///
/// T9.8 SEAM: full `basic_auth.ini` parsing (`default_permission`,
/// `database_uri`, `read_database_uri`, admin bootstrap, cache config) lands in
/// T9.8. Until then we take an env-driven shortcut: the config-path presence is
/// the enable flag, and the auth DB URI comes from `MLFLOW_AUTH_DATABASE_URI`
/// (default `sqlite:///basic_auth.db`, matching the shipped ini's
/// `database_uri`).
const MLFLOW_AUTH_CONFIG_PATH_ENV: &str = "MLFLOW_AUTH_CONFIG_PATH";
const MLFLOW_AUTH_DATABASE_URI_ENV: &str = "MLFLOW_AUTH_DATABASE_URI";
const MLFLOW_AUTH_READ_DATABASE_URI_ENV: &str = "MLFLOW_AUTH_READ_DATABASE_URI";
/// The shipped `basic_auth.ini` default (`database_uri = sqlite:///basic_auth.db`).
const DEFAULT_AUTH_DATABASE_URI: &str = "sqlite:///basic_auth.db";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let config = ServerConfig::from_cli(cli)?;

    // Serve the tracking API when a backend store is configured; otherwise run
    // the ops-only app (health/version/metrics). The store is verified against
    // the expected Alembic head at connect time.
    let app = match &config.backend_store_uri {
        Some(uri) => {
            let db = Db::connect_and_verify_with(uri, PoolConfig::from_env()).await?;
            let artifact_root = config
                .default_artifact_root
                .clone()
                .unwrap_or_else(|| DEFAULT_ARTIFACT_ROOT.to_string());
            // The webhook store shares the tracking DB pool (`Db` is a cheap
            // Arc-backed clone). Its Fernet cipher is resolved from
            // `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` (ephemeral key when unset).
            let webhook_store = WebhookStore::new(db.clone())?;
            // The async delivery engine (T8.3) over the same store. Scoped to the
            // default workspace (single-tenant server); T8.4's registry event
            // triggers call `dispatcher.fire(event, payload)`.
            let webhook_dispatcher = WebhookDispatcher::new(
                webhook_store.clone(),
                mlflow_server::workspace::DEFAULT_WORKSPACE_NAME,
            );
            let store = TrackingStore::new(db.clone(), artifact_root);
            // The registry tables live in the same Alembic-migrated database as
            // the tracking tables, so the registry store shares the same `Db`
            // pool (`_get_model_registry_store()`, `handlers.py:674`).
            let registry_store = RegistryStore::new(store.db().clone());

            // Resolve the `--artifacts-destination` proxy repo once (parity with
            // Python's memoized `_artifact_repo`). Only local-FS/`file:` URIs are
            // wired in v1; cloud schemes error at request time.
            let proxied_repo = match &config.artifacts_destination {
                Some(dest) => Some(mlflow_artifacts::factory::repo_from_uri(dest)?),
                None => None,
            };
            let mut app_state = AppState::with_registry(
                store,
                registry_store,
                config.serve_artifacts,
                proxied_repo,
                config.artifacts_destination.clone(),
            )
            .with_webhook_store(webhook_store, webhook_dispatcher);

            // Enable the auth/RBAC API (T9.2 users, later T9.3 roles) when the
            // basic-auth app is configured. See the T9.8 SEAM note above.
            if let Some(auth_store) = build_auth_store().await? {
                app_state = app_state.with_auth_store(auth_store);
            }

            // Workspace REST endpoints (T10.2) are enabled iff
            // `MLFLOW_ENABLE_WORKSPACES` is truthy (`MLFLOW_ENABLE_WORKSPACES.get()`,
            // default False). When on, the workspace store shares the tracking DB
            // pool; `MLFLOW_WORKSPACE_STORE_URI` (unset → tracking URI) only names
            // the `mlflow gc` hint. When off, the endpoints return a 503.
            if workspaces_enabled() {
                let workspace_uri = std::env::var("MLFLOW_WORKSPACE_STORE_URI")
                    .ok()
                    .unwrap_or_else(|| uri.clone());
                let workspace_store = WorkspaceStore::new(db.clone(), workspace_uri);
                app_state = app_state.with_workspace_store(workspace_store);
            }
            build_app_with_state(&config, app_state)
        }
        None => build_app(&config),
    };

    let listener = TcpListener::bind((config.host.as_str(), config.port)).await?;
    let local_addr = listener.local_addr()?;
    tracing::info!(address = %local_addr, static_prefix = ?config.static_prefix, "mlflow-server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("mlflow-server shut down");
    Ok(())
}

/// Build the auth store when the basic-auth app is enabled, else `None`.
///
/// T9.8 SEAM: this reads env vars instead of parsing `basic_auth.ini`. Presence
/// of `MLFLOW_AUTH_CONFIG_PATH` (set by `--app-name basic-auth`) is the enable
/// flag; the DB URI comes from `MLFLOW_AUTH_DATABASE_URI` (default
/// `sqlite:///basic_auth.db`) and the optional read replica from
/// `MLFLOW_AUTH_READ_DATABASE_URI`. The admin-user bootstrap
/// (`create_admin_user`) also belongs to T9.8's full config wiring.
async fn build_auth_store() -> anyhow::Result<Option<AuthStore>> {
    if std::env::var_os(MLFLOW_AUTH_CONFIG_PATH_ENV).is_none() {
        return Ok(None);
    }
    let db_uri = std::env::var(MLFLOW_AUTH_DATABASE_URI_ENV)
        .unwrap_or_else(|_| DEFAULT_AUTH_DATABASE_URI.to_string());
    let read_uri = std::env::var(MLFLOW_AUTH_READ_DATABASE_URI_ENV).ok();
    let auth_db = AuthDb::connect_and_verify(&db_uri, read_uri.as_deref()).await?;
    tracing::info!(db_uri = %db_uri, "basic-auth app enabled; auth API mounted");
    Ok(Some(AuthStore::new(auth_db)))
}

/// `MLFLOW_ENABLE_WORKSPACES.get()` (`mlflow/environment_variables.py:116`,
/// default `False`): truthy iff the env var is `"true"`/`"1"` (case-insensitive).
fn workspaces_enabled() -> bool {
    std::env::var("MLFLOW_ENABLE_WORKSPACES")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "true" | "1"))
        .unwrap_or(false)
}

/// Resolves once SIGINT (Ctrl-C) or SIGTERM is received, so
/// `with_graceful_shutdown` can stop accepting new connections and let
/// in-flight requests finish.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install SIGINT handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT, shutting down"),
        () = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}
