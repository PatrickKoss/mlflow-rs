//! `mlflow-server`: the Rust MLflow tracking server binary.
//!
//! Thin entry point: parses CLI args, initializes tracing, builds the
//! `Router` via [`mlflow_server::build_app`], and serves it with graceful
//! shutdown on SIGINT/SIGTERM. All actual app wiring lives in `lib.rs` so it
//! can be exercised directly in tests.

use clap::Parser;
use mlflow_server::{build_app, build_app_with_state, AppState, Cli, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use mlflow_webhooks::WebhookStore;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

/// `DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH` (`mlflow/store/tracking/__init__.py:12`).
const DEFAULT_ARTIFACT_ROOT: &str = "./mlruns";

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
            let store = TrackingStore::new(db, artifact_root);

            // Resolve the `--artifacts-destination` proxy repo once (parity with
            // Python's memoized `_artifact_repo`). Only local-FS/`file:` URIs are
            // wired in v1; cloud schemes error at request time.
            let proxied_repo = match &config.artifacts_destination {
                Some(dest) => Some(mlflow_artifacts::factory::repo_from_uri(dest)?),
                None => None,
            };
            let app_state = AppState::with_artifacts(
                store,
                config.serve_artifacts,
                proxied_repo,
                config.artifacts_destination.clone(),
            )
            .with_webhook_store(webhook_store);
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
