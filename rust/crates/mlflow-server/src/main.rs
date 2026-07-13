//! `mlflow-server`: the Rust MLflow tracking server binary.
//!
//! Thin entry point: parses CLI args, initializes tracing, builds the
//! `Router` via [`mlflow_server::build_app`], and serves it with graceful
//! shutdown on SIGINT/SIGTERM. All actual app wiring lives in `lib.rs` so it
//! can be exercised directly in tests.

use clap::Parser;
use mlflow_server::{build_app, Cli, ServerConfig};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let config = ServerConfig::from_cli(cli)?;

    let app = build_app(&config);

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
