//! `mlflow-server`: the Rust MLflow tracking server binary.
//!
//! Thin entry point: parses CLI args, initializes tracing, builds the
//! `Router` via [`mlflow_server::build_app`], and serves it with graceful
//! shutdown on SIGINT/SIGTERM. All actual app wiring lives in `lib.rs` so it
//! can be exercised directly in tests.

use clap::Parser;
use mlflow_auth::{
    create_admin_user, warn_if_default_admin_password, AuthConfig, AuthDb, AuthStore,
};
use mlflow_registry::RegistryStore;
use mlflow_server::job_runner::{JobRunner, JobRunnerConfig};
use mlflow_server::native_worker::{
    native_job_functions, resolve_worker_program, NativeWorkerExecutor,
};
use mlflow_server::{build_app, build_app_with_state, AppState, Cli, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use mlflow_webhooks::{WebhookDispatcher, WebhookStore};
use std::sync::Arc;
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

    // `--workers` is accepted for deploy-script parity but the Rust server runs
    // a single async tokio runtime, so there are no worker processes to spawn.
    if let Some(workers) = config.workers {
        tracing::info!(
            workers,
            "--workers accepted but ignored: the Rust server is async (single tokio runtime)"
        );
    }
    // `--read-replica-backend-store-uri` is accepted and carried in the config,
    // but the Rust tracking store does not yet split reads onto a replica (SEAM,
    // see CLI_PARITY.md). Warn so the operator knows reads still hit the primary.
    if let Some(replica) = &config.read_replica_backend_store_uri {
        tracing::warn!(
            replica_uri = %replica,
            "--read-replica-backend-store-uri accepted but not yet wired: all reads use the primary backend store"
        );
    }

    let listener = TcpListener::bind((config.host.as_str(), config.port)).await?;
    let local_addr = listener.local_addr()?;

    // Serve the tracking API when a backend store is configured; otherwise run
    // the ops-only app (health/version/metrics). The store is verified against
    // the expected Alembic head at connect time.
    let mut online_scoring_scheduler = None;
    let (app, job_runner) = match &config.backend_store_uri {
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
            // basic-auth app is configured (`--app-name basic-auth` or
            // `MLFLOW_AUTH_CONFIG_PATH`; resolved in `ServerConfig`, T11.1).
            if config.auth_enabled {
                if let Some(auth_store) = build_auth_store().await? {
                    app_state = app_state.with_auth_store(auth_store);
                }
            }

            // Workspace REST endpoints (T10.2) are enabled iff the resolved
            // workspaces signal is on (`--enable-workspaces`/`--disable-workspaces`
            // overriding `MLFLOW_ENABLE_WORKSPACES`, T11.1). When on, the
            // workspace store shares the tracking DB pool; `--workspace-store-uri`
            // (unset → tracking URI) only names the `mlflow gc` hint. When off,
            // the endpoints return a 503.
            let mut runner_workspaces =
                vec![mlflow_server::workspace::DEFAULT_WORKSPACE_NAME.to_string()];
            if config.enable_workspaces {
                let workspace_uri = config
                    .workspace_store_uri
                    .clone()
                    .unwrap_or_else(|| uri.clone());
                let workspace_store = WorkspaceStore::new(db.clone(), workspace_uri);
                runner_workspaces = workspace_store
                    .list_workspaces()
                    .await?
                    .into_iter()
                    .map(|workspace| workspace.name)
                    .collect();
                app_state = app_state.with_workspace_store(workspace_store);
            } else {
                // Single-tenant startup guard (T10.3): if a previous
                // workspaces-enabled run left root entities outside the `default`
                // workspace, refuse to boot single-tenant (they'd be unreachable).
                // Mirrors Python's store-construction guard
                // (`INVALID_STATE`); byte-matched messages live in
                // `verify_single_tenant_data`.
                mlflow_store::verify_single_tenant_data(
                    &db,
                    &[
                        ("experiments", "experiments"),
                        ("registered_models", "registered models"),
                        ("webhooks", "webhooks"),
                    ],
                )
                .await?;
            }
            if config.job_execution_enabled {
                online_scoring_scheduler = Some(
                    mlflow_server::online_scoring_scheduler::OnlineScoringScheduler::new(
                        app_state.tracking_store().clone(),
                        app_state.workspace_store().cloned(),
                    ),
                );
            }
            let job_runner = if config.job_execution_enabled {
                let worker = resolve_worker_program().map_err(|error| {
                    anyhow::anyhow!(
                        "native job execution is enabled but mlflow-genai-worker is unavailable: {error}"
                    )
                })?;
                let server_uri = worker_server_uri(local_addr);
                let gateway_uri =
                    std::env::var("MLFLOW_GATEWAY_URI").unwrap_or_else(|_| server_uri.clone());
                let internal_token = std::env::var("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN")
                    .unwrap_or_else(|_| uuid::Uuid::new_v4().simple().to_string());
                let executor = NativeWorkerExecutor::new(worker)
                    .tracking_uri(server_uri)
                    .gateway_uri(gateway_uri)
                    .internal_gateway_token(internal_token);
                JobRunner::new(
                    app_state.job_store(),
                    Arc::new(executor),
                    native_job_functions()?,
                    runner_workspaces,
                    JobRunnerConfig::from_server_config(&config)?,
                )
                .start()
                .await?
            } else {
                None
            };
            (build_app_with_state(&config, app_state), job_runner)
        }
        None => (build_app(&config), None),
    };

    tracing::info!(address = %local_addr, static_prefix = ?config.static_prefix, "mlflow-server listening");

    let scheduler_task =
        online_scoring_scheduler.map(|scheduler| tokio::spawn(scheduler.run_periodic()));
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    if let Some(task) = scheduler_task {
        task.abort();
    }
    if let Some(runner) = job_runner {
        runner.shutdown().await;
    }
    serve_result?;

    tracing::info!("mlflow-server shut down");
    Ok(())
}

fn worker_server_uri(address: std::net::SocketAddr) -> String {
    let host = if address.ip().is_unspecified() {
        if address.is_ipv4() {
            "127.0.0.1".to_string()
        } else {
            "[::1]".to_string()
        }
    } else if address.is_ipv6() {
        format!("[{}]", address.ip())
    } else {
        address.ip().to_string()
    };
    format!("http://{host}:{}", address.port())
}

/// Build the auth store when the basic-auth app is enabled, else `None`.
///
/// Mirrors Python's `create_app` sequence (`auth/__init__.py:4648-4653`): parse
/// the config, connect + verify the auth DB at `database_uri` (+ optional
/// `read_database_uri` replica), then bootstrap the admin user
/// (`create_admin_user`) and warn if it still uses the shipped default password.
/// The parsed [`AuthConfig`] is carried into the [`AuthStore`] so the permission
/// validators read `default_permission` and the credential cache honours the
/// `auth_cache_*` fields.
///
/// The caller gates this on `config.auth_enabled` (`--app-name basic-auth` or
/// `MLFLOW_AUTH_CONFIG_PATH` present, resolved in `ServerConfig`). The ini file
/// is selected by `MLFLOW_AUTH_CONFIG_PATH` (unset → packaged `basic_auth.ini`),
/// and the DB URIs come from the ini's `database_uri` / `read_database_uri`,
/// exactly as Python resolves them, so the same file drives both servers. Note:
/// Python's `create_app` also requires `MLFLOW_FLASK_SERVER_SECRET_KEY` for Flask
/// CSRF; per plan D12 the Rust server owns its own per-process signup-CSRF secret
/// (see `auth_api::signup`), so that env var is intentionally *not* required here.
async fn build_auth_store() -> anyhow::Result<Option<AuthStore>> {
    let config = AuthConfig::read()?;
    let auth_db =
        AuthDb::connect_and_verify(&config.database_uri, config.read_database_uri.as_deref())
            .await?;
    let store = AuthStore::with_config(auth_db, config.clone());

    // Admin bootstrap, matching `create_app`'s create_admin_user +
    // _warn_if_default_admin_password (`auth/__init__.py:4652-4653`).
    create_admin_user(&store, &config.admin_username, &config.admin_password).await?;
    warn_if_default_admin_password(&config.admin_password);

    tracing::info!(
        db_uri = %config.database_uri,
        default_permission = %config.default_permission,
        credential_cache = store.credential_cache_enabled(),
        "basic-auth app enabled; auth API mounted"
    );
    Ok(Some(store))
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
