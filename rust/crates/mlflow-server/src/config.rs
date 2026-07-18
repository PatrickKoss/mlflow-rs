//! CLI configuration for `mlflow-server`.
//!
//! Mirrors the `mlflow server` command (`mlflow/cli/__init__.py`) flag surface
//! as closely as the Rust server can. Flag names, defaults, and env-var
//! fallbacks match Python; the flag wins over its env var (clap `env = ...`
//! semantics, matching click's `envvar=`). See `CLI_PARITY.md` for the full
//! flag → status matrix.
//!
//! Static-prefix validation matches `_validate_static_prefix`
//! (`mlflow/cli/__init__.py:353-364`) exactly: must start with `/`, must not
//! end with `/`.

use std::fmt;

use clap::Parser;

/// Env var Python's `--static-prefix` reads as a fallback
/// (`mlflow/cli/__init__.py`).
pub const MLFLOW_STATIC_PREFIX_ENV_VAR: &str = "MLFLOW_STATIC_PREFIX";

/// Security-middleware env vars (`mlflow/environment_variables.py`).
/// Python has no CLI flags for these on some paths — they are env-backed — but
/// we expose CLI flags too (T11.2), the flag taking precedence over the env var.
pub const MLFLOW_SERVER_ALLOWED_HOSTS_ENV_VAR: &str = "MLFLOW_SERVER_ALLOWED_HOSTS";
pub const MLFLOW_SERVER_CORS_ALLOWED_ORIGINS_ENV_VAR: &str = "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS";
pub const MLFLOW_SERVER_X_FRAME_OPTIONS_ENV_VAR: &str = "MLFLOW_SERVER_X_FRAME_OPTIONS";
pub const MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE_ENV_VAR: &str =
    "MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE";
/// Python's job-runner enable gate (default `True`).
pub const MLFLOW_SERVER_ENABLE_JOB_EXECUTION_ENV_VAR: &str = "MLFLOW_SERVER_ENABLE_JOB_EXECUTION";

/// `MLFLOW_ENABLE_WORKSPACES` (`environment_variables.py`, default `False`).
pub const MLFLOW_ENABLE_WORKSPACES_ENV_VAR: &str = "MLFLOW_ENABLE_WORKSPACES";
/// `MLFLOW_WORKSPACE_STORE_URI` (`environment_variables.py`).
pub const MLFLOW_WORKSPACE_STORE_URI_ENV_VAR: &str = "MLFLOW_WORKSPACE_STORE_URI";
/// `MLFLOW_AUTH_CONFIG_PATH`: Python's enable signal for the basic-auth app,
/// set by `mlflow server --app-name basic-auth`.
pub const MLFLOW_AUTH_CONFIG_PATH_ENV_VAR: &str = "MLFLOW_AUTH_CONFIG_PATH";

/// Default `X-Frame-Options` value (`MLFLOW_SERVER_X_FRAME_OPTIONS` default).
pub const DEFAULT_X_FRAME_OPTIONS: &str = "SAMEORIGIN";

/// The only `--app-name` value the Rust server understands. Python's choice set
/// comes from the `mlflow.app` entry points (`pyproject.toml`), which currently
/// contains exactly `basic-auth`.
pub const APP_NAME_BASIC_AUTH: &str = "basic-auth";

#[derive(Debug, Parser)]
#[command(name = "mlflow-server", about = "MLflow tracking server (Rust)")]
pub struct Cli {
    /// The network address to listen on (`--host`/`-h`, env `MLFLOW_HOST`).
    #[arg(long, short = 'H', env = "MLFLOW_HOST", default_value = "127.0.0.1")]
    pub host: String,

    /// The port to listen on (`--port`/`-p`, env `MLFLOW_PORT`).
    #[arg(long, short, env = "MLFLOW_PORT", default_value_t = 5000)]
    pub port: u16,

    /// Number of worker processes. Accepted for deploy-script parity but the
    /// Rust server is async (single tokio runtime); the value is logged and
    /// otherwise ignored (`--workers`/`-w`, env `MLFLOW_WORKERS`).
    #[arg(long, short, env = "MLFLOW_WORKERS")]
    pub workers: Option<u32>,

    /// A prefix prepended to the path of all routes (`--static-prefix`, env
    /// `MLFLOW_STATIC_PREFIX`).
    #[arg(long, env = MLFLOW_STATIC_PREFIX_ENV_VAR)]
    pub static_prefix: Option<String>,

    /// The SQLAlchemy-style backend store URI (`--backend-store-uri`, env
    /// `MLFLOW_BACKEND_STORE_URI`), e.g. `sqlite:///mlflow.db`,
    /// `postgresql://...`. Required to serve the tracking API.
    #[arg(long, env = "MLFLOW_BACKEND_STORE_URI")]
    pub backend_store_uri: Option<String>,

    /// URI for a read-only database replica (`--read-replica-backend-store-uri`,
    /// env `MLFLOW_READ_REPLICA_BACKEND_STORE_URI`). Accepted and carried into
    /// [`ServerConfig`]; the Rust tracking store does not yet split reads onto a
    /// replica (SEAM: see `CLI_PARITY.md`).
    #[arg(long, env = "MLFLOW_READ_REPLICA_BACKEND_STORE_URI")]
    pub read_replica_backend_store_uri: Option<String>,

    /// URI to persist registered models (`--registry-store-uri`, env
    /// `MLFLOW_REGISTRY_STORE_URI`). If unset, the backend store URI is used.
    /// The Rust registry shares the tracking DB, so a *different* URI is
    /// rejected (fail-loud) — see [`ServerConfig::from_cli`].
    #[arg(long, env = "MLFLOW_REGISTRY_STORE_URI")]
    pub registry_store_uri: Option<String>,

    /// The default artifact root URI (`--default-artifact-root`, env
    /// `MLFLOW_DEFAULT_ARTIFACT_ROOT`).
    #[arg(long, env = "MLFLOW_DEFAULT_ARTIFACT_ROOT")]
    pub default_artifact_root: Option<String>,

    /// Enable serving of artifacts through the `mlflow-artifacts` proxy
    /// (`--serve-artifacts`/`--no-serve-artifacts`, env `MLFLOW_SERVE_ARTIFACTS`).
    /// Python defaults this to `True`; we mirror that.
    #[arg(
        long,
        env = "MLFLOW_SERVE_ARTIFACTS",
        default_value_t = true,
        overrides_with = "no_serve_artifacts"
    )]
    pub serve_artifacts: bool,

    /// Disables the artifact proxy (`--no-serve-artifacts`). Clap flag pair for
    /// the click `--serve-artifacts/--no-serve-artifacts` toggle.
    #[arg(long, conflicts_with = "serve_artifacts")]
    pub no_serve_artifacts: bool,

    /// Serve only the artifact proxy routes, disabling tracking endpoints
    /// (`--artifacts-only`, env `MLFLOW_ARTIFACTS_ONLY`). Default False.
    #[arg(long, env = "MLFLOW_ARTIFACTS_ONLY", default_value_t = false)]
    pub artifacts_only: bool,

    /// The base artifact-store URI the `mlflow-artifacts` proxy reads/writes
    /// (`--artifacts-destination`, env `MLFLOW_ARTIFACTS_DESTINATION`). Python
    /// defaults to `./mlartifacts`.
    #[arg(long, env = "MLFLOW_ARTIFACTS_DESTINATION")]
    pub artifacts_destination: Option<String>,

    /// Comma-separated allowed `Host` headers (`--allowed-hosts`, env
    /// `MLFLOW_SERVER_ALLOWED_HOSTS`). `*` disables the check.
    #[arg(long, env = MLFLOW_SERVER_ALLOWED_HOSTS_ENV_VAR)]
    pub allowed_hosts: Option<String>,

    /// Comma-separated allowed CORS origins (`--cors-allowed-origins`, env
    /// `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`).
    #[arg(long, env = MLFLOW_SERVER_CORS_ALLOWED_ORIGINS_ENV_VAR)]
    pub cors_allowed_origins: Option<String>,

    /// `X-Frame-Options` header value (`--x-frame-options`, env
    /// `MLFLOW_SERVER_X_FRAME_OPTIONS`, default `SAMEORIGIN`).
    #[arg(long, env = MLFLOW_SERVER_X_FRAME_OPTIONS_ENV_VAR)]
    pub x_frame_options: Option<String>,

    /// Activate the Prometheus exporter on `/metrics` (`--expose-prometheus`,
    /// env `MLFLOW_EXPOSE_PROMETHEUS`). In Python this is a directory path for
    /// the multiprocess collector; the Rust server has no multiprocess model, so
    /// any value is treated as "enable" (the path is otherwise unused).
    #[arg(long, env = "MLFLOW_EXPOSE_PROMETHEUS")]
    pub expose_prometheus: Option<String>,

    /// Application name (`--app-name`). Only `basic-auth` is accepted; it enables
    /// the auth/RBAC API. Any other value fails loudly.
    #[arg(long)]
    pub app_name: Option<String>,

    /// Workspace provider backend URI (`--workspace-store-uri`, env
    /// `MLFLOW_WORKSPACE_STORE_URI`). Only takes effect with workspaces enabled.
    #[arg(long, env = MLFLOW_WORKSPACE_STORE_URI_ENV_VAR)]
    pub workspace_store_uri: Option<String>,

    /// Enable workspaces mode (`--enable-workspaces`). Overrides
    /// `MLFLOW_ENABLE_WORKSPACES` when passed.
    #[arg(long, conflicts_with = "disable_workspaces")]
    pub enable_workspaces: bool,

    /// Disable workspaces mode (`--disable-workspaces`). Overrides
    /// `MLFLOW_ENABLE_WORKSPACES` when passed.
    #[arg(long)]
    pub disable_workspaces: bool,
}

/// Error returned when CLI/env resolution fails loudly. Includes both the
/// static-prefix validation (`_validate_static_prefix`) and the parity
/// fail-loud cases required by T11.1's AC ("unsupported flags fail loudly").
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error("--static-prefix must begin with a '/'.")]
    StaticPrefixMissingLeadingSlash,
    #[error("--static-prefix should not end with a '/'.")]
    StaticPrefixTrailingSlash,
    #[error(
        "--app-name {0:?} is not supported by the Rust MLflow server. \
         The only supported value is 'basic-auth' (enables the auth/RBAC API)."
    )]
    UnsupportedAppName(String),
    #[error(
        "--registry-store-uri ({registry:?}) differs from --backend-store-uri \
         ({backend:?}). The Rust MLflow server keeps the model registry in the \
         same database as the tracking store, so a separate registry URI is not \
         supported. Point both at the same database or omit --registry-store-uri."
    )]
    RegistryUriMismatch { registry: String, backend: String },
    #[error(
        "{name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), but got {value}"
    )]
    InvalidBooleanEnvironment { name: &'static str, value: String },
}

/// Back-compat alias: pre-T11.1 code referred to `StaticPrefixError`.
pub type StaticPrefixError = ConfigError;

/// Resolved, validated server configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// `--workers`: accepted for parity, not acted on (async server). Stored so
    /// startup can log the interpretation.
    pub workers: Option<u32>,
    pub static_prefix: Option<String>,
    pub backend_store_uri: Option<String>,
    /// `--read-replica-backend-store-uri`: accepted and stored; not yet wired
    /// into the tracking store's read path (SEAM).
    pub read_replica_backend_store_uri: Option<String>,
    /// `--registry-store-uri`: after validation this equals the backend store
    /// URI (the registry shares the tracking DB). Stored for observability.
    pub registry_store_uri: Option<String>,
    pub default_artifact_root: Option<String>,
    /// Whether the `mlflow-artifacts` proxy surface is enabled
    /// (`--serve-artifacts`). Mirrors `_is_serving_proxied_artifacts()`.
    pub serve_artifacts: bool,
    /// Artifacts-only mode (`--artifacts-only`): only the artifact proxy routes
    /// are registered; tracking endpoints are omitted.
    pub artifacts_only: bool,
    /// The `--artifacts-destination` base URI for the proxy repo, if configured.
    pub artifacts_destination: Option<String>,
    pub allowed_hosts: Option<Vec<String>>,
    pub cors_allowed_origins: Option<Vec<String>>,
    pub x_frame_options: String,
    /// Disable Host/CORS/security-header middleware, matching
    /// `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true` on Python servers.
    pub disable_security_middleware: bool,
    /// Whether the Prometheus `/metrics` exporter is enabled
    /// (`--expose-prometheus`). Python gates the exporter on the env var being
    /// set; we gate route registration on this being true.
    pub expose_prometheus: bool,
    /// Whether the basic-auth app is enabled (`--app-name basic-auth`, or
    /// `MLFLOW_AUTH_CONFIG_PATH` present).
    pub auth_enabled: bool,
    /// Whether the DB-backed job runner may start. Python defaults this gate to
    /// enabled and disables it only for `false`/`0`.
    pub job_execution_enabled: bool,
    /// Resolved workspaces-enabled signal (flags override
    /// `MLFLOW_ENABLE_WORKSPACES`).
    pub enable_workspaces: bool,
    /// `--workspace-store-uri` (or `MLFLOW_WORKSPACE_STORE_URI`); only meaningful
    /// when `enable_workspaces` is true.
    pub workspace_store_uri: Option<String>,
}

impl Default for ServerConfig {
    /// Matches the `mlflow server` defaults for the fields that have one, so
    /// tests can spread `..ServerConfig::default()` and only set what they
    /// exercise. `expose_prometheus` defaults to `true` here (unlike the CLI,
    /// which defaults it off) so the `/metrics` route is available in HTTP
    /// integration tests without every fixture opting in.
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 0,
            workers: None,
            static_prefix: None,
            backend_store_uri: None,
            read_replica_backend_store_uri: None,
            registry_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_only: false,
            artifacts_destination: None,
            allowed_hosts: None,
            cors_allowed_origins: None,
            x_frame_options: DEFAULT_X_FRAME_OPTIONS.to_string(),
            disable_security_middleware: false,
            expose_prometheus: true,
            auth_enabled: false,
            job_execution_enabled: true,
            enable_workspaces: false,
            workspace_store_uri: None,
        }
    }
}

impl fmt::Display for ServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

impl ServerConfig {
    /// Builds a `ServerConfig` from parsed CLI args plus the process
    /// environment, applying the same precedence and validation as Python's
    /// click-based CLI. Fails loudly for unsupported flag values (per T11.1 AC).
    pub fn from_cli(cli: Cli) -> Result<Self, ConfigError> {
        let static_prefix = validate_static_prefix(cli.static_prefix)?;

        let allowed_hosts = split_csv(cli.allowed_hosts);
        let cors_allowed_origins = split_csv(cli.cors_allowed_origins);
        let x_frame_options = cli
            .x_frame_options
            .unwrap_or_else(|| DEFAULT_X_FRAME_OPTIONS.to_string());

        // `--serve-artifacts` defaults true; `--no-serve-artifacts` flips it.
        let serve_artifacts = cli.serve_artifacts && !cli.no_serve_artifacts;

        // `--app-name`: only `basic-auth` is accepted. Python's create_app path
        // exports `MLFLOW_AUTH_CONFIG_PATH`; we also honour that env var as an
        // enable signal (matching `main.rs`'s pre-T11.1 behaviour).
        let auth_enabled = match cli.app_name.as_deref() {
            Some(APP_NAME_BASIC_AUTH) => true,
            Some(other) => return Err(ConfigError::UnsupportedAppName(other.to_string())),
            None => std::env::var_os(MLFLOW_AUTH_CONFIG_PATH_ENV_VAR).is_some(),
        };

        // `--registry-store-uri`: the Rust registry shares the tracking DB, so a
        // *different* URI cannot be honoured. Match Python's "defaults to backend
        // URI" and fail loudly if a distinct one is requested.
        let registry_store_uri = match (&cli.registry_store_uri, &cli.backend_store_uri) {
            (Some(reg), Some(backend)) if reg != backend => {
                return Err(ConfigError::RegistryUriMismatch {
                    registry: reg.clone(),
                    backend: backend.clone(),
                });
            }
            (Some(reg), _) => Some(reg.clone()),
            (None, backend) => backend.clone(),
        };

        // `--enable-workspaces`/`--disable-workspaces` override the env var,
        // matching Python's ParameterSource precedence (COMMANDLINE wins).
        let enable_workspaces = if cli.enable_workspaces {
            true
        } else if cli.disable_workspaces {
            false
        } else {
            env_truthy(MLFLOW_ENABLE_WORKSPACES_ENV_VAR)
        };
        let job_execution_enabled = env_bool(MLFLOW_SERVER_ENABLE_JOB_EXECUTION_ENV_VAR, true)?;

        Ok(Self {
            host: cli.host,
            port: cli.port,
            workers: cli.workers,
            static_prefix,
            backend_store_uri: cli.backend_store_uri,
            read_replica_backend_store_uri: cli.read_replica_backend_store_uri,
            registry_store_uri,
            default_artifact_root: cli.default_artifact_root,
            serve_artifacts,
            artifacts_only: cli.artifacts_only,
            artifacts_destination: cli.artifacts_destination,
            allowed_hosts,
            cors_allowed_origins,
            x_frame_options,
            disable_security_middleware: env_truthy(
                MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE_ENV_VAR,
            ),
            expose_prometheus: cli.expose_prometheus.is_some(),
            auth_enabled,
            job_execution_enabled,
            enable_workspaces,
            workspace_store_uri: cli.workspace_store_uri,
        })
    }
}

/// Split a comma-separated flag/env value, mirroring
/// `get_allowed_hosts_from_env` / `get_allowed_origins_from_env`
/// (`security_utils.py`): split on `,`, trim each entry. Empty/unset → `None`.
fn split_csv(value: Option<String>) -> Option<Vec<String>> {
    let raw = value?;
    if raw.is_empty() {
        return None;
    }
    Some(raw.split(',').map(|s| s.trim().to_string()).collect())
}

/// `_EnvironmentVariable`-style truthiness for the boolean workspaces env var:
/// truthy iff `true`/`1` (case-insensitive), matching Python's parser.
fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "true" | "1"))
        .unwrap_or(false)
}

fn env_bool(name: &'static str, default: bool) -> Result<bool, ConfigError> {
    let Ok(value) = std::env::var(name) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(ConfigError::InvalidBooleanEnvironment { name, value }),
    }
}

/// Validates a static prefix the same way Python's `_validate_static_prefix`
/// does: `None` passes through untouched, otherwise the value must start with
/// `/` and must not end with `/`.
fn validate_static_prefix(value: Option<String>) -> Result<Option<String>, ConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.starts_with('/') {
        return Err(ConfigError::StaticPrefixMissingLeadingSlash);
    }
    if value.ends_with('/') {
        return Err(ConfigError::StaticPrefixTrailingSlash);
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate process-global env vars, since `cargo test`
    /// runs tests concurrently.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn parse(args: &[&str]) -> Cli {
        let mut full = vec!["mlflow-server"];
        full.extend_from_slice(args);
        Cli::parse_from(full)
    }

    #[test]
    fn none_is_valid() {
        assert_eq!(validate_static_prefix(None), Ok(None));
    }

    #[test]
    fn valid_prefix_passes_through() {
        assert_eq!(
            validate_static_prefix(Some("/mlflow".to_string())),
            Ok(Some("/mlflow".to_string()))
        );
    }

    #[test]
    fn missing_leading_slash_is_rejected() {
        assert_eq!(
            validate_static_prefix(Some("mlflow".to_string())),
            Err(ConfigError::StaticPrefixMissingLeadingSlash)
        );
    }

    #[test]
    fn trailing_slash_is_rejected() {
        assert_eq!(
            validate_static_prefix(Some("/mlflow/".to_string())),
            Err(ConfigError::StaticPrefixTrailingSlash)
        );
    }

    #[test]
    fn root_prefix_is_valid() {
        // "/" starts with '/' AND ends with '/'; Python's checks are two
        // independent `if`s (not elif), so "/" hits the trailing-slash branch.
        assert_eq!(
            validate_static_prefix(Some("/".to_string())),
            Err(ConfigError::StaticPrefixTrailingSlash)
        );
    }

    #[test]
    fn defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&[])).unwrap();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 5000);
        assert_eq!(config.static_prefix, None);
        assert!(config.serve_artifacts);
        assert!(!config.artifacts_only);
        assert_eq!(config.x_frame_options, "SAMEORIGIN");
        assert!(!config.disable_security_middleware);
        assert!(!config.expose_prometheus);
        assert!(!config.auth_enabled);
        assert!(config.job_execution_enabled);
        assert!(!config.enable_workspaces);
        assert_eq!(config.workers, None);
    }

    #[test]
    fn no_serve_artifacts_flips_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&["--no-serve-artifacts"])).unwrap();
        assert!(!config.serve_artifacts);
    }

    #[test]
    fn artifacts_only_parsed() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&["--artifacts-only"])).unwrap();
        assert!(config.artifacts_only);
    }

    #[test]
    fn workers_accepted_but_stored() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&["--workers", "8"])).unwrap();
        assert_eq!(config.workers, Some(8));
    }

    #[test]
    fn expose_prometheus_sets_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config =
            ServerConfig::from_cli(parse(&["--expose-prometheus", "/tmp/metrics"])).unwrap();
        assert!(config.expose_prometheus);
    }

    #[test]
    fn app_name_basic_auth_enables_auth() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&["--app-name", "basic-auth"])).unwrap();
        assert!(config.auth_enabled);
    }

    #[test]
    fn unknown_app_name_fails_loudly() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let err = ServerConfig::from_cli(parse(&["--app-name", "bogus"])).unwrap_err();
        assert_eq!(err, ConfigError::UnsupportedAppName("bogus".to_string()));
    }

    #[test]
    fn auth_config_path_env_enables_auth() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        // SAFETY: serialized by ENV_LOCK.
        unsafe {
            std::env::set_var(MLFLOW_AUTH_CONFIG_PATH_ENV_VAR, "/etc/basic_auth.ini");
        }
        let config = ServerConfig::from_cli(parse(&[])).unwrap();
        assert!(config.auth_enabled);
        unsafe {
            std::env::remove_var(MLFLOW_AUTH_CONFIG_PATH_ENV_VAR);
        }
    }

    #[test]
    fn registry_uri_matching_backend_is_ok() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&[
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--registry-store-uri",
            "sqlite:///mlflow.db",
        ]))
        .unwrap();
        assert_eq!(
            config.registry_store_uri.as_deref(),
            Some("sqlite:///mlflow.db")
        );
    }

    #[test]
    fn registry_uri_defaults_to_backend() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config =
            ServerConfig::from_cli(parse(&["--backend-store-uri", "sqlite:///mlflow.db"])).unwrap();
        assert_eq!(
            config.registry_store_uri.as_deref(),
            Some("sqlite:///mlflow.db")
        );
    }

    #[test]
    fn registry_uri_mismatch_fails_loudly() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let err = ServerConfig::from_cli(parse(&[
            "--backend-store-uri",
            "sqlite:///a.db",
            "--registry-store-uri",
            "postgresql://other/db",
        ]))
        .unwrap_err();
        assert!(matches!(err, ConfigError::RegistryUriMismatch { .. }));
    }

    #[test]
    fn enable_workspaces_flag_overrides_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_ENABLE_WORKSPACES_ENV_VAR, "false");
        }
        let config = ServerConfig::from_cli(parse(&["--enable-workspaces"])).unwrap();
        assert!(config.enable_workspaces);
        unsafe {
            std::env::remove_var(MLFLOW_ENABLE_WORKSPACES_ENV_VAR);
        }
    }

    #[test]
    fn disable_workspaces_flag_overrides_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_ENABLE_WORKSPACES_ENV_VAR, "true");
        }
        let config = ServerConfig::from_cli(parse(&["--disable-workspaces"])).unwrap();
        assert!(!config.enable_workspaces);
        unsafe {
            std::env::remove_var(MLFLOW_ENABLE_WORKSPACES_ENV_VAR);
        }
    }

    #[test]
    fn workspaces_falls_back_to_env_when_no_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_ENABLE_WORKSPACES_ENV_VAR, "true");
        }
        let config = ServerConfig::from_cli(parse(&[])).unwrap();
        assert!(config.enable_workspaces);
        unsafe {
            std::env::remove_var(MLFLOW_ENABLE_WORKSPACES_ENV_VAR);
        }
    }

    #[test]
    fn cli_flag_takes_precedence_over_static_prefix_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_STATIC_PREFIX_ENV_VAR, "/env-prefix");
        }
        let config = ServerConfig::from_cli(parse(&["--static-prefix", "/cli-prefix"])).unwrap();
        assert_eq!(config.static_prefix.as_deref(), Some("/cli-prefix"));
        unsafe {
            std::env::remove_var(MLFLOW_STATIC_PREFIX_ENV_VAR);
        }
    }

    #[test]
    fn falls_back_to_static_prefix_env_when_flag_absent() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_STATIC_PREFIX_ENV_VAR, "/env-prefix");
        }
        let config = ServerConfig::from_cli(parse(&[])).unwrap();
        assert_eq!(config.static_prefix.as_deref(), Some("/env-prefix"));
        unsafe {
            std::env::remove_var(MLFLOW_STATIC_PREFIX_ENV_VAR);
        }
    }

    #[test]
    fn allowed_hosts_and_cors_split_and_trim() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = ServerConfig::from_cli(parse(&[
            "--allowed-hosts",
            "a.com, b.com",
            "--cors-allowed-origins",
            "https://x.com,https://y.com",
        ]))
        .unwrap();
        assert_eq!(
            config.allowed_hosts,
            Some(vec!["a.com".to_string(), "b.com".to_string()])
        );
        assert_eq!(
            config.cors_allowed_origins,
            Some(vec![
                "https://x.com".to_string(),
                "https://y.com".to_string()
            ])
        );
    }

    #[test]
    fn disable_security_middleware_env_is_honored() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE_ENV_VAR, "true");
        }
        let config = ServerConfig::from_cli(parse(&[])).unwrap();
        assert!(config.disable_security_middleware);
        unsafe {
            std::env::remove_var(MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE_ENV_VAR);
        }
    }

    #[test]
    fn job_execution_gate_matches_python_boolean_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var(MLFLOW_SERVER_ENABLE_JOB_EXECUTION_ENV_VAR, "0");
        }
        assert!(
            !ServerConfig::from_cli(parse(&[]))
                .unwrap()
                .job_execution_enabled
        );

        unsafe {
            std::env::set_var(MLFLOW_SERVER_ENABLE_JOB_EXECUTION_ENV_VAR, "invalid");
        }
        assert!(matches!(
            ServerConfig::from_cli(parse(&[])),
            Err(ConfigError::InvalidBooleanEnvironment { .. })
        ));
        unsafe {
            std::env::remove_var(MLFLOW_SERVER_ENABLE_JOB_EXECUTION_ENV_VAR);
        }
    }

    /// Clears every env var the parser reads so a test's outcome does not depend
    /// on the ambient environment (clap `env = ...` reads the live process env).
    fn clear_env() {
        for var in [
            MLFLOW_STATIC_PREFIX_ENV_VAR,
            MLFLOW_SERVER_ALLOWED_HOSTS_ENV_VAR,
            MLFLOW_SERVER_CORS_ALLOWED_ORIGINS_ENV_VAR,
            MLFLOW_SERVER_X_FRAME_OPTIONS_ENV_VAR,
            MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE_ENV_VAR,
            MLFLOW_SERVER_ENABLE_JOB_EXECUTION_ENV_VAR,
            MLFLOW_ENABLE_WORKSPACES_ENV_VAR,
            MLFLOW_WORKSPACE_STORE_URI_ENV_VAR,
            MLFLOW_AUTH_CONFIG_PATH_ENV_VAR,
            "MLFLOW_HOST",
            "MLFLOW_PORT",
            "MLFLOW_WORKERS",
            "MLFLOW_BACKEND_STORE_URI",
            "MLFLOW_READ_REPLICA_BACKEND_STORE_URI",
            "MLFLOW_REGISTRY_STORE_URI",
            "MLFLOW_DEFAULT_ARTIFACT_ROOT",
            "MLFLOW_SERVE_ARTIFACTS",
            "MLFLOW_ARTIFACTS_ONLY",
            "MLFLOW_ARTIFACTS_DESTINATION",
            "MLFLOW_EXPOSE_PROMETHEUS",
        ] {
            // SAFETY: serialized by ENV_LOCK held in each caller.
            unsafe {
                std::env::remove_var(var);
            }
        }
    }
}
