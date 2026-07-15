//! CLI configuration for `mlflow-server`.
//!
//! Mirrors the subset of `mlflow server` flags relevant to this task
//! (`mlflow/cli/__init__.py`): `--host`, `--port`, `--static-prefix`
//! (env var `MLFLOW_STATIC_PREFIX`, CLI flag wins). Static-prefix validation
//! matches `_validate_static_prefix` (`mlflow/cli/__init__.py:353-364`)
//! exactly: must start with `/`, must not end with `/`.

use std::fmt;

use clap::Parser;

/// Environment variable Python's `mlflow server --static-prefix` reads as a
/// fallback (`mlflow/cli/__init__.py:434`).
pub const MLFLOW_STATIC_PREFIX_ENV_VAR: &str = "MLFLOW_STATIC_PREFIX";

#[derive(Debug, Parser)]
#[command(name = "mlflow-server", about = "MLflow tracking server (Rust)")]
pub struct Cli {
    /// The network address to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// The port to listen on.
    #[arg(long, default_value_t = 5000)]
    pub port: u16,

    /// A prefix prepended to the path of all routes. Falls back to the
    /// `MLFLOW_STATIC_PREFIX` env var when not passed on the CLI.
    #[arg(long)]
    pub static_prefix: Option<String>,

    /// The SQLAlchemy-style backend store URI (`mlflow server
    /// --backend-store-uri`), e.g. `sqlite:///mlflow.db`,
    /// `postgresql://...`. Required to serve the tracking API.
    #[arg(long)]
    pub backend_store_uri: Option<String>,

    /// The default artifact root URI (`mlflow server --default-artifact-root`).
    /// Used as the parent of per-experiment default artifact locations.
    #[arg(long)]
    pub default_artifact_root: Option<String>,

    /// Enable serving of artifacts through the `mlflow-artifacts` proxy
    /// endpoints (`mlflow server --serve-artifacts` / `--no-serve-artifacts`).
    /// When on, the server sets `_SERVE_ARTIFACTS_ENV_VAR="true"`, gating the
    /// `MlflowArtifactsService` surface (`_disable_unless_serve_artifacts`).
    /// Python defaults this to `True`; we mirror that default so the proxy is
    /// live out of the box.
    #[arg(long, default_value_t = true)]
    pub serve_artifacts: bool,

    /// The base artifact-store URI the `mlflow-artifacts` proxy reads/writes
    /// (`mlflow server --artifacts-destination`, env
    /// `_MLFLOW_SERVER_ARTIFACT_DESTINATION`). Only local-FS / `file:` URIs are
    /// wired in v1 (cloud schemes return NOT_IMPLEMENTED via
    /// `mlflow_artifacts::factory::repo_from_uri`).
    #[arg(long)]
    pub artifacts_destination: Option<String>,
}

/// Error returned when `--static-prefix` (or `MLFLOW_STATIC_PREFIX`) fails
/// validation. Mirrors `_validate_static_prefix`
/// (`mlflow/cli/__init__.py:353-364`).
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum StaticPrefixError {
    #[error("--static-prefix must begin with a '/'.")]
    MissingLeadingSlash,
    #[error("--static-prefix should not end with a '/'.")]
    TrailingSlash,
}

/// Resolved, validated server configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub static_prefix: Option<String>,
    pub backend_store_uri: Option<String>,
    pub default_artifact_root: Option<String>,
    /// Whether the `mlflow-artifacts` proxy surface is enabled
    /// (`--serve-artifacts`). Mirrors `_is_serving_proxied_artifacts()`.
    pub serve_artifacts: bool,
    /// The `--artifacts-destination` base URI for the proxy repo, if configured.
    pub artifacts_destination: Option<String>,
}

impl fmt::Display for ServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

impl ServerConfig {
    /// Builds a `ServerConfig` from parsed CLI args plus the process
    /// environment, applying the same precedence and validation as Python's
    /// click-based CLI: the `--static-prefix` flag takes precedence over
    /// `MLFLOW_STATIC_PREFIX`, and the resolved value (from either source)
    /// is validated.
    pub fn from_cli(cli: Cli) -> Result<Self, StaticPrefixError> {
        let raw_prefix = cli
            .static_prefix
            .or_else(|| std::env::var(MLFLOW_STATIC_PREFIX_ENV_VAR).ok());
        let static_prefix = validate_static_prefix(raw_prefix)?;
        Ok(Self {
            host: cli.host,
            port: cli.port,
            static_prefix,
            backend_store_uri: cli.backend_store_uri,
            default_artifact_root: cli.default_artifact_root,
            serve_artifacts: cli.serve_artifacts,
            artifacts_destination: cli.artifacts_destination,
        })
    }
}

/// Validates a static prefix the same way Python's `_validate_static_prefix`
/// does: `None` passes through untouched (Python only validates when a
/// value is present), otherwise the value must start with `/` and must not
/// end with `/`.
fn validate_static_prefix(value: Option<String>) -> Result<Option<String>, StaticPrefixError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.starts_with('/') {
        return Err(StaticPrefixError::MissingLeadingSlash);
    }
    if value.ends_with('/') {
        return Err(StaticPrefixError::TrailingSlash);
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate `MLFLOW_STATIC_PREFIX`, since env vars
    /// are process-global and `cargo test` runs tests concurrently.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
            Err(StaticPrefixError::MissingLeadingSlash)
        );
    }

    #[test]
    fn trailing_slash_is_rejected() {
        assert_eq!(
            validate_static_prefix(Some("/mlflow/".to_string())),
            Err(StaticPrefixError::TrailingSlash)
        );
    }

    #[test]
    fn root_prefix_is_valid() {
        // "/" starts with '/' AND ends with '/'; Python's checks are two
        // independent `if`s (not elif), so "/" hits the trailing-slash
        // branch and is rejected too.
        assert_eq!(
            validate_static_prefix(Some("/".to_string())),
            Err(StaticPrefixError::TrailingSlash)
        );
    }

    #[test]
    fn cli_flag_takes_precedence_over_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized by ENV_LOCK; no other test reads/writes this
        // var without holding the same lock.
        unsafe {
            std::env::set_var(MLFLOW_STATIC_PREFIX_ENV_VAR, "/env-prefix");
        }
        let cli = Cli {
            host: "127.0.0.1".to_string(),
            port: 5000,
            static_prefix: Some("/cli-prefix".to_string()),
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_destination: None,
        };
        let config = ServerConfig::from_cli(cli).unwrap();
        assert_eq!(config.static_prefix.as_deref(), Some("/cli-prefix"));
        unsafe {
            std::env::remove_var(MLFLOW_STATIC_PREFIX_ENV_VAR);
        }
    }

    #[test]
    fn falls_back_to_env_var_when_cli_flag_absent() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(MLFLOW_STATIC_PREFIX_ENV_VAR, "/env-prefix");
        }
        let cli = Cli {
            host: "127.0.0.1".to_string(),
            port: 5000,
            static_prefix: None,
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_destination: None,
        };
        let config = ServerConfig::from_cli(cli).unwrap();
        assert_eq!(config.static_prefix.as_deref(), Some("/env-prefix"));
        unsafe {
            std::env::remove_var(MLFLOW_STATIC_PREFIX_ENV_VAR);
        }
    }

    #[test]
    fn defaults_to_no_prefix() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var(MLFLOW_STATIC_PREFIX_ENV_VAR);
        }
        let cli = Cli {
            host: "127.0.0.1".to_string(),
            port: 5000,
            static_prefix: None,
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_destination: None,
        };
        let config = ServerConfig::from_cli(cli).unwrap();
        assert_eq!(config.static_prefix, None);
    }
}
