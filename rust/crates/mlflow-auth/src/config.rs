//! Auth configuration parsing (plan T9.8), a byte-faithful port of
//! `mlflow/server/auth/config.py` and the shipped `basic_auth.ini`.
//!
//! [`AuthConfig`] mirrors Python's `AuthConfig` `NamedTuple` field-for-field.
//! [`read_auth_config`] resolves the config path from the
//! `MLFLOW_AUTH_CONFIG_PATH` env var (falling back to the packaged
//! `basic_auth.ini`, whose shipped values *are* the defaults contract) and
//! parses it with the minimal INI reader in this module.
//!
//! ## Parity notes
//!
//! * The same config file drives both servers; the defaults below match
//!   `config.py`'s `configparser` fallbacks exactly (see the field docs).
//! * Python honours **no** env-var overrides for the individual fields — only
//!   `MLFLOW_AUTH_CONFIG_PATH` selects the file. The pre-T9.8 Rust seam invented
//!   `MLFLOW_AUTH_DATABASE_URI` / `MLFLOW_AUTH_READ_DATABASE_URI` /
//!   `MLFLOW_AUTH_DEFAULT_PERMISSION`; those are dropped here so the two servers
//!   read the identical config surface.
//! * `authorization_function` is Python's pluggable auth backend hook. The Rust
//!   server only implements the shipped default
//!   (`mlflow.server.auth:authenticate_request_basic_auth`); any other value is
//!   a *loud* startup error rather than a silent no-op (see
//!   [`AuthConfig::validate`]).
//!
//! ## INI reader
//!
//! Python uses `configparser`. The shipped file is a single `[mlflow]` section
//! of `key = value` lines with `#`/`;` comments; we implement exactly that shape
//! (no interpolation, no multi-section merging beyond what the file needs). This
//! avoids a heavyweight INI dependency while reading every field Python reads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlflow_error::MlflowError;

use crate::permissions::ALL_PERMISSIONS;

/// `MLFLOW_AUTH_CONFIG_PATH` (`environment_variables.py:471`): selects the ini
/// file. Unset → the packaged `basic_auth.ini`.
pub const MLFLOW_AUTH_CONFIG_PATH_ENV: &str = "MLFLOW_AUTH_CONFIG_PATH";

/// `DEFAULT_AUTHORIZATION_FUNCTION` (`config.py:7`): the only auth backend the
/// Rust server implements (HTTP Basic against the auth DB).
pub const DEFAULT_AUTHORIZATION_FUNCTION: &str =
    "mlflow.server.auth:authenticate_request_basic_auth";

// ---------------------------------------------------------------------------
// Shipped defaults (basic_auth.ini) — the defaults *contract*.
// ---------------------------------------------------------------------------

/// `default_permission = READ` (basic_auth.ini).
pub const DEFAULT_PERMISSION: &str = "READ";
/// `database_uri = sqlite:///basic_auth.db` (basic_auth.ini).
pub const DEFAULT_DATABASE_URI: &str = "sqlite:///basic_auth.db";
/// `admin_username = admin` (basic_auth.ini).
pub const DEFAULT_ADMIN_USERNAME: &str = "admin";
/// `admin_password = password1234` (basic_auth.ini).
pub const DEFAULT_ADMIN_PASSWORD: &str = "password1234";
/// `grant_default_workspace_access` fallback (`config.py:44`, `fallback=False`).
pub const DEFAULT_GRANT_DEFAULT_WORKSPACE_ACCESS: bool = false;
/// `workspace_cache_max_size` fallback (`config.py:47`, `fallback=10000`).
pub const DEFAULT_WORKSPACE_CACHE_MAX_SIZE: u64 = 10_000;
/// `workspace_cache_ttl_seconds` fallback (`config.py:50`, `fallback=3600`).
pub const DEFAULT_WORKSPACE_CACHE_TTL_SECONDS: u64 = 3_600;
/// `auth_cache_max_size` fallback (`config.py:53`, `fallback=10000`).
pub const DEFAULT_AUTH_CACHE_MAX_SIZE: u64 = 10_000;
/// `auth_cache_ttl_seconds` fallback (`config.py:57`, `fallback=0` → cache off).
pub const DEFAULT_AUTH_CACHE_TTL_SECONDS: u64 = 0;

/// `AuthConfig` (`config.py:11-22`): every field the auth app reads from the
/// ini file, with the shipped defaults as the parity contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthConfig {
    /// `default_permission` — the permission floor for authenticated users
    /// without a matching grant. `config["mlflow"]["default_permission"]`.
    pub default_permission: String,
    /// `database_uri` — the auth DB SQLAlchemy URI.
    /// `config["mlflow"]["database_uri"]`.
    pub database_uri: String,
    /// `admin_username` — bootstrap admin username.
    /// `config["mlflow"]["admin_username"]`.
    pub admin_username: String,
    /// `admin_password` — bootstrap admin password.
    /// `config["mlflow"]["admin_password"]`.
    pub admin_password: String,
    /// `authorization_function` — pluggable backend id (`config.py:36`,
    /// `fallback=DEFAULT_AUTHORIZATION_FUNCTION`). Rust supports only the
    /// default; [`AuthConfig::validate`] rejects any other value.
    pub authorization_function: String,
    /// `grant_default_workspace_access` — when true, users inherit
    /// `default_permission` for the reserved `default` workspace
    /// (`config.py:43`, `getboolean`, `fallback=False`).
    pub grant_default_workspace_access: bool,
    /// `workspace_cache_max_size` — resource→workspace TTL-cache capacity
    /// (`config.py:46`, `getint`, `fallback=10000`).
    pub workspace_cache_max_size: u64,
    /// `workspace_cache_ttl_seconds` — resource→workspace TTL (`config.py:49`,
    /// `getint`, `fallback=3600`).
    pub workspace_cache_ttl_seconds: u64,
    /// `auth_cache_max_size` — credential-cache capacity (`config.py:52`,
    /// `getint`, `fallback=10000`).
    pub auth_cache_max_size: u64,
    /// `auth_cache_ttl_seconds` — credential-cache TTL; `0` disables the cache
    /// (`config.py:55`, `getint`, `fallback=0`).
    pub auth_cache_ttl_seconds: u64,
    /// `read_database_uri` — optional read-replica URI
    /// (`config.py:58`, `.get("read_database_uri", None)`).
    pub read_database_uri: Option<String>,
}

impl Default for AuthConfig {
    /// The shipped `basic_auth.ini` values — the defaults contract used when a
    /// field is absent, and by callers (tests) that want the packaged config.
    fn default() -> Self {
        Self {
            default_permission: DEFAULT_PERMISSION.to_string(),
            database_uri: DEFAULT_DATABASE_URI.to_string(),
            admin_username: DEFAULT_ADMIN_USERNAME.to_string(),
            admin_password: DEFAULT_ADMIN_PASSWORD.to_string(),
            authorization_function: DEFAULT_AUTHORIZATION_FUNCTION.to_string(),
            grant_default_workspace_access: DEFAULT_GRANT_DEFAULT_WORKSPACE_ACCESS,
            workspace_cache_max_size: DEFAULT_WORKSPACE_CACHE_MAX_SIZE,
            workspace_cache_ttl_seconds: DEFAULT_WORKSPACE_CACHE_TTL_SECONDS,
            auth_cache_max_size: DEFAULT_AUTH_CACHE_MAX_SIZE,
            auth_cache_ttl_seconds: DEFAULT_AUTH_CACHE_TTL_SECONDS,
            read_database_uri: None,
        }
    }
}

impl AuthConfig {
    /// Resolve the config path (`MLFLOW_AUTH_CONFIG_PATH` or the packaged
    /// `basic_auth.ini`) and parse it. Mirrors `read_auth_config` (`config.py:26`).
    pub fn read() -> Result<Self, MlflowError> {
        match std::env::var_os(MLFLOW_AUTH_CONFIG_PATH_ENV) {
            Some(p) if !p.is_empty() => Self::read_from_path(PathBuf::from(p)),
            _ => Ok(Self::default()),
        }
    }

    /// Parse an explicit ini path. A missing/unreadable file is a loud error
    /// (Python's `configparser.read` silently ignores a missing file and then
    /// `KeyError`s on `config["mlflow"]`; we surface the clearer "not found").
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self, MlflowError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|e| {
            MlflowError::invalid_parameter_value(format!(
                "Could not read auth config file '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_ini_str(&contents)
    }

    /// Parse the ini text and fill defaults for absent fields, mirroring
    /// `read_auth_config`'s per-field `configparser` reads and fallbacks.
    pub fn from_ini_str(contents: &str) -> Result<Self, MlflowError> {
        let section = parse_mlflow_section(contents)?;
        let get = |key: &str| section.get(key).map(String::as_str);

        // The four required-ish keys use `config["mlflow"][key]` in Python,
        // which raises `KeyError` when absent. The packaged file always ships
        // them; a custom file that drops one is a loud error here.
        let required = |key: &str| -> Result<String, MlflowError> {
            get(key).map(str::to_string).ok_or_else(|| {
                MlflowError::invalid_parameter_value(format!(
                    "Auth config is missing required key '{key}' in the [mlflow] section."
                ))
            })
        };

        let config = Self {
            default_permission: required("default_permission")?,
            database_uri: required("database_uri")?,
            admin_username: required("admin_username")?,
            admin_password: required("admin_password")?,
            authorization_function: get("authorization_function")
                .unwrap_or(DEFAULT_AUTHORIZATION_FUNCTION)
                .to_string(),
            grant_default_workspace_access: get_bool(
                &section,
                "grant_default_workspace_access",
                DEFAULT_GRANT_DEFAULT_WORKSPACE_ACCESS,
            )?,
            workspace_cache_max_size: get_int(
                &section,
                "workspace_cache_max_size",
                DEFAULT_WORKSPACE_CACHE_MAX_SIZE,
            )?,
            workspace_cache_ttl_seconds: get_int(
                &section,
                "workspace_cache_ttl_seconds",
                DEFAULT_WORKSPACE_CACHE_TTL_SECONDS,
            )?,
            auth_cache_max_size: get_int(
                &section,
                "auth_cache_max_size",
                DEFAULT_AUTH_CACHE_MAX_SIZE,
            )?,
            auth_cache_ttl_seconds: get_int(
                &section,
                "auth_cache_ttl_seconds",
                DEFAULT_AUTH_CACHE_TTL_SECONDS,
            )?,
            read_database_uri: get("read_database_uri").map(str::to_string),
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate fields the Rust server can only partly honour.
    ///
    /// * `default_permission` must be a known permission name (Python would
    ///   `KeyError` later in `get_permission`; we fail at parse time instead).
    /// * `authorization_function` must be the shipped default — Rust does not
    ///   support pluggable auth backends, so a non-default value is a loud error
    ///   rather than a silently-ignored setting.
    pub fn validate(&self) -> Result<(), MlflowError> {
        if !ALL_PERMISSIONS
            .iter()
            .any(|p| p.name == self.default_permission)
        {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid default_permission '{}' in auth config. Valid permissions are: {}",
                self.default_permission,
                ALL_PERMISSIONS
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", "),
            )));
        }
        if self.authorization_function != DEFAULT_AUTHORIZATION_FUNCTION {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Unsupported authorization_function '{}' in auth config. The Rust MLflow \
                 server only supports the built-in HTTP Basic backend \
                 ('{DEFAULT_AUTHORIZATION_FUNCTION}'); pluggable authorization functions are \
                 not implemented. Remove the setting or run the Python server.",
                self.authorization_function
            )));
        }
        Ok(())
    }
}

/// Parse the single `[mlflow]` section into a key→value map. The reader accepts
/// the shape of the shipped file: `#`/`;` comment lines, blank lines, and
/// `key = value` / `key: value` pairs. Keys are lower-cased (configparser
/// default). Values keep their inline text verbatim (no interpolation).
fn parse_mlflow_section(contents: &str) -> Result<HashMap<String, String>, MlflowError> {
    let mut current_section: Option<String> = None;
    let mut mlflow: HashMap<String, String> = HashMap::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            current_section = Some(inner.trim().to_ascii_lowercase());
            continue;
        }
        // configparser accepts `=` or `:` as the key/value delimiter.
        let sep = line.find('=').or_else(|| line.find(':'));
        let Some(sep) = sep else {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Malformed auth config line (expected 'key = value'): {raw:?}"
            )));
        };
        if current_section.as_deref() == Some("mlflow") {
            let key = line[..sep].trim().to_ascii_lowercase();
            let value = line[sep + 1..].trim().to_string();
            mlflow.insert(key, value);
        }
    }
    if current_section.is_none() && mlflow.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Auth config has no [mlflow] section.".to_string(),
        ));
    }
    Ok(mlflow)
}

/// `configparser.getboolean` parity: accepts `1/yes/true/on` and
/// `0/no/false/off` (case-insensitive), else a loud error.
fn get_bool(
    section: &HashMap<String, String>,
    key: &str,
    fallback: bool,
) -> Result<bool, MlflowError> {
    match section.get(key) {
        None => Ok(fallback),
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "yes" | "true" | "on" => Ok(true),
            "0" | "no" | "false" | "off" => Ok(false),
            other => Err(MlflowError::invalid_parameter_value(format!(
                "Invalid boolean value '{other}' for auth config key '{key}'."
            ))),
        },
    }
}

/// `configparser.getint` parity: parse as a base-10 integer, else a loud error.
fn get_int(
    section: &HashMap<String, String>,
    key: &str,
    fallback: u64,
) -> Result<u64, MlflowError> {
    match section.get(key) {
        None => Ok(fallback),
        Some(v) => v.trim().parse::<u64>().map_err(|_| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid integer value '{v}' for auth config key '{key}'."
            ))
        }),
    }
}
