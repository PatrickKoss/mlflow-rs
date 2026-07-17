//! Config-parity tests (plan T9.8 **VER**): the Rust [`AuthConfig`] parser must
//! read the *real* packaged `basic_auth.ini` and land the exact same field
//! defaults the Python `read_auth_config` (`config.py`) produces, so the same
//! file drives both servers.

use std::path::{Path, PathBuf};

use mlflow_auth::config::{AuthConfig, DEFAULT_AUTHORIZATION_FUNCTION};

/// The real shipped `mlflow/server/auth/basic_auth.ini`, relative to this crate.
fn packaged_ini_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../mlflow/server/auth/basic_auth.ini")
}

#[test]
fn packaged_basic_auth_ini_defaults_match_python() {
    let config =
        AuthConfig::read_from_path(packaged_ini_path()).expect("parse packaged basic_auth.ini");

    // Every field + default from `mlflow/server/auth/config.py`'s
    // `read_auth_config` against the shipped `basic_auth.ini`.
    assert_eq!(config.default_permission, "READ");
    assert_eq!(config.database_uri, "sqlite:///basic_auth.db");
    assert_eq!(config.admin_username, "admin");
    assert_eq!(config.admin_password, "password1234");
    // authorization_function is set explicitly in the shipped file (to the
    // default value), so it round-trips to the same string config.py uses.
    assert_eq!(
        config.authorization_function,
        DEFAULT_AUTHORIZATION_FUNCTION
    );
    // grant_default_workspace_access = false (shipped) → getboolean fallback False.
    assert!(!config.grant_default_workspace_access);
    // The cache fields are commented out in the shipped file → configparser
    // fallbacks apply verbatim.
    assert_eq!(config.workspace_cache_max_size, 10_000);
    assert_eq!(config.workspace_cache_ttl_seconds, 3_600);
    assert_eq!(config.auth_cache_max_size, 10_000);
    // auth_cache_ttl_seconds = 0 → credential cache OFF by default.
    assert_eq!(config.auth_cache_ttl_seconds, 0);
    // read_database_uri absent → None.
    assert_eq!(config.read_database_uri, None);
}

#[test]
fn default_impl_equals_packaged_file() {
    // `AuthConfig::default()` is the defaults contract; it must equal parsing the
    // shipped file, so callers that skip the ini get identical behaviour.
    let from_file = AuthConfig::read_from_path(packaged_ini_path()).expect("parse packaged ini");
    assert_eq!(from_file, AuthConfig::default());
}

#[test]
fn env_var_selects_config_file() {
    // A temp ini with overridden values; `AuthConfig::read` must resolve it via
    // MLFLOW_AUTH_CONFIG_PATH. (Serialized-ish: this is the only test touching
    // the env var, and it restores it.)
    let dir = std::env::temp_dir().join(format!("mlflow_auth_cfg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("custom.ini");
    std::fs::write(
        &path,
        "[mlflow]\n\
         default_permission = MANAGE\n\
         database_uri = sqlite:///custom.db\n\
         read_database_uri = sqlite:///replica.db\n\
         admin_username = root\n\
         admin_password = a-very-long-password\n\
         grant_default_workspace_access = true\n\
         workspace_cache_max_size = 5\n\
         workspace_cache_ttl_seconds = 7\n\
         auth_cache_max_size = 9\n\
         auth_cache_ttl_seconds = 60\n",
    )
    .unwrap();

    let prev = std::env::var_os("MLFLOW_AUTH_CONFIG_PATH");
    std::env::set_var("MLFLOW_AUTH_CONFIG_PATH", &path);
    let config = AuthConfig::read().expect("read via env var");
    match prev {
        Some(v) => std::env::set_var("MLFLOW_AUTH_CONFIG_PATH", v),
        None => std::env::remove_var("MLFLOW_AUTH_CONFIG_PATH"),
    }

    assert_eq!(config.default_permission, "MANAGE");
    assert_eq!(config.database_uri, "sqlite:///custom.db");
    assert_eq!(
        config.read_database_uri.as_deref(),
        Some("sqlite:///replica.db")
    );
    assert_eq!(config.admin_username, "root");
    assert_eq!(config.admin_password, "a-very-long-password");
    assert!(config.grant_default_workspace_access);
    assert_eq!(config.workspace_cache_max_size, 5);
    assert_eq!(config.workspace_cache_ttl_seconds, 7);
    assert_eq!(config.auth_cache_max_size, 9);
    assert_eq!(config.auth_cache_ttl_seconds, 60);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_config_file_is_a_loud_error() {
    let err = AuthConfig::read_from_path("/nonexistent/definitely/not/here.ini")
        .expect_err("missing file must error");
    assert!(
        err.to_string().contains("Could not read auth config file"),
        "unexpected error: {err}"
    );
}

#[test]
fn unsupported_authorization_function_errors_loudly() {
    let ini = "[mlflow]\n\
        default_permission = READ\n\
        database_uri = sqlite:///basic_auth.db\n\
        admin_username = admin\n\
        admin_password = password1234\n\
        authorization_function = my.custom:auth_backend\n";
    let err = AuthConfig::from_ini_str(ini).expect_err("custom auth fn must error");
    let msg = err.to_string();
    assert!(
        msg.contains("Unsupported authorization_function 'my.custom:auth_backend'"),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("only supports the built-in HTTP Basic backend"),
        "unexpected error: {msg}"
    );
}

#[test]
fn default_authorization_function_is_accepted() {
    // The shipped default value must NOT trip the unsupported-function check.
    let ini = format!(
        "[mlflow]\n\
         default_permission = READ\n\
         database_uri = sqlite:///basic_auth.db\n\
         admin_username = admin\n\
         admin_password = password1234\n\
         authorization_function = {DEFAULT_AUTHORIZATION_FUNCTION}\n"
    );
    AuthConfig::from_ini_str(&ini).expect("default auth fn accepted");
}

#[test]
fn bad_permission_string_errors() {
    let ini = "[mlflow]\n\
        default_permission = SUPERUSER\n\
        database_uri = sqlite:///basic_auth.db\n\
        admin_username = admin\n\
        admin_password = password1234\n";
    let err = AuthConfig::from_ini_str(ini).expect_err("bad permission must error");
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid default_permission 'SUPERUSER'"),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("Valid permissions are:"),
        "unexpected error: {msg}"
    );
}

#[test]
fn bad_integer_and_boolean_values_error() {
    let bad_int = "[mlflow]\n\
        default_permission = READ\n\
        database_uri = sqlite:///basic_auth.db\n\
        admin_username = admin\n\
        admin_password = password1234\n\
        auth_cache_max_size = not-a-number\n";
    assert!(AuthConfig::from_ini_str(bad_int)
        .unwrap_err()
        .to_string()
        .contains("Invalid integer value 'not-a-number'"));

    let bad_bool = "[mlflow]\n\
        default_permission = READ\n\
        database_uri = sqlite:///basic_auth.db\n\
        admin_username = admin\n\
        admin_password = password1234\n\
        grant_default_workspace_access = maybe\n";
    assert!(AuthConfig::from_ini_str(bad_bool)
        .unwrap_err()
        .to_string()
        .contains("Invalid boolean value 'maybe'"));
}

#[test]
fn missing_required_key_errors() {
    // Drop `database_uri`: Python `config["mlflow"]["database_uri"]` KeyErrors;
    // we surface a clear message instead.
    let ini = "[mlflow]\n\
        default_permission = READ\n\
        admin_username = admin\n\
        admin_password = password1234\n";
    let err = AuthConfig::from_ini_str(ini).expect_err("missing key must error");
    assert!(
        err.to_string()
            .contains("missing required key 'database_uri'"),
        "unexpected error: {err}"
    );
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let ini = "; a comment\n\
        [mlflow]\n\
        # another comment\n\
        \n\
        default_permission = EDIT\n\
        database_uri = sqlite:///basic_auth.db\n\
        admin_username = admin\n\
        admin_password = password1234\n";
    let config = AuthConfig::from_ini_str(ini).expect("parse with comments");
    assert_eq!(config.default_permission, "EDIT");
}
