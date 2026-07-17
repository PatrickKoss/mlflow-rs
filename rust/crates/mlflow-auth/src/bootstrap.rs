//! Admin-user bootstrap, mirroring `mlflow/server/auth/__init__.py:3670-3700`.
//!
//! On startup the Python server calls `create_admin_user(admin_username,
//! admin_password)` then `_warn_if_default_admin_password(admin_password)`
//! (`__init__.py:4652-4653`). We reproduce both, including the exact log
//! wording, so the Rust server's startup is observably identical.
//!
//! Config defaults (from `mlflow/server/auth/basic_auth.ini`): admin username
//! `admin`, admin password `password1234`.

use mlflow_error::{ErrorCode, MlflowError};

use crate::store::AuthStore;

/// Default admin username shipped in `basic_auth.ini` (`admin_username = admin`).
pub const DEFAULT_ADMIN_USERNAME: &str = "admin";

/// Default admin password shipped in `basic_auth.ini`
/// (`_DEFAULT_ADMIN_PASSWORD`, `__init__.py:3689`; `admin_password = password1234`).
pub const DEFAULT_ADMIN_PASSWORD: &str = "password1234";

/// The route the info/warning messages point users to for changing the
/// password (`UPDATE_USER_PASSWORD`, `routes.py:27`:
/// `/api/2.0/mlflow/users/update-password`).
const UPDATE_USER_PASSWORD: &str = "/api/2.0/mlflow/users/update-password";

/// `create_admin_user(username, password)` (`__init__.py:3670-3685`): create the
/// admin user with `is_admin=True` iff it does not already exist. A concurrent
/// duplicate insert (multi-worker race) is swallowed, matching Python's
/// `IntegrityError` handling. On success, logs the same info message.
pub async fn create_admin_user(
    store: &AuthStore,
    username: &str,
    password: &str,
) -> Result<(), MlflowError> {
    if store.has_user(username).await? {
        return Ok(());
    }
    match store.create_user(username, password, true).await {
        Ok(_) => {
            tracing::info!(
                "Created admin user '{username}'. It is recommended that you set a new password \
                 as soon as possible on {UPDATE_USER_PASSWORD}."
            );
            Ok(())
        }
        // Multi-worker race: another worker won the insert. Python returns
        // quietly on the IntegrityError-caused RESOURCE_ALREADY_EXISTS.
        Err(e) if e.error_code == ErrorCode::ResourceAlreadyExists => Ok(()),
        Err(e) => Err(e),
    }
}

/// `_warn_if_default_admin_password(password)` (`__init__.py:3692-3700`). Logs
/// the exact warning when the configured admin password is the shipped default.
pub fn warn_if_default_admin_password(password: &str) {
    if password == DEFAULT_ADMIN_PASSWORD {
        tracing::warn!(
            "The MLflow basic auth admin account is using the default password shipped in \
             basic_auth.ini. Change it before exposing this server beyond localhost. To \
             override, set the MLFLOW_AUTH_CONFIG_PATH environment variable to point to a custom \
             basic_auth.ini with a non-default admin_password, or update the password via \
             {UPDATE_USER_PASSWORD} after startup."
        );
    }
}
