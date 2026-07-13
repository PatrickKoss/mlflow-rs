//! Connection-pool tuning, mapped from MLflow's SQLAlchemy env vars.
//!
//! MLflow drives SQLAlchemy's `create_engine` pool from four env vars
//! (`mlflow/store/db/utils.py:349`):
//!
//! | MLflow env var | SQLAlchemy meaning | sqlx equivalent |
//! |---|---|---|
//! | `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE` | steady-state pool size | part of `max_connections` |
//! | `MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW` | extra connections above pool_size | added to `max_connections` |
//! | `MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE` | recycle a connection after N seconds | `max_lifetime` |
//! | `MLFLOW_SQLALCHEMYSTORE_ECHO` | log all SQL | (informational; sqlx logs via `tracing`) |
//!
//! **Mapping rationale.** SQLAlchemy's `QueuePool` keeps `pool_size` persistent
//! connections and allows up to `max_overflow` additional transient ones, so the
//! hard ceiling is `pool_size + max_overflow`. sqlx has no separate overflow
//! concept — it has a single `max_connections` ceiling and a `min_connections`
//! floor. We therefore map:
//!
//! * `max_connections = pool_size + max_overflow` (the SQLAlchemy ceiling),
//! * `min_connections = pool_size` (the persistent set),
//! * `max_lifetime = pool_recycle` seconds.
//!
//! When neither pool var is set, MLflow lets SQLAlchemy use its defaults
//! (`pool_size=5`, `max_overflow=10`); we mirror that with `max_connections=15`,
//! `min_connections=0` (sqlx lazily opens connections, which is the more
//! conservative default for a low-idle-RSS server, plan §5.5).

use std::time::Duration;

/// SQLAlchemy default `pool_size` (used when the env var is unset/zero).
const DEFAULT_POOL_SIZE: u32 = 5;
/// SQLAlchemy default `max_overflow` (used when the env var is unset/zero).
const DEFAULT_MAX_OVERFLOW: u32 = 10;

/// Resolved pool tuning, ready to apply to a sqlx pool builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolConfig {
    /// Upper bound on open connections (`pool_size + max_overflow`).
    pub max_connections: u32,
    /// Number of persistent connections kept warm (`pool_size`).
    pub min_connections: u32,
    /// Connection recycle interval (`pool_recycle` seconds), if configured.
    pub max_lifetime: Option<Duration>,
    /// Whether to echo SQL (`MLFLOW_SQLALCHEMYSTORE_ECHO`).
    pub echo: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_POOL_SIZE + DEFAULT_MAX_OVERFLOW,
            min_connections: 0,
            max_lifetime: None,
            echo: false,
        }
    }
}

impl PoolConfig {
    /// Resolve pool config from the current process environment.
    pub fn from_env() -> Self {
        Self::from_values(
            env_u32("MLFLOW_SQLALCHEMYSTORE_POOL_SIZE"),
            env_u32("MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW"),
            env_u32("MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE"),
            env_bool("MLFLOW_SQLALCHEMYSTORE_ECHO"),
        )
    }

    /// Build a config from raw values (factored out for testing).
    ///
    /// `pool_size` / `max_overflow` of `None` (or `0`) fall back to the
    /// SQLAlchemy defaults, matching MLflow's "only send if injected" behavior.
    pub fn from_values(
        pool_size: Option<u32>,
        max_overflow: Option<u32>,
        pool_recycle: Option<u32>,
        echo: bool,
    ) -> Self {
        let size = pool_size.filter(|v| *v > 0).unwrap_or(DEFAULT_POOL_SIZE);
        let overflow = max_overflow
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_MAX_OVERFLOW);
        let min = pool_size.filter(|v| *v > 0).unwrap_or(0);
        Self {
            max_connections: size.saturating_add(overflow),
            min_connections: min,
            max_lifetime: pool_recycle
                .filter(|v| *v > 0)
                .map(|secs| Duration::from_secs(secs as u64)),
            echo,
        }
    }
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn env_bool(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_unset() {
        let cfg = PoolConfig::from_values(None, None, None, false);
        assert_eq!(cfg.max_connections, 15);
        assert_eq!(cfg.min_connections, 0);
        assert_eq!(cfg.max_lifetime, None);
        assert!(!cfg.echo);
    }

    #[test]
    fn maps_pool_size_and_overflow_to_max_connections() {
        let cfg = PoolConfig::from_values(Some(8), Some(4), None, false);
        assert_eq!(cfg.max_connections, 12);
        assert_eq!(cfg.min_connections, 8);
    }

    #[test]
    fn pool_size_only_uses_default_overflow() {
        let cfg = PoolConfig::from_values(Some(3), None, None, false);
        assert_eq!(cfg.max_connections, 3 + DEFAULT_MAX_OVERFLOW);
        assert_eq!(cfg.min_connections, 3);
    }

    #[test]
    fn zero_is_treated_as_unset() {
        let cfg = PoolConfig::from_values(Some(0), Some(0), Some(0), false);
        assert_eq!(cfg, PoolConfig::from_values(None, None, None, false));
    }

    #[test]
    fn pool_recycle_maps_to_max_lifetime() {
        let cfg = PoolConfig::from_values(Some(5), Some(10), Some(1800), true);
        assert_eq!(cfg.max_lifetime, Some(Duration::from_secs(1800)));
        assert!(cfg.echo);
    }
}
