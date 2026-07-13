//! SQLAlchemy-style database URI parsing.
//!
//! MLflow accepts SQLAlchemy connection URIs whose scheme may carry a `+driver`
//! suffix, e.g. `postgresql+psycopg2://...`, `mysql+pymysql://...`,
//! `sqlite:///path`. The Rust server uses `sqlx`, which does not understand the
//! `+driver` forms, so we strip the driver, map the base scheme to a
//! [`Dialect`], and hand `sqlx` a scheme it accepts.
//!
//! This mirrors `extract_db_type_from_uri` (`mlflow/utils/uri.py:256`): the
//! scheme may contain at most one `+`; anything else is rejected. MSSQL is a
//! recognized-but-unsupported scheme here (deferred, plan T0.3) and yields a
//! clear "not yet supported by the Rust server" error rather than a generic
//! parse failure.

use crate::dialect::Dialect;
use std::fmt;

/// Error produced while parsing a database URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UriError {
    /// The URI had no `scheme://` prefix.
    MissingScheme { uri: String },
    /// The scheme contained more than one `+` (invalid SQLAlchemy driver form).
    InvalidScheme { uri: String },
    /// A recognized SQLAlchemy scheme that the Rust server does not support yet.
    UnsupportedDialect { db_type: String, uri: String },
    /// An entirely unknown scheme.
    UnknownScheme { scheme: String, uri: String },
}

impl fmt::Display for UriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UriError::MissingScheme { uri } => {
                write!(f, "Invalid database URI: '{uri}'. Expected a scheme like 'sqlite:///', 'postgresql://', or 'mysql://'.")
            }
            UriError::InvalidScheme { uri } => {
                write!(f, "Invalid database URI: '{uri}'. The scheme may contain at most one '+driver' suffix.")
            }
            UriError::UnsupportedDialect { db_type, uri } => {
                write!(f, "Database dialect '{db_type}' (from URI '{uri}') is not yet supported by the Rust server.")
            }
            UriError::UnknownScheme { scheme, uri } => {
                write!(f, "Unsupported database scheme '{scheme}' in URI '{uri}'. Supported schemes: sqlite, postgresql, mysql.")
            }
        }
    }
}

impl std::error::Error for UriError {}

/// A database URI parsed into a [`Dialect`] plus a `sqlx`-compatible connection
/// string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedUri {
    /// The resolved dialect.
    pub dialect: Dialect,
    /// A connection string that `sqlx` accepts (driver suffix stripped, scheme
    /// normalized). Query parameters and credentials are preserved.
    pub sqlx_url: String,
    /// The original URI as provided.
    pub original: String,
}

/// Parse a SQLAlchemy-style database URI.
///
/// Handles `+driver` suffixes (`postgresql+psycopg2`, `mysql+pymysql`,
/// `mssql+pyodbc`, ...), preserves query parameters (e.g. `?charset=utf8mb4`)
/// and userinfo, and normalizes the scheme for `sqlx`:
///
/// * `sqlite` -> `sqlite` (unchanged; `sqlite:///path` and `sqlite://` forms).
/// * `postgresql` / `postgres` -> `postgres` (sqlx's canonical scheme).
/// * `mysql` / `mariadb` -> `mysql`.
/// * `mssql` -> [`UriError::UnsupportedDialect`] ("not yet supported").
pub fn parse(uri: &str) -> Result<ParsedUri, UriError> {
    let Some((scheme, rest)) = split_scheme(uri) else {
        return Err(UriError::MissingScheme {
            uri: uri.to_string(),
        });
    };

    let db_type = match scheme.split('+').collect::<Vec<_>>().as_slice() {
        [base] => (*base).to_string(),
        [base, _driver] => (*base).to_string(),
        _ => {
            return Err(UriError::InvalidScheme {
                uri: uri.to_string(),
            })
        }
    };

    let (dialect, sqlx_scheme) = match db_type.to_ascii_lowercase().as_str() {
        "sqlite" => (Dialect::Sqlite, "sqlite"),
        "postgresql" | "postgres" => (Dialect::Postgres, "postgres"),
        "mysql" | "mariadb" => (Dialect::MySql, "mysql"),
        "mssql" => {
            return Err(UriError::UnsupportedDialect {
                db_type,
                uri: uri.to_string(),
            })
        }
        other => {
            return Err(UriError::UnknownScheme {
                scheme: other.to_string(),
                uri: uri.to_string(),
            })
        }
    };

    Ok(ParsedUri {
        dialect,
        sqlx_url: format!("{sqlx_scheme}://{rest}"),
        original: uri.to_string(),
    })
}

/// Split a URI into `(scheme, rest_after_"://")`.
///
/// Returns `None` when there is no `://` separator.
fn split_scheme(uri: &str) -> Option<(&str, &str)> {
    let idx = uri.find("://")?;
    let scheme = &uri[..idx];
    if scheme.is_empty() {
        return None;
    }
    Some((scheme, &uri[idx + 3..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_supported_forms() {
        let cases: &[(&str, Dialect, &str)] = &[
            (
                "sqlite:///abs/mlflow.db",
                Dialect::Sqlite,
                "sqlite:///abs/mlflow.db",
            ),
            ("sqlite:///./rel.db", Dialect::Sqlite, "sqlite:///./rel.db"),
            (
                "postgresql://u:p@host:5432/db",
                Dialect::Postgres,
                "postgres://u:p@host:5432/db",
            ),
            (
                "postgresql+psycopg2://u:p@host/db",
                Dialect::Postgres,
                "postgres://u:p@host/db",
            ),
            (
                "postgres://u@host/db",
                Dialect::Postgres,
                "postgres://u@host/db",
            ),
            (
                "mysql+pymysql://u:p@host:3306/db",
                Dialect::MySql,
                "mysql://u:p@host:3306/db",
            ),
            (
                "mysql://u:p@host/db?charset=utf8mb4",
                Dialect::MySql,
                "mysql://u:p@host/db?charset=utf8mb4",
            ),
            (
                "mariadb+mariadbconnector://u@host/db",
                Dialect::MySql,
                "mysql://u@host/db",
            ),
            (
                "postgresql+psycopg2://u:p@host/db?sslmode=require&connect_timeout=10",
                Dialect::Postgres,
                "postgres://u:p@host/db?sslmode=require&connect_timeout=10",
            ),
        ];
        for (input, dialect, sqlx_url) in cases {
            let parsed = parse(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!(parsed.dialect, *dialect, "dialect for {input}");
            assert_eq!(parsed.sqlx_url, *sqlx_url, "sqlx_url for {input}");
            assert_eq!(parsed.original, *input);
        }
    }

    #[test]
    fn rejects_mssql_with_not_supported_message() {
        let err = parse("mssql+pyodbc://u:p@host/db").unwrap_err();
        assert_eq!(
            err,
            UriError::UnsupportedDialect {
                db_type: "mssql".to_string(),
                uri: "mssql+pyodbc://u:p@host/db".to_string(),
            }
        );
        assert!(err
            .to_string()
            .contains("not yet supported by the Rust server"));
    }

    #[test]
    fn rejects_double_plus_scheme() {
        assert_eq!(
            parse("postgresql+psycopg2+extra://host/db").unwrap_err(),
            UriError::InvalidScheme {
                uri: "postgresql+psycopg2+extra://host/db".to_string(),
            }
        );
    }

    #[test]
    fn rejects_missing_scheme() {
        assert!(matches!(
            parse("/just/a/path").unwrap_err(),
            UriError::MissingScheme { .. }
        ));
        assert!(matches!(
            parse("://host/db").unwrap_err(),
            UriError::MissingScheme { .. }
        ));
    }

    #[test]
    fn rejects_unknown_scheme() {
        assert!(matches!(
            parse("oracle://host/db").unwrap_err(),
            UriError::UnknownScheme { .. }
        ));
    }
}
