//! Per-dialect SQL generation for the tracking store.
//!
//! MLflow supports SQLite, PostgreSQL, MySQL, and (deferred here) MSSQL. The
//! observable SQL differs per backend in a handful of well-defined ways that
//! this module encapsulates so that the store layer (T2.4+) can stay
//! dialect-agnostic:
//!
//! * **Upserts** — SQLite/Postgres use `INSERT ... ON CONFLICT (...) DO UPDATE`
//!   while MySQL uses `INSERT ... ON DUPLICATE KEY UPDATE`
//!   (`sqlalchemy_store.py:9806` `_upsert_batch`).
//! * **Case-sensitive `LIKE`** — SQLite honors `PRAGMA case_sensitive_like`
//!   (set on every connection, see [`crate::db`]); Postgres `LIKE` is already
//!   case-sensitive; MySQL `LIKE` is collation-dependent and case-insensitive by
//!   default, so MLflow wraps case-sensitive comparisons with `BINARY`
//!   (`search_utils.py:284-306`). [`Dialect::case_sensitive_like`] mirrors that.
//! * **Identifier quoting** — `"ident"` for SQLite/Postgres, `` `ident` `` for
//!   MySQL.
//! * **Capabilities** — e.g. `RETURNING` support, which the store can use to
//!   avoid a follow-up `SELECT` on Postgres/SQLite.
//!
//! The abstraction is intentionally minimal-but-extensible: [`Dialect`] is an
//! enum (cheap to `Copy`, easy to `match`), and MSSQL can be added as a variant
//! later (T0.3, `tiberius` fast-follow) without touching call sites that only
//! use the methods here.

/// A supported SQL backend dialect.
///
/// MSSQL is intentionally omitted: it is deferred to a `tiberius`-based
/// fast-follow (plan T0.3). Adding it later means adding a variant and filling
/// in the `match` arms below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    Sqlite,
    Postgres,
    MySql,
}

impl Dialect {
    /// The MLflow `db_type` string for this dialect (matches
    /// `mlflow/store/db/db_types.py`).
    pub fn db_type(self) -> &'static str {
        match self {
            Dialect::Sqlite => "sqlite",
            Dialect::Postgres => "postgresql",
            Dialect::MySql => "mysql",
        }
    }

    /// Quote an SQL identifier (table/column name) for this dialect.
    ///
    /// Any embedded quote character is doubled to avoid injection through
    /// identifiers, matching the standard escaping rule for each backend.
    pub fn quote_ident(self, ident: &str) -> String {
        match self {
            Dialect::Sqlite | Dialect::Postgres => {
                format!("\"{}\"", ident.replace('"', "\"\""))
            }
            Dialect::MySql => format!("`{}`", ident.replace('`', "``")),
        }
    }

    /// The positional bind-parameter placeholder for the 1-based `index`.
    ///
    /// Postgres uses `$1, $2, ...`; SQLite and MySQL both accept `?`.
    pub fn placeholder(self, index: usize) -> String {
        match self {
            Dialect::Postgres => format!("${index}"),
            Dialect::Sqlite | Dialect::MySql => "?".to_string(),
        }
    }

    /// Whether `INSERT ... RETURNING` is supported.
    ///
    /// SQLite (>= 3.35) and Postgres support it; MySQL does not, so the store
    /// must fall back to a follow-up `SELECT` there.
    pub fn supports_returning(self) -> bool {
        match self {
            Dialect::Sqlite | Dialect::Postgres => true,
            Dialect::MySql => false,
        }
    }

    /// Whether `NULLS LAST` / `NULLS FIRST` ordering is supported natively.
    ///
    /// Postgres and SQLite (>= 3.30) support the explicit syntax; MySQL does
    /// not and needs the `col IS NULL` emulation trick.
    pub fn supports_nulls_ordering(self) -> bool {
        match self {
            Dialect::Sqlite | Dialect::Postgres => true,
            Dialect::MySql => false,
        }
    }

    /// Build an idempotent `INSERT ... ON CONFLICT/ON DUPLICATE KEY UPDATE`
    /// statement.
    ///
    /// Mirrors `_upsert_batch` (`sqlalchemy_store.py:9806`): SQLite/Postgres use
    /// `ON CONFLICT (<pk>) DO UPDATE SET col = excluded.col`; MySQL uses
    /// `ON DUPLICATE KEY UPDATE col = VALUES(col)`. When `update_columns` is
    /// empty the statement degrades to a conflict-ignoring no-op (matching the
    /// Python `on_conflict_do_nothing` / self-update-on-PK behavior).
    ///
    /// A single row is emitted (one placeholder per column). Batching multiple
    /// rows is the caller's concern; this keeps the generated SQL analyzable and
    /// unit-testable.
    pub fn upsert(&self, spec: &UpsertSpec<'_>) -> String {
        let table = self.quote_ident(spec.table);
        let cols: Vec<String> = spec.columns.iter().map(|c| self.quote_ident(c)).collect();
        let mut ph = 0usize;
        let values: Vec<String> = spec
            .columns
            .iter()
            .map(|c| {
                ph += 1;
                let p = self.placeholder(ph);
                if matches!(self, Dialect::Postgres) && spec.json_columns.contains(c) {
                    format!("CAST({p} AS json)")
                } else {
                    p
                }
            })
            .collect();
        let insert = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table,
            cols.join(", "),
            values.join(", ")
        );

        match self {
            Dialect::Sqlite | Dialect::Postgres => {
                let pk: Vec<String> = spec
                    .pk_columns
                    .iter()
                    .map(|c| self.quote_ident(c))
                    .collect();
                if spec.update_columns.is_empty() {
                    format!("{insert} ON CONFLICT ({}) DO NOTHING", pk.join(", "))
                } else {
                    let sets: Vec<String> = spec
                        .update_columns
                        .iter()
                        .map(|c| {
                            let q = self.quote_ident(c);
                            format!("{q} = excluded.{q}")
                        })
                        .collect();
                    format!(
                        "{insert} ON CONFLICT ({}) DO UPDATE SET {}",
                        pk.join(", "),
                        sets.join(", ")
                    )
                }
            }
            Dialect::MySql => {
                if spec.update_columns.is_empty() {
                    // Self-assign the first PK column to make the statement a
                    // silent no-op on duplicate (mirrors the Python fallback).
                    let first_pk = self.quote_ident(spec.pk_columns[0]);
                    format!("{insert} ON DUPLICATE KEY UPDATE {first_pk} = {first_pk}")
                } else {
                    let sets: Vec<String> = spec
                        .update_columns
                        .iter()
                        .map(|c| {
                            let q = self.quote_ident(c);
                            format!("{q} = VALUES({q})")
                        })
                        .collect();
                    format!("{insert} ON DUPLICATE KEY UPDATE {}", sets.join(", "))
                }
            }
        }
    }

    /// Render a case-sensitive `LIKE` predicate for a string column.
    ///
    /// * SQLite: plain `col LIKE ?` — case sensitivity comes from the
    ///   connection-level `PRAGMA case_sensitive_like=true`.
    /// * Postgres: plain `col LIKE ?` — `LIKE` is case-sensitive already.
    /// * MySQL: `(col LIKE ? AND BINARY col LIKE ?)` — the default collation is
    ///   case-insensitive, so a `BINARY` comparison is added to force
    ///   case-sensitivity (`search_utils.py:294`). Both placeholders bind the
    ///   same value.
    ///
    /// `column` must already be a safe, quoted identifier or qualified name.
    pub fn case_sensitive_like(self, column: &str, ph_index: usize) -> String {
        match self {
            Dialect::Sqlite | Dialect::Postgres => {
                format!("{column} LIKE {}", self.placeholder(ph_index))
            }
            Dialect::MySql => {
                let p1 = self.placeholder(ph_index);
                let p2 = self.placeholder(ph_index + 1);
                format!("({column} LIKE {p1} AND BINARY {column} LIKE {p2})")
            }
        }
    }

    /// Render a case-insensitive `LIKE` (i.e. SQL `ILIKE`) predicate.
    ///
    /// * Postgres: native `ILIKE`.
    /// * SQLite: `LIKE` is case-insensitive for ASCII by default, but the store
    ///   forces case sensitivity via `PRAGMA case_sensitive_like=true`, so
    ///   `LOWER(col) LIKE LOWER(?)` is used to get case-insensitive behavior.
    /// * MySQL: plain `col LIKE ?` — the default collation is already
    ///   case-insensitive.
    pub fn case_insensitive_like(self, column: &str, ph_index: usize) -> String {
        let ph = self.placeholder(ph_index);
        match self {
            Dialect::Postgres => format!("{column} ILIKE {ph}"),
            Dialect::Sqlite => format!("LOWER({column}) LIKE LOWER({ph})"),
            Dialect::MySql => format!("{column} LIKE {ph}"),
        }
    }
}

/// Inputs to [`Dialect::upsert`].
#[derive(Debug, Clone, Default)]
pub struct UpsertSpec<'a> {
    /// Target table name (unquoted).
    pub table: &'a str,
    /// All columns to insert, in order (unquoted).
    pub columns: &'a [&'a str],
    /// Primary-key columns that define the conflict target (unquoted).
    pub pk_columns: &'a [&'a str],
    /// Non-PK columns to overwrite on conflict (unquoted). Empty => do-nothing.
    pub update_columns: &'a [&'a str],
    /// Columns with a `json` SQL type whose text bind must be cast on
    /// Postgres (`CAST($n AS json)`); sqlite/mysql accept plain text binds.
    pub json_columns: &'a [&'a str],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec<'a>(
        table: &'a str,
        columns: &'a [&'a str],
        pk: &'a [&'a str],
        update: &'a [&'a str],
    ) -> UpsertSpec<'a> {
        UpsertSpec {
            table,
            columns,
            pk_columns: pk,
            update_columns: update,
            ..Default::default()
        }
    }

    #[test]
    fn upsert_sqlite() {
        let s = spec(
            "latest_metrics",
            &["key", "run_uuid", "value", "step"],
            &["key", "run_uuid"],
            &["value", "step"],
        );
        assert_eq!(
            Dialect::Sqlite.upsert(&s),
            "INSERT INTO \"latest_metrics\" (\"key\", \"run_uuid\", \"value\", \"step\") \
             VALUES (?, ?, ?, ?) ON CONFLICT (\"key\", \"run_uuid\") DO UPDATE SET \
             \"value\" = excluded.\"value\", \"step\" = excluded.\"step\""
        );
    }

    #[test]
    fn upsert_postgres_uses_dollar_placeholders() {
        let s = spec(
            "latest_metrics",
            &["key", "run_uuid", "value"],
            &["key", "run_uuid"],
            &["value"],
        );
        assert_eq!(
            Dialect::Postgres.upsert(&s),
            "INSERT INTO \"latest_metrics\" (\"key\", \"run_uuid\", \"value\") \
             VALUES ($1, $2, $3) ON CONFLICT (\"key\", \"run_uuid\") DO UPDATE SET \
             \"value\" = excluded.\"value\""
        );
    }

    #[test]
    fn upsert_mysql_on_duplicate_key() {
        let s = spec(
            "latest_metrics",
            &["key", "run_uuid", "value"],
            &["key", "run_uuid"],
            &["value"],
        );
        assert_eq!(
            Dialect::MySql.upsert(&s),
            "INSERT INTO `latest_metrics` (`key`, `run_uuid`, `value`) \
             VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)"
        );
    }

    #[test]
    fn upsert_no_update_columns_do_nothing() {
        let s = spec("inputs", &["a", "b"], &["a", "b"], &[]);
        assert_eq!(
            Dialect::Sqlite.upsert(&s),
            "INSERT INTO \"inputs\" (\"a\", \"b\") VALUES (?, ?) ON CONFLICT (\"a\", \"b\") DO NOTHING"
        );
        assert_eq!(
            Dialect::MySql.upsert(&s),
            "INSERT INTO `inputs` (`a`, `b`) VALUES (?, ?) ON DUPLICATE KEY UPDATE `a` = `a`"
        );
    }

    #[test]
    fn quote_ident_escapes() {
        assert_eq!(Dialect::Postgres.quote_ident("a\"b"), "\"a\"\"b\"");
        assert_eq!(Dialect::MySql.quote_ident("a`b"), "`a``b`");
    }

    #[test]
    fn case_sensitive_like_forms() {
        assert_eq!(
            Dialect::Sqlite.case_sensitive_like("runs.name", 1),
            "runs.name LIKE ?"
        );
        assert_eq!(
            Dialect::Postgres.case_sensitive_like("runs.name", 3),
            "runs.name LIKE $3"
        );
        assert_eq!(
            Dialect::MySql.case_sensitive_like("runs.name", 1),
            "(runs.name LIKE ? AND BINARY runs.name LIKE ?)"
        );
    }

    #[test]
    fn case_insensitive_like_forms() {
        assert_eq!(
            Dialect::Postgres.case_insensitive_like("runs.name", 2),
            "runs.name ILIKE $2"
        );
        assert_eq!(
            Dialect::Sqlite.case_insensitive_like("runs.name", 1),
            "LOWER(runs.name) LIKE LOWER(?)"
        );
        assert_eq!(
            Dialect::MySql.case_insensitive_like("runs.name", 1),
            "runs.name LIKE ?"
        );
    }

    #[test]
    fn capabilities() {
        assert!(Dialect::Sqlite.supports_returning());
        assert!(Dialect::Postgres.supports_returning());
        assert!(!Dialect::MySql.supports_returning());
        assert!(!Dialect::MySql.supports_nulls_ordering());
    }
}
