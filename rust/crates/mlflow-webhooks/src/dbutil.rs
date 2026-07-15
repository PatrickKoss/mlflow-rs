//! Dialect-agnostic query-execution helpers over `mlflow_store::Db`.
//!
//! A trimmed local copy of the same binder/executor pattern that
//! `mlflow-registry/src/dbutil.rs` carries. `mlflow-store` already contains an
//! identical helper module, but its `Val`/`Tx`/`RowLike` types and `Db` query
//! methods are `pub(crate)` — invisible outside `mlflow-store` — and the plan
//! forbids editing `mlflow-store` (a concurrent agent owns it). So this crate
//! carries the subset it needs (text/int binds; string/opt-string/opt-i64
//! reads), byte-for-byte behaviorally identical to the registry copy pending
//! consolidation into a shared `mlflow-db` crate.

use mlflow_store::Db;
use sqlx::{MySql, Postgres, Row, Sqlite, Transaction};

/// A bindable SQL value covering the column types the webhook store writes.
#[derive(Debug, Clone)]
pub(crate) enum Val {
    Text(String),
    OptText(Option<String>),
    Int(i64),
}

macro_rules! bind_val {
    ($q:expr, $v:expr) => {{
        match $v {
            Val::Text(s) => $q.bind(s.clone()),
            Val::OptText(s) => $q.bind(s.clone()),
            Val::Int(i) => $q.bind(*i),
        }
    }};
}

/// An open transaction on one of the backends.
pub(crate) enum Tx<'c> {
    Sqlite(Transaction<'c, Sqlite>),
    Postgres(Transaction<'c, Postgres>),
    MySql(Transaction<'c, MySql>),
}

/// Extension trait adding the query helpers this crate needs onto
/// `mlflow_store::Db` (whose own equivalents are crate-private).
pub(crate) trait DbExt {
    async fn begin_tx(&self) -> Result<Tx<'static>, sqlx::Error>;
    async fn exec(&self, sql: &str, vals: &[Val]) -> Result<u64, sqlx::Error>;
    async fn fetch_all<T, F>(&self, sql: &str, vals: &[Val], f: F) -> Result<Vec<T>, sqlx::Error>
    where
        F: Fn(&dyn RowLike) -> Result<T, sqlx::Error>;
    async fn fetch_optional<T, F>(
        &self,
        sql: &str,
        vals: &[Val],
        f: F,
    ) -> Result<Option<T>, sqlx::Error>
    where
        F: Fn(&dyn RowLike) -> Result<T, sqlx::Error>;
}

impl DbExt for Db {
    async fn begin_tx(&self) -> Result<Tx<'static>, sqlx::Error> {
        match self {
            Db::Sqlite(p) => Ok(Tx::Sqlite(p.begin().await?)),
            Db::Postgres(p) => Ok(Tx::Postgres(p.begin().await?)),
            Db::MySql(p) => Ok(Tx::MySql(p.begin().await?)),
        }
    }

    async fn exec(&self, sql: &str, vals: &[Val]) -> Result<u64, sqlx::Error> {
        match self {
            Db::Sqlite(p) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                Ok(q.execute(p).await?.rows_affected())
            }
            Db::Postgres(p) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                Ok(q.execute(p).await?.rows_affected())
            }
            Db::MySql(p) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                Ok(q.execute(p).await?.rows_affected())
            }
        }
    }

    async fn fetch_all<T, F>(&self, sql: &str, vals: &[Val], f: F) -> Result<Vec<T>, sqlx::Error>
    where
        F: Fn(&dyn RowLike) -> Result<T, sqlx::Error>,
    {
        match self {
            Db::Sqlite(p) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                let rows = q.fetch_all(p).await?;
                rows.iter().map(|r| f(r as &dyn RowLike)).collect()
            }
            Db::Postgres(p) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                let rows = q.fetch_all(p).await?;
                rows.iter().map(|r| f(r as &dyn RowLike)).collect()
            }
            Db::MySql(p) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                let rows = q.fetch_all(p).await?;
                rows.iter().map(|r| f(r as &dyn RowLike)).collect()
            }
        }
    }

    async fn fetch_optional<T, F>(
        &self,
        sql: &str,
        vals: &[Val],
        f: F,
    ) -> Result<Option<T>, sqlx::Error>
    where
        F: Fn(&dyn RowLike) -> Result<T, sqlx::Error>,
    {
        Ok(self.fetch_all(sql, vals, f).await?.into_iter().next())
    }
}

impl Tx<'_> {
    /// Execute a non-returning statement inside the transaction.
    pub(crate) async fn exec(&mut self, sql: &str, vals: &[Val]) -> Result<u64, sqlx::Error> {
        match self {
            Tx::Sqlite(tx) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                Ok(q.execute(&mut **tx).await?.rows_affected())
            }
            Tx::Postgres(tx) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                Ok(q.execute(&mut **tx).await?.rows_affected())
            }
            Tx::MySql(tx) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                Ok(q.execute(&mut **tx).await?.rows_affected())
            }
        }
    }

    /// Commit the transaction.
    pub(crate) async fn commit(self) -> Result<(), sqlx::Error> {
        match self {
            Tx::Sqlite(tx) => tx.commit().await,
            Tx::Postgres(tx) => tx.commit().await,
            Tx::MySql(tx) => tx.commit().await,
        }
    }
}

/// A tiny row-accessor abstraction so mapping closures can read columns without
/// naming a concrete backend `Row` type.
pub(crate) trait RowLike {
    fn get_string(&self, col: &str) -> Result<String, sqlx::Error>;
    fn get_opt_string(&self, col: &str) -> Result<Option<String>, sqlx::Error>;
    fn get_opt_i64(&self, col: &str) -> Result<Option<i64>, sqlx::Error>;
}

macro_rules! impl_rowlike {
    ($t:ty) => {
        impl RowLike for $t {
            fn get_string(&self, col: &str) -> Result<String, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_string(&self, col: &str) -> Result<Option<String>, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_i64(&self, col: &str) -> Result<Option<i64>, sqlx::Error> {
                self.try_get(col)
            }
        }
    };
}

impl_rowlike!(sqlx::sqlite::SqliteRow);
impl_rowlike!(sqlx::mysql::MySqlRow);
impl_rowlike!(sqlx::postgres::PgRow);
