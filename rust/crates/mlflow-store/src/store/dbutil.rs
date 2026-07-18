//! Small dialect-agnostic query execution helpers over the [`Db`] enum.
//!
//! The `Db` enum holds a concrete per-backend pool, which means every query
//! would otherwise need a three-arm `match`. To keep the store methods readable
//! we funnel all runtime SQL through a tiny typed-value binder here: build a
//! statement string (with placeholders from [`crate::dialect::Dialect`]),
//! collect the bind values into a `Vec<Val>`, and run it via one of the
//! executors below. This is the same pattern the store layer uses throughout
//! T2.4/T2.5.
//!
//! A [`Tx`] wraps a per-backend transaction so a whole `log_batch` (plan Q6) or
//! a run/experiment lifecycle change runs atomically.

use sqlx::{MySql, Postgres, Row, Sqlite, Transaction};

use crate::db::Db;

/// A bindable SQL value covering every column type the store writes.
#[derive(Debug, Clone)]
pub(crate) enum Val {
    Text(String),
    OptText(Option<String>),
    Int(i64),
    OptInt(Option<i64>),
    Float(f64),
    OptFloat(Option<f64>),
    OptJson(Option<serde_json::Value>),
    Bool(bool),
    Bytes(Vec<u8>),
}

/// Bind a single [`Val`] onto a sqlx query for one backend. A macro keeps the
/// three backend arms in lockstep.
macro_rules! bind_val {
    ($q:expr, $v:expr) => {{
        match $v {
            Val::Text(s) => $q.bind(s.clone()),
            Val::OptText(s) => $q.bind(s.clone()),
            Val::Int(i) => $q.bind(*i),
            Val::OptInt(i) => $q.bind(*i),
            Val::Float(f) => $q.bind(*f),
            Val::OptFloat(f) => $q.bind(*f),
            Val::OptJson(value) => $q.bind(value.clone().map(sqlx::types::Json)),
            Val::Bool(b) => $q.bind(*b),
            Val::Bytes(bytes) => $q.bind(bytes.clone()),
        }
    }};
}

/// An open transaction on one of the backends.
pub(crate) enum Tx<'c> {
    Sqlite(Transaction<'c, Sqlite>),
    Postgres(Transaction<'c, Postgres>),
    MySql(Transaction<'c, MySql>),
}

impl Db {
    /// Begin a transaction.
    pub(crate) async fn begin_tx(&self) -> Result<Tx<'static>, sqlx::Error> {
        match self {
            Db::Sqlite(p) => Ok(Tx::Sqlite(p.begin().await?)),
            Db::Postgres(p) => Ok(Tx::Postgres(p.begin().await?)),
            Db::MySql(p) => Ok(Tx::MySql(p.begin().await?)),
        }
    }

    /// Execute a non-returning statement outside a transaction; return the
    /// number of affected rows.
    pub(crate) async fn exec(&self, sql: &str, vals: &[Val]) -> Result<u64, sqlx::Error> {
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
}

impl Tx<'_> {
    /// Execute a non-returning statement inside the transaction; return affected
    /// rows.
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

    /// Fetch all rows inside the transaction, mapping each with `f`.
    pub(crate) async fn fetch_all<T, F>(
        &mut self,
        sql: &str,
        vals: &[Val],
        f: F,
    ) -> Result<Vec<T>, sqlx::Error>
    where
        F: Fn(&dyn RowLike) -> Result<T, sqlx::Error>,
    {
        match self {
            Tx::Sqlite(tx) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                let rows = q.fetch_all(&mut **tx).await?;
                rows.iter().map(|r| f(r as &dyn RowLike)).collect()
            }
            Tx::Postgres(tx) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                let rows = q.fetch_all(&mut **tx).await?;
                rows.iter().map(|r| f(r as &dyn RowLike)).collect()
            }
            Tx::MySql(tx) => {
                let mut q = sqlx::query(sql);
                for v in vals {
                    q = bind_val!(q, v);
                }
                let rows = q.fetch_all(&mut **tx).await?;
                rows.iter().map(|r| f(r as &dyn RowLike)).collect()
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

impl Db {
    /// Fetch all rows outside a transaction, mapping each with `f`.
    pub(crate) async fn fetch_all<T, F>(
        &self,
        sql: &str,
        vals: &[Val],
        f: F,
    ) -> Result<Vec<T>, sqlx::Error>
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

    /// Fetch at most one row.
    pub(crate) async fn fetch_optional<T, F>(
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

/// A tiny row-accessor abstraction so mapping closures can read columns without
/// naming a concrete backend `Row` type. Backed by each backend's `Row`.
pub(crate) trait RowLike {
    fn get_i64(&self, col: &str) -> Result<i64, sqlx::Error>;
    fn get_opt_i64(&self, col: &str) -> Result<Option<i64>, sqlx::Error>;
    fn get_string(&self, col: &str) -> Result<String, sqlx::Error>;
    fn get_opt_string(&self, col: &str) -> Result<Option<String>, sqlx::Error>;
    fn get_f64(&self, col: &str) -> Result<f64, sqlx::Error>;
    fn get_opt_f64(&self, col: &str) -> Result<Option<f64>, sqlx::Error>;
    fn get_bool(&self, col: &str) -> Result<bool, sqlx::Error>;
    fn get_opt_json(&self, col: &str) -> Result<Option<serde_json::Value>, sqlx::Error>;
    fn get_bytes(&self, col: &str) -> Result<Vec<u8>, sqlx::Error>;

    /// Read an SQLAlchemy `Integer` column (e.g. `experiment_id`), widening to
    /// `i64`. On Postgres this maps to `INT4`/`i32`; SQLite and MySQL store it
    /// as `i64`, so the widening is a no-op there. Kept distinct from
    /// [`RowLike::get_i64`] (used for `BigInteger`/`INT8` columns) so we never
    /// mis-decode a 32-bit column as 64-bit on Postgres.
    fn get_int(&self, col: &str) -> Result<i64, sqlx::Error>;
    fn get_opt_int(&self, col: &str) -> Result<Option<i64>, sqlx::Error>;
}

/// Backends that store SQLAlchemy `Integer` as a native 64-bit column
/// (`get_int` == `get_i64`).
macro_rules! impl_rowlike_i64_int {
    ($t:ty) => {
        impl RowLike for $t {
            fn get_i64(&self, col: &str) -> Result<i64, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_i64(&self, col: &str) -> Result<Option<i64>, sqlx::Error> {
                self.try_get(col)
            }
            fn get_string(&self, col: &str) -> Result<String, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_string(&self, col: &str) -> Result<Option<String>, sqlx::Error> {
                self.try_get(col)
            }
            fn get_f64(&self, col: &str) -> Result<f64, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_f64(&self, col: &str) -> Result<Option<f64>, sqlx::Error> {
                self.try_get(col)
            }
            fn get_bool(&self, col: &str) -> Result<bool, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_json(&self, col: &str) -> Result<Option<serde_json::Value>, sqlx::Error> {
                self.try_get::<Option<sqlx::types::Json<serde_json::Value>>, _>(col)
                    .map(|value| value.map(|value| value.0))
            }
            fn get_bytes(&self, col: &str) -> Result<Vec<u8>, sqlx::Error> {
                self.try_get(col)
            }
            fn get_int(&self, col: &str) -> Result<i64, sqlx::Error> {
                self.try_get(col)
            }
            fn get_opt_int(&self, col: &str) -> Result<Option<i64>, sqlx::Error> {
                self.try_get(col)
            }
        }
    };
}

impl_rowlike_i64_int!(sqlx::sqlite::SqliteRow);
impl_rowlike_i64_int!(sqlx::mysql::MySqlRow);

// Postgres: `Integer` columns are `INT4` (`i32`), `BigInteger` are `INT8`
// (`i64`). `get_int` reads `i32` and widens.
impl RowLike for sqlx::postgres::PgRow {
    fn get_i64(&self, col: &str) -> Result<i64, sqlx::Error> {
        self.try_get(col)
    }
    fn get_opt_i64(&self, col: &str) -> Result<Option<i64>, sqlx::Error> {
        self.try_get(col)
    }
    fn get_string(&self, col: &str) -> Result<String, sqlx::Error> {
        self.try_get(col)
    }
    fn get_opt_string(&self, col: &str) -> Result<Option<String>, sqlx::Error> {
        self.try_get(col)
    }
    fn get_f64(&self, col: &str) -> Result<f64, sqlx::Error> {
        self.try_get(col)
    }
    fn get_opt_f64(&self, col: &str) -> Result<Option<f64>, sqlx::Error> {
        self.try_get(col)
    }
    fn get_bool(&self, col: &str) -> Result<bool, sqlx::Error> {
        self.try_get(col)
    }
    fn get_opt_json(&self, col: &str) -> Result<Option<serde_json::Value>, sqlx::Error> {
        self.try_get::<Option<sqlx::types::Json<serde_json::Value>>, _>(col)
            .map(|value| value.map(|value| value.0))
    }
    fn get_bytes(&self, col: &str) -> Result<Vec<u8>, sqlx::Error> {
        self.try_get(col)
    }
    fn get_int(&self, col: &str) -> Result<i64, sqlx::Error> {
        let v: i32 = self.try_get(col)?;
        Ok(i64::from(v))
    }
    fn get_opt_int(&self, col: &str) -> Result<Option<i64>, sqlx::Error> {
        let value: Option<i32> = self.try_get(col)?;
        Ok(value.map(i64::from))
    }
}
