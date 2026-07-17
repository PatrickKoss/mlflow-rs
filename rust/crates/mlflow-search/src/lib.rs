//! `mlflow-search`: search filter/order-by DSL parsers.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§4 item 8, Phase 2 T2.3), this crate
//! ports the filter/order-by grammars from `mlflow/utils/search_utils.py` for
//! runs, experiments, registered models, model versions, traces, and logged
//! models, **including error classification and verbatim error messages**.
//!
//! ## Approach
//!
//! MLflow tokenizes with the `sqlparse` library, so parity means reproducing
//! sqlparse's *observable* lexing + grouping, not its internals:
//!
//! - [`token`] ports the relevant `sqlparse` `SQL_REGEX` rows (keyword table
//!   embedded verbatim in `keywords_generated.rs`).
//! - [`group`] ports the `sqlparse.engine.grouping.group()` passes the grammar
//!   observes (parenthesis / period / identifier / order / comparison / aliased
//!   / identifier-list).
//! - [`common`] ports the base `SearchUtils` clause splitting and the
//!   `_join_in_comparison_tokens` IN / NOT IN / IS NULL grouping.
//! - [`domains`] holds one module per `Search*Utils` subclass.
//!
//! Parsers return a crate-local AST ([`Comparison`] list, [`OrderBy`]) plus a
//! [`SearchError`] carrying an `INVALID_PARAMETER_VALUE`-equivalent code and a
//! message that matches Python's `MlflowException` message byte-for-byte. The
//! crate is intentionally dependency-light (no axum); callers convert
//! [`SearchError`] into `mlflow-error`'s type at the boundary.

mod ast;
mod common;
mod domains;
mod error;
mod group;
mod keywords_generated;
mod literal_eval;
mod page_token;
mod token;

pub use ast::{Comparison, OrderBy, Value};
pub use error::{ErrorCode, Result, SearchError};
pub use page_token::{create_page_token, parse_start_offset_from_page_token};

pub use domains::logged_models::{
    AscendingValue, LoggedModelOrderBy, OrderByInput as LoggedModelOrderByInput, SqlaComparison,
    SqlaEntityType, SqlaValue,
};

/// Filter/order-by parsers, one module per search domain. Each mirrors the
/// corresponding `Search*Utils` class.
pub mod parse {
    pub use crate::domains::experiments::{
        parse_order_by as experiments_order_by, parse_search_filter as experiments_filter,
    };
    pub use crate::domains::logged_models::{
        parse_filter_string_sqlalchemy as logged_models_filter_sqlalchemy,
        parse_order_by as logged_models_order_by, parse_search_filter as logged_models_filter,
    };
    pub use crate::domains::model_versions::{
        parse_order_by as model_versions_order_by, parse_search_filter as model_versions_filter,
    };
    pub use crate::domains::registered_models::{
        parse_order_by as registered_models_order_by,
        parse_order_by_store as registered_models_order_by_store,
        parse_search_filter as registered_models_filter,
    };
    pub use crate::domains::runs::{
        parse_order_by as runs_order_by, parse_search_filter as runs_filter,
    };
    pub use crate::domains::traces::{
        parse_order_by as traces_order_by, parse_search_filter as traces_filter,
    };
}
