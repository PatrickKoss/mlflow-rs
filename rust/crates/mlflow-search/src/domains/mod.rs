//! Per-domain filter/order_by parsers.
//!
//! Each domain mirrors the corresponding `Search*Utils` class in
//! `mlflow/utils/search_utils.py`. They share the tokenizer ([`crate::token`]),
//! grouper ([`crate::group`]), and clause-splitting ([`crate::common`]), but
//! differ in valid identifiers, alias maps, comparator validation timing, and
//! value typing — so each is its own module.

pub mod evaluation_datasets;
pub mod experiments;
pub mod logged_models;
pub mod model_versions;
pub mod registered_models;
pub mod runs;
pub mod traces;

mod shared;
