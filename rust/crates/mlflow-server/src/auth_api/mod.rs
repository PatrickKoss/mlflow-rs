//! HTTP surface for the auth/RBAC API (Phase 9).
//!
//! T9.3 owns the role / permission / assignment endpoints in [`roles`]. Other
//! auth endpoints (user CRUD, login) are T9.2; this module is intentionally
//! minimal so the two tasks compose without conflict — the orchestrator merges
//! the additional submodules when T9.2 lands.

pub mod roles;

pub use roles::register_role_routes;
