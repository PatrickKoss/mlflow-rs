//! `mlflow-auth`: RBAC authentication and authorization.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§3.16, Phase 9), this crate owns the
//! four-table auth DB (users, roles, role_permissions,
//! user_role_assignments), werkzeug-compatible password hash
//! verification/generation (shared `basic_auth.db` with the Python server),
//! the `READ < USE < EDIT < MANAGE` permission model with synthetic
//! `__user_<id>__` roles for per-user grants, and the tower middleware that
//! enforces authentication/authorization for every Rust-served route
//! (mirroring `mlflow/server/auth/__init__.py`).

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
