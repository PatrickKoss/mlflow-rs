# MLflow Rust tracking server

This directory contains the Rust reimplementation of the MLflow tracking,
tracing, model registry, webhooks, auth/RBAC, and workspaces server. See
[`RUST_TRACKING_SERVER_PLAN.md`](../RUST_TRACKING_SERVER_PLAN.md) at the repo
root for the full design, scope, and work breakdown — this README only
covers how the `rust/` tree is laid out.

GenAI functionality (gateway, scorers, evaluation, assistant, jobs, etc.)
stays on the Python server; see the plan's §1/§2.2 for the full routing
split.

## Layout

```
rust/
├── Cargo.toml            # virtual workspace manifest, [workspace.dependencies]
├── rust-toolchain.toml   # pinned toolchain
├── rustfmt.toml
├── crates/
│   ├── mlflow-server      # binary: axum HTTP app tying everything together
│   ├── mlflow-proto        # generated protobuf types + MLflow JSON wire codec
│   ├── mlflow-store        # tracking/tracing backend store (sqlx)
│   ├── mlflow-registry     # model registry store (sqlx)
│   ├── mlflow-auth          # RBAC auth: users/roles/permissions, enforcement
│   ├── mlflow-search        # search filter DSL parsers
│   ├── mlflow-artifacts     # artifact proxy / streaming (object_store)
│   └── mlflow-webhooks      # webhook storage + delivery engine
└── spikes/                # standalone spike project (not a workspace member)
```

`spikes/` is excluded from the workspace (`exclude = ["spikes"]`) and owns
its own `Cargo.toml`; it is used for throwaway proofs-of-concept (e.g.
verifying the werkzeug password hash and Fernet decryption crates, per plan
task T0.4) that shouldn't be built as part of `cargo build --workspace`.

## Building and testing

```bash
cd rust
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs the same commands on every push/PR touching `rust/**`; see
[`.github/workflows/rust.yml`](../.github/workflows/rust.yml).
