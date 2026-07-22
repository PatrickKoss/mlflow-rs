# MLflow Rust server

This directory contains the completed Rust reimplementation of the MLflow
server. It covers tracking and tracing, model registry, webhooks, auth/RBAC,
workspaces, local and S3-compatible artifact proxying, GraphQL and OTLP, and
the GenAI gateway, scorers, evaluation, Assistant, and jobs. The production
serving image contains the `mlflow-server` and `mlflow-genai-worker` native
binaries and no Python runtime. See the
[full implementation plan and record](../docs/rust-tracking-server-plan/README.md)
for the design, decisions, and verification journey.

## Why Rust?

The rewrite is intended to preserve MLflow's client and UI contracts while
substantially reducing the resources and latency of the server plane.

| Benchmark (Python → Rust) | Result |
|---|---|
| [Process-tree memory, idle / loaded](bench/memory.md) | **106.5× / 67.3× less RSS**; about 47 MiB vs 3,145 MiB loaded |
| [Core tracking reads under one-hour mixed load](bench/soak.md) | **13–40× lower p95 latency** for run search, experiment list/search, and metric history; 0 errors for both servers |
| [Heavyweight analytical queries on a 2.4 GB SQLite database](bench/RESULTS.md) | **1.7–5.4× lower p95 latency**; the prompt anti-join was at parity |
| [GenAI evaluation mixed soak](bench/genai_eval.md) | **117.8× less RSS and 9.9× less CPU** at equivalent throughput |

These results were measured on WSL2. Read the linked reports for the exact
hardware, workloads, methodology, caveats, and reproduction commands. In
particular, the 1.7–5.4× large-dataset query gains are smaller than the gains
for frequently used tracking endpoints during the mixed-load soak.

## Quickstart

The release Compose stack runs Postgres, a one-shot Alembic migration, the
published Rust server image, and the published nginx UI image:

```bash
cd rust/deploy
docker compose -f docker-compose.release.yml up -d --wait
```

MLflow is then available at <http://localhost:80>. The default image tag is
`latest`; select a release explicitly with, for example,
`MLFLOW_RUST_VERSION=0.1.0`. See the
[deployment guide](deploy/README.md) for verification and cleanup commands.

The images can also be pulled independently:

```bash
docker pull ghcr.io/patrickkoss/mlflow-rust:latest
docker pull ghcr.io/patrickkoss/mlflow-rust-ui:latest
```

Linux x86-64 archives containing both native binaries are available from the
[GitHub Releases page](https://github.com/PatrickKoss/mlflow-rs/releases). To
install the server and its sibling native job worker directly from Git, use
Rust 1.89 or newer:

```bash
cargo install --locked --git https://github.com/PatrickKoss/mlflow-rs.git --tag rust-v0.1.0 mlflow-server
cargo install --locked --git https://github.com/PatrickKoss/mlflow-rs.git --tag rust-v0.1.0 mlflow-genai-worker
```

The database must already be at MLflow's expected Alembic head before a native
binary starts. The release Compose stack performs this migration automatically.

## Compatibility

The Rust server provides wire parity for the MLflow REST and ajax APIs, GraphQL
endpoints, and OTLP ingestion contract used by the stock `mlflow` Python client
and the stock MLflow React UI. The Python client/SDK is unchanged and remains
the supported client; upstream documentation for client-side tracking,
tracing, registry, and GenAI workflows applies to this server.

Local/file and S3-compatible artifact destinations, including MinIO, are
implemented. GCS and Azure artifact proxy destinations are not yet implemented;
use client-direct uploads or a separately managed artifact plane for those
schemes.

## Versioning and releases

Rust server releases use independent semantic versions tagged as
`rust-vMAJOR.MINOR.PATCH`. The initial release is `rust-v0.1.0`. Each GitHub
release records the upstream MLflow commit/version to which it was synced and
publishes:

- `ghcr.io/patrickkoss/mlflow-rust:MAJOR.MINOR.PATCH` and `:latest`;
- `ghcr.io/patrickkoss/mlflow-rust-ui:MAJOR.MINOR.PATCH` and `:latest`;
- `mlflow-rust-MAJOR.MINOR.PATCH-linux-x86_64.tar.gz`, containing
  `mlflow-server`, `mlflow-genai-worker`, `LICENSE.txt`, and a short README;
- a matching SHA-256 sums file.

The Rust version advances independently of upstream MLflow's version. The fork
does not publish its path-dependent workspace crates to crates.io; use release
binaries or `cargo install --git` for native installs. It tracks upstream master
through periodic syncs; see the
[sync documentation](../docs/rust-sync/README.md) for the sync state and
process.

## Layout

```text
rust/
├── Cargo.toml                    # virtual workspace manifest
├── rust-toolchain.toml           # pinned Rust toolchain
├── crates/
│   ├── mlflow-artifacts          # local and S3-compatible artifact proxy
│   ├── mlflow-auth               # users, roles, permissions, and enforcement
│   ├── mlflow-error              # shared API and storage errors
│   ├── mlflow-genai              # GenAI domain stores and execution logic
│   ├── mlflow-genai-worker       # native background-job executable
│   ├── mlflow-proto              # generated protobuf types and JSON wire codec
│   ├── mlflow-registry           # model and prompt registry store
│   ├── mlflow-search             # search-filter DSL parsers
│   ├── mlflow-server             # Axum HTTP server executable and library
│   ├── mlflow-store              # tracking, tracing, and workspace stores
│   ├── mlflow-test-support       # shared Rust test infrastructure
│   └── mlflow-webhooks           # webhook persistence and delivery
├── bench/                       # benchmark harnesses and committed reports
├── compliance/                  # Python/Rust differential replay suites
├── deploy/                      # local and published-image Compose stacks
├── e2e/                         # end-to-end integration tests
├── genai-inventory/             # generated GenAI route/provider manifests
├── tests/                       # cross-crate integration tests
├── tools/                       # generation and maintenance tools
└── spikes/                      # excluded standalone proof-of-concept workspace
```

`spikes/` is excluded from the workspace and owns its own `Cargo.toml`; it is
not built by workspace commands.

## Building and testing

```bash
cd rust
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

For optimized native executables:

```bash
cargo build --release -p mlflow-server -p mlflow-genai-worker
```

CI runs the same workspace checks on changes that touch the Rust server; see
[the Rust workflow](../.github/workflows/rust.yml).
