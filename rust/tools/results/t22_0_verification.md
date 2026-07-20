# T22.0 verification — S3 artifact proxy/factory support

Date: 2026-07-20

## Automated gates

| Gate | Result |
|---|---:|
| `cargo test -p mlflow-artifacts --features aws` | PASS — 43 tests |
| `cargo test -p mlflow-store` | PASS — 288 tests |
| `cargo test -p mlflow-server` | PASS — 762 tests |
| `cargo clippy -p mlflow-artifacts --features aws --all-targets -- -D warnings` | PASS |
| `cargo clippy -p mlflow-store --all-targets -- -D warnings` | PASS |
| `cargo clippy -p mlflow-server --all-targets -- -D warnings` | PASS |
| `cargo build --release -p mlflow-server` | PASS |
| `cargo fmt --all -- --check` | PASS |
| Ruff check for the differential tool | PASS |

## Live MinIO verification

MinIO came from `rust/bench/docker-compose.soak.yml` with the test endpoint
exported through `MLFLOW_TEST_S3_ENDPOINT`. All services and volumes started
for the run were removed afterward.

| Suite | Result |
|---|---:|
| Artifact repository put/get/list/delete, presigned GET, MPU complete + abort | PASS — 2 tests |
| Rust HTTP artifact proxy round trips, presigned GET, MPU complete + abort | PASS — 1 test |
| Trace archival archive/read/delete cycle using an `s3://` archive repository | PASS — 1 test |

## Python/Rust differential

`rust/tools/s3_artifact_proxy_differential.py` ran equivalent artifact-proxy
requests against Python and release-mode Rust servers backed by isolated MinIO
prefixes. Ordinary put/list/get/delete, presigned download, multipart
create/upload/complete, and multipart abort all matched.

- Result: PASS
- Cases: 10/10
- Non-allowlisted differences: 0
- Machine-readable evidence: `rust/tools/results/t22_0_s3_differential.json`
