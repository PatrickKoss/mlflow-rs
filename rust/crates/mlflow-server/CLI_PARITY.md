# `mlflow-server` CLI / env parity matrix (T11.1)

This document is the parity contract between Python's `mlflow server` command
(`mlflow/cli/__init__.py`, `mlflow/utils/cli_args.py`) and the Rust
`mlflow-server` binary (`src/config.rs`, `src/main.rs`, `src/lib.rs`).

Legend for **Status**:

- **supported** — flag is wired end-to-end; observable behaviour matches Python.
- **mapped** — accepted and translated to the Rust equivalent (different
  mechanism, same observable effect).
- **accepted-noop** — accepted for deploy-script parity, no runtime effect
  (documented interpretation; does **not** fail).
- **fail-loud** — a value the Rust server cannot honour is rejected at startup
  with an error naming the flag (per the AC "unsupported flags fail loudly").

Precedence matches click: the CLI flag wins over its env var (clap `env = ...`).

## `mlflow server` flags

| Python flag | Env var | Python default | Rust status | Notes |
|---|---|---|---|---|
| `--backend-store-uri` | `MLFLOW_BACKEND_STORE_URI` | `None` (→ sqlite/mlruns) | supported | Tracking + registry DB. When unset, Rust runs the ops-only app (health/version). |
| `--read-replica-backend-store-uri` | `MLFLOW_READ_REPLICA_BACKEND_STORE_URI` | `None` | accepted-noop (SEAM) | Stored on `ServerConfig`; a startup warning is logged. The Rust tracking `Db` does not yet split reads onto a replica (auth's `AuthDb` does; tracking read-splitting is a follow-up). All reads use the primary. |
| `--registry-store-uri` | `MLFLOW_REGISTRY_STORE_URI` | `None` (→ backend URI) | supported / fail-loud | If equal to (or defaulting from) `--backend-store-uri`, honoured. A *different* URI is rejected: the Rust registry shares the tracking DB. |
| `--default-artifact-root` | `MLFLOW_DEFAULT_ARTIFACT_ROOT` | `None` (→ `./mlruns` or proxy) | supported | Falls back to `./mlruns` when unset. |
| `--serve-artifacts` / `--no-serve-artifacts` | `MLFLOW_SERVE_ARTIFACTS` | `True` | supported | Gates the `mlflow-artifacts` proxy surface. `--no-serve-artifacts` flips the default off. |
| `--artifacts-destination` | `MLFLOW_ARTIFACTS_DESTINATION` | `./mlartifacts` | supported | Local-FS / `file:` URIs wired; cloud schemes error at request time (`NOT_IMPLEMENTED`). |
| `--artifacts-only` | `MLFLOW_ARTIFACTS_ONLY` | `False` | supported | Only the `MlflowArtifactsService` proxy routes + root `/get-artifact` and `/upload-artifact` are registered (the two endpoints Python leaves enabled). Tracking RPCs are omitted → 404 (Python returns 503 via `_disable_if_artifacts_only`). |
| `--host` / `-h` | `MLFLOW_HOST` | `127.0.0.1` | supported | Rust uses `-H` for the short flag (clap reserves `-h` for `--help`); the long `--host` and env var match Python. |
| `--port` / `-p` | `MLFLOW_PORT` | `5000` | supported | |
| `--workers` / `-w` | `MLFLOW_WORKERS` | `None` (→ 4) | accepted-noop | The Rust server is async (single tokio runtime); there are no worker processes. The value is logged and ignored. Does **not** fail (real deploy scripts pass it). |
| `--static-prefix` | `MLFLOW_STATIC_PREFIX` | `None` | supported | Same validation as `_validate_static_prefix`: must start with `/`, must not end with `/` (else fail-loud). |
| `--allowed-hosts` | `MLFLOW_SERVER_ALLOWED_HOSTS` | `None` (localhost+private) | supported (T11.2) | |
| `--cors-allowed-origins` | `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | `None` (localhost) | supported (T11.2) | |
| `--x-frame-options` | `MLFLOW_SERVER_X_FRAME_OPTIONS` | `SAMEORIGIN` | supported (T11.2) | `NONE` disables the header. |
| `--expose-prometheus` | `MLFLOW_EXPOSE_PROMETHEUS` | `None` | mapped | Python treats it as the multiprocess collector dir and gates the `/metrics` exporter on it being set. The Rust server has no multiprocess model: any value **enables** `/metrics`; unset → `/metrics` 404s. The path itself is unused. |
| `--app-name` | (none) | `None` (→ default app) | mapped / fail-loud | Only `basic-auth` is accepted; it enables the auth/RBAC API (equivalent to Python's `create_app` exporting `MLFLOW_AUTH_CONFIG_PATH`). Any other value fails loudly. Unset + `MLFLOW_AUTH_CONFIG_PATH` present also enables auth. |
| `--workspace-store-uri` | `MLFLOW_WORKSPACE_STORE_URI` | `None` (→ backend URI) | supported | Only takes effect when workspaces are enabled; else the tracking URI is used as the `gc` hint. |
| `--enable-workspaces` / `--disable-workspaces` | `MLFLOW_ENABLE_WORKSPACES` | `False` | supported | Flags override the env var (click COMMANDLINE precedence). When neither flag is passed, the env var (`true`/`1`) decides. |
| `--gunicorn-opts` | `MLFLOW_GUNICORN_OPTS` | `None` | fail-loud | Gunicorn-specific; not applicable to the async Rust server. Passing it (an unknown flag to clap) exits 2. |
| `--waitress-opts` | (none) | `None` | fail-loud | Waitress-specific (Windows WSGI); N/A. Unknown flag → exit 2. |
| `--uvicorn-opts` | `MLFLOW_UVICORN_OPTS` | `None` | fail-loud | Uvicorn-specific; N/A. Unknown flag → exit 2. |
| `--dev` | (none) | `False` | fail-loud | Python dev auto-reload; N/A. Unknown flag → exit 2. |
| `--trace-archival-config` | `MLFLOW_TRACE_ARCHIVAL_CONFIG` | `None` | supported (T21.1) | CLI wins over env. YAML schema/repository validation and errors match Python; incompatible with `--artifacts-only`. Runtime reads use a thread-safe 5s TTL cache with stale-on-refresh-error tolerance. Store/archive paths and scheduling are later T21 tasks. |
| `--secrets-cache-ttl` | (none) | `60` | fail-loud | Secrets cache not ported. Unknown flag → exit 2. |
| `--secrets-cache-max-size` | (none) | `1000` | fail-loud | Secrets cache not ported. Unknown flag → exit 2. |

> **Fail-loud mechanism.** clap rejects genuinely unknown flags (`--gunicorn-opts`,
> `--dev`, …) with exit code 2 and a message naming the flag — no silent
> ignore. Value-level parity failures (`--app-name` other than `basic-auth`, a
> mismatched `--registry-store-uri`, an invalid `--static-prefix`) are rejected
> post-parse by `ServerConfig::from_cli` with a message naming the flag.

## SQLAlchemy pool family (`MLFLOW_SQLALCHEMYSTORE_*`)

Mapped in `mlflow-store/src/pool.rs` (`PoolConfig::from_env`). SQLAlchemy's
`QueuePool` (pool_size persistent + max_overflow transient) maps onto sqlx's
single `max_connections` ceiling and `min_connections` floor.

| Python env var | SQLAlchemy meaning | Default | Rust mapping |
|---|---|---|---|
| `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE` | steady-state pool size | SQLAlchemy `5` | `min_connections = pool_size`; contributes to `max_connections`. |
| `MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW` | extra transient connections | SQLAlchemy `10` | `max_connections = pool_size + max_overflow`. |
| `MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE` | recycle after N seconds | `None` | `max_lifetime = N seconds`. |
| `MLFLOW_SQLALCHEMYSTORE_ECHO` | log all SQL | `False` | `echo` flag (sqlx logs via `tracing`). |
| `MLFLOW_SQLALCHEMYSTORE_POOLCLASS` | swap SQLAlchemy pool class | `None` | accepted-noop — sqlx has a single pool implementation with no swappable class. |

When neither `POOL_SIZE` nor `MAX_OVERFLOW` is set, the Rust default is
`max_connections = 15` (SQLAlchemy's `5 + 10`), `min_connections = 0` (sqlx opens
connections lazily — the conservative low-idle-RSS default, plan §5.5).

## Verification

- Unit tests: `src/config.rs` (`mod tests`) — flag parsing, env fallback,
  flag-overrides-env precedence, fail-loud cases.
- Pool mapping: `mlflow-store/src/pool.rs` (`mod tests`).
- CLI integration: `tests/cli_parity.rs` — spawns the binary; `--help` exit 0
  and flag presence, unknown-flag exit 2, fail-loud `--app-name` /
  `--registry-store-uri` / `--static-prefix`.
- Artifacts-only routing: `tests/artifacts_http.rs::artifacts_only_serves_proxy_but_not_tracking`.
