# Rust server deployment

T22.4 is the serving-plane cutover: nginx and the Rust server are the only
long-running application containers. Rust owns tracking, tracing, artifacts,
registry, auth/RBAC, workspaces, GraphQL, the complete GenAI/gateway/Assistant
surface, PromptLab, and all six native job kinds.

For an existing Python installation, follow
[MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md). To reverse the cutover, follow
[ROLLBACK.md](ROLLBACK.md); rollback now requires restoring a Python service,
not merely changing an already-present upstream.

## Topology and runtime boundary

```text
clients -> nginx -> Rust mlflow-server -> shared databases/artifact store
            |              |
            |              +-> mlflow-genai-worker subprocesses
            +-> static React shell/assets

Python/Alembic migration init -> exits before Rust starts
```

`mlflow-server` and `mlflow-genai-worker` are executable siblings in the same
Python-free production image. The server resolves the worker at startup when
job execution is enabled and fails closed if it is missing or non-executable.
Do not set `MLFLOW_GENAI_WORKER_PATH` for the stock image; sibling resolution is
the intended layout.

Database migrations remain Python/Alembic-owned. Run the digest-pinned Python
image as a one-shot init job for tracking and, when enabled, auth migrations.
The reference compose mounts this checkout's migration revisions read-only and
copies them into the ephemeral init container before `mlflow db upgrade`, so a
release image reaches the schema head expected by the branch. The job must
finish before Rust starts and must not be exposed as a Service or nginx
upstream. This is the sole Python exception in the reference deployment.

## nginx routing contract

The final contract has no Python rows:

| Route | Backend | Notes |
|---|---|---|
| `/` | nginx static | React `index.html`; fail 503 if the required build is absent |
| `/static-files/*` | nginx static | hashed assets; missing files return 404 |
| `/ajax-api/3.0/mlflow/assistant/*` | Rust | SSE; buffering off, 3600s read timeout |
| `/gateway/*` | Rust | streaming; buffering off, 3600s read timeout |
| `/ajax-api/2.0/mlflow/gateway-proxy` | Rust | streaming settings retained |
| `/(api|ajax-api)/3.0/mlflow/scorer/invoke` | Rust | streaming settings retained |
| everything else | Rust | includes every tracking, GenAI, jobs, PromptLab, artifact, registry, auth, workspace, and ops route |

The committed contract is [`../deploy/nginx.conf`](../deploy/nginx.conf).
It defines only `rust_backend`; `/python/health` and the Python static fallbacks
were removed. Every response is tagged `X-MLflow-Backend: rust` or `static`.
Keep `client_max_body_size 0` for artifact uploads and preserve the explicit
streaming locations if adapting the config to another proxy.

When `--static-prefix` is configured, apply the same prefix consistently to
the proxy locations and frontend asset URLs.

## Reference Docker Compose

[`../deploy/docker-compose.yml`](../deploy/docker-compose.yml) is the canonical
local/full-stack example:

```text
postgres --healthy--> migrate --completed--> rust --healthy--> nginx
```

The `migrate` init container runs `mlflow db upgrade` and exits. There is no
Python server service and nginx depends only on Rust. The Rust image is built by
[`../deploy/Dockerfile.rust`](../deploy/Dockerfile.rust), which compiles and
copies both release binaries into `/usr/local/bin` and installs no Python
runtime.

```bash
bash rust/deploy/build_placeholder_ui.sh  # or build the real UI
docker compose -f rust/deploy/docker-compose.yml build
docker compose -f rust/deploy/docker-compose.yml up -d --wait
bash rust/deploy/smoke.sh
bash rust/deploy/smoke_frontend.sh

IMAGE=$(docker compose -f rust/deploy/docker-compose.yml images -q rust)
bash rust/deploy/audit_image.sh "$IMAGE"

docker compose -f rust/deploy/docker-compose.yml down -v
```

The audit fails if the serving image contains a Python executable on `PATH`, a
libpython shared object, any `site-packages` directory, or any `.py` payload. It
also launches the image against an isolated migrated database with native jobs
enabled and requires `/health` 200.

## Kubernetes and other orchestrators

Use the same dependency order:

1. Run tracking and auth Alembic migrations as pinned, non-serving Jobs.
2. Deploy the Rust image with `mlflow-server` as PID 1. The image already
   contains `mlflow-genai-worker`; do not mount Python packages into it.
3. Require the Rust `/health` readiness probe before exposing nginx/Ingress.
4. Serve the React build from nginx, a static image, or a CDN.
5. Route every application request to the Rust Service. Retain buffering-off
   and long timeouts for Assistant/gateway streaming routes.

There must be no `mlflow-python` Deployment/Service, `python_backend` upstream,
or route-specific Python fallback after cutover. Pin the Rust image by digest
and run `audit_image.sh` on the exact promoted tag before rollout.

One Rust replica is the conservative default when signup is enabled because
the signup-CSRF secret is process-local. If signup is unused, or the signup
GET/POST pair has session affinity, scale after representative load testing.

## Required runtime configuration

CLI flags override their environment variables. Important settings are:

| Flag | Environment | Behavior |
|---|---|---|
| `--backend-store-uri` | `MLFLOW_BACKEND_STORE_URI` | required for application APIs; schema head is verified at startup |
| `--default-artifact-root` | `MLFLOW_DEFAULT_ARTIFACT_ROOT` | defaults to `./mlruns` |
| `--serve-artifacts` / `--no-serve-artifacts` | `MLFLOW_SERVE_ARTIFACTS` | proxy artifact serving defaults on |
| `--artifacts-destination` | `MLFLOW_ARTIFACTS_DESTINATION` | local/file and S3 are supported |
| `--app-name basic-auth` | `MLFLOW_AUTH_CONFIG_PATH` | enables auth/RBAC |
| `--enable-workspaces` / `--disable-workspaces` | `MLFLOW_ENABLE_WORKSPACES` | flags override environment |
| `--trace-archival-config` | `MLFLOW_TRACE_ARCHIVAL_CONFIG` | optional archival scheduler config |
| `--host`, `--port` | `MLFLOW_HOST`, `MLFLOW_PORT` | bind address; compose uses `0.0.0.0:5000` |
| none | `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` | enabled by default; startup resolves the native worker |

Use a persistent `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` anywhere encrypted
webhook secrets exist. The auth database has its own Alembic lineage and must
be migrated with the matching auth migration command.

SQL pool variables retain their Python names:

- `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE`
- `MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW`
- `MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE`
- `MLFLOW_SQLALCHEMYSTORE_ECHO`

Without overrides, Rust uses at most 15 connections and starts with zero idle
connections.

## Artifact destination support

The all-Rust serving deployment is supported for local paths, `file:` URIs,
and `s3://` destinations, including S3-compatible endpoints and multipart
uploads. S3 configuration uses the standard `AWS_*` variables plus
`MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_S3_IGNORE_TLS`, and
`MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE`.

> **Known limitation:** `gs://`, `gcs://`, `wasbs://`, `abfss://`, `az://`, and
> `azure://` artifact destinations remain fail-loud `NOT_IMPLEMENTED` seams in
> Rust.

For GCS/Azure, use client-direct uploads or deliberately retain a separately
managed Python artifact-only plane and routing rule. That exception is outside
the full-Rust reference compose and means the deployment is not yet
Python-free. Never silently route an unsupported destination to Python.
