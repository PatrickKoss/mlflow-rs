# MLflow all-Rust reference deployment

This is the post-T22.4 reference stack: nginx exposes one front door on `:80`,
serves the built React shell and hashed assets directly, and sends every other
route to `mlflow-server`. The serving image also contains the sibling
`mlflow-genai-worker`, used for all six native job kinds. No Python interpreter,
library, site-packages directory, or `.py` payload is present in that image.

The `migrate` service is the deliberate exception. MLflow schemas remain owned
by Alembic, so a digest-pinned stock MLflow image runs `mlflow db upgrade` as a
one-shot init job and exits before the Rust serving container starts. The
repository's migration revisions are mounted read-only into that ephemeral
container so it reaches the branch's expected head. It never receives traffic
and is not a fallback server.

## Published images quickstart

From a checkout of this repository, run the release stack without compiling
the server or UI locally:

```bash
cd rust/deploy
docker compose -f docker-compose.release.yml up -d --wait

bash smoke.sh
bash smoke_frontend.sh

docker compose -f docker-compose.release.yml down -v
```

This uses `ghcr.io/patrickkoss/mlflow-rust:latest` and
`ghcr.io/patrickkoss/mlflow-rust-ui:latest`. Pin both images to a release by
setting the shared version before the command, for example:

```bash
MLFLOW_RUST_VERSION=0.2.0 docker compose -f docker-compose.release.yml up -d --wait
```

The release stack still uses the digest-pinned stock MLflow image for its
one-shot `mlflow db upgrade`. It mounts the migration revisions from this
checkout, exits before serving starts, and requires no local image build. The
request-serving Rust and nginx containers contain no Python runtime.

## Build-from-source quickstart

```bash
cd rust

# Use the real UI build, or the hermetic placeholder for deployment smoke.
yarn --cwd ../mlflow/server/js install --immutable
yarn --cwd ../mlflow/server/js build
# Alternative: bash deploy/build_placeholder_ui.sh

docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d --wait
bash deploy/smoke.sh
bash deploy/smoke_frontend.sh

RUST_IMAGE=$(docker compose -f deploy/docker-compose.yml images -q rust)
bash deploy/audit_image.sh "$RUST_IMAGE"

docker compose -f deploy/docker-compose.yml down -v
```

MLflow is reachable at `http://localhost:80` while the stack is running.
MinIO's S3 API is published at `http://localhost:9000` and its console at
`http://localhost:9001`. The local defaults are bucket `mlflow` and credentials
`minioadmin` / `minioadmin`; override them with `MLFLOW_S3_BUCKET`,
`MINIO_ROOT_USER`, and `MINIO_ROOT_PASSWORD` before starting Compose.

## Service graph

```text
postgres --healthy--> migrate ---------\
                                      +--> rust --healthy--> nginx :80
minio --healthy--> minio-init --------/
```

- `postgres`: tracking, registry, GenAI, and job state.
- `migrate`: Python/Alembic one-shot init; exits before serving begins.
- `minio`: S3-compatible artifact storage; publishes the API on `:9000` and
  console on `:9001`.
- `minio-init`: one-shot creation of the configured artifact bucket.
- `rust`: `mlflow-server` plus the co-installed `mlflow-genai-worker`; owns all
  API, gateway, Assistant, PromptLab, and job traffic.
- `nginx`: serves the UI build and proxies every non-static request to Rust.

The compose artifact destination is `s3://mlflow` in MinIO and the default
artifact root is `mlflow-artifacts:/`, so normal clients upload artifacts through
the tracking server. Full-Rust serving is supported for local/file and
S3-compatible destinations. GCS and Azure artifact proxy destinations remain
fail-loud `NOT_IMPLEMENTED` seams; use client-direct uploads or retain a
separately managed Python artifact plane until those backends are ported.

## Seed object-backed traces

From the repository root, seed 6,000 traces split evenly across two experiments:

```bash
uv run --frozen python rust/tools/seed_minio_traces.py --total 6000
```

The `Tracking Server Proxy` experiment uses an `mlflow-artifacts:/` root. Its
`traces.json` payloads travel through nginx and the Rust artifact proxy before
being written to MinIO. The `Client Direct` experiment uses an `s3://` root, so
the Python client uploads `traces.json` directly to MinIO. Trace metadata is
always registered with the tracking server in both modes; only the artifact
payload route differs.

Use `--mode proxy` or `--mode direct` to exercise one route. The script accepts
`--tracking-uri`, `--s3-endpoint`, bucket and credential flags, prints progress,
and reports the experiment UI URLs plus the resulting object counts.

## Building the UI

nginx bind-mounts `mlflow/server/js/build/` read-only at
`/usr/share/mlflow-ui`. Create a production build with:

```bash
yarn --cwd mlflow/server/js install --immutable
yarn --cwd mlflow/server/js build
```

For an offline smoke, `rust/deploy/build_placeholder_ui.sh` writes a minimal
shell plus one hashed JavaScript asset. Missing builds no longer fall back to a
serving container: `/` returns 503 and missing assets return 404.

Release builds bake the same directory and `nginx.conf` into
`ghcr.io/patrickkoss/mlflow-rust-ui`. `Dockerfile.ui` uses the repository root
as its build context:

```bash
docker build -f rust/deploy/Dockerfile.ui -t mlflow-rust-ui:local .
```

## Routing and attribution

| Route | Backend | Notes |
|---|---|---|
| `/` | nginx static | `index.html`; `Cache-Control: no-cache` |
| `/static-files/*` | nginx static | hashed assets; 28-day cache |
| Assistant, `/gateway/*`, gateway-proxy, scorer invoke | Rust | explicit nginx locations retain buffering-off and long-read-timeout settings |
| every other route | Rust | tracking, tracing, GenAI, jobs, artifacts, registry, auth, workspaces, GraphQL, and ops |

nginx emits `X-MLflow-Backend: static` or `rust` on every response. There is no
`python_backend`, `/python/health`, or static fallback. `smoke.sh` records the
headers from every request and ends with a global assertion that none is
`X-MLflow-Backend: python`.

`client_max_body_size 0` preserves unlimited artifact uploads. The explicit
Assistant, gateway, gateway-proxy, and scorer-invoke locations disable proxy
buffering and use a 3600-second read timeout for streaming responses.

## Verification scripts

- `smoke.sh` exercises tracking, traces, registry, webhooks, artifacts, and all
  formerly split GenAI route families. It creates a trace, submits a
  deterministic `ResponseLength` scorer, polls the jobs API to `SUCCEEDED`, and
  proves the native worker launches without provider calls.
- `smoke_frontend.sh` verifies the UI shell and hashed asset cache policy, a
  GenAI hash-route shell, and a successful Rust-backed provider-discovery API.
- `audit_image.sh IMAGE_TAG` rejects any `python*` executable on `PATH`,
  `libpython*.so*`, `site-packages`, or `.py` file in the target image. It then
  migrates an isolated Postgres database with the one-shot init image, launches
  the audited image with native jobs enabled, requires `/health` 200, and
  verifies the server resolved its executable worker sibling. All temporary
  audit containers and its network are removed by a trap.
