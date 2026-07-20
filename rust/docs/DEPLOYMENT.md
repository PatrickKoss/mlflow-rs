# Rust server deployment

T22.4 is the serving-plane cutover: nginx and the Rust server are the only
long-running application containers. Rust owns tracking, tracing, artifacts,
registry, auth/RBAC, workspaces, GraphQL, the complete GenAI/gateway/Assistant
surface, PromptLab, and all six native job kinds.

For an existing Python installation, follow
[MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md). To reverse the cutover, follow
[ROLLBACK.md](ROLLBACK.md); rollback now requires restoring a Python service,
not merely changing an already-present upstream. To operate server-owned trace
archival, follow [ARCHIVAL_RUNBOOK.md](ARCHIVAL_RUNBOOK.md).

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
| `--trace-archival-config` | `MLFLOW_TRACE_ARCHIVAL_CONFIG` | optional archival YAML; validated at startup and incompatible with `--artifacts-only` |
| `--host`, `--port` | `MLFLOW_HOST`, `MLFLOW_PORT` | bind address; compose uses `0.0.0.0:5000` |
| none | `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` | enabled by default; startup resolves the native worker |
| none | `MLFLOW_CRYPTO_KEK_PASSPHRASE` | gateway-secret KEK root; unset silently uses the development-only default |
| none | `MLFLOW_CRYPTO_KEK_VERSION` | version stamped on new/rewritten secrets; defaults to `1` |
| none | `MLFLOW_GENAI_WORKER_PATH` | native worker executable; defaults to the `mlflow-genai-worker` sibling and fails startup if unavailable |
| none | `MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS` | per-kind process cap for scorer invoke, issue detection, and GenAI evaluation; defaults to `10` |
| none | `MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS` | per-kind process cap for trace and session online scoring; defaults to `5` |
| none | `MLFLOW_GENAI_EVAL_MAX_WORKERS` | row-level fan-out inside applicable native jobs; those paths default to `10` |
| none | `MLFLOW_JUDGE_MAX_ITERATIONS` | judge/tool loop cap; defaults to `20` |
| none | `MLFLOW_GATEWAY_BUDGET_REDIS_URL` | selects the shared Redis budget tracker; unset selects process-local memory |
| none | `MLFLOW_GATEWAY_BUDGET_REFRESH_INTERVAL` | budget-policy refresh interval in seconds; defaults to `600` |
| none | `PATH` | must contain `claude` and/or `codex` only when the corresponding Assistant CLI provider is used |
| none | `OPENAI_API_KEY` | one Codex authentication option; `codex login` is the other |

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

## Gateway-secret KEK

Always set `MLFLOW_CRYPTO_KEK_PASSPHRASE` in production. Generate at least 32
random bytes (for example, `openssl rand -base64 48`), store the result in the
deployment's secret manager, and inject it into every server replica. Do not
commit it to compose files, Kubernetes manifests, shell history, logs, or the
database backup. Back up the secret-manager version identifier with the
database backup.

An unset passphrase does **not** fail startup. Rust silently derives the KEK
from `mlflow-default-kek-passphrase-for-development-only`; a wrong passphrase
also lets the server start and only fails when an encrypted secret is used.
Check the configuration signal after every rollout:

```bash
curl -fsS "$MLFLOW_URL/ajax-api/3.0/mlflow/gateway/secrets/config"
# Require: "using_default_passphrase": false
```

That signal detects an absent or empty setting, not a wrong value. Prove the
value by invoking a known canary endpoint backed by an encrypted fake-provider
secret. A wrong value returns `Failed to decrypt secret. Check KEK passphrase,
secret metadata, or database integrity.` at read time. `/health` alone is not a
KEK check. Secret IDs and names are encryption AAD and must not be edited in the
database; create a replacement rather than renaming a secret.

### Version rotation

`MLFLOW_CRYPTO_KEK_VERSION` selects the version for new or rewritten rows.
Every row stores its own `kek_version`, and reads derive the KEK for that stored
version. For a routine version rotation that keeps the root passphrase:

1. Back up the database and confirm the current passphrase is recoverable from
   the secret manager. Inventory `secret_name` and `kek_version` from the
   `secrets` table without selecting ciphertext or plaintext.
2. Keep `MLFLOW_CRYPTO_KEK_PASSPHRASE` unchanged, increment
   `MLFLOW_CRYPTO_KEK_VERSION`, and roll all replicas.
3. Create or rewrite a canary secret and exercise it through a credential-free
   provider stub. Confirm the canary row has the new version and an untouched
   row still decrypts with its stored old version.
4. Re-enter each remaining secret value through the gateway secret update API.
   Confirm no old `kek_version` rows remain, then retain the backup according to
   policy.

The all-Rust deployment has no bulk re-key endpoint or tool. A version bump
does not rewrite existing rows. It also does not remediate a compromised root
passphrase: Rust uses the one current passphrase to derive every stored version.
Changing that passphrase makes all not-yet-rewritten rows unreadable. If the
passphrase itself must change, use a maintenance window, keep authoritative
copies of every plaintext in the external secret manager, switch the
passphrase and version, re-enter every value, and verify every dependent
endpoint before reopening traffic. Roll back both environment values together
if the rewrite cannot finish.

## Native worker capacity and supervision

The server launches one `mlflow-genai-worker` process per active job. The stock
image resolves the executable beside `mlflow-server`; setting
`MLFLOW_GENAI_WORKER_PATH` overrides that path. With job execution enabled, a
missing, non-file, or non-executable worker fails startup. There is no Python
fallback.

The server-side process caps are independent for each job kind:

| Job kinds | Cap | Default |
|---|---|---:|
| `invoke_scorer`, `invoke_issue_detection`, `invoke_genai_evaluate` | `MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS` | 10 each |
| `run_online_trace_scorer`, `run_online_session_scorer` | `MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS` | 5 each |
| `optimize_prompts` | fixed | 2 |

The default worst-case worker-process count per server replica is therefore
`3*10 + 2*5 + 2 = 42`, plus the server process. With configured values `J` and
`O`, size for `3*J + 2*O + 2`. Both server cap variables must be positive
integers; zero or malformed values fail startup. `MLFLOW_GENAI_EVAL_MAX_WORKERS`
adds row-level asynchronous fan-out inside applicable jobs (default `10` in
invoke and issue-detection paths); invalid or zero values fail the affected job.
`MLFLOW_JUDGE_MAX_ITERATIONS` limits a judge's provider/tool loop (default
`20`); use a positive integer. A malformed value falls back to `20`, while `0`
causes the judge to fail without an iteration.

There is no MLflow worker-memory setting. Apply an aggregate container/cgroup
limit and choose the process caps from measured peak resident memory. For
compose:

```yaml
services:
  rust:
    mem_limit: 4g
    environment:
      MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS: "4"
      MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS: "2"
      MLFLOW_GENAI_EVAL_MAX_WORKERS: "4"
```

The analogous Kubernetes control is `resources.requests.memory` plus
`resources.limits.memory` on the Rust container. Alert on cgroup OOM events,
container restarts, job failure/timeout counts, process count, and RSS. The
launcher places each job in a supervised process group with kill-on-drop; the
database job runner owns cancellation and timeout, so production workers have
no independent launcher timeout. Run `mlflow-server` as PID 1 under the
orchestrator and let its restart policy supervise server failure.

## Pinned scorer, provider, and optimizer manifests

`rust/genai-inventory/scorers.json`, `providers.json`,
`provider_manifest.json`, and `algorithms.json` are release inputs, not runtime
configuration. Their `pin`/`reference` fields bind scorer behavior, LiteLLM
provider transforms/pricing/token limits, GEPA (`0.0.27`), DSPy (`3.2.1`), and
other third-party behavior to the verified snapshot. The ledger stores SHA-256
checksums for `scorers.json`, `providers.json`, and `algorithms.json`; checksum
drift makes `validate_ledger.py` fail. `provider_manifest.json` and
`providers.json` are compiled into `mlflow-server`, so mounting edited JSON
beside a running container has no effect. The ledger does not checksum the
derived `provider_manifest.json`, so both its generated diff and the
`build_provider_manifest.py --check` result require separate review.

Treat an upgrade as a release change:

1. Update the intended pins, source references, generators, and native code in
   one review. Regenerate the inventory with the exact target package versions
   (replace the versions below), then derive the runtime provider manifest:

   ```bash
   uv run --with 'litellm==1.91.2' --with 'dspy==3.2.1' \
     python rust/genai-inventory/build_inventory.py
   uv run python rust/tools/build_provider_manifest.py
   uv run python rust/genai-inventory/validate_ledger.py
   uv run python rust/tools/build_provider_manifest.py --check
   ```

2. Regenerate changed semantic/recorder fixtures only against exact pinned
   packages and credential-free loopback providers. Review source provenance,
   license changes, manifest counts, and every checksum change.
3. Require the `compliance` job (request corpus + required semantic corpus),
   `python-conformance`, and `sse-recorders` to pass. When third-party semantic
   goldens change, also run the manual `semantic-oracle-refresh` job and review
   its artifact. Do not weaken comparisons or add an allowlist entry merely to
   accept drift.
4. Rebuild the Rust image, run `audit_image.sh` and the compose smoke tests,
   then deploy the new immutable image digest through the normal canary.

## Shared Redis budget tracking

When `MLFLOW_GATEWAY_BUDGET_REDIS_URL` is nonempty, the gateway uses Redis Lua
operations and `mlflow:budget:` keys for atomic, shared spend windows. When it
is unset or empty, the tracker is process-local memory. The in-memory backend
is suitable only for one server process: every horizontally scaled deployment
**must** set the Redis URL on every replica or budget enforcement is
independently calculated per replica.

`MLFLOW_GATEWAY_BUDGET_REFRESH_INTERVAL` controls policy refresh and defaults
to 600 seconds. Redis is selected from configuration without a startup
connection probe or backend-selection info log; exercise a fake-cost canary and
monitor the Redis keys and the server's budget refresh/record errors after each
rollout. Tracker refresh failures are fail-open, so Redis reachability is part
of the enforcement SLO. Restrict and encrypt Redis access, persist it according
to the budget-state recovery objective, and do not share the keyspace between
independent MLflow installations.

## Optional Assistant CLI providers

The production image needs neither Node.js nor an Assistant CLI unless the
`claude_code` or `codex` provider is offered. For those providers, the server
looks for executable `claude` or `codex` files on its own `PATH`; an ordinary
network-only sidecar is insufficient. Use either an organization-maintained
extended Rust image, or an init/sidecar pattern that places the Node runtime and
CLI installation on a read-only shared volume included in the Rust container's
`PATH`:

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
```

Pin the Node base and CLI package versions in the image build, persist only the
required CLI login state, and rerun `rust/deploy/audit_image.sh` on the result
to preserve the Python-free boundary. Authenticate Claude with `claude login`.
For Codex, inject `OPENAI_API_KEY` from the secret manager or run `codex login`.
The health probes execute a minimal CLI request, so test them only where such
provider access is intended:

```bash
curl -i "$MLFLOW_URL/ajax-api/3.0/mlflow/assistant/providers/claude_code/health"
curl -i "$MLFLOW_URL/ajax-api/3.0/mlflow/assistant/providers/codex/health"
```

A missing CLI returns 412 with its installation command; failed authentication
returns 401 with `claude login` or `OPENAI_API_KEY`/`codex login` remediation.
The rest of MLflow remains available and the UI shows the Assistant setup
state, so omit these binaries entirely when Assistant CLI use is not desired.

## Trace archival

Trace archival requires a validated YAML file, durable archive storage, and the
tracking/job scheduler. See [ARCHIVAL_RUNBOOK.md](ARCHIVAL_RUNBOOK.md) for
enablement, sizing, monitoring, retention overrides, forced archival, reads,
and recovery.

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
