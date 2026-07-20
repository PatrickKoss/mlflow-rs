# Python-server to Rust-server migration

This runbook replaces a long-running Python MLflow server with the all-Rust
serving stack. Python remains only as a pinned, one-shot Alembic migration job.
Keep [ROLLBACK.md](ROLLBACK.md) open during the change.

The final serving topology is nginx/static UI -> Rust for every application
route. It is valid for local/file and S3-compatible artifact destinations. GCS
and Azure artifact proxy destinations remain Rust `NOT_IMPLEMENTED` seams; move
those installations to client-direct uploads or retain a documented Python
artifact-only exception instead of claiming a Python-free cutover.

## 1. Pin, audit, and back up

1. Pin the Python migration image and Rust image to the same MLflow source
   revision. Record immutable image digests.
2. Confirm the database heads expected by the build. At this revision:

   | Database | Version table | Required head |
   |---|---|---|
   | tracking + registry | `alembic_version` | `c4a9b7d3e812` |
   | auth | `alembic_version_auth` | `f1a2b3c4d5e6` |

3. Back up tracking/registry and auth databases with native database tools and
   test restoring both backups into disposable databases.
4. Retain the auth config, webhook encryption key, object-store credentials,
   `MLFLOW_CRYPTO_KEK_PASSPHRASE`, and
   `MLFLOW_CRYPTO_KEK_VERSION`, plus their secret-manager version identifiers.
   A missing or wrong KEK does not fail Rust startup; it makes existing gateway
   secrets fail only when read. Do not log any secret value.
5. If gateway budgets use Redis, retain
   `MLFLOW_GATEWAY_BUDGET_REDIS_URL`, the Redis data, and access credentials.
   If trace archival is enabled, retain its YAML, archive-store credentials,
   and a database/archive backup pair. See
   [ARCHIVAL_RUNBOOK.md](ARCHIVAL_RUNBOOK.md).
6. Audit the exact Rust image before it is eligible for promotion:

   ```bash
   bash rust/deploy/audit_image.sh YOUR_RUST_IMAGE@sha256:...
   ```

   The audit must prove the image has no Python executable, libpython,
   site-packages, or `.py` payload and that a jobs-enabled server starts with
   its native worker sibling resolved.

## 2. Run the Python-owned migrations

Put schema-changing maintenance in the appropriate maintenance window, then
run the pinned migration image as short-lived jobs:

```bash
docker run --rm --network YOUR_DB_NETWORK YOUR_MIGRATION_IMAGE \
  mlflow db upgrade "$BACKEND_STORE_URI"

docker run --rm --network YOUR_DB_NETWORK YOUR_MIGRATION_IMAGE \
  python -m mlflow.server.auth db upgrade \
  --url "$AUTH_DATABASE_URI" --revision head
```

Verify both version tables afterward. Do not expose these containers through a
Service or nginx. Rust never initializes or upgrades the schema and refuses to
start against a missing, older, or newer head.

## 3. Stage Rust privately

Start the audited Rust image on an internal address with the production backend
URI, artifact configuration, auth config, workspace mode, and persistent
`MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`. Inject the existing
`MLFLOW_CRYPTO_KEK_PASSPHRASE` and `MLFLOW_CRYPTO_KEK_VERSION` unchanged. Keep
public traffic on Python during this staging step.

Do not set `MLFLOW_GENAI_WORKER_PATH` for the stock image: the worker is
co-installed beside `mlflow-server`. Leave native job execution enabled so a
missing worker is a startup failure.

Verify privately:

```bash
curl -fsS "$RUST_URL/health"
curl -fsS "$RUST_URL/version"
curl -fsS -H 'Content-Type: application/json' \
  -d '{"max_results":1}' \
  "$RUST_URL/api/2.0/mlflow/experiments/search"
curl -fsS "$RUST_URL/ajax-api/3.0/mlflow/gateway/supported-providers"
curl -fsS "$RUST_URL/ajax-api/3.0/mlflow/gateway/secrets/config"
```

Require the secrets response to show `using_default_passphrase:false`, then
exercise one existing encrypted secret through a deterministic provider mock.
The configuration response cannot detect a wrong passphrase. Also run
deterministic gateway/Assistant mocks and one native scorer job. The reference
`rust/deploy/smoke.sh` demonstrates a provider-free `ResponseLength` job and
requires it to reach `SUCCEEDED` through the jobs API. Size worker process caps
and memory using [DEPLOYMENT.md](DEPLOYMENT.md#native-worker-capacity-and-supervision)
before production traffic.

For more than one Rust replica, set
`MLFLOW_GATEWAY_BUDGET_REDIS_URL` on every replica before canary traffic. An
unset URL makes enforcement per-replica. If archival will continue through the
cutover, mount and validate the same config and archive root using the archival
runbook; do not enable a second scheduler against a different root.

For S3, verify PUT/GET plus multipart create/upload/complete using the exact
endpoint and addressing-style configuration intended for production.

## 4. Canary Rust

Because Rust now implements every application route, canary either selected
read-only routes first or a stable percentage of complete user sessions. Keep
`X-MLflow-Backend` attribution visible. Do not allow both Python and Rust job
runners to claim the shared jobs table simultaneously: disable Python job
execution before enabling Rust job traffic.

During the canary, exercise:

- experiment and run create/read round trips;
- trace creation and retrieval;
- artifact upload/download;
- registry and webhook operations;
- GenAI discovery/CRUD and mock gateway streaming;
- Assistant stub streaming;
- a deterministic native job to terminal success;
- authenticated requests and workspace selection, when enabled.

Watch error rate, latency, database locks, pool use, worker process count, and
memory against existing SLOs. A GCS/Azure artifact proxy installation must not
route those requests to Rust.

## 5. Full nginx cutover

1. Deploy the final nginx contract from `rust/deploy/nginx.conf`: only
   `rust_backend`, explicit Rust streaming locations, nginx static locations,
   and the default Rust location.
2. Require the frontend build to be present. Missing static content must fail
   closed; there is no Python UI fallback.
3. Route all traffic to Rust and run:

   ```bash
   bash rust/deploy/smoke.sh
   bash rust/deploy/smoke_frontend.sh
   ```

4. Require the smoke footer to report the native worker job succeeded and zero
   responses carried `X-MLflow-Backend: python`.
5. Record the Rust image digest, database heads, nginx config checksum, KEK
   passphrase/version identifiers, Redis budget backend, archival config/archive
   root, and cutover timestamp.

## 6. Remove the Python serving plane

After the observation window:

1. Scale the Python server to zero, then remove its Deployment/Service or
   compose service.
2. Remove `python_backend`, `/python/health`, every Python route exception, and
   every Python static fallback from proxy configuration.
3. Remove Python runtime images from the serving release manifests. Keep only
   the pinned migration Job definition and make its non-serving role explicit.
4. Confirm runtime discovery contains no Python server endpoint and nginx logs
   show only `backend=rust` or `backend=static`.
5. Preserve the pre-cutover release manifest or commit SHA required by
   [ROLLBACK.md](ROLLBACK.md).

If any required verification fails, restore the full Python service using the
rollback procedure. Do not downgrade or restore a database as the first
response; both servers use the same schemas and successful Rust writes remain
valid.
