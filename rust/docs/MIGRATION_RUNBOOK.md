# Python-only to split-server migration

This runbook moves an existing Python MLflow server behind nginx and adds the
Rust server. Both servers continue to use the same tracking/registry database,
auth database, and artifact store. Read [DEPLOYMENT.md](DEPLOYMENT.md) before
starting and keep [ROLLBACK.md](ROLLBACK.md) open during the change.

## 1. Pin the release and check prerequisites

1. Use a Python image and Rust image built from the same MLflow source revision.
   This Rust build requires these exact database heads:

   | Database | Version table | Required head |
   |---|---|---|
   | tracking + registry | `alembic_version` | `c4a9b7d3e812` |
   | auth | `alembic_version_auth` | `f1a2b3c4d5e6` |

2. Confirm that the target Python image contains both revisions before touching
   a database:

   ```bash
   docker run --rm YOUR_PYTHON_IMAGE python - <<'PY'
   from mlflow.store.db import utils
   from mlflow.server.auth.db.utils import _get_alembic_config
   from alembic.script import ScriptDirectory

   print("tracking:", utils._get_latest_schema_revision())
   print("auth:", ScriptDirectory.from_config(_get_alembic_config("sqlite://")).get_current_head())
   PY
   ```

   Expected output is `tracking: c4a9b7d3e812` and `auth: f1a2b3c4d5e6`.

3. Record the current revisions and retain the output with the change record:

   ```bash
   psql "$BACKEND_STORE_URI" -Atc 'select version_num from alembic_version'
   psql "$AUTH_DATABASE_URI" -Atc 'select version_num from alembic_version_auth'
   ```

   Adapt the query client for MySQL or SQLite. Do not start Rust yet. Rust does
   not initialize or migrate either schema; it refuses to boot unless both
   version tables contain exactly the heads above. A database ahead of the
   pinned build is also rejected.

## 2. Back up both databases and secrets

1. Stop schema-changing maintenance and retain a database-native backup of both
   databases. PostgreSQL example:

   ```bash
   pg_dump --format=custom --file=tracking-before-split.dump "$BACKEND_STORE_URI"
   pg_dump --format=custom --file=auth-before-split.dump "$AUTH_DATABASE_URI"
   pg_restore --list tracking-before-split.dump >/dev/null
   pg_restore --list auth-before-split.dump >/dev/null
   ```

2. Back up the current auth INI, `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`,
   `MLFLOW_FLASK_SERVER_SECRET_KEY`, object-store credentials, and their secret
   manager version identifiers. Do not print secret values into the change log.

3. Test restore into disposable databases. The backup step is incomplete until
   both restored databases can be queried.

Verification:

- both backup files are non-empty and pass the database tool's integrity/list check;
- the restored copies contain the two Alembic version tables;
- the secret manager can return the pinned key versions to both workloads.

## 3. Upgrade both databases with the pinned Python image

Put the Python service in maintenance/read-only mode if required by the database
and migration sizes. Migrations remain Python-owned.

1. Upgrade tracking and registry:

   ```bash
   docker run --rm --network YOUR_DB_NETWORK YOUR_PYTHON_IMAGE \
     mlflow db upgrade "$BACKEND_STORE_URI"
   ```

2. Upgrade the separate auth lineage:

   ```bash
   docker run --rm --network YOUR_DB_NETWORK YOUR_PYTHON_IMAGE \
     python -m mlflow.server.auth db upgrade --url "$AUTH_DATABASE_URI" --revision head
   ```

3. Query the heads again:

   ```bash
   test "$(psql "$BACKEND_STORE_URI" -Atc 'select version_num from alembic_version')" = c4a9b7d3e812
   test "$(psql "$AUTH_DATABASE_URI" -Atc 'select version_num from alembic_version_auth')" = f1a2b3c4d5e6
   ```

Do not substitute the tracking `mlflow db upgrade` command for the auth command:
the auth migrations use a different script directory and version table.

## 4. Make identity and key configuration identical where required

1. Mount the same auth INI into both servers and set the same
   `MLFLOW_AUTH_CONFIG_PATH`. Its `database_uri` must be the existing shared auth
   database. Rust reads the Python-created Werkzeug password hashes directly;
   do not reset or rehash users during cutover.

2. Supply the exact same persistent Fernet key to both servers:

   ```text
   MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY=<same url-safe-base64 32-byte key>
   ```

   If this key differs, a server cannot decrypt webhook secrets written by the
   other server. If it is absent, each process generates an ephemeral key; that
   is not safe for a split or replicated deployment. Generate a new key only
   for an installation that has no encrypted webhook secrets:

   ```bash
   openssl rand -base64 32 | tr '+/' '-_' | tr -d '\n'
   ```

3. Keep `MLFLOW_FLASK_SERVER_SECRET_KEY` on Python only. It signs Python Flask
   sessions and must remain stable across Python replicas and restarts.

4. Rust owns a separate signup-CSRF secret. The current Rust build generates it
   once per process; it has no configuration variable. A restart invalidates
   outstanding signup tokens, and multiple Rust replicas require session
   affinity for `GET /signup` and `POST /api/2.0/mlflow/users/create-ui`.
   Run one Rust replica unless that affinity is configured.

Verification:

- an existing user authenticates through a Rust-owned route and a Python-owned
  GenAI route with the same credentials;
- an existing encrypted webhook can be read/tested from both pinned images;
- neither server logs a generated/ephemeral webhook-key warning.

## 5. Deploy Rust beside Python with no public traffic

1. Start Rust with the same backend URI, artifact root, auth INI, Fernet key,
   workspace mode, and security settings. Do not put it in nginx yet.
2. If the artifact destination is S3, GCS, or Azure, start Rust with
   `--no-serve-artifacts`; keep Python's `--serve-artifacts` destination and
   route the artifact-proxy family to Python as shown in
   [DEPLOYMENT.md](DEPLOYMENT.md#known-limitation-cloud-artifact-proxy).
3. Wait for startup. A schema error is a hard stop; do not bypass the head pin.

Verification against the private Rust Service:

```bash
curl -fsS "$RUST_URL/health"
curl -fsS "$RUST_URL/version"
curl -fsS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  "$RUST_URL/api/2.0/mlflow/experiments/search" \
  -H 'Content-Type: application/json' -d '{"max_results":1}'
```

Check that Rust's logs show both database connections and no Alembic mismatch.

## 6. Canary

Route canaries only among endpoints implemented by both servers. Never send the
GenAI exception routes to Rust, and never send a cloud-backed artifact-proxy
request to Rust.

Choose one method:

1. Per-route canary: move read-only tracking routes first, for example
   `experiments/search`, `runs/get`, and `runs/search`.
2. Percentage canary: use nginx `split_clients` to assign a stable cookie or
   client address to Rust, then select an upstream in a normal location. Keep a
   header such as `X-MLflow-Backend` on every response so attribution is visible.

During the canary, execute a write/read round trip through Rust:

```bash
NAME="split-canary-$(date +%s)"
CREATE=$(curl -fsS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"$NAME\"}" \
  "$PUBLIC_URL/api/2.0/mlflow/experiments/create")
EXPERIMENT_ID=$(printf '%s' "$CREATE" | jq -r .experiment_id)
RUN=$(curl -fsS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d "{\"experiment_id\":\"$EXPERIMENT_ID\",\"start_time\":0}" \
  "$PUBLIC_URL/api/2.0/mlflow/runs/create")
RUN_ID=$(printf '%s' "$RUN" | jq -r '.run.info.run_id')
curl -fsS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  "$PUBLIC_URL/api/2.0/mlflow/runs/get?run_id=$RUN_ID" | jq -e \
  --arg id "$RUN_ID" '.run.info.run_id == $id'
```

Also verify:

- `/health` is Rust and `/python/health` is Python;
- an authenticated Rust request returns neither `401` nor `403`;
- a GenAI or jobs request carries `X-MLflow-Backend: python`;
- error rate, database locks, pool use, and p95 latency remain within the
  operator's existing SLO. For measured reference behavior, use
  [`../bench/soak.md`](../bench/soak.md), not new capacity assumptions.

## 7. Verify webhook delivery

Start a disposable HTTP receiver that records request headers, then create and
test a webhook through the Rust route:

```bash
WEBHOOK=$(curl -fsS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d '{"name":"split-test","url":"https://YOUR_RECEIVER.example/hook","events":[{"entity":"REGISTERED_MODEL","action":"CREATED"}],"secret":"one-time-test-secret"}' \
  "$PUBLIC_URL/api/2.0/mlflow/webhooks")
WEBHOOK_ID=$(printf '%s' "$WEBHOOK" | jq -r '.webhook.webhook_id')
curl -fsS -u "$MLFLOW_USER:$MLFLOW_PASSWORD" -X POST \
  "$PUBLIC_URL/api/2.0/mlflow/webhooks/$WEBHOOK_ID/test" | jq -e \
  '.result.success == true'
```

Confirm the receiver saw `X-MLflow-Signature`, `X-MLflow-Timestamp`, and
`X-MLflow-Delivery-Id`. Remove the disposable webhook after the test.

## 8. Full cutover

1. Apply the nginx default-to-Rust configuration from
   [DEPLOYMENT.md](DEPLOYMENT.md#nginx-routing-contract).
2. Keep Python at full capacity until the observation window passes.
3. Run the health, run round-trip, authenticated request, GenAI attribution,
   artifact upload/download, and webhook checks again.
4. Record image digests, both Alembic heads, secret versions, nginx config
   checksum, and the cutover time.
5. Scale Python only to the capacity required by GenAI, jobs, UI fallback, and
   cloud artifact-proxy traffic.

If any verification fails, follow [ROLLBACK.md](ROLLBACK.md). Do not downgrade a
database as part of the first response.
