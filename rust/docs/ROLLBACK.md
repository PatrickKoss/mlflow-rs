# Split-server rollback

Rollback means changing nginx routing back to Python. It normally requires no
data copy or data migration because Rust and Python share the tracking/registry
database, auth database, and artifact store. See [MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md)
for the forward procedure and [DEPLOYMENT.md](DEPLOYMENT.md) for the normal route map.

## 1. Decide whether routing-only rollback is safe

1. Confirm the old Python image can run against the current database heads:

   ```bash
   psql "$BACKEND_STORE_URI" -Atc 'select version_num from alembic_version'
   psql "$AUTH_DATABASE_URI" -Atc 'select version_num from alembic_version_auth'
   ```

2. If the old Python release knows the current heads, do a routing-only
   rollback. This is the normal and fastest path.
3. If the database was upgraded past the old Python release's expected head,
   routing can still be flipped immediately, but that old image may not start or
   may mis-handle the newer schema. Route to the pinned migration Python image,
   not an incompatible old image. Schema downgrade is a separate, backed-up
   maintenance operation in section 4.

Never restore a pre-cutover database over a live database: both servers may
have committed valid writes after the backup.

## 2. Flip traffic to Python

1. Keep Rust running for diagnosis. Replace the default Rust location with the
   Python upstream. Preserve streaming settings on GenAI locations.

   ```nginx
   location / {
       proxy_pass http://python_backend;
       proxy_http_version 1.1;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       proxy_set_header X-Forwarded-Proto $scheme;
       proxy_set_header X-MLFLOW-WORKSPACE $http_x_mlflow_workspace;
       add_header X-MLflow-Backend python always;
   }
   ```

2. Validate and reload atomically:

   ```bash
   nginx -t
   nginx -s reload
   ```

   Kubernetes alternative: apply the rollback ConfigMap, run
   `kubectl rollout restart deployment/mlflow-nginx`, and wait for
   `kubectl rollout status deployment/mlflow-nginx`.

3. If a percentage canary was used, set its Rust percentage to zero first.
   Do not leave route-specific Rust locations above the Python default.

## 3. Verify and stabilize

1. Confirm attribution and health:

   ```bash
   curl -fsSI "$PUBLIC_URL/health" | grep -i '^X-MLflow-Backend: python'
   curl -fsS "$PUBLIC_URL/health"
   ```

2. Repeat the run create/read round trip and authenticated request from
   [MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md#6-canary).
3. Upload and download an artifact through the configured mode. For a cloud
   proxy, confirm `/(api|ajax-api)/2.0/mlflow-artifacts/*` reaches Python.
4. Test a webhook delivery and confirm its signature headers.
5. Watch HTTP error rate, Python worker memory, DB locks, and pool saturation.
6. Only after the observation window, scale Rust down. Preserve its logs and
   exact image digest for the incident record.

No database reconciliation is required: successful Rust writes were made to the
same stores Python now reads.

## 4. Optional schema downgrade

Do this only when an older Python binary must be restored and its documented
schema compatibility requires it. Take a new backup, stop all MLflow writers,
test the downgrade on a restored copy, and obtain the database owner's approval.
`mlflow db` intentionally has no downgrade command; invoke the repository's
Alembic scripts explicitly with the same MLflow source that supplied the
migrations.

The two newest tracking revisions have implemented downgrades:

| Revision | Downgrade target | Effect |
|---|---|---|
| `c4a9b7d3e812` | `a3f8c21d9b47` | Drops `index_span_attributes_key_value` and the `span_attributes` table. All extracted/backfilled attribute rows are lost. Original span JSON in `spans.content` is not deleted, but any new searchable rows that exist only in `span_attributes` are lost. |
| `a3f8c21d9b47` | `b7e4c1a90f23` | Drops the five query-performance indexes. No application rows are deleted; query latency may regress. The migration includes MySQL-safe foreign-key handling. |

Run one revision at a time and inspect the version after each step:

```bash
export DOWNGRADE_TARGET=a3f8c21d9b47  # then b7e4c1a90f23 if required
uv run python - "$BACKEND_STORE_URI" "$DOWNGRADE_TARGET" <<'PY'
import sys
from pathlib import Path
from alembic.command import downgrade
from alembic.config import Config

url, target = sys.argv[1:]
migrations = Path("mlflow/store/db_migrations").resolve()
cfg = Config(str(migrations / "alembic.ini"))
cfg.set_main_option("script_location", str(migrations))
cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
downgrade(cfg, target)
PY
psql "$BACKEND_STORE_URI" -Atc 'select version_num from alembic_version'
```

The Rust build documented here will refuse to start after either downgrade
because it requires `c4a9b7d3e812`. Keep routing on a Python image compatible
with the selected target. The auth database has a separate lineage; do not
downgrade it merely because the tracking database was downgraded.

## 5. Close the rollback

Record the nginx checksum, image digest, database heads, start/end times, lost
or replayed requests, and whether any schema downgrade occurred. If routing was
the only change, the forward migration can resume at the canary step after the
fault is corrected.
