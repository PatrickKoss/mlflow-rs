# Rust cutover rollback

After T22.4, rollback is no longer an nginx-only operation: the normal deploy
stack contains no Python server to receive traffic. Restore a compatible Python
service first, verify it privately, then change routing. Rust and Python use the
same databases and artifact store, so routing rollback normally requires no
data copy or schema downgrade.

The pre-cutover reference is commit `543081c33`. Prefer a release tag or an
archived, reviewed manifest derived from that commit. For incident comparison,
the old files can be inspected without changing the worktree:

```bash
git show 543081c33:rust/deploy/docker-compose.yml
git show 543081c33:rust/deploy/nginx.conf
```

Do not blindly deploy those files: reapply current secrets, image digests,
database URLs, artifact configuration, and any fixes made after that commit.

## 1. Confirm compatibility and choose scope

1. Query `alembic_version` and, when auth is enabled,
   `alembic_version_auth`.
2. Select a Python image that recognizes those exact heads. The pinned
   migration image for the current release is the safest default.
3. Decide whether to restore only selected routes or all application traffic.
   GCS/Azure artifact proxy rollback may be limited to the artifact family;
   a broad incident rollback can send every non-static route to Python.
4. Keep nginx/static serving in place if it is healthy. A Python UI fallback is
   optional rollback scope, not a prerequisite.
5. Recover the exact `MLFLOW_CRYPTO_KEK_PASSPHRASE` used by Rust. Preserve
   `MLFLOW_CRYPTO_KEK_VERSION` as well so new writes remain on the intended
   version. A successful health check does not prove the KEK can decrypt rows.
6. Decide whether the restored service continues Redis budget tracking and
   trace archival. Keep the Redis data and archive repository intact while
   making that decision; see [ARCHIVAL_RUNBOOK.md](ARCHIVAL_RUNBOOK.md).

Never restore a pre-cutover database over a live database. Valid writes may
have committed after the backup.

## 2. Re-add the Python service

Create a Python Service/compose service using the compatible image, current
backend URI, auth config, webhook encryption key, KEK passphrase/version,
artifact destination, and shared volumes. Start it privately with job execution
disabled:

```yaml
python:
  image: YOUR_COMPATIBLE_MLFLOW_IMAGE
  environment:
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION: "false"
  command:
    - mlflow
    - server
    - --host
    - 0.0.0.0
    - --port
    - "5001"
    - --backend-store-uri
    - YOUR_BACKEND_STORE_URI
```

Add a private healthcheck and wait for `/health` 200. Verify authenticated
tracking, GenAI discovery, an existing encrypted gateway secret through a
credential-free provider mock, and artifact access directly against port 5001
before adding `python_backend` to nginx.

If the Python service will own artifact proxying, include its exact production
artifact settings. Do not infer credentials or destination from the Rust
configuration.

If budgets used Redis before rollback, pass the same
`MLFLOW_GATEWAY_BUDGET_REDIS_URL` to a compatible Python service. Do not delete
the `mlflow:budget:` keys: switching to an unset URL creates process-local
windows and abandons shared in-flight state, so enforcement can differ during
the transition. Keep only the chosen serving plane handling requests while
verifying a fake-cost budget canary. If archival continues, transfer scheduler
ownership with the job runner and preserve the same config and archive root; if
it pauses, leave the archive readable for existing traces.

## 3. Restore routing atomically

Add the upstream only after Python is healthy:

```nginx
upstream python_backend {
    server python:5001;
}
```

For a full rollback, replace the default Rust proxy with Python while retaining
forwarded headers:

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

Restore buffering-off and long read timeouts on Python Assistant/gateway
streaming locations. Validate with `nginx -t`, then reload or roll out the proxy
atomically. Do not leave a route-specific Rust location above the Python
default unless it is deliberately part of a partial rollback.

## 4. Transfer job-runner ownership

The jobs table is the queue. Rust and Python job runners must never claim it at
the same time.

1. Keep Python job execution disabled while routing changes.
2. Stop new native job submissions or wait until Rust-owned jobs are terminal.
3. Restart Rust with `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` or scale Rust to
   zero.
4. Only then enable Python job execution and restart the Python service.
5. Submit one deterministic job and require terminal success.

## 5. Verify and stabilize

Confirm:

- `/health` and representative application routes carry
  `X-MLflow-Backend: python` for the selected rollback scope;
- experiment/run create and read round trips succeed;
- GenAI discovery and stubbed streaming work;
- artifact upload/download matches the configured destination;
- auth, workspace selection, webhook signatures, and one job succeed;
- an existing encrypted gateway secret decrypts with the carried KEK;
- Redis budget and archival ownership match the rollback decision;
- only one job runtime is enabled.

Watch error rate, Python worker memory, database locks, and pool saturation.
Preserve Rust/nginx logs and exact image digests for incident analysis.

## 6. Schema downgrade is a separate operation

Only consider downgrade if no compatible Python image can run against the
current heads. Take a fresh backup, stop every MLflow writer, test downgrade on
a restored copy, and obtain database-owner approval. Never downgrade merely to
speed up a routing rollback.

Record the restored Python image digest, manifest/pre-cutover reference, nginx
checksum, database heads, KEK version identifier, Redis/archive ownership,
job-runner handoff time, and rollback start/end times.
