# MLflow reference deployment (Rust + Python behind nginx)

Reference deployment for the split MLflow server (RUST_TRACKING_SERVER_PLAN.md
T11.3). One nginx front door on `:80` routes every request to the Rust tracking
server **except** the genai / gateway / jobs surface, which goes to the stock
Python MLflow server. Both share one Alembic-migrated Postgres database and a
shared artifact volume.

## Quick start

```bash
cd rust
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d --wait
bash deploy/smoke.sh              # asserts backend attribution; exits non-zero on mismatch
docker compose -f deploy/docker-compose.yml down -v
```

MLflow is then reachable at `http://localhost:80`.

## Service graph & startup ordering

```
postgres ──(healthy)──▶ migrate ──(completed)──▶ rust  ┐
                                              └──▶ python ┴──(both healthy)──▶ nginx :80
```

- **postgres** (`postgres:16`): backing DB, `pg_isready` healthcheck. No host
  port published.
- **migrate** (`ghcr.io/mlflow/mlflow`): one-shot `mlflow db upgrade` that brings
  the empty DB to the current Alembic head, then **exits 0**. This is the crux
  of the ordering: the Rust server refuses to start against an unmigrated DB
  (`Db::connect_and_verify`), so migration must finish before `rust` boots. Both
  `rust` and `python` `depends_on` it with `service_completed_successfully`.
- **rust**: the `mlflow-server` binary built from `Dockerfile.rust`. Serves on
  `:5000`, `--serve-artifacts`, artifacts to the shared `/mlartifacts` volume.
- **python** (`ghcr.io/mlflow/mlflow`): stock `mlflow server` on `:5001`, same
  Postgres URI and artifact volume. Hit only for the genai/gateway surface, and
  it ships the React UI build served at `/` for this task.
- **nginx** (`nginx:1.27-alpine`): reverse proxy on `:80` mounting `nginx.conf`.

## Routing table (nginx.conf, from plan §2.2)

Default rule: **everything not listed goes to Rust.**

| Prefix (regex) | Backend | Notes |
|---|---|---|
| `/(api\|ajax-api)/3.0/mlflow/{gateway,scorers,datasets,issues,genai,label-schemas,review-queues}/*` | Python | genai/gateway family |
| `/ajax-api/3.0/mlflow/assistant/*` | Python | SSE, `proxy_buffering off` |
| `/ajax-api/3.0/jobs/*` | Python | jobs API |
| `/gateway/*` | Python | streaming, `proxy_buffering off` |
| `/ajax-api/2.0/mlflow/gateway-proxy` | Python | `proxy_buffering off` |
| `/ajax-api/2.0/mlflow/runs/create-promptlab-run` | Python | genai authoring |
| `/(api\|ajax-api)/3.0/mlflow/scorer/invoke` | Python | genai eval, `proxy_buffering off` |
| `/python/health` | Python | rewritten to `/health` for ops |
| `/`, `/static-files/*` | Python | **T11.4** will move the UI build to nginx |
| **everything else** | **Rust** | tracking, runs, metrics, tracing, OTLP `/v1/traces`, artifacts (`/get-artifact`, `/model-versions/get-artifact`, `/mlflow-artifacts/*`, `/ajax-api/2.0/mlflow/upload-artifact`), logged models, registry, webhooks, `/graphql`, users/roles/permissions, `/signup`, workspaces, `server-info`, ui-telemetry, `/health`, `/version`, `/metrics` |

### Attribution

Every proxied response carries `X-MLflow-Backend: rust` or `python` (emitted with
`add_header ... always`, so it survives 4xx/5xx). `smoke.sh` asserts this header
per request. nginx also logs `backend=<tag>` per line via a custom `log_format`:

```bash
docker compose -f deploy/docker-compose.yml logs nginx
```

### Streaming / body-size tuning

- `proxy_buffering off` on the SSE/streaming Python locations (assistant,
  gateway, gateway-proxy, scorer/invoke) so tokens flush immediately.
- `client_max_body_size 0` (unlimited) at the server level so large artifact
  uploads through the proxy artifact plane are not capped.

## Ports & volumes

| | |
|---|---|
| nginx | `:80` (host) |
| rust | `:5000` (internal) |
| python | `:5001` (internal) |
| postgres | `:5432` (internal) |
| `artifacts` volume | mounted at `/mlartifacts` in both `rust` and `python` |

## smoke.sh

Waits for `/health`, then exercises the SDK surface through nginx and asserts
backend attribution: experiments (create / get-by-name / search), runs
(create / log-metric / log-parameter / get-history / search), traces (V3 start +
OTLP `/v1/traces`), registry (registered model + version + get), webhooks (list),
users endpoint (rust-attributed whether or not auth is enabled), artifact
upload+download+list via the proxy. Finally it fires genai/gateway/jobs requests
and asserts they attribute to **python** (a 404/501 from Python is fine —
attribution is the assertion). Exits non-zero on any mismatch.
