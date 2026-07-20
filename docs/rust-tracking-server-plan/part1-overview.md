## 1. Goal and Non-Goals

### Goals

- Rust HTTP server serving the tracking + tracing + registry + auth + workspaces API with
  **byte-compatible JSON wire behavior** (same paths, field names, error format,
  pagination semantics).
- nginx in front routes **everything to Rust by default**; only genai paths go to Python.
- The React frontend is a **separate static deliverable** served by nginx directly
  (fully static, `mlflow/server/__init__.py:186-198`, no templating).
- Same backend databases: **SQLite, PostgreSQL, MySQL, MSSQL** (`mlflow/store/db/db_types.py`).
- Auth/RBAC enforcement in Rust for all Rust-served routes, sharing the auth DB with
  Python (which keeps enforcing for the genai routes it still serves).
- Drastically lower memory footprint vs. `N × Python worker` processes.
- Query layer designed for a **~100 GB database**: keyset pagination, semi-joins instead
  of join+DISTINCT, proper indexes, no unindexable full-text LIKE scans.
- A **compliance test harness** that runs the existing Python test suites against the
  Rust server (proven pattern from the Go store effort, see §6).

### Non-Goals (genai — stay in Python, routed there by nginx) — Part I boundary

> **2026-07-17:** every item below is now planned in Part II (Phases 15–22),
> except the Databricks-only endpoints, the UC registry/prompt services, and the
> deprecated standalone YAML gateway, which remain permanently out of scope
> (§11.3). This list stays authoritative as the Part I boundary and as the
> interim nginx routing contract until the matching Part II phase lands.

- Gateway (`/gateway/*`, `gateway-proxy`, secrets/endpoints/model-definitions/budget/guardrails)
- Scorers (`/3.0/mlflow/scorers/*`, scorer invoke, online scoring configs)
- GenAI evaluate (`/3.0/mlflow/genai/evaluate/invoke`), evaluation datasets
  (`/3.0/mlflow/datasets/*`)
- Issues CRUD + detection, label schemas, review queues
- Prompt **optimization** jobs and the UC prompt protos (note: the Prompts *registry* UI
  works purely on registry endpoints — covered by the Rust registry, §3.14)
- Assistant (`/ajax-api/3.0/mlflow/assistant/*`, SSE), promptlab
  (`create-promptlab-run`), server-side jobs API + Huey runner (`/ajax-api/3.0/jobs/*`) —
  jobs exist to run genai workloads (scorers/eval), so they stay Python (D10)
- Databricks-only endpoints (`unified-traces`, `get-online-trace-details`), Unity Catalog
  registry/prompt services
- Trace archival (`ARCHIVE_REPO`, `--trace-archival-config`) — v1 treats all spans as
  `TRACKING_STORE` (see D6)

---

## 2. Target Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                    nginx                     │
                        │  - serves React build (static)               │
                        │  - default route → Rust                     │
                        │  - genai paths → Python                     │
                        └───────┬──────────────────────────┬───────────┘
                                │ default                  │ genai only
                                ▼                          ▼
                    ┌───────────────────┐        ┌────────────────────┐
                    │   Rust server     │        │   Python mlflow    │
                    │  tracking/tracing │        │   server (uvicorn) │
                    │  registry/webhooks│        │   gateway, scorers │
                    │  auth/RBAC        │        │   eval, assistant, │
                    │  workspaces       │        │   jobs, issues …   │
                    └────┬─────────┬────┘        └────┬──────────┬────┘
                         │         │                  │          │
                         ▼         ▼                  ▼          ▼
              ┌────────────────┐ ┌──────────────────────┐ ┌────────────┐
              │ backend DB     │ │ auth DB              │ │ artifact   │
              │ (tracking +    │ │ (users/roles/grants, │ │ store      │
              │  registry +    │ │  alembic_version_auth│ │ (FS/S3/…)  │
              │  workspaces)   │ │  — shared by both)   │ │            │
              └────────────────┘ └──────────────────────┘ └────────────┘
```

Both servers point at the **same** backend DB, the same auth DB, and the same artifact
storage. Schema ownership stays with Python/alembic (§5.4); Rust verifies both alembic
heads at startup (`alembic_version` for the backend store, `alembic_version_auth` for the
auth DB) and refuses to run on a mismatch (mirrors `_verify_schema`,
`mlflow/store/db/utils.py:123-134`, and `mlflow/server/auth/db/utils.py`).

### 2.1 Routing insights

- Every proto endpoint is registered under **both** `/api/{v}.0/...` and
  `/ajax-api/{v}.0/...` with identical handlers (`mlflow/server/handlers.py:6737-6744`);
  the auth app mirrors the same dual registration for its own routes
  (`mlflow/server/auth/routes.py`). SDK uses `/api/`, UI uses `/ajax-api/`.
- With registry + auth + workspaces in Rust, the split becomes **"Rust by default,
  enumerate the genai exceptions"** — far simpler and more future-proof than enumerating
  Rust paths (new upstream genai endpoints fail safe to Python).
- `POST /v1/traces` (OTLP) exists only in the FastAPI wrapper
  (`mlflow/server/otel_api.py:92`) — high-volume tracing write path, Rust-native.
- `/graphql` serves tracking + registry reads (`mlflow/server/graphql/autogenerated_graphql_schema.py:303,341`)
  — fully in Rust scope now, including `mlflowSearchModelVersions`.
- `MLFLOW_STATIC_PREFIX` (`--static-prefix`) prepends a path prefix to *every* route
  (`handlers.py:6731`) — nginx config and Rust server must both honor it.
- **Auth enforcement is per-request inside each app**, so it must exist in both planes:
  Rust enforces for Rust-served routes; Python's existing auth app keeps enforcing for
  genai routes. Both read the same auth DB, so users/roles/grants are consistent (D1).
- SSE/streaming only exists on Python routes (assistant, gateway) — nginx needs
  `proxy_buffering off` + long timeouts for those locations only.

### 2.2 nginx routing table (the contract)

Default rule: **everything not listed below goes to Rust.** (Interim contract
for the split deployment — Part II's cutover, T22.4, deletes the Python rows one
phase at a time.)

| Route pattern (under optional static prefix) | Backend |
|---|---|
| `/`, `/static-files/*` | nginx static (React build from `mlflow/server/js/build/`) |
| `/(api\|ajax-api)/3.0/mlflow/{gateway,scorers,datasets,issues,genai,label-schemas,review-queues}/*` | Python |
| `/ajax-api/3.0/mlflow/assistant/*` (SSE) | Python |
| `/ajax-api/3.0/jobs/*` | Python |
| `/gateway/*` (streaming), `/ajax-api/2.0/mlflow/gateway-proxy` | Python |
| `/ajax-api/2.0/mlflow/runs/create-promptlab-run` | Python |
| `/ajax-api/3.0/mlflow/scorer/invoke`, genai-evaluate/issues invoke routes | Python |
| `/python/health` (rewritten `/health` for ops) | Python |
| **everything else** — tracking, tracing, OTLP `/v1/traces`, metrics, artifacts (`/get-artifact`, `/model-versions/get-artifact`, `/mlflow-artifacts/*`), logged models, registry (`registered-models/*`, `model-versions/*`), webhooks, `/graphql`, users/roles/permissions, `/signup`, workspaces, `server-info`, ui-telemetry, `/health`, `/version`, `/metrics` | **Rust** |

Frontend feasibility confirmed: all UI API calls are relative URLs
(`mlflow/server/js/src/common/utils/FetchUtils.ts:60`), hash router, un-templated
`index.html`. The admin (`src/admin/`) and account (`src/account/`) UIs call
users/roles/permissions endpoints that Rust now serves; the workspace selector sends the
`X-MLFLOW-WORKSPACE` header per request (`src/workspaces/utils/WorkspaceUtils.ts`).

### 2.3 Tech stack decision (proposed defaults)

| Concern | Choice | Rationale |
|---|---|---|
| HTTP framework | `axum` (tokio + tower + hyper) | tower middleware fits the before/after-request auth model |
| DB access | `sqlx` + hand-built SQL AST for the search DSLs | sqlite/postgres/mysql native; MSSQL via `tiberius` adapter (D2) |
| Protobuf | `prost` compiling `service.proto`, `model_registry.proto`, `webhooks.proto`, `assessments.proto`, `databricks.proto`, `mlflow_artifacts.proto`, OTLP protos | single source of truth shared with Python |
| JSON | custom serializer over prost types replicating `message_to_json` quirks (§4) | serde-default output will NOT match the wire |
| Object storage | `object_store` crate (S3/GCS/Azure/local) | one API, all major backends |
| Password hashing | `pbkdf2`/`scrypt` crates emitting/verifying **werkzeug's format** (`method$salt$hash`) | credential compatibility with existing auth DBs (§4.13) |
| Secrets encryption | `fernet` crate | webhook secrets are Fernet-encrypted (`MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`) |
| Webhook signing | HMAC-SHA256, `v1,<b64>` Standard-Webhooks format | parity with `mlflow/webhooks/delivery.py:121` |
| Metrics | `metrics` + prometheus exporter on `/metrics` | replaces gunicorn prometheus multiprocess |
| Config | clap CLI + env vars matching `mlflow server` flags | drop-in ops compatibility |

