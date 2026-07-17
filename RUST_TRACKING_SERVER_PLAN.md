# Rust MLflow Server — Implementation Plan (everything except genai)

Status: **in progress — Phases 2, 3, 4, 5, 6, 7, 8 complete; Phase 9 complete
except T9.9 (admin UI validation); Phase 10 T10.1/T10.2 done (T10.3 in flight);
Phase 11 T11.2/T11.3/T11.4/T11.5 done (T11.1 in flight); Phase 12 T12.1/T12.2
done, T12.4 harness landed (not yet green).** · Branch:
`feature/rust-tracking-server` · Last updated: 2026-07-17

**Resume notes (2026-07-17):** Phase 9 (auth/RBAC) is complete through T9.8.
T12.4's differential harness was salvaged and landed as foundation, and the two
real parity bugs it surfaced are fixed (experiment `workspace` proto field 9,
and `searchExperiments` `view_type` proto2 default → ACTIVE_ONLY). All merged
and green (fmt/clippy/full workspace suite by exit code).

**In flight:** T10.3 (workspace request scoping), T11.1 (CLI/env parity).

**Open:**
- **T12.4 (differential replay harness)** — scaffolding + 133-case corpus +
  engine landed under `rust/compliance/` (runs via `uv run python
  rust/compliance/replay.py`; `--list` works). NOT yet green end-to-end: needs
  a full dual-server run to triage the remaining reported diffs (tag ordering,
  search pagination page-count, duplicate-experiment error message). Keep the
  checkbox unticked until the run is zero-non-allowlisted-diffs.
- **T12.6 (chaos test)** — partial WIP checkpointed on
  `worktree-agent-ad634b20ba0a07ff0` @ `cdd95e09f` (`chaos.rs` + `rust.yml` CI
  job), NOT verified/complete. Salvage or redo on resume.
- Remaining pre-existing: **T9.9** (admin/account UI validation),
  **T10.4** (workspace-aware auth — several T10.4 seams already marked in the
  auth code; needs T10.3 first), **T11.6** (UI smoke checklist),
  **T12.3/T12.5** (client-suite conformance + CI matrix), and **Phases 13–14**
  (scale benchmarks, memory/soak validation).
- Parity backlog opened by T12.1 (see its note): #1 type-mismatch validation
  messages (serde text vs Python's "Invalid value … for parameter …"); #3
  HTTP reason-phrase casing (gated in tests via `MLFLOW_RUST_STORE_TESTING`).
  #2 (auth enforcement) was closed by T9.4.

This document is the master plan for reimplementing the MLflow server in Rust for all
**non-genai** functionality: tracking, tracing, artifacts, GraphQL, **model registry,
webhooks, auth/RBAC (incl. the admin/account UI backend), and workspaces** — fronted by
nginx, with full wire-level feature parity against the Python implementation. GenAI
features (gateway, scorers, evaluation, issues, label schemas, review queues, prompt
optimization, assistant, jobs) stay on the Python server.

It is written so that individual tasks can be picked up by other contributors/models:
every task has a checkbox, acceptance criteria (AC), and a verification method (VER).
All facts were derived from the current codebase (file/line references included).
When in doubt, the Python implementation is the spec.

---

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

### Non-Goals (genai — stay in Python, routed there by nginx)

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

Default rule: **everything not listed below goes to Rust.**

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

---

## 3. API Surface to Implement (complete inventory)

Authoritative endpoint definitions: `mlflow/protos/service.proto`,
`mlflow/protos/model_registry.proto`, `mlflow/protos/webhooks.proto` (`databricks.rpc`
options), route generation `mlflow/server/handlers.py:6723-6807`, `HANDLERS` dict
(`handlers.py:7663`), auth routes `mlflow/server/auth/routes.py`. Every proto endpoint is
served on both `/api/` and `/ajax-api/` prefixes unless marked ajax-only.

### 3.1 Experiments (9 endpoints)

create, get, get-by-name, search (POST **and** GET), delete, restore, update,
set-experiment-tag, delete-experiment-tag. Search: `max_results` (int64, default 1000),
`page_token`, `filter`, `order_by[]`, `view_type` (`ACTIVE_ONLY=1, DELETED_ONLY=2, ALL=3`).

### 3.2 Runs (14 endpoints)

create, update, delete, restore, get, search, log-metric, log-parameter, set-tag,
delete-tag, log-batch, log-model (legacy), log-inputs, outputs (log-outputs).

Limits to replicate exactly: search `max_results` int32 default 1000 max **50000**
(`handlers.py:1934`); log-batch ≤1000 metrics / ≤1000 total entities; param value ≤6000
bytes; default view `ACTIVE_ONLY`; tiebreak ordering `start_time DESC, run_id`.

### 3.3 Metrics (2 + 2 ajax-only endpoints)

- `GET /mlflow/metrics/get-history` (paginated, store cap 25000)
- `GET /mlflow/metrics/get-history-bulk-interval` (proto + ajax route;
  `MAX_RESULTS_PER_RUN=2500`, sampling in `handlers.py:2223`; UI default 320 pts/chart)
- ajax-only: `GET /ajax-api/2.0/mlflow/metrics/get-history-bulk` (≤100 run_ids, cap
  25000; hand-rolled JSON, **not** proto-serialized, `handlers.py:2112`)

### 3.4 Datasets / inputs

- `POST (mlflow/)experiments/search-datasets` (proto path lacks leading `/`,
  `service.proto:684`; explicit ajax route `mlflow/server/__init__.py:135`)

### 3.5 Logged models (10 endpoints)

CRUD + finalize (PATCH), search (POST, default 50 max 50, dataset-scoped metric ordering,
encoded token `SearchLoggedModelsPaginationToken`), tags set/delete, artifact directory
listing, log-params; ajax-only artifact file download.

### 3.6 Tracing V3 (13 endpoints)

startTraceV3, getTraceInfoV3, getTrace (`?allow_partial=`), batchGetTraces,
batchGetTraceInfos, searchTracesV3 (default 100, max 500), deleteTracesV3 (time-based OR
id-based, `HasField` semantics), setTraceTagV3/deleteTraceTagV3, linkTracesToRun (≤100),
linkPromptsToTrace (stores name/version link only), calculateTraceFilterCorrelation
(NPMI), queryTraceMetrics (aggregations over traces/spans/assessments).

### 3.7 Tracing V2 (deprecated but still served — 7 endpoints)

startTrace, endTrace, getTraceInfo, searchTraces (GET), deleteTraces, setTraceTag,
deleteTraceTag under `/api/2.0/mlflow/traces...`. Thin adapters over V3 store paths.
The UI still calls `GET /ajax-api/2.0/mlflow/traces` for "contains traces".

### 3.8 OTLP ingestion

`POST /v1/traces` — `ExportTraceServiceRequest` as `application/x-protobuf` or JSON,
gzip `Content-Encoding`, required header `x-mlflow-experiment-id`, optional
`x-mlflow-run-id` (links completed traces to run). Persists via `log_spans` semantics
(`sqlalchemy_store.py:4971-5362`). Status codes 200/400/422/501
(`mlflow/server/otel_api.py:95-231`). Under auth: requires experiment UPDATE resolved
from the header (`auth/__init__.py:4441`).

### 3.9 Assessments (4 endpoints)

create / get / update (FieldMask paths: `assessment_name`, `expectation`, `feedback`,
`rationale`, `metadata`, `valid`) / delete under
`/api/3.0/mlflow/traces/{trace_id}/assessments...`. Override/supersede model
(`overrides` + `valid`).

### 3.10 Trace artifact fetch (ajax-only)

`GET /ajax-api/{2,3}.0/mlflow/get-trace-artifact?request_id=&path=` — span JSON
(`traces.json`) or trace attachment. Dispatch on `mlflow.trace.spansLocation` tag:
`TRACKING_STORE` → DB; `ARTIFACT_REPO` → artifact store; `ARCHIVE_REPO` → out of scope v1
(`handlers.py:4177-4234`).

### 3.11 Artifacts

- `GET /get-artifact?run_id=&path=` — stream run artifact (`validate_path_is_safe`,
  `handlers.py:1519`).
- `GET /model-versions/get-artifact?name=&version=&path=` — resolves
  `storage_location or source` via the registry store, then streams from the resolved
  repo, honoring `mlflow-artifacts:` proxying + workspace prefixes (`handlers.py:3033`).
- `MlflowArtifactsService` (8 endpoints): download/upload/list/delete under
  `/(api|ajax-api)/2.0/mlflow-artifacts/artifacts...`, multipart create/complete/abort,
  presigned URL. Gated by `--serve-artifacts`.
- ajax-only `POST /ajax-api/2.0/mlflow/upload-artifact`.

### 3.12 GraphQL (`/graphql`, GET+POST)

Query: `mlflowGetExperiment`, `mlflowGetRun`, `mlflowGetMetricHistoryBulkInterval`,
`mlflowListArtifacts`, `mlflowSearchModelVersions`.
Mutation: `mlflowSearchRuns`, `mlflowSearchDatasets`. All delegate to the same store
logic — with registry in Rust, **all resolvers are now in scope** (no proxying).
Replicate the query-safety/no-batching guard (`graphql_no_batching.py`) and the GraphQL
auth middleware behavior (§3.16, `auth/__init__.py:4139`).

### 3.13 Misc

`/health`, `/version`, `GET/POST /(api|ajax-api)/3.0/mlflow/server-info` (UI boot — Rust
owns it now, must report feature flags consistent with the split, D5),
`/ajax-api/3.0/mlflow/ui-telemetry` (GET/POST sink), `/metrics` (prometheus).

### 3.14 Model Registry (21 endpoints)

Proto `mlflow/protos/model_registry.proto:13-380`, handlers `handlers.py:7705-7726`.
All `since.major=2`.

**Registered models:** create, rename (POST), update (PATCH), delete (DELETE), get,
search (GET; default 100, threshold 1000), get-latest-versions (**POST and GET**),
set-tag, delete-tag.

**Model versions:** create (resolves `models:/`/`runs:/` sources to `storage_location`,
MAX+1 versioning with retry, `sqlalchemy_store.py:981-1096`), update (PATCH),
transition-stage (canonical stage names; `archive_existing_versions` archives all other
versions in the target stage, only valid for Staging/Production,
`sqlalchemy_store.py:1192-1230`), delete (soft-delete → `Deleted_Internal` stage with
source/run_id redaction and alias removal, `:1244`), get, search (GET; **store default
10000 / threshold 200000** — proto says default 200000, the store wins), get-download-uri,
set-tag, delete-tag.

**Aliases:** `/mlflow/registered-models/alias` is **HTTP-method-overloaded**: POST=set,
DELETE=delete, GET=get-version-by-alias.

Entity quirks: `ModelVersion.version` is a string in proto but Integer in DB;
`latest_versions` only contains READY versions per stage; OSS
`SqlModelVersion.to_mlflow_entity` populates aliases separately; `user_id` not returned
on RegisteredModel.

**Prompts ride on the registry** (`mlflow/store/model_registry/abstract_store.py:517-1160`):
no separate OSS prompt endpoints. The Rust registry must support: arbitrary tags
(template text lives in `mlflow.prompt.text`); the **prompt-exclusion anti-join** —
default searches EXCLUDE rows tagged `mlflow.prompt.is_prompt='true'` unless the filter
explicitly queries that tag (`sqlalchemy_store.py:776-832`); and the special semantics
where `is_prompt != 'true'` / `= 'false'` matches rows lacking the tag entirely
(`search_utils.py:1304,1499`). Get that right and the Prompts UI works for free.

**copy_model_version** is client-side composition (create RM + create MV with
`models:/{src}/{ver}` source) — no extra endpoint, but `models:/` source resolution in
create must work.

### 3.15 Webhooks (6 endpoints + delivery engine)

Proto `mlflow/protos/webhooks.proto:14-116`, handlers `handlers.py:3367-3462`. REST-style
path params: `POST/GET /mlflow/webhooks`, `GET/PATCH/DELETE /mlflow/webhooks/{webhook_id}`,
`POST /mlflow/webhooks/{webhook_id}/test`.

Delivery engine (`mlflow/webhooks/delivery.py`): async in-process pool (fire-and-forget,
no durable queue), HMAC-SHA256 signature `v1,<b64>` over `"{delivery_id}.{timestamp}.{payload}"`
when a secret is set, headers `X-MLflow-Signature`/`X-MLflow-Timestamp`/`X-MLflow-Delivery-Id`,
HTTP retries on [429,500,502,503,504] with backoff, **SSRF protection** (public-IP
validation at connect time, no proxy env), TTL cache of webhooks-by-event. Events fired
from registry mutations (registered model created; model version created; MV tag
set/deleted; MV alias set/deleted; PROMPT_* mirrors) — trigger sites
`handlers.py:2638-3334`. Secrets stored Fernet-encrypted
(`MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`). Under auth: all webhook endpoints are
admin-only (`WEBHOOK_BEFORE_REQUEST_HANDLERS`, `auth/__init__.py:2772`).

### 3.16 Auth & RBAC (`--app-name basic-auth` equivalent)

Important: this repo's auth app is a full **RBAC system**, not upstream's per-resource
basic-auth. Legacy per-resource permission tables exist on disk but are dead at runtime —
model the RBAC design only (`mlflow/server/auth/`).

**Users API** (`/api/2.0/mlflow/users/*` + ajax, `auth/routes.py:3-34`): create,
create-ui (form + CSRF), get, current (returns `{user, is_basic_auth}`), list (with
roles), update-password (self-service requires `current_password`), update-admin,
delete (cannot delete self). Hand-rolled JSON shapes (not proto).

**Per-user permission APIs** (v3): current/permissions, permissions/list, grant, revoke,
get (`{allowed, permission}`). Grants write to a **synthetic `__user_<id>__` role** per
workspace — the single most non-obvious mechanism
(`auth/sqlalchemy_store.py:259-543`).

**Roles API** (`/api/3.0/mlflow/roles/*`): create/get/list/update/delete role;
add/remove/list/update role permissions; assign/unassign; list user-roles / role-users.
Role names with prefix `__user_` are rejected.

**Permission model** (`auth/permissions.py`): `READ < USE < EDIT < MANAGE` (+
`NO_PERMISSIONS`); resource types `experiment, registered_model, prompt, scorer,
gateway_secret, gateway_endpoint, gateway_model_definition, workspace`; workspace-scope
grants only USE (member) or MANAGE (workspace admin) with pattern `*`. Resolution: scan
user's roles in the resource's workspace, max-merge matching grants
(`resource_pattern IN ('*', resource_id)`; workspace-`*` grants fold in only when MANAGE),
then floor against `default_permission`, preserving `NO_PERMISSIONS` as the
workspace-boundary deny (`auth/__init__.py:556-1022`,
`auth/sqlalchemy_store.py:2010`).

**Enforcement** (before-request): skip unprotected routes → authenticate (HTTP Basic
against werkzeug-hashed passwords; pluggable `authorization_function`) → **admin bypass**
→ validator lookup (exact map from `BEFORE_REQUEST_HANDLERS` `auth/__init__.py:2480-2617`
+ regex matchers for parameterized trace/logged-model/webhook paths + artifact-proxy path
inspection; unknown `/mlflow/traces/` paths **fail closed**). Permission-per-endpoint
matrix documented in the validators map (runs inherit experiment perms; model versions
inherit registered-model perms; createModelVersion additionally requires READ on the
source run/model).

**After-request hooks** (`auth/__init__.py:3594-3650`): creator gets MANAGE on create;
**search/list responses are filtered to readable rows and re-fetched to fill
max_results** (`_role_based_read_predicate`, filter_search_experiments /
registered_models / model_versions / logged_models); grant cleanup/rename cascades on
delete/rename; workspace role seeding/cleanup on workspace create/delete.

**GraphQL auth middleware**: per-field READ checks + experiment-id narrowing for
`mlflowSearchRuns`, post-filter for `mlflowSearchModelVersions`
(`auth/__init__.py:4139-4262`).

**Signup UI**: server-rendered `/signup` HTML form + CSRF token posting to
`create-user-ui`; requires a static `MLFLOW_FLASK_SERVER_SECRET_KEY`. No login page —
HTTP Basic only. The React **admin console** (`src/admin/`) and **account page**
(`src/account/`) consume the users/roles/permissions endpoints above.

**Auth config** (`basic_auth.ini` / `MLFLOW_AUTH_CONFIG_PATH`): default_permission,
database_uri, read_database_uri (read-replica routing), admin bootstrap user
(`create_admin_user`), grant_default_workspace_access, cache sizes/TTLs,
authorization_function pluggability (Rust v1: basic-auth built-in; custom Python
functions are out of scope — D9).

### 3.17 Workspaces (5 endpoints + scoping layer)

Proto `service.proto:1051-1132`; handlers `handlers.py:1351-1496`, registered at
`/api/3.0/mlflow/workspaces...`: list, create (201), get, update (PATCH), delete (204,
`?mode=RESTRICT|CASCADE|SET_DEFAULT`). Disabled → 503 (`_disable_if_workspaces_disabled`).
Name rules: k8s-style regex, len 2-63, reserved names `{workspaces, api, ajax-api,
static-files}`; `default` is reserved/undeletable.

Scoping mechanics: request workspace from **`X-MLFLOW-WORKSPACE` header** (fallback
`default`); workspace-aware store variants add `WHERE workspace = ?` to every query and
prefix artifact locations with `workspaces/<name>/`
(`mlflow/store/tracking/sqlalchemy_workspace_store.py:62`,
`mlflow/store/model_registry/sqlalchemy_workspace_store.py`); `workspaces` table is
provider-pluggable via a separate store URI (`--workspace-store-uri`). Single-tenant mode
(workspaces disabled): pin `workspace='default'`, refuse startup if non-default
workspaces exist (`sqlalchemy_store.py:446-462`).

---

## 4. Wire-Compatibility Contract (must-match behaviors)

Source: `mlflow/utils/proto_json_utils.py`, `mlflow/exceptions.py`,
`mlflow/utils/search_utils.py`, `mlflow/server/auth/`.

1. **snake_case JSON** (`preserving_proto_field_name=True`). No camelCase.
2. **int64 as JSON numbers** — MLflow un-does Google's string encoding
   (`_mark_int64_fields`, `proto_json_utils.py:47`). Exception: int64 **map keys** stay
   strings.
3. **Pretty-printed output** (`indent=2`) — replicate (D4).
4. **Enums serialize by name** (`"FINISHED"`, `"READY"`).
5. **Unknown request fields are ignored** (`ParseDict(..., ignore_unknown_fields=True)`).
6. **Error format**: `{"error_code": "<ENUM_NAME>", "message": "..."}` with the exact
   status map in `ERROR_CODE_TO_HTTP_STATUS` (`exceptions.py:30`). Unknown endpoints →
   proto-style 404 (`_not_implemented`). Auth failures: 401 with
   `WWW-Authenticate: Basic realm="mlflow"`; authorization failures: 403.
7. **Pagination tokens** are opaque; currently base64(JSON `{"offset": N}`)
   (`search_utils.py:942-987`). Rust may switch to keyset tokens (opaque strings) since
   nginx routes each endpoint to exactly one backend (D3).
8. **Search filter DSL** — full grammar parity for runs, experiments, logged models,
   traces, registered models (`SearchModelUtils:1267` — name only, LIKE auto-wrapped
   `%x%` per proto doc), model versions (`SearchModelVersionUtils:1452` — name,
   version_number, run_id (+IN), source_path aliases).
9. **GET-with-body quirk**: several search endpoints accept POST and GET; GET handlers
   parse query args into proto messages (repeated fields via repeated query params).
10. **`HasField` semantics**: unset-vs-zero distinctions (deleteTraces
    `max_timestamp_millis`, run search `run_view_type`) — prost `Option<T>` covers this.
11. **Trace write ordering discipline**: sorted metadata/metric keys, sorted trace ids,
    bounded deadlock retry (2 retries, backoff) — Postgres correctness requirement
    (commit `4c5548c39`).
12. **Workspace plumbing**: `experiments.workspace` + unique `(workspace, name)`;
    registry tables have workspace-leading composite PKs; `X-MLFLOW-WORKSPACE` header is
    the wire contract.
13. **Werkzeug password hash format** (`method$salt$hash`, pbkdf2:sha256 or scrypt
    depending on pinned werkzeug — verify at implementation time): Rust must **verify**
    existing hashes and **generate** hashes Python can verify (shared auth DB).
14. **Registry constants**: store-side max_results defaults/thresholds win over proto
    declarations (RM 100/1000, MV 10000/200000).
15. **Prompt anti-join semantics** (§3.14) are part of the observable search contract.
16. **Webhook signature**: `v1,<base64(hmac_sha256(id.ts.payload))>` + the three
    `X-MLflow-*` headers; receivers already verify this format.
17. **Auth JSON endpoints are hand-rolled** (users/roles/permissions) — match the exact
    shapes in `auth/entities.py` (`Role.to_json:358`, `RolePermission.to_json:410`,
    `UserRoleAssignment.to_json:448`) and the handler dicts, not proto rules.

---

## 5. Storage & Database Strategy

### 5.1 Backend-store tables owned by the Rust server (read+write)

Tracking/tracing (`mlflow/store/tracking/dbmodels/models.py`): `experiments`,
`experiment_tags`, `runs`, `params`, `tags`, `metrics`, `latest_metrics`, `datasets`,
`inputs`, `input_tags`, `logged_models`, `logged_model_params`, `logged_model_tags`,
`logged_model_metrics`, `trace_info`, `trace_tags`, `trace_request_metadata`,
`trace_metrics`, `spans`, `span_metrics`, `assessments`, `entity_associations`.

Registry (`mlflow/store/model_registry/dbmodels/models.py`): `registered_models`,
`model_versions`, `registered_model_tags`, `model_version_tags`,
`registered_model_aliases` — all with **workspace-leading composite PKs**. Webhooks:
`webhooks` (Fernet-encrypted `secret`, soft-delete `deleted_timestamp`),
`webhook_events`. Workspaces: `workspaces` (PK `name`).

Semantics to replicate exactly:

- **Wide composite PKs as dedup**: `metrics` 6-col PK; `logged_model_metrics` 5-col PK.
- **`latest_metrics` maintenance**: Python does select-for-update + compare
  (`sqlalchemy_store.py:1366-1483`); Rust implements the atomic form (§5.2 Q5) with
  identical observable semantics.
- **`spans.duration_ns`** stored generated column (`models.py:2010`).
- **Dialect upserts**: `ON CONFLICT DO UPDATE` (sqlite/pg), `ON DUPLICATE KEY UPDATE`
  (mysql), per-row merge (mssql) (`sqlalchemy_store.py:9806`).
- **SQLite session config**: `PRAGMA foreign_keys=ON`, `busy_timeout=20000`,
  `case_sensitive_like=true` (`store/db/utils.py:154-157`).
- **Registry**: `model_versions.storage_location` is DB-only (resolved artifact path,
  distinct from proto `source`); MV soft-delete redacts source/run_id/run_link; version
  numbering is `MAX(version)+1` in a retry loop (contention→retry, not a sequence);
  stage transition archiving is transactional over sibling versions.

### 5.2 Known query inefficiencies to fix (the 100 GB story)

From `mlflow/store/tracking/sqlalchemy_store.py` and
`mlflow/store/model_registry/sqlalchemy_store.py`:

| # | Problem | Location | Fix in Rust |
|---|---|---|---|
| Q1 | OFFSET pagination everywhere (O(offset)/page) | `_search_runs` L2054/2067, `search_traces` L3812, `get_metric_history` L1506, `search_logged_models` L3429, registry searches (offset via page token, `model_registry/sqlalchemy_store.py:518-557`) | keyset/seek pagination behind opaque tokens |
| Q2 | `SELECT DISTINCT` over full run rows after N filter joins | tracking L2059 | EXISTS semi-joins → no fan-out, no DISTINCT |
| Q3 | One subquery JOIN per filter clause and per order-by clause | `_get_sqlalchemy_filter_clauses` L9151, `_get_orderby_clauses` L9194 | correlated EXISTS per predicate; single CTE for order-by keys |
| Q4 | Missing indexes: `runs(experiment_id, lifecycle_stage, start_time)`, `logged_models.experiment_id`, `inputs.source_id`, `model_versions.run_id`, `model_versions.current_stage`; two empty `Index()` decls (tracking models.py L832/L863) | schema | alembic migrations (Python-owned, §5.4) |
| Q5 | `latest_metrics` read-lock-compare-write holds FOR UPDATE locks | L1366-1483 | atomic `INSERT ... ON CONFLICT DO UPDATE ... WHERE excluded.(step,timestamp,value) > current` |
| Q6 | `log_batch` non-atomic (separate sessions per entity type) + redundant run lookup | L1240/L1322/L1241 | one transaction per log-batch |
| Q7 | Span attribute search via `content LIKE '%"attr"value%'` (full scan) | L9550 (TODO L9526) | indexed `span_attributes` table or DB JSON operators (Phase 13) |
| Q8 | Eager loading returns ALL metrics/params/tags per run per page | `_get_eager_run_query_options` L1056 | batched IN-queries + streamed serialization |
| Q9 | Prompt-exclusion anti-join added to **every** registry search | `model_registry/sqlalchemy_store.py:776` | keep semantics; consider partial index / `NOT EXISTS` form measured per dialect |
| Q10 | Auth after-request search filtering re-fetches pages in a loop to fill max_results | `auth/__init__.py:1586` (`_role_based_read_predicate` + refetch) | push readable-resource filter into the search query (semi-join against grants) when auth is native to the same process |

Rules: **wire-invisible query improvements land with the port** (Q1-Q3, Q5, Q6, Q10);
**schema changes** (Q4, Q7, Q9-index) go to Phase 13 via alembic.

### 5.3 Auth DB

Separate database (default `sqlite:///basic_auth.db`, `basic_auth.ini`), 4 live tables
(`mlflow/server/auth/db/models.py`): `users` (id, username unique, password_hash,
is_admin), `roles` (unique `(workspace, name)`), `role_permissions` (unique
`(role_id, resource_type, resource_pattern)`), `user_role_assignments` (unique
`(user_id, role_id)`). Version table **`alembic_version_auth`**, head `f1a2b3c4d5e6`.
Legacy per-resource permission tables are dead at runtime — Rust ignores them.
Read-replica routing supported (`read_database_uri`). Synthetic `__user_<id>__` roles
carry per-user grants (§3.16).

### 5.4 Schema/migration ownership

- Alembic remains the schema owner for **both** DBs; migrations live in
  `mlflow/store/db_migrations/` (head `b7e4c1a90f23`) and
  `mlflow/server/auth/db/migrations/` (head `f1a2b3c4d5e6`); run via Python.
- Rust embeds both expected head revisions, reads the version tables at startup, refuses
  to start on mismatch with a "run `mlflow db upgrade`" message. Rust never writes the
  version tables.
- New indexes/tables needed by Rust are contributed as normal alembic migrations so both
  servers stay on one lineage.

### 5.5 Connection/memory model

- One async pool per database (backend, auth, optional read replicas), replacing
  `workers × (pool_size + max_overflow)` Python connections.
- Streaming JSON serialization for large responses.
- In-process TTL caches with Python-equivalent semantics: resource→workspace cache,
  optional credential cache (HMAC-keyed, off by default), webhooks-by-event cache,
  workspace artifact-root cache.
- Target: idle RSS < 100 MB (Python baseline measured in Phase 14).

---

## 6. Compliance Testing Strategy

Everything needed already exists in the repo:

1. **`tests/tracking/test_rest_tracking.py`** (5,529 lines): live-server HTTP tests for
   experiments, runs, metrics, search, errors, traces, spans, assessments, OTLP, GraphQL,
   **and registry-over-HTTP** (model version source validation, lifecycle, GraphQL
   search, lines 1597-2525). Fixture boots the server via `ServerThread`/`_init_server`
   (`tests/tracking/integration_test_utils.py`) — parametrize to launch the Rust binary.
2. **Go-store precedent**: `_MLFLOW_GO_STORE_TESTING`
   (`mlflow/environment_variables.py:1074`) gates representational-difference assertions.
   Add `MLFLOW_RUST_STORE_TESTING` identically.
3. **`MLFLOW_TRACKING_URI=http://rust-server`** re-points the whole Python client stack
   (`mlflow/tracking/_tracking_service/utils.py:252`) — fluent/client suites become
   conformance suites (silence `tests/conftest.py:326` warning).
4. **Wire spec**: `tests/store/tracking/test_rest_store.py` + registry
   `tests/store/model_registry/test_rest_store.py` / `test_rest_store_webhooks.py` assert
   exact request payloads per endpoint.
5. **Registry behavior**: `tests/store/model_registry/test_sqlalchemy_store.py`
   (2,518 lines — CRUD, search, tags, aliases, stages, pagination) mirrored over HTTP;
   `test_sqlalchemy_workspace_store.py` for workspace variants.
6. **Auth**: `tests/server/auth/` — `auth_test_utils.py` launches a real server with an
   isolated `basic_auth.ini` (`MLFLOW_AUTH_CONFIG_PATH`) and provides
   `create_user`/`grant_role_permission`/`User` helpers; suites `test_auth.py`,
   `test_permissions.py`, `test_client_rbac.py`, `test_auth_workspace.py`,
   `test_sqlalchemy_store_rbac.py`. Point the launcher at the Rust binary.
7. **Workspaces**: `tests/server/test_workspace_endpoints.py`,
   `test_workspace_middleware.py`, `tests/store/workspace/*`,
   workspace store variants in tracking/registry test trees.
8. **DB matrix CI**: copy the `database` job in `.github/workflows/master.yml`
   (postgres/mysql/mssql/sqlite via `tests/db/compose.yml`), add `mlflow-rust` service.
9. **Artifacts**: `tests/tracking/test_mlflow_artifacts.py` (subprocess `mlflow server`
   fixture) against the Rust artifact proxy.
10. **Differential (golden) testing**: new harness replaying identical requests against
    Python and Rust and diffing normalized responses — catches drift assertions miss.

---

## 7. Work Breakdown

Legend: every task has **AC** (acceptance criteria) and **VER** (how to confirm).
Tick a box only when both hold. Suggested execution order is phase order; Phases 5-10
have internal independence (artifacts/GraphQL/registry can proceed in parallel once
Phase 2 lands; auth needs registry + tracking APIs to protect).

### Phase 0 — Decisions & foundations

- [x] **T0.1 Confirm scope freeze**: "everything except genai" per §1/§3, with the genai
      exception list (§2.2) approved.
      **AC:** endpoint tables in §3 marked approved; ambiguous items (queryTraceMetrics,
      correlation, ui-telemetry) have in/out decisions in §9.
      **VER:** review sign-off recorded in this file. *(Approved 2026-07-13 —
      implementation kicked off on this scope; queryTraceMetrics, correlation, and
      ui-telemetry are IN scope as listed in §3.6/§3.13.)*
- [x] **T0.2 Auth enforcement architecture** (D1): Rust enforces natively for Rust routes;
      Python keeps its auth app for genai routes; both share the auth DB. Confirm
      custom `authorization_function` plugins are out of scope for Rust v1 (D9).
      **AC:** decision + consequences documented (incl. `MLFLOW_FLASK_SERVER_SECRET_KEY`
      only needed on Python side; Rust needs its own CSRF secret for /signup).
      **VER:** §9 updated. *(D1/D9 already decided; consequences as documented in §9.)*
- [x] **T0.3 MSSQL support tier** (D2): full via `tiberius`, or postgres/mysql/sqlite v1
      with mssql fast-follow.
      **AC/VER:** decision in §9; CI matrix reflects it. *(Decided: v1 =
      sqlite/postgres/mysql via sqlx; MSSQL fast-follow via tiberius.)*
- [x] **T0.4 Crate stack** per §2.3 or record deviations; verify a Rust
      werkzeug-compatible password-hash implementation and a Fernet crate exist and are
      audited (both are hard blockers for auth/webhooks).
      **AC:** `rust/` workspace `Cargo.toml` lists deps; spike code verifies a werkzeug
      hash from a real `basic_auth.db` and decrypts a Fernet token from Python.
      **VER:** `rust/spikes/` proof tests green. *(Done 2026-07-13: werkzeug 3.1.8,
      default method scrypt N=32768/r=8/p=1 dklen=64; pbkdf2:sha256:1000000 also covered;
      RustCrypto `scrypt`+`pbkdf2`+`sha2` verify AND generate hashes accepted by Python
      both directions; `fernet` 0.2.2 round-trips with `cryptography` 46. Deviation:
      sqlx pinned 0.8.6 — 0.9 needs Rust ≥1.94, toolchain pinned 1.89.)*

### Phase 1 — Scaffolding & protocol layer

- [x] **T1.1 Cargo workspace** at `rust/`: `mlflow-server` (bin), `mlflow-proto`,
      `mlflow-store` (tracking store), `mlflow-registry` (registry store), `mlflow-auth`,
      `mlflow-search` (filter DSLs), `mlflow-artifacts`, `mlflow-webhooks`.
      **AC:** `cargo build --workspace` green; CI with clippy + rustfmt.
      **VER:** GitHub Actions run green. *(Done 2026-07-13: workspace + 8 crates,
      toolchain pinned 1.89.0, `.github/workflows/rust.yml` (fmt/clippy/build/test);
      local gates green — Actions run pending first push.)*
- [x] **T1.2 Proto codegen** from `service.proto`, `model_registry.proto`,
      `webhooks.proto`, `assessments.proto`, `databricks.proto`, `mlflow_artifacts.proto`
      + OTLP protos; extract `databricks.rpc` endpoint options at build time to generate
      the route table.
      **AC:** generated route table covers every §3 proto endpoint (method/path/version);
      snapshot test against a dump of Python `get_endpoints()`.
      **VER:** `rust/tools/route_parity.py` diff empty (modulo documented Python-only routes).
      *(Done 2026-07-13: protox+prost via extension-preserving descriptor pool
      (prost-reflect) — the `protox::compile()` convenience API silently drops the
      `mlflow.rpc` extension, so build.rs drives `protox::Compiler` directly. 186 raw →
      372 expanded routes; parity vs Python's 391 endpoints exact for all 372
      proto-backed; 19 allowlisted non-proto routes (15 genai out-of-scope, 4 in-scope
      hand-crafted: /graphql ×2, server-info ×2 — to be implemented in T6.1/T11.5).
      OTLP trace_service.proto vendored under `rust/crates/mlflow-proto/vendor/` (only
      compiled _pb2 ships in the opentelemetry-proto wheel). The §3.4 missing-leading-
      slash quirk (`/api/2.0mlflow/experiments/search-datasets`) reproduced + tested.)*
- [x] **T1.3 MLflow-compatible JSON codec** (§4 items 1-5) + deserializer with
      unknown-field tolerance and `HasField` awareness.
      **AC:** golden round-trips over Run, TraceInfoV3, SearchRuns, RegisteredModel,
      ModelVersion, Webhook messages byte-identical to Python `message_to_json`.
      **VER:** `rust/tests/json_golden.rs`; goldens generated by `rust/tools/gen_goldens.py`.
      *(Done 2026-07-13: hand-rolled walker over prost-reflect DynamicMessage +
      Python-json.dumps-parity formatter (ensure_ascii, indent=2, repr floats, field
      order = field number); 13 goldens byte-identical. Known deviations, documented in
      json.rs: (1) map keys emitted sorted — Python's own map order is
      process-nondeterministic, so byte-parity there is impossible for either side;
      (2) float last-digit dtoa tie-break on ~0.01% of bit patterns; (3) bare
      Infinity/NaN doubles untested vs real MLflow. Parse side = prost-reflect
      deserializer, unknown-field tolerant.)*
- [x] **T1.4 Error model** with the full `ErrorCode`→HTTP map + auth 401/403 forms.
      **AC:** table-driven test covers every code; `_not_implemented` 404 parity;
      `WWW-Authenticate` header on 401.
      **VER:** golden diff against Python for forced errors. *(Done 2026-07-13: new
      `mlflow-error` crate. Findings: error bodies use json.dumps DEFAULT separators
      (not indent=2) and conditionally carry `sqlstate`/`error_class` derived from
      error_code via mlflow/error_classification.py client tables; 21-entry status map,
      remaining 58 of 79 ErrorCode variants default to 500; `_not_implemented` = empty
      body, 404, text/html; auth 401/403 are plain-text responses, not
      MlflowException-shaped. 8 golden fixtures byte-identical; exhaustive
      variant-coverage test guards new proto enum values.)*
- [x] **T1.5 Server skeleton**: axum app, `/health`, `/version`, request logging,
      `MLFLOW_STATIC_PREFIX`, graceful shutdown, `/metrics`.
      **AC:** `curl /health` → `OK`; `/version` matches the targeted MLflow version.
      **VER:** integration test. *(Done 2026-07-13: lib+bin split (`build_app(config)`),
      /health + /version byte/content-type-matched to Flask (version parsed from
      mlflow/version.py at build time), /metrics via metrics-exporter-prometheus with
      http_requests_total + duration histogram, --static-prefix/MLFLOW_STATIC_PREFIX
      with Python's verbatim validation errors, TraceLayer logging, SIGINT/SIGTERM
      graceful shutdown; 16 tests incl. real-socket + manual curl verification.)*

### Phase 2 — Tracking/tracing storage layer

- [x] **T2.1 Schema model + startup verification** for the backend store: structs for all
      §5.1 tracking tables; verify `alembic_version` head on boot.
      **AC:** starts on a Python-migrated DB; refuses stale DB with "run `mlflow db upgrade`".
      **VER:** test with head and head-minus-one sqlite DBs. *(Done 2026-07-13: 22 tables
      in `mlflow-store::schema`; head `b7e4c1a90f23` verified against a real
      alembic-migrated fixture (`rust/tools/make_test_db.py` →
      tests/fixtures/tracking.db); stale-head + uninitialized-DB refusal with
      Python-matching wording; Rust never creates DB files.)*
- [x] **T2.2 Dialect abstraction** (sqlite/postgres/mysql via sqlx, + mssql per T0.3):
      upsert forms, LIKE/ILIKE case semantics, pagination SQL, SQLite PRAGMAs.
      **AC:** store suite (T2.4+) passes on all enabled dialects.
      **VER:** `tests/db/compose.yml`-based matrix locally + CI. *(Foundation landed
      2026-07-13: `Db` pool enum + `Dialect` (upsert per backend, BINARY LIKE for mysql,
      quoting, placeholders, capability flags), SQLAlchemy pool env-var mapping,
      SQLAlchemy-URI parser incl. +driver suffixes, SQLite PRAGMAs via after_connect.
      Live pg/mysql tests gated behind MLFLOW_RUST_TEST_{PG,MYSQL}_URI — box stays
      unticked until the T2.4+ suite runs on the full dialect matrix in CI.
      2026-07-14: all env-gated tests + both trace-write stress tests (1000
      iters) green on live dockerized Postgres 16 + MySQL 8; 4 dialect bugs
      found+fixed (commit 9f5e9f70d), incl. session-level ANSI_QUOTES on MySQL
      so hand-written SQL can use standard `"quoted"` identifiers on all
      dialects. 2026-07-17: new `mlflow-test-support` crate generalizes every
      `mlflow-store`/`mlflow-registry` integration test file's hand-rolled
      `TempDb` (sqlite fixture copy) into a dialect-dispatching helper — set
      `MLFLOW_RUST_TEST_DIALECT=postgres|mysql` + the existing
      MLFLOW_RUST_TEST_{PG,MYSQL}_URI and the *same* ~275 test bodies across 12
      files run against a live, already-migrated (`mlflow db upgrade`, Rust
      never migrates) Postgres/MySQL schema instead of a fresh sqlite copy,
      truncating + re-seeding the shared schema before each test
      (`--test-threads=1` required on the live dialects only). 3 corpus-replay
      files with their own private per-corpus fixture DBs
      (`search_runs_corpus.rs`, `sampling_corpus.rs`, `search_corpus.rs`) and
      `mlflow-server`'s HTTP-layer tests are intentionally out of scope (own
      fixtures / separate crate). Found+fixed 3 more dialect bugs: `status`/
      `present`/`one` read via `get_i64` instead of `get_int` on physical
      `Integer`/bare-literal columns (Postgres `INT4` widening,
      logged_models.rs/traces.rs), and `SELECT key, value FROM
      experiment_tags` missing the `"key"`/`"value"` quoting every other
      `key`-column reference already had (MySQL reserved word,
      search_experiments.rs). `rust/tests/db/compose.yml` (Postgres 16 + MySQL
      8) for local runs; CI job `dialect-matrix` in `.github/workflows/rust.yml`
      runs the full matrix via GitHub Actions service containers on every
      rust/**-touching PR + master push, alongside the existing fmt/clippy/
      build/test jobs.)*
- [x] **T2.3 Search DSL parser** (`mlflow-search`): runs, experiments, logged models,
      traces grammars from `mlflow/utils/search_utils.py` incl. aliases, quoting,
      comparator validation, order_by.
      **AC:** ported Python parser test corpus passes 1:1 incl. error classification.
      **VER:** `cargo test -p mlflow-search`; error-message parity in Phase 12.
      *(Done 2026-07-13: observable-behavior port of sqlparse 0.5.5 (lexer SQL_REGEX,
      grouping passes, embedded 801-entry keyword table) + all six Search*Utils domains
      incl. registered models + model versions. 1,816-case corpus (460 valid/1,356
      invalid) generated from the REAL Python parsers replays 1:1, plus ported
      test_search_utils.py cases. Two genuine Python bugs reproduced faithfully:
      the _join_in_comparison_tokens duplicate-token fall-through, and tags.x IS NULL
      on registry domains raising an uncaught ValueError (→ 500 not 400; flagged for
      the Phase 12 differential allowlist). Known gaps documented for Phase 12: exotic
      `;` statement-splitting, shlex unterminated-quote parity.)*
- [x] **T2.4 Store: experiments + runs + params/tags** (CRUD, lifecycle,
      `(workspace,name)` uniqueness, cascades, param immutability).
      **AC:** parity with `tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_{core,experiments,runs}.py`
      behaviors over HTTP.
      **VER:** Rust unit tests + Phase 12 suite. *(Store layer done 2026-07-13:
      `TrackingStore` in mlflow-store/src/store/, all methods workspace-scoped;
      validation.py caps/messages ported verbatim; runName↔tag sync both directions;
      deleted-experiment name conflict; 78 tests green. HTTP-parity re-check in
      Phase 12.)*
- [x] **T2.5 Store: metrics + latest_metrics**: atomic upsert (Q5), exact compare
      semantics on (step, timestamp, value), single-transaction `log_batch` (Q6),
      duplicate-metric idempotency.
      **AC:** concurrent-logging stress produces correct latest_metrics, no deadlocks on
      pg + mysql.
      **VER:** `rust/tests/stress_latest_metrics.rs` against dockerized pg + mysql.
      *(Done 2026-07-13: atomic upsert — sqlite/pg row-value `ON CONFLICT ... DO UPDATE
      ... WHERE (excluded.step,timestamp,value) > (...)`; mysql per-column `IF(<greater>)`
      expansion. NaN→(0.0,is_nan) / ±Inf→f64::MAX clamp per sanitize_metric_value;
      lexicographic (step,timestamp,value) tie-break on sanitized values. 200-writer
      stress PASSED on real Postgres 16 (migrated container); mysql variant written,
      env-gated (MLFLOW_RUST_TEST_MYSQL_URI), pending CI. Found+fixed pg INT4 widening
      gotcha via RowLike::get_int. 2026-07-14: mysql variant executed green on live
      MySQL 8 after the ANSI_QUOTES/`"key"` dialect fixes (commit 9f5e9f70d).)*
- [x] **T2.6 Store: search_runs**: EXISTS semi-joins (Q2/Q3), keyset pagination behind
      opaque tokens (Q1), NULLS LAST emulation parity, inline
      params/metrics/tags/inputs/outputs per page.
      **AC:** ordering + page boundaries identical to Python across dialects; postgres
      EXPLAIN shows index usage once Q4 indexes exist.
      **VER:** differential harness (T12.4) on seeded DB; EXPLAIN artifacts in PR.
      *(Done 2026-07-14: wired the pre-existing but never-compiled `search.rs`
      into the store; EXISTS semi-joins (Q2/Q3), keyset pagination behind
      opaque tokens (Q1/D3), batched inputs/outputs/params/metrics/tags eager
      loading (Q8), dataset-inputs-only in search results matching
      `RunInputs(dataset_inputs=...)`. Parity fixes: per-entity comparator
      validation (Python validates at filter-apply time, not parse time),
      SQLite `start_time DESC` NULLS placement (LAST, not FIRST), numeric
      attributes bind numerically for Postgres. Differential corpus
      (`rust/tools/gen_search_runs_corpus.py` → 31 cases, full pagination
      walks against the genuine Python SqlAlchemyStore) generated + replay
      test active; first runs caught two more real bugs, both fixed:
      dataset.context positional-bind order (0 rows on sqlite/mysql) and NULL
      numeric order keys decoded as 0.0 poisoning the keyset cursor
      (boundary-row duplication). 14 unit + corpus replay tests.
      Cross-dialect (pg/mysql) differential + EXPLAIN artifacts pending
      Phase 12/13.)*
- [x] **T2.7 Store: metric history** (get-history, bulk, bulk-interval sampling ported
      exactly from `handlers.py:2223`).
      **AC:** identical sampled point sets vs Python on dense histories.
      **VER:** differential test (>2500 points). *(Done 2026-07-13: verbatim port of the
      SqlAlchemyStore SQL-override path (sqlalchemy_store.py:1611) incl. f64
      interval-index truncation + forced endpoint + min/max union; 17-case/8,079-point
      corpus generated via the real Python store replays byte-identical. Handler-level
      caps/validation documented for Phase 3 (bulk = hand-rolled JSON, bulk-interval =
      proto).)*
- [x] **T2.8 Store: datasets/inputs/outputs**.
      **AC/VER:** parity via Phase 12 suite. *(Store layer done 2026-07-13: log_inputs
      (dataset dedup on (experiment_id,name,digest), input dedup, input_tags),
      log_outputs (RUN_OUTPUT→MODEL_OUTPUT edges in `inputs` table — NOT
      entity_associations), search_datasets (DISTINCT + LEFT JOIN context tag, cap
      1000), Run.inputs/outputs assembly. mlflow-store at 93 tests. HTTP parity in
      Phase 12.)*
- [x] **T2.9 Store: logged models** (CRUD, finalize state machine, search with
      dataset-scoped ordering + encoded token, tags, params).
      **AC/VER:** Phase 12 suite.
      *(Done 2026-07-14: `store/logged_models.rs` — CRUD, finalize (no
      state-machine guard, matches Python exactly), tags/params,
      search_logged_models with attribute/metric/param/tag filters (EXISTS
      semi-joins, not literal JOINs — Python's join-based filter can silently
      drop models from a page under pagination, verified against a live
      SqlAlchemyStore; deliberately not reproduced), dataset-scoped metric
      ordering via RANK() OVER (...), byte-for-byte
      SearchLoggedModelsPaginationToken port. Discovered the SqlAlchemyStore
      filter parser is NOT the sqlparse-grammar SearchLoggedModelsUtils ported
      in T2.3 (that one is FileStore's) — ported the actual
      `mlflow.utils.search_logged_model_utils.parse_filter_string` separately
      in mlflow-search, preserving its quirks (dotted attributes.<numeric-alias>
      skips alias resolution → 500; validate_op error always names string_ops).
      Followed the real Python default/cap
      (SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT=100, no enforced max) over this
      plan's earlier "default 50 max 50" note, which doesn't match the source.
      Workspace-scoped throughout. 28 integration + 17 parser tests.
      HTTP/proto-layer concerns deferred to Phase 12. Gap CLOSED 2026-07-14:
      `_log_model_metrics` fully ported (`log_model_metrics_tx`, wired into
      log_batch/log_metric) — per-call dedup on the full Metric tuple incl.
      model_id (one call can span multiple models), shared `_validate_metric`,
      NaN→0.0 (no is_nan column). Deviations: explicit workspace-scoped
      model-existence check replaces Python's untested FK-violation error
      path; write joins log_batch's single transaction per Q6. +11 tests.)*
- [x] **T2.10 Store: traces** (start_trace V3 with sorted-merge discipline, get/batch-get,
      search filters incl. span/assessment/run_id special cases, delete both modes, tags,
      entity_associations, deadlock retry).
      **AC:** parity with `test_sqlalchemy_store_traces.py` over HTTP; 1000-iteration
      parallel start_trace+log_spans on postgres with zero deadlock failures.
      **VER:** Phase 12 suite + `rust/tests/stress_trace_writes.rs`.
      *(Store layer done 2026-07-14: start_trace V3 (sorted-merge + 2x deadlock
      retry per commit 4c5548c39), get/batch-get, search_traces
      (span/assessment/run_id/tag/metadata filters via EXISTS semi-joins,
      order-by + `(timestamp_ms DESC, request_id ASC)` tiebreak, offset
      pagination = Python page contents), delete_traces (both modes +
      HasField Some(0)-vs-None), tag CRUD, link_traces_to_run (≤100,
      entity_associations; delete leaves associations orphaned — matches
      Python). 23 sqlite tests. Gaps: archive-backed delete (Phase 4),
      session-scoped assessments + LINKED_PROMPTS/issue filters (Phase 12).)*
- [x] **T2.11 Store: spans** (`log_spans` bulk upsert, trace time-range update,
      span_metrics, lazy content reads, `content=""` = cleared payload).
      **AC:** OTLP payload → both servers → identical `traces/get` output.
      **VER:** differential test.
      *(Store layer done 2026-07-14: log_spans bulk dialect upsert + atomic
      trace time-range update (skipped when finalized) + span_metrics +
      SPANS_LOCATION tag; lazy content reads (no `spans.content` on TraceInfo
      reads; `content=""` skipped); `duration_ns` read-only generated column.
      10 sqlite tests. OTLP→row translation deferred to the HTTP layer
      (Phase 3); token-usage/cost/session/resource-tag aggregation Phase 3/12.
      PG concurrency stress test executed 2026-07-14 on live Postgres 16 AND
      MySQL 8 at 1000 iterations: zero deadlock failures (T2.10 AC met). The
      run caught 4 dialect bugs, all fixed in commit 9f5e9f70d (MySQL `key`
      reserved word → ANSI_QUOTES + `"key"` sweep; pg ON CONFLICT ambiguous
      preview merge; pg json cast for `spans.dimension_attributes`).
      OTLP→traces/get differential deferred to Phase 4/12.)*
- [x] **T2.12 Store: assessments** (FieldMask update, overrides/valid,
      feedback/expectation/issue JSON encoding).
      **AC/VER:** parity with `test_sqlalchemy_store_assessments.py` via Phase 12.
      *(Store layer done 2026-07-14: create/get/update/delete_assessment with
      `a-<uuid4>` id gen, override/supersede (valid flip + un-invalidation on
      delete), feedback/expectation/issue JSON encoding matching `json.dumps`
      semantics, workspace-scoped trace joins. mlflow-store at 112 tests. HTTP
      FieldMask translation deferred to Phase 3/12 (`valid` has no store-layer
      setter).)*

### Phase 3 — Tracking HTTP API

- [x] **T3.1 Experiments endpoints** (§3.1) incl. POST+GET search.
      **AC:** experiment sections of `test_rest_tracking.py` pass against Rust.
      **VER:** Phase 12 runner `-k experiment`.
      *(Done 2026-07-14: 9 experiment endpoints wired on both `/api/2.0` +
      `/ajax-api/2.0` from the mlflow-proto route table. Shared HTTP foundation
      landed: `AppState`+`TrackingStore`, `proto_http` adapter (JSON/GET →
      proto → codec → response), `Workspace` extractor (`X-MLFLOW-WORKSPACE`,
      fallback `default`); to add an endpoint: implement the handler + one arm
      in `handler_for`. `search_experiments` was missing from the T2.4 store —
      added in `mlflow-store/src/store/search_experiments.rs` (EXISTS tag
      semi-joins, offset page tokens, [1,50000] max_results,
      unspecified-view_type→empty parity). Error-body parity for
      missing-param / RESOURCE_DOES_NOT_EXIST / RESOURCE_ALREADY_EXISTS /
      bad-max_results. 15 HTTP + 7 store tests. Gaps: handler-level
      type-coercion error messages and `_validate_storage_location_uri`
      deferred to a shared validation layer; cross-dialect + Python
      differential deferred to Phase 12.)*
- [x] **T3.2 Runs endpoints** (§3.2) incl. limits, param-length errors, view-type,
      deprecated `user_id`.
      **AC:** run sections pass; limit-violation error payloads byte-match.
      **VER:** Phase 12 runner `-k run` + golden diffs.
      *(Done 2026-07-15: all 14 endpoints in `mlflow-server/src/runs.rs`, wired
      via `handler_for` on both prefixes. One new store method
      (`record_logged_model` for legacy `runs/log-model`) — byte-parity
      `mlflow.log-model.history` tag needed insertion-ordered,
      json.dumps-compatible serialization (serde_json `preserve_order` enabled;
      proto codec unaffected, verified by goldens). Parity subtleties: search
      `max_results` handler check fires only when the field is present and
      byte-matches Python's `invalid_value` AssertionError message (distinct
      from the store threshold error); omitted `user_id`/`start_time` stored as
      `""`/`0` per proto2 defaults; `run_id or run_uuid` fallback where the
      proto has both; `RunInfo.to_proto` emit rules mirrored exactly.
      `_validate_batch_log_api_req` found to be a dead no-op in Python (counts
      parsed dict keys, not bytes — can never fire) and deliberately omitted
      with a doc comment. 22 HTTP tests + 3 store tests. Deferred:
      cross-dialect differential (Phase 12); `_disable_if_artifacts_only` mode
      (T11.1).)*
- [x] **T3.3 Metrics endpoints** (§3.3) incl. both ajax-only bulk routes with hand-rolled
      JSON shape.
      **AC:** UI charts render identically; metric suite sections pass.
      **VER:** Phase 12 runner `-k metric` + UI smoke (T11.6).
      *(Done 2026-07-15: `mlflow-server/src/metric_history.rs` — get-history +
      get-history-bulk-interval via the route table; ajax-only get-history-bulk
      hand-registered in `lib.rs` (not proto-backed). Hand-rolled JSON =
      Flask-jsonify parity verified against a live Flask app: sorted keys,
      compact separators, trailing newline, bare `NaN`/`Infinity` literals
      (serde_json can't emit those — body built by hand reusing the codec's
      exported `python_float_repr`/`quote_json_string`). Quirks reproduced:
      non-integer `max_results` on get-history-bulk hits an uncaught Python
      `ValueError` → generic HTML 500 (not JSON); bulk-interval int-coercion
      failures match `_assert_intlike`'s double-space message via raw-query
      pre-validation; get-history's `max_results` has NO validator in Python —
      non-numeric values are silently dropped, reproduced by scrubbing the
      field pre-parse. 25 HTTP tests. UI chart smoke deferred to T11.6.)*
- [x] **T3.4 Logged models + search-datasets endpoints** (§3.4, §3.5).
      **AC/VER:** Phase 12 runner `-k "logged_model or dataset"`.
      *(Done 2026-07-15: 8 logged-model endpoints
      (create/get/finalize PATCH/delete/search/set-tags PATCH/delete-tag/
      log-params) in `mlflow-server/src/logged_models.rs` + search-datasets in
      `datasets.rs`, incl. the missing-leading-slash quirk path from the route
      table AND the hand-registered correctly-slashed ajax route
      (`mlflow/server/__init__.py:135`). New reusable path-param
      infrastructure for later phases (traces/registry/webhooks):
      `to_axum_path` converts Flask `<param>` → axum `{param}`, and
      `proto_http::parse_request_with_path_params` overlays captured segments
      onto the parsed proto before validation (path wins over body — a
      documented, tested deviation from Flask's separate view args,
      behaviorally identical for real clients). `SetLoggedModelTags.Response`'s
      optional `model` field left unpopulated like Python. 20 HTTP tests.
      Deferred to Phase 5 (T5.1/T5.3): `listLoggedModelArtifacts` + ajax-only
      logged-model artifact file download (need the artifact-repo layer).)*
- [x] **T3.5 GET-request proto parsing** (repeated params, nested fields).
      **AC:** V2 trace search via GET and experiments/get round-trip correctly.
      **VER:** unit tests + suite.
      *(Done 2026-07-14: GET query→proto in `mlflow-proto::from_query_pairs` /
      `dynamic_from_query_pairs`, descriptor-driven — repeated fields via
      repeated query params (single occurrence still a list), bool true/false
      coercion with Python's verbatim error, scalars coerced by the codec
      deserializer (int64/enum-by-name). Wired through
      `proto_http::parse_request`. Unit tests + experiments/search GET with
      repeated order_by + experiments/get GET round-trips in the HTTP suite.
      V2 trace search GET re-check when T4.2 lands.)*

### Phase 4 — Tracing HTTP API

- [x] **T4.1 V3 trace endpoints** (§3.6).
      **AC:** trace sections of `test_rest_tracking.py` pass (start/search/delete/tags/
      link/correlation/metrics).
      **VER:** Phase 12 runner `-k trace`.
      *(Done 2026-07-15: all 13 endpoints in `mlflow-server/src/traces.rs` on
      both `/api/3.0` + `/ajax-api/3.0`. Store additions: `get_trace` (retry +
      `num_spans` completeness check, ARCHIVE_REPO → NOT_IMPLEMENTED per D6),
      `link_prompts_to_trace`, new `traces_analytics.rs`
      (calculate_trace_filter_correlation + query_trace_metrics reusing the
      trace-search filter machinery) and `trace_correlation.rs` (pure-math NPMI
      port, 6 unit tests). `proto_http` GET parsing fixed to Python's
      `GET and flask_request.args` (query args only when non-empty, else JSON
      body — the GET-with-body quirk, §4.9). getTrace assembles TraceInfoV3 +
      OTLP spans decoded from stored span-dict JSON. 22 HTTP tests. Deferred,
      documented in doc comments: queryTraceMetrics PERCENTILE +
      time_interval_seconds bucketing (→ INVALID_PARAMETER_VALUE; pagination
      matches Python's always-None token); assessments on TraceInfoV3
      responses start empty (Phase 12); OTLP span links not decoded;
      MySQL float casts use DECIMAL(38,10) — advanced-view dialect
      verification rides Phase 12.)*
- [x] **T4.2 V2 trace endpoints** (§3.7) as adapters.
      **AC:** UI contains-traces check works; V2 tests pass.
      **VER:** Phase 12 runner + UI smoke.
      *(Done 2026-07-15: all 7 endpoints in `mlflow-server/src/traces_v2.rs`;
      new store methods `deprecated_start_trace_v2`/`deprecated_end_trace_v2` —
      deliberately NOT reusing V3 `start_trace`, since V2 start never writes
      the `TRACE_INFO_FINALIZED` marker and generates its own `uuid4().hex`
      request id; endTrace is fully implemented in Python (execution_time from
      request_time, status, metadata/tag merge-upsert) and ported as such.
      `TraceInfoV2.to_proto` field truncation (250/250/4096) applied.
      `_assert_map_key_present` ported with Python's exact invalid_value
      message. UI contains-traces GET shape covered by test. 20 HTTP tests.
      Known gap, documented, shared with V3: `_validate_trace_tag_handler_
      mutation` (spansLocation/archiveLocation tag-mutation guard) not yet
      implemented for either version — kept symmetric, follow-up in Phase 12.)*
- [x] **T4.3 OTLP `/v1/traces`** (§3.8): protobuf + JSON, gzip, headers, all-or-nothing,
      200/400/422/501.
      **AC:** `test_rest_store_logs_spans_via_otel_endpoint` passes; Rust + Python OTel
      exporters both ingest.
      **VER:** Phase 12 runner + `rust/tests/otlp_ingest.rs`.
      *(Done 2026-07-15: hand-registered `/v1/traces` in `lib.rs`; new
      `mlflow-server/src/otlp/{mod,json,translate}.rs`. Handler reproduces
      `otel_api.py:95-259` status/body split exactly incl. FastAPI's compact
      `{"detail": ...}` 422 shape (verified against Starlette internals);
      gzip/deflate via flate2. `translate.rs` is the OTLP→row translation
      T2.11 deferred: fused `Span.from_otel_proto`/`to_dict`/
      `sanitize_attributes` (base64 big-endian ids, OTel status names in
      `content` vs plain names in the status column, ensure_ascii escaping,
      dimension_attributes, LLM-cost span_metrics, root-span service.name
      allowlist). Hand-rolled OTLP/JSON→prost decoder (hex ids,
      camel+snake case). 30 unit + 15 integration tests. Deferred, documented:
      the 11 vendor OTEL attribute-inference translators
      (OpenInference/Traceloop/…) — spans persist fine without vendor-inferred
      attributes; server-side telemetry recording (no Rust telemetry infra).)*
- [x] **T4.4 Assessments endpoints** (§3.9).
      **AC:** `test_assessments_end_to_end` passes.
      **VER:** Phase 12 runner `-k assessment`.
      *(Done 2026-07-15: 4 endpoints in `mlflow-server/src/assessments.rs`
      (create/get/update PATCH/delete) with path params, both prefixes.
      FieldMask JSON parsing came free from prost-reflect (canonical
      well-known-type mapping, verified). Shared codec gap fixed:
      `google.protobuf.Value`/`Struct`/`ListValue` support added to
      `mlflow-proto/src/json.rs` (byte-matched vs Python MessageToJson incl.
      int→`4.0` double widening, unset Value → null) — needed for
      Feedback/Expectation values, useful beyond assessments. Python quirks
      reproduced: `valid` mask path raises an uncaught TypeError in Python
      (update has no such kwarg) → 500, reproduced + flagged for the Phase 12
      differential allowlist; oneof auto-vivification defers type-mismatch
      errors to the store-side check. No store changes needed. 17 HTTP tests.)*
- [x] **T4.5 get-trace-artifact** (§3.10): DB spans + ARTIFACT_REPO fallback + attachments
      with path validation.
      **AC:** `test_get_trace_artifact_handler` passes; trace explorer renders both
      storage locations.
      **VER:** Phase 12 runner + UI smoke.
      *(Done 2026-07-15: `mlflow-server/src/trace_artifact.rs`, hand-registered
      on both ajax prefixes. Ports `_fetch_trace_data_from_store` + handler
      fallback exactly — NOT via the T4.1 `get_trace` store method, whose
      getTrace-endpoint semantics return empty spans where this handler must
      fall through to the artifact repo. TRACKING_STORE → `{"spans": [...]}`
      from stored span-dict JSON (compact json.dumps separators);
      other/untagged → artifact repo `traces.json` via
      `mlflow_artifacts::repo_from_uri` (local FS/`file://`; cloud schemes
      NOT_IMPLEMENTED until Phase 5); attachments via `attachments/{uuid}`
      with `_validate_attachment_path` + `validate_path_is_safe` (traversal →
      400 exact body); ARCHIVE_REPO → 501 per D6. Response headers mirror
      `_response_with_file_attachment_headers`. 11 HTTP + 3 unit tests.
      Deviations documented: not-found messages use repo-relative paths (not
      Python's per-call temp-dir absolute paths); third-party OTEL spanType
      backfill (`translate_loaded_span`) not ported.)*

### Phase 5 — Artifacts

- [x] **T5.1 `/get-artifact` streaming download** with artifact-URI resolution and path
      safety.
      **AC:** artifact browser works; traversal attempts → 400.
      **VER:** artifact suite sections + explicit traversal tests.
      *(Done 2026-07-15: root-mounted `/get-artifact` in
      `mlflow-server/src/artifacts.rs`; run artifact-URI resolution incl.
      `mlflow-artifacts://` + `http(s)://…/mlflow-artifacts/artifacts/`
      proxied forms (`AppState::resolve_artifact` ports
      `_is_servable_proxied_run_artifact_root` + destination-path logic);
      streamed via object_store `into_stream` (no buffering);
      traversal → 400. Deviation: missing-`path` → JSON 400 (Python:
      Flask KeyError HTML 400). Workspace prefixing elided at the
      `resolve_artifact` seam for Phase 10.)*
- [x] **T5.2 `mlflow-artifacts` proxy** (§3.11) over `object_store`: full surface,
      streamed both directions (Python's WSGI bridge buffers whole bodies —
      `fastapi_app.py:41` — Rust must not).
      **AC:** `tests/tracking/test_mlflow_artifacts.py` passes; 5 GB upload keeps Rust
      RSS growth < 100 MB.
      **VER:** Phase 12 runner + memory probe.
      *(Done 2026-07-15: all 8 endpoints on both prefixes via a new
      `MlflowArtifactsService` branch in `handler_for` (previously
      early-returned for non-MlflowService); `to_axum_path` extended for
      Flask `<path:…>` → axum `{*…}` wildcards. Uploads stream chunk-by-chunk
      into object_store multipart put; downloads stream out; 256 MiB
      RSS-probe test (<128 MiB growth) — the 5 GB probe is Phase 12.
      `--serve-artifacts` (default true) + `--artifacts-destination` added to
      Cli/ServerConfig; disabled mode returns Python's exact 503. Local
      FS/`file:` backend; cloud schemes + multipart/presigned →
      NOT_IMPLEMENTED (parity with LocalArtifactRepository, which lacks those
      mixins). Real bug fixed: `local_repo` now create_dir_all's the root
      (run artifact dirs exist only on first write; uploads 500'd before).
      Python-suite run (test_mlflow_artifacts.py) rides Phase 12.)*
- [x] **T5.3 ajax `upload-artifact` + logged-model artifact routes.**
      **AC/VER:** UI artifact upload/download smoke + suite sections.
      *(Done 2026-07-15: ajax `upload-artifact` (buffered with Python's 10 MB
      cap — parity, not a streaming path); `listLoggedModelArtifacts`
      (proto route `…/artifacts/directories`) + ajax logged-model file
      download (`…/artifacts/files`) — closes the T3.4 deferral. 13 HTTP
      tests + 5 state unit tests across T5.1-T5.3. UI smoke rides T11.6.)*
- [x] **T5.4 `/model-versions/get-artifact`** (§3.11): registry-store URI resolution
      (`storage_location or source`), proxied-artifact handling, workspace prefixes.
      **AC:** model artifact downloads work for `models:/`-sourced and directly-sourced
      versions.
      **VER:** registry artifact tests + UI smoke on model page.
      *(Done 2026-07-15: `get_model_version_artifact` in
      `mlflow-server/src/artifacts.rs`, root route only (matches
      `__init__.py:117`). Resolution via
      `RegistryStore::get_model_version_download_uri` through the T5.1
      `resolve_artifact` seam; `models:/`-sourced versions work because the
      store resolves them to storage_location at create time. AppState gained
      `Option<RegistryStore>` (`with_registry` constructor); main.rs wires it
      from the same Db pool. Python quirk reproduced: missing `version` →
      `int(None)` TypeError escapes `_validate_model_version`'s ValueError
      catch → 500 INTERNAL_ERROR with verbatim message (present-but-invalid
      version → 400). 10 HTTP tests. Workspace prefixing stays the Phase 10
      seam.)*

### Phase 6 — GraphQL

- [x] **T6.1 `/graphql` endpoint** implementing all §3.12 operations against the Rust
      stores (registry included); replicate no-batching guard and error shapes.
      **AC:** run detail page + chart polling work against Rust; GraphQL depth tests in
      `test_rest_tracking.py` pass; `mlflowSearchModelVersions` returns parity results.
      **VER:** Phase 12 runner `-k graphql` + UI smoke of run page.
      *(Done 2026-07-17: `mlflow-server/src/graphql/` (graphql-parser 0.4,
      hand-rolled executor). Query: mlflowGetExperiment/GetRun (+run
      extension experiment/modelVersions)/GetMetricHistoryBulkInterval/
      ListArtifacts/SearchModelVersions; Mutation: mlflowSearchRuns/
      SearchDatasets; test/testMutation. Quirks ported: in-band bare-string
      errors + null root field (handlers.py:3682), apiError always null,
      LongString → quoted int, NaN metric → null, no-batching guard verbatim
      (depth 10 / 1000 selections / env-var-named messages), GET parses JSON
      body. Hardened vs Python (documented): array body / missing query →
      clean in-band error instead of 500. 24 HTTP + 5 unit tests.)*
- [x] **T6.2 GraphQL schema parity check**: generate from the same
      `autogenerated_schema.gql`; add schema-diff test (D8).
      **AC:** schema diff empty for implemented fields.
      **VER:** CI schema-diff job.
      *(Done 2026-07-17: embedded SDL asserted byte-identical to
      `mlflow/server/js/src/graphql/autogenerated_schema.gql` (the only
      location of the .gql) + every implemented root field checked present
      in the SDL. Runs in `graphql_http.rs`.)*

### Phase 7 — Model Registry

- [x] **T7.1 Registry schema + store core**: 5 tables with workspace-leading composite
      PKs; registered model CRUD incl. rename (cascades via FK onupdate), tag CRUD,
      alias CRUD; `storage_location` handling.
      **AC:** behaviors match `tests/store/model_registry/test_sqlalchemy_store.py`
      (mirrored over HTTP in Phase 12).
      **VER:** Rust unit tests + Phase 12 registry sections.
      *(Done 2026-07-15: `mlflow-registry/src/store/` — much of the groundwork
      (schema/entities/stages/validation/dbutil, RM CRUD, rename cascade,
      get_latest_versions via ROW_NUMBER, tag/alias CRUD) predated this task
      and was verified against the Python spec rather than rewritten.
      Workspace-scoped throughout; `user_id` never returned on RegisteredModel.
      Registry crate at 50 tests incl. 2 env-gated live pg/mysql smokes.)*
- [x] **T7.2 Model version lifecycle**: create with `models:/`/`runs:/` source resolution
      + `MAX(version)+1` retry loop; update; **transition-stage** with canonical stage
      names + archive_existing_versions; **soft-delete** with redaction + alias removal;
      get-download-uri.
      **AC:** stage-transition and soft-delete edge cases from the store suite pass;
      copy-model-version flow (client-side) works end-to-end.
      **VER:** Phase 12 suite + an explicit copy_model_version client test.
      *(Done 2026-07-15: `store/model_versions.rs` — create with
      `models:/name/version` → storage_location resolution (via download-uri
      path); `runs:/` and bare `models:/<id>` stored verbatim, matching the
      Python registry store (the logged-model-id lookup is a cross-store
      MlflowClient call — caller/HTTP-layer responsibility, documented);
      MAX+1 retry; update; transition-stage (case-insensitive canonical
      stages, archive_existing_versions transactional over siblings,
      Python's `['Staging', 'Production']` list-repr error); soft-delete →
      `Deleted_Internal` with redaction sentinels + alias removal, tags kept;
      `get_model_version_including_deleted` mirror of Python's test helper.
      copy_model_version end-to-end + Phase 12 HTTP checks ride T7.4/T12.)*
- [x] **T7.3 Registry search**: `search_registered_models` / `search_model_versions` with
      the DSL (§4.8), AND-of-tags HAVING-count subquery, **prompt-exclusion anti-join**
      + `_is_querying_prompt` bypass, order_by defaults and tiebreakers, offset-token
      contract (`limit(N+1)`), latest-versions via ROW_NUMBER window batch.
      **AC:** search parity incl. prompt in/exclusion and the "`is_prompt != 'true'`
      matches untagged rows" semantics; `get-latest-versions` returns READY-only per
      stage.
      **VER:** Phase 12 suite + differential harness with a corpus mixing models and
      prompts.
      *(Done 2026-07-15: `mlflow-registry/src/store/search.rs` using the T2.3
      parsers. AND-of-tags HAVING-count port; prompt anti-join with the
      untagged-row semantics (MV's anti-join reads model_version_tags grouped
      by (workspace,name) — a prompt-tagged RM does NOT hide its versions);
      offset tokens byte-match Python; deleted-MV exclusion; MV search omits
      aliases; store thresholds RM 1000/MV 200000 (defaults are handler-level
      → T7.4). Confirmed the store does NOT auto-wrap `name = 'x'` into
      `%x%` (client-side per proto doc). Parser gap fixed additively:
      the store's RM order-by uses `parse_order_by_for_search_registered_
      models` key set (timestamp, not creation_timestamp) — new
      `registered_models_order_by_store` in mlflow-search. Differential
      corpus (`rust/tools/gen_registry_search_corpus.py`, 40 cases via the
      genuine Python registry SqlAlchemyStore, full page walks + tokens)
      replays clean; bring-up caught 2 real bugs (outer-alias reference in
      subquery; positional-bind ordering on sqlite/mysql), both fixed.
      13 behavioral + 2 corpus tests + live pg/mysql smoke (env-gated).)*
- [x] **T7.4 Registry REST endpoints** (§3.14, 21 endpoints) incl. the method-overloaded
      alias route and GET+POST get-latest-versions; store-side max_results limits
      (RM 100/1000, MV 10000/200000).
      **AC:** registry sections of `test_rest_tracking.py` (model version source
      validation tests, lines 1597-2018) pass against Rust.
      **VER:** Phase 12 runner `-k "registered_model or model_version"`.
      *(Done 2026-07-15: all 21 in `mlflow-server/src/registry.rs` via a
      `ModelRegistryService` branch in `handler_for`. Method-overloaded alias
      route falls out naturally: axum 0.8 merges MethodRouters for a repeated
      path with disjoint methods, one route-table entry each. Full port of
      createModelVersion source validation
      (`source_validation.rs`: relative-path/encoded-traversal/%00 rejection,
      run/model source checks, validation-regex gate; std-only URI/path
      helpers) with byte-matched errors + traversal negative tests.
      MV search max_results: raw-request presence check distinguishes omitted
      (→ store default 10000) from the proto's declared default 200000.
      Alias-not-found is 400 INVALID_PARAMETER_VALUE (matches
      sqlalchemy_store.py:1592). `// WEBHOOK SEAM:` markers left at every
      Python trigger site for T8.4. Deferred: the `model_id` back-link tag
      Python writes via the tracking store after MV create (cross-store
      boundary, same as T7.2's logged-model-id note). 21 HTTP + 7 unit tests.)*
- [x] **T7.5 Prompts-on-registry validation**: the Prompts UI (list/create/version/alias
      prompt) works against the Rust registry unchanged.
      **AC:** UI smoke: prompts pages function; models pages never show prompts.
      **VER:** T11.6 checklist items.
      *(Done 2026-07-17: prompts share the RegisteredModel/ModelVersion REST
      surface unchanged (tagged `mlflow.prompt.is_prompt='true'`); T7.3 already
      ported the search-time anti-join that hides prompts from default model
      listings, and T7.4 already ported the handler-level prompt branches
      (source-validation prompt path, webhook seams). The one real gap was
      `registered_models.rs`'s `create_registered_model` collision path, which
      only had the plain "already exists" message — ported Python's
      `handle_resource_already_exist_error` (`mlflow/prompt/registry_utils.py:
      264-290`) so a name collision between a model and a prompt gets the
      cross-type message (with `repr()`-correct quoting) instead of a generic
      one. Added `registry_http.rs` coverage driving the Prompts UI's exact
      REST call shapes end-to-end (create prompt, create prompt version via
      the "dummy-source" placeholder, tag set/delete, alias set, the
      `tags.\`mlflow.prompt.is_prompt\` = 'true'` search filter) plus both
      collision directions and a default-search prompt-exclusion check. 3 new
      HTTP tests (24 total in the file); one `registry_store.rs` assertion
      updated for the message's new trailing period.)*

### Phase 8 — Webhooks

- [x] **T8.1 Webhook storage**: `webhooks` + `webhook_events` tables, Fernet-encrypted
      secrets (`MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`), soft delete, workspace scoping.
      **AC:** secrets written by Python decrypt in Rust and vice versa.
      **VER:** cross-language crypto test in `rust/tests/`.
      *(Done 2026-07-15: `mlflow-webhooks` crate — entities, Fernet
      `SecretCipher` mirroring the `EncryptedString` TypeDecorator,
      validation (URL scheme/hostname + non-public-IP resolve gate, event
      entity/action combos, byte-matched messages), `WebhookStore`
      (workspace-scoped CRUD, soft delete, offset tokens,
      list_webhooks_by_event), HMAC `v1` signer in `signing.rs` for T8.3
      reuse, per-event example payloads. Cross-language fixtures generated by
      Python `cryptography`/`hmac` (committed + regen script): Python→Rust
      Fernet decrypt, wire-format check, HMAC byte-match. Plan-text
      correction: Python has NO missing-key error — `EncryptedString` falls
      back to an ephemeral `Fernet.generate_key()`; reproduced faithfully.
      26 crate tests.)*
- [x] **T8.2 Webhook REST endpoints** (§3.15, 6 endpoints incl. `/test` with
      `WebhookTestResult`).
      **AC:** `tests/store/model_registry/test_rest_store_webhooks.py` payload parity;
      `tests/tracking/test_client_webhooks.py` passes over HTTP.
      **VER:** Phase 12 runner `-k webhook`.
      *(Done 2026-07-15: `mlflow-server/src/webhooks.rs` via a new
      `WebhookService` branch in `handler_for` + path-param overlay for
      `{webhook_id}`. `/test` fires a real HTTP POST (example payload, three
      `X-MLflow-*` headers, `v1,<b64>` signature) with the resolve-time
      SSRF gate; secrets never returned (proto has no secret field).
      `AppState::with_webhook_store` additive builder; main.rs shares the
      tracking Db pool. 6 HTTP tests incl. a local receiver recomputing the
      signature. Deferred to T8.3 as scoped: async delivery engine, retries,
      TTL cache, connect-time SSRF adapter. Python-suite parity runs ride
      Phase 12.)*
- [x] **T8.3 Delivery engine**: async task pool, HMAC `v1` signing + `X-MLflow-*`
      headers, HTTP retries on [429,5xx] with backoff, **SSRF guard** (public-IP
      validation at connect, no proxy env), TTL cache by event, fire-and-forget error
      logging.
      **AC:** a local receiver verifies signatures Python receivers accept; SSRF suite
      (RFC1918/link-local/redirect tricks) blocked identically to Python.
      **VER:** `rust/tests/webhook_delivery.rs` incl. SSRF matrix.
      *(Done 2026-07-17: `mlflow-webhooks/src/{dispatcher,http_send}.rs`.
      Retry/backoff byte-matched to `mlflow/webhooks/delivery.py` defaults:
      statuses [429,500,502,503,504], total=3, factor=1.0 max=60 jitter=1.0,
      Retry-After honored, 30s per-attempt timeout. SSRF: own resolver →
      all resolved IPs must be global → TCP connect to the validated IP
      (closes TOCTOU) → `getpeername` re-check, gate re-run per redirect hop
      (cap 30), `trust_env=False` equivalent, `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS`
      escape hatch. `WebhookDispatcher::fire(event, data)` fire-and-forget
      (TTL cache maxsize=1000, Semaphore(10), tokio::spawn per send) exposed
      via `AppState::webhook_dispatcher()` for T8.4's 12 seam sites. Quirks:
      TTL-only cache invalidation; ISO8601 `+00:00` not `Z`; `/test` has
      timeout but no retries. Deferred (doc-commented): HTTPS delivery fails
      closed pending TLS stack; at-most-once per D11. 21 unit + 11
      integration tests incl. 11-target SSRF matrix.)*
- [x] **T8.4 Event triggers** wired into registry mutations (RM created; MV created; MV
      tag set/deleted; MV alias set/deleted; PROMPT_* mirrors by is_prompt
      classification).
      **AC:** trigger matrix test: each mutation fires exactly the events Python fires
      (entity/action pairs from `webhooks.proto` enums).
      **VER:** differential trigger-capture test with a recording receiver.
      *(Done 2026-07-17: all 12 seam sites wired post-commit in `registry.rs`
      + payload builders in `registry/webhook_events.rs`. Matrix matched to
      handlers.py:2636-3341: prompt events fire INSTEAD OF model events;
      RM tag set/delete fires nothing for non-prompts (Python's
      deliver_webhook is inside `if _is_prompt` with no else). Classification
      parity: create paths use presence-only request-tag check
      (`_is_prompt_request`), tag/alias paths re-query the stored tag
      (`_is_prompt`, value.lower()=='true'). Payloads byte-matched to
      webhooks/types.py TypedDict order incl. proto2 `""` description on RM
      create vs `or None` on MV create; prompt_version pops
      mlflow.prompt.text as `template` and omits source/run_id. fire_event
      no-ops without a dispatcher; lookup failure degrades to not-a-prompt
      (best-effort, documented). 9 unit + 4 HTTP trigger-matrix tests with a
      recording receiver.)*

### Phase 9 — Auth & RBAC

- [x] **T9.1 Auth DB layer**: 4 RBAC tables, `alembic_version_auth` head check
      (`f1a2b3c4d5e6`), werkzeug-compatible hash verify/generate, read-replica routing,
      admin bootstrap (`create_admin_user` + default-password warning).
      **AC:** Rust authenticates users created by Python and vice versa on a shared
      `basic_auth.db`.
      **VER:** cross-language auth-DB test.
      *(Done 2026-07-17: `mlflow-auth` crate fleshed out — schema/entities for
      users/roles/role_permissions/user_role_assignments (legacy per-resource
      tables ignored per §5.3), `AuthDb` with write pool + optional read
      replica (sqlalchemy_store.py:111-134 parity incl. same-URI warning),
      head `f1a2b3c4d5e6`/`alembic_version_auth` refusal with Python wording.
      `hash.rs` byte-matched to werkzeug 3.1.8: verifies scrypt+pbkdf2
      (constant-time via subtle), generates scrypt:32768:8:1 dklen=64 with
      62-char SALT_CHARS salt; RFC 7914 vector + live-Python cross-checks
      both directions. Bootstrap: admin/password1234 + verbatim warning
      (auth/__init__.py:3694). Alembic-migrated fixture via
      `rust/tools/make_auth_test_db.py` (scrypt + pbkdf2 users, seeded RBAC)
      committed with .gitignore exception. Quirks: authenticate→false not
      error; >12-chars password rule by code points; sqlite 0/1 bool
      tolerance; MySQL last_insert_id fallback. 9 unit + 5 cross-language
      tests.)*
- [x] **T9.2 Users API** (§3.16): all 8 endpoints incl. self-service password rules and
      cannot-delete-self; hand-rolled JSON shapes.
      **AC:** `tests/server/auth/test_client.py` user sections pass against Rust.
      **VER:** Phase 12 auth runner.
      *(Done 2026-07-17: `mlflow-server/src/auth_api/{mod,users}.rs` — all 8
      endpoints under /api/2.0 + /ajax-api/2.0, mounted only when
      `MLFLOW_AUTH_CONFIG_PATH` is set (absent → 404, matching the Python
      auth app). Hand-rolled shapes: `{"user":{id,username,is_admin}}`, list
      includes role objects, `/users/current` adds `is_basic_auth: true`.
      Handler-level checks ported (self-service current_password rules,
      cannot-delete-self, >12-code-point password, duplicate → 400 not 409
      per ERROR_CODE_TO_HTTP_STATUS, first-colon Basic-cred split like
      werkzeug); authorization gating deferred to T9.4 behind
      `// AUTH SEAM (T9.4):` markers; ini parsing/bootstrap behind
      `// T9.8 SEAM:`. `AuthStore::delete_user` cascades the synthetic
      user-role rows (sqlalchemy_store.py:222-241). 28 HTTP + 1 store tests.
      Deferred: create-ui CSRF/HTML → T9.7; list_users role-visibility
      narrowing → T9.3/T9.4.)*
- [x] **T9.3 Roles + permissions APIs**: role CRUD, role-permission CRUD, assignments,
      per-user grant/revoke/get via **synthetic `__user_<id>__` roles** (SAVEPOINT-safe
      get-or-create, `__user_` prefix rejection), scorer pattern key encoding.
      **AC:** `test_client_rbac.py` + `test_sqlalchemy_store_rbac.py` behaviors pass over
      HTTP.
      **VER:** Phase 12 auth runner.
      *(Done 2026-07-17: `mlflow-auth/src/{permissions,roles,user_grants}.rs`
      + `mlflow-server/src/auth_api/roles.rs` — 15 endpoints on /api/3.0 +
      /ajax-api/3.0 (auth/__init__.py:3001-4058). Python's begin_nested()
      SAVEPOINT get-or-create reproduced as `INSERT .. ON CONFLICT DO
      NOTHING` + re-select via Dialect::upsert (12-task race test: exactly
      one role+assignment); other-assignee collision defense
      (sqlalchemy_store.py:315-337); `__user_` whole-prefix rejection. Scorer
      pattern `<exp_id>/<url_quote(name, safe='')>` round-trips. Permission
      enums READ<USE<EDIT<MANAGE + NO_PERMISSIONS, 8 resource types, tuple-
      repr error messages byte-matched incl. single-element trailing comma.
      Quirks: 400 not 409 on exists; unknown resource → deny-by-default
      allowed=false. Role routes merged post-with_state as a self-contained
      Router<AuthStore> (orchestrator wired into build; caller-scope
      filtering of list_user_roles rides T9.4). 37 store + 17 HTTP + 8 unit
      tests.)*
- [x] **T9.4 Permission resolution + enforcement middleware**: tower layer implementing
      authenticate → admin bypass → validator dispatch (exact-path map + regex matchers
      for trace/logged-model/webhook paths + artifact-proxy path inspection with
      experiment-id extraction incl. `workspaces/<ws>/` prefixes); fail-closed on unknown
      `/mlflow/traces/` paths; permission matrix per §3.16 (runs/logged-models inherit
      experiment; MV create requires model UPDATE + source READ; webhooks admin-only;
      OTLP requires experiment UPDATE from header).
      **AC:** `tests/server/auth/test_auth.py` + `test_permissions.py` pass against Rust
      (server launched with `MLFLOW_AUTH_CONFIG_PATH`).
      **VER:** Phase 12 auth runner; fail-closed paths covered by explicit tests.
      *(Done: `mlflow-server/src/auth_middleware/{mod,validators,path_matchers}.rs` — the
      tower layer applied at the top of the app router when auth is enabled. Dispatch is
      driven from the proto `ROUTE_TABLE` keyed on `(service, method)` [mirroring Python's
      `BEFORE_REQUEST_HANDLERS`] plus the hand-registered auth/artifact/OTLP routes; four
      groups [logged-model / webhook / exact / trace-parameterized] consulted in Python's
      `_find_validator` order with fail-closed on unknown `/mlflow/traces/` subpaths.
      Permission resolution reuses T9.3's `get_role_permission_for_resource` folded against
      `default_permission` (env `MLFLOW_AUTH_DEFAULT_PERMISSION`, default READ — `// T9.8
      SEAM:`); workspace resolved to `"default"` pre-T10.4. Body-buffering for JSON-reading
      validators [MV create, start/search/batch traces, search datasets]. `// T9.5 SEAM:`
      left for after-request hooks. 25 HTTP tests [`auth_middleware_http.rs` +
      `auth_middleware_no_default_http.rs`] + 10 unit tests; the lattice-only cases of
      `test_permissions.py` are covered by `mlflow-auth`'s `permissions` unit tests.
      Python-suite runs ride Phase 12. Merge reconciliation (orchestrator): /signup
      re-registered via `.route()` so the auth-layered fallback survives (a `.merge()`
      clobbered it → unmatched traces paths 404'd instead of the fail-closed 403), the
      layer re-applied to /signup directly + `(SIGNUP, GET)→validate_can_create_user`
      added (auth/__init__.py:2649), and the T9.7/T11.5 tests moved to Python's
      authenticated flow — `_before_request` gates /signup, create-user-ui, and
      server-info too (the `:3638` server-info exemption is after-request-only).
      Closes parity-backlog item #2 from T12.1.)*
- [x] **T9.5 After-request hooks**: creator-MANAGE grants on create; search/list response
      filtering for experiments/registered-models/model-versions/logged-models — prefer
      the query-integrated form (Q10) with a flag-gated fallback to Python-identical
      refetch behavior; grant cascade on delete/rename; workspace role seed/cleanup.
      **AC:** a non-admin user sees exactly the same filtered search results from Rust
      and Python on a seeded permission fixture, including page-fill behavior.
      **VER:** differential test with multi-user fixtures.
      *(Done 2026-07-17: `auth_middleware/after_request.rs` dispatched by a new
      `dispatch_after_request` (mirrors AFTER_REQUEST_HANDLERS). Creator MANAGE
      grants (createExperiment; createRegisteredModel with prompt-vs-model
      namespace from the response is_prompt tag). Search filtering
      (experiments/RM/MV/logged-models) via `_role_based_read_predicate`
      (list_role_grants_for_user_in_workspace); admins skip. **Default =
      Python-identical refetch** (exact truncate-then-token math incl. the
      logged-models opaque token and Python's `if next_page_token:`-only
      overwrite quirk; MV is drop-only, no page-fill). Q10 query-integrated
      form NOT implemented — documented env seam
      `MLFLOW_RUST_AUTH_QUERY_INTEGRATED_FILTERING` (no effect). Grant cascade
      on RM delete + rename over both registered_model/prompt namespaces.
      Store methods added: delete_grants_for_resource, rename_grants_for_resource,
      create_page_token (mlflow-search), logged_models token offset round-trip.
      Deferred (T10.4 seams): workspace seed/cleanup + ListWorkspaces filter +
      workspaces-enabled deny fallback; scorer/gateway/review-queue hooks not
      served by this binary. Merge reconciliation (orchestrator): kept the
      admin path running the after-request hook (not early-returning), sourced
      is_admin from T9.8's cached authenticate_and_get_user, stamped the T9.6
      AuthContext; default_permission()/readable_set fallback rewired to
      AuthConfig; tests moved to AuthStore::with_config. 10 tests across 2
      binaries.)*
- [x] **T9.6 GraphQL auth middleware**: per-field READ checks, experiment-id narrowing
      for searchRuns, post-filter for searchModelVersions, admin bypass,
      `MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH` toggle.
      **AC:** GraphQL requests by non-admin users match Python results/errors.
      **VER:** auth GraphQL tests in Phase 12.
      *(Done 2026-07-17: `graphql/auth.rs` mirroring
      GraphQLAuthorizationMiddleware (auth/__init__.py:4139, applied
      handlers.py:3671). Toggle default ON (anything but false/0); checks
      skipped when auth app off. /graphql: authenticated-but-no-validator at
      request level (Dispatched::Allow, matches Python), fine-grained checks
      in execution: PROTECTED_FIELDS map (get-experiment/run/artifacts/
      metric-history-bulk-interval → experiment READ incl. parent lookup;
      searchRuns/searchDatasets → id narrowing with empty-list allow +
      none-readable deny; searchModelVersions → in-place drop, NO page-fill).
      Denied field → null with NO errors entry (graphene resolve None);
      MlflowException during resolution swallowed to deny. AuthContext
      stamped on request extensions by the T9.4 layer; T9.4 permission
      helpers extracted as pub(crate) free functions, not duplicated.
      Toggle-off case in its own test binary (env is process-global).
      10 + 1 HTTP tests; graphql_http (24) still green.)*
- [x] **T9.7 Signup UI + CSRF**: `/signup` server-rendered form (port template), CSRF
      token validation on `create-user-ui`, flash-alert redirect behavior.
      **AC:** browser signup flow works; CSRF-less POST rejected.
      **VER:** HTTP tests + manual browser check.
      *(Done 2026-07-17: `auth_api/signup.rs` + logo SVG ported verbatim.
      flask_wtf CSRF reproduced observably (own per-process secret per D12,
      NOT MLFLOW_FLASK_SERVER_SECRET_KEY): session cookie + timed form token
      as two HMAC-SHA256 envelopes mirroring itsdangerous; validation order
      + all 5 error texts byte-matched (missing/session-missing/expired
      3600s/invalid/do-not-match — the mismatch case reachable cross-wise);
      form-field → X-CSRFToken → X-CSRF-Token fallback chain. create_user_ui
      CSRF-gates BEFORE content-type like Python's csrf.protect(). Flash
      behavior: alert()+redirect HTML (auth/__init__.py:3703), duplicate →
      /signup, success → home. /signup registered outside static_prefix
      nesting (Python's one raw add_url_rule exception). Deferred: the
      is_secure referrer branch (unreachable over plain HTTP). 6 unit + 9
      HTTP tests; 2 T9.2 tests updated for CSRF.)*
- [x] **T9.8 Auth config + caches**: `basic_auth.ini` parsing (all fields incl.
      `default_permission`, `grant_default_workspace_access`, cache TTLs),
      `MLFLOW_AUTH_CONFIG_PATH`, optional credential cache (HMAC-keyed, default off),
      resource→workspace TTL cache.
      **AC:** same config file drives both servers; defaults identical.
      **VER:** config-parity unit tests.
      *(Done 2026-07-17: `mlflow-auth/src/{config,credential_cache,workspace_cache}.rs`.
      `AuthConfig` ports `config.py`'s NamedTuple field-for-field with a minimal
      INI reader (single `[mlflow]` section, `#`/`;` comments, `=`/`:`
      delimiter, no interpolation — the exact shape configparser reads for the
      shipped file). Defaults = the packaged `basic_auth.ini` verbatim
      (default_permission READ, database_uri sqlite:///basic_auth.db, admin
      admin/password1234, grant_default_workspace_access false, workspace cache
      10000/3600, auth cache 10000/0=off). `AuthConfig::default()` equals parsing
      the shipped file (asserted). Only `MLFLOW_AUTH_CONFIG_PATH` is honoured —
      the pre-T9.8 Rust-invented `MLFLOW_AUTH_DATABASE_URI` /
      `MLFLOW_AUTH_READ_DATABASE_URI` / `MLFLOW_AUTH_DEFAULT_PERMISSION` env
      overrides are dropped so both servers read the identical surface (Python
      honours no per-field env override). `authorization_function` != the shipped
      default → loud startup error (pluggable backends unsupported); bad
      permission / bad int / bad bool / missing required key / missing file all
      error loudly. Credential cache: HMAC-SHA256(random per-process key,
      password) → TTL'd `Mutex<HashMap>` (workspaces.rs pattern), off unless
      auth_cache_ttl_seconds>0; wired into `AuthStore::authenticate_and_get_user`
      (`_authenticate_cached`) which the middleware now calls (one query, admin
      check reuses the user); update_user/delete_user invalidate. Resource→
      workspace TTL cache implemented + unit-tested in `workspace_cache.rs` as
      the T10.4 seam (nothing consults it pre-T10.4 — the resolver still uses
      "default"; plug point documented at `validators.rs::experiment_permission`).
      `AuthStore` carries `Arc<AuthConfig>` + both caches; `AuthStore::new` uses
      defaults, `with_config` the production path. `main.rs::build_auth_store`
      parses the ini → connect+verify → create_admin_user +
      _warn_if_default_admin_password (Python create_app order). D12: Rust keeps
      its own per-process signup-CSRF secret; `MLFLOW_FLASK_SERVER_SECRET_KEY`
      stays Python-side and is intentionally NOT required at startup. Tests: 10
      config-parity + 4 credential-cache-store + 6 cache unit (5 workspace + …) +
      6 credential-cache unit; no_default_http switched from the retired env var
      to `AuthConfig{default_permission:NO_PERMISSIONS}`.)*
- [ ] **T9.9 Admin/account UI validation**: React admin console (`src/admin/`) and
      account page (`src/account/`) fully functional against Rust (user CRUD, role CRUD,
      grants via EditAccessModal, current-user permissions list, `is_basic_auth` logout
      behavior).
      **AC:** UI smoke checklist items green with auth enabled.
      **VER:** T11.6 with auth-enabled deployment.

### Phase 10 — Workspaces

- [x] **T10.1 Workspace store + table**: `workspaces` table, name validation (k8s regex,
      reserved names, `default` undeletable), delete modes RESTRICT/CASCADE/SET_DEFAULT
      walking all workspace-root models, artifact-root + trace-archival config
      resolution with TTL caches.
      **AC:** parity with `tests/store/workspace/test_sqlalchemy_store.py` +
      `test_workspace_validator.py`.
      **VER:** Phase 12 workspace runner.
      *(Done 2026-07-17: `mlflow-store/src/store/workspaces{,_cascade}.rs`.
      Schema matched to dbmodels/models.py:18-24 (name 63 PK, description,
      default_artifact_root, trace_archival_location/retention). Validator
      byte-matched (abstract_store.py:146-181; pattern without a regex dep).
      Delete modes per sqlalchemy_store.py:195-266 over the 10
      `_WORKSPACE_ROOT_MODELS`; Python's ORM `cascade="all"` reproduced as
      explicit FK-ordered DELETEs; RESTRICT counts + message parity;
      SET_DEFAULT preflight name-conflict check with `{name!r}` repr,
      transactional rollback. Quirks: inputs/input_tags/entity_associations
      left for `mlflow gc` like Python; `mlflow gc` hint logged on CASCADE.
      TTL caches 128 cap/60s matching env-var defaults, primed post-commit,
      evicted on delete. Trace-archival validation ported except the
      Databricks get_artifact_repository branch (doc-commented, needs the
      Python repo registry). 9 unit + 31 integration tests.)*
- [x] **T10.2 Workspace REST endpoints** (§3.17) incl. 201/204 status codes, `?mode=`,
      503-when-disabled.
      **AC:** `tests/server/test_workspace_endpoints.py` passes against Rust.
      **VER:** Phase 12 workspace runner.
      *(Done: `mlflow-server/src/workspaces_api.rs` — 5 `MlflowService` workspace
      RPCs wired via `handler_for` in `lib.rs`; `AppState` gains an
      `Option<Arc<WorkspaceStore>>` (`with_workspace_store`), `main.rs` builds it
      when `MLFLOW_ENABLE_WORKSPACES` is truthy. Handler-level trace-archival /
      artifact-root validators reproduce Python's `parameter_name` messages;
      create=201, delete=204 with `?mode=` (default RESTRICT, unknown → 400),
      disabled → plain-text 503. 8 unit + 30 integration tests.)*
- [x] **T10.3 Request workspace context + scoping**: `X-MLFLOW-WORKSPACE` header
      resolution (skip for server-info), per-request context threaded through tracking
      **and** registry queries (`WHERE workspace = ?` on every scoped model), artifact
      location prefixing `workspaces/<name>/`, forbid explicit artifact_location on
      experiment create when enabled, single-tenant startup guard.
      **AC:** `test_workspace_middleware.py` + workspace-store variants
      (`tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py`,
      `tests/store/model_registry/test_sqlalchemy_workspace_store.py`) pass over HTTP;
      cross-workspace leakage tests all negative.
      **VER:** Phase 12 workspace runner + explicit isolation tests.
      *(Done 2026-07-17: `mlflow-server/src/workspace.rs` gains
      `workspace_middleware` (tower `from_fn_with_state`) + a `ResolvedWorkspace`
      extension. Layer order (`lib.rs`): security (outermost) → workspace → auth
      (innermost) → routes, mirroring Python installing
      `workspace_before_request_handler` on the base app after
      `security.init_security_middleware` but before the auth app's
      `_before_request` (`__init__.py:82-84`). Enabled (workspace store present):
      normalize header (`_normalize_workspace`), validate non-`default` names via
      `WorkspaceNameValidator` (400), look up in the store (missing → 404
      `"Workspace '{name}' not found"`); absent header → `get_default_workspace`.
      Server-info skipped (`.../mlflow/server-info` suffix). Disabled: header
      ignored → `default`. The resolved workspace flows to the auth middleware
      via the extension (T10.4 grant-partitioning seam; pre-T10.4 validators
      still resolve in `default`). Cross-workspace isolation rides the existing
      store `WHERE workspace = ?` (tracking + registry scoped since Phase 2/T5.4)
      — no new query plumbing. Artifact prefixing + forbid in the
      `create_experiment` handler: enabled → forbid explicit `artifact_location`
      (`INVALID_PARAMETER_VALUE`, byte-matched), else derive via new store
      `create_experiment_workspace_scoped` + `WorkspaceArtifactRoot::Scoped`
      (`resolve_artifact_root` → `should_append` ? `<root>/workspaces/<ws>/<id>`
      : `<root>/<id>`; empty root → `"Cannot determine an artifact root"`).
      Applies to `default` too when enabled. Single-tenant startup guard
      `mlflow_store::verify_single_tenant_data` (INVALID_STATE, byte-matched
      "Cannot disable workspaces because {experiments|registered models|webhooks}
      exist outside the default workspace"), called from `main.rs` on the
      workspaces-disabled path; missing tables skipped. Tests: 19 HTTP
      (`workspace_scoping_http.rs`) + 5 workspace.rs unit + 3 guard unit
      (`workspaces_store.rs`). Quirks: search assertions use `view_type=ALL`
      (the ActiveOnly-default search is T3 territory). Deferred to T10.4:
      workspace-aware auth grant partitioning; the reserved-artifact-root startup
      guards (root ending `workspaces` / already scoped) are not yet ported —
      only the disable-with-non-default-data guard is.)*
- [ ] **T10.4 Workspace-aware auth integration**: role workspace partitioning, workspace
      USE/MANAGE grants, default-workspace inheritance, `NO_PERMISSIONS` boundary deny,
      workspace admin capabilities, `filter_list_workspaces`.
      **AC:** `test_auth_workspace.py` + `test_client_workspace.py` pass against Rust.
      **VER:** Phase 12 auth+workspace runner; UI workspace selector smoke.

### Phase 11 — Server config, nginx, deployment

- [x] **T11.1 CLI/env parity**: `--backend-store-uri`, `--read-replica-backend-store-uri`,
      `--registry-store-uri`, `--default-artifact-root`, `--serve-artifacts`,
      `--artifacts-destination`, `--artifacts-only`, `--host/--port/--workers` (threads),
      `--static-prefix`, `--allowed-hosts`, `--cors-allowed-origins`,
      `--x-frame-options`, `--expose-prometheus`, `--app-name basic-auth` equivalent
      (auth-enabled flag + `MLFLOW_AUTH_CONFIG_PATH`), `--workspace-store-uri`,
      `--enable-workspaces/--disable-workspaces`, `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE`
      family mapped to the Rust pool.
      **AC:** documented parity matrix; unsupported flags fail loudly.
      **VER:** CLI integration tests.
      *(Done 2026-07-17: every `server` flag wired in `config.rs` with
      Python-matching name/default/env. Supported: backend/registry/default-
      artifact-root/serve-artifacts/artifacts-destination/artifacts-only
      (gates routes to the proxy + root get/upload-artifact)/host(-H)/port/
      static-prefix/allowed-hosts/cors/x-frame/workspace-store-uri/
      enable-disable-workspaces (flag overrides env). Mapped: expose-prometheus
      (gates `/metrics`), app-name (only `basic-auth`). Accepted-noop:
      --workers (async server, logged+ignored), read-replica (stored+warned —
      tracking `Db` has no read-split seam yet; only `AuthDb` does), POOLCLASS.
      Fail-loud (clap exit 2 / ConfigError): registry-uri ≠ backend, unknown
      flags (--dev/--gunicorn-opts/--uvicorn-opts/--waitress-opts/
      --trace-archival-config/--secrets-cache-*). Pool env family mapped in
      `pool.rs`. Parity matrix doc at `mlflow-server/CLI_PARITY.md`. 18 config
      + 5 CLI-integration + 1 artifacts-only tests. Deferred: tracking
      read-replica split (SEAM).)*
- [x] **T11.2 Security middleware parity**: host-header allowlist, CORS, X-Frame-Options
      (mirror `mlflow/server/security.py`).
      **AC:** identical responses to disallowed Host/CORS preflights.
      **VER:** table-driven HTTP tests vs both servers.
      *(Done 2026-07-17: `mlflow-server/src/security.rs` tower layer, outermost
      (runs before auth → disallowed Host is 403 before any 401, matching
      Python installing security on the base app before the auth app). Ported
      `security.py`/`security_utils.py`: Host allowlist
      (`--allowed-hosts`/`MLFLOW_SERVER_ALLOWED_HOSTS`, default localhost +
      RFC1918/4193 patterns, `*` disables, fnmatch only when pattern has `*`),
      CORS (`--cors-allowed-origins`/env, localhost always allowed, wildcard
      disables credentials, preflight 204 + exact ACA-* header set + `Vary:
      Origin`, echoes request-headers), cross-origin state-change block, and
      X-Frame-Options (`--x-frame-options`/env, default SAMEORIGIN, NONE
      disables) + `X-Content-Type-Options: nosniff` on every response incl.
      403/404. Byte-matched rejection bodies. Quirks reproduced: flask-cors
      decorates even the 403 rejection with CORS headers for an allowed
      origin; bare `[::1]` matched by equality not glob. Python has env only —
      added the 3 CLI flags (flag>env) mirroring `--static-prefix`; full CLI
      parity is T11.1. Deferred: notebook-renderer X-Frame exemption (no such
      route here), FastAPI/OTLP security variant,
      MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE kill-switch. 20 HTTP + 14 unit
      tests; 24 existing test files updated for the 3 new ServerConfig
      fields.)*
- [x] **T11.3 nginx reference config** implementing §2.2 ("default → Rust, genai →
      Python"), `proxy_buffering off` for Python SSE/streaming locations,
      client_max_body_size for artifact uploads; `rust/deploy/nginx.conf` +
      docker-compose (nginx + rust + python + postgres).
      **AC:** `docker compose up` yields working MLflow at `:80`; UI works end to end;
      genai requests observably hit Python, everything else Rust (access logs).
      **VER:** `rust/deploy/smoke.sh` (SDK: experiments/runs/metrics/traces/models/
      webhooks/users; asserts backend attribution via distinctive server headers).
      *(Done: `rust/deploy/{nginx.conf,Dockerfile.rust,Dockerfile.rust.dockerignore,
      docker-compose.yml,smoke.sh,README.md}`. Migration ordering: one-shot `migrate`
      service runs `mlflow db upgrade` on Postgres, then rust/python wait on
      `service_completed_successfully` (Rust refuses an unmigrated DB). Every proxied
      response carries `X-MLflow-Backend: rust|python` + custom access-log format.
      `docker compose up -d --wait` all-healthy; `smoke.sh` = 28/28 requests with
      correct attribution (14 python, 44 rust across log lines). UI static (`/`,
      `/static-files/*`) proxied to Python for now — the nginx→JS-build split is
      T11.4. Deviations: Dockerfile build context is the repo root, not `rust/`, so
      `build.rs`/`mlflow-proto`/graphql-schema `include_str!` can read
      `mlflow/{version.py,protos/,server/js/.../autogenerated_schema.gql}`; python +
      migrate healthchecks use the image's bundled `python3` since the mlflow image
      ships no `curl`.)*
- [x] **T11.4 Frontend split**: nginx serves `mlflow/server/js/build/`; document
      `yarn build`, cache headers (28-day hashed assets, no-cache `index.html`).
      **AC:** UI fully loads with the Python container stopped, except genai pages.
      **VER:** compose smoke with Python paused.
      *(Done: `nginx.conf`'s `/` and `/static-files/*` locations now serve directly
      from the build dir bind-mounted at `/usr/share/mlflow-ui` (`docker-compose.yml`
      mounts `../../mlflow/server/js/build`), mirroring Python's
      `mlflow/server/__init__.py` static handlers (`serve_static_file()` ->
      `max_age=2419200`; `serve()` -> `index.html`, no cache header — we go one step
      further and pin `no-cache`). Response header `X-MLflow-Backend: static`
      distinguishes nginx-served UI from proxied `rust`/`python`. If the build dir
      is empty/missing, `try_files` falls back to proxying Python (pre-T11.4
      behavior), so `docker compose up` still works without a build. New
      `rust/deploy/smoke_frontend.sh`: baseline cache-header checks, then stops the
      `python` service and re-verifies `/` + hashed asset still 200/`static`, Rust
      API still works, and a genai request now 502s (expected — genai has no Rust
      implementation); restarts `python` via trap. `rust/deploy/build_placeholder_ui.sh`
      generates a minimal stand-in build (`index.html` + one hashed
      `static/js/main.<hash>.js`) for smoke runs without a real, network-heavy
      `yarn build`; README.md documents both the real `yarn build` path and this
      fallback. Verified: `docker compose up -d --wait` all-healthy, `smoke.sh`
      28/28 PASS, `smoke_frontend.sh` 16/16 PASS (incl. Python-stopped section),
      `down -v` clean; `cargo fmt --check`/`cargo build --release`/`cargo test
      --release` all exit 0 (crate code untouched). Deferred:
      `cargo clippy --all-targets -D warnings` fails on a pre-existing,
      unrelated `needless_splitn` lint in `mlflow-auth/src/hash.rs` (predates this
      task, out of scope per "you shouldn't touch crate code").)*
- [x] **T11.5 `server-info` in Rust** (D5): serve
      `/(api|ajax-api)/3.0/mlflow/server-info` with flags consistent with the deployment
      (workspaces on/off, auth on/off); verify `useServerInfo` consumers behave.
      **AC:** UI boots with no console errors; feature gates render correctly.
      **VER:** browser console check in UI smoke.
      *(Done 2026-07-17: `mlflow-server/src/server_info.rs`, hand-registered
      like /graphql (handlers.py:6799 — GET only; the plan prose saying
      GET/POST was wrong). Exactly Python's 3 fields (handlers.py:6612-6616 /
      useServerInfo.tsx:9-13): store_type always "SqlStore" (Rust has no
      FileStore), workspaces_enabled from AppState, trace_archival_enabled
      hardcoded false (no archival in Rust yet). There is NO auth_enabled
      wire field in Python — D5's "auth on/off" is deployment intent, not a
      field. Reachable unauthenticated + carved out of the workspace-header
      gate like Python. 7 HTTP tests incl. byte-exact body. Browser check
      rides T11.6.)*
- [ ] **T11.6 UI smoke checklist** (manual or playwright): experiment list, runs table
      (Load more), run detail (GraphQL), charts (bulk-interval), compare runs, metric
      page, artifact browser, traces tab (list, span tree, attachments, assessments),
      logged models tab, datasets dropdown, **model registry pages (models list, version
      detail, stages, aliases, model artifact download), prompts pages, admin console
      (users/roles/grants), account page, workspace selector**.
      **AC:** all render against Rust; network tab shows Python only for genai.
      **VER:** recorded checklist in PR; optional Playwright under `rust/e2e/`.

### Phase 12 — Compliance harness & CI

- [x] **T12.1 Server-launch integration**: extend
      `tests/tracking/integration_test_utils.py` `_init_server` with
      `server_type="rust"`; parametrize `mlflow_client` fixture in
      `test_rest_tracking.py`; point `tests/server/auth/auth_test_utils.py` launcher at
      the Rust binary (auth config env).
      **AC:** `pytest tests/tracking/test_rest_tracking.py` and `tests/server/auth/`
      runnable against Rust via a switch.
      **VER:** local runs; failures triaged into a parity backlog issue.
      *(Done 2026-07-17: switch `MLFLOW_SERVER_TYPE=rust` (+
      `MLFLOW_RUST_SERVER_BIN` override, auto-cargo-build) drives both the
      tracking `mlflow_client` fixture (subprocess `_init_server` path; file
      store variants skip) and `test_auth.py::client` via
      `resolve_auth_server_launch()` (init_db + admin seed through the real
      Python auth store; ini's database_uri → MLFLOW_AUTH_DATABASE_URI).
      Python default paths unchanged/verified. First smoke: experiments
      subset 7/8 PASS against Rust. **Parity backlog opened:** (1)
      type-mismatch validation errors surface serde text instead of
      Python's "Invalid value 123 for parameter 'name'"; (2) auth
      enforcement absent pre-T9.4 (unauth requests succeed) — expected to
      close when T9.4 merges; (3) HTTP reason-phrase casing "Bad Request" vs
      Python's "BAD REQUEST". Remaining auth fixtures (test_client*.py,
      fastapi_*) still flask-only — extend during T12.3.)*
- [x] **T12.2 `MLFLOW_RUST_STORE_TESTING` flag** mirroring the Go flag; every use links a
      justification.
      **AC:** flag exists; zero unexplained uses.
      **VER:** grep + review.
      *(Done 2026-07-17: `_MLFLOW_RUST_STORE_TESTING` declared beside
      `_MLFLOW_GO_STORE_TESTING` (environment_variables.py:1074 precedent —
      Go has NO launcher coupling, it's purely an assertion deviation gate,
      so the Rust flag stays orthogonal to T12.1's MLFLOW_SERVER_TYPE;
      documented in the docstring). One justified use: reason-phrase casing
      gate in test_auth.py (backlog #3), verified live both ways. Backlog #1
      (validation messages) deliberately ungated until T12.3's mass sweep;
      #2 left to close via T9.4.)*
- [ ] **T12.3 Client-suite conformance**: `tests/tracking/test_tracking.py`, client
      tests, and `tests/store/model_registry/test_rest_store*.py`-derived HTTP checks
      with `MLFLOW_TRACKING_URI=http://rust`; DB reset between tests.
      **AC:** suites green (modulo flag-gated diffs).
      **VER:** CI logs.
- [ ] **T12.4 Differential replay harness** (`rust/compliance/`): request corpus covering
      every §3 endpoint (success + error + pagination walks + multi-user auth scenarios +
      workspace headers), replayed against Python and Rust, normalized diff; token
      fields checked for opacity only.
      **AC:** zero non-allowlisted diffs on sqlite + postgres.
      **VER:** CI artifact.
- [ ] **T12.5 CI matrix** modeled on the `database` job (`master.yml`): `mlflow-rust`
      service in `tests/db/compose.yml`, runs tracking + registry + auth + workspace
      compliance subsets on postgres/mysql/sqlite (+ mssql per T0.3); tracing job modeled
      on `tracing.yml`.
      **AC:** required-check workflow green on the feature branch.
      **VER:** GitHub Actions.
- [ ] **T12.6 Concurrency/chaos**: parallel log-batch + start_trace + log_spans +
      registry version creation + searches on postgres; no client-visible 5xx in 10k
      mixed ops; MV `MAX+1` race resolves via retry like Python.
      **AC:** 0 unexpected errors; version numbers dense and unique.
      **VER:** `rust/tests/chaos.rs` nightly.

### Phase 13 — Schema evolution for 100 GB scale (after parity is proven)

- [ ] **T13.1 Index migrations** (alembic): `runs(experiment_id, lifecycle_stage,
      start_time)`, `logged_models(experiment_id)`, `inputs(source_id)`,
      `model_versions(run_id)`, `model_versions(current_stage)`; fix the two
      empty `Index()` declarations (tracking models.py:832,863).
      **AC:** upgrade+downgrade clean on all dialects (`tests/db/check_migration.sh`);
      EXPLAIN shows index usage; no Python-suite regression.
      **VER:** migration CI + query plans on a seeded 10M-run dataset.
- [ ] **T13.2 `span_attributes` extraction table** (Q7): indexed key/value maintained on
      span ingest; span-content LIKE filters rewritten; shared alembic migration with a
      Python-compatible write path or capability-gated Rust-only (D7).
      **AC:** span-attribute filter on 50M spans < 1s on postgres; results identical to
      LIKE baseline on the corpus.
      **VER:** benchmark report + differential search results.
- [ ] **T13.3 Benchmark suite + seeded dataset generator** (`rust/bench/`): ~100 GB-scale
      synthetic DB (millions of runs, dense metrics, 10M+ traces with spans, 100k+ model
      versions); scenarios: runs/search with metric filters + ordering, deep pagination,
      bulk-interval history, traces/search with span filters, OTLP ingest throughput,
      registry search with prompt anti-join.
      **AC:** documented p50/p95 Python vs Rust; targets: p95 run-search < 500 ms,
      deep-page O(1), OTLP ingest ≥ 5x Python.
      **VER:** `rust/bench/RESULTS.md` with hardware notes.
- [ ] **T13.4 Deeper restructures** informed by T13.3 (metric partitioning, narrower
      metrics PK with dedup hash, trace hot/cold split, auth grant semi-join
      materialization). Out of scope until benchmarks prove need.
      **AC:** written proposal per change with migration + rollback story.
      **VER:** design doc reviewed.

### Phase 14 — Memory & production validation

- [ ] **T14.1 Memory baseline**: Python server RSS (4 uvicorn workers, idle + load) vs
      Rust on identical workloads.
      **AC:** report with Rust idle/loaded RSS; target ≥ 5x total reduction.
      **VER:** `rust/bench/memory.md` (cgroup memory.current sampling).
- [ ] **T14.2 Soak test**: 24 h mixed workload (ingest + search + UI polling + webhook
      deliveries) at realistic rates on postgres; RSS growth, pool health, error rates.
      **AC:** no monotonic RSS growth; error rate < 0.01%; no webhook-delivery task
      leaks.
      **VER:** soak dashboard/logs in release notes.
- [ ] **T14.3 Operational docs**: deployment guide (compose + k8s), migration runbook
      (Python-only → split; auth DB sharing; secret/key management for Fernet + CSRF),
      rollback procedure (nginx flips routes back to Python — zero data migration, both
      DBs shared).
      **AC:** a fresh operator deploys the split from docs alone.
      **VER:** doc walkthrough by someone not on the project.

---

## 8. Verification quick-reference (how to confirm the whole thing works)

1. **Route parity**: `rust/tools/route_parity.py` — Python `get_endpoints()` +
   auth-route dump equals the Rust route table (modulo documented genai routes).
2. **Wire parity**: JSON golden tests (T1.3) + differential replay harness (T12.4) with
   zero non-allowlisted diffs on sqlite and postgres, including auth'd multi-user and
   workspace-header scenarios.
3. **Behavioral parity**: `tests/tracking/test_rest_tracking.py`, client suites,
   registry REST checks, `tests/server/auth/`, and workspace endpoint/middleware suites
   green against Rust on the DB matrix (T12.1-T12.5).
4. **UI parity**: T11.6 smoke — tracking, tracing, registry, prompts, admin/account, and
   workspace-selector flows all green through nginx with the frontend served statically;
   genai features still functional via Python.
5. **Interop**: users/webhook secrets created by either server work on the other
   (shared auth DB, Fernet, werkzeug hashes); alembic head pins enforced on both DBs.
6. **Scale**: T13.3 benchmarks meet latency targets; EXPLAIN plans show index usage on
   hot paths.
7. **Memory**: T14.1/T14.2 reports demonstrate reduction and stability.

Local dev loop once Phase 12 lands:

```bash
# 1. build rust server
cargo build --release --manifest-path rust/Cargo.toml

# 2. run compliance suites against it (sqlite)
MLFLOW_RUST_SERVER_BIN=rust/target/release/mlflow-server \
  uv run pytest tests/tracking/test_rest_tracking.py --server-impl rust
MLFLOW_RUST_SERVER_BIN=rust/target/release/mlflow-server \
  uv run pytest tests/server/auth --server-impl rust

# 3. full stack smoke
docker compose -f rust/deploy/compose.yaml up
bash rust/deploy/smoke.sh
```

---

## 9. Open decisions & risks

| ID | Decision/Risk | Notes | Status |
|---|---|---|---|
| D1 | **Auth enforcement split**: Rust enforces natively for its routes; Python's auth app keeps covering genai routes; both share the auth DB. Requires werkzeug-hash + synthetic-role + resolution-chain parity (§3.16). Custom `authorization_function` plugins: Rust v1 supports built-in basic-auth only (Python already restricts custom functions on FastAPI routes too, `auth/__init__.py:4521`). | **decided by scope change (2026-07-13)** — auth is in Rust scope | decided |
| D2 | **MSSQL**: sqlx has no MSSQL driver; needs `tiberius` adapter. | v1 = sqlite/postgres/mysql, mssql fast-follow. | **decided (2026-07-13)** |
| D3 | **Pagination tokens**: keyset tokens change token contents (still opaque). nginx routes each endpoint to exactly one backend, so mixed-backend paging cannot occur. | Keyset tokens approved; verify no test decodes tokens; flag-gate if any do. | **decided (2026-07-13)** |
| D4 | **Pretty-printed JSON (indent=2)**: replicate for simpler golden/differential testing. | Replicate. | **decided (2026-07-13)** |
| D5 | **`server-info`**: Rust owns it (T11.5) and must report feature flags matching the deployment (auth, workspaces, and genai features that live on Python). | Audit `useServerInfo` consumers for flags that gate genai UI — those must still reflect Python availability. | open |
| D6 | **Trace archival (`ARCHIVE_REPO`)**: not in v1; Rust returns clear NOT_IMPLEMENTED for archived traces. Deployments using archival keep tracing on Python until supported. | | open |
| D7 | **`span_attributes` table** (T13.2): shared alembic migration + Python write path vs Rust-only capability. | Shared migration safer for mixed deployments. | open |
| D8 | **Version skew**: Python and Rust evolve independently. | Route-parity test (T1.2) + dual schema-head pins (T2.1/T9.1) + GraphQL schema diff (T6.2) turn skew into CI failures. Pin supported MLflow version per Rust release. | accepted |
| D9 | **Custom auth plugins** (`authorization_function`) are Python callables — not portable. | Rust v1: built-in basic-auth; document JWT/OIDC as nginx-level or future Rust plugin API. Deployments with custom Python auth functions can't split auth'd routes yet. | decided (v1 limitation) |
| D10 | **Jobs API stays Python**: `/ajax-api/3.0/jobs/*` + Huey runner exist to execute genai workloads (scorer invocations, evaluations). | Revisit only if a non-genai consumer appears. | decided |
| D11 | **Webhook delivery durability**: Python is fire-and-forget from an in-process pool — deliveries die with the process. Rust replicates this for parity; a durable outbox table is a possible improvement but changes semantics. | Keep parity in v1; log a proposal for later. | open |
| D12 | **Key management**: `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` (Fernet) must be identical on both servers; Rust needs its own signup-CSRF secret; `MLFLOW_FLASK_SERVER_SECRET_KEY` remains Python-side. Document in T14.3 runbook. | | open |
| D13 | **Werkzeug hash algorithm** depends on the pinned werkzeug version (pbkdf2:sha256 vs scrypt). | Spike result: werkzeug 3.1.8 defaults to `scrypt:32768:8:1` (salt = 16 ASCII chars, hex digest). Rust verifies scrypt + pbkdf2:sha256 (param-driven from the stored hash) and generates 3.1.8-default scrypt. Other methods (pbkdf2:sha1/sha512, legacy) rejected loudly — add if seen in real DBs. | **decided (2026-07-13)** |
| R1 | **Behavioral corners not covered by tests** (exact error strings, odd query-arg parsing, after-request filter edge cases). | Differential replay harness (T12.4) is the safety net; grow the corpus on every wild diff. | mitigated |
| R2 | **`spans.content` size** (LONGTEXT) — heavy row fetches for large traces. | Lazy content reads (T2.11) + Phase 13 payload evaluation. | mitigated |
| R3 | **Auth after-request filtering is subtle** (refetch-to-fill-page, workspace deny semantics, prompt-vs-model classification per row). | Dedicated multi-user differential fixtures (T9.5); keep a Python-identical fallback mode behind a flag. | mitigated |

---

## 10. Research appendix (where the facts came from)

Tracking/tracing:
- Endpoint generation & handler map: `mlflow/server/handlers.py:6723-6807, 7663`;
  proto options in `mlflow/protos/service.proto`.
- JSON codec quirks: `mlflow/utils/proto_json_utils.py:32-168`. Error model:
  `mlflow/exceptions.py:30-140`.
- Search DSLs: `mlflow/utils/search_utils.py` (`SearchUtils:172`,
  `SearchExperimentsUtils:1053`, `SearchModelUtils:1267`,
  `SearchModelVersionUtils:1452`, `SearchTraceUtils:1688`,
  `SearchLoggedModelsUtils:2504`).
- Store internals: `mlflow/store/tracking/sqlalchemy_store.py` (search_runs L2006-2096,
  latest_metrics L1366-1483, filter/orderby builders L9054-9220, search_traces
  L3755-3839, log_spans L4971-5362, bulk upsert L9806). Schema:
  `mlflow/store/tracking/dbmodels/models.py`; migrations `mlflow/store/db_migrations/`
  (head `b7e4c1a90f23`); verification `mlflow/store/db/utils.py:109-134`.
- Server runtime: `mlflow/server/__init__.py`, `mlflow/server/fastapi_app.py`,
  `mlflow/server/otel_api.py`, `mlflow/cli/__init__.py:367-538`.

Registry/webhooks:
- Proto `mlflow/protos/model_registry.proto:13-483`, `mlflow/protos/webhooks.proto:14-116`;
  handlers `mlflow/server/handlers.py:2638-3462, 7705-7733`.
- Store `mlflow/store/model_registry/sqlalchemy_store.py` (search filters 589-832, prompt
  anti-join 776, latest versions 297, create MV 981-1096, transition 1192, soft delete
  1244); models `mlflow/store/model_registry/dbmodels/models.py`; constants
  `mlflow/store/model_registry/__init__.py`.
- Prompts-on-registry: `mlflow/store/model_registry/abstract_store.py:434-1160`,
  `mlflow/prompt/constants.py`.
- Webhook delivery: `mlflow/webhooks/delivery.py` (HMAC 121, retries 80, pool 51),
  `mlflow/webhooks/ssrf.py`, `mlflow/webhooks/constants.py`.

Auth/workspaces:
- `mlflow/server/auth/__init__.py` (validators map 2480-2617, before_request 2912,
  after_request 3594-3650, resolution chain 556-1022, GraphQL middleware 4139, FastAPI
  middleware 4287-4521, create_app 4610); `auth/routes.py`, `auth/permissions.py`,
  `auth/entities.py`, `auth/config.py`, `auth/basic_auth.ini`;
  store `auth/sqlalchemy_store.py` (synthetic roles 259-543, resolver 2010);
  DB `auth/db/models.py`, migrations `auth/db/migrations/` (head `f1a2b3c4d5e6`,
  version table `alembic_version_auth`).
- Workspaces: proto `service.proto:1051-1132`; handlers `handlers.py:1351-1496`;
  `mlflow/store/workspace/` (abstract/sqlalchemy/rest stores);
  `mlflow/server/workspace_helpers.py`; `mlflow/utils/workspace_utils.py`
  (`X-MLFLOW-WORKSPACE`), `mlflow/utils/workspace_context.py`;
  workspace-aware stores `mlflow/store/tracking/sqlalchemy_workspace_store.py:62`,
  `mlflow/store/model_registry/sqlalchemy_workspace_store.py`.

Frontend:
- `mlflow/server/js/src/experiment-tracking/sdk/MlflowService.ts`;
  runs table `.../experiment-page/hooks/useExperimentRuns.tsx`
  (`RUNS_SEARCH_MAX_RESULTS=100`); charts `.../useSampledMetricHistory*.tsx` (320 pts);
  trace source dispatch `experiment-tracking/utils/TraceUtils.ts`;
  GraphQL `graphql/client.ts` + four hook files; registry `model-registry/services.ts`;
  admin/account `src/admin/api.ts`, `src/account/api.ts`;
  workspaces `src/workspaces/utils/WorkspaceUtils.ts`;
  relative URLs `common/utils/FetchUtils.ts:60`; asset base `package.json`
  (`"homepage": "static-files"`).

Compliance infra:
- `tests/tracking/integration_test_utils.py`, `tests/tracking/test_rest_tracking.py`,
  `tests/tracking/test_mlflow_artifacts.py`, `tests/store/tracking/test_rest_store.py`,
  `tests/store/model_registry/` (sqlalchemy/rest/webhooks/workspace suites),
  `tests/server/auth/auth_test_utils.py` + suites, `tests/server/test_workspace_*.py`,
  `tests/store/workspace/`, `tests/db/compose.yml`, `.github/workflows/master.yml`
  (database job), `mlflow/environment_variables.py:1074`,
  `mlflow/tracking/_tracking_service/utils.py:252`.
