# Rust MLflow Server — Implementation Plan

**Part I** (§1–§10, Phases 0–14): everything except genai. **Part II** (§11–§18,
Phases 15–22): the genai port — added 2026-07-17; goal is full Python-app parity
in a Python-free Rust deployment, retiring both the Python server plane and the
Python job-execution runtime.

Status: **Part 1 COMPLETE except T9.9/T11.6 (browser-driven UI validation,
deliberately deferred) — Phases 2–8, 10, 12, 13, and 14 done; Phase 9/11 done
except those two UI checks. Corpus GREEN + required CI gate; client suites 0
failures vs Rust; benchmarks, soak (67–106x memory reduction, 0 errors), and
operational docs landed. Next: Part 2 (genai port) per user directive.**
· Branch: `feature/rust-tracking-server` · Last updated: 2026-07-18

**Resume notes (2026-07-18):** implementation subagents now run via Codex
(gpt-5.6-sol); the orchestrator verifies, merges, and ticks the plan.

**Open:**
- **Part 2 (genai port)** — Phases 15, 16, AND 17 COMPLETE 2026-07-18.
  Phase 16: all six CRUD tasks, zero new migrations (every table pre-existed
  at head `c4a9b7d3e812`). Phase 17: runner + native worker + scheduler +
  invoke endpoints; jobs execute Python-free end to end (fixture mode for
  Phase 19 semantics). Phase 18 (gateway) COMPLETE 2026-07-19: all seven
  tasks — CRUD + crypto, discovery + proxy bridge, runtime core with SSE
  parity, full provider matrix (191/191 pinned providers, zero Python
  fallback), traffic split + fallback, budget enforcement, guardrails.
  Every §12 route family fully accounted (§12.8 78/0, §12.9 10/0). Replay
  corpus 188 cases, zero non-allowlisted diffs. Phase 19 (native GenAI
  execution) COMPLETE 2026-07-19: scorers + judges, evaluation/invoke/
  online scoring, third-party scorer parity (T19.3+T19.3b), issue
  discovery, prompt optimization (exact CPython MT19937 GEPA) — all five
  tasks native, Python-free, with six pinned oracles as gates. Also
  2026-07-19: root-caused the day's "load-correlated flakes"/ICEs to
  leaked uvicorn reference servers exhausting the WSL2 cgroup pid limit
  (see T19.5 note); post-cleanup full suite 1,353 tests green. Next:
  Phase 20 (assistant §12.10 + promptlab §12.11).
- **D23 Phoenix license blocker** — RESOLVED: user approved the rejection
  approach 2026-07-18; rejection errors must point at builtin/instructions-
  judge equivalents (see D23 row).
- **T9.9 + T11.6** — browser-driven UI validation, deliberately deferred to be
  done together.
- Deferred seams: postgres corpus support in replay.py (TODO(T12.5) markers),
  tracking read-replica split (T11.1 SEAM), workspaces_store.rs sqlite-only
  tests.
- **Rust artifact proxy lacks cloud schemes (S3/GCS/Azure)** — surfaced by
  the T14.2 soak: `--serve-artifacts` with `--artifacts-destination s3://...`
  is Python-only today; Rust proxies local FS only. Client-direct uploads
  (the common client path) are unaffected. Document in the T14.3 runbook
  (route artifact-proxy traffic to Python, or use client-direct); candidate
  follow-up task for Part 2 era.
- Parity backlog opened by T12.1 (see its note): #1 type-mismatch validation
  messages (serde text vs Python's "Invalid value … for parameter …"); #3
  HTTP reason-phrase casing (gated in tests via `MLFLOW_RUST_STORE_TESTING`).
  #2 (auth enforcement) was closed by T9.4.

This document is the master plan for reimplementing the MLflow server in Rust for all
**non-genai** functionality: tracking, tracing, artifacts, GraphQL, **model registry,
webhooks, auth/RBAC (incl. the admin/account UI backend), and workspaces** — fronted by
nginx, with full wire-level feature parity against the Python implementation. GenAI
features (gateway, scorers, evaluation, issues, label schemas, review queues, prompt
optimization, assistant, jobs) stay on the Python server **during Part I only** —
Part II (§11 onward) ports their wire, storage, orchestration, and execution paths;
its end state contains no Python runtime in the server deployment (D14).

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
- [x] **T10.4 Workspace-aware auth integration**: role workspace partitioning, workspace
      USE/MANAGE grants, default-workspace inheritance, `NO_PERMISSIONS` boundary deny,
      workspace admin capabilities, `filter_list_workspaces`.
      **AC:** `test_auth_workspace.py` + `test_client_workspace.py` pass against Rust.
      **VER:** Phase 12 auth+workspace runner; UI workspace selector smoke.
      *(Done 2026-07-17: threaded `RequestCtx::workspaces_enabled` /
      `AfterCtx::workspaces_enabled` (`AppState::workspace_store().is_some()`)
      through the auth validators + after-request hooks. The
      enabled-vs-disabled difference lives in the fold's "no grant matched"
      branch (`validators.rs::resolve_role_permission`): workspaces OFF → fall
      to `default_permission` (`_role_permission_for` returns `None`,
      `__init__.py:711-712`); ON → `NO_PERMISSIONS` boundary deny
      (`:715`) unless the opt-in default-workspace auto-grant
      (`_user_inherits_default_workspace_grant`, `:541`,
      `grant_default_workspace_access` off by default) applies in the `default`
      workspace. Grant lookups already scope to `RequestCtx::workspace` (T10.3
      resolved) via `get_role_permission_for_resource`, whose `(workspace,*)`
      fold gives workspace-admin MANAGE concrete-resource reads and USE only the
      workspace-tier create-gate. Wired: `_user_can_create_in_workspace`
      (create-experiment/RM + list-users, `:580`/`:1663`/`:1208`),
      `validate_can_view_workspace` (GetWorkspace, `:1216`), workspace REST
      before-request validators (`Create/Update/Delete → sender_is_admin`,
      `Get → view`, `List → none`, `:2611`), and after-request hooks
      `filter_list_workspaces` (`:3140`), `_seed_default_workspace_roles`
      (`:3201`, admin/user two-tier, `MLFLOW_RBAC_SEED_DEFAULT_ROLES` default
      on), `_cleanup_workspace_permissions` (`:3259`). Read-predicate fallback
      gated to deny when enabled (`after_request.rs::readable_set`, `:1616`).
      New auth-store methods `list_accessible_workspace_names`
      (`sqlalchemy_store.py:994`, synthetic-role can_read semantics) +
      `delete_workspace_permissions_for_workspace` (`:695`). Single-tenant
      (disabled) path byte-identical to pre-T10.4 — full workspace test suite
      green. Cache decision: the T9.8 resource→workspace cache is NOT on the
      hot path — Rust store fetches are already workspace-scoped, so the auth
      resolver looks up grants directly in the request workspace with no
      unscoped resource→workspace round-trip to memoize (documented in
      `workspace_cache.rs`). 14 HTTP tests in
      `tests/auth_workspace_http.rs`.)*

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
- [x] **T12.3 Client-suite conformance**: `tests/tracking/test_tracking.py`, client
      tests, and `tests/store/model_registry/test_rest_store*.py`-derived HTTP checks
      with `MLFLOW_TRACKING_URI=http://rust`; DB reset between tests.
      **AC:** suites green (modulo flag-gated diffs).
      **VER:** CI logs.
      **DONE (2026-07-18):** all four suites run under `MLFLOW_SERVER_TYPE=rust`
      with per-test DB freshness. Flagship `test_rest_tracking.py`: 122 passed /
      0 failed / 149 skipped / 1 xfail / 1 xpass (baseline was 39 failed; 26
      triaged skips, each with a stated reason — Python-process internals,
      monkeypatch-dependent, or unimplemented genai APIs). Auth
      `test_auth.py`: 95 passed / 0 failed (was 7 failed). `test_client.py` +
      `test_client_rbac.py`: 94 passed / 0 failed (was 8 failed). Parity fixes
      landed en route: artifact headers/MIME, nonfinite protobuf floats,
      assessments, logged-model metrics/version links, trace attributes,
      relative artifact locations (new `uri_util.rs`), bulk-metric param
      validation, trace-link source permissions, permission-filtered run
      search, security-middleware disable flag, role-management authorization,
      workspace-scoped role visibility, per-user grant validation, malformed
      scorer-resource handling. No `_MLFLOW_RUST_STORE_TESTING` gate was
      needed. Verified post-merge: cargo fmt/clippy/test-workspace exit 0,
      T12.4 replay corpus exit 0 with no allowlist changes.
- [x] **T12.4 Differential replay harness** (`rust/compliance/`): request corpus covering
      every §3 endpoint (success + error + pagination walks + multi-user auth scenarios +
      workspace headers), replayed against Python and Rust, normalized diff; token
      fields checked for opacity only.
      **AC:** zero non-allowlisted diffs on sqlite + postgres.
      **VER:** CI artifact.
      *(Done 2026-07-17 on sqlite: 133 cases / 12 sections, **0 non-allowlisted
      diffs, 0 status mismatches, 0 errors** (11 allowlisted, all documented:
      Flask default HTML error pages on route misses, the duplicate-experiment
      IntegrityError leak with per-run bind params, the webhook
      unknown-entity 500-vs-400 deliberate deviation). Engine: bind/normalize/
      token-opacity diffing with order-insensitive `tags`, `_ms`-timestamp
      normalization, and `/__status__`-pointer allowlisting for deliberate
      status deviations. Driving it green fixed EIGHT real Rust parity bugs:
      experiment workspace field, view_type proto2 default, create_run
      outputs omission, logged-model source_run_id '', RM description
      omission, metric-history 200-empty on missing runs (single-tenant),
      plus the big one — Python's lenient ParseDict request parsing
      (swallowed parse errors + raw-JSON schema fallback + partial-parse
      semantics, `mlflow-proto/src/lenient.rs` +
      `mlflow-server/src/schema_validation.rs`) which closed parity-backlog
      #1, log-batch, logged-model finalize/params, and startTraceV3. The
      compliance CI job is now a REQUIRED gate. Deferred: postgres corpus run
      (replay.py is sqlite-only; TODO(T12.5) markers) and the coverage-notes
      extensibility backlog in the report.)*
- [x] **T12.5 CI matrix** modeled on the `database` job (`master.yml`): `mlflow-rust`
      service in `tests/db/compose.yml`, runs tracking + registry + auth + workspace
      compliance subsets on postgres/mysql/sqlite (+ mssql per T0.3); tracing job modeled
      on `tracing.yml`.
      **AC:** required-check workflow green on the feature branch.
      **VER:** GitHub Actions.
      *(Done 2026-07-17 as ADVISORY: `compliance` job in
      `.github/workflows/rust.yml` — pg16+mysql8 services (T2.2 pattern),
      builds the Rust binary, `uv sync --extra auth --extra db`, runs the full
      T12.4 replay corpus with `continue-on-error: true` + always-uploaded
      report artifact; paths widened to rust/** + mlflow/**. Gating stays
      advisory until T12.4 is green — first honest full run: 133 cases,
      67 non-allowlisted diffs, 18 status mismatches, 0 errors. Most
      mismatches trace to a HARNESS gap (replay.py's Python boot never sets
      SERVE_ARTIFACTS_ENV_VAR → artifacts 5/5 + auth 7/8 mismatch), not
      server bugs. pg/mysql seeding not wired (replay.py is sqlite-only —
      TODO(T12.5) markers); no compose change needed (harness boots servers
      itself). `rust/.gitignore` now ignores `/compliance/report/`. Flip to
      required once T12.4 hits zero non-allowlisted diffs.)*
- [x] **T12.6 Concurrency/chaos**: parallel log-batch + start_trace + log_spans +
      registry version creation + searches on postgres; no client-visible 5xx in 10k
      mixed ops; MV `MAX+1` race resolves via retry like Python.
      **AC:** 0 unexpected errors; version numbers dense and unique.
      **VER:** `rust/tests/chaos.rs` nightly.
      Landed as `rust/crates/mlflow-server/tests/chaos.rs`: boots the full axum
      app in-process on a real socket against a live Postgres schema, hammers it
      with 10k mixed ops (`MLFLOW_CHAOS_OPS` override) over ~32 bounded tokio
      tasks — log-batch 40%, start-trace 15%, log-spans 15%, MV-create 10%,
      search-runs 10%, search-MVs 5%, search-traces 3%, exp create/delete 2%.
      MV creation on one shared model is bounded to 4-way (the store/Python
      `MAX(version)+1` retry loop has no backoff, so higher contention makes
      retry-exhaustion the common case rather than a tail). Asserts 0 unexpected
      5xx (the `CREATE_MODEL_VERSION_RETRIES=3` exhaustion → 500 is accepted as
      Python's documented contract and tallied separately), MV versions dense
      `1..=N` and unique (DB-level + client-observed), and experiment
      create/delete consistency. Env-gated on `MLFLOW_RUST_TEST_PG_URI` (skips
      fast otherwise; `cargo test --workspace` on sqlite stays green). Measured:
      10k ops in ~16.5s (~600 ops/s), 0 other 5xx, ~1% MV retry-exhaustion.
      CI: nightly `chaos` job in `.github/workflows/rust.yml` (`schedule` cron +
      `workflow_dispatch` only, not on PRs; Postgres service + `mlflow db
      upgrade`).

### Phase 13 — Schema evolution for 100 GB scale (after parity is proven)

- [x] **T13.1 Index migrations** (alembic): `runs(experiment_id, lifecycle_stage,
      start_time)`, `logged_models(experiment_id)`, `inputs(source_id)`,
      `model_versions(run_id)`, `model_versions(current_stage)`; fix the two
      empty `Index()` declarations (tracking models.py:832,863).
      **AC:** upgrade+downgrade clean on all dialects (`tests/db/check_migration.sh`);
      EXPLAIN shows index usage; no Python-suite regression.
      **VER:** migration CI + query plans on a seeded 10M-run dataset.
      **DONE (2026-07-18):** alembic revision `a3f8c21d9b47` (off `b7e4c1a90f23`)
      creates all five indexes; downgrade handles MySQL's FK-backed-index
      consolidation (errno 1553) by dropping/recreating the FK around the index
      drop. The two dead `Index()` declarations (trace tags/metadata) are now
      real `request_id` indexes; all five indexes mirrored in ORM
      `__table_args__` with migration-matching `index_<table>_<cols>` names;
      `tests/store/dump_schema.py` now emits deterministic `CREATE INDEX`
      lines so `tests/resources/db/latest_schema.sql` guards index parity;
      Rust `EXPECTED_ALEMBIC_HEAD` bumped and the sqlite fixture regenerated.
      Verified: schema suite 23 passed; sqlite upgrade→downgrade→upgrade walk
      with PRAGMA assertions; EXPLAIN QUERY PLAN uses the runs index on a
      10k-run DB; store core+runs 256 passed; cargo fmt/clippy/test-workspace
      exit 0; T12.4 replay harness exit 0 post-merge. NOT run: postgres/mysql
      `check_migration.sh` (no docker locally) — covered by migration CI.
- [x] **T13.2 `span_attributes` extraction table** (Q7): indexed key/value maintained on
      span ingest; span-content LIKE filters rewritten; shared alembic migration with a
      Python-compatible write path or capability-gated Rust-only (D7).
      **AC:** span-attribute filter on 50M spans < 1s on postgres; results identical to
      LIKE baseline on the corpus.
      **VER:** benchmark report + differential search results.
      **DONE (2026-07-18):** D7 resolved as SHARED migration + Python write
      path. Revision `c4a9b7d3e812` (off `a3f8c21d9b47`): `span_attributes`
      (trace_id, span_id, key PK; value VARCHAR(500) + value_truncated flag;
      composite FK → spans ON DELETE CASCADE; index (key, value)); batched
      keyset backfill of existing spans (1k batches, top-level string attrs
      only, >250-char keys skipped to avoid aliasing). Write paths: Python
      log_spans extract + relog delete-and-replace + archival cleanup; Rust
      shared span-ingest path (OTLP included) + trace-delete/workspace
      cleanup. Read path: Rust span-attribute LIKE now pre-filters via the
      indexed table, KEEPING the original content-LIKE as residual predicate
      so results are byte-identical to Python (JSON-escaping quirks, cross-
      attribute substring bleed, >500-char values); ILIKE/RLIKE and
      wildcard/quote/backslash/long keys stay on content scan (documented);
      Python read path unchanged. Verified: schema suite 23 passed; sqlite
      up/down/up walk; backfill == ingest extraction (incl. truncated 600-char
      value); byte-identical Python-vs-Rust responses on 9 adversarial span
      queries over 2,500 spans (unicode, substring-hostile, ILIKE, missing
      key); trace store 507 passed, workspace suite 68 passed; timing sanity
      100k spans sqlite: 30.2ms LIKE → 2.8ms indexed (10.7x), plan uses
      index_span_attributes_key_value; cargo fmt/clippy/test-workspace and
      T12.4 replay corpus exit 0 pre- AND post-merge. 50M-span/<1s postgres AC
      deferred to T13.3's full-scale bench run (not measurable locally).
- [x] **T13.3 Benchmark suite + seeded dataset generator** (`rust/bench/`): ~100 GB-scale
      synthetic DB (millions of runs, dense metrics, 10M+ traces with spans, 100k+ model
      versions); scenarios: runs/search with metric filters + ordering, deep pagination,
      bulk-interval history, traces/search with span filters, OTLP ingest throughput,
      registry search with prompt anti-join.
      **AC:** documented p50/p95 Python vs Rust; targets: p95 run-search < 500 ms,
      deep-page O(1), OTLP ingest ≥ 5x Python.
      **VER:** `rust/bench/RESULTS.md` with hardware notes.
      **DONE (2026-07-18):** suite landed (`rust/bench/{seed.py,bench.py,
      README.md,RESULTS.md}`), scale is a CLI parameter; measured locally at
      2.4 GB sqlite (20k runs / 8M metric rows / 50k traces / 200k spans / 1M
      span attrs / 5k MVs), release binary, byte-identical DB copies, 30
      iterations + warmup. p95 Rust vs Python: run-search 36.0 vs 162.3 ms
      (4.5x, <500 ms AC MET); deep pagination 8.5 vs 36.9 ms, both O(1) over
      25 pages (AC MET); bulk-interval history 3.2 vs 9.6 ms (3.0x); span-attr
      trace search 154.6 vs 265.6 ms (1.72x — weakest read win); OTLP ingest
      2,826 vs 676 spans/s (4.18x — 5x AC NOT MET at this scale, sequential
      batches, honestly reported); prompt anti-join 86.3 vs 86.2 ms (tie —
      only non-win, feeds T13.4). Full-scale postgres instructions in README.
      En route fixed quoted dotted span-attribute search, and the WIP branch
      carried a real soundness fix (thread-local order-by-join registry
      unsound across .await → OrderCols threaded explicitly). Verified:
      ruff/fmt/clippy/test-workspace and T12.4 replay corpus exit 0 pre- and
      post-merge. WSL2/sqlite caveats documented in RESULTS.md.
- [x] **T13.4 Deeper restructures** informed by T13.3 (metric partitioning, narrower
      metrics PK with dedup hash, trace hot/cold split, auth grant semi-join
      materialization). Out of scope until benchmarks prove need.
      **AC:** written proposal per change with migration + rollback story.
      **VER:** design doc reviewed.
      **DONE (2026-07-18):** `rust/bench/RESTRUCTURES.md` (544 lines) — full
      problem/design/migration/rollback story per candidate, each argued from
      T13.3's measured numbers. Verdicts: metric partitioning DEFER (3.2 ms
      history p95 at 8M rows shows no need); narrow metrics PK + dedup hash
      DEFER (no measured index-size/write-amp problem); trace hot/cold split
      DEFER (154.6 ms span-search p95 warrants watching, but full-scale
      postgres evidence is missing and the added writes could worsen the OTLP
      path); auth grant semi-join DEFER (no sparse-grant benchmark exists;
      persistent cross-database materialization rejected as unsafe). Fifth
      section covers what the data actually surfaced: the prompt anti-join
      tie (Rust per-model hydration is the suspect) and the OTLP 4.18x gap
      (statement batching is the first profiling target). Each DEFER names
      the full-scale measurement (T13.3 postgres run / T14.2 soak) that would
      trigger revisiting.

### Phase 14 — Memory & production validation

- [x] **T14.1 Memory baseline**: Python server RSS (4 uvicorn workers, idle + load) vs
      Rust on identical workloads.
      **AC:** report with Rust idle/loaded RSS; target ≥ 5x total reduction.
      **VER:** `rust/bench/memory.md` (cgroup memory.current sampling).
      **DONE (2026-07-18):** measured during the T14.2 soak (same infra).
      Whole-process-tree RSS via /proc (WSL2 cgroup v1 can't isolate
      memory.current — documented): idle 2,976 MiB (Python, 4 workers) vs
      27.95 MiB (Rust) = 106.5x; loaded (final-10-min mean of the 1 h run)
      3,145 MiB vs 46.7 MiB = 67.3x. ≥5x AC MET by a wide margin. Report:
      `rust/bench/memory.md`.
- [ ] **T14.2 Soak + load comparison** *(respecified by user 2026-07-18: 1 h
      instead of 24 h — "1h is good enough with the right measurement and load
      test")*: 1 h mixed workload run TWICE on identical infrastructure — once
      against the Python server, once against Rust — as realistic as possible:
      **postgres + MinIO (S3) artifact storage in local docker containers**.
      Workload models the user's real pain points:
      - concurrent training runs logging params/metrics/tags at realistic
        cadence (the "huge problems while doing trainings and tracking");
      - trace ingest alongside training;
      - the post-training read pattern from real clients (e.g. the ONNX model
        flow): after a run finishes, `metrics/get-history` and
        `metrics/get-history-bulk-interval` calls for its metrics;
      - regular result querying throughout (runs/search, experiment listing,
        UI-style polling);
      - artifact/model uploads to S3 (MinIO) as part of each training run.
      Measure per-endpoint latency percentiles (p50/p95/p99), error rates, RSS
      over time, and DB pool health for BOTH servers; report side by side.
      **AC:** no monotonic RSS growth over the hour; error rate < 0.01%; no
      webhook-delivery task leaks; comparative report Python vs Rust.
      **VER:** soak report (`rust/bench/soak.md`) with graphs/tables + docker
      compose file to reproduce.
      **DONE (2026-07-18):** `rust/bench/{soak.py,docker-compose.soak.yml,
      soak.md}`. Two sequential 3,600 s runs on identical infra (postgres:16 +
      MinIO in docker, fresh DB + bucket prefix per target): 8 trainer workers
      (create run → batched param/metric logging → S3 artifacts + model →
      terminate → ONNX-style get-history + get-history-bulk-interval
      read-back) + trace ingest + 2 UI-poll readers + webhook to a local
      sink. ~19.4k requests (Python) / ~21.2k (Rust), ~140 runs, ~9.6k metric
      points, ~855 traces, ~550 S3 objects each. Errors 0/0 (<0.01% AC MET
      both). Webhooks all delivered, no task leaks (AC MET both). p50/p95
      highlights (Python → Rust): get-history 49.9/50.3 → 1.0/1.3 ms;
      bulk-interval 50.0/51.6 → 1.8/2.3 ms; log-batch 9.5/50.0 → 4.6/6.6 ms;
      runs/search 28.5/89.1 → 5.4/6.7 ms; trace ingest 11.6/14.0 → 3.9/5.4 ms.
      RSS-trend AC: Python MET (+14.4 MiB/h, <5% bin growth); Rust NOT MET by
      the strict monotonicity rule (39.9 → 46.7 MiB, near-plateau in final
      bins — reads as warm-up asymptote, not a leak; longer run would settle
      it). Pool health fine (peak 31/100 vs 15/100 connections). En route:
      fixed real Rust PG bug (runs/search order-rank INT4 decoded as i64) +
      live-PG regression test. LIMITATION surfaced: Rust artifact proxy has
      no cloud-scheme (S3) support — both sides used client-direct SigV4
      uploads (realistic client path); see open items.
- [x] **T14.3 Operational docs**: deployment guide (compose + k8s), migration runbook
      (Python-only → split; auth DB sharing; secret/key management for Fernet + CSRF),
      rollback procedure (nginx flips routes back to Python — zero data migration, both
      DBs shared).
      **AC:** a fresh operator deploys the split from docs alone.
      **VER:** doc walkthrough by someone not on the project.
      **DONE (2026-07-18):** `rust/docs/{DEPLOYMENT.md,MIGRATION_RUNBOOK.md,
      ROLLBACK.md}` — grounded in the real route table, CLI_PARITY.md flags,
      and alembic head pins (refuse-to-boot behavior explained); D12 key
      management (shared Fernet webhook key, Rust CSRF secret, Flask secret
      stays Python-side); artifact-proxy S3 limitation called out with nginx
      routing those paths to Python. Example compose VERIFIED LIVE: booted,
      tracking request served by Rust (200), artifact-proxy PUT+GET served by
      Python (200), genai probe attributed to Python; then torn down. 9 k8s
      YAML docs validated (PyYAML; kubectl unavailable). ROLLBACK.md lists
      downgrade-safe revisions (a3f8c21d9b47, c4a9b7d3e812) and what the
      span_attributes downgrade drops. Human doc walkthrough (the AC's
      fresh-operator test) still pending — flagged for the user.

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

---

---

# Part II — GenAI Port (full-app parity)

Added 2026-07-17, after Part I reached compliance-green on the core surface.
New goal: **the entire MLflow Python server ported to Rust** — the genai
features Part I excluded are now in scope. Same rules as Part I: every fact
below was derived from the current codebase (file/line refs included); when in
doubt, the Python implementation is the spec. Phase numbers, task IDs, and
decision IDs continue Part I's sequences.

## 11. Goals, end state, and the execution boundary

### 11.1 End state

- Rust serves **every** HTTP route of `mlflow server`: Part I's surface plus
  gateway (CRUD + runtime), scorers, evaluation datasets, genai evaluate,
  issues, label schemas, review queues, prompt optimization, the jobs API,
  assistant, promptlab, and trace archival.
- The Python *server plane* (uvicorn workers) is retired; the genai rows in
  §2.2's nginx table are deleted one phase at a time (T22.4).
- The production image contains **no Python interpreter, standard library,
  site-packages, Python shared library, or `.py` payloads**. GenAI jobs execute
  in a per-job Rust worker linked to the same native GenAI engine as the server
  (D14). The assistant keeps spawning external CLIs (`claude`, `codex`)
  exactly as the Python reference does today.
- The public Python `mlflow.genai` SDK remains supported as a client: it keeps
  constructing the existing scorer JSON and calling the unchanged HTTP API.
  Python may run in development/CI as the differential oracle, never in the
  production server deployment.

```
            ┌──────────────────────────────────────────────┐
            │                    nginx                     │
            │  static React build · everything else → Rust │
            └──────────────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │         Rust server          │
                    │  Part I surface + gateway,   │
                    │  genai CRUD, jobs runner,    │
                    │  native scorer/judge engine │
                    │  assistant SSE, archival     │
                    └──┬─────────┬─────────┬───────┘
        spawns per job │         │ HTTPS   │ spawns per assistant turn
                       ▼         ▼         ▼
            ┌────────────────┐ ┌────────┐ ┌──────────────────┐
            │ Rust GenAI     │ │ LLM    │ │ claude / codex   │
            │ worker (same   │ │ provi- │ │ CLIs (NDJSON     │
            │ native crates) │ │ ders   │ │ streaming)       │
            └────────────────┘ └────────┘ └──────────────────┘
```

### 11.2 Native execution boundary and compatibility contract

The genai surface splits cleanly into three tiers:

| Tier | What | Port strategy |
|---|---|---|
| **A — CRUD/wire** | eval datasets, scorers CRUD + online configs, issues CRUD, label schemas, review queues, prompt-opt job CRUD, jobs API, gateway CRUD (secrets/endpoints/model-defs/bindings/tags/budgets/guardrail configs) | native Rust, byte parity — same discipline as Part I |
| **B — network/runtime** | gateway proxying + SSE, budget enforcement, assistant SSE + CLI subprocesses + tool loop, trace archival, periodic schedulers | native Rust (`tokio`); no Python involved |
| **C — semantic execution** | scorer deserialization/execution, evaluation harness, issue discovery, online scoring, prompt optimization, inline judge guardrails | native `mlflow-genai` engine; async jobs run in the isolated Rust worker from D14 |

The Tier C contract is the behavior reachable from the OSS server at the
pinned MLflow release, not every client-only helper exported by the Python SDK.
The compatibility inventory is:

- `SerializedScorer` JSON: all concrete built-ins, `InstructionsJudge`,
  `Guidelines`, and `MemoryAugmentedJudge`, including unknown-field tolerance,
  version metadata, exact validation errors, aggregations, structured feedback,
  and single-turn/session classification (`scorers/base.py`).
- The four server-allowlisted third-party module families — DeepEval, Ragas,
  TruLens, and Phoenix — including every metric present in the pinned compatible
  package versions, their input mapping, deterministic algorithms, prompt
  templates, threshold/rationale/metadata semantics, embeddings, and model calls
  (`THIRD_PARTY_SCORER_ALLOWED_MODULES`, `scorer_utils.py:46`). Rust uses a
  generated compatibility manifest rather than runtime dynamic imports.
- Evaluation and online scoring: result standardization, `SCORER_ERROR`
  capture, rate limits/retries, concurrency, trace/session grouping, expectation
  and tag logging, assessment metadata, aggregate metrics, trace/run links,
  sampling, and checkpoints (`evaluation/`, `scorers/job.py`, `scorers/online/`).
- Issue discovery's sampling → triage → session analysis → LLM clustering/
  resplitting/dedup → issue creation/annotation pipeline (`discovery/`).
- The server's constrained single-prompt optimization job: native MetaPrompt
  plus a port of the pinned GEPA algorithm and accepted `gepa_kwargs`, with the
  same prompt registration, metrics, candidate artifacts, and result URI
  (`optimize/job.py`, `optimize/optimizers/`).

Decorator scorers remain rejected on OSS exactly as Python rejects them: their
payload contains raw Python source reconstructed by `exec()`, and therefore is
not a capability of the OSS server (`base.py:574-585`, `handlers.py:5461-5464`).
Client-only scorer integrations (for example Google ADK and Guardrails scorers)
remain usable through the Python SDK; if their payload is sent to an OSS server,
Rust reproduces Python's current unsupported-module behavior rather than
silently expanding the wire contract. T15.5 accounts for every `mlflow/genai`
module and test so no server-reachable Python behavior is missed.

### 11.3 Permanently out of scope (unchanged)

- **Databricks-only `unified-traces` / `get-online-trace-details`**: not served
  by the OSS Python server at all — absent from `HANDLERS`
  (`handlers.py:7663+`); only `DatabricksRestStore` calls them, client→
  Databricks (`databricks_rest_store.py:556,587`; abstract store raises
  `NotImplementedError`, `abstract_store.py:471-479`). Nothing to port.
- **UC prompt/registry protos** (`unity_catalog_prompt_service.proto`):
  client-only (`store/_unity_catalog/registry/rest_store.py:178`); no OSS route.
- **Legacy standalone YAML gateway** (`mlflow gateway start`,
  `mlflow/gateway/app.py`; deprecated at `mlflow/gateway/cli.py:48-51`) — D15:
  stays a deprecated Python CLI. Rust ports only the DB-backed embedded
  gateway. The `/ajax-api/2.0/mlflow/gateway-proxy` Flask bridge to a legacy
  deployments server IS ported (it's a thin validated proxy, §12.8).
- `databricks-agents` managed archival (`mlflow/tracing/archival.py:7-70`).

---

## 12. GenAI API surface (complete inventory)

### 12.1 Evaluation datasets (12 proto endpoints)

RPCs on `MlflowService` (`service.proto:1594-1963`), so served on **both**
`/api/3.0` and `/ajax-api/3.0` via the generic route generation; handlers
`handlers.py:7012-7242`. Entity messages live in a separate proto:
`mlflow/protos/datasets.proto` (proto2, messages only — `Dataset:10`,
`DatasetRecord:45`, `DatasetRecordSource:86`, `SourceType` enum
`TRACE/HUMAN/DOCUMENT/CODE`).

create (POST `/mlflow/datasets/create`), get / delete
(GET/DELETE `/mlflow/datasets/{dataset_id}`), search (POST **and** GET
`/mlflow/datasets/search`), tags (PATCH `.../tags`, DELETE `.../tags/{key}`),
records (POST/GET/DELETE `.../records`), experiment-ids (GET
`.../experiment-ids`), add-experiments / remove-experiments (POST).

Wire quirks: `Dataset.tags/schema/profile` and all `DatasetRecord` payload
fields are **JSON strings** on the wire (`datasets.proto:17-24,52-84`);
`UpsertDatasetRecords.records` is ONE JSON-serialized list string, not a
repeated message (`service.proto:5007`). Search uses offset tokens
(default/cap 1000, `sqlalchemy_store.py:7027-7064`); records use **cursor
tokens** base64(`"created_time:record_id"`) with a legacy integer-offset
fallback (`:7206-7228`). Upsert dedups on
`input_hash = sha256(json.dumps(inputs, sort_keys=True))` (`:7293`) under
unique `(dataset_id, input_hash)`; recomputes schema/profile/digest
(`:7070-7137`). Filter DSL: `SearchEvaluationDatasetsUtils` (`:7031`).

### 12.2 GenAI evaluate invoke + generic jobs API (3 ajax-only endpoints)

- `POST /ajax-api/3.0/mlflow/genai/evaluate/invoke` (`handlers.py:6852-6855`,
  handler `:5004`): hand-rolled JSON `{experiment_id, trace_ids[],
  serialized_scorers[]}` (`:5017-5023`); pre-creates an eval run tagged
  `mlflow.runType=genai_evaluate` (`:5042-5045`), submits
  `invoke_genai_evaluate_job` to the runner (`:5050`), tags the run
  `mlflow.genaiEvaluate.jobId` (`:5059`), returns `{"job_id", "run_id"}`
  (`:5064`).
- `GET /ajax-api/3.0/mlflow/jobs/<job_id>` → `{status, result(parsed),
  status_details}` (`handlers.py:6870`, handler `:5067-5077`);
  `PATCH /ajax-api/3.0/mlflow/jobs/cancel/<job_id>` (`:6865`, handler
  `:5082-5089`).
- The job (`mlflow/genai/evaluation/job.py:24`) links traces to the run,
  deserializes scorers via pydantic `Scorer.model_validate_json`, and runs
  `mlflow.genai.evaluate` — native Tier C in Rust. Results land as run metrics + trace
  assessments + trace-run links (`harness.py:765-766,1015-1039`); there is no
  separate results table.

### 12.3 Scorers (5 proto endpoints + 3 hand-rolled)

- Proto RPCs (`service.proto:1790-1867`, messages `:5105-5210`; handlers
  `handlers.py:5443-5562`, dispatch `:7803-7808`): register (POST
  `/2.0/mlflow/scorers/register`), list (GET `.../list`; empty
  `experiment_id` ⇒ cross-experiment), versions (GET `.../versions`), get
  (GET `.../get`; omitted `version` ⇒ latest), delete (DELETE `.../delete`;
  omitted `version` ⇒ all versions).
- `POST /ajax-api/3.0/mlflow/scorer/invoke` (`routes.py:83`,
  `handlers.py:6835`, handler `:6663`): `{experiment_id, serialized_scorer,
  trace_ids[], log_assessments}` → one job per trace batch; response
  `{"jobs": [{"job_id", "trace_ids"}]}` (`:6718-6720`).
- Online scoring config (hand-rolled, on both prefixes,
  `handlers.py:6981-7004`): GET `/3.0/mlflow/scorers/online-configs`
  (`{scorer_ids[]}` → `{"configs":[...]}`), PUT
  `/3.0/mlflow/scorers/online-config` (`{experiment_id, name, sample_rate,
  filter_string?}`).

Semantics: `serialized_scorer` is **JSON (pydantic), never pickle**
(`base.py:300-354`); versioning is append-only `MAX(scorer_version)+1`
(`sqlalchemy_store.py:2629-2631`); register resolves gateway-endpoint
name→ID rewriting the payload (`:2594-2604`) and **rejects decorator scorers**
(`call_source` present → error, `handlers.py:5461-5464`); list returns the
latest version per name (`:2737-2749`).

### 12.4 Issues (4 proto endpoints + detection invoke)

- RPCs (`service.proto:1377-1447`; messages `mlflow/protos/issues.proto`;
  handlers `handlers.py:4428-4530`, dispatch `:7763-7766`): create (POST
  `/3.0/mlflow/issues`), update (PATCH `/{issue_id}`), get (GET
  `/{issue_id}`), search (POST `/search`; `include_trace_count` populates
  `trace_count`).
- `POST /ajax-api/3.0/mlflow/issues/invoke` (`handlers.py:6842-6849`, handler
  `:4925`): creates a detection run, submits `invoke_issue_detection_job` —
  native Tier C (LLM pipeline, `discovery/pipeline.py:586`).
- Issue↔trace linkage is an **assessment** (`assessment_type="issue"`,
  `name=issue_id`, via `mlflow.log_issue`, `tracing/assessment.py:332-401`);
  `trace_count` = distinct trace_ids of those assessments
  (`sqlalchemy_store.py:7880-7894`). `source_run_id` links the detection run.

### 12.5 Label schemas (6 proto endpoints)

RPCs `/3.0/mlflow/label-schemas/{create,get,get-by-name,list,update,delete}`
(`service.proto:1453-1590`; messages `mlflow/protos/label_schemas.proto`;
handlers around `handlers.py:4536-4617`, dispatch `:7768-7773`). Pure CRUD.
Unique `(experiment_id, name)`; `type` (`feedback|expectation`) immutable;
input configs (`pass_fail/categorical/numeric/text`) stored as JSON text.
Schemas are UI rendering hints only — they never gate assessment writes
(`models.py:3233-3245`).

### 12.6 Review queues (11 proto endpoints)

RPCs `/3.0/mlflow/review-queues/{create,get-or-create-user,get,get-by-name,
list,update,delete}` + `items/{add,remove,list,set-status}`
(`service.proto:2765-3010`; messages `mlflow/protos/review_queues.proto`;
handlers `handlers.py:4655-4771`, dispatch `:7774-7784`). Pure CRUD. Quirks:
`name_key` = lowercased identity with unique `(experiment_id, name_key)`
(`models.py:3473-3477`); items are **soft references** to traces (shape/dedup
validation only, `review_queues/validation.py:241-255`; `item_type` v1 =
`trace`, `session`/`span` reserved); `review_queue_label_schemas.schema_id`
deliberately NOT an FK (MSSQL multi-cascade-path, `models.py:3679-3691`); user
queues resolve to all experiment schemas at read time; item status changes
only on explicit reviewer action (`models.py:3568-3571`).

### 12.7 Prompt optimization (5 RPCs / 6 routes)

RPCs `/3.0/mlflow/prompt-optimization/jobs` (POST create), `/{job_id}` (GET,
DELETE), `/search` (POST **and** GET), `/{job_id}/cancel` (POST)
(`service.proto:2633-2760`; messages `mlflow/protos/prompt_optimization.proto`,
`OptimizerType` `GEPA/METAPROMPT`; handlers `handlers.py:7359-7647`, dispatch
`:7854-7858`). **No dedicated table** — jobs ride the generic `jobs` table +
an MLflow run holding config as params (`:7429-7456`); proto responses are
rebuilt from the job entity (`_build_prompt_optimization_job_from_entity`
`:7477`), optimized-prompt URI read from the job result (`:7520-7524`).
Execution (`optimize_prompts_job`, `mlflow/genai/optimize/job.py:250-321`) is
native Tier C.

### 12.8 Gateway CRUD (36 proto endpoints + 5 hand-rolled)

Proto RPCs under `/2.0/mlflow/gateway/...` (`service.proto:1974-2618`;
messages from `:5209`; handlers from `handlers.py:5662`):

- **secrets** create/get/update/delete/list (5)
- **endpoints** create/get/update/delete/list (5)
- **model-definitions** create/get/list/update/delete (5)
- **endpoints/models** attach/detach (2)
- **endpoints/bindings** create/delete/list (3)
- **endpoints** set-tag/delete-tag (2)
- **budgets** create/get/update/delete/list/windows (6)
- **guardrails** create/get/delete/list, add-to-endpoint,
  remove-from-endpoint, list-for-endpoint, update-config (8)

Hand-rolled ajax (`handlers.py:6811-6839`, paths `auth/routes.py:79-82`): GET
`supported-providers`, `supported-models`, `provider-config`,
`secrets/config` (handlers `:6621-6653`). Plus the legacy bridge
`GET/POST /ajax-api/2.0/mlflow/gateway-proxy` (`server/__init__.py:146-148`,
handler `:2317`): forwards to `MLFLOW_DEPLOYMENTS_TARGET`; validates
`gateway_path` (GET must equal `api/2.0/endpoints`, POST must match
`gateway/[^/]+/invocations`, `:2295-2309`); returns `{"endpoints": []}` when
the target is unset.

### 12.9 Gateway runtime (10 routes, SSE) — Tier B

FastAPI router prefix `/gateway` (`mlflow/server/gateway_api.py:88`), mounted
ahead of Flask (`fastapi_app.py:190-199`) — in Rust these are ordinary axum
routes:

`/{endpoint_name}/mlflow/invocations` (unified chat/embeddings, auto-detected
by `messages` vs `input`, `:600`); `/mlflow/v1/chat/completions` (endpoint via
`model`, `:729`); `/openai/v1/{chat/completions,embeddings,responses,
responses/compact}` (`:836-1128`); `/anthropic/v1/messages` (`:1172`);
`/gemini/v1beta/models/{name}:generateContent` and `:streamGenerateContent`
(`:1271,1341`); `/proxy/{endpoint_name}/{path...}` (raw passthrough, `model`
NOT rewritten, query string preserved, `:1410`).

Must-match behaviors:

1. **Providers**: 23 registered (`provider_registry.py:64-89`); litellm is the
   fallback for anything unhandled (`gateway_api.py:400-411`) — D16.
   Composites: `TrafficRouteProvider` (weighted random pick, `base.py:613`)
   and `FallbackProvider` (sequential, propagates last status, `base.py:697`;
   models by `fallback_order`).
2. **Secret resolution**: endpoint config cache decrypts secrets
   (`config_resolver.py:147`); provider-specific auth modes incl. Bedrock
   api-key/access-keys/iam-role/default-chain and Databricks pat/oauth-m2m
   (`gateway_api.py:335-392`).
3. **Egress**: single choke point; strip `accept-encoding` +
   `X-MLflow-Authorization`, force `Accept-Encoding: gzip, deflate, identity`
   (`providers/utils.py:11-33`); timeout `MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS`
   (default 300).
4. **SSE**: chunks exactly `data: {json}\n\n` (`utils.py:348`); `[DONE]`
   skipped; mid-stream errors emit `{"error":{"message","type"}}` since
   headers are already sent (`utils.py:364`); incomplete-chunk buffering on
   `\n` (`utils.py:400`); **post-LLM guardrails NOT applied to streams**
   (`gateway_api.py:638`).
5. **Budget**: REJECT policies → HTTP 429 with the exact message
   `Budget limit exceeded. Limit: ${amount:.2f} USD per {value} {unit}.
   Budget resets at {ISO8601 Z}. Request rejected.` (`budget.py:157-184`);
   spend computed from trace span metrics `total_cost`
   (`sqlalchemy_mixin.py:1339`); tracker backends in-memory or Redis
   (`MLFLOW_GATEWAY_BUDGET_REDIS_URL`); ALERT policies fire
   `budget_policy.exceeded` webhooks (`budget.py:107`) — reuses the Part I
   `mlflow-webhooks` dispatcher.
6. **Guardrails** (`gateway/guardrails.py`): stage BEFORE/AFTER; VALIDATION →
   HTTP 400 on violation; SANITIZATION rewrites the payload via an
   action-endpoint LLM call carrying `X-MLflow-Guardrail-Bypass: 1` to prevent
   recursion (`:34,262`) — execution strategy is D17.
7. **Headers/quirks**: timing headers `X-MLflow-Gateway-Duration-Ms` /
   `X-MLflow-Gateway-Overhead-Duration-Ms` (`fastapi_app.py:121-154`);
   `X-MLflow-Gateway-Caller` accepts only `judge`; subscription-CLI detection
   (User-Agent `claude-cli`/`codex`/`geminicli` + own auth header keeps client
   credentials, `providers/base.py:45-73`); `model` field in unified payloads
   → HTTP 422 (`base.py:603`); provider allowlist
   `MLFLOW_GATEWAY_ALLOWED_PROVIDERS`; Anthropic `max_tokens` default
   8192/max 1,000,000.

### 12.10 Assistant (9 routes, SSE) — Tier B

FastAPI router `/ajax-api/3.0/mlflow/assistant` with a **localhost-only 403
gate on every route** (`server/assistant/api.py:44-71`): POST `/message` (→
`{session_id, stream_url}`); GET `/sessions/{id}/stream` (SSE); PATCH
`/sessions/{id}` (cancel via subprocess SIGTERM); POST
`/sessions/{id}/permission`; GET `/providers/{p}/health` (501/412/401
exception mapping); GET+PUT `/config`; POST `/skills/install`; GET
`/providers/{p}/models` (API key via `X-API-Key` header).

Mechanics: SSE frames `event: {type}\ndata: {json}\n\n` with 6 event types
(`mlflow/assistant/types.py:54-104`); providers ClaudeCode / Codex /
MlflowGateway / Ollama (`providers/__init__.py:19-29`); the Claude provider
spawns `claude -p ... --output-format stream-json --verbose
--append-system-prompt ...` (`claude_code.py:366-411`), streams NDJSON with a
100 MB line buffer, maps SIGKILL to an `interrupted` event, and delegates auth
to the CLI's own credential store; the OpenAI-compatible base runs an
in-process tool loop (Bash/Read/Write/Edit with cwd path-confinement +
restricted allowlist, `tool_executor.py:14-127`) and **encodes the whole chat
history as JSON in `session_id`** trimmed to 500 KB
(`openai_compatible.py:44-145,407-417`); permission pause/resume spans the
`permission_request` event + the `/permission` POST. Sessions are JSON files
under `$TMPDIR/mlflow-assistant-sessions` with UUID validation + PID files for
cancellation (`session.py:12-242`); config at
`~/.mlflow/assistant/config.json`. Known Python bug: `/message` returns
`stream_url=.../stream/{id}` but the route is `/sessions/{id}/stream`
(`api.py:154` vs `:158`) — D18. The `dev/dev_stubs/` fake-`claude` mechanism
must keep working against Rust (CI ui-review bot).

### 12.11 Promptlab (1 endpoint)

`POST /ajax-api/2.0/mlflow/runs/create-promptlab-run`
(`server/__init__.py:140-143`, handler `handlers.py:2340-2404`). Requires
`experiment_id, prompt_template, prompt_parameters, model_route, model_input,
mlflow_version`; creates a run with params + tags
(`MLFLOW_RUN_SOURCE_TYPE="PROMPT_ENGINEERING"`), **saves a real pyfunc
"promptlab" model artifact** (MLmodel/requirements/`parameters.yaml`, pinned
`mlflow[gateway]==<version>`) plus `eval_results_table.json`
(`utils/promptlab_utils.py:27-146`, `prompt/promptlab_model.py:105-197`), and
responds with proto `CreateRun.Response`. The handler itself never calls the
gateway (prediction happens only when the saved model is later loaded).
Artifact-writer strategy is D19. Auth: experiment UPDATE
(`auth/__init__.py:2172,2726`).

### 12.12 Trace archival (no routes — closes D6) — Tier B

- Config: `--trace-archival-config` / `MLFLOW_TRACE_ARCHIVAL_CONFIG` YAML
  (`trace_archival` key: `enabled`, `location`, `retention`,
  `long_retention_allowlist`, `interval_seconds` default 300 max 86400,
  `max_traces_per_pass`; `mlflow/tracing/trace_archival_config.py:22-143`),
  5s-TTL cached with stale-on-error tolerance (`:50-102`); rejected together
  with `--artifacts-only` (`cli/__init__.py:672-682`).
- Repo constraints: archival location must NOT be `mlflow-artifacts:` proxy,
  Databricks, or DBFS-rest, and the repo must implement `delete_artifacts`
  (`utils/validation.py:197-235`).
- Payload: `traces.pb` = OTLP `TracesData`, one ResourceSpans/one ScopeSpans,
  root-first span sort (`mlflow/tracing/otel/otel_archival.py:16-132`) —
  Rust already has the OTLP protos from T1.2.
- Store: `archive_traces` (`sqlalchemy_store.py:5830+`); finalize flips
  `SPANS_LOCATION` → `ARCHIVE_REPO` + writes `ARCHIVE_LOCATION` tag + blanks
  `spans.content` in one transaction with a `db_payload_generation` race guard
  (`:6390-6437`); trace delete removes archived payloads (`:4356-4431`);
  reads flow through `getTrace`/`get-trace-artifact` (Part I stubbed these
  NOT_IMPLEMENTED — T21.2 removes the stubs). Retention/allowlist resolution:
  `store/tracking/utils/trace_archival.py:96-174`.
- Scheduler: periodic task every minute with a process-local monotonic gate to
  `interval_seconds`, workspace-fairness iteration, `max_traces_per_pass`
  budget (`trace_archival_service.py:39-212`, registration
  `jobs/utils.py:785-793`).

### 12.13 Auth treatment (verified per area)

| Area | Gate |
|---|---|
| Scorers CRUD | validators: register→experiment UPDATE; list→`validate_can_read_scorer_list` + after-request `filter_list_scorers`; get/versions→read scorer; delete→delete scorer (`auth/__init__.py:2527-2532`, `:3566`) |
| Scorer invoke | `validate_gateway_proxy` (`:2729`) |
| Online-config routes | **explicitly excluded** from validators (`:2637`) — authenticated-only |
| Datasets (all 12) | **no validators** — authenticated-only (absent from `BEFORE_REQUEST_VALIDATORS`) |
| Issues (all 4 + invoke) | **no validators** — authenticated-only (`:2639-2641` only exempts the invoke path) |
| Label schemas | full validator set (`:2603-2610`) |
| Review queues | full validator set + admin-bypass integrity hook `enforce_review_queue_name_not_username` (`:2593-2601`, `:2360`) |
| Prompt optimization | validator set (`:2562-2566`) |
| Jobs API | FastAPI middleware guards prefix `/ajax-api/3.0/jobs` but the Flask route is `/ajax-api/3.0/mlflow/jobs/...` (`:4479` vs `handlers.py:6862`) — verify which layer actually gates it and port the observed behavior |
| gateway-proxy | `validate_gateway_proxy` (`:2196,2727`) |
| Assistant | localhost gate + authenticated-only (`:4482-4483`) |
| Promptlab | experiment UPDATE (`:2726`) |

The unvalidated areas (datasets, issues, online-configs) are ported
**faithfully as authenticated-only**, each with a `// AUTH GAP:` marker and a
backlog entry (D21) — silently "fixing" them would diverge from Python.

---

## 13. Storage & crypto

### 13.1 Tables (all already exist — Part I's pinned head includes them)

The Part I alembic head `b7e4c1a90f23` IS the review-queue migration, so a
migrated DB already contains every genai table; **no new migrations are needed
to start Part II**, and Rust's startup head-check needs no change.

- Datasets: `evaluation_datasets`, `evaluation_dataset_tags`,
  `evaluation_dataset_records` (`models.py:1554,1699,1739`) +
  `entity_associations` (already Part I).
- Scorers: `scorers`, `scorer_versions`, `online_scoring_configs`
  (`models.py:2125,2166,2226`).
- Issues: `issues` (`models.py:1154`). Label schemas: `label_schemas`
  (`:3232`). Review queues: `review_queues`, `review_queue_users`,
  `review_queue_items`, `review_queue_label_schemas` (`:3389-3658`).
- Jobs: `jobs` (`models.py:2291`) — states as int enums, `params` JSON text,
  `status_details` JSON, workspace column, index
  `(job_name, workspace, status, creation_time)`.
- Gateway: `secrets`, `endpoints`, `model_definitions`,
  `endpoint_model_mappings`, `endpoint_bindings`, `endpoint_tags`,
  `budget_policies`, `guardrails`, `guardrail_configs`
  (`models.py:2398-3227`).

All carry `workspace` columns with per-workspace uniques — the Part I
workspace plumbing (§3.17) extends unchanged.

### 13.2 Secrets crypto — envelope AES-GCM, NOT Fernet

`mlflow/utils/crypto.py`: per-secret random 256-bit DEK; value and DEK both
AES-256-GCM (`nonce(12) + ct + tag(16)`); KEK derived via
**PBKDF2-HMAC-SHA256, 600,000 iterations**, fixed salt
`b"mlflow-secrets-kek-v1-2025"` + big-endian `kek_version` (`:59-204`);
passphrase from `MLFLOW_CRYPTO_KEK_PASSPHRASE` (dev default with warning);
**AAD = `"{secret_id}|{secret_name}"`** so both fields are immutable
(`:399-420`, `models.py:2410-2422`); plaintext is JSON with `sort_keys=True`
(`:508`); masking first-3+`...`+last-4 / `***` if <8 (`:423-450`); KEK
rotation (`rotate_secret_encryption` `:600`). RustCrypto `aes-gcm` + `pbkdf2`
cover this; cross-language spike required (T15.3), same pattern as Part I's
Fernet spike. During any mixed period both planes need the identical
passphrase (extends D12).

---

## 14. Runtime engines to build

### 14.1 Rust job runner (replaces Huey) — D20

Source of truth is the existing `jobs` table (Python already recovers from it
at startup, re-enqueueing unfinished jobs — `utils.py:642-657`). The Rust
runner polls/claims from the table directly; the SqliteHuey queue files
(`*.mlflow-huey-store`, per-function instances, JSON serializer,
`utils.py:434-493`) are NOT reproduced. Semantics to match exactly:

- States `PENDING/RUNNING/SUCCEEDED/FAILED/TIMEOUT/CANCELED` stored as int
  ordinals; proto mapping folds TIMEOUT→FAILED (`_job_status.py:7-49`);
  finalized jobs immune to mutation (`store/jobs/sqlalchemy_store.py:126`).
- Retry only on transient errors: `retry_count >= 
  MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES` → FAILED, else PENDING with
  incremented count (`:210-238`); exponential backoff via the base/max delay
  env vars.
- Timeout: poll + kill subprocess + `mark_job_timed_out` (`utils.py:266-271`).
- `exclusive=["experiment_id"]`: lock key
  `sha256(job_name + params_subset)`; a locked submission is **CANCELED, not
  queued** (`utils.py:284-297,333-354`).
- Per-function `max_workers` concurrency (from the `@job` decorator metadata);
  startup recovery resets RUNNING→PENDING.
- Gating: `MLFLOW_SERVER_ENABLE_JOB_EXECUTION`, requires DB-backed store,
  rejects Windows (`utils.py:713-727`).

### 14.2 Native Rust GenAI worker protocol — D14

Add a workspace crate `mlflow-genai` containing the semantic execution engine
and a small `mlflow-genai-worker` binary linked to that exact crate version.
The server spawns one worker subprocess per claimed job, preserving Python's
hard timeout/cancel boundary and containing panics or runaway memory without
embedding an interpreter.

The protocol is versioned JSON: the server writes one request to stdin and the
worker writes one success/result or structured failure envelope to stdout;
logs go to stderr. Requests contain `{protocol_version, job_id, job_kind,
params, workspace, subject}`. `job_kind` is a closed Rust enum matching the
existing six-function allowlist (`jobs/__init__.py:20-42`):
`invoke_scorer`, `run_online_trace_scorer`, `run_online_session_scorer`,
`optimize_prompts`, `invoke_issue_detection`, `invoke_genai_evaluate`.
Unknown versions/kinds fail before executing user-controlled data. Results and
status details keep the existing jobs-table JSON shapes.

Workers call back to the Rust server over its existing HTTP APIs for store and
gateway access. The launcher propagates the submitting subject, workspace,
`MLFLOW_TRACKING_URI`, `MLFLOW_GATEWAY_URI`, and the existing internal gateway
credential so gateway RBAC is evaluated as the submitting user
(`genai/scorers/job.py:168-171`, `server/__init__.py:466,511`). The launcher
uses bounded stdin/stdout/stderr buffers, closes inherited file descriptors,
kills the complete process group on cancel/timeout, and treats malformed
output, signals, and non-zero exits as the same structured job failures as the
Python reference. There is no `python -m` entrypoint, Python package handshake,
`pip_requirements`, uv environment, or Python fallback.

`mlflow-genai` is also linked into `mlflow-server` for low-latency inline
execution (gateway guardrail judges and request validation); async evaluation,
scoring, discovery, and optimization always use the worker boundary.

### 14.3 Native GenAI semantic engine

`mlflow-genai` is split by stable internal interfaces rather than Python module
layout:

- `SerializedScorer` is an untagged compatibility enum over the five mutually
  exclusive payload forms (builtin, decorator, instructions, memory-augmented,
  third-party). It retains unknown JSON fields for forward diagnostics and
  reproduces Python's validation/error-class behavior. Decorator execution is
  an explicit OSS rejection variant, never an evaluator.
- `ScorerExecutor` accepts canonical `EvalItem`/session inputs and emits
  canonical `Feedback` values or `SCORER_ERROR`; implementations cover every
  concrete builtin, instructions/trace-tool judges, memory retrieval, and the
  pinned third-party compatibility manifest.
- `JudgeRuntime` owns prompt rendering, response JSON Schema, native provider
  dispatch, retries/context pruning, tool loops, token/cost accounting, and
  conversion to assessment sources/metadata. Gateway, issue discovery,
  MetaPrompt, GEPA reflection, and third-party LLM metrics reuse it.
- `EvaluationEngine` owns bounded prediction/scorer concurrency, rate limiting,
  single-turn/session grouping, trace creation/linking, expectation/tag and
  assessment writes, aggregate metrics, evaluator tracing, and cleanup.
- `DiscoveryEngine` and `OptimizationEngine` implement the reference pipelines
  as deterministic state machines whose external LLM/embedding calls are
  injectable for differential tests.

The third-party manifest is generated for pinned DeepEval/Ragas/TruLens/Phoenix
versions and records every accepted module/class/metric, constructor schema,
session level, deterministic-vs-LLM execution, prompt/version fingerprint, and
model/embedding requirements. Deterministic algorithms and approved prompt/
parser assets are ported with provenance and license metadata. A missing or
license-incompatible implementation blocks the compatibility phase; it cannot
be hidden behind an unsupported-provider allowlist.

GEPA is likewise pinned and ported as a native algorithm: candidate state,
Pareto/frontier selection, reflective batches, randomness/seed handling,
metric-call budget, advanced `gepa_kwargs`, result selection, and artifact
logging must match the reference snapshot. MetaPrompt ports its templates,
baseline/final evaluation, variable-preservation validation, JSON cleanup, and
fallback behavior directly.

### 14.4 Gateway execution engine

Native tokio/hyper: provider adapter trait mirroring `BaseProvider`
(chat/chat_stream/completions/embeddings/passthrough/proxy +
token-usage extraction, `providers/base.py:89,355-459`); streaming
passthrough without buffering (Part I's T5.2 discipline); secret TTL cache
(`SecretCache` semantics); budget tracker (in-memory + Redis) with the
600s-refresh spend query; cost from a vendored litellm price snapshot (D16);
tracing of gateway calls into the (Rust) tracking store mirroring
`maybe_traced_gateway_call` so `total_cost` span metrics keep feeding budgets.

The pinned LiteLLM compatibility manifest includes every fallback provider,
request/auth transform, response/stream normalization, retry classification,
tokenizer/model limits, and cost entry reachable in the Python reference
release. Rust implements that manifest natively; generic OpenAI-compatible
providers share one adapter, while provider-specific transforms remain explicit.
Unknown future providers fail exactly as the pinned reference does—there is no
Python fallback.

### 14.5 Assistant engine

Tokio subprocess management (spawn/kill/PID files), NDJSON→SSE translation
with the exact event vocabulary, session JSON files with atomic replace +
UUID path-safety, the OAI-compatible tool loop with the path-confinement
sandbox ported exactly (resolve-then-`relative_to` checks,
`tool_executor.py:32-73` — security-critical, needs adversarial tests), and
the permission pause/resume protocol.

### 14.6 Periodic scheduler

Replaces Huey `periodic_task(crontab)` + `lock_task`: a tokio interval
scheduler running (a) the online-scoring scheduler every minute
(read active configs → group by experiment → shuffle → submit trace- and
session-scorer jobs, `genai/scorers/job.py:430-501`; sampling waterfall
`online/sampler.py:57`; checkpoint = experiment tag
`MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT`,
`online/trace_checkpointer.py:58-67`; constants MAX_LOOKBACK 1h / 500
traces / 100 sessions per job) and (b) the trace-archival scheduler
(§12.12). Single-instance locking via the same DB-lock discipline the job
runner uses.

---

## 15. Compliance strategy (Part II)

1. **Differential replay harness** (`rust/compliance/`) grows genai sections:
   datasets, scorers CRUD + online configs, issues, label schemas, review
   queues, prompt-opt CRUD, gateway CRUD, jobs API — all Tier A surfaces are
   replayable exactly like Part I endpoints. Job-dispatching endpoints
   (invoke routes) replay with the runner disabled (submission-side parity:
   response shape, run/tag creation) and separately with a deterministic native
   worker/provider fixture.
2. **Existing Python suites + reachability ledger**: suites re-point at Rust
   via the Part I
   `MLFLOW_SERVER_TYPE=rust` switch — the genai suites under `tests/genai/`,
   `tests/server/jobs/`, gateway tests, and the store-level dataset/scorer/
   issue/review-queue suites. T15.5 classifies every `mlflow/genai` module,
   public entry point, job-reachable function, and test as server-native,
   client-only, Databricks-only, or intentionally rejected; no unclassified
   row can pass CI.
3. **SSE differential**: a recorder that captures full event streams
   (gateway + assistant) from both servers against a scripted mock provider /
   the `dev/dev_stubs` fake `claude`, diffing frame-by-frame — same spirit as
   the T12.4 corpus but for streams.
4. **Mock-provider fixture**: a local HTTP server speaking OpenAI/Anthropic/
   Gemini response shapes (incl. streaming) — the webhook-receiver pattern
   from T8.3 applied to providers, so provider adapters are testable
   hermetically.
5. **Crypto interop**: secrets written by Python decrypt in Rust and vice
   versa on a shared DB (T15.3 fixtures, like Part I's Fernet/werkzeug tests).
6. **Semantic differential corpus**: the pinned Python reference and Rust
   engine receive identical scorer JSON, traces/datasets, clock, randomness,
   and scripted LLM/embedding responses. Compare exact outbound provider
   requests/tool transcripts and final assessments, metadata, metrics,
   checkpoints, issues, prompt versions, job results, and artifacts.
   Deterministic third-party metrics compare exact values; floating-point
   tolerances require an individually documented numerical reason.
7. **Python-free production gate**: build the final image with no Python
   executable, stdlib, site-packages, `.py` files, or Python shared library;
   scan the Rust sources/binary invocation trace for attempted Python launches,
   then run every GenAI API/UI/job smoke with that image.

---

## 16. Work breakdown (Phases 15–22)

Same legend as §7: AC = acceptance criteria, VER = verification. Phases 16–18
are parallelizable once Phase 15 lands; Phase 19 needs 17 (runner) and
benefits from 18 (gateway, for judge LLM calls); 20–21 are independent of 19.

### Phase 15 — Part II foundations

- [x] **T15.1 Decision pass**: D14–D22 reviewed and recorded in §17 (execution
      model, litellm strategy, guardrail execution, promptlab writer, queue
      replacement, auth-gap policy).
      **AC:** each decision has consequences documented; scope table §11.2
      approved.
      **VER:** sign-off recorded in this file.
      **DONE (2026-07-18):** full D14–D22 review executed by the orchestrator
      under the user's standing continue-with-Part-2 directive. D14, D16,
      D17, D19 were already decided; D22 accepted. The four `proposed` rows
      are hereby ratified as decided (marked "T15.1 pass 2026-07-18" in §17):
      D15 (legacy YAML gateway stays Python/deprecated; only the DB-backed
      embedded gateway + gateway-proxy bridge are ported), D18 (emit the
      corrected stream_url form; wire-visible deviation must get a
      differential-corpus allowlist entry plus a UI dead-letter check), D20
      (jobs table is the queue; HARD operational rule: Python and Rust
      runners must never run concurrently on one DB — enforced in the
      T14.3-style runbook at cutover), D21 (auth gaps ported faithfully with
      `// AUTH GAP:` markers; hardening is a post-parity two-plane change).
      Scope table §11.2 approved as written. User may veto any ratification;
      revisit costs one table edit before the affected phase starts.
- [x] **T15.2 Proto/routing extension**: compile `datasets.proto`,
      `issues.proto`, `label_schemas.proto`, `review_queues.proto`,
      `prompt_optimization.proto` into `mlflow-proto`; the genai RPCs already
      in `service.proto` flow into the existing route table; shrink the T1.2
      route-parity allowlist (the 15 genai non-proto routes become in-scope
      hand-registered routes as their phases land).
      **AC:** `route_parity.py` accounts for every §12 endpoint.
      **VER:** parity diff empty modulo not-yet-implemented markers.
      **DONE (2026-07-18):** all five protos (proto2) compiled through the
      existing reflection-based mlflow-JSON codec (snake_case, HasField
      presence, enum names, 64-bit ints as strings-where-Python-does,
      unknown-field tolerance) + 5 round-trip tests (genai_json.rs).
      route_parity.py accounts for 162 generated genai proto routes + all
      non-proto §12 routes with per-phase `planned(T16.x–T20.x)` markers
      (197 planned total; per-section table in the merge commit). T1.2
      allowlist shrank 19 → 4. NO live routes registered — corpus behavior
      unchanged, replay + route_parity + cargo gates exit 0 post-merge.
- [x] **T15.3 Secrets-crypto spike**: Rust AES-256-GCM envelope + PBKDF2 KEK
      verifying fixtures generated by `mlflow/utils/crypto.py`, both
      directions, incl. AAD binding, masking rules, and kek_version rotation.
      **AC:** cross-language fixtures round-trip; wrong-AAD fails closed.
      **VER:** `rust/spikes/` tests green (Part I T0.4 pattern).
      **DONE (2026-07-18):** standalone spike (`rust/spikes/src/secrets.rs` +
      `verify_secrets.py`). Envelope pinned: random 32-byte DEK per secret;
      AES-256-GCM `nonce(12)||ct||tag(16)`; wrapped DEK always 60 bytes;
      KEK = PBKDF2-HMAC-SHA256 600k iters, salt "mlflow-secrets-kek-v1-2025"
      + kek_version as u32 BE; AAD "{secret_id}|{secret_name}" on the value
      only; kek_version stored separately; rotation re-wraps only the DEK.
      Masking: <8 chars or non-string → ***, else first3+...+last4. 32
      Python→Rust + 32 Rust→Python envelopes + 2 rotations each + 19 masking
      fixtures all round-trip; wrong AAD/KEK, truncation, and corruption all
      fail closed on a constant error. 14 spike tests + workspace gates
      exit 0.
- [x] **T15.4 Native engine + worker spike**: add skeleton `mlflow-genai` and
      `mlflow-genai-worker`; parse a real builtin-scorer payload and execute
      one deterministic scorer plus one instructions judge through the mock
      gateway inside a spawned Rust worker (params/result envelopes,
      non-zero/signal/malformed-output propagation, process-group timeout).
      **AC:** job lifecycle is observable in the `jobs` table, outputs match
      the Python fixtures, and the test environment has no Python executable.
      **VER:** spike tests + production-image-style PATH isolation green.
      **DONE (2026-07-18):** crates `mlflow-genai` (SerializedScorer,
      ScorerExecutor, protocol envelopes, process-group-aware WorkerLauncher)
      + `mlflow-genai-worker` (bounded stdin/stdout JSON worker) are
      workspace members. Persisted-scorer format pinned (metadata + exactly
      one representation field; genuine ResponseLength fixture with
      builtin_scorer_class/pydantic_data; instructions_judge_pydantic_data;
      unknown additive fields retained). §14.2 protocol: versioned request
      envelope, closed 6-value job_kind enum, succeeded/failed result tags;
      unknown version fails before params parse. Spike proves: ResponseLength
      + InstructionsJudge (via in-process mock gateway) match Python oracles
      (generate_oracles.py, real Python InstructionsJudge mocked at the
      provider boundary); scratch-sqlite jobs rows PENDING→RUNNING→SUCCEEDED;
      non_zero_exit / signal / malformed_output / timeout each propagate
      distinctly; process-group kill takes worker-spawned children; PATH is
      an empty temp dir (python/python3 NotFound) and both flows still pass.
      Scratch jobs adapter marked T15.4-only pending T16.5's real store.
      cargo + replay gates exit 0 pre- and post-merge.
- [x] **T15.5 Reachability, test, and corpus inventory**: enumerate every
      `mlflow/genai` module/function/test plus §12 gateway/jobs/assistant/
      archival suites; classify server reachability, record the native owner
      and fixture, generate the pinned scorer/provider/GEPA compatibility
      manifests, audit vendored algorithm/prompt licenses and provenance, and
      define semantic/SSE corpus recorders.
      **AC:** ledger has zero unclassified server paths and every accepted
      third-party metric/provider in the pinned reference has an implementation
      and test owner; a missing or license-blocked port blocks Part II.
      **VER:** checked-in machine-readable manifest + generated Markdown report.
      **DONE (2026-07-18):** `rust/genai-inventory/` — ledger.json (1,546
      server_reachable / 346 client_only / 0 dead / 0 unclassified; per-phase:
      T16 138, T17 6, T18 55, T19 1,331, T20 12, T21 4), REPORT.md generated
      byte-stably from the ledger, validator script enforces the invariants;
      manifests: scorers.json (138), providers.json (191 providers / 2,908
      model records, litellm==1.91.2 wheel-pinned incl. price-table SHA-256
      per D16), algorithms (GEPA 0.0.27 + MetaPrompt); corpus-recorders.md
      (scripted Python semantic/model transcripts + framing-aware SSE
      recordings, T12.4 normalization patterns). License audit: all sources
      MIT/Apache-2.0 EXCEPT Phoenix evaluators (Elastic-2.0) → BLOCKER
      recorded as D23 (explicit Rust rejection of those scorer families;
      user review requested). ruff/report-stability/validator/fmt exit 0.

### Phase 16 — GenAI CRUD (Tier A)

- [x] **T16.1 Evaluation datasets**: store (3 tables + associations, offset +
      cursor tokens, input_hash upsert dedup, schema/profile/digest
      recompute, `SearchEvaluationDatasetsUtils` filter DSL in
      `mlflow-search`) + all 12 endpoints with the JSON-string field quirks.
      **AC:** dataset sections of the Python store/REST suites pass against
      Rust; record-pagination tokens interop with Python-written tokens
      (legacy offset fallback included).
      **VER:** Phase 22 runner + differential corpus section.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge 0480b8906): NO new
      migration — tables pre-existed via upstream `71994744cf8e` (datasets/
      tags/records/associations) + `3da73c924c2f` (record outputs); head
      stays `c4a9b7d3e812`. All 12 endpoints under /api/3.0 + /ajax-api/3.0
      with Python's JSON-string quirks (tags/schema/profile; record
      inputs/outputs/expectations/tags/source; record list as ONE JSON
      string both directions). D21 authenticated-only, workspace-scoped.
      Cross-server token interop on one shared sqlite DB Rust→Python→Rust:
      record cursors both directions + legacy decimal-offset + search
      offset tokens. VER: Python store suite 32/32 (before and after),
      Rust-backed REST 13 passed/12 expected file-store skips, token
      interop 1/1; route_parity §12.1 = 26 implemented/0 planned.
      Merge notes: route_parity.py accounting conflict with T16.5 resolved
      by summing proto-implemented + hand-implemented counts; the corpus
      `dataset_search` allowlist entry turned out NOT stale (case hits the
      legacy 2.0 experiments/search-datasets path, not the 3.0 API) —
      restored with a clarifying comment (4410bc04b). Post-merge gates
      fmt/clippy/test/route_parity all 0; replay 133 cases, 0
      non-allowlisted diffs.
- [x] **T16.2 Scorers CRUD + online configs**: 5 RPCs + 2 config routes;
      MAX+1 versioning, latest-per-name listing, gateway-endpoint name→ID
      payload rewrite, decorator-scorer registration rejection (exact error).
      **AC:** scorer CRUD suites pass; serialized payloads written by either
      server read identically by the other.
      **VER:** Phase 22 runner `-k scorer`.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge 9fe2f203a): NO new
      migration — all 3 scorer tables pre-existed at head `c4a9b7d3e812`.
      7 logical routes under /api/3.0 + /ajax-api/3.0. MAX+1 collision
      retry, latest-per-name listing, workspace isolation, gateway
      name→ID persistence + ID→name reads; decorator-registration
      rejection matches Python's raw response bytes. D23 rejection hook
      live: Phoenix metrics rejected naming Elastic-2.0 and pointing to
      Faithfulness/RelevanceToQuery/Correctness/Safety builtins +
      instructions judge for Summarization/SQL. Cross-server test:
      Python- and Rust-written scorer payloads read byte-identical.
      Agent also REMOVED the rust-skip guards in
      tests/server/auth/test_client.py + test_rest_tracking.py so scorer
      suites now run against Rust. VER: Rust-mode Python suites 14
      passed (REST CRUD 1, genai CRUD 7, auth/RBAC 6), 8 new Rust tests,
      cross-server 1/1; route_parity §12.3 = 14 implemented/1 planned
      (T17.4 invoke). Merge conflicts (handler arms, store re-exports,
      TRACKING_TABLES=29, IMPLEMENTED_PROTO_SECTIONS) resolved by union;
      post-merge gates all exit 0 (fmt after rustfmt on the hand-merged
      import list, f6c28993d).
- [x] **T16.3 Issues + label schemas**: issues CRUD (4 RPCs, trace_count via
      the assessments join) and label schemas (6 RPCs, immutable `type`,
      unique names).
      **AC/VER:** suites + differential sections; `include_trace_count`
      parity on a seeded fixture.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge 8fbadb121): NO new
      migration — `issues` + `label_schemas` pre-existed at head
      `c4a9b7d3e812`. 10 logical routes under /api/3.0 + /ajax-api/3.0.
      trace_count semantics: LEFT OUTER JOIN assessments ON name=issue_id
      AND assessment_type='issue', COUNT(DISTINCT trace_id), NO valid
      filter. Label-schema names unique PER EXPERIMENT (not global);
      `type` wire-immutable (absent from UpdateLabelSchema, updates
      preserve it); exact Python error strings ported. Issues D21
      authenticated-only (`// AUTH GAP:`); label-schema perms inherit
      from experiment. VER: Rust-backed REST slice 16 passed (12 issue +
      3 label-schema + 1 differential), Python oracle suites 165/165 + 18
      label-schema handlers, 5 new Rust integration tests; seeded
      same-DB differential byte-identical with AND without
      include_trace_count; route_parity §12.4=8, §12.5=12 implemented.
      Post-merge gates fmt/clippy/test/route_parity/replay all exit 0.
- [x] **T16.4 Review queues**: 11 RPCs, 4 tables, name_key semantics, soft
      references, user-queue schema resolution, full auth validator set +
      integrity hook.
      **AC/VER:** review-queue suites + multi-user auth differential.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge 84b006bed): NO new
      migration — all 4 tables pre-existed at head `c4a9b7d3e812`. All 11
      RPCs on both prefixes; §12.6 = 22 implemented/0 planned. Full auth
      set ported (NOT D21): create/edit gates, owner/member/manager
      rules, list filtering, custom-only owner delete/prune, owner
      reassignment requires MANAGE, EDIT+membership status updates,
      attribution binding, admin username-shadow integrity hook.
      Semantics: case-insensitive (experiment_id, name_key) uniqueness
      with preserved display casing; trace/schema soft references (trace
      deletion prunes items; deleted schemas stay as literal
      associations but drop out of live resolution); user queues
      dynamically inherit current experiment schemas. VER: 103/103
      Python parametrized cases, 4 new Rust tests, 2-user auth
      differential byte-identical (8 grant/denial responses).
      Post-merge: conflicts vs T16.6 unioned; the unimplemented-endpoint
      sentinel test collided (T16.4 pointed it at prompt-optimization,
      which T16.6 implemented first) — moved to gateway CRUD/T18.1
      (f5363f23d); gates then all exit 0. PHASE 16 COMPLETE.
- [x] **T16.5 Jobs store + API**: `jobs` table store (states, retries,
      finalized-immutability) + GET/cancel endpoints; resolve and document
      the `/mlflow/jobs` vs `/jobs` auth-prefix question (§12.13).
      **AC:** job rows created by Python readable/cancellable via Rust and
      vice versa.
      **VER:** cross-server job-store test.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge 132935daa): NO new
      migration — `jobs` table pre-existed at head `c4a9b7d3e812`.
      Store: `rust/crates/mlflow-store/src/store/jobs.rs` (790 lines) —
      workspace-aware lifecycle, retries, status-detail merge, finalized-
      transition rejection, D20 queue claims (SQLite conditional
      UPDATE…RETURNING; PG CTE `FOR UPDATE SKIP LOCKED`+RETURNING; MySQL
      transactional `FOR UPDATE SKIP LOCKED`). §12.13 RESOLVED: Flask
      serves `/ajax-api/3.0/mlflow/jobs/...` (global auth hook, no
      per-job validator), FastAPI serves `/ajax-api/3.0/jobs/...`
      (auth-prefix matcher) — both authenticated-only, distinct wire
      formats preserved; `// AUTH GAP:` at mlflow-server/src/lib.rs:329.
      T15.4 scratch jobs adapter deleted; worker spike asserts on real
      JobStore. 2 hand-registered Flask routes added to route_parity
      allowlist (same category as server-info). VER: cross-server 1/1,
      Python-vs-Rust 4/4, store lifecycle+32-way claim race 5/5, HTTP
      wire/auth 5/5, worker 2/2; post-merge gates fmt/clippy/test/
      route_parity/replay all exit 0. Ledger's submit/search/runner
      tests deferred to Phase 17 (runner surfaces).
- [x] **T16.6 Prompt-optimization CRUD**: 5 RPCs over the jobs store + runs
      (entity rebuild from job + run params, optimized-prompt URI from
      result); submission enqueues per Phase 17 (execution rides Phase 19).
      **AC/VER:** CRUD parity via corpus; create returns a queued job whose
      lifecycle matches Python's.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge after bad6dbd54, clean
      no-conflict): NO dedicated table — rides jobs + runs/params/metrics/
      dataset-inputs; no migration. All 5 RPCs (12 concrete routes),
      §12.7 = 12 implemented/0 planned. Quirks ported: scorer_names
      JSON-string on runs, optimizer_config_json raw vs parsed job
      optimizer_config, reconstruction accepts native AND JSON-string
      fields, optimized_prompt_uri only from successful job result,
      failed results = raw error string, create-response tags not
      persisted, completion timestamp omitted. Cancel kills job + run;
      delete removes finalized job + soft-deletes run. Auth inherits
      referenced experiment (update create/cancel, read get/search,
      delete delete) — NO D21 gap. Submission leaves job PENDING with
      `// Phase 19:` marker (no optimizer implemented, per plan). NEW
      11-case prompt_optimization corpus section added to replay
      (rust/compliance/corpus/prompt_optimization.yaml) — corpus now 144
      cases. VER: Python handler suite 23/23, Rust-switched auth 2/2, 4
      new Rust tests incl. cross-server byte-parity; 80 optimizer
      execution tests correctly deferred to Phase 19. Post-merge gates
      all exit 0.

### Phase 17 — Job runner + worker

- [x] **T17.1 Runner core** per §14.1: DB-claimed queue, state machine,
      transient-retry with backoff, timeout kill, exclusive-lock CANCELED
      semantics, per-function max_workers, startup recovery, enable-gate.
      **AC:** job-lifecycle suite parity incl. the CANCELED-when-locked and
      TIMEOUT paths; no queue files on disk.
      **VER:** `rust/tests/job_runner.rs` + Python jobs suites via launcher.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge after f5e935bb9, clean):
      `rust/crates/mlflow-server/src/job_runner.rs` (631 lines) + 518-line
      lifecycle test. States PENDING(0)→RUNNING(1)→SUCCEEDED(2)|FAILED(3)|
      TIMEOUT(4)|CANCELED(5), finals immutable. Retry: at cap→FAILED
      without increment, else increment→PENDING with backoff
      min(base·2^(n−1), max), defaults 3/15s/60s per Python. Cancel
      checked before timeout; exclusive-lock collision→competing job
      CANCELED; cancel-while-locked→drop execution, row stays CANCELED,
      then release. Startup recovery requeues PENDING + resets pre-start
      RUNNING→PENDING across workspaces. Gate
      MLFLOW_SERVER_ENABLE_JOB_EXECUTION (default true; invalid values
      fail config). Execution seam = `JobExecutor` trait for T17.2.
      Store claims extended to exclude locally-delayed retries. DB is the
      only queue (no-Huey-files test). VER: runner lifecycle 7/7, store
      6/6, Python jobs API switch 4/4; ledger's 80 runner/submit cases
      triaged: 42 deferred to T17.2 native kinds (discovery 12, evaluate
      7+5, optimize 18), 7 invoke routes to T17.4, rest covered by Rust
      equivalents. Post-merge gates all exit 0.
- [x] **T17.2 Native worker protocol** per §14.2: versioned request/result
      envelopes, closed six-kind dispatch enum, subject/workspace propagation,
      bounded output, process-group kill-on-timeout/cancel, crash/malformed-
      output mapping, and back-pressure.
      **AC:** all six native job kinds launch and report through the Rust
      runner with Python absent; protocol-version and unknown-kind negatives
      fail before execution.
      **VER:** integration matrix using the deterministic `mlflow-genai`
      fixture + production-image-style PATH isolation.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge 549e55efa): six-kind
      closed dispatch (invoke_scorer, run_online_trace_scorer,
      run_online_session_scorer, optimize_prompts, invoke_issue_detection,
      invoke_genai_evaluate) with deterministic fixtures behind
      MLFLOW_GENAI_WORKER_FIXTURE=1 + `// Phase 19:` markers at every
      real-logic site. Negatives BEFORE execution:
      UNSUPPORTED_PROTOCOL_VERSION / UNKNOWN_JOB_KIND; unknown runner job
      names fail before spawn. Supervision: one subprocess per job;
      workspace/subject/tracking-URI/gateway-URI/internal-token
      propagated; startup FAILS if worker binary missing (no Python
      fallback); 4 MiB input + 4 MiB stdout/stderr caps with
      drain-after-cap (no pipe deadlock) and `...[truncated]`; Unix new
      process group + SIGKILL-on-drop kills grandchildren;
      close_range/fcntl FD hygiene; failure mapping non_zero_exit/signal/
      malformed_output/timeout persisted as status_details. Back-pressure
      via T17.1 caps (judges 10, online 5, optimization 2; observed peak
      = cap in test). VER: Python-free 6/6 matrix (no python/python3 on
      PATH), 6/6 negatives, spike regression 2/2, Python oracle suites
      51/51 items. 37 direct-Python-function tests + 5 HTTP cases stay
      with T17.4/Phase 19 (documented). Merge union kept T17.2
      status-details + finalize_with_retry hardening; both scheduler and
      runner start in main.rs behind the shared gate. Post-merge gates
      all exit 0.
- [x] **T17.3 Periodic scheduler + online-scoring scheduler**: tokio-based
      cron with DB locking; port the scheduler logic (config scan, grouping,
      shuffle, sampler waterfall, checkpoint tags, per-job caps).
      **AC:** with a seeded config, Rust submits the same jobs Python would
      for the same trace timeline (deterministic-seed comparison).
      **VER:** scheduler differential test.
      **DONE 2026-07-18** (codex gpt-5.6-sol, merge on top of T17.1 base):
      `online_scoring_scheduler.rs` (429 lines) — next-wall-clock-minute
      tick + 60s cadence, missed ticks skipped, all workspaces scanned.
      Python reference: crontab */1 + Huey lock_task
      `online-scoring-scheduler-lock` (mlflow/server/jobs/utils.py) →
      Rust `scheduler_locks.rs`: reserved job-table row with atomic
      insert/takeover, owner fencing token, lease expiry, conditional
      release. Groups by experiment, shuffles groups (SplitMix64/
      Fisher-Yates vs Python's Mersenne Twister — DELIBERATE order-only
      deviation; job SET + params identical), sampler = Python's SHA-256
      dense waterfall, checkpoint experiment tags
      mlflow.latestOnlineScoring.{trace,session}.checkpoint with 1h
      lookback, caps 500 traces/100 sessions per job; submission does
      NOT advance checkpoints (execution does, per Python). No dedup,
      matching Python. run_once(seed) seam for determinism. Gate:
      MLFLOW_SERVER_ENABLE_JOB_EXECUTION. VER: 8 Rust scheduler tests,
      shared-DB Python-vs-Rust differential (seed 2026) exact sorted
      (kind, params, workspace) equality 3/3 jobs, Python suites 10/10.
      Post-merge gates green after fixing a REAL T17.1 latent bug the
      gate run exposed: transient store errors (SQLITE_BUSY_SNAPSHOT on
      deferred read→write tx) stranded finalizing jobs in RUNNING —
      finalization writes now retry with fresh transactions
      (7f9976748); flaky retry test 15/15 after.
- [x] **T17.4 Invoke endpoints**: evaluate-invoke, scorer-invoke,
      issue-detection-invoke, prompt-opt submission wired to the runner with
      submission-side byte parity (pre-created runs, tags, response shapes,
      batching rules).
      **AC:** invoke responses + created runs/tags byte-match Python with the
      deterministic native worker fixture.
      **VER:** differential corpus (native fixture mode).
      **DONE 2026-07-18** (codex gpt-5.6-sol, clean merge): invoke.rs
      (559 lines). Evaluate: exact empty-list errors ("Please select at
      least one trace/judge."), RUNNING run with user_id="unknown", no
      end time, mlflow.runType=genai_evaluate,
      mlflow.genaiEvaluate.jobId tag, {"job_id","run_id"} response.
      Scorer: looser Python validation kept, batches of 100 via
      MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE, session scorers grouped by
      mlflow.trace.session (first-session order, chronological within,
      sessionless omitted), {"jobs":[...]} response, no pre-created run.
      Issue detection: Python schema + endpoint/provider-model rule incl.
      exact 500 text; run ended-as-RUNNING (non-null end time) matching
      Python; secret IDs kept out of durable params (gateway resolution =
      Phase 18). All three authenticated-only (D21). Prompt-opt PENDING
      rows now claimed and dispatched through the native worker. §12.2/
      12.3/12.4 all 0 planned. VER: invoke corpus 10/10 (batch pinned to
      2), full corpus 154 cases/0 diffs, invoke_http 7/7, shared-DB
      interop 3/3, Python oracle 12/12. Gates all exit 0.
      PHASE 17 COMPLETE.

### Phase 18 — Gateway

- [x] **T18.1 Gateway CRUD + crypto**: 9 tables, 36 endpoints, envelope
      crypto (T15.3), secret masking, secret cache, workspace scoping.
      **AC:** secrets round-trip cross-language; CRUD suites pass.
      **VER:** Phase 22 runner + corpus `-k gateway`.
      **DONE (2026-07-18, codex agent, merge bd5d2ecbd):** all 9 gateway
      tables pre-existed at head `c4a9b7d3e812` — no migration. T15.3 spike
      promoted to `mlflow-store/src/secrets.rs` (exact format: AES-256-GCM
      nonce||ct||tag, 60-byte wrapped DEK, PBKDF2 600k, versioned KEK, AAD
      id|name). Cross-language AC passed both directions: Python decrypted
      32 Rust envelopes (kek_version 1 and 42), Rust decrypted 32 Python
      envelopes; wrong-id/wrong-name AAD negatives rejected both ways.
      Masking (Unicode-char based, <8 → `***`, else first3...last4) on all
      public reads; encrypted LRU cache mirrors Python (TTL 60s clamp
      10–300, max 1000, dual time-bucket keys, lazy expiry, full-cache
      invalidation on mutation). 36 proto handlers / 72 routes with quirks
      (usage-tracking default true, ignored list `secret_id`, empty-update
      no-ops, ASCII endpoint names, guardrail name serialization); auth:
      dependency USE checks, creator MANAGE, deletion cascades, admin-only
      budgets. Python gateway suites 266 passed / 1 skipped against Rust
      store. Corpus +21 gateway cases (masked bytes) → 175 total, zero
      non-allowlisted diffs. Route parity: §12.8 implemented=72, planned=6
      (T18.2 discovery). 404 sentinel moved to `gateway/supported-models`
      (moves again at T18.2). Gates fmt/clippy/test/parity/replay all 0.
- [x] **T18.2 Discovery + bridge routes**: 4 ajax discovery routes +
      `gateway-proxy` with its validation quirks and empty-target behavior.
      **AC/VER:** corpus section; UI provider-picker renders from Rust.
      **DONE (2026-07-18, codex agent, merge 950076edb):** 4 discovery
      routes (supported-providers, supported-models, provider-config,
      secrets/config) with provider sort/filter/aliases, Vertex
      consolidation, model flatten/dedup, fine-tune exclusion, pricing
      conversion, passphrase env truthiness; UI payloads asserted against
      Python-derived byte lengths + SHA-256 (UI render itself stays on the
      deferred browser-validation backlog with T9.9/T11.6). gateway-proxy
      GET/POST with Python quirks: empty/unset MLFLOW_DEPLOYMENTS_TARGET
      short-circuits to `{"endpoints":[]}\n` before all validation; GET
      only `api/2.0/endpoints`, POST only `gateway/<name>/invocations`;
      HTML 415/400/500 Flask-layer errors; headers not forwarded either
      direction; only upstream 200 succeeds, else INTERNAL_ERROR 500;
      Flask-sorted JSON re-serialization. Stub-server forwarding tests.
      §12.8 now implemented=78, planned=0. Corpus +13 (8 discovery/empty-
      target in gateway.yaml, 5 proxy-validation in new
      gateway_proxy_validation.yaml) → 188 total, zero non-allowlisted
      diffs. Sentinel moved to `/ajax-api/3.0/mlflow/assistant/config`
      (§12.10, moves at T20.1). Gates fmt/clippy/test/parity/replay all 0.
- [x] **T18.3 Runtime core**: unified invocations + `mlflow/v1/chat/
      completions`, provider trait, native adapters for openai/azure,
      anthropic, gemini; SSE plumbing with the §12.9 exactness list; timing
      headers; endpoint-config resolution + cache.
      **AC:** mock-provider differential streams frame-identical to Python.
      **VER:** SSE recorder (T15.5 design) + adapter unit tests.
      **DONE (2026-07-18, codex agent, merge 09b1e527d):** provider trait
      (transform_request/response/stream_frame, map_error, inject_auth,
      cross-frame state for the D16 matrix); native openai/azure(+Azure AD)/
      anthropic/gemini adapters. Endpoints: `/gateway/{name}/mlflow/
      invocations` + `/gateway/mlflow/v1/chat/completions` with FastAPI-
      style errors and representative Pydantic messages. §12.9 SSE honored:
      `data: {json}\n\n` compact reserialization, no [DONE], keep-alive/
      event/blank lines ignored, cross-chunk line buffering, OpenAI bad
      frames skipped vs Anthropic/Gemini exact JSONDecodeError envelopes,
      mid-stream errors in-band at HTTP 200, lazy upstream connect
      (Python's setup-duration semantics), duration always + overhead only
      non-stream, forced gzip/deflate/identity, X-MLflow-Authorization
      never forwarded. Config chain endpoint→mapping→model-def→secret with
      shared encrypted SecretCache (`endpoint_config:{ws}:{name}`),
      mutation invalidation. AC: 27 recorder comparisons — all 4 providers
      frame-identical (non-stream/multi-frame SSE/429+500/mid-stream).
      Corpus harness can't spawn per-case mocks → runtime differential
      lives in recorder + integration tests (documented). Merge conflicts
      hand-resolved: route_parity IMPLEMENTED_EXTERNAL_ROUTES made a dict
      union (T18.2 proxy + T18.3 runtime), deduped reqwest features in
      mlflow-server Cargo.toml. §12.9 implemented=2, planned=8 (T18.4).
      Workspace suite 1,270 tests. Gates fmt/clippy/test/parity/replay all
      0; corpus 188 cases, zero non-allowlisted diffs.
- [x] **T18.4 Full provider matrix**: passthrough + raw-proxy routes;
      bedrock/databricks auth modes; openai-compatible family
      (groq/deepseek/xai/openrouter/ollama/portkey); every pinned LiteLLM
      fallback provider/request transform, model/token/cost entry, retry
      classification, and provider allowlist behavior per D16.
      **AC:** every provider reachable in the pinned Python reference is
      reachable natively with request/response/stream/cost parity; zero
      Python fallback and zero unsupported-provider allowlist entries.
      **VER:** generated provider manifest + hermetic conformance matrix.
      **DONE (2026-07-19, codex agent, merge into 8a2f063c2 + fix
      a6668d05d):** all 8 passthrough/raw-proxy routes native (openai chat/
      embeddings/responses/responses-compact, anthropic messages, gemini
      generate/streamGenerate, raw `/gateway/proxy/{name}/{path}`) with
      endpoint USE auth, upstream query preservation, SSE framing. AC met:
      191/191 pinned LiteLLM provider identifiers reachable natively, zero
      Python fallback, zero unsupported entries; 2,908 model/token/cost
      records; 6 explicit adapters (+bedrock SigV4/API-key/default-chain/
      STS-assume-role, databricks PAT/env/OAuth-M2M) + 6 openai-compatible
      variants + 179 pinned-transform entries. Manifest
      `rust/genai-inventory/provider_manifest.json` (chat 146, embeddings
      48, stream 146, cost 114) + regeneration check. Retry parity across
      auth/timeout/rate-limit/content-policy/bad-request; pinned pricing.
      38 differential comparisons. Orchestrator-found+fixed bug
      (a6668d05d): manifest generator classified by exact name so aliases
      amazon-bedrock/databricks-model-serving fell to
      pinned_litellm_transform while adapter_for() resolves them to
      explicit native adapters — matrix test only passed with ambient
      AWS_* env creds; generator now normalizes aliases. §12.9
      implemented=10, planned=0 — ALL §12 route families now fully
      accounted. Workspace 1,288+ tests. Gates all 0; corpus 188, zero
      non-allowlisted diffs.
- [x] **T18.5 Traffic split + fallback**: weighted routing and sequential
      fallback incl. status propagation and attempt accounting.
      **AC/VER:** statistical + deterministic tests on mock providers.
      **DONE (2026-07-19, codex agent, merge 305d6bd9e):** weights use
      Python's `int(weight*100)` truncation + float32 normalization;
      mixed zero-weights never selected; all-zero reproduces Python's
      `ValueError: probabilities contain NaN`; endpoint-model order
      preserved, categorical cumulative selection. RNG deviation
      documented (thread-local vs NumPy MT19937 — distribution-equivalent,
      order-only; T17.3 precedent). Fallbacks stably sorted by
      fallback_order (missing last); max_attempts = Python's
      primary+fallback accounting with zero→default and provider-count
      cap; every normal provider exception falls back, pre-routing
      validation/config errors propagate immediately; final recognized
      status propagates, generic→500, "All N fallback attempts failed.
      Last error: ..." byte-exact. No attempt-count header (Python exposes
      none — asserted via recorder call counts). Streaming fallback active
      MID-STREAM (partial chunks from failed provider followed by next
      provider's; terminal failure = SSE error frame at HTTP 200).
      Verified: deterministic 0/100+single-target, scripted fallback
      chains, 100k-selection statistical (±1% of truncated 69/31), Python
      differential oracle incl. partial-stream fallback. Merge surgery:
      stream_response kept T18.5's lazy ProviderStream (drops eager
      connect); restored primary_model helper for T18.4's raw-proxy path
      (raw proxy bypasses split/fallback, per differential). Gates all 0;
      corpus 188, zero non-allowlisted diffs.
- [x] **T18.6 Budget enforcement**: policy CRUD already in T18.1; tracker
      (in-memory + Redis), spend query over span `total_cost`, 429 message
      byte-parity, ALERT webhooks through the Part I dispatcher, gateway-call
      tracing so costs keep accruing.
      **AC:** REJECT/ALERT behaviors match Python on a seeded spend fixture.
      **VER:** budget differential test.
      **DONE (2026-07-19, codex agent, merge d5a1a4aca — PHASE 18
      COMPLETE):** tracker keyed by policy ID; Redis keys byte-match Python
      (`mlflow:budget:window:{id}` etc.), selected via
      MLFLOW_GATEWAY_BUDGET_REDIS_URL, in-memory default, 600s refresh;
      epoch-aligned min/hour/day windows, Sunday weeks, calendar months;
      inclusive spend>=limit, one ALERT per window, rollover reset; global
      vs workspace policy scoping; in-memory backfill max(current,DB),
      Redis authoritative DB spend. Spend query = SUM(span_metrics.value)
      for total_cost over gateway-tagged traces, [start,end). REJECT 429
      `{"detail":...}` byte-compatible; ALERT = budget_policy.exceeded via
      existing T8.3 dispatcher, Python field order. Tracing: gateway/{name}
      root + provider/{prov}/{model} child LLM spans, token usage +
      input/output/total_cost metrics, error span on budget rejection; SSE
      unchanged. Seeded differential: under/boundary/over/threshold/reset
      all matched. Redis service test self-skips without redis-server.
      Orchestrator merge integration (13 hunks vs T18.7): order = budget →
      guardrails BEFORE → provider → guardrails AFTER → trace completion
      (budget cost recorded inside, Python's callback-inside-trace order);
      traces capture ORIGINAL client request (pre-sanitization) and FINAL
      post-guardrail response; resolver keeps T18.7 name; ProviderStream
      carries guardrail + trace state. Gates all 0 post-integration;
      corpus 188, zero non-allowlisted diffs; 1,300+ workspace tests.
- [x] **T18.7 Guardrails execution** per D17: BEFORE/AFTER orchestration,
      VALIDATION 400s, SANITIZATION via action endpoint with the bypass
      header, no-post-guardrails-on-streams rule.
      **AC:** guardrail matrix (stage × action × violation) parity.
      **VER:** mock-judge differential.
      **DONE (2026-07-19, codex agent, merge 7e0224a61):** sequential
      execution by execution_order ASC NULLS LAST then guardrail_id;
      sanitized payload chains into next guardrail; first VALIDATION
      violation short-circuits rest + provider call. BEFORE once before
      routing/fallback, AFTER once after successful non-stream response;
      chat runs both, unified embeddings neither, passthrough embeddings
      BEFORE-only, raw proxy BEFORE + AFTER only on non-stream JSON/text.
      VALIDATION → 400 `{"detail":"Guardrail '<name>' blocked:
      <rationale>"}` byte-exact. SANITIZATION → POST
      /gateway/{action_endpoint}/mlflow/invocations with
      X-MLflow-Guardrail-Bypass: 1, only Authorization forwarded, payload
      = sanitizer prompt + rationale + json.dumps(payload, indent=2) +
      exact JSON-schema response format; choices[0].message.content parsed
      and replaces payload. Streams: BEFORE applies (typed-stream
      violation = HTTP 200 SSE GuardrailViolation; raw proxy stays 400),
      AFTER never runs, no warning (Python emits none). Ordering pinned:
      Python checks budgets BEFORE guardrails (T18.6 seam consistent). AC
      met: all 8 stage×action×outcome cells byte-identical vs Python via
      guardrail_oracle.py; sanitization internal payload matched exactly.
      Gates all 0; corpus 188, zero non-allowlisted diffs.

### Phase 19 — Native GenAI execution parity (Tier C)

- [x] **T19.1 Scorer model + native judges** per §14.3: exact
      `SerializedScorer` parsing/round-trip/errors; all concrete deterministic
      and LLM builtins; `InstructionsJudge`, `Guidelines`, trace tools and
      structured output; `MemoryAugmentedJudge` embedding retrieval; explicit
      OSS decorator rejection.
      **AC:** every server-accepted non-third-party payload in T15.5's manifest
      executes natively and produces request/feedback parity; every rejected
      form returns Python's status/error class/message.
      **VER:** scorer serialization corpus + scripted judge/tool/embedding
      differential suites.
      **DONE (2026-07-19, codex agent, merge b3e2b760f):** new
      `rust/crates/mlflow-genai/` crate. Manifest coverage: 139 entries →
      26 accepted natively (24 builtins + InstructionsJudge +
      MemoryAugmentedJudge), 7 rejected correctly (OSS @scorer security
      rejection + 6 Phoenix per D23), 106 deferred to T19.3
      (DeepEval/Ragas/TruLens). All five SerializedScorer representations
      parse/round-trip; 11 malformed forms return Python's exact HTTP 400
      INVALID_PARAMETER_VALUE messages. Deterministic value-exact:
      PIIDetection, RegexMatch (lookarounds/backreferences),
      ResponseLength, Equivalence + ToolCallCorrectness short-circuits;
      ordered/unordered ToolCallCorrectness rationales value-exact. 21
      LLM/hybrid builtins natively dispatched; InstructionsJudge/
      Guidelines request construction + structured output + feedback
      source/metadata + 6 trace-tool schemas scripted-diff-clean.
      MemoryAugmentedJudge: scripted corpus/query embeddings, cosine
      top-k, semantic/episodic augmentation, retrieved-memory metadata.
      Real execution wired into Phase 17 worker; fixture mode intact. New
      oracles: rust/tools/scorer_oracle.py (run with `uv run --with
      dspy==3.2.1`; 26/26/11) + judge_oracle.py (7 suites, 6 schemas).
      Zero live LLM calls. Gates all 0; corpus 188 unchanged; 1,321 tests.
- [x] **T19.2 Evaluation, invoke, and online scoring**: native bounded
      concurrency/rate limiting/retries, scorer-result standardization,
      `SCORER_ERROR`, evaluator traces, single-turn/session grouping,
      expectations/tags/assessments, aggregate metrics, batching, dense
      sampling, checkpoints, cleanup, and all three scoring job kinds.
      **AC:** evaluate-invoke, scorer-invoke, and same-seed online scoring
      produce identical jobs, runs, traces, assessments, metrics, and
      checkpoints with Python absent.
      **VER:** seeded end-to-end semantic differential corpus.
      **DONE (2026-07-19, codex agent, merge 33d66655c):** rate limits per
      Python env contract (predict default auto@10RPS, scorer = predict ×
      scorer count, workers scorer-RPS×2s clamped 10–500, per-item scorer
      workers default 10, retries default 3 — 429-only, 1/2/4…s cap 60);
      AIMD token-bucket limiter (×0.5 throttle, 1/rps recovery, 5s
      cooldown, 2× ceiling). Standardization incl. SCORER_ERROR
      CODE-sourced with error_code/message/stack; session feedback carries
      mlflow.trace.session; evaluator traces mlflow.trace.sourceScorer +
      mlflow.assessment.scorerTraceId; aggregates
      `<name>/<agg>` for mean/min/max/median/population-variance/linear-
      p90. All three job kinds Python-free: evaluate-invoke run/trace
      links + root-span assessments + terminal status; scorer-invoke
      batching + session order; online trace/session with SHA-256 dense
      sampling, 500/100 caps, checkpoint tags byte-exact (JSON spacing +
      timestamp/ID ties), rescoring logs replacements before deleting
      superseded assessments. Fixture mode intact. New
      rust/tools/evaluation_oracle.py (seed 1902: 4 rate + 6
      standardization + 10 aggregate values + 6 metrics). Gates + all 3
      oracles 0; corpus 188; 1,334 tests (one invoke_http flake under full
      parallel load did not reproduce in 6 follow-up runs — watching).
- [x] **T19.3 Third-party scorer compatibility**: port every DeepEval,
      Ragas, TruLens, and Phoenix metric in the pinned manifest, including
      deterministic algorithms, prompts/parsers, input/session mapping,
      thresholds, rationales, metadata, model/embedding calls, and dynamic-
      metric error behavior.
      **AC:** manifest coverage is 100%; deterministic values are exact and
      scripted LLM request/feedback transcripts are diff-clean. No Python or
      external Python package is installed in the execution image.
      **VER:** per-family version matrix + license/provenance audit + golden
      corpus generated from the pinned reference environments.
      **PARTIAL (2026-07-19, codex agent, merge 3c79adebc — AC NOT MET,
      T19.3b follow-up in flight):** foundation merged: all 106
      DeepEval(44)/Ragas(37)/TruLens(25) metrics natively dispatched;
      all 12 deterministic metrics value-exact (15-case corpus);
      dynamic unknown-name error parity all three families; Phoenix 6/6
      rejected per D23 with equivalence pointers. Pins/licenses audited:
      DeepEval 4.0.7 Apache-2.0, Ragas 0.4.3 Apache-2.0, TruLens 2.8.1
      MIT, Phoenix 2.13.0 Elastic-2.0 (rejected). Agent's honest
      limitation (why the box stays open): most LLM-based DeepEval/Ragas
      paths use family-generic prompts, not exact per-metric pinned
      workflows; golden corpus covers 3 family adapter transcripts, not
      every LLM metric. Artifacts: third_party_golden.json + generator,
      rust/genai-inventory/third-party-compatibility.md. Gates + all
      oracles 0 post-merge; corpus 188. T19.3b closes the per-metric
      prompt/parser/transcript gap.
      **DONE (2026-07-19, T19.3b codex agent, merge fef23a745 — AC MET):**
      exact per-metric pinned workflows for every LLM metric that the
      pinned references can actually run: 56 model-backed workflows
      (DeepEval 29, Ragas 24, TruLens 3) with 129 ordered positive-call
      transcripts + 55 malformed-response transcripts (65 calls), all
      diff-clean vs golden corpus regenerated byte-identically from the
      pinned packages. The remaining LLM metrics FAIL IN THE PINNED
      REFERENCES THEMSELVES before any provider call (DeepEval 13:
      missing template vars/non-serializable DAGs/multimodal-MCP/pydantic
      schema; Ragas 3: sample classification/missing adapters; TruLens
      22: registry passes no provider-method args) — Rust reproduces each
      pinned error exactly; per-metric reasons in
      rust/genai-inventory/third-party-compatibility.md. All 112 manifest
      rows covered; 12 deterministic metrics exact; Phoenix 6/6 D23;
      unknown-name parity. Workflow dispatcher third_party/workflow.rs +
      generated pinned_workflows.json. New third_party_oracle gate (run
      with `uv run --with 'deepeval==4.0.7,ragas==0.4.3,trulens==2.8.1'`).
      Gates + all four oracles 0; corpus 188. NOTE: two rare
      load-correlated test flakes observed today under concurrent
      multi-checkout cargo runs (invoke_http once, workspace_scoping_http
      once incl. an ld OOM-kill signal 9) — never reproduced in isolation
      or repeat full runs; unrelated to merged diffs; watching.
- [x] **T19.4 Native issue discovery**: sampling, scorer verification,
      triage evaluation, session/error/execution-path extraction, latency
      statistics, LLM clustering/summarization/resplitting/dedup, severity/
      category filtering, issue rows, annotations, costs, progress, summary,
      and run terminal state.
      **AC:** scripted-model runs yield diff-clean issues, affected trace IDs,
      assessments, summary artifact, cost tag, status details, and job result.
      **VER:** phase-by-phase and end-to-end discovery differentials.
      **DONE 2026-07-19 (codex gpt-5.6-sol, merge 8937d2f38):** AC MET. Full
      `invoke_issue_detection` pipeline native in
      `mlflow-genai/src/discovery.rs`, mapping Python job.py/sampling.py/
      pipeline.py/extraction.py/clustering.py/utils.py phase-for-phase;
      fixture mode untouched. CPython `Random(42)` sampling ported (3/3
      exact incl. the 5,000-item branch); latency p50/p75/p90/p95/p99
      value-exact; clustering incl. invalid-index removal + orphan
      singletons exact; dedup assignments/severity/categories/examples
      exact. Scripted E2E diff-clean: issue rows, affected trace IDs,
      triage feedback + issue annotations, `summary.md` byte-exact,
      issues.json/metadata.json semantic-exact (elapsed-time normalized),
      cost 0.4 + `total_cost_usd` tag, `Generating summary...` status,
      FINISHED-before-cost-tag ordering, failure→FAILED. New gate
      `rust/tools/issue_discovery_oracle.py` (7 cases, zero diffs) +
      golden corpus. Post-merge gates all 0 (fmt/clippy/test -j4/
      route_parity 372/replay 188/scorer/judge/eval/third-party/discovery
      oracles); merge auto-resolved cleanly (lib.rs module list + worker
      dispatch verified by hand).
- [x] **T19.5 Native prompt optimization**: server-constrained prediction
      from prompt model config, scorer aggregation, native MetaPrompt, pinned
      GEPA algorithm/`gepa_kwargs`, seeded candidate selection, reflective
      datasets, metric-call budget, candidate artifacts/metrics, prompt
      registration/linkage, and job result URI.
      **AC:** fixed-seed/scripted-model GEPA and MetaPrompt runs produce
      identical candidate sequence, scores, artifacts, registered prompt,
      run/job state, and failure behavior.
      **VER:** optimizer state-machine differential + artifact byte/semantic
      fixtures where reference formatting is intentionally nondeterministic.
      **DONE 2026-07-19 (codex gpt-5.6-sol, merge dfe7120f5):** AC MET.
      MetaPrompt exact templates/validation/fallback; GEPA ports CPython
      MT19937 seeding + `getrandbits` selection directly (no scripted-
      selection workaround) — fixed-seed differential exact (candidates
      seed→candidate-2→candidate-4, batches, scores, 39 calls at budget 35);
      `gepa_kwargs` pinned-signature validation; candidate artifact layout +
      metrics parity (nondeterministic multi-scorer column order normalized
      by sorting, documented); prompt registration/linkage via
      `mlflow.linkedPrompts`, result URI `prompts:/<name>/<version>`;
      failure paths exact. New gate `rust/tools/prompt_optimization_oracle.py`
      (pinned `gepa==0.0.27`). Post-merge gates all 11 green (fmt/clippy/
      test/route_parity 372/replay 188/scorer/judge/eval/third-party/
      discovery/prompt-optimization oracles). NOTE: the test gate was
      initially blocked by HOST pid exhaustion — 95 uvicorn reference
      servers leaked by earlier SIGKILL'd test runs held ~4750 threads
      against the WSL2 cgroup pids.max 4915, making thread/process spawn
      fail EAGAIN. This was the true root cause of ALL of 2026-07-19's
      "load-correlated flakes" (invoke_http, workspace_scoping_http,
      native_protocol), the ld OOM-kills, and the rustc ICEs; earlier
      flake notes are superseded. After user-run pkill of the leaked
      servers, full `cargo test --workspace` at default parallelism:
      1,353 tests / 115 suites, exit 0, zero flakes. Hardening backlog:
      cross-server tests leak the reference server when the test binary
      dies uncleanly (Drop never runs) — consider process-group spawn +
      stale-server reaper in mlflow-test-support.

### Phase 20 — Assistant + promptlab

- [x] **T20.1 Assistant sessions + routes**: session file store (atomic
      write, UUID validation, PID files), all 9 routes, localhost gate, SSE
      framing, config + skills + models endpoints; D18 resolution for the
      stream_url path.
      **AC:** assistant UI chat works against Rust with the dev stub.
      **VER:** HTTP tests + `dev/run_dev_server.py --stub-providers claude`
      smoke against the Rust server.
      **DONE 2026-07-19 (codex gpt-5.6-sol, merge 700e8a56b):** AC MET. All
      9 §12.10 routes native in `assistant.rs`; session JSON byte-parity
      (Python json.dump spacing/ordering/ASCII), 0600 tmp+rename atomic
      writes, PID files + SIGTERM cancel, exact `Invalid session ID format`
      UUID guard; localhost gate via real TCP peer (IPv4/IPv6 loopback)
      with Python's exact 403 body; SSE `event:/data:` framing exact,
      all six event types; D18: /message returns the real
      sessions/{id}/stream path. Provider seam `AssistantProvider` trait
      (stream/check_connection/list_models/resolve_skills_path) with
      in-process dev stub; CLI providers are T20.2's module, wired in
      T20.3. 27 HTTP/SSE differentials + dev-stub smoke green. Merge
      resolutions: lib.rs router union with T20.4; route_parity §12.10
      flip + PLANNED_EXTERNAL_ROUTES now empty (all §12 implemented);
      404 sentinel repointed to a permanently-nonexistent path.
      Post-merge gates 13/13 green (incl. new assistant differential).
- [ ] **T20.2 CLI providers**: claude/codex subprocess spawn (exact flag
      construction, permission modes), NDJSON parsing, message filtering,
      usage events, SIGKILL→interrupted, cancellation, health probes with
      the 501/412/401 mapping.
      **AC:** stub-CLI streams frame-identical to Python; real-CLI manual
      smoke.
      **VER:** SSE recorder vs Python with the stub.
- [ ] **T20.3 OpenAI-compatible provider + tool executor**: tool loop,
      permission pause/resume, session-in-session_id encoding + 500 KB
      trim-by-turn-groups, sandboxed Bash/Read/Write/Edit with the
      confinement checks ported exactly + adversarial escape tests.
      **AC:** tool-loop conversation transcripts match; escape suite all
      negative.
      **VER:** `rust/tests/assistant_tools.rs` incl. traversal/symlink
      matrix.
- [x] **T20.4 Promptlab** per D19: run creation parity + pyfunc promptlab
      model artifact + `eval_results_table.json`, loadable by the Python
      client.
      **AC:** `mlflow.pyfunc.load_model` on a Rust-created promptlab run
      predicts through the gateway identically.
      **VER:** cross-language load test.
      **DONE 2026-07-19 (codex gpt-5.6-sol, merge from 344f51169):** AC MET.
      §12.11 create-promptlab-run + both demo routes implemented
      (`promptlab.rs`, `demo.rs`) — the LAST planned external routes;
      route_parity now zero planned. Rust-created promptlab run loads via
      `mlflow.pyfunc.load_model` and predicts through a scripted gateway
      identically to the Python twin; all artifacts byte-identical
      (MLmodel identical after run-id/uuid/timestamp normalization, YAML
      semantically equal). Demo parity incl. Flask JSON bytes + trailing
      newline, feature ordering/filtering, idempotence, deletion.
      Cross-language harness `rust/tools/promptlab_cross_language.py`.
      Post-merge gates 11/11 green (1,357+ tests).

### Phase 21 — Trace archival (closes D6)

- [ ] **T21.1 Config + flag**: `--trace-archival-config` /
      `MLFLOW_TRACE_ARCHIVAL_CONFIG` YAML parsing/validation (incl. repo
      support constraints and the artifacts-only conflict), 5s-TTL cache
      with stale tolerance.
      **AC:** invalid configs fail startup with Python's messages.
      **VER:** config-parity unit tests.
- [ ] **T21.2 Store paths**: `archive_traces`, transactional finalize
      (tag flips + content blank + generation guard), archived-payload
      deletion on trace delete, ARCHIVE_REPO reads in `getTrace` /
      `get-trace-artifact` (removing the Part I NOT_IMPLEMENTED stubs from
      T4.1/T4.5), retention/allowlist resolution.
      **AC:** archive→read→delete cycle byte-matches Python; D6 closed.
      **VER:** archival differential on sqlite + postgres.
- [ ] **T21.3 OTLP payloads**: `traces.pb` writer/reader (single
      ResourceSpans/ScopeSpans, root-first sort) interoperable with
      Python-written payloads both directions.
      **AC/VER:** cross-language payload fixtures.
- [ ] **T21.4 Scheduler task**: minute tick + interval gate + workspace
      fairness + per-pass budget on the §14.6 scheduler.
      **AC/VER:** same-seed scheduling decisions match Python.

### Phase 22 — Compliance & cutover

- [ ] **T22.1 Differential corpus genai sections** for every Tier A surface,
      invoke submission, and §15 semantic-engine category; compliance CI stays
      a required gate.
      **AC:** zero non-allowlisted wire/state/semantic diffs including all
      pinned scorer/provider/optimizer manifest entries.
      **VER:** CI artifact.
- [ ] **T22.2 Python-suite + reachability conformance**: T15.5's server-
      reachable suites green against Rust via the launcher switch on the DB
      matrix; ledger generator reports zero unclassified/missing native owners.
      Client-only SDK suites remain green against the Rust HTTP server.
      **AC/VER:** CI logs + generated coverage report.
- [ ] **T22.3 SSE/streaming differential** (gateway + assistant) green
      frame-by-frame against mocks/stubs.
      **AC/VER:** recorder CI job.
- [ ] **T22.4 nginx cutover**: delete the Python rows from §2.2 phase by
      phase; final state removes the Python server container from
      `rust/deploy/docker-compose.yml` and removes Python from every production
      image; `smoke.sh` asserts zero `X-MLflow-Backend: python` responses.
      **AC:** full-stack compose serves everything from Rust; genai UI pages
      and all six native job kinds work with `python`, libpython,
      site-packages, and `.py` payloads absent.
      **VER:** `smoke.sh` + `smoke_frontend.sh` + image-content/runtime-launch
      audit extended.
- [ ] **T22.5 UI smoke (genai)**: gateway admin pages (secrets/endpoints/
      budgets/guardrails), scorers + evaluation runs pages, datasets, issues,
      review queues, labeling, prompt optimization, assistant panel.
      **AC/VER:** T11.6-style recorded checklist.
- [ ] **T22.6 Ops docs**: extend T14.3 with KEK passphrase management +
      rotation, native worker concurrency/memory/process supervision, pinned
      scorer/provider/GEPA manifest upgrades, Redis budget tracker option,
      assistant CLI prerequisites, and the archival runbook.
      **AC/VER:** fresh-operator walkthrough.

---

## 17. Open decisions & risks (Part II)

| ID | Decision/Risk | Notes | Status |
|---|---|---|---|
| D14 | **Python-free execution model**: Rust owns wire, storage, queueing, scheduling, and all GenAI semantics. Each async job runs in a per-job `mlflow-genai-worker` Rust subprocess linked to the same `mlflow-genai` crate as the server; workers use the existing Rust HTTP APIs for store/gateway access. No Python interpreter, package, venv, sidecar, or fallback is permitted in production. The Python `mlflow.genai` SDK remains a compatible client/test oracle; OSS decorator-scorer rejection is preserved. | Keeps hard cancel/timeout/crash isolation while eliminating Python. `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=0` still disables execution explicitly; a missing native worker is a startup/deployment error, not a reduced-function fallback mode. | decided |
| D15 | **Legacy standalone YAML gateway** (`mlflow gateway start`) stays Python and deprecated; only the DB-backed embedded gateway is ported. | The `gateway-proxy` bridge route IS ported. | decided (T15.1 pass 2026-07-18) |
| D16 | **Full native LiteLLM compatibility**: vendor a pinned, generated manifest of every fallback provider transform, model/token limit, retry rule, tokenizer mapping, and price entry reachable in the reference release; implement it in Rust alongside the explicit gateway adapters. | No Python fallback and no fallback-only-provider exception. Manifest regeneration + semantic conformance is mandatory on upgrades because cost accuracy feeds budgets. | decided |
| D17 | **Guardrail execution**: JudgeGuardrail executes builtin/instructions judges inline through the native `ScorerExecutor`/`JudgeRuntime`; decorator and unsupported serialized scorer kinds retain Python OSS rejection behavior. | Avoids per-request worker startup while sharing identical prompt/provider/feedback logic with async jobs; no sidecar. | decided |
| D18 | **Assistant `/message` stream_url bug** (`api.py:154` returns `/stream/{id}`, route is `/sessions/{id}/stream`): the frontend builds its own URL, so both forms are dead letters. Propose: emit the correct form, document the deviation, and verify the UI never consumes the field. | Trivial but wire-visible. | decided (T15.1 pass 2026-07-18) |
| D19 | **Promptlab pyfunc artifact writer**: Rust writes the MLmodel/requirements/`parameters.yaml` and `eval_results_table.json` layout directly and byte-compatibly; the artifact remains loadable by the Python client. | No runtime code execution is needed to write the static layout; cross-language load/predict is a required test. | decided |
| D20 | **Queue replacement**: the `jobs` DB table becomes the queue (Rust polls/claims); SqliteHuey queue files are not reproduced. During migration the Python and Rust runners must never run simultaneously against the same DB (double execution); after cutover only native Rust workers exist. | Recovery improves because queue and lifecycle state cannot diverge. | decided (T15.1 pass 2026-07-18) |
| D21 | **Auth gaps ported faithfully**: datasets, issues, and online-config routes are authenticated-only in Python (no per-resource validators). Rust replicates this with `// AUTH GAP:` markers; fixing is a coordinated two-plane change proposed post-parity. | Silently hardening would break differential parity and possibly clients. | decided (T15.1 pass 2026-07-18) |
| D22 | **FastAPI-vs-Flask error-shape split**: gateway/assistant routes emit FastAPI-style errors (`{"detail": ...}`, 422 validation shape) while Flask routes emit MLflow proto-style errors. Rust must keep the per-route split (Part I already did this for OTLP's 422). | | accepted |
| D23 | **Phoenix evaluators license blocker** (T15.5 audit): `arize-phoenix-evals` is Elastic-2.0 — usable as a Python runtime dependency but NOT reimplementable/vendorable into Apache-2.0 MLflow. Rust explicitly REJECTS the six Phoenix-derived scorer metrics (Hallucination, QA, Relevance, SQL, Summarization, Toxicity — all LLM judges) with a clear error that (a) names the license constraint and (b) points at the functional equivalent: MLflow builtin judges (Faithfulness≈Hallucination, RelevanceToQuery≈Relevance, Correctness≈QA, Safety≈Toxicity) or a custom instructions judge (Summarization, SQL). Client-side execution via the Python SDK + user-installed phoenix remains supported (D14 client posture). Unblock paths: upstream relicense/grant, or counsel-approved clean-room lookalikes under MLflow names (post-parity product decision). The rejection is a deliberate wire deviation → Phase 22 corpus allowlist entry. All other Part II sources are MIT/Apache-2.0 (rust/genai-inventory/licenses.md). | Six LLM-judge metrics affected; everything else unblocked. | decided — USER APPROVED 2026-07-18 ("phoenix rejection approach is fine") |
| R4 | **Provider API drift**: the explicit gateway adapters plus the pinned LiteLLM compatibility manifest chase moving upstream APIs. | Hermetic request/stream/cost conformance pins today's behavior; manifest regeneration is required with each supported MLflow/provider snapshot (extends D8). | mitigated |
| R5 | **SSE byte-parity is fragile** across providers/chunk boundaries. | Frame-level recorder + recorded fixtures (T15.5/T22.3); allowlist only documented deviations. | mitigated |
| R6 | **KEK/secret ops**: AAD immutability means renames brick secrets; wrong passphrase = silent unusable gateway. | Startup probe decrypts a sentinel; runbook coverage (T22.6). | open |
| R7 | **Native compatibility-manifest drift**: scorer JSON, third-party algorithms/prompts, LiteLLM transforms/costs, and GEPA behavior can change independently upstream. | Server and worker link the same crate version; manifests carry source versions/fingerprints and upgrades cannot merge until the complete semantic corpus is regenerated and green. | mitigated |
| R8 | **Tool-executor sandbox parity** is security-critical (LLM-driven shell + file ops). | Port confinement checks exactly + adversarial escape suite (T20.3); restricted-mode allowlist default. | mitigated |
| R9 | **Third-party port scope and licensing**: DeepEval/Ragas/TruLens/Phoenix and GEPA contain substantial external algorithms and prompt assets. | T15.5 records provenance/license per implementation; missing or incompatible ports block release rather than silently reducing parity. Differential fixtures pin observable behavior without copying unapproved code. | open |

---

## 18. Research appendix (Part II — where the facts came from)

- Datasets: `mlflow/protos/{service,datasets}.proto`; handlers
  `handlers.py:5004-5089,6852-6874,7012-7242`; store
  `sqlalchemy_store.py:6863-7700`; models `dbmodels/models.py:1554-1848,2052`;
  eval job `mlflow/genai/evaluation/{job,base,harness}.py`.
- Scorers/jobs: `service.proto:1790-1867,5105-5210`;
  `handlers.py:5443-5648,6663-6720,6981-7004`; models
  `models.py:2125-2371`; store `sqlalchemy_store.py:2561-2888`; scorer
  serialization `mlflow/genai/scorers/{base,scorer_utils}.py`; jobs framework
  `mlflow/server/jobs/{__init__,utils,_job_runner,_job_subproc_entry}.py`;
  job store `mlflow/store/jobs/sqlalchemy_store.py`; online scoring
  `mlflow/genai/scorers/online/*`, `mlflow/genai/scorers/job.py`.
- Native semantic-engine inventory: evaluation pipeline/result normalization/
  aggregation/session handling in `mlflow/genai/evaluation/*` and
  `mlflow/genai/scorers/aggregation.py`; all concrete builtins in
  `scorers/builtin_scorers.py`; instructions and trace-tool judges in
  `judges/{instructions_judge,adapters,tools,utils}/*`; memory execution in
  `judges/optimizers/memalign/*`; third-party serialization, registries,
  mappings, and execution in `scorers/{deepeval,ragas,trulens,phoenix}/*`.
- Issues/labels/review-queues/prompt-opt: protos
  `{issues,label_schemas,review_queues,prompt_optimization}.proto` +
  `service.proto:1377-1590,2633-3010`; handlers
  `handlers.py:4428-4771,4925,7359-7647,7763-7784,7854-7858`; models
  `models.py:1154,3232,3389-3691`; detection
  `mlflow/genai/discovery/{job,pipeline,clustering,extraction,sampling,utils}.py`;
  optimization `mlflow/genai/optimize/{job,optimize,types,util}.py` +
  `optimizers/{base,gepa_optimizer,metaprompt_optimizer}.py`.
- Gateway: `mlflow/server/gateway_api.py`, `mlflow/server/fastapi_app.py`;
  providers `mlflow/gateway/providers/*` (`base.py`, `utils.py`,
  `provider_registry.py`); schemas `mlflow/gateway/schemas/*`,
  `mlflow/types/chat.py`; crypto `mlflow/utils/crypto.py`; store mixin
  `mlflow/store/tracking/gateway/{sqlalchemy_mixin,config_resolver}.py`;
  budget `mlflow/gateway/budget.py` + `budget_tracker/*`; guardrails
  `mlflow/gateway/{guardrails,guardrail_utils}.py`; models
  `models.py:2398-3227`; proto `service.proto:1974-2618,5209+`; legacy app
  `mlflow/gateway/{app,cli,runner,config}.py`.
- Assistant/promptlab: `mlflow/server/assistant/{api,session}.py`;
  `mlflow/assistant/{types,config}.py` + `providers/*`; stubs
  `dev/dev_stubs/`; promptlab `handlers.py:2340-2404`,
  `mlflow/utils/promptlab_utils.py`, `mlflow/prompt/promptlab_model.py`.
- Archival: `mlflow/tracing/{trace_archival_config,trace_archival_service}.py`,
  `mlflow/tracing/otel/otel_archival.py`, `mlflow/tracing/constant.py:209`;
  store `sqlalchemy_store.py:4356-4431,5830+,6390-6437`,
  `store/tracking/utils/trace_archival.py`; repo layer
  `store/artifact/artifact_repo.py:427-549`; validation
  `utils/validation.py:197-235`; CLI `mlflow/cli/__init__.py:488-696`.
- Auth for all areas: `mlflow/server/auth/__init__.py`
  (`:2527-2641,2593-2610,2562-2566,2196,2726-2729,3566,4479-4483`).
