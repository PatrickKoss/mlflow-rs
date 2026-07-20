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

