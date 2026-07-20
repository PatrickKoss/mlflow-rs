## 8. Verification quick-reference (how to confirm the whole thing works)

1. **Route parity**: `rust/tools/route_parity.py` — Python `get_endpoints()` +
   auth-route dump equals the Rust route table (modulo documented genai routes).
2. **Wire parity**: JSON golden tests (T1.3) + differential replay harness (T12.4) with
   zero non-allowlisted diffs on sqlite and postgres, including auth'd multi-user and
   workspace-header scenarios.
3. **Behavioral parity**: `tests/tracking/test_rest_tracking.py`, client suites,
   registry REST checks, `tests/server/auth/`, and workspace endpoint/middleware suites
   green against Rust on the DB matrix (T12.1-T12.5).
4. **UI parity**: T11.6/T9.9 smoke CLOSED 2026-07-20 — 21/21 tracking, tracing,
   registry, prompts, and workspace-selector surfaces plus 5/5 auth-enabled
   admin/account flows are green through nginx with the frontend served statically;
   the response audit observed zero Python-attributed responses.
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
