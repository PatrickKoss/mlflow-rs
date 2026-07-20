# Rust upstream sync plan — 2026-07-20

Status: **OPEN** · From: `ee51c63ca0f541a47eb65a811b627755717da0a2` · To: `9f872c10cd88eaa062e612b6726b06cedbd75084`

| Bucket | Commits |
|---|---:|
| `server-api` | 13 |
| `ui` | 9 |
| `client-sdk` | 9 |
| `infra` | 10 |

Upstream merge commit: `e9c4da346` (merged 2026-07-20; one trivial import
conflict in `tests/server/auth/test_auth.py`).

When in doubt, the merged upstream Python implementation is the behavioral spec.

## Tasks

- [ ] T-S1 MCP server registry backend
  - **Upstream refs:** `14414ab00` Add MCP registry backend and clients (#24380); `d52771638` Harden MCP registry auth, validation, and migration edges (#24479)
  - **Rust target:** `rust/crates/mlflow-store` (entities + sqlalchemy-mixin parity for mcp_servers/mcp_server_versions/mcp_access_endpoints tables incl. tags/aliases + search), `rust/crates/mlflow-server` (24 endpoints under `/api/3.0/mlflow/mcp-servers` and the ajax prefix, per `mlflow/server/mcp_server_api.py`), `rust/crates/mlflow-auth` (new MCP permissions per `mlflow/server/auth/permissions.py`), workspace scoping per `sqlalchemy_workspace_store.py`
  - **AC:** Every MCP registry endpoint returns byte-identical (allowlist-aware) responses vs the merged Python server on sqlite+postgres, including validation errors (semver rules in `semver_utils.py`, name rules in `validation.py`), auth enforcement, and workspace scoping. Alembic migration `a8b9c0d1e2f3` is schema owner — Rust consumes, never migrates.
  - **VER:** New conformance suite `mcp_server_registry` in `rust/genai-inventory/run_conformance.py` matrix (Python-HTTP baseline, sqlite+postgres) + unary corpus cases in `rust/compliance/` for CRUD/search/tags/aliases/error paths; Rust store + auth unit tests.

- [x] T-S2 Trace token-usage rollup parity fix — DONE 2026-07-20 (`5a47f20ce`): real Rust defect — trace-level token usage was never aggregated; now recomputed from the persisted span tree with rollup parents suppressing descendants. 14 spans_store tests + 2 new OTLP corpus cases; coordinator-verified: focused tests green, replay 267 cases / 0 non-allowlisted diffs.
  - **Upstream refs:** `bdc41820c` Fix double-counted trace token usage for rollup parent spans in `SqlAlchemyStore.log_spans` (#24339)
  - **Rust target:** `rust/crates/mlflow-store/src/store/` span-logging path (log_spans token aggregation)
  - **AC:** Logging spans where a parent span rolls up child token usage yields the same trace-level token totals as merged Python (no double count), for both single-batch and incremental log_spans; matches upstream's `test_sqlalchemy_store_traces.py` additions.
  - **VER:** Rust store test mirroring upstream's regression test + corpus replay case exercising rollup-parent span ingestion, zero non-allowlisted diffs.

- [x] T-S3 Assistant server behavior parity — DONE 2026-07-20 (`1b4ffb39c`): per-route remote-access policies (`MLFLOW_ENABLE_REMOTE_ASSISTANT` + API-based mlflow_gateway provider only; CLI providers localhost-only), empty-error fallback, prompt text byte-identical (claude 14,156 B / codex 14,528 B). Coordinator-verified: full recorder suite 31 passed, twice.
  - **Upstream refs:** `a716c25e78` Allow remote access for API-based providers (#24040; `MLFLOW_ENABLE_REMOTE_ASSISTANT`, localhost-gating rework in `mlflow/server/assistant/api.py`, provider `supports_remote` capability); `2dd3419be` Never surface an empty error message (#24417); `2447f582e` Prompt scope guard + response-length discipline (#24445)
  - **Rust target:** `rust/crates/mlflow-server/src/assistant_providers/` + assistant API routes
  - **AC:** Remote (non-localhost) assistant requests are allowed iff the env flag is set AND the selected provider is API-based (CLI providers remain localhost-only); error payloads never carry an empty message; provider system prompts match upstream's updated text.
  - **VER:** `uv run --no-sync pytest -q rust/compliance/recorders/` (assistant SSE recorder cases extended for the remote-access matrix + empty-error case); conformance rows for the localhost/remote × provider-type matrix.

- [x] T-S4 Gateway trace kvlist normalization — DONE 2026-07-20 (`5854dc738`): real divergence found — Rust materialized optional `null` fields into stored span inputs where Python strips them (`model_dump(exclude_none=True)`); recursive normalization added + persisted-spanInputs differential. Coordinator-verified in the full recorder suite.
  - **Upstream refs:** `abc652d7c` Fix gateway Try-in-Browser traces rendering as raw `kvlist` data (#24400; `mlflow/gateway/tracing_utils.py`)
  - **Rust target:** Rust gateway trace-emission path (`rust/crates/mlflow-server`/`mlflow-genai` gateway tracing utils)
  - **AC:** Traces produced by gateway Try-in-Browser store normalized attribute values (not raw kvlist protobuf shapes), matching merged Python's span attribute JSON.
  - **VER:** Conformance/corpus case comparing gateway-produced trace span attributes vs Python baseline; UI smoke assertion on the gateway playground trace rendering path if reachable in `rust/e2e`.

- [ ] T-S5 Model catalog + UC proto wire parity
  - **Upstream refs:** `b7ad14743` + `9f872c10c` model catalog updates (gemini/vertex_ai/fireworks JSONs, consumed via `mlflow/utils/providers.py` by `mlflow/server/handlers.py` and genai judges/gateway adapters); `2efc3f844` UC-native model-registry protos (additive, #24412); `abb91a264` Restore oneof on `TemporaryCredentials.credentials` (#24489)
  - **Rust target:** `rust/genai-inventory/` manifests (+ `validate_ledger.py` checksums) and compiled-in provider matrix (`gateway_provider_matrix.rs`); `rust/crates/mlflow-proto` regeneration from updated `.proto` files
  - **AC:** Rust-served model/provider catalog data matches merged Python for the updated catalogs; regenerated protos compile and preserve wire compatibility (oneof restored); any UC-native surface Rust does not serve is explicitly recorded as out-of-scope with the reason.
  - **VER:** `rust/genai-inventory/validate_ledger.py` green; conformance rows touching catalog-backed responses; `cargo test -p mlflow-proto` + full corpus replay for wire regressions.

- [x] T-S6 Job-runner orphan shutdown + periodic lock recovery semantics — DONE 2026-07-20 (`b5e584717`): Rust design already correct (process-group guard + kill_on_drop; scheduler lease takeover); proof tests added (job_runner.rs, trace_archival_scheduler.rs), coordinator-verified green.
  - **Upstream refs:** `2b4acea79` Fix Huey orphan shutdown and periodic lock recovery (#24492)
  - **Rust target:** `rust/crates/mlflow-server/src/job_runner.rs`, `rust/crates/mlflow-genai/src/jobs.rs`
  - **AC:** Rust's native job runner exhibits the behaviors upstream fixed in the Huey implementation: orphaned job processes are shut down when the runner stops, and stale periodic-task locks are recovered after an unclean crash (no permanently wedged periodic tasks). Where Rust's design already guarantees this (kill_on_drop, DB-lock lease expiry), prove it with a test instead of porting code.
  - **VER:** Rust job-runner tests covering orphan-on-shutdown and stale-lock recovery (mirror upstream's `test_jobs.py` additions where applicable).

- [ ] T-S7 Rebuild the upstream UI and run UI smoke
  - **Upstream refs:** all 9 `ui` commits + js portions of `7240aa5a2` (saved-view tag envelopes — client-side encoding over existing tag APIs), `a716c25e78`, `abc652d7c`, `49408bd845`
  - **Rust target:** production React static build served by the Rust deployment
  - **AC:** The merged UI builds cleanly and all e2e-covered surfaces work against Rust with zero `X-MLflow-Backend: python` responses and no unexpected 4xx.
  - **VER:** `bash rust/e2e/run.sh` — all three phases green (production build already rebuilt in phase 1).

## Skipped

- `client-sdk` (9 commits: openai/langchain/llama_index/pydantic_ai autologging, pytorch/keras/pyspark flavors, tracing export client, uri utils): the Python client remains the supported client; client-side code is never ported to Rust.
- `infra` (10 commits: CI workflows, TypeScript gates, uv.lock, test-only hardening, docs): no server behavior; arrives via the merge itself.

## Completion checklist

- [ ] All port tasks ticked with dated DONE notes and evidence
- [ ] `uv run --no-sync python rust/compliance/replay.py` — exit 0
- [ ] `uv run --no-sync python rust/genai-inventory/run_conformance.py --profile required` — matrix identical
- [ ] `uv run --no-sync pytest -q rust/compliance/recorders/` — green
- [ ] `bash rust/e2e/run.sh` — three phases green
- [ ] `rust/sync/state.json` advanced to `9f872c10cd88eaa062e612b6726b06cedbd75084` with history entry
