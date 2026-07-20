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

