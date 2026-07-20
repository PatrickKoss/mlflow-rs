# T22.5 GenAI UI smoke — recorded checklist

Recorded: 2026-07-20 against the nginx-fronted all-Rust compose stack at `http://127.0.0.1`.

Result: **16/16 surfaces passed; 0 failed.** Each row was rendered in headless Chromium from the real production React build. Browser page/console errors and unexpected failed same-origin responses fail the owning test, and every same-origin response is audited for zero `X-MLflow-Backend: python`.

<!-- prettier-ignore-start -->
| Surface | Route | What was asserted | Result | Screenshot | Notes |
|---|---|---|---:|---|---|
| Gateway secrets config + endpoints | `#/gateway` | Gateway secrets/config request completed and the populated endpoints surface rendered; zero Python-attributed responses. | **PASS** | `gateway-secrets-endpoints.png` | The OSS page consumes secrets/config as a capability gate; when available, its rendered state is the endpoints panel. |
| Gateway endpoint creation | `#/gateway/endpoints/create` | Create-endpoint form rendered provider/model/name controls; zero Python-attributed responses. | **PASS** | `gateway-endpoint-create.png` | No provider call or credential validation was submitted. |
| Gateway budgets | `#/gateway/budgets` | Budgets heading and seeded policy rendered; zero Python-attributed responses. | **PASS** | `gateway-budgets.png` | Populated policy state. |
| Gateway usage | `#/gateway/usage` | Usage page controls rendered; zero Python-attributed responses. | **PASS** | `gateway-usage.png` | Honest empty-usage state; Rust time-bucket and percentile metric requests completed, and no live inference traffic was generated. |
| Gateway endpoint guardrails | `#/gateway/endpoints/e-b75ed6b5bf64423faedd5266723994c6?tab=guardrails` | Endpoint detail Guardrails tab and seeded guardrail rendered; zero Python-attributed responses. | **PASS** | `gateway-guardrails.png` | Guardrail was attached through Rust RPCs; no model invocation occurred. |
| Evaluation runs | `#/experiments/1/evaluation-runs` | Evaluation-runs page and populated deterministic run rendered; zero Python-attributed responses. | **PASS** | `evaluation-runs.png` | Populated state includes a completed GenAI evaluation run and issue-detection run. |
| Issues | `#/experiments/1/evaluation-runs/cab620693bb74d32967b20603c8f5aa5/issues` | Issue-detection run Issues tab and seeded run-linked issue rendered; zero Python-attributed responses. | **PASS** | `issues.png` | Populated pending issue state. |
| Datasets | `#/experiments/1/datasets` | Datasets list and seeded dataset rendered; zero Python-attributed responses. | **PASS** | `datasets.png` | Populated dataset state. |
| Dataset records | `#/experiments/1/datasets/d-03751154358b4bbbaf3b4147b9a1fc93` | Dataset detail and populated records table rendered; zero Python-attributed responses. | **PASS** | `dataset-records.png` | Two deterministic records, one linked to a seeded trace. |
| Scorers / judges | `#/experiments/1/judges` | Judges page and registered deterministic scorer rendered; zero Python-attributed responses. | **PASS** | `scorers.png` | Registered ResponseLength scorer and seeded a successful native worker job; no external model. |
| Review queues | `#/experiments/1/review-queue?selectedQueueId=rq-6bab7e8759814366a4b9331243a71d41` | Selected populated review queue and pending trace rendered; zero Python-attributed responses. | **PASS** | `review-queues.png` | Populated custom queue state. |
| Labeling / focused review | `#/experiments/1/review-queue?selectedQueueId=rq-6bab7e8759814366a4b9331243a71d41&selectedItemId=tr-11111111111111111111111111111111` | FocusedReview rendered the trace and label-schema question controls; zero Python-attributed responses. | **PASS** | `labeling.png` | Read-only browser smoke; it does not mutate the seeded answer. |
| Experiment prompts | `#/experiments/1/prompts` | Experiment-scoped prompts list and seeded prompt rendered; zero Python-attributed responses. | **PASS** | `experiment-prompts.png` | Prompt is linked to the seeded experiment. |
| Global prompts | `#/prompts` | Global prompts list and seeded prompt rendered; zero Python-attributed responses. | **PASS** | `global-prompts.png` | Populated global registry state. |
| Prompt optimization | `#/experiments/1/prompts/t22-5-support-prompt?promptVersion=2` | Prompt details rendered and Optimize Prompt modal opened; zero Python-attributed responses. | **PASS** | `prompt-optimization.png` | Instruction modal only; no optimizer/provider job was submitted. |
| Assistant panel (compose unauthenticated state) | `#/experiments/1/evaluation-runs` | Global Assistant drawer opened and setup/unauthenticated state rendered; zero Python-attributed responses, including assistant config. | **PASS** | `assistant.png` | The expected config response was Rust-attributed 403. Authenticated CLI chat frame parity is covered by rust/compliance/recorders/test_assistant_cli_provider_differential.py (T20.2). |
<!-- prettier-ignore-end -->

Expected auth-disabled capability probes are narrowly allowlisted by exact path and status: current/list users (404), Assistant config (403), and UI telemetry (404). Their responses remain subject to the zero-Python attribution audit; every other same-origin 4xx/5xx fails the suite.

## Findings resolved during the smoke

- Gateway usage initially exposed 400s because Rust rejected the dashboard's time-bucketed and percentile trace-metrics queries. The smoke added cross-dialect time buckets and Postgres `PERCENTILE_CONT`; the final dashboard pass has no failed metrics responses.
- Judges initially exposed a trace prefetch with `locations: []` because `ExperimentPageTabs` computed the experiment trace location but did not provide its `SqlWarehouseContext`. The outlet is now wrapped with that context; the final judges pass searches the seeded experiment successfully.

No open browser-rendering finding remains in the recorded surface set.

Screenshots are generated under `rust/e2e/screenshots/` and intentionally gitignored; rerunning the suite refreshes both screenshots and this checklist.

The seed uses only deterministic Rust-backed HTTP RPCs and fake credential/model names. It creates real OTLP spans, trace previews and an assessment; a dataset with records; a registered scorer and successful native ResponseLength worker job; two evaluation-style runs and a run-linked issue; a label schema plus populated review queue; two prompt versions; and gateway secret/model/endpoint/guardrail/budget records. It performs no live provider calls.
