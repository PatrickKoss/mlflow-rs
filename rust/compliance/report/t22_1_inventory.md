# T22.1 GenAI differential corpus inventory

Inventory date: 2026-07-20. Counts are executable cases, not route counts.

- Request replay before T22.1: **188** cases.
- Request replay after T22.1: **271** cases (**+83**).
- Manifest-generated semantic corpus: **374** cases.
- Combined required compliance coverage: **645** cases.

## Tier A wire/state and submission inventory

| Surface | Corpus section | Before | After | T22.1 status |
|---|---|---:|---:|---|
| Evaluation datasets (all 12 RPC operations, both search verbs, record cursor walk/dedup) | `datasets` | 0 | 17 | Added; complete |
| Scorers CRUD/versioning + online configs | `scorers` | 0 | 13 | Added; complete |
| Issues CRUD/search | `issues` | 0 | 7 | Added; complete |
| Label schemas CRUD | `label_schemas` | 0 | 9 | Added; complete |
| Review queues (all 11 RPC operations + trace prerequisite) | `review_queues` | 0 | 14 | Added; complete |
| Prompt-optimization job CRUD | `prompt_optimization` | 11 | 13 | Extended; complete |
| Generic jobs get/cancel/get-state | `prompt_optimization` | 0 | 3 | Added; complete |
| Gateway secrets CRUD | `gateway` | 5 | 5 | Pre-existing; complete |
| Gateway endpoints CRUD | `gateway` | 4 | 5 | Extended with update; complete |
| Gateway model-definition lifecycle | `gateway` | 5 | 7 | Extended with a second attachable definition; complete |
| Gateway endpoint model attach/detach | `gateway` | 0 | 2 | Added; complete |
| Gateway endpoint bindings | `gateway` | 0 | 3 | Added; complete |
| Gateway endpoint tags | `gateway` | 1 | 2 | Extended with delete; complete |
| Gateway budgets/windows | `gateway` | 6 | 6 | Pre-existing; complete |
| Gateway guardrail configs (all 8 RPCs) | `gateway` | 0 | 8 | Added; complete |
| Gateway discovery + disabled legacy bridge | `gateway` | 8 | 8 | Pre-existing; complete |
| Gateway guardrail scorer prerequisite | `gateway` | 0 | 1 | Added deterministic setup case |
| Gateway closed-target bridge validation | `gateway_proxy_validation` | 5 | 5 | Pre-existing; complete |
| Evaluate/scorer/issue invoke submission + durable run/tag state | `invoke` | 10 | 10 | Pre-existing; complete |
| Prompt-optimization submission | `prompt_optimization` | 1 | 2 | Extended with isolated jobs-API fixture |

The `gateway` section is 47 cases overall (29 before T22.1). The
`prompt_optimization` section is 16 cases overall (11 before T22.1).

## Tier C semantic and pinned-manifest inventory

These cases are generated from the checked-in manifests/fixtures by
`rust/compliance/semantic.py`; no live provider is reachable.

| Semantic category / manifest | Semantic section | Cases | Status |
|---|---|---:|---|
| Scorer deserialization/execution: 24 builtins, 2 serialized judges, 112 third-party rows, 1 rejected-payload inventory row | `semantic_scorer_execution` | 139 | Complete; pinned scorer manifest fully covered |
| Pinned LiteLLM provider adapters/accounting/retry inventory | `semantic_provider_manifest` | 191 | Complete; 191/191, zero unsupported/Python fallback |
| Evaluation harness: rate parsing, result standardization, aggregate values/metrics | `semantic_evaluation_harness` | 26 | Complete |
| Issue discovery: sampling, latency, clustering, dedup, end-to-end persistence | `semantic_issue_discovery` | 7 | Complete |
| Online trace/session scoring submission with identical seed/shared DB | `semantic_online_scoring` | 1 | Complete |
| Pinned GEPA 0.0.27 + MetaPrompt algorithms | `semantic_prompt_optimization` | 2 | Complete; optimizer manifest fully covered |
| Inline judge guardrails: stage × action × outcome matrix | `semantic_inline_judge_guardrails` | 8 | Complete |

## Replay result

- Request replay: **271 cases**, **0 non-allowlisted diffs**, **11 existing
  allowlisted diffs**, **0 status mismatches**, **0 errors**, **0 skipped**.
- Semantic oracle-refresh replay: **374 cases**, **0 non-allowlisted diffs**,
  **0 allowlisted diffs**, **0 live provider calls**.
- New allowlist entries: **none**.

Evidence: `last_run.json`, `last_run.md`, `semantic_last_run.json`, and
`semantic_last_run.md` in this directory.
