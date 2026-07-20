# Differential Request Replay - Last Run

- Cases run: **267**  |  Non-allowlisted diffs: **0**  |  Allowlisted: **11**  |  Status mismatches: **0**  |  Errors: **0**

## Per-section

| Section | Cases | Status mismatch | Diffs | Allowlisted | Errors |
|---|---|---|---|---|---|
| artifacts | 5 | 0 | 0 | 0 | 0 |
| datasets | 17 | 0 | 0 | 0 | 0 |
| experiments | 19 | 0 | 0 | 1 | 0 |
| gateway | 47 | 0 | 0 | 0 | 0 |
| gateway_proxy_validation | 5 | 0 | 0 | 0 | 0 |
| graphql | 6 | 0 | 0 | 0 | 0 |
| invoke | 10 | 0 | 0 | 0 | 0 |
| issues | 7 | 0 | 0 | 0 | 0 |
| label_schemas | 9 | 0 | 0 | 0 | 0 |
| logged_models | 12 | 0 | 0 | 1 | 0 |
| metrics | 8 | 0 | 0 | 0 | 0 |
| prompt_optimization | 16 | 0 | 0 | 0 | 0 |
| registry | 26 | 0 | 0 | 0 | 0 |
| review_queues | 14 | 0 | 0 | 0 | 0 |
| runs | 20 | 0 | 0 | 0 | 0 |
| scorers | 13 | 0 | 0 | 0 | 0 |
| server_info | 3 | 0 | 0 | 0 | 0 |
| traces | 14 | 0 | 0 | 3 | 0 |
| webhooks | 7 | 0 | 0 | 6 | 0 |
| workspaces | 9 | 0 | 0 | 0 | 0 |

## Skipped sections

- **auth**: auth DB migration failed: The MLflow basic auth app requires the Flask-WTF package to perform CSRF validation. Please run `pip install mlflow[auth]` to install it.

## Allowlisted diffs (known, tolerated)

- experiments::experiment_create_duplicate `/message` - Python leaks the raw SQLAlchemy IntegrityError into the message, including the INSERT statement and its bound parameters (which contain the request's creation_time) — the text differs even between two Python runs, so byte-parity is impossible by construction. Rust returns the same leading "Experiment(name=...) already exists." sentence with a stable tail. Error code and status match.
- logged_models::dataset_search `/__raw_text__` - Flask default HTML 404 page vs empty axum body on an unmatched route; status matches.
- webhooks::webhook_create_bad_event `/__status__` - DELIBERATE DEVIATION - Python raises an unhandled exception on an unknown webhook entity (HTTP 500 with the Flask HTML error page); Rust returns a clean 400 INVALID_PARAMETER_VALUE naming the bad entity. Revisit if the Phase 12 Python-suite run asserts the 500.
- webhooks::webhook_create_bad_event `/message` - DELIBERATE DEVIATION - Python raises an unhandled exception on an unknown webhook entity (HTTP 500 with the Flask HTML error page); Rust returns a clean 400 INVALID_PARAMETER_VALUE naming the bad entity. Revisit if the Phase 12 Python-suite run asserts the 500.
- webhooks::webhook_create_bad_event `/__raw_text__` - DELIBERATE DEVIATION - Python raises an unhandled exception on an unknown webhook entity (HTTP 500 with the Flask HTML error page); Rust returns a clean 400 INVALID_PARAMETER_VALUE naming the bad entity. Revisit if the Phase 12 Python-suite run asserts the 500.
- webhooks::webhook_create_bad_event `/sqlstate` - DELIBERATE DEVIATION - Python raises an unhandled exception on an unknown webhook entity (HTTP 500 with the Flask HTML error page); Rust returns a clean 400 INVALID_PARAMETER_VALUE naming the bad entity. Revisit if the Phase 12 Python-suite run asserts the 500.
- webhooks::webhook_create_bad_event `/error_code` - DELIBERATE DEVIATION - Python raises an unhandled exception on an unknown webhook entity (HTTP 500 with the Flask HTML error page); Rust returns a clean 400 INVALID_PARAMETER_VALUE naming the bad entity. Revisit if the Phase 12 Python-suite run asserts the 500.
- webhooks::webhook_create_bad_event `/error_class` - DELIBERATE DEVIATION - Python raises an unhandled exception on an unknown webhook entity (HTTP 500 with the Flask HTML error page); Rust returns a clean 400 INVALID_PARAMETER_VALUE naming the bad entity. Revisit if the Phase 12 Python-suite run asserts the 500.
- traces::trace_get_info_v3 `/__raw_text__` - Flask default HTML 404 page vs empty axum body on an unmatched route; status matches.
- traces::trace_set_tag_v3 `/__raw_text__` - Flask default HTML 405 page vs empty axum body; status matches.
- traces::trace_get_info_missing `/__raw_text__` - Flask default HTML 404 page vs empty axum body on an unmatched route; status matches.

## Coverage notes

Corpus sections map to plan section 3 as follows:

- experiments -> 3.1 (CRUD, search POST+GET, pagination-walk, tags, errors)
- runs -> 3.2 (CRUD, log-metric/param/tag, log-batch, search-walk, errors)
- metrics -> 3.3 (get-history, get-history-bulk-interval, bulk ajax)
- logged_models -> 3.5 (create/get/search-walk/tags/artifacts-list, datasets 3.4)
- traces -> 3.6/3.7 (startTraceV3/end/search-walk/tag, OTLP 3.8)
- registry -> 3.14 (RM+MV CRUD/search-walk/stages/aliases/download-uri/errors)
- webhooks -> 3.15 (CRUD/test; local receiver skipped if unavailable)
- graphql -> 3.12 (getExperiment/getRun/searchModelVersions)
- server_info -> 3.13 (health/version/server-info)
- artifacts -> 3.11 (upload/list/download via proxy)
- auth (separate boot) -> 3.16 (401/403/admin/non-admin)
- workspaces (separate boot) -> 3.17 (X-MLFLOW-WORKSPACE scoping)
- datasets -> 12.1 (metadata/tags/records/associations, dedup + cursor walk)
- scorers -> 12.3 (CRUD/versioning, decorator rejection, online configs)
- issues -> 12.4 (CRUD/search; invoke lives in the isolated invoke section)
- label_schemas -> 12.5 (CRUD, lookup/list, immutable input-type validation)
- review_queues -> 12.6 (all 11 RPC operations + item status lifecycle)
- prompt_optimization -> 12.2/12.7 (CRUD + generic jobs get/cancel/state)
- invoke -> 12.2-12.4 (invoke handles, validation, batching, pre-created runs/tags)
- gateway -> 12.8 (all CRUD families + discovery and empty-target bridge behavior)
- gateway_proxy_validation -> 12.8 (GET/POST validation before a closed local target)

Deliberately deferred to follow-up (documented, not covered here): assessments
FieldMask update paths (3.9) beyond create/get; trace artifact fetch dispatch
on spansLocation (3.10); tracing V2 deprecated adapters (3.7) beyond the search
smoke; queryTraceMetrics / calculateTraceFilterCorrelation aggregations (3.6);
multipart artifact create/complete/abort + presigned URLs (3.11); full RBAC
role/permission matrix and after-request search filtering (3.16); workspace
delete modes RESTRICT/CASCADE/SET_DEFAULT (3.17). These are enumerated as the
extensibility backlog for the corpus.
