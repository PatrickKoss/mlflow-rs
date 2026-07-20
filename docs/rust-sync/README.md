# Rust MLflow server — upstream sync process

Status: **ACTIVE MAINTENANCE PROCESS** · Baseline: `ee51c63ca0f541a47eb65a811b627755717da0a2`
· Started: 2026-07-20

The Rust tracking-server rewrite is complete, but the fork is maintained long-term and
upstream `mlflow/mlflow` remains the source of truth for features and server behavior.
Every server-facing upstream change must therefore be triaged and, when applicable,
reimplemented in Rust. Python client, SDK, autologging, `pyfunc`, and flavor changes are
permanently out of scope for the port: the Python client stays the client.

This process deliberately preserves fork history, produces tickable plans in the same
style as the completed rewrite, and proves each port against the existing differential
compliance system. The analyzer reports drift; it never merges or modifies source code.

---

## The sync loop

Invoke `.claude/skills/upstream-sync/SKILL.md` with `analyze`, `plan`, `execute`, or
`full`. A full sync follows these phases in order:

1. **Analyze.** Fetch `upstream/master`, read the anchor in `rust/sync/state.json`, and
   run `rust/sync/analyze_upstream.py`. Save `drift-report.md` under
   `docs/rust-sync/<YYYY-MM-DD>-<short-from>-<short-to>/`. Review every
   `server-api` commit before proceeding.
2. **Merge upstream.** Create `sync/upstream-<YYYY-MM-DD>` and merge
   `upstream/master`; never rebase the fork. Resolve pure-Python conflicts toward
   upstream and preserve the fork for Rust, rewrite-plan, sync-plan, and deployment
   machinery. Rebuild a changed UI and run the affected Python server tests so the
   merged Python reference is sound.
3. **Plan.** Copy `TEMPLATE-sync-plan.md` to the dated folder as `plan.md`. Cluster
   related `server-api` commits into tasks with upstream refs, Rust targets,
   acceptance criteria (AC), and a concrete verification method (VER). Represent all
   UI drift with one rebuild/UI-smoke task. Record client/infra drift as explicitly
   skipped, with a reason.
4. **Execute.** The coordinator gives each unchecked task to a fresh Codex executor in
   its own worktree. The executor commits the implementation, but the coordinator
   independently reruns the task's VER gate before a fast-forward merge and only then
   ticks the checkbox.
5. **Close.** Once every task and completion item is checked, run all four compliance
   gates below, advance `state.json` to the merged upstream commit, append its history
   record, and publish the final summary. Tear down all reference servers, containers,
   and volumes.

`analyze` stops after phase 0. `plan` stops after phase 2. `execute` resumes an existing
unchecked dated plan at phase 3. `full` performs phases 0–4.

## Drift classification

Rules live in the `CLASSIFICATION` structure at the top of
`rust/sync/analyze_upstream.py`. A commit can touch several buckets. Its primary bucket
is the highest match in `server-api > ui > client-sdk > infra`, and `mixed` records all
matches.

| Bucket | Meaning | Port action |
|---|---|---|
| `server-api` | Server handlers/auth/jobs, stores, protos, entities, webhooks, server-side tracing and GenAI, gateway/deployment-server code, Assistant runtime, or a UI file that defines a persisted/ajax contract. | Inspect and normally reimplement the behavior in Rust. Add or update differential coverage. |
| `ui` | `mlflow/server/js/**`. The Rust deployment serves the same production static build. | Usually free after rebuilding, but verify with UI smoke. Serialized, GraphQL, and ajax contract changes are also `server-api`. |
| `client-sdk` | Tracking and tracing clients, `pyfunc`, flavors, integrations, and autologging. | Do not port. Merge the Python implementation because Python remains the client. |
| `infra` | CI, docs, tests, developer tooling, locks, libraries, and otherwise non-runtime maintenance. | Merge normally; port nothing unless manual review finds server behavior hidden by an incomplete rule. |

Classification is deterministic path triage, not a substitute for reading the report.
Maintain the rules when upstream moves a server responsibility to a new path.

## Mandatory compliance gates

Every sync must pass all four gates before its plan or state can be marked complete:

1. **Unary corpus replay:**
   `uv run --no-sync python rust/compliance/replay.py`. Exit 0 means zero
   non-allowlisted Python-vs-Rust diffs.
2. **Python-over-HTTP conformance matrix:**
   `uv run --no-sync python rust/genai-inventory/run_conformance.py --profile required`.
   Run the full profile as appropriate for touched optional integrations.
3. **SSE recorder differentials:** build the release server and run
   `uv run --no-sync pytest -q rust/compliance/recorders/`. Preserve byte/frame-level
   checks; do not weaken them to make a sync green.
4. **Playwright UI smoke:** `bash rust/e2e/run.sh`. The harness rebuilds the production
   UI and runs GenAI, Part I, and auth-enabled admin/account phases (twice) against the
   nginx-fronted Rust stacks.

New upstream endpoints require new unary corpus or conformance cases in addition to the
Rust implementation. Existing green tests do not prove a previously unrepresented
route. Streaming endpoints also require recorder coverage, and new UI surfaces require
Playwright coverage.

## Sync state

`rust/sync/state.json` is the single machine-readable anchor:

| Field | Type | Meaning |
|---|---|---|
| `last_synced_upstream_commit` | full commit SHA string | Inclusive upstream commit whose server behavior is implemented and verified in Rust. |
| `last_synced_at` | `YYYY-MM-DD` string | Date the last full sync closed. |
| `note` | string | Human context for the current anchor. |
| `history` | array | Completed sync records, oldest first. |

Each `history` item has `from`, `to`, `date`, `plan_doc`, `commits_total`, and
`commits_relevant`. Use full upstream SHAs for `from`/`to`, a repository-relative path
for `plan_doc`, the analyzer's first-parent total for `commits_total`, and its primary
`server-api` count for `commits_relevant`.

| History field | Type | Meaning |
|---|---|---|
| `from` | full commit SHA string | Previous sync anchor. |
| `to` | full commit SHA string | Merged and verified upstream head. |
| `date` | `YYYY-MM-DD` string | Date the sync closed. |
| `plan_doc` | string | Repository-relative path to the completed `plan.md`. |
| `commits_total` | integer | First-parent commits analyzed in `from..to`. |
| `commits_relevant` | integer | Commits whose primary bucket was `server-api`. |

Advance state only after the merged upstream reference, every plan task, and every
compliance gate are verified. Append the history item and set
`last_synced_upstream_commit` to the same `to` SHA and `last_synced_at` to the close
date in one commit. Never advance the anchor for an analysis-only run.

## Cadence

Run the drift check weekly; `.github/workflows/upstream-drift.yml` does this
automatically. Perform a full sync at least once per upstream minor release, or sooner
when the `server-api` bucket exceeds roughly 20 commits. Small, frequent syncs keep
behavioral clusters and differential failures reviewable.

## Worked example: 2026-07-20

From the rewrite baseline `ee51c63ca` to `9f872c10c`, the analyzer found 41
first-parent commits: **13 `server-api`, 9 `ui`, 9 `client-sdk`, and 10 `infra`**.
The committed report is
[`2026-07-20-ee51c63ca-9f872c10c/drift-report.md`](2026-07-20-ee51c63ca-9f872c10c/drift-report.md).

Representative results show the intended boundary: MCP registry and model-registry
proto work is `server-api`; the runs tag-multiselect sort is `ui`; the PyTorch export
fix is `client-sdk`; Keras package/test maintenance is primary `client-sdk` and mixed
with `infra`; lockfile-only commits are `infra`. The experiment saved-view envelope is
primary `server-api` and mixed with `ui` because it changes a server-persisted contract
even though its implementation currently lives in the frontend.
