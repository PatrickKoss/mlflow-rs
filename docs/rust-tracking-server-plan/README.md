# Rust MLflow Server — Implementation Plan (index)

**Part I** (§1–§10, Phases 0–14): everything except genai. **Part II** (§11–§18,
Phases 15–23): the genai port — added 2026-07-17; goal is full Python-app parity
in a Python-free Rust deployment, retiring both the Python server plane and the
Python job-execution runtime.

Status: **Part 1 COMPLETE except T9.9/T11.6 (browser-driven UI validation,
deliberately deferred) — Phases 2–8, 10, 12, 13, and 14 done; Phase 9/11 done
except those two UI checks. Corpus GREEN + required CI gate; client suites 0
failures vs Rust; benchmarks, soak (67–106x memory reduction, 0 errors), and
operational docs landed. Part 2 (genai port) Phases 15–21 + 23 COMPLETE;
Phase 22 is in progress (T22.0–T22.3 done; T22.4 next) and is the only
remaining planned work.**
· Branch: `feature/rust-tracking-server` · Last updated: 2026-07-20

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
  (see T19.5 note); post-cleanup full suite 1,353 tests green. Phase 20
  (assistant + promptlab) COMPLETE 2026-07-19: all four tasks — sessions +
  9 routes with SSE/localhost parity, CLI providers (20/20 stub-frame
  differentials), OpenAI-compatible provider + openat2 tool sandbox with
  all-negative escape suite, promptlab + demo routes with cross-language
  pyfunc load. ALL §12 external route families now implemented — zero
  planned routes remain. Phase 21 (trace archival) COMPLETE 2026-07-19,
  **D6 CLOSED**: config/flag with 20/20 byte-identical error parity +
  5 s-TTL cache, store paths (archive→read→delete cycle matching Python
  on sqlite + live Postgres 16, T4.1/T4.5 stubs removed), OTLP traces.pb
  codec with cross-language golden fixtures byte-exact both directions,
  scheduler with same-seed decision differential (fairness = name-sort +
  shuffle, shared per-pass budget). Phase 23 added 2026-07-19 per user
  directive: genai perf/resource evaluation Python-vs-Rust (Phase 14
  style, deterministic fake providers, 1k–10k+ reqs/cell). **PHASE 23
  COMPLETE** 2026-07-20: all of T23.1–T23.5 done; soak passed the no-leak
  RSS check both sides; final report `rust/bench/genai_eval.md` written
  (Rust faster across families, 117.8× less RSS in soak, all regressions
  disclosed). **Phase 22 STARTED 2026-07-20 per user directive** (the earlier
  stop-after-Phase-23 directive is fulfilled): T22.0 (S3 artifact proxy/
  factory support) and T22.1 (Tier A + semantic differential corpus) are
  COMPLETE 2026-07-20. T22.1 expanded request replay from 188 to 271 cases
  and added 374 manifest-generated semantic cases; all 645 are green with
  zero non-allowlisted differences and no new allowlist entries. T22.2
  refreshed the 1,892-item reachability ledger at the T22.1 merge head
  (1,546 server-reachable / 346 client-only / 0 dead; zero unclassified or
  missing owners). The 2026-07-20 follow-up replaced the hand-written-only
  test band with an AST-derived inventory and a Python-over-HTTP mechanical
  baseline. One store-only review-queue test was excluded by that baseline,
  leaving 35 repointable definitions (41 collected cases), 183 client-only
  definitions, and 3,344 Python-internal definitions, all with explicit
  reasons. Fresh-server/fresh-database isolation made the 41-case baseline and
  release-Rust matrix green on SQLite + Postgres 16; client-only remained 292
  passed / 1 documented skip per backend. **T22.2 also found and fixed a genuine
  Rust parity bug: StartTraceV3 discarded embedded assessments; fix
  `84340d2e2` persists them atomically and returns them in TraceInfo.** Required
  core + validator run on every Rust CI build; the full optional-dependency
  matrix is nightly/manual. **T22.3 is COMPLETE 2026-07-20:** all 26 recorder
  pytest items are green against the release Rust server (gateway 1/1 with
  38/38 comparisons, Assistant HTTP/SSE 1/1 with 27/27 comparisons, CLI
  providers 21/21, OpenAI-compatible provider 3/3), with no parity regression
  and no weakened byte/frame checks. Required CI job `sse-recorders` builds
  the release server and standalone recorders and retains pytest/JUnit output
  on failure. T22.4 (nginx cutover) is next. See
  `part2-work-breakdown.md` §16 Phase 22. Phase 23 already
  finished, so T22.4's precondition on the Python container is satisfied.
- **D23 Phoenix license blocker** — RESOLVED: user approved the rejection
  approach 2026-07-18; rejection errors must point at builtin/instructions-
  judge equivalents (see D23 row in `part2-decisions-and-appendix.md`).
- **T9.9 + T11.6** — browser-driven UI validation, deliberately deferred to be
  done together.
- Deferred seams: postgres corpus support in replay.py (TODO(T12.5) markers),
  tracking read-replica split (T11.1 SEAM), workspaces_store.rs sqlite-only
  tests.
- **GCS/Azure artifact proxy schemes remain Python-only.** The S3 gap surfaced
  by T14.2/T23.4 is CLOSED by **T22.0** (2026-07-20): stock Rust builds now
  proxy S3-compatible destinations, including MinIO, multipart uploads,
  presigned downloads, and trace archival. GCS/Azure retain their documented
  `NOT_IMPLEMENTED` factory seams until separately implemented.
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

## How this plan is split (read this before picking up a task)

The plan was one 4,100-line file; it is now split across the files below so a
single agent can load only the parts a task needs. **The section numbering (§1–§18)
is continuous and unchanged across files** — a cross-reference like "see §12.8" or
"per D14" points to whatever file contains that section/decision, per the map below.
This README is the only file that carries the live **Status/Open** block above; the
section files carry the durable spec. When you complete a task, update **two** places:
the task's checkbox + DONE note in its work-breakdown file, and the Status/Open block
here if the phase-level state changed.

| File | Sections | What it is | Read it when… |
|---|---|---|---|
| `README.md` (this file) | Status/Open + this map | Live project state + navigation | Always start here — it tells you what's done and where to look. |
| `part1-overview.md` | §1 Goal & Non-Goals, §2 Target Architecture | Part I framing: scope, the nginx-fronted Rust architecture, tech choices | You need the big picture or the crate/deployment layout for non-genai work. |
| `part1-api-and-contracts.md` | §3 API Surface, §4 Wire-Compat Contract, §5 Storage & DB, §6 Compliance Strategy | The Part I *spec*: every endpoint to match, must-match wire behaviors, storage/migration strategy, how compliance is proven | Implementing or verifying a Part I (non-genai) task — this is the contract you match against. |
| `part1-work-breakdown.md` | §7 Work Breakdown (Phases 0–14) | Every Part I task with checkbox/AC/VER and DONE notes | Picking up or checking off a Part I task (T0.x–T14.x). |
| `part1-verification-and-decisions.md` | §8 Verification quick-ref, §9 Open decisions & risks, §10 Research appendix | How to confirm the whole Part I stack works; Part I decision log; where Part I facts came from | You need the end-to-end smoke commands, a Part I decision rationale (D1–D13), or a source citation. |
| `part2-overview.md` | Part II banner, §11 Goals & execution boundary, §12 GenAI API surface, §13 Storage & crypto, §14 Runtime engines, §15 Compliance (Part II) | The Part II *spec*: the Python-free execution boundary, every genai route (§12.1–§12.12), crypto/storage, the runtime engines to build, Part II compliance | Implementing or verifying any genai (Part II) task — start here for the genai contract, especially the §12 route inventory. |
| `part2-work-breakdown.md` | §16 Work Breakdown (Phases 15–23) | Every Part II task with checkbox/AC/VER and DONE notes | Picking up or checking off a genai task (T15.x–T23.x). **Phase 22 lives here and is the only unfinished work.** |
| `part2-decisions-and-appendix.md` | §17 Open decisions & risks (Part II), §18 Research appendix (Part II) | Part II decision log (D14–D23, incl. the Phoenix/D23 resolution) and Part II source citations | You need a genai decision rationale (e.g. D14 worker-subprocess model, D23 Phoenix rejection) or a Part II source citation. |

### Typical reading paths

- **Continue Phase 22 (the remaining work):** README (status) → `part2-work-breakdown.md` (§16, find the first `- [ ]` T22.x) → `part2-overview.md` (§12/§15 for the contract that task matches) → `part2-decisions-and-appendix.md` (any D-row the task cites).
- **Pick up a non-genai task:** README → `part1-work-breakdown.md` (find the task) → `part1-api-and-contracts.md` (the contract) → `part1-verification-and-decisions.md` (how to verify + relevant decision).
- **Just orienting:** README → `part1-overview.md` then `part2-overview.md`.

### Conventions (unchanged from the original plan)

- Every task has a checkbox (`- [ ]` / `- [x]`), **AC** (acceptance criteria), and **VER** (verification method). Completed tasks carry a **DONE** note with date, merge commit, and evidence.
- When in doubt, the Python implementation is the spec.
- Decisions are referenced by ID (D1, D14, …) and live in the two decision sections; risks/seams are tracked in the Open block above and the decision sections.
