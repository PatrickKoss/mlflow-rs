---
name: upstream-sync
description: Sync the Rust tracking server with upstream mlflow/mlflow — analyze drift, merge upstream, produce a sync plan, orchestrate Codex executors to port server-relevant changes, verify compliance gates, tick and advance state
disable-model-invocation: true
argument-hint: "[analyze|plan|execute|full] (default: full)"
arguments: [mode]
---

# Sync the Rust tracking server with upstream

## Usage and mode

Interpret `$mode` as one of `analyze`, `plan`, `execute`, or `full`; default to `full`.

- `analyze`: run phase 0 and stop.
- `plan`: run phases 0–2 and stop after writing the plan.
- `execute`: locate the newest unchecked `docs/rust-sync/*/plan.md`, confirm its
  upstream merge is present, then run phases 3–4.
- `full`: run phases 0–4 in order.

Read `docs/rust-sync/README.md` before starting. Never perform a merge in `analyze`
mode or advance `rust/sync/state.json` before phase 4 passes.

## Phase 0 — analyze

1. Read `rust/sync/state.json`. Record the full `from` SHA.
2. Run:

   ```bash
   python rust/sync/analyze_upstream.py --fetch \
     --output-dir docs/rust-sync/<YYYY-MM-DD>-<shortfrom>-<shortto>/
   ```

   Use nine-character `shortfrom` and `shortto` values. The analyzer writes
   `drift-report.md`; read the entire report and inspect every `server-api` commit.
3. If the mode is `analyze`, report the counts and stop without changing git history or
   sync state.

## Phase 1 — merge upstream

1. Confirm the working tree is clean and create `sync/upstream-<YYYY-MM-DD>` from the
   intended fork branch.
2. Run `git merge upstream/master`. **Do not rebase:** the merge commit must preserve
   the fork/upstream ancestry.
3. Resolve conflicts by responsibility:
   - Favor upstream for pure-Python files.
   - Favor the fork for `rust/**`, `docs/rust-tracking-server-plan/**`,
     `docs/rust-sync/**`, and fork deployment files (especially `rust/deploy/**`).
   - Reconcile mixed contract/generated files deliberately; do not apply a blanket
     ours/theirs resolution.
4. If `mlflow/server/js/**` changed, install with the pinned package manager and rebuild
   the production UI.
5. Run the Python test subset covering every touched server file. Fix merge errors in
   the Python reference before planning the Rust port.

## Phase 2 — plan

1. Copy `docs/rust-sync/TEMPLATE-sync-plan.md` to `plan.md` in the phase-0 dated
   folder and fill in the full from/to SHAs and exact analyzer counts.
2. Cluster related `server-api` commits into cohesive port tasks. Every task must have:
   - an unchecked `- [ ] T-S<N> <title>` checkbox;
   - upstream commit SHAs, subjects, and PR references;
   - concrete Rust target crates/files;
   - **AC** describing observable behavior parity with merged upstream Python; and
   - **VER** naming the compliance command and new/existing cases that prove the AC.
3. If the UI bucket is non-empty, create one task covering production rebuild plus
   `bash rust/e2e/run.sh` instead of one task per UI commit.
4. Record `client-sdk` and `infra` commits in **Skipped**, with one line of reasoning
   per bucket. Client/SDK/flavor code is intentionally never ported.
5. If the mode is `plan`, summarize the plan and stop.

## Phase 3 — execute

Act as coordinator. Use one task branch and one worktree per unchecked port task under
`/home/patrick/projects/mlflow-wt/`. Do not let executors share a worktree. Write a
self-contained `prompt.md` for each task containing the repository instructions, sync
plan task, upstream diff/refs, Rust targets, AC, VER, and current merge head.

Start each executor as a fresh background invocation from the coordinator shell:

```bash
codex exec --skip-git-repo-check -m gpt-5.6-sol \
  --config model_reasoning_effort="high" \
  --sandbox danger-full-access -C <worktree> - \
  < prompt.md > log 2>&1 &
```

Never pass `--full-auto`. Never use `codex exec resume`; make every retry a fresh
invocation with the complete current context in its prompt.

Require each executor to run its VER gate and commit with `git commit -s` plus:

```text
Co-Authored-By: Codex <codex@openai.com>
```

For each completed executor:

1. Review its diff and signed commit against the task AC.
2. If the coordinator head advanced, rebase only the disposable task branch onto that
   head. This does not replace the required upstream merge with a rebase.
3. Independently rerun the task's VER gate in the rebased task worktree.
4. Only after the gate passes, integrate with `git merge --ff-only <task-branch>`.
5. Rerun any gate affected by integration, add a dated DONE note/evidence, and tick the
   task checkbox. Never tick based only on an executor's report.

Resolve overlapping tasks serially or refresh later worktrees/prompts from the new
coordinator head. Keep all failures and follow-up fixes attached to the same task AC.

## Phase 4 — close

Proceed only when all task checkboxes are ticked. Run all four gates from the final
coordinator head:

```bash
uv run --no-sync python rust/compliance/replay.py
uv run --no-sync python rust/genai-inventory/run_conformance.py --profile required
uv run --no-sync pytest -q rust/compliance/recorders/
bash rust/e2e/run.sh
```

New upstream endpoints must have added unary corpus/conformance cases; streaming and UI
surfaces must also have recorder/Playwright coverage. Code-only parity is insufficient.

After every gate is green:

1. Tick the completion checklist in `plan.md` and mark the plan complete.
2. Append `{from, to, date, plan_doc, commits_total, commits_relevant}` to
   `rust/sync/state.json.history`.
3. Set `last_synced_upstream_commit` to the merged upstream `to` SHA and
   `last_synced_at` to the close date in the same commit.
4. Report the merge, port commits, task/gate evidence, skipped buckets, and new anchor.

Before finishing on WSL2, stop every server/executor and tear down containers and
volumes with the applicable `docker compose down -v --remove-orphans`. Leaked reference
servers can exhaust the pid cgroup and cause misleading flakes; run
`cargo run -p mlflow-test-support --bin reap-reference-servers` when pressure or
cross-test interference appears, then rerun the affected gate.
