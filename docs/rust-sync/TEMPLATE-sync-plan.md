# Rust upstream sync plan — <YYYY-MM-DD>

Status: **OPEN** · From: `<full-from-commit>` · To: `<full-to-commit>`

| Bucket       | Commits |
| ------------ | ------: |
| `server-api` | <count> |
| `ui`         | <count> |
| `client-sdk` | <count> |
| `infra`      | <count> |

When in doubt, the merged upstream Python implementation is the behavioral spec.

## Tasks

- [ ] T-S1 <title>
  - **Upstream refs:** `<sha>` <subject/PR>
  - **Rust target:** `rust/crates/<crate>/src/<file>.rs`
  - **AC:** <observable behavior that must match upstream Python>
  - **VER:** <specific corpus, conformance, recorder, or UI-smoke command/cases>

- [ ] T-S2 Rebuild the upstream UI and run UI smoke
  - **Upstream refs:** <all UI-bucket commits, or `N/A`>
  - **Rust target:** production React static build served by the Rust deployment
  - **AC:** The changed UI builds and all affected surfaces work against Rust without
    Python-attributed responses.
  - **VER:** `bash rust/e2e/run.sh`

## Skipped

- `client-sdk`: <commit refs> — not ported; the Python client, SDK, autologging, and
  flavors remain the client implementation.
- `infra`: <commit refs> — merged as upstream maintenance with no Rust behavior to port.

## Completion checklist

- [ ] Unary differential corpus replay is green:
      `uv run --no-sync python rust/compliance/replay.py`.
- [ ] Required Python-over-HTTP conformance matrix is green:
      `uv run --no-sync python rust/genai-inventory/run_conformance.py --profile required`.
- [ ] SSE/streaming recorder differentials are green:
      `uv run --no-sync pytest -q rust/compliance/recorders/`.
- [ ] Three-phase Playwright UI smoke is green: `bash rust/e2e/run.sh`.
- [ ] Production UI was rebuilt if the `ui` bucket was non-empty.
- [ ] New upstream endpoints have new corpus/conformance cases, not code-only coverage.
- [ ] `rust/sync/state.json` advances to `<full-to-commit>` and records this plan.
