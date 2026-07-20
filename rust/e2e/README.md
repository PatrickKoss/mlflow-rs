# T22.5 GenAI UI smoke

This self-contained Playwright suite renders the GenAI UI against the nginx-fronted all-Rust reference compose stack. It seeds deterministic data through HTTP, fails on browser page/console errors, rejects every same-origin response carrying `X-MLflow-Backend: python`, captures one screenshot per surface, and rewrites `rust/compliance/report/t22_5_ui_smoke.md`.

Run the complete verification from the repository root:

```bash
bash rust/e2e/run.sh
```

The wrapper uses Node 24.14.0 with the repository-pinned Yarn 4.12.0 release for the production React build, applies the repository's seven-day npm cooldown to all `npx` calls, installs Playwright Chromium, creates a fresh compose database, runs the suite twice, and removes the compose stack and volumes through its exit trap.

For iteration against an already-running fresh stack and an existing real UI build:

```bash
cd rust/e2e
npm ci --min-release-age=7
npx --min-release-age=7 playwright install chromium
node seed.mjs
npm test
```

Set `MLFLOW_E2E_BASE_URL` to override the default `http://127.0.0.1`. Screenshots, Playwright output, dependencies, and generated seed state are gitignored; the Markdown checklist is committed.
