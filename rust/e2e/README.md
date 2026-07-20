# T22.5 + T11.6/T9.9 UI smoke

This self-contained Playwright harness renders both the GenAI and Part 1 UI against nginx-fronted all-Rust reference compose stacks. It seeds deterministic data through HTTP, fails on browser page/console errors, rejects every same-origin response carrying `X-MLflow-Backend: python`, captures one screenshot per surface, and rewrites `rust/compliance/report/t22_5_ui_smoke.md` plus `rust/compliance/report/t11_6_ui_smoke.md`.

Run the complete verification from the repository root:

```bash
bash rust/e2e/run.sh
```

The wrapper uses Node 24.14.0 with the repository-pinned Yarn 4.12.0 release for the production React build, applies the repository's seven-day npm cooldown to all `npx` calls, installs Playwright Chromium, and runs two complete rounds. Each round creates a fresh auth-disabled stack with `--enable-workspaces` for the GenAI and Part 1 suites, then a fresh auth-enabled stack with `--enable-workspaces --app-name basic-auth` for the admin/account suite. Every stack and volume is removed after its phase and again through the exit trap.

For iteration against an already-running fresh stack and an existing real UI build:

```bash
cd rust/e2e
npm ci --min-release-age=7
npx --min-release-age=7 playwright install chromium
node seed.mjs
npm run test:genai
npm run test:part1

# Against a separately started basic-auth stack:
npm run test:auth
```

Set `MLFLOW_E2E_BASE_URL` to override the default `http://127.0.0.1`. Screenshots, Playwright output, dependencies, and generated seed state are gitignored; the Markdown checklist is committed.
