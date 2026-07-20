# T11.6 + T9.9 Part 1 UI smoke — recorded checklist

Recorded: 2026-07-20 against nginx-fronted all-Rust compose stacks at `http://127.0.0.1`.

Result: **26/26 surfaces passed; 0 failed.** Every owning browser test rejects page/console errors, unexpected same-origin 4xx/5xx, and any response carrying `X-MLflow-Backend: python`. Screenshots come from the real production React build in headless Chromium.

## T11.6 auth-disabled Part 1

<!-- prettier-ignore-start -->
| Surface | Route | What was asserted | Result | Screenshot | Notes |
|---|---|---|---:|---|---|
| Experiment list | `#/experiments` | The seeded experiment rendered in the experiment list. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-experiment-list.png` | Populated tracking state. |
| Runs table + Load more | `#/experiments/1/runs` | The runs grid rendered more than one page and Load more fetched the next client page. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-runs-load-more.png` | The seed creates 110 ordinary runs; the UI requests 100 per page. |
| Run detail (GraphQL) | `#/experiments/1/runs/26c2f00df9de486881f8b7b6a4b4e6e9` | Run overview rendered the seeded run and its GraphQL request completed from Rust. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-run-detail.png` | GraphQL run-details feature flag is enabled in the production OSS build. |
| Run charts (bulk interval) | `#/experiments/1/runs?compareRunsMode=CHART` | The accuracy chart rendered and Rust served interval-sampled histories for both metric runs. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-runs-charts.png` | Each metric run has six deterministic steps. |
| Compare runs | `#/compare-runs?runs=["26c2f00df9de486881f8b7b6a4b4e6e9","a6ee4b593e0e4cc0a8033fe21b5dd9a8"]&experiments=["1"]` | The two seeded runs rendered on the comparison page with their accuracy data. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-compare-runs.png` | Direct route uses the production getCompareRunPageRoute shape. |
| Metric page | `#/metric/?runs=["26c2f00df9de486881f8b7b6a4b4e6e9","a6ee4b593e0e4cc0a8033fe21b5dd9a8"]&metric=%22accuracy%22&experiments=["1"]&plot_metric_keys=%5B%22accuracy%22%5D&plot_layout={}&x_axis=step&y_axis_scale=linear&line_smoothness=1&show_point=false&deselected_curves=[]&last_linear_y_axis_range=[]` | The accuracy metric plot rendered histories for both seeded runs. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-metric-page.png` | Direct route uses the production getMetricPageRoute query shape. |
| Run artifact browser | `#/experiments/1/runs/26c2f00df9de486881f8b7b6a4b4e6e9/artifacts` | The uploaded model directory and deterministic artifact rendered in the run artifact browser. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-run-artifacts.png` | Artifact bytes were uploaded through the Rust proxy plane. |
| Traces tab — list | `#/experiments/1/traces` | The populated traces table rendered deterministic request previews. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-traces-list.png` | The same real OTLP traces support the detail checks. |
| Trace detail — span tree | `#/experiments/1/traces/tr-11111111111111111111111111111111` | The trace detail rendered the root and child spans. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-trace-span-tree.png` | Two-level OTLP span tree. |
| Trace detail — attachment | `#/experiments/1/traces/tr-11111111111111111111111111111111` | The attachment URI rendered and Rust returned the seeded attachment bytes. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-trace-attachment.png` | Real text attachment stored below the trace artifact root. |
| Trace detail — assessment | `#/experiments/1/traces/tr-11111111111111111111111111111111` | The seeded correctness assessment name and true feedback value rendered. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-trace-assessment.png` | Read-only smoke of the real assessment record. |
| Logged models tab | `#/experiments/1/models` | The finalized logged model rendered in the experiment model list. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-logged-models.png` | Created and finalized through /api/2.0/mlflow/logged-models. |
| Logged model detail | `#/experiments/1/models/m-4237a5c382e247f9bccb6a78d4eaf852` | The finalized model detail rendered its ID, name, status, and parameters. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-logged-model-detail.png` | Populated finalized state. |
| Datasets dropdown + records | `#/experiments/1/datasets/d-cd78708689094e69abae111559e0de07` | The experiment Datasets navigation and populated dataset records rendered. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `dataset-records.png` | Shared with the T22.5 test; intentionally not duplicated. |
| Model Registry — models list | `#/models` | The seeded non-prompt registered model rendered in the global registry list. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-registry-models.png` | Prompt-tagged models remain excluded from this assertion. |
| Model Registry — model, versions, stages | `#/models/t11-6-registered-model` | The model overview rendered two versions, then version 1 rendered the seeded Staging transition. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-registry-model.png` | Both version rows are backed by real run artifacts. |
| Model Registry — version, alias, artifact download | `#/models/t11-6-registered-model/versions/2` | Version 2 rendered the champion alias; its Source Run link opened the artifact UI and downloaded MLmodel from Rust. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-registry-version.png` | The test verifies the response body, not only the download control. |
| Experiment prompts | `#/experiments/1/prompts` | The experiment-scoped prompt list rendered the seeded prompt. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `experiment-prompts.png` | Shared with the T22.5 test; intentionally not duplicated. |
| Global prompts | `#/prompts` | The global prompt registry rendered the seeded prompt. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `global-prompts.png` | Shared with the T22.5 test; intentionally not duplicated. |
| Prompt detail + optimization entry | `#/experiments/1/prompts/t22-5-support-prompt?promptVersion=2` | Prompt version detail rendered and the deterministic optimization instruction modal opened. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `prompt-optimization.png` | Shared with T22.5; no provider job was submitted. |
| Workspace selector | `#/experiments` | The production sidebar selector listed default and the seeded second workspace and switched context. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `part1-workspace-selector.png` | Fully enabled by Rust server-info from --enable-workspaces. |
<!-- prettier-ignore-end -->

## T9.9 auth-enabled admin/account

<!-- prettier-ignore-start -->
| Surface | Route | What was asserted | Result | Screenshot | Notes |
|---|---|---|---:|---|---|
| Admin console — user CRUD | `#/admin` | Created a disposable user through the UI, rendered its detail, then deleted it through the UI. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `auth-admin-user.png` | Bootstrap admin/password1234; cleanup verified in the users table. |
| Admin console — role CRUD | `#/admin?tab=roles` | Created a disposable role through the UI, rendered its detail, then deleted it through the UI. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `auth-admin-role.png` | Role belongs to the default workspace. |
| Admin console — EditAccessModal grant | `#/admin/users/t11-ui-user` | Assigned the disposable role to the user through EditAccessModal and verified the role on user detail. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `auth-edit-access.png` | Functional role grant, not a render-only modal check. |
| Account — current-user permissions | `#/account?tab=permissions` | The admin identity and its current-user permissions table rendered from authenticated Rust APIs. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `auth-account-permissions.png` | /users/current is asserted 200 with is_basic_auth=true. |
| Basic-auth logout behavior | `#/account` | Logout appeared only after is_basic_auth=true; clicking it cleared auth cookies and issued the deliberate bogus-credential users/current XHR. Zero Python-attributed responses and no unexpected failed same-origin response. | **PASS** | `auth-logout.png` | The one deliberate logout 401 is dynamically allowlisted; all normal authenticated probes must be 200. |
<!-- prettier-ignore-end -->

## Stack, fixture, and audit mechanics

The auth-disabled stack enables the production workspace feature through `--enable-workspaces`. Its exact capability-probe allowlist remains current/list users (404), Assistant config (403), and UI telemetry (404). The auth-enabled stack adds `--app-name basic-auth` and a one-shot copy of the migrated auth fixture into a disposable volume; current/list users must succeed, and only the deliberate bogus-credential logout request may return 401.

The deterministic Part 1 seed creates 110 ordinary runs, six-step accuracy histories on two newest runs, workspace-scoped run/model/trace artifacts, a finalized logged model, a non-prompt registered model with two versions, Staging and champion metadata, and a second workspace. The T22.5 dataset and prompt fixtures are reused rather than duplicated.

## Findings resolved during the smoke

- The production artifact browser and compare page exposed a missing Rust `GET /ajax-api/2.0/mlflow/artifacts/list` registration. A Rust handler and HTTP regression test now cover run ID aliases, root listings, and nested paths.
- Auth-enabled cleanup exposed a foreign-key failure when deleting a user assigned to a normal role. `AuthStore::delete_user` now removes all assignments owned by the user transactionally, with a focused regression test.

No open browser-rendering or functional finding remains in the recorded surface set.

Screenshots are generated under `rust/e2e/screenshots/` and intentionally gitignored. The seed is deterministic and makes no live provider calls.
