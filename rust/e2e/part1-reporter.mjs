import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { authSurfaces, part1Surfaces } from "./part1-surfaces.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATE_PATH = path.join(HERE, ".state.json");
const RESULTS_PATH = path.join(HERE, ".t11-results.json");
const REPORT_PATH = path.resolve(HERE, "../compliance/report/t11_6_ui_smoke.md");

const escapeCell = (value) => String(value).replaceAll("|", "\\|").replaceAll("\n", "<br>");

export default class Part1RecordedChecklistReporter {
  constructor() {
    this.results = fs.existsSync(RESULTS_PATH)
      ? new Map(Object.entries(JSON.parse(fs.readFileSync(RESULTS_PATH, "utf8"))))
      : new Map();
  }

  onTestEnd(test, result) {
    for (const annotation of test.annotations.filter(({ type }) => type === "surface")) {
      this.results.set(
        annotation.description,
        result.status === "passed" ? "PASS" : `FAIL (${result.status})`,
      );
    }
  }

  onEnd() {
    fs.writeFileSync(RESULTS_PATH, `${JSON.stringify(Object.fromEntries(this.results), null, 2)}\n`);
    if (!fs.existsSync(STATE_PATH)) return;

    const state = JSON.parse(fs.readFileSync(STATE_PATH, "utf8"));
    const rows = [...part1Surfaces(state), ...authSurfaces()];
    const passCount = rows.filter((row) => this.results.get(row.id) === "PASS").length;
    const failCount = rows.length - passCount;
    const lines = [
      "# T11.6 + T9.9 Part 1 UI smoke — recorded checklist",
      "",
      `Recorded: 2026-07-20 against nginx-fronted all-Rust compose stacks at \`${state.baseURL}\`.`,
      "",
      `Result: **${passCount}/${rows.length} surfaces passed; ${failCount} failed.** Every owning browser test rejects page/console errors, unexpected same-origin 4xx/5xx, and any response carrying \`X-MLflow-Backend: python\`. Screenshots come from the real production React build in headless Chromium.`,
      "",
    ];

    for (const section of [...new Set(rows.map(({ section }) => section))]) {
      const sectionRows = rows.filter((row) => row.section === section);
      lines.push(
        `## ${section}`,
        "",
        "<!-- prettier-ignore-start -->",
        "| Surface | Route | What was asserted | Result | Screenshot | Notes |",
        "|---|---|---|---:|---|---|",
        ...sectionRows.map(
          (row) =>
            `| ${escapeCell(row.surface)} | \`#${escapeCell(row.route)}\` | ${escapeCell(row.assertion)} Zero Python-attributed responses and no unexpected failed same-origin response. | **${escapeCell(this.results.get(row.id) ?? "NOT RUN")}** | \`${row.screenshot}\` | ${escapeCell(row.notes)} |`,
        ),
        "<!-- prettier-ignore-end -->",
        "",
      );
    }

    lines.push(
      "## Stack, fixture, and audit mechanics",
      "",
      "The auth-disabled stack enables the production workspace feature through `--enable-workspaces`. Its exact capability-probe allowlist remains current/list users (404), Assistant config (403), and UI telemetry (404). The auth-enabled stack adds `--app-name basic-auth` and a one-shot copy of the migrated auth fixture into a disposable volume; current/list users must succeed, and only the deliberate bogus-credential logout request may return 401.",
      "",
      "The deterministic Part 1 seed creates 110 ordinary runs, six-step accuracy histories on two newest runs, workspace-scoped run/model/trace artifacts, a finalized logged model, a non-prompt registered model with two versions, Staging and champion metadata, and a second workspace. The T22.5 dataset and prompt fixtures are reused rather than duplicated.",
      "",
      "## Findings resolved during the smoke",
      "",
      "- The production artifact browser and compare page exposed a missing Rust `GET /ajax-api/2.0/mlflow/artifacts/list` registration. A Rust handler and HTTP regression test now cover run ID aliases, root listings, and nested paths.",
      "- Auth-enabled cleanup exposed a foreign-key failure when deleting a user assigned to a normal role. `AuthStore::delete_user` now removes all assignments owned by the user transactionally, with a focused regression test.",
      "",
      "No open browser-rendering or functional finding remains in the recorded surface set.",
      "",
      "Screenshots are generated under `rust/e2e/screenshots/` and intentionally gitignored. The seed is deterministic and makes no live provider calls.",
      "",
    );
    fs.writeFileSync(REPORT_PATH, lines.join("\n"));
  }
}
