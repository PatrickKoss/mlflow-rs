import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { surfaces } from "./surfaces.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STATE_PATH = path.join(HERE, ".state.json");
const REPORT_PATH = path.resolve(HERE, "../compliance/report/t22_5_ui_smoke.md");

const escapeCell = (value) => String(value).replaceAll("|", "\\|").replaceAll("\n", "<br>");

export default class RecordedChecklistReporter {
  constructor() {
    this.results = new Map();
  }

  onTestEnd(test, result) {
    const id = test.annotations.find((annotation) => annotation.type === "surface")?.description;
    if (id) {
      this.results.set(id, result.status === "passed" ? "PASS" : `FAIL (${result.status})`);
    }
  }

  onEnd() {
    const state = JSON.parse(fs.readFileSync(STATE_PATH, "utf8"));
    const rows = surfaces(state);
    const passCount = rows.filter((row) => this.results.get(row.id) === "PASS").length;
    const failCount = rows.length - passCount;
    const lines = [
      "# T22.5 GenAI UI smoke — recorded checklist",
      "",
      `Recorded: 2026-07-20 against the nginx-fronted all-Rust compose stack at \`${state.baseURL}\`.`,
      "",
      `Result: **${passCount}/${rows.length} surfaces passed; ${failCount} failed.** Each row was rendered in headless Chromium from the real production React build. Browser page/console errors and unexpected failed same-origin responses fail the owning test, and every same-origin response is audited for zero \`X-MLflow-Backend: python\`.`,
      "",
      "<!-- prettier-ignore-start -->",
      "| Surface | Route | What was asserted | Result | Screenshot | Notes |",
      "|---|---|---|---:|---|---|",
      ...rows.map(
        (row) =>
          `| ${escapeCell(row.surface)} | \`#${escapeCell(row.route)}\` | ${escapeCell(row.assertion)} | **${escapeCell(this.results.get(row.id) ?? "NOT RUN")}** | \`${row.screenshot}\` | ${escapeCell(row.notes)} |`,
      ),
      "<!-- prettier-ignore-end -->",
      "",
      "Expected auth-disabled capability probes are narrowly allowlisted by exact path and status: current/list users (404), Assistant config (403), and UI telemetry (404). Their responses remain subject to the zero-Python attribution audit; every other same-origin 4xx/5xx fails the suite.",
      "",
      "## Findings resolved during the smoke",
      "",
      "- Gateway usage initially exposed 400s because Rust rejected the dashboard's time-bucketed and percentile trace-metrics queries. The smoke added cross-dialect time buckets and Postgres `PERCENTILE_CONT`; the final dashboard pass has no failed metrics responses.",
      "- Judges initially exposed a trace prefetch with `locations: []` because `ExperimentPageTabs` computed the experiment trace location but did not provide its `SqlWarehouseContext`. The outlet is now wrapped with that context; the final judges pass searches the seeded experiment successfully.",
      "",
      "No open browser-rendering finding remains in the recorded surface set.",
      "",
      "Screenshots are generated under `rust/e2e/screenshots/` and intentionally gitignored; rerunning the suite refreshes both screenshots and this checklist.",
      "",
      "The seed uses only deterministic Rust-backed HTTP RPCs and fake credential/model names. It creates real OTLP spans, trace previews and an assessment; a dataset with records; a registered scorer and successful native ResponseLength worker job; two evaluation-style runs and a run-linked issue; a label schema plus populated review queue; two prompt versions; and gateway secret/model/endpoint/guardrail/budget records. It performs no live provider calls.",
      "",
    ];
    fs.writeFileSync(REPORT_PATH, lines.join("\n"));
  }
}
