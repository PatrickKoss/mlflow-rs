import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test as base } from "@playwright/test";

import { SCREENSHOT_DIR, surfaces } from "../surfaces.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const state = JSON.parse(fs.readFileSync(path.resolve(HERE, "../.state.json"), "utf8"));
const surfaceById = new Map(surfaces(state).map((surface) => [surface.id, surface]));
const screenshotDir = fileURLToPath(SCREENSHOT_DIR);
const expectedCapabilityFailures = new Set([
  "/ajax-api/2.0/mlflow/users/current|404",
  "/ajax-api/2.0/mlflow/users/list|404",
  "/ajax-api/3.0/mlflow/assistant/config|403",
  "/ajax-api/3.0/mlflow/ui-telemetry|404",
]);

const test = base.extend({
  _browserAudit: [
    async ({ page }, use) => {
      const consoleErrors = [];
      const pageErrors = [];
      const pythonResponses = [];
      const unexpectedHttpFailures = [];

      page.on("console", (message) => {
        if (message.type() === "error") {
          consoleErrors.push({ text: message.text(), url: message.location().url });
        }
      });
      page.on("pageerror", (error) => pageErrors.push(error.message));
      page.on("response", (response) => {
        if (!response.url().startsWith(state.baseURL)) return;
        if (response.headers()["x-mlflow-backend"]?.toLowerCase() === "python") {
          pythonResponses.push(`${response.status()} ${response.url()}`);
        }
        if (response.status() >= 400) {
          const url = new URL(response.url());
          const key = `${url.pathname}|${response.status()}`;
          if (!expectedCapabilityFailures.has(key)) {
            unexpectedHttpFailures.push(
              `${response.status()} ${response.request().method()} ${url.pathname}`,
            );
          }
        }
      });

      await use();

      const unexpectedConsoleErrors = consoleErrors.filter(({ text, url }) => {
        // Auth-disabled OSS performs these capability probes intentionally. Chrome reports
        // their exact, separately-audited 403/404 responses as generic console errors.
        if (text.startsWith("Failed to load resource:") && url) {
          const pathname = new URL(url).pathname;
          return ![...expectedCapabilityFailures].some((key) => key.startsWith(`${pathname}|`));
        }
        // Chromium can emit this layout-only diagnostic while the review panes resize;
        // pageerror still catches actual uncaught ResizeObserver exceptions.
        return text !== "null ResizeObserver loop completed with undelivered notifications.";
      });
      expect(pageErrors, "uncaught browser errors").toEqual([]);
      expect(unexpectedHttpFailures, "unexpected failed same-origin responses").toEqual([]);
      expect(unexpectedConsoleErrors, "browser console errors").toEqual([]);
      expect(pythonResponses, "responses attributed to the Python backend").toEqual([]);
    },
    { auto: true },
  ],
});

function surface(id) {
  const definition = surfaceById.get(id);
  if (!definition) throw new Error(`unknown surface: ${id}`);
  test.info().annotations.push({ type: "surface", description: id });
  return definition;
}

async function openSurface(page, id) {
  const definition = surface(id);
  await page.goto(`/#${definition.route}`, { waitUntil: "domcontentloaded" });
  return definition;
}

async function capture(page, definition) {
  await page.screenshot({ path: path.join(screenshotDir, definition.screenshot), fullPage: true });
}

test("gateway-secrets-endpoints", async ({ page }) => {
  const configResponse = page.waitForResponse((response) =>
    response.url().includes("/gateway/secrets/config"),
  );
  const definition = await openSurface(page, "gateway-secrets-endpoints");
  const response = await configResponse;
  expect(response.status()).toBe(200);
  expect(await response.json()).toMatchObject({ secrets_available: true });
  await expect(page.getByRole("heading", { name: "Endpoints", exact: true })).toBeVisible();
  await expect(page.getByText("t22-5-deterministic-endpoint", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("gateway-endpoint-create", async ({ page }) => {
  const definition = await openSurface(page, "gateway-endpoint-create");
  await expect(page.getByRole("heading", { name: "Create endpoint", exact: true })).toBeVisible();
  await expect(page.getByText("Name", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Model", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("gateway-budgets", async ({ page }) => {
  const definition = await openSurface(page, "gateway-budgets");
  await expect(page.getByRole("heading", { name: "Budgets", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "$25.5", exact: true })).toBeVisible();
  await capture(page, definition);
});

test("gateway-usage", async ({ page }) => {
  const definition = await openSurface(page, "gateway-usage");
  await expect(page.getByRole("heading", { name: "Usage", exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Usage", exact: true })).toBeVisible();
  await capture(page, definition);
});

test("gateway-guardrails", async ({ page }) => {
  const definition = await openSurface(page, "gateway-guardrails");
  await expect(page.getByRole("tab", { name: "Guardrails", exact: true })).toBeVisible();
  await expect(
    page.getByText("T22.5 deterministic input guardrail", { exact: true }),
  ).toBeVisible();
  await capture(page, definition);
});

test("evaluation-runs", async ({ page }) => {
  const definition = await openSurface(page, "evaluation-runs");
  await expect(page.getByText("T22.5 deterministic evaluation", { exact: true })).toBeVisible();
  await expect(page.getByText("T22.5 issue detection", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("issues", async ({ page }) => {
  const definition = await openSurface(page, "issues");
  await expect(page.getByRole("tab", { name: "Issues", exact: true })).toBeVisible();
  await expect(page.getByText("T22.5 deterministic quality issue", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("datasets", async ({ page }) => {
  const definition = await openSurface(page, "datasets");
  await expect(page.getByText("T22.5 evaluation dataset", { exact: true })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Name", exact: true })).toBeVisible();
  await capture(page, definition);
});

test("dataset-records", async ({ page }) => {
  const definition = await openSurface(page, "dataset-records");
  await expect(page.getByText("T22.5 evaluation dataset", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Inputs", exact: true })).toBeVisible();
  await expect(page.getByText("What stack answered this request?", { exact: false })).toBeVisible();
  await capture(page, definition);
});

test("scorers", async ({ page }) => {
  const definition = await openSurface(page, "scorers");
  await expect(page.getByText("T22.5 response length", { exact: true })).toBeVisible();
  await expect(page.getByText("T22.5 input guardrail scorer", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("review-queues", async ({ page }) => {
  const definition = await openSurface(page, "review-queues");
  await expect(
    page.getByText("T22.5 deterministic review queue", { exact: true }).first(),
  ).toBeVisible();
  await expect(page.getByRole("radio", { name: "All (2)", exact: true })).toBeChecked();
  await expect(page.getByText("All (2)", { exact: true })).toBeVisible();
  await expect(page.getByText("deterministic question 1", { exact: false })).toBeVisible();
  await capture(page, definition);
});

test("labeling", async ({ page }) => {
  const definition = await openSurface(page, "labeling");
  await expect(
    page.getByText("Is the deterministic answer correct?", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Correct", { exact: true })).toBeVisible();
  await expect(page.getByText("Incorrect", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("experiment-prompts", async ({ page }) => {
  const definition = await openSurface(page, "experiment-prompts");
  await expect(page.getByText(state.promptName, { exact: true })).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Latest version", exact: true }),
  ).toBeVisible();
  await capture(page, definition);
});

test("global-prompts", async ({ page }) => {
  const definition = await openSurface(page, "global-prompts");
  await expect(page.getByRole("heading", { name: "Prompts", exact: true })).toBeVisible();
  await expect(page.getByText(state.promptName, { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("prompt-optimization", async ({ page }) => {
  const definition = await openSurface(page, "prompt-optimization");
  await expect(
    page.getByText("Answer {{question}} clearly and concisely.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Optimize", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("Optimize Prompt");
  await expect(page.getByRole("dialog")).toContainText("mlflow.genai.optimize_prompts");
  await capture(page, definition);
});

test("assistant", async ({ page }) => {
  const assistantConfigResponse = page.waitForResponse((response) =>
    response.url().includes("/assistant/config"),
  );
  const definition = await openSurface(page, "assistant");
  const response = await assistantConfigResponse;
  expect(response.status()).toBe(403);
  const toggle = page.locator('[data-assistant-ui="true"]').first();
  await expect(toggle).toBeVisible();
  if ((await toggle.getAttribute("aria-pressed")) !== "true") await toggle.click();
  await expect(
    page.getByRole("heading", { name: "Welcome to MLflow Assistant", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Get Started", exact: true })).toBeVisible();
  await capture(page, definition);
});
