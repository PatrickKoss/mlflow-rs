import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "@playwright/test";

import { AUTH_DISABLED_CAPABILITY_FAILURES, createAuditedTest } from "../browser-audit.mjs";
import { part1Surfaces } from "../part1-surfaces.mjs";
import { SCREENSHOT_DIR } from "../surfaces.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const state = JSON.parse(fs.readFileSync(path.resolve(HERE, "../.state.json"), "utf8"));
const surfaceById = new Map(part1Surfaces(state).map((definition) => [definition.id, definition]));
const screenshotDir = fileURLToPath(SCREENSHOT_DIR);
const test = createAuditedTest({
  baseURL: state.baseURL,
  expectedFailures: AUTH_DISABLED_CAPABILITY_FAILURES,
});

function surface(...ids) {
  for (const id of ids) {
    if (!surfaceById.has(id)) throw new Error(`unknown surface: ${id}`);
    test.info().annotations.push({ type: "surface", description: id });
  }
  return surfaceById.get(ids[0]);
}

async function openSurface(page, id) {
  const definition = surface(id);
  const separator = definition.route.includes("?") ? "&" : "?";
  await page.goto(`/#${definition.route}${separator}workspace=default`, {
    waitUntil: "domcontentloaded",
  });
  return definition;
}

async function capture(page, definition) {
  await page.screenshot({ path: path.join(screenshotDir, definition.screenshot), fullPage: true });
}

test("experiment-list", async ({ page }) => {
  const definition = await openSurface(page, "experiment-list");
  await expect(page.getByRole("heading", { name: "Experiments", exact: true })).toBeVisible();
  await expect(page.getByText("T22.5 GenAI UI Smoke", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("runs-table-pagination", async ({ page }) => {
  const definition = await openSurface(page, "runs-table-pagination");
  await expect(page.getByText("T11.6 pagination run 30", { exact: true })).toBeVisible();
  const loadMore = page.getByRole("button", { name: "Load more", exact: true });
  await expect(loadMore).toBeVisible();
  await loadMore.click();
  await expect(page.getByText("T11.6 metric run 1", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("run-detail-graphql", async ({ page }) => {
  const graphqlResponse = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/graphql" && response.request().method() === "POST",
  );
  const definition = await openSurface(page, "run-detail-graphql");
  const response = await graphqlResponse;
  expect(response.status()).toBe(200);
  expect((await response.json()).errors ?? null).toBeNull();
  await expect(page.getByText("T11.6 metric run 1", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("optimizer", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("charts-bulk-interval", async ({ page }) => {
  const intervalResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/mlflow/metrics/get-history-bulk-interval"),
  );
  const definition = await openSurface(page, "charts-bulk-interval");
  const response = await intervalResponse;
  expect(response.status()).toBe(200);
  const body = await response.json();
  expect(body.metrics?.length).toBeGreaterThanOrEqual(2);
  await expect(page.getByText("accuracy", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("compare-runs", async ({ page }) => {
  const definition = await openSurface(page, "compare-runs");
  await expect(page.getByText("T11.6 metric run 1", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("T11.6 metric run 2", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("accuracy", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("metric-page", async ({ page }) => {
  const definition = await openSurface(page, "metric-page");
  await expect(page.getByText("accuracy", { exact: true }).first()).toBeVisible();
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible();
  await capture(page, definition);
});

test("artifact-browser", async ({ page }) => {
  const definition = await openSurface(page, "artifact-browser");
  await expect(page.getByText("model", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("notes", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("traces-list", async ({ page }) => {
  const definition = await openSurface(page, "traces-list");
  await expect(page.getByText("deterministic question 1", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("deterministic question 2", { exact: false }).first()).toBeVisible();
  await capture(page, definition);
});

test("trace-span-tree-and-attachment", async ({ page }) => {
  const definition = surface("trace-span-tree", "trace-attachment");
  const attachmentResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname.endsWith("/mlflow/get-trace-artifact") && url.searchParams.has("path");
  });
  await page.goto(`/#${definition.route}?workspace=default`, { waitUntil: "domcontentloaded" });
  const attachmentLink = page.getByText("Download text/plain (35 B)", { exact: true });
  await expect(attachmentLink).toBeVisible();
  await attachmentLink.click();
  const response = await attachmentResponse;
  expect(response.status()).toBe(200);
  expect(await response.text()).toBe("T11.6 deterministic trace attachment");
  await page.getByRole("tab", { name: "Details & Timeline", exact: true }).click();
  await expect(page.getByText("answer-question-1", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("deterministic-tool-1", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("trace-assessment", async ({ page }) => {
  const definition = await openSurface(page, "trace-assessment");
  await expect(page.getByText("correctness", { exact: true }).first()).toBeVisible();
  await expect(page.locator('[role="status"]').filter({ hasText: "True" })).toBeVisible();
  await capture(page, definition);
});

test("logged-models-list", async ({ page }) => {
  const definition = await openSurface(page, "logged-models-list");
  await expect(page.getByText("T11-6 deterministic logged model", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("logged-model-detail", async ({ page }) => {
  const definition = await openSurface(page, "logged-model-detail");
  await expect(page.getByRole("heading", { name: "T11-6 deterministic logged model", exact: true })).toBeVisible();
  await expect(page.getByText(state.loggedModelId, { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Ready", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("framework", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("registry-models-list", async ({ page }) => {
  const definition = await openSurface(page, "registry-models-list");
  await expect(page.getByRole("heading", { name: "Registered Models", exact: true })).toBeVisible();
  await expect(page.getByText(state.registeredModelName, { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("registry-model-overview", async ({ page }) => {
  const definition = await openSurface(page, "registry-model-overview");
  await expect(page.getByRole("heading", { name: state.registeredModelName, exact: true })).toBeVisible();
  await expect(page.getByText("Version 1", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Version 2", { exact: true }).first()).toBeVisible();
  await page.getByRole("link", { name: "Version 1", exact: true }).click();
  await expect(page.getByText("Staging", { exact: true })).toBeVisible();
  await capture(page, definition);
});

test("registry-version-detail", async ({ page }) => {
  const artifactResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname.endsWith("/model-versions/get-artifact") && url.searchParams.get("path") === "MLmodel";
  });
  const definition = await openSurface(page, "registry-version-detail");
  await expect(page.getByRole("heading", { name: "Version 2", exact: true })).toBeVisible();
  await expect(page.getByRole("status", { name: "champion", exact: true })).toBeVisible();
  const response = await artifactResponse;
  expect(response.status()).toBe(200);
  await expect(response.text()).resolves.toContain("artifact_path: model");
  await page.getByRole("link", { name: "T11.6 metric run 1", exact: true }).click();
  await expect(page.getByText("MLmodel", { exact: true }).first()).toBeVisible();
  await capture(page, definition);
});

test("workspace-selector", async ({ page }) => {
  test.info().annotations.push({ type: "surface", description: "basic-auth-logout" });
  const definition = await openSurface(page, "workspace-selector");
  await expect(page.getByText("Logout", { exact: true })).toHaveCount(0);
  const selector = page.getByRole("combobox", { name: "Workspace" });
  await expect(selector).toBeVisible();
  await selector.click();
  await expect(page.getByText("default", { exact: true }).last()).toBeVisible();
  const second = page.getByText(state.secondWorkspaceName, { exact: true }).last();
  await expect(second).toBeVisible();
  await second.click();
  await expect(page).toHaveURL(new RegExp(`workspace=${state.secondWorkspaceName}`));
  await capture(page, definition);
});
