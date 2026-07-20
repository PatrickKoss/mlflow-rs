import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.MLFLOW_E2E_BASE_URL ?? "http://127.0.0.1";
const suite = process.env.MLFLOW_E2E_SUITE ?? "genai";
const testMatch = {
  genai: "genai-ui-smoke.spec.mjs",
  part1: "part1-ui-smoke.spec.mjs",
  auth: "auth-ui-smoke.spec.mjs",
}[suite];
if (!testMatch) throw new Error(`unknown MLFLOW_E2E_SUITE: ${suite}`);
const reporters = [["line"], ["./part1-reporter.mjs"]];
if (suite === "genai") reporters.push(["./reporter.mjs"]);

export default defineConfig({
  testDir: "./tests",
  testMatch,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 45_000,
  expect: { timeout: 15_000 },
  reporter: reporters,
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    viewport: { width: 1440, height: 1000 },
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
    trace: "retain-on-failure",
    video: "off",
    ...(suite === "auth"
      ? { httpCredentials: { username: "admin", password: "password1234" } }
      : {}),
  },
});
