import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.MLFLOW_E2E_BASE_URL ?? "http://127.0.0.1";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 45_000,
  expect: { timeout: 15_000 },
  reporter: [["line"], ["./reporter.mjs"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    viewport: { width: 1440, height: 1000 },
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
    trace: "retain-on-failure",
    video: "off",
  },
});
