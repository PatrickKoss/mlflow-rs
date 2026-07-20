import { expect, test as base } from "@playwright/test";

export function createAuditedTest({ baseURL, expectedFailures = [] }) {
  const staticExpectedFailures = new Set(expectedFailures);

  return base.extend({
    browserAudit: [
      async ({ page }, use) => {
        const consoleErrors = [];
        const pageErrors = [];
        const pythonResponses = [];
        const unexpectedHttpFailures = [];
        const dynamicExpectedFailures = new Set();
        const expectedFailure = (pathname, status) =>
          staticExpectedFailures.has(`${pathname}|${status}`) ||
          dynamicExpectedFailures.has(`${pathname}|${status}`);

        page.on("console", (message) => {
          if (message.type() === "error") {
            consoleErrors.push({ text: message.text(), url: message.location().url });
          }
        });
        page.on("pageerror", (error) => pageErrors.push(error.message));
        page.on("response", (response) => {
          if (!response.url().startsWith(baseURL)) return;
          if (response.headers()["x-mlflow-backend"]?.toLowerCase() === "python") {
            pythonResponses.push(`${response.status()} ${response.url()}`);
          }
          if (response.status() >= 400) {
            const url = new URL(response.url());
            if (!expectedFailure(url.pathname, response.status())) {
              unexpectedHttpFailures.push(
                `${response.status()} ${response.request().method()} ${url.pathname}`,
              );
            }
          }
        });

        await use({
          allowFailure: (pathname, status) =>
            dynamicExpectedFailures.add(`${pathname}|${status}`),
        });

        const unexpectedConsoleErrors = consoleErrors.filter(({ text, url }) => {
          if (text.startsWith("Failed to load resource:") && url) {
            const responseUrl = new URL(url);
            return ![...staticExpectedFailures, ...dynamicExpectedFailures].some((entry) =>
              entry.startsWith(`${responseUrl.pathname}|`),
            );
          }
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
}

export const SHARED_CAPABILITY_FAILURES = [
  "/ajax-api/3.0/mlflow/assistant/config|403",
  "/ajax-api/3.0/mlflow/ui-telemetry|404",
];

export const AUTH_DISABLED_CAPABILITY_FAILURES = [
  "/ajax-api/2.0/mlflow/users/current|404",
  "/ajax-api/2.0/mlflow/users/list|404",
  ...SHARED_CAPABILITY_FAILURES,
];
