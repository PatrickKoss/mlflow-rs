import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "@playwright/test";

import { createAuditedTest, SHARED_CAPABILITY_FAILURES } from "../browser-audit.mjs";
import { authSurfaces } from "../part1-surfaces.mjs";
import { SCREENSHOT_DIR } from "../surfaces.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const state = JSON.parse(fs.readFileSync(path.resolve(HERE, "../.state.json"), "utf8"));
const screenshotDir = fileURLToPath(SCREENSHOT_DIR);
const authSurfaceById = new Map(authSurfaces().map((definition) => [definition.id, definition]));
const test = createAuditedTest({
  baseURL: state.baseURL,
  expectedFailures: SHARED_CAPABILITY_FAILURES,
});

const username = "t11-ui-user";
const roleName = "t11-ui-role";

function annotate(...ids) {
  for (const id of ids) {
    if (!authSurfaceById.has(id)) throw new Error(`unknown auth surface: ${id}`);
    test.info().annotations.push({ type: "surface", description: id });
  }
}

async function capture(page, id) {
  await page.screenshot({
    path: path.join(screenshotDir, authSurfaceById.get(id).screenshot),
    fullPage: true,
  });
}

async function rustJson(page, route, method = "GET", body) {
  return page.evaluate(
    async ({ route, method, body }) => {
      const response = await fetch(route, {
        method,
        headers: body === undefined ? undefined : { "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      return {
        status: response.status,
        backend: response.headers.get("x-mlflow-backend"),
        body: await response.json().catch(() => ({})),
      };
    },
    { route, method, body },
  );
}

test("auth-enabled admin/account functional flow", async ({ page, browserAudit }) => {
  annotate(
    "admin-users-crud",
    "admin-roles-crud",
    "admin-edit-access",
    "account-permissions",
    "basic-auth-logout",
  );

  await page.goto("/#/admin?workspace=default", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Platform Admin", exact: true })).toBeVisible();
  await expect(page.getByText("admin", { exact: true }).first()).toBeVisible();

  const createUserResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/mlflow/users/create"),
  );
  await page.getByRole("button", { name: "Create User", exact: true }).click();
  const createUserDialog = page.getByRole("dialog", { name: "Create User" });
  await createUserDialog.getByPlaceholder("Enter username").fill(username);
  await createUserDialog.getByPlaceholder("Enter password").fill("t11-password1234");
  await createUserDialog.getByRole("button", { name: "Create user", exact: true }).click();
  expect((await createUserResponse).status()).toBe(200);
  await expect(page.getByRole("link", { name: username, exact: true })).toBeVisible();
  await capture(page, "admin-users-crud");

  await page.getByRole("tab", { name: "Roles", exact: true }).click();
  const createRoleResponsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/mlflow/roles/create"),
  );
  await page.getByRole("button", { name: "Create Role", exact: true }).click();
  const createRoleDialog = page.getByRole("dialog", { name: "Create Role" });
  await createRoleDialog.getByPlaceholder("Enter role name").fill(roleName);
  await createRoleDialog.getByPlaceholder("Enter description (optional)").fill("T11.6 UI CRUD role");
  await createRoleDialog.getByRole("button", { name: "Create role", exact: true }).click();
  const createRoleResponse = await createRoleResponsePromise;
  expect(createRoleResponse.status()).toBe(200);
  const roleId = (await createRoleResponse.json()).role.id;
  await expect(page.getByRole("link", { name: roleName, exact: true })).toBeVisible();
  await capture(page, "admin-roles-crud");

  const permissionSeed = await rustJson(page, "/ajax-api/3.0/mlflow/roles/permissions/add", "POST", {
    role_id: roleId,
    resource_type: "experiment",
    resource_pattern: "*",
    permission: "READ",
  });
  expect(permissionSeed).toMatchObject({ status: 200, backend: "rust" });
  const adminAssignment = await rustJson(page, "/ajax-api/3.0/mlflow/roles/assign", "POST", {
    username: "admin",
    role_id: roleId,
  });
  expect(adminAssignment).toMatchObject({ status: 200, backend: "rust" });

  await page.goto(`/#/admin/users/${username}?workspace=default`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: username, exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Edit access", exact: true }).click();
  const editDialog = page.getByRole("dialog", { name: `Edit access for ${username}` });
  const rolePicker = editDialog.getByRole("combobox", { name: "Roles" });
  await expect(rolePicker).toBeVisible();
  await rolePicker.click();
  await page
    .getByRole("option", { name: `default/${roleName} — T11.6 UI CRUD role`, exact: true })
    .click();
  await editDialog.getByRole("button", { name: "Review changes", exact: true }).click();
  await expect(editDialog.getByText(`default/${roleName}`, { exact: true })).toBeVisible();
  const assignResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname.endsWith("/mlflow/roles/assign") &&
      response.request().postDataJSON()?.username === username,
  );
  await editDialog.getByRole("button", { name: "Apply changes", exact: true }).click();
  expect((await assignResponse).status()).toBe(200);
  await expect(editDialog).toBeHidden();
  await expect(page.getByText(roleName, { exact: true }).first()).toBeVisible();
  await capture(page, "admin-edit-access");

  const currentUserResponsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/mlflow/users/current"),
  );
  await page.goto("/#/account?tab=permissions&workspace=default", { waitUntil: "domcontentloaded" });
  const currentUserResponse = await currentUserResponsePromise;
  expect(currentUserResponse.status()).toBe(200);
  expect(await currentUserResponse.json()).toMatchObject({
    user: { username: "admin", is_admin: true },
    is_basic_auth: true,
  });
  await expect(page.getByRole("heading", { name: "Account", exact: true })).toBeVisible();
  await expect(page.getByText("Logged in as", { exact: false })).toContainText("admin");
  await expect(page.getByRole("tab", { name: "Permissions", exact: true })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByText("experiment:all", { exact: true })).toBeVisible();
  await expect(page.getByText("READ", { exact: true })).toBeVisible();
  await expect(page.getByText(roleName, { exact: true })).toBeVisible();
  await capture(page, "account-permissions");

  await page.goto("/#/admin?workspace=default", { waitUntil: "domcontentloaded" });
  await page.getByRole("checkbox", { name: `Select user ${username}` }).check();
  await page.getByRole("button", { name: "Delete (1)", exact: true }).click();
  const deleteUserResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/mlflow/users/delete"),
  );
  await page.getByRole("dialog", { name: "Delete users" }).getByRole("button", { name: "Delete" }).click();
  expect((await deleteUserResponse).status()).toBe(200);
  await expect(page.getByRole("link", { name: username, exact: true })).toHaveCount(0);

  await page.getByRole("tab", { name: "Roles", exact: true }).click();
  await page.getByRole("checkbox", { name: `Select role ${roleName}` }).check();
  await page.getByRole("button", { name: "Delete (1)", exact: true }).click();
  const deleteRoleResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/mlflow/roles/delete"),
  );
  await page.getByRole("dialog", { name: "Delete roles" }).getByRole("button", { name: "Delete" }).click();
  expect((await deleteRoleResponse).status()).toBe(200);
  await expect(page.getByRole("link", { name: roleName, exact: true })).toHaveCount(0);

  await page.goto("/#/account?workspace=default", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("button", { name: "Change password", exact: true })).toBeVisible();
  const accountTrigger = page.getByRole("button", { name: "Account menu for admin", exact: true });
  await expect(accountTrigger).toBeVisible();
  await accountTrigger.click();
  const logout = page.getByRole("menuitem", { name: "Logout", exact: true });
  await expect(logout).toBeVisible();
  await page.evaluate(() => {
    document.cookie = "mlflow_user=t11-marker; path=/";
    document.cookie = "mlflow-request-header-Authorization=t11-marker; path=/";
  });
  await capture(page, "basic-auth-logout");
  browserAudit.allowFailure("/ajax-api/2.0/mlflow/users/current", 401);
  const logoutResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === "/ajax-api/2.0/mlflow/users/current" && response.status() === 401,
  );
  await logout.click();
  await logoutResponse;
  await expect
    .poll(async () => (await page.context().cookies()).map(({ name }) => name))
    .not.toContain("mlflow_user");
  expect((await page.context().cookies()).map(({ name }) => name)).not.toContain(
    "mlflow-request-header-Authorization",
  );
});
