#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="${REPO_ROOT}/rust/e2e"
COMPOSE_FILE="${REPO_ROOT}/rust/deploy/docker-compose.yml"
WORKSPACES_OVERRIDE="${E2E_DIR}/docker-compose.workspaces.yml"
AUTH_OVERRIDE="${E2E_DIR}/docker-compose.auth.yml"

cleanup() {
  docker compose -f "${COMPOSE_FILE}" -f "${AUTH_OVERRIDE}" down -v --remove-orphans
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
npx --min-release-age=7 --yes --package node@24.14.0 node mlflow/server/js/yarn/releases/yarn-4.12.0.cjs --cwd mlflow/server/js install
npx --min-release-age=7 --yes --package node@24.14.0 node mlflow/server/js/yarn/releases/yarn-4.12.0.cjs --cwd mlflow/server/js build

cd "${E2E_DIR}"
npm ci --min-release-age=7
npx --min-release-age=7 playwright install chromium

rm -f .t11-results.json

for round in 1 2; do
  echo "UI smoke round ${round}/2: auth-disabled GenAI + Part 1"
  docker compose -f "${COMPOSE_FILE}" down -v
  docker compose -f "${COMPOSE_FILE}" -f "${WORKSPACES_OVERRIDE}" up -d --build
  node seed.mjs
  MLFLOW_E2E_SUITE=genai npm test
  MLFLOW_E2E_SUITE=part1 npm test
  docker compose -f "${COMPOSE_FILE}" -f "${WORKSPACES_OVERRIDE}" down -v

  echo "UI smoke round ${round}/2: auth-enabled admin + account"
  docker compose -f "${COMPOSE_FILE}" -f "${AUTH_OVERRIDE}" up -d --build
  MLFLOW_E2E_SUITE=auth npm test
  docker compose -f "${COMPOSE_FILE}" -f "${AUTH_OVERRIDE}" down -v
done
