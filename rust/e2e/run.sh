#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="${REPO_ROOT}/rust/e2e"
COMPOSE_FILE="${REPO_ROOT}/rust/deploy/docker-compose.yml"

cleanup() {
  docker compose -f "${COMPOSE_FILE}" down -v
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
npx --min-release-age=7 --yes --package node@24.14.0 node mlflow/server/js/yarn/releases/yarn-4.12.0.cjs --cwd mlflow/server/js install
npx --min-release-age=7 --yes --package node@24.14.0 node mlflow/server/js/yarn/releases/yarn-4.12.0.cjs --cwd mlflow/server/js build

cd "${E2E_DIR}"
npm ci --min-release-age=7
npx --min-release-age=7 playwright install chromium

docker compose -f "${COMPOSE_FILE}" down -v
docker compose -f "${COMPOSE_FILE}" up -d --build
node seed.mjs
npm test
npm test
