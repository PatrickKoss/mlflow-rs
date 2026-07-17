#!/usr/bin/env bash
# Frontend-split smoke test (T11.4).
#
# Two things this asserts that smoke.sh doesn't:
#   1. The UI (`/`, `/static-files/*`) is served by nginx directly from the
#      mounted build dir — attributed `X-MLflow-Backend: static`, not `python` —
#      with the cache-header matrix from RUST_TRACKING_SERVER_PLAN.md T11.4:
#      hashed assets get a 28-day `Cache-Control`, `index.html` gets `no-cache`.
#   2. AC: "UI fully loads with the Python container stopped, except genai
#      pages." — stops the `python` compose service, re-checks `/` and the
#      hashed asset both still 200 (nginx serves them with Python down), then
#      confirms a genai request now fails (502/504, Python unreachable) while
#      a plain Rust-backed API request still works. Restarts `python` at the end
#      (best-effort, even on failure) so a later `smoke.sh` run isn't left broken.
#
# Requires: the compose stack already up (`docker compose ... up -d --wait`)
# AND a UI build populated at `mlflow/server/js/build/` — run
# `bash build_placeholder_ui.sh` first if you don't have a real `yarn build`.
#
# Usage: BASE_URL=http://localhost:80 bash smoke_frontend.sh
set -u

BASE_URL="${BASE_URL:-http://localhost:80}"
COMPOSE_FILE="${COMPOSE_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml}"
PASS=0
FAIL=0

bold() { printf '\033[1m%s\033[0m\n' "$1"; }

# check DESC ACTUAL EXPECTED
check() {
  local desc="$1" actual="$2" expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf '  \033[32mPASS\033[0m %-45s got=%s\n' "$desc" "$actual"
    ((PASS++))
  else
    printf '  \033[31mFAIL\033[0m %-45s got=%s expected=%s\n' "$desc" "$actual" "$expected"
    ((FAIL++))
  fi
}

# check_status_in DESC ACTUAL EXPECTED_SET...
check_status_in() {
  local desc="$1" actual="$2"; shift 2
  local expect
  for expect in "$@"; do
    if [[ "$actual" == "$expect" ]]; then
      printf '  \033[32mPASS\033[0m %-45s got=%s (in {%s})\n' "$desc" "$actual" "$*"
      ((PASS++))
      return
    fi
  done
  printf '  \033[31mFAIL\033[0m %-45s got=%s expected one of {%s}\n' "$desc" "$actual" "$*"
  ((FAIL++))
}

# fetch METHOD PATH -> sets STATUS, BODY, and HDR_<name> (lowercased, '-'->'_')
# for every response header, via globals.
fetch() {
  local method="$1" path="$2"
  local tmp_headers tmp_body
  tmp_headers="$(mktemp)"; tmp_body="$(mktemp)"
  STATUS="$(curl -s -o "$tmp_body" -D "$tmp_headers" -w '%{http_code}' -X "$method" "${BASE_URL}${path}")"
  BODY="$(cat "$tmp_body")"
  HDR_CACHE_CONTROL="$(grep -i '^cache-control:' "$tmp_headers" | tail -1 | cut -d: -f2- | tr -d '\r' | sed 's/^ *//')"
  HDR_BACKEND="$(grep -i '^x-mlflow-backend:' "$tmp_headers" | tail -1 | cut -d: -f2- | tr -d '\r' | sed 's/^ *//' | tr '[:upper:]' '[:lower:]')"
  rm -f "$tmp_headers" "$tmp_body"
}

restart_python() {
  bold "Restarting python container..."
  docker compose -f "$COMPOSE_FILE" start python >/dev/null
  local tries=60
  while (( tries-- > 0 )); do
    fetch GET "/python/health"
    [[ "$STATUS" == "200" ]] && { echo "  python back up."; return 0; }
    sleep 2
  done
  echo "  WARNING: python did not come back healthy in time" >&2
}

bold "== Static UI, Python up (baseline) =="

fetch GET "/"
check "GET / -> 200" "$STATUS" "200"
check "GET / -> backend=static" "$HDR_BACKEND" "static"
check "GET / -> Cache-Control: no-cache" "$HDR_CACHE_CONTROL" "no-cache"
[[ "$BODY" == *"<div id=\"root\""* ]] && { echo "    body looks like index.html"; ((PASS++)); } || { echo "    NOTE: body didn't look like index.html: ${BODY:0:120}"; ((FAIL++)); }

ASSET_PATH="$(printf '%s' "$BODY" | grep -o '/static-files/static/js/main\.[A-Za-z0-9]*\.js' | head -1)"
if [[ -z "$ASSET_PATH" ]]; then
  echo "  WARNING: could not discover a hashed asset path from index.html; skipping asset checks." >&2
else
  fetch GET "$ASSET_PATH"
  check "GET \$asset -> 200" "$STATUS" "200"
  check "GET \$asset -> backend=static" "$HDR_BACKEND" "static"
  check "GET \$asset -> Cache-Control 28d" "$HDR_CACHE_CONTROL" "public, max-age=2419200"
fi

fetch GET "/api/2.0/mlflow/experiments/search"
check_status_in "GET tracking API still 200/4xx (rust up)" "$STATUS" "200" "400" "422"
check "GET tracking API -> backend=rust" "$HDR_BACKEND" "rust"

bold "== Stopping python (AC: UI loads except genai) =="
docker compose -f "$COMPOSE_FILE" stop python >/dev/null
trap restart_python EXIT

fetch GET "/"
check "python down: GET / -> 200" "$STATUS" "200"
check "python down: GET / -> backend=static" "$HDR_BACKEND" "static"

if [[ -n "$ASSET_PATH" ]]; then
  fetch GET "$ASSET_PATH"
  check "python down: GET \$asset -> 200" "$STATUS" "200"
  check "python down: GET \$asset -> backend=static" "$HDR_BACKEND" "static"
fi

fetch GET "/api/2.0/mlflow/experiments/search"
check_status_in "python down: tracking API still works (rust)" "$STATUS" "200" "400" "422"
check "python down: tracking API -> backend=rust" "$HDR_BACKEND" "rust"

# Expected failure: genai lives only on Python. With Python stopped, nginx's
# upstream connect fails -> 502 (connection refused) or 504 (timeout).
fetch GET "/api/3.0/mlflow/genai/does-not-exist"
check_status_in "python down: genai request now fails (expected)" "$STATUS" "502" "503" "504"

echo
bold "== Summary =="
echo "  PASS=${PASS}  FAIL=${FAIL}"
if (( FAIL > 0 )); then
  echo "FRONTEND SMOKE FAILED: ${FAIL} check(s) failed." >&2
  exit 1
fi
echo "FRONTEND SMOKE OK: all ${PASS} checks passed."
