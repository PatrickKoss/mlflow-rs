#!/usr/bin/env bash
# Frontend smoke test for the all-Rust cutover (T22.4).
#
# Two things this asserts that smoke.sh doesn't:
#   1. The UI (`/`, `/static-files/*`) is served by nginx directly from the
#      mounted build dir — attributed `X-MLflow-Backend: static` —
#      with the cache-header matrix from RUST_TRACKING_SERVER_PLAN.md T11.4:
#      hashed assets get a 28-day `Cache-Control`, `index.html` gets `no-cache`.
#   2. GenAI hash routes load the static SPA shell and a deterministic GenAI
#      discovery API succeeds from Rust. There is no Python service to stop.
#
# Requires: the compose stack already up (`docker compose ... up -d --wait`)
# AND a UI build populated at `mlflow/server/js/build/` — run
# `bash build_placeholder_ui.sh` first if you don't have a real `yarn build`.
#
# Usage: BASE_URL=http://localhost:80 bash smoke_frontend.sh
set -u

BASE_URL="${BASE_URL:-http://localhost:80}"
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

bold "== Static UI =="

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

bold "== GenAI UI and API after all-Rust cutover =="
# React uses hash routing, so a GenAI deep link fetches the same static shell.
fetch GET "/#/gateway"
check "GenAI deep link shell -> 200" "$STATUS" "200"
check "GenAI deep link shell -> backend=static" "$HDR_BACKEND" "static"

if [[ -n "$ASSET_PATH" ]]; then
  fetch GET "$ASSET_PATH"
  check "GenAI page asset -> 200" "$STATUS" "200"
  check "GenAI page asset -> backend=static" "$HDR_BACKEND" "static"
fi

fetch GET "/ajax-api/3.0/mlflow/gateway/supported-providers"
check "GenAI discovery API -> 200" "$STATUS" "200"
check "GenAI discovery API -> backend=rust" "$HDR_BACKEND" "rust"

echo
bold "== Summary =="
echo "  PASS=${PASS}  FAIL=${FAIL}"
if (( FAIL > 0 )); then
  echo "FRONTEND SMOKE FAILED: ${FAIL} check(s) failed." >&2
  exit 1
fi
echo "FRONTEND SMOKE OK: all ${PASS} checks passed."
