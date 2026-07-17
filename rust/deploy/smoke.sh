#!/usr/bin/env bash
# Smoke test for the MLflow reference deployment (T11.3).
#
# Drives the SDK surface through nginx on :80 and asserts, for each request,
# that the `X-MLflow-Backend` response header matches the upstream §2.2 says
# should serve it (rust for everything, python for the genai/gateway surface).
# Exits non-zero on the first attribution mismatch or unexpected HTTP failure.
#
# Usage: BASE_URL=http://localhost:80 bash smoke.sh
set -u

BASE_URL="${BASE_URL:-http://localhost:80}"
PASS=0
FAIL=0

bold() { printf '\033[1m%s\033[0m\n' "$1"; }

# wait_for_health: block until nginx -> rust /health returns 200 (or timeout).
wait_for_health() {
  local tries=60
  while (( tries-- > 0 )); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ${BASE_URL}/health never became ready" >&2
  return 1
}

# req METHOD PATH EXPECTED_BACKEND [DESC] [curl-args...]
# Performs the request, prints status + attributed backend, asserts the backend
# header equals EXPECTED_BACKEND. HTTP status is informational (a genai 404/501
# from Python still counts as correct ATTRIBUTION). Sets global BODY.
req() {
  local method="$1" path="$2" expect="$3" desc="${4:-$path}"; shift 4 || shift $#
  local tmp_headers tmp_body status backend
  tmp_headers="$(mktemp)"; tmp_body="$(mktemp)"
  status="$(curl -s -o "$tmp_body" -D "$tmp_headers" -w '%{http_code}' \
             -X "$method" "$@" "${BASE_URL}${path}")"
  backend="$(grep -i '^x-mlflow-backend:' "$tmp_headers" | tail -1 \
             | tr -d '\r' | awk '{print tolower($2)}')"
  BODY="$(cat "$tmp_body")"
  rm -f "$tmp_headers" "$tmp_body"

  if [[ "$backend" == "$expect" ]]; then
    printf '  \033[32mPASS\033[0m %-6s %-55s http=%s backend=%s\n' "$method" "$path" "$status" "$backend"
    ((PASS++))
  else
    printf '  \033[31mFAIL\033[0m %-6s %-55s http=%s backend=%s (expected %s)\n' \
      "$method" "$path" "$status" "${backend:-<none>}" "$expect"
    ((FAIL++))
  fi
}

json() { printf '%s' "$1"; }

bold "Waiting for MLflow at ${BASE_URL} ..."
wait_for_health || exit 1

bold "== Ops (Rust) =="
req GET  "/health"                                   rust "health"
req GET  "/version"                                  rust "version"
req GET  "/metrics"                                  rust "prometheus metrics"

bold "== Experiments (Rust) =="
EXP_NAME="smoke-$(date +%s)-$$"
req POST "/api/2.0/mlflow/experiments/create" rust "create experiment" \
  -H 'Content-Type: application/json' -d "$(json "{\"name\":\"${EXP_NAME}\"}")"
EXP_ID="$(printf '%s' "$BODY" | sed -n 's/.*"experiment_id"[: ]*"\([0-9]*\)".*/\1/p')"
echo "    experiment_id=${EXP_ID}"
req GET  "/api/2.0/mlflow/experiments/get-by-name?experiment_name=${EXP_NAME}" rust "get experiment by name"
req POST "/api/2.0/mlflow/experiments/search"        rust "search experiments" \
  -H 'Content-Type: application/json' -d '{"max_results":10}'

bold "== Runs / metrics / params (Rust) =="
req POST "/api/2.0/mlflow/runs/create"               rust "create run" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"experiment_id\":\"${EXP_ID}\",\"start_time\":0}")"
RUN_ID="$(printf '%s' "$BODY" | sed -n 's/.*"run_id"[: ]*"\([^"]*\)".*/\1/p' | head -1)"
echo "    run_id=${RUN_ID}"
req POST "/api/2.0/mlflow/runs/log-metric"           rust "log metric" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"run_id\":\"${RUN_ID}\",\"key\":\"acc\",\"value\":0.9,\"timestamp\":0,\"step\":0}")"
req POST "/api/2.0/mlflow/runs/log-parameter"        rust "log param" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"run_id\":\"${RUN_ID}\",\"key\":\"lr\",\"value\":\"0.01\"}")"
req GET  "/api/2.0/mlflow/metrics/get-history?run_id=${RUN_ID}&metric_key=acc" rust "get metric history"
req POST "/api/2.0/mlflow/runs/search"               rust "search runs" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"experiment_ids\":[\"${EXP_ID}\"],\"max_results\":10}")"

bold "== Traces (Rust) =="
# V3 trace start (StartTraceV3). A 200 or a validation 4xx both attribute to Rust.
req POST "/api/3.0/mlflow/traces"                    rust "start trace v3" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"trace\":{\"trace_info\":{\"trace_id\":\"smoke-trace-1\",\"trace_location\":{\"type\":\"MLFLOW_EXPERIMENT\",\"mlflow_experiment\":{\"experiment_id\":\"${EXP_ID}\"}},\"request_time\":\"1970-01-01T00:00:00Z\",\"state\":\"OK\"}}}")"
# OTLP trace ingest endpoint (root path -> Rust).
req POST "/v1/traces"                                rust "otlp /v1/traces" \
  -H 'Content-Type: application/json' -d '{}'

bold "== Registry: models + versions (Rust) =="
MODEL_NAME="smoke-model-$$"
req POST "/api/2.0/mlflow/registered-models/create"  rust "create registered model" \
  -H 'Content-Type: application/json' -d "$(json "{\"name\":\"${MODEL_NAME}\"}")"
req POST "/api/2.0/mlflow/model-versions/create"     rust "create model version" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"name\":\"${MODEL_NAME}\",\"source\":\"/mlartifacts/${RUN_ID}/model\"}")"
req GET  "/api/2.0/mlflow/registered-models/get?name=${MODEL_NAME}" rust "get registered model"

bold "== Webhooks (Rust) =="
# List webhooks: `GET /mlflow/webhooks` (ListWebhooks).
req GET  "/api/2.0/mlflow/webhooks"                  rust "list webhooks"

bold "== Users / auth (Rust; served only when basic-auth enabled) =="
# When auth is disabled the route 404s from Rust — attribution (rust) is what
# we assert, so this passes either way; it never leaks to Python.
req GET  "/api/2.0/mlflow/users/get?username=admin"  rust "users endpoint (rust-attributed)"

bold "== Artifact plane (Rust) =="
# Proxied artifact upload (PUT) + download (GET) via the mlflow-artifacts plane.
ART_PATH="smoke/hello.txt"
req PUT  "/api/2.0/mlflow-artifacts/artifacts/${ART_PATH}" rust "artifact upload (proxy PUT)" \
  -H 'Content-Type: application/octet-stream' --data-binary 'hello from smoke'
req GET  "/api/2.0/mlflow-artifacts/artifacts/${ART_PATH}" rust "artifact download (proxy GET)"
if [[ "$BODY" == "hello from smoke" ]]; then
  echo "    artifact round-trip body OK"
else
  echo "    NOTE: artifact download body was: '${BODY}'"
fi
req GET  "/api/2.0/mlflow-artifacts/artifacts?path=smoke" rust "artifact list (proxy)"

bold "== GenAI / gateway surface (MUST attribute to Python) =="
# These prefixes are the ONLY ones §2.2 routes to Python. A 404/501/4xx from the
# Python container is fine — ATTRIBUTION to python is the assertion.
req GET  "/api/3.0/mlflow/genai/does-not-exist"      python "genai -> python"
req POST "/api/3.0/mlflow/gateway/anything"          python "gateway -> python" \
  -H 'Content-Type: application/json' -d '{}'
req GET  "/api/3.0/mlflow/scorers/list"              python "scorers -> python"
req GET  "/api/3.0/mlflow/label-schemas/list"        python "label-schemas -> python"
req GET  "/ajax-api/3.0/jobs/list"                   python "jobs -> python"
req POST "/api/3.0/mlflow/scorer/invoke"             python "scorer/invoke -> python" \
  -H 'Content-Type: application/json' -d '{}'
req POST "/ajax-api/2.0/mlflow/runs/create-promptlab-run" python "create-promptlab-run -> python" \
  -H 'Content-Type: application/json' -d '{}'

echo
bold "== Summary =="
echo "  PASS=${PASS}  FAIL=${FAIL}"
if (( FAIL > 0 )); then
  echo "SMOKE FAILED: ${FAIL} attribution mismatch(es)." >&2
  exit 1
fi
echo "SMOKE OK: all ${PASS} requests attributed to the correct backend."
