#!/usr/bin/env bash
# Smoke test for the all-Rust MLflow reference deployment (T22.4).
#
# Drives the SDK surface through nginx on :80 and asserts, for each request,
# that the `X-MLflow-Backend` response header is `rust`. It also submits and
# completes a deterministic native scorer job through the co-installed worker,
# then globally rejects any recorded `X-MLflow-Backend: python` response.
#
# Usage: BASE_URL=http://localhost:80 bash smoke.sh
set -u

BASE_URL="${BASE_URL:-http://localhost:80}"
PASS=0
FAIL=0
TMP_DIR="$(mktemp -d)"
ALL_HEADERS="${TMP_DIR}/all-headers"
: >"$ALL_HEADERS"
trap 'rm -rf "$TMP_DIR"' EXIT

bold() { printf '\033[1m%s\033[0m\n' "$1"; }

# wait_for_health: block until nginx -> rust /health returns 200 (or timeout).
wait_for_health() {
  local tries=60 tmp_headers
  while (( tries-- > 0 )); do
    tmp_headers="$(mktemp "${TMP_DIR}/health.XXXXXX")"
    if curl -fsS -D "$tmp_headers" "${BASE_URL}/health" >/dev/null 2>&1; then
      cat "$tmp_headers" >>"$ALL_HEADERS"
      rm -f "$tmp_headers"
      return 0
    fi
    cat "$tmp_headers" >>"$ALL_HEADERS"
    rm -f "$tmp_headers"
    sleep 2
  done
  echo "ERROR: ${BASE_URL}/health never became ready" >&2
  return 1
}

# req METHOD PATH EXPECTED_BACKEND [DESC] [curl-args...]
# Performs the request, prints status + attributed backend, asserts the backend
# header equals EXPECTED_BACKEND. HTTP status is informational unless the caller
# explicitly checks global STATUS. Sets global BODY and STATUS.
req() {
  local method="$1" path="$2" expect="$3" desc="${4:-$path}"; shift 4 || shift $#
  local tmp_headers tmp_body backend
  tmp_headers="$(mktemp "${TMP_DIR}/headers.XXXXXX")"
  tmp_body="$(mktemp "${TMP_DIR}/body.XXXXXX")"
  STATUS="$(curl -s -o "$tmp_body" -D "$tmp_headers" -w '%{http_code}' \
             -X "$method" "$@" "${BASE_URL}${path}")"
  backend="$(grep -i '^x-mlflow-backend:' "$tmp_headers" | tail -1 \
             | tr -d '\r' | awk '{print tolower($2)}')"
  BODY="$(cat "$tmp_body")"
  cat "$tmp_headers" >>"$ALL_HEADERS"
  rm -f "$tmp_headers" "$tmp_body"

  if [[ "$backend" == "$expect" ]]; then
    printf '  \033[32mPASS\033[0m %-6s %-55s http=%s backend=%s (%s)\n' "$method" "$path" "$STATUS" "$backend" "$desc"
    ((PASS++))
  else
    printf '  \033[31mFAIL\033[0m %-6s %-55s http=%s backend=%s (expected %s)\n' \
      "$method" "$path" "$STATUS" "${backend:-<none>}" "$expect"
    ((FAIL++))
  fi
}

check_equal() {
  local desc="$1" actual="$2" expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf '  \033[32mPASS\033[0m %-55s got=%s\n' "$desc" "$actual"
    ((PASS++))
  else
    printf '  \033[31mFAIL\033[0m %-55s got=%s expected=%s\n' \
      "$desc" "${actual:-<empty>}" "$expected"
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
# This concrete trace is reused by the deterministic native scorer exercise.
TRACE_ID="smoke-trace-$(date +%s)-$$"
req POST "/api/3.0/mlflow/traces"                    rust "start trace v3" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"trace\":{\"trace_info\":{\"trace_id\":\"${TRACE_ID}\",\"trace_location\":{\"type\":\"MLFLOW_EXPERIMENT\",\"mlflow_experiment\":{\"experiment_id\":\"${EXP_ID}\"}},\"request_time\":\"1970-01-01T00:00:00Z\",\"state\":\"OK\"}}}")"
check_equal "start trace v3 status" "$STATUS" "200"
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

bold "== GenAI / gateway surface (Rust cutover) =="
# Some probes intentionally return a Rust 4xx/404; attribution proves every
# formerly split route family now reaches Rust.
req GET  "/api/3.0/mlflow/genai/does-not-exist"      rust "genai -> rust"
req POST "/api/3.0/mlflow/gateway/anything"          rust "gateway -> rust" \
  -H 'Content-Type: application/json' -d '{}'
req GET  "/api/3.0/mlflow/scorers/list"              rust "scorers -> rust"
req GET  "/api/3.0/mlflow/label-schemas/list"        rust "label-schemas -> rust"
req GET  "/ajax-api/3.0/jobs/list"                   rust "jobs -> rust"
req POST "/api/3.0/mlflow/scorer/invoke"             rust "scorer/invoke -> rust" \
  -H 'Content-Type: application/json' -d '{}'
req POST "/ajax-api/2.0/mlflow/runs/create-promptlab-run" rust "create-promptlab-run -> rust" \
  -H 'Content-Type: application/json' -d '{}'

bold "== Native GenAI job (Rust server -> mlflow-genai-worker) =="
SCORER_JSON='{"name":"smoke-response-length","builtin_scorer_class":"ResponseLength","builtin_scorer_pydantic_data":{"max_length":100,"unit":"chars"}}'
SCORER_ESCAPED="${SCORER_JSON//\\/\\\\}"
SCORER_ESCAPED="${SCORER_ESCAPED//\"/\\\"}"
req POST "/ajax-api/3.0/mlflow/scorer/invoke" rust "submit native scorer job" \
  -H 'Content-Type: application/json' \
  -d "$(json "{\"experiment_id\":\"${EXP_ID}\",\"serialized_scorer\":\"${SCORER_ESCAPED}\",\"trace_ids\":[\"${TRACE_ID}\"],\"log_assessments\":false}")"
check_equal "native scorer submission status" "$STATUS" "200"
JOB_ID="$(printf '%s' "$BODY" | sed -n 's/.*"job_id"[: ]*"\([^"]*\)".*/\1/p' | head -1)"
if [[ -n "$JOB_ID" ]]; then
  printf '    job_id=%s\n' "$JOB_ID"
else
  echo "  FAIL native scorer response omitted job_id" >&2
  ((FAIL++))
fi

JOB_STATUS=""
if [[ -n "$JOB_ID" ]]; then
  for _ in $(seq 1 30); do
    req GET "/ajax-api/3.0/jobs/${JOB_ID}" rust "poll native scorer job"
    JOB_STATUS="$(printf '%s' "$BODY" | sed -n 's/.*"status"[: ]*"\([A-Z]*\)".*/\1/p' | head -1)"
    case "$JOB_STATUS" in
      SUCCEEDED | FAILED | TIMEOUT | CANCELED) break ;;
    esac
    sleep 1
  done
fi
check_equal "native scorer job completed via worker" "$JOB_STATUS" "SUCCEEDED"

bold "== Global Python attribution audit =="
if grep -qi '^X-MLflow-Backend:[[:space:]]*python[[:space:]]*$' "$ALL_HEADERS"; then
  echo "  FAIL at least one smoke response carried X-MLflow-Backend: python" >&2
  ((FAIL++))
else
  echo "  PASS zero smoke responses carried X-MLflow-Backend: python"
  ((PASS++))
fi

echo
bold "== Summary =="
echo "  PASS=${PASS}  FAIL=${FAIL}"
if (( FAIL > 0 )); then
  echo "SMOKE FAILED: ${FAIL} check(s) failed." >&2
  exit 1
fi
echo "SMOKE OK: all ${PASS} checks passed; native worker succeeded; zero Python headers."
