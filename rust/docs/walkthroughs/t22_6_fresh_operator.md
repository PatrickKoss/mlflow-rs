# T22.6 fresh-operator walkthrough

Date: 2026-07-20. Result: **PASS**. This walkthrough used Docker 29.1.3,
Docker Compose 5.0.0, the locally built `deploy-rust:latest` image
(`af9216a46086`), Postgres 16, the reference nginx, and only obvious fake
credentials. No live provider was contacted.

## 1. Prepare explicit runtime configuration — PASS

I created a temporary compose override with this environment:

```yaml
services:
  rust:
    environment:
      MLFLOW_CRYPTO_KEK_PASSPHRASE: t22-6-obvious-fake-passphrase-7ec6d9f61566459ebf7f01904a5d25da
      MLFLOW_CRYPTO_KEK_VERSION: "1"
      MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS: "1"
      MLFLOW_GENAI_EVAL_MAX_WORKERS: "1"
      MLFLOW_TRACE_ARCHIVAL_CONFIG: /etc/mlflow/trace-archival.yaml
    volumes:
      - ./.t22_6-trace-archival-valid.yaml:/etc/mlflow/trace-archival.yaml:ro
```

The mounted valid file was:

```yaml
trace_archival:
  enabled: true
  location: file:///mlartifacts/trace-archive
  retention: 30d
  long_retention_allowlist: []
  interval_seconds: 60
  max_traces_per_pass: 1000
```

I then ran:

```bash
bash rust/deploy/build_placeholder_ui.sh
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml down -v
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml up -d --wait --no-build
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml ps
```

Observed: Postgres was healthy, `migrate` exited 0, Rust was healthy, and nginx
was healthy on port 80. Rust logged:

```text
mlflow-server listening address=0.0.0.0:5000 static_prefix=None
trace archival scheduler pass completed archived_total=0 scope_count=1
```

The configuration checks were:

```bash
curl -fsS http://localhost/ajax-api/3.0/mlflow/gateway/secrets/config
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml exec -T rust sh -c \
  'printf "%s\n" "$MLFLOW_CRYPTO_KEK_VERSION" \
    "$MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS" \
    "$MLFLOW_GENAI_EVAL_MAX_WORKERS" "$MLFLOW_TRACE_ARCHIVAL_CONFIG"'
```

Observed:

```text
{"secrets_available":true,"using_default_passphrase":false}
1
1
1
/etc/mlflow/trace-archival.yaml
```

## 2. Create secrets and prove decryption — PASS

I started the repository's deterministic provider in a disposable container on
the compose network:

```bash
docker run -d --rm --name t22-6-mock-provider --network deploy_default \
  -v "$PWD:/repo:ro" \
  ghcr.io/mlflow/mlflow@sha256:73365f742d67ef9e59e50118010bf14ff825a157c8244051baea887b8587e772 \
  python -c 'import sys; sys.path.insert(0,"/repo"); from rust.bench.genai.mock_provider import DeterministicProvider; DeterministicProvider(("0.0.0.0",8080),2206).serve_forever()'
docker exec deploy-rust-1 curl -fsS \
  http://t22-6-mock-provider:8080/v1/models
```

Observed: `{"data":[{"id":"genai-bench-model","object":"model"}]}`.

I created two secrets through the gateway create route. Each used provider
`openai`, an `auth_config.api_base` of
`http://t22-6-mock-provider:8080/v1`, and an obvious fake `api_key`:

```bash
BASE_URL=http://localhost
create_secret() {
  curl -fsS -H 'Content-Type: application/json' \
    -d "{\"secret_name\":\"$1\",\"secret_value\":{\"api_key\":\"$2\"},\"provider\":\"openai\",\"auth_config\":{\"api_base\":\"http://t22-6-mock-provider:8080/v1\"},\"created_by\":\"t22.6-walkthrough\"}" \
    "$BASE_URL/api/3.0/mlflow/gateway/secrets/create"
}
OLD_SECRET_JSON=$(create_secret t22-6-old obvious-fake-key-old-v1)
ROTATE_SECRET_JSON=$(create_secret t22-6-rotate obvious-fake-key-rotate-v1)
OLD_SECRET_ID=$(printf '%s' "$OLD_SECRET_JSON" | jq -r '.secret.secret_id')
ROTATE_SECRET_ID=$(printf '%s' "$ROTATE_SECRET_JSON" | jq -r '.secret.secret_id')
```

The actual returned IDs were:

```text
t22-6-old    -> s-1cc38b159d5545f4998ced402aa60493
t22-6-rotate -> s-6fbfddae776547edace6b2bc1b67d803
```

For each secret I created a model definition and endpoint:

```bash
create_model() {
  curl -fsS -H 'Content-Type: application/json' \
    -d "{\"name\":\"$1-model\",\"secret_id\":\"$2\",\"provider\":\"openai\",\"model_name\":\"genai-bench-model\",\"created_by\":\"t22.6-walkthrough\"}" \
    "$BASE_URL/api/3.0/mlflow/gateway/model-definitions/create"
}
OLD_MODEL_JSON=$(create_model t22-6-old "$OLD_SECRET_ID")
ROTATE_MODEL_JSON=$(create_model t22-6-rotate "$ROTATE_SECRET_ID")
OLD_MODEL_ID=$(printf '%s' "$OLD_MODEL_JSON" | jq -r \
  '.model_definition.model_definition_id')
ROTATE_MODEL_ID=$(printf '%s' "$ROTATE_MODEL_JSON" | jq -r \
  '.model_definition.model_definition_id')

create_endpoint() {
  curl -fsS -H 'Content-Type: application/json' \
    -d "{\"name\":\"$1-endpoint\",\"routing_strategy\":\"REQUEST_BASED_TRAFFIC_SPLIT\",\"model_configs\":[{\"model_definition_id\":\"$2\",\"weight\":1.0,\"linkage_type\":\"PRIMARY\"}],\"usage_tracking\":false,\"created_by\":\"t22.6-walkthrough\"}" \
    "$BASE_URL/api/3.0/mlflow/gateway/endpoints/create"
}
OLD_ENDPOINT_JSON=$(create_endpoint t22-6-old "$OLD_MODEL_ID")
ROTATE_ENDPOINT_JSON=$(create_endpoint t22-6-rotate "$ROTATE_MODEL_ID")
```

The returned model IDs were `d-f5612af32c344ce98ac0b364d93230be` and
`d-527582640c024189a4af401eb83051dc`; the endpoints were
`t22-6-old-endpoint` and `t22-6-rotate-endpoint`. I invoked both with:

```bash
curl -fsS -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"KEK decrypt probe"}],"stream":false}' \
  http://localhost/gateway/t22-6-old-endpoint/mlflow/invocations
curl -fsS -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"KEK decrypt probe"}],"stream":false}' \
  http://localhost/gateway/t22-6-rotate-endpoint/mlflow/invocations
```

Observed: both returned model `genai-bench-model`, `finish_reason=stop`, and
deterministic response ID `chatcmpl-9559801cb700281139bb`. The request cannot
reach the fake provider until Rust decrypts the stored API-key map, so these
successful invocations proved read-time decryption. The database showed:

```bash
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml exec -T postgres \
  psql -U mlflow -d mlflow -Atc \
  "SELECT secret_name || ':' || kek_version FROM secrets WHERE secret_name LIKE 't22-6-%' ORDER BY secret_name;"
```

```text
t22-6-old:1
t22-6-rotate:1
```

## 3. Rotate the KEK write version — PASS

I changed only `MLFLOW_CRYPTO_KEK_VERSION` from `1` to `2`, kept the passphrase
unchanged, and recreated Rust and nginx:

```bash
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml up -d --no-deps \
  --force-recreate --wait rust
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml up -d --no-deps \
  --force-recreate --wait nginx
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml exec -T rust sh -c \
  'echo MLFLOW_CRYPTO_KEK_VERSION=$MLFLOW_CRYPTO_KEK_VERSION'
```

Observed: `MLFLOW_CRYPTO_KEK_VERSION=2`, `/health` returned `OK`, and the
archival scheduler completed its startup pass.

I re-entered only the second secret:

```bash
curl -fsS -H 'Content-Type: application/json' \
  -d '{"secret_id":"s-6fbfddae776547edace6b2bc1b67d803","secret_value":{"api_key":"obvious-fake-key-rotate-v2"},"updated_by":"t22.6-walkthrough"}' \
  http://localhost/api/3.0/mlflow/gateway/secrets/update
```

Observed: its masked value changed to `obv...e-v2`, and the same SQL audit now
reported mixed stored versions:

```text
t22-6-old:1
t22-6-rotate:2
```

After the restart and update had cleared any process-local secret cache, I
invoked both endpoints again. Both returned HTTP 200, model
`genai-bench-model`, `finish_reason=stop`, and response ID
`chatcmpl-b624af9004081004ec9b`. This proved an untouched version-1 row and a
rewritten version-2 row decrypt concurrently from their stored versions.

## 4. Verify native worker concurrency handling — PASS

The serving stack was running with both
`MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS=1` and
`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`. I ran the reference smoke:

```bash
BASE_URL=http://localhost:80 bash rust/deploy/smoke.sh
```

Observed: `PASS=35 FAIL=0`; native job
`0d6b05b8-c0b2-4b0c-9be2-1c4c733ad96e` reached `SUCCEEDED`, and zero responses
had a Python backend header.

To prove the server consumes and validates the server-side cap rather than
merely inheriting an unused environment variable, I ran a disposable server
with the same setting changed to zero:

```bash
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml run --rm --no-deps \
  -e MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS=0 rust
```

Observed: exit 1 with:

```text
Error: InvalidParameterValue: MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS must be greater than zero.
```

## 5. Validate archival config at startup — PASS with one finding

The valid config had already passed startup validation and emitted the
completed scheduler-pass log in steps 1 and 3. I created a second disposable
file whose only invalid field was `retention: someday`, then ran:

```bash
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml run --rm --no-deps \
  -e MLFLOW_TRACE_ARCHIVAL_CONFIG=/etc/mlflow/trace-archival-invalid.yaml \
  -v "$PWD/rust/deploy/.t22_6-trace-archival-invalid.yaml:/etc/mlflow/trace-archival-invalid.yaml:ro" \
  rust
```

Observed: exit 1 before serving, with:

```text
Error: Invalid value for 'trace_archival.retention'. Expected a duration in the form `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h').
```

Finding: despite the enabled scheduler and its completed-pass log, this command
returned a false signal:

```bash
curl -fsS http://localhost/api/3.0/mlflow/server-info
```

```json
{"store_type":"SqlStore","workspaces_enabled":false,"trace_archival_enabled":false}
```

The handler currently hard-codes the archival field. No Rust source was changed
for this docs task; [ARCHIVAL_RUNBOOK.md](../ARCHIVAL_RUNBOOK.md) documents the
reliable monitoring signals.

The optional Redis exercise was not run. Source verification confirmed that a
nonempty `MLFLOW_GATEWAY_BUDGET_REDIS_URL` selects Redis, but there is currently
no startup backend-selection log to use as the suggested proof.

## 6. Tear down — PASS

I removed the fake provider, application containers, network, named volume, and
temporary config/override files:

```bash
docker stop t22-6-mock-provider
docker compose -f rust/deploy/docker-compose.yml \
  -f rust/deploy/.t22_6-compose.override.yml down -v --remove-orphans
docker ps --format '{{.Names}}' | grep -E '^(deploy-|t22-6-)' || true
docker volume ls --format '{{.Name}}' | grep '^deploy_' || true
```

Observed: compose removed nginx, Rust, migration, Postgres, `deploy_default`,
and `deploy_artifacts`. Both final inventory commands returned no matches. I
then deleted the three temporary `.t22_6-*` files from the worktree; `git
status --short` showed only the requested documentation deliverables. Images
were retained as required.
