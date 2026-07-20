#!/usr/bin/env bash
# Audit the production Rust image and prove that its server resolves the
# co-installed native GenAI worker at startup.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 IMAGE_TAG" >&2
  exit 2
fi

IMAGE="$1"
MIGRATE_IMAGE="${MIGRATE_IMAGE:-ghcr.io/mlflow/mlflow@sha256:73365f742d67ef9e59e50118010bf14ff825a157c8244051baea887b8587e772}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_ID="mlflow-rust-audit-$$"
NETWORK="${AUDIT_ID}-net"
POSTGRES="${AUDIT_ID}-postgres"
MIGRATE="${AUDIT_ID}-migrate"
SERVER="${AUDIT_ID}-server"

cleanup() {
  docker rm -f "$SERVER" "$MIGRATE" "$POSTGRES" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker image inspect "$IMAGE" >/dev/null

echo "== Image content: $IMAGE =="
docker run --rm --entrypoint /bin/sh "$IMAGE" -ceu '
  fail=0

  path_hits=""
  old_ifs=$IFS
  IFS=:
  for dir in $PATH; do
    IFS=$old_ifs
    [ -d "$dir" ] || continue
    hits=$(find -L "$dir" -maxdepth 1 -type f -executable -name "python*" -print)
    if [ -n "$hits" ]; then
      path_hits="${path_hits}${hits}
"
    fi
    IFS=:
  done
  IFS=$old_ifs
  if [ -n "$path_hits" ]; then
    echo "FAIL: Python executable(s) found on PATH:" >&2
    printf "%s" "$path_hits" >&2
    fail=1
  fi

  lib_hits=$(find / -xdev \( -type f -o -type l \) -name "libpython*.so*" -print)
  if [ -n "$lib_hits" ]; then
    echo "FAIL: libpython shared object(s) found:" >&2
    printf "%s\n" "$lib_hits" >&2
    fail=1
  fi

  site_hits=$(find / -xdev -type d -name site-packages -print)
  if [ -n "$site_hits" ]; then
    echo "FAIL: site-packages directories found:" >&2
    printf "%s\n" "$site_hits" >&2
    fail=1
  fi

  py_hits=$(find / -xdev -type f -name "*.py" -print)
  if [ -n "$py_hits" ]; then
    echo "FAIL: Python source payload(s) found:" >&2
    printf "%s\n" "$py_hits" >&2
    fail=1
  fi

  test "$fail" -eq 0
  test -x /usr/local/bin/mlflow-server
  test -x /usr/local/bin/mlflow-genai-worker
  echo "PASS: no python* executable, libpython, site-packages, or .py payload"
  echo "PASS: mlflow-server and mlflow-genai-worker are executable siblings"
'

echo "== Runtime launch with native jobs enabled =="
docker network create "$NETWORK" >/dev/null
docker run -d --name "$POSTGRES" --network "$NETWORK" \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=mlflow \
  -e POSTGRES_DB=mlflow \
  postgres:16 >/dev/null

for _ in $(seq 1 60); do
  if docker exec "$POSTGRES" pg_isready -U mlflow -d mlflow >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec "$POSTGRES" pg_isready -U mlflow -d mlflow >/dev/null

# Alembic remains the deliberate Python-owned, one-shot migration exception.
# This container exits before the audited Rust serving container starts.
docker run -d --name "$MIGRATE" --network "$NETWORK" \
  -v "$REPO_ROOT/mlflow/store/db_migrations/versions:/cutover-migrations:ro" \
  --entrypoint /bin/sh "$MIGRATE_IMAGE" -c \
  'migration_dir="$(python -c '\''import mlflow.store.db_migrations, os; print(os.path.join(os.path.dirname(mlflow.store.db_migrations.__file__), "versions"))'\'')";
   cp /cutover-migrations/*.py "$migration_dir/";
   exec mlflow db upgrade "$1"' _ \
  "postgresql://mlflow:mlflow@$POSTGRES:5432/mlflow" >/dev/null
if [[ "$(docker wait "$MIGRATE")" != "0" ]]; then
  docker logs "$MIGRATE" >&2
  exit 1
fi

docker run -d --name "$SERVER" --network "$NETWORK" "$IMAGE" \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://mlflow:mlflow@"$POSTGRES":5432/mlflow \
  --serve-artifacts \
  --artifacts-destination /mlartifacts \
  --default-artifact-root /mlartifacts >/dev/null

for _ in $(seq 1 60); do
  if docker exec "$SERVER" curl -fsS http://127.0.0.1:5000/health >/dev/null 2>&1; then
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "$SERVER" | grep -qx true; then
    docker logs "$SERVER" >&2
    exit 1
  fi
  sleep 1
done
docker exec "$SERVER" curl -fsS http://127.0.0.1:5000/health >/dev/null

test "$(docker exec "$SERVER" readlink /proc/1/exe)" = "/usr/local/bin/mlflow-server"
docker exec "$SERVER" test -x /usr/local/bin/mlflow-genai-worker
if docker logs "$SERVER" 2>&1 | grep -q 'native job execution is enabled but mlflow-genai-worker is unavailable'; then
  docker logs "$SERVER" >&2
  exit 1
fi
docker logs "$SERVER" 2>&1 | grep -q 'mlflow-server listening'

echo "PASS: /health returned 200 from mlflow-server"
echo "PASS: jobs-enabled startup resolved executable sibling mlflow-genai-worker"
echo "IMAGE AUDIT OK"
