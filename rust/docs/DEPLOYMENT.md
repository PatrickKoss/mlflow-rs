# Split-server deployment

This guide deploys nginx in front of a Rust MLflow tracking server and a Python
MLflow server. Use the same pinned MLflow source revision for both images. For
an existing installation, follow [MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md);
for reversal, use [ROLLBACK.md](ROLLBACK.md).

## Topology

```text
clients -> nginx -> Rust: tracking, tracing, OTLP, registry, auth/RBAC,
                   webhooks, workspaces, GraphQL, local artifact proxy, ops
                 -> Python: GenAI, gateway, assistant/SSE, jobs, PromptLab,
                    cloud-backed artifact proxy, packaged UI
        -> shared tracking/registry DB
        -> shared auth DB
        -> shared artifact store
```

Both applications enforce auth for the requests they receive. They must share
the auth database and webhook Fernet key. Python/Alembic owns both schemas; Rust
only verifies their exact heads at startup.

### nginx routing contract

Default every API route to Rust, then declare these Python exceptions before
the default location:

| Route | Backend | Notes |
|---|---|---|
| `/(api|ajax-api)/3.0/mlflow/(gateway|scorers|datasets|issues|genai|label-schemas|review-queues)[/*]` | Python | Part 1 GenAI families |
| `/ajax-api/3.0/mlflow/assistant[/*]` | Python | SSE; disable proxy buffering |
| `/ajax-api/3.0/jobs[/*]` | Python | job-execution runtime |
| `/gateway[/*]` | Python | streaming; disable proxy buffering |
| `/ajax-api/2.0/mlflow/gateway-proxy` | Python | streaming |
| `/ajax-api/2.0/mlflow/runs/create-promptlab-run` | Python | PromptLab execution |
| `/(api|ajax-api)/3.0/mlflow/scorer/invoke` | Python | scorer execution |
| `/python/health` | Python | rewrite to `/health` |
| `/(api|ajax-api)/2.0/mlflow-artifacts[/*]` | Python only for S3/GCS/Azure proxy destinations | Keep on Rust only for local/file proxy destinations |
| `/`, `/static-files/*` | static image/CDN; Python fallback in the example | UI, not an API plane |
| everything else | Rust | Includes tracking, registry, auth, webhooks, `/graphql`, `/v1/traces`, `/health`, `/version`, and `/metrics` |

The artifact-proxy routes are exactly the eight `MlflowArtifactsService`
endpoints under both `/api/2.0/mlflow-artifacts/...` and
`/ajax-api/2.0/mlflow-artifacts/...`: `artifacts`, `mpu/create`,
`mpu/complete`, `mpu/abort`, and `presigned`, including their path suffixes.
Root `/get-artifact`, `/model-versions/get-artifact`, and
`/ajax-api/2.0/mlflow/upload-artifact` are separate routes.

## Docker Compose walkthrough

The example uses PostgreSQL for tracking/registry, the default shared SQLite
auth database on a named volume, MinIO, local source builds for both MLflow
images, and the cloud-artifact exception: Rust has `--no-serve-artifacts`, while
Python proxies the `mlflow-artifacts` routes to MinIO. The committed source must
contain tracking head `c4a9b7d3e812` and auth head `f1a2b3c4d5e6`.

1. Save the following as `docker-compose.yml` in the repository root. The fixed
   passwords and keys are for this local walkthrough only.

```yaml
name: mlflow-split-docs

x-python-image: &python-image
  image: mlflow-python-split:local
  build:
    context: .
    dockerfile_inline: |
      FROM python:3.10-slim
      WORKDIR /opt/mlflow
      COPY . .
      RUN pip install --no-cache-dir '.[db,auth,gateway,genai]'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow-local-only
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow -d mlflow"]
      interval: 2s
      timeout: 5s
      retries: 30
    volumes:
      - postgres-data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address :9001
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 2s
      timeout: 5s
      retries: 30
    volumes:
      - minio-data:/data

  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint:
      - /bin/sh
      - -ec
      - mc alias set local http://minio:9000 minioadmin minioadmin && mc mb --ignore-existing local/mlflow
    restart: "no"

  migrate-tracking:
    <<: *python-image
    depends_on:
      postgres:
        condition: service_healthy
    command:
      - mlflow
      - db
      - upgrade
      - postgresql://mlflow:mlflow-local-only@postgres:5432/mlflow
    restart: "no"

  migrate-auth:
    <<: *python-image
    command:
      - python
      - -m
      - mlflow.server.auth
      - db
      - upgrade
      - --url
      - sqlite:////auth/basic_auth.db
      - --revision
      - head
    volumes:
      - auth-data:/auth
    restart: "no"

  rust:
    image: mlflow-rust-split:local
    build:
      context: .
      dockerfile: rust/deploy/Dockerfile.rust
    depends_on:
      migrate-tracking:
        condition: service_completed_successfully
      migrate-auth:
        condition: service_completed_successfully
    environment:
      RUST_LOG: info
      MLFLOW_AUTH_CONFIG_PATH: /etc/mlflow/auth.ini
      MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY: AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=
    command:
      - --host
      - 0.0.0.0
      - --port
      - "5000"
      - --backend-store-uri
      - postgresql://mlflow:mlflow-local-only@postgres:5432/mlflow
      - --app-name
      - basic-auth
      - --no-serve-artifacts
      - --default-artifact-root
      - mlflow-artifacts:/
    configs:
      - source: auth-ini
        target: /etc/mlflow/auth.ini
    volumes:
      - auth-data:/auth
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:5000/health || exit 1"]
      interval: 3s
      timeout: 5s
      retries: 30

  python:
    <<: *python-image
    depends_on:
      migrate-tracking:
        condition: service_completed_successfully
      migrate-auth:
        condition: service_completed_successfully
      minio-init:
        condition: service_completed_successfully
    environment:
      MLFLOW_AUTH_CONFIG_PATH: /etc/mlflow/auth.ini
      MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY: AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=
      MLFLOW_FLASK_SERVER_SECRET_KEY: local-python-session-key-change-me
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      AWS_DEFAULT_REGION: us-east-1
      MLFLOW_SERVER_ENABLE_JOB_EXECUTION: "false"
    command:
      - mlflow
      - server
      - --host
      - 0.0.0.0
      - --port
      - "5001"
      - --backend-store-uri
      - postgresql://mlflow:mlflow-local-only@postgres:5432/mlflow
      - --app-name
      - basic-auth
      - --serve-artifacts
      - --artifacts-destination
      - s3://mlflow/artifacts
    configs:
      - source: auth-ini
        target: /etc/mlflow/auth.ini
    volumes:
      - auth-data:/auth
    healthcheck:
      test:
        - CMD-SHELL
        - python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')"
      interval: 3s
      timeout: 5s
      retries: 30

  nginx:
    image: nginx:1.27-alpine
    depends_on:
      rust:
        condition: service_healthy
      python:
        condition: service_healthy
    ports:
      - "8080:80"
    configs:
      - source: nginx-conf
        target: /etc/nginx/conf.d/default.conf
    healthcheck:
      test: ["CMD", "wget", "-q", "-O", "/dev/null", "http://127.0.0.1/health"]
      interval: 3s
      timeout: 5s
      retries: 30

configs:
  auth-ini:
    content: |
      [mlflow]
      default_permission = READ
      database_uri = sqlite:////auth/basic_auth.db
      admin_username = admin
      admin_password = change-this-local-password
      authorization_function = mlflow.server.auth:authenticate_request_basic_auth
      grant_default_workspace_access = false
      auth_cache_ttl_seconds = 0

  nginx-conf:
    content: |
      upstream rust_backend { server rust:5000; }
      upstream python_backend { server python:5001; }

      server {
        listen 80;
        server_name _;
        client_max_body_size 0;

        location ~ ^/(api|ajax-api)/3\.0/mlflow/(gateway|scorers|datasets|issues|genai|label-schemas|review-queues)(/|$$) {
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }
        location ~ ^/ajax-api/3\.0/mlflow/assistant(/|$$) {
          proxy_pass http://python_backend;
          proxy_http_version 1.1;
          proxy_set_header Connection "";
          proxy_buffering off;
          proxy_read_timeout 3600s;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }
        location ~ ^/ajax-api/3\.0/jobs(/|$$) {
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }
        location ~ ^/gateway(/|$$) {
          proxy_pass http://python_backend;
          proxy_http_version 1.1;
          proxy_set_header Connection "";
          proxy_buffering off;
          proxy_read_timeout 3600s;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }
        location = /ajax-api/2.0/mlflow/gateway-proxy {
          proxy_pass http://python_backend;
          proxy_buffering off;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }
        location = /ajax-api/2.0/mlflow/runs/create-promptlab-run {
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }
        location ~ ^/(api|ajax-api)/3\.0/mlflow/scorer/invoke$$ {
          proxy_pass http://python_backend;
          proxy_buffering off;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }

        # Rust's artifact proxy cannot use S3/GCS/Azure destinations.
        location ~ ^/(api|ajax-api)/2\.0/mlflow-artifacts(/|$$) {
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }

        location = /python/health {
          rewrite ^ /health break;
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python always;
        }

        # The local example uses the UI packaged in the Python image. Use a
        # static nginx image or CDN in production.
        location = / {
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python-ui always;
        }
        location /static-files/ {
          proxy_pass http://python_backend;
          proxy_set_header Host $$host;
          add_header X-MLflow-Backend python-ui always;
        }

        location / {
          proxy_pass http://rust_backend;
          proxy_set_header Host $$host;
          proxy_set_header X-Real-IP $$remote_addr;
          proxy_set_header X-Forwarded-For $$proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $$scheme;
          proxy_set_header X-MLFLOW-WORKSPACE $$http_x_mlflow_workspace;
          add_header X-MLflow-Backend rust always;
        }
      }

volumes:
  postgres-data:
  minio-data:
  auth-data:
```

2. Build and start it:

   ```bash
   docker compose build
   docker compose up -d --wait
   ```

3. Verify the route split. Auth is enabled, so use the walkthrough admin:

   ```bash
   curl -i -u admin:change-this-local-password \
     -H 'Content-Type: application/json' -d '{"max_results":1}' \
     http://localhost:8080/api/2.0/mlflow/experiments/search

   curl -i -u admin:change-this-local-password \
     -X PUT --data-binary 'artifact-through-python' \
     http://localhost:8080/api/2.0/mlflow-artifacts/artifacts/smoke/hello.txt

   curl -i -u admin:change-this-local-password \
     http://localhost:8080/api/3.0/mlflow/genai/does-not-exist
   ```

   The first response must be successful and contain `X-MLflow-Backend: rust`.
   The artifact upload must contain `X-MLflow-Backend: python`; a following GET
   of the same URL must return `artifact-through-python`. The GenAI request may
   return a version-dependent 404, but its backend header must be `python`.

4. Always tear the walkthrough down, including data volumes:

   ```bash
   docker compose down -v
   ```

## Kubernetes

Use managed PostgreSQL and object storage in production. This manifest keeps
the default SQLite auth database on a shared `ReadWriteMany` volume so both
servers see one file; replace it with an externally managed auth database only
after validating that database dialect against the pinned Rust build. Build and
push the two images from the same checkout, then replace the two
`YOUR_REGISTRY` image names and every `REPLACE_...` value below. The Python image must include
`[db,auth,gateway,genai]`. Apply database migrations from
[MIGRATION_RUNBOOK.md](MIGRATION_RUNBOOK.md#3-upgrade-both-databases-with-the-pinned-python-image)
before applying these workloads.

Build and push example (replace the registry and immutable tag first):

```bash
export SPLIT_TAG=c4a9b7d3e812
export SPLIT_REGISTRY=registry.example/mlflow
docker build -f rust/deploy/Dockerfile.rust \
  -t "$SPLIT_REGISTRY/mlflow-rust:$SPLIT_TAG" .
docker build -t "$SPLIT_REGISTRY/mlflow-python:$SPLIT_TAG" -f - . <<'DOCKERFILE'
FROM python:3.10-slim
WORKDIR /opt/mlflow
COPY . .
RUN pip install --no-cache-dir '.[db,auth,gateway,genai]'
DOCKERFILE
docker push "$SPLIT_REGISTRY/mlflow-rust:$SPLIT_TAG"
docker push "$SPLIT_REGISTRY/mlflow-python:$SPLIT_TAG"
```

The example intentionally runs one Rust replica because its signup-CSRF secret
is per process. If `/signup` is unused, or ingress session affinity covers the
signup GET/POST pair, scale it after load testing. Resource sizing must be based
on local load tests; [`../bench/memory.md`](../bench/memory.md) and
[`../bench/soak.md`](../bench/soak.md) are reference measurements only.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-split-secrets
type: Opaque
stringData:
  backend-store-uri: postgresql://mlflow:REPLACE_PASSWORD@postgres.example:5432/mlflow
  webhook-fernet-key: REPLACE_WITH_URLSAFE_BASE64_32_BYTE_KEY
  flask-session-key: REPLACE_WITH_LONG_RANDOM_PYTHON_ONLY_KEY
  aws-access-key-id: REPLACE_ACCESS_KEY
  aws-secret-access-key: REPLACE_SECRET_KEY
  auth.ini: |
    [mlflow]
    default_permission = READ
    database_uri = sqlite:////auth/basic_auth.db
    admin_username = admin
    admin_password = REPLACE_BOOTSTRAP_PASSWORD
    authorization_function = mlflow.server.auth:authenticate_request_basic_auth
    grant_default_workspace_access = false
    auth_cache_ttl_seconds = 0
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-auth-db
spec:
  accessModes: [ReadWriteMany]
  storageClassName: REPLACE_RWX_STORAGE_CLASS
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-nginx
data:
  default.conf: |
    upstream rust_backend { server mlflow-rust:5000; }
    upstream python_backend { server mlflow-python:5001; }
    server {
      listen 8080;
      server_name _;
      client_max_body_size 0;
      location ~ ^/(api|ajax-api)/3\.0/mlflow/(gateway|scorers|datasets|issues|genai|label-schemas|review-queues)(/|$) {
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location ~ ^/ajax-api/3\.0/mlflow/assistant(/|$) {
        proxy_pass http://python_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location ~ ^/ajax-api/3\.0/jobs(/|$) {
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location ~ ^/gateway(/|$) {
        proxy_pass http://python_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location = /ajax-api/2.0/mlflow/gateway-proxy {
        proxy_pass http://python_backend;
        proxy_buffering off;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location = /ajax-api/2.0/mlflow/runs/create-promptlab-run {
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location ~ ^/(api|ajax-api)/3\.0/mlflow/scorer/invoke$ {
        proxy_pass http://python_backend;
        proxy_buffering off;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location ~ ^/(api|ajax-api)/2\.0/mlflow-artifacts(/|$) {
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location = /python/health {
        rewrite ^ /health break;
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python always;
      }
      location = / {
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python-ui always;
      }
      location /static-files/ {
        proxy_pass http://python_backend;
        proxy_set_header Host $host;
        add_header X-MLflow-Backend python-ui always;
      }
      location / {
        proxy_pass http://rust_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-MLFLOW-WORKSPACE $http_x_mlflow_workspace;
        add_header X-MLflow-Backend rust always;
      }
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-rust
spec:
  replicas: 1
  selector:
    matchLabels: {app: mlflow-rust}
  template:
    metadata:
      labels: {app: mlflow-rust}
    spec:
      containers:
        - name: server
          image: YOUR_REGISTRY/mlflow-rust:c4a9b7d3e812
          args:
            - --host
            - 0.0.0.0
            - --port
            - "5000"
            - --app-name
            - basic-auth
            - --no-serve-artifacts
            - --default-artifact-root
            - mlflow-artifacts:/
          env:
            - name: MLFLOW_BACKEND_STORE_URI
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: backend-store-uri}
            - name: MLFLOW_AUTH_CONFIG_PATH
              value: /etc/mlflow/auth.ini
            - name: MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: webhook-fernet-key}
          ports:
            - {name: http, containerPort: 5000}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health, port: http}
          volumeMounts:
            - {name: auth-config, mountPath: /etc/mlflow, readOnly: true}
            - {name: auth-db, mountPath: /auth}
      volumes:
        - name: auth-config
          secret:
            secretName: mlflow-split-secrets
            items:
              - {key: auth.ini, path: auth.ini}
        - name: auth-db
          persistentVolumeClaim:
            claimName: mlflow-auth-db
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-rust
spec:
  selector: {app: mlflow-rust}
  ports:
    - {name: http, port: 5000, targetPort: http}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-python
spec:
  replicas: 2
  selector:
    matchLabels: {app: mlflow-python}
  template:
    metadata:
      labels: {app: mlflow-python}
    spec:
      containers:
        - name: server
          image: YOUR_REGISTRY/mlflow-python:c4a9b7d3e812
          command: [mlflow]
          args:
            - server
            - --host
            - 0.0.0.0
            - --port
            - "5001"
            - --app-name
            - basic-auth
            - --serve-artifacts
            - --artifacts-destination
            - s3://REPLACE_BUCKET/mlflow
          env:
            - name: MLFLOW_BACKEND_STORE_URI
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: backend-store-uri}
            - {name: MLFLOW_AUTH_CONFIG_PATH, value: /etc/mlflow/auth.ini}
            - name: MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: webhook-fernet-key}
            - name: MLFLOW_FLASK_SERVER_SECRET_KEY
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: flask-session-key}
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: aws-access-key-id}
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef: {name: mlflow-split-secrets, key: aws-secret-access-key}
            - {name: AWS_DEFAULT_REGION, value: us-east-1}
            - {name: MLFLOW_S3_ENDPOINT_URL, value: https://REPLACE_OBJECT_STORE_ENDPOINT}
            - {name: MLFLOW_SERVER_ENABLE_JOB_EXECUTION, value: "true"}
          ports:
            - {name: http, containerPort: 5001}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health, port: http}
          volumeMounts:
            - {name: auth-config, mountPath: /etc/mlflow, readOnly: true}
            - {name: auth-db, mountPath: /auth}
      volumes:
        - name: auth-config
          secret:
            secretName: mlflow-split-secrets
            items:
              - {key: auth.ini, path: auth.ini}
        - name: auth-db
          persistentVolumeClaim:
            claimName: mlflow-auth-db
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-python
spec:
  selector: {app: mlflow-python}
  ports:
    - {name: http, port: 5001, targetPort: http}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-nginx
spec:
  replicas: 2
  selector:
    matchLabels: {app: mlflow-nginx}
  template:
    metadata:
      labels: {app: mlflow-nginx}
    spec:
      containers:
        - name: nginx
          image: nginx:1.27-alpine
          ports:
            - {name: http, containerPort: 8080}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health, port: http}
          volumeMounts:
            - {name: config, mountPath: /etc/nginx/conf.d, readOnly: true}
      volumes:
        - name: config
          configMap:
            name: mlflow-nginx
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-nginx
spec:
  type: LoadBalancer
  selector: {app: mlflow-nginx}
  ports:
    - {name: http, port: 80, targetPort: http}
```

Apply and wait:

```bash
kubectl apply -f mlflow-split.yaml
kubectl rollout status deployment/mlflow-rust
kubectl rollout status deployment/mlflow-python
kubectl rollout status deployment/mlflow-nginx
kubectl get endpoints mlflow-rust mlflow-python mlflow-nginx
```

Do not commit the populated Secret. Prefer the cluster's external secret
operator and pin secret versions so both server Deployments roll together.

## Rust flag and environment reference

CLI flags override their environment variables. The table is the operational
subset of `rust/crates/mlflow-server/CLI_PARITY.md`; unsupported process-manager
flags are listed because deploy scripts commonly carry them forward.

| Flag | Environment | Default / Rust behavior |
|---|---|---|
| `--backend-store-uri` | `MLFLOW_BACKEND_STORE_URI` | Unset: ops-only server; required for APIs |
| `--read-replica-backend-store-uri` | `MLFLOW_READ_REPLICA_BACKEND_STORE_URI` | Accepted, but all tracking reads still use primary; warning logged |
| `--registry-store-uri` | `MLFLOW_REGISTRY_STORE_URI` | Backend URI; a different URI is rejected |
| `--default-artifact-root` | `MLFLOW_DEFAULT_ARTIFACT_ROOT` | `./mlruns` when unset |
| `--serve-artifacts` / `--no-serve-artifacts` | `MLFLOW_SERVE_ARTIFACTS` | `true` |
| `--artifacts-destination` | `MLFLOW_ARTIFACTS_DESTINATION` | `./mlartifacts`; only local/file is implemented |
| `--artifacts-only` | `MLFLOW_ARTIFACTS_ONLY` | `false`; registers proxy plus root get/upload routes only |
| `--host` / `-H` | `MLFLOW_HOST` | `127.0.0.1` |
| `--port` / `-p` | `MLFLOW_PORT` | `5000` |
| `--workers` / `-w` | `MLFLOW_WORKERS` | Accepted and ignored; one async Tokio runtime |
| `--static-prefix` | `MLFLOW_STATIC_PREFIX` | Unset; must start with `/` and not end with `/` |
| `--allowed-hosts` | `MLFLOW_SERVER_ALLOWED_HOSTS` | localhost and private-address defaults |
| `--cors-allowed-origins` | `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | localhost defaults |
| `--x-frame-options` | `MLFLOW_SERVER_X_FRAME_OPTIONS` | `SAMEORIGIN`; `NONE` disables |
| none | `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE` | `false`; `true` disables host/CORS/security-header middleware |
| `--expose-prometheus` | `MLFLOW_EXPOSE_PROMETHEUS` | Unset: `/metrics` is 404; any value enables it |
| `--app-name basic-auth` | `MLFLOW_AUTH_CONFIG_PATH` also enables auth | Other app names are rejected |
| `--workspace-store-uri` | `MLFLOW_WORKSPACE_STORE_URI` | Backend URI; used only with workspaces |
| `--enable-workspaces` / `--disable-workspaces` | `MLFLOW_ENABLE_WORKSPACES` | `false`; flags override env |
| `--trace-archival-config` | `MLFLOW_TRACE_ARCHIVAL_CONFIG` | Unset; CLI wins over env; Python-compatible YAML validation; incompatible with `--artifacts-only`; runtime config reloads on a 5s TTL and keeps the last valid value after refresh errors |
| `--gunicorn-opts`, `--waitress-opts`, `--uvicorn-opts`, `--dev` | corresponding Python env where applicable | Unknown/rejected; exit 2 |
| `--secrets-cache-ttl`, `--secrets-cache-max-size` | none | Not ported; unknown/rejected, exit 2 |

SQL pool environment mapping:

| Environment | Rust mapping |
|---|---|
| `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE` | `min_connections`; contributes to maximum |
| `MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW` | maximum = pool size + overflow |
| `MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE` | connection maximum lifetime in seconds |
| `MLFLOW_SQLALCHEMYSTORE_ECHO` | SQL logging through tracing |
| `MLFLOW_SQLALCHEMYSTORE_POOLCLASS` | accepted no-op |

Without pool variables, Rust uses maximum 15 and minimum 0 connections.

## Known limitation: cloud artifact proxy

> **KNOWN LIMITATION:** Rust's artifact proxy supports local paths and `file:`
> URIs only. `--serve-artifacts --artifacts-destination s3://...`, `gs://...`,
> or Azure destinations fail at request time with `NOT_IMPLEMENTED`.

Use one of these configurations:

1. Keep `--serve-artifacts` on Python, start Rust with
   `--no-serve-artifacts`, and route
   `^/(api|ajax-api)/2\.0/mlflow-artifacts(/|$)` to Python, as above.
2. Use client-direct uploads to S3/GCS/Azure and do not expose an artifact proxy.
3. Use Rust's proxy only with a shared local/file volume.

Do not point Rust's proxy at MinIO merely because MinIO is local to the cluster;
an `s3://` URI is still a cloud-scheme backend. The soak used client-direct S3
uploads for Rust for this reason; see [`../bench/soak.md`](../bench/soak.md).
