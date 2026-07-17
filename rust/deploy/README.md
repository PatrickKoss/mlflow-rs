# MLflow reference deployment (Rust + Python behind nginx)

Reference deployment for the split MLflow server (RUST_TRACKING_SERVER_PLAN.md
T11.3/T11.4). One nginx front door on `:80` routes every request to the Rust
tracking server **except** the genai / gateway / jobs surface, which goes to
the stock Python MLflow server, and the UI (`/`, `/static-files/*`), which
nginx serves directly from the React build. Rust and Python share one
Alembic-migrated Postgres database and a shared artifact volume.

## Quick start

```bash
cd rust

# Populate the UI build nginx will serve (see "Building the UI" below).
# Real build:
yarn --cwd ../mlflow/server/js install --frozen-lockfile
yarn --cwd ../mlflow/server/js build
# ...or, for a quick smoke without yarn/network, a minimal placeholder:
bash deploy/build_placeholder_ui.sh

docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d --wait
bash deploy/smoke.sh              # asserts backend attribution; exits non-zero on mismatch
bash deploy/smoke_frontend.sh     # UI cache headers + "UI survives Python down" (T11.4 AC)
docker compose -f deploy/docker-compose.yml down -v
```

MLflow is then reachable at `http://localhost:80`.

## Building the UI

nginx serves the UI directly from `mlflow/server/js/build/` (bind-mounted
read-only into the `nginx` container at `/usr/share/mlflow-ui`), the same
directory the stock Python server ships (`REL_STATIC_DIR = "js/build"` in
`mlflow/server/__init__.py`). Build it with:

```bash
cd mlflow/server/js
yarn install --frozen-lockfile
yarn build          # -> mlflow/server/js/build/ (index.html + static/{js,css}/*)
```

This is a full CRA production build (source maps disabled, `--max_old_space_size=8192`)
and can take several minutes and needs network access for `yarn install`.
`mlflow/server/js/build/` is already gitignored; nothing about this step is
committed.

### Build availability fallback

If `mlflow/server/js/build/` doesn't exist (bind-mounting a missing host path
makes Docker create it as an empty directory), nginx's `try_files` on `/` and
`/static-files/*` falls through to a named location that proxies the request to
the Python container instead — i.e. **without a build present, behavior reverts
to the pre-T11.4 (T11.3) proxy-to-Python UI**, so `docker compose up` always
comes up with a working UI either way. This is a deliberate choice over a
compose profile: no extra `--profile` flag to remember, and it degrades
gracefully instead of nginx erroring on a missing file.

For a quick smoke run without the real (network-heavy) `yarn build`,
`deploy/build_placeholder_ui.sh` writes a minimal stand-in build (one
`index.html` + one hashed `static/js/main.<hash>.js`) — enough to exercise the
nginx static-serving + cache-header paths, but **not** the real MLflow UI.

## Service graph & startup ordering

```
postgres ──(healthy)──▶ migrate ──(completed)──▶ rust  ┐
                                              └──▶ python ┴──(both healthy)──▶ nginx :80
```

- **postgres** (`postgres:16`): backing DB, `pg_isready` healthcheck. No host
  port published.
- **migrate** (`ghcr.io/mlflow/mlflow`): one-shot `mlflow db upgrade` that brings
  the empty DB to the current Alembic head, then **exits 0**. This is the crux
  of the ordering: the Rust server refuses to start against an unmigrated DB
  (`Db::connect_and_verify`), so migration must finish before `rust` boots. Both
  `rust` and `python` `depends_on` it with `service_completed_successfully`.
- **rust**: the `mlflow-server` binary built from `Dockerfile.rust`. Serves on
  `:5000`, `--serve-artifacts`, artifacts to the shared `/mlartifacts` volume.
- **python** (`ghcr.io/mlflow/mlflow`): stock `mlflow server` on `:5001`, same
  Postgres URI and artifact volume. Hit only for the genai/gateway surface, and
  it ships the React UI build served at `/` for this task.
- **nginx** (`nginx:1.27-alpine`): reverse proxy on `:80` mounting `nginx.conf`.

## Routing table (nginx.conf, from plan §2.2)

Default rule: **everything not listed goes to Rust.**

| Prefix (regex) | Backend | Notes |
|---|---|---|
| `/(api\|ajax-api)/3.0/mlflow/{gateway,scorers,datasets,issues,genai,label-schemas,review-queues}/*` | Python | genai/gateway family |
| `/ajax-api/3.0/mlflow/assistant/*` | Python | SSE, `proxy_buffering off` |
| `/ajax-api/3.0/jobs/*` | Python | jobs API |
| `/gateway/*` | Python | streaming, `proxy_buffering off` |
| `/ajax-api/2.0/mlflow/gateway-proxy` | Python | `proxy_buffering off` |
| `/ajax-api/2.0/mlflow/runs/create-promptlab-run` | Python | genai authoring |
| `/(api\|ajax-api)/3.0/mlflow/scorer/invoke` | Python | genai eval, `proxy_buffering off` |
| `/python/health` | Python | rewritten to `/health` for ops |
| `/` | nginx (static) | `index.html` from the build dir; falls back to Python proxy if the build dir is empty |
| `/static-files/*` | nginx (static) | hashed JS/CSS from the build dir; same fallback |
| **everything else** | **Rust** | tracking, runs, metrics, tracing, OTLP `/v1/traces`, artifacts (`/get-artifact`, `/model-versions/get-artifact`, `/mlflow-artifacts/*`, `/ajax-api/2.0/mlflow/upload-artifact`), logged models, registry, webhooks, `/graphql`, users/roles/permissions, `/signup`, workspaces, `server-info`, ui-telemetry, `/health`, `/version`, `/metrics` |

### Cache policy (T11.4)

Mirrors `mlflow/server/__init__.py`'s Flask static handlers exactly, with one
deliberate strengthening on `index.html` (see note):

| Path | `Cache-Control` | Matches Python? |
|---|---|---|
| `/static-files/*` (hashed `static/js/*.js`, `static/css/*.css`) | `public, max-age=2419200` (28 days) | Yes — `send_from_directory(..., max_age=2419200)` in `serve_static_file()` |
| `/` (`index.html`) | `no-cache` | **Stricter than Python.** Flask's `serve()` calls `send_from_directory(..., "index.html")` with no `max_age`, so Python sets no explicit `Cache-Control` at all (browsers fall back to heuristic/no caching, but nothing forces revalidation). We pin `no-cache` explicitly so an intermediary proxy/CDN can never serve a stale shell after a new build ships new hashed asset URLs. |

### Attribution

Every response carries `X-MLflow-Backend: rust`, `python`, or (T11.4) `static`
(emitted with `add_header ... always`, so it survives 4xx/5xx):

- `rust` / `python` — proxied to that backend.
- `static` — served by nginx directly from the mounted UI build (`/`,
  `/static-files/*`). If the build dir is missing, these fall back to the
  Python proxy and are tagged `python` instead, same as pre-T11.4 behavior.

`smoke.sh` and `smoke_frontend.sh` assert this header per request. nginx also
logs `backend=<tag>` per line via a custom `log_format`:

```bash
docker compose -f deploy/docker-compose.yml logs nginx
```

### Streaming / body-size tuning

- `proxy_buffering off` on the SSE/streaming Python locations (assistant,
  gateway, gateway-proxy, scorer/invoke) so tokens flush immediately.
- `client_max_body_size 0` (unlimited) at the server level so large artifact
  uploads through the proxy artifact plane are not capped.

## Ports & volumes

| | |
|---|---|
| nginx | `:80` (host) |
| rust | `:5000` (internal) |
| python | `:5001` (internal) |
| postgres | `:5432` (internal) |
| `artifacts` volume | mounted at `/mlartifacts` in both `rust` and `python` |
| `mlflow/server/js/build` (host) | bind-mounted read-only at `/usr/share/mlflow-ui` in `nginx` (T11.4) |

## smoke.sh

Waits for `/health`, then exercises the SDK surface through nginx and asserts
backend attribution: experiments (create / get-by-name / search), runs
(create / log-metric / log-parameter / get-history / search), traces (V3 start +
OTLP `/v1/traces`), registry (registered model + version + get), webhooks (list),
users endpoint (rust-attributed whether or not auth is enabled), artifact
upload+download+list via the proxy. Finally it fires genai/gateway/jobs requests
and asserts they attribute to **python** (a 404/501 from Python is fine —
attribution is the assertion). Exits non-zero on any mismatch.

## smoke_frontend.sh (T11.4)

Requires the stack already up AND a UI build at `mlflow/server/js/build/`
(real or `build_placeholder_ui.sh`). With Python still running, checks `/` and
a hashed asset (auto-discovered from `index.html`'s `<script src>`) both come
back `backend=static` with the right `Cache-Control` (`no-cache` for `/`,
`public, max-age=2419200` for the asset), and a plain tracking API call still
hits Rust. Then it stops the `python` compose service and re-checks: `/` and
the hashed asset still 200 + `backend=static` (nginx doesn't need Python for
these), the tracking API still works (Rust doesn't need Python either), and a
genai request now fails with 502/503/504 (**expected** — genai has no Rust
implementation and Python is down). Restarts `python` on exit (even on
failure) via a trap, so a subsequent `smoke.sh` run isn't left with Python
stopped. Exits non-zero on any check failure.
