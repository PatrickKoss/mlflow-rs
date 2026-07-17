#!/usr/bin/env bash
# Generates a MINIMAL placeholder UI build under `mlflow/server/js/build/` so the
# T11.4 nginx static-serving path (and its smoke coverage) can be exercised
# without running the real, network-heavy `yarn build` (see README.md "Building
# the UI"). NOT a substitute for the real build — swap it out with
# `yarn --cwd mlflow/server/js build` for anything beyond this smoke test.
#
# Layout mirrors CRA's actual output (`homepage: "static-files"` in
# mlflow/server/js/package.json bakes this shape into a real build):
#   build/index.html                  -- SPA shell, references the hashed asset
#   build/static/js/main.<hash>.js    -- one hashed "bundle" so the 28-day
#                                         Cache-Control path has something real
#                                         to serve through nginx's /static-files/
#                                         location.
#
# Usage: bash build_placeholder_ui.sh [build_dir]
set -euo pipefail

BUILD_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../mlflow/server/js" && pwd)/build}"
ASSET_HASH="deadbeef01"
JS_REL="static/js/main.${ASSET_HASH}.js"

mkdir -p "${BUILD_DIR}/static/js" "${BUILD_DIR}/static/css"

cat > "${BUILD_DIR}/${JS_REL}" <<EOF
// Placeholder hashed bundle (T11.4 smoke only — not the real MLflow UI build).
console.log("mlflow-ui placeholder build ${ASSET_HASH}");
EOF

cat > "${BUILD_DIR}/index.html" <<EOF
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>MLflow (placeholder build)</title></head>
  <body>
    <div id="root">T11.4 placeholder UI build — not the real MLflow React app.</div>
    <script src="/static-files/${JS_REL}"></script>
  </body>
</html>
EOF

echo "Placeholder UI build written to ${BUILD_DIR}"
