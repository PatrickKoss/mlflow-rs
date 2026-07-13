//! Proto-backed HTTP route table for the MLflow server.
//!
//! [`ROUTE_TABLE`] is generated at build time (see `build.rs`) by decoding the
//! `databricks.rpc` `MethodOptions` extension off every RPC in `MlflowService`,
//! `ModelRegistryService`, `MlflowArtifactsService`, and `WebhookService`. The
//! table stores the RAW proto-level endpoint (path + `since` version); use
//! [`RouteSpec::expand`] to obtain the concrete Flask paths.

use serde::Serialize;

/// A single raw proto endpoint: one `HttpEndpoint` of a `databricks.rpc` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RouteSpec {
    /// Owning gRPC service name, e.g. `"MlflowService"`.
    pub service: &'static str,
    /// RPC method name, e.g. `"createExperiment"`.
    pub method: &'static str,
    /// HTTP method for this endpoint, e.g. `"POST"` / `"GET"` / `"PATCH"`.
    pub http_method: &'static str,
    /// Proto-declared path, e.g. `"/mlflow/experiments/create"`. Note some paths
    /// intentionally lack a leading slash (e.g. the search-datasets endpoint);
    /// this is preserved verbatim to match Python string concatenation.
    pub path: &'static str,
    /// `since.major` — drives the URL version component (`2` → `/api/2.0/...`).
    pub since_major: i32,
    /// `since.minor`.
    pub since_minor: i32,
}

/// A concrete, registrable route: one HTTP method + one fully-qualified path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpandedRoute {
    pub service: &'static str,
    pub method: &'static str,
    pub http_method: &'static str,
    pub path: String,
}

impl RouteSpec {
    /// Expand this raw route into its two concrete Flask paths (`/api/{v}.0...`
    /// and `/ajax-api/{v}.0...`), mirroring `handlers.py::_get_paths`.
    ///
    /// `static_prefix` (if non-empty) is prepended to every path exactly like
    /// `handlers._add_static_prefix` (with a trailing slash stripped),
    /// honoring `MLFLOW_STATIC_PREFIX`.
    pub fn expand(&self, static_prefix: &str) -> Vec<ExpandedRoute> {
        let flask_path = convert_path_parameter_to_flask_format(self.path);
        let version = self.since_major;
        [
            format!("/api/{version}.0{flask_path}"),
            format!("/ajax-api/{version}.0{flask_path}"),
        ]
        .into_iter()
        .map(|route| ExpandedRoute {
            service: self.service,
            method: self.method,
            http_method: self.http_method,
            path: add_static_prefix(static_prefix, &route),
        })
        .collect()
    }
}

/// Mirrors `handlers._add_static_prefix`: prepend the prefix (trailing slash
/// stripped) when set, otherwise return the route unchanged.
fn add_static_prefix(static_prefix: &str, route: &str) -> String {
    if static_prefix.is_empty() {
        route.to_string()
    } else {
        format!("{}{route}", static_prefix.trim_end_matches('/'))
    }
}

/// Mirrors `handlers._convert_path_parameter_to_flask_format`: `{trace_id}` ->
/// `<trace_id>`, and the Databricks-specific `{assessment.trace_id}` ->
/// `<trace_id>`.
fn convert_path_parameter_to_flask_format(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut chars = path.char_indices().peekable();
    while let Some((_, c)) = chars.next() {
        if c != '{' {
            out.push(c);
            continue;
        }
        // Collect until the matching '}'.
        let mut inner = String::new();
        let mut closed = false;
        for (_, ic) in chars.by_ref() {
            if ic == '}' {
                closed = true;
                break;
            }
            inner.push(ic);
        }
        if !closed {
            // Unbalanced brace: emit literally, matching a no-op regex.
            out.push('{');
            out.push_str(&inner);
            continue;
        }
        // `{assessment.trace_id}` -> `<trace_id>`; otherwise take the last
        // dotted segment is NOT what Python does — Python's first regex only
        // rewrites `\w+` (no dots), so `{a.b}` is untouched by it, then the
        // second regex maps the specific `{assessment.trace_id}` token. Emulate
        // both precisely.
        if inner == "assessment.trace_id" {
            out.push_str("<trace_id>");
        } else if is_word(&inner) {
            out.push('<');
            out.push_str(&inner);
            out.push('>');
        } else {
            // Not a bare word and not the special token: leave the braces (the
            // `{\w+}` regex would not have matched).
            out.push('{');
            out.push_str(&inner);
            out.push('}');
        }
    }
    out
}

/// True when `s` is a non-empty run of `\w` characters (ASCII word chars),
/// matching Python's `\w+` for the endpoint paths in scope.
fn is_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

include!(concat!(env!("OUT_DIR"), "/routes_generated.rs"));
