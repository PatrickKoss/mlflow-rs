//! HTTP integration tests for the security middleware (plan T11.2), mirroring
//! `mlflow/server/security.py` + `tests/server/test_security*.py`.
//!
//! Byte-checks the host-header allowlist rejection, CORS preflight/actual
//! headers, cross-origin state-change block, `X-Frame-Options` /
//! `X-Content-Type-Options` decoration, and the security-before-auth ordering.
//! Each test boots a real listener over the ops-only app (no store): the
//! security layer wraps every route, so its behavior is observable without a
//! backend — allowed requests fall through to a 404, but the security layer
//! runs (and rejects) first.

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, ServerConfig};
use tokio::net::TcpListener;

/// A booted server plus its base URL and shutdown handle.
struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl TestServer {
    async fn start(config: ServerConfig) -> Self {
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(&config, recorder, None);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("server error");
        });
        TestServer {
            base: format!("http://{addr}"),
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

/// Builds a `ServerConfig` with the given security knobs and inert values for
/// everything else (ops-only app).
fn config(
    allowed_hosts: Option<&[&str]>,
    cors_allowed_origins: Option<&[&str]>,
    x_frame_options: &str,
) -> ServerConfig {
    ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_prefix: None,
        backend_store_uri: None,
        default_artifact_root: None,
        serve_artifacts: true,
        artifacts_destination: None,
        allowed_hosts: allowed_hosts.map(|hs| hs.iter().map(|s| s.to_string()).collect()),
        cors_allowed_origins: cors_allowed_origins
            .map(|os| os.iter().map(|s| s.to_string()).collect()),
        x_frame_options: x_frame_options.to_string(),
        ..Default::default()
    }
}

struct Resp {
    status: StatusCode,
    body: String,
    content_type: Option<String>,
    acao: Option<String>,
    acac: Option<String>,
    acam: Option<String>,
    acah: Option<String>,
    vary: Option<String>,
    xfo: Option<String>,
    xcto: Option<String>,
}

/// Header list for the request builder: (name, value) pairs.
async fn request(base: &str, method: Method, path: &str, headers: &[(&str, &str)]) -> Resp {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{base}{path}");
    let mut builder = Request::builder().method(method).uri(uri);
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let req = builder.body(Full::new(Bytes::new())).unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let get = |name: &str| {
        resp.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let content_type = get("content-type");
    let acao = get("access-control-allow-origin");
    let acac = get("access-control-allow-credentials");
    let acam = get("access-control-allow-methods");
    let acah = get("access-control-allow-headers");
    let vary = get("vary");
    let xfo = get("x-frame-options");
    let xcto = get("x-content-type-options");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    Resp {
        status,
        body,
        content_type,
        acao,
        acac,
        acam,
        acah,
        vary,
        xfo,
        xcto,
    }
}

const INVALID_HOST_MSG: &str = "Invalid Host header - possible DNS rebinding attack detected";
const CORS_BLOCKED_MSG: &str = "Cross-origin request blocked";
const ALLOW_METHODS: &str = "DELETE, GET, OPTIONS, PATCH, POST, PUT";

// ---- Host-header allowlist ----

#[tokio::test]
async fn host_allowlist_matrix() {
    // (host, allowed?) — allowlist is localhost,127.0.0.1.
    let cases = [
        ("localhost", true),
        ("127.0.0.1", true),
        ("localhost:5000", false), // not in the explicit list (no :* pattern)
        ("evil.attacker.com", false),
    ];
    let srv = TestServer::start(config(
        Some(&["localhost", "127.0.0.1"]),
        None,
        "SAMEORIGIN",
    ))
    .await;
    for (host, allowed) in cases {
        // Hit a path that is not exempt and not routed: allowed -> 404 (falls
        // through), disallowed -> 403 with the byte-exact rejection body.
        let resp = request(&srv.base, Method::GET, "/some/path", &[("host", host)]).await;
        if allowed {
            assert_ne!(
                resp.status,
                StatusCode::FORBIDDEN,
                "host={host} should pass"
            );
        } else {
            assert_eq!(resp.status, StatusCode::FORBIDDEN, "host={host}");
            assert_eq!(resp.body, INVALID_HOST_MSG);
            assert_eq!(
                resp.content_type.as_deref(),
                Some("text/plain; charset=utf-8")
            );
        }
    }
}

#[tokio::test]
async fn default_host_allowlist_accepts_private_and_localhost() {
    // Unset allowed_hosts -> defaults (localhost + :* + private ranges).
    let srv = TestServer::start(config(None, None, "SAMEORIGIN")).await;
    for host in [
        "localhost",
        "127.0.0.1",
        "127.0.0.1:5000",
        "localhost:8080",
        "192.168.1.1",
        "10.0.0.1",
        "172.16.0.1",
        "[::1]",
        "[::1]:8080",
    ] {
        let resp = request(&srv.base, Method::GET, "/some/path", &[("host", host)]).await;
        assert_ne!(
            resp.status,
            StatusCode::FORBIDDEN,
            "host={host} should pass"
        );
    }
    let resp = request(
        &srv.base,
        Method::GET,
        "/some/path",
        &[("host", "evil.com")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(resp.body, INVALID_HOST_MSG);
}

#[tokio::test]
async fn wildcard_host_disables_validation() {
    let srv = TestServer::start(config(Some(&["*"]), None, "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::GET,
        "/some/path",
        &[("host", "any.domain.com")],
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn wildcard_subdomain_host_pattern() {
    let cases = [
        ("app.example.com", true),
        ("sub.app.example.com", true),
        ("evil.com", false),
    ];
    let srv = TestServer::start(config(Some(&["*.example.com"]), None, "SAMEORIGIN")).await;
    for (host, allowed) in cases {
        let resp = request(&srv.base, Method::GET, "/some/path", &[("host", host)]).await;
        if allowed {
            assert_ne!(resp.status, StatusCode::FORBIDDEN, "host={host}");
        } else {
            assert_eq!(resp.status, StatusCode::FORBIDDEN, "host={host}");
        }
    }
}

#[tokio::test]
async fn health_and_version_exempt_from_host_validation() {
    let srv = TestServer::start(config(Some(&["localhost"]), None, "SAMEORIGIN")).await;
    for path in ["/health", "/version"] {
        let resp = request(&srv.base, Method::GET, path, &[("host", "evil.com")]).await;
        assert_eq!(resp.status, StatusCode::OK, "path={path}");
    }
    // A non-exempt path with the same bad host is rejected.
    let resp = request(&srv.base, Method::GET, "/metrics", &[("host", "evil.com")]).await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
}

// ---- CORS preflight ----

#[tokio::test]
async fn preflight_allowed_origin_full_header_set() {
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::OPTIONS,
        "/api/2.0/mlflow/experiments/list",
        &[
            ("host", "localhost"),
            ("origin", "http://localhost:3000"),
            ("access-control-request-method", "POST"),
            ("access-control-request-headers", "Content-Type"),
        ],
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT);
    assert_eq!(resp.body, "");
    assert_eq!(resp.acao.as_deref(), Some("http://localhost:3000"));
    assert_eq!(resp.acac.as_deref(), Some("true"));
    assert_eq!(resp.acam.as_deref(), Some(ALLOW_METHODS));
    assert_eq!(resp.acah.as_deref(), Some("Content-Type"));
    assert_eq!(resp.vary.as_deref(), Some("Origin"));
    // after_request security headers apply to the preflight response too.
    assert_eq!(resp.xcto.as_deref(), Some("nosniff"));
    assert_eq!(resp.xfo.as_deref(), Some("SAMEORIGIN"));
}

#[tokio::test]
async fn preflight_without_request_headers_omits_allow_headers() {
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::OPTIONS,
        "/api/2.0/mlflow/experiments/list",
        &[
            ("host", "localhost"),
            ("origin", "http://localhost:3000"),
            ("access-control-request-method", "POST"),
        ],
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT);
    assert_eq!(resp.acao.as_deref(), Some("http://localhost:3000"));
    assert_eq!(resp.acam.as_deref(), Some(ALLOW_METHODS));
    assert_eq!(resp.acah, None);
}

#[tokio::test]
async fn preflight_disallowed_origin_no_cors_headers() {
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::OPTIONS,
        "/api/2.0/mlflow/experiments/list",
        &[
            ("host", "localhost"),
            ("origin", "http://evil.com"),
            ("access-control-request-method", "POST"),
        ],
    )
    .await;
    // flask-cors still returns 204 for a disallowed preflight, but with no
    // CORS headers (only the security headers).
    assert_eq!(resp.status, StatusCode::NO_CONTENT);
    assert_eq!(resp.acao, None);
    assert_eq!(resp.acac, None);
    assert_eq!(resp.acam, None);
    assert_eq!(resp.xcto.as_deref(), Some("nosniff"));
    assert_eq!(resp.xfo.as_deref(), Some("SAMEORIGIN"));
}

#[tokio::test]
async fn preflight_localhost_origin_always_allowed() {
    // A localhost origin not in the configured list is still allowed (the
    // localhost patterns are appended to the CORS allowlist).
    let srv = TestServer::start(config(None, Some(&["https://trusted.com"]), "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::OPTIONS,
        "/api/2.0/mlflow/experiments/list",
        &[
            ("host", "localhost"),
            ("origin", "http://127.0.0.1:9999"),
            ("access-control-request-method", "POST"),
        ],
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT);
    assert_eq!(resp.acao.as_deref(), Some("http://127.0.0.1:9999"));
}

// ---- CORS actual requests ----

#[tokio::test]
async fn actual_request_allowed_origin_headers() {
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    // GET /health (200, exempt from host validation) with an allowed origin.
    let resp = request(
        &srv.base,
        Method::GET,
        "/health",
        &[("host", "localhost"), ("origin", "http://localhost:3000")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.acao.as_deref(), Some("http://localhost:3000"));
    assert_eq!(resp.acac.as_deref(), Some("true"));
    assert_eq!(resp.vary.as_deref(), Some("Origin"));
}

#[tokio::test]
async fn actual_request_disallowed_origin_no_cors_headers() {
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::GET,
        "/health",
        &[("host", "localhost"), ("origin", "http://evil.com")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.acao, None);
    assert_eq!(resp.acac, None);
    assert_eq!(resp.vary, None);
}

// ---- Cross-origin state-change block ----

#[tokio::test]
async fn state_change_block_matrix() {
    // (method, origin, blocked?) with allowlist http://localhost:3000.
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    let cases = [
        (Method::POST, "http://evil.com", true),
        (Method::PUT, "http://evil.com", true),
        (Method::DELETE, "http://evil.com", true),
        (Method::PATCH, "http://evil.com", true),
        (Method::GET, "http://evil.com", false), // GET never blocked
        (Method::POST, "http://localhost:3000", false), // allowed origin
        (Method::POST, "http://127.0.0.1:9999", false), // localhost bypass
    ];
    for (method, origin, blocked) in cases {
        let resp = request(
            &srv.base,
            method.clone(),
            "/api/2.0/mlflow/experiments/create",
            &[("host", "localhost"), ("origin", origin)],
        )
        .await;
        if blocked {
            assert_eq!(resp.status, StatusCode::FORBIDDEN, "{method} {origin}");
            assert_eq!(resp.body, CORS_BLOCKED_MSG);
            assert_eq!(
                resp.content_type.as_deref(),
                Some("text/plain; charset=utf-8")
            );
        } else {
            assert_ne!(resp.status, StatusCode::FORBIDDEN, "{method} {origin}");
        }
    }
}

#[tokio::test]
async fn state_change_block_only_on_api_endpoints() {
    let srv = TestServer::start(config(None, Some(&["http://localhost:3000"]), "SAMEORIGIN")).await;
    // POST from a disallowed origin to a non-API path is not blocked.
    let resp = request(
        &srv.base,
        Method::POST,
        "/some/non-api/path",
        &[("host", "localhost"), ("origin", "http://evil.com")],
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn state_change_block_with_no_configured_origins_blocks_any_cross_origin() {
    // No configured origins -> any non-localhost cross-origin state change is
    // blocked (`should_block_cors_request` returns True in the fall-through).
    let srv = TestServer::start(config(None, None, "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/create",
        &[("host", "localhost"), ("origin", "http://evil.com")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(resp.body, CORS_BLOCKED_MSG);
}

// ---- Wildcard CORS ----

#[tokio::test]
async fn wildcard_cors_reflects_origin_without_credentials() {
    let srv = TestServer::start(config(None, Some(&["*"]), "SAMEORIGIN")).await;
    // Actual request: origin reflected, but no credentials header.
    let resp = request(
        &srv.base,
        Method::GET,
        "/health",
        &[("host", "localhost"), ("origin", "http://evil.com")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.acao.as_deref(), Some("http://evil.com"));
    assert_ne!(resp.acac.as_deref(), Some("true"));
}

#[tokio::test]
async fn wildcard_cors_disables_state_change_block() {
    let srv = TestServer::start(config(None, Some(&["*"]), "SAMEORIGIN")).await;
    // POST from any origin is not blocked in wildcard mode.
    let resp = request(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/create",
        &[("host", "localhost"), ("origin", "http://evil.com")],
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN);
}

// ---- X-Frame-Options ----

#[tokio::test]
async fn x_frame_options_matrix() {
    // (configured value, expected header value or None).
    let cases: [(&str, Option<&str>); 5] = [
        ("SAMEORIGIN", Some("SAMEORIGIN")),
        ("DENY", Some("DENY")),
        ("deny", Some("DENY")), // uppercased
        ("NONE", None),         // disabled
        ("none", None),         // disabled (case-insensitive)
    ];
    for (configured, expected) in cases {
        let srv = TestServer::start(config(Some(&["*"]), None, configured)).await;
        let resp = request(&srv.base, Method::GET, "/health", &[("host", "localhost")]).await;
        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(resp.xfo.as_deref(), expected, "configured={configured}");
        // X-Content-Type-Options is always present regardless of XFO.
        assert_eq!(resp.xcto.as_deref(), Some("nosniff"));
    }
}

// ---- Security headers on error/404 responses ----

#[tokio::test]
async fn security_headers_on_404() {
    let srv = TestServer::start(config(Some(&["*"]), None, "SAMEORIGIN")).await;
    let resp = request(
        &srv.base,
        Method::GET,
        "/does-not-exist",
        &[("host", "localhost")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
    assert_eq!(resp.xcto.as_deref(), Some("nosniff"));
    assert_eq!(resp.xfo.as_deref(), Some("SAMEORIGIN"));
}

#[tokio::test]
async fn security_headers_on_host_rejection() {
    let srv = TestServer::start(config(Some(&["localhost"]), None, "SAMEORIGIN")).await;
    let resp = request(&srv.base, Method::GET, "/metrics", &[("host", "evil.com")]).await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(resp.body, INVALID_HOST_MSG);
    // after_request decorates even the rejection.
    assert_eq!(resp.xcto.as_deref(), Some("nosniff"));
    assert_eq!(resp.xfo.as_deref(), Some("SAMEORIGIN"));
    // No CORS headers on the rejection.
    assert_eq!(resp.acao, None);
}

// ---- Host rejection wins over CORS (ordering) ----

#[tokio::test]
async fn host_rejection_precedes_cors_and_wins_even_with_allowed_origin() {
    // A disallowed Host is rejected even when the Origin would be CORS-allowed:
    // host validation runs first (Flask `before_request` ordering), so the 403
    // body is the Invalid-Host message, NOT a CORS block. The `after_request`
    // CORS decoration still runs on the rejection (flask-cors parity), so the
    // allowed origin's CORS headers ARE present on the 403.
    let srv = TestServer::start(config(
        Some(&["localhost"]),
        Some(&["http://good.com"]),
        "SAMEORIGIN",
    ))
    .await;
    let resp = request(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/create",
        &[("host", "evil.com"), ("origin", "http://good.com")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(resp.body, INVALID_HOST_MSG);
    // Rejection body is Invalid-Host (not the CORS-block message), proving host
    // validation ran and won over the state-change block.
    assert_ne!(resp.body, CORS_BLOCKED_MSG);
    // flask-cors after_request still decorates the rejection for the allowed
    // origin.
    assert_eq!(resp.acao.as_deref(), Some("http://good.com"));
    assert_eq!(resp.acac.as_deref(), Some("true"));
}
