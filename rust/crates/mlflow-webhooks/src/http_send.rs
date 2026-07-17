//! Connect-time SSRF-guarded HTTP delivery for webhooks, porting
//! `mlflow/webhooks/ssrf.py` (the `SSRFProtectedHTTPAdapter` connection-time
//! peer-IP check) and the retry/redirect behavior of the `requests.Session`
//! configured in `mlflow/webhooks/delivery.py:67-102`.
//!
//! ## Why a hand-rolled sender instead of `reqwest`/hyper's high-level client
//!
//! Python's SSRF protection (`ssrf.py`) subclasses urllib3's connection pool so
//! the peer IP of the *actual connected socket* is validated immediately after
//! `connect()` returns, before any TLS/HTTP bytes are exchanged. That closes the
//! DNS-rebinding TOCTOU gap: `_validate_webhook_url` resolves + checks the
//! hostname, but a naive `requests.post` re-resolves independently, so a
//! rebinding attacker could return a public IP at validation and
//! `169.254.169.254` at request time.
//!
//! To reproduce that in Rust we do the resolution ourselves ([`Resolver`]),
//! reject any non-global resolved IP up front, then connect a raw `TcpStream`
//! **to a validated IP** (no second, independent DNS lookup) and re-check the
//! socket's real peer address via `getpeername`. Because the same validated IP
//! is both checked and connected, rebinding has nothing to exploit. This mirrors
//! `ssrf.py`'s `_assert_public_peer(sock)` running on the post-`connect()`
//! socket.
//!
//! ## No proxy env
//!
//! `delivery.py:98` sets `session.trust_env = False` so a configured proxy can't
//! become the validated peer (bypassing the destination check). We connect
//! directly to the resolved destination IP and never consult proxy env vars.
//!
//! ## Redirects
//!
//! `requests` follows redirects by default, and `ssrf.py` mounts the SSRF
//! adapter on the session so **every** redirect hop's connection is peer-IP
//! validated. We follow redirects (bounded by [`MAX_REDIRECTS`]) and re-run the
//! full resolve + connect-time SSRF gate on each hop, so a redirect to a private
//! IP fails closed exactly as it would in Python.
//!
//! ## Escape hatch
//!
//! `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` (via [`crate::validation::allow_private_ips`])
//! disables the peer-IP gate, matching `ssrf.py`'s `_assert_public_peer` early
//! return. Dev servers set it to deliver to `localhost`; tests use it to target
//! a local listener.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::header::{CONTENT_TYPE, HOST, LOCATION, RETRY_AFTER};
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;

use crate::validation::{allow_private_ips, is_global_ip};

/// `requests`' default redirect cap (`urllib3`'s `Retry(redirect=...)` / the
/// session's 30-hop `resolve_redirects` limit). We use the same default 30.
const MAX_REDIRECTS: usize = 30;

/// A resolved HTTP response: status plus the fully-read body. Mirrors the slice
/// of `requests.Response` the delivery/test paths read (`status_code`, `text`).
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
    /// The parsed `Retry-After` header (delta-seconds form only), used to honor
    /// `respect_retry_after_header=True` on 429/503 retries.
    pub(crate) retry_after: Option<Duration>,
}

/// Errors from a webhook send attempt. [`SendError::Ssrf`] is never retried and
/// fails closed, mirroring `ssrf.py`'s `SSRFProtectionError` (not a urllib3
/// exception type, so urllib3 never retries it).
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// A resolved / connected peer IP was not public (DNS-rebinding guard), or
    /// the connection peer could not be determined. Fails closed, not retried.
    #[error("{0}")]
    Ssrf(String),
    /// Malformed URL (no host, unparseable, unsupported scheme).
    #[error("{0}")]
    InvalidUrl(String),
    /// Too many redirect hops.
    #[error("exceeded {MAX_REDIRECTS} redirects")]
    TooManyRedirects,
    /// A transport/IO/protocol error (connect refused, reset, timeout, …).
    #[error("{0}")]
    Transport(String),
}

/// A hostname → IP resolver, extracted as a seam so the SSRF matrix can be
/// tested without real DNS. The default implementation ([`SystemResolver`]) uses
/// the OS resolver (`getaddrinfo`), exactly like `socket.getaddrinfo` /
/// urllib3's `create_connection`.
pub trait Resolver: Send + Sync {
    /// Resolve `host` to one or more IPs (any address family). Returning an
    /// empty vec is treated as "host did not resolve".
    fn resolve(&self, host: &str) -> Result<Vec<IpAddr>, String>;
}

/// The default OS resolver.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemResolver;

impl Resolver for SystemResolver {
    fn resolve(&self, host: &str) -> Result<Vec<IpAddr>, String> {
        use std::net::ToSocketAddrs;
        // Resolve with a throwaway port; discard it, we only need the IPs.
        let addrs = (host, 0u16)
            .to_socket_addrs()
            .map_err(|e| format!("cannot resolve {host:?}: {e}"))?;
        Ok(addrs.map(|a| a.ip()).collect())
    }
}

/// A parsed webhook target: scheme, host, port, and request path+query.
struct Target {
    scheme: Scheme,
    host: String,
    port: u16,
    /// The `path[?query]` for the HTTP request line (always begins with `/`).
    request_target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scheme {
    Http,
    Https,
}

impl Target {
    fn parse(url: &str) -> Result<Self, SendError> {
        let uri: hyper::Uri = url
            .parse()
            .map_err(|e| SendError::InvalidUrl(format!("invalid webhook URL: {e}")))?;
        let scheme = match uri.scheme_str() {
            Some("http") => Scheme::Http,
            Some("https") => Scheme::Https,
            other => {
                return Err(SendError::InvalidUrl(format!(
                    "unsupported webhook URL scheme: {other:?}"
                )));
            }
        };
        let host = uri
            .host()
            .ok_or_else(|| SendError::InvalidUrl("webhook URL has no host".to_string()))?
            .to_string();
        let port = uri.port_u16().unwrap_or(match scheme {
            Scheme::Http => 80,
            Scheme::Https => 443,
        });
        let request_target = match uri.path_and_query() {
            Some(pq) => pq.as_str().to_string(),
            None => "/".to_string(),
        };
        Ok(Target {
            scheme,
            host,
            port,
            request_target,
        })
    }

    /// The `Host` header value (`host[:port]`, omitting the default port).
    fn host_header(&self) -> String {
        let default = matches!(
            (self.scheme, self.port),
            (Scheme::Http, 80) | (Scheme::Https, 443)
        );
        if default {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// A single outbound webhook POST, already fully assembled: the target URL, the
/// serialized JSON body, and the `X-MLflow-*` headers (delivery id, timestamp,
/// and — when signed — signature). This is what both the async delivery engine
/// and the `/test` path hand to [`send_with_ssrf_guard`].
#[derive(Debug, Clone)]
pub struct SignedRequest {
    pub url: String,
    pub body: String,
    /// `(name, value)` header pairs beyond `Content-Type`/`Host` (which the
    /// sender always sets). Order-preserving to keep parity with Python.
    pub headers: Vec<(&'static str, String)>,
}

/// The tuning knobs the sender needs, resolved from the environment once and
/// passed in so the hot path does no env lookups. Mirrors
/// `MLFLOW_WEBHOOK_REQUEST_TIMEOUT` and the `Retry(...)` config in `delivery.py`.
#[derive(Debug, Clone, Copy)]
pub struct SendConfig {
    /// `MLFLOW_WEBHOOK_REQUEST_TIMEOUT` (default 30s) — per-attempt wall clock.
    pub timeout: Duration,
    /// `Retry.total` (`MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES`, default 3) — the
    /// number of *retries* after the first attempt.
    pub max_retries: u32,
    /// `Retry.backoff_factor` (1.0). Backoff for retry `n` (1-indexed) is
    /// `backoff_factor * 2^(n-1)` seconds, capped at `backoff_max`.
    pub backoff_factor: f64,
    /// `Retry.backoff_max` (60s).
    pub backoff_max: Duration,
    /// `Retry.backoff_jitter` (1.0, urllib3 >= 2.0): add up to this many seconds
    /// of uniform jitter to each backoff.
    pub backoff_jitter: f64,
    /// `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` — the SSRF escape hatch. When true the
    /// connect-time peer-IP gate is disabled (`ssrf.py`'s `_assert_public_peer`
    /// early return). Resolved from env in [`SendConfig::from_env`]; injected
    /// directly in tests so the SSRF matrix runs deterministically in parallel
    /// without mutating a process-global env var.
    pub allow_private_ips: bool,
}

impl SendConfig {
    /// Resolve the config from the same env vars Python reads
    /// (`delivery.py:73-89`, `environment_variables.py:1373-1389`), including
    /// the private-IP escape hatch.
    pub fn from_env() -> Self {
        Self {
            timeout: Duration::from_secs(env_u64("MLFLOW_WEBHOOK_REQUEST_TIMEOUT", 30)),
            max_retries: env_u64("MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES", 3) as u32,
            backoff_factor: 1.0,
            backoff_max: Duration::from_secs_f64(60.0),
            backoff_jitter: 1.0,
            allow_private_ips: allow_private_ips(),
        }
    }
}

/// `Retry.status_forcelist` (`delivery.py:82`): retry only on these statuses.
const RETRYABLE_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Send `req` with the connect-time SSRF guard, per-attempt timeout, redirect
/// following (each hop re-validated), and retry-on-status backoff — the Rust
/// analogue of `session.post(...)` through the `SSRFProtectedHTTPAdapter` with
/// the configured `Retry` strategy.
///
/// A [`SendError::Ssrf`] is never retried (fails closed). A retryable status is
/// retried up to `config.max_retries` times with exponential backoff (honoring a
/// `Retry-After` header on 429/503, like `respect_retry_after_header=True`).
pub async fn send_with_ssrf_guard(
    req: &SignedRequest,
    config: SendConfig,
    resolver: Arc<dyn Resolver>,
) -> Result<HttpResponse, SendError> {
    let mut attempt: u32 = 0;
    loop {
        let result = send_once_following_redirects(req, config, resolver.as_ref()).await;

        match &result {
            Ok(resp)
                if RETRYABLE_STATUSES.contains(&resp.status) && attempt < config.max_retries =>
            {
                backoff(config, attempt, resp.retry_after).await;
                attempt += 1;
                continue;
            }
            // Transport errors are retried too (urllib3's `Retry.connect`/`read`
            // default to `total`), matching Python retrying connection failures.
            Err(SendError::Transport(_)) if attempt < config.max_retries => {
                backoff(config, attempt, None).await;
                attempt += 1;
                continue;
            }
            _ => return result,
        }
    }
}

/// One logical send: follow up to [`MAX_REDIRECTS`] redirects, SSRF-validating
/// every hop, returning the final (non-redirect) response.
async fn send_once_following_redirects(
    req: &SignedRequest,
    config: SendConfig,
    resolver: &dyn Resolver,
) -> Result<HttpResponse, SendError> {
    let mut current_url = req.url.clone();
    for _ in 0..=MAX_REDIRECTS {
        let (response, location) = send_single_hop(&current_url, req, config, resolver).await?;
        match location {
            Some(next) if is_redirect_status(response.status) => {
                current_url = resolve_redirect_url(&current_url, &next)?;
            }
            _ => return Ok(response),
        }
    }
    Err(SendError::TooManyRedirects)
}

/// A single request/response with no redirect following: resolve the host, run
/// the connect-time SSRF gate, connect to a validated IP, send the POST, and
/// return the response plus any `Location` header.
async fn send_single_hop(
    url: &str,
    req: &SignedRequest,
    config: SendConfig,
    resolver: &dyn Resolver,
) -> Result<(HttpResponse, Option<String>), SendError> {
    let target = Target::parse(url)?;
    if target.scheme == Scheme::Https {
        // TLS is out of scope for the low-level sender: the same limitation the
        // T8.2 `/test` path documents. The SSRF gate and full engine are what
        // T8.3 delivers; wiring a TLS stack (`hyper-rustls`) is a follow-up.
        // Fail closed with a clear error rather than silently downgrading.
        return Err(SendError::Transport(
            "https webhook delivery requires a TLS-enabled build (not yet wired); \
             see http_send.rs"
                .to_string(),
        ));
    }

    let ip = resolve_and_validate_peer(&target.host, resolver, config.allow_private_ips)?;
    let addr = SocketAddr::new(ip, target.port);

    let response = tokio::time::timeout(config.timeout, async {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| SendError::Transport(format!("connect to {addr} failed: {e}")))?;
        // Re-validate the *actual* connected peer (getpeername), closing any gap
        // between the validated resolution and the socket — `ssrf.py`'s
        // `_assert_public_peer(sock)` on the post-connect socket.
        assert_public_peer(&stream, config.allow_private_ips)?;
        send_on_stream(stream, &target, req).await
    })
    .await
    .map_err(|_| {
        SendError::Transport(format!(
            "webhook request timed out after {}s",
            config.timeout.as_secs()
        ))
    })??;

    Ok(response)
}

/// Resolve `host` and reject any non-global resolved IP (the up-front half of
/// the SSRF gate), returning the first validated IP to connect to. When
/// `allow_private_ips` is set (the `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` escape
/// hatch) the check is skipped (dev/test).
fn resolve_and_validate_peer(
    host: &str,
    resolver: &dyn Resolver,
    allow_private_ips: bool,
) -> Result<IpAddr, SendError> {
    // A bare IP literal skips DNS but still goes through the peer check below.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if !allow_private_ips && !is_global_ip(ip) {
            return Err(SendError::Ssrf(format!(
                "Webhook connection blocked: {ip} is not a public IP address."
            )));
        }
        return Ok(ip);
    }

    let ips = resolver.resolve(host).map_err(SendError::Transport)?;
    let first = ips
        .first()
        .copied()
        .ok_or_else(|| SendError::Transport(format!("host {host:?} did not resolve")))?;

    if allow_private_ips {
        return Ok(first);
    }
    // Every resolved IP must be global (a rebinding response mixing a public and
    // a private IP is rejected), matching `_validate_webhook_url`'s "all resolved
    // addresses must be global" and `ssrf.py`'s per-connection check.
    for ip in &ips {
        if !is_global_ip(*ip) {
            return Err(SendError::Ssrf(format!(
                "Webhook connection blocked: {host} resolves to {ip}, which is not a \
                 public IP address. This may indicate a DNS rebinding attempt."
            )));
        }
    }
    Ok(first)
}

/// `ssrf.py`'s `_assert_public_peer(sock)`: check the socket's real peer address
/// (`getpeername`) is global. Fails closed on any error.
fn assert_public_peer(stream: &TcpStream, allow_private_ips: bool) -> Result<(), SendError> {
    if allow_private_ips {
        return Ok(());
    }
    let peer = stream.peer_addr().map_err(|e| {
        SendError::Ssrf(format!(
            "Could not determine webhook connection peer address: {e}"
        ))
    })?;
    if !is_global_ip(peer.ip()) {
        return Err(SendError::Ssrf(format!(
            "Webhook connection blocked: {} is not a public IP address. \
             This may indicate a DNS rebinding attempt.",
            peer.ip()
        )));
    }
    Ok(())
}

/// Perform the HTTP/1.1 request over an already-connected, already-validated
/// stream and read the full response body.
async fn send_on_stream(
    stream: TcpStream,
    target: &Target,
    req: &SignedRequest,
) -> Result<(HttpResponse, Option<String>), SendError> {
    let io = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, Full<Bytes>>(io)
        .await
        .map_err(|e| SendError::Transport(format!("http handshake failed: {e}")))?;

    // Drive the connection in the background; it completes when the response is
    // fully read and the stream closes.
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let mut builder = Request::builder()
        .method("POST")
        .uri(&target.request_target)
        .header(HOST, target.host_header())
        .header(CONTENT_TYPE, "application/json");
    for (name, value) in &req.headers {
        builder = builder.header(*name, value);
    }
    let request = builder
        .body(Full::new(Bytes::from(req.body.clone())))
        .map_err(|e| SendError::Transport(format!("failed to build request: {e}")))?;

    let response = sender
        .send_request(request)
        .await
        .map_err(|e| SendError::Transport(format!("request failed: {e}")))?;

    let status = response.status().as_u16();
    let location = response
        .headers()
        .get(LOCATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    let body_bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|e| SendError::Transport(format!("failed to read response body: {e}")))?
        .to_bytes();
    let body = String::from_utf8_lossy(&body_bytes).into_owned();
    Ok((
        HttpResponse {
            status,
            body,
            retry_after,
        },
        location,
    ))
}

fn is_redirect_status(status: u16) -> bool {
    matches!(
        StatusCode::from_u16(status).ok(),
        Some(
            StatusCode::MOVED_PERMANENTLY
                | StatusCode::FOUND
                | StatusCode::SEE_OTHER
                | StatusCode::TEMPORARY_REDIRECT
                | StatusCode::PERMANENT_REDIRECT
        )
    )
}

/// Resolve a possibly-relative `Location` against the current URL.
fn resolve_redirect_url(current: &str, location: &str) -> Result<String, SendError> {
    if location.contains("://") {
        return Ok(location.to_string());
    }
    // Relative redirect: reuse scheme+authority from the current URL.
    let cur: hyper::Uri = current
        .parse()
        .map_err(|e| SendError::InvalidUrl(format!("invalid current URL: {e}")))?;
    let scheme = cur.scheme_str().unwrap_or("http");
    let authority = cur
        .authority()
        .map(|a| a.as_str().to_string())
        .ok_or_else(|| SendError::InvalidUrl("current URL has no authority".to_string()))?;
    if let Some(abs_path) = location.strip_prefix('/') {
        Ok(format!("{scheme}://{authority}/{abs_path}"))
    } else {
        Ok(format!("{scheme}://{authority}/{location}"))
    }
}

/// Sleep the backoff for retry `attempt` (0-indexed: the wait *before* the
/// `attempt+1`-th retry). `backoff = min(backoff_factor * 2^attempt, backoff_max)
/// + U(0, backoff_jitter)`, or the `Retry-After` value when present — matching
/// urllib3's `Retry.get_backoff_time` + `backoff_jitter` and
/// `respect_retry_after_header`.
async fn backoff(config: SendConfig, attempt: u32, retry_after: Option<Duration>) {
    let wait = retry_after.unwrap_or_else(|| backoff_duration(config, attempt));
    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }
}

/// Pure backoff computation (unit-tested): the deterministic exponential term
/// plus uniform jitter in `[0, backoff_jitter)`.
fn backoff_duration(config: SendConfig, attempt: u32) -> Duration {
    let base = config.backoff_factor * 2f64.powi(attempt as i32);
    let capped = base.min(config.backoff_max.as_secs_f64());
    let jitter = if config.backoff_jitter > 0.0 {
        unit_random() * config.backoff_jitter
    } else {
        0.0
    };
    Duration::from_secs_f64(capped + jitter)
}

/// A cheap uniform `[0, 1)` sample for backoff jitter. Jitter only needs to
/// de-synchronize retries, not cryptographic quality, so we derive it from the
/// wall clock's sub-second nanos rather than pulling in a `rand` dependency.
fn unit_random() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    f64::from(nanos) / 1_000_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_target_defaults() {
        let t = Target::parse("http://example.com/hook?x=1").unwrap();
        assert_eq!(t.host, "example.com");
        assert_eq!(t.port, 80);
        assert_eq!(t.request_target, "/hook?x=1");
        assert_eq!(t.host_header(), "example.com");
    }

    #[test]
    fn parses_explicit_port_into_host_header() {
        let t = Target::parse("http://example.com:8080/").unwrap();
        assert_eq!(t.port, 8080);
        assert_eq!(t.host_header(), "example.com:8080");
    }

    #[test]
    fn rejects_unsupported_scheme() {
        assert!(matches!(
            Target::parse("ftp://example.com/x"),
            Err(SendError::InvalidUrl(_))
        ));
    }

    #[test]
    fn redirect_status_detection() {
        assert!(is_redirect_status(301));
        assert!(is_redirect_status(302));
        assert!(is_redirect_status(307));
        assert!(is_redirect_status(308));
        assert!(!is_redirect_status(200));
        assert!(!is_redirect_status(404));
    }

    #[test]
    fn relative_redirect_resolution() {
        let u = resolve_redirect_url("http://host:5000/a/b", "/c/d").unwrap();
        assert_eq!(u, "http://host:5000/c/d");
        let u = resolve_redirect_url("http://host/a", "http://other/x").unwrap();
        assert_eq!(u, "http://other/x");
    }

    #[test]
    fn backoff_is_exponential_and_capped() {
        let cfg = SendConfig {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            backoff_factor: 1.0,
            backoff_max: Duration::from_secs(60),
            backoff_jitter: 0.0,
            allow_private_ips: false,
        };
        assert_eq!(backoff_duration(cfg, 0).as_secs_f64(), 1.0);
        assert_eq!(backoff_duration(cfg, 1).as_secs_f64(), 2.0);
        assert_eq!(backoff_duration(cfg, 2).as_secs_f64(), 4.0);
        // 2^10 = 1024 > 60 → capped.
        assert_eq!(backoff_duration(cfg, 10).as_secs_f64(), 60.0);
    }

    #[test]
    fn ip_literal_peer_gate_blocks_private() {
        // With the escape hatch off, a private IP literal is rejected up front.
        let resolver = SystemResolver;
        // 10.0.0.1 is RFC1918 → not global.
        let err = resolve_and_validate_peer("10.0.0.1", &resolver, false);
        assert!(matches!(err, Err(SendError::Ssrf(_))));
        // The escape hatch lets it through.
        assert!(resolve_and_validate_peer("10.0.0.1", &resolver, true).is_ok());
    }
}
