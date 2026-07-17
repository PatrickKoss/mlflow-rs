//! Webhook input validation, mirroring `_validate_webhook_name`,
//! `_validate_webhook_url`, `_validate_webhook_events`
//! (`mlflow/utils/validation.py:867-947`) and the `WebhookEvent.__init__`
//! entity/action combination check (`mlflow/entities/webhook.py:172`).
//!
//! ## URL validation and SSRF
//!
//! Python's `_validate_webhook_url` (a) checks the scheme against
//! `MLFLOW_WEBHOOK_ALLOWED_SCHEMES` (default `["https"]`), (b) requires a
//! hostname, and (c) unless `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` is set, resolves
//! the hostname and rejects any non-public (`ip.is_global == False`) resolved
//! address. We port (a) and (b) exactly. For (c) — the public-IP resolution
//! gate that doubles as the test-path SSRF guard — see [`resolve_public_ips`].

use std::net::{IpAddr, ToSocketAddrs};

use mlflow_error::MlflowError;

use crate::entities::{WebhookAction, WebhookEntity, WebhookEvent};

/// `MLFLOW_WEBHOOK_ALLOWED_SCHEMES` (default `["https"]`,
/// `mlflow/environment_variables.py:1361`). Comma-separated, stripped.
const ALLOWED_SCHEMES_ENV: &str = "MLFLOW_WEBHOOK_ALLOWED_SCHEMES";
/// `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` (default `false`,
/// `mlflow/environment_variables.py:1395`).
const ALLOW_PRIVATE_IPS_ENV: &str = "MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS";

/// The allowed URL schemes from the environment (default `["https"]`).
fn allowed_schemes() -> Vec<String> {
    match std::env::var(ALLOWED_SCHEMES_ENV) {
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => vec!["https".to_string()],
    }
}

/// `_MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS.get()` — truthy env values.
pub fn allow_private_ips() -> bool {
    matches!(
        std::env::var(ALLOW_PRIVATE_IPS_ENV).ok().as_deref(),
        Some("true" | "True" | "TRUE" | "1")
    )
}

/// `_validate_webhook_name(name)`: 1-63 chars, must start and end with a letter
/// or digit, and contain only letters, digits, `.`, `_`, `-` (case-insensitive,
/// matching `_WEBHOOK_NAME_REGEX`).
pub fn validate_webhook_name(name: &str) -> Result<(), MlflowError> {
    if is_valid_webhook_name(name) {
        return Ok(());
    }
    Err(MlflowError::invalid_parameter_value(format!(
        "Webhook name {} is invalid. It must start and end with a letter or digit, \
         be less than 63 characters long, and contain only letters, digits, dots (.), \
         underscores (_), and hyphens (-).",
        py_repr(name)
    )))
}

fn is_valid_webhook_name(name: &str) -> bool {
    // `^(?=.{1,63}$)[a-z0-9]([a-z0-9._-]*[a-z0-9])?$` (case-insensitive).
    // The regex operates on characters; the lookahead bounds the character
    // count to 1..=63.
    let chars: Vec<char> = name.chars().collect();
    if chars.is_empty() || chars.len() > 63 {
        return false;
    }
    let is_alnum = |c: char| c.is_ascii_alphanumeric();
    let is_mid = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
    if !is_alnum(chars[0]) {
        return false;
    }
    if chars.len() == 1 {
        return true;
    }
    if !is_alnum(*chars.last().unwrap()) {
        return false;
    }
    chars[1..chars.len() - 1].iter().all(|&c| is_mid(c))
}

/// `_validate_webhook_url(url)`: non-empty, allowed scheme, has a hostname, and
/// (unless private IPs are allowed) resolves only to public IPs.
pub fn validate_webhook_url(url: &str) -> Result<(), MlflowError> {
    if url.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Webhook URL cannot be empty or just whitespace: {}",
            py_repr(url)
        )));
    }

    let parsed = ParsedUrl::parse(url);
    let schemes = allowed_schemes();
    if !schemes.contains(&parsed.scheme) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid webhook URL scheme: {}. Allowed schemes are: {}.",
            py_repr(&parsed.scheme),
            schemes.join(", ")
        )));
    }

    let Some(hostname) = parsed.hostname.filter(|h| !h.is_empty()) else {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Webhook URL must include a hostname: {}",
            py_repr(url)
        )));
    };

    if !allow_private_ips() {
        resolve_public_ips(&hostname)?;
    }
    Ok(())
}

/// Resolve `hostname` and error (matching `_validate_webhook_url`'s
/// non-public-IP branch) if it fails to resolve or resolves to any non-global
/// address. Returns the resolved public IPs on success.
///
/// This is the connect-time SSRF gate for the `/test` path. Python's test path
/// (`test_webhook` -> `_send_webhook_request` -> `_validate_webhook_url` +
/// `SSRFProtectedHTTPAdapter`) both re-validates the URL and checks the peer IP
/// at connect time; the full connect-time TOCTOU adapter is T8.3. Here we apply
/// the same resolve-and-check the create/update path already ran, which is the
/// conservative behavior: a hostname that resolves to a private IP is rejected
/// before any request is sent.
pub fn resolve_public_ips(hostname: &str) -> Result<Vec<IpAddr>, MlflowError> {
    // `getaddrinfo(hostname, None)` — resolve with a throwaway port.
    let addrs = (hostname, 0u16).to_socket_addrs().map_err(|e| {
        MlflowError::invalid_parameter_value(format!(
            "Cannot resolve webhook URL hostname {}: {e}",
            py_repr(hostname)
        ))
    })?;
    let mut ips = Vec::new();
    for addr in addrs {
        let ip = addr.ip();
        if !is_global_ip(ip) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Webhook URL must not resolve to a non-public IP address. \
                 {} resolves to {ip}.",
                py_repr(hostname)
            )));
        }
        ips.push(ip);
    }
    Ok(ips)
}

/// `ipaddress.ip_address(...).is_global` for the address families we resolve.
/// Conservative: anything not clearly global (loopback, private, link-local,
/// multicast, unspecified, documentation/benchmark ranges, IPv6 ULA, etc.) is
/// treated as non-global.
///
/// Public so the connect-time SSRF gate in [`crate::http_send`] applies the
/// exact same globality test the resolve-time `_validate_webhook_url` gate uses,
/// mirroring how `ssrf.py` reuses `ipaddress`'s `is_global`.
pub fn is_global_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            if v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || v4.is_multicast()
            {
                return false;
            }
            let o = v4.octets();
            // 100.64.0.0/10 (CGNAT, not global per Python's is_global).
            if o[0] == 100 && (64..=127).contains(&o[1]) {
                return false;
            }
            // 192.0.0.0/24 (IETF protocol assignments) and 198.18.0.0/15
            // (benchmarking) are not global.
            if o[0] == 192 && o[1] == 0 && o[2] == 0 {
                return false;
            }
            if o[0] == 198 && (o[1] == 18 || o[1] == 19) {
                return false;
            }
            // 240.0.0.0/4 reserved.
            if o[0] >= 240 {
                return false;
            }
            true
        }
        IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
                return false;
            }
            let seg0 = v6.segments()[0];
            // fc00::/7 unique-local, fe80::/10 link-local.
            if (seg0 & 0xfe00) == 0xfc00 || (seg0 & 0xffc0) == 0xfe80 {
                return false;
            }
            // Map IPv4-mapped addresses to their v4 global check.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_global_ip(IpAddr::V4(v4));
            }
            true
        }
    }
}

/// `_validate_webhook_events(events)`: the list must be non-empty.
pub fn validate_webhook_events(events: &[WebhookEvent]) -> Result<(), MlflowError> {
    if events.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Webhook events must be a non-empty list of WebhookEvent objects: [].".to_string(),
        ));
    }
    Ok(())
}

/// `WebhookEvent.__init__`'s combination check
/// (`VALID_ENTITY_ACTIONS`, `mlflow/entities/webhook.py:109`): reject an
/// `(entity, action)` pair that is not a valid subscription, with the same
/// message (valid actions sorted by their lowercase value).
pub fn validate_event_combination(
    entity: WebhookEntity,
    action: WebhookAction,
) -> Result<(), MlflowError> {
    let valid = valid_actions(entity);
    if valid.contains(&action) {
        return Ok(());
    }
    let mut names: Vec<&str> = valid.iter().map(|a| a.as_db_str()).collect();
    names.sort_unstable();
    let list = names
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(MlflowError::invalid_parameter_value(format!(
        "Invalid action '{}' for entity '{}'. Valid actions are: [{}]",
        action.as_db_str(),
        entity.as_db_str(),
        list
    )))
}

/// `VALID_ENTITY_ACTIONS[entity]`.
fn valid_actions(entity: WebhookEntity) -> Vec<WebhookAction> {
    use WebhookAction::*;
    use WebhookEntity::*;
    match entity {
        RegisteredModel => vec![Created],
        ModelVersion => vec![Created],
        ModelVersionTag => vec![Set, Deleted],
        ModelVersionAlias => vec![Created, Deleted],
        Prompt => vec![Created],
        PromptVersion => vec![Created],
        PromptTag => vec![Set, Deleted],
        PromptVersionTag => vec![Set, Deleted],
        PromptAlias => vec![Created, Deleted],
        BudgetPolicy => vec![Exceeded],
    }
}

/// A minimally-parsed URL: scheme + hostname, matching what
/// `urllib.parse.urlparse` exposes for validation (`scheme`, `hostname`).
struct ParsedUrl {
    scheme: String,
    hostname: Option<String>,
}

impl ParsedUrl {
    /// Parse `scheme://[userinfo@]host[:port]/...`. Only the pieces the Python
    /// validation reads are extracted. `hostname` is lowercased and IPv6
    /// brackets are stripped, as `urllib`'s `.hostname` does.
    fn parse(url: &str) -> Self {
        let (scheme, rest) = match url.split_once("://") {
            Some((s, r)) => (s.to_ascii_lowercase(), r),
            None => {
                // No `scheme://` — urllib would yield an empty scheme; the
                // scheme check then fails with the empty-scheme message.
                return ParsedUrl {
                    scheme: String::new(),
                    hostname: None,
                };
            }
        };
        // Authority is up to the first '/', '?', or '#'.
        let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..authority_end];
        // Strip userinfo.
        let host_port = match authority.rsplit_once('@') {
            Some((_userinfo, hp)) => hp,
            None => authority,
        };
        let hostname = if let Some(stripped) = host_port.strip_prefix('[') {
            // IPv6 literal: `[::1]:port`.
            stripped
                .split_once(']')
                .map(|(h, _)| h.to_ascii_lowercase())
        } else {
            host_port
                .split_once(':')
                .map(|(h, _)| h)
                .or(Some(host_port))
                .filter(|h| !h.is_empty())
                .map(|h| h.to_ascii_lowercase())
        };
        ParsedUrl { scheme, hostname }
    }
}

/// Python `repr()` of a string (single-quoted unless it contains a single
/// quote and no double quote), for byte-matching validation messages that
/// embed `{url!r}` / `{name!r}`.
fn py_repr(s: &str) -> String {
    let use_double = s.contains('\'') && !s.contains('"');
    let quote = if use_double { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names() {
        for n in ["a", "webhook-1", "my.hook_name", "A1", "abc123"] {
            assert!(validate_webhook_name(n).is_ok(), "{n} should be valid");
        }
    }

    #[test]
    fn invalid_names() {
        for n in ["", "-abc", "abc-", ".x", "a b", &"x".repeat(64)] {
            assert!(validate_webhook_name(n).is_err(), "{n:?} should be invalid");
        }
    }

    #[test]
    fn scheme_check_default_https_only() {
        // Default allowed scheme is https; http is rejected before DNS.
        let err = validate_webhook_url("http://example.com/hook").unwrap_err();
        assert!(err.message.contains("Invalid webhook URL scheme"));
    }

    #[test]
    fn empty_url_rejected() {
        assert!(validate_webhook_url("   ").is_err());
    }

    #[test]
    fn parses_hostname_and_scheme() {
        let p = ParsedUrl::parse("https://user:pw@Example.COM:8443/path?q=1");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.hostname.as_deref(), Some("example.com"));
    }

    #[test]
    fn event_combination_rules() {
        assert!(
            validate_event_combination(WebhookEntity::RegisteredModel, WebhookAction::Created)
                .is_ok()
        );
        let err =
            validate_event_combination(WebhookEntity::RegisteredModel, WebhookAction::Deleted)
                .unwrap_err();
        assert!(err.message.contains("Invalid action 'deleted'"));
        assert!(err.message.contains("['created']"));
    }
}
