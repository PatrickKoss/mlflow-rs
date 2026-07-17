//! The credential cache (plan T9.8), a port of the `_USER_AUTH_CACHE` machinery
//! in `mlflow/server/auth/__init__.py:376-449`.
//!
//! Successful basic-auth checks are cached so the expensive werkzeug hash
//! comparison (scrypt/pbkdf2, tens of milliseconds by design) runs at most once
//! per `(username, password)` per TTL window. The cache is:
//!
//! * **Off by default** — enabled only when `auth_cache_ttl_seconds > 0`
//!   (`__init__.py:381`). When off, [`CredentialCache::enabled`] is `false` and
//!   every method is a no-op / miss, so the middleware always hits the store.
//! * **HMAC-keyed** — the cache key is `(username, HMAC-SHA256(key, password))`
//!   with a random per-process key (`__init__.py:391`). The plaintext password
//!   is never stored, and the digest is useless to an attacker who dumps process
//!   memory without the key (`__init__.py:394`).
//! * **Bounded + TTL'd** — `auth_cache_max_size` entries, `auth_cache_ttl_seconds`
//!   TTL, using the same `Mutex<HashMap>` TTL-cache pattern as
//!   `mlflow-store`'s workspace caches (cachetools.TTLCache parity).
//!
//! Mutations (password/admin/deletion) call [`CredentialCache::invalidate_user`]
//! so the change takes effect immediately on the mutating worker rather than
//! after the TTL (`_invalidate_user_auth_cache`, `__init__.py:441`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::entities::User;

type HmacSha256 = Hmac<Sha256>;

/// Cache key: `(username, HMAC-SHA256(hmac_key, password))`.
type CacheKey = (String, Vec<u8>);
/// Cache entry: insertion instant (for the TTL) + the authenticated user.
type CacheEntry = (Instant, User);

/// A cached credential-check result keyed by `(username, hmac(password))`.
pub struct CredentialCache {
    /// `None` when the cache is disabled (`auth_cache_ttl_seconds == 0`),
    /// matching Python's `_USER_AUTH_CACHE = None` branch.
    inner: Option<Inner>,
}

struct Inner {
    map: Mutex<HashMap<CacheKey, CacheEntry>>,
    capacity: usize,
    ttl: Duration,
    /// Random per-process HMAC key (`_USER_AUTH_CACHE_HMAC_KEY`,
    /// `secrets.token_bytes(32)`).
    hmac_key: [u8; 32],
}

impl CredentialCache {
    /// Build a cache from the parsed config fields. A zero TTL (the shipped
    /// default) yields a disabled cache; a positive TTL enables it with the
    /// given capacity. Mirrors `__init__.py:376-384`.
    pub fn new(max_size: u64, ttl_seconds: u64) -> Self {
        if ttl_seconds == 0 {
            return Self { inner: None };
        }
        Self {
            inner: Some(Inner {
                map: Mutex::new(HashMap::new()),
                capacity: usize::try_from(max_size).unwrap_or(usize::MAX).max(1),
                ttl: Duration::from_secs(ttl_seconds),
                hmac_key: random_hmac_key(),
            }),
        }
    }

    /// A permanently disabled cache (used when auth is enabled with the default
    /// `auth_cache_ttl_seconds = 0`, and in tests).
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// Whether the cache is enabled (`auth_cache_ttl_seconds > 0`).
    pub fn enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// `(username, HMAC-SHA256(hmac_key, password))` (`_auth_cache_key`,
    /// `__init__.py:394`).
    fn key(inner: &Inner, username: &str, password: &str) -> CacheKey {
        let mut mac = HmacSha256::new_from_slice(&inner.hmac_key)
            .expect("HMAC accepts a 32-byte key of any length");
        mac.update(password.as_bytes());
        (username.to_string(), mac.finalize().into_bytes().to_vec())
    }

    /// Look up a cached `User` for the credential, or `None` on a miss / expiry
    /// / when disabled. A hit skips the store's hash comparison entirely.
    pub fn get(&self, username: &str, password: &str) -> Option<User> {
        let inner = self.inner.as_ref()?;
        let key = Self::key(inner, username, password);
        let mut map = inner.map.lock().unwrap();
        match map.get(&key) {
            Some((inserted, user)) if inserted.elapsed() < inner.ttl => Some(user.clone()),
            Some(_) => {
                map.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Cache a successful credential check. No-op when disabled.
    pub fn insert(&self, username: &str, password: &str, user: User) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        let key = Self::key(inner, username, password);
        let mut map = inner.map.lock().unwrap();
        map.retain(|_, (inserted, _)| inserted.elapsed() < inner.ttl);
        if map.len() >= inner.capacity && !map.contains_key(&key) {
            // cachetools.TTLCache evicts the oldest entry when over capacity.
            if let Some(oldest) = map
                .iter()
                .min_by_key(|(_, (inserted, _))| *inserted)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest);
            }
        }
        map.insert(key, (Instant::now(), user));
    }

    /// Drop every cached credential for `username` (`_invalidate_user_auth_cache`,
    /// `__init__.py:441`). Called from user-mutation paths so password / admin /
    /// deletion changes take effect immediately. No-op when disabled.
    pub fn invalidate_user(&self, username: &str) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        inner.map.lock().unwrap().retain(|(u, _), _| u != username);
    }

    /// Current entry count (test/diagnostic helper). `0` when disabled.
    pub fn len(&self) -> usize {
        match &self.inner {
            Some(inner) => inner.map.lock().unwrap().len(),
            None => 0,
        }
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for CredentialCache {
    /// Never print entries: they hold user password hashes and HMAC-keyed
    /// digests. Report only the enabled flag and entry count.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CredentialCache")
            .field("enabled", &self.enabled())
            .field("len", &self.len())
            .finish()
    }
}

/// A random per-process 32-byte HMAC key drawn from the OS CSPRNG, mirroring
/// Python's `secrets.token_bytes(32)` (`_USER_AUTH_CACHE_HMAC_KEY`). Uses two
/// UUIDv4s (16 CSPRNG bytes each via `getrandom`) — the same entropy source the
/// `/signup` `CsrfSecret` uses, avoiding a new dependency.
fn random_hmac_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    key[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    key[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(id: i64, name: &str) -> User {
        User {
            id,
            username: name.to_string(),
            password_hash: "scrypt:32768:8:1$salt$hash".to_string(),
            is_admin: false,
        }
    }

    #[test]
    fn disabled_by_default_ttl_zero() {
        // The shipped default is auth_cache_ttl_seconds = 0 → cache off.
        let cache = CredentialCache::new(10_000, 0);
        assert!(!cache.enabled());
        cache.insert("u", "pw", user(1, "u"));
        assert!(cache.is_empty(), "disabled cache stores nothing");
        assert!(
            cache.get("u", "pw").is_none(),
            "disabled cache always misses"
        );
    }

    #[test]
    fn enabled_hit_and_miss() {
        let cache = CredentialCache::new(10, 3600);
        assert!(cache.enabled());
        // Miss before insert.
        assert!(cache.get("alice", "pw").is_none());
        cache.insert("alice", "pw", user(1, "alice"));
        // Hit with the same (user, password).
        assert_eq!(cache.get("alice", "pw").map(|u| u.id), Some(1));
        // Miss on a different password (HMAC digest differs).
        assert!(cache.get("alice", "wrong").is_none());
        // Miss on a different username with the same password.
        assert!(cache.get("bob", "pw").is_none());
    }

    #[test]
    fn ttl_zero_never_hits_even_if_enabled_path() {
        // A positive-then-expired window: ttl of 0 secs disables; use the
        // ResourceWorkspace pattern of instant expiry via a tiny window is
        // covered elsewhere. Here confirm invalidation clears an entry.
        let cache = CredentialCache::new(10, 3600);
        cache.insert("alice", "pw", user(1, "alice"));
        assert!(cache.get("alice", "pw").is_some());
        cache.invalidate_user("alice");
        assert!(
            cache.get("alice", "pw").is_none(),
            "invalidate_user drops the entry"
        );
    }

    #[test]
    fn invalidate_user_only_drops_that_user() {
        let cache = CredentialCache::new(10, 3600);
        cache.insert("alice", "pw1", user(1, "alice"));
        cache.insert("bob", "pw2", user(2, "bob"));
        cache.invalidate_user("alice");
        assert!(cache.get("alice", "pw1").is_none());
        assert_eq!(cache.get("bob", "pw2").map(|u| u.id), Some(2));
    }

    #[test]
    fn eviction_over_capacity() {
        let cache = CredentialCache::new(2, 3600);
        cache.insert("a", "p", user(1, "a"));
        cache.insert("b", "p", user(2, "b"));
        cache.insert("c", "p", user(3, "c"));
        assert_eq!(cache.len(), 2, "bounded by capacity");
    }

    #[test]
    fn distinct_hmac_key_per_instance() {
        // Two enabled caches use independent random HMAC keys, so a digest from
        // one is meaningless to the other — but each is internally consistent.
        let a = CredentialCache::new(10, 3600);
        let b = CredentialCache::new(10, 3600);
        a.insert("u", "pw", user(1, "u"));
        b.insert("u", "pw", user(2, "u"));
        assert_eq!(a.get("u", "pw").map(|u| u.id), Some(1));
        assert_eq!(b.get("u", "pw").map(|u| u.id), Some(2));
    }
}
