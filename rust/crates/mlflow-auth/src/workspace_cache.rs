//! The resource→workspace TTL cache (plan T9.8), a port of
//! `_RESOURCE_WORKSPACE_CACHE` in `mlflow/server/auth/__init__.py:363-367`.
//!
//! Permission checks resolve a resource id (experiment / registered model /
//! prompt / …) to the workspace that owns it, then scope the role lookup to
//! that workspace. That resource→workspace relationship is **immutable**
//! (`__init__.py:622`), so the resolved mapping is cached with a bounded TTL
//! cache keyed by `"<resource_label>:<workspace_scope>:<resource_id>"`
//! (`__init__.py:636-657`).
//!
//! ## Status: not on the T10.4 hot path (Rust resolves differently)
//!
//! Python needs this cache because `_get_resource_workspace` fetches the
//! resource *unscoped* (base `_get_query` has no workspace filter,
//! `sqlalchemy_store.py:414-418`) to read its owning workspace, then scopes the
//! role lookup to that workspace — an extra DB round-trip per permission check
//! that the immutable resource→workspace mapping lets it cache.
//!
//! The Rust store fetches are **already workspace-scoped** (`get_experiment(ws,
//! id)`, `get_run(ws, id)`, … filter on `workspace = ?`), and the T10.4 auth
//! resolver (`auth_middleware/validators.rs`) looks up grants directly in the
//! request's resolved workspace (`RequestCtx::workspace`, stamped by T10.3)
//! without a separate resource→workspace fetch. There is therefore no
//! resource→workspace resolution step to memoize on the Rust permission path,
//! and nothing consults this cache. It is kept (implemented + unit-tested, its
//! `workspace_cache_max_size` / `workspace_cache_ttl_seconds` config read from
//! the same ini as Python) so a future resolver that does an unscoped
//! resource→workspace lookup (e.g. to reject a cross-workspace resource id whose
//! grant string collides) can drop it in front via `get_or_insert` without
//! re-deriving the cache.
//!
//! Uses the same `Mutex<HashMap>` TTL-cache pattern as `mlflow-store`'s
//! workspace caches (cachetools.TTLCache parity: `maxsize` + `ttl`, oldest-entry
//! eviction).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A bounded TTL cache mapping a resource key to its (optional) workspace name.
/// `None` is a cacheable value (Python caches `workspace_name: str | None`; an
/// unresolvable resource maps to `None`).
#[derive(Debug)]
pub struct ResourceWorkspaceCache {
    map: Mutex<HashMap<String, (Instant, Option<String>)>>,
    capacity: usize,
    ttl: Duration,
}

impl ResourceWorkspaceCache {
    /// Build from the parsed config fields (`workspace_cache_max_size`,
    /// `workspace_cache_ttl_seconds`). Mirrors `__init__.py:363-367`.
    pub fn new(max_size: u64, ttl_seconds: u64) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            capacity: usize::try_from(max_size).unwrap_or(usize::MAX).max(1),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// The cache key Python builds: `"<resource_label>:<workspace_scope>:<resource_id>"`
    /// (`__init__.py:636`). Exposed so the T10.4 resolver builds identical keys.
    pub fn cache_key(resource_label: &str, workspace_scope: &str, resource_id: &str) -> String {
        format!("{resource_label}:{workspace_scope}:{resource_id}")
    }

    /// Look up a cached mapping, honouring the TTL. `None` = miss/expired.
    /// (`Some(None)` = a cached "no workspace"; `Some(Some(ws))` = a hit.)
    pub fn get(&self, key: &str) -> Option<Option<String>> {
        let mut map = self.map.lock().unwrap();
        match map.get(key) {
            Some((inserted, v)) if inserted.elapsed() < self.ttl => Some(v.clone()),
            Some(_) => {
                map.remove(key);
                None
            }
            None => None,
        }
    }

    /// Cache a resolved mapping, evicting the oldest entry when over capacity
    /// (cachetools.TTLCache semantics).
    pub fn insert(&self, key: String, workspace: Option<String>) {
        let mut map = self.map.lock().unwrap();
        map.retain(|_, (inserted, _)| inserted.elapsed() < self.ttl);
        if map.len() >= self.capacity && !map.contains_key(&key) {
            if let Some(oldest) = map
                .iter()
                .min_by_key(|(_, (inserted, _))| *inserted)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest);
            }
        }
        map.insert(key, (Instant::now(), workspace));
    }

    /// Return the cached mapping or compute + cache it (the shape T10.4's
    /// resolver will call: `cache.get_or_insert(key, || resolve(id))`).
    pub fn get_or_insert(
        &self,
        key: &str,
        resolve: impl FnOnce() -> Option<String>,
    ) -> Option<String> {
        if let Some(cached) = self.get(key) {
            return cached;
        }
        let resolved = resolve();
        self.insert(key.to_string(), resolved.clone());
        resolved
    }

    /// Current entry count (test/diagnostic helper).
    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_matches_python_shape() {
        assert_eq!(
            ResourceWorkspaceCache::cache_key("experiment", "default", "12"),
            "experiment:default:12"
        );
    }

    #[test]
    fn hit_miss_and_none_are_cacheable() {
        let cache = ResourceWorkspaceCache::new(10, 3600);
        assert_eq!(cache.get("experiment:default:1"), None); // miss
        cache.insert("experiment:default:1".into(), Some("ws-a".into()));
        assert_eq!(
            cache.get("experiment:default:1"),
            Some(Some("ws-a".to_string()))
        );
        // A resolved "no workspace" is itself a cache hit, not a miss.
        cache.insert("experiment:default:2".into(), None);
        assert_eq!(cache.get("experiment:default:2"), Some(None));
    }

    #[test]
    fn get_or_insert_computes_once() {
        let cache = ResourceWorkspaceCache::new(10, 3600);
        let calls = std::cell::Cell::new(0);
        let resolve = || {
            calls.set(calls.get() + 1);
            Some("ws".to_string())
        };
        assert_eq!(
            cache.get_or_insert("experiment:default:1", resolve),
            Some("ws".to_string())
        );
        assert_eq!(
            cache.get_or_insert("experiment:default:1", || {
                calls.set(calls.get() + 1);
                Some("ws".to_string())
            }),
            Some("ws".to_string())
        );
        assert_eq!(calls.get(), 1, "second call must hit the cache");
    }

    #[test]
    fn ttl_expires_entries() {
        let cache = ResourceWorkspaceCache::new(10, 0);
        // ttl_seconds = 0 → every entry is immediately stale.
        cache.insert("experiment:default:1".into(), Some("ws".into()));
        // Duration::from_secs(0) means elapsed() is never < ttl, so it expires.
        assert_eq!(cache.get("experiment:default:1"), None);
    }

    #[test]
    fn evicts_oldest_over_capacity() {
        let cache = ResourceWorkspaceCache::new(2, 3600);
        cache.insert("a".into(), Some("1".into()));
        cache.insert("b".into(), Some("2".into()));
        cache.insert("c".into(), Some("3".into()));
        assert_eq!(cache.len(), 2);
        // "a" (oldest) was evicted.
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("c"), Some(Some("3".to_string())));
    }
}
