//! The async webhook delivery engine, porting the fire-and-forget delivery in
//! `mlflow/webhooks/delivery.py` (`deliver_webhook` L310, `_deliver_webhook_impl`
//! L288, the `ThreadPoolExecutor` pool L51, and `_get_cached_webhooks_by_event`
//! L228) onto tokio.
//!
//! ## Fire-and-forget, no durable queue (design decision D11)
//!
//! Python submits each delivery to a process-global `ThreadPoolExecutor` and
//! returns immediately; nothing is persisted, so pending deliveries are lost if
//! the process dies. We replicate that observable semantics on tokio: [`fire`]
//! looks up the subscribed webhooks (through the TTL cache), signs a request for
//! each active one, and `tokio::spawn`s the send — returning without awaiting.
//! Concurrency is bounded to `MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS` (default 10)
//! by a [`tokio::sync::Semaphore`], the analogue of the pool's `max_workers`; the
//! queue in front of the permits is effectively unbounded, matching the
//! `ThreadPoolExecutor`'s unbounded work queue.
//!
//! ### Future work: a durable outbox
//!
//! D11 accepts the parity-with-Python at-most-once behavior for now. A durable
//! outbox (persist the intended delivery in a `webhook_deliveries` table inside
//! the same transaction as the triggering mutation, then a background worker
//! drains it with retries and marks terminal state) would upgrade this to
//! at-least-once and survive restarts. That is intentionally **not** built here;
//! this doc-comment is the proposal pointer.
//!
//! ## Public API for T8.4 (event triggers)
//!
//! T8.4's registry handlers call [`WebhookDispatcher::fire`] with just an event
//! and its data payload — they need zero webhook knowledge (no lookup, no
//! signing, no HTTP). The dispatcher is built once in `main.rs`, stored in
//! `AppState`, and shared (`Arc`-cloned) into handlers.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::{Mutex, Semaphore};

use crate::delivery::build_signed_request;
use crate::entities::{Webhook, WebhookEvent};
use crate::http_send::{send_with_ssrf_guard, Resolver, SendConfig, SystemResolver};
use crate::store::WebhookStore;

/// `MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS` (default 10,
/// `environment_variables.py:1377`).
const MAX_WORKERS_ENV: &str = "MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS";
/// `MLFLOW_WEBHOOK_CACHE_TTL` (default 60s, `environment_variables.py:1389`).
const CACHE_TTL_ENV: &str = "MLFLOW_WEBHOOK_CACHE_TTL";
/// `TTLCache(maxsize=1000, ...)` (`delivery.py:223`).
const CACHE_MAX_SIZE: usize = 1000;

/// The async delivery engine: looks up webhooks-by-event (TTL-cached), signs,
/// and fire-and-forget enqueues HTTP deliveries with the connect-time SSRF
/// guard.
///
/// Cheap to clone (all shared state is behind `Arc`), matching the `AppState`
/// clone requirement.
#[derive(Clone)]
pub struct WebhookDispatcher {
    inner: Arc<Inner>,
}

struct Inner {
    store: WebhookStore,
    /// The workspace deliveries are scoped to. Python's `deliver_webhook` reads
    /// `MLFLOW_ENABLE_WORKSPACES`; the single-tenant server uses one workspace.
    workspace: String,
    /// Bounds concurrent in-flight deliveries to `max_workers` (the pool size).
    semaphore: Arc<Semaphore>,
    cache: Arc<Mutex<EventCache>>,
    ttl: Duration,
    send_config: SendConfig,
    resolver: Arc<dyn Resolver>,
}

impl WebhookDispatcher {
    /// Build the dispatcher over a webhook store, resolving pool size / cache
    /// TTL / retry config from the environment (parity with the module-level
    /// globals Python initializes at import time).
    pub fn new(store: WebhookStore, workspace: impl Into<String>) -> Self {
        Self::with_resolver(store, workspace, Arc::new(SystemResolver))
    }

    /// Build the dispatcher with an explicit [`Resolver`] — the seam the SSRF
    /// matrix tests use to map a hostname to a chosen IP without real DNS.
    pub fn with_resolver(
        store: WebhookStore,
        workspace: impl Into<String>,
        resolver: Arc<dyn Resolver>,
    ) -> Self {
        Self::with_config(store, workspace, resolver, SendConfig::from_env())
    }

    /// Build the dispatcher with an explicit resolver and [`SendConfig`] — the
    /// fully-injectable constructor tests use to drive delivery against a local
    /// listener (private-IP escape hatch on, no retry backoff) and to force a
    /// short cache TTL. Production paths use [`new`](Self::new).
    pub fn with_config(
        store: WebhookStore,
        workspace: impl Into<String>,
        resolver: Arc<dyn Resolver>,
        send_config: SendConfig,
    ) -> Self {
        let max_workers = env_usize(MAX_WORKERS_ENV, 10).max(1);
        let ttl = Duration::from_secs(env_u64(CACHE_TTL_ENV, 60));
        Self {
            inner: Arc::new(Inner {
                store,
                workspace: workspace.into(),
                semaphore: Arc::new(Semaphore::new(max_workers)),
                cache: Arc::new(Mutex::new(EventCache::new(CACHE_MAX_SIZE))),
                ttl,
                send_config,
                resolver,
            }),
        }
    }

    /// `deliver_webhook(event=..., payload=..., store=...)` (`delivery.py:310`).
    ///
    /// Looks up webhooks subscribed to `event` (TTL-cached), and for each
    /// **active** one signs a request and `tokio::spawn`s a fire-and-forget send.
    /// Returns immediately without awaiting deliveries. Never propagates a
    /// delivery error to the caller (each failure is logged), matching Python's
    /// top-level `try/except` in `deliver_webhook`.
    ///
    /// `payload` is the event's `data` object (the same shape Python passes as
    /// `WebhookPayload`); the wrapped `{entity, action, timestamp, data}` and the
    /// signature are built per webhook inside the spawned task.
    pub async fn fire(&self, event: WebhookEvent, payload: Value) {
        for handle in self.fire_handles(event, payload).await {
            // Detach: fire-and-forget. The task runs to completion on the runtime
            // independently; we do not await it.
            drop(handle);
        }
    }

    /// Like [`fire`](Self::fire) but returns the spawned delivery task handles so
    /// callers (tests) can await completion. Production code uses `fire`; this
    /// exists so integration tests can deterministically observe deliveries
    /// without the fire-and-forget race.
    #[doc(hidden)]
    pub async fn fire_handles(
        &self,
        event: WebhookEvent,
        payload: Value,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let webhooks = match self.cached_webhooks_by_event(event).await {
            Ok(w) => w,
            Err(e) => {
                tracing::error!(
                    error = %e.message,
                    "Failed to deliver webhook for event {}.{}: webhook lookup failed",
                    event.entity.as_db_str(),
                    event.action.as_db_str(),
                );
                return Vec::new();
            }
        };

        webhooks
            .iter()
            .filter(|w| w.status.is_active())
            .map(|w| self.spawn_delivery(w.clone(), event, payload.clone()))
            .collect()
    }

    /// Sign + spawn a single fire-and-forget delivery, gated by a pool permit.
    fn spawn_delivery(
        &self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Value,
    ) -> tokio::task::JoinHandle<()> {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            // Acquire a pool permit (bounded concurrency = max_workers). The
            // permit is held for the whole delivery incl. retries/backoff.
            let _permit = match inner.semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => return, // semaphore closed (shutdown)
            };

            let signed = match build_signed_request(&webhook, event, &payload, &inner.workspace) {
                Ok(s) => s,
                Err(msg) => {
                    tracing::error!(
                        url = %webhook.url,
                        "Failed to send webhook for event {}.{}: {msg}",
                        event.entity.as_db_str(),
                        event.action.as_db_str(),
                    );
                    return;
                }
            };

            if let Err(e) =
                send_with_ssrf_guard(&signed, inner.send_config, inner.resolver.clone()).await
            {
                // `_send_webhook_with_error_handling` logs and swallows.
                tracing::error!(
                    url = %webhook.url,
                    "Failed to send webhook to {} for event {}.{}: {e}",
                    webhook.url,
                    event.entity.as_db_str(),
                    event.action.as_db_str(),
                );
            }
        })
    }

    /// `_get_cached_webhooks_by_event` (`delivery.py:228`): serve from the TTL
    /// cache, else fetch every page from the store and repopulate. TTL-only
    /// invalidation, exactly like Python's `TTLCache` (no explicit invalidation
    /// on webhook mutation — a create/update is visible after at most `ttl`).
    async fn cached_webhooks_by_event(
        &self,
        event: WebhookEvent,
    ) -> Result<Arc<Vec<Webhook>>, mlflow_error::MlflowError> {
        let mut cache = self.inner.cache.lock().await;
        if let Some(hit) = cache.get(&event) {
            return Ok(hit);
        }

        // Miss: fetch all pages for this specific event.
        let mut webhooks: Vec<Webhook> = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let page = self
                .inner
                .store
                .list_webhooks_by_event(
                    &self.inner.workspace,
                    event,
                    Some(100),
                    page_token.as_deref(),
                )
                .await?;
            webhooks.extend(page.webhooks);
            match page.next_page_token {
                Some(t) => page_token = Some(t),
                None => break,
            }
        }

        let value = Arc::new(webhooks);
        cache.insert(event, value.clone(), self.inner.ttl);
        Ok(value)
    }
}

/// A tiny TTL cache keyed by [`WebhookEvent`], mirroring
/// `cachetools.TTLCache(maxsize, ttl)`: entries expire after `ttl`, and a bounded
/// size evicts the oldest-inserted entry when full.
struct EventCache {
    max_size: usize,
    entries: std::collections::HashMap<WebhookEvent, Entry>,
    /// Insertion order for size-based eviction (oldest first).
    order: std::collections::VecDeque<WebhookEvent>,
}

struct Entry {
    value: Arc<Vec<Webhook>>,
    expires_at: Instant,
}

impl EventCache {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn get(&mut self, event: &WebhookEvent) -> Option<Arc<Vec<Webhook>>> {
        let expired = match self.entries.get(event) {
            Some(entry) if entry.expires_at > Instant::now() => return Some(entry.value.clone()),
            Some(_) => true,
            None => false,
        };
        if expired {
            self.entries.remove(event);
            self.order.retain(|e| e != event);
        }
        None
    }

    fn insert(&mut self, event: WebhookEvent, value: Arc<Vec<Webhook>>, ttl: Duration) {
        if !self.entries.contains_key(&event) {
            self.order.push_back(event);
            while self.order.len() > self.max_size {
                if let Some(evicted) = self.order.pop_front() {
                    self.entries.remove(&evicted);
                }
            }
        }
        self.entries.insert(
            event,
            Entry {
                value,
                expires_at: Instant::now() + ttl,
            },
        );
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{WebhookAction, WebhookEntity};

    fn event() -> WebhookEvent {
        WebhookEvent::new(WebhookEntity::RegisteredModel, WebhookAction::Created)
    }

    #[test]
    fn cache_hits_within_ttl_and_expires_after() {
        let mut cache = EventCache::new(10);
        let ev = event();
        cache.insert(ev, Arc::new(vec![]), Duration::from_secs(60));
        assert!(cache.get(&ev).is_some());

        // A zero TTL entry is immediately expired on the next get.
        cache.insert(ev, Arc::new(vec![]), Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.get(&ev).is_none());
    }

    #[test]
    fn cache_evicts_oldest_when_full() {
        let mut cache = EventCache::new(1);
        let a = WebhookEvent::new(WebhookEntity::RegisteredModel, WebhookAction::Created);
        let b = WebhookEvent::new(WebhookEntity::ModelVersion, WebhookAction::Created);
        cache.insert(a, Arc::new(vec![]), Duration::from_secs(60));
        cache.insert(b, Arc::new(vec![]), Duration::from_secs(60));
        assert!(cache.get(&a).is_none());
        assert!(cache.get(&b).is_some());
    }
}
