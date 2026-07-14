//! Shared application state for the HTTP layer.
//!
//! `AppState` is the single value threaded through every axum handler via
//! `State`. For Phase 3 it carries the tracking [`TrackingStore`]; later phases
//! (runs, metrics, traces, registry, auth, webhooks) add their own stores here,
//! so this is the extension point for the whole server.
//!
//! The state is cheap to clone (`TrackingStore` holds an `Arc`-backed pool),
//! which is what axum's `State` extractor requires.

use std::sync::Arc;

use mlflow_store::TrackingStore;

/// Application state shared across all HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    tracking_store: TrackingStore,
}

impl AppState {
    /// Build the state from an already-constructed [`TrackingStore`]. Tests
    /// inject a store over a temp DB; `main`/`build_app` construct one from the
    /// configured backend-store URI.
    pub fn new(tracking_store: TrackingStore) -> Self {
        Self {
            inner: Arc::new(AppStateInner { tracking_store }),
        }
    }

    /// The tracking store (experiments, runs, metrics, traces, …).
    pub fn tracking_store(&self) -> &TrackingStore {
        &self.inner.tracking_store
    }
}
