//! `mlflow-webhooks`: webhook storage + the `/test` delivery slice.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§3.15, §4.16, Phase 8, tasks T8.1/T8.2),
//! this crate owns the `webhooks`/`webhook_events` tables and the pieces the
//! REST endpoints need:
//!
//! * [`entities`] — the owned `Webhook`/`WebhookEvent`/`WebhookTestResult`
//!   entities and the `WebhookStatus`/`WebhookEntity`/`WebhookAction` enums
//!   (the enums carry the lowercase DB strings and know their proto numbers).
//! * [`crypto`] — Fernet encryption of the `secret` column, mirroring the
//!   `EncryptedString` `TypeDecorator` (validated Rust<->Python-compatible in
//!   `rust/spikes/`, T0.4). A missing `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`
//!   falls back to an ephemeral key, exactly as Python does — see the note
//!   below.
//! * [`signing`] — HMAC-SHA256 `v1,<b64>` signing + the `X-MLflow-*` header
//!   names, placed here so the T8.3 async delivery engine reuses the same code.
//! * [`validation`] — name/url/events validation and the entity/action
//!   combination check, byte-matching `mlflow/utils/validation.py` and
//!   `WebhookEvent.__init__`.
//! * [`payloads`] — the example event payloads the `/test` endpoint sends.
//! * [`WebhookStore`] — create/get/list/update/delete + `list_webhooks_by_event`,
//!   all workspace-scoped with soft delete (**T8.1**).
//! * [`delivery`] — the wrapped/signed request assembly + the `/test` endpoint
//!   delivery (**T8.2**).
//! * [`http_send`] — the connect-time SSRF-guarded HTTP sender (resolve +
//!   validate peer IP, connect to a pinned IP, retry-on-status backoff, redirect
//!   re-validation), porting `mlflow/webhooks/ssrf.py` + the session's `Retry`
//!   config (**T8.3**).
//! * [`WebhookDispatcher`] — the fire-and-forget async delivery engine: TTL
//!   cache of webhooks-by-event, bounded tokio task pool, signing, enqueue.
//!   `WebhookDispatcher::fire(event, payload)` is the API T8.4's registry event
//!   triggers call (**T8.3**).
//!
//! ## Deviation: missing encryption key
//!
//! The T8.1 task text says "missing key when a secret is supplied → match
//! Python's error". Python's `EncryptedString` (`models.py:300`) has **no such
//! error**: it does `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY.get() or
//! Fernet.generate_key()`, silently using an ephemeral key when unset. Matching
//! Python faithfully therefore means *not* erroring — [`crypto::SecretCipher::from_env`]
//! generates an ephemeral key. A *present-but-malformed* key is an error (as
//! Python's `Fernet(key)` raises). See that module's docs.
//!
//! ## Reuse of `mlflow-store`
//!
//! Like `mlflow-registry`, this crate depends on `mlflow-store` for its
//! connection/dialect infrastructure ([`mlflow_store::Db`],
//! [`mlflow_store::Dialect`]). The crate-private query helpers there are copied
//! into [`dbutil`] pending consolidation.

mod dbutil;

pub mod crypto;
pub mod delivery;
pub mod dispatcher;
pub mod entities;
pub mod http_send;
pub mod payloads;
pub mod schema;
pub mod signing;
pub mod store;
pub mod validation;

pub use crypto::SecretCipher;
pub use dispatcher::WebhookDispatcher;
pub use entities::{
    Webhook, WebhookAction, WebhookEntity, WebhookEvent, WebhookStatus, WebhookTestResult,
};
pub use http_send::{Resolver, SystemResolver};
pub use store::{WebhookPage, WebhookStore};
