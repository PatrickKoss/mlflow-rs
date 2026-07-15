//! Table names for the two webhook tables (plan §5.1).
//!
//! Source of truth: `mlflow/store/model_registry/dbmodels/models.py:314-384`
//! (`SqlWebhook`, `SqlWebhookEvent`). Both tables carry a `workspace` column
//! (`webhooks` has a plain `webhook_id` PK; `webhook_events` PKs on
//! `(webhook_id, entity, action)` with an `ON DELETE CASCADE` FK to `webhooks`).

/// The `webhooks` table.
pub const WEBHOOKS: &str = "webhooks";
/// The `webhook_events` table.
pub const WEBHOOK_EVENTS: &str = "webhook_events";

/// All webhook table names owned by the Rust store (plan §5.1).
pub const WEBHOOK_TABLES: &[&str] = &[WEBHOOKS, WEBHOOK_EVENTS];
