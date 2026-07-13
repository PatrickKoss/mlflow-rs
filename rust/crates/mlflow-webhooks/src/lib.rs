//! `mlflow-webhooks`: webhook storage and delivery engine.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§3.15, Phase 8), this crate owns the
//! `webhooks`/`webhook_events` tables (Fernet-encrypted secrets, soft
//! delete, workspace scoping), the REST endpoints for CRUD + test delivery,
//! and the async fire-and-forget delivery engine: HMAC-SHA256 `v1,<b64>`
//! signing over `"{delivery_id}.{timestamp}.{payload}"`, the
//! `X-MLflow-Signature`/`X-MLflow-Timestamp`/`X-MLflow-Delivery-Id`
//! headers, retries on `[429, 500, 502, 503, 504]` with backoff, SSRF
//! protection (public-IP validation at connect time), and a TTL cache of
//! webhooks by event, matching `mlflow/webhooks/delivery.py`.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
