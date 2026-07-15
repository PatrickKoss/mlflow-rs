//! Webhook HMAC-SHA256 signing and the `X-MLflow-*` header names, porting
//! `mlflow/webhooks/delivery.py:121` (`_generate_hmac_signature`) and
//! `mlflow/webhooks/constants.py`.
//!
//! Placed in `mlflow-webhooks` (not the server) so the async delivery engine
//! (T8.3) reuses the exact same signing code the `/test` path uses here.

use base64::Engine;
use hmac::{Hmac, Mac};
use sha2::Sha256;

/// `WEBHOOK_SIGNATURE_HEADER` (`mlflow/webhooks/constants.py:2`).
pub const WEBHOOK_SIGNATURE_HEADER: &str = "X-MLflow-Signature";
/// `WEBHOOK_TIMESTAMP_HEADER` (`mlflow/webhooks/constants.py:3`).
pub const WEBHOOK_TIMESTAMP_HEADER: &str = "X-MLflow-Timestamp";
/// `WEBHOOK_DELIVERY_ID_HEADER` (`mlflow/webhooks/constants.py:4`).
pub const WEBHOOK_DELIVERY_ID_HEADER: &str = "X-MLflow-Delivery-Id";
/// `WEBHOOK_SIGNATURE_VERSION` (`mlflow/webhooks/constants.py:7`).
pub const WEBHOOK_SIGNATURE_VERSION: &str = "v1";

/// Generate the webhook HMAC-SHA256 signature over
/// `"{delivery_id}.{timestamp}.{payload}"`, returned as
/// `"{version},{base64(digest)}"` — byte-for-byte
/// `_generate_hmac_signature(secret, delivery_id, timestamp, payload)`.
pub fn generate_hmac_signature(
    secret: &str,
    delivery_id: &str,
    timestamp: &str,
    payload: &str,
) -> String {
    let signed_content = format!("{delivery_id}.{timestamp}.{payload}");
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(signed_content.as_bytes());
    let digest = mac.finalize().into_bytes();
    let signature_b64 = base64::engine::general_purpose::STANDARD.encode(digest);
    format!("{WEBHOOK_SIGNATURE_VERSION},{signature_b64}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_has_version_prefix() {
        let sig = generate_hmac_signature("secret", "d", "1", "{}");
        assert!(sig.starts_with("v1,"));
    }
}
