//! Fernet encryption of webhook secrets, mirroring the `EncryptedString`
//! SQLAlchemy `TypeDecorator` in
//! `mlflow/store/model_registry/dbmodels/models.py:287-311`.
//!
//! Python builds the cipher once per `EncryptedString` instance:
//!
//! ```python
//! encryption_key = MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY.get() or Fernet.generate_key()
//! self.cipher = Fernet(encryption_key)
//! ```
//!
//! then `process_bind_param` encrypts on write and `process_result_value`
//! decrypts on read (both `None`-passthrough). Crucially, **a missing key is
//! NOT an error**: Python falls back to an ephemeral `Fernet.generate_key()`.
//! We reproduce that faithfully — [`SecretCipher::from_env`] generates a random
//! key when `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` is unset, so a secret written
//! and read within the same process round-trips, but (as in Python) secrets do
//! not survive a restart without a configured key. See the crate docs for this
//! deviation note relative to the T8.1 task phrasing.
//!
//! The `fernet` crate (0.2) is validated Rust<->Python-compatible against
//! Python `cryptography.fernet` in `rust/spikes/` (T0.4): tokens produced by
//! either side decrypt on the other, so secrets written by Python decrypt in
//! Rust and vice versa (the T8.1 cross-language AC).

use fernet::Fernet;
use mlflow_error::MlflowError;

/// The webhook-secret encryption key env var
/// (`MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`,
/// `mlflow/environment_variables.py:1367`).
pub const SECRET_ENCRYPTION_KEY_ENV: &str = "MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY";

/// A Fernet cipher for webhook secrets, holding the key exactly as Python's
/// `EncryptedString.cipher` does.
#[derive(Clone)]
pub struct SecretCipher {
    fernet: Fernet,
    /// The url-safe-base64 key string, retained so a store can be cloned with a
    /// stable cipher (Python shares one `EncryptedString` instance per column).
    key: String,
}

impl std::fmt::Debug for SecretCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the key material.
        f.debug_struct("SecretCipher").finish_non_exhaustive()
    }
}

impl SecretCipher {
    /// Build a cipher from `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`, falling back
    /// to a freshly generated ephemeral key when unset/empty — matching
    /// `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY.get() or Fernet.generate_key()`.
    ///
    /// A *present but malformed* key is an error (Python's `Fernet(key)` raises
    /// `ValueError`), surfaced here as an `INTERNAL_ERROR` since it is a server
    /// misconfiguration, not a client input.
    pub fn from_env() -> Result<Self, MlflowError> {
        match std::env::var(SECRET_ENCRYPTION_KEY_ENV) {
            Ok(k) if !k.is_empty() => Self::from_key(&k),
            _ => Ok(Self::generate()),
        }
    }

    /// Build a cipher from an explicit url-safe-base64 Fernet key.
    pub fn from_key(key: &str) -> Result<Self, MlflowError> {
        let fernet = Fernet::new(key).ok_or_else(|| {
            MlflowError::internal_error(format!(
                "Invalid {SECRET_ENCRYPTION_KEY_ENV}: must be 32 url-safe base64-encoded bytes."
            ))
        })?;
        Ok(Self {
            fernet,
            key: key.to_string(),
        })
    }

    /// Generate an ephemeral cipher (`Fernet.generate_key()` fallback).
    pub fn generate() -> Self {
        let key = Fernet::generate_key();
        let fernet = Fernet::new(&key).expect("freshly generated Fernet key is valid");
        Self { fernet, key }
    }

    /// The key string (url-safe base64). Only exposed so callers that must
    /// re-derive an identical cipher (e.g. tests) can; never logged.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// `process_bind_param`: encrypt a plaintext secret to a Fernet token
    /// string. `None` passes through.
    pub fn encrypt(&self, plaintext: Option<&str>) -> Option<String> {
        plaintext.map(|p| self.fernet.encrypt(p.as_bytes()))
    }

    /// `process_result_value`: decrypt a stored Fernet token back to plaintext.
    /// `None` passes through. A token that fails to decrypt (wrong key /
    /// corruption) is an error, matching `self.cipher.decrypt(...)` raising.
    pub fn decrypt(&self, token: Option<&str>) -> Result<Option<String>, MlflowError> {
        match token {
            None => Ok(None),
            Some(t) => {
                let bytes = self.fernet.decrypt(t).map_err(|e| {
                    MlflowError::internal_error(format!("Failed to decrypt webhook secret: {e:?}"))
                })?;
                let s = String::from_utf8(bytes).map_err(|e| {
                    MlflowError::internal_error(format!(
                        "Decrypted webhook secret is not valid UTF-8: {e}"
                    ))
                })?;
                Ok(Some(s))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_within_one_cipher() {
        let cipher = SecretCipher::generate();
        let token = cipher.encrypt(Some("hunter2")).unwrap();
        assert_eq!(
            cipher.decrypt(Some(&token)).unwrap(),
            Some("hunter2".into())
        );
    }

    #[test]
    fn none_passes_through() {
        let cipher = SecretCipher::generate();
        assert_eq!(cipher.encrypt(None), None);
        assert_eq!(cipher.decrypt(None).unwrap(), None);
    }

    #[test]
    fn malformed_key_is_error() {
        assert!(SecretCipher::from_key("not-a-valid-fernet-key").is_err());
    }
}
