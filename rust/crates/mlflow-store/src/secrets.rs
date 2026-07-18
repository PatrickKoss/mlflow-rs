//! Gateway secret envelope encryption and the encrypted in-process cache.
//!
//! The envelope format is shared with `mlflow.utils.crypto`: a random 32-byte
//! DEK encrypts the UTF-8 secret with AES-256-GCM and AAD
//! `"{secret_id}|{secret_name}"`; the KEK wraps that DEK without AAD. Both
//! ciphertext blobs are `nonce(12) || ciphertext || tag(16)`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use hmac::Hmac;
use pbkdf2::pbkdf2;
use rand::{rngs::OsRng, RngCore};
use serde_json::{Map, Value};
use sha2::Sha256;

pub const AES_256_KEY_LENGTH: usize = 32;
pub const GCM_NONCE_LENGTH: usize = 12;
pub const GCM_TAG_LENGTH: usize = 16;
pub const PBKDF2_ITERATIONS: u32 = 600_000;
pub const MLFLOW_KEK_SALT: &[u8] = b"mlflow-secrets-kek-v1-2025";
pub const DEFAULT_KEK_PASSPHRASE: &str = "mlflow-default-kek-passphrase-for-development-only";
pub const CRYPTO_KEK_PASSPHRASE_ENV_VAR: &str = "MLFLOW_CRYPTO_KEK_PASSPHRASE";
pub const CRYPTO_KEK_VERSION_ENV_VAR: &str = "MLFLOW_CRYPTO_KEK_VERSION";
pub const SECRETS_CACHE_TTL_ENV_VAR: &str = "MLFLOW_SERVER_SECRETS_CACHE_TTL";
pub const SECRETS_CACHE_MAX_SIZE_ENV_VAR: &str = "MLFLOW_SERVER_SECRETS_CACHE_MAX_SIZE";

const MIN_CACHE_TTL: u64 = 10;
const MAX_CACHE_TTL: u64 = 300;
const DEFAULT_CACHE_TTL: u64 = 60;
const DEFAULT_CACHE_MAX_SIZE: usize = 1000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CryptoError;

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("secret decryption failed")
    }
}

impl std::error::Error for CryptoError {}

#[derive(Clone)]
pub struct Kek {
    key: [u8; AES_256_KEY_LENGTH],
    pub version: u32,
}

impl Kek {
    pub fn derive(passphrase: &str, version: u32) -> Result<Self, CryptoError> {
        let mut salt = Vec::with_capacity(MLFLOW_KEK_SALT.len() + 4);
        salt.extend_from_slice(MLFLOW_KEK_SALT);
        salt.extend_from_slice(&version.to_be_bytes());
        let mut key = [0_u8; AES_256_KEY_LENGTH];
        pbkdf2::<Hmac<Sha256>>(passphrase.as_bytes(), &salt, PBKDF2_ITERATIONS, &mut key)
            .map_err(|_| CryptoError)?;
        Ok(Self { key, version })
    }

    pub fn from_environment() -> Result<Self, CryptoError> {
        let passphrase = std::env::var(CRYPTO_KEK_PASSPHRASE_ENV_VAR)
            .unwrap_or_else(|_| DEFAULT_KEK_PASSPHRASE.to_string());
        let version = std::env::var(CRYPTO_KEK_VERSION_ENV_VAR)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        Self::derive(&passphrase, version)
    }

    pub fn for_stored_version(version: u32) -> Result<Self, CryptoError> {
        let passphrase = std::env::var(CRYPTO_KEK_PASSPHRASE_ENV_VAR)
            .unwrap_or_else(|_| DEFAULT_KEK_PASSPHRASE.to_string());
        Self::derive(&passphrase, version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedSecret {
    pub encrypted_value: Vec<u8>,
    pub wrapped_dek: Vec<u8>,
    pub kek_version: u32,
}

pub fn create_aad(secret_id: &str, secret_name: &str) -> Vec<u8> {
    format!("{secret_id}|{secret_name}").into_bytes()
}

fn encrypt_blob(plaintext: &[u8], key: &[u8; AES_256_KEY_LENGTH], aad: &[u8]) -> Vec<u8> {
    let cipher = Aes256Gcm::new_from_slice(key).expect("fixed-size AES-256 key");
    let mut nonce = [0_u8; GCM_NONCE_LENGTH];
    OsRng.fill_bytes(&mut nonce);
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .expect("AES-GCM encryption with valid inputs");
    let mut envelope = Vec::with_capacity(GCM_NONCE_LENGTH + ciphertext.len());
    envelope.extend_from_slice(&nonce);
    envelope.extend_from_slice(&ciphertext);
    envelope
}

fn decrypt_blob(
    envelope: &[u8],
    key: &[u8; AES_256_KEY_LENGTH],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if envelope.len() < GCM_NONCE_LENGTH + GCM_TAG_LENGTH {
        return Err(CryptoError);
    }
    let (nonce, ciphertext) = envelope.split_at(GCM_NONCE_LENGTH);
    Aes256Gcm::new_from_slice(key)
        .map_err(|_| CryptoError)?
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| CryptoError)
}

pub fn encrypt_secret(
    plaintext: &[u8],
    kek: &Kek,
    secret_id: &str,
    secret_name: &str,
) -> EncryptedSecret {
    let mut dek = [0_u8; AES_256_KEY_LENGTH];
    OsRng.fill_bytes(&mut dek);
    EncryptedSecret {
        encrypted_value: encrypt_blob(plaintext, &dek, &create_aad(secret_id, secret_name)),
        wrapped_dek: encrypt_blob(&dek, &kek.key, &[]),
        kek_version: kek.version,
    }
}

pub fn decrypt_secret(
    encrypted_value: &[u8],
    wrapped_dek: &[u8],
    kek: &Kek,
    secret_id: &str,
    secret_name: &str,
) -> Result<Vec<u8>, CryptoError> {
    let dek: [u8; AES_256_KEY_LENGTH] = decrypt_blob(wrapped_dek, &kek.key, &[])?
        .try_into()
        .map_err(|_| CryptoError)?;
    decrypt_blob(encrypted_value, &dek, &create_aad(secret_id, secret_name))
}

/// Python masks by Unicode code point, not UTF-8 byte.
pub fn mask_string_value(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() < 8 {
        return "***".to_string();
    }
    let prefix: String = chars[..3].iter().collect();
    let suffix: String = chars[chars.len() - 4..].iter().collect();
    format!("{prefix}...{suffix}")
}

pub fn mask_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(mask_string_value)
        .unwrap_or_else(|| "***".to_string())
}

pub fn mask_secret_value(secret: &Map<String, Value>) -> Map<String, Value> {
    secret
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(mask_json_value(value))))
        .collect()
}

struct CacheEntry {
    encrypted: Vec<u8>,
    bucket: u64,
    expires_at: SystemTime,
}

#[derive(Default)]
struct BucketKeys {
    active: Option<(u64, [u8; AES_256_KEY_LENGTH])>,
    previous: Option<(u64, [u8; AES_256_KEY_LENGTH])>,
}

struct CacheState {
    entries: HashMap<String, CacheEntry>,
    lru: VecDeque<String>,
    keys: BucketKeys,
}

/// Thread-safe LRU whose values remain encrypted in memory. Keys rotate with
/// the TTL and only the current/previous time buckets are retained, matching
/// Python's one-bucket boundary tolerance and forward-expiry behavior.
pub struct SecretCache {
    ttl: Duration,
    max_size: usize,
    state: Mutex<CacheState>,
}

impl std::fmt::Debug for SecretCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretCache")
            .field("ttl", &self.ttl)
            .field("max_size", &self.max_size)
            .finish_non_exhaustive()
    }
}

impl SecretCache {
    pub fn from_environment() -> Result<Self, String> {
        let ttl = std::env::var(SECRETS_CACHE_TTL_ENV_VAR)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_CACHE_TTL);
        let max_size = std::env::var(SECRETS_CACHE_MAX_SIZE_ENV_VAR)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_CACHE_MAX_SIZE);
        Self::new(ttl, max_size)
    }

    pub fn new(ttl_seconds: u64, max_size: usize) -> Result<Self, String> {
        if !(MIN_CACHE_TTL..=MAX_CACHE_TTL).contains(&ttl_seconds) {
            return Err(format!(
                "Cache TTL must be between {MIN_CACHE_TTL} and {MAX_CACHE_TTL} seconds. Got: {ttl_seconds}. Lower values (10-30s) are more secure but impact performance. Higher values (120-300s) improve performance but increase exposure window."
            ));
        }
        Ok(Self::with_ttl(Duration::from_secs(ttl_seconds), max_size))
    }

    fn with_ttl(ttl: Duration, max_size: usize) -> Self {
        Self {
            ttl,
            max_size,
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                lru: VecDeque::new(),
                keys: BucketKeys::default(),
            }),
        }
    }

    fn bucket(&self, now: SystemTime) -> u64 {
        let divisor = self.ttl.as_secs_f64();
        (now.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            / divisor) as u64
    }

    fn bucket_key(
        keys: &mut BucketKeys,
        requested: u64,
        current: u64,
    ) -> Option<[u8; AES_256_KEY_LENGTH]> {
        if keys
            .active
            .as_ref()
            .is_some_and(|(bucket, _)| *bucket != current)
        {
            let old = keys.active.take();
            keys.previous = old.filter(|(bucket, _)| bucket.saturating_add(1) == current);
        }
        if keys
            .previous
            .as_ref()
            .is_some_and(|(bucket, _)| current.abs_diff(*bucket) > 1)
        {
            keys.previous = None;
        }
        if let Some((_, key)) = keys
            .active
            .as_ref()
            .filter(|(bucket, _)| *bucket == requested)
        {
            return Some(*key);
        }
        if let Some((_, key)) = keys
            .previous
            .as_ref()
            .filter(|(bucket, _)| *bucket == requested)
        {
            return Some(*key);
        }
        if requested != current {
            return None;
        }
        let mut key = [0_u8; AES_256_KEY_LENGTH];
        OsRng.fill_bytes(&mut key);
        keys.active = Some((current, key));
        Some(key)
    }

    pub fn set(&self, cache_key: &str, value: &Value) {
        let now = SystemTime::now();
        let bucket = self.bucket(now);
        let mut state = self.state.lock().expect("secret-cache mutex poisoned");
        let key = Self::bucket_key(&mut state.keys, bucket, bucket)
            .expect("the current bucket always has a key");
        let plaintext = match value {
            Value::String(value) => value.clone(),
            _ => serde_json::to_string(value).expect("JSON values serialize"),
        };
        state.entries.insert(
            cache_key.to_string(),
            CacheEntry {
                encrypted: encrypt_blob(plaintext.as_bytes(), &key, &[]),
                bucket,
                expires_at: now + self.ttl,
            },
        );
        state.lru.retain(|key| key != cache_key);
        state.lru.push_back(cache_key.to_string());
        while state.entries.len() > self.max_size {
            if let Some(oldest) = state.lru.pop_front() {
                state.entries.remove(&oldest);
            }
        }
    }

    pub fn get(&self, cache_key: &str) -> Option<Value> {
        let now = SystemTime::now();
        let current = self.bucket(now);
        let mut state = self.state.lock().expect("secret-cache mutex poisoned");
        let entry = state.entries.remove(cache_key)?;
        if now > entry.expires_at || current.abs_diff(entry.bucket) > 1 {
            state.lru.retain(|key| key != cache_key);
            return None;
        }
        let Some(key) = Self::bucket_key(&mut state.keys, entry.bucket, current) else {
            state.lru.retain(|key| key != cache_key);
            return None;
        };
        let plaintext = decrypt_blob(&entry.encrypted, &key, &[]).ok()?;
        let plaintext = String::from_utf8(plaintext).ok()?;
        let value = if plaintext.starts_with('{') && plaintext.ends_with('}') {
            serde_json::from_str(&plaintext).unwrap_or(Value::String(plaintext))
        } else {
            Value::String(plaintext)
        };
        state.entries.insert(cache_key.to_string(), entry);
        state.lru.retain(|key| key != cache_key);
        state.lru.push_back(cache_key.to_string());
        Some(value)
    }

    pub fn clear(&self) {
        let mut state = self.state.lock().expect("secret-cache mutex poisoned");
        state.entries.clear();
        state.lru.clear();
    }

    pub fn size(&self) -> usize {
        self.state
            .lock()
            .expect("secret-cache mutex poisoned")
            .entries
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct Fixtures {
        cases: Vec<Fixture>,
    }

    #[derive(Deserialize)]
    struct Fixture {
        plaintext: String,
        passphrase: String,
        kek_version: u32,
        secret_id: String,
        secret_name: String,
        encrypted_value_b64: String,
        wrapped_dek_b64: String,
    }

    #[test]
    fn decrypts_python_envelopes_and_rejects_aad_mismatch() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../spikes/fixtures/secrets_python.json");
        let fixtures: Fixtures =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(fixtures.cases.len(), 32);
        for case in fixtures.cases {
            let kek = Kek::derive(&case.passphrase, case.kek_version).unwrap();
            let encrypted = STANDARD.decode(case.encrypted_value_b64).unwrap();
            let wrapped = STANDARD.decode(case.wrapped_dek_b64).unwrap();
            assert_eq!(wrapped.len(), 60);
            assert_eq!(
                decrypt_secret(
                    &encrypted,
                    &wrapped,
                    &kek,
                    &case.secret_id,
                    &case.secret_name
                )
                .unwrap(),
                case.plaintext.as_bytes()
            );
            assert!(
                decrypt_secret(&encrypted, &wrapped, &kek, "wrong", &case.secret_name).is_err()
            );
            assert!(decrypt_secret(&encrypted, &wrapped, &kek, &case.secret_id, "wrong").is_err());
        }
    }

    #[test]
    fn masking_matches_python_edges() {
        assert_eq!(mask_string_value("1234567"), "***");
        assert_eq!(mask_string_value("12345678"), "123...5678");
        assert_eq!(mask_string_value("abc😀efgh"), "abc...efgh");
        assert_eq!(mask_json_value(&Value::Bool(true)), "***");
        assert_eq!(mask_json_value(&Value::Null), "***");
    }

    #[test]
    fn cache_is_lru_and_clear_invalidates() {
        let cache = SecretCache::new(10, 2).unwrap();
        cache.set("one", &serde_json::json!({"api_key": "fake-one"}));
        cache.set("two", &Value::String("fake-two".to_string()));
        assert!(cache.get("one").is_some());
        cache.set("three", &Value::String("fake-three".to_string()));
        assert!(cache.get("two").is_none());
        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn cache_expires_and_drops_unreadable_entry() {
        let cache = SecretCache::with_ttl(Duration::from_millis(2), 2);
        cache.set("secret", &Value::String("obvious-fake".to_string()));
        std::thread::sleep(Duration::from_millis(4));
        assert!(cache.get("secret").is_none());
        assert_eq!(cache.size(), 0);
    }
}
