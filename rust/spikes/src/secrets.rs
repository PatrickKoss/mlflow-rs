//! T15.3 spike for the gateway-secret envelope in `mlflow/utils/crypto.py`.

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

const DECRYPTION_ERROR: &str = "secret decryption failed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CryptoError;

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(DECRYPTION_ERROR)
    }
}

impl std::error::Error for CryptoError {}

#[derive(Clone)]
pub struct Kek {
    key: [u8; AES_256_KEY_LENGTH],
    pub version: u32,
}

impl Kek {
    /// Match Python's PBKDF2-HMAC-SHA256 derivation. The version is appended to
    /// the fixed salt as four unsigned big-endian bytes.
    pub fn derive(passphrase: &str, version: u32) -> Result<Self, CryptoError> {
        let mut versioned_salt = Vec::with_capacity(MLFLOW_KEK_SALT.len() + 4);
        versioned_salt.extend_from_slice(MLFLOW_KEK_SALT);
        versioned_salt.extend_from_slice(&version.to_be_bytes());

        let mut key = [0_u8; AES_256_KEY_LENGTH];
        pbkdf2::<Hmac<Sha256>>(
            passphrase.as_bytes(),
            &versioned_salt,
            PBKDF2_ITERATIONS,
            &mut key,
        )
        .map_err(|_| CryptoError)?;
        Ok(Self { key, version })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedSecret {
    pub encrypted_value: Vec<u8>,
    pub wrapped_dek: Vec<u8>,
    pub kek_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotatedSecret {
    pub encrypted_value: Vec<u8>,
    pub wrapped_dek: Vec<u8>,
}

pub fn create_aad(secret_id: &str, secret_name: &str) -> Vec<u8> {
    format!("{secret_id}|{secret_name}").into_bytes()
}

fn encrypt_blob(plaintext: &[u8], key: &[u8; AES_256_KEY_LENGTH], aad: &[u8]) -> Vec<u8> {
    let cipher = Aes256Gcm::new_from_slice(key).expect("AES-256 key has a fixed valid length");
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
        .expect("AES-GCM encryption with valid inputs cannot fail");

    let mut result = Vec::with_capacity(GCM_NONCE_LENGTH + ciphertext.len());
    result.extend_from_slice(&nonce);
    result.extend_from_slice(&ciphertext);
    result
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
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError)?;
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| CryptoError)
}

fn wrap_dek(dek: &[u8; AES_256_KEY_LENGTH], kek: &Kek) -> Vec<u8> {
    encrypt_blob(dek, &kek.key, &[])
}

fn unwrap_dek(wrapped_dek: &[u8], kek: &Kek) -> Result<[u8; AES_256_KEY_LENGTH], CryptoError> {
    let dek = decrypt_blob(wrapped_dek, &kek.key, &[])?;
    dek.try_into().map_err(|_| CryptoError)
}

pub fn encrypt_secret(
    plaintext: &[u8],
    kek: &Kek,
    secret_id: &str,
    secret_name: &str,
) -> EncryptedSecret {
    let mut dek = [0_u8; AES_256_KEY_LENGTH];
    OsRng.fill_bytes(&mut dek);
    let aad = create_aad(secret_id, secret_name);
    EncryptedSecret {
        encrypted_value: encrypt_blob(plaintext, &dek, &aad),
        wrapped_dek: wrap_dek(&dek, kek),
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
    let dek = unwrap_dek(wrapped_dek, kek)?;
    decrypt_blob(encrypted_value, &dek, &create_aad(secret_id, secret_name))
}

pub fn rotate_secret_encryption(
    encrypted_value: &[u8],
    wrapped_dek: &[u8],
    old_kek: &Kek,
    new_kek: &Kek,
) -> Result<RotatedSecret, CryptoError> {
    let dek = unwrap_dek(wrapped_dek, old_kek)?;
    Ok(RotatedSecret {
        encrypted_value: encrypted_value.to_vec(),
        wrapped_dek: wrap_dek(&dek, new_kek),
    })
}

/// Python counts Unicode code points for `len` and slicing; Rust `chars`
/// provides the corresponding behavior for valid UTF-8 strings.
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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct FixtureFile {
        cases: Vec<FixtureCase>,
        rotations: Vec<RotationFixture>,
        masking: MaskingFixtures,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        plaintext: String,
        passphrase: String,
        kek_version: u32,
        secret_id: String,
        secret_name: String,
        aad_b64: String,
        encrypted_value_b64: String,
        wrapped_dek_b64: String,
    }

    #[derive(Deserialize)]
    struct RotationFixture {
        plaintext: String,
        secret_id: String,
        secret_name: String,
        old_passphrase: String,
        old_kek_version: u32,
        new_passphrase: String,
        new_kek_version: u32,
        encrypted_value_b64: String,
        old_wrapped_dek_b64: String,
        new_wrapped_dek_b64: String,
    }

    #[derive(Deserialize)]
    struct MaskingFixtures {
        strings: Vec<MaskStringFixture>,
        dictionaries: Vec<MaskDictionaryFixture>,
    }

    #[derive(Deserialize)]
    struct MaskStringFixture {
        input: Value,
        masked: String,
    }

    #[derive(Deserialize)]
    struct MaskDictionaryFixture {
        input: Map<String, Value>,
        masked: Map<String, Value>,
    }

    fn load_fixtures() -> FixtureFile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("secrets_python.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn decode(value: &str) -> Vec<u8> {
        STANDARD.decode(value).unwrap()
    }

    #[test]
    fn decrypts_every_python_envelope() {
        let fixtures = load_fixtures();
        assert_eq!(fixtures.cases.len(), 32);
        let mut keks = HashMap::new();
        for case in fixtures.cases {
            let kek = keks
                .entry((case.passphrase.clone(), case.kek_version))
                .or_insert_with(|| Kek::derive(&case.passphrase, case.kek_version).unwrap());
            assert_eq!(
                create_aad(&case.secret_id, &case.secret_name),
                decode(&case.aad_b64)
            );
            assert_eq!(
                decode(&case.encrypted_value_b64).len(),
                case.plaintext.len() + GCM_NONCE_LENGTH + GCM_TAG_LENGTH
            );
            assert_eq!(decode(&case.wrapped_dek_b64).len(), 60);
            let plaintext = decrypt_secret(
                &decode(&case.encrypted_value_b64),
                &decode(&case.wrapped_dek_b64),
                kek,
                &case.secret_id,
                &case.secret_name,
            )
            .unwrap();
            assert_eq!(plaintext, case.plaintext.as_bytes());
        }
    }

    #[test]
    fn matches_python_kek_rotation() {
        let fixtures = load_fixtures();
        assert_eq!(fixtures.rotations.len(), 2);
        for case in fixtures.rotations {
            let old_kek = Kek::derive(&case.old_passphrase, case.old_kek_version).unwrap();
            let new_kek = Kek::derive(&case.new_passphrase, case.new_kek_version).unwrap();
            let encrypted_value = decode(&case.encrypted_value_b64);
            let old_wrapped_dek = decode(&case.old_wrapped_dek_b64);
            let expected_new_wrapped_dek = decode(&case.new_wrapped_dek_b64);

            let rotated =
                rotate_secret_encryption(&encrypted_value, &old_wrapped_dek, &old_kek, &new_kek)
                    .unwrap();
            assert_eq!(rotated.encrypted_value, encrypted_value);
            assert_eq!(
                decrypt_secret(
                    &rotated.encrypted_value,
                    &rotated.wrapped_dek,
                    &new_kek,
                    &case.secret_id,
                    &case.secret_name,
                )
                .unwrap(),
                case.plaintext.as_bytes()
            );
            assert_eq!(
                decrypt_secret(
                    &encrypted_value,
                    &expected_new_wrapped_dek,
                    &new_kek,
                    &case.secret_id,
                    &case.secret_name,
                )
                .unwrap(),
                case.plaintext.as_bytes()
            );
        }
    }

    #[test]
    fn masking_matches_python_fixtures() {
        let fixtures = load_fixtures();
        for case in fixtures.masking.strings {
            assert_eq!(
                mask_json_value(&case.input).as_bytes(),
                case.masked.as_bytes()
            );
        }
        for case in fixtures.masking.dictionaries {
            assert_eq!(mask_secret_value(&case.input), case.masked);
        }
    }

    fn sample_envelope() -> (String, Kek, EncryptedSecret) {
        let plaintext = "do-not-leak-this-plaintext".to_string();
        let kek = Kek::derive("correct test passphrase", 1).unwrap();
        let encrypted = encrypt_secret(plaintext.as_bytes(), &kek, "secret-id", "secret-name");
        (plaintext, kek, encrypted)
    }

    fn assert_closed(error: CryptoError, plaintext: &str) {
        let message = error.to_string();
        assert_eq!(message, DECRYPTION_ERROR);
        assert!(!message.contains(plaintext));
    }

    #[test]
    fn wrong_aad_fails_closed_without_plaintext_leak() {
        let (plaintext, kek, encrypted) = sample_envelope();
        let error = decrypt_secret(
            &encrypted.encrypted_value,
            &encrypted.wrapped_dek,
            &kek,
            "secret-id",
            "wrong-name",
        )
        .unwrap_err();
        assert_closed(error, &plaintext);
    }

    #[test]
    fn wrong_kek_fails_closed_without_plaintext_leak() {
        let (plaintext, _kek, encrypted) = sample_envelope();
        let wrong_kek = Kek::derive("wrong test passphrase", 1).unwrap();
        let error = decrypt_secret(
            &encrypted.encrypted_value,
            &encrypted.wrapped_dek,
            &wrong_kek,
            "secret-id",
            "secret-name",
        )
        .unwrap_err();
        assert_closed(error, &plaintext);
    }

    #[test]
    fn truncated_envelopes_fail_closed_without_plaintext_leak() {
        let (plaintext, kek, encrypted) = sample_envelope();
        for (value, wrapped) in [
            (
                &encrypted.encrypted_value[..11],
                encrypted.wrapped_dek.as_slice(),
            ),
            (
                encrypted.encrypted_value.as_slice(),
                &encrypted.wrapped_dek[..27],
            ),
        ] {
            let error =
                decrypt_secret(value, wrapped, &kek, "secret-id", "secret-name").unwrap_err();
            assert_closed(error, &plaintext);
        }
    }

    #[test]
    fn corrupted_envelopes_fail_closed_without_plaintext_leak() {
        let (plaintext, kek, encrypted) = sample_envelope();
        let mut corrupted_value = encrypted.encrypted_value.clone();
        *corrupted_value.last_mut().unwrap() ^= 0x80;
        let error = decrypt_secret(
            &corrupted_value,
            &encrypted.wrapped_dek,
            &kek,
            "secret-id",
            "secret-name",
        )
        .unwrap_err();
        assert_closed(error, &plaintext);

        let mut corrupted_dek = encrypted.wrapped_dek.clone();
        *corrupted_dek.last_mut().unwrap() ^= 0x80;
        let error = decrypt_secret(
            &encrypted.encrypted_value,
            &corrupted_dek,
            &kek,
            "secret-id",
            "secret-name",
        )
        .unwrap_err();
        assert_closed(error, &plaintext);
    }
}
