//! T0.4 crypto spike: prove Rust <-> Python (werkzeug / cryptography.fernet)
//! interop for the MLflow tracking-server reimplementation.
//!
//! Two blockers are covered:
//!
//! 1. Werkzeug password hashes (auth DB stores `method$salt$hexdigest`).
//!    - `scrypt:<N>:<r>:<p>$<salt>$<hexdigest>`  (werkzeug default, dklen=64)
//!    - `pbkdf2:sha256:<iterations>$<salt>$<hexdigest>`  (dklen=32)
//!    The salt is raw ASCII; werkzeug feeds `salt.encode()` straight into the
//!    KDF (it does NOT base64-decode it). The digest is lowercase hex.
//!
//! 2. Fernet tokens (webhook secret encryption, `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`).
//!
//! The functions here are deliberately small and dependency-light so they can
//! be lifted into the real server crate later.

use hmac::Hmac;
use pbkdf2::pbkdf2;
use rand::Rng;
use scrypt::{scrypt, Params as ScryptParams};
use sha2::Sha256;
use subtle::ConstantTimeEq;

/// A parsed werkzeug password hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WerkzeugHash {
    Scrypt {
        n: u32,
        r: u32,
        p: u32,
        salt: String,
        digest: Vec<u8>,
    },
    Pbkdf2Sha256 {
        iterations: u32,
        salt: String,
        digest: Vec<u8>,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum HashError {
    BadStructure(String),
    UnsupportedMethod(String),
    BadNumber(String),
    BadHex(String),
    Kdf(String),
}

impl std::fmt::Display for HashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashError::BadStructure(s) => write!(f, "bad hash structure: {s}"),
            HashError::UnsupportedMethod(s) => write!(f, "unsupported method: {s}"),
            HashError::BadNumber(s) => write!(f, "bad number: {s}"),
            HashError::BadHex(s) => write!(f, "bad hex digest: {s}"),
            HashError::Kdf(s) => write!(f, "kdf error: {s}"),
        }
    }
}

impl std::error::Error for HashError {}

/// Parse a werkzeug password hash string of the form `method$salt$hexdigest`.
///
/// werkzeug splits on the FIRST two `$`; the salt itself is guaranteed by
/// werkzeug not to contain `$` (it draws from `[A-Za-z0-9]`), so a plain
/// 3-way split is correct.
pub fn parse(hash: &str) -> Result<WerkzeugHash, HashError> {
    let parts: Vec<&str> = hash.splitn(3, '$').collect();
    let [method_part, salt, hexdigest] = parts.as_slice() else {
        return Err(HashError::BadStructure(format!(
            "expected method$salt$digest, got {} segments",
            parts.len()
        )));
    };
    let digest = hex::decode(hexdigest).map_err(|e| HashError::BadHex(e.to_string()))?;

    // The method segment is itself `:`-delimited.
    let method_fields: Vec<&str> = method_part.split(':').collect();
    match method_fields.as_slice() {
        ["scrypt", n, r, p] => Ok(WerkzeugHash::Scrypt {
            n: parse_u32(n)?,
            r: parse_u32(r)?,
            p: parse_u32(p)?,
            salt: (*salt).to_string(),
            digest,
        }),
        ["pbkdf2", "sha256", iterations] => Ok(WerkzeugHash::Pbkdf2Sha256 {
            iterations: parse_u32(iterations)?,
            salt: (*salt).to_string(),
            digest,
        }),
        _ => Err(HashError::UnsupportedMethod(method_part.to_string())),
    }
}

fn parse_u32(s: &str) -> Result<u32, HashError> {
    s.parse::<u32>().map_err(|_| HashError::BadNumber(s.to_string()))
}

/// Recompute the KDF output for `password` using the parameters/salt of a
/// parsed hash. Returns the raw digest bytes.
fn derive(hash: &WerkzeugHash, password: &str) -> Result<Vec<u8>, HashError> {
    match hash {
        WerkzeugHash::Scrypt { n, r, p, salt, digest } => {
            // werkzeug: hashlib.scrypt(pw, salt=salt.encode(), n=N, r=r, p=p, dklen=64)
            // scrypt crate takes log2(N) as the cost parameter.
            let log_n = log2_exact(*n)?;
            let params = ScryptParams::new(log_n, *r, *p, digest.len())
                .map_err(|e| HashError::Kdf(e.to_string()))?;
            let mut out = vec![0u8; digest.len()];
            scrypt(password.as_bytes(), salt.as_bytes(), &params, &mut out)
                .map_err(|e| HashError::Kdf(e.to_string()))?;
            Ok(out)
        }
        WerkzeugHash::Pbkdf2Sha256 { iterations, salt, digest } => {
            // werkzeug: hashlib.pbkdf2_hmac('sha256', pw, salt.encode(), iterations)
            // dklen defaults to the hash digest size (32 for sha256).
            let mut out = vec![0u8; digest.len()];
            pbkdf2::<Hmac<Sha256>>(password.as_bytes(), salt.as_bytes(), *iterations, &mut out)
                .map_err(|e| HashError::Kdf(e.to_string()))?;
            Ok(out)
        }
    }
}

fn log2_exact(n: u32) -> Result<u8, HashError> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(HashError::BadNumber(format!("scrypt N={n} is not a power of two")));
    }
    Ok(n.trailing_zeros() as u8)
}

/// Verify a plaintext `password` against a werkzeug hash string.
/// Equivalent to werkzeug's `check_password_hash`.
pub fn verify(hash: &str, password: &str) -> Result<bool, HashError> {
    let parsed = parse(hash)?;
    let expected = match &parsed {
        WerkzeugHash::Scrypt { digest, .. } => digest.clone(),
        WerkzeugHash::Pbkdf2Sha256 { digest, .. } => digest.clone(),
    };
    let actual = derive(&parsed, password)?;
    Ok(actual.ct_eq(&expected).into())
}

fn random_salt(len: usize) -> String {
    // werkzeug's DEFAULT_PBKDF2_ITERATIONS salt alphabet is [A-Za-z0-9], 16 chars.
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char)
        .collect()
}

/// Generate a werkzeug-compatible scrypt hash string that Python's
/// `check_password_hash` will accept. Uses werkzeug's default params.
pub fn generate_scrypt(password: &str) -> Result<String, HashError> {
    const N: u32 = 32768;
    const R: u32 = 8;
    const P: u32 = 1;
    const DKLEN: usize = 64;
    let salt = random_salt(16);
    let params = ScryptParams::new(log2_exact(N)?, R, P, DKLEN)
        .map_err(|e| HashError::Kdf(e.to_string()))?;
    let mut out = vec![0u8; DKLEN];
    scrypt(password.as_bytes(), salt.as_bytes(), &params, &mut out)
        .map_err(|e| HashError::Kdf(e.to_string()))?;
    Ok(format!("scrypt:{N}:{R}:{P}${salt}${}", hex::encode(out)))
}

/// Generate a werkzeug-compatible pbkdf2:sha256 hash string.
pub fn generate_pbkdf2_sha256(password: &str) -> Result<String, HashError> {
    const ITERATIONS: u32 = 1_000_000;
    const DKLEN: usize = 32;
    let salt = random_salt(16);
    let mut out = vec![0u8; DKLEN];
    pbkdf2::<Hmac<Sha256>>(password.as_bytes(), salt.as_bytes(), ITERATIONS, &mut out)
        .map_err(|e| HashError::Kdf(e.to_string()))?;
    Ok(format!("pbkdf2:sha256:{ITERATIONS}${salt}${}", hex::encode(out)))
}

// ---------------------------------------------------------------------------
// Fernet (webhook secret encryption)
// ---------------------------------------------------------------------------

/// Decrypt a Fernet token produced by Python `cryptography.fernet`.
/// `key` is the urlsafe-base64 32-byte key string.
pub fn fernet_decrypt(key: &str, token: &str) -> Result<String, String> {
    let fernet = fernet::Fernet::new(key).ok_or_else(|| "invalid fernet key".to_string())?;
    let plaintext = fernet
        .decrypt(token)
        .map_err(|e| format!("fernet decrypt failed: {e:?}"))?;
    String::from_utf8(plaintext).map_err(|e| format!("plaintext not utf-8: {e}"))
}

/// Encrypt a plaintext into a Fernet token that Python can decrypt.
pub fn fernet_encrypt(key: &str, plaintext: &str) -> Result<String, String> {
    let fernet = fernet::Fernet::new(key).ok_or_else(|| "invalid fernet key".to_string())?;
    Ok(fernet.encrypt(plaintext.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct WerkzeugFixture {
        entries: Vec<Entry>,
    }

    #[derive(Deserialize)]
    struct Entry {
        password: String,
        method: String,
        hash: String,
    }

    #[derive(Deserialize)]
    struct FernetFixture {
        key: String,
        plaintext: String,
        token: String,
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name)
    }

    fn load_werkzeug() -> WerkzeugFixture {
        let raw = std::fs::read_to_string(fixture_path("werkzeug_hashes.json")).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    #[test]
    fn parse_scrypt_default_shape() {
        let h = parse("scrypt:32768:8:1$abcd$00ff").unwrap();
        assert_eq!(
            h,
            WerkzeugHash::Scrypt {
                n: 32768,
                r: 8,
                p: 1,
                salt: "abcd".into(),
                digest: vec![0x00, 0xff],
            }
        );
    }

    #[test]
    fn parse_pbkdf2_shape() {
        let h = parse("pbkdf2:sha256:1000000$abcd$00ff").unwrap();
        assert_eq!(
            h,
            WerkzeugHash::Pbkdf2Sha256 {
                iterations: 1_000_000,
                salt: "abcd".into(),
                digest: vec![0x00, 0xff],
            }
        );
    }

    #[test]
    fn parse_rejects_unknown_method() {
        assert!(matches!(
            parse("md5$abcd$00ff"),
            Err(HashError::UnsupportedMethod(_))
        ));
    }

    /// Direction 1: Rust verifies every Python-werkzeug-generated fixture.
    #[test]
    fn verifies_all_python_fixtures() {
        let fx = load_werkzeug();
        assert!(!fx.entries.is_empty());
        for e in &fx.entries {
            assert!(
                verify(&e.hash, &e.password).unwrap(),
                "Rust failed to verify {} hash for password {:?}",
                e.method,
                e.password
            );
            // A wrong password must be rejected.
            assert!(
                !verify(&e.hash, &format!("{}x", e.password)).unwrap(),
                "Rust accepted a wrong password for {:?}",
                e.password
            );
        }
    }

    /// Direction 2 (self-consistency): Rust-generated hashes verify with the
    /// Rust verifier. The Python-accepts-them proof is in verify.py, exercised
    /// by the `rust_generated_hashes` integration test.
    #[test]
    fn round_trips_rust_generated_hashes() {
        for pw in ["hunter2", "", "üñîçödé-\u{1f510}", &"a".repeat(100)] {
            let s = generate_scrypt(pw).unwrap();
            assert!(verify(&s, pw).unwrap());
            assert!(!verify(&s, "wrong").unwrap());

            let p = generate_pbkdf2_sha256(pw).unwrap();
            assert!(verify(&p, pw).unwrap());
            assert!(!verify(&p, "wrong").unwrap());
        }
    }

    /// Direction 1: Rust decrypts a Python-fernet-generated token.
    #[test]
    fn decrypts_python_fernet_token() {
        let raw = std::fs::read_to_string(fixture_path("fernet.json")).unwrap();
        let fx: FernetFixture = serde_json::from_str(&raw).unwrap();
        let plaintext = fernet_decrypt(&fx.key, &fx.token).unwrap();
        assert_eq!(plaintext, fx.plaintext);
    }

    /// Direction 2 (self-consistency): Rust encrypt -> Rust decrypt.
    #[test]
    fn round_trips_rust_fernet() {
        let key = fernet::Fernet::generate_key();
        let token = fernet_encrypt(&key, "webhook-secret \u{1f511}").unwrap();
        assert_eq!(fernet_decrypt(&key, &token).unwrap(), "webhook-secret \u{1f511}");
    }
}
