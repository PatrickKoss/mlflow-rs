//! Werkzeug-compatible password hashing (`generate_password_hash` /
//! `check_password_hash`).
//!
//! Source of truth: `werkzeug/security.py` (Werkzeug 3.1.8, the version pinned
//! in `uv.lock`), used by `mlflow/server/auth/sqlalchemy_store.py:9,140,147,216`.
//! MLflow calls both functions with **default arguments**, so the hashes it
//! writes are always `scrypt:32768:8:1$<salt>$<hex>`; this module must both
//! **verify** any werkzeug hash (scrypt and pbkdf2, salted) and **generate**
//! hashes werkzeug's `check_password_hash` accepts.
//!
//! ## Hash string format
//!
//! `f"{method}${salt}${hex}"` — three `$`-separated fields, split on the first
//! two `$` (`security.py:138`, `split("$", 2)`):
//!
//! * `method` — `scrypt:<n>:<r>:<p>` or `pbkdf2:<hash_name>:<iterations>`.
//! * `salt` — the raw salt *string* (its UTF-8 bytes are the KDF salt).
//! * `hex` — lowercase hex of the derived key.
//!
//! ## Parameters (werkzeug 3.1.8 defaults)
//!
//! * scrypt: `n = 2**15 = 32768`, `r = 8`, `p = 1`; `maxmem = 132 * n * r * p`
//!   (`security.py:41-58`, `84-86`, `92-93`). Default salt length 16.
//! * pbkdf2: `sha256`, `1_000_000` iterations (`DEFAULT_PBKDF2_ITERATIONS`,
//!   `security.py:10,59-79`).
//!
//! ## Comparison
//!
//! werkzeug compares the recomputed hex against the stored hex with
//! `hmac.compare_digest` (constant-time, `security.py:142`). We mirror that with
//! `subtle::ConstantTimeEq` over the ASCII hex bytes.
//!
//! ## Salt alphabet
//!
//! `SALT_CHARS = a-zA-Z0-9` (62 chars, `security.py:9`); `gen_salt` draws each
//! char uniformly from a CSPRNG (`secrets.choice`, `security.py:28-33`). We draw
//! from the OS CSPRNG with rejection sampling to keep the distribution uniform.

use pbkdf2::pbkdf2_hmac;
use scrypt::{scrypt, Params as ScryptParams};
use sha2::{Sha256, Sha512};
use subtle::ConstantTimeEq;

/// The werkzeug salt alphabet (`SALT_CHARS`, `security.py:9`).
const SALT_CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// werkzeug `generate_password_hash` default `salt_length` (`security.py:85`).
const DEFAULT_SALT_LENGTH: usize = 16;

/// werkzeug scrypt defaults (`security.py:43-45`): `n = 2**15`, `r = 8`, `p = 1`.
const SCRYPT_DEFAULT_LOG_N: u8 = 15;
const SCRYPT_DEFAULT_R: u32 = 8;
const SCRYPT_DEFAULT_P: u32 = 1;

/// werkzeug pbkdf2 default iterations (`DEFAULT_PBKDF2_ITERATIONS`,
/// `security.py:10`, raised to 1,000,000 in werkzeug 3.1).
const DEFAULT_PBKDF2_ITERATIONS: u32 = 1_000_000;

/// A password-hash parse/compute error. These map to the werkzeug `ValueError`
/// paths; `check_password_hash` swallows a malformed hash into `false`, so most
/// callers never see these — they surface only from the (rarely-hit) generate
/// path or an explicitly requested non-default method.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HashError {
    #[error("Invalid hash method '{0}'.")]
    InvalidMethod(String),

    #[error("'scrypt' takes 3 arguments.")]
    ScryptArgs,

    #[error("'pbkdf2' takes 2 arguments.")]
    Pbkdf2Args,

    #[error("Salt length must be at least 1.")]
    EmptySalt,

    #[error("scrypt parameter error: {0}")]
    ScryptParams(String),
}

/// Compute the raw derived-key hex plus the canonical method string for a
/// `method`/`salt`/`password` triple, mirroring `_hash_internal`
/// (`security.py:36-81`). Returns `(hex, canonical_method)`.
fn hash_internal(method: &str, salt: &str, password: &str) -> Result<(String, String), HashError> {
    let salt_bytes = salt.as_bytes();
    let password_bytes = password.as_bytes();

    let mut parts = method.split(':');
    let kdf = parts.next().unwrap_or("");
    let args: Vec<&str> = parts.collect();

    match kdf {
        "scrypt" => {
            let (n, r, p) = match args.as_slice() {
                [] => (
                    1u64 << SCRYPT_DEFAULT_LOG_N,
                    SCRYPT_DEFAULT_R,
                    SCRYPT_DEFAULT_P,
                ),
                [n, r, p] => {
                    let n = n.parse::<u64>().map_err(|_| HashError::ScryptArgs)?;
                    let r = r.parse::<u32>().map_err(|_| HashError::ScryptArgs)?;
                    let p = p.parse::<u32>().map_err(|_| HashError::ScryptArgs)?;
                    (n, r, p)
                }
                _ => return Err(HashError::ScryptArgs),
            };
            let hex = scrypt_hex(password_bytes, salt_bytes, n, r, p)?;
            Ok((hex, format!("scrypt:{n}:{r}:{p}")))
        }
        "pbkdf2" => {
            // 0 args -> sha256/default iters; 1 arg -> hash_name/default iters;
            // 2 args -> hash_name/iters; else error (`security.py:60-72`).
            let (hash_name, iterations) = match args.as_slice() {
                [] => ("sha256", DEFAULT_PBKDF2_ITERATIONS),
                [h] => (*h, DEFAULT_PBKDF2_ITERATIONS),
                [h, i] => (*h, i.parse::<u32>().map_err(|_| HashError::Pbkdf2Args)?),
                _ => return Err(HashError::Pbkdf2Args),
            };
            let hex = pbkdf2_hex(hash_name, password_bytes, salt_bytes, iterations)?;
            Ok((hex, format!("pbkdf2:{hash_name}:{iterations}")))
        }
        other => Err(HashError::InvalidMethod(other.to_string())),
    }
}

/// `hashlib.scrypt(...).hex()` with werkzeug's `maxmem = 132 * n * r * p`.
///
/// The RustCrypto `scrypt` crate takes `log2(n)` and derives its own memory
/// bound; werkzeug's `maxmem` only guards Python's `hashlib` allocation, so it
/// does not affect the derived bytes — only `(n, r, p, salt, password, dklen)`
/// do, and werkzeug uses the scrypt default `dklen = 64`.
fn scrypt_hex(password: &[u8], salt: &[u8], n: u64, r: u32, p: u32) -> Result<String, HashError> {
    if !n.is_power_of_two() || n < 2 {
        return Err(HashError::ScryptParams(format!(
            "n must be a power of two >= 2, got {n}"
        )));
    }
    let log_n = n.trailing_zeros() as u8;
    let params =
        ScryptParams::new(log_n, r, p, 64).map_err(|e| HashError::ScryptParams(e.to_string()))?;
    // hashlib.scrypt default output length (dklen) is 64 bytes.
    let mut out = [0u8; 64];
    scrypt(password, salt, &params, &mut out)
        .map_err(|e| HashError::ScryptParams(e.to_string()))?;
    Ok(to_hex(&out))
}

/// `hashlib.pbkdf2_hmac(hash_name, password, salt, iterations).hex()`.
///
/// werkzeug's derived-key length defaults to the digest size of `hash_name`
/// (`hashlib.pbkdf2_hmac` `dklen=None`), i.e. 32 bytes for sha256, 64 for
/// sha512.
fn pbkdf2_hex(
    hash_name: &str,
    password: &[u8],
    salt: &[u8],
    iterations: u32,
) -> Result<String, HashError> {
    // `pbkdf2_hmac` cannot fail for HMAC (any key length is valid), so no
    // error path here — only an unsupported digest name is rejected.
    match hash_name {
        "sha256" => {
            let mut out = [0u8; 32];
            pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut out);
            Ok(to_hex(&out))
        }
        "sha512" => {
            let mut out = [0u8; 64];
            pbkdf2_hmac::<Sha512>(password, salt, iterations, &mut out);
            Ok(to_hex(&out))
        }
        other => Err(HashError::InvalidMethod(format!("pbkdf2:{other}"))),
    }
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Verify a plaintext `password` against a werkzeug-format `pwhash`, mirroring
/// `check_password_hash` (`security.py:123-142`).
///
/// A malformed hash (missing `$` fields, unknown method) yields `false` rather
/// than an error, exactly as werkzeug returns `False` on `ValueError`. The
/// final comparison is constant-time over the hex bytes.
pub fn check_password_hash(pwhash: &str, password: &str) -> bool {
    // `pwhash.split("$", 2)` -> exactly [method, salt, hashval].
    let mut it = pwhash.splitn(3, '$');
    let (Some(method), Some(salt), Some(hashval)) = (it.next(), it.next(), it.next()) else {
        return false;
    };
    match hash_internal(method, salt, password) {
        Ok((computed, _)) => computed.as_bytes().ct_eq(hashval.as_bytes()).into(),
        Err(_) => false,
    }
}

/// Generate a werkzeug-format password hash for `password` using the default
/// method (`scrypt:32768:8:1`) and a fresh 16-char salt, mirroring
/// `generate_password_hash(password)` (`security.py:84-120`). The result is
/// accepted by werkzeug's `check_password_hash`.
pub fn generate_password_hash(password: &str) -> Result<String, HashError> {
    generate_password_hash_with(password, "scrypt", DEFAULT_SALT_LENGTH)
}

/// Like [`generate_password_hash`] but with an explicit `method` and
/// `salt_length` (the werkzeug default parameters). Used by tests to pin
/// pbkdf2 and to exercise both KDFs.
pub fn generate_password_hash_with(
    password: &str,
    method: &str,
    salt_length: usize,
) -> Result<String, HashError> {
    let salt = gen_salt(salt_length)?;
    let (hex, actual_method) = hash_internal(method, &salt, password)?;
    Ok(format!("{actual_method}${salt}${hex}"))
}

/// `gen_salt(length)` (`security.py:28-33`): `length` chars drawn uniformly
/// from `SALT_CHARS` via a CSPRNG. Rejection sampling keeps the distribution
/// unbiased (matching `secrets.choice`).
fn gen_salt(length: usize) -> Result<String, HashError> {
    if length == 0 {
        return Err(HashError::EmptySalt);
    }
    let n = SALT_CHARS.len() as u8; // 62 chars in the alphabet.
                                    // Largest multiple of n that fits in a u8, for rejection sampling.
    let limit = (u8::MAX / n) * n;
    let mut out = String::with_capacity(length);
    let mut buf = [0u8; 1];
    while out.len() < length {
        os_random(&mut buf);
        let b = buf[0];
        if b < limit {
            out.push(SALT_CHARS[(b % n) as usize] as char);
        }
    }
    Ok(out)
}

/// Fill `buf` with OS CSPRNG bytes. sqlx already pulls `getrandom` transitively,
/// but to avoid a direct dependency we read `/dev/urandom` on unix and fall back
/// to a `RandomState`-seeded stream elsewhere. The salt only needs to be
/// unpredictable, not cryptographically certified beyond the OS RNG.
fn os_random(buf: &mut [u8]) {
    #[cfg(unix)]
    {
        use std::io::Read;
        if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
            if f.read_exact(buf).is_ok() {
                return;
            }
        }
    }
    // Fallback: hash-based stream seeded from the system RNG state + time.
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    for (i, b) in buf.iter_mut().enumerate() {
        hasher.write_usize(i);
        *b = (hasher.finish() & 0xff) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_then_check_scrypt_round_trips() {
        let h = generate_password_hash("correct horse battery staple").unwrap();
        assert!(h.starts_with("scrypt:32768:8:1$"));
        assert!(check_password_hash(&h, "correct horse battery staple"));
        assert!(!check_password_hash(&h, "wrong password here!!"));
    }

    #[test]
    fn generate_then_check_pbkdf2_round_trips() {
        let h = generate_password_hash_with("hunter2hunter2", "pbkdf2:sha256", 16).unwrap();
        assert!(h.starts_with("pbkdf2:sha256:1000000$"));
        assert!(check_password_hash(&h, "hunter2hunter2"));
        assert!(!check_password_hash(&h, "nope"));
    }

    #[test]
    fn salt_is_default_length_and_alphabet() {
        let h = generate_password_hash("passwordpassword").unwrap();
        // scrypt:32768:8:1$<16-char-salt>$<hex>
        let salt = h.splitn(3, '$').nth(1).unwrap();
        assert_eq!(salt.len(), 16);
        assert!(salt.bytes().all(|b| SALT_CHARS.contains(&b)));
    }

    #[test]
    fn malformed_hash_is_false_not_panic() {
        assert!(!check_password_hash("not-a-hash", "x"));
        assert!(!check_password_hash("scrypt:32768:8:1$onlytwo", "x"));
        assert!(!check_password_hash("bogus:1$salt$deadbeef", "x"));
    }

    #[test]
    fn known_pbkdf2_vector_matches_hashlib() {
        // Independently computed with Python:
        //   hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 1000).hex()
        // == "632c2812e46d4604102ba7618e9d6d7d2f8128f6266b4a03264d2a0460b7dcb3"
        let hex = super::pbkdf2_hex("sha256", b"password", b"salt", 1000).unwrap();
        assert_eq!(
            hex,
            "632c2812e46d4604102ba7618e9d6d7d2f8128f6266b4a03264d2a0460b7dcb3"
        );
    }

    #[test]
    fn known_scrypt_vector_matches_hashlib() {
        // RFC 7914 test vector (n=1024, r=8, p=16, dklen=64) — also what
        // hashlib.scrypt(b"password", salt=b"NaCl", n=1024, r=8, p=16).hex()
        // returns. Confirms our (n,r,p,dklen) wiring is byte-exact.
        let hex = super::scrypt_hex(b"password", b"NaCl", 1024, 8, 16).unwrap();
        assert_eq!(
            hex,
            "fdbabe1c9d3472007856e7190d01e9fe7c6ad7cbc8237830e77376634b373162\
             2eaf30d92e22a3886ff109279d9830dac727afb94a83ee6d8360cbdfa2cc0640"
        );
    }
}
