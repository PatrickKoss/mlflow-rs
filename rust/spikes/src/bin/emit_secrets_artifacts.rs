//! Emit Rust-produced T15.3 envelopes for Python-side verification.

use std::path::PathBuf;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use mlflow_crypto_spike::secrets::{create_aad, encrypt_secret, rotate_secret_encryption, Kek};
use serde_json::{json, Value};

const PASSPHRASES: [&str; 2] = [
    "alpha-test-passphrase-with-at-least-32-characters",
    "βeta-test-passphrase-with-unicode-and-sufficient-length",
];
const AADS: [(&str, &str); 2] = [
    ("123e4567-e89b-12d3-a456-426614174000", "provider-api-key"),
    ("秘密-id-🔐", "名前|with-delimiter"),
];
const VERSIONS: [u32; 2] = [1, 42];

fn b64(value: &[u8]) -> String {
    STANDARD.encode(value)
}

fn main() {
    let out_path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: emit_secrets_artifacts <out.json>");
    let plaintexts = [
        String::new(),
        "sk-test-1234567890abcdef".to_string(),
        "🔐 Secret with emoji 密钥 and pässwörd".to_string(),
        format!("long:{}", "0123456789abcdef".repeat(256)),
    ];
    let keks: Vec<(&str, u32, Kek)> = PASSPHRASES
        .iter()
        .flat_map(|passphrase| {
            VERSIONS.iter().map(|version| {
                (
                    *passphrase,
                    *version,
                    Kek::derive(passphrase, *version).expect("derive KEK"),
                )
            })
        })
        .collect();

    let mut cases = Vec::new();
    for plaintext in &plaintexts {
        for (passphrase, version, kek) in &keks {
            for (secret_id, secret_name) in AADS {
                let encrypted = encrypt_secret(plaintext.as_bytes(), kek, secret_id, secret_name);
                cases.push(json!({
                    "case_id": format!("rust-{:02}", cases.len()),
                    "plaintext": plaintext,
                    "passphrase": passphrase,
                    "kek_version": version,
                    "secret_id": secret_id,
                    "secret_name": secret_name,
                    "aad_b64": b64(&create_aad(secret_id, secret_name)),
                    "encrypted_value_b64": b64(&encrypted.encrypted_value),
                    "wrapped_dek_b64": b64(&encrypted.wrapped_dek),
                }));
            }
        }
    }

    let old_kek = Kek::derive(PASSPHRASES[0], VERSIONS[0]).expect("derive old KEK");
    let new_kek = Kek::derive(PASSPHRASES[1], VERSIONS[1]).expect("derive new KEK");
    let mut rotations = Vec::new();
    for (index, plaintext) in plaintexts[1..3].iter().enumerate() {
        let (secret_id, secret_name) = AADS[index];
        let encrypted = encrypt_secret(plaintext.as_bytes(), &old_kek, secret_id, secret_name);
        let rotated = rotate_secret_encryption(
            &encrypted.encrypted_value,
            &encrypted.wrapped_dek,
            &old_kek,
            &new_kek,
        )
        .expect("rotate DEK");
        rotations.push(json!({
            "case_id": format!("rust-rotation-{index}"),
            "plaintext": plaintext,
            "secret_id": secret_id,
            "secret_name": secret_name,
            "old_passphrase": PASSPHRASES[0],
            "old_kek_version": VERSIONS[0],
            "new_passphrase": PASSPHRASES[1],
            "new_kek_version": VERSIONS[1],
            "encrypted_value_b64": b64(&rotated.encrypted_value),
            "old_wrapped_dek_b64": b64(&encrypted.wrapped_dek),
            "new_wrapped_dek_b64": b64(&rotated.wrapped_dek),
        }));
    }

    let document: Value = json!({
        "format": {
            "cipher": "AES-256-GCM",
            "nonce_bytes": 12,
            "tag_bytes": 16,
            "dek_bytes": 32,
            "pbkdf2_hash": "HMAC-SHA256",
            "pbkdf2_iterations": 600000,
            "kek_salt_b64": b64(b"mlflow-secrets-kek-v1-2025"),
            "kek_version_encoding": "unsigned-u32-big-endian appended to salt",
            "aad_encoding": "utf-8(secret_id + '|' + secret_name)",
        },
        "cases": cases,
        "rotations": rotations,
    });
    std::fs::write(
        &out_path,
        format!("{}\n", serde_json::to_string_pretty(&document).unwrap()),
    )
    .expect("write artifact fixture");
    eprintln!(
        "wrote {} envelopes and {} rotations to {}",
        document["cases"].as_array().unwrap().len(),
        document["rotations"].as_array().unwrap().len(),
        out_path.display()
    );
}
