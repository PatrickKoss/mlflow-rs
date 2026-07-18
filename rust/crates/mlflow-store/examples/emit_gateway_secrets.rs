//! Emit production `mlflow-store` envelopes for Python round-trip verification.

use std::path::PathBuf;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use mlflow_store::{create_aad, encrypt_secret, Kek};
use serde_json::json;

fn main() {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("output path");
    let plaintexts = [
        "",
        "obvious-fake-api-key-123456",
        "obvious fake unicode 🔐 密钥",
        "obvious-fake-long-value-0123456789abcdef0123456789abcdef",
    ];
    let passphrases = [
        "obvious-fake-alpha-passphrase-at-least-32-characters",
        "obvious-fake-beta-passphrase-at-least-32-characters",
    ];
    let versions = [1_u32, 42_u32];
    let aads = [
        ("fake-secret-id", "fake-secret-name"),
        ("秘密-id-🔐", "名前|fake"),
    ];
    let mut cases = Vec::new();
    for plaintext in plaintexts {
        for passphrase in passphrases {
            for version in versions {
                let kek = Kek::derive(passphrase, version).expect("derive fake KEK");
                for (secret_id, secret_name) in aads {
                    let encrypted =
                        encrypt_secret(plaintext.as_bytes(), &kek, secret_id, secret_name);
                    cases.push(json!({
                        "case_id": format!("production-rust-{:02}", cases.len()),
                        "plaintext": plaintext,
                        "passphrase": passphrase,
                        "kek_version": version,
                        "secret_id": secret_id,
                        "secret_name": secret_name,
                        "aad_b64": STANDARD.encode(create_aad(secret_id, secret_name)),
                        "encrypted_value_b64": STANDARD.encode(encrypted.encrypted_value),
                        "wrapped_dek_b64": STANDARD.encode(encrypted.wrapped_dek),
                    }));
                }
            }
        }
    }
    let document = json!({"cases": cases, "rotations": []});
    std::fs::write(output, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
}
