//! Emit Rust-generated crypto artifacts as JSON for Python-side verification.
//!
//!     cargo run --bin emit_artifacts -- /path/to/artifacts.json
//!
//! Then: `uv run --frozen python verify.py /path/to/artifacts.json`
//! proves the Rust -> Python direction (werkzeug + fernet).

use std::io::Write;
use std::path::PathBuf;

use mlflow_crypto_spike::{fernet_encrypt, generate_pbkdf2_sha256, generate_scrypt};

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let out_path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: emit_artifacts <out.json>");

    let passwords = ["hunter2", "", "üñîçödé-\u{1f510}", "correct horse battery staple"];

    let mut werkzeug_items = Vec::new();
    for pw in &passwords {
        for hash in [
            generate_scrypt(pw).expect("scrypt gen"),
            generate_pbkdf2_sha256(pw).expect("pbkdf2 gen"),
        ] {
            werkzeug_items.push(format!(
                "{{\"password\":\"{}\",\"hash\":\"{}\"}}",
                json_escape(pw),
                json_escape(&hash)
            ));
        }
    }

    let fernet_key = fernet::Fernet::generate_key();
    let fernet_plaintext = "rust-encrypted-webhook-secret \u{1f511}";
    let fernet_token = fernet_encrypt(&fernet_key, fernet_plaintext).expect("fernet encrypt");

    let doc = format!(
        "{{\"werkzeug\":[{}],\"fernet\":{{\"key\":\"{}\",\"token\":\"{}\",\"expected_plaintext\":\"{}\"}}}}",
        werkzeug_items.join(","),
        json_escape(&fernet_key),
        json_escape(&fernet_token),
        json_escape(fernet_plaintext),
    );

    let mut f = std::fs::File::create(&out_path).expect("create out file");
    f.write_all(doc.as_bytes()).expect("write");
    eprintln!("wrote artifacts to {}", out_path.display());
}
