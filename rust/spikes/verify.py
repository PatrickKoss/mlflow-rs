"""Verify Rust-generated crypto artifacts from the Python side.

The Rust tests emit artifacts (hashes / fernet tokens produced by the Rust
crates) to a JSON file; this script feeds them to the *authoritative* Python
implementations (werkzeug.check_password_hash / cryptography.fernet) and exits
non-zero if any fail. This closes the round-trip: Rust -> Python.

Usage:
    uv run --frozen python rust/spikes/verify.py <artifacts.json>

Artifacts schema:
    {
      "werkzeug": [{"password": "...", "hash": "..."}, ...],
      "fernet":   {"key": "...", "token": "...", "expected_plaintext": "..."}
    }
"""

import json
import sys
from pathlib import Path

from cryptography.fernet import Fernet
from werkzeug.security import check_password_hash


def main() -> int:
    artifacts = json.loads(Path(sys.argv[1]).read_text())
    failures = []

    for item in artifacts.get("werkzeug", []):
        ok = check_password_hash(item["hash"], item["password"])
        status = "OK" if ok else "FAIL"
        print(f"[werkzeug] {status} password={item['password']!r} hash={item['hash'][:40]}...")
        if not ok:
            failures.append(f"werkzeug hash rejected for password {item['password']!r}")
        # Negative control: a wrong password must be rejected.
        if check_password_hash(item["hash"], item["password"] + "x"):
            failures.append(f"werkzeug hash accepted a WRONG password for {item['password']!r}")

    fernet = artifacts.get("fernet")
    if fernet:
        plaintext = Fernet(fernet["key"].encode()).decrypt(fernet["token"].encode()).decode()
        ok = plaintext == fernet["expected_plaintext"]
        print(f"[fernet]   {'OK' if ok else 'FAIL'} decrypted={plaintext!r}")
        if not ok:
            failures.append(
                f"fernet plaintext mismatch: got {plaintext!r} "
                f"expected {fernet['expected_plaintext']!r}"
            )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll Python-side verifications passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
