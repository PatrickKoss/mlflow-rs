# Crypto Spikes — Rust/Python Interop

Proves two hard blockers for the Rust MLflow tracking-server reimplementation
(plan §4.13, D13) are feasible:

1. **Werkzeug password-hash compatibility** — Rust must both *verify* hashes
   produced by Python `werkzeug.security.generate_password_hash` (the auth DB
   format, see `mlflow/server/auth/sqlalchemy_store.py`) and *generate* hashes
   that Python `check_password_hash` accepts.
2. **Fernet compatibility** — Rust must decrypt/encrypt Fernet tokens
   interchangeably with Python `cryptography.fernet`, used to encrypt webhook
   secrets via `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`
   (see `mlflow/store/model_registry/dbmodels/models.py`, `EncryptedString`).

This package is **standalone**: its `Cargo.toml` has an empty `[workspace]`
table so it does not join the parent `rust/` workspace.

## Observed Python environment (this repo)

| Library        | Version  | Source                                   |
| -------------- | -------- | ---------------------------------------- |
| `werkzeug`     | 3.1.8    | `importlib.metadata.version('werkzeug')` |
| `cryptography` | 46.0.7   | `importlib.metadata.version(...)`        |

**Default hash method** observed for `generate_password_hash('test')`:
`scrypt` (werkzeug ≥ 2.3 default). Example:

```
scrypt:32768:8:1$gy6UKUSHvEdBefij$ba9c801b2de2...9330ad   (128 hex chars)
```

MLflow's auth store (`mlflow/server/auth/sqlalchemy_store.py`) calls
`generate_password_hash(password)` with **no `method=` argument**, so it inherits
whatever werkzeug's default is — currently scrypt. A real auth DB may also
contain `pbkdf2:sha256` hashes (older werkzeug default / explicit choice), so
both are handled.

## Hash format grammar

Werkzeug stores `method$salt$hexdigest`, split on the first two `$`. The salt
is drawn from `[A-Za-z0-9]` (16 chars by default) so it never contains `$` — a
plain 3-way split is safe. The salt is fed to the KDF as **raw ASCII bytes**
(`salt.encode()`), NOT base64-decoded. The digest is **lowercase hex**.

```
hash        := method_part "$" salt "$" hexdigest
method_part := "scrypt:" N ":" r ":" p
             | "pbkdf2:sha256:" iterations
salt        := [A-Za-z0-9]{16}          ; werkzeug default length
hexdigest   := HEX                        ; scrypt -> 128 chars (64 bytes)
                                          ; pbkdf2:sha256 -> 64 chars (32 bytes)
```

### Parameters observed

| Method          | Params seen                     | KDF call (werkzeug internals)                                     | dklen |
| --------------- | ------------------------------- | ---------------------------------------------------------------- | ----- |
| `scrypt`        | `N=32768, r=8, p=1`             | `hashlib.scrypt(pw, salt=salt, n=N, r=r, p=p, maxmem=…, dklen=64)`| 64 B  |
| `pbkdf2:sha256` | `iterations=1000000`            | `hashlib.pbkdf2_hmac('sha256', pw, salt, iterations)` (dklen=32) | 32 B  |

> **scrypt cost param gotcha:** werkzeug's format stores `N` (the raw work
> factor, 32768), but the RustCrypto `scrypt` crate's `Params::new` takes
> `log_n = log2(N)` (= 15). The spike converts via `n.trailing_zeros()` and
> rejects non-power-of-two `N`.

## Rust crates (RustCrypto + fernet)

Locked versions (`Cargo.lock`):

| Crate       | Version | Role                                   |
| ----------- | ------- | -------------------------------------- |
| `scrypt`    | 0.11.0  | scrypt KDF                             |
| `pbkdf2`    | 0.12.2  | PBKDF2 KDF                             |
| `sha2`      | 0.10.9  | SHA-256 for PBKDF2-HMAC                |
| `hmac`      | 0.12.1  | HMAC-SHA256 wrapper                    |
| `hex`       | 0.4.3   | hex encode/decode of digests           |
| `subtle`    | 2.6.1   | constant-time digest comparison        |
| `rand`      | 0.8.7   | salt generation                        |
| `fernet`    | 0.2.2   | Fernet decrypt/encrypt                 |
| `base64`    | 0.22.1  | (available; fernet handles b64 itself) |

## Cross-language verification results

Both directions, both algorithms, plus Fernet — all **executed and passing**.

### Direction 1 — Rust verifies Python output (`cargo test`)

- `verifies_all_python_fixtures`: Rust `verify()` accepts all 10 werkzeug
  fixtures (5 passwords × {scrypt, pbkdf2:sha256}, incl. empty, 128-char, and
  unicode+emoji) and rejects wrong passwords.
- `decrypts_python_fernet_token`: Rust `fernet_decrypt` recovers the exact
  Python plaintext (unicode key emoji included).

```
running 7 tests
test tests::parse_scrypt_default_shape ... ok
test tests::parse_pbkdf2_shape ... ok
test tests::parse_rejects_unknown_method ... ok
test tests::decrypts_python_fernet_token ... ok
test tests::round_trips_rust_fernet ... ok
test tests::verifies_all_python_fixtures ... ok
test tests::round_trips_rust_generated_hashes ... ok
test result: ok. 7 passed; 0 failed
```

### Direction 2 — Python verifies Rust output (`verify.py`)

Rust generates hashes/tokens (`emit_artifacts` binary), Python `werkzeug.
check_password_hash` / `cryptography.fernet` verify them:

```
[werkzeug] OK password='hunter2'                      scrypt         ...
[werkzeug] OK password='hunter2'                      pbkdf2:sha256  ...
[werkzeug] OK password=''                             scrypt / pbkdf2 ...
[werkzeug] OK password='üñîçödé-🔐'                    scrypt / pbkdf2 ...
[werkzeug] OK password='correct horse battery staple' scrypt / pbkdf2 ...
[fernet]   OK decrypted='rust-encrypted-webhook-secret 🔑'
All Python-side verifications passed.   (exit 0)
```

## T15.3 gateway-secret envelope

`gen_secrets_fixtures.py` calls the implementation in
`mlflow/utils/crypto.py` directly. Its checked-in `secrets_python.json`
contains 32 Python envelopes (4 plaintexts × 2 passphrases × 2 AAD pairs × 2
KEK versions), two Python KEK rotations, and 19 masking fixtures. The
plaintexts cover empty, ASCII, Unicode, and a 4 KiB value.

`src/secrets.rs` reproduces the database representation:

- A fresh random 32-byte DEK encrypts each value with AES-256-GCM.
- `encrypted_value` is `nonce(12) || ciphertext || tag(16)`.
- The KEK wraps the 32-byte DEK without AAD, so `wrapped_dek` is always 60
  bytes with the same nonce/ciphertext/tag layout.
- The KEK is PBKDF2-HMAC-SHA256(passphrase UTF-8, 600,000 iterations, 32-byte
  output). Its salt is `b"mlflow-secrets-kek-v1-2025"` followed by
  `kek_version` encoded as an unsigned four-byte big-endian integer.
- Value AAD is UTF-8 `secret_id + "|" + secret_name`; it is authenticated and
  is not stored in either encrypted byte field.
- Rotation unwraps the DEK with the old KEK and wraps it with the new KEK. It
  leaves `encrypted_value` byte-for-byte unchanged.

`emit_secrets_artifacts` produces the reverse 32-case matrix in
`secrets_rust.json`, plus two Rust rotations. `verify_secrets.py` decrypts all
of them through Python's `_decrypt_secret`. Rust tests decrypt every Python
fixture, compare masking output, and verify wrong AAD, wrong KEK, truncated
value/wrapped-DEK, and corrupted value/wrapped-DEK errors fail closed with a
constant message that contains no plaintext.

## Reproduce

```bash
# 1. (Re)generate Python fixtures  (Rust reads these)
uv run --frozen python rust/spikes/gen_fixtures.py

# 2. Direction 1: Rust verifies Python fixtures
cd rust/spikes && cargo test

# 3. Direction 2: emit Rust artifacts, Python verifies them
cargo run --release --bin emit_artifacts -- /tmp/rust_artifacts.json
cd ../.. && uv run --frozen python rust/spikes/verify.py /tmp/rust_artifacts.json

# 4. T15.3 Python -> Rust fixtures and spike tests
uv run --frozen python rust/spikes/gen_secrets_fixtures.py
cargo test --manifest-path rust/spikes/Cargo.toml

# 5. T15.3 Rust -> Python fixture and verification
cargo run --release --manifest-path rust/spikes/Cargo.toml \
  --bin emit_secrets_artifacts -- rust/spikes/fixtures/secrets_rust.json
uv run --frozen python rust/spikes/verify_secrets.py
```

## Risks / not-yet-covered variants

- **Only `scrypt` and `pbkdf2:sha256` are parsed.** werkzeug also supports
  `pbkdf2:sha1` / `pbkdf2:sha512` and, historically, `plain`/salted-hash
  methods. Any real DB using a non-sha256 pbkdf2 variant or a legacy method
  would need extra arms in `parse()`/`derive()`. The parser rejects these
  loudly (`UnsupportedMethod`) rather than silently verifying incorrectly.
- **Default method drift.** werkzeug's default has changed across versions
  (pbkdf2 → scrypt). The Rust generator hardcodes werkzeug 3.1.8 defaults
  (`scrypt:32768:8:1`, `pbkdf2:sha256:1000000`). If the server pins a
  different werkzeug and we want byte-identical params, revisit the constants.
  (Verification is param-driven and unaffected — it reads N/r/p/iterations from
  the stored hash.)
- **scrypt memory/perf.** `N=32768, r=8, p=1` ≈ 32 MiB per hash; 1M-iteration
  pbkdf2 is ~1s unoptimized (fast in `--release`). Fine for login, but the auth
  path should keep werkzeug's existing rate-limit/caching behaviour.
- **Salt alphabet.** The Rust generator uses `[A-Za-z0-9]` (matches werkzeug).
  Not security-load-bearing for verification, but keeps generated hashes
  visually identical to werkzeug's.
- **`fernet` crate 0.2.2** does not expose key rotation (`MultiFernet`). MLflow
  currently uses a single key, so this is fine today; multi-key rotation would
  need a different crate or manual layering.
```
