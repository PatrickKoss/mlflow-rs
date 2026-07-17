//! `SearchUtils.parse_start_offset_from_page_token` port.
//!
//! The offset page token is `base64(json.dumps({"offset": N}))`. This mirrors
//! the Python decode + validation, including the quirk that a falsy offset
//! (`0`, `""`, missing) is rejected via `if not offset_str`, and the exact
//! error messages (which embed a Python `repr()` of the decoded value / parsed
//! dict).

use crate::error::{Result, SearchError};
use serde_json::Value;

/// Parse the base64/JSON `{"offset": N}` page token into a start offset.
pub fn parse_start_offset_from_page_token(page_token: Option<&str>) -> Result<i64> {
    let token = match page_token {
        None | Some("") => return Ok(0),
        Some(t) => t,
    };

    let decoded = base64_decode(token).ok_or_else(|| {
        SearchError::invalid_parameter_value("Invalid page token, could not base64-decode")
    })?;

    let parsed: Value = match serde_json::from_slice(&decoded) {
        Ok(v) => v,
        Err(_) => {
            let decoded_str = String::from_utf8_lossy(&decoded);
            return Err(SearchError::invalid_parameter_value(format!(
                "Invalid page token, decoded value={decoded_str}"
            )));
        }
    };

    // offset_str = parsed_token.get("offset"); if not offset_str: raise
    let offset_val = parsed.get("offset");
    let is_falsy = match offset_val {
        None => true,
        Some(Value::Null) => true,
        Some(Value::Bool(false)) => true,
        Some(Value::Number(n)) => n.as_f64() == Some(0.0),
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    };
    if is_falsy {
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid page token, parsed value={}",
            py_repr_json(&parsed)
        )));
    }

    let offset_val = offset_val.unwrap();
    // Python: int(offset_str); on failure "Invalid page token, not stringable".
    match offset_val {
        Value::Number(n) if n.is_i64() => Ok(n.as_i64().unwrap()),
        Value::Number(n) if n.is_u64() => Ok(n.as_u64().unwrap() as i64),
        Value::String(s) => s.parse::<i64>().map_err(|_| {
            SearchError::invalid_parameter_value(format!("Invalid page token, not stringable {s}"))
        }),
        // int(float) truncates in Python; a float offset is unusual but valid.
        Value::Number(n) => Ok(n.as_f64().unwrap() as i64),
        other => Err(SearchError::invalid_parameter_value(format!(
            "Invalid page token, not stringable {}",
            py_repr_json(other)
        ))),
    }
}

/// `SearchUtils.create_page_token`: `base64(json.dumps({"offset": N}))`.
/// `json.dumps` default separators render `{"offset": N}` (space after colon),
/// so the emitted token byte-matches the store's own `create_page_token`.
/// Encoded with the same hand-rolled standard base64 alphabet [`base64_decode`]
/// reads (no external dependency).
pub fn create_page_token(offset: i64) -> String {
    let json = format!("{{\"offset\": {offset}}}");
    base64_encode(json.as_bytes())
}

/// Standard base64 encode (with `=` padding), the inverse of [`base64_decode`].
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Standard base64 decode (the alphabet `json.dumps` output is encoded with).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255u8; 256];
    for (i, &b) in ALPHABET.iter().enumerate() {
        lookup[b as usize] = i as u8;
    }
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in &bytes {
        if b == b'=' {
            break;
        }
        let v = lookup[b as usize];
        if v == 255 {
            return None;
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

/// Python `repr()` of a decoded JSON value (dicts render `{'k': v}` etc.).
fn py_repr_json(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => crate::literal_eval::py_repr_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr_json).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, val)| {
                    format!(
                        "{}: {}",
                        crate::literal_eval::py_repr_str(k),
                        py_repr_json(val)
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}
