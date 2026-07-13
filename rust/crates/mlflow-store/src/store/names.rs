//! Random run-name generation (`_generate_random_name`, non-parity RNG — see
//! `names_data`). Produces `<predicate>-<noun>-<int>` truncated to 20 chars.

use uuid::Uuid;

use super::names_data::{GENERATOR_NOUNS, GENERATOR_PREDICATES};

const MAX_LENGTH: usize = 20;
const INTEGER_SCALE: u32 = 3;

/// Generate a random run name of the form `<predicate>-<noun>-<int>`.
///
/// Mirrors `_generate_random_name` (retry up to 10 times to stay under 20 chars,
/// else truncate). Randomness comes from `Uuid::new_v4` bytes rather than
/// Python's `random` module — the exact word is not part of any wire contract.
pub(crate) fn generate_random_name() -> String {
    let mut name = String::new();
    for _ in 0..10 {
        name = generate_string();
        if name.chars().count() <= MAX_LENGTH {
            return name;
        }
    }
    name.chars().take(MAX_LENGTH).collect()
}

fn generate_string() -> String {
    let bytes = Uuid::new_v4().into_bytes();
    let predicate = GENERATOR_PREDICATES[pick(&bytes[0..4], GENERATOR_PREDICATES.len())];
    let noun = GENERATOR_NOUNS[pick(&bytes[4..8], GENERATOR_NOUNS.len())];
    // random.randint(0, 10**integer_scale) is inclusive of 10**scale.
    let modulus = 10u32.pow(INTEGER_SCALE) + 1;
    let num = pick(&bytes[8..12], modulus as usize);
    format!("{predicate}-{noun}-{num}")
}

/// Reduce four random bytes to `[0, upper)`.
fn pick(bytes: &[u8], upper: usize) -> usize {
    let mut acc: u32 = 0;
    for &b in bytes {
        acc = acc.wrapping_shl(8) | u32::from(b);
    }
    (acc as usize) % upper
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_length() {
        for _ in 0..200 {
            let name = generate_random_name();
            assert!(name.chars().count() <= MAX_LENGTH, "too long: {name}");
            let parts: Vec<&str> = name.split('-').collect();
            // predicate-noun-int (some nouns/predicates may themselves lack
            // dashes, so at least 3 dash-separated parts).
            assert!(parts.len() >= 3, "unexpected shape: {name}");
            let predicate = parts[0];
            assert!(
                GENERATOR_PREDICATES.contains(&predicate),
                "first segment not a predicate: {name}"
            );
        }
    }
}
