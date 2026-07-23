//! Column codecs and their frozen tag registry.
//!
//! Decision #28: every stored column carries a one-byte codec tag from an
//! **append-only** integer registry, the same pattern as the frozen
//! column-type and logical-type registries — a new codec is a new entry,
//! never a format migration, and `0 = uncompressed` is a legitimate
//! permanent answer (today's ruling for `f64`, pending #30), not a
//! placeholder.
//!
//! Decision #29: ordered `i64` columns — the ordering key above all —
//! use **delta-of-delta**, the TSDB standard for clock-like sequences:
//! regular cadence makes the second difference almost always zero, and
//! zigzag + LEB128 varint packs those zeros into one byte each. All
//! arithmetic is wrapping, so the encoding is exact for every `i64`
//! including `i64::MIN`/`MAX` (wrapping subtraction and addition are
//! inverse bijections; no widening needed).
//!
//! The confirm-against-plain-delta measurement #29 asked for lives with
//! this module's tests (`measure_29_*`, `#[ignore]`, run explicitly);
//! plain delta itself is deliberately not a registered codec unless that
//! measurement ever says otherwise.

use std::fmt;

/// The frozen codec registry (#28). Tags are serialization format:
/// append entries, never renumber or remove.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    /// Raw little-endian values, exactly as in memory.
    Uncompressed = 0,
    /// Delta-of-delta with zigzag + LEB128 varints, for `i64` columns.
    DeltaOfDeltaI64 = 1,
}

impl Codec {
    /// This codec's frozen registry tag.
    pub fn tag(self) -> u8 {
        self as u8
    }

    /// Looks a tag up in the registry; unknown tags are a data error the
    /// caller must surface, never guess around.
    pub fn from_tag(tag: u8) -> Option<Codec> {
        match tag {
            0 => Some(Codec::Uncompressed),
            1 => Some(Codec::DeltaOfDeltaI64),
            _ => None,
        }
    }
}

/// Why encoded bytes could not be decoded.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CodecError {
    /// The stream ended mid-value.
    Truncated,
    /// A varint ran past the 10-byte maximum for a `u64`.
    VarintTooLong,
    /// Bytes remained after the expected number of values.
    TrailingBytes { extra: usize },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::Truncated => write!(f, "encoded column ends mid-value"),
            CodecError::VarintTooLong => write!(f, "varint exceeds 10 bytes"),
            CodecError::TrailingBytes { extra } => {
                write!(f, "{extra} bytes remain after the last value")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// Maps a signed value onto an unsigned one with small magnitudes first
/// (0, -1, 1, -2, …), so varints stay short for small deltas of either
/// sign.
fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Appends `value` as an LEB128 varint (7 bits per byte, high bit =
/// "more").
fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Reads one varint at `*position`, advancing it.
fn read_varint(bytes: &[u8], position: &mut usize) -> Result<u64, CodecError> {
    let mut value = 0u64;
    for shift in 0..10 {
        let &byte = bytes.get(*position).ok_or(CodecError::Truncated)?;
        *position += 1;
        value |= u64::from(byte & 0x7f) << (7 * shift);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(CodecError::VarintTooLong)
}

/// Encodes `values` as delta-of-delta: the first value, then the first
/// delta, then second differences — each zigzag-varint packed. Empty
/// input encodes to empty bytes.
pub fn encode_delta_of_delta(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() + 16);
    let Some(&first) = values.first() else {
        return out;
    };
    push_varint(&mut out, zigzag(first));
    let mut previous = first;
    let mut previous_delta = 0i64;
    for (index, &value) in values.iter().enumerate().skip(1) {
        let delta = value.wrapping_sub(previous);
        if index == 1 {
            push_varint(&mut out, zigzag(delta));
        } else {
            push_varint(&mut out, zigzag(delta.wrapping_sub(previous_delta)));
        }
        previous = value;
        previous_delta = delta;
    }
    out
}

/// Decodes exactly `count` values; anything else in `bytes` is an error.
pub fn decode_delta_of_delta(bytes: &[u8], count: usize) -> Result<Vec<i64>, CodecError> {
    let mut values = Vec::with_capacity(count);
    let mut position = 0usize;
    if count > 0 {
        let mut value = unzigzag(read_varint(bytes, &mut position)?);
        values.push(value);
        let mut delta = 0i64;
        for index in 1..count {
            let step = unzigzag(read_varint(bytes, &mut position)?);
            delta = if index == 1 {
                step
            } else {
                delta.wrapping_add(step)
            };
            value = value.wrapping_add(delta);
            values.push(value);
        }
    }
    if position != bytes.len() {
        return Err(CodecError::TrailingBytes {
            extra: bytes.len() - position,
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn round_trip(values: &[i64]) {
        let encoded = encode_delta_of_delta(values);
        assert_eq!(
            decode_delta_of_delta(&encoded, values.len()).unwrap(),
            values,
            "{values:?}"
        );
    }

    #[test]
    fn registry_is_frozen() {
        assert_eq!(Codec::Uncompressed.tag(), 0);
        assert_eq!(Codec::DeltaOfDeltaI64.tag(), 1);
        assert_eq!(Codec::from_tag(0), Some(Codec::Uncompressed));
        assert_eq!(Codec::from_tag(1), Some(Codec::DeltaOfDeltaI64));
        assert_eq!(Codec::from_tag(2), None);
        assert_eq!(Codec::from_tag(255), None);
    }

    #[test]
    fn hand_picked_sequences_round_trip() {
        round_trip(&[]);
        round_trip(&[0]);
        round_trip(&[42]);
        round_trip(&[1, 2]);
        round_trip(&[1_000, 2_000, 3_000, 4_000]); // constant cadence
        round_trip(&[1, 1, 1, 1]); // constant value
        round_trip(&[5, 3, 8, 8, -20, 40]); // disorderly
        round_trip(&[i64::MIN, i64::MAX, 0, i64::MIN, -1, 1]); // extremes
        round_trip(&[i64::MAX, i64::MIN]); // maximal wrapping delta
    }

    #[test]
    fn regular_cadence_costs_about_a_byte_per_row() {
        // The compression claim in its simplest observable form: a fixed
        // cadence has zero second difference, one varint byte per row.
        let values: Vec<i64> = (0..1000).map(|i| 1_700_000_000_000 + i * 1_000).collect();
        let encoded = encode_delta_of_delta(&values);
        // First value + first delta cost a few bytes; the rest cost one.
        assert!(encoded.len() < values.len() + 16, "{}", encoded.len());
        assert!(encoded.len() >= values.len(), "{}", encoded.len());
    }

    #[test]
    fn malformed_bytes_are_errors_not_garbage() {
        let encoded = encode_delta_of_delta(&[1, 2, 3]);
        // Truncated stream.
        assert_eq!(
            decode_delta_of_delta(&encoded[..encoded.len() - 1], 3),
            Err(CodecError::Truncated)
        );
        // Count larger than the stream holds.
        assert_eq!(
            decode_delta_of_delta(&encoded, 4),
            Err(CodecError::Truncated)
        );
        // Count smaller than the stream holds.
        assert_eq!(
            decode_delta_of_delta(&encoded, 2),
            Err(CodecError::TrailingBytes { extra: 1 })
        );
        // A varint that never terminates.
        assert_eq!(
            decode_delta_of_delta(&[0x80; 11], 1),
            Err(CodecError::VarintTooLong)
        );
        // Empty bytes decode to exactly zero values, nothing else.
        assert_eq!(decode_delta_of_delta(&[], 0), Ok(vec![]));
        assert_eq!(decode_delta_of_delta(&[], 1), Err(CodecError::Truncated));
    }

    /// Plain delta with the same zigzag varints — the comparator #29
    /// asked to be measured against, deliberately NOT a registered codec.
    fn encode_plain_delta(values: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() + 16);
        let mut previous = 0i64;
        for &value in values {
            push_varint(&mut out, zigzag(value.wrapping_sub(previous)));
            previous = value;
        }
        out
    }

    /// The #29 confirmation measurement (a measurement, not a decision —
    /// the ruling chose delta-of-delta; this records the margin on the
    /// corpus). Run explicitly, in release mode:
    ///
    /// ```text
    /// cargo test -p storage-lite --release codec::tests::measure_29 \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn measure_29_delta_of_delta_vs_plain_delta() {
        for (name, spec) in [
            ("ticks", corpus::Spec::ticks(1_000_000, 29)),
            ("telemetry", corpus::Spec::telemetry(1_000_000, 29)),
        ] {
            let timestamps: Vec<i64> = spec.generate().iter().map(|row| row.ts).collect();
            let raw = timestamps.len() * 8;
            let dod = encode_delta_of_delta(&timestamps);
            let delta = encode_plain_delta(&timestamps);
            let start = std::time::Instant::now();
            let decoded = decode_delta_of_delta(&dod, timestamps.len()).unwrap();
            let elapsed = start.elapsed();
            assert_eq!(decoded, timestamps);
            println!(
                "{name}: raw {raw} B; delta-of-delta {} B ({:.2}x vs raw); \
                 plain delta {} B ({:.2}x vs raw); dod/delta {:.3}; \
                 decode {:.0}M values/s",
                dod.len(),
                raw as f64 / dod.len() as f64,
                delta.len(),
                raw as f64 / delta.len() as f64,
                delta.len() as f64 / dod.len() as f64,
                timestamps.len() as f64 / elapsed.as_secs_f64() / 1e6,
            );
        }
    }

    proptest! {
        #[test]
        fn any_sequence_round_trips(values in prop::collection::vec(any::<i64>(), 0..300)) {
            round_trip(&values);
        }

        #[test]
        fn clock_like_sequences_round_trip_and_shrink(
            start in 0i64..2_000_000_000_000,
            cadence in 1i64..10_000,
            jitter in prop::collection::vec(-50i64..50, 200)
        ) {
            let mut value = start;
            let values: Vec<i64> = jitter
                .iter()
                .map(|j| {
                    value += cadence + j;
                    value
                })
                .collect();
            let encoded = encode_delta_of_delta(&values);
            prop_assert_eq!(decode_delta_of_delta(&encoded, values.len()).unwrap(), values.clone());
            // Jittered cadence stays near two bytes per row — the shape
            // the codec exists for.
            prop_assert!(encoded.len() <= values.len() * 3);
        }
    }
}
