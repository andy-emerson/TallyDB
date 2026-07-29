//! ALP and its integer sibling: the value-column codecs (#42).
//!
//! **ALP** (Adaptive Lossless floating-Point, Afroozeh & Boncz 2023) is
//! the ruled codec for `f64` columns (#30, 2026-07-24). Its insight is
//! that most stored doubles began life as *decimals* — prices, rounded
//! readings — so `value · 10ᵉ / 10ᶠ` lands exactly on an integer, and
//! integers compress with machinery floats defeat. Per vector of 1024
//! values the encoder picks the exponent pair by sampling, converts,
//! **verifies each value round-trips bit-exactly**, and stores the
//! integers frame-of-reference + bit-packed; any value that fails the
//! verification is an exception, stored verbatim. Losslessness is
//! therefore enforced per value at encode time — by construction, not
//! by hope: NaN payloads, ±0, subnormals and ±inf all either
//! reconstruct exactly or ride verbatim.
//!
//! **ALP-RD** ("real doubles") is the fallback for vectors that are not
//! decimal at heart (sensor reals). It cuts each double's bits into a
//! left part (sign, exponent, top mantissa bits) and a right part;
//! across a vector the left parts take few distinct values, so they
//! dictionary-encode into 3-bit codes while the right parts store
//! verbatim, bit-packed. Reassembly is bit-exact by construction — no
//! verification needed, no value can fail.
//!
//! The encoder computes both candidates per vector (plus raw) and keeps
//! the smallest — "adaptive" made literal. Encode runs at freeze and
//! compaction, off the read path, so it spends cycles to buy decode
//! speed: decode is unpack + one fused multiply per value.
//!
//! The **integer sibling** shares the frame-of-reference + bit-packing
//! backend for non-key `i64` columns and `u32` symbol codes — the
//! columns delta-of-delta (#29) is wrong for, since neither is
//! clock-like.
//!
//! Measured (release, seed 42, 1M rows/family, container hardware,
//! 2026-07-29 — `measure_42_*`, run explicitly): ticks prices (penny
//! grid) compress **4.18×** vs raw through the ALP path; telemetry
//! reals (continuous) **1.16×** through the RD path — lossless real
//! doubles are near-incompressible, and ~1.2× is the published family
//! for this shape; symbol codes **6.4×** (32 symbols) and **10.5×**
//! (8); integer cents through the i64 sibling **4.0–4.2×**. Decode
//! runs 31–93M values/s, the same band as the shipped delta-of-delta —
//! and decode happens once per segment at open, not per query: reads
//! serve from the decoded, zero-copy buffers.

use crate::codec::CodecError;

/// Values per vector: each gets its own scheme, exponents, reference
/// frame and bit width — the adaptivity is per-vector, as in the paper.
const CHUNK: usize = 1024;

/// `10^e` for `e` in `0..=18`, as literals: every entry below `1e23` is
/// exactly representable, and literals (unlike `powi`) are the same
/// bits on every platform — these constants are serialization format.
const POW10: [f64; 19] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18,
];

/// `10^-e` for `e` in `0..=18` — nearest-representable doubles, again
/// as platform-independent literals.
const INV10: [f64; 19] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
    1e-15, 1e-16, 1e-17, 1e-18,
];

/// Chunk mode tags (serialization format, frozen like every registry).
const MODE_ALP: u8 = 0;
const MODE_RD: u8 = 1;
const MODE_RAW: u8 = 2;

// ---------------------------------------------------------------------
// The shared backend: frame-of-reference + bit-packing.
// ---------------------------------------------------------------------

/// Appends `values` bit-packed at `width` bits each, little-endian
/// within a running bit stream. `width == 0` appends nothing (every
/// value is the frame base). Values must fit `width` bits.
fn pack_bits(out: &mut Vec<u8>, values: &[u64], width: u8) {
    debug_assert!(width <= 64);
    if width == 0 {
        return;
    }
    // A u128 accumulator holds the worst case whole (7 carried bits +
    // 64 incoming), so the loop needs no split-write case at all.
    let mut acc: u128 = 0;
    let mut bits: u32 = 0;
    for &value in values {
        debug_assert!(width == 64 || value >> width == 0);
        acc |= u128::from(value) << bits;
        bits += u32::from(width);
        while bits >= 8 {
            out.push((acc & 0xff) as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push((acc & 0xff) as u8);
    }
}

/// Reads `count` values of `width` bits from `bytes` at `*position`,
/// advancing it past the ceil(count·width/8) bytes consumed.
fn unpack_bits(
    bytes: &[u8],
    position: &mut usize,
    count: usize,
    width: u8,
) -> Result<Vec<u64>, CodecError> {
    if width == 0 {
        return Ok(vec![0; count]);
    }
    let total_bits = count
        .checked_mul(width as usize)
        .ok_or(CodecError::Truncated)?;
    let total_bytes = total_bits.div_ceil(8);
    let end = position
        .checked_add(total_bytes)
        .ok_or(CodecError::Truncated)?;
    if end > bytes.len() {
        return Err(CodecError::Truncated);
    }
    let stream = &bytes[*position..end];
    *position = end;
    let mut values = Vec::with_capacity(count);
    let mut bit: usize = 0;
    for _ in 0..count {
        let mut value: u64 = 0;
        let mut got: u32 = 0;
        while got < u32::from(width) {
            let byte = stream[bit / 8];
            let offset = (bit % 8) as u32;
            let available = 8 - offset;
            let need = u32::from(width) - got;
            let take = available.min(need);
            let chunk = (u64::from(byte) >> offset) & ((1u64 << take) - 1);
            value |= chunk << got;
            got += take;
            bit += take as usize;
        }
        values.push(value);
    }
    Ok(values)
}

/// Bits needed to represent `value` (0 for 0).
fn width_of(value: u64) -> u8 {
    (64 - value.leading_zeros()) as u8
}

// ---------------------------------------------------------------------
// ALP proper: the decimal path.
// ---------------------------------------------------------------------

/// Decode of one converted integer — THE reconstruction, shared
/// verbatim by encode-time verification and decode, so the two can
/// never disagree.
#[inline]
fn alp_reconstruct(digits: i64, e: usize, f: usize) -> f64 {
    digits as f64 * POW10[f] * INV10[e]
}

/// Encode of one value under `(e, f)`: the converted integer, or
/// `None` when the value must ride as an exception (does not convert,
/// does not fit, or does not round-trip bit-exactly).
#[inline]
fn alp_convert(value: f64, e: usize, f: usize) -> Option<i64> {
    let scaled = value * POW10[e] * INV10[f];
    // The float-to-int cast saturates (NaN to 0), so guard the domain
    // first; the bitwise verification below would catch these anyway,
    // but the guard keeps the arithmetic honest.
    if !(scaled.is_finite() && (-9.0e18..=9.0e18).contains(&scaled)) {
        return None;
    }
    let digits = scaled.round() as i64;
    (alp_reconstruct(digits, e, f).to_bits() == value.to_bits()).then_some(digits)
}

/// Picks `(e, f)` for a chunk by scoring a small sample under every
/// candidate pair — exceptions dominate the score, digit range breaks
/// ties. Deterministic: fixed sample positions, fixed tie order.
fn choose_exponents(values: &[f64]) -> (usize, usize) {
    let step = (values.len() / 32).max(1);
    let sample: Vec<f64> = values.iter().copied().step_by(step).collect();
    let mut best = (0usize, 0usize);
    let mut best_score = u64::MAX;
    for e in 0..POW10.len() {
        for f in 0..=e {
            let mut exceptions = 0u64;
            let (mut low, mut high) = (i64::MAX, i64::MIN);
            for &value in &sample {
                match alp_convert(value, e, f) {
                    Some(digits) => {
                        low = low.min(digits);
                        high = high.max(digits);
                    }
                    None => exceptions += 1,
                }
            }
            let width = if low <= high {
                u64::from(width_of(high.wrapping_sub(low) as u64))
            } else {
                64
            };
            // An exception costs its verbatim 8 bytes + position; a
            // converted value costs its packed width.
            let score = exceptions * (64 + 16) + (sample.len() as u64 - exceptions) * width;
            if score < best_score {
                best_score = score;
                best = (e, f);
            }
        }
    }
    best
}

/// Encodes one chunk in ALP mode, returning `None` when every value is
/// an exception (nothing decimal here — RD or raw will win anyway).
fn encode_alp_chunk(values: &[f64]) -> Option<Vec<u8>> {
    let (e, f) = choose_exponents(values);
    let mut digits: Vec<Option<i64>> = Vec::with_capacity(values.len());
    let (mut low, mut high) = (i64::MAX, i64::MIN);
    let mut exceptions = 0usize;
    for &value in values {
        let converted = alp_convert(value, e, f);
        match converted {
            Some(digit) => {
                low = low.min(digit);
                high = high.max(digit);
            }
            None => exceptions += 1,
        }
        digits.push(converted);
    }
    if low > high {
        return None; // all exceptions
    }
    let width = width_of(high.wrapping_sub(low) as u64);
    let mut out = Vec::with_capacity(2 + 8 + 1 + values.len() * usize::from(width) / 8);
    out.push(MODE_ALP);
    out.push(e as u8);
    out.push(f as u8);
    out.extend_from_slice(&low.to_le_bytes());
    out.push(width);
    // Exception slots pack the frame base — a placeholder the decoder
    // overwrites, kept in-stream so packing stays uniform.
    let packed: Vec<u64> = digits
        .iter()
        .map(|digit| digit.unwrap_or(low).wrapping_sub(low) as u64)
        .collect();
    pack_bits(&mut out, &packed, width);
    out.extend_from_slice(&(exceptions as u16).to_le_bytes());
    for (position, (digit, &value)) in digits.iter().zip(values).enumerate() {
        if digit.is_none() {
            out.extend_from_slice(&(position as u16).to_le_bytes());
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    Some(out)
}

fn decode_alp_chunk(
    bytes: &[u8],
    position: &mut usize,
    count: usize,
) -> Result<Vec<f64>, CodecError> {
    let e = take_u8(bytes, position)? as usize;
    let f = take_u8(bytes, position)? as usize;
    if e >= POW10.len() || f > e {
        return Err(CodecError::Invalid("ALP exponents out of range"));
    }
    let base = i64::from_le_bytes(take_array::<8>(bytes, position)?);
    let width = take_u8(bytes, position)?;
    if width > 64 {
        return Err(CodecError::Invalid("ALP bit width over 64"));
    }
    let packed = unpack_bits(bytes, position, count, width)?;
    let mut values: Vec<f64> = packed
        .into_iter()
        .map(|delta| alp_reconstruct(base.wrapping_add(delta as i64), e, f))
        .collect();
    let exceptions = u16::from_le_bytes(take_array::<2>(bytes, position)?) as usize;
    for _ in 0..exceptions {
        let slot = u16::from_le_bytes(take_array::<2>(bytes, position)?) as usize;
        let bits = u64::from_le_bytes(take_array::<8>(bytes, position)?);
        if slot >= count {
            return Err(CodecError::Invalid("ALP exception past the chunk"));
        }
        values[slot] = f64::from_bits(bits);
    }
    Ok(values)
}

// ---------------------------------------------------------------------
// ALP-RD: the real-doubles fallback.
// ---------------------------------------------------------------------

/// Left parts a dictionary may hold: 3-bit codes.
const RD_DICT: usize = 8;
/// Left-part widths searched (right parts stay under 64 bits and left
/// parts fit `u16`).
const RD_LEFT_MAX: u8 = 16;

/// The `(left_width, dictionary)` cut that minimizes this chunk's
/// encoded size. Deterministic: candidates counted with a sort, ties
/// broken by value.
fn choose_cut(values: &[f64]) -> (u8, Vec<u16>) {
    let mut best_width = 1u8;
    let mut best_dict: Vec<u16> = Vec::new();
    let mut best_size = usize::MAX;
    for left_width in 1..=RD_LEFT_MAX {
        let shift = 64 - u32::from(left_width);
        let mut lefts: Vec<u16> = values
            .iter()
            .map(|value| (value.to_bits() >> shift) as u16)
            .collect();
        lefts.sort_unstable();
        // (left, count) runs, then the RD_DICT most frequent —
        // count-descending, value-ascending, so the choice (and the
        // bytes) never depend on iteration order.
        let mut runs: Vec<(u16, usize)> = Vec::new();
        for &left in &lefts {
            match runs.last_mut() {
                Some((current, count)) if *current == left => *count += 1,
                _ => runs.push((left, 1)),
            }
        }
        runs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let dict: Vec<u16> = runs.iter().take(RD_DICT).map(|&(left, _)| left).collect();
        let covered: usize = runs.iter().take(RD_DICT).map(|&(_, count)| count).sum();
        let exceptions = values.len() - covered;
        let size = (values.len() * 3).div_ceil(8)
            + (values.len() * (64 - left_width as usize)).div_ceil(8)
            + dict.len() * 2
            + exceptions * 4;
        if size < best_size {
            best_size = size;
            best_width = left_width;
            best_dict = dict;
        }
    }
    (best_width, best_dict)
}

/// Encodes one chunk in RD mode — total by construction: every double
/// splits into a left part (dictionary code or exception) and a right
/// part stored verbatim, so reassembly is bit-exact for any input.
fn encode_rd_chunk(values: &[f64]) -> Vec<u8> {
    let (left_width, dict) = choose_cut(values);
    let shift = 64 - u32::from(left_width);
    let right_width = 64 - left_width;
    let mut out = Vec::with_capacity(3 + values.len() * usize::from(right_width) / 8);
    out.push(MODE_RD);
    out.push(left_width);
    out.push(dict.len() as u8);
    for &entry in &dict {
        out.extend_from_slice(&entry.to_le_bytes());
    }
    let mut codes: Vec<u64> = Vec::with_capacity(values.len());
    let mut exceptions: Vec<(u16, u16)> = Vec::new();
    for (position, value) in values.iter().enumerate() {
        let left = (value.to_bits() >> shift) as u16;
        match dict.iter().position(|&entry| entry == left) {
            Some(code) => codes.push(code as u64),
            None => {
                codes.push(0); // placeholder; the exception list wins
                exceptions.push((position as u16, left));
            }
        }
    }
    pack_bits(&mut out, &codes, 3);
    let rights: Vec<u64> = values
        .iter()
        .map(|value| {
            if right_width == 64 {
                value.to_bits()
            } else {
                value.to_bits() & ((1u64 << right_width) - 1)
            }
        })
        .collect();
    pack_bits(&mut out, &rights, right_width);
    out.extend_from_slice(&(exceptions.len() as u16).to_le_bytes());
    for (position, left) in exceptions {
        out.extend_from_slice(&position.to_le_bytes());
        out.extend_from_slice(&left.to_le_bytes());
    }
    out
}

fn decode_rd_chunk(
    bytes: &[u8],
    position: &mut usize,
    count: usize,
) -> Result<Vec<f64>, CodecError> {
    let left_width = take_u8(bytes, position)?;
    if left_width == 0 || left_width > RD_LEFT_MAX {
        return Err(CodecError::Invalid("RD left width out of range"));
    }
    let dict_len = take_u8(bytes, position)? as usize;
    if dict_len == 0 || dict_len > RD_DICT {
        return Err(CodecError::Invalid("RD dictionary size out of range"));
    }
    let mut dict = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        dict.push(u16::from_le_bytes(take_array::<2>(bytes, position)?));
    }
    let codes = unpack_bits(bytes, position, count, 3)?;
    let right_width = 64 - left_width;
    let rights = unpack_bits(bytes, position, count, right_width)?;
    let shift = 64 - u32::from(left_width);
    let mut values = Vec::with_capacity(count);
    for (&code, &right) in codes.iter().zip(&rights) {
        let left = *dict
            .get(code as usize)
            .ok_or(CodecError::Invalid("RD code past its dictionary"))?;
        values.push(f64::from_bits((u64::from(left) << shift) | right));
    }
    let exceptions = u16::from_le_bytes(take_array::<2>(bytes, position)?) as usize;
    for _ in 0..exceptions {
        let slot = u16::from_le_bytes(take_array::<2>(bytes, position)?) as usize;
        let left = u16::from_le_bytes(take_array::<2>(bytes, position)?);
        if slot >= count {
            return Err(CodecError::Invalid("RD exception past the chunk"));
        }
        let right = rights[slot];
        values[slot] = f64::from_bits((u64::from(left) << shift) | right);
    }
    Ok(values)
}

// ---------------------------------------------------------------------
// The f64 column codec: per-chunk adaptive dispatch.
// ---------------------------------------------------------------------

/// Encodes an `f64` column: per chunk of 1024, the smallest of ALP, RD
/// and raw — decided by actual encoded size, not by estimate, so the
/// choice can never be pessimal.
pub fn encode_alp_f64(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for chunk in values.chunks(CHUNK) {
        let alp = encode_alp_chunk(chunk);
        let rd = encode_rd_chunk(chunk);
        let raw_size = 1 + chunk.len() * 8;
        let best = match alp {
            Some(alp) if alp.len() <= rd.len() && alp.len() < raw_size => alp,
            _ if rd.len() < raw_size => rd,
            _ => {
                let mut raw = Vec::with_capacity(raw_size);
                raw.push(MODE_RAW);
                for value in chunk {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
                raw
            }
        };
        out.extend_from_slice(&best);
    }
    out
}

/// Decodes exactly `count` values; anything else in `bytes` is an
/// error, never garbage.
pub fn decode_alp_f64(bytes: &[u8], count: usize) -> Result<Vec<f64>, CodecError> {
    let mut values = Vec::with_capacity(count);
    let mut position = 0usize;
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(CHUNK);
        match take_u8(bytes, &mut position)? {
            MODE_ALP => values.extend(decode_alp_chunk(bytes, &mut position, chunk)?),
            MODE_RD => values.extend(decode_rd_chunk(bytes, &mut position, chunk)?),
            MODE_RAW => {
                for _ in 0..chunk {
                    let bits = u64::from_le_bytes(take_array::<8>(bytes, &mut position)?);
                    values.push(f64::from_bits(bits));
                }
            }
            _ => return Err(CodecError::Invalid("unknown ALP chunk mode")),
        }
        remaining -= chunk;
    }
    if position != bytes.len() {
        return Err(CodecError::TrailingBytes {
            extra: bytes.len() - position,
        });
    }
    Ok(values)
}

// ---------------------------------------------------------------------
// The integer sibling: frame-of-reference + bit-packing for the
// columns delta-of-delta is wrong for.
// ---------------------------------------------------------------------

/// Encodes an `i64` column: per chunk, `base` (the minimum) + packed
/// offsets. Exact over the full range including `i64::MIN`/`MAX` —
/// offsets are wrapping differences, which fit `u64` whenever `base`
/// is the true minimum.
pub fn encode_for_i64(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for chunk in values.chunks(CHUNK) {
        let base = chunk.iter().copied().min().expect("chunks are non-empty");
        let deltas: Vec<u64> = chunk
            .iter()
            .map(|&value| value.wrapping_sub(base) as u64)
            .collect();
        let width = deltas.iter().copied().max().map_or(0, width_of);
        out.extend_from_slice(&base.to_le_bytes());
        out.push(width);
        pack_bits(&mut out, &deltas, width);
    }
    out
}

/// Decodes exactly `count` values (see [`encode_for_i64`]).
pub fn decode_for_i64(bytes: &[u8], count: usize) -> Result<Vec<i64>, CodecError> {
    let mut values = Vec::with_capacity(count);
    let mut position = 0usize;
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(CHUNK);
        let base = i64::from_le_bytes(take_array::<8>(bytes, &mut position)?);
        let width = take_u8(bytes, &mut position)?;
        if width > 64 {
            return Err(CodecError::Invalid("FOR bit width over 64"));
        }
        for delta in unpack_bits(bytes, &mut position, chunk, width)? {
            values.push(base.wrapping_add(delta as i64));
        }
        remaining -= chunk;
    }
    if position != bytes.len() {
        return Err(CodecError::TrailingBytes {
            extra: bytes.len() - position,
        });
    }
    Ok(values)
}

/// As [`encode_for_i64`], for `u32` symbol codes — low-cardinality
/// dictionaries make the offsets a few bits wide.
pub fn encode_for_u32(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len());
    for chunk in values.chunks(CHUNK) {
        let base = chunk.iter().copied().min().expect("chunks are non-empty");
        let deltas: Vec<u64> = chunk.iter().map(|&value| u64::from(value - base)).collect();
        let width = deltas.iter().copied().max().map_or(0, width_of);
        out.extend_from_slice(&base.to_le_bytes());
        out.push(width);
        pack_bits(&mut out, &deltas, width);
    }
    out
}

/// Decodes exactly `count` values (see [`encode_for_u32`]).
pub fn decode_for_u32(bytes: &[u8], count: usize) -> Result<Vec<u32>, CodecError> {
    let mut values = Vec::with_capacity(count);
    let mut position = 0usize;
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(CHUNK);
        let base = u32::from_le_bytes(take_array::<4>(bytes, &mut position)?);
        let width = take_u8(bytes, &mut position)?;
        if width > 32 {
            return Err(CodecError::Invalid("FOR u32 bit width over 32"));
        }
        for delta in unpack_bits(bytes, &mut position, chunk, width)? {
            let value = u64::from(base) + delta;
            values.push(u32::try_from(value).map_err(|_| CodecError::Invalid("FOR u32 overflow"))?);
        }
        remaining -= chunk;
    }
    if position != bytes.len() {
        return Err(CodecError::TrailingBytes {
            extra: bytes.len() - position,
        });
    }
    Ok(values)
}

// ---------------------------------------------------------------------

fn take_u8(bytes: &[u8], position: &mut usize) -> Result<u8, CodecError> {
    let &byte = bytes.get(*position).ok_or(CodecError::Truncated)?;
    *position += 1;
    Ok(byte)
}

fn take_array<const N: usize>(bytes: &[u8], position: &mut usize) -> Result<[u8; N], CodecError> {
    let end = position.checked_add(N).ok_or(CodecError::Truncated)?;
    let slice = bytes.get(*position..end).ok_or(CodecError::Truncated)?;
    *position = end;
    Ok(slice.try_into().expect("length checked"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn round_trip_f64(values: &[f64]) {
        let encoded = encode_alp_f64(values);
        let decoded = decode_alp_f64(&encoded, values.len()).unwrap();
        let bits: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
        let decoded_bits: Vec<u64> = decoded.iter().map(|value| value.to_bits()).collect();
        assert_eq!(decoded_bits, bits, "bit-exactness is the contract");
    }

    #[test]
    fn adversarial_ieee_values_round_trip_bit_exactly() {
        // The evidence #42 names: every value either reconstructs
        // exactly or is an exception by construction.
        let nasty = [
            0.0,
            -0.0,
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001), // NaN payload
            f64::from_bits(0xfff0_0000_dead_beef), // signaling-ish payload
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            f64::from_bits(1),  // smallest subnormal
            -f64::from_bits(1), // and negative
            f64::MAX,
            f64::MIN,
            1e300,
            -1e-300,
            0.1,
            -0.1,
            101.37,
            1.0 / 3.0,
            f64::EPSILON,
        ];
        round_trip_f64(&nasty);
        // Interleaved with decimal values, so they ride as exceptions
        // inside an ALP chunk rather than tipping the whole chunk RD.
        let mixed: Vec<f64> = (0..2000)
            .map(|i| {
                if i % 97 == 0 {
                    nasty[i / 97 % nasty.len()]
                } else {
                    (i as f64) * 0.01
                }
            })
            .collect();
        round_trip_f64(&mixed);
    }

    #[test]
    fn shapes_pick_the_right_scheme_and_round_trip() {
        // Penny prices: the ALP front door.
        let prices: Vec<f64> = (0..3000).map(|i| 100.0 + (i % 500) as f64 * 0.01).collect();
        let encoded = encode_alp_f64(&prices);
        assert_eq!(encoded[0], MODE_ALP, "decimal data takes the ALP path");
        assert!(
            encoded.len() * 3 < prices.len() * 8,
            "pennies should compress well over 2x: {} vs {}",
            encoded.len(),
            prices.len() * 8
        );
        round_trip_f64(&prices);
        // Continuous reals near a level: the RD path.
        let reals: Vec<f64> = (0..3000)
            .map(|i| 100.0 + (i as f64 * 0.7182818).sin() * (1.0 + i as f64 * 1e-7))
            .collect();
        let encoded = encode_alp_f64(&reals);
        assert_eq!(encoded[0], MODE_RD, "continuous data takes the RD path");
        assert!(encoded.len() < reals.len() * 8, "RD still beats raw");
        round_trip_f64(&reals);
        // Uniform random bits: nothing helps; raw, not bloat.
        let noise: Vec<f64> = (0..2000)
            .map(|i| {
                f64::from_bits(
                    (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .rotate_left(17),
                )
            })
            .collect();
        let encoded = encode_alp_f64(&noise);
        assert!(
            encoded.len() <= noise.len() * 8 + 2 * (1 + 2),
            "worst case pays only the chunk headers: {}",
            encoded.len()
        );
        round_trip_f64(&noise);
        // Empty and tiny columns.
        round_trip_f64(&[]);
        round_trip_f64(&[42.5]);
    }

    #[test]
    fn integer_sibling_round_trips_and_packs() {
        for values in [
            vec![],
            vec![0i64],
            vec![i64::MIN, i64::MAX, 0, -1, 1],
            (0..5000).map(|i| 1_000_000 + (i % 37)).collect::<Vec<_>>(),
            vec![-5; 3000],
        ] {
            let encoded = encode_for_i64(&values);
            assert_eq!(decode_for_i64(&encoded, values.len()).unwrap(), values);
        }
        // The shape it exists for: near-constant non-clock integers.
        let near: Vec<i64> = (0..4096).map(|i| 7_000 + (i % 16)).collect();
        let encoded = encode_for_i64(&near);
        // 4 bits per value plus 9 bytes per chunk header.
        assert!(encoded.len() < near.len(), "{}", encoded.len());
        for values in [
            vec![],
            vec![0u32, u32::MAX],
            (0..5000u32).map(|i| i % 8).collect::<Vec<_>>(),
        ] {
            let encoded = encode_for_u32(&values);
            assert_eq!(decode_for_u32(&encoded, values.len()).unwrap(), values);
        }
        // Symbol codes: 8 distinct values pack into 3 bits each.
        let codes: Vec<u32> = (0..4096u32).map(|i| i % 8).collect();
        let encoded = encode_for_u32(&codes);
        assert!(encoded.len() < codes.len(), "{}", encoded.len());
    }

    #[test]
    fn malformed_bytes_are_errors_not_garbage() {
        let encoded = encode_alp_f64(&[1.25, 2.5, 3.75]);
        for cut in 0..encoded.len() {
            assert!(
                decode_alp_f64(&encoded[..cut], 3).is_err(),
                "truncation at {cut} must error"
            );
        }
        // A wrong count desynchronizes the chunk walk; which error
        // surfaces depends on where the walk falls apart — erroring at
        // all is the contract.
        assert!(decode_alp_f64(&encoded, 2).is_err());
        assert!(decode_alp_f64(&encoded, 4).is_err());
        assert!(decode_alp_f64(&[9], 1).is_err(), "unknown mode byte");
        let encoded = encode_for_i64(&[1, 2, 3]);
        for cut in 0..encoded.len() {
            assert!(decode_for_i64(&encoded[..cut], 3).is_err());
        }
        let encoded = encode_for_u32(&[1, 2, 3]);
        for cut in 0..encoded.len() {
            assert!(decode_for_u32(&encoded[..cut], 3).is_err());
        }
    }

    /// The #42 ratio/throughput measurement over the corpus. Run
    /// explicitly, in release mode:
    ///
    /// ```text
    /// cargo test -p storage-lite --release alp::tests::measure_42 \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn measure_42_ratio_and_throughput_on_the_corpus() {
        for (name, spec) in [
            ("ticks", corpus::Spec::ticks(1_000_000, 42)),
            ("telemetry", corpus::Spec::telemetry(1_000_000, 42)),
        ] {
            let rows = spec.generate();
            let values: Vec<f64> = rows.iter().map(|row| row.value).collect();
            let raw = values.len() * 8;
            let encoded = encode_alp_f64(&values);
            let start = std::time::Instant::now();
            let decoded = decode_alp_f64(&encoded, values.len()).unwrap();
            let elapsed = start.elapsed();
            assert!(decoded
                .iter()
                .zip(&values)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            let codes: Vec<u32> = rows.iter().map(|row| row.key).collect();
            let codes_encoded = encode_for_u32(&codes);
            // A size-like integer column (cents), for the i64 sibling.
            let cents: Vec<i64> = values.iter().map(|v| (v * 100.0).round() as i64).collect();
            let cents_encoded = encode_for_i64(&cents);
            assert_eq!(decode_for_i64(&cents_encoded, cents.len()).unwrap(), cents);
            println!(
                "{name}: f64 raw {raw} B, alp {} B ({:.2}x), decode {:.0}M values/s; \
                 codes raw {} B, for {} B ({:.2}x); i64 cents {} B ({:.2}x)",
                encoded.len(),
                raw as f64 / encoded.len() as f64,
                values.len() as f64 / elapsed.as_secs_f64() / 1e6,
                codes.len() * 4,
                codes_encoded.len(),
                (codes.len() * 4) as f64 / codes_encoded.len() as f64,
                cents_encoded.len(),
                (cents.len() * 8) as f64 / cents_encoded.len() as f64,
            );
        }
    }

    proptest! {
        #[test]
        fn any_doubles_round_trip_bit_exactly(bits in prop::collection::vec(any::<u64>(), 0..2500)) {
            let values: Vec<f64> = bits.into_iter().map(f64::from_bits).collect();
            round_trip_f64(&values);
        }

        #[test]
        fn decimal_like_doubles_round_trip(
            cents in prop::collection::vec(-10_000_000i64..10_000_000, 0..2500)
        ) {
            let values: Vec<f64> = cents.into_iter().map(|c| c as f64 * 0.01).collect();
            round_trip_f64(&values);
        }

        #[test]
        fn any_i64s_round_trip(values in prop::collection::vec(any::<i64>(), 0..2500)) {
            let encoded = encode_for_i64(&values);
            prop_assert_eq!(decode_for_i64(&encoded, values.len()).unwrap(), values);
        }

        #[test]
        fn any_u32s_round_trip(values in prop::collection::vec(any::<u32>(), 0..2500)) {
            let encoded = encode_for_u32(&values);
            prop_assert_eq!(decode_for_u32(&encoded, values.len()).unwrap(), values);
        }

        #[test]
        fn random_bytes_never_panic_the_decoders(
            bytes in prop::collection::vec(any::<u8>(), 0..400),
            count in 0usize..2000
        ) {
            let _ = decode_alp_f64(&bytes, count);
            let _ = decode_for_i64(&bytes, count);
            let _ = decode_for_u32(&bytes, count);
        }
    }
}
