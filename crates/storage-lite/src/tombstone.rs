//! Tombstones: the delete half of tombstone + reinsert.
//!
//! Decision #1 fixes what a tombstone is: a set of **internal row ids**
//! — never key tuples, never positions. `DELETE` tombstones the rows its
//! predicate matched; `UPDATE` tombstones them and reappends corrected
//! copies (which get fresh row ids at the tail of the ingest sequence);
//! an out-of-order correction is the same append. Reads resolve
//! tombstones by masking; compaction resolves them for good.
//!
//! Persistence is an append-only log of delete files beside the
//! segments: each mutation writes one `del-…` object holding the row
//! ids it killed (sorted, delta-varint, CRC-checked, same header
//! discipline as segments). Reopen unions the logs; compaction rewrites
//! storage and removes them. Nothing is ever edited in place.

use crate::codec::{decode_delta_of_delta, encode_delta_of_delta};
use crate::format::FormatError;
use std::collections::BTreeSet;

/// First bytes of every delete-log file.
pub const TOMBSTONE_MAGIC: [u8; 8] = *b"TALLYDEL";

/// Encodes one delete log: the header discipline of the segment format
/// (magic, version, CRC over the payload), then the sorted row ids
/// delta-of-delta packed (sorted ids are exactly the ascending-integer
/// shape that codec compresses best).
pub fn encode_tombstones(ids: &BTreeSet<u64>) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() + 32);
    out.extend_from_slice(&TOMBSTONE_MAGIC);
    out.extend_from_slice(&1u16.to_le_bytes()); // version
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
    out.extend_from_slice(&(ids.len() as u64).to_le_bytes());
    let signed: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
    out.extend_from_slice(&encode_delta_of_delta(&signed));
    let crc = crc32(&out[16..]);
    out[12..16].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Decodes one delete log, verifying magic, version, and checksum.
pub fn decode_tombstones(bytes: &[u8]) -> Result<BTreeSet<u64>, FormatError> {
    if bytes.len() < 24 {
        return Err(FormatError::Corrupt("delete log too short".to_owned()));
    }
    if bytes[..8] != TOMBSTONE_MAGIC {
        return Err(FormatError::BadMagic);
    }
    let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    if version != 1 {
        return Err(FormatError::UnsupportedVersion(version));
    }
    if bytes[10..12] != [0, 0] {
        return Err(FormatError::Corrupt(
            "reserved header bytes are not zero".to_owned(),
        ));
    }
    let stored = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let computed = crc32(&bytes[16..]);
    if stored != computed {
        return Err(FormatError::ChecksumMismatch { stored, computed });
    }
    let count = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
        .map_err(|_| FormatError::Corrupt("row-id count exceeds this platform".to_owned()))?;
    let signed = decode_delta_of_delta(&bytes[24..], count)?;
    let mut ids = BTreeSet::new();
    let mut previous: Option<u64> = None;
    for value in signed {
        let id = value as u64;
        if previous.is_some_and(|previous| id <= previous) {
            return Err(FormatError::Corrupt(
                "delete log row ids are not strictly ascending".to_owned(),
            ));
        }
        previous = Some(id);
        ids.insert(id);
    }
    Ok(ids)
}

/// The same IEEE CRC-32 the segment format uses.
fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut bit = 0;
            while bit < 8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
                bit += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc = (crc >> 8) ^ TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_stays_small() {
        for ids in [
            BTreeSet::new(),
            BTreeSet::from([0u64]),
            BTreeSet::from([5, 6, 7, 8, 1000, u64::from(u32::MAX)]),
            (0u64..10_000).collect::<BTreeSet<u64>>(),
        ] {
            let bytes = encode_tombstones(&ids);
            assert_eq!(decode_tombstones(&bytes).unwrap(), ids);
        }
        // A dense run of ids costs about a byte each, not eight.
        let dense: BTreeSet<u64> = (0u64..10_000).collect();
        assert!(encode_tombstones(&dense).len() < 11_000);
    }

    #[test]
    fn corruption_is_loud() {
        let ids: BTreeSet<u64> = (0u64..100).step_by(3).collect();
        let bytes = encode_tombstones(&ids);
        for position in 0..bytes.len() {
            let mut corrupt = bytes.clone();
            corrupt[position] ^= 0x10;
            assert!(
                decode_tombstones(&corrupt).is_err(),
                "flipped byte {position} decoded"
            );
        }
        for len in 0..bytes.len() {
            assert!(decode_tombstones(&bytes[..len]).is_err());
        }
    }
}
