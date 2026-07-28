//! The v1 segment format's spec tests: round-trips, corruption handling,
//! and the golden byte lock.
//!
//! Per the design's blast-radius ranking, storage bytes are the highest
//! evidence priority: corruption is silent and the format entrenches the
//! day real data exists in it. The golden test locks the bytes — any
//! change that moves them is a behavioral change whose review includes
//! re-blessing `tests/golden/segment_v1.bin`, never a refactor.

use arrow_lite::{Column, ColumnType, Field, LogicalType, NumericData, Schema};
use storage_lite::{
    decode_manifest, decode_segment, encode_manifest, encode_segment, FormatError,
    ManifestSections, RowValue, Segment, WriteBuffer,
};

/// A fixture exercising every format feature: all three column types, a
/// logical annotation, nulls (and a nullable column without nulls), an
/// ordered ordering key (delta-of-delta), negatives, NaN (kept out of
/// the zone map), and a non-zero base row id.
fn fixture_segment() -> Segment {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false).with_logical(LogicalType::TimestampNs),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
        Field::new("y", ColumnType::F64, true),
        Field::new("n", ColumnType::I64, true),
    ]);
    type FixtureRow<'a> = (i64, &'a str, f64, Option<f64>, Option<i64>);
    let mut buffer = WriteBuffer::new(schema, 0).unwrap();
    let rows: &[FixtureRow<'_>] = &[
        (1_000, "AAPL", 1.5, Some(-2.5), Some(-7)),
        (2_000, "MSFT", f64::NAN, None, Some(0)),
        (3_000, "AAPL", -0.0, Some(4.0), None),
        (3_000, "TSLA", 1e300, Some(0.25), Some(i64::MAX)),
        (4_500, "MSFT", -1e-300, None, Some(i64::MIN)),
    ];
    for &(ts, sym, x, y, n) in rows {
        buffer
            .append(&[
                RowValue::I64(ts),
                RowValue::Key(sym),
                RowValue::F64(x),
                y.map_or(RowValue::Null, RowValue::F64),
                n.map_or(RowValue::Null, RowValue::I64),
            ])
            .unwrap();
    }
    buffer.snapshot_at(12_345).unwrap()
}

/// Equality through the deterministic encoding: bit-level and total, so
/// NaN payloads and negative zero compare exactly (the fixture holds a
/// NaN precisely because `PartialEq` on `f64` cannot).
fn assert_segments_equal(left: &Segment, right: &Segment) {
    assert_eq!(left.batch().schema(), right.batch().schema());
    assert_eq!(encode_segment(left), encode_segment(right));
    assert_eq!(left.ordering_key(), right.ordering_key());
    assert_eq!(left.is_ordered(), right.is_ordered());
    assert_eq!(left.base_row_id(), right.base_row_id());
}

#[test]
fn fixture_round_trips_exactly() {
    let segment = fixture_segment();
    let bytes = encode_segment(&segment);
    let decoded = decode_segment(&bytes).unwrap();
    assert_segments_equal(&segment, &decoded);
    // And the re-encode is byte-identical — determinism end to end.
    assert_eq!(encode_segment(&decoded), bytes);
}

#[test]
fn empty_and_single_row_segments_round_trip() {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("x", ColumnType::F64, false),
    ]);
    let empty = WriteBuffer::new(schema.clone(), 0)
        .unwrap()
        .freeze()
        .unwrap();
    assert_segments_equal(&empty, &decode_segment(&encode_segment(&empty)).unwrap());
    let mut buffer = WriteBuffer::new(schema, 0).unwrap();
    buffer
        .append(&[RowValue::I64(i64::MIN), RowValue::F64(f64::INFINITY)])
        .unwrap();
    let one = buffer.freeze().unwrap();
    assert_segments_equal(&one, &decode_segment(&encode_segment(&one)).unwrap());
}

#[test]
fn unordered_segment_stays_unordered_and_uncompressed() {
    let schema = Schema::new(vec![Field::new("ts", ColumnType::I64, false)]);
    let mut buffer = WriteBuffer::new(schema, 0).unwrap();
    for ts in [5, 3, 9] {
        buffer.append(&[RowValue::I64(ts)]).unwrap();
    }
    let segment = buffer.freeze().unwrap();
    assert!(!segment.is_ordered());
    let decoded = decode_segment(&encode_segment(&segment)).unwrap();
    assert!(!decoded.is_ordered());
    assert_segments_equal(&segment, &decoded);
}

#[test]
fn every_truncation_is_an_error_never_a_panic() {
    let bytes = encode_segment(&fixture_segment());
    for len in 0..bytes.len() {
        assert!(
            decode_segment(&bytes[..len]).is_err(),
            "prefix of {len} bytes decoded"
        );
    }
}

#[test]
fn every_single_byte_corruption_is_caught() {
    // The CRC covers the payload; magic/version/reserved/crc corruption
    // is caught structurally. Flipping any single byte anywhere must
    // fail loudly — this is what "silent corruption" defense means.
    let bytes = encode_segment(&fixture_segment());
    for position in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[position] ^= 0x40;
        assert!(
            decode_segment(&corrupt).is_err(),
            "flipped byte {position} decoded"
        );
    }
}

/// The manifest fixture: every schema feature the manifest serializes
/// (all three column types, nullability, a logical annotation), a
/// non-zero ordering key, and a non-zero generation.
fn fixture_manifest() -> (Schema, usize, u64) {
    let schema = Schema::new(vec![
        Field::new("sym", ColumnType::Key, false),
        Field::new("ts", ColumnType::I64, false).with_logical(LogicalType::TimestampNs),
        Field::new("x", ColumnType::F64, true),
    ]);
    (schema, 1, 42)
}

#[test]
fn manifest_round_trips_exactly() {
    let (schema, ordering_key, generation) = fixture_manifest();
    let bytes = encode_manifest(
        &schema,
        ordering_key,
        generation,
        &ManifestSections::default(),
    );
    let manifest = decode_manifest(&bytes).unwrap();
    assert_eq!(manifest.schema, schema);
    assert_eq!(manifest.ordering_key, ordering_key);
    assert_eq!(manifest.generation, generation);
}

#[test]
fn every_manifest_truncation_is_an_error_never_a_panic() {
    let (schema, ordering_key, generation) = fixture_manifest();
    let bytes = encode_manifest(
        &schema,
        ordering_key,
        generation,
        &ManifestSections::default(),
    );
    for len in 0..bytes.len() {
        assert!(
            decode_manifest(&bytes[..len]).is_err(),
            "prefix of {len} bytes decoded"
        );
    }
}

#[test]
fn every_manifest_single_byte_corruption_is_caught() {
    let (schema, ordering_key, generation) = fixture_manifest();
    let bytes = encode_manifest(
        &schema,
        ordering_key,
        generation,
        &ManifestSections::default(),
    );
    for position in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[position] ^= 0x40;
        assert!(
            decode_manifest(&corrupt).is_err(),
            "flipped byte {position} decoded"
        );
    }
}

#[test]
fn manifest_rejects_segment_bytes_and_vice_versa() {
    let (schema, ordering_key, generation) = fixture_manifest();
    assert!(matches!(
        decode_manifest(&encode_segment(&fixture_segment())),
        Err(FormatError::BadMagic)
    ));
    assert!(matches!(
        decode_segment(&encode_manifest(
            &schema,
            ordering_key,
            generation,
            &ManifestSections::default()
        )),
        Err(FormatError::BadMagic)
    ));
}

/// The manifest golden lock — same contract as the segment's below.
#[test]
fn manifest_golden_bytes_are_locked() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("manifest_v1.bin");
    let (schema, ordering_key, generation) = fixture_manifest();
    let bytes = encode_manifest(
        &schema,
        ordering_key,
        generation,
        &ManifestSections::default(),
    );
    if !path.exists() && std::env::var_os("TALLYDB_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        panic!("golden blessed at {path:?}; rerun without TALLYDB_BLESS_GOLDEN");
    }
    let golden = std::fs::read(&path).unwrap_or_else(|_| {
        panic!("missing {path:?} — bless it with TALLYDB_BLESS_GOLDEN=1 if intentional")
    });
    assert_eq!(
        bytes, golden,
        "manifest bytes moved off the committed golden — a behavioral \
         change; re-bless deliberately, in review"
    );
    let decoded = decode_manifest(&golden).unwrap();
    assert_eq!(decoded.schema, schema);
    assert_eq!(decoded.ordering_key, ordering_key);
    assert_eq!(decoded.generation, generation);
}

#[test]
fn format_errors_are_specific() {
    let bytes = encode_segment(&fixture_segment());
    assert!(matches!(
        decode_segment(b"NOTTALLY-rest-doesnt-matter"),
        Err(FormatError::BadMagic)
    ));
    let mut wrong_version = bytes.clone();
    wrong_version[8] = 99;
    assert!(matches!(
        decode_segment(&wrong_version),
        Err(FormatError::UnsupportedVersion(99))
    ));
    let mut flipped_payload = bytes.clone();
    let last = flipped_payload.len() - 1;
    flipped_payload[last] ^= 0xFF;
    assert!(matches!(
        decode_segment(&flipped_payload),
        Err(FormatError::ChecksumMismatch { .. })
    ));
}

#[test]
fn ordering_key_compresses_and_f64_does_not() {
    // The writer policy in observable form: the ordered ordering key's
    // varints undercut raw by a wide margin, while f64 stays raw
    // byte-for-byte (the #30 interim ruling made visible).
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("x", ColumnType::F64, false),
    ]);
    let mut buffer = WriteBuffer::new(schema, 0).unwrap();
    let rows = 10_000;
    for i in 0..rows {
        buffer
            .append(&[RowValue::I64(1_000 * i), RowValue::F64(i as f64)])
            .unwrap();
    }
    let bytes = encode_segment(&buffer.freeze().unwrap());
    // Raw would be ~16 bytes per row for the two columns; the encoded
    // file must sit far below the i64-raw half and above the f64 half.
    let f64_raw = rows as usize * 8;
    assert!(bytes.len() > f64_raw, "{}", bytes.len());
    assert!(bytes.len() < f64_raw + rows as usize * 2, "{}", bytes.len());
}

/// The golden lock. `segment_v1.bin` is committed; these bytes freezing
/// IS the format freezing.
///
/// If this test fails, the format changed: that is a behavioral change,
/// not a refactor. Bless it deliberately — delete the file, rerun with
/// `TALLYDB_BLESS_GOLDEN=1`, commit the new bytes, and say so in review.
#[test]
fn golden_bytes_are_locked() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("segment_v1.bin");
    let bytes = encode_segment(&fixture_segment());
    if !path.exists() && std::env::var_os("TALLYDB_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        panic!("golden blessed at {path:?}; rerun without TALLYDB_BLESS_GOLDEN");
    }
    let golden = std::fs::read(&path).unwrap_or_else(|_| {
        panic!("missing {path:?} — bless it with TALLYDB_BLESS_GOLDEN=1 if intentional")
    });
    assert_eq!(
        bytes, golden,
        "segment bytes moved off the committed golden — a behavioral \
         change; re-bless deliberately, in review"
    );
    // The committed bytes also still decode to the fixture.
    assert_segments_equal(&decode_segment(&golden).unwrap(), &fixture_segment());
}

#[test]
fn decoded_columns_read_correctly() {
    // Spot-check decoded values (not just round-trip equality) so an
    // encode/decode bug that cancels itself out cannot hide.
    let decoded = decode_segment(&encode_segment(&fixture_segment())).unwrap();
    let batch = decoded.batch();
    let Column::Numeric(NumericData::I64(ts)) = &batch.columns()[0] else {
        panic!("ts type")
    };
    assert_eq!(ts.values().as_slice(), &[1_000, 2_000, 3_000, 3_000, 4_500]);
    let Column::Key(sym) = &batch.columns()[1] else {
        panic!("sym type")
    };
    assert_eq!(sym.value_at(0), Some("AAPL"));
    assert_eq!(sym.value_at(4), Some("MSFT"));
    assert_eq!(sym.codes().as_slice(), &[0, 1, 0, 2, 1]);
    let Column::Numeric(NumericData::F64(x)) = &batch.columns()[2] else {
        panic!("x type")
    };
    assert!(x.values().as_slice()[1].is_nan());
    assert_eq!(x.values().as_slice()[3], 1e300);
    let Column::Numeric(NumericData::F64(y)) = &batch.columns()[3] else {
        panic!("y type")
    };
    assert_eq!(y.null_count(), 2);
    assert!(!y.is_valid(1));
    assert!(!y.is_valid(4));
    let Column::Numeric(NumericData::I64(n)) = &batch.columns()[4] else {
        panic!("n type")
    };
    assert_eq!(n.values().as_slice()[3], i64::MAX);
    assert_eq!(n.values().as_slice()[4], i64::MIN);
    assert!(!n.is_valid(2));
    assert_eq!(
        batch.schema().fields()[0].logical(),
        Some(LogicalType::TimestampNs)
    );
}

#[test]
fn sectioned_manifest_round_trips_and_v1_stays_byte_identical() {
    // M4.4: empty sections encode plain v1 (the golden above already
    // locks those bytes); content switches to v2, round-trips, and a
    // v1 reader's world is untouched until a table actually diverges.
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("x", ColumnType::F64, true),
    ]);
    let empty = ManifestSections::default();
    let v1 = encode_manifest(&schema, 0, 3, &empty);
    assert_eq!(&v1[8..10], &1u16.to_le_bytes(), "empty sections stay v1");
    let sections = ManifestSections {
        diverged: true,
        history: vec!["seg-g0000000001-00000000000000000000.tlyseg".to_owned()],
    };
    let v2 = encode_manifest(&schema, 0, 3, &sections);
    assert_eq!(&v2[8..10], &2u16.to_le_bytes(), "content is v2");
    let decoded = decode_manifest(&v2).unwrap();
    assert_eq!(decoded.generation, 3);
    assert_eq!(decoded.sections, sections);
    // A v1 manifest decodes with empty sections.
    assert_eq!(decode_manifest(&v1).unwrap().sections, empty);
}

#[test]
fn unknown_manifest_sections_are_skipped_whole() {
    // The forward-compatibility contract: a reader ignores tags it
    // does not know (zone maps arrive later under tag 1), so sections
    // fill in over time without a version bump.
    let schema = Schema::new(vec![Field::new("ts", ColumnType::I64, false)]);
    let sections = ManifestSections {
        diverged: true,
        history: Vec::new(),
    };
    let mut bytes = encode_manifest(&schema, 0, 0, &sections);
    // Append one unknown section (tag 999, 4-byte payload) by hand,
    // bump the count, and re-stamp the checksum the way the encoder
    // lays it out: crc32c over everything past the 16-byte header.
    let count_offset = bytes.len() - (2 + 2 + 4 + 1); // count + known section
    let count = u16::from_le_bytes([bytes[count_offset], bytes[count_offset + 1]]);
    bytes[count_offset..count_offset + 2].copy_from_slice(&(count + 1).to_le_bytes());
    bytes.extend_from_slice(&999u16.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&[0xAA; 4]);
    let crc = crc32c_reference(&bytes[16..]);
    bytes[12..16].copy_from_slice(&crc.to_le_bytes());
    let decoded = decode_manifest(&bytes).unwrap();
    assert!(decoded.sections.diverged, "known section still read");
}

/// An independent CRC-32C (Castagnoli, reflected 0x82F63B78) so the
/// unknown-section test re-stamps checksums without borrowing the
/// implementation under test.
fn crc32c_reference(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x82F6_3B78
            } else {
                crc >> 1
            };
        }
    }
    !crc
}
