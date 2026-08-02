//! The on-disk segment format, version 1.
//!
//! A segment file is **self-describing** — it embeds its schema, so a
//! segment can be opened, verified, and read with no table metadata in
//! hand (the per-segment-dictionary ruling #6, applied to the whole
//! file). The bytes are **deterministic**: the same segment encodes to
//! the same bytes on every backend and platform (everything is explicit
//! little-endian, dictionaries serialize in intern order, and nothing
//! iterates a hash map), which is what lets golden tests lock the format.
//!
//! ## Layout (all integers little-endian)
//!
//! ```text
//! magic        8B  "TALLYSEG"
//! version      u16 (1, or 2 when a trailer follows the columns — see
//!                  [`VERSION_TRAILERED`])
//! reserved     u16 (zero)
//! crc32c       u32 — CRC-32C (Castagnoli) of every byte after this
//!                  field; chosen over IEEE CRC-32 because it is the
//!                  polynomial with hardware instructions on both
//!                  x86_64 (SSE4.2) and ARMv8, and identical software
//!                  cost everywhere else including WASM (ruled
//!                  2026-07-24; the accelerated implementation is a
//!                  future additive optimization — this module's
//!                  table-driven form defines the function)
//! base_row_id  u64 (decision #1: id of the segment's first row)
//! row_count    u64
//! ordering_key u32 (column index)
//! flags        u32 (bit 0: ordering key arrived non-decreasing)
//! column_count u32
//! then per column:
//!   name_len u16, name bytes (UTF-8)
//!   column_type u8   — frozen registry (0 f64, 1 i64, 2 key)
//!   nullable    u8
//!   logical     u8 tag + u8 payload — frozen registry (0,0 = none)
//!   codec       u8   — frozen registry (decision #28)
//!   zone map    u8 bitfield (bit 0 present, bit 1 has-NaN, f64 only);
//!                  if present: min 8B, max 8B (f64 bits or i64, per
//!                  column type; min/max over valid non-NaN values,
//!                  canonical NaN when every valid value is NaN)
//!   validity    u8 present; if 1: u32 byte length + LSB bitmap bytes
//!   values      u64 byte length + encoded bytes (per the codec; key
//!                  columns store their u32 codes here)
//!   dictionary  (key columns only) u32 entry count,
//!                  i32 offsets × (count + 1), u32 byte length, bytes
//! ```
//!
//! Version 1 is decode-on-open: `decode` materializes in-memory columns,
//! so encoded buffers carry no alignment padding. A future zero-copy
//! open for uncompressed columns would be a new version — cheap under
//! the append-only registry discipline, and not worth speculative bytes
//! today.
//!
//! ## The manifest format, versions 1 and 2
//!
//! The table manifest is its own small record (it used to be an encoded
//! empty segment whose `base_row_id` field smuggled the generation — a
//! pun retired 2026-07-24). It carries exactly what reopen needs: the
//! schema to verify against, the ordering key, and the committed
//! generation. Same conventions as the segment: little-endian,
//! deterministic, golden-locked.
//!
//! ```text
//! magic        8B  "TALLYMFT"
//! version      u16 (1 while no section carries content; 2 once one
//!                   does — a table that never corrects keeps
//!                   byte-identical v1 bytes)
//! reserved     u16 (zero)
//! crc32c       u32 — CRC-32C of every byte after this field
//! generation   u64 — the committed compaction generation
//! ordering_key u32 (column index)
//! column_count u32
//! then per column (the segment format's schema prefix, same registries):
//!   name_len u16, name bytes (UTF-8)
//!   column_type u8, nullable u8, logical u8 tag + u8 payload
//! then (v2 only) the sections, each: tag u16, length u32, payload
//!   tag 1 — segment records: u32 count, then per segment
//!             name_len u16 + name, base_row_id u64, row_count u64,
//!             ordered u8, sequence kind u8 (0 row-ids;
//!             1 contiguous + base u64; 2 explicit + end u64),
//!             zone-map count u32, then per column the segment
//!             format's zone-map field verbatim
//!   tag 2 — the ingest-sequence watermark
//!   tag 3 — the history segments' names
//! ```
//!
//! Sections are additive and skippable: a decoder ignores tags it does
//! not know, which is what let the knowledge axis (tag 2, M4.4) and the
//! residency design's segment records (tag 1, 2026-07-30) share one
//! revision. Tag 1 makes the manifest the authoritative list of the
//! generation's live segments: reopen reads exactly the named files,
//! and a manifest without the section (an older writer's) falls back to
//! scanning the backend.

use crate::alp::{
    decode_alp_f64, decode_for_i64, decode_for_u32, encode_alp_f64, encode_for_i64, encode_for_u32,
};
use crate::codec::{decode_delta_of_delta, encode_delta_of_delta, Codec, CodecError};
use crate::mem::{RowValue, Segment, SequenceInfo, ZoneMap};
use arrow_lite::{
    Bitmap, Buffer, Column, ColumnType, Dictionary, Field, KeyColumn, LogicalType, NumericColumn,
    NumericData, RecordBatch, Schema,
};
use std::fmt;

/// First bytes of every segment file.
pub const MAGIC: [u8; 8] = *b"TALLYSEG";
/// The format version this module writes for segments whose sequences
/// are virtual (`sequence == row id` — every segment of a never-diverged
/// table). Byte-identical to the original golden-locked format, so
/// tables that never retain history never see a version bump.
pub const VERSION: u16 = 1;
/// The trailered segment version (M4.4, issue #75): v1's exact layout,
/// then — after the last column — the same section scheme the manifest
/// uses: `section_count: u16`, then per section `tag: u16, length: u32,
/// payload`. Readers skip unknown tags. Only diverged tables ever write
/// v2 segments. Assigned trailer tags (a registry separate from the
/// manifest's): 1 = birth sequences (see [`crate::mem::SequenceInfo`]) —
/// a state byte (1 contiguous, 2 explicit), then for contiguous the
/// `u64` base, for explicit the delta-of-delta-coded per-row array;
/// 2 = kill coordinates (history segments only) — the delta-of-delta-
/// coded per-row array of sequences at which each row's tombstone
/// landed (see [`crate::mem::Segment::superseded`]).
pub const VERSION_TRAILERED: u16 = 2;

const TRAILER_SEQUENCE: u16 = 1;
const TRAILER_SUPERSEDED: u16 = 2;
const SEQUENCE_CONTIGUOUS: u8 = 1;
const SEQUENCE_EXPLICIT: u8 = 2;

/// Why segment bytes could not be decoded.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FormatError {
    /// The file does not start with [`MAGIC`].
    BadMagic,
    /// A version this build does not read.
    UnsupportedVersion(u16),
    /// The CRC-32C over the payload disagrees with the header.
    ChecksumMismatch { stored: u32, computed: u32 },
    /// Structurally invalid bytes; names what was wrong.
    Corrupt(String),
    /// A column's encoded values failed to decode.
    Codec(CodecError),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::BadMagic => write!(f, "not a TallyDB segment or manifest (bad magic)"),
            FormatError::UnsupportedVersion(version) => {
                write!(f, "format version {version} is not supported")
            }
            FormatError::ChecksumMismatch { stored, computed } => write!(
                f,
                "checksum mismatch (stored {stored:#010x}, computed {computed:#010x})"
            ),
            FormatError::Corrupt(what) => write!(f, "corrupt file: {what}"),
            FormatError::Codec(error) => write!(f, "corrupt file: {error}"),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<CodecError> for FormatError {
    fn from(error: CodecError) -> Self {
        FormatError::Codec(error)
    }
}

/// CRC-32C (the Castagnoli polynomial), table-driven. This software
/// form defines the function; a hardware implementation (SSE4.2 /
/// ARMv8 CRC instructions compute exactly this polynomial) is a future
/// additive optimization, never a format change.
///
/// Public because it is the storage layer's one checksum: sidecar
/// records (a maintained view's definition, #83) reuse it rather than
/// growing a second CRC implementation.
pub fn crc32c(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut bit = 0;
            while bit < 8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0x82F6_3B78
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

/// Offset of the CRC field; the checksum covers everything after it.
const CRC_OFFSET: usize = 12;
const PAYLOAD_OFFSET: usize = 16;

/// Encodes a segment: byte-identical v1 while its sequences are virtual
/// (the golden-locked layout every never-diverged table keeps forever),
/// the trailered v2 once sequence data must be carried.
pub fn encode_segment(segment: &Segment) -> Vec<u8> {
    let version = match (segment.sequence_info(), segment.superseded()) {
        (SequenceInfo::RowIds, None) => VERSION,
        _ => VERSION_TRAILERED,
    };
    let batch = segment.batch();
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
    out.extend_from_slice(&segment.base_row_id().to_le_bytes());
    out.extend_from_slice(&(batch.num_rows() as u64).to_le_bytes());
    out.extend_from_slice(&(segment.ordering_key() as u32).to_le_bytes());
    out.extend_from_slice(&u32::from(segment.is_ordered()).to_le_bytes());
    out.extend_from_slice(&(batch.schema().fields().len() as u32).to_le_bytes());
    for (index, (field, column)) in batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .enumerate()
    {
        let is_ordering_key = index == segment.ordering_key();
        encode_column(
            &mut out,
            segment,
            field,
            column,
            is_ordering_key,
            segment.zone_map(index),
        );
    }
    if version == VERSION_TRAILERED {
        let mut sections: Vec<(u16, Vec<u8>)> = Vec::new();
        match segment.sequence_info() {
            // A history segment can carry kill coordinates while its
            // births are virtual only in principle; in practice every
            // trailered segment of a diverged table has sequence data,
            // and a RowIds state simply writes no sequence section.
            SequenceInfo::RowIds => {}
            SequenceInfo::Contiguous { base } => {
                let mut payload = Vec::with_capacity(9);
                payload.push(SEQUENCE_CONTIGUOUS);
                payload.extend_from_slice(&base.to_le_bytes());
                sections.push((TRAILER_SEQUENCE, payload));
            }
            SequenceInfo::Explicit(values) => {
                // Sequences are u64; the codec is i64 with wrapping
                // arithmetic throughout, so the bit-preserving cast
                // round-trips every value exactly.
                let signed: Vec<i64> = values.iter().map(|&value| value as i64).collect();
                let mut payload = encode_delta_of_delta(&signed);
                payload.insert(0, SEQUENCE_EXPLICIT);
                sections.push((TRAILER_SEQUENCE, payload));
            }
        }
        if let Some(superseded) = segment.superseded() {
            let signed: Vec<i64> = superseded.iter().map(|&value| value as i64).collect();
            sections.push((TRAILER_SUPERSEDED, encode_delta_of_delta(&signed)));
        }
        out.extend_from_slice(&(sections.len() as u16).to_le_bytes());
        for (tag, payload) in sections {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&payload);
        }
    }
    let crc = crc32c(&out[PAYLOAD_OFFSET..]);
    out[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    out
}

/// The codec the writer chooses for a column — the one place that
/// policy lives. Decision #29: the ordering key of an ordered segment
/// is clock-like and takes delta-of-delta. Decision #42 (2026-07-29):
/// `f64` columns take ALP (with its RD and raw per-chunk fallbacks —
/// the codec can never bloat, so the policy needs no size check here),
/// and the non-clock integers — `i64` columns off the ordering key,
/// `u32` symbol codes — take frame-of-reference + bit-packing.
fn writer_codec(segment: &Segment, column: &Column, is_ordering_key: bool) -> Codec {
    match column {
        Column::Numeric(NumericData::I64(_)) if is_ordering_key && segment.is_ordered() => {
            Codec::DeltaOfDeltaI64
        }
        Column::Numeric(NumericData::I64(_)) => Codec::ForI64,
        Column::Numeric(NumericData::F64(_)) => Codec::AlpF64,
        Column::Key(_) => Codec::ForU32,
    }
}

/// The field prefix both containers share — name, type, nullability,
/// logical annotation — written identically by segments (per column)
/// and manifests (per schema field).
fn encode_field(out: &mut Vec<u8>, field: &Field) {
    out.extend_from_slice(&(field.name().len() as u16).to_le_bytes());
    out.extend_from_slice(field.name().as_bytes());
    out.push(field.column_type() as u8);
    out.push(u8::from(field.nullable()));
    match field.logical() {
        None => out.extend_from_slice(&[0, 0]),
        Some(logical) => {
            let payload = match logical {
                LogicalType::Decimal64 { scale } => scale,
                LogicalType::TimestampNs => 0,
            };
            out.extend_from_slice(&[logical.tag(), payload]);
        }
    }
}

/// Decodes the shared field prefix (the inverse of [`encode_field`]).
fn decode_field(reader: &mut Reader<'_>) -> Result<Field, FormatError> {
    let name_len = reader.u16()? as usize;
    let name = std::str::from_utf8(reader.take(name_len)?)
        .map_err(|_| FormatError::Corrupt("column name is not UTF-8".to_owned()))?
        .to_owned();
    let column_type = ColumnType::from_tag(reader.u8()?)
        .ok_or_else(|| FormatError::Corrupt(format!("unknown column type for '{name}'")))?;
    let nullable = reader.u8()? != 0;
    let logical_tag = reader.u8()?;
    let logical_payload = reader.u8()?;
    let mut field = Field::new(name.clone(), column_type, nullable);
    if logical_tag != 0 {
        field = field.with_logical(
            LogicalType::from_parts(logical_tag, logical_payload).ok_or_else(|| {
                FormatError::Corrupt(format!("unknown logical type {logical_tag} for '{name}'"))
            })?,
        );
    }
    Ok(field)
}

fn encode_column(
    out: &mut Vec<u8>,
    segment: &Segment,
    field: &Field,
    column: &Column,
    is_ordering_key: bool,
    zone_map: Option<&ZoneMap>,
) {
    encode_field(out, field);
    let codec = writer_codec(segment, column, is_ordering_key);
    out.push(codec.tag());
    encode_zone_map(out, zone_map);
    let validity = match column {
        Column::Numeric(numeric) => numeric.validity(),
        Column::Key(keys) => keys.validity(),
    };
    encode_validity(out, validity);
    match column {
        Column::Numeric(NumericData::F64(numeric)) => {
            let bytes = match codec {
                Codec::AlpF64 => encode_alp_f64(numeric.values().as_slice()),
                _ => {
                    let mut bytes = Vec::with_capacity(numeric.len() * 8);
                    for value in numeric.values().as_slice() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    bytes
                }
            };
            push_values(out, &bytes);
        }
        Column::Numeric(NumericData::I64(numeric)) => {
            let bytes = match codec {
                Codec::DeltaOfDeltaI64 => encode_delta_of_delta(numeric.values().as_slice()),
                Codec::ForI64 => encode_for_i64(numeric.values().as_slice()),
                _ => {
                    let mut bytes = Vec::with_capacity(numeric.len() * 8);
                    for value in numeric.values().as_slice() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    bytes
                }
            };
            push_values(out, &bytes);
        }
        Column::Key(keys) => {
            let bytes = match codec {
                Codec::ForU32 => encode_for_u32(keys.codes().as_slice()),
                _ => {
                    let mut bytes = Vec::with_capacity(keys.len() * 4);
                    for code in keys.codes().as_slice() {
                        bytes.extend_from_slice(&code.to_le_bytes());
                    }
                    bytes
                }
            };
            push_values(out, &bytes);
            let dictionary = keys.dictionary();
            out.extend_from_slice(&(dictionary.len() as u32).to_le_bytes());
            for offset in dictionary.offsets() {
                out.extend_from_slice(&offset.to_le_bytes());
            }
            out.extend_from_slice(&(dictionary.bytes().len() as u32).to_le_bytes());
            out.extend_from_slice(dictionary.bytes());
        }
    }
}

fn push_values(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Zone map: the segment's precomputed min/max (see
/// [`crate::mem::ZoneMap`]), absent when no valid value exists or the
/// column is a key. The presence byte is a small bitfield: bit 0 =
/// present, bit 1 = the column holds at least one valid NaN (`f64`
/// only — pruning soundness under the NaN-is-greatest comparison
/// relation, D2 ruling 2026-07-24).
fn encode_zone_map(out: &mut Vec<u8>, zone_map: Option<&ZoneMap>) {
    let encoded: Option<(u8, [u8; 8], [u8; 8])> = zone_map.map(|zone_map| match zone_map {
        ZoneMap::F64 { min, max, has_nan } => (
            1 | (u8::from(*has_nan) << 1),
            min.to_le_bytes(),
            max.to_le_bytes(),
        ),
        ZoneMap::I64 { min, max } => (1, min.to_le_bytes(), max.to_le_bytes()),
    });
    match encoded {
        None => out.push(0),
        Some((presence, min, max)) => {
            out.push(presence);
            out.extend_from_slice(&min);
            out.extend_from_slice(&max);
        }
    }
}

/// The zone-map field's reader — shared by the segment format's column
/// entries and the manifest's segment records, which is what keeps the
/// two encodings one encoding.
fn decode_zone_map(
    reader: &mut Reader<'_>,
    column_type: ColumnType,
    name: &str,
) -> Result<Option<ZoneMap>, FormatError> {
    let presence = reader.u8()?;
    if presence & !0b11 != 0 || (presence & 0b10 != 0 && (presence & 1 == 0)) {
        return Err(FormatError::Corrupt(format!(
            "invalid zone-map presence byte {presence:#04x} for '{name}'"
        )));
    }
    if presence & 1 == 0 {
        return Ok(None);
    }
    let has_nan = presence & 0b10 != 0;
    let min = reader.take(8)?;
    let max = reader.take(8)?;
    Ok(Some(match column_type {
        ColumnType::F64 => ZoneMap::F64 {
            min: f64::from_le_bytes(min.try_into().unwrap()),
            max: f64::from_le_bytes(max.try_into().unwrap()),
            has_nan,
        },
        ColumnType::I64 if has_nan => {
            return Err(FormatError::Corrupt(format!(
                "i64 column '{name}' carries the NaN zone-map bit"
            )))
        }
        ColumnType::I64 => ZoneMap::I64 {
            min: i64::from_le_bytes(min.try_into().unwrap()),
            max: i64::from_le_bytes(max.try_into().unwrap()),
        },
        ColumnType::Key => {
            return Err(FormatError::Corrupt(format!(
                "key column '{name}' carries a zone map"
            )))
        }
    }))
}

fn encode_validity(out: &mut Vec<u8>, validity: Option<&Bitmap>) {
    match validity {
        None => out.push(0),
        Some(bitmap) => {
            out.push(1);
            let bytes = bitmap.as_bytes();
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }
    }
}

/// A bounds-checked little-endian reader; every truncation is an error,
/// never a panic.
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], FormatError> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| FormatError::Corrupt("unexpected end of file".to_owned()))?;
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, FormatError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, FormatError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Decodes segment bytes (v1 or trailered v2), verifying magic,
/// version, and checksum.
pub fn decode_segment(bytes: &[u8]) -> Result<Segment, FormatError> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(8)? != MAGIC {
        return Err(FormatError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION && version != VERSION_TRAILERED {
        return Err(FormatError::UnsupportedVersion(version));
    }
    if reader.u16()? != 0 {
        return Err(FormatError::Corrupt(
            "reserved header bytes are not zero".to_owned(),
        ));
    }
    let stored = reader.u32()?;
    let computed = crc32c(&bytes[PAYLOAD_OFFSET..]);
    if stored != computed {
        return Err(FormatError::ChecksumMismatch { stored, computed });
    }
    let base_row_id = reader.u64()?;
    let row_count = usize::try_from(reader.u64()?)
        .map_err(|_| FormatError::Corrupt("row count exceeds this platform".to_owned()))?;
    let ordering_key = reader.u32()? as usize;
    let flags = reader.u32()?;
    let ordered = flags & 1 != 0;
    let column_count = reader.u32()? as usize;
    if ordering_key >= column_count {
        return Err(FormatError::Corrupt(format!(
            "ordering key index {ordering_key} out of range for {column_count} columns"
        )));
    }
    let mut fields = Vec::with_capacity(column_count);
    let mut columns = Vec::with_capacity(column_count);
    let mut zone_maps = Vec::with_capacity(column_count);
    for _ in 0..column_count {
        let (field, column, zone_map) = decode_column(&mut reader, row_count)?;
        fields.push(field);
        columns.push(column);
        zone_maps.push(zone_map);
    }
    let mut sequence = SequenceInfo::RowIds;
    let mut superseded = None;
    if version == VERSION_TRAILERED {
        let section_count = reader.u16()?;
        for _ in 0..section_count {
            let tag = reader.u16()?;
            let length = reader.u32()? as usize;
            let payload = reader.take(length)?;
            match tag {
                TRAILER_SEQUENCE => {
                    sequence = decode_sequence_section(payload, row_count)?;
                }
                TRAILER_SUPERSEDED => {
                    let signed = decode_delta_of_delta(payload, row_count)?;
                    superseded = Some(
                        signed
                            .into_iter()
                            .map(|value| value as u64)
                            .collect::<Vec<u64>>(),
                    );
                }
                // Unknown trailer tags are skipped whole — the same
                // forward-compatibility contract as manifest sections.
                _ => {}
            }
        }
    }
    if reader.position != bytes.len() {
        return Err(FormatError::Corrupt(format!(
            "{} bytes remain after the last {}",
            bytes.len() - reader.position,
            if version == VERSION_TRAILERED {
                "trailer section"
            } else {
                "column"
            }
        )));
    }
    let batch = RecordBatch::new(Schema::new(fields), columns);
    Ok(Segment::from_parts(
        batch,
        ordering_key,
        ordered,
        base_row_id,
        zone_maps,
        sequence,
        superseded,
    ))
}

/// Decodes a v2 segment's sequence trailer section.
fn decode_sequence_section(payload: &[u8], rows: usize) -> Result<SequenceInfo, FormatError> {
    match payload.first() {
        Some(&SEQUENCE_CONTIGUOUS) => {
            let base: [u8; 8] = payload[1..]
                .try_into()
                .map_err(|_| FormatError::Corrupt("contiguous sequence base is 8 bytes".into()))?;
            Ok(SequenceInfo::Contiguous {
                base: u64::from_le_bytes(base),
            })
        }
        Some(&SEQUENCE_EXPLICIT) => {
            let signed = decode_delta_of_delta(&payload[1..], rows)?;
            Ok(SequenceInfo::Explicit(
                signed.into_iter().map(|value| value as u64).collect(),
            ))
        }
        Some(&state) => Err(FormatError::Corrupt(format!(
            "unknown sequence state byte {state}"
        ))),
        None => Err(FormatError::Corrupt("empty sequence section".into())),
    }
}

/// First bytes of every manifest file.
pub const MANIFEST_MAGIC: [u8; 8] = *b"TALLYMFT";
/// The manifest format version this module writes for tables with no
/// section content — byte-identical to the original format, so
/// untouched tables never see a version bump.
pub const MANIFEST_VERSION: u16 = 1;
/// The sectioned manifest version (M4.4): v1's exact layout, then
/// `section_count: u16` followed by `tag: u16, length: u32, payload`
/// per section. Readers skip unknown tags, so sections fill in over
/// time without another version bump. Assigned tags: 1 = segment
/// records (the residency design, 2026-07-30 — see [`SegmentRecord`]:
/// per live segment its name, row span, ordering flag, sequence
/// summary, and zone maps, so an open prunes and plans before touching
/// any segment file), 2 = knowledge state (the `u64` ingest-sequence
/// watermark — see [`ManifestSections::next_sequence`]), 3 = history
/// segments.
pub const MANIFEST_VERSION_SECTIONED: u16 = 2;

/// What the manifest knows about one flushed segment without opening
/// its file: everything query planning prunes on and everything reopen
/// verifies — the metadata half of the residency design (ruled
/// 2026-07-30: prune-metadata lives in a manifest section, so an open
/// touches no segment file until a query actually needs its rows).
#[derive(Clone, PartialEq, Debug)]
pub struct SegmentRecord {
    /// The backend object name (`seg-<generation>-<base>.tlyseg`).
    pub name: String,
    /// Id of the segment's first row.
    pub base_row_id: u64,
    /// Rows in the segment.
    pub rows: u64,
    /// Whether the ordering key arrived non-decreasing.
    pub ordered: bool,
    /// Where the segment's birth sequences stand — enough for the
    /// watermark fold and the reader's supersession-visibility rule;
    /// the per-row array (if any) stays in the segment file.
    pub sequence: SequenceSummary,
    /// Per-column zone maps, aligned to the schema; `None` per the
    /// segment format's meaning (no valid comparable values).
    pub zone_maps: Vec<Option<ZoneMap>>,
}

impl SegmentRecord {
    /// Derives the record from a decoded segment — the one constructor,
    /// so a record can never disagree with the segment it describes.
    pub fn of(name: String, segment: &Segment) -> SegmentRecord {
        // Every stored segment carries zone maps (the write buffer and
        // compaction both compute them at freeze). A maps-free segment
        // here would record an all-`None` row, which *means* every
        // column is all-null — a silent pruning corruption, refused.
        assert!(
            segment.zone_maps_present(),
            "a stored segment always carries zone maps"
        );
        let rows = segment.batch().num_rows();
        SegmentRecord {
            name,
            base_row_id: segment.base_row_id(),
            rows: rows as u64,
            ordered: segment.is_ordered(),
            sequence: match segment.sequence_info() {
                SequenceInfo::RowIds => SequenceSummary::RowIds,
                SequenceInfo::Contiguous { base } => SequenceSummary::Contiguous { base: *base },
                SequenceInfo::Explicit(_) => SequenceSummary::Explicit {
                    end: segment.sequence_end(),
                },
            },
            zone_maps: (0..segment.batch().columns().len())
                .map(|index| segment.zone_map(index).copied())
                .collect(),
        }
    }

    /// The sequence one past the segment's largest birth — the value
    /// [`Segment::sequence_end`] computes from data, here from metadata.
    pub fn sequence_end(&self) -> u64 {
        match self.sequence {
            SequenceSummary::RowIds => self.base_row_id + self.rows,
            SequenceSummary::Contiguous { base } => base + self.rows,
            SequenceSummary::Explicit { end } => end,
        }
    }

    /// Whether this segment's sequences have left the virtual state
    /// (the record-level form of `sequence_info() != RowIds`).
    pub fn diverged(&self) -> bool {
        self.sequence != SequenceSummary::RowIds
    }
}

/// A [`SequenceInfo`] with the per-row array reduced to its end: the
/// manifest carries where a segment's sequences *stand*, not what they
/// *are*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SequenceSummary {
    /// Sequence == row id.
    RowIds,
    /// Row `i` carries sequence `base + i`.
    Contiguous {
        /// The first row's sequence.
        base: u64,
    },
    /// Non-contiguous per-row births; only the fold survives here.
    Explicit {
        /// One past the largest birth sequence.
        end: u64,
    },
}

/// Manifest section content beyond the v1 core. `Default` is the
/// empty state, which encodes as plain v1.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ManifestSections {
    /// The generation's live segments in base-row-id order, with the
    /// metadata an open needs before touching any segment file. Empty
    /// on manifests from writers older than this section (or tables
    /// with nothing flushed): reopen then falls back to scanning the
    /// backend and re-earns the section at its next manifest write.
    pub segments: Vec<SegmentRecord>,
    /// The ingest-sequence watermark: the sequence the next appended
    /// row will receive. `Some` exactly when the table has diverged —
    /// run a retaining compaction that broke sequence == row id — and
    /// the counter must advance independently of reassigned row ids
    /// (ids compact downward; birth sequences never reuse). `None`
    /// while virtual: the next sequence *is* the next row id, derived,
    /// nothing stored.
    pub next_sequence: Option<u64>,
    /// Segments holding superseded row versions: excluded from
    /// latest-knowledge scans, entered only under `ASOF`.
    pub history: Vec<String>,
}

impl ManifestSections {
    /// Whether encoding needs the sectioned version at all.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.next_sequence.is_none() && self.history.is_empty()
    }

    /// Whether the table has ever diverged (sequence != row id).
    pub fn diverged(&self) -> bool {
        self.next_sequence.is_some()
    }
}

const SECTION_SEGMENTS: u16 = 1;
const SECTION_KNOWLEDGE: u16 = 2;
const SECTION_HISTORY: u16 = 3;

/// A decoded table manifest: what reopen verifies against and the
/// generation the backend is committed to.
#[derive(Clone, PartialEq, Debug)]
pub struct Manifest {
    /// The table's schema.
    pub schema: Schema,
    /// Index of the ordering-key column.
    pub ordering_key: usize,
    /// The committed compaction generation.
    pub generation: u64,
    /// Section content (v2); empty for v1 manifests.
    pub sections: ManifestSections,
}

/// Encodes a manifest: byte-identical v1 while `sections` is empty
/// (the golden-locked layout untouched tables keep forever), the
/// sectioned v2 once any section has content.
pub fn encode_manifest(
    schema: &Schema,
    ordering_key: usize,
    generation: u64,
    sections: &ManifestSections,
) -> Vec<u8> {
    let version = if sections.is_empty() {
        MANIFEST_VERSION
    } else {
        MANIFEST_VERSION_SECTIONED
    };
    let mut out = Vec::new();
    out.extend_from_slice(&MANIFEST_MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
    out.extend_from_slice(&generation.to_le_bytes());
    out.extend_from_slice(&(ordering_key as u32).to_le_bytes());
    out.extend_from_slice(&(schema.fields().len() as u32).to_le_bytes());
    for field in schema.fields() {
        encode_field(&mut out, field);
    }
    if version == MANIFEST_VERSION_SECTIONED {
        let mut section_bytes: Vec<(u16, Vec<u8>)> = Vec::new();
        if !sections.segments.is_empty() {
            let mut payload = Vec::new();
            payload.extend_from_slice(&(sections.segments.len() as u32).to_le_bytes());
            for record in &sections.segments {
                payload.extend_from_slice(&(record.name.len() as u16).to_le_bytes());
                payload.extend_from_slice(record.name.as_bytes());
                payload.extend_from_slice(&record.base_row_id.to_le_bytes());
                payload.extend_from_slice(&record.rows.to_le_bytes());
                payload.push(u8::from(record.ordered));
                match record.sequence {
                    SequenceSummary::RowIds => payload.push(0),
                    SequenceSummary::Contiguous { base } => {
                        payload.push(1);
                        payload.extend_from_slice(&base.to_le_bytes());
                    }
                    SequenceSummary::Explicit { end } => {
                        payload.push(2);
                        payload.extend_from_slice(&end.to_le_bytes());
                    }
                }
                payload.extend_from_slice(&(record.zone_maps.len() as u32).to_le_bytes());
                for zone_map in &record.zone_maps {
                    encode_zone_map(&mut payload, zone_map.as_ref());
                }
            }
            section_bytes.push((SECTION_SEGMENTS, payload));
        }
        if let Some(next_sequence) = sections.next_sequence {
            section_bytes.push((SECTION_KNOWLEDGE, next_sequence.to_le_bytes().to_vec()));
        }
        if !sections.history.is_empty() {
            let mut payload = Vec::new();
            payload.extend_from_slice(&(sections.history.len() as u32).to_le_bytes());
            for name in &sections.history {
                payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
                payload.extend_from_slice(name.as_bytes());
            }
            section_bytes.push((SECTION_HISTORY, payload));
        }
        out.extend_from_slice(&(section_bytes.len() as u16).to_le_bytes());
        for (tag, payload) in section_bytes {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&payload);
        }
    }
    let crc = crc32c(&out[PAYLOAD_OFFSET..]);
    out[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Decodes manifest bytes of either version, verifying magic, version,
/// and checksum; unknown v2 section tags are skipped, not refused.
pub fn decode_manifest(bytes: &[u8]) -> Result<Manifest, FormatError> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(8)? != MANIFEST_MAGIC {
        return Err(FormatError::BadMagic);
    }
    let version = reader.u16()?;
    if version != MANIFEST_VERSION && version != MANIFEST_VERSION_SECTIONED {
        return Err(FormatError::UnsupportedVersion(version));
    }
    if reader.u16()? != 0 {
        return Err(FormatError::Corrupt(
            "reserved header bytes are not zero".to_owned(),
        ));
    }
    let stored = reader.u32()?;
    let computed = crc32c(&bytes[PAYLOAD_OFFSET..]);
    if stored != computed {
        return Err(FormatError::ChecksumMismatch { stored, computed });
    }
    let generation = reader.u64()?;
    let ordering_key = reader.u32()? as usize;
    let column_count = reader.u32()? as usize;
    if ordering_key >= column_count {
        return Err(FormatError::Corrupt(format!(
            "ordering key index {ordering_key} out of range for {column_count} columns"
        )));
    }
    let mut fields = Vec::with_capacity(column_count);
    for _ in 0..column_count {
        fields.push(decode_field(&mut reader)?);
    }
    let mut sections = ManifestSections::default();
    if version == MANIFEST_VERSION_SECTIONED {
        let section_count = reader.u16()? as usize;
        for _ in 0..section_count {
            let tag = reader.u16()?;
            let length = reader.u32()? as usize;
            let payload = reader.take(length)?;
            match tag {
                SECTION_SEGMENTS => {
                    let mut inner = Reader {
                        bytes: payload,
                        position: 0,
                    };
                    let count = inner.u32()? as usize;
                    for _ in 0..count {
                        let len = inner.u16()? as usize;
                        let name = std::str::from_utf8(inner.take(len)?)
                            .map_err(|_| {
                                FormatError::Corrupt("segment record name is not UTF-8".to_owned())
                            })?
                            .to_owned();
                        let base_row_id = inner.u64()?;
                        let rows = inner.u64()?;
                        let ordered = match inner.u8()? {
                            0 => false,
                            1 => true,
                            other => {
                                return Err(FormatError::Corrupt(format!(
                                    "segment record '{name}' has ordered byte {other}"
                                )))
                            }
                        };
                        let sequence = match inner.u8()? {
                            0 => SequenceSummary::RowIds,
                            1 => SequenceSummary::Contiguous { base: inner.u64()? },
                            2 => SequenceSummary::Explicit { end: inner.u64()? },
                            other => {
                                return Err(FormatError::Corrupt(format!(
                                    "segment record '{name}' has sequence kind {other}"
                                )))
                            }
                        };
                        let map_count = inner.u32()? as usize;
                        if map_count != fields.len() {
                            return Err(FormatError::Corrupt(format!(
                                "segment record '{name}' carries {map_count} zone maps \
                                 for {} columns",
                                fields.len()
                            )));
                        }
                        let mut zone_maps = Vec::with_capacity(map_count);
                        for field in &fields {
                            zone_maps.push(decode_zone_map(
                                &mut inner,
                                field.column_type(),
                                &name,
                            )?);
                        }
                        sections.segments.push(SegmentRecord {
                            name,
                            base_row_id,
                            rows,
                            ordered,
                            sequence,
                            zone_maps,
                        });
                    }
                    if inner.position != payload.len() {
                        return Err(FormatError::Corrupt(
                            "trailing bytes in the segments section".to_owned(),
                        ));
                    }
                }
                SECTION_KNOWLEDGE => {
                    let watermark: [u8; 8] = payload.try_into().map_err(|_| {
                        FormatError::Corrupt("knowledge section is 8 bytes".to_owned())
                    })?;
                    sections.next_sequence = Some(u64::from_le_bytes(watermark));
                }
                SECTION_HISTORY => {
                    let mut inner = Reader {
                        bytes: payload,
                        position: 0,
                    };
                    let count = inner.u32()? as usize;
                    for _ in 0..count {
                        let len = inner.u16()? as usize;
                        let name = std::str::from_utf8(inner.take(len)?)
                            .map_err(|_| {
                                FormatError::Corrupt("history segment name is not UTF-8".to_owned())
                            })?
                            .to_owned();
                        sections.history.push(name);
                    }
                    if inner.position != payload.len() {
                        return Err(FormatError::Corrupt(
                            "trailing bytes in the history section".to_owned(),
                        ));
                    }
                }
                // Unknown tags (zone maps to come, and anything newer
                // than this reader) are skipped whole — that is the
                // sectioned format's forward-compatibility contract.
                _ => {}
            }
        }
    }
    if reader.position != bytes.len() {
        return Err(FormatError::Corrupt(format!(
            "{} bytes remain after the last {}",
            bytes.len() - reader.position,
            if version == MANIFEST_VERSION_SECTIONED {
                "section"
            } else {
                "column"
            }
        )));
    }
    Ok(Manifest {
        schema: Schema::new(fields),
        ordering_key,
        generation,
        sections,
    })
}

fn decode_column(
    reader: &mut Reader<'_>,
    rows: usize,
) -> Result<(Field, Column, Option<ZoneMap>), FormatError> {
    let field = decode_field(reader)?;
    let name = field.name().to_owned();
    let column_type = field.column_type();
    let codec = Codec::from_tag(reader.u8()?)
        .ok_or_else(|| FormatError::Corrupt(format!("unknown codec for '{name}'")))?;
    let zone_map = decode_zone_map(reader, column_type, &name)?;
    let validity = if reader.u8()? != 0 {
        let byte_len = reader.u32()? as usize;
        let bytes = reader.take(byte_len)?;
        if byte_len < rows.div_ceil(8) {
            return Err(FormatError::Corrupt(format!(
                "validity bitmap for '{name}' is shorter than {rows} rows"
            )));
        }
        Some(Bitmap::from_bools(
            (0..rows).map(|row| bytes[row / 8] >> (row % 8) & 1 == 1),
        ))
    } else {
        None
    };
    let values_len = usize::try_from(reader.u64()?)
        .map_err(|_| FormatError::Corrupt("column exceeds this platform".to_owned()))?;
    let values = reader.take(values_len)?;
    let column = match column_type {
        ColumnType::F64 => {
            let buffer = match codec {
                Codec::Uncompressed => decode_f64(values, rows, &name)?,
                Codec::AlpF64 => Buffer::from_slice(&decode_alp_f64(values, rows)?),
                other => {
                    return Err(FormatError::Corrupt(format!(
                        "f64 column '{name}' carries codec {other:?}"
                    )))
                }
            };
            Column::Numeric(NumericData::F64(match validity {
                Some(bitmap) => NumericColumn::new_nullable(buffer, bitmap),
                None => NumericColumn::new_non_null(buffer),
            }))
        }
        ColumnType::I64 => {
            let buffer = match codec {
                Codec::Uncompressed => decode_i64(values, rows, &name)?,
                Codec::DeltaOfDeltaI64 => Buffer::from_slice(&decode_delta_of_delta(values, rows)?),
                Codec::ForI64 => Buffer::from_slice(&decode_for_i64(values, rows)?),
                other => {
                    return Err(FormatError::Corrupt(format!(
                        "i64 column '{name}' carries codec {other:?}"
                    )))
                }
            };
            Column::Numeric(NumericData::I64(match validity {
                Some(bitmap) => NumericColumn::new_nullable(buffer, bitmap),
                None => NumericColumn::new_non_null(buffer),
            }))
        }
        ColumnType::Key => {
            let codes = match codec {
                Codec::ForU32 => Buffer::from_slice(&decode_for_u32(values, rows)?),
                Codec::Uncompressed => {
                    if values.len() != rows * 4 {
                        return Err(FormatError::Corrupt(format!(
                            "key column '{name}' holds {} bytes of codes, expected {}",
                            values.len(),
                            rows * 4
                        )));
                    }
                    let mut codes = Buffer::with_capacity(rows);
                    for chunk in values.chunks_exact(4) {
                        codes.push(u32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    codes
                }
                other => {
                    return Err(FormatError::Corrupt(format!(
                        "key column '{name}' carries codec {other:?}"
                    )))
                }
            };
            let dictionary = decode_dictionary(reader, &name)?;
            for (row, &code) in codes.as_slice().iter().enumerate() {
                let in_range = (code as usize) < dictionary.len();
                let null_slot = code == 0 && dictionary.is_empty();
                if !in_range && !null_slot {
                    return Err(FormatError::Corrupt(format!(
                        "key column '{name}' code {code} at row {row} exceeds its dictionary"
                    )));
                }
            }
            Column::Key(match validity {
                Some(bitmap) => KeyColumn::new_nullable(codes, bitmap, dictionary),
                None => KeyColumn::new_non_null(codes, dictionary),
            })
        }
    };
    Ok((field, column, zone_map))
}

fn decode_f64(bytes: &[u8], rows: usize, name: &str) -> Result<Buffer<f64>, FormatError> {
    if bytes.len() != rows * 8 {
        return Err(FormatError::Corrupt(format!(
            "f64 column '{name}' holds {} bytes, expected {}",
            bytes.len(),
            rows * 8
        )));
    }
    let mut buffer = Buffer::with_capacity(rows);
    for chunk in bytes.chunks_exact(8) {
        buffer.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(buffer)
}

fn decode_i64(bytes: &[u8], rows: usize, name: &str) -> Result<Buffer<i64>, FormatError> {
    if bytes.len() != rows * 8 {
        return Err(FormatError::Corrupt(format!(
            "i64 column '{name}' holds {} bytes, expected {}",
            bytes.len(),
            rows * 8
        )));
    }
    let mut buffer = Buffer::with_capacity(rows);
    for chunk in bytes.chunks_exact(8) {
        buffer.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(buffer)
}

fn decode_dictionary(reader: &mut Reader<'_>, name: &str) -> Result<Dictionary, FormatError> {
    let entries = reader.u32()? as usize;
    let mut offsets = Vec::with_capacity(entries + 1);
    for _ in 0..entries + 1 {
        offsets.push(reader.i32()?);
    }
    let bytes_len = reader.u32()? as usize;
    let bytes = reader.take(bytes_len)?;
    let mut dictionary = Dictionary::new();
    for pair in offsets.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let valid = 0 <= start && start <= end && (end as usize) <= bytes.len();
        if !valid {
            return Err(FormatError::Corrupt(format!(
                "dictionary offsets for '{name}' are not monotonic in range"
            )));
        }
        let value = std::str::from_utf8(&bytes[start as usize..end as usize]).map_err(|_| {
            FormatError::Corrupt(format!("dictionary value for '{name}' is not UTF-8"))
        })?;
        let code = dictionary.intern(value);
        if code as usize != dictionary.len() - 1 {
            return Err(FormatError::Corrupt(format!(
                "dictionary for '{name}' repeats the value '{value}'"
            )));
        }
    }
    Ok(dictionary)
}

// ---------------------------------------------------------------- WAL

/// WAL file magic. The write-ahead log is a sidecar: it never changes
/// the segment format (whose bytes stay locked by the committed
/// golden), and it exists only between flushes — truncated whenever the
/// rows it guards become segment-durable.
pub(crate) const WAL_MAGIC: [u8; 8] = *b"TALLYWAL";
/// WAL format version. Version 2 (M4.5, issue #73) added one control
/// record — the supersession bracket — to the record grammar; v1 logs
/// (which cannot contain control records) still replay.
pub(crate) const WAL_VERSION: u16 = 2;
/// The header's encoded size: magic (8) + version (2) + generation (8)
/// plus base row id (8) plus their CRC-32C (4) — the header carries a
/// checksum like every other structure in the format. A log file
/// shorter than this holds no acknowledged record (records follow the
/// header in the same file) and reads as an empty log, not corruption.
pub(crate) const WAL_HEADER_LEN: usize = 30;

/// Encodes the WAL header: magic, version, generation, and the row id
/// of the first record — replay skips records already covered by
/// flushed segments (a crash between segment publish and WAL truncate
/// leaves such a prefix) and refuses logs from another generation
/// (compaction reassigns row ids, so cross-generation replay would be
/// in the wrong id space).
pub(crate) fn encode_wal_header(generation: u64, base_row_id: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(WAL_HEADER_LEN);
    out.extend_from_slice(&WAL_MAGIC);
    out.extend_from_slice(&WAL_VERSION.to_le_bytes());
    out.extend_from_slice(&generation.to_le_bytes());
    out.extend_from_slice(&base_row_id.to_le_bytes());
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// One row as a WAL record: `u32` payload length, the payload (one
/// presence-tagged cell per schema column), `u32` CRC-32C of the
/// payload. A record whose length, payload, or CRC cannot be read
/// whole is a torn tail — the crash boundary — never an error.
///
/// Cell presence tags are 0..=3; a payload whose first byte is `0xFF`
/// is a **control record** instead (see [`encode_wal_supersession`]).
pub(crate) fn encode_wal_record(row: &[RowValue<'_>]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(16 * row.len());
    for cell in row {
        match cell {
            RowValue::Null => payload.push(0u8),
            RowValue::I64(value) => {
                payload.push(1);
                payload.extend_from_slice(&value.to_le_bytes());
            }
            RowValue::F64(value) => {
                payload.push(2);
                payload.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            RowValue::Key(value) => {
                payload.push(3);
                payload.extend_from_slice(&(value.len() as u32).to_le_bytes());
                payload.extend_from_slice(value.as_bytes());
            }
        }
    }
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&crc32c(&payload).to_le_bytes());
    out
}

/// The supersession bracket (issue #73): announces that the next
/// `replacements` row records belong to one mutation whose every row is
/// born at `sequence` — the single knowledge coordinate at which the
/// mutation's victims also die. The bracket's *commit evidence* is the
/// delete log carrying `superseding == sequence`; replay finding the
/// bracket at the log's clean tail without that evidence drops the
/// bracketed rows whole, which is what makes a crashed `UPDATE`
/// old-or-new instead of torn.
pub(crate) fn encode_wal_supersession(sequence: u64, replacements: u64) -> Vec<u8> {
    let mut payload = Vec::with_capacity(18);
    payload.push(0xFF);
    payload.push(1); // control tag: begin supersession
    payload.extend_from_slice(&sequence.to_le_bytes());
    payload.extend_from_slice(&replacements.to_le_bytes());
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&crc32c(&payload).to_le_bytes());
    out
}

/// A WAL row, owned (`RowValue` borrows; replay needs owned cells).
#[derive(Debug, PartialEq)]
pub(crate) enum WalCell {
    Null,
    I64(i64),
    F64(f64),
    Key(String),
}

impl WalCell {
    /// The borrowed view [`crate::Store::append`] takes.
    pub fn as_row_value(&self) -> RowValue<'_> {
        match self {
            WalCell::Null => RowValue::Null,
            WalCell::I64(value) => RowValue::I64(*value),
            WalCell::F64(value) => RowValue::F64(*value),
            WalCell::Key(value) => RowValue::Key(value),
        }
    }
}

/// One replayable WAL entry.
#[derive(Debug, PartialEq)]
pub(crate) enum WalEntry {
    /// An ordinary appended row.
    Row(Vec<WalCell>),
    /// A supersession bracket: the next `replacements` rows share the
    /// birth coordinate `sequence` (see [`encode_wal_supersession`]).
    Supersession { sequence: u64, replacements: u64 },
}

/// A decoded WAL: where its records start in the row-id space, and the
/// clean-prefix entries that survived.
pub(crate) struct WalContents {
    pub generation: u64,
    pub base_row_id: u64,
    pub entries: Vec<WalEntry>,
}

/// Decodes a WAL file. Header corruption is an error (the file is not a
/// WAL); a torn or corrupt *record* ends the clean prefix silently —
/// that is the crash boundary working as designed, and everything
/// before it is intact (each record carries its own CRC).
pub(crate) fn decode_wal(bytes: &[u8], columns: usize) -> Result<WalContents, FormatError> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(8)? != WAL_MAGIC {
        return Err(FormatError::Corrupt("not a WAL file".to_owned()));
    }
    let version = reader.u16()?;
    if version != 1 && version != WAL_VERSION {
        return Err(FormatError::Corrupt(format!(
            "WAL version {version} is not readable (this build reads 1 and {WAL_VERSION})"
        )));
    }
    let generation = reader.u64()?;
    let base_row_id = reader.u64()?;
    let stored_crc = reader.u32()?;
    if crc32c(&bytes[..WAL_HEADER_LEN - 4]) != stored_crc {
        return Err(FormatError::Corrupt(
            "WAL header checksum mismatch".to_owned(),
        ));
    }
    let mut entries = Vec::new();
    // Records are self-delimiting, so a failed read simply ends the
    // clean prefix — no start offset needs remembering.
    while let Ok(length) = reader.u32() {
        let Ok(payload) = reader.take(length as usize) else {
            break;
        };
        let Ok(stored_crc) = reader.u32() else { break };
        if crc32c(payload) != stored_crc {
            break; // torn or corrupt record: the clean prefix ends here
        }
        if payload.first() == Some(&0xFF) {
            // A control record. Unknown control tags or malformed
            // payloads end the clean prefix like any other damage.
            if payload.len() != 18 || payload[1] != 1 {
                break;
            }
            entries.push(WalEntry::Supersession {
                sequence: u64::from_le_bytes(payload[2..10].try_into().unwrap()),
                replacements: u64::from_le_bytes(payload[10..18].try_into().unwrap()),
            });
            continue;
        }
        let mut cells = Reader {
            bytes: payload,
            position: 0,
        };
        let mut row = Vec::with_capacity(columns);
        let mut clean = true;
        for _ in 0..columns {
            let cell = match cells.u8() {
                Ok(0) => WalCell::Null,
                Ok(1) => match cells.take(8) {
                    Ok(b) => WalCell::I64(i64::from_le_bytes(b.try_into().unwrap())),
                    Err(_) => {
                        clean = false;
                        break;
                    }
                },
                Ok(2) => match cells.take(8) {
                    Ok(b) => {
                        WalCell::F64(f64::from_bits(u64::from_le_bytes(b.try_into().unwrap())))
                    }
                    Err(_) => {
                        clean = false;
                        break;
                    }
                },
                Ok(3) => {
                    let ok = cells.u32().ok().and_then(|len| {
                        cells
                            .take(len as usize)
                            .ok()
                            .and_then(|b| std::str::from_utf8(b).ok().map(str::to_owned))
                    });
                    match ok {
                        Some(text) => WalCell::Key(text),
                        None => {
                            clean = false;
                            break;
                        }
                    }
                }
                _ => {
                    clean = false;
                    break;
                }
            };
            row.push(cell);
        }
        if !clean || row.len() != columns {
            break; // CRC passed but shape is wrong: stop, don't guess
        }
        entries.push(WalEntry::Row(row));
    }
    Ok(WalContents {
        generation,
        base_row_id,
        entries,
    })
}
