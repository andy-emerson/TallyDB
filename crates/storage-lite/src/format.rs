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
//! version      u16 (this module writes 1)
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
//!   zone map    u8 present; if 1: min 8B, max 8B (f64 bits or i64,
//!                  per column type; computed over valid, non-NaN values)
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
//! ## The manifest format, version 1
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
//! version      u16 (this module writes 1)
//! reserved     u16 (zero)
//! crc32c       u32 — CRC-32C of every byte after this field
//! generation   u64 — the committed compaction generation
//! ordering_key u32 (column index)
//! column_count u32
//! then per column (the segment format's schema prefix, same registries):
//!   name_len u16, name bytes (UTF-8)
//!   column_type u8, nullable u8, logical u8 tag + u8 payload
//! ```

use crate::codec::{decode_delta_of_delta, encode_delta_of_delta, Codec, CodecError};
use crate::mem::{Segment, ZoneMap};
use arrow_lite::{
    Bitmap, Buffer, Column, ColumnType, Dictionary, Field, KeyColumn, LogicalType, NumericColumn,
    NumericData, RecordBatch, Schema,
};
use std::fmt;

/// First bytes of every segment file.
pub const MAGIC: [u8; 8] = *b"TALLYSEG";
/// The format version this module writes.
pub const VERSION: u16 = 1;

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
fn crc32c(bytes: &[u8]) -> u32 {
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

/// Encodes a segment into its v1 bytes.
pub fn encode_segment(segment: &Segment) -> Vec<u8> {
    let batch = segment.batch();
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
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
    let crc = crc32c(&out[PAYLOAD_OFFSET..]);
    out[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    out
}

/// The codec the writer chooses for a column — the one place that
/// policy lives. Decision #29: the ordering key of an ordered segment is
/// clock-like and takes delta-of-delta; everything else (including every
/// `f64` column, per the #30 interim ruling) is uncompressed.
fn writer_codec(segment: &Segment, column: &Column, is_ordering_key: bool) -> Codec {
    match column {
        Column::Numeric(NumericData::I64(_)) if is_ordering_key && segment.is_ordered() => {
            Codec::DeltaOfDeltaI64
        }
        _ => Codec::Uncompressed,
    }
}

fn encode_column(
    out: &mut Vec<u8>,
    segment: &Segment,
    field: &Field,
    column: &Column,
    is_ordering_key: bool,
    zone_map: Option<&ZoneMap>,
) {
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
            let mut bytes = Vec::with_capacity(numeric.len() * 8);
            for value in numeric.values().as_slice() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            push_values(out, &bytes);
        }
        Column::Numeric(NumericData::I64(numeric)) => {
            let bytes = match codec {
                Codec::DeltaOfDeltaI64 => encode_delta_of_delta(numeric.values().as_slice()),
                Codec::Uncompressed => {
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
            let mut bytes = Vec::with_capacity(keys.len() * 4);
            for code in keys.codes().as_slice() {
                bytes.extend_from_slice(&code.to_le_bytes());
            }
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
/// [`crate::mem::ZoneMap`]), absent when no valid non-NaN value exists
/// or the column is a key.
fn encode_zone_map(out: &mut Vec<u8>, zone_map: Option<&ZoneMap>) {
    let bounds: Option<([u8; 8], [u8; 8])> = zone_map.map(|zone_map| match zone_map {
        ZoneMap::F64 { min, max } => (min.to_le_bytes(), max.to_le_bytes()),
        ZoneMap::I64 { min, max } => (min.to_le_bytes(), max.to_le_bytes()),
    });
    match bounds {
        None => out.push(0),
        Some((min, max)) => {
            out.push(1);
            out.extend_from_slice(&min);
            out.extend_from_slice(&max);
        }
    }
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

/// Decodes v1 segment bytes, verifying magic, version, and checksum.
pub fn decode_segment(bytes: &[u8]) -> Result<Segment, FormatError> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(8)? != MAGIC {
        return Err(FormatError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
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
    if reader.position != bytes.len() {
        return Err(FormatError::Corrupt(format!(
            "{} bytes remain after the last column",
            bytes.len() - reader.position
        )));
    }
    let batch = RecordBatch::new(Schema::new(fields), columns);
    Ok(Segment::from_parts(
        batch,
        ordering_key,
        ordered,
        base_row_id,
        zone_maps,
    ))
}

/// First bytes of every manifest file.
pub const MANIFEST_MAGIC: [u8; 8] = *b"TALLYMFT";
/// The manifest format version this module writes.
pub const MANIFEST_VERSION: u16 = 1;

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
}

/// Encodes a manifest into its v1 bytes.
pub fn encode_manifest(schema: &Schema, ordering_key: usize, generation: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MANIFEST_MAGIC);
    out.extend_from_slice(&MANIFEST_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
    out.extend_from_slice(&generation.to_le_bytes());
    out.extend_from_slice(&(ordering_key as u32).to_le_bytes());
    out.extend_from_slice(&(schema.fields().len() as u32).to_le_bytes());
    for field in schema.fields() {
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
    let crc = crc32c(&out[PAYLOAD_OFFSET..]);
    out[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Decodes v1 manifest bytes, verifying magic, version, and checksum.
pub fn decode_manifest(bytes: &[u8]) -> Result<Manifest, FormatError> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(8)? != MANIFEST_MAGIC {
        return Err(FormatError::BadMagic);
    }
    let version = reader.u16()?;
    if version != MANIFEST_VERSION {
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
        fields.push(field);
    }
    if reader.position != bytes.len() {
        return Err(FormatError::Corrupt(format!(
            "{} bytes remain after the last column",
            bytes.len() - reader.position
        )));
    }
    Ok(Manifest {
        schema: Schema::new(fields),
        ordering_key,
        generation,
    })
}

fn decode_column(
    reader: &mut Reader<'_>,
    rows: usize,
) -> Result<(Field, Column, Option<ZoneMap>), FormatError> {
    let name_len = reader.u16()? as usize;
    let name = std::str::from_utf8(reader.take(name_len)?)
        .map_err(|_| FormatError::Corrupt("column name is not UTF-8".to_owned()))?
        .to_owned();
    let column_type = ColumnType::from_tag(reader.u8()?)
        .ok_or_else(|| FormatError::Corrupt(format!("unknown column type for '{name}'")))?;
    let nullable = reader.u8()? != 0;
    let logical_tag = reader.u8()?;
    let logical_payload = reader.u8()?;
    let logical = match logical_tag {
        0 => None,
        tag => Some(
            LogicalType::from_parts(tag, logical_payload).ok_or_else(|| {
                FormatError::Corrupt(format!("unknown logical type {tag} for '{name}'"))
            })?,
        ),
    };
    let codec = Codec::from_tag(reader.u8()?)
        .ok_or_else(|| FormatError::Corrupt(format!("unknown codec for '{name}'")))?;
    let zone_map = if reader.u8()? != 0 {
        let min = reader.take(8)?;
        let max = reader.take(8)?;
        Some(match column_type {
            ColumnType::F64 => ZoneMap::F64 {
                min: f64::from_le_bytes(min.try_into().unwrap()),
                max: f64::from_le_bytes(max.try_into().unwrap()),
            },
            ColumnType::I64 => ZoneMap::I64 {
                min: i64::from_le_bytes(min.try_into().unwrap()),
                max: i64::from_le_bytes(max.try_into().unwrap()),
            },
            ColumnType::Key => {
                return Err(FormatError::Corrupt(format!(
                    "key column '{name}' carries a zone map"
                )))
            }
        })
    } else {
        None
    };
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
            if codec != Codec::Uncompressed {
                return Err(FormatError::Corrupt(format!(
                    "f64 column '{name}' uses an i64 codec"
                )));
            }
            let buffer = decode_f64(values, rows, &name)?;
            Column::Numeric(NumericData::F64(match validity {
                Some(bitmap) => NumericColumn::new_nullable(buffer, bitmap),
                None => NumericColumn::new_non_null(buffer),
            }))
        }
        ColumnType::I64 => {
            let buffer = match codec {
                Codec::Uncompressed => decode_i64(values, rows, &name)?,
                Codec::DeltaOfDeltaI64 => Buffer::from_slice(&decode_delta_of_delta(values, rows)?),
            };
            Column::Numeric(NumericData::I64(match validity {
                Some(bitmap) => NumericColumn::new_nullable(buffer, bitmap),
                None => NumericColumn::new_non_null(buffer),
            }))
        }
        ColumnType::Key => {
            if codec != Codec::Uncompressed {
                return Err(FormatError::Corrupt(format!(
                    "key column '{name}' uses an i64 codec"
                )));
            }
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
    let mut field = Field::new(name, column_type, nullable);
    if let Some(logical) = logical {
        field = field.with_logical(logical);
    }
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
