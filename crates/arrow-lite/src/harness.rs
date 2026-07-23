//! `extern "C"` hooks for the PyArrow round-trip script (dev/CI only).
//!
//! Compiled only under the `oracle-harness` feature, which exists so
//! `tests/pyarrow_roundtrip.py` can drive this crate as a shared library:
//! build with `cargo build -p arrow-lite --features oracle-harness`, then
//! run the script against `target/debug/libarrow_lite.so`. Nothing here
//! is part of the engine's API.
//!
//! The script and this module agree on two fixtures, defined twice on
//! purpose — once here, once in Python — so each side is an independent
//! statement of the expected data:
//!
//! - the **canonical batch**: `ts` (timestamp ns), `sym` (u32-dictionary
//!   utf8), `px` (nullable f64 with real nulls), `qty` (i64), `amt`
//!   (decimal64(18, 2));
//! - the **stream slices**: three two-column (`x` f64, `n` i64) batches.

use crate::bitmap::Bitmap;
use crate::buffer::{Buffer, Element, NumericColumn};
use crate::cdata::{
    export_batch, export_stream, import_batch, ArrowArray, ArrowArrayStream, ArrowSchema,
    StreamReader,
};
use crate::column::{Column, ColumnType, NumericData};
use crate::key::KeyColumn;
use crate::logical::LogicalType;
use crate::schema::{Field, RecordBatch, Schema};

/// The canonical batch, in our types.
fn canonical_batch() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false).with_logical(LogicalType::TimestampNs),
        Field::new("sym", ColumnType::Key, false),
        Field::new("px", ColumnType::F64, true),
        Field::new("qty", ColumnType::I64, false),
        Field::new("amt", ColumnType::I64, false).with_logical(LogicalType::Decimal64 { scale: 2 }),
    ]);
    RecordBatch::new(
        schema,
        vec![
            Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                Buffer::from_slice(&[1_000, 2_000, 3_000, 4_000]),
            ))),
            Column::Key(KeyColumn::from_values(["AAPL", "MSFT", "AAPL", "TSLA"])),
            Column::Numeric(NumericData::F64(NumericColumn::new_nullable(
                Buffer::from_slice(&[101.5, 0.0, 99.25, 0.0]),
                Bitmap::from_bools([true, false, true, false]),
            ))),
            Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                Buffer::from_slice(&[10, 20, 30, 40]),
            ))),
            Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                Buffer::from_slice(&[19_999, 250, -75, 0]),
            ))),
        ],
    )
}

/// Stream slice `i`, in our types.
fn slice_batch(i: i64) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("x", ColumnType::F64, false),
        Field::new("n", ColumnType::I64, false),
    ]);
    RecordBatch::new(
        schema,
        vec![
            Column::Numeric(NumericData::F64(NumericColumn::new_non_null(
                Buffer::from_slice(&[i as f64, i as f64 + 0.5]),
            ))),
            Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                Buffer::from_slice(&[i * 10, i * 10 + 1]),
            ))),
        ],
    )
}

/// Logical equality: schema, validity, and values at valid rows (values
/// under null slots are unspecified across the interface).
fn logically_equal(actual: &RecordBatch, expected: &RecordBatch) -> bool {
    fn numeric_eq<T: Element>(a: &NumericColumn<T>, e: &NumericColumn<T>) -> bool {
        a.len() == e.len()
            && (0..a.len()).all(|row| {
                a.is_valid(row) == e.is_valid(row)
                    && (!a.is_valid(row) || a.values()[row] == e.values()[row])
            })
    }
    actual.schema() == expected.schema()
        && actual.num_rows() == expected.num_rows()
        && actual
            .columns()
            .iter()
            .zip(expected.columns())
            .all(|(a, e)| match (a, e) {
                (Column::Numeric(NumericData::F64(a)), Column::Numeric(NumericData::F64(e))) => {
                    numeric_eq(a, e)
                }
                (Column::Numeric(NumericData::I64(a)), Column::Numeric(NumericData::I64(e))) => {
                    numeric_eq(a, e)
                }
                (Column::Key(a), Column::Key(e)) => {
                    a.len() == e.len() && (0..a.len()).all(|row| a.value_at(row) == e.value_at(row))
                }
                _ => false,
            })
}

/// Return codes shared by the verify hooks.
const OK: i32 = 0;
const IMPORT_FAILED: i32 = 1;
const MISMATCH: i32 = 2;
const PANICKED: i32 = 3;

fn catch(f: impl FnOnce() -> i32 + std::panic::UnwindSafe) -> i32 {
    std::panic::catch_unwind(f).unwrap_or(PANICKED)
}

/// Writes the canonical batch into caller-provided structs.
///
/// # Safety
/// Both pointers must be valid, writable, and uninitialized (any prior
/// live export must have been released by the caller).
#[no_mangle]
pub unsafe extern "C" fn tallydb_oracle_export_batch(
    schema_out: *mut ArrowSchema,
    array_out: *mut ArrowArray,
) {
    let (schema, array) = export_batch(canonical_batch());
    // SAFETY: caller provides valid destinations.
    unsafe {
        schema_out.write(schema);
        array_out.write(array);
    }
}

/// Writes a stream of the three slice batches into a caller-provided
/// struct.
///
/// # Safety
/// As for [`tallydb_oracle_export_batch`].
#[no_mangle]
pub unsafe extern "C" fn tallydb_oracle_export_stream(out: *mut ArrowArrayStream) {
    let stream = export_stream(slice_batch(0).schema().clone(), (0..3).map(slice_batch));
    // SAFETY: caller provides a valid destination.
    unsafe { out.write(stream) };
}

/// Imports a foreign batch and verifies it equals the canonical batch.
/// Takes ownership of (and releases) both structs. Returns 0 on success.
///
/// # Safety
/// Both pointers must point at valid, unreleased C Data exports.
#[no_mangle]
pub unsafe extern "C" fn tallydb_oracle_verify_batch(
    schema: *mut ArrowSchema,
    array: *mut ArrowArray,
) -> i32 {
    // SAFETY: taking ownership of the caller's live structs by value.
    let (schema, array) = unsafe { (schema.read(), array.read()) };
    catch(move || {
        // SAFETY: live structs whose ownership we now hold.
        match unsafe { import_batch(schema, array) } {
            Ok(batch) if logically_equal(&batch, &canonical_batch()) => OK,
            Ok(batch) => {
                eprintln!("mismatch: imported {batch:?}");
                MISMATCH
            }
            Err(error) => {
                eprintln!("import failed: {error}");
                IMPORT_FAILED
            }
        }
    })
}

/// Imports a foreign stream and verifies it yields exactly the three
/// slice batches. Takes ownership of (and releases) the struct. Returns 0
/// on success.
///
/// # Safety
/// `stream` must point at a valid, unreleased C Data stream export.
#[no_mangle]
pub unsafe extern "C" fn tallydb_oracle_verify_stream(stream: *mut ArrowArrayStream) -> i32 {
    // SAFETY: taking ownership of the caller's live struct by value.
    let stream = unsafe { stream.read() };
    catch(move || {
        // SAFETY: a live stream whose ownership we now hold.
        let reader = match unsafe { StreamReader::new(stream) } {
            Ok(reader) => reader,
            Err(error) => {
                eprintln!("stream open failed: {error}");
                return IMPORT_FAILED;
            }
        };
        let batches: Result<Vec<RecordBatch>, _> = reader.collect();
        match batches {
            Ok(batches)
                if batches.len() == 3
                    && batches
                        .iter()
                        .enumerate()
                        .all(|(i, b)| logically_equal(b, &slice_batch(i as i64))) =>
            {
                OK
            }
            Ok(batches) => {
                eprintln!("mismatch: read {} batches: {batches:?}", batches.len());
                MISMATCH
            }
            Err(error) => {
                eprintln!("stream read failed: {error}");
                IMPORT_FAILED
            }
        }
    })
}
