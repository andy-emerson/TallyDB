//! Round-trip oracle: arrow-rs (dev-only) versus our hand-rolled layout.
//!
//! Every test crosses the C Data Interface in one direction or the other:
//! we export and the real implementation imports, or the reverse. The
//! oracle never links into the runtime — this file is the differential
//! harness issue #15 calls for, and it is what upgrades the M0 layout
//! claims (#10, #11, #13, #14) to cross-checked.
//!
//! The two sides' ABI structs are layout-identical by specification, so a
//! struct crosses sides by `transmute` — asserted below before any test
//! relies on it.

use arrow::array::{
    Array, ArrayData, ArrayRef, Decimal64Array, DictionaryArray, Float64Array, Int64Array,
    RecordBatchIterator, StringArray, StructArray, TimestampNanosecondArray, UInt32Array,
};
use arrow::datatypes::{
    DataType, Field as ArrowField, Schema as ArrowRsSchema, TimeUnit, UInt32Type,
};
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::RecordBatchReader;
use arrow_lite::{
    export_batch, export_stream, import_batch, Bitmap, Buffer, Column, ColumnType, Dictionary,
    Field, KeyColumn, LogicalType, NumericColumn, NumericData, RecordBatch, Schema,
};
use std::sync::Arc;

/// The transmute the whole harness rests on: both sides' structs are the
/// same C ABI struct.
#[test]
fn ffi_structs_are_layout_compatible() {
    use std::mem::{align_of, size_of};
    assert_eq!(
        size_of::<arrow_lite::ArrowSchema>(),
        size_of::<FFI_ArrowSchema>()
    );
    assert_eq!(
        size_of::<arrow_lite::ArrowArray>(),
        size_of::<FFI_ArrowArray>()
    );
    assert_eq!(
        size_of::<arrow_lite::ArrowArrayStream>(),
        size_of::<FFI_ArrowArrayStream>()
    );
    assert_eq!(
        align_of::<arrow_lite::ArrowSchema>(),
        align_of::<FFI_ArrowSchema>()
    );
    assert_eq!(
        align_of::<arrow_lite::ArrowArray>(),
        align_of::<FFI_ArrowArray>()
    );
}

/// Ours → arrow-rs: export a batch and let the oracle import it.
fn oracle_import(batch: RecordBatch) -> StructArray {
    let (schema, array) = export_batch(batch);
    // SAFETY: layout-compatible per the assertion test; ownership moves
    // to arrow-rs, whose drop calls our release callbacks.
    let (ffi_schema, ffi_array) = unsafe {
        (
            std::mem::transmute::<arrow_lite::ArrowSchema, FFI_ArrowSchema>(schema),
            std::mem::transmute::<arrow_lite::ArrowArray, FFI_ArrowArray>(array),
        )
    };
    // SAFETY: a live pair our exporter just produced.
    let data: ArrayData = unsafe { from_ffi(ffi_array, &ffi_schema) }.expect("oracle imports");
    data.validate_full().expect("oracle validates");
    StructArray::from(data)
}

/// arrow-rs → ours: let the oracle export and import it ourselves.
fn import_from_oracle(expected: &StructArray) -> RecordBatch {
    let (ffi_array, ffi_schema) = to_ffi(&expected.to_data()).expect("oracle exports");
    // SAFETY: layout-compatible; ownership moves to our importer, which
    // releases via arrow-rs's callbacks.
    let (schema, array) = unsafe {
        (
            std::mem::transmute::<FFI_ArrowSchema, arrow_lite::ArrowSchema>(ffi_schema),
            std::mem::transmute::<FFI_ArrowArray, arrow_lite::ArrowArray>(ffi_array),
        )
    };
    // SAFETY: a live pair the oracle just produced.
    unsafe { import_batch(schema, array) }.expect("we import")
}

/// The canonical batch: both numeric subtypes, a dictionary column, a
/// nullable column with real nulls, and both logical annotations.
fn ours_full() -> RecordBatch {
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

/// The same batch in the oracle's own types.
fn arrow_full() -> StructArray {
    let sym: DictionaryArray<UInt32Type> = DictionaryArray::try_new(
        UInt32Array::from(vec![0, 1, 0, 2]),
        Arc::new(StringArray::from(vec!["AAPL", "MSFT", "TSLA"])),
    )
    .expect("dictionary builds");
    let fields_and_arrays: Vec<(ArrowField, ArrayRef)> = vec![
        (
            ArrowField::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Arc::new(TimestampNanosecondArray::from(vec![
                1_000, 2_000, 3_000, 4_000,
            ])),
        ),
        (
            ArrowField::new("sym", sym.data_type().clone(), false),
            Arc::new(sym),
        ),
        (
            ArrowField::new("px", DataType::Float64, true),
            Arc::new(Float64Array::from(vec![
                Some(101.5),
                None,
                Some(99.25),
                None,
            ])),
        ),
        (
            ArrowField::new("qty", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
        ),
        (
            ArrowField::new("amt", DataType::Decimal64(18, 2), false),
            Arc::new(
                Decimal64Array::from_iter_values([19_999, 250, -75, 0])
                    .with_precision_and_scale(18, 2)
                    .expect("precision fits"),
            ),
        ),
    ];
    StructArray::from(
        fields_and_arrays
            .into_iter()
            .map(|(field, array)| (Arc::new(field), array))
            .collect::<Vec<_>>(),
    )
}

/// Logical comparison of our batch against ours-expected, ignoring values
/// under null slots (the oracle's exports leave them unspecified).
fn assert_logically_equal(actual: &RecordBatch, expected: &RecordBatch) {
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.num_rows(), expected.num_rows());
    for (a, e) in actual.columns().iter().zip(expected.columns()) {
        match (a, e) {
            (Column::Numeric(NumericData::F64(a)), Column::Numeric(NumericData::F64(e))) => {
                assert_numeric_eq(a, e)
            }
            (Column::Numeric(NumericData::I64(a)), Column::Numeric(NumericData::I64(e))) => {
                assert_numeric_eq(a, e)
            }
            (Column::Key(a), Column::Key(e)) => {
                assert_eq!(a.len(), e.len());
                for row in 0..a.len() {
                    assert_eq!(a.value_at(row), e.value_at(row), "key row {row}");
                }
            }
            _ => panic!("column type mismatch"),
        }
    }
}

fn assert_numeric_eq<T: arrow_lite::Element>(a: &NumericColumn<T>, e: &NumericColumn<T>) {
    assert_eq!(a.len(), e.len());
    for row in 0..a.len() {
        assert_eq!(a.is_valid(row), e.is_valid(row), "validity row {row}");
        if a.is_valid(row) {
            assert_eq!(a.values()[row], e.values()[row], "value row {row}");
        }
    }
}

#[test]
fn oracle_imports_our_export() {
    let imported = oracle_import(ours_full());
    assert_eq!(imported, arrow_full());
}

#[test]
fn we_import_oracle_export() {
    let batch = import_from_oracle(&arrow_full());
    assert_logically_equal(&batch, &ours_full());
}

#[test]
fn empty_batch_crosses_both_ways() {
    let schema = Schema::new(vec![
        Field::new("x", ColumnType::F64, false),
        Field::new("k", ColumnType::Key, false),
    ]);
    let ours = RecordBatch::new(
        schema,
        vec![
            Column::Numeric(NumericData::F64(NumericColumn::new_non_null(Buffer::new()))),
            Column::Key(KeyColumn::from_values(std::iter::empty())),
        ],
    );
    let via_oracle = oracle_import(ours.clone());
    assert_eq!(via_oracle.len(), 0);
    let back = import_from_oracle(&via_oracle);
    assert_logically_equal(&back, &ours);
}

#[test]
fn nullable_key_crosses_both_ways() {
    let mut dict = Dictionary::new();
    dict.intern("a");
    dict.intern("b");
    let schema = Schema::new(vec![Field::new("k", ColumnType::Key, true)]);
    let ours = RecordBatch::new(
        schema,
        vec![Column::Key(KeyColumn::new_nullable(
            Buffer::from_slice(&[0, 1, 0]),
            Bitmap::from_bools([true, true, false]),
            dict,
        ))],
    );
    let via_oracle = oracle_import(ours.clone());
    let expected: DictionaryArray<UInt32Type> = DictionaryArray::try_new(
        UInt32Array::from(vec![Some(0), Some(1), None]),
        Arc::new(StringArray::from(vec!["a", "b"])),
    )
    .expect("dictionary builds");
    let expected = StructArray::from(vec![(
        Arc::new(ArrowField::new("k", expected.data_type().clone(), true)),
        Arc::new(expected) as ArrayRef,
    )]);
    assert_eq!(via_oracle, expected);
    let back = import_from_oracle(&via_oracle);
    assert_logically_equal(&back, &ours);
}

#[test]
fn oracle_reads_our_stream() {
    let batches: Vec<RecordBatch> = (0..3).map(slice_batch).collect();
    let stream = export_stream(batches[0].schema().clone(), batches.clone().into_iter());
    // SAFETY: layout-compatible; ownership moves to the oracle's reader.
    let ffi_stream = unsafe {
        std::mem::transmute::<arrow_lite::ArrowArrayStream, FFI_ArrowArrayStream>(stream)
    };
    let reader = ArrowArrayStreamReader::try_new(ffi_stream).expect("oracle opens stream");
    assert_eq!(
        reader.schema().as_ref(),
        &ArrowRsSchema::new(vec![
            ArrowField::new("x", DataType::Float64, false),
            ArrowField::new("n", DataType::Int64, false),
        ])
    );
    let read: Vec<arrow::record_batch::RecordBatch> =
        reader.collect::<Result<_, _>>().expect("oracle reads");
    assert_eq!(read.len(), 3);
    for (i, batch) in read.iter().enumerate() {
        assert_eq!(batch, &arrow_slice_batch(i as i64));
    }
}

#[test]
fn we_read_oracle_stream() {
    let arrow_batches: Vec<arrow::record_batch::RecordBatch> =
        (0..3).map(arrow_slice_batch).collect();
    let schema = arrow_batches[0].schema();
    let reader = RecordBatchIterator::new(arrow_batches.into_iter().map(Ok), schema);
    let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
    // SAFETY: layout-compatible; ownership moves to our reader.
    let stream = unsafe {
        std::mem::transmute::<FFI_ArrowArrayStream, arrow_lite::ArrowArrayStream>(ffi_stream)
    };
    // SAFETY: a live stream the oracle just produced.
    let reader = unsafe { arrow_lite::StreamReader::new(stream) }.expect("we open stream");
    let read: Vec<RecordBatch> = reader.collect::<Result<_, _>>().expect("we read");
    assert_eq!(read.len(), 3);
    for (i, batch) in read.iter().enumerate() {
        assert_logically_equal(batch, &slice_batch(i as i64));
    }
}

/// One small two-column batch per stream slice, in our types.
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

/// The same slice in the oracle's types.
fn arrow_slice_batch(i: i64) -> arrow::record_batch::RecordBatch {
    arrow::record_batch::RecordBatch::try_new(
        Arc::new(ArrowRsSchema::new(vec![
            ArrowField::new("x", DataType::Float64, false),
            ArrowField::new("n", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![i as f64, i as f64 + 0.5])),
            Arc::new(Int64Array::from(vec![i * 10, i * 10 + 1])),
        ],
    )
    .expect("batch builds")
}
