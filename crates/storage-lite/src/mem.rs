//! The in-memory building blocks: an append buffer that freezes into an
//! immutable segment.
//!
//! These are the pieces [`crate::store::Store`] composes into a table's
//! storage — the buffer validates and accumulates arriving rows, the
//! segment is the immutable unit readers see. Still ahead, in build
//! order: the on-disk format and I/O backend trait (M2.2 — designed
//! together, so the trait doesn't freeze memory-only assumptions), then
//! tombstones and compaction (M2.3).
//!
//! What this layer holds to:
//!
//! - **Append-optimized:** one row at a time, O(1) amortized, into
//!   per-column builders.
//! - **Ordered:** the schema names its ordering key (`i64`, `NOT NULL`);
//!   ingest is *expected* roughly sorted on it, and the frozen segment
//!   reports [`Segment::is_ordered`] and [`Segment::ordering_bounds`] so
//!   readers that require strict order (the window executor) can check
//!   instead of silently mis-computing.
//! - **Numeric-or-key:** rows are checked cell-by-cell against the schema
//!   on append — wrong type, null into `NOT NULL`, wrong arity are all
//!   errors at the door, not corruption later.

use arrow_lite::{
    Bitmap, Buffer, Column, ColumnType, Dictionary, KeyColumn, NumericColumn, NumericData,
    RecordBatch, Schema,
};
use std::fmt;

/// One cell of an arriving row.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RowValue<'a> {
    /// A value for an `f64` column.
    F64(f64),
    /// A value for an `i64` column.
    I64(i64),
    /// A value for a key column (interned on ingest).
    Key(&'a str),
    /// No value — legal only where the schema says nullable.
    Null,
}

/// Why an append or freeze was refused.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StorageError {
    /// The row has the wrong number of cells.
    WrongArity { expected: usize, got: usize },
    /// A cell's type disagrees with its column.
    TypeMismatch {
        column: String,
        expected: ColumnType,
    },
    /// A null arrived for a `NOT NULL` column.
    NullNotAllowed { column: String },
    /// The declared ordering key must be an `i64 NOT NULL` column.
    BadOrderingKey { reason: String },
    /// A nullable key column ended up all-null with an empty dictionary —
    /// not representable as a `KeyColumn` (known limitation, kept).
    AllNullKeyColumn { column: String },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::WrongArity { expected, got } => {
                write!(f, "row has {got} cells, schema has {expected} columns")
            }
            StorageError::TypeMismatch { column, expected } => {
                write!(f, "column '{column}' expects {expected:?}")
            }
            StorageError::NullNotAllowed { column } => {
                write!(f, "column '{column}' is NOT NULL")
            }
            StorageError::BadOrderingKey { reason } => write!(f, "bad ordering key: {reason}"),
            StorageError::AllNullKeyColumn { column } => write!(
                f,
                "key column '{column}' is entirely null; an all-null key column is unsupported"
            ),
        }
    }
}

impl std::error::Error for StorageError {}

/// Per-column accumulation state while rows arrive.
///
/// Cloning is cheap by design: numeric and code buffers are copy-on-write
/// handles (O(1)), and the null flags and dictionary are bounded by the
/// write buffer's row count and distinct-key count. [`WriteBuffer::snapshot`]
/// leans on this to freeze a point-in-time segment without consuming the
/// buffer.
#[derive(Clone)]
enum ColumnBuilder {
    F64 {
        values: Buffer<f64>,
        nulls: Vec<bool>,
    },
    I64 {
        values: Buffer<i64>,
        nulls: Vec<bool>,
    },
    Key {
        codes: Buffer<u32>,
        nulls: Vec<bool>,
        dictionary: Dictionary,
    },
}

/// The append path: one row at a time against a declared schema, frozen
/// into a [`Segment`] when done.
///
/// ```
/// use arrow_lite::{ColumnType, Field, Schema};
/// use storage_lite::{RowValue, WriteBuffer};
///
/// let schema = Schema::new(vec![
///     Field::new("ts", ColumnType::I64, false),
///     Field::new("sym", ColumnType::Key, false),
///     Field::new("x", ColumnType::F64, false),
/// ]);
/// let mut buffer = WriteBuffer::new(schema, 0).unwrap();
/// buffer
///     .append(&[RowValue::I64(1), RowValue::Key("A"), RowValue::F64(0.5)])
///     .unwrap();
/// let segment = buffer.freeze().unwrap();
/// assert_eq!(segment.batch().num_rows(), 1);
/// assert!(segment.is_ordered());
/// ```
#[derive(Clone)]
pub struct WriteBuffer {
    schema: Schema,
    ordering_key: usize,
    builders: Vec<ColumnBuilder>,
    rows: usize,
    /// Whether the ordering key has been non-decreasing so far.
    ordered: bool,
    last_ordering_value: Option<i64>,
}

impl WriteBuffer {
    /// A buffer for `schema`, whose column `ordering_key` is the declared
    /// ordering key — required to be `i64` and `NOT NULL`.
    pub fn new(schema: Schema, ordering_key: usize) -> Result<WriteBuffer, StorageError> {
        let field =
            schema
                .fields()
                .get(ordering_key)
                .ok_or_else(|| StorageError::BadOrderingKey {
                    reason: format!("index {ordering_key} out of range"),
                })?;
        if field.column_type() != ColumnType::I64 {
            return Err(StorageError::BadOrderingKey {
                reason: format!("'{}' is not i64", field.name()),
            });
        }
        if field.nullable() {
            return Err(StorageError::BadOrderingKey {
                reason: format!("'{}' must be NOT NULL", field.name()),
            });
        }
        let builders = schema
            .fields()
            .iter()
            .map(|f| match f.column_type() {
                ColumnType::F64 => ColumnBuilder::F64 {
                    values: Buffer::new(),
                    nulls: Vec::new(),
                },
                ColumnType::I64 => ColumnBuilder::I64 {
                    values: Buffer::new(),
                    nulls: Vec::new(),
                },
                ColumnType::Key => ColumnBuilder::Key {
                    codes: Buffer::new(),
                    nulls: Vec::new(),
                    dictionary: Dictionary::new(),
                },
            })
            .collect();
        Ok(WriteBuffer {
            schema,
            ordering_key,
            builders,
            rows: 0,
            ordered: true,
            last_ordering_value: None,
        })
    }

    /// Appends one row, checking every cell against the schema.
    pub fn append(&mut self, row: &[RowValue<'_>]) -> Result<(), StorageError> {
        let fields = self.schema.fields();
        if row.len() != fields.len() {
            return Err(StorageError::WrongArity {
                expected: fields.len(),
                got: row.len(),
            });
        }
        // Validate the whole row before touching any builder, so a
        // rejected row leaves the buffer exactly as it was.
        for (field, cell) in fields.iter().zip(row) {
            let ok = match (field.column_type(), cell) {
                (_, RowValue::Null) => {
                    if !field.nullable() {
                        return Err(StorageError::NullNotAllowed {
                            column: field.name().to_owned(),
                        });
                    }
                    true
                }
                (ColumnType::F64, RowValue::F64(_)) => true,
                (ColumnType::I64, RowValue::I64(_)) => true,
                (ColumnType::Key, RowValue::Key(_)) => true,
                _ => false,
            };
            if !ok {
                return Err(StorageError::TypeMismatch {
                    column: field.name().to_owned(),
                    expected: field.column_type(),
                });
            }
        }
        for (builder, cell) in self.builders.iter_mut().zip(row) {
            builder.push(cell);
        }
        if let RowValue::I64(value) = row[self.ordering_key] {
            if let Some(last) = self.last_ordering_value {
                if value < last {
                    self.ordered = false;
                }
            }
            self.last_ordering_value = Some(value);
        }
        self.rows += 1;
        Ok(())
    }

    /// Rows appended so far.
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether no rows have been appended.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Freezes into an immutable segment whose rows begin at row id 0.
    pub fn freeze(self) -> Result<Segment, StorageError> {
        self.freeze_at(0)
    }

    /// Freezes into an immutable segment whose first row carries the
    /// internal row id `base_row_id` (see [`Segment::base_row_id`]).
    pub fn freeze_at(self, base_row_id: u64) -> Result<Segment, StorageError> {
        let mut columns = Vec::with_capacity(self.builders.len());
        for (field, builder) in self.schema.fields().iter().zip(self.builders) {
            columns.push(builder.finish(field.name())?);
        }
        Ok(Segment {
            batch: RecordBatch::new(self.schema, columns),
            ordering_key: self.ordering_key,
            ordered: self.ordered,
            base_row_id,
        })
    }

    /// Freezes a point-in-time copy without consuming the buffer: the
    /// segment holds exactly the rows appended so far, and later appends
    /// leave it untouched (the copy-on-write buffers make this cheap —
    /// no row data is copied at snapshot time).
    pub fn snapshot_at(&self, base_row_id: u64) -> Result<Segment, StorageError> {
        self.clone().freeze_at(base_row_id)
    }
}

impl ColumnBuilder {
    /// Appends one pre-validated cell. Null slots store a placeholder
    /// value (0 / code 0) under a false validity bit.
    fn push(&mut self, cell: &RowValue<'_>) {
        match (self, cell) {
            (ColumnBuilder::F64 { values, nulls }, RowValue::F64(v)) => {
                values.push(*v);
                nulls.push(false);
            }
            (ColumnBuilder::F64 { values, nulls }, RowValue::Null) => {
                values.push(0.0);
                nulls.push(true);
            }
            (ColumnBuilder::I64 { values, nulls }, RowValue::I64(v)) => {
                values.push(*v);
                nulls.push(false);
            }
            (ColumnBuilder::I64 { values, nulls }, RowValue::Null) => {
                values.push(0);
                nulls.push(true);
            }
            (
                ColumnBuilder::Key {
                    codes,
                    nulls,
                    dictionary,
                },
                RowValue::Key(value),
            ) => {
                codes.push(dictionary.intern(value));
                nulls.push(false);
            }
            (ColumnBuilder::Key { codes, nulls, .. }, RowValue::Null) => {
                codes.push(0);
                nulls.push(true);
            }
            _ => unreachable!("append validated the cell against the schema"),
        }
    }

    /// Builds the frozen column. The bitmap exists only if a null actually
    /// arrived — a nullable column that saw no nulls freezes bitmap-free,
    /// same as `NOT NULL`.
    fn finish(self, name: &str) -> Result<Column, StorageError> {
        fn validity(nulls: &[bool]) -> Option<Bitmap> {
            if nulls.iter().any(|&n| n) {
                Some(Bitmap::from_bools(nulls.iter().map(|&n| !n)))
            } else {
                None
            }
        }
        Ok(match self {
            ColumnBuilder::F64 { values, nulls } => {
                Column::Numeric(NumericData::F64(match validity(&nulls) {
                    Some(bitmap) => NumericColumn::new_nullable(values, bitmap),
                    None => NumericColumn::new_non_null(values),
                }))
            }
            ColumnBuilder::I64 { values, nulls } => {
                Column::Numeric(NumericData::I64(match validity(&nulls) {
                    Some(bitmap) => NumericColumn::new_nullable(values, bitmap),
                    None => NumericColumn::new_non_null(values),
                }))
            }
            ColumnBuilder::Key {
                codes,
                nulls,
                dictionary,
            } => {
                if dictionary.is_empty() && !codes.is_empty() {
                    return Err(StorageError::AllNullKeyColumn {
                        column: name.to_owned(),
                    });
                }
                Column::Key(match validity(&nulls) {
                    Some(bitmap) => KeyColumn::new_nullable(codes, bitmap, dictionary),
                    None => KeyColumn::new_non_null(codes, dictionary),
                })
            }
        })
    }
}

/// An immutable, in-memory segment: a record batch plus what storage knows
/// about it.
pub struct Segment {
    batch: RecordBatch,
    ordering_key: usize,
    ordered: bool,
    base_row_id: u64,
}

impl Segment {
    /// Reassembles a segment from decoded parts (the format module's
    /// doorway; nothing else constructs segments directly).
    pub(crate) fn from_parts(
        batch: RecordBatch,
        ordering_key: usize,
        ordered: bool,
        base_row_id: u64,
    ) -> Segment {
        Segment {
            batch,
            ordering_key,
            ordered,
            base_row_id,
        }
    }

    /// The segment's data.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Index of the declared ordering key column.
    pub fn ordering_key(&self) -> usize {
        self.ordering_key
    }

    /// Whether the ordering key arrived non-decreasing. Readers that
    /// require strict order (M1's window executor) must check this rather
    /// than assume.
    pub fn is_ordered(&self) -> bool {
        self.ordered
    }

    /// The internal row id of this segment's first row (decision #1: every
    /// row carries a monotonic id assigned at ingest — row `i` of this
    /// segment has id `base_row_id + i`). Tombstones and "newest version
    /// wins" resolution address rows by these ids, never by key tuples.
    pub fn base_row_id(&self) -> u64 {
        self.base_row_id
    }

    /// First and last values of the ordering key, or `None` if the
    /// segment is empty. Readers use this to check that a *sequence* of
    /// segments is globally ordered: each segment internally ordered, and
    /// each boundary non-decreasing.
    pub fn ordering_bounds(&self) -> Option<(i64, i64)> {
        let Column::Numeric(NumericData::I64(column)) = &self.batch.columns()[self.ordering_key]
        else {
            unreachable!("the ordering key is validated as i64 at construction")
        };
        let values = column.values().as_slice();
        Some((*values.first()?, *values.last()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_lite::Field;

    fn m1_schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, true),
        ])
    }

    fn row<'a>(ts: i64, sym: &'a str, x: f64, y: Option<f64>) -> Vec<RowValue<'a>> {
        vec![
            RowValue::I64(ts),
            RowValue::Key(sym),
            RowValue::F64(x),
            y.map_or(RowValue::Null, RowValue::F64),
        ]
    }

    #[test]
    fn append_freeze_round_trip() {
        let mut buffer = WriteBuffer::new(m1_schema(), 0).unwrap();
        buffer.append(&row(1, "A", 0.5, Some(1.0))).unwrap();
        buffer.append(&row(2, "B", 1.5, None)).unwrap();
        buffer.append(&row(3, "A", 2.5, Some(3.0))).unwrap();
        let segment = buffer.freeze().unwrap();
        let batch = segment.batch();
        assert_eq!(batch.num_rows(), 3);
        assert!(segment.is_ordered());
        let Column::Numeric(NumericData::I64(ts)) = &batch.columns()[0] else {
            panic!("ts type")
        };
        assert_eq!(ts.values().as_slice(), &[1, 2, 3]);
        let Column::Key(sym) = &batch.columns()[1] else {
            panic!("sym type")
        };
        assert_eq!(sym.value_at(0), Some("A"));
        assert_eq!(sym.value_at(1), Some("B"));
        assert_eq!(sym.codes().as_slice(), &[0, 1, 0]);
        let Column::Numeric(NumericData::F64(y)) = &batch.columns()[3] else {
            panic!("y type")
        };
        assert_eq!(y.null_count(), 1);
        assert!(!y.is_valid(1));
    }

    #[test]
    fn out_of_order_ingest_is_recorded_not_rejected() {
        let mut buffer = WriteBuffer::new(m1_schema(), 0).unwrap();
        buffer.append(&row(5, "A", 0.0, None)).unwrap();
        buffer.append(&row(3, "A", 0.0, None)).unwrap(); // late arrival
        let segment = buffer.freeze().unwrap();
        assert!(!segment.is_ordered());
    }

    #[test]
    fn schema_violations_are_rejected_at_the_door() {
        let mut buffer = WriteBuffer::new(m1_schema(), 0).unwrap();
        // Wrong arity.
        assert!(matches!(
            buffer.append(&[RowValue::I64(1)]),
            Err(StorageError::WrongArity {
                expected: 4,
                got: 1
            })
        ));
        // Wrong type (f64 into the i64 ordering key).
        assert!(matches!(
            buffer.append(&[
                RowValue::F64(1.0),
                RowValue::Key("A"),
                RowValue::F64(0.0),
                RowValue::Null
            ]),
            Err(StorageError::TypeMismatch { .. })
        ));
        // Null into NOT NULL.
        assert!(matches!(
            buffer.append(&[
                RowValue::Null,
                RowValue::Key("A"),
                RowValue::F64(0.0),
                RowValue::Null
            ]),
            Err(StorageError::NullNotAllowed { .. })
        ));
        // A rejected row leaves the buffer untouched.
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.freeze().unwrap().batch().num_rows(), 0);
    }

    #[test]
    fn ordering_key_must_be_i64_not_null() {
        let schema = m1_schema();
        assert!(matches!(
            WriteBuffer::new(schema.clone(), 1), // sym: key, not i64
            Err(StorageError::BadOrderingKey { .. })
        ));
        assert!(matches!(
            WriteBuffer::new(schema.clone(), 3), // y: nullable
            Err(StorageError::BadOrderingKey { .. })
        ));
        assert!(matches!(
            WriteBuffer::new(schema, 9), // out of range
            Err(StorageError::BadOrderingKey { .. })
        ));
    }

    #[test]
    fn nullable_column_without_nulls_freezes_bitmap_free() {
        let mut buffer = WriteBuffer::new(m1_schema(), 0).unwrap();
        buffer.append(&row(1, "A", 0.0, Some(1.0))).unwrap();
        let segment = buffer.freeze().unwrap();
        let Column::Numeric(NumericData::F64(y)) = &segment.batch().columns()[3] else {
            panic!("y type")
        };
        assert!(y.validity().is_none());
    }

    #[test]
    fn all_null_key_column_is_an_error() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("k", ColumnType::Key, true),
        ]);
        let mut buffer = WriteBuffer::new(schema, 0).unwrap();
        buffer.append(&[RowValue::I64(1), RowValue::Null]).unwrap();
        assert!(matches!(
            buffer.freeze(),
            Err(StorageError::AllNullKeyColumn { .. })
        ));
    }

    #[test]
    fn empty_freeze_is_fine() {
        let segment = WriteBuffer::new(m1_schema(), 0).unwrap().freeze().unwrap();
        assert_eq!(segment.batch().num_rows(), 0);
        assert!(segment.is_ordered());
    }
}
