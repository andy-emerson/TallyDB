//! Schemas and record batches: named, typed collections of columns.
//!
//! A [`Field`] declares one column — name, physical type, nullability, and
//! an optional export annotation. A [`Schema`] is the ordered field list; a
//! [`RecordBatch`] binds a schema to actual columns and is the unit that
//! crosses the C Data Interface (results leave the engine as a *stream of
//! record batches*).
//!
//! Construction validates the bindings — type match, equal lengths, and
//! the no-bitmap-when-`NOT NULL` rule — so an exported batch is consistent
//! by construction.

use crate::column::{Column, ColumnType};
use crate::logical::LogicalType;

/// One column's declaration: name, physical type, nullability, optional
/// export annotation.
///
/// ```
/// use arrow_lite::{ColumnType, Field, LogicalType};
///
/// let ts = Field::new("ts", ColumnType::I64, false).with_logical(LogicalType::TimestampNs);
/// assert_eq!(ts.name(), "ts");
/// assert!(!ts.nullable());
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Field {
    name: String,
    column_type: ColumnType,
    nullable: bool,
    logical: Option<LogicalType>,
}

impl Field {
    /// A field with no logical annotation.
    pub fn new(name: impl Into<String>, column_type: ColumnType, nullable: bool) -> Self {
        Field {
            name: name.into(),
            column_type,
            nullable,
            logical: None,
        }
    }

    /// Adds a logical annotation.
    ///
    /// # Panics
    /// If the annotation does not apply to this field's physical type
    /// (annotations sit on `i64` columns only).
    pub fn with_logical(mut self, logical: LogicalType) -> Self {
        assert!(
            logical.valid_for(self.column_type),
            "{logical:?} does not apply to {:?}",
            self.column_type
        );
        self.logical = Some(logical);
        self
    }

    /// The column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The physical type.
    pub fn column_type(&self) -> ColumnType {
        self.column_type
    }

    /// Whether rows may be null. `false` also promises the bound column
    /// carries no validity bitmap.
    pub fn nullable(&self) -> bool {
        self.nullable
    }

    /// The export annotation, if any.
    pub fn logical(&self) -> Option<LogicalType> {
        self.logical
    }
}

/// An ordered list of fields.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    /// A schema over `fields`.
    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }

    /// The fields, in column order.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}

/// A schema bound to columns: the unit of data interchange.
///
/// ```
/// use arrow_lite::{
///     Buffer, Column, ColumnType, Field, KeyColumn, NumericColumn, NumericData, RecordBatch,
///     Schema,
/// };
///
/// let schema = Schema::new(vec![
///     Field::new("sym", ColumnType::Key, false),
///     Field::new("px", ColumnType::F64, false),
/// ]);
/// let batch = RecordBatch::new(
///     schema,
///     vec![
///         Column::Key(KeyColumn::from_values(["A", "B"])),
///         Column::Numeric(NumericData::F64(NumericColumn::new_non_null(
///             Buffer::from_slice(&[1.0, 2.0]),
///         ))),
///     ],
/// );
/// assert_eq!(batch.num_rows(), 2);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct RecordBatch {
    schema: Schema,
    columns: Vec<Column>,
    num_rows: usize,
}

impl RecordBatch {
    /// Binds `columns` to `schema`.
    ///
    /// # Panics
    /// If the column count, any column's physical type, or any column's
    /// length disagrees with the schema — or a `NOT NULL` field's column
    /// carries a validity bitmap (the contract says it must not exist).
    pub fn new(schema: Schema, columns: Vec<Column>) -> Self {
        assert_eq!(
            schema.fields().len(),
            columns.len(),
            "schema has {} fields but {} columns were bound",
            schema.fields().len(),
            columns.len()
        );
        let num_rows = columns.first().map_or(0, Column::len);
        for (field, column) in schema.fields().iter().zip(&columns) {
            assert_eq!(
                field.column_type(),
                column.column_type(),
                "column '{}' bound with wrong physical type",
                field.name()
            );
            assert_eq!(
                column.len(),
                num_rows,
                "column '{}' has {} rows, expected {num_rows}",
                field.name(),
                column.len()
            );
            let has_bitmap = match column {
                Column::Numeric(n) => n.validity().is_some(),
                Column::Key(k) => k.validity().is_some(),
            };
            assert!(
                field.nullable() || !has_bitmap,
                "NOT NULL column '{}' carries a validity bitmap",
                field.name()
            );
        }
        RecordBatch {
            schema,
            columns,
            num_rows,
        }
    }

    /// The schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The columns, in schema order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Number of rows (every column agrees).
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, NumericColumn};
    use crate::column::NumericData;
    use crate::Bitmap;

    fn f64_column(values: &[f64]) -> Column {
        Column::Numeric(NumericData::F64(NumericColumn::new_non_null(
            Buffer::from_slice(values),
        )))
    }

    #[test]
    #[should_panic(expected = "does not apply")]
    fn logical_on_float_rejected() {
        Field::new("x", ColumnType::F64, false).with_logical(LogicalType::TimestampNs);
    }

    #[test]
    #[should_panic(expected = "wrong physical type")]
    fn type_mismatch_rejected() {
        let schema = Schema::new(vec![Field::new("k", ColumnType::Key, false)]);
        RecordBatch::new(schema, vec![f64_column(&[1.0])]);
    }

    #[test]
    #[should_panic(expected = "expected 2")]
    fn ragged_lengths_rejected() {
        let schema = Schema::new(vec![
            Field::new("a", ColumnType::F64, false),
            Field::new("b", ColumnType::F64, false),
        ]);
        RecordBatch::new(schema, vec![f64_column(&[1.0, 2.0]), f64_column(&[1.0])]);
    }

    #[test]
    #[should_panic(expected = "carries a validity bitmap")]
    fn not_null_with_bitmap_rejected() {
        let schema = Schema::new(vec![Field::new("a", ColumnType::F64, false)]);
        let col = Column::Numeric(NumericData::F64(NumericColumn::new_nullable(
            Buffer::from_slice(&[1.0]),
            Bitmap::new_set(1),
        )));
        RecordBatch::new(schema, vec![col]);
    }

    #[test]
    fn empty_batch_is_fine() {
        let batch = RecordBatch::new(Schema::new(vec![]), vec![]);
        assert_eq!(batch.num_rows(), 0);
    }
}
