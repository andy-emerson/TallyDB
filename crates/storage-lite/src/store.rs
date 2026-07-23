//! The multi-segment container: one [`Store`] per table.
//!
//! A store is an active [`WriteBuffer`] plus the immutable segments it
//! has already frozen. Appends always go to the buffer; when the buffer
//! reaches the store's segment-row threshold it is flushed automatically,
//! so a long-lived store is a growing sequence of bounded segments.
//! Readers never see the buffer directly — [`Store::snapshot`] freezes a
//! point-in-time copy of it (cheap, copy-on-write) and returns the full
//! segment sequence, so appends and queries interleave freely without
//! either blocking the other.
//!
//! ## Row identity starts here (decision #1)
//!
//! The store assigns every appended row an internal monotonic row id and
//! stamps each segment with the id of its first row. Duplicates are
//! first-class — nothing here inspects key values or collapses rows.
//! These ids are what tombstones and "newest version wins" resolution
//! will address when mutation arrives; today they only need to exist and
//! be right.
//!
//! ## What a snapshot promises
//!
//! The segments come back in append order, each internally consistent,
//! covering exactly the rows appended before the call. Global ordering is
//! *not* promised — ingest is only expected roughly sorted — so each
//! segment reports [`Segment::is_ordered`] and [`Segment::ordering_bounds`]
//! and readers that require order (the window executor) check instead of
//! assuming, exactly as they did for a single segment.

use crate::mem::{RowValue, Segment, StorageError, WriteBuffer};
use arrow_lite::Schema;
use std::sync::Arc;

/// Rows per segment before an automatic flush. Large enough that segment
/// bookkeeping is noise, small enough that a segment is a reasonable unit
/// of compaction and (at M2.2) I/O.
pub const DEFAULT_SEGMENT_ROWS: usize = 65_536;

/// A table's storage: an active write buffer plus frozen segments.
///
/// ```
/// use arrow_lite::{ColumnType, Field, Schema};
/// use storage_lite::{RowValue, Store};
///
/// let schema = Schema::new(vec![
///     Field::new("ts", ColumnType::I64, false),
///     Field::new("x", ColumnType::F64, false),
/// ]);
/// // A tiny threshold so the example spans segments.
/// let mut store = Store::with_segment_rows(schema, 0, 2).unwrap();
/// for i in 0..5 {
///     let id = store.append(&[RowValue::I64(i), RowValue::F64(i as f64)]).unwrap();
///     assert_eq!(id, i as u64); // row ids are assigned in ingest order
/// }
/// let segments = store.snapshot().unwrap();
/// // Two full segments plus the live buffer's single row.
/// let rows: Vec<usize> = segments.iter().map(|s| s.batch().num_rows()).collect();
/// assert_eq!(rows, [2, 2, 1]);
/// assert_eq!(segments[2].base_row_id(), 4);
/// ```
pub struct Store {
    schema: Schema,
    ordering_key: usize,
    segment_rows: usize,
    buffer: WriteBuffer,
    /// Row id of the buffer's first row.
    buffer_base: u64,
    segments: Vec<Arc<Segment>>,
    rows: u64,
}

impl Store {
    /// A store for `schema` ordered on column `ordering_key`, flushing
    /// every [`DEFAULT_SEGMENT_ROWS`] rows.
    pub fn new(schema: Schema, ordering_key: usize) -> Result<Store, StorageError> {
        Store::with_segment_rows(schema, ordering_key, DEFAULT_SEGMENT_ROWS)
    }

    /// As [`Store::new`], with an explicit segment-row threshold
    /// (`>= 1`; tests use small thresholds to exercise many segments).
    pub fn with_segment_rows(
        schema: Schema,
        ordering_key: usize,
        segment_rows: usize,
    ) -> Result<Store, StorageError> {
        assert!(segment_rows >= 1, "segment_rows must be at least 1");
        let buffer = WriteBuffer::new(schema.clone(), ordering_key)?;
        Ok(Store {
            schema,
            ordering_key,
            segment_rows,
            buffer,
            buffer_base: 0,
            segments: Vec::new(),
            rows: 0,
        })
    }

    /// The store's schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Index of the declared ordering key column.
    pub fn ordering_key(&self) -> usize {
        self.ordering_key
    }

    /// Total rows appended over the store's lifetime — also the id the
    /// next appended row will receive.
    pub fn len(&self) -> u64 {
        self.rows
    }

    /// Whether no rows have ever been appended.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Frozen segments so far (not counting the live buffer).
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Appends one row and returns its internal row id. Flushes
    /// automatically when the buffer reaches the segment-row threshold.
    pub fn append(&mut self, row: &[RowValue<'_>]) -> Result<u64, StorageError> {
        self.buffer.append(row)?;
        let id = self.rows;
        self.rows += 1;
        if self.buffer.len() >= self.segment_rows {
            self.flush()?;
        }
        Ok(id)
    }

    /// Freezes the live buffer into a segment now (a no-op when empty).
    ///
    /// Flushing goes through a snapshot first, so a buffer that cannot
    /// freeze (an all-null key column) returns the error with the buffer
    /// — and its rows — intact.
    pub fn flush(&mut self) -> Result<(), StorageError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let segment = self.buffer.snapshot_at(self.buffer_base)?;
        self.segments.push(Arc::new(segment));
        self.buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        self.buffer_base = self.rows;
        Ok(())
    }

    /// A point-in-time view: every frozen segment, plus (if the buffer
    /// holds rows) a segment frozen from a copy of it. Appends after the
    /// call don't affect the returned segments.
    pub fn snapshot(&self) -> Result<Vec<Arc<Segment>>, StorageError> {
        let mut segments = self.segments.clone();
        if !self.buffer.is_empty() {
            segments.push(Arc::new(self.buffer.snapshot_at(self.buffer_base)?));
        }
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_lite::{Column, ColumnType, Field, NumericData};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ])
    }

    fn append_n(store: &mut Store, range: std::ops::Range<i64>) {
        for i in range {
            store
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                    RowValue::F64(i as f64),
                ])
                .unwrap();
        }
    }

    #[test]
    fn auto_flush_partitions_ingest_into_bounded_segments() {
        let mut store = Store::with_segment_rows(schema(), 0, 4).unwrap();
        append_n(&mut store, 0..10);
        assert_eq!(store.segment_count(), 2); // two full, two rows live
        let segments = store.snapshot().unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(
            segments
                .iter()
                .map(|s| s.batch().num_rows())
                .collect::<Vec<_>>(),
            [4, 4, 2]
        );
    }

    #[test]
    fn row_ids_are_monotonic_across_segments() {
        let mut store = Store::with_segment_rows(schema(), 0, 3).unwrap();
        for i in 0..8i64 {
            let id = store
                .append(&[RowValue::I64(i), RowValue::Key("A"), RowValue::F64(0.0)])
                .unwrap();
            assert_eq!(id, i as u64);
        }
        let segments = store.snapshot().unwrap();
        assert_eq!(
            segments.iter().map(|s| s.base_row_id()).collect::<Vec<_>>(),
            [0, 3, 6]
        );
        assert_eq!(store.len(), 8);
    }

    #[test]
    fn snapshot_is_isolated_from_later_appends() {
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..5);
        let before = store.snapshot().unwrap();
        append_n(&mut store, 5..9);
        // The old snapshot still sees exactly its five rows...
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].batch().num_rows(), 5);
        let Column::Numeric(NumericData::I64(ts)) = &before[0].batch().columns()[0] else {
            panic!("ts type")
        };
        assert_eq!(ts.values().as_slice(), &[0, 1, 2, 3, 4]);
        // ...and a fresh one sees all nine.
        let after = store.snapshot().unwrap();
        assert_eq!(after[0].batch().num_rows(), 9);
    }

    #[test]
    fn snapshot_of_live_buffer_shares_row_data() {
        // The buffer snapshot is copy-on-write: until the next append,
        // the segment and the buffer share the same numeric allocation.
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..4);
        let first = store.snapshot().unwrap();
        let second = store.snapshot().unwrap();
        let ptr = |segment: &Segment| {
            let Column::Numeric(NumericData::F64(x)) = &segment.batch().columns()[2] else {
                panic!("x type")
            };
            x.values().as_ptr()
        };
        assert_eq!(ptr(&first[0]), ptr(&second[0]));
    }

    #[test]
    fn explicit_flush_then_snapshot_has_no_live_tail() {
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..5);
        store.flush().unwrap();
        assert_eq!(store.segment_count(), 1);
        assert_eq!(store.snapshot().unwrap().len(), 1);
        // Flushing an empty buffer is a no-op, not an empty segment.
        store.flush().unwrap();
        assert_eq!(store.segment_count(), 1);
    }

    #[test]
    fn ordering_bounds_expose_cross_segment_order() {
        let mut store = Store::with_segment_rows(schema(), 0, 3).unwrap();
        append_n(&mut store, 0..9);
        let segments = store.snapshot().unwrap();
        let bounds: Vec<_> = segments
            .iter()
            .map(|s| s.ordering_bounds().unwrap())
            .collect();
        assert_eq!(bounds, [(0, 2), (3, 5), (6, 8)]);
        assert!(segments.iter().all(|s| s.is_ordered()));
    }

    #[test]
    fn failed_flush_keeps_the_rows() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("k", ColumnType::Key, true),
        ]);
        let mut store = Store::with_segment_rows(schema, 0, 100).unwrap();
        store.append(&[RowValue::I64(1), RowValue::Null]).unwrap();
        // All-null key column: unsupported, so the flush fails...
        assert!(matches!(
            store.flush(),
            Err(StorageError::AllNullKeyColumn { .. })
        ));
        // ...but the rows are still there, and interning a real key later
        // makes the same buffer freezable.
        store
            .append(&[RowValue::I64(2), RowValue::Key("A")])
            .unwrap();
        store.flush().unwrap();
        assert_eq!(store.snapshot().unwrap()[0].batch().num_rows(), 2);
    }

    #[test]
    fn rejected_rows_get_no_row_id() {
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..2);
        assert!(store.append(&[RowValue::I64(9)]).is_err()); // wrong arity
        assert_eq!(
            store.append(&[RowValue::I64(9), RowValue::Key("A"), RowValue::F64(0.0)]),
            Ok(2)
        );
    }
}
