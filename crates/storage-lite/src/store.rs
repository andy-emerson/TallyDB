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

use crate::format::{decode_segment, encode_segment};
use crate::io::{IoError, StorageBackend};
use crate::mem::{RowValue, Segment, StorageError, WriteBuffer};
use crate::tombstone::{decode_tombstones, encode_tombstones};
use arrow_lite::{Bitmap, Column, NumericData, Schema};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Rows per segment before an automatic flush. Large enough that segment
/// bookkeeping is noise, small enough that a segment is a reasonable unit
/// of compaction and I/O.
pub const DEFAULT_SEGMENT_ROWS: usize = 65_536;

/// The backend object holding the table manifest — an encoded empty
/// segment, which carries exactly what a manifest needs (schema and
/// ordering key) with the segment format's own magic, CRC, and
/// versioning. The manifest's otherwise-unused `base_row_id` field
/// stores the table's current **generation** (see below).
const MANIFEST: &str = "table.tlym";

/// Segment and delete-log names carry a generation number, and the
/// manifest names the current one. This is what makes compaction
/// crash-safe: a compaction writes the whole next generation first,
/// then commits it with one atomic manifest write, then cleans up the
/// old objects — a crash at any point leaves a backend whose manifest
/// still names exactly one complete, self-consistent generation, and
/// reopen ignores every object from any other.
fn segment_name(generation: u64, base_row_id: u64) -> String {
    format!("seg-g{generation:010}-{base_row_id:020}.tlyseg")
}

fn delete_log_name(generation: u64, sequence: u64) -> String {
    format!("del-g{generation:010}-{sequence:020}.tlyd")
}

/// The `name`s of a generation's objects start with these.
fn segment_prefix(generation: u64) -> String {
    format!("seg-g{generation:010}-")
}

/// The cell at (`column`, `row`) as the row value that would recreate
/// it — how compaction replays live rows through the ordinary append
/// path.
fn cell_value(column: &Column, row: usize) -> RowValue<'_> {
    match column {
        Column::Numeric(NumericData::F64(numeric)) => {
            if numeric.is_valid(row) {
                RowValue::F64(numeric.values().as_slice()[row])
            } else {
                RowValue::Null
            }
        }
        Column::Numeric(NumericData::I64(numeric)) => {
            if numeric.is_valid(row) {
                RowValue::I64(numeric.values().as_slice()[row])
            } else {
                RowValue::Null
            }
        }
        Column::Key(keys) => keys.value_at(row).map_or(RowValue::Null, RowValue::Key),
    }
}

fn delete_log_prefix(generation: u64) -> String {
    format!("del-g{generation:010}-")
}

/// One segment as a reader sees it: the immutable segment plus the live
/// mask tombstones impose on it. `live: None` means every row is live —
/// the common case, and the one downstream keeps zero-copy.
#[derive(Clone)]
pub struct SegmentView {
    /// The stored segment.
    pub segment: Arc<Segment>,
    /// Bit per row, `true` = live; `None` when nothing is tombstoned.
    pub live: Option<Bitmap>,
}

impl SegmentView {
    /// A view with every row live.
    pub fn all_live(segment: Arc<Segment>) -> SegmentView {
        SegmentView {
            segment,
            live: None,
        }
    }

    /// Rows a reader will actually see.
    pub fn live_rows(&self) -> usize {
        match &self.live {
            None => self.segment.batch().num_rows(),
            Some(mask) => mask.count_set(),
        }
    }

    /// Whether local row `row` is live.
    pub fn is_live(&self, row: usize) -> bool {
        self.live.as_ref().is_none_or(|mask| mask.get(row))
    }
}

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
/// let rows: Vec<usize> = segments.iter().map(|s| s.segment.batch().num_rows()).collect();
/// assert_eq!(rows, [2, 2, 1]);
/// assert_eq!(segments[2].segment.base_row_id(), 4);
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
    /// Row ids the table has tombstoned (decision #1: ids, never keys).
    tombstones: BTreeSet<u64>,
    /// Sequence number for the next delete log.
    delete_log_sequence: u64,
    /// The current storage generation (bumped by each compaction).
    generation: u64,
    /// Where flushed segments also go, if the store is persistent.
    backend: Option<Arc<dyn StorageBackend>>,
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
            tombstones: BTreeSet::new(),
            delete_log_sequence: 0,
            generation: 0,
            backend: None,
        })
    }

    /// A persistent store over `backend`, flushing every
    /// [`DEFAULT_SEGMENT_ROWS`] rows. Creates the table if the backend is
    /// empty; otherwise reopens it, verifying the manifest against
    /// `schema`/`ordering_key` and every stored segment's checksum,
    /// schema, and row-id contiguity.
    ///
    /// **Durability boundary: [`Store::flush`].** Rows in the write
    /// buffer exist only in memory until flushed (automatically at the
    /// segment-row threshold, or explicitly); a crash loses them and
    /// reopen sees exactly the flushed segments.
    pub fn persistent(
        backend: Arc<dyn StorageBackend>,
        schema: Schema,
        ordering_key: usize,
    ) -> Result<Store, StorageError> {
        Store::persistent_with_segment_rows(backend, schema, ordering_key, DEFAULT_SEGMENT_ROWS)
    }

    /// As [`Store::persistent`], with an explicit segment-row threshold.
    pub fn persistent_with_segment_rows(
        backend: Arc<dyn StorageBackend>,
        schema: Schema,
        ordering_key: usize,
        segment_rows: usize,
    ) -> Result<Store, StorageError> {
        let mut store = Store::with_segment_rows(schema, ordering_key, segment_rows)?;
        let generation = match backend.read(MANIFEST) {
            Ok(bytes) => {
                let manifest = decode_segment(&bytes)?;
                if manifest.batch().schema() != &store.schema {
                    return Err(StorageError::SchemaMismatch {
                        reason: "manifest schema differs from the schema given".to_owned(),
                    });
                }
                if manifest.ordering_key() != ordering_key {
                    return Err(StorageError::SchemaMismatch {
                        reason: format!(
                            "manifest orders on column {}, caller asked for {ordering_key}",
                            manifest.ordering_key()
                        ),
                    });
                }
                manifest.base_row_id() // the current generation
            }
            Err(IoError::NotFound(_)) => {
                let empty = WriteBuffer::new(store.schema.clone(), ordering_key)?.freeze()?;
                backend.write(MANIFEST, &encode_segment(&empty))?;
                0
            }
            Err(error) => return Err(error.into()),
        };
        let mut segments = Vec::new();
        let mut tombstones = BTreeSet::new();
        let mut next_sequence = 0u64;
        for name in backend.list()? {
            // Objects from other generations are a crashed compaction's
            // leftovers — invisible here, removed by the next compaction.
            if let Some(sequence) = name
                .strip_prefix(&delete_log_prefix(generation))
                .and_then(|rest| rest.strip_suffix(".tlyd"))
            {
                let sequence: u64 = sequence.parse().map_err(|_| StorageError::SchemaMismatch {
                    reason: format!("delete log '{name}' has a malformed name"),
                })?;
                tombstones.extend(decode_tombstones(&backend.read(&name)?)?);
                next_sequence = next_sequence.max(sequence + 1);
                continue;
            }
            if !(name.starts_with(&segment_prefix(generation)) && name.ends_with(".tlyseg")) {
                continue;
            }
            let segment = decode_segment(&backend.read(&name)?)?;
            if segment.batch().schema() != &store.schema {
                return Err(StorageError::SchemaMismatch {
                    reason: format!("segment '{name}' was written under a different schema"),
                });
            }
            if segment.ordering_key() != ordering_key {
                return Err(StorageError::SchemaMismatch {
                    reason: format!("segment '{name}' orders on a different column"),
                });
            }
            segments.push(Arc::new(segment));
        }
        segments.sort_by_key(|segment| segment.base_row_id());
        let mut expected_base = 0u64;
        for segment in &segments {
            if segment.base_row_id() != expected_base {
                return Err(StorageError::MissingRows { expected_base });
            }
            expected_base += segment.batch().num_rows() as u64;
        }
        store.segments = segments;
        store.rows = expected_base;
        store.buffer_base = expected_base;
        store.tombstones = tombstones;
        store.delete_log_sequence = next_sequence;
        store.generation = generation;
        store.backend = Some(backend);
        Ok(store)
    }

    /// The store's schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Index of the declared ordering key column.
    pub fn ordering_key(&self) -> usize {
        self.ordering_key
    }

    /// Total rows appended over the store's lifetime, tombstoned or not
    /// — also the id the next appended row will receive.
    pub fn len(&self) -> u64 {
        self.rows
    }

    /// Rows a reader sees: appended minus tombstoned.
    pub fn live_len(&self) -> u64 {
        self.rows - self.tombstones.len() as u64
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
    /// On a persistent store this is the durability boundary: the
    /// segment's bytes are published to the backend before the segment
    /// is registered, so a failure at any point leaves both the backend
    /// and the buffer — rows included — exactly as they were.
    pub fn flush(&mut self) -> Result<(), StorageError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let segment = self.buffer.snapshot_at(self.buffer_base)?;
        if let Some(backend) = &self.backend {
            backend.write(
                &segment_name(self.generation, self.buffer_base),
                &encode_segment(&segment),
            )?;
        }
        self.segments.push(Arc::new(segment));
        self.buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        self.buffer_base = self.rows;
        Ok(())
    }

    /// Tombstones rows by id: they disappear from every later snapshot,
    /// and — on a persistent store — from every reopen, via one
    /// append-only delete log per call. Already-dead ids are ignored
    /// (idempotent); ids never assigned are an error. Returns how many
    /// rows died. The physical rows remain until [`Store::compact`]
    /// resolves them.
    pub fn tombstone(&mut self, ids: &[u64]) -> Result<u64, StorageError> {
        if let Some(&bad) = ids.iter().find(|&&id| id >= self.rows) {
            return Err(StorageError::TombstoneOutOfRange { id: bad });
        }
        let newly: BTreeSet<u64> = ids
            .iter()
            .copied()
            .filter(|id| !self.tombstones.contains(id))
            .collect();
        if newly.is_empty() {
            return Ok(0);
        }
        if let Some(backend) = &self.backend {
            backend.write(
                &delete_log_name(self.generation, self.delete_log_sequence),
                &encode_tombstones(&newly),
            )?;
            self.delete_log_sequence += 1;
        }
        let count = newly.len() as u64;
        self.tombstones.extend(newly);
        Ok(count)
    }

    /// Compacts the table: merges every live row — buffer included —
    /// into fresh segments **sorted by (ordering key, ingest sequence)**,
    /// resolves all tombstones, and reassigns contiguous internal row
    /// ids in the new order. This is where "resolved at the next
    /// compaction" happens: deleted rows physically disappear, and the
    /// disorder left by late arrivals or `UPDATE`'s reappends is sorted
    /// away, so a store is always globally ordered right after
    /// compaction. Ties on the ordering key keep ingest order (stable
    /// sort by row id), so duplicates stay first-class and "newest
    /// version wins" stays meaningful.
    ///
    /// On a persistent store the rewrite is crash-safe: the entire next
    /// generation is written first, one atomic manifest write commits
    /// it, and only then are the old generation's objects removed — a
    /// crash anywhere leaves one complete generation to reopen.
    pub fn compact(&mut self) -> Result<(), StorageError> {
        // Collect every live row's (ordering value, row id, location),
        // buffer included via an ephemeral snapshot.
        let views = self.snapshot()?;
        let mut order: Vec<(i64, u64, usize, usize)> = Vec::with_capacity(self.live_len() as usize);
        for (view_index, view) in views.iter().enumerate() {
            let Column::Numeric(NumericData::I64(ordering)) =
                &view.segment.batch().columns()[self.ordering_key]
            else {
                unreachable!("the ordering key is validated as i64 at construction")
            };
            let base = view.segment.base_row_id();
            for (row, &value) in ordering.values().as_slice().iter().enumerate() {
                if view.is_live(row) {
                    order.push((value, base + row as u64, view_index, row));
                }
            }
        }
        order.sort_by_key(|&(value, id, _, _)| (value, id));
        // Rebuild into fresh segments of the configured size.
        let mut new_segments: Vec<Segment> = Vec::new();
        let mut buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        let mut base = 0u64;
        for &(_, _, view_index, row) in &order {
            let batch = views[view_index].segment.batch();
            let cells: Vec<RowValue<'_>> = batch
                .columns()
                .iter()
                .map(|column| cell_value(column, row))
                .collect();
            buffer.append(&cells)?;
            if buffer.len() >= self.segment_rows {
                let rows = buffer.len() as u64;
                let full = std::mem::replace(
                    &mut buffer,
                    WriteBuffer::new(self.schema.clone(), self.ordering_key)?,
                );
                new_segments.push(full.freeze_at(base)?);
                base += rows;
            }
        }
        if !buffer.is_empty() {
            let rows = buffer.len() as u64;
            new_segments.push(buffer.freeze_at(base)?);
            base += rows;
        }
        // Persist the next generation, commit it, then clean up.
        if let Some(backend) = &self.backend {
            let next = self.generation + 1;
            // Pre-clean: a compaction that crashed after writing some
            // next-generation objects left strays under exactly this
            // generation. They must go before we write, or a stray whose
            // base the new layout doesn't overwrite would be loaded as
            // real data after the commit.
            for name in backend.list()? {
                if name.starts_with(&segment_prefix(next))
                    || name.starts_with(&delete_log_prefix(next))
                {
                    backend.remove(&name)?;
                }
            }
            for segment in &new_segments {
                backend.write(
                    &segment_name(next, segment.base_row_id()),
                    &encode_segment(segment),
                )?;
            }
            let manifest =
                WriteBuffer::new(self.schema.clone(), self.ordering_key)?.freeze_at(next)?;
            backend.write(MANIFEST, &encode_segment(&manifest))?;
            // Committed. Old objects are now garbage; reopen would
            // ignore them regardless, but a clean run leaves nothing.
            for name in backend.list()? {
                let current = name.starts_with(&segment_prefix(next))
                    || name.starts_with(&delete_log_prefix(next));
                let stale = (name.starts_with("seg-") || name.starts_with("del-")) && !current;
                if stale {
                    backend.remove(&name)?;
                }
            }
            self.generation = next;
        }
        self.segments = new_segments.into_iter().map(Arc::new).collect();
        self.rows = base;
        self.buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        self.buffer_base = base;
        self.tombstones.clear();
        self.delete_log_sequence = 0;
        Ok(())
    }

    /// A point-in-time view: every frozen segment plus (if the buffer
    /// holds rows) a segment frozen from a copy of it, each carrying the
    /// live mask its tombstones impose. Untombstoned segments come back
    /// mask-free — the zero-copy common case. Appends and tombstones
    /// after the call don't affect the returned views.
    pub fn snapshot(&self) -> Result<Vec<SegmentView>, StorageError> {
        let mut segments = self.segments.clone();
        if !self.buffer.is_empty() {
            segments.push(Arc::new(self.buffer.snapshot_at(self.buffer_base)?));
        }
        Ok(segments
            .into_iter()
            .map(|segment| {
                let base = segment.base_row_id();
                let end = base + segment.batch().num_rows() as u64;
                if self.tombstones.range(base..end).next().is_none() {
                    SegmentView::all_live(segment)
                } else {
                    let live =
                        Bitmap::from_bools((base..end).map(|id| !self.tombstones.contains(&id)));
                    SegmentView {
                        segment,
                        live: Some(live),
                    }
                }
            })
            .collect())
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
                .map(|s| s.segment.batch().num_rows())
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
            segments
                .iter()
                .map(|s| s.segment.base_row_id())
                .collect::<Vec<_>>(),
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
        assert_eq!(before[0].segment.batch().num_rows(), 5);
        let Column::Numeric(NumericData::I64(ts)) = &before[0].segment.batch().columns()[0] else {
            panic!("ts type")
        };
        assert_eq!(ts.values().as_slice(), &[0, 1, 2, 3, 4]);
        // ...and a fresh one sees all nine.
        let after = store.snapshot().unwrap();
        assert_eq!(after[0].segment.batch().num_rows(), 9);
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
        assert_eq!(ptr(&first[0].segment), ptr(&second[0].segment));
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
            .map(|s| s.segment.ordering_bounds().unwrap())
            .collect();
        assert_eq!(bounds, [(0, 2), (3, 5), (6, 8)]);
        assert!(segments.iter().all(|s| s.segment.is_ordered()));
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
        assert_eq!(store.snapshot().unwrap()[0].segment.batch().num_rows(), 2);
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
