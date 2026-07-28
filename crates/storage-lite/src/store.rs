//! The multi-segment container: one [`Store`] per table.
//!
//! A store is an active [`WriteBuffer`] plus the immutable segments it
//! has already frozen. Appends always go to the buffer; when the buffer
//! reaches the store's segment-row threshold it is flushed automatically,
//! so a long-lived store is a growing sequence of bounded segments.
//! Readers never see the buffer directly — [`Store::snapshot`] freezes a
//! point-in-time copy of it (cheap — value and code buffers are
//! copy-on-write; the per-snapshot cost is the null flags and dictionary
//! index, which are copied) and returns the full
//! segment sequence, so appends and queries interleave freely without
//! either blocking the other.
//!
//! ## Row identity starts here (decision #1)
//!
//! The store assigns every appended row an internal monotonic row id and
//! stamps each segment with the id of its first row. Duplicates are
//! first-class — nothing here inspects key values or collapses rows.
//! Tombstones address rows by these ids ([`Store::tombstone`]), and
//! [`Store::compact`] resolves them: live rows merge into fresh
//! segments sorted by (ordering key, ingest sequence) with contiguous
//! new ids, while superseded rows are retained as history segments
//! addressed by ingest sequence alone — crash-safely on a persistent
//! store (see the generation protocol below).
//!
//! ## What a snapshot promises
//!
//! One [`SegmentView`] per segment, in append order, covering exactly
//! the rows appended before the call, each carrying the live mask its
//! tombstones impose. Global ordering is *not* promised — ingest is
//! only expected roughly sorted, and `UPDATE`'s reappends can disorder
//! a table until compaction — so readers that require order (the window
//! executor) check [`Segment::is_ordered`] and the live ordering bounds
//! instead of assuming.

use crate::format::{decode_manifest, decode_segment, encode_manifest, encode_segment};
use crate::io::{IoError, StorageBackend};
use crate::mem::{RowValue, Segment, SequenceInfo, StorageError, WriteBuffer};
use crate::tombstone::{decode_tombstones, encode_tombstones};
use arrow_lite::{Bitmap, Column, ColumnType, NumericData, Schema};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

/// Rows per segment before an automatic flush. Large enough that segment
/// bookkeeping is noise, small enough that a segment is a reasonable unit
/// of compaction and I/O.
pub const DEFAULT_SEGMENT_ROWS: usize = 65_536;

/// The backend object holding the table manifest — a dedicated small
/// record (schema, ordering key, and the table's current **generation**,
/// see below) with its own magic, CRC, and versioning; the format lives
/// in `format.rs` beside the segment's.
const MANIFEST: &str = "table.tlym";

/// The write-ahead log's one name per table. Its header carries the
/// generation, so a log stranded by a crashed compaction is recognized
/// and ignored rather than replayed into the wrong row-id space.
const WAL: &str = "wal.tlyw";

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

/// History segments live outside the generation protocol: their names
/// are recorded in the manifest (which is what makes them real — a
/// crash can strand unlisted `hist-` files, pre-cleaned by the next
/// compaction), and they are never rewritten once listed, so history
/// accumulates without write amplification.
fn history_name(index: usize) -> String {
    format!("hist-{index:010}.tlyseg")
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
    rows: u64,
    /// Sequence number for the next delete log.
    delete_log_sequence: u64,
    /// The current storage generation (bumped by each compaction).
    generation: u64,
    /// Section content the manifest carries (knowledge state, history
    /// segments) — empty until the table first diverges (M4.4).
    manifest_sections: crate::format::ManifestSections,
    /// Where flushed segments also go, if the store is persistent.
    backend: Option<Arc<dyn StorageBackend>>,
    /// The open write-ahead log, when `wal_sync` is not `Off` and the
    /// store is persistent.
    wal: Option<Box<dyn crate::LogWriter>>,
    wal_sync: WalSync,
    last_wal_sync: std::time::Instant,
    /// The reader-visible state, shared with every [`StoreReader`]. The
    /// lock is held only to read or swap it — never across encoding,
    /// backend I/O, or compaction's merge — so a reader's `snapshot()`
    /// waits microseconds at worst (bounded by one write-buffer copy).
    shared: Arc<Mutex<Shared>>,
}

/// What a snapshot reads: the published segments, the live write buffer,
/// and the tombstone set. Everything else in [`Store`] belongs to the
/// single writer alone.
struct Shared {
    segments: Vec<Arc<Segment>>,
    buffer: WriteBuffer,
    /// Row id of the buffer's first row.
    buffer_base: u64,
    /// Birth sequence of the buffer's first row, once the table has
    /// diverged (sequence != row id — see [`crate::SequenceInfo`]);
    /// `None` while virtual. Lives here, not on [`Store`], because
    /// reader snapshots stamp the buffer's segment with it. The
    /// ingest-sequence watermark is derived: `base + buffer.len()`.
    buffer_sequence_base: Option<u64>,
    /// Row ids the table has tombstoned (decision #1: ids, never keys),
    /// each mapped to the sequence its kill landed at (0 = unknown,
    /// from a v1 delete log) — what compaction moves into history
    /// segments' kill coordinates.
    tombstones: BTreeMap<u64, u64>,
    /// History segments: superseded row versions a retaining compaction
    /// preserved. Never in a latest-knowledge snapshot; entered only
    /// under `AS OF`.
    history: Vec<Arc<Segment>>,
}

/// A cheap, cloneable handle that mints point-in-time snapshots while
/// the single writer proceeds — the concurrent-reader half of the
/// single-writer/concurrent-readers cut (#51). `Send`: hand one to a
/// reader thread; every [`StoreReader::snapshot`] briefly takes the
/// same per-store lock the writer takes around its state swaps, and the
/// returned views are fully detached (`Arc`-backed, immutable).
#[derive(Clone)]
pub struct StoreReader {
    shared: Arc<Mutex<Shared>>,
}

impl StoreReader {
    /// A point-in-time view, exactly as [`Store::snapshot`] — callable
    /// from any thread while the writer appends, mutates, or compacts.
    pub fn snapshot(&self) -> Result<Vec<SegmentView>, StorageError> {
        snapshot_of(&lock(&self.shared))
    }

    /// As [`Store::knowledge_snapshot`], from any thread.
    pub fn knowledge_snapshot(&self) -> Result<KnowledgeSnapshot, StorageError> {
        knowledge_snapshot_of(&lock(&self.shared))
    }
}

/// A point-in-time capture of everything an `AS OF` read needs, taken
/// under one lock so the three parts can never be torn against each
/// other: the latest-knowledge views, the history segments, and the
/// pending (uncompacted) tombstones' kill stamps.
pub struct KnowledgeSnapshot {
    /// The latest-knowledge views, exactly as [`Store::snapshot`].
    pub views: Vec<SegmentView>,
    /// History segments (see [`Store::history`]).
    pub history: Vec<Arc<Segment>>,
    /// Pending tombstones: row id → the sequence its kill landed at.
    pub stamps: BTreeMap<u64, u64>,
}

impl KnowledgeSnapshot {
    /// The table as it was known at ingest-sequence `cut`: rows born at
    /// or before the cut and not superseded by it. Live segments keep
    /// rows whose pending tombstone (if any) landed after the cut;
    /// history rows return where their kill came later. The result runs
    /// through the ordinary executor — the knowledge mask is just a
    /// live mask.
    pub fn as_of(&self, cut: u64) -> Vec<SegmentView> {
        let mut out = Vec::with_capacity(self.views.len() + self.history.len());
        for view in &self.views {
            let segment = Arc::clone(&view.segment);
            let base = segment.base_row_id();
            let rows = segment.batch().num_rows();
            let mut mask = Vec::with_capacity(rows);
            let mut all_live = true;
            for row in 0..rows {
                let born = segment.sequence_at(row) <= cut;
                let killed = self
                    .stamps
                    .get(&(base + row as u64))
                    .is_some_and(|&kill| kill <= cut);
                let live = born && !killed;
                all_live &= live;
                mask.push(live);
            }
            out.push(if all_live {
                SegmentView::all_live(segment)
            } else {
                SegmentView {
                    segment,
                    live: Some(Bitmap::from_bools(mask.into_iter())),
                }
            });
        }
        for segment in &self.history {
            let rows = segment.batch().num_rows();
            let kills = segment.superseded();
            let mask = (0..rows).map(|row| {
                // A history row without a kill array (which the engine
                // never writes) reads as killed-at-unknown: never
                // visible — the same conservative reading as a v1
                // delete log.
                let kill = kills.map_or(0, |kills| kills[row]);
                segment.sequence_at(row) <= cut && kill > cut
            });
            out.push(SegmentView {
                segment: Arc::clone(segment),
                live: Some(Bitmap::from_bools(mask)),
            });
        }
        out
    }
}

/// The [`KnowledgeSnapshot`] algorithm over locked state.
fn knowledge_snapshot_of(shared: &Shared) -> Result<KnowledgeSnapshot, StorageError> {
    Ok(KnowledgeSnapshot {
        views: snapshot_of(shared)?,
        history: shared.history.clone(),
        stamps: shared.tombstones.clone(),
    })
}

/// Takes the shared-state lock. A poisoned lock means a writer panicked
/// mid-operation and the buffer may hold a torn row: refuse loudly to
/// serve possibly-torn state rather than limp on.
fn lock(shared: &Arc<Mutex<Shared>>) -> std::sync::MutexGuard<'_, Shared> {
    shared
        .lock()
        .expect("table state lock poisoned: a writer panicked mid-operation")
}

/// When acknowledged appends become durable — decision #43, ruled on a
/// measurement (2026-07-27: group commit at ≤ 100ms cost +0.4µs on a
/// 1.11µs append; fsync-per-append cost ~670×).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalSync {
    /// No write-ahead log: the durability boundary is the flush, the
    /// original contract — for replayable upstreams that re-ingest
    /// from an offset after a crash.
    Off,
    /// Append every row to the log, sync when this much time has
    /// passed since the last sync (in-thread group commit): under a
    /// steady append stream, a crash loses at most this window of
    /// appends. Precisely: the sync rides the next append after the
    /// interval elapses, so a tail written and then left idle stays
    /// unsynced (OS-buffered — surviving a process crash but not power
    /// loss) until the next append, a flush, or the store's drop — a
    /// clean close syncs the tail. The default is 100 ms.
    Group(std::time::Duration),
    /// Sync every append: zero loss window, measured ~670× slower per
    /// append on ordinary disks. For the caller who insists.
    Full,
}

impl Default for WalSync {
    fn default() -> WalSync {
        WalSync::Group(std::time::Duration::from_millis(100))
    }
}

/// Store configuration beyond the required schema and ordering key.
#[derive(Clone, Copy, Debug, Default)]
pub struct StoreOptions {
    /// Rows the write buffer accumulates before freezing a segment;
    /// `None` means [`DEFAULT_SEGMENT_ROWS`] (unless `segment_bytes`
    /// decides instead).
    pub segment_rows: Option<usize>,
    /// The buffer's memory bound in bytes — the knob an embedder
    /// actually budgets (#44). Numeric-or-key makes every column
    /// fixed-width (`i64`/`f64`: 8 bytes; key codes: 4), so bytes
    /// convert exactly to a per-schema row count at construction; key
    /// dictionaries (bounded by distinct values) sit outside the
    /// bound, as documented. Setting both this and `segment_rows` is
    /// refused loudly.
    pub segment_bytes: Option<usize>,
    /// The durability level (persistent stores only; an in-memory
    /// store has nothing to sync to).
    pub wal_sync: WalSync,
}

/// One stored row's fixed width under `schema` — the #44 conversion.
fn row_width(schema: &Schema) -> usize {
    schema
        .fields()
        .iter()
        .map(|field| match field.column_type() {
            ColumnType::I64 | ColumnType::F64 => 8,
            ColumnType::Key => 4,
        })
        .sum()
}

/// The snapshot algorithm over locked state: every frozen segment plus
/// (if the buffer holds rows) a segment frozen from a copy of it, each
/// carrying the live mask its tombstones impose.
fn snapshot_of(shared: &Shared) -> Result<Vec<SegmentView>, StorageError> {
    let mut segments = shared.segments.clone();
    if !shared.buffer.is_empty() {
        let mut segment = shared.buffer.snapshot_at(shared.buffer_base)?;
        // Post-divergence, the buffer's rows carry sequences from the
        // watermark, not their row ids; stamp the snapshot so readers
        // (and flush, which reuses this path's shape) see them.
        if let Some(base) = shared.buffer_sequence_base {
            segment = segment.with_sequence(SequenceInfo::Contiguous { base });
        }
        segments.push(Arc::new(segment));
    }
    Ok(segments
        .into_iter()
        .map(|segment| {
            let base = segment.base_row_id();
            let end = base + segment.batch().num_rows() as u64;
            if shared.tombstones.range(base..end).next().is_none() {
                SegmentView::all_live(segment)
            } else {
                let live =
                    Bitmap::from_bools((base..end).map(|id| !shared.tombstones.contains_key(&id)));
                SegmentView {
                    segment,
                    live: Some(live),
                }
            }
        })
        .collect())
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
            rows: 0,
            delete_log_sequence: 0,
            generation: 0,
            manifest_sections: crate::format::ManifestSections::default(),
            backend: None,
            wal: None,
            wal_sync: WalSync::Off,
            last_wal_sync: std::time::Instant::now(),
            shared: Arc::new(Mutex::new(Shared {
                segments: Vec::new(),
                buffer,
                buffer_base: 0,
                buffer_sequence_base: None,
                tombstones: BTreeMap::new(),
                history: Vec::new(),
            })),
        })
    }

    /// A persistent store over `backend`, flushing every
    /// [`DEFAULT_SEGMENT_ROWS`] rows. Creates the table if the backend is
    /// empty; otherwise reopens it, verifying the manifest against
    /// `schema`/`ordering_key` and every stored segment's checksum,
    /// schema, and row-id contiguity.
    ///
    /// **Durability:** governed by [`WalSync`] (default
    /// `Group(100ms)`): acknowledged appends are logged to a sidecar
    /// WAL and survive a crash up to the sync window; reopen replays
    /// the log's clean prefix on top of the flushed segments. Under
    /// [`WalSync::Off`] the durability boundary is [`Store::flush`] —
    /// rows in the write buffer exist only in memory until flushed,
    /// and a crash loses them.
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
        Store::persistent_with(
            backend,
            schema,
            ordering_key,
            StoreOptions {
                segment_rows: Some(segment_rows),
                ..StoreOptions::default()
            },
        )
    }

    /// Reopens an existing persistent store, taking the schema and
    /// ordering key from its manifest — the doorway a shell or server
    /// uses to open a table it did not create. Errors if the backend
    /// holds no manifest.
    pub fn open_existing(
        backend: Arc<dyn StorageBackend>,
        options: StoreOptions,
    ) -> Result<Store, StorageError> {
        let manifest = decode_manifest(&backend.read(MANIFEST)?)?;
        let schema = manifest.schema.clone();
        let ordering_key = manifest.ordering_key;
        Store::persistent_with(backend, schema, ordering_key, options)
    }

    /// As [`Store::persistent`], with explicit [`StoreOptions`] — the
    /// segment threshold and the durability level (#43).
    pub fn persistent_with(
        backend: Arc<dyn StorageBackend>,
        schema: Schema,
        ordering_key: usize,
        options: StoreOptions,
    ) -> Result<Store, StorageError> {
        let segment_rows = match (options.segment_rows, options.segment_bytes) {
            (Some(_), Some(_)) => {
                return Err(StorageError::Options(
                    "set segment_rows or segment_bytes, not both".to_owned(),
                ))
            }
            (Some(rows), None) => rows,
            (None, Some(bytes)) => (bytes / row_width(&schema)).max(1),
            (None, None) => DEFAULT_SEGMENT_ROWS,
        };
        let mut store = Store::with_segment_rows(schema, ordering_key, segment_rows)?;
        store.wal_sync = options.wal_sync;
        let generation = match backend.read(MANIFEST) {
            Ok(bytes) => {
                let manifest = decode_manifest(&bytes)?;
                if manifest.schema != store.schema {
                    return Err(StorageError::SchemaMismatch {
                        reason: "manifest schema differs from the schema given".to_owned(),
                    });
                }
                if manifest.ordering_key != ordering_key {
                    return Err(StorageError::SchemaMismatch {
                        reason: format!(
                            "manifest orders on column {}, caller asked for {ordering_key}",
                            manifest.ordering_key
                        ),
                    });
                }
                store.manifest_sections = manifest.sections.clone();
                manifest.generation
            }
            Err(IoError::NotFound(_)) => {
                backend.write(
                    MANIFEST,
                    &encode_manifest(
                        &store.schema,
                        ordering_key,
                        0,
                        &crate::format::ManifestSections::default(),
                    ),
                )?;
                0
            }
            Err(error) => return Err(error.into()),
        };
        let mut segments = Vec::new();
        let mut tombstones = BTreeMap::new();
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
                let log = decode_tombstones(&backend.read(&name)?)?;
                tombstones.extend(log.ids.iter().map(|&id| (id, log.stamped_at)));
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
        // A tombstone naming a row id that was never made durable is the
        // fingerprint of a torn mutation written by a pre-fix build (or a
        // corrupt log). Reject it loudly rather than carrying it: left in
        // place it underflows live_len and shadow-kills reissued ids.
        if let Some(&bad) = tombstones.keys().find(|&&id| id >= expected_base) {
            return Err(StorageError::TombstoneOutOfRange { id: bad });
        }
        // History segments are exactly the ones the manifest names — a
        // `hist-` file the manifest does not know is a crashed
        // compaction's stray, invisible here and pre-cleaned by the
        // next compaction.
        let mut history = Vec::with_capacity(store.manifest_sections.history.len());
        for name in &store.manifest_sections.history {
            let segment = decode_segment(&backend.read(name)?)?;
            if segment.batch().schema() != &store.schema {
                return Err(StorageError::SchemaMismatch {
                    reason: format!(
                        "history segment '{name}' was written under a different schema"
                    ),
                });
            }
            history.push(Arc::new(segment));
        }
        // A diverged table's watermark: the manifest records it at each
        // compaction, but flushes advance sequences without rewriting
        // the manifest, so fold the stored segments' ends on top. Rows
        // the WAL replays below then take sequences from here — the
        // same values they had before the crash, since assignment is
        // deterministic in append order. (History needs no fold: it
        // only changes at compaction, where the manifest catches up.)
        let watermark = store.manifest_sections.next_sequence.map(|recorded| {
            segments
                .iter()
                .map(|segment| segment.sequence_end())
                .fold(recorded, u64::max)
        });
        {
            let mut shared = lock(&store.shared);
            shared.segments = segments;
            shared.buffer_base = expected_base;
            shared.buffer_sequence_base = watermark;
            shared.tombstones = tombstones;
            shared.history = history;
        }
        store.rows = expected_base;
        store.delete_log_sequence = next_sequence;
        store.generation = generation;
        store.backend = Some(backend);
        store.replay_wal()?;
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

    /// Rows a reader sees: appended minus tombstoned. Saturating as
    /// defense in depth — a healthy store never tombstones more than it
    /// appended, and the reopen check rejects any log that would, but an
    /// underflow must degrade to zero rather than wrap.
    pub fn live_len(&self) -> u64 {
        self.rows
            .saturating_sub(lock(&self.shared).tombstones.len() as u64)
    }

    /// Whether no rows have ever been appended.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Frozen segments so far (not counting the live buffer).
    pub fn segment_count(&self) -> usize {
        lock(&self.shared).segments.len()
    }

    /// The history segments: superseded row versions preserved by
    /// retaining compactions, in the order they were retained. Never
    /// part of [`Store::snapshot`] — latest-knowledge reads pay nothing
    /// for them; `AS OF` reads walk them explicitly.
    pub fn history(&self) -> Vec<Arc<Segment>> {
        lock(&self.shared).history.clone()
    }

    /// The ingest-sequence watermark: the sequence the next appended
    /// row will receive — also one past the newest knowledge the table
    /// holds, so `AS OF next_sequence() - 1` (on a non-empty table) is
    /// the latest state. Equal to [`Store::len`] until the table
    /// diverges.
    pub fn next_sequence(&self) -> u64 {
        let shared = lock(&self.shared);
        shared
            .buffer_sequence_base
            .map_or(self.rows, |base| base + shared.buffer.len() as u64)
    }

    /// Everything an `AS OF` read needs, captured atomically — see
    /// [`KnowledgeSnapshot`].
    pub fn knowledge_snapshot(&self) -> Result<KnowledgeSnapshot, StorageError> {
        knowledge_snapshot_of(&lock(&self.shared))
    }

    /// A cheap, cloneable reader handle: mints point-in-time snapshots
    /// from any thread while this store's single writer proceeds. See
    /// [`StoreReader`].
    pub fn reader(&self) -> StoreReader {
        StoreReader {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Replays the write-ahead log at reopen: rows past the flushed
    /// segments re-enter the write buffer (and a fresh log), rows a
    /// crash left as a torn tail end the clean prefix silently, and a
    /// log stranded by a crashed compaction (wrong generation) is
    /// ignored — its rows are already in the new generation's segments.
    /// Ends with the log in steady state for the configured level:
    /// recreated and synced under `Group`/`Full`, removed under `Off`.
    fn replay_wal(&mut self) -> Result<(), StorageError> {
        let backend = self.backend.as_ref().expect("replay is a reopen step");
        let recovered = match backend.read(WAL) {
            // Shorter than one header is the crash window between log
            // creation (which truncates in place) and the header sync:
            // no record was ever synced under this log — records follow
            // the header in the same file — so there is nothing to
            // recover, and treating it as corruption would leave the
            // store permanently unopenable over intact segments.
            Ok(bytes) if bytes.len() < crate::format::WAL_HEADER_LEN => Vec::new(),
            Ok(bytes) => {
                let wal = crate::format::decode_wal(&bytes, self.schema.fields().len())?;
                if wal.generation == self.generation {
                    let skip = usize::try_from(self.rows.saturating_sub(wal.base_row_id))
                        .expect("row counts fit usize");
                    wal.rows.into_iter().skip(skip).collect()
                } else {
                    Vec::new()
                }
            }
            Err(IoError::NotFound(_)) => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        if self.wal_sync == WalSync::Off {
            // Recovered rows re-enter the buffer under the flush-boundary
            // contract; the log itself goes away.
            for row in &recovered {
                let cells: Vec<RowValue<'_>> = row
                    .iter()
                    .map(crate::format::WalCell::as_row_value)
                    .collect();
                let mut shared = lock(&self.shared);
                shared.buffer.append(&cells)?;
                self.rows += 1;
            }
            match backend.remove(WAL) {
                Ok(()) | Err(IoError::NotFound(_)) => {}
                Err(error) => return Err(error.into()),
            }
            return Ok(());
        }
        // Assemble the replacement log whole — header plus every
        // recovered record — and publish it atomically *over* the old
        // one. The old log stays the durable copy until the publishing
        // rename commits, so a crash at any instant of recovery leaves
        // exactly one complete log to recover from; truncate-then-
        // rewrite would destroy the only copy first.
        let mut bytes = crate::format::encode_wal_header(self.generation, self.rows);
        let rows: Vec<Vec<RowValue<'_>>> = recovered
            .iter()
            .map(|row| {
                row.iter()
                    .map(crate::format::WalCell::as_row_value)
                    .collect()
            })
            .collect();
        for cells in &rows {
            bytes.extend_from_slice(&crate::format::encode_wal_record(cells));
        }
        backend.write(WAL, &bytes)?;
        for cells in &rows {
            let mut shared = lock(&self.shared);
            shared.buffer.append(cells)?;
            self.rows += 1;
        }
        self.wal = Some(backend.open_log(WAL)?);
        self.last_wal_sync = std::time::Instant::now();
        Ok(())
    }

    /// Replaces the log with an empty one at the current row watermark
    /// — the truncation that follows a flush or compaction, once every
    /// row the old log guarded is segment-durable. Atomic publish, so
    /// no crash instant sees a headerless log.
    fn reset_wal(&mut self) -> Result<(), StorageError> {
        if self.wal.is_none() {
            return Ok(());
        }
        let backend = self.backend.as_ref().expect("a WAL implies a backend");
        backend.write(
            WAL,
            &crate::format::encode_wal_header(self.generation, self.rows),
        )?;
        self.wal = Some(backend.open_log(WAL)?);
        self.last_wal_sync = std::time::Instant::now();
        Ok(())
    }

    /// Logs one appended row and applies the sync level.
    fn wal_append(&mut self, row: &[RowValue<'_>]) -> Result<(), StorageError> {
        let Some(wal) = self.wal.as_mut() else {
            return Ok(());
        };
        wal.append(&crate::format::encode_wal_record(row))?;
        match self.wal_sync {
            WalSync::Full => {
                wal.sync()?;
                self.last_wal_sync = std::time::Instant::now();
            }
            WalSync::Group(interval) => {
                if self.last_wal_sync.elapsed() >= interval {
                    wal.sync()?;
                    self.last_wal_sync = std::time::Instant::now();
                }
            }
            WalSync::Off => unreachable!("Off never opens a log"),
        }
        Ok(())
    }

    /// Appends one row and returns its internal row id. Flushes
    /// automatically when the buffer reaches the segment-row threshold.
    pub fn append(&mut self, row: &[RowValue<'_>]) -> Result<u64, StorageError> {
        // Validate before logging: a rejected row must leave neither
        // buffer nor WAL changed. Logged-then-rejected would plant a
        // phantom record that ends replay's clean prefix early (dropping
        // every acknowledged row after it) or replays a row the caller
        // was told failed.
        {
            let shared = lock(&self.shared);
            shared.buffer.validate(row)?;
        }
        self.wal_append(row)?;
        let must_flush = {
            let mut shared = lock(&self.shared);
            shared.buffer.append(row)?;
            shared.buffer.len() >= self.segment_rows
        };
        let id = self.rows;
        self.rows += 1;
        if must_flush {
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
        // Copy the buffer under the brief lock; encode and publish with
        // the lock released. Readers between the two locks still see the
        // rows — in the buffer, where they were — so every snapshot is
        // consistent; the single-writer cut means nothing else moves.
        let segment = {
            let shared = lock(&self.shared);
            if shared.buffer.is_empty() {
                return Ok(());
            }
            let segment = shared.buffer.snapshot_at(shared.buffer_base)?;
            match shared.buffer_sequence_base {
                // Diverged: the frozen segment records where in the
                // sequence space its rows were born.
                Some(base) => segment.with_sequence(SequenceInfo::Contiguous { base }),
                None => segment,
            }
        };
        if let Some(backend) = &self.backend {
            backend.write(
                &segment_name(self.generation, segment.base_row_id()),
                &encode_segment(&segment),
            )?;
        }
        // Built before the lock so adoption below cannot fail partway.
        let fresh = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        {
            let mut shared = lock(&self.shared);
            let flushed = segment.batch().num_rows() as u64;
            shared.segments.push(Arc::new(segment));
            shared.buffer = fresh;
            shared.buffer_base = self.rows;
            // The next buffered row's sequence follows the flushed rows.
            if let Some(base) = shared.buffer_sequence_base {
                shared.buffer_sequence_base = Some(base + flushed);
            }
        }
        // Every row the log guarded is now segment-durable: truncate.
        // A crash before this point replays a prefix the segments
        // already cover — the header's base row id makes replay skip it.
        self.reset_wal()?;
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
        let (newly, buffer_base, stamp) = {
            let shared = lock(&self.shared);
            let newly: BTreeSet<u64> = ids
                .iter()
                .copied()
                .filter(|id| !shared.tombstones.contains_key(id))
                .collect();
            // The kill's coordinate on the knowledge axis: the current
            // watermark. It is not consumed — the next appended row
            // takes the same sequence, and `AS OF n` applies both ("all
            // knowledge with coordinate <= n"); a flush below moves the
            // buffer without moving the watermark, so the stamp holds.
            let stamp = shared
                .buffer_sequence_base
                .map_or(self.rows, |base| base + shared.buffer.len() as u64);
            (newly, shared.buffer_base, stamp)
        };
        if newly.is_empty() {
            return Ok(0);
        }
        // A delete log must never name a row that is not yet durable. Any
        // id in the current write buffer (>= buffer_base) is in-memory
        // only, so flush before writing the log: this makes every
        // tombstoned row — and any replacement rows a mutation appended
        // ahead of the tombstone — durable first. Without it, a crash
        // after the (synced) delete log but before a flush would apply a
        // delete against a row that never reached disk, and reopen would
        // carry a tombstone for a row id it then reissues (silent
        // shadow-kill of future rows).
        if self.backend.is_some() && newly.iter().any(|&id| id >= buffer_base) {
            self.flush()?;
        }
        // The companion rule: no delete log may commit ahead of the
        // rows that *supersede* its victims. UPDATE appends its
        // replacements before tombstoning the originals, and the delete
        // log below is synced immediately — so if the replacements were
        // still riding an unsynced WAL tail (or, under `Off`, the
        // buffer), a crash here would recover the deletion without the
        // replacements: the one middle state that loses data forever.
        // Sync the log (cheap) — or without one, flush — first.
        if self.backend.is_some() {
            if let Some(wal) = self.wal.as_mut() {
                wal.sync().map_err(StorageError::from)?;
                self.last_wal_sync = std::time::Instant::now();
            } else {
                let buffered = !lock(&self.shared).buffer.is_empty();
                if buffered {
                    self.flush()?;
                }
            }
        }
        if let Some(backend) = &self.backend {
            backend.write(
                &delete_log_name(self.generation, self.delete_log_sequence),
                &encode_tombstones(&newly, stamp),
            )?;
            self.delete_log_sequence += 1;
        }
        let count = newly.len() as u64;
        lock(&self.shared)
            .tombstones
            .extend(newly.into_iter().map(|id| (id, stamp)));
        Ok(count)
    }

    /// Compacts the table: merges every live row — buffer included —
    /// into fresh segments **sorted by (ordering key, ingest sequence)**,
    /// resolves all tombstones, and reassigns contiguous internal row
    /// ids in the new order. This is where "resolved at the next
    /// compaction" happens: deleted rows leave the live set — **retained
    /// as history segments** with their birth and kill coordinates
    /// (the corrections model, #75; latest-knowledge reads never touch
    /// them) — and the disorder left by late arrivals or `UPDATE`'s
    /// reappends is sorted away, so a store is always globally ordered
    /// right after compaction. Ties on the ordering key keep ingest
    /// order (stable sort by row id), so duplicates stay first-class
    /// and "newest version wins" stays meaningful. A compaction that
    /// retains rows or moves any row id **diverges** the table: birth
    /// sequences freeze as they were, row ids renumber freely, and the
    /// two axes never rejoin (an ordered, untombstoned table compacts
    /// to itself and stays virtual).
    ///
    /// On a persistent store the rewrite is crash-safe: the entire next
    /// generation is written first, one atomic manifest write commits
    /// it, and only then are the old generation's objects removed
    /// (best-effort — a removal failure leaves ignorable garbage, never a
    /// stranded generation) — a crash anywhere leaves one complete
    /// generation to reopen.
    ///
    /// Memory: compaction holds the old generation and the fully rebuilt
    /// new one at once, plus a sort index (~32 bytes/row), so it peaks at
    /// roughly twice the table's footprint — and it is the only way to
    /// release `UPDATE`/`DELETE` debt, so it runs precisely when the table
    /// is already inflated (interacts with #43/#44, #56).
    pub fn compact(&mut self) -> Result<(), StorageError> {
        // The ingest-sequence watermark as recorded so far, plus the
        // kill stamps and prior history, captured before the merge: a
        // diverged table's compaction must carry all three through —
        // row ids reassign below, knowledge coordinates never do.
        let (recorded_watermark, stamps, old_history) = {
            let shared = lock(&self.shared);
            (
                shared
                    .buffer_sequence_base
                    .map(|base| base + shared.buffer.len() as u64),
                shared.tombstones.clone(),
                shared.history.clone(),
            )
        };
        // Collect every row's (ordering value, row id, location),
        // buffer included via an ephemeral snapshot — live rows headed
        // for the new generation, dead rows for history.
        let views = self.snapshot()?;
        let capacity = views.iter().map(SegmentView::live_rows).sum();
        let mut order: Vec<(i64, u64, usize, usize)> = Vec::with_capacity(capacity);
        let mut dead_order: Vec<(i64, u64, usize, usize)> = Vec::new();
        for (view_index, view) in views.iter().enumerate() {
            let Column::Numeric(NumericData::I64(ordering)) =
                &view.segment.batch().columns()[self.ordering_key]
            else {
                unreachable!("the ordering key is validated as i64 at construction")
            };
            let base = view.segment.base_row_id();
            for (row, &value) in ordering.values().as_slice().iter().enumerate() {
                let entry = (value, base + row as u64, view_index, row);
                if view.is_live(row) {
                    order.push(entry);
                } else {
                    dead_order.push(entry);
                }
            }
        }
        order.sort_by_key(|&(value, id, _, _)| (value, id));
        dead_order.sort_by_key(|&(value, id, _, _)| (value, id));
        // Does this compaction break sequence == row id? Yes if it
        // already broke (diverged), if anything is retained (a dead
        // row's sequence outlives its id, so future ids must not reuse
        // it), or if the merge moves any live row to a new id. A
        // compaction of an ordered, untombstoned table changes nothing
        // and the table stays virtual.
        let diverging = recorded_watermark.is_some()
            || !dead_order.is_empty()
            || order
                .iter()
                .enumerate()
                .any(|(new_id, &(_, id, _, _))| id != new_id as u64);
        let watermark = if diverging {
            Some(recorded_watermark.unwrap_or(self.rows))
        } else {
            None
        };
        // Rebuild into fresh segments of the configured size. A
        // diverged table's rows keep their birth sequences through the
        // merge — gathered in merge order, so the new segments carry
        // them explicitly (the merge follows the ordering key, not
        // ingest, making sequences non-contiguous).
        let mut new_segments: Vec<Segment> = Vec::new();
        let mut buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        let mut sequences: Vec<u64> = Vec::new();
        let mut base = 0u64;
        for &(_, _, view_index, row) in &order {
            let view_segment = &views[view_index].segment;
            let cells: Vec<RowValue<'_>> = view_segment
                .batch()
                .columns()
                .iter()
                .map(|column| cell_value(column, row))
                .collect();
            buffer.append(&cells)?;
            if watermark.is_some() {
                sequences.push(view_segment.sequence_at(row));
            }
            if buffer.len() >= self.segment_rows {
                let rows = buffer.len() as u64;
                let full = std::mem::replace(
                    &mut buffer,
                    WriteBuffer::new(self.schema.clone(), self.ordering_key)?,
                );
                let mut segment = full.freeze_at(base)?;
                if watermark.is_some() {
                    segment = segment
                        .with_sequence(SequenceInfo::Explicit(std::mem::take(&mut sequences)));
                }
                new_segments.push(segment);
                base += rows;
            }
        }
        if !buffer.is_empty() {
            let rows = buffer.len() as u64;
            let mut segment = buffer.freeze_at(base)?;
            if watermark.is_some() {
                segment = segment.with_sequence(SequenceInfo::Explicit(sequences));
            }
            new_segments.push(segment);
            base += rows;
        }
        // Dead rows become history segments: same merge order, same
        // chunking, each row carrying its birth and kill coordinates.
        // Their base row id is 0 — history rows have no live identity;
        // they are addressed by sequence alone.
        let mut new_history: Vec<Segment> = Vec::new();
        if !dead_order.is_empty() {
            let mut buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
            let mut births: Vec<u64> = Vec::new();
            let mut kills: Vec<u64> = Vec::new();
            for &(_, id, view_index, row) in &dead_order {
                let view_segment = &views[view_index].segment;
                let cells: Vec<RowValue<'_>> = view_segment
                    .batch()
                    .columns()
                    .iter()
                    .map(|column| cell_value(column, row))
                    .collect();
                buffer.append(&cells)?;
                births.push(view_segment.sequence_at(row));
                kills.push(*stamps.get(&id).expect("a dead row has a tombstone"));
                if buffer.len() >= self.segment_rows {
                    let full = std::mem::replace(
                        &mut buffer,
                        WriteBuffer::new(self.schema.clone(), self.ordering_key)?,
                    );
                    new_history.push(
                        full.freeze()?
                            .with_sequence(SequenceInfo::Explicit(std::mem::take(&mut births)))
                            .with_superseded(std::mem::take(&mut kills)),
                    );
                }
            }
            if !buffer.is_empty() {
                new_history.push(
                    buffer
                        .freeze()?
                        .with_sequence(SequenceInfo::Explicit(births))
                        .with_superseded(kills),
                );
            }
        }
        // Built now, before the commit point, so adopting the new
        // generation in memory below cannot fail partway.
        let fresh_buffer = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        // The manifest the commit will write: the current watermark (the
        // one durable record that sequences run ahead of row ids) and
        // the accumulated history names. Built as a local and adopted
        // only at the commit point, so a failed compaction never leaves
        // memory claiming history the backend does not hold.
        let mut sections = self.manifest_sections.clone();
        if watermark.is_some() {
            sections.next_sequence = watermark;
        }
        let new_history_names: Vec<String> = (0..new_history.len())
            .map(|index| history_name(sections.history.len() + index))
            .collect();
        sections.history.extend(new_history_names.iter().cloned());

        // Persist the next generation and commit it atomically.
        if let Some(backend) = &self.backend {
            let next = self.generation + 1;
            // Pre-clean: a compaction that crashed after writing some
            // next-generation objects left strays under exactly this
            // generation — and possibly `hist-` files the manifest never
            // came to name. They must go before we write, or a stray
            // whose base the new layout doesn't overwrite would be
            // loaded as real data after the commit.
            for name in backend.list()? {
                let stray_generation = name.starts_with(&segment_prefix(next))
                    || name.starts_with(&delete_log_prefix(next));
                let stray_history =
                    name.starts_with("hist-") && !self.manifest_sections.history.contains(&name);
                if stray_generation || stray_history {
                    backend.remove(&name)?;
                }
            }
            for (name, segment) in new_history_names.iter().zip(&new_history) {
                backend.write(name, &encode_segment(segment))?;
            }
            for segment in &new_segments {
                backend.write(
                    &segment_name(next, segment.base_row_id()),
                    &encode_segment(segment),
                )?;
            }
            // The manifest write is the commit point.
            backend.write(
                MANIFEST,
                &encode_manifest(&self.schema, self.ordering_key, next, &sections),
            )?;
            self.generation = next;
        }
        self.manifest_sections = sections;

        // In-memory commit: adopt the new generation. Infallible, taken
        // under one brief lock, and run immediately after the durable
        // commit, so no later error can leave memory describing the old
        // generation while disk holds the new one — the stranding that
        // made every subsequent write vanish at reopen (R1). A reader
        // holding pre-swap views keeps its segments alive through their
        // `Arc`s — read-copy-update, no coordination needed.
        {
            let mut shared = lock(&self.shared);
            shared.segments = new_segments.into_iter().map(Arc::new).collect();
            shared.buffer = fresh_buffer;
            shared.buffer_base = base;
            shared.buffer_sequence_base = watermark;
            shared.tombstones.clear();
            shared.history = old_history
                .into_iter()
                .chain(new_history.into_iter().map(Arc::new))
                .collect();
        }
        self.rows = base;
        self.delete_log_sequence = 0;
        // The old log's rows — buffer included — are all in the new
        // generation's segments; recreate it under the new generation.
        self.reset_wal()?;

        // Best-effort cleanup of the now-stale prior generation. A failure
        // here only leaves garbage that reopen already ignores (it loads
        // the one generation the manifest names); it must never fail the
        // compaction or strand the generation, so a remove error is
        // swallowed rather than propagated.
        if let Some(backend) = &self.backend {
            let current = self.generation;
            if let Ok(names) = backend.list() {
                for name in names {
                    let belongs = name.starts_with(&segment_prefix(current))
                        || name.starts_with(&delete_log_prefix(current));
                    let stale = (name.starts_with("seg-") || name.starts_with("del-")) && !belongs;
                    if stale {
                        let _ = backend.remove(&name);
                    }
                }
            }
        }
        Ok(())
    }

    /// Marks the table diverged at `watermark`: from here on, appended
    /// rows take birth sequences from the watermark instead of their
    /// row ids, and the manifest records it. [`Store::compact`] crosses
    /// this seam itself the first time it retains or renumbers; tests
    /// drive it directly to pin the plumbing at chosen watermarks.
    #[cfg(test)]
    fn diverge(&mut self, watermark: u64) -> Result<(), StorageError> {
        self.manifest_sections.next_sequence = Some(watermark);
        lock(&self.shared).buffer_sequence_base = Some(watermark);
        if let Some(backend) = &self.backend {
            backend.write(
                MANIFEST,
                &encode_manifest(
                    &self.schema,
                    self.ordering_key,
                    self.generation,
                    &self.manifest_sections,
                ),
            )?;
        }
        Ok(())
    }

    /// A point-in-time view: every frozen segment plus (if the buffer
    /// holds rows) a segment frozen from a copy of it, each carrying the
    /// live mask its tombstones impose. Untombstoned segments come back
    /// mask-free — the zero-copy common case. Appends and tombstones
    /// after the call don't affect the returned views.
    pub fn snapshot(&self) -> Result<Vec<SegmentView>, StorageError> {
        snapshot_of(&lock(&self.shared))
    }
}

impl Drop for Store {
    /// A clean close syncs the log's tail (best-effort): the last
    /// group-commit window must not depend on a next append that never
    /// comes. Power loss while idle can still take the OS-buffered
    /// tail — tests that model power loss leak the store
    /// (`std::mem::forget`) instead of dropping it.
    fn drop(&mut self) {
        if let Some(wal) = self.wal.as_mut() {
            let _ = wal.sync();
        }
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

    #[test]
    fn a_diverged_table_stamps_sequences_along_the_watermark() {
        // M4.4 step 3: the two axes come apart at divergence — row ids
        // keep numbering storage positions, birth sequences continue
        // from the watermark — and every snapshot and flush carries the
        // sequence data readers will need under `AS OF`.
        let mut store = Store::with_segment_rows(schema(), 0, 4).unwrap();
        append_n(&mut store, 0..2);
        let views = store.snapshot().unwrap();
        assert_eq!(views[0].segment.sequence_info(), &SequenceInfo::RowIds);
        assert_eq!(views[0].segment.sequence_at(1), 1);
        store.flush().unwrap();
        store.diverge(100).unwrap();
        append_n(&mut store, 2..4);
        let views = store.snapshot().unwrap();
        // The buffer snapshot is stamped; row ids stay their own axis.
        assert_eq!(
            views[1].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 100 }
        );
        assert_eq!(views[1].segment.base_row_id(), 2);
        assert_eq!(views[1].segment.sequence_at(1), 101);
        // The flushed segment carries the same stamp, and the next
        // buffer picks up where the flush left the sequence space.
        store.flush().unwrap();
        append_n(&mut store, 4..5);
        let views = store.snapshot().unwrap();
        assert_eq!(
            views[1].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 100 }
        );
        assert_eq!(
            views[2].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 102 }
        );
    }

    #[test]
    fn compaction_preserves_birth_sequences_and_the_watermark() {
        // Compaction reassigns row ids downward and reorders rows on
        // the ordering key; a diverged table's birth sequences must
        // survive both, and the watermark must never rewind (sequences
        // are permanent even when their rows die).
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        store.diverge(10).unwrap();
        for ts in [5, 1, 3] {
            store
                .append(&[RowValue::I64(ts), RowValue::Key("A"), RowValue::F64(0.0)])
                .unwrap();
        }
        // Kill ts=1 (row id 1, sequence 11).
        store.tombstone(&[1]).unwrap();
        store.compact().unwrap();
        let views = store.snapshot().unwrap();
        // Merge order is ts 3, 5 — sequences 12, 10, explicit because
        // the merge follows the ordering key, not ingest.
        assert_eq!(
            views[0].segment.sequence_info(),
            &SequenceInfo::Explicit(vec![12, 10])
        );
        assert_eq!(views[0].segment.base_row_id(), 0);
        // The next append's sequence follows the dead row's, not the
        // compacted row count: 13, never 12 again.
        store
            .append(&[RowValue::I64(9), RowValue::Key("A"), RowValue::F64(0.0)])
            .unwrap();
        let views = store.snapshot().unwrap();
        assert_eq!(
            views[1].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 13 }
        );
    }

    #[test]
    fn a_diverged_reopen_recovers_the_watermark_past_the_manifest() {
        // Flushes advance sequences without rewriting the manifest, so
        // reopen folds segment ends over the recorded watermark — and
        // WAL replay hands the buffered rows the same sequences they
        // had before the close.
        let backend: Arc<dyn crate::StorageBackend> = Arc::new(crate::MemBackend::new());
        let mut store =
            Store::persistent_with_segment_rows(Arc::clone(&backend), schema(), 0, 4).unwrap();
        store.diverge(50).unwrap(); // manifest records 50
        append_n(&mut store, 0..6); // auto-flush at 4; two rows stay buffered
        drop(store);
        let store =
            Store::persistent_with_segment_rows(Arc::clone(&backend), schema(), 0, 4).unwrap();
        let views = store.snapshot().unwrap();
        assert_eq!(
            views[0].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 50 }
        );
        // The replayed buffer continues at 54 — past the stale
        // manifest's 50, because the flushed segment's end wins.
        assert_eq!(
            views[1].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 54 }
        );
        // And a compaction commits the full state durably: explicit
        // sequences in the segments, the watermark in the manifest.
        let mut store = store;
        store.compact().unwrap();
        drop(store);
        let mut store =
            Store::persistent_with_segment_rows(Arc::clone(&backend), schema(), 0, 4).unwrap();
        let views = store.snapshot().unwrap();
        assert_eq!(
            views[0].segment.sequence_info(),
            &SequenceInfo::Explicit(vec![50, 51, 52, 53])
        );
        store
            .append(&[RowValue::I64(9), RowValue::Key("A"), RowValue::F64(0.0)])
            .unwrap();
        let views = store.snapshot().unwrap();
        assert_eq!(
            views.last().unwrap().segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 56 }
        );
    }
}
