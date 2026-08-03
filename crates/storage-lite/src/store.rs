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

use crate::format::{
    decode_manifest, decode_segment, encode_manifest, encode_segment, SegmentRecord,
};
use crate::io::{IoError, StorageBackend};
use crate::mem::{RowValue, Segment, SequenceInfo, StorageError, WriteBuffer, ZoneMap};
use crate::tombstone::{decode_tombstones, encode_tombstones, DeleteLog};
use arrow_lite::{Bitmap, Column, ColumnType, NumericData, Schema};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Rows per segment before an automatic flush. Large enough that segment
/// bookkeeping is noise, small enough that a segment is a reasonable unit
/// of compaction and I/O.
pub const DEFAULT_SEGMENT_ROWS: usize = 65_536;

/// The backend object holding the table manifest — a dedicated small
/// record (schema, ordering key, and the table's current **generation**,
/// see below) with its own magic, CRC, and versioning; the format lives
/// in `format.rs` beside the segment's.
/// The manifest's filename — public because it is the marker a
/// directory scanner (the console, the oracle harness) tests to
/// recognize a store directory, and two hand-rolled copies of the
/// literal had already grown before this was exported.
pub const MANIFEST: &str = "table.tlym";

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

/// A recovered WAL unit after supersession brackets resolve (see
/// [`Store::replay_wal`]).
enum Replayed {
    Row(Vec<crate::format::WalCell>),
    Supersession {
        sequence: u64,
        rows: Vec<Vec<crate::format::WalCell>>,
    },
}

impl Replayed {
    fn row_count(&self) -> usize {
        match self {
            Replayed::Row(_) => 1,
            Replayed::Supersession { rows, .. } => rows.len(),
        }
    }
}

/// Owned WAL cells as the borrowed row the append path takes.
fn owned_cells(cells: &[crate::format::WalCell]) -> Vec<RowValue<'_>> {
    cells
        .iter()
        .map(crate::format::WalCell::as_row_value)
        .collect()
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

/// Residency accounting shared by every lazy slot of one store — the
/// engineering half of the 2026-07-30 residency ruling (option b):
/// decoded segments are a cache over the compressed files, retained
/// under a byte budget and evicted least-recently-used.
///
/// The budget is **advisory over retention, never over correctness**:
/// a segment some snapshot or query still holds (its `Arc` is shared)
/// is never evicted, so one query's working set may transiently exceed
/// the budget — the documented bound is budget + the largest concurrent
/// working set. `None` means unbounded (the interim default; the
/// default's final value is decision #87).
pub(crate) struct SegmentCache {
    budget: Option<u64>,
    /// Bytes currently retained by slots (not counting evicted segments
    /// queries still pin — those are theirs, not the cache's).
    resident: Mutex<u64>,
    /// A monotone use-clock; slots stamp it on every touch.
    clock: AtomicU64,
    /// Every evictable slot ever minted for this store; dead weak refs
    /// are swept during eviction scans.
    registry: Mutex<Vec<Weak<SegmentSlot>>>,
}

impl SegmentCache {
    fn new(budget: Option<u64>) -> Arc<SegmentCache> {
        Arc::new(SegmentCache {
            budget,
            resident: Mutex::new(0),
            clock: AtomicU64::new(0),
            registry: Mutex::new(Vec::new()),
        })
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn register(&self, slot: &Arc<SegmentSlot>) {
        self.registry
            .lock()
            .expect("cache registry lock poisoned")
            .push(Arc::downgrade(slot));
    }

    /// Accounts `bytes` of freshly decoded segment and evicts the
    /// least-recently-used unpinned slots until the total fits the
    /// budget again. A pinned slot (its segment `Arc` is shared with a
    /// snapshot or query) is skipped — eviction reclaims only memory
    /// nothing is reading.
    fn admit(&self, bytes: u64) {
        let mut resident = self.resident.lock().expect("cache lock poisoned");
        *resident += bytes;
        let Some(budget) = self.budget else { return };
        if *resident <= budget {
            return;
        }
        let mut candidates: Vec<(u64, Arc<SegmentSlot>)> = {
            let mut registry = self.registry.lock().expect("cache registry lock poisoned");
            registry.retain(|weak| weak.strong_count() > 0);
            registry
                .iter()
                .filter_map(Weak::upgrade)
                .map(|slot| (slot.last_touch.load(Ordering::Relaxed), slot))
                .collect()
        };
        candidates.sort_by_key(|(touch, _)| *touch);
        for (_, slot) in candidates {
            if *resident <= budget {
                break;
            }
            if let Some(freed) = slot.evict() {
                *resident = resident.saturating_sub(freed);
            }
        }
        // Still over budget here means everything left is pinned — the
        // advisory overshoot; it drains as those readers finish.
    }
}

/// What a read-only refresh carries from the previous open: the slot
/// context (cache included) and the slots themselves, so anything the
/// new manifest still names keeps its decoded state.
struct RefreshCarry {
    shared: Arc<SlotShared>,
    segments: Vec<Arc<SegmentSlot>>,
    history: Vec<Arc<SegmentSlot>>,
}

impl RefreshCarry {
    fn slot_named(&self, name: &str) -> Option<Arc<SegmentSlot>> {
        self.segments
            .iter()
            .find(|slot| slot.name.as_deref() == Some(name))
            .cloned()
    }

    fn history_named(&self, name: &str) -> Option<Arc<SegmentSlot>> {
        self.history
            .iter()
            .find(|slot| slot.name.as_deref() == Some(name))
            .cloned()
    }
}

/// Per-store context every slot shares: what a fault-in needs to read,
/// verify, and account a segment.
pub(crate) struct SlotShared {
    schema: Schema,
    ordering_key: usize,
    backend: Option<Arc<dyn StorageBackend>>,
    cache: Arc<SegmentCache>,
}

/// What a slot knows about its segment without decoding it — served
/// from the manifest's segment record (tag 1) or derived from a
/// decoded segment; either way it answers planning (pruning, live
/// masks, ordering) with zero I/O.
pub(crate) struct SegmentMeta {
    base_row_id: u64,
    rows: usize,
    ordered: bool,
    /// One past the largest birth sequence (open-time watermark folds).
    sequence_end: u64,
    /// Whether sequences have left the virtual state.
    diverged: bool,
    zone_maps: Vec<Option<ZoneMap>>,
}

impl SegmentMeta {
    fn of(segment: &Segment) -> SegmentMeta {
        SegmentMeta {
            base_row_id: segment.base_row_id(),
            rows: segment.batch().num_rows(),
            ordered: segment.is_ordered(),
            sequence_end: segment.sequence_end(),
            diverged: segment.sequence_info() != &SequenceInfo::RowIds,
            // Preserve map-less-ness: a segment built without zone maps
            // (`from_batch_unpruned`) must stay "no maps at all" here,
            // not become a map of all-`None` columns — a per-column
            // `None` means "no valid values, prune", and promoting the
            // former into the latter silently prunes every scratch
            // segment a numeric predicate touches.
            zone_maps: if segment.zone_maps_present() {
                (0..segment.batch().columns().len())
                    .map(|index| segment.zone_map(index).copied())
                    .collect()
            } else {
                Vec::new()
            },
        }
    }

    fn from_record(record: &SegmentRecord) -> SegmentMeta {
        SegmentMeta {
            base_row_id: record.base_row_id,
            rows: record.rows as usize,
            ordered: record.ordered,
            sequence_end: record.sequence_end(),
            diverged: record.diverged(),
            zone_maps: record.zone_maps.clone(),
        }
    }
}

/// One segment's residency slot: metadata always in memory, the decoded
/// segment faulted in from the backend on first data access and
/// evictable under the cache budget. The store's shared state holds
/// slots; snapshots hold [`SegmentHandle`]s over them; the decoded
/// [`Segment`] exists only while resident or pinned.
pub(crate) struct SegmentSlot {
    /// `None` on history slots, which are only ever read whole under
    /// `AS OF` — they fault for data, never answer metadata.
    meta: Option<SegmentMeta>,
    /// The backend object to fault from; `None` for permanently
    /// resident slots (in-memory stores, buffer snapshots, scratch).
    name: Option<String>,
    shared: Arc<SlotShared>,
    state: Mutex<Option<Arc<Segment>>>,
    /// Decoded footprint, known once decoded (0 until then).
    bytes: AtomicU64,
    last_touch: AtomicU64,
}

impl SegmentSlot {
    /// A slot already holding its decoded segment. Evictable exactly
    /// when `name` names a backend object to fault it back from.
    fn resident(
        segment: Arc<Segment>,
        shared: Arc<SlotShared>,
        name: Option<String>,
    ) -> Arc<SegmentSlot> {
        let bytes = segment.resident_bytes();
        let slot = Arc::new(SegmentSlot {
            meta: Some(SegmentMeta::of(&segment)),
            name,
            state: Mutex::new(Some(segment)),
            bytes: AtomicU64::new(bytes),
            last_touch: AtomicU64::new(shared.cache.tick()),
            shared,
        });
        if slot.evictable() {
            slot.shared.cache.register(&slot);
            slot.shared.cache.admit(bytes);
        }
        slot
    }

    /// A slot that has never decoded: metadata from the manifest
    /// record, data faulted on first touch.
    fn lazy(record: &SegmentRecord, shared: Arc<SlotShared>) -> Arc<SegmentSlot> {
        let slot = Arc::new(SegmentSlot {
            meta: Some(SegmentMeta::from_record(record)),
            name: Some(record.name.clone()),
            state: Mutex::new(None),
            bytes: AtomicU64::new(0),
            last_touch: AtomicU64::new(shared.cache.tick()),
            shared,
        });
        slot.shared.cache.register(&slot);
        slot
    }

    /// A history slot: no metadata, faulted whole under `AS OF`.
    fn history(name: String, shared: Arc<SlotShared>) -> Arc<SegmentSlot> {
        let slot = Arc::new(SegmentSlot {
            meta: None,
            name: Some(name),
            state: Mutex::new(None),
            bytes: AtomicU64::new(0),
            last_touch: AtomicU64::new(shared.cache.tick()),
            shared,
        });
        slot.shared.cache.register(&slot);
        slot
    }

    fn evictable(&self) -> bool {
        self.name.is_some() && self.shared.backend.is_some()
    }

    fn meta(&self) -> &SegmentMeta {
        self.meta
            .as_ref()
            .expect("only history slots lack metadata, and they are never asked")
    }

    /// The decoded segment: resident, or faulted in now. The fault
    /// reads the whole file (the backend is whole-object), decodes,
    /// verifies schema and metadata agreement, and accounts the bytes —
    /// evicting colder slots if a budget is set.
    fn segment(&self) -> Result<Arc<Segment>, StorageError> {
        self.last_touch
            .store(self.shared.cache.tick(), Ordering::Relaxed);
        let mut state = self.state.lock().expect("slot lock poisoned");
        if let Some(segment) = &*state {
            return Ok(Arc::clone(segment));
        }
        let name = self.name.as_ref().expect("an empty slot is named");
        let backend = self
            .shared
            .backend
            .as_ref()
            .expect("an empty slot has a backend");
        let segment = decode_segment(&backend.read(name)?)?;
        if segment.batch().schema() != &self.shared.schema {
            return Err(StorageError::SchemaMismatch {
                reason: format!("segment '{name}' was written under a different schema"),
            });
        }
        if let Some(meta) = &self.meta {
            if segment.base_row_id() != meta.base_row_id
                || segment.batch().num_rows() != meta.rows
                || segment.is_ordered() != meta.ordered
                || segment.sequence_end() != meta.sequence_end
            {
                return Err(StorageError::SchemaMismatch {
                    reason: format!("segment '{name}' disagrees with its manifest record"),
                });
            }
        }
        let segment = Arc::new(segment);
        let bytes = segment.resident_bytes();
        self.bytes.store(bytes, Ordering::Relaxed);
        *state = Some(Arc::clone(&segment));
        drop(state);
        self.shared.cache.admit(bytes);
        Ok(segment)
    }

    /// Drops the resident segment if nothing outside the slot holds it.
    /// Returns the bytes freed. `try_lock`: a slot mid-fault is simply
    /// not a candidate this round.
    fn evict(&self) -> Option<u64> {
        let mut state = self.state.try_lock().ok()?;
        let segment = state.as_ref()?;
        if Arc::strong_count(segment) > 1 {
            return None; // pinned by a snapshot or a running query
        }
        *state = None;
        Some(self.bytes.load(Ordering::Relaxed))
    }
}

/// One segment as a snapshot addresses it: always-available metadata
/// (row span, ordering, zone maps, the live mask) plus [`Self::view`],
/// which faults the decoded segment in on first data access. Queries
/// prune on the metadata and pay decode only for the segments that
/// survive — the query-side half of the residency design.
#[derive(Clone)]
pub struct SegmentHandle {
    slot: Arc<SegmentSlot>,
    live: Option<Bitmap>,
}

impl SegmentHandle {
    /// A handle over an already-decoded, permanently resident segment —
    /// the shape `AS OF` reads, buffer snapshots, and tests produce.
    pub fn resident(segment: Arc<Segment>, live: Option<Bitmap>) -> SegmentHandle {
        let shared = Arc::new(SlotShared {
            schema: segment.batch().schema().clone(),
            ordering_key: segment.ordering_key(),
            backend: None,
            cache: SegmentCache::new(None),
        });
        SegmentHandle {
            slot: SegmentSlot::resident(segment, shared, None),
            live,
        }
    }

    /// Rows in the segment (live or not).
    pub fn rows(&self) -> usize {
        self.slot.meta().rows
    }

    /// Id of the segment's first row.
    pub fn base_row_id(&self) -> u64 {
        self.slot.meta().base_row_id
    }

    /// Whether the ordering key arrived non-decreasing.
    pub fn is_ordered(&self) -> bool {
        self.slot.meta().ordered
    }

    /// Index of the declared ordering-key column.
    pub fn ordering_key(&self) -> usize {
        self.slot.shared.ordering_key
    }

    /// The zone map for column `index` — same meaning as
    /// [`Segment::zone_map`], served without touching the segment file.
    pub fn zone_map(&self, index: usize) -> Option<&ZoneMap> {
        self.slot
            .meta()
            .zone_maps
            .get(index)
            .and_then(Option::as_ref)
    }

    /// Whether this segment carries zone maps at all — same meaning as
    /// [`Segment::zone_maps_present`].
    pub fn zone_maps_present(&self) -> bool {
        !self.slot.meta().zone_maps.is_empty()
    }

    /// One past the largest birth sequence in this segment — same
    /// meaning as [`Segment::sequence_end`], served without touching
    /// the segment file. A maintained view's refresh (#83) skips every
    /// segment where this is at or below its stamp, which is what
    /// keeps refresh cost proportional to what changed rather than to
    /// the table.
    pub fn sequence_end(&self) -> u64 {
        self.slot.meta().sequence_end
    }

    /// Bit per row, `true` = live; `None` when nothing is tombstoned.
    pub fn live(&self) -> Option<&Bitmap> {
        self.live.as_ref()
    }

    /// Rows a reader will actually see.
    pub fn live_rows(&self) -> usize {
        match &self.live {
            None => self.rows(),
            Some(mask) => mask.count_set(),
        }
    }

    /// Whether local row `row` is live.
    pub fn is_live(&self, row: usize) -> bool {
        self.live.as_ref().is_none_or(|mask| mask.get(row))
    }

    /// The data view: the decoded segment plus this handle's live mask.
    /// This is the fault point — everything above answers from metadata.
    pub fn view(&self) -> Result<SegmentView, StorageError> {
        Ok(SegmentView {
            segment: self.slot.segment()?,
            live: self.live.clone(),
        })
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
/// let rows: Vec<usize> = segments.iter().map(|s| s.rows()).collect();
/// assert_eq!(rows, [2, 2, 1]);
/// assert_eq!(segments[2].base_row_id(), 4);
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
    /// A read-only handle (F4): opened over a directory another process
    /// writes. Every mutating operation refuses; [`Store::refresh`]
    /// re-reads the durable state.
    read_only: bool,
    /// The per-store context every segment slot shares: schema,
    /// backend, and the residency cache. Rebuilt when the store gains
    /// its backend at open.
    slot_shared: Arc<SlotShared>,
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
    segments: Vec<Arc<SegmentSlot>>,
    buffer: WriteBuffer,
    /// Row id of the buffer's first row.
    buffer_base: u64,
    /// The diverged half of the knowledge axis; `None` while virtual
    /// (sequence == row id and the watermark is the row count). Lives
    /// here, not on [`Store`], because reader snapshots stamp the
    /// buffer's segment from it.
    knowledge: Option<Knowledge>,
    /// Row ids the table has tombstoned (decision #1: ids, never keys),
    /// each mapped to the sequence its kill landed at (0 = unknown,
    /// from a v1 delete log) — what compaction moves into history
    /// segments' kill coordinates.
    tombstones: BTreeMap<u64, u64>,
    /// History segments: superseded row versions a retaining compaction
    /// preserved. Never in a latest-knowledge snapshot; entered only
    /// under `AS OF` — which is why their slots carry no metadata and
    /// fault whole on first use.
    history: Vec<Arc<SegmentSlot>>,
}

/// A diverged table's live knowledge state: where the ingest-sequence
/// watermark stands and where the buffered rows were born.
struct Knowledge {
    /// The sequence the next ordinary append receives.
    next: u64,
    /// The buffer's first row's birth sequence; with `explicit` absent,
    /// buffered row `i` was born at `buffer_base + i`.
    buffer_base: u64,
    /// Per-row births, materialized once a supersession (issue #73)
    /// landed rows in the buffer at a shared coordinate — the one thing
    /// that breaks the buffer's contiguity.
    explicit: Option<Vec<u64>>,
}

impl Shared {
    /// The ingest-sequence watermark; `rows` is the store's row count,
    /// which IS the watermark while virtual.
    fn watermark(&self, rows: u64) -> u64 {
        self.knowledge
            .as_ref()
            .map_or(rows, |knowledge| knowledge.next)
    }

    /// Appends one ordinary row, advancing the knowledge axis with it.
    fn append_row(&mut self, row: &[RowValue<'_>]) -> Result<(), StorageError> {
        self.buffer.append(row)?;
        if let Some(knowledge) = &mut self.knowledge {
            if let Some(explicit) = &mut knowledge.explicit {
                explicit.push(knowledge.next);
            }
            knowledge.next += 1;
        }
        Ok(())
    }

    /// Consumes one knowledge coordinate without a row: a `DELETE`'s
    /// kill (ruled 2026-07-29). Consuming is what makes a delete a
    /// knowledge event of its own — the next append lands *above* it,
    /// so the cut that shows the deletion is a stable one no later
    /// arrival can join. It also diverges a virtual table on the spot:
    /// there is now a coordinate no row carries.
    fn consume_sequence(&mut self, virtual_watermark: u64) {
        let buffered = self.buffer.len() as u64;
        let knowledge = self.knowledge.get_or_insert_with(|| Knowledge {
            next: virtual_watermark,
            buffer_base: virtual_watermark - buffered,
            explicit: None,
        });
        knowledge.next += 1;
        if buffered == 0 {
            // Nothing to describe: the empty buffer simply restarts at
            // the new watermark, as it does after a flush.
            knowledge.buffer_base = knowledge.next;
            knowledge.explicit = None;
        } else if knowledge.explicit.is_none() {
            // The buffer straddles the gap — its rows were born below
            // the consumed coordinate, later ones will be born above —
            // so contiguity is gone and births go per row.
            let base = knowledge.buffer_base;
            knowledge.explicit = Some((0..buffered).map(|offset| base + offset).collect());
        }
    }

    /// Appends one row born at `shared_sequence` — a supersession's
    /// replacement (live or replayed). Diverges a virtual table (the
    /// shared coordinate is what breaks sequence == row id) and
    /// materializes the buffer's explicit births.
    fn append_superseding(
        &mut self,
        row: &[RowValue<'_>],
        shared_sequence: u64,
        virtual_watermark: u64,
    ) -> Result<(), StorageError> {
        let buffered = self.buffer.len() as u64;
        let knowledge = self.knowledge.get_or_insert_with(|| Knowledge {
            next: virtual_watermark,
            buffer_base: virtual_watermark - buffered,
            explicit: None,
        });
        if knowledge.explicit.is_none() {
            let base = knowledge.buffer_base;
            knowledge.explicit = Some((0..buffered).map(|offset| base + offset).collect());
        }
        self.buffer.append(row)?;
        let knowledge = self.knowledge.as_mut().expect("diverged above");
        knowledge
            .explicit
            .as_mut()
            .expect("materialized above")
            .push(shared_sequence);
        knowledge.next = knowledge.next.max(shared_sequence + 1);
        Ok(())
    }

    /// The buffer's sequence stamp for snapshots and flushes; `None`
    /// while virtual.
    fn buffer_sequence(&self) -> Option<SequenceInfo> {
        self.knowledge
            .as_ref()
            .map(|knowledge| match &knowledge.explicit {
                Some(explicit) => SequenceInfo::Explicit(explicit.clone()),
                None => SequenceInfo::Contiguous {
                    base: knowledge.buffer_base,
                },
            })
    }
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
    pub fn snapshot(&self) -> Result<Vec<SegmentHandle>, StorageError> {
        snapshot_of(&lock(&self.shared))
    }

    /// As [`Store::knowledge_snapshot`], from any thread.
    pub fn knowledge_snapshot(&self) -> Result<KnowledgeSnapshot, StorageError> {
        knowledge_snapshot_of(&lock(&self.shared))
    }
}

/// A point-in-time capture of everything an `AS OF` read needs, taken
/// under one lock so the three parts can never be torn against each
/// other: the latest-knowledge handles, the history slots, and the
/// pending (uncompacted) tombstones' kill stamps.
pub struct KnowledgeSnapshot {
    /// The latest-knowledge handles, exactly as [`Store::snapshot`].
    latest: Vec<SegmentHandle>,
    /// History slots (see [`Store::history`]); faulted by `as_of`.
    history: Vec<Arc<SegmentSlot>>,
    /// Pending tombstones: row id → the sequence its kill landed at.
    stamps: BTreeMap<u64, u64>,
}

impl KnowledgeSnapshot {
    /// The latest-knowledge handles — what a plain (no `AS OF`) query
    /// runs over.
    pub fn latest(&self) -> &[SegmentHandle] {
        &self.latest
    }

    /// The table as it was known at ingest-sequence `cut`: rows born at
    /// or before the cut and not superseded by it. Live segments keep
    /// rows whose pending tombstone (if any) landed after the cut;
    /// history rows return where their kill came later. The result runs
    /// through the ordinary executor — the knowledge mask is just a
    /// live mask.
    ///
    /// This **decodes** every segment it consults (masks come from
    /// per-row birth and kill sequences): `AS OF` pays the fault-in for
    /// the axis it walks, which is the residency design's intended
    /// trade — history is cold by definition.
    pub fn as_of(&self, cut: u64) -> Result<Vec<SegmentHandle>, StorageError> {
        let mut out = Vec::with_capacity(self.latest.len() + self.history.len());
        for handle in &self.latest {
            let segment = handle.view()?.segment;
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
            let live = (!all_live).then(|| Bitmap::from_bools(mask));
            out.push(SegmentHandle::resident(segment, live));
        }
        for slot in &self.history {
            let segment = slot.segment()?;
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
            let live = Some(Bitmap::from_bools(mask));
            out.push(SegmentHandle::resident(segment, live));
        }
        Ok(out)
    }

    /// Calls `touch` with the ordering-key value of every row **born or
    /// killed** by an ingest-sequence coordinate at or after `since` —
    /// the watermark below which the caller's derived state is
    /// complete (the first coordinate it does *not* cover,
    /// [`Store::next_sequence`]'s convention), so the filter is
    /// inclusive. This is the derivation a maintained view's refresh
    /// (#83) runs to learn which buckets need re-folding; the dirty
    /// list is derivable state, which is why a view needs no durable
    /// bookkeeping beyond its watermark. ("Stamp" in this function's
    /// comments means a KILL's coordinate — the sense this module
    /// already used — never the view-side watermark.)
    ///
    /// Cost is proportional to what changed, not to the table: a live
    /// segment whose every birth is at or below `since`
    /// ([`SegmentHandle::sequence_end`], answered from metadata) is
    /// skipped without touching its file, kills are walked from the
    /// pending-tombstone map (one faulted segment per killed row's
    /// home), and only **history** segments are scanned
    /// unconditionally — their kill coordinates live in the segment,
    /// not the metadata, and history only exists after a correction has
    /// already been compacted. (An additive manifest field for a
    /// history segment's largest kill would remove that scan; worth it
    /// only if refresh-over-corrected-history ever measures hot.)
    pub fn touched_ordering_keys(
        &self,
        since: u64,
        mut touch: impl FnMut(i64),
    ) -> Result<(), StorageError> {
        self.touched_walk(since, |segment, row| {
            touch(ordering_value(segment, row));
        })
    }

    /// As [`KnowledgeSnapshot::touched_ordering_keys`], additionally
    /// yielding the
    /// touched row's value in `key_column` — the seam a maintained
    /// **join** view's refresh needs (#83 tranche 3): a quote-side
    /// correction's blast radius is a fact-key range *per symbol*, and
    /// a symbol-blind endpoint is unsound, so the walk must say whose
    /// row changed. `key_column` must be a key column; a null key
    /// yields `None`, exactly as it matches nothing in the join.
    pub fn touched_rows(
        &self,
        since: u64,
        key_column: usize,
        mut touch: impl FnMut(i64, Option<&str>),
    ) -> Result<(), StorageError> {
        // The snapshot carries no schema of its own; the column is
        // validated against the first touched segment's (all segments
        // share the store's), and a misuse is a loud error, not a
        // panic.
        let mut misuse: Option<StorageError> = None;
        self.touched_walk(since, |segment, row| {
            if misuse.is_some() {
                return;
            }
            match segment.batch().columns().get(key_column) {
                Some(Column::Key(keys)) => touch(ordering_value(segment, row), keys.value_at(row)),
                _ => {
                    misuse = Some(StorageError::TypeMismatch {
                        column: segment
                            .batch()
                            .schema()
                            .fields()
                            .get(key_column)
                            .map(|field| field.name().to_owned())
                            .unwrap_or_else(|| format!("column {key_column}")),
                        expected: arrow_lite::ColumnType::Key,
                    })
                }
            }
        })?;
        match misuse {
            None => Ok(()),
            Some(error) => Err(error),
        }
    }

    /// The shared walk behind both touched derivations: every row born
    /// or killed at or after `since`, yielded as `(segment, row)`. One
    /// body, so the two public forms cannot drift.
    fn touched_walk(
        &self,
        since: u64,
        mut touch: impl FnMut(&Segment, usize),
    ) -> Result<(), StorageError> {
        // Births at or after the stamp, in live segments.
        for handle in &self.latest {
            if handle.sequence_end() <= since {
                // One past the largest birth is at most the stamp:
                // every birth here is covered. Metadata only, no fault.
                continue;
            }
            let segment = handle.view()?.segment;
            for row in 0..segment.batch().num_rows() {
                if segment.sequence_at(row) >= since {
                    touch(&segment, row);
                }
            }
        }
        // Kills after the stamp, still pending (uncompacted): the map
        // names the row id; its home segment names the value.
        for (&row_id, &kill) in &self.stamps {
            // kill == 0 is the v1 sentinel, "killed at an unknown
            // coordinate". `as_of` reads it conservatively (never
            // visible); this walk must be conservative in the OTHER
            // direction — an unknown kill may postdate any watermark,
            // so it always touches. (`kill >= since` covers it only
            // while since is 0.)
            if kill != 0 && kill < since {
                continue;
            }
            let home = self.latest.iter().find(|handle| {
                handle.base_row_id() <= row_id
                    && row_id < handle.base_row_id() + handle.rows() as u64
            });
            if let Some(handle) = home {
                let segment = handle.view()?.segment;
                let row = (row_id - segment.base_row_id()) as usize;
                touch(&segment, row);
            }
        }
        // Kills (and births — a row both born and killed since the
        // stamp reports the same value) already compacted into history.
        for slot in &self.history {
            let segment = slot.segment()?;
            let kills = segment.superseded();
            for row in 0..segment.batch().num_rows() {
                let kill = kills.map_or(0, |kills| kills[row]);
                // kill == 0: the v1 killed-at-unknown sentinel —
                // always touched, as in the pending walk above.
                if kill == 0 || kill >= since || segment.sequence_at(row) >= since {
                    touch(&segment, row);
                }
            }
        }
        Ok(())
    }
}

/// A row's ordering-key value — validated `i64 NOT NULL` at
/// construction, so there is no other case.
fn ordering_value(segment: &Segment, row: usize) -> i64 {
    let Column::Numeric(NumericData::I64(column)) =
        &segment.batch().columns()[segment.ordering_key()]
    else {
        unreachable!("the ordering key is validated as i64 at construction")
    };
    column.values().as_slice()[row]
}

/// The [`KnowledgeSnapshot`] algorithm over locked state.
fn knowledge_snapshot_of(shared: &Shared) -> Result<KnowledgeSnapshot, StorageError> {
    Ok(KnowledgeSnapshot {
        latest: snapshot_of(shared)?,
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
    /// The residency budget: decoded segments retained in memory, in
    /// bytes (2026-07-30 ruling, option b). `None` — the interim
    /// default, decision #87 — retains everything ever touched, which
    /// is the pre-residency behavior exactly. The budget bounds
    /// *retention*, not correctness: segments a snapshot or running
    /// query still holds are never evicted, so peak memory is the
    /// budget plus the largest concurrent working set.
    pub cache_bytes: Option<u64>,
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
fn snapshot_of(shared: &Shared) -> Result<Vec<SegmentHandle>, StorageError> {
    let live_mask = |base: u64, end: u64| {
        if shared.tombstones.range(base..end).next().is_none() {
            None
        } else {
            Some(Bitmap::from_bools(
                (base..end).map(|id| !shared.tombstones.contains_key(&id)),
            ))
        }
    };
    let mut out: Vec<SegmentHandle> = shared
        .segments
        .iter()
        .map(|slot| {
            let base = slot.meta().base_row_id;
            let end = base + slot.meta().rows as u64;
            SegmentHandle {
                slot: Arc::clone(slot),
                live: live_mask(base, end),
            }
        })
        .collect();
    if !shared.buffer.is_empty() {
        let mut segment = shared.buffer.snapshot_at(shared.buffer_base)?;
        // Post-divergence, the buffer's rows carry sequences from the
        // watermark, not their row ids; stamp the snapshot so readers
        // (and flush, which reuses this path's shape) see them.
        if let Some(sequence) = shared.buffer_sequence() {
            segment = segment.with_sequence(sequence);
        }
        let base = segment.base_row_id();
        let end = base + segment.batch().num_rows() as u64;
        let live = live_mask(base, end);
        out.push(SegmentHandle::resident(Arc::new(segment), live));
    }
    Ok(out)
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
        let slot_shared = Arc::new(SlotShared {
            schema: schema.clone(),
            ordering_key,
            backend: None,
            cache: SegmentCache::new(None),
        });
        Ok(Store {
            read_only: false,
            slot_shared,
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
                knowledge: None,
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
        Store::persistent_inner(backend, schema, ordering_key, options, Some(manifest))
    }

    /// Opens an existing persistent store **read-only** (F4): the
    /// cross-process reader half of the beta shape — one feed-writer
    /// process, any number of console or binding readers over the same
    /// directory. Takes no directory lock and refuses every mutation.
    ///
    /// **What a reader sees: the durable prefix, consistently.** The
    /// view is built from the manifest, the flushed segments, and the
    /// delete logs — never the writer's WAL or write buffer — so it
    /// lags the writer by at most the flush boundary, and it is
    /// old-or-new per mutation, never a torn middle: a supersession's
    /// delete log whose replacement rows have not been flushed yet is
    /// skipped *whole* (kill and replacements both invisible — the
    /// pre-mutation state), which is decidable from the log itself:
    /// replacements born at coordinate `c` are flushed exactly when
    /// some flushed row carries a sequence past `c`, because rows
    /// reach segments in row-id order. Plain deletes need no such
    /// test — their victims are flushed before the log commits.
    ///
    /// **Compaction races retry.** The writer's compaction publishes a
    /// whole new generation and then removes the old one's objects; a
    /// reader that catches the middle sees a named file vanish, and
    /// the open retries from the manifest (bounded), which is atomic.
    ///
    /// Refresh by [`Store::refresh`], which re-reads the metadata and
    /// keeps every already-decoded segment it can (same names, same
    /// schema — the immutable files guarantee the bytes).
    pub fn open_read_only(backend: Arc<dyn StorageBackend>) -> Result<Store, StorageError> {
        Store::open_read_only_with_cache(backend, None)
    }

    /// As [`Store::open_read_only`], with a residency budget (see
    /// [`StoreOptions::cache_bytes`]) — the reader-side knob: a console
    /// riding alongside a feed writer bounds what it retains decoded.
    pub fn open_read_only_with_cache(
        backend: Arc<dyn StorageBackend>,
        cache_bytes: Option<u64>,
    ) -> Result<Store, StorageError> {
        Store::open_read_only_inner(backend, cache_bytes, None)
    }

    fn open_read_only_inner(
        backend: Arc<dyn StorageBackend>,
        cache_bytes: Option<u64>,
        previous: Option<&RefreshCarry>,
    ) -> Result<Store, StorageError> {
        let mut missing = String::new();
        for _ in 0..8 {
            match Store::load_read_only(backend.clone(), cache_bytes, previous) {
                Ok(store) => return Ok(store),
                // A compaction moved the generation mid-scan: a named
                // object vanished. The next manifest read names the
                // complete new generation — go again.
                Err(StorageError::Io(IoError::NotFound(name))) => missing = name,
                Err(error) => return Err(error),
            }
        }
        // Eight straight misses on a manifest-named object is either a
        // writer compacting continuously (rare, resolved by retrying)
        // or an object the store has actually lost — say which object,
        // so the second case is diagnosable.
        Err(StorageError::Misuse(format!(
            "read-only open kept racing generation changes ('{missing}' \
             stayed unreadable); retry — if this persists with an idle \
             writer, the store has lost that object"
        )))
    }

    /// Re-reads the durable state (read-only stores only): new flushed
    /// segments, new delete logs, a new generation after compaction.
    /// Snapshots minted before a refresh stay valid — their segments
    /// are immutable and decoded in memory.
    pub fn refresh(&mut self) -> Result<(), StorageError> {
        if !self.read_only {
            return Err(StorageError::Misuse(
                "refresh is the read-only handle's doorway; the writer sees \
                 its own state"
                    .to_owned(),
            ));
        }
        let backend = self
            .backend
            .clone()
            .expect("read-only stores are persistent");
        // The residency budget survives the refresh, and so does the
        // decoded cache: slots whose names the new manifest still
        // carries are reused whole (the files are immutable, so same
        // name = same bytes), which is what keeps a reader's hot
        // window hot across every refresh.
        let cache_bytes = self.slot_shared.cache.budget;
        let previous = {
            let shared = lock(&self.shared);
            RefreshCarry {
                shared: Arc::clone(&self.slot_shared),
                segments: shared.segments.clone(),
                history: shared.history.clone(),
            }
        };
        *self = Store::open_read_only_inner(backend, cache_bytes, Some(&previous))?;
        Ok(())
    }

    /// Every mutating doorway starts here: a read-only handle mutates
    /// nothing, ever — the writer process owns the directory.
    fn refuse_read_only(&self) -> Result<(), StorageError> {
        if self.read_only {
            return Err(StorageError::Misuse(
                "this handle opened the store read-only; the writer process \
                 owns mutation"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn load_read_only(
        backend: Arc<dyn StorageBackend>,
        cache_bytes: Option<u64>,
        previous: Option<&RefreshCarry>,
    ) -> Result<Store, StorageError> {
        let manifest = decode_manifest(&backend.read(MANIFEST)?)?;
        let schema = manifest.schema.clone();
        let ordering_key = manifest.ordering_key;
        let generation = manifest.generation;
        let mut store = Store::with_segment_rows(schema, ordering_key, DEFAULT_SEGMENT_ROWS)?;
        store.read_only = true;
        store.manifest_sections = manifest.sections;
        // A refresh keeps the previous open's slot context whole —
        // same cache instance, so accounting and eviction stay
        // continuous and carried slots stay registered — as long as
        // the manifest still describes the same table.
        let slot_shared = match previous {
            Some(carry)
                if carry.shared.schema == store.schema
                    && carry.shared.ordering_key == ordering_key =>
            {
                Arc::clone(&carry.shared)
            }
            _ => Arc::new(SlotShared {
                schema: store.schema.clone(),
                ordering_key,
                backend: Some(backend.clone()),
                cache: SegmentCache::new(cache_bytes),
            }),
        };
        store.slot_shared = Arc::clone(&slot_shared);
        let mut slots: Vec<Arc<SegmentSlot>> = Vec::new();
        // Delete logs are held back until the segment watermark is
        // known: the supersession-visibility rule needs it. They are
        // always discovered by listing — logs commit between manifest
        // writes, so the manifest cannot name them.
        let mut logs: Vec<DeleteLog> = Vec::new();
        let recorded_layout = !store.manifest_sections.segments.is_empty();
        let listing = backend.list()?;
        for name in &listing {
            if name
                .strip_prefix(&delete_log_prefix(generation))
                .and_then(|rest| rest.strip_suffix(".tlyd"))
                .is_some()
            {
                logs.push(decode_tombstones(&backend.read(name)?)?);
                continue;
            }
            if recorded_layout
                || !(name.starts_with(&segment_prefix(generation)) && name.ends_with(".tlyseg"))
            {
                continue;
            }
            let segment = decode_segment(&backend.read(name)?)?;
            if segment.batch().schema() != &store.schema {
                return Err(StorageError::SchemaMismatch {
                    reason: format!("segment '{name}' was written under a different schema"),
                });
            }
            slots.push(SegmentSlot::resident(
                Arc::new(segment),
                Arc::clone(&slot_shared),
                Some(name.clone()),
            ));
        }
        if recorded_layout {
            // The manifest names the layout: lazy slots, nothing
            // decoded. A named file missing from the listing is a
            // compaction moving the generation mid-read — surfaced as
            // NotFound so the open's bounded retry re-reads the
            // manifest, which is atomic. (Later faults hit the same
            // NotFound if the race lands mid-query; refresh resolves.)
            // A refresh reuses the previous state's slot for any name
            // it still carries — resident stays resident.
            for record in &store.manifest_sections.segments {
                if !listing.contains(&record.name) {
                    return Err(StorageError::Io(IoError::NotFound(record.name.clone())));
                }
                if let Some(kept) = previous.and_then(|carry| carry.slot_named(&record.name)) {
                    slots.push(kept);
                    continue;
                }
                slots.push(SegmentSlot::lazy(record, Arc::clone(&slot_shared)));
            }
        }
        slots.sort_by_key(|slot| slot.meta().base_row_id);
        let mut expected_base = 0u64;
        for slot in &slots {
            if slot.meta().base_row_id != expected_base {
                return Err(StorageError::MissingRows { expected_base });
            }
            expected_base += slot.meta().rows as u64;
        }
        // The flushed watermark, from metadata alone — the test the
        // visibility rule runs against.
        let flushed = slots
            .iter()
            .map(|slot| slot.meta().sequence_end)
            .fold(0, u64::max);
        let mut tombstones = BTreeMap::new();
        for log in &logs {
            // A supersession whose replacements are still in the
            // writer's WAL: applying its kill without them would show
            // the one middle state that loses rows. Skip it whole —
            // the reader sees the pre-mutation state until the flush.
            if log.superseding != 0 && flushed <= log.superseding {
                continue;
            }
            tombstones.extend(log.ids.iter().map(|&id| (id, log.stamped_at)));
        }
        // A tombstone past the durable rows would be a torn write; on
        // the reader it may simply be a race — surface it as one.
        if tombstones.keys().any(|&id| id >= expected_base) {
            return Err(StorageError::Io(IoError::NotFound(
                "a delete log ran ahead of its rows; retrying".to_owned(),
            )));
        }
        let history: Vec<Arc<SegmentSlot>> = store
            .manifest_sections
            .history
            .iter()
            .map(|name| {
                // History files are never rewritten once listed, so a
                // refresh reuses their slots by name unconditionally.
                previous
                    .and_then(|carry| carry.history_named(name))
                    .unwrap_or_else(|| SegmentSlot::history(name.clone(), Arc::clone(&slot_shared)))
            })
            .collect();
        let recorded = store.manifest_sections.next_sequence;
        let killed_at = tombstones.values().copied().max().unwrap_or(0);
        let diverged =
            recorded.is_some() || killed_at > 0 || slots.iter().any(|slot| slot.meta().diverged);
        let watermark = diverged.then(|| {
            let spent = if killed_at > 0 { killed_at + 1 } else { 0 };
            slots
                .iter()
                .map(|slot| slot.meta().sequence_end)
                .fold(recorded.unwrap_or(0).max(spent), u64::max)
        });
        {
            let mut shared = lock(&store.shared);
            shared.segments = slots;
            shared.buffer_base = expected_base;
            shared.knowledge = watermark.map(|next| Knowledge {
                next,
                buffer_base: next,
                explicit: None,
            });
            shared.tombstones = tombstones;
            shared.history = history;
        }
        store.rows = expected_base;
        store.generation = generation;
        store.backend = Some(backend);
        Ok(store)
    }

    /// As [`Store::persistent`], with explicit [`StoreOptions`] — the
    /// segment threshold and the durability level (#43).
    pub fn persistent_with(
        backend: Arc<dyn StorageBackend>,
        schema: Schema,
        ordering_key: usize,
        options: StoreOptions,
    ) -> Result<Store, StorageError> {
        Store::persistent_inner(backend, schema, ordering_key, options, None)
    }

    /// The one persistent-open path; `preloaded` carries the manifest a
    /// caller already decoded ([`Store::open_existing`]) so nothing is
    /// read twice.
    fn persistent_inner(
        backend: Arc<dyn StorageBackend>,
        schema: Schema,
        ordering_key: usize,
        options: StoreOptions,
        preloaded: Option<crate::format::Manifest>,
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
        let stored_manifest = match preloaded {
            Some(manifest) => Some(manifest),
            None => match backend.read(MANIFEST) {
                Ok(bytes) => Some(decode_manifest(&bytes)?),
                Err(IoError::NotFound(_)) => None,
                Err(error) => return Err(error.into()),
            },
        };
        let generation = match stored_manifest {
            Some(manifest) => {
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
                store.manifest_sections = manifest.sections;
                manifest.generation
            }
            None => {
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
        };
        let slot_shared = Arc::new(SlotShared {
            schema: store.schema.clone(),
            ordering_key,
            backend: Some(backend.clone()),
            cache: SegmentCache::new(options.cache_bytes),
        });
        store.slot_shared = Arc::clone(&slot_shared);
        let mut slots: Vec<Arc<SegmentSlot>> = Vec::new();
        let mut tombstones = BTreeMap::new();
        let mut committed_supersessions: BTreeSet<u64> = BTreeSet::new();
        let mut next_sequence = 0u64;
        // Whether the manifest names the segments (tag 1). When it
        // does, it is authoritative: the open builds lazy slots from
        // the records and decodes nothing — the residency design's
        // instant open — and a stray segment file (a crash between a
        // flush's segment write and its manifest write) is invisible,
        // its rows still in the WAL, replayed below. Without the
        // section (an older writer's manifest) the backend scan is the
        // list, as it always was, and the section is earned right after
        // the load so the next open takes the short path.
        let recorded_layout = !store.manifest_sections.segments.is_empty();
        let listing = backend.list()?;
        for name in &listing {
            // Objects from other generations are a crashed compaction's
            // leftovers — invisible here, removed by the next compaction.
            if let Some(sequence) = name
                .strip_prefix(&delete_log_prefix(generation))
                .and_then(|rest| rest.strip_suffix(".tlyd"))
            {
                let sequence: u64 = sequence.parse().map_err(|_| StorageError::SchemaMismatch {
                    reason: format!("delete log '{name}' has a malformed name"),
                })?;
                let log = decode_tombstones(&backend.read(name)?)?;
                tombstones.extend(log.ids.iter().map(|&id| (id, log.stamped_at)));
                if log.superseding != 0 {
                    committed_supersessions.insert(log.superseding);
                }
                next_sequence = next_sequence.max(sequence + 1);
                continue;
            }
            if recorded_layout
                || !(name.starts_with(&segment_prefix(generation)) && name.ends_with(".tlyseg"))
            {
                continue;
            }
            let segment = decode_segment(&backend.read(name)?)?;
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
            slots.push(SegmentSlot::resident(
                Arc::new(segment),
                Arc::clone(&slot_shared),
                Some(name.clone()),
            ));
        }
        if recorded_layout {
            for record in &store.manifest_sections.segments {
                // The writer holds the directory lock, so no compaction
                // can be moving the generation under it: a named file
                // absent from the listing is lost rows, not a race —
                // said now, at open, rather than at first fault.
                if !listing.contains(&record.name) {
                    return Err(StorageError::MissingRows {
                        expected_base: record.base_row_id,
                    });
                }
                slots.push(SegmentSlot::lazy(record, Arc::clone(&slot_shared)));
            }
        }
        slots.sort_by_key(|slot| slot.meta().base_row_id);
        let mut expected_base = 0u64;
        for slot in &slots {
            if slot.meta().base_row_id != expected_base {
                return Err(StorageError::MissingRows { expected_base });
            }
            expected_base += slot.meta().rows as u64;
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
        // next compaction. Slots only: history is read whole under
        // `AS OF`, so nothing is decoded (or schema-checked) until then.
        let history: Vec<Arc<SegmentSlot>> = store
            .manifest_sections
            .history
            .iter()
            .map(|name| SegmentSlot::history(name.clone(), Arc::clone(&slot_shared)))
            .collect();
        // A diverged table's watermark: the manifest records it at each
        // compaction, but flushes — and supersessions, which diverge a
        // table without a compaction — advance sequences without
        // rewriting the manifest. So divergence is detected from the
        // manifest OR from any stored segment carrying sequence data,
        // and the watermark folds the segments' ends over whatever the
        // manifest recorded. Rows the WAL replays below then take
        // sequences from here — the same values they had before the
        // crash, since assignment is deterministic in append order.
        // (History needs no fold: it only changes at compaction, where
        // the manifest catches up.)
        //
        // Delete logs join the fold for the same reason: a kill
        // consumes its coordinate, so a stamp is evidence of a
        // coordinate one past it — and on a table whose only mutation
        // was a delete, the stamp is the *only* evidence, since no row
        // carries the consumed sequence. (A stamp of 0 is a v1 log's
        // "unknown", never a real kill: coordinate 0 can only be spent
        // by a delete on a table with no rows to delete.)
        let recorded = store.manifest_sections.next_sequence;
        let killed_at = tombstones.values().copied().max().unwrap_or(0);
        let diverged =
            recorded.is_some() || killed_at > 0 || slots.iter().any(|slot| slot.meta().diverged);
        let watermark = diverged.then(|| {
            let spent = if killed_at > 0 { killed_at + 1 } else { 0 };
            slots
                .iter()
                .map(|slot| slot.meta().sequence_end)
                .fold(recorded.unwrap_or(0).max(spent), u64::max)
        });
        // A legacy manifest (no tag 1) earns its segment records now —
        // derived from the segments just decoded, named by the same
        // deterministic rule that wrote them — so the next open reads
        // exactly the named files.
        if !recorded_layout && !slots.is_empty() {
            let mut records = Vec::with_capacity(slots.len());
            for slot in &slots {
                let name = slot.name.clone().expect("legacy slots are named");
                let segment = slot.segment()?;
                records.push(crate::format::SegmentRecord::of(name, &segment));
            }
            let mut sections = store.manifest_sections.clone();
            sections.segments = records;
            backend.write(
                MANIFEST,
                &encode_manifest(&store.schema, ordering_key, generation, &sections),
            )?;
            store.manifest_sections = sections;
        }
        {
            let mut shared = lock(&store.shared);
            shared.segments = slots;
            shared.buffer_base = expected_base;
            shared.knowledge = watermark.map(|next| Knowledge {
                next,
                buffer_base: next,
                explicit: None,
            });
            shared.tombstones = tombstones;
            shared.history = history;
        }
        store.rows = expected_base;
        store.delete_log_sequence = next_sequence;
        store.generation = generation;
        store.backend = Some(backend);
        store.replay_wal(&committed_supersessions)?;
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
    /// for them; `AS OF` reads walk them explicitly. **Decodes** every
    /// history segment not already resident (history is lazy — the
    /// residency design).
    pub fn history(&self) -> Result<Vec<Arc<Segment>>, StorageError> {
        let slots = lock(&self.shared).history.clone();
        slots.iter().map(|slot| slot.segment()).collect()
    }

    /// The ingest-sequence watermark: the sequence the next appended
    /// row will receive. Every coordinate the table has spent is `<`
    /// this, so **`AS OF next_sequence() - 1` is the latest state** — in
    /// every mutation shape, since appends, supersessions and kills all
    /// consume exactly one coordinate (the delete-consumes ruling,
    /// 2026-07-29). Equal to [`Store::len`] until the table diverges.
    ///
    /// That form is the idiom rather than `next_sequence()` itself
    /// because it addresses a coordinate that has been *spent*: its
    /// answer is fixed forever, while the watermark is the address the
    /// next arrival will take, so a cut there silently absorbs it.
    pub fn next_sequence(&self) -> u64 {
        lock(&self.shared).watermark(self.rows)
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
    /// Supersession brackets (issue #73) resolve to old-or-new: a
    /// bracket whose delete log committed replays with its shared
    /// coordinate; an uncommitted bracket at the log's tail — the
    /// crashed-mutation window — is dropped whole. Ends with the log in
    /// steady state for the configured level: recreated and synced
    /// under `Group`/`Full`, removed under `Off`.
    fn replay_wal(&mut self, committed: &BTreeSet<u64>) -> Result<(), StorageError> {
        let backend = self.backend.as_ref().expect("replay is a reopen step");
        let (base_row_id, entries) = match backend.read(WAL) {
            // Shorter than one header is the crash window between log
            // creation (which truncates in place) and the header sync:
            // no record was ever synced under this log — records follow
            // the header in the same file — so there is nothing to
            // recover, and treating it as corruption would leave the
            // store permanently unopenable over intact segments.
            Ok(bytes) if bytes.len() < crate::format::WAL_HEADER_LEN => (self.rows, Vec::new()),
            Ok(bytes) => {
                let wal = crate::format::decode_wal(&bytes, self.schema.fields().len())?;
                if wal.generation == self.generation {
                    (wal.base_row_id, wal.entries)
                } else {
                    (self.rows, Vec::new())
                }
            }
            Err(IoError::NotFound(_)) => (self.rows, Vec::new()),
            Err(error) => return Err(error.into()),
        };
        // Resolve brackets into replayable units (old-or-new).
        let mut units: Vec<Replayed> = Vec::new();
        let mut iter = entries.into_iter().peekable();
        while let Some(entry) = iter.next() {
            match entry {
                crate::format::WalEntry::Row(cells) => units.push(Replayed::Row(cells)),
                crate::format::WalEntry::Supersession {
                    sequence,
                    replacements,
                } => {
                    let mut rows = Vec::new();
                    while (rows.len() as u64) < replacements {
                        match iter.next() {
                            Some(crate::format::WalEntry::Row(cells)) => rows.push(cells),
                            // Truncated mid-bracket: the crash window.
                            _ => break,
                        }
                    }
                    let complete = rows.len() as u64 == replacements;
                    let at_tail = iter.peek().is_none();
                    // A complete bracket replays if its delete log
                    // committed, or if records follow it (the process
                    // outlived a failed commit and the rows were
                    // visible). A complete uncommitted bracket at the
                    // tail is a crashed mutation — dropped whole, so
                    // the originals stand: old, never torn.
                    if complete && (committed.contains(&sequence) || !at_tail) {
                        units.push(Replayed::Supersession { sequence, rows });
                    }
                    if !complete {
                        break; // nothing after a torn bracket replays
                    }
                }
            }
        }
        // Skip the prefix already covered by flushed segments (a crash
        // between segment publish and WAL truncate leaves one). Flushes
        // take the whole buffer, so the boundary never splits a bracket.
        let mut skip =
            usize::try_from(self.rows.saturating_sub(base_row_id)).expect("row counts fit usize");
        let mut replay: Vec<Replayed> = Vec::new();
        for unit in units {
            let rows = unit.row_count();
            if skip == 0 {
                replay.push(unit);
            } else if rows <= skip {
                skip -= rows;
            } else {
                // Unreachable by construction; drop the split unit
                // rather than replay half a mutation.
                skip = 0;
            }
        }
        if self.wal_sync == WalSync::Off {
            // Recovered rows re-enter the buffer under the flush-boundary
            // contract; the log itself goes away.
            for unit in replay {
                self.apply_replayed(unit)?;
            }
            let backend = self.backend.as_ref().expect("replay is a reopen step");
            match backend.remove(WAL) {
                Ok(()) | Err(IoError::NotFound(_)) => {}
                Err(error) => return Err(error.into()),
            }
            return Ok(());
        }
        // Assemble the replacement log whole — header plus every
        // recovered record, brackets preserved so a second crash still
        // resolves old-or-new — and publish it atomically *over* the
        // old one. The old log stays the durable copy until the
        // publishing rename commits, so a crash at any instant of
        // recovery leaves exactly one complete log to recover from;
        // truncate-then-rewrite would destroy the only copy first.
        let mut bytes = crate::format::encode_wal_header(self.generation, self.rows);
        for unit in &replay {
            match unit {
                Replayed::Row(cells) => {
                    bytes.extend_from_slice(&crate::format::encode_wal_record(&owned_cells(cells)));
                }
                Replayed::Supersession { sequence, rows } => {
                    bytes.extend_from_slice(&crate::format::encode_wal_supersession(
                        *sequence,
                        rows.len() as u64,
                    ));
                    for cells in rows {
                        bytes.extend_from_slice(&crate::format::encode_wal_record(&owned_cells(
                            cells,
                        )));
                    }
                }
            }
        }
        let backend = self.backend.as_ref().expect("replay is a reopen step");
        backend.write(WAL, &bytes)?;
        for unit in replay {
            self.apply_replayed(unit)?;
        }
        let backend = self.backend.as_ref().expect("replay is a reopen step");
        self.wal = Some(backend.open_log(WAL)?);
        self.last_wal_sync = std::time::Instant::now();
        Ok(())
    }

    /// Applies one recovered WAL unit to the buffer and row counter.
    fn apply_replayed(&mut self, unit: Replayed) -> Result<(), StorageError> {
        match unit {
            Replayed::Row(cells) => {
                let row = owned_cells(&cells);
                lock(&self.shared).append_row(&row)?;
                self.rows += 1;
            }
            Replayed::Supersession { sequence, rows } => {
                for cells in &rows {
                    let row = owned_cells(cells);
                    let virtual_watermark = self.rows;
                    lock(&self.shared).append_superseding(&row, sequence, virtual_watermark)?;
                    self.rows += 1;
                }
            }
        }
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
        self.refuse_read_only()?;
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
            shared.append_row(row)?;
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
        self.refuse_read_only()?;
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
            match shared.buffer_sequence() {
                // Diverged: the frozen segment records where in the
                // sequence space its rows were born.
                Some(sequence) => segment.with_sequence(sequence),
                None => segment,
            }
        };
        let name = self
            .backend
            .as_ref()
            .map(|_| segment_name(self.generation, segment.base_row_id()));
        if let (Some(backend), Some(name)) = (&self.backend, &name) {
            backend.write(name, &encode_segment(&segment))?;
            // The manifest names the segment (tag 1) — written after the
            // segment so a crash between the two leaves an orphan file
            // the manifest never adopted (its rows still in the WAL),
            // and before the WAL reset so the log is only truncated once
            // the layout that covers it is committed. Adopted in memory
            // only after the write succeeds, like compaction's commit.
            let mut sections = self.manifest_sections.clone();
            sections
                .segments
                .push(crate::format::SegmentRecord::of(name.clone(), &segment));
            backend.write(
                MANIFEST,
                &encode_manifest(&self.schema, self.ordering_key, self.generation, &sections),
            )?;
            self.manifest_sections = sections;
        }
        // Built before the lock so adoption below cannot fail partway.
        // The fresh segment stays resident (the writer just built it)
        // but is evictable: named, it can always fault back in.
        let slot = SegmentSlot::resident(Arc::new(segment), self.slot_shared.clone(), name);
        let fresh = WriteBuffer::new(self.schema.clone(), self.ordering_key)?;
        {
            let mut shared = lock(&self.shared);
            shared.segments.push(slot);
            shared.buffer = fresh;
            shared.buffer_base = self.rows;
            // The empty buffer restarts contiguous at the watermark.
            if let Some(knowledge) = &mut shared.knowledge {
                knowledge.buffer_base = knowledge.next;
                knowledge.explicit = None;
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
    ///
    /// A delete **consumes a knowledge coordinate** (ruled 2026-07-29):
    /// the kill is stamped at the current watermark and the watermark
    /// then advances, so no later append can share the deletion's
    /// coordinate. That is what makes the deletion addressable — `AS OF`
    /// the stamp is the table with those rows gone, permanently, rather
    /// than a cut whose meaning drifts as ingest continues. The price is
    /// that the first delete **diverges** the table (a coordinate no row
    /// carries is precisely `sequence != row id`), and that on a
    /// persistent store the buffer is flushed first: with the WAL
    /// truncated at the deletion, every replayed row is one that arrived
    /// after it, so recovery cannot renumber rows across the gap.
    pub fn tombstone(&mut self, ids: &[u64]) -> Result<u64, StorageError> {
        self.refuse_read_only()?;
        if let Some(&bad) = ids.iter().find(|&&id| id >= self.rows) {
            return Err(StorageError::TombstoneOutOfRange { id: bad });
        }
        let newly: BTreeSet<u64> = {
            let shared = lock(&self.shared);
            ids.iter()
                .copied()
                .filter(|id| !shared.tombstones.contains_key(id))
                .collect()
        };
        if newly.is_empty() {
            return Ok(0);
        }
        // Two rules meet in this flush. A delete log must never name a
        // row that is not yet durable: an id in the write buffer is
        // in-memory only, and a crash after the (synced) log would apply
        // a delete against a row that never reached disk, leaving reopen
        // with a tombstone for an id it then reissues (silent
        // shadow-kill of future rows). And because the kill consumes a
        // coordinate, everything born *below* it must be out of the WAL
        // before it lands — replay assigns buffered rows their
        // sequences positionally from the watermark, which after
        // recovery already counts the gap. Flushing unconditionally
        // satisfies both, and costs nothing when the buffer is empty.
        if self.backend.is_some() {
            self.flush()?;
        }
        // The kill's coordinate: the current watermark, consumed below
        // once the log that records it has committed.
        let stamp = lock(&self.shared).watermark(self.rows);
        // A delete log must also never commit ahead of rows that
        // *supersede* its victims — recovering the kill without the
        // replacement is the one middle state that loses data forever.
        // The flush above discharges that rule too: every row the store
        // holds is segment-durable before the log is written, so there
        // is no unsynced tail left to lose. ([`Store::supersede`] keeps
        // its own WAL bracket; it does not come through here.)
        if let Some(backend) = &self.backend {
            backend.write(
                &delete_log_name(self.generation, self.delete_log_sequence),
                &encode_tombstones(&newly, stamp, 0),
            )?;
            self.delete_log_sequence += 1;
        }
        let count = newly.len() as u64;
        {
            let mut shared = lock(&self.shared);
            shared
                .tombstones
                .extend(newly.into_iter().map(|id| (id, stamp)));
            // Committed: the coordinate is spent, and the table has
            // diverged if it had not already.
            shared.consume_sequence(self.rows);
        }
        Ok(count)
    }

    /// Supersedes rows as **one knowledge event** (issue #73): appends
    /// `replacements` and tombstones `victims`, all at a single ingest
    /// sequence — the mutation's coordinate, consumed exactly once.
    /// `AS OF` that coordinate sees every replacement and no victim;
    /// the cut before it sees the originals untouched — old-or-new on
    /// the knowledge axis, with no cut in between. The same property
    /// holds across a crash: the WAL brackets the replacements, and
    /// the delete log — written last, carrying the coordinate — is the
    /// commit record; replay finding the bracket without it drops the
    /// replacements whole. (Under [`WalSync::Off`] the flush boundary
    /// remains the durability contract: a crash between the mutation's
    /// flush and its delete log can leave both versions live, exactly
    /// as that mode already trades.)
    ///
    /// The shared coordinate is precisely `sequence != row id`, so a
    /// supersession diverges a virtual table on the spot.
    ///
    /// At least one victim is required. The commit record is a delete
    /// log carrying the coordinate, and that field spells "no
    /// supersession" as `0` — so a mutation whose coordinate genuinely
    /// *is* 0 (only reachable superseding nothing on an empty table)
    /// would write evidence indistinguishable from a plain delete, and
    /// replay would drop its acknowledged replacements. Rather than
    /// leave that hole open, the shape is refused: a supersession with
    /// no victim is an append, and [`Store::append`] is that operation.
    /// Making the field a presence flag instead of a magic value is a
    /// format revision, deferred until insert-as-supersession is
    /// actually wanted.
    pub fn supersede(
        &mut self,
        replacements: &[Vec<RowValue<'_>>],
        victims: &[u64],
    ) -> Result<u64, StorageError> {
        self.refuse_read_only()?;
        // Validate everything up front: a refused mutation changes
        // nothing anywhere.
        if victims.is_empty() {
            return Err(StorageError::Misuse(
                "supersede needs at least one victim row (a supersession with \
                 nothing to supersede is an append — use append)"
                    .to_owned(),
            ));
        }
        {
            let shared = lock(&self.shared);
            for row in replacements {
                shared.buffer.validate(row)?;
            }
        }
        if let Some(&bad) = victims.iter().find(|&&id| id >= self.rows) {
            return Err(StorageError::TombstoneOutOfRange { id: bad });
        }
        let (newly, buffer_base) = {
            let shared = lock(&self.shared);
            let newly: BTreeSet<u64> = victims
                .iter()
                .copied()
                .filter(|id| !shared.tombstones.contains_key(id))
                .collect();
            (newly, shared.buffer_base)
        };
        if newly.is_empty() && replacements.is_empty() {
            return Ok(0);
        }
        // Buffered victims must be segment-durable before the bracket
        // opens: any later flush would truncate the bracket out of the
        // WAL mid-mutation, which is also why the threshold flush is
        // deferred to the very end.
        if self.backend.is_some() && newly.iter().any(|&id| id >= buffer_base) {
            self.flush()?;
        }
        // The mutation's coordinate (the pre-flush above never moves it).
        let sequence = lock(&self.shared).watermark(self.rows);
        if let Some(wal) = self.wal.as_mut() {
            wal.append(&crate::format::encode_wal_supersession(
                sequence,
                replacements.len() as u64,
            ))?;
        }
        for row in replacements {
            self.wal_append(row)?;
            let virtual_watermark = self.rows;
            lock(&self.shared).append_superseding(row, sequence, virtual_watermark)?;
            self.rows += 1;
        }
        // Superseding rows must be durable before the delete log
        // commits — the same rule [`Store::tombstone`] enforces: sync
        // the WAL, or without one, flush.
        if self.backend.is_some() {
            if let Some(wal) = self.wal.as_mut() {
                wal.sync().map_err(StorageError::from)?;
                self.last_wal_sync = std::time::Instant::now();
            } else if !lock(&self.shared).buffer.is_empty() {
                self.flush()?;
            }
        }
        // The commit record: the delete log carrying the bracket's
        // coordinate — written even when every victim was already dead,
        // because the bracket's replacements need the evidence.
        if let Some(backend) = &self.backend {
            backend.write(
                &delete_log_name(self.generation, self.delete_log_sequence),
                &encode_tombstones(&newly, sequence, sequence),
            )?;
            self.delete_log_sequence += 1;
        }
        let count = newly.len() as u64;
        lock(&self.shared)
            .tombstones
            .extend(newly.into_iter().map(|id| (id, sequence)));
        // The threshold flush deferred while the bracket was open.
        if lock(&self.shared).buffer.len() >= self.segment_rows {
            self.flush()?;
        }
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
        self.refuse_read_only()?;
        // The ingest-sequence watermark as recorded so far, plus the
        // kill stamps and prior history, captured before the merge: a
        // diverged table's compaction must carry all three through —
        // row ids reassign below, knowledge coordinates never do.
        let (recorded_watermark, stamps, old_history) = {
            let shared = lock(&self.shared);
            (
                shared.knowledge.as_ref().map(|knowledge| knowledge.next),
                shared.tombstones.clone(),
                shared.history.clone(),
            )
        };
        // Collect every row's (ordering value, row id, location),
        // buffer included via an ephemeral snapshot — live rows headed
        // for the new generation, dead rows for history. The merge
        // reads every row, so every segment is materialized here: a
        // compaction's working set is the whole table by construction
        // (the 2× peak, decision #82 — the residency budget is advisory
        // and this is its documented overshoot case).
        let views = self
            .snapshot()?
            .iter()
            .map(SegmentHandle::view)
            .collect::<Result<Vec<SegmentView>, StorageError>>()?;
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
            // The new generation's segment records, in base order — the
            // manifest is the authoritative segment list (tag 1), so the
            // commit below publishes layout and metadata in one write.
            sections.segments = new_segments
                .iter()
                .map(|segment| {
                    crate::format::SegmentRecord::of(
                        segment_name(next, segment.base_row_id()),
                        segment,
                    )
                })
                .collect();
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
            // The merged segments enter as resident slots (named under
            // the new generation, so they can fault back in if evicted);
            // new history segments likewise, under their `hist-` names.
            let new_slots: Vec<Arc<SegmentSlot>> = new_segments
                .into_iter()
                .map(|segment| {
                    let name = self
                        .backend
                        .as_ref()
                        .map(|_| segment_name(self.generation, segment.base_row_id()));
                    SegmentSlot::resident(Arc::new(segment), self.slot_shared.clone(), name)
                })
                .collect();
            let new_history_slots: Vec<Arc<SegmentSlot>> = new_history
                .into_iter()
                .zip(&new_history_names)
                .map(|(segment, name)| {
                    let name = self.backend.as_ref().map(|_| name.clone());
                    SegmentSlot::resident(Arc::new(segment), self.slot_shared.clone(), name)
                })
                .collect();
            let mut shared = lock(&self.shared);
            shared.segments = new_slots;
            shared.buffer = fresh_buffer;
            shared.buffer_base = base;
            shared.knowledge = watermark.map(|next| Knowledge {
                next,
                buffer_base: next,
                explicit: None,
            });
            shared.tombstones.clear();
            shared.history = old_history.into_iter().chain(new_history_slots).collect();
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
        lock(&self.shared).knowledge = Some(Knowledge {
            next: watermark,
            buffer_base: watermark,
            explicit: None,
        });
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

    /// A point-in-time view: one [`SegmentHandle`] per frozen segment
    /// plus (if the buffer holds rows) one frozen from a copy of it,
    /// each carrying the live mask its tombstones impose. Untombstoned
    /// segments come back mask-free — the zero-copy common case.
    /// Appends and tombstones after the call don't affect the returned
    /// handles. Metadata (row spans, zone maps, ordering) answers with
    /// no I/O; [`SegmentHandle::view`] faults the decoded segment in —
    /// the residency design's query seam.
    pub fn snapshot(&self) -> Result<Vec<SegmentHandle>, StorageError> {
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

    fn materialized(store: &Store) -> Vec<SegmentView> {
        store
            .snapshot()
            .unwrap()
            .iter()
            .map(|handle| handle.view().unwrap())
            .collect()
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
        let segments = materialized(&store);
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
        let segments = materialized(&store);
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
        let before = materialized(&store);
        append_n(&mut store, 5..9);
        // The old snapshot still sees exactly its five rows...
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].segment.batch().num_rows(), 5);
        let Column::Numeric(NumericData::I64(ts)) = &before[0].segment.batch().columns()[0] else {
            panic!("ts type")
        };
        assert_eq!(ts.values().as_slice(), &[0, 1, 2, 3, 4]);
        // ...and a fresh one sees all nine.
        let after = materialized(&store);
        assert_eq!(after[0].segment.batch().num_rows(), 9);
    }

    #[test]
    fn snapshot_of_live_buffer_shares_row_data() {
        // The buffer snapshot is copy-on-write: until the next append,
        // the segment and the buffer share the same numeric allocation.
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..4);
        let first = materialized(&store);
        let second = materialized(&store);
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
        assert_eq!(materialized(&store).len(), 1);
        // Flushing an empty buffer is a no-op, not an empty segment.
        store.flush().unwrap();
        assert_eq!(store.segment_count(), 1);
    }

    #[test]
    fn ordering_bounds_expose_cross_segment_order() {
        let mut store = Store::with_segment_rows(schema(), 0, 3).unwrap();
        append_n(&mut store, 0..9);
        let segments = materialized(&store);
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
        assert_eq!(materialized(&store)[0].segment.batch().num_rows(), 2);
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
        let views = materialized(&store);
        assert_eq!(views[0].segment.sequence_info(), &SequenceInfo::RowIds);
        assert_eq!(views[0].segment.sequence_at(1), 1);
        store.flush().unwrap();
        store.diverge(100).unwrap();
        append_n(&mut store, 2..4);
        let views = materialized(&store);
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
        let views = materialized(&store);
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
        // Kill ts=1 (row id 1, sequence 11). The kill consumes the
        // watermark, 13 — a coordinate no row will ever carry.
        store.tombstone(&[1]).unwrap();
        assert_eq!(store.next_sequence(), 14);
        store.compact().unwrap();
        let views = materialized(&store);
        // Merge order is ts 3, 5 — sequences 12, 10, explicit because
        // the merge follows the ordering key, not ingest.
        assert_eq!(
            views[0].segment.sequence_info(),
            &SequenceInfo::Explicit(vec![12, 10])
        );
        assert_eq!(views[0].segment.base_row_id(), 0);
        // The next append's sequence follows the dead row's *and* the
        // kill's, not the compacted row count: 14, never 12 again.
        store
            .append(&[RowValue::I64(9), RowValue::Key("A"), RowValue::F64(0.0)])
            .unwrap();
        let views = materialized(&store);
        assert_eq!(
            views[1].segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 14 }
        );
    }

    #[test]
    fn a_delete_consumes_a_coordinate_mid_buffer() {
        // The awkward shape for an in-memory store: the kill lands
        // while the rows it splits are still in the write buffer, so
        // the buffer straddles the gap — births below it, later
        // arrivals above — and contiguity is gone.
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..3);
        assert_eq!(store.next_sequence(), 3);
        store.tombstone(&[1]).unwrap();
        assert_eq!(store.next_sequence(), 4, "the kill spent coordinate 3");
        append_n(&mut store, 3..4);
        assert_eq!(store.next_sequence(), 5);
        let views = materialized(&store);
        assert_eq!(
            views[0].segment.sequence_info(),
            &SequenceInfo::Explicit(vec![0, 1, 2, 4]),
            "the arrival is born above the kill, not beside it"
        );
        // And the cut at the kill is the deletion, with nothing else in
        // it — the property the whole ruling was about.
        let knowledge = store.knowledge_snapshot().unwrap();
        let live_at = |cut: u64| -> usize {
            knowledge
                .as_of(cut)
                .unwrap()
                .iter()
                .map(crate::SegmentHandle::live_rows)
                .sum()
        };
        assert_eq!(live_at(2), 3);
        assert_eq!(live_at(3), 2);
        assert_eq!(live_at(4), 3);
    }

    #[test]
    fn a_supersession_is_one_knowledge_event() {
        let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
        append_n(&mut store, 0..3);
        let replacement = vec![vec![
            RowValue::I64(1),
            RowValue::Key("A"),
            RowValue::F64(9.0),
        ]];
        // The mutation lands whole at coordinate 3 and consumes it.
        assert_eq!(store.supersede(&replacement, &[1]).unwrap(), 1);
        assert_eq!(store.next_sequence(), 4);
        let knowledge = store.knowledge_snapshot().unwrap();
        let xs_at = |cut: u64| -> Vec<f64> {
            let mut xs: Vec<f64> = knowledge
                .as_of(cut)
                .unwrap()
                .iter()
                .flat_map(|view| {
                    let view = view.view().unwrap();
                    let Column::Numeric(NumericData::F64(x)) = &view.segment.batch().columns()[2]
                    else {
                        panic!("x type")
                    };
                    (0..view.segment.batch().num_rows())
                        .filter(|&row| view.is_live(row))
                        .map(|row| x.values().as_slice()[row])
                        .collect::<Vec<_>>()
                })
                .collect();
            xs.sort_by(f64::total_cmp);
            xs
        };
        // The cut before the mutation is all-old, the cut at it is
        // all-new; no cut sees both versions.
        assert_eq!(xs_at(2), [0.0, 1.0, 2.0]);
        assert_eq!(xs_at(3), [0.0, 2.0, 9.0]);
        assert_eq!(xs_at(4), [0.0, 2.0, 9.0]);
    }

    /// A backend that, once armed, fails delete-log writes — the crash
    /// window between a supersession's appends and its commit record.
    struct FailingDeleteLogs {
        inner: crate::MemBackend,
        armed: std::sync::atomic::AtomicBool,
    }

    impl crate::StorageBackend for FailingDeleteLogs {
        fn open_log(&self, name: &str) -> Result<Box<dyn crate::LogWriter>, crate::IoError> {
            self.inner.open_log(name)
        }
        fn write(&self, name: &str, bytes: &[u8]) -> Result<(), crate::IoError> {
            if name.starts_with("del-") && self.armed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(crate::IoError::Backend("injected log failure".to_owned()));
            }
            self.inner.write(name, bytes)
        }
        fn read(&self, name: &str) -> Result<Vec<u8>, crate::IoError> {
            self.inner.read(name)
        }
        fn list(&self) -> Result<Vec<String>, crate::IoError> {
            self.inner.list()
        }
        fn remove(&self, name: &str) -> Result<(), crate::IoError> {
            self.inner.remove(name)
        }
    }

    #[test]
    fn a_crashed_supersession_recovers_old_never_torn() {
        let backend = Arc::new(FailingDeleteLogs {
            inner: crate::MemBackend::new(),
            armed: std::sync::atomic::AtomicBool::new(false),
        });
        let mut store =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
        append_n(&mut store, 0..3);
        let replacement = vec![vec![
            RowValue::I64(1),
            RowValue::Key("A"),
            RowValue::F64(9.0),
        ]];
        // The commit record fails: the mutation errors, and a crash at
        // this instant (no clean close) must recover the OLD state —
        // the WAL bracket at the tail has no commit evidence.
        backend
            .armed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(store.supersede(&replacement, &[1]).is_err());
        std::mem::forget(store);
        let store = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
        assert_eq!(store.live_len(), 3);
        assert_eq!(store.next_sequence(), 3, "old state, still virtual");
        // The same mutation committed replays as NEW — bracket plus
        // commit record — across the same crash-shaped close.
        backend
            .armed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let mut store = store;
        assert_eq!(store.supersede(&replacement, &[1]).unwrap(), 1);
        std::mem::forget(store);
        let store = Store::persistent_with_segment_rows(backend, schema(), 0, 100).unwrap();
        assert_eq!(store.live_len(), 3);
        assert_eq!(store.next_sequence(), 4);
        let views = materialized(&store);
        let total_live: usize = views.iter().map(SegmentView::live_rows).sum();
        assert_eq!(total_live, 3, "replacement in, victim out — never both");
    }

    #[test]
    fn a_supersession_diverges_durably_through_flushed_segments() {
        // A supersession diverges without a compaction, so no manifest
        // records it; reopen must detect divergence from the flushed
        // segment's sequence data alone.
        let backend: Arc<dyn crate::StorageBackend> = Arc::new(crate::MemBackend::new());
        let mut store =
            Store::persistent_with_segment_rows(Arc::clone(&backend), schema(), 0, 100).unwrap();
        append_n(&mut store, 0..3);
        let replacement = vec![vec![
            RowValue::I64(1),
            RowValue::Key("A"),
            RowValue::F64(9.0),
        ]];
        store.supersede(&replacement, &[1]).unwrap();
        store.flush().unwrap();
        drop(store);
        let store = Store::persistent_with_segment_rows(backend, schema(), 0, 100).unwrap();
        assert_eq!(store.next_sequence(), 4);
        assert_eq!(store.live_len(), 3);
        // The mutation pre-flushed its buffered victims as a still-
        // virtual segment; the replacement then landed in an explicit
        // one — and that second segment alone is what tells reopen the
        // table diverged.
        let views = materialized(&store);
        assert_eq!(views[0].segment.sequence_info(), &SequenceInfo::RowIds);
        assert_eq!(
            views[1].segment.sequence_info(),
            &SequenceInfo::Explicit(vec![3]),
        );
        assert!(!views[0].is_live(1), "the victim stays masked");
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
        let views = materialized(&store);
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
        let views = materialized(&store);
        assert_eq!(
            views[0].segment.sequence_info(),
            &SequenceInfo::Explicit(vec![50, 51, 52, 53])
        );
        store
            .append(&[RowValue::I64(9), RowValue::Key("A"), RowValue::F64(0.0)])
            .unwrap();
        let views = materialized(&store);
        assert_eq!(
            views.last().unwrap().segment.sequence_info(),
            &SequenceInfo::Contiguous { base: 56 }
        );
    }

    #[test]
    fn touched_rows_carry_the_join_key_through_births_kills_and_history() {
        // The #83 tranche-3 seam: a maintained join view's refresh
        // needs to know WHOSE row changed (a symbol-blind interval
        // endpoint under-repairs), so the touched walk also yields the
        // named key column's value — from a pending kill, from a kill
        // compacted into history, and from a birth alike.
        let mut store = Store::with_segment_rows(schema(), 0, 4).unwrap();
        append_n(&mut store, 0..6); // sequences 0..=5; A even, B odd
                                    // Kill ts=1 (row id 1, symbol B): the kill consumes sequence 6.
        store.tombstone(&[1]).unwrap();
        let pairs = |store: &Store, since: u64| {
            let mut touched: Vec<(i64, Option<String>)> = Vec::new();
            store
                .knowledge_snapshot()
                .unwrap()
                .touched_rows(since, 1, |ts, sym| {
                    touched.push((ts, sym.map(str::to_owned)));
                })
                .unwrap();
            touched
        };
        // Only the pending kill sits at or after 6 — and it names B.
        assert_eq!(pairs(&store, 6), [(1, Some("B".to_owned()))]);
        // Compaction moves the kill to history; the walk still sees it
        // and still names B (the history branch reads the same row).
        store.compact().unwrap();
        assert_eq!(pairs(&store, 6), [(1, Some("B".to_owned()))]);
        // A birth after the watermark reports its own symbol.
        store
            .append(&[RowValue::I64(9), RowValue::Key("C"), RowValue::F64(0.0)])
            .unwrap(); // sequence 7
        assert_eq!(pairs(&store, 7), [(9, Some("C".to_owned()))]);
        // The two public walks share one body: same rows, same order.
        let snapshot = store.knowledge_snapshot().unwrap();
        let mut keys = Vec::new();
        let mut with_syms = Vec::new();
        snapshot
            .touched_ordering_keys(0, |ts| keys.push(ts))
            .unwrap();
        snapshot
            .touched_rows(0, 1, |ts, _| with_syms.push(ts))
            .unwrap();
        assert_eq!(keys, with_syms);
        assert!(!keys.is_empty());
        // Misuse is a loud error, never a panic: neither the i64
        // ordering key nor the f64 payload is a key column.
        for column in [0usize, 2] {
            let error = snapshot.touched_rows(0, column, |_, _| {}).unwrap_err();
            assert!(
                matches!(error, StorageError::TypeMismatch { .. }),
                "{error:?}"
            );
        }
    }
}
