//! Residency spec (ruled 2026-07-30, option b): a table opens without
//! decoding any segment file, faults segments in on first data access,
//! and retains them under a byte budget — least-recently-used out,
//! pinned (in-use) segments never. The tests observe all of it from
//! outside, by counting backend reads: a fault is a read, an eviction
//! is a re-read on the next touch, and retention is the absence of one.

use arrow_lite::{ColumnType, Field, Schema};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use storage_lite::{
    FsBackend, IoError, LogWriter, MemBackend, RowValue, StorageBackend, Store, StoreOptions,
    WalSync,
};

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ])
}

/// Counts every `read` per object name; the observable that makes
/// laziness and eviction testable without reaching into the store.
struct CountingBackend {
    inner: Arc<dyn StorageBackend>,
    reads: Mutex<HashMap<String, usize>>,
}

impl CountingBackend {
    fn new(inner: Arc<dyn StorageBackend>) -> Arc<CountingBackend> {
        Arc::new(CountingBackend {
            inner,
            reads: Mutex::new(HashMap::new()),
        })
    }

    /// Total reads of segment files (`seg-*` / `hist-*`), not counting
    /// the manifest, the WAL, or delete logs.
    fn segment_reads(&self) -> usize {
        self.reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.ends_with(".tlyseg"))
            .map(|(_, count)| count)
            .sum()
    }

    fn reads_of(&self, name: &str) -> usize {
        self.reads.lock().unwrap().get(name).copied().unwrap_or(0)
    }

    fn reset(&self) {
        self.reads.lock().unwrap().clear();
    }
}

impl StorageBackend for CountingBackend {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
        self.inner.write(name, bytes)
    }
    fn read(&self, name: &str) -> Result<Vec<u8>, IoError> {
        *self
            .reads
            .lock()
            .unwrap()
            .entry(name.to_owned())
            .or_insert(0) += 1;
        self.inner.read(name)
    }
    fn list(&self) -> Result<Vec<String>, IoError> {
        self.inner.list()
    }
    fn remove(&self, name: &str) -> Result<(), IoError> {
        self.inner.remove(name)
    }
    fn open_log(&self, name: &str) -> Result<Box<dyn LogWriter>, IoError> {
        self.inner.open_log(name)
    }
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

fn options(segment_rows: usize, cache_bytes: Option<u64>) -> StoreOptions {
    StoreOptions {
        segment_rows: Some(segment_rows),
        wal_sync: WalSync::Off,
        cache_bytes,
        ..StoreOptions::default()
    }
}

#[test]
fn a_recorded_open_reads_no_segment_files_and_a_fault_reads_exactly_one() {
    let counting = CountingBackend::new(Arc::new(MemBackend::new()));
    {
        let backend: Arc<dyn StorageBackend> = counting.clone();
        let mut store = Store::persistent_with(backend, schema(), 0, options(4, None)).unwrap();
        append_n(&mut store, 0..12); // three flushed segments
        store.flush().unwrap();
    }
    counting.reset();
    let backend: Arc<dyn StorageBackend> = counting.clone();
    let store = Store::persistent_with(backend, schema(), 0, options(4, None)).unwrap();
    assert_eq!(
        counting.segment_reads(),
        0,
        "the open served itself from the manifest's records alone"
    );
    // Metadata answers without I/O; the first view is the fault.
    let handles = store.snapshot().unwrap();
    assert_eq!(handles.len(), 3);
    assert_eq!(handles[1].base_row_id(), 4);
    assert_eq!(counting.segment_reads(), 0, "metadata is free");
    handles[1].view().unwrap();
    assert_eq!(counting.segment_reads(), 1, "one fault, one read");
    // Resident now: touching it again reads nothing.
    handles[1].view().unwrap();
    assert_eq!(counting.segment_reads(), 1);
}

#[test]
fn a_read_only_open_and_refresh_read_no_segment_files() {
    let dir = std::env::temp_dir().join(format!("tallydb-residency-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let writer_backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    let mut writer = Store::persistent_with(writer_backend, schema(), 0, options(4, None)).unwrap();
    append_n(&mut writer, 0..8);
    writer.flush().unwrap();

    let counting = CountingBackend::new(Arc::new(FsBackend::open_read_only(&dir).unwrap()));
    let backend: Arc<dyn StorageBackend> = counting.clone();
    let mut reader = Store::open_read_only(backend).unwrap();
    assert_eq!(counting.segment_reads(), 0, "the reader opened lazily");
    assert_eq!(reader.live_len(), 8, "row counts come from metadata");
    append_n(&mut writer, 8..12);
    writer.flush().unwrap();
    reader.refresh().unwrap();
    assert_eq!(counting.segment_reads(), 0, "the refresh stayed lazy too");
    assert_eq!(reader.live_len(), 12);
    drop(writer);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_budget_evicts_cold_segments_and_never_pinned_ones() {
    // Six segments of four rows each (~150 decoded bytes apiece); a
    // budget of ~two segments. Touch them all in order: the early ones
    // get evicted (their next touch re-reads), the pinned one never is
    // (its next touch reads nothing new).
    let counting = CountingBackend::new(Arc::new(MemBackend::new()));
    {
        let backend: Arc<dyn StorageBackend> = counting.clone();
        let mut store = Store::persistent_with(backend, schema(), 0, options(4, None)).unwrap();
        append_n(&mut store, 0..24);
        store.flush().unwrap();
    }
    counting.reset();
    let backend: Arc<dyn StorageBackend> = counting.clone();
    let store = Store::persistent_with(backend, schema(), 0, options(4, Some(300))).unwrap();
    let handles = store.snapshot().unwrap();
    assert_eq!(handles.len(), 6);
    let name = |index: usize| format!("seg-g0000000000-{:020}.tlyseg", index * 4);

    // Pin segment 0 by holding its view across the whole walk.
    let pinned = handles[0].view().unwrap();
    for handle in &handles[1..] {
        handle.view().unwrap(); // fault in, then release
    }
    // The walk exceeded the budget, so the coldest unpinned segments
    // were evicted along the way: touching segment 1 again re-reads it.
    let before = counting.reads_of(&name(1));
    handles[1].view().unwrap();
    assert_eq!(
        counting.reads_of(&name(1)),
        before + 1,
        "segment 1 was evicted and re-faulted"
    );
    // The pinned segment survived the entire walk: one read, ever.
    handles[0].view().unwrap();
    assert_eq!(
        counting.reads_of(&name(0)),
        1,
        "a pinned segment is never evicted"
    );
    drop(pinned);
}

#[test]
fn an_unbounded_store_retains_everything_it_touches() {
    // The interim default (decision #87): no budget, no eviction —
    // exactly the pre-residency retention behavior.
    let counting = CountingBackend::new(Arc::new(MemBackend::new()));
    {
        let backend: Arc<dyn StorageBackend> = counting.clone();
        let mut store = Store::persistent_with(backend, schema(), 0, options(4, None)).unwrap();
        append_n(&mut store, 0..24);
        store.flush().unwrap();
    }
    counting.reset();
    let backend: Arc<dyn StorageBackend> = counting.clone();
    let store = Store::persistent_with(backend, schema(), 0, options(4, None)).unwrap();
    let handles = store.snapshot().unwrap();
    for _ in 0..3 {
        for handle in &handles {
            handle.view().unwrap();
        }
    }
    assert_eq!(
        counting.segment_reads(),
        6,
        "each segment read exactly once"
    );
}
