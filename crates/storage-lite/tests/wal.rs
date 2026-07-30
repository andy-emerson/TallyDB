//! Crash-injection tests for the write-ahead log (#43): what survives
//! a crash is exactly what the configured sync level promises. The
//! MemBackend models power loss — appended-but-unsynced log bytes
//! vanish when the store drops — which is the adversarial case; the fs
//! backend is additionally exercised where its semantics differ
//! (OS-buffered bytes *do* survive a process crash).

use std::sync::Arc;
use storage_lite::{FsBackend, MemBackend, RowValue, StorageBackend, Store, StoreOptions, WalSync};

use arrow_lite::{ColumnType, Field, NumericData, Schema};

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ])
}

fn open(backend: Arc<dyn StorageBackend>, wal_sync: WalSync) -> Store {
    Store::persistent_with(
        backend,
        schema(),
        0,
        StoreOptions {
            segment_rows: Some(1000), // large: nothing auto-flushes
            wal_sync,
            ..StoreOptions::default()
        },
    )
    .unwrap()
}

fn append_n(store: &mut Store, range: std::ops::Range<i64>) {
    for i in range {
        store
            .append(&[
                RowValue::I64(i),
                RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                RowValue::F64(i as f64 * 0.5),
            ])
            .unwrap();
    }
}

fn ts_values(store: &Store) -> Vec<i64> {
    let mut out = Vec::new();
    for view in store.snapshot().unwrap() {
        let view = view.view().unwrap();
        let arrow_lite::Column::Numeric(NumericData::I64(ts)) = &view.segment.batch().columns()[0]
        else {
            panic!("ts is i64")
        };
        for (row, &value) in ts.values().as_slice().iter().enumerate() {
            if view.is_live(row) {
                out.push(value);
            }
        }
    }
    out
}

#[test]
fn full_sync_survives_power_loss_exactly() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..7);
    } // power loss: no flush, but every append was synced
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 7);
    assert_eq!(ts_values(&store), (0..7).collect::<Vec<_>>());
}

#[test]
fn group_sync_loses_at_most_the_window() {
    // A one-hour group interval: no sync ever fires during the test,
    // so power loss takes every unflushed row — the window's worst
    // case. Power loss means drop never runs (a drop syncs — see the
    // clean-close test below), so the store is leaked, not dropped.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(
            backend.clone(),
            WalSync::Group(std::time::Duration::from_secs(3600)),
        );
        append_n(&mut store, 0..7);
        std::mem::forget(store);
    }
    let store = open(backend.clone(), WalSync::Full);
    assert_eq!(store.len(), 0, "unsynced tail lost, as the window allows");
    // A zero interval syncs every append: nothing lost even at power
    // loss.
    {
        let mut store = open(backend.clone(), WalSync::Group(std::time::Duration::ZERO));
        append_n(&mut store, 0..7);
        std::mem::forget(store);
    }
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 7);
}

#[test]
fn a_clean_close_syncs_the_group_tail() {
    // The idle-writer case: rows appended, interval never elapsed, and
    // the process exits cleanly. Drop syncs the tail — durability must
    // not depend on a next append that never comes.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(
            backend.clone(),
            WalSync::Group(std::time::Duration::from_secs(3600)),
        );
        append_n(&mut store, 0..7);
    } // dropped cleanly
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 7);
    assert_eq!(ts_values(&store), (0..7).collect::<Vec<_>>());
}

#[test]
fn a_torn_tail_ends_the_clean_prefix_silently() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..5);
    }
    // Tear the last record: strip three bytes off the log.
    let bytes = backend.read("wal.tlyw").unwrap();
    backend
        .write("wal.tlyw", &bytes[..bytes.len() - 3])
        .unwrap();
    let store = open(backend, WalSync::Full);
    assert_eq!(
        store.len(),
        4,
        "clean prefix recovered, torn record dropped"
    );
    assert_eq!(ts_values(&store), (0..4).collect::<Vec<_>>());
}

#[test]
fn a_corrupt_record_ends_the_clean_prefix() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..5);
    }
    // Flip a byte inside the third record's payload (past the 30-byte
    // header and two records; each record here is 4 + 26 + 4 bytes).
    let mut bytes = backend.read("wal.tlyw").unwrap();
    let offset = 30 + 2 * 34 + 8;
    bytes[offset] ^= 0xFF;
    backend.write("wal.tlyw", &bytes).unwrap();
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 2, "recovery stops at the corrupt record");
}

#[test]
fn flush_truncates_and_replay_skips_the_flushed_prefix() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..5);
        store.flush().unwrap(); // rows 0..5 now segment-durable; log reset
        append_n(&mut store, 5..8);
    }
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 8);
    assert_eq!(ts_values(&store), (0..8).collect::<Vec<_>>());
}

#[test]
fn recovery_is_durable_and_repeatable() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..4);
    }
    // First recovery re-logs the recovered rows; crash again at once.
    {
        let store = open(backend.clone(), WalSync::Full);
        assert_eq!(store.len(), 4);
    }
    // Second recovery still sees them, and the store keeps working.
    let mut store = open(backend.clone(), WalSync::Full);
    assert_eq!(store.len(), 4);
    append_n(&mut store, 4..6);
    drop(store);
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 6);
    assert_eq!(ts_values(&store), (0..6).collect::<Vec<_>>());
}

#[test]
fn compaction_resets_the_log_into_the_new_generation() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..6);
        store.tombstone(&[1, 3]).unwrap();
        store.compact().unwrap();
        append_n(&mut store, 100..103); // post-compaction, WAL-guarded
    }
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 7); // 4 compacted survivors + 3 recovered
    assert_eq!(ts_values(&store), vec![0, 2, 4, 5, 100, 101, 102]);
}

#[test]
fn a_rejected_append_leaves_no_phantom_record() {
    // A row the schema rejects must leave the WAL untouched: logged
    // before validation, it would occupy a record replay chokes on —
    // ending the clean prefix early (dropping every acknowledged row
    // after it) or replaying a row the caller was told failed.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..3);
        assert!(store
            .append(&[
                RowValue::F64(3.5), // F64 into the I64 ordering column
                RowValue::Key("A"),
                RowValue::F64(0.0),
            ])
            .is_err());
        assert!(store.append(&[RowValue::I64(3)]).is_err(), "wrong arity");
        append_n(&mut store, 3..6);
    } // power loss
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 6, "every acknowledged row, nothing else");
    assert_eq!(ts_values(&store), (0..6).collect::<Vec<_>>());
}

#[test]
fn a_short_or_empty_log_reads_as_empty_not_corrupt() {
    // The log is born and rotated by atomic publish, so our own writes
    // never leave a sub-header file — but a filesystem that truncates
    // on power loss can. No record can be acknowledged under an
    // unsynced header, so reopen must treat a short log as empty — not
    // as corruption that bricks a store whose segments are intact.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..5);
        store.flush().unwrap(); // rows are segment-durable
    }
    for stub in [&[][..], &[0x54, 0x41, 0x4C][..]] {
        backend.write("wal.tlyw", stub).unwrap();
        let store = open(backend.clone(), WalSync::Full);
        assert_eq!(store.len(), 5, "stub of {} bytes", stub.len());
        assert_eq!(ts_values(&store), (0..5).collect::<Vec<_>>());
        drop(store);
    }
}

#[test]
fn a_corrupt_header_is_loud_not_silent() {
    // A full-length header whose checksum fails is real corruption —
    // the CRC-everything rule covers the header too — and recovery
    // refuses loudly instead of guessing at the generation.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..3);
        std::mem::forget(store); // keep the log; a drop-sync is fine too
    }
    let mut bytes = backend.read("wal.tlyw").unwrap();
    bytes[12] ^= 0xFF; // inside the generation field
    backend.write("wal.tlyw", &bytes).unwrap();
    let Err(error) = Store::persistent_with(
        backend,
        schema(),
        0,
        StoreOptions {
            segment_rows: Some(1000),
            wal_sync: WalSync::Full,
            ..StoreOptions::default()
        },
    ) else {
        panic!("corrupt header must be refused");
    };
    assert!(
        error.to_string().contains("checksum"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_delete_log_never_commits_ahead_of_superseding_rows() {
    // UPDATE's shape at the storage layer: replacements appended first,
    // then the originals tombstoned. The delete log commits the moment
    // it is written — so the replacements must be made durable *before*
    // it does, or a crash in the group window recovers the deletion
    // without the replacements: originals gone, replacements gone, the
    // one middle state that loses data forever. (The tombstone's
    // unconditional flush is what discharges this now; it used to be a
    // WAL sync, which held only while nothing was buffered.)
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(
            backend.clone(),
            WalSync::Group(std::time::Duration::from_secs(3600)),
        );
        append_n(&mut store, 0..4);
        store.flush().unwrap(); // originals segment-durable
        append_n(&mut store, 100..104); // replacements: logged, unsynced
        store.tombstone(&[0, 1, 2, 3]).unwrap(); // must sync the WAL first
        std::mem::forget(store); // power loss
    }
    let store = open(backend, WalSync::Full);
    // Eight ids in the row-id space (originals stay until compaction);
    // the four live rows are the replacements.
    assert_eq!(store.len(), 8);
    assert_eq!(ts_values(&store), vec![100, 101, 102, 103]);
}

#[test]
fn a_delete_log_never_commits_ahead_of_buffered_rows_under_off() {
    // The same invariant without a WAL: under `Off` the replacements
    // live only in the write buffer, so the tombstone must flush them
    // into a segment before its delete log commits — the same flush,
    // reached by a path that never had a log to sync.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Off);
        append_n(&mut store, 0..4);
        store.flush().unwrap();
        append_n(&mut store, 100..104);
        store.tombstone(&[0, 1, 2, 3]).unwrap(); // must flush first
        std::mem::forget(store); // power loss
    }
    let store = open(backend, WalSync::Off);
    assert_eq!(store.len(), 8);
    assert_eq!(ts_values(&store), vec![100, 101, 102, 103]);
}

/// A delete consumes a coordinate, and replay assigns buffered rows
/// their sequences positionally from the recovered watermark — which
/// already counts the gap. So nothing born *below* the kill may still
/// be in the log when it lands, or recovery would renumber those rows
/// above it. The tombstone's flush is what guarantees that; this is
/// the shape that would catch its absence: rows buffered when a kill
/// lands on a row that is already durable.
#[test]
fn replay_across_a_consumed_coordinate_keeps_births_below_it() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(
            backend.clone(),
            WalSync::Group(std::time::Duration::from_secs(3600)),
        );
        append_n(&mut store, 0..4); // sequences 0..3
        store.flush().unwrap(); // durable, and the WAL is truncated
        append_n(&mut store, 100..103); // sequences 4..6, logged only
        store.tombstone(&[0]).unwrap(); // kills a durable row, spends 7
        assert_eq!(store.next_sequence(), 8);
        std::mem::forget(store); // power loss
    }
    let store = open(backend, WalSync::Full);
    // The spent coordinate is recovered from the delete log alone: no
    // row carries 7, and no later row may be issued below 8.
    assert_eq!(store.next_sequence(), 8);
    // The three rows that arrived before the kill are back — the
    // tombstone's flush made them durable — and still carry sequences
    // below it, rather than being renumbered above the gap.
    assert_eq!(ts_values(&store), vec![1, 2, 3, 100, 101, 102]);
    let births: Vec<u64> = store
        .snapshot()
        .unwrap()
        .iter()
        .flat_map(|view| {
            let view = view.view().unwrap();
            let segment = &view.segment;
            (0..segment.batch().num_rows())
                .filter(|&row| view.is_live(row))
                .map(|row| segment.sequence_at(row))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(births, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn off_means_no_log_at_all() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    let mut store = open(backend.clone(), WalSync::Off);
    append_n(&mut store, 0..5);
    assert!(
        !backend
            .list()
            .unwrap()
            .iter()
            .any(|name| name == "wal.tlyw"),
        "Off writes no log"
    );
}

#[test]
fn a_stale_wal_from_before_compaction_is_ignored() {
    // Simulate the crash window between compaction's manifest commit
    // and its log reset: save the old-generation log, compact, restore
    // the stale bytes over the fresh log, reopen.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..5);
        let stale = backend.read("wal.tlyw").unwrap();
        store.compact().unwrap();
        drop(store);
        backend.write("wal.tlyw", &stale).unwrap();
    }
    let store = open(backend, WalSync::Full);
    // The stale log's generation mismatches: ignored, not replayed —
    // its rows are already in the compacted segments, and replaying
    // would duplicate them in the wrong id space.
    assert_eq!(store.len(), 5);
    assert_eq!(ts_values(&store), (0..5).collect::<Vec<_>>());
}

#[test]
fn the_fs_backend_recovers_the_same_way() {
    let dir = std::env::temp_dir().join(format!(
        "tallydb-wal-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    {
        let mut store = open(backend.clone(), WalSync::Full);
        append_n(&mut store, 0..5);
        store.flush().unwrap();
        append_n(&mut store, 5..9);
    }
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 9);
    assert_eq!(ts_values(&store), (0..9).collect::<Vec<_>>());
    drop(store);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The ingest measurement behind #43's ruling, now resident (#52's
/// first entry): per-append cost under each sync level, on the real
/// append path. Ratios within one run are the durable numbers.
///
/// ```text
/// cargo test -p storage-lite --release --test wal measure -- --ignored --nocapture
/// ```
#[test]
#[ignore = "measurement — run explicitly in release mode"]
fn measure_wal_regimes() {
    let dir = std::env::temp_dir().join(format!("tallydb-wal-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    println!("trickle ingest, one row per append, fs backend, segment threshold 4096:");
    for (name, wal_sync, rows) in [
        ("off (flush boundary)", WalSync::Off, 200_000i64),
        (
            "group 100ms (default)",
            WalSync::Group(std::time::Duration::from_millis(100)),
            200_000,
        ),
        (
            "group 10ms",
            WalSync::Group(std::time::Duration::from_millis(10)),
            200_000,
        ),
        ("full (every append)", WalSync::Full, 2_000),
    ] {
        let sub = dir.join(name.replace(' ', "-"));
        std::fs::create_dir_all(&sub).unwrap();
        let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&sub).unwrap());
        let mut store = Store::persistent_with(
            backend,
            schema(),
            0,
            StoreOptions {
                segment_rows: Some(4096),
                wal_sync,
                ..StoreOptions::default()
            },
        )
        .unwrap();
        let start = std::time::Instant::now();
        append_n(&mut store, 0..rows);
        let per_append = start.elapsed().as_secs_f64() / rows as f64;
        store.flush().unwrap();
        println!(
            "  {name:<24} {:>9.2}us/append  ({rows} rows)",
            per_append * 1e6
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A backend that fails every manifest write once armed — the crash
/// injection for the window a flush opens between publishing a segment
/// file and the manifest write that adopts it (tag 1, the residency
/// design 2026-07-30).
struct FailManifestWrites {
    inner: Arc<dyn StorageBackend>,
    armed: std::sync::atomic::AtomicBool,
}

impl StorageBackend for FailManifestWrites {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), storage_lite::IoError> {
        if name == "table.tlym" && self.armed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(storage_lite::IoError::Backend(
                "injected: manifest write lost".to_owned(),
            ));
        }
        self.inner.write(name, bytes)
    }
    fn read(&self, name: &str) -> Result<Vec<u8>, storage_lite::IoError> {
        self.inner.read(name)
    }
    fn list(&self) -> Result<Vec<String>, storage_lite::IoError> {
        self.inner.list()
    }
    fn remove(&self, name: &str) -> Result<(), storage_lite::IoError> {
        self.inner.remove(name)
    }
    fn open_log(
        &self,
        name: &str,
    ) -> Result<Box<dyn storage_lite::LogWriter>, storage_lite::IoError> {
        self.inner.open_log(name)
    }
}

#[test]
fn a_crash_between_the_segment_write_and_its_manifest_write_loses_nothing() {
    // The flush order is segment file, then the manifest that names it,
    // then the WAL reset. A crash between the first two leaves an
    // orphan segment file the manifest never adopted — and every one of
    // its rows still in the WAL, because the reset never ran. Reopen
    // must serve each row exactly once: the orphan is invisible, the
    // WAL replays, and the re-flush overwrites the orphan under the
    // same deterministic name.
    let inner: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    let failing = Arc::new(FailManifestWrites {
        inner: inner.clone(),
        armed: std::sync::atomic::AtomicBool::new(false),
    });
    {
        let backend: Arc<dyn StorageBackend> = failing.clone();
        let mut store = Store::persistent_with(
            backend,
            schema(),
            0,
            StoreOptions {
                segment_rows: Some(4),
                wal_sync: WalSync::Full,
                ..StoreOptions::default()
            },
        )
        .unwrap();
        append_n(&mut store, 0..3);
        failing
            .armed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // The fourth append reaches the threshold; its flush writes the
        // segment file and then dies on the manifest write.
        let result = store.append(&[RowValue::I64(3), RowValue::Key("B"), RowValue::F64(1.5)]);
        assert!(result.is_err(), "the injected manifest failure surfaces");
        std::mem::forget(store); // crash: drop never runs
    }
    // The window was real: the orphan segment file exists on disk...
    assert!(
        inner
            .list()
            .unwrap()
            .iter()
            .any(|name| name.starts_with("seg-")),
        "the segment file was published before the manifest failure"
    );
    // ...and the manifest never adopted it.
    let manifest = storage_lite::decode_manifest(&inner.read("table.tlym").unwrap()).unwrap();
    assert!(manifest.sections.segments.is_empty(), "no record adopted");
    // Reopen on the healed backend: all four rows, exactly once.
    let mut store = Store::persistent_with(
        inner.clone(),
        schema(),
        0,
        StoreOptions {
            segment_rows: Some(4),
            wal_sync: WalSync::Full,
            ..StoreOptions::default()
        },
    )
    .unwrap();
    assert_eq!(store.len(), 4);
    assert_eq!(ts_values(&store), vec![0, 1, 2, 3]);
    // And the store is fully live: the next flush adopts the layout.
    store.flush().unwrap();
    let manifest = storage_lite::decode_manifest(&inner.read("table.tlym").unwrap()).unwrap();
    assert_eq!(manifest.sections.segments.len(), 1);
    assert_eq!(ts_values(&store), vec![0, 1, 2, 3]);
}
