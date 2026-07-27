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
    // so power loss takes every unflushed row — the window's worst case.
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    {
        let mut store = open(
            backend.clone(),
            WalSync::Group(std::time::Duration::from_secs(3600)),
        );
        append_n(&mut store, 0..7);
    }
    let store = open(backend.clone(), WalSync::Full);
    assert_eq!(store.len(), 0, "unsynced tail lost, as the window allows");
    // A zero interval syncs every append: nothing lost.
    {
        let mut store = open(backend.clone(), WalSync::Group(std::time::Duration::ZERO));
        append_n(&mut store, 0..7);
    }
    let store = open(backend, WalSync::Full);
    assert_eq!(store.len(), 7);
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
    // Flip a byte inside the third record's payload (past the 26-byte
    // header and two records; each record here is 4 + 26 + 4 bytes).
    let mut bytes = backend.read("wal.tlyw").unwrap();
    let offset = 26 + 2 * 34 + 8;
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
