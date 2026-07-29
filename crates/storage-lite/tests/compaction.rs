//! Compaction spec tests: the tests are the spec for what "resolved at
//! the next compaction" means — tombstones gone, order restored, row
//! ids contiguous again, durably and crash-safely.

use arrow_lite::{Column, ColumnType, Field, NumericData, Schema};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use storage_lite::{FsBackend, IoError, MemBackend, RowValue, SequenceInfo, StorageBackend, Store};

/// A backend that, once armed, fails every `remove` — used to model a
/// post-commit cleanup failure during compaction (R1).
struct FailingRemoves {
    inner: MemBackend,
    armed: AtomicBool,
}

impl FailingRemoves {
    fn new() -> FailingRemoves {
        FailingRemoves {
            inner: MemBackend::new(),
            armed: AtomicBool::new(false),
        }
    }
    fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl StorageBackend for FailingRemoves {
    fn open_log(&self, name: &str) -> Result<Box<dyn storage_lite::LogWriter>, IoError> {
        self.inner.open_log(name)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
        self.inner.write(name, bytes)
    }
    fn read(&self, name: &str) -> Result<Vec<u8>, IoError> {
        self.inner.read(name)
    }
    fn list(&self) -> Result<Vec<String>, IoError> {
        self.inner.list()
    }
    fn remove(&self, name: &str) -> Result<(), IoError> {
        if self.armed.load(Ordering::SeqCst) {
            return Err(IoError::Backend("injected remove failure".to_owned()));
        }
        self.inner.remove(name)
    }
}

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ])
}

fn append(store: &mut Store, ts: i64, sym: &str, x: f64) -> u64 {
    store
        .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
        .unwrap()
}

/// Live rows as (ts, sym, x) triples across the snapshot.
fn rows(store: &Store) -> Vec<(i64, String, f64)> {
    store
        .snapshot()
        .unwrap()
        .iter()
        .flat_map(|view| {
            let batch = view.segment.batch();
            let Column::Numeric(NumericData::I64(ts)) = &batch.columns()[0] else {
                panic!("ts type")
            };
            let Column::Key(sym) = &batch.columns()[1] else {
                panic!("sym type")
            };
            let Column::Numeric(NumericData::F64(x)) = &batch.columns()[2] else {
                panic!("x type")
            };
            (0..batch.num_rows())
                .filter(|&row| view.is_live(row))
                .map(|row| {
                    (
                        ts.values().as_slice()[row],
                        sym.value_at(row).unwrap().to_owned(),
                        x.values().as_slice()[row],
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn each_backend(test: impl Fn(Arc<dyn StorageBackend>)) {
    test(Arc::new(MemBackend::new()));
    let dir = std::env::temp_dir().join(format!(
        "tallydb-compact-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    test(Arc::new(FsBackend::new(&dir).unwrap()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn compaction_drops_tombstones_and_restores_contiguity() {
    let mut store = Store::with_segment_rows(schema(), 0, 3).unwrap();
    for i in 0..10i64 {
        append(&mut store, i, if i % 2 == 0 { "A" } else { "B" }, i as f64);
    }
    store.tombstone(&[0, 4, 5, 9]).unwrap();
    store.compact().unwrap();
    assert_eq!(store.len(), 6); // ids reassigned contiguously
    assert_eq!(store.live_len(), 6);
    assert_eq!(
        rows(&store).iter().map(|row| row.0).collect::<Vec<_>>(),
        [1, 2, 3, 6, 7, 8]
    );
    // Bases are contiguous over the new ids and everything is live.
    let views = store.snapshot().unwrap();
    assert!(views.iter().all(|view| view.live.is_none()));
    assert_eq!(
        views
            .iter()
            .map(|view| view.segment.base_row_id())
            .collect::<Vec<_>>(),
        [0, 3]
    );
    // The next append picks up after the survivors.
    assert_eq!(append(&mut store, 99, "A", 0.0), 6);
}

#[test]
fn compaction_sorts_late_arrivals_and_keeps_ingest_order_on_ties() {
    let mut store = Store::with_segment_rows(schema(), 0, 4).unwrap();
    append(&mut store, 10, "A", 1.0);
    append(&mut store, 30, "A", 2.0);
    append(&mut store, 20, "B", 3.0); // late arrival
    append(&mut store, 20, "B", 4.0); // duplicate ordering value, later ingest
    append(&mut store, 5, "C", 5.0); // very late
    let unordered = store.snapshot().unwrap();
    assert!(unordered.iter().any(|view| !view.segment.is_ordered()));
    store.compact().unwrap();
    // Sorted by ts; the tie at 20 keeps ingest order (x=3 before x=4);
    // duplicates survive — nothing collapses them.
    assert_eq!(
        rows(&store),
        [
            (5, "C".to_owned(), 5.0),
            (10, "A".to_owned(), 1.0),
            (20, "B".to_owned(), 3.0),
            (20, "B".to_owned(), 4.0),
            (30, "A".to_owned(), 2.0),
        ]
    );
    assert!(store
        .snapshot()
        .unwrap()
        .iter()
        .all(|view| view.segment.is_ordered()));
}

#[test]
fn compaction_merges_dictionaries_per_segment() {
    // Two segments with disjoint intern orders collapse into fresh
    // segments whose dictionaries are self-contained (#6) and minimal.
    let mut store = Store::with_segment_rows(schema(), 0, 2).unwrap();
    append(&mut store, 1, "B", 1.0);
    append(&mut store, 2, "A", 2.0);
    append(&mut store, 3, "C", 3.0);
    append(&mut store, 4, "A", 4.0);
    store.tombstone(&[2]).unwrap(); // C never survives
    store.compact().unwrap();
    let views = store.snapshot().unwrap();
    let mut values: Vec<String> = Vec::new();
    for view in &views {
        let Column::Key(sym) = &view.segment.batch().columns()[1] else {
            panic!("sym type")
        };
        let dictionary = sym.dictionary();
        for code in 0..dictionary.len() as u32 {
            values.push(dictionary.value(code).to_owned());
        }
    }
    values.sort();
    values.dedup();
    assert_eq!(values, ["A", "B"]); // C is gone from every dictionary
}

#[test]
fn compaction_is_durable_and_leaves_no_stale_objects() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 3).unwrap();
            for i in 0..9i64 {
                append(&mut store, 9 - i, "A", i as f64); // fully reversed ingest
            }
            store.tombstone(&[0, 8]).unwrap();
            store.compact().unwrap();
            assert_eq!(store.live_len(), 7);
        }
        // Reopen sees the compacted generation only.
        let store = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 3).unwrap();
        assert_eq!(store.len(), 7);
        assert_eq!(
            rows(&store).iter().map(|row| row.0).collect::<Vec<_>>(),
            [2, 3, 4, 5, 6, 7, 8] // sorted; ts 9 (id 0) and ts 1 (id 8) died
        );
        // No delete logs or stale-generation segments remain.
        let names = backend.list().unwrap();
        assert!(
            names.iter().all(|name| !name.starts_with("del-")),
            "{names:?}"
        );
        assert!(
            names
                .iter()
                .filter(|name| name.starts_with("seg-"))
                .all(|name| name.starts_with("seg-g0000000001-")),
            "{names:?}"
        );
    });
}

#[test]
fn crashed_compaction_before_commit_is_invisible() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2).unwrap();
            for i in 0..6i64 {
                append(&mut store, i, "A", i as f64);
            }
            store.tombstone(&[1]).unwrap();
        }
        // Simulate a compaction that wrote next-generation segments but
        // crashed before the manifest commit: plant a gen-1 stray at a
        // base the real gen-1 layout will NOT overwrite — the dangerous
        // case, since after a later commit to generation 1 it would
        // otherwise be loaded as real data.
        {
            let donor = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2)
                .unwrap()
                .snapshot()
                .unwrap();
            let bytes = storage_lite::encode_segment(&donor[0].segment);
            backend
                .write("seg-g0000000001-00000000000000000999.tlyseg", &bytes)
                .unwrap();
        }
        // Reopen: the manifest still names generation 0 — the crashed
        // generation's object is ignored, tombstones intact.
        let mut store =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2).unwrap();
        assert_eq!(store.len(), 6);
        assert_eq!(store.live_len(), 5);
        // The next successful compaction pre-cleans the stray, commits
        // generation 1, and the reopened table holds exactly the five
        // live rows — the stray's rows never leak in.
        store.compact().unwrap();
        let names = backend.list().unwrap();
        let segment_names: Vec<&String> = names
            .iter()
            .filter(|name| name.starts_with("seg-"))
            .collect();
        assert!(
            segment_names
                .iter()
                .all(|name| name.starts_with("seg-g0000000001-")),
            "{names:?}"
        );
        assert!(!names
            .iter()
            .any(|name| name.ends_with("00000000000000000999.tlyseg")));
        let reopened =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2).unwrap();
        assert_eq!(reopened.len(), 5);
    });
}

#[test]
fn compaction_cleanup_failure_does_not_strand_the_generation() {
    // R1: the post-commit cleanup of stale objects is best-effort. If a
    // `remove` fails after the manifest commit, the store must still adopt
    // the new generation — otherwise memory stays at gen N while disk is
    // gen N+1, and every later write (gen-N names) is silently dropped at
    // reopen.
    let backend = Arc::new(FailingRemoves::new());
    {
        let mut store =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 3).unwrap();
        for i in 0..6i64 {
            append(&mut store, i, "A", i as f64);
        }
        store.flush().unwrap(); // gen-0 objects now exist -> cleanup has work
                                // Arm the fault: the post-commit removal of the stale gen-0
                                // segments will fail.
        backend.arm();
        // Compaction must still succeed — cleanup is best-effort — and the
        // generation must advance despite the failed removes.
        store.compact().unwrap();
        // A write after the failed-cleanup compaction must land in the new
        // generation, not a stranded old one.
        append(&mut store, 100, "A", 100.0);
        store.flush().unwrap();
    }
    // Reopen: all seven rows survive; none were stranded under gen N.
    let reopened = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 3).unwrap();
    assert_eq!(reopened.live_len(), 7);
    assert_eq!(
        rows(&reopened).iter().map(|row| row.0).collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5, 100]
    );
}

#[test]
fn compacting_an_empty_or_untouched_store_is_sound() {
    let mut store = Store::with_segment_rows(schema(), 0, 4).unwrap();
    store.compact().unwrap();
    assert_eq!(store.len(), 0);
    // Untouched (no tombstones, ordered): compaction is an identity on
    // the data.
    for i in 0..5i64 {
        append(&mut store, i, "A", i as f64);
    }
    let before = rows(&store);
    store.compact().unwrap();
    assert_eq!(rows(&store), before);
    assert_eq!(store.len(), 5);
}

/// A supersession with no victim is refused, and refused *before* it
/// changes anything. The commit record spells "no supersession" as the
/// coordinate `0`, so a mutation whose coordinate genuinely is 0 — only
/// reachable superseding nothing on an empty table — would write
/// evidence indistinguishable from a plain delete and lose its
/// acknowledged rows on reopen. The shape is an append; `append` is
/// that operation.
#[test]
fn a_supersession_with_no_victim_is_refused_and_changes_nothing() {
    let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
    let replacement = vec![vec![
        RowValue::I64(10),
        RowValue::Key("A"),
        RowValue::F64(1.0),
    ]];
    let error = store
        .supersede(&replacement, &[])
        .expect_err("a victimless supersession must be refused");
    assert!(
        format!("{error}").contains("use append"),
        "the refusal must name the right operation: {error}"
    );
    assert_eq!(store.len(), 0, "a refused mutation changes nothing");
    // With a victim, the same call works and the coordinate is >= 1.
    append(&mut store, 10, "A", 1.0);
    store.supersede(&replacement, &[0]).unwrap();
    assert!(store.next_sequence() >= 1);
}

#[test]
fn retaining_compaction_moves_superseded_rows_to_history() {
    // The corrections model (#75): a deleted row leaves the live set
    // but its version is retained — birth sequence, kill coordinate,
    // full cells — in history segments that latest-knowledge reads
    // never touch.
    let mut store = Store::with_segment_rows(schema(), 0, 100).unwrap();
    append(&mut store, 10, "A", 1.0); // id 0
    append(&mut store, 20, "B", 2.0); // id 1
    append(&mut store, 30, "A", 3.0); // id 2
    store.tombstone(&[0]).unwrap(); // stamped at watermark 3
    append(&mut store, 40, "B", 4.0); // id 3
    store.tombstone(&[2]).unwrap(); // stamped at watermark 4
    store.compact().unwrap();
    // Live reads: unchanged semantics, history invisible.
    assert_eq!(
        rows(&store),
        [(20, "B".to_owned(), 2.0), (40, "B".to_owned(), 4.0)]
    );
    assert_eq!(store.snapshot().unwrap().len(), 1);
    // The dead rows live on, addressed by sequence alone: births are
    // their virtual-era row ids, kills the watermark each delete
    // landed at, cells intact, merge-ordered (ts 10 before ts 30).
    let history = store.history();
    assert_eq!(history.len(), 1);
    assert_eq!(
        history[0].sequence_info(),
        &SequenceInfo::Explicit(vec![0, 2])
    );
    assert_eq!(history[0].superseded(), Some(&[3, 4][..]));
    let Column::Numeric(NumericData::I64(ts)) = &history[0].batch().columns()[0] else {
        panic!("ts type")
    };
    assert_eq!(ts.values().as_slice(), &[10, 30]);
    // The live rows diverged with their birth sequences preserved.
    let views = store.snapshot().unwrap();
    assert_eq!(
        views[0].segment.sequence_info(),
        &SequenceInfo::Explicit(vec![1, 3])
    );
    // A second round accumulates history; it never rewrites what an
    // earlier compaction retained.
    store.tombstone(&[0]).unwrap(); // ts 20, birth 1, stamped at 4
    store.compact().unwrap();
    let history = store.history();
    assert_eq!(history.len(), 2);
    assert_eq!(
        history[0].sequence_info(),
        &SequenceInfo::Explicit(vec![0, 2])
    );
    assert_eq!(history[1].sequence_info(), &SequenceInfo::Explicit(vec![1]));
    assert_eq!(history[1].superseded(), Some(&[4][..]));
}

#[test]
fn an_ordered_untombstoned_table_stays_virtual_through_compaction() {
    let mut store = Store::with_segment_rows(schema(), 0, 3).unwrap();
    for i in 0..7i64 {
        append(&mut store, i, "A", i as f64);
    }
    store.compact().unwrap();
    // Nothing retained, nothing moved: no history, still virtual.
    assert!(store.history().is_empty());
    let views = store.snapshot().unwrap();
    assert!(views
        .iter()
        .all(|view| view.segment.sequence_info() == &SequenceInfo::RowIds));
    // But mere disorder — no delete anywhere — moves row ids, and
    // moved ids diverge the table: birth sequences freeze as they
    // were while ids renumber under the sort.
    append(&mut store, 3, "A", 99.0); // late arrival: id 7, sequence 7
    store.compact().unwrap();
    assert!(store.history().is_empty());
    let sequences: Vec<u64> = store
        .snapshot()
        .unwrap()
        .iter()
        .flat_map(|view| {
            (0..view.segment.batch().num_rows())
                .map(|row| view.segment.sequence_at(row))
                .collect::<Vec<_>>()
        })
        .collect();
    // The late row sorts into the middle carrying its birth sequence.
    assert_eq!(sequences, [0, 1, 2, 3, 7, 4, 5, 6]);
}

#[test]
fn history_survives_reopen_and_unlisted_strays_are_invisible() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
            for i in 0..5i64 {
                append(&mut store, i, "A", i as f64);
            }
            store.tombstone(&[1, 3]).unwrap(); // one event, stamped at 5
            store.compact().unwrap();
        }
        // Plant a stray: a hist- file the manifest never named — a
        // crashed compaction's leftover.
        {
            let donor = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100)
                .unwrap()
                .history()[0]
                .clone();
            backend
                .write(
                    "hist-0000009999.tlyseg",
                    &storage_lite::encode_segment(&donor),
                )
                .unwrap();
        }
        let mut store =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
        // The listed history came back whole; the stray was not loaded.
        let history = store.history();
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].sequence_info(),
            &SequenceInfo::Explicit(vec![1, 3])
        );
        assert_eq!(history[0].superseded(), Some(&[5, 5][..]));
        assert_eq!(store.live_len(), 3);
        // The next compaction pre-cleans the stray and keeps — never
        // rewrites — the listed files.
        store.tombstone(&[0]).unwrap();
        store.compact().unwrap();
        let names = backend.list().unwrap();
        assert!(
            !names.contains(&"hist-0000009999.tlyseg".to_owned()),
            "{names:?}"
        );
        assert!(
            names.contains(&"hist-0000000000.tlyseg".to_owned()),
            "{names:?}"
        );
        assert!(
            names.contains(&"hist-0000000001.tlyseg".to_owned()),
            "{names:?}"
        );
        let reopened =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
        assert_eq!(reopened.history().len(), 2);
        assert_eq!(reopened.live_len(), 2);
    });
}

#[test]
fn nulls_survive_compaction() {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("y", ColumnType::F64, true),
    ]);
    let mut store = Store::with_segment_rows(schema, 0, 2).unwrap();
    store
        .append(&[RowValue::I64(2), RowValue::F64(1.0)])
        .unwrap();
    store.append(&[RowValue::I64(1), RowValue::Null]).unwrap();
    store.compact().unwrap();
    let views = store.snapshot().unwrap();
    let Column::Numeric(NumericData::F64(y)) = &views[0].segment.batch().columns()[1] else {
        panic!("y type")
    };
    // Sorted: the null row (ts 1) now comes first, still null.
    assert!(!y.is_valid(0));
    assert!(y.is_valid(1));
    assert_eq!(y.values().as_slice()[1], 1.0);
}
