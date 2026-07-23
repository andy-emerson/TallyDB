//! Compaction spec tests: the tests are the spec for what "resolved at
//! the next compaction" means — tombstones gone, order restored, row
//! ids contiguous again, durably and crash-safely.

use arrow_lite::{Column, ColumnType, Field, NumericData, Schema};
use std::sync::Arc;
use storage_lite::{FsBackend, MemBackend, RowValue, StorageBackend, Store};

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
