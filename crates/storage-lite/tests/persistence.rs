//! Persistence spec tests: what a persistent store promises across a
//! close-and-reopen, and how it fails when the stored bytes are wrong.
//! Run against both backends — the contract, not the filesystem, is the
//! spec.

use arrow_lite::{Column, ColumnType, Field, NumericData, Schema};
use std::sync::Arc;
use storage_lite::{
    encode_segment, FsBackend, IoError, MemBackend, RowValue, StorageBackend, StorageError, Store,
};

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

fn ts_values(store: &Store) -> Vec<i64> {
    store
        .snapshot()
        .unwrap()
        .iter()
        .flat_map(|segment| {
            let Column::Numeric(NumericData::I64(ts)) = &segment.batch().columns()[0] else {
                panic!("ts type")
            };
            ts.values().as_slice().to_vec()
        })
        .collect()
}

fn each_backend(test: impl Fn(Arc<dyn StorageBackend>)) {
    test(Arc::new(MemBackend::new()));
    let dir = std::env::temp_dir().join(format!(
        "tallydb-persist-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    test(Arc::new(FsBackend::new(&dir).unwrap()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reopen_sees_exactly_the_flushed_rows() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 4).unwrap();
            append_n(&mut store, 0..10); // 8 rows auto-flushed, 2 live
            assert_eq!(store.len(), 10);
        } // dropped without a final flush — the live rows are gone
        let mut store =
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 4).unwrap();
        assert_eq!(store.len(), 8, "unflushed rows do not survive");
        assert_eq!(ts_values(&store), (0..8).collect::<Vec<_>>());
        // Row ids continue where the flushed data ended (#1).
        let id = store
            .append(&[RowValue::I64(99), RowValue::Key("A"), RowValue::F64(0.0)])
            .unwrap();
        assert_eq!(id, 8);
    });
}

#[test]
fn explicit_flush_makes_everything_durable() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
            append_n(&mut store, 0..7);
            store.flush().unwrap();
        }
        let store = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 100).unwrap();
        assert_eq!(store.len(), 7);
        assert_eq!(ts_values(&store), (0..7).collect::<Vec<_>>());
        assert_eq!(store.segment_count(), 1);
    });
}

#[test]
fn reopened_data_is_bit_identical() {
    each_backend(|backend| {
        let before: Vec<Vec<u8>>;
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 3).unwrap();
            append_n(&mut store, 0..9);
            before = store
                .snapshot()
                .unwrap()
                .iter()
                .map(|segment| encode_segment(segment))
                .collect();
        }
        let store = Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 3).unwrap();
        let after: Vec<Vec<u8>> = store
            .snapshot()
            .unwrap()
            .iter()
            .map(|segment| encode_segment(segment))
            .collect();
        assert_eq!(before, after);
    });
}

#[test]
fn schema_disagreement_is_refused_at_open() {
    each_backend(|backend| {
        Store::persistent(backend.clone(), schema(), 0).unwrap();
        // Different column type.
        let other = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::I64, false),
        ]);
        assert!(matches!(
            Store::persistent(backend.clone(), other, 0),
            Err(StorageError::SchemaMismatch { .. })
        ));
        // Same schema, different ordering key: `ts` stays a valid choice
        // of i64 NOT NULL column, but disagrees with the manifest.
        let reordered = Schema::new(vec![
            Field::new("other_ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ]);
        assert!(matches!(
            Store::persistent(backend.clone(), reordered, 0),
            Err(StorageError::SchemaMismatch { .. })
        ));
    });
}

#[test]
fn missing_segment_is_loud() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2).unwrap();
            append_n(&mut store, 0..6); // three segments
        }
        // Remove the middle segment: rows 2..4 vanish from the backend.
        let victim = backend
            .list()
            .unwrap()
            .into_iter()
            .find(|name| name.contains("seg-") && name.contains("2"))
            .unwrap();
        backend.remove(&victim).unwrap();
        assert_eq!(
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2)
                .err()
                .unwrap(),
            StorageError::MissingRows { expected_base: 2 }
        );
    });
}

#[test]
fn corrupt_segment_is_loud() {
    each_backend(|backend| {
        {
            let mut store =
                Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2).unwrap();
            append_n(&mut store, 0..2);
        }
        let name = backend
            .list()
            .unwrap()
            .into_iter()
            .find(|name| name.starts_with("seg-"))
            .unwrap();
        let mut bytes = backend.read(&name).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        backend.write(&name, &bytes).unwrap();
        assert!(matches!(
            Store::persistent_with_segment_rows(backend.clone(), schema(), 0, 2),
            Err(StorageError::Format(_))
        ));
    });
}

#[test]
fn in_memory_stores_never_touch_a_backend() {
    // A plain Store with no backend still works exactly as before —
    // persistence is opt-in, not a tax.
    let mut store = Store::with_segment_rows(schema(), 0, 2).unwrap();
    append_n(&mut store, 0..5);
    assert_eq!(store.len(), 5);
    assert_eq!(ts_values(&store), vec![0, 1, 2, 3, 4]);
}

#[test]
fn backend_read_errors_surface() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemBackend::new());
    // A manifest that isn't a segment file at all.
    backend.write("table.tlym", b"garbage").unwrap();
    assert!(matches!(
        Store::persistent(backend.clone(), schema(), 0),
        Err(StorageError::Format(_))
    ));
    // IoError conversion sanity.
    assert_eq!(
        StorageError::from(IoError::NotFound("x".into())),
        StorageError::Io(IoError::NotFound("x".into()))
    );
}
