//! F4: cross-process readers — a read-only store over a directory
//! another handle writes.
//!
//! The contract under test: a reader sees the **durable prefix,
//! consistently** — flushed segments and committed deletes, never the
//! writer's buffer or WAL, and never a mutation's torn middle — and a
//! refresh advances the view. Tests run two `Store`s over one real
//! directory, which is the cross-process seam minus the process
//! boundary (the directory lock itself is per-file-handle, so the
//! two-writer refusal is exercised in-process too, in `io`'s tests).

use arrow_lite::{Column, ColumnType, Field, NumericData, Schema};
use std::sync::Arc;
use storage_lite::{FsBackend, RowValue, StorageBackend, StorageError, Store, WalSync};

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
                RowValue::Key("A"),
                RowValue::F64(i as f64),
            ])
            .unwrap();
    }
}

fn live_ts(store: &Store) -> Vec<i64> {
    store
        .snapshot()
        .unwrap()
        .iter()
        .flat_map(|view| {
            let view = view.view().unwrap();
            let Column::Numeric(NumericData::I64(ts)) = &view.segment.batch().columns()[0] else {
                panic!("ts type")
            };
            (0..view.segment.batch().num_rows())
                .filter(|&row| view.is_live(row))
                .map(|row| ts.values().as_slice()[row])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tallydb-f4-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn a_reader_sees_the_flushed_prefix_and_refresh_advances_it() {
    let dir = fresh_dir("prefix");
    let writer_backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    let mut writer = Store::persistent_with_segment_rows(writer_backend, schema(), 0, 100).unwrap();
    append_n(&mut writer, 0..5);
    writer.flush().unwrap();
    append_n(&mut writer, 5..8); // buffered: invisible to readers

    let reader_backend: Arc<dyn StorageBackend> =
        Arc::new(FsBackend::open_read_only(&dir).unwrap());
    let mut reader = Store::open_read_only(reader_backend).unwrap();
    assert_eq!(live_ts(&reader), vec![0, 1, 2, 3, 4]);
    assert_eq!(reader.next_sequence(), 5);

    // The writer flushes; the reader's existing view must NOT move
    // (snapshot stability), and a refresh must.
    writer.flush().unwrap();
    assert_eq!(live_ts(&reader), vec![0, 1, 2, 3, 4]);
    reader.refresh().unwrap();
    assert_eq!(live_ts(&reader), (0..8).collect::<Vec<_>>());

    // A snapshot minted before a refresh keeps answering.
    let held = reader.snapshot().unwrap();
    append_n(&mut writer, 8..9);
    writer.flush().unwrap();
    reader.refresh().unwrap();
    assert_eq!(live_ts(&reader), (0..9).collect::<Vec<_>>());
    let held_rows: usize = held.iter().map(|view| view.live_rows()).sum();
    assert_eq!(held_rows, 8);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_reader_refuses_every_mutation() {
    let dir = fresh_dir("refuse");
    {
        let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
        let mut writer = Store::persistent_with_segment_rows(backend, schema(), 0, 100).unwrap();
        append_n(&mut writer, 0..3);
        writer.flush().unwrap();
    }
    let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::open_read_only(&dir).unwrap());
    let mut reader = Store::open_read_only(backend).unwrap();
    let refused = |result: Result<u64, StorageError>| {
        assert!(
            matches!(result, Err(StorageError::Misuse(_))),
            "must refuse as misuse"
        );
    };
    refused(reader.append(&[RowValue::I64(9), RowValue::Key("A"), RowValue::F64(9.0)]));
    refused(reader.tombstone(&[0]));
    refused(reader.supersede(
        &[vec![
            RowValue::I64(0),
            RowValue::Key("A"),
            RowValue::F64(1.0),
        ]],
        &[0],
    ));
    assert!(matches!(reader.compact(), Err(StorageError::Misuse(_))));
    assert!(matches!(reader.flush(), Err(StorageError::Misuse(_))));
    // And a writer store refuses refresh — it sees its own state.
    let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    let mut writer = Store::open_existing(backend, storage_lite::StoreOptions::default()).unwrap();
    assert!(matches!(writer.refresh(), Err(StorageError::Misuse(_))));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_committed_delete_is_visible_and_a_buffered_row_is_not() {
    let dir = fresh_dir("delete");
    let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    let mut writer = Store::persistent_with_segment_rows(backend, schema(), 0, 100).unwrap();
    append_n(&mut writer, 0..5);
    writer.flush().unwrap();
    writer.tombstone(&[1, 3]).unwrap(); // flush-first, then the log

    let reader_backend: Arc<dyn StorageBackend> =
        Arc::new(FsBackend::open_read_only(&dir).unwrap());
    let reader = Store::open_read_only(reader_backend).unwrap();
    assert_eq!(live_ts(&reader), vec![0, 2, 4]);
    // The kill consumed a coordinate; the reader's watermark knows.
    assert_eq!(reader.next_sequence(), writer.next_sequence());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_unflushed_supersession_shows_the_pre_state_never_the_torn_middle() {
    // THE consistency case. An UPDATE appends replacements (WAL only,
    // under the threshold) and commits its delete log immediately. A
    // reader that applied the log without the replacements would show
    // rows missing — the torn middle. The rule: a supersession log
    // whose replacements are not flushed is skipped whole, so the
    // reader sees the complete PRE state; after the writer's flush and
    // a refresh, the complete POST state. Old-or-new, nothing between.
    let dir = fresh_dir("supersede");
    let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    let mut writer = Store::persistent_with(
        backend,
        schema(),
        0,
        storage_lite::StoreOptions {
            segment_rows: Some(100),
            wal_sync: WalSync::Group(std::time::Duration::from_secs(3600)),
            ..storage_lite::StoreOptions::default()
        },
    )
    .unwrap();
    append_n(&mut writer, 0..4);
    writer.flush().unwrap();
    // The correction: ts=1 becomes x=99. Replacement rides the WAL.
    writer
        .supersede(
            &[vec![
                RowValue::I64(1),
                RowValue::Key("A"),
                RowValue::F64(99.0),
            ]],
            &[1],
        )
        .unwrap();

    let reader_backend: Arc<dyn StorageBackend> =
        Arc::new(FsBackend::open_read_only(&dir).unwrap());
    let mut reader = Store::open_read_only(reader_backend).unwrap();
    // Pre-state: all four originals live, none replaced, none missing.
    assert_eq!(live_ts(&reader), vec![0, 1, 2, 3]);
    assert_eq!(reader.live_len(), 4);

    // The writer flushes; the refresh sees the complete post-state.
    writer.flush().unwrap();
    reader.refresh().unwrap();
    assert_eq!(live_ts(&reader), vec![0, 2, 3, 1]); // replacement appended last
    assert_eq!(reader.live_len(), 4);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn refresh_follows_a_compaction_into_the_new_generation() {
    let dir = fresh_dir("compact");
    let backend: Arc<dyn StorageBackend> = Arc::new(FsBackend::new(&dir).unwrap());
    let mut writer = Store::persistent_with_segment_rows(backend, schema(), 0, 2).unwrap();
    append_n(&mut writer, 0..6);
    writer.flush().unwrap();
    writer.tombstone(&[2]).unwrap();

    let reader_backend: Arc<dyn StorageBackend> =
        Arc::new(FsBackend::open_read_only(&dir).unwrap());
    let mut reader = Store::open_read_only(reader_backend).unwrap();
    let before = reader.snapshot().unwrap();
    assert_eq!(live_ts(&reader), vec![0, 1, 3, 4, 5]);

    // The writer compacts: new generation, old objects removed.
    writer.compact().unwrap();
    reader.refresh().unwrap();
    assert_eq!(live_ts(&reader), vec![0, 1, 3, 4, 5]);
    // The pre-compaction snapshot still answers from memory.
    let rows: usize = before.iter().map(|view| view.live_rows()).sum();
    assert_eq!(rows, 5);
    // And history survives into the reader: the tombstoned row is
    // addressable AS OF the pre-delete cut.
    let knowledge = reader.knowledge_snapshot().unwrap();
    let live_at = |cut: u64| -> usize {
        knowledge
            .as_of(cut)
            .unwrap()
            .iter()
            .map(storage_lite::SegmentHandle::live_rows)
            .sum()
    };
    assert_eq!(live_at(5), 6, "before the kill: all six");
    assert_eq!(live_at(6), 5, "at the kill: five");
    std::fs::remove_dir_all(&dir).unwrap();
}
