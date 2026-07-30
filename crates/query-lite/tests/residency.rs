//! The query-side residency claim (ruled 2026-07-30): zone-map pruning
//! runs on segment-handle metadata *before* the fault, so a pruned
//! segment's file is never read — pruning saves I/O, not just
//! evaluation. Observed from outside by counting backend reads.

use arrow_lite::{ColumnType, Field, Schema};
use query_lite::{execute, plan, Registry};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use storage_lite::{
    IoError, LogWriter, MemBackend, RowValue, StorageBackend, Store, StoreOptions, WalSync,
};

struct CountingBackend {
    inner: Arc<dyn StorageBackend>,
    reads: Mutex<HashMap<String, usize>>,
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

#[test]
fn zone_pruning_never_reads_a_pruned_segments_file() {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ]);
    let counting = Arc::new(CountingBackend {
        inner: Arc::new(MemBackend::new()),
        reads: Mutex::new(HashMap::new()),
    });
    {
        let backend: Arc<dyn StorageBackend> = counting.clone();
        let mut store = Store::persistent_with(
            backend,
            schema.clone(),
            0,
            StoreOptions {
                segment_rows: Some(4),
                wal_sync: WalSync::Off,
                ..StoreOptions::default()
            },
        )
        .unwrap();
        for ts in 0..12 {
            store
                .append(&[
                    RowValue::I64(ts),
                    RowValue::Key("A"),
                    RowValue::F64(ts as f64),
                ])
                .unwrap();
        }
        store.flush().unwrap();
    }
    counting.reads.lock().unwrap().clear();
    let backend: Arc<dyn StorageBackend> = counting.clone();
    let store = Store::persistent_with(
        backend,
        schema.clone(),
        0,
        StoreOptions {
            segment_rows: Some(4),
            wal_sync: WalSync::Off,
            ..StoreOptions::default()
        },
    )
    .unwrap();
    // Three segments: ts 0..4, 4..8, 8..12. The predicate can only
    // match the last; the first two prune on their zone maps.
    let handles = store.snapshot().unwrap();
    let output = execute(
        &schema,
        &handles,
        &plan("SELECT x FROM t WHERE ts >= 8").unwrap(),
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(output.num_rows(), 4, "the answer is right");
    let reads = counting.reads.lock().unwrap();
    let segment_reads: Vec<(&String, &usize)> = reads
        .iter()
        .filter(|(name, _)| name.ends_with(".tlyseg"))
        .collect();
    assert_eq!(
        segment_reads.len(),
        1,
        "exactly one segment file was read: {segment_reads:?}"
    );
    assert!(
        segment_reads[0].0.ends_with("-00000000000000000008.tlyseg"),
        "and it was the matching one: {segment_reads:?}"
    );
}
