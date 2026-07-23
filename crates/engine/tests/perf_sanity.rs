//! Perf sanity, not a benchmark: rough throughput numbers for the record,
//! so a regression that costs an order of magnitude is noticed at the
//! increment where it happened. Run explicitly, in release mode:
//!
//! ```text
//! cargo test -p engine --release --test perf_sanity -- --ignored --nocapture
//! ```
//!
//! The numbers printed are Observed evidence (one machine, one run) —
//! cite them with their run, never as stable facts. Real measurement
//! (ratios against a named peer on the corpus) arrives with the corpus
//! at M2.2.

use arrow_lite::{ColumnType, Field, Schema};
use engine::{RowValue, Table};
use std::time::Instant;

const ROWS: i64 = 1_000_000;

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
        Field::new("y", ColumnType::F64, false),
    ])
}

/// Deterministic fixture values, cheap enough not to dominate timing.
fn row(i: i64) -> [RowValue<'static>; 4] {
    const SYMBOLS: [&str; 8] = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG", "META", "NFLX",
    ];
    let x = (i % 1000) as f64 * 0.25;
    [
        RowValue::I64(i),
        RowValue::Key(SYMBOLS[(i % 8) as usize]),
        RowValue::F64(x),
        RowValue::F64(2.0 * x + (i % 7) as f64),
    ]
}

#[test]
#[ignore = "perf sanity — run explicitly in release mode"]
fn ingest_and_query_throughput() {
    let mut table = Table::new("trades", schema(), "ts").unwrap();

    let start = Instant::now();
    for i in 0..ROWS {
        table.append(&row(i)).unwrap();
    }
    let ingest = start.elapsed();

    let start = Instant::now();
    let output = table.query("SELECT ts, sym, x, y FROM trades").unwrap();
    let passthrough = start.elapsed();
    assert_eq!(output.num_rows(), ROWS as usize);

    let start = Instant::now();
    let output = table
        .query(
            "SELECT regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS beta FROM trades",
        )
        .unwrap();
    let regression = start.elapsed();
    assert_eq!(output.num_rows(), ROWS as usize);

    println!(
        "ingest: {ROWS} rows in {ingest:.2?} ({:.1}M rows/s)",
        ROWS as f64 / ingest.as_secs_f64() / 1e6
    );
    println!(
        "passthrough query: {passthrough:.2?} ({:.1}M rows/s)",
        ROWS as f64 / passthrough.as_secs_f64() / 1e6
    );
    println!(
        "rolling regression (20-row windows, 8 partitions): {regression:.2?} \
         ({:.2}M windows/s)",
        ROWS as f64 / regression.as_secs_f64() / 1e6
    );
}
