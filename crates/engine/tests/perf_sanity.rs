//! Perf sanity, not a benchmark: rough throughput numbers for the record,
//! so a regression that costs an order of magnitude is noticed at the
//! increment where it happened. Run explicitly, in release mode:
//!
//! ```text
//! cargo test -p engine --release --test perf_sanity -- --ignored --nocapture
//! ```
//!
//! The numbers printed are Observed evidence (one machine, one run) —
//! cite them with their run, never as stable facts. Measurement against
//! named peers lives in `tests/m2_compute_latency_bench.py`, which
//! reports ratios against both NumPy over TallyDB's own export and the
//! DuckDB+NumPy stack; this file is only a smoke check that the shapes
//! stay in their expected order of magnitude.

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

/// The #83 scaling claim, measured: refresh cost is proportional to
/// the arriving batch, not the table. Two tables, 4x apart in size,
/// each fully folded, then given the SAME small batch: the second
/// refresh must not scale with the table. The contrast column — a full
/// recompute of the definition — IS proportional to the table, which
/// is what the maintained view exists to avoid.
#[test]
#[ignore = "perf sanity — run explicitly in release mode"]
fn view_refresh_scales_with_the_batch_not_the_table() {
    use engine::Database;
    let batch = 2_000i64;
    let mut costs = Vec::new();
    for &rows in &[250_000i64, 1_000_000] {
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", schema(), "ts", 8_192).unwrap())
            .unwrap();
        for i in 0..rows {
            db.append("trades", &row(i)).unwrap();
        }
        db.create_materialized_view(
            "bars",
            "SELECT sym, ts / 1000 AS bar, count(*) AS n, sum(x) AS s, \
             min(x) AS lo, max(x) AS hi FROM trades GROUP BY sym, ts / 1000",
        )
        .unwrap();
        db.refresh_view("bars").unwrap();
        for i in rows..rows + batch {
            db.append("trades", &row(i)).unwrap();
        }
        // The staleness-premium check first (the read-semantics
        // ruling's honesty condition on #83): a query against the STALE view
        // (union read: materialized + live fold of the 2k tail) vs the
        // same query after refresh. The ruling's bet is that the tail-
        // bounded union costs little over the fresh read.
        let start = Instant::now();
        let stale = db.query("SELECT bar, s FROM bars").unwrap();
        let stale_read = start.elapsed();
        let start = Instant::now();
        let folded = db.refresh_view("bars").unwrap();
        let refresh = start.elapsed();
        let start = Instant::now();
        let fresh = db.query("SELECT bar, s FROM bars").unwrap();
        let fresh_read = start.elapsed();
        assert_eq!(stale.num_rows(), fresh.num_rows());
        println!("rows {rows}: stale union read {stale_read:?} vs fresh read {fresh_read:?}");
        let start = Instant::now();
        let _ = db
            .table("trades")
            .unwrap()
            .query(db.view("bars").unwrap().sql())
            .unwrap();
        let recompute = start.elapsed();
        println!(
            "rows {rows}: batch refresh {refresh:?} ({folded} buckets) vs full recompute {recompute:?}"
        );
        costs.push(refresh);
    }
    let ratio = costs[1].as_secs_f64() / costs[0].as_secs_f64().max(1e-9);
    println!("refresh cost ratio at 4x the table: {ratio:.2}");
    assert!(
        ratio < 2.5,
        "refresh scaled with the table (x4 table -> x{ratio:.1} refresh)"
    );
}

/// Tranche 2's repair claim, measured: correcting one row of a RUNNING
/// view re-folds one hidden bucket, priced against the full recompute
/// the partials representation exists to avoid (the O(suffix) rewrite
/// never happens because no suffix is stored). Same run, same table —
/// the ratio is the evidence, the absolutes are just the record.
#[test]
#[ignore = "perf sanity — run explicitly in release mode"]
fn running_correction_repairs_one_bucket_not_the_answer() {
    use engine::Database;
    let rows = 1_000_000i64;
    let mut db = Database::new();
    db.add_table(Table::with_segment_rows("trades", schema(), "ts", 8_192).unwrap())
        .unwrap();
    for i in 0..rows {
        db.append("trades", &row(i)).unwrap();
    }
    db.create_materialized_view(
        "totals",
        "SELECT sym, count(*) AS n, sum(x) AS s, avg(x) AS a, \
         min(x) AS lo, max(x) AS hi FROM trades GROUP BY sym",
    )
    .unwrap();
    db.refresh_view("totals").unwrap();
    db.mutate("UPDATE trades SET x = 12345.0 WHERE ts = 500000")
        .unwrap();
    let start = Instant::now();
    let folded = db.refresh_view("totals").unwrap();
    let repair = start.elapsed();
    assert_eq!(folded, 1, "a one-row correction re-folds one hidden bucket");
    let start = Instant::now();
    let _ = db
        .table("trades")
        .unwrap()
        .query(db.view("totals").unwrap().sql())
        .unwrap();
    let recompute = start.elapsed();
    let ratio = repair.as_secs_f64() / recompute.as_secs_f64().max(1e-9);
    println!(
        "one-row correction over {rows} rows: repair {repair:?} (1 bucket) \
         vs full recompute {recompute:?} — ratio {ratio:.3}"
    );
    assert!(
        ratio < 0.5,
        "one-bucket repair cost approached full recompute (ratio {ratio:.2})"
    );
}

/// Tranche 2's read claim, measured: a CUMULATIVE view's ranged read
/// prices the requested range (boundary combine over ~span/width
/// partial rows + assembly over the suffix), not the table — against
/// the full read of the same view in the same run, which recomputes
/// every window over every row.
#[test]
#[ignore = "perf sanity — run explicitly in release mode"]
fn cumulative_range_read_prices_the_range_not_the_table() {
    use engine::Database;
    let rows = 1_000_000i64;
    let floor = rows - 10_000;
    let mut db = Database::new();
    db.add_table(Table::with_segment_rows("trades", schema(), "ts", 8_192).unwrap())
        .unwrap();
    for i in 0..rows {
        db.append("trades", &row(i)).unwrap();
    }
    db.create_materialized_view(
        "cum",
        "SELECT ts, sym, \
         sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cs, \
         avg(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ca \
         FROM trades",
    )
    .unwrap();
    db.refresh_view("cum").unwrap();
    let start = Instant::now();
    let ranged = db
        .query(&format!(
            "SELECT ts, sym, cs, ca FROM cum WHERE ts >= {floor}"
        ))
        .unwrap();
    let ranged_read = start.elapsed();
    assert_eq!(ranged.num_rows(), 10_000);
    let start = Instant::now();
    let full = db.query("SELECT ts, sym, cs, ca FROM cum").unwrap();
    let full_read = start.elapsed();
    assert_eq!(full.num_rows(), rows as usize);
    let ratio = ranged_read.as_secs_f64() / full_read.as_secs_f64().max(1e-9);
    println!(
        "cumulative read of the last 10k of {rows} rows: ranged {ranged_read:?} \
         vs full recompute {full_read:?} — ratio {ratio:.3}"
    );
    assert!(
        ratio < 0.5,
        "the ranged read cost approached the full recompute (ratio {ratio:.2})"
    );
}
