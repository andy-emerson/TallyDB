//! Bucketed grouping streams: its live state is the open bucket, not
//! the result (F1 = d, ruled 2026-07-29).
//!
//! The ruling chose monotone integer arithmetic on the ordering key
//! over general `GROUP BY` expressions *because* monotonicity lets the
//! grouping stream — once a bucket is left it cannot come back, so
//! every group inside it can be closed there and then. That is a claim
//! about memory, so this measures memory.
//!
//! The comparison is a ratio against the same query over the same rows
//! with the ordering broken, which forces the hash path: an absolute
//! byte count would move with allocator and platform, while the ratio
//! is the property being claimed. Breaking the order changes nothing
//! else — same rows, same buckets, same answer — so the difference is
//! the path and only the path.

mod common;

use arrow_lite::{ColumnType, Field, Schema};
use common::peak_of;
use query_lite::{execute, plan, Registry};
use storage_lite::{RowValue, SegmentHandle, Store};

#[global_allocator]
static ALLOCATOR: common::Counting = common::Counting;

/// Rows, and the bucket width: 200,000 rows over 20,000 buckets, so
/// there are far more groups than the eight symbols inside any one of
/// them — which is the shape the two paths differ on.
const ROWS: i64 = 200_000;
const WIDTH: i64 = 10;
const SYMBOLS: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ])
}

/// `ordered`: timestamps ascending, so buckets arrive in order and the
/// grouping streams. Otherwise every row's timestamp is reflected about
/// the midpoint, which leaves the same multiset of rows and the same
/// buckets but arriving descending — so the segments are not ordered
/// and the same query must take the hash path.
fn store(ordered: bool) -> Store {
    let mut store = Store::with_segment_rows(schema(), 0, 8192).unwrap();
    for row in 0..ROWS {
        let ts = if ordered { row } else { ROWS - 1 - row };
        store
            .append(&[
                RowValue::I64(ts),
                RowValue::Key(SYMBOLS[(row % 8) as usize]),
                RowValue::F64(row as f64),
            ])
            .unwrap();
    }
    store
}

#[test]
fn a_bucketed_grouping_holds_the_open_bucket_not_the_whole_result() {
    let registry = Registry::new();
    let schema = schema();
    let run = |views: &[SegmentHandle], sql: &str| {
        let plan = plan(sql).unwrap();
        let output = execute(&schema, views, &plan, &registry).unwrap();
        std::hint::black_box(output.num_rows())
    };
    let ordered: Vec<SegmentHandle> = store(true).snapshot().unwrap();
    let disordered: Vec<SegmentHandle> = store(false).snapshot().unwrap();

    // The bar query: per symbol, per bucket, four aggregates. Every
    // group's accumulators are live at once on the hash path; only the
    // open bucket's eight are, streaming.
    let sql = &format!(
        "SELECT sym, ts / {WIDTH} AS bar, count(*) AS n, sum(x) AS s, \
         first(x) AS o, last(x) AS c FROM t GROUP BY sym, ts / {WIDTH}"
    );
    run(&ordered, sql); // warm

    let rows_ordered = run(&ordered, sql);
    let rows_disordered = run(&disordered, sql);
    assert_eq!(
        rows_ordered, rows_disordered,
        "both paths must produce the same groups"
    );

    let streaming = peak_of(|| {
        run(&ordered, sql);
    });
    let hashing = peak_of(|| {
        run(&disordered, sql);
    });
    // What streaming removes is the ACCUMULATOR state: on the hash
    // path every group's accumulators are live at once, streaming only
    // the open bucket's. What it does not remove is the result, nor the
    // group keys retained to label it — both inherently one per group.
    // So the ratio is bounded well below the group-count ratio, and the
    // threshold here is deliberately loose enough to survive allocator
    // differences while still failing if the streaming path stops being
    // taken (which would make the two equal).
    //
    // Measured 2026-07-30 on this fixture: 40.4 MB streaming vs 66.7 MB
    // hashing over 160,000 groups — a ratio of about 1.65.
    assert!(
        hashing > streaming + streaming / 4,
        "streaming should hold materially less than hashing: {streaming} vs {hashing} \
         bytes over {rows_ordered} groups"
    );
    println!(
        "streaming {streaming} bytes vs hashing {hashing} bytes over {rows_ordered} groups \
         (ratio {:.2})",
        hashing as f64 / streaming as f64
    );
}
