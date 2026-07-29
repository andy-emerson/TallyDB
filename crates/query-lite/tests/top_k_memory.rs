//! `ORDER BY … LIMIT k` must cost memory in k, not in n (#80).
//!
//! The claim is about allocation, so the check measures allocation: a
//! counting allocator records the peak bytes live above a baseline, and
//! each query is compared against **the same query without `ORDER BY`**.
//! That subtraction is the point — it removes the per-batch bookkeeping
//! every query pays and leaves the sort's own working set, which is the
//! thing top-k changes.

mod common;

use arrow_lite::{ColumnType, Field, Schema};
use common::peak_of;
use query_lite::{execute, plan, Registry};
use storage_lite::{RowValue, SegmentView, Store};

#[global_allocator]
static ALLOCATOR: common::Counting = common::Counting;

const ROWS: i64 = 200_000;

#[test]
fn a_bounded_order_by_pays_for_k_not_for_n() {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ]);
    let mut store = Store::with_segment_rows(schema.clone(), 0, 8192).unwrap();
    for i in 0..ROWS {
        // A scrambled sort key, so no path can be lucky about order.
        let x = ((i * 2_654_435_761u64 as i64) % 1_000_003) as f64;
        store
            .append(&[
                RowValue::I64(i),
                RowValue::Key(["A", "B", "C", "D"][(i % 4) as usize]),
                RowValue::F64(x),
            ])
            .unwrap();
    }
    let views: Vec<SegmentView> = store.snapshot().unwrap();
    let registry = Registry::new();
    let run = |sql: &str| {
        let plan = plan(sql).unwrap();
        let output = execute(&schema, &views, &plan, &registry).unwrap();
        std::hint::black_box(output.num_rows());
    };
    // Warm: first-call allocations (parser tables, dictionaries) are
    // not what is being measured.
    run("SELECT ts, sym, x FROM t ORDER BY x LIMIT 10");
    let sort_cost = |sql: &str, unsorted: &str| -> usize {
        let sorted = peak_of(|| run(sql));
        let plain = peak_of(|| run(unsorted));
        sorted.saturating_sub(plain)
    };

    // The full sort holds an entry per row and then materializes a
    // sorted copy of the whole result; ten rows hold ten. (Only
    // numbers can be sorted by — symbol columns are unordered labels,
    // #58 — so an f64 key is the shape to measure.)
    let unbounded = sort_cost(
        "SELECT ts, x FROM t ORDER BY x",
        "SELECT ts, x FROM t LIMIT 10",
    );
    let bounded = sort_cost(
        "SELECT ts, x FROM t ORDER BY x LIMIT 10",
        "SELECT ts, x FROM t LIMIT 10",
    );
    assert!(
        unbounded > 4_000_000,
        "sorting {ROWS} rows should cost megabytes, measured {unbounded}"
    );
    assert!(
        bounded < 8_192,
        "a ten-row answer's sort allocated {bounded} bytes"
    );

    // k is what moves it, and n is not: a thousand rows costs more than
    // ten and still a small fraction of the whole.
    let thousand = sort_cost(
        "SELECT ts, x FROM t ORDER BY x LIMIT 1000",
        "SELECT ts, x FROM t LIMIT 10",
    );
    assert!(
        thousand > bounded && thousand < unbounded / 8,
        "LIMIT 1000 measured {thousand}, against {bounded} for ten and \
         {unbounded} for all {ROWS}"
    );
}
