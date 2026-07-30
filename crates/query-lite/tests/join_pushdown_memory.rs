//! A join gathers only the dimension columns the query reads (#81).
//!
//! Every gathered attribute becomes a full column at *fact*
//! cardinality — a dimension of eight attributes joined against a
//! hundred thousand rows is eight hundred thousand cells, however many
//! of them the SELECT list wanted. So the claim is about allocation,
//! and this measures allocation: the same join, reading one attribute
//! or all of them, under a counting allocator.

mod common;

use arrow_lite::{ColumnType, Field, Schema};
use common::peak_of;
use query_lite::{execute_join, plan, Registry};
use storage_lite::{RowValue, SegmentHandle, Store};

#[global_allocator]
static ALLOCATOR: common::Counting = common::Counting;

const ROWS: i64 = 100_000;
/// Attribute columns on the dimension, beyond its key.
const ATTRIBUTES: usize = 8;

#[test]
fn a_join_gathers_only_the_columns_the_query_reads() {
    let fact_schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("x", ColumnType::F64, false),
    ]);
    let mut attributes: Vec<Field> = vec![
        Field::new("id", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
    ];
    for column in 0..ATTRIBUTES {
        attributes.push(Field::new(format!("a{column}"), ColumnType::F64, false));
    }
    let dimension_schema = Schema::new(attributes);

    let mut fact = Store::with_segment_rows(fact_schema.clone(), 0, 8192).unwrap();
    for i in 0..ROWS {
        fact.append(&[
            RowValue::I64(i),
            RowValue::Key(["A", "B", "C", "D"][(i % 4) as usize]),
            RowValue::F64(i as f64),
        ])
        .unwrap();
    }
    let mut dimension = Store::with_segment_rows(dimension_schema.clone(), 0, 8).unwrap();
    for (id, sym) in ["A", "B", "C", "D"].iter().enumerate() {
        let mut row = vec![RowValue::I64(id as i64), RowValue::Key(sym)];
        for column in 0..ATTRIBUTES {
            row.push(RowValue::F64((id * 10 + column) as f64));
        }
        dimension.append(&row).unwrap();
    }
    let fact_views: Vec<SegmentHandle> = fact.snapshot().unwrap();
    let dimension_views: Vec<SegmentHandle> = dimension.snapshot().unwrap();
    let registry = Registry::new();
    let run = |sql: &str| {
        let plan = plan(sql).unwrap();
        let output = execute_join(
            &fact_schema,
            &fact_views,
            &dimension_schema,
            &dimension_views,
            &plan,
            &registry,
        )
        .unwrap();
        std::hint::black_box(output.num_rows());
    };
    let join = "FROM fact JOIN dim ON fact.sym = dim.sym";
    let all: String = (0..ATTRIBUTES)
        .map(|column| format!("a{column}"))
        .collect::<Vec<_>>()
        .join(", ");
    run(&format!("SELECT ts, a0 {join}")); // warm

    let one = peak_of(|| run(&format!("SELECT ts, a0 {join}")));
    let every = peak_of(|| run(&format!("SELECT ts, {all} {join}")));
    // Eight attributes cost about eight times one — the gather is the
    // dominant term, and it is per referenced column.
    assert!(
        every > one * 4,
        "reading {ATTRIBUTES} attributes ({every} bytes) should cost far more \
         than reading one ({one} bytes)"
    );
    // A column read only by the WHERE is still gathered — it has to be
    // — but the seven the query never mentions are not.
    let filtered = peak_of(|| run(&format!("SELECT ts {join} WHERE a3 > 1.0")));
    assert!(
        filtered < one * 2,
        "filtering on one unprojected attribute ({filtered} bytes) should cost \
         about what projecting one does ({one} bytes)"
    );
    // And the join that reads no attribute at all pays for none.
    let none = peak_of(|| run(&format!("SELECT ts {join}")));
    assert!(
        none < one,
        "a join reading no attribute ({none} bytes) should cost less than one \
         that reads one ({one} bytes)"
    );
}
