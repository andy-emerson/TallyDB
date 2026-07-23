//! Execution: the plan over a snapshot of segment views.
//!
//! ## The compute seam
//!
//! Window-aggregate *implementations* are not defined here. The embedder
//! (`engine`) registers them in a [`Registry`] — that is how compute
//! (BLAS/LAPACK-backed regressions) reaches SQL while `query-lite` itself
//! stays compute-free. An implementation sees plain `&[f64]` window
//! slices and returns one value per window (`None` for windows where the
//! aggregate is undefined — too few rows, degenerate inputs — which
//! surfaces as SQL `NULL`, matching standard aggregate semantics).
//!
//! ## One batch per segment
//!
//! A query runs over a storage snapshot — one [`SegmentView`] per
//! segment of one table, in append order — and produces one output batch
//! per view with any live rows, Arrow's own model for a chunked result.
//! That shape is what keeps passthrough zero-copy: each batch's
//! passthrough columns share its segment's buffers (copy-on-write
//! clones), and each batch's key columns keep their segment's own
//! dictionary (per-segment dictionaries, decision #6). Callers that want
//! a single contiguous result pay for the concatenation themselves,
//! knowingly.
//!
//! ## Tombstones and where the copies are
//!
//! A view's live mask is how deletion reaches a reader: tombstoned rows
//! simply do not exist here — not in passthrough, not in windows, not in
//! partitions. The zero-copy path survives untombstoned: a mask-free
//! view over a single segment slices the stored buffers directly, while
//! a masked view is filter-materialized once per query — a bounded
//! O(rows) gather, the price of mutation on the read side, paid only
//! where mutation actually happened. Windows that *span segments* and
//! `PARTITION BY` gather the same way they did before; for `PARTITION
//! BY` across segments, each segment's dictionary codes are first
//! remapped into a query-lifetime key space (the query-time remap
//! decision #6 accepted).

use crate::plan::{Plan, PlanItem, QueryError};
use arrow_lite::{
    Bitmap, Buffer, Column, ColumnType, Field, KeyColumn, NumericColumn, NumericData, RecordBatch,
    Schema,
};
use std::collections::HashMap;
use std::sync::Arc;
use storage_lite::SegmentView;

/// One window-aggregate implementation, registered by the embedder.
pub trait WindowAggregate: Send + Sync {
    /// Number of column arguments the function takes.
    fn arity(&self) -> usize;

    /// Evaluates one window. `args` holds one slice per argument, all the
    /// same length (the window's rows, oldest first). `Ok(None)` means
    /// the aggregate is undefined for this window and becomes SQL `NULL`;
    /// `Err` aborts the query.
    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String>;
}

/// The window-aggregate registry: SQL name → implementation.
#[derive(Clone, Default)]
pub struct Registry {
    aggregates: HashMap<String, Arc<dyn WindowAggregate>>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Registry::default()
    }

    /// Registers `aggregate` under `name` (lower-cased; last one wins).
    pub fn register(&mut self, name: &str, aggregate: Arc<dyn WindowAggregate>) {
        self.aggregates.insert(name.to_lowercase(), aggregate);
    }

    fn get(&self, name: &str) -> Option<&Arc<dyn WindowAggregate>> {
        self.aggregates.get(name)
    }
}

/// A query's result: the output schema plus one batch per segment with
/// live rows, in append order. The schema is carried explicitly so an
/// empty table still yields a well-formed (zero-batch) result.
#[derive(Debug)]
pub struct QueryOutput {
    /// Schema of every batch.
    pub schema: Schema,
    /// One batch per segment with live rows.
    pub batches: Vec<RecordBatch>,
}

impl QueryOutput {
    /// Total rows across all batches.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// Runs `plan` over `views` — one table's snapshot, in append order, all
/// sharing `schema` — resolving window functions in `registry`.
///
/// The embedder has already resolved the plan's table name to this
/// snapshot; nothing here re-checks it.
pub fn execute(
    schema: &Schema,
    views: &[SegmentView],
    plan: &Plan,
    registry: &Registry,
) -> Result<QueryOutput, QueryError> {
    // Views with no live rows contribute nothing; dropping them up front
    // means "one batch per segment" below never emits an empty batch.
    let views: Vec<&SegmentView> = views.iter().filter(|view| view.live_rows() > 0).collect();
    let mut fields = Vec::with_capacity(plan.items.len());
    let mut columns_per_view: Vec<Vec<Column>> = views.iter().map(|_| Vec::new()).collect();
    for item in &plan.items {
        let (field, columns) = match item {
            PlanItem::Column { name, alias } => {
                passthrough(schema, &views, name, alias.as_deref())?
            }
            PlanItem::WindowAgg {
                function,
                args,
                partition_by,
                order_by,
                preceding,
                alias,
            } => window_aggregate(
                schema,
                &views,
                registry,
                function,
                args,
                partition_by.as_deref(),
                order_by,
                *preceding,
                alias.as_deref(),
            )?,
        };
        fields.push(field);
        for (out, column) in columns_per_view.iter_mut().zip(columns) {
            out.push(column);
        }
    }
    let schema = Schema::new(fields);
    let batches = columns_per_view
        .into_iter()
        .map(|columns| RecordBatch::new(schema.clone(), columns))
        .collect();
    Ok(QueryOutput { schema, batches })
}

/// Looks up a column by name in the table schema.
fn resolve<'a>(schema: &'a Schema, name: &str) -> Result<(usize, &'a Field), QueryError> {
    schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == name)
        .ok_or_else(|| QueryError::UnknownColumn(name.to_owned()))
}

/// Local row indices a reader sees, in stored order.
fn live_rows<'a>(view: &'a SegmentView) -> impl Iterator<Item = usize> + 'a {
    (0..view.segment.batch().num_rows()).filter(move |&row| view.is_live(row))
}

/// Rebuilds `column` with only the mask's live rows — the bounded copy a
/// tombstoned segment costs its readers.
fn filter_column(column: &Column, view: &SegmentView) -> Column {
    let keep: Vec<usize> = live_rows(view).collect();
    let validity = |bitmap: Option<&Bitmap>| {
        bitmap.map(|bitmap| Bitmap::from_bools(keep.iter().map(|&row| bitmap.get(row))))
    };
    match column {
        Column::Numeric(NumericData::F64(numeric)) => {
            let values = numeric.values().as_slice();
            let buffer: Buffer<f64> = keep.iter().map(|&row| values[row]).collect();
            Column::Numeric(NumericData::F64(match validity(numeric.validity()) {
                Some(bitmap) => NumericColumn::new_nullable(buffer, bitmap),
                None => NumericColumn::new_non_null(buffer),
            }))
        }
        Column::Numeric(NumericData::I64(numeric)) => {
            let values = numeric.values().as_slice();
            let buffer: Buffer<i64> = keep.iter().map(|&row| values[row]).collect();
            Column::Numeric(NumericData::I64(match validity(numeric.validity()) {
                Some(bitmap) => NumericColumn::new_nullable(buffer, bitmap),
                None => NumericColumn::new_non_null(buffer),
            }))
        }
        Column::Key(keys) => {
            let codes = keys.codes().as_slice();
            let buffer: Buffer<u32> = keep.iter().map(|&row| codes[row]).collect();
            let dictionary = keys.dictionary().clone();
            Column::Key(match validity(keys.validity()) {
                Some(bitmap) => KeyColumn::new_nullable(buffer, bitmap, dictionary),
                None => KeyColumn::new_non_null(buffer, dictionary),
            })
        }
    }
}

/// A stored column, passed through — by shared handle where the view is
/// mask-free (no row data copied), filter-materialized where it isn't.
fn passthrough(
    schema: &Schema,
    views: &[&SegmentView],
    name: &str,
    alias: Option<&str>,
) -> Result<(Field, Vec<Column>), QueryError> {
    let (index, field) = resolve(schema, name)?;
    let mut out = Field::new(alias.unwrap_or(name), field.column_type(), field.nullable());
    if let Some(logical) = field.logical() {
        out = out.with_logical(logical);
    }
    let columns = views
        .iter()
        .map(|view| {
            let column = &view.segment.batch().columns()[index];
            match &view.live {
                None => column.clone(),
                Some(_) => filter_column(column, view),
            }
        })
        .collect();
    Ok((out, columns))
}

/// The window slice for row `position` in a run of rows: `preceding` rows
/// back through the current row, ragged at the start of the run.
fn window_bounds(position: usize, preceding: usize) -> (usize, usize) {
    (position.saturating_sub(preceding), position + 1)
}

/// First and last live values of the ordering key, or `None` if no live
/// rows.
fn live_ordering_bounds(view: &SegmentView) -> Option<(i64, i64)> {
    let Column::Numeric(NumericData::I64(column)) =
        &view.segment.batch().columns()[view.segment.ordering_key()]
    else {
        unreachable!("the ordering key is validated as i64 at construction")
    };
    let values = column.values().as_slice();
    let mut rows = live_rows(view);
    let first = rows.next()?;
    let last = rows.last().unwrap_or(first);
    Some((values[first], values[last]))
}

/// Checks that the snapshot is globally ordered on the window's ORDER BY
/// column: it must be the declared ordering key, each segment must be
/// internally ordered, and each boundary between live rows must be
/// non-decreasing. Checked, never assumed — a window over misordered
/// rows silently computes the wrong thing. (A segment whose ingest was
/// misordered is refused even if the offending rows are now tombstoned —
/// conservative, and resolved for good by compaction.)
fn check_order(
    views: &[&SegmentView],
    order_index: usize,
    order_by: &str,
) -> Result<(), QueryError> {
    let mut previous_last: Option<i64> = None;
    for view in views {
        if order_index != view.segment.ordering_key() {
            return Err(QueryError::Unsupported(format!(
                "ORDER BY '{order_by}' — windows order by the declared ordering key only"
            )));
        }
        if !view.segment.is_ordered() {
            return Err(QueryError::Unordered(format!(
                "ingest was not sorted on '{order_by}' (compaction restores order)"
            )));
        }
        let Some((first, last)) = live_ordering_bounds(view) else {
            continue;
        };
        if previous_last.is_some_and(|previous| first < previous) {
            return Err(QueryError::Unordered(format!(
                "ingest was not sorted on '{order_by}' across segments"
            )));
        }
        previous_last = Some(last);
    }
    Ok(())
}

/// One argument column's live values in one view: a shared slice where
/// the zero-copy path holds, a gathered copy where a mask is in the way.
enum ArgValues<'a> {
    Shared(&'a [f64]),
    Gathered(Vec<f64>),
}

impl ArgValues<'_> {
    fn as_slice(&self) -> &[f64] {
        match self {
            ArgValues::Shared(values) => values,
            ArgValues::Gathered(values) => values,
        }
    }
}

/// Per-view live `f64` values for one argument column, validated
/// (f64, no nulls among live rows).
fn argument_values<'a>(
    views: &[&'a SegmentView],
    index: usize,
    name: &str,
) -> Result<Vec<ArgValues<'a>>, QueryError> {
    let mut result = Vec::with_capacity(views.len());
    for view in views {
        let Column::Numeric(NumericData::F64(column)) = &view.segment.batch().columns()[index]
        else {
            return Err(QueryError::TypeError(format!(
                "window argument '{name}' must be f64"
            )));
        };
        let any_live_null =
            column.validity().is_some() && live_rows(view).any(|row| !column.is_valid(row));
        if any_live_null {
            return Err(QueryError::Unsupported(format!(
                "window argument '{name}' has nulls (unsupported as a window argument)"
            )));
        }
        let values = column.values().as_slice();
        result.push(match &view.live {
            None => ArgValues::Shared(values),
            Some(_) => ArgValues::Gathered(live_rows(view).map(|row| values[row]).collect()),
        });
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn window_aggregate(
    schema: &Schema,
    views: &[&SegmentView],
    registry: &Registry,
    function: &str,
    arg_names: &[String],
    partition_by: Option<&str>,
    order_by: &str,
    preceding: usize,
    alias: Option<&str>,
) -> Result<(Field, Vec<Column>), QueryError> {
    let aggregate = registry
        .get(function)
        .ok_or_else(|| QueryError::UnknownFunction(function.to_owned()))?;
    if arg_names.len() != aggregate.arity() {
        return Err(QueryError::TypeError(format!(
            "{function} takes {} arguments, got {}",
            aggregate.arity(),
            arg_names.len()
        )));
    }
    let (order_index, _) = resolve(schema, order_by)?;
    check_order(views, order_index, order_by)?;
    // args[a][v]: argument `a`'s live values in view `v`.
    let mut args: Vec<Vec<ArgValues<'_>>> = Vec::with_capacity(arg_names.len());
    for name in arg_names {
        let (index, _) = resolve(schema, name)?;
        args.push(argument_values(views, index, name)?);
    }
    // One result slot per live row, per view.
    let mut results: Vec<Vec<Option<f64>>> = views
        .iter()
        .map(|view| vec![None; view.live_rows()])
        .collect();
    match partition_by {
        None => unpartitioned(aggregate.as_ref(), &args, preceding, &mut results)?,
        Some(partition_column) => partitioned(
            schema,
            views,
            aggregate.as_ref(),
            &args,
            partition_column,
            preceding,
            &mut results,
        )?,
    }
    let name = alias.unwrap_or(function);
    let columns = results.into_iter().map(assemble_f64).collect();
    Ok((Field::new(name, ColumnType::F64, true), columns))
}

/// Unpartitioned windows run over the snapshot's live rows in append
/// order. A single mask-free view is the pure zero-copy path — every
/// window is a direct slice of the stored buffer. Otherwise each
/// argument is gathered once into contiguous scratch (windows span view
/// boundaries; the stored buffers don't) and the windows slice that.
fn unpartitioned(
    aggregate: &dyn WindowAggregate,
    args: &[Vec<ArgValues<'_>>],
    preceding: usize,
    results: &mut [Vec<Option<f64>>],
) -> Result<(), QueryError> {
    let gathered: Vec<Vec<f64>>;
    let arg_slices: Vec<&[f64]> = if args.first().is_none_or(|slices| slices.len() == 1) {
        args.iter().map(|slices| slices[0].as_slice()).collect()
    } else {
        gathered = args
            .iter()
            .map(|slices| {
                slices
                    .iter()
                    .flat_map(|values| values.as_slice().iter().copied())
                    .collect()
            })
            .collect();
        gathered.iter().map(Vec::as_slice).collect()
    };
    let mut windows: Vec<&[f64]> = Vec::with_capacity(arg_slices.len());
    let mut global = 0usize;
    for result in results.iter_mut() {
        for slot in result.iter_mut() {
            let (start, end) = window_bounds(global, preceding);
            windows.clear();
            windows.extend(arg_slices.iter().map(|values| &values[start..end]));
            *slot = aggregate.evaluate(&windows).map_err(QueryError::Compute)?;
            global += 1;
        }
    }
    Ok(())
}

/// Partitioned windows track each key separately across the whole
/// snapshot's live rows. Dictionary codes are per-segment (decision #6),
/// so each segment's codes are remapped into a query-lifetime key space
/// first; each partition's rows are then gathered into contiguous
/// scratch (they are scattered even within one segment) and results
/// scattered back to their view and live position.
fn partitioned(
    schema: &Schema,
    views: &[&SegmentView],
    aggregate: &dyn WindowAggregate,
    args: &[Vec<ArgValues<'_>>],
    partition_column: &str,
    preceding: usize,
    results: &mut [Vec<Option<f64>>],
) -> Result<(), QueryError> {
    let (index, _) = resolve(schema, partition_column)?;
    // The query-lifetime key space: value → unified code, built once per
    // distinct value per segment (cheap under low cardinality).
    let mut unified: HashMap<String, usize> = HashMap::new();
    // Per partition: scratch per argument, plus where each row came from
    // (view index, live position within the view).
    let mut scratch: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut origins: Vec<Vec<(usize, usize)>> = Vec::new();
    for (view_index, view) in views.iter().enumerate() {
        let Column::Key(keys) = &view.segment.batch().columns()[index] else {
            return Err(QueryError::TypeError(format!(
                "PARTITION BY '{partition_column}' must be a key column"
            )));
        };
        let any_live_null =
            keys.validity().is_some() && live_rows(view).any(|row| !keys.is_valid(row));
        if any_live_null {
            return Err(QueryError::Unsupported(format!(
                "PARTITION BY '{partition_column}' has nulls (unsupported as a partition key)"
            )));
        }
        let dictionary = keys.dictionary();
        let remap: Vec<usize> = (0..dictionary.len() as u32)
            .map(|code| {
                let next = unified.len();
                *unified
                    .entry(dictionary.value(code).to_owned())
                    .or_insert(next)
            })
            .collect();
        // One slot per unified code — allocated eagerly, because a code
        // can enter the unified space from a dictionary entry whose live
        // rows come later or never (a tombstoned row's key, a value seen
        // only in another segment). Codeless partitions stay empty and
        // cost nothing below.
        while scratch.len() < unified.len() {
            scratch.push(vec![Vec::new(); args.len()]);
            origins.push(Vec::new());
        }
        let codes = keys.codes().as_slice();
        for (live_position, row) in live_rows(view).enumerate() {
            let partition = remap[codes[row] as usize];
            for (argument, per_view) in args.iter().enumerate() {
                scratch[partition][argument].push(per_view[view_index].as_slice()[live_position]);
            }
            origins[partition].push((view_index, live_position));
        }
    }
    let mut windows: Vec<&[f64]> = Vec::with_capacity(args.len());
    for (values, rows) in scratch.iter().zip(&origins) {
        for (position, &(view_index, live_position)) in rows.iter().enumerate() {
            let (start, end) = window_bounds(position, preceding);
            windows.clear();
            windows.extend(values.iter().map(|argument| &argument[start..end]));
            results[view_index][live_position] =
                aggregate.evaluate(&windows).map_err(QueryError::Compute)?;
        }
    }
    Ok(())
}

/// One view's output column: nullable f64, bitmap only if a window
/// actually came back undefined.
fn assemble_f64(results: Vec<Option<f64>>) -> Column {
    let values = results.iter().map(|v| v.unwrap_or(0.0)).collect();
    let column = if results.iter().any(Option::is_none) {
        NumericColumn::new_nullable(
            values,
            Bitmap::from_bools(results.iter().map(Option::is_some)),
        )
    } else {
        NumericColumn::new_non_null(values)
    };
    Column::Numeric(NumericData::F64(column))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan;
    use storage_lite::{RowValue, Store, WriteBuffer};

    /// Mean of the first argument — enough to test frame arithmetic
    /// without any compute dependency.
    struct Mean;

    impl WindowAggregate for Mean {
        fn arity(&self) -> usize {
            1
        }
        fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
            let window = args[0];
            if window.is_empty() {
                return Ok(None);
            }
            Ok(Some(window.iter().sum::<f64>() / window.len() as f64))
        }
    }

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry.register("mean", Arc::new(Mean));
        registry
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ])
    }

    /// One mask-free view holding `rows`, as the M1 tests built.
    fn segment(rows: &[(i64, &str, f64)]) -> Vec<SegmentView> {
        let mut buffer = WriteBuffer::new(schema(), 0).unwrap();
        for &(ts, sym, x) in rows {
            buffer
                .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
                .unwrap();
        }
        vec![SegmentView::all_live(Arc::new(buffer.freeze().unwrap()))]
    }

    /// The same rows split into segments of `segment_rows` via a Store —
    /// the multi-segment shape queries actually run over.
    fn store(rows: &[(i64, &str, f64)], segment_rows: usize) -> Store {
        let mut store = Store::with_segment_rows(schema(), 0, segment_rows).unwrap();
        for &(ts, sym, x) in rows {
            store
                .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
                .unwrap();
        }
        store
    }

    fn segmented(rows: &[(i64, &str, f64)], segment_rows: usize) -> Vec<SegmentView> {
        store(rows, segment_rows).snapshot().unwrap()
    }

    fn f64_column(batch: &RecordBatch, index: usize) -> &NumericColumn<f64> {
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            panic!("expected f64 column")
        };
        column
    }

    /// Flattens one output column of a multi-batch result into
    /// `Option<f64>` per row, for comparison against a reference.
    fn flatten(output: &QueryOutput, index: usize) -> Vec<Option<f64>> {
        output
            .batches
            .iter()
            .flat_map(|batch| {
                let column = f64_column(batch, index);
                (0..column.len())
                    .map(|row| column.is_valid(row).then(|| column.values()[row]))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn run(views: &[SegmentView], sql: &str) -> Result<QueryOutput, QueryError> {
        execute(&schema(), views, &plan(sql).unwrap(), &registry())
    }

    #[test]
    fn unpartitioned_trailing_mean_matches_hand_computation() {
        let views = segment(&[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)]);
        let output = run(
            &views,
            "SELECT ts, mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS m \
             FROM t",
        )
        .unwrap();
        assert_eq!(output.schema.fields()[1].name(), "m");
        // Windows: [1], [1,2], [2,3], [3,4] — ragged only at the start.
        assert_eq!(
            f64_column(&output.batches[0], 1).values().as_slice(),
            &[1.0, 1.5, 2.5, 3.5]
        );
    }

    #[test]
    fn segmentation_never_changes_results() {
        // The golden invariant of the multi-segment executor: the same
        // rows produce the same values whether they sit in one segment or
        // many — windows span boundaries as if storage were contiguous.
        let rows: Vec<(i64, &str, f64)> = (0..23)
            .map(|i| {
                (
                    i,
                    ["A", "B", "C"][(i % 3) as usize],
                    (i as f64) * 1.5 - (i % 5) as f64,
                )
            })
            .collect();
        for sql in [
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) FROM t",
            "SELECT mean(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
        ] {
            let reference = flatten(&run(&segment(&rows), sql).unwrap(), 0);
            for segment_rows in [1, 4, 7, 23, 100] {
                let output = run(&segmented(&rows, segment_rows), sql).unwrap();
                assert_eq!(flatten(&output, 0), reference, "{sql} @ {segment_rows}");
            }
        }
    }

    #[test]
    fn tombstoned_rows_compute_exactly_like_absent_rows() {
        // The equivalence golden for mutation's read side: a table where
        // rows were deleted computes exactly what a table that never
        // held them computes — passthrough, windows, and partitions.
        let rows: Vec<(i64, &str, f64)> = (0..20)
            .map(|i| (i, ["A", "B"][(i % 2) as usize], i as f64 * 0.5))
            .collect();
        let dead: &[u64] = &[0, 3, 4, 5, 11, 19]; // ends, a run, scattered
        let surviving: Vec<(i64, &str, f64)> = rows
            .iter()
            .enumerate()
            .filter(|(index, _)| !dead.contains(&(*index as u64)))
            .map(|(_, &row)| row)
            .collect();
        for sql in [
            "SELECT ts, sym, x FROM t",
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) FROM t",
            "SELECT mean(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
        ] {
            for segment_rows in [4, 100] {
                let mut mutated = store(&rows, segment_rows);
                mutated.tombstone(dead).unwrap();
                let output = run(&mutated.snapshot().unwrap(), sql).unwrap();
                let reference = run(&segmented(&surviving, segment_rows), sql).unwrap();
                // Compare every column that is f64-flattenable; ts and
                // sym are checked via row counts + the x column carrying
                // position-sensitive values.
                assert_eq!(output.num_rows(), reference.num_rows(), "{sql}");
                let index = output.schema.fields().len() - 1;
                assert_eq!(
                    flatten(&output, index),
                    flatten(&reference, index),
                    "{sql} @ {segment_rows}"
                );
            }
        }
    }

    #[test]
    fn masked_key_passthrough_keeps_dictionary_and_values() {
        let mut store = store(
            &[(1, "B", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "C", 4.0)],
            100,
        );
        store.tombstone(&[1, 3]).unwrap(); // drop the A@2 and C rows
        let output = run(&store.snapshot().unwrap(), "SELECT sym FROM t").unwrap();
        assert_eq!(output.num_rows(), 2);
        let Column::Key(keys) = &output.batches[0].columns()[0] else {
            panic!("sym type")
        };
        assert_eq!(keys.value_at(0), Some("B"));
        assert_eq!(keys.value_at(1), Some("A"));
    }

    #[test]
    fn mask_free_views_stay_zero_copy_and_masked_ones_do_not_leak() {
        let mut store = store(
            &[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        store.tombstone(&[0]).unwrap(); // first segment masked
        let views = store.snapshot().unwrap();
        let output = run(&views, "SELECT x FROM t").unwrap();
        // Masked segment: one row survives, materialized.
        assert_eq!(
            f64_column(&output.batches[0], 0).values().as_slice(),
            &[2.0]
        );
        // Mask-free segment: still the stored allocation, shared.
        let stored = f64_column(views[1].segment.batch(), 2);
        assert_eq!(
            f64_column(&output.batches[1], 0).values().as_ptr(),
            stored.values().as_ptr()
        );
    }

    #[test]
    fn fully_tombstoned_segments_vanish_from_results() {
        let mut store = store(
            &[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        store.tombstone(&[0, 1]).unwrap(); // entire first segment
        let output = run(&store.snapshot().unwrap(), "SELECT x FROM t").unwrap();
        assert_eq!(output.batches.len(), 1);
        assert_eq!(
            f64_column(&output.batches[0], 0).values().as_slice(),
            &[3.0, 4.0]
        );
        // The whole table gone: schema survives, batches don't.
        let mut store = store2();
        store.tombstone(&[0, 1]).unwrap();
        let output = run(&store.snapshot().unwrap(), "SELECT ts, x FROM t").unwrap();
        assert_eq!(output.batches.len(), 0);
        assert_eq!(output.schema.fields()[0].name(), "ts");
    }

    fn store2() -> Store {
        store(&[(1, "A", 1.0), (2, "B", 2.0)], 100)
    }

    #[test]
    fn partition_codes_remap_across_segment_dictionaries() {
        // Segment 1 interns B first (code 0), segment 2 interns C then A:
        // the same symbol gets different codes in different segments, so
        // only the query-time remap makes partitions line up.
        let views = segmented(
            &[
                (1, "B", 10.0),
                (2, "A", 1.0),
                (3, "C", 100.0),
                (4, "A", 3.0),
                (5, "B", 30.0),
                (6, "C", 300.0),
            ],
            3,
        );
        assert_eq!(views.len(), 2);
        let output = run(
            &views,
            "SELECT mean(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        assert_eq!(
            flatten(&output, 0),
            [
                Some(10.0),  // B: [10]
                Some(1.0),   // A: [1]
                Some(100.0), // C: [100]
                Some(2.0),   // A: [1,3]
                Some(20.0),  // B: [10,30]
                Some(200.0)  // C: [100,300]
            ]
        );
    }

    #[test]
    fn passthrough_shares_each_segments_buffer() {
        let views = segmented(
            &[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        let output = run(&views, "SELECT x FROM t").unwrap();
        assert_eq!(output.batches.len(), 2);
        for (view, batch) in views.iter().zip(&output.batches) {
            let stored = f64_column(view.segment.batch(), 2);
            let out = f64_column(batch, 0);
            // Zero-copy: each result batch is its segment's buffer, shared.
            assert_eq!(out.values().as_ptr(), stored.values().as_ptr());
        }
    }

    #[test]
    fn empty_table_yields_schema_and_no_batches() {
        let output = run(&[], "SELECT ts, x FROM t").unwrap();
        assert_eq!(output.batches.len(), 0);
        assert_eq!(output.num_rows(), 0);
        assert_eq!(output.schema.fields()[0].name(), "ts");
        // Window plans over an empty table are also fine.
        let output = run(
            &[],
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        assert_eq!(output.batches.len(), 0);
        assert_eq!(output.schema.fields()[0].column_type(), ColumnType::F64);
    }

    #[test]
    fn undefined_windows_surface_as_null() {
        struct NeedsTwo;
        impl WindowAggregate for NeedsTwo {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok((args[0].len() >= 2).then(|| args[0][0]))
            }
        }
        let mut registry = Registry::new();
        registry.register("needs_two", Arc::new(NeedsTwo));
        let views = segment(&[(1, "A", 1.0), (2, "A", 2.0)]);
        let plan = plan(
            "SELECT needs_two(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) \
             FROM t",
        )
        .unwrap();
        let output = execute(&schema(), &views, &plan, &registry).unwrap();
        let column = f64_column(&output.batches[0], 0);
        assert!(!column.is_valid(0)); // one-row window: undefined -> NULL
        assert!(column.is_valid(1));
    }

    #[test]
    fn execution_errors_are_specific() {
        let views = segment(&[(1, "A", 1.0)]);
        let cases = [
            ("SELECT nope FROM t", "unknown column"),
            (
                "SELECT nope(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "unknown window function",
            ),
            (
                "SELECT mean(sym) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "must be f64",
            ),
            (
                "SELECT mean(x) OVER (ORDER BY x ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "ordering key",
            ),
            (
                "SELECT mean(x, x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "takes 1 arguments",
            ),
            (
                "SELECT mean(x) OVER (PARTITION BY x ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "must be a key column",
            ),
        ];
        for (sql, needle) in cases {
            let error = run(&views, sql).expect_err(sql).to_string();
            assert!(error.contains(needle), "{sql}: got '{error}'");
        }
    }

    #[test]
    fn unordered_data_is_refused_within_and_across_segments() {
        let sql =
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t";
        // Within one segment.
        let views = segment(&[(5, "A", 1.0), (3, "A", 2.0)]);
        assert!(matches!(run(&views, sql), Err(QueryError::Unordered(_))));
        // Each segment ordered, but the boundary goes backwards.
        let views = segmented(
            &[(1, "A", 1.0), (5, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        assert!(views.iter().all(|view| view.segment.is_ordered()));
        assert!(matches!(run(&views, sql), Err(QueryError::Unordered(_))));
        // Touching boundaries (equal values) are fine — "roughly sorted"
        // allows ties.
        let views = segmented(
            &[(1, "A", 1.0), (3, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        run(&views, sql).unwrap();
        // Tombstoning the offending boundary row resolves the
        // cross-segment disorder — live bounds, not raw bounds.
        let mut disordered = store(
            &[(1, "A", 1.0), (5, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        disordered.tombstone(&[1]).unwrap();
        run(&disordered.snapshot().unwrap(), sql).unwrap();
    }
}
