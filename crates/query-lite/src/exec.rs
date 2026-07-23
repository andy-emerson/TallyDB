//! Execution: the plan over a snapshot of segments.
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
//! A query runs over a storage snapshot — the segments of one table, in
//! append order — and produces one output batch per segment, Arrow's own
//! model for a chunked result. That shape is what keeps passthrough
//! zero-copy: each batch's passthrough columns share its segment's
//! buffers outright (copy-on-write clones), and each batch's key columns
//! keep their segment's own dictionary (per-segment dictionaries,
//! decision #6). Callers that want a single contiguous result pay for the
//! concatenation themselves, knowingly.
//!
//! ## Where the copies are
//!
//! Over a single segment, an unpartitioned window is a slice of the
//! stored column's own buffer — pointer arithmetic, no copy. Two cases
//! need rows gathered into contiguous scratch first, both the same
//! bounded O(rows) shape as the design-matrix gather recorded in
//! deferred issue #4: windows that *span segments* (the stored buffers
//! are per-segment, a window is not), and `PARTITION BY` (a partition's
//! rows are scattered). For `PARTITION BY` across segments, each
//! segment's dictionary codes are first remapped into a query-lifetime
//! key space — the query-time remap that decision #6 accepted, bounded by
//! the low-cardinality assumption.

use crate::plan::{Plan, PlanItem, QueryError};
use arrow_lite::{
    Bitmap, Column, ColumnType, Field, NumericColumn, NumericData, RecordBatch, Schema,
};
use std::collections::HashMap;
use std::sync::Arc;
use storage_lite::Segment;

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

/// A query's result: the output schema plus one batch per non-empty
/// segment of the snapshot, in append order. The schema is carried
/// explicitly so an empty table still yields a well-formed (zero-batch)
/// result.
#[derive(Debug)]
pub struct QueryOutput {
    /// Schema of every batch.
    pub schema: Schema,
    /// One batch per non-empty segment.
    pub batches: Vec<RecordBatch>,
}

impl QueryOutput {
    /// Total rows across all batches.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// Runs `plan` over `segments` — one table's snapshot, in append order,
/// all sharing `schema` — resolving window functions in `registry`.
///
/// The embedder has already resolved the plan's table name to this
/// snapshot; nothing here re-checks it.
pub fn execute(
    schema: &Schema,
    segments: &[Arc<Segment>],
    plan: &Plan,
    registry: &Registry,
) -> Result<QueryOutput, QueryError> {
    // Empty segments contribute no rows; dropping them up front means
    // "one batch per segment" below never emits an empty batch.
    let segments: Vec<&Segment> = segments
        .iter()
        .map(Arc::as_ref)
        .filter(|segment| segment.batch().num_rows() > 0)
        .collect();
    let mut fields = Vec::with_capacity(plan.items.len());
    let mut columns_per_segment: Vec<Vec<Column>> = segments.iter().map(|_| Vec::new()).collect();
    for item in &plan.items {
        let (field, columns) = match item {
            PlanItem::Column { name, alias } => {
                passthrough(schema, &segments, name, alias.as_deref())?
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
                &segments,
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
        for (out, column) in columns_per_segment.iter_mut().zip(columns) {
            out.push(column);
        }
    }
    let schema = Schema::new(fields);
    let batches = columns_per_segment
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

/// A stored column, passed through by shared handle — no row data copied,
/// and each segment's key columns keep their own dictionary.
fn passthrough(
    schema: &Schema,
    segments: &[&Segment],
    name: &str,
    alias: Option<&str>,
) -> Result<(Field, Vec<Column>), QueryError> {
    let (index, field) = resolve(schema, name)?;
    let mut out = Field::new(alias.unwrap_or(name), field.column_type(), field.nullable());
    if let Some(logical) = field.logical() {
        out = out.with_logical(logical);
    }
    let columns = segments
        .iter()
        .map(|segment| segment.batch().columns()[index].clone())
        .collect();
    Ok((out, columns))
}

/// The window slice for row `position` in a run of rows: `preceding` rows
/// back through the current row, ragged at the start of the run.
fn window_bounds(position: usize, preceding: usize) -> (usize, usize) {
    (position.saturating_sub(preceding), position + 1)
}

/// Checks that the snapshot is globally ordered on the window's ORDER BY
/// column: it must be the declared ordering key, each segment must be
/// internally ordered, and each segment boundary must be non-decreasing.
/// Checked, never assumed — a window over misordered rows silently
/// computes the wrong thing.
fn check_order(
    segments: &[&Segment],
    order_index: usize,
    order_by: &str,
) -> Result<(), QueryError> {
    let mut previous_last: Option<i64> = None;
    for segment in segments {
        if order_index != segment.ordering_key() {
            return Err(QueryError::Unsupported(format!(
                "ORDER BY '{order_by}' — windows order by the declared ordering key only"
            )));
        }
        if !segment.is_ordered() {
            return Err(QueryError::Unordered(format!(
                "ingest was not sorted on '{order_by}'"
            )));
        }
        let Some((first, last)) = segment.ordering_bounds() else {
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

/// Per-segment `&[f64]` slices for one argument column, validated
/// (f64, no nulls).
fn argument_slices<'a>(
    segments: &[&'a Segment],
    index: usize,
    name: &str,
) -> Result<Vec<&'a [f64]>, QueryError> {
    let mut slices = Vec::with_capacity(segments.len());
    for segment in segments {
        let Column::Numeric(NumericData::F64(column)) = &segment.batch().columns()[index] else {
            return Err(QueryError::TypeError(format!(
                "window argument '{name}' must be f64"
            )));
        };
        if column.null_count() > 0 {
            return Err(QueryError::Unsupported(format!(
                "window argument '{name}' has nulls (unsupported as a window argument)"
            )));
        }
        slices.push(column.values().as_slice());
    }
    Ok(slices)
}

#[allow(clippy::too_many_arguments)]
fn window_aggregate(
    schema: &Schema,
    segments: &[&Segment],
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
    check_order(segments, order_index, order_by)?;
    // Per-argument, per-segment slices: args[a][s] is argument `a` in
    // segment `s`.
    let mut args: Vec<Vec<&[f64]>> = Vec::with_capacity(arg_names.len());
    for name in arg_names {
        let (index, _) = resolve(schema, name)?;
        args.push(argument_slices(segments, index, name)?);
    }
    // One result slot per row, per segment.
    let mut results: Vec<Vec<Option<f64>>> = segments
        .iter()
        .map(|segment| vec![None; segment.batch().num_rows()])
        .collect();
    match partition_by {
        None => unpartitioned(segments, aggregate.as_ref(), &args, preceding, &mut results)?,
        Some(partition_column) => partitioned(
            schema,
            segments,
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

/// Unpartitioned windows run over the snapshot's rows in append order.
/// A single segment is the pure zero-copy path — every window is a
/// direct slice of the stored buffer. Across segments, each argument is
/// gathered once into contiguous scratch (windows span segment
/// boundaries; the stored buffers don't) and the windows slice that.
fn unpartitioned(
    segments: &[&Segment],
    aggregate: &dyn WindowAggregate,
    args: &[Vec<&[f64]>],
    preceding: usize,
    results: &mut [Vec<Option<f64>>],
) -> Result<(), QueryError> {
    let gathered: Vec<Vec<f64>>;
    let arg_slices: Vec<&[f64]> = if segments.len() == 1 {
        args.iter().map(|slices| slices[0]).collect()
    } else {
        gathered = args.iter().map(|slices| slices.concat()).collect();
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
/// snapshot. Dictionary codes are per-segment (decision #6), so each
/// segment's codes are remapped into a query-lifetime key space first;
/// each partition's rows are then gathered into contiguous scratch (they
/// are scattered even within one segment) and results scattered back to
/// their segment and row.
fn partitioned(
    schema: &Schema,
    segments: &[&Segment],
    aggregate: &dyn WindowAggregate,
    args: &[Vec<&[f64]>],
    partition_column: &str,
    preceding: usize,
    results: &mut [Vec<Option<f64>>],
) -> Result<(), QueryError> {
    let (index, _) = resolve(schema, partition_column)?;
    // The query-lifetime key space: value → unified code, built once per
    // distinct value per segment (cheap under low cardinality).
    let mut unified: HashMap<String, usize> = HashMap::new();
    // Per partition: scratch per argument, plus where each row came from.
    let mut scratch: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut origins: Vec<Vec<(usize, usize)>> = Vec::new();
    for (segment_index, segment) in segments.iter().enumerate() {
        let Column::Key(keys) = &segment.batch().columns()[index] else {
            return Err(QueryError::TypeError(format!(
                "PARTITION BY '{partition_column}' must be a key column"
            )));
        };
        if keys.null_count() > 0 {
            return Err(QueryError::Unsupported(format!(
                "PARTITION BY '{partition_column}' has nulls (unsupported as a partition key)"
            )));
        }
        // This segment's code → unified code.
        let dictionary = keys.dictionary();
        let remap: Vec<usize> = (0..dictionary.len() as u32)
            .map(|code| {
                let next = unified.len();
                *unified
                    .entry(dictionary.value(code).to_owned())
                    .or_insert(next)
            })
            .collect();
        for (row, &code) in keys.codes().iter().enumerate() {
            let partition = remap[code as usize];
            if partition == scratch.len() {
                scratch.push(vec![Vec::new(); args.len()]);
                origins.push(Vec::new());
            }
            for (argument, slices) in args.iter().enumerate() {
                scratch[partition][argument].push(slices[segment_index][row]);
            }
            origins[partition].push((segment_index, row));
        }
    }
    let mut windows: Vec<&[f64]> = Vec::with_capacity(args.len());
    for (values, rows) in scratch.iter().zip(&origins) {
        for (position, &(segment_index, row)) in rows.iter().enumerate() {
            let (start, end) = window_bounds(position, preceding);
            windows.clear();
            windows.extend(values.iter().map(|argument| &argument[start..end]));
            results[segment_index][row] =
                aggregate.evaluate(&windows).map_err(QueryError::Compute)?;
        }
    }
    Ok(())
}

/// One segment's output column: nullable f64, bitmap only if a window
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

    /// One segment holding `rows`, as the M1 tests built.
    fn segment(rows: &[(i64, &str, f64)]) -> Vec<Arc<Segment>> {
        let mut buffer = WriteBuffer::new(schema(), 0).unwrap();
        for &(ts, sym, x) in rows {
            buffer
                .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
                .unwrap();
        }
        vec![Arc::new(buffer.freeze().unwrap())]
    }

    /// The same rows split into segments of `segment_rows` via a Store —
    /// the multi-segment shape queries actually run over.
    fn segmented(rows: &[(i64, &str, f64)], segment_rows: usize) -> Vec<Arc<Segment>> {
        let mut store = Store::with_segment_rows(schema(), 0, segment_rows).unwrap();
        for &(ts, sym, x) in rows {
            store
                .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
                .unwrap();
        }
        store.snapshot().unwrap()
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

    fn run(segments: &[Arc<Segment>], sql: &str) -> Result<QueryOutput, QueryError> {
        execute(&schema(), segments, &plan(sql).unwrap(), &registry())
    }

    #[test]
    fn unpartitioned_trailing_mean_matches_hand_computation() {
        let segments = segment(&[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)]);
        let output = run(
            &segments,
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
    fn partitioned_windows_track_each_key_separately() {
        let segments = segment(&[
            (1, "A", 1.0),
            (2, "B", 10.0),
            (3, "A", 2.0),
            (4, "B", 20.0),
            (5, "A", 3.0),
        ]);
        let output = run(
            &segments,
            "SELECT mean(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        // A's runs: [1], [1,2], [2,3]; B's: [10], [10,20] — interleaved
        // back into ingest positions.
        assert_eq!(
            f64_column(&output.batches[0], 0).values().as_slice(),
            &[1.0, 10.0, 1.5, 15.0, 2.5]
        );
    }

    #[test]
    fn partition_codes_remap_across_segment_dictionaries() {
        // Segment 1 interns B first (code 0), segment 2 interns C then A:
        // the same symbol gets different codes in different segments, so
        // only the query-time remap makes partitions line up.
        let segments = segmented(
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
        assert_eq!(segments.len(), 2);
        let output = run(
            &segments,
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
        let segments = segmented(
            &[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        let output = run(&segments, "SELECT x FROM t").unwrap();
        assert_eq!(output.batches.len(), 2);
        for (segment, batch) in segments.iter().zip(&output.batches) {
            let stored = f64_column(segment.batch(), 2);
            let out = f64_column(batch, 0);
            // Zero-copy: each result batch is its segment's buffer, shared.
            assert_eq!(out.values().as_ptr(), stored.values().as_ptr());
        }
    }

    #[test]
    fn key_passthrough_keeps_per_segment_dictionaries() {
        let segments = segmented(&[(1, "B", 1.0), (2, "A", 2.0), (3, "A", 3.0)], 2);
        let output = run(&segments, "SELECT sym FROM t").unwrap();
        let dictionaries: Vec<Vec<&str>> = output
            .batches
            .iter()
            .map(|batch| {
                let Column::Key(keys) = &batch.columns()[0] else {
                    panic!("sym type")
                };
                (0..keys.dictionary().len() as u32)
                    .map(|code| keys.dictionary().value(code))
                    .collect()
            })
            .collect();
        assert_eq!(dictionaries, [vec!["B", "A"], vec!["A"]]);
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
        let segments = segment(&[(1, "A", 1.0), (2, "A", 2.0)]);
        let plan = plan(
            "SELECT needs_two(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) \
             FROM t",
        )
        .unwrap();
        let output = execute(&schema(), &segments, &plan, &registry).unwrap();
        let column = f64_column(&output.batches[0], 0);
        assert!(!column.is_valid(0)); // one-row window: undefined -> NULL
        assert!(column.is_valid(1));
    }

    #[test]
    fn execution_errors_are_specific() {
        let segments = segment(&[(1, "A", 1.0)]);
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
            let error = run(&segments, sql).expect_err(sql).to_string();
            assert!(error.contains(needle), "{sql}: got '{error}'");
        }
    }

    #[test]
    fn unordered_data_is_refused_within_and_across_segments() {
        let sql =
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t";
        // Within one segment.
        let segments = segment(&[(5, "A", 1.0), (3, "A", 2.0)]);
        assert!(matches!(run(&segments, sql), Err(QueryError::Unordered(_))));
        // Each segment ordered, but the boundary goes backwards.
        let segments = segmented(
            &[(1, "A", 1.0), (5, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        assert!(segments.iter().all(|s| s.is_ordered()));
        assert!(matches!(run(&segments, sql), Err(QueryError::Unordered(_))));
        // Touching boundaries (equal values) are fine — "roughly sorted"
        // allows ties.
        let segments = segmented(
            &[(1, "A", 1.0), (3, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            2,
        );
        run(&segments, sql).unwrap();
    }
}
