//! Execution: the M1 plan over one frozen segment.
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
//! ## The zero-copy shape
//!
//! Without `PARTITION BY`, every window is a slice of the stored column's
//! own buffer — pointer arithmetic, no copy. With `PARTITION BY`, a
//! partition's rows are not contiguous in the segment, so each argument
//! column is gathered once per partition into a scratch buffer and the
//! windows slice *that* — one bounded O(rows) gather per column per
//! partition, same shape as the design-matrix gather recorded in
//! deferred issue #4. Passthrough columns share the segment's buffers
//! outright (copy-on-write clones).

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

/// Runs `plan` over `segment`, resolving window functions in `registry`.
///
/// The embedder has already resolved the plan's table name to this
/// segment; nothing here re-checks it.
pub fn execute(
    segment: &Segment,
    plan: &Plan,
    registry: &Registry,
) -> Result<RecordBatch, QueryError> {
    let batch = segment.batch();
    let mut fields = Vec::with_capacity(plan.items.len());
    let mut columns = Vec::with_capacity(plan.items.len());
    for item in &plan.items {
        let (field, column) = match item {
            PlanItem::Column { name, alias } => passthrough(batch, name, alias.as_deref())?,
            PlanItem::WindowAgg {
                function,
                args,
                partition_by,
                order_by,
                preceding,
                alias,
            } => window_aggregate(
                segment,
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
        columns.push(column);
    }
    Ok(RecordBatch::new(Schema::new(fields), columns))
}

/// Looks up a column by name.
fn resolve<'a>(batch: &'a RecordBatch, name: &str) -> Result<(usize, &'a Field), QueryError> {
    batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == name)
        .ok_or_else(|| QueryError::UnknownColumn(name.to_owned()))
}

/// A stored column, passed through by shared handle — no row data copied.
fn passthrough(
    batch: &RecordBatch,
    name: &str,
    alias: Option<&str>,
) -> Result<(Field, Column), QueryError> {
    let (index, field) = resolve(batch, name)?;
    let mut out = Field::new(alias.unwrap_or(name), field.column_type(), field.nullable());
    if let Some(logical) = field.logical() {
        out = out.with_logical(logical);
    }
    Ok((out, batch.columns()[index].clone()))
}

/// The window slice for row `position` in a run of rows: `preceding` rows
/// back through the current row, ragged at the start of the run.
fn window_bounds(position: usize, preceding: usize) -> (usize, usize) {
    (position.saturating_sub(preceding), position + 1)
}

#[allow(clippy::too_many_arguments)]
fn window_aggregate(
    segment: &Segment,
    registry: &Registry,
    function: &str,
    arg_names: &[String],
    partition_by: Option<&str>,
    order_by: &str,
    preceding: usize,
    alias: Option<&str>,
) -> Result<(Field, Column), QueryError> {
    let batch = segment.batch();
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
    // The ORDER BY column must be the declared ordering key, and the data
    // must actually be in that order — checked, never assumed, because a
    // window over misordered rows silently computes the wrong thing.
    let (order_index, _) = resolve(batch, order_by)?;
    if order_index != segment.ordering_key() {
        return Err(QueryError::Unsupported(format!(
            "ORDER BY '{order_by}' — M1 windows order by the declared ordering key only"
        )));
    }
    if !segment.is_ordered() {
        return Err(QueryError::Unordered(format!(
            "ingest was not sorted on '{order_by}'"
        )));
    }
    // Argument columns: f64, with no nulls (M1).
    let mut arg_slices: Vec<&[f64]> = Vec::with_capacity(arg_names.len());
    for name in arg_names {
        let (index, _) = resolve(batch, name)?;
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            return Err(QueryError::TypeError(format!(
                "window argument '{name}' must be f64"
            )));
        };
        if column.null_count() > 0 {
            return Err(QueryError::Unsupported(format!(
                "window argument '{name}' has nulls (M1 requires non-null arguments)"
            )));
        }
        arg_slices.push(column.values().as_slice());
    }
    let num_rows = batch.num_rows();
    let mut results: Vec<Option<f64>> = vec![None; num_rows];
    match partition_by {
        None => {
            // Unpartitioned: every window is a direct slice of the stored
            // buffers — the pure zero-copy path.
            let mut windows: Vec<&[f64]> = Vec::with_capacity(arg_slices.len());
            for (row, result) in results.iter_mut().enumerate() {
                let (start, end) = window_bounds(row, preceding);
                windows.clear();
                windows.extend(arg_slices.iter().map(|values| &values[start..end]));
                *result = aggregate.evaluate(&windows).map_err(QueryError::Compute)?;
            }
        }
        Some(partition_column) => {
            let (index, _) = resolve(batch, partition_column)?;
            let Column::Key(keys) = &batch.columns()[index] else {
                return Err(QueryError::TypeError(format!(
                    "PARTITION BY '{partition_column}' must be a key column"
                )));
            };
            if keys.null_count() > 0 {
                return Err(QueryError::Unsupported(format!(
                    "PARTITION BY '{partition_column}' has nulls (M1 requires non-null keys)"
                )));
            }
            // Rows of each partition, in stored (= ordering-key) order.
            let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); keys.dictionary().len()];
            for (row, &code) in keys.codes().iter().enumerate() {
                partitions[code as usize].push(row);
            }
            for rows in partitions.iter().filter(|rows| !rows.is_empty()) {
                // The bounded gather: a partition's rows are scattered, so
                // pull each argument into contiguous scratch once; the
                // windows below are slices of that scratch (see the
                // module docs and issue #4).
                let scratch: Vec<Vec<f64>> = arg_slices
                    .iter()
                    .map(|values| rows.iter().map(|&row| values[row]).collect())
                    .collect();
                let mut windows: Vec<&[f64]> = Vec::with_capacity(arg_slices.len());
                for (position, &row) in rows.iter().enumerate() {
                    let (start, end) = window_bounds(position, preceding);
                    windows.clear();
                    windows.extend(scratch.iter().map(|values| &values[start..end]));
                    results[row] = aggregate.evaluate(&windows).map_err(QueryError::Compute)?;
                }
            }
        }
    }
    // Assemble the output column: nullable f64, bitmap only if a window
    // actually came back undefined.
    let values = results.iter().map(|v| v.unwrap_or(0.0)).collect();
    let column = if results.iter().any(Option::is_none) {
        NumericColumn::new_nullable(
            values,
            Bitmap::from_bools(results.iter().map(Option::is_some)),
        )
    } else {
        NumericColumn::new_non_null(values)
    };
    let name = alias.unwrap_or(function);
    Ok((
        Field::new(name, ColumnType::F64, true),
        Column::Numeric(NumericData::F64(column)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan;
    use storage_lite::{RowValue, WriteBuffer};

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

    fn segment(rows: &[(i64, &str, f64)]) -> Segment {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ]);
        let mut buffer = WriteBuffer::new(schema, 0).unwrap();
        for &(ts, sym, x) in rows {
            buffer
                .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
                .unwrap();
        }
        buffer.freeze().unwrap()
    }

    fn f64_column(batch: &RecordBatch, index: usize) -> &NumericColumn<f64> {
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            panic!("expected f64 column")
        };
        column
    }

    #[test]
    fn unpartitioned_trailing_mean_matches_hand_computation() {
        let segment = segment(&[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)]);
        let plan = plan(
            "SELECT ts, mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS m \
             FROM t",
        )
        .unwrap();
        let batch = execute(&segment, &plan, &registry()).unwrap();
        assert_eq!(batch.schema().fields()[1].name(), "m");
        // Windows: [1], [1,2], [2,3], [3,4] — ragged only at the start.
        assert_eq!(
            f64_column(&batch, 1).values().as_slice(),
            &[1.0, 1.5, 2.5, 3.5]
        );
    }

    #[test]
    fn partitioned_windows_track_each_key_separately() {
        let segment = segment(&[
            (1, "A", 1.0),
            (2, "B", 10.0),
            (3, "A", 2.0),
            (4, "B", 20.0),
            (5, "A", 3.0),
        ]);
        let plan = plan(
            "SELECT mean(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        let batch = execute(&segment, &plan, &registry()).unwrap();
        // A's runs: [1], [1,2], [2,3]; B's: [10], [10,20] — interleaved
        // back into ingest positions.
        assert_eq!(
            f64_column(&batch, 0).values().as_slice(),
            &[1.0, 10.0, 1.5, 15.0, 2.5]
        );
    }

    #[test]
    fn passthrough_shares_the_segment_buffer() {
        let segment = segment(&[(1, "A", 1.0), (2, "A", 2.0)]);
        let plan = plan("SELECT x FROM t").unwrap();
        let batch = execute(&segment, &plan, &registry()).unwrap();
        let stored = f64_column(segment.batch(), 2);
        let out = f64_column(&batch, 0);
        // Zero-copy: the result column is the stored buffer, shared.
        assert_eq!(out.values().as_ptr(), stored.values().as_ptr());
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
        let segment = segment(&[(1, "A", 1.0), (2, "A", 2.0)]);
        let plan = plan(
            "SELECT needs_two(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) \
             FROM t",
        )
        .unwrap();
        let batch = execute(&segment, &plan, &registry).unwrap();
        let column = f64_column(&batch, 0);
        assert!(!column.is_valid(0)); // one-row window: undefined -> NULL
        assert!(column.is_valid(1));
    }

    #[test]
    fn execution_errors_are_specific() {
        let segment = segment(&[(1, "A", 1.0)]);
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
            let error = execute(&segment, &plan(sql).unwrap(), &registry())
                .expect_err(sql)
                .to_string();
            assert!(error.contains(needle), "{sql}: got '{error}'");
        }
    }

    #[test]
    fn unordered_segment_is_refused() {
        let segment = segment(&[(5, "A", 1.0), (3, "A", 2.0)]);
        let plan = plan(
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        assert!(matches!(
            execute(&segment, &plan, &registry()),
            Err(QueryError::Unordered(_))
        ));
    }
}
