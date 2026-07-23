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

use crate::plan::{AggCall, AggFunction, AggItem, OrderBy, Plan, PlanItem, Projection, QueryError};
use crate::predicate::{can_match, evaluate as evaluate_predicate};
use arrow_lite::{
    Bitmap, Buffer, Column, ColumnType, Dictionary, Field, KeyColumn, NumericColumn, NumericData,
    RecordBatch, Schema,
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
    // WHERE first, standard SQL order of operations: the predicate folds
    // into each view's live mask, so everything downstream — windows
    // included — sees only the surviving rows.
    let filtered: Vec<SegmentView>;
    let views: &[SegmentView] = match &plan.predicate {
        None => views,
        Some(predicate) => {
            filtered = views
                .iter()
                .map(|view| {
                    // Zone-map pruning: skip evaluating segments whose
                    // value ranges provably cannot match. Correctness
                    // never depends on this — the pruned outcome is
                    // exactly an all-false match.
                    let rows = view.segment.batch().num_rows();
                    let live = if !can_match(predicate, schema, view) {
                        Bitmap::new_unset(rows)
                    } else {
                        let matched = evaluate_predicate(predicate, schema, view)?;
                        match &view.live {
                            None => matched,
                            Some(live) => live.and(&matched),
                        }
                    };
                    Ok(SegmentView {
                        segment: view.segment.clone(),
                        live: Some(live),
                    })
                })
                .collect::<Result<Vec<SegmentView>, QueryError>>()?;
            &filtered
        }
    };
    // Views with no live rows contribute nothing; dropping them up front
    // means "one batch per segment" below never emits an empty batch.
    let views: Vec<&SegmentView> = views.iter().filter(|view| view.live_rows() > 0).collect();
    let mut output = match &plan.projection {
        Projection::Items(items) => project_items(schema, &views, items, registry)?,
        Projection::Aggregate { keys, items } => project_aggregate(schema, &views, keys, items)?,
    };
    if let Some(order_by) = &plan.order_by {
        output = sort_output(output, order_by)?;
    }
    if plan.limit.is_some() || plan.offset.is_some() {
        output = limit_output(output, plan.offset.unwrap_or(0), plan.limit);
    }
    Ok(output)
}

/// The row-per-row projection: plain columns and window calls, one
/// output batch per view.
fn project_items(
    schema: &Schema,
    views: &[&SegmentView],
    items: &[PlanItem],
    registry: &Registry,
) -> Result<QueryOutput, QueryError> {
    let mut fields = Vec::with_capacity(items.len());
    let mut columns_per_view: Vec<Vec<Column>> = views.iter().map(|_| Vec::new()).collect();
    for item in items {
        let (field, columns) = match item {
            PlanItem::Column { name, alias } => passthrough(schema, views, name, alias.as_deref())?,
            PlanItem::WindowAgg {
                function,
                args,
                partition_by,
                order_by,
                preceding,
                alias,
            } => window_aggregate(
                schema,
                views,
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

/// A group key's per-row code: a unified dictionary code, or the null
/// group.
type GroupCode = Option<usize>;

/// One aggregate accumulator. The variant is chosen from the call and
/// its argument column's type; every variant tracks whether it has seen
/// a (non-null) value, because SQL aggregates over nothing are NULL —
/// except COUNT, which is 0.
#[derive(Clone)]
enum Accumulator {
    CountStar(i64),
    CountColumn(i64),
    SumF64 { sum: f64, seen: bool },
    SumI64 { sum: i64, seen: bool },
    Avg { sum: f64, count: i64 },
    MinMaxF64 { value: f64, seen: bool, max: bool },
    MinMaxI64 { value: i64, seen: bool, max: bool },
}

impl Accumulator {
    /// The starting accumulator for `call` over a column of
    /// `argument_type` (`None` for `COUNT(*)`).
    fn new(call: &AggCall, argument_type: Option<ColumnType>) -> Result<Accumulator, QueryError> {
        let type_error = |what: &str| {
            QueryError::TypeError(format!(
                "{what} needs a numeric argument, got {:?}",
                argument_type
            ))
        };
        Ok(match (call.function, argument_type) {
            (AggFunction::Count, None) => Accumulator::CountStar(0),
            (AggFunction::Count, Some(_)) => Accumulator::CountColumn(0),
            (AggFunction::Sum, Some(ColumnType::F64)) => Accumulator::SumF64 {
                sum: 0.0,
                seen: false,
            },
            (AggFunction::Sum, Some(ColumnType::I64)) => Accumulator::SumI64 {
                sum: 0,
                seen: false,
            },
            (AggFunction::Avg, Some(ColumnType::F64 | ColumnType::I64)) => {
                Accumulator::Avg { sum: 0.0, count: 0 }
            }
            (AggFunction::Min | AggFunction::Max, Some(ColumnType::F64)) => {
                Accumulator::MinMaxF64 {
                    value: 0.0,
                    seen: false,
                    max: call.function == AggFunction::Max,
                }
            }
            (AggFunction::Min | AggFunction::Max, Some(ColumnType::I64)) => {
                Accumulator::MinMaxI64 {
                    value: 0,
                    seen: false,
                    max: call.function == AggFunction::Max,
                }
            }
            (AggFunction::Sum, _) => return Err(type_error("SUM")),
            (AggFunction::Avg, _) => return Err(type_error("AVG")),
            (AggFunction::Min, _) => return Err(type_error("MIN")),
            (AggFunction::Max, _) => return Err(type_error("MAX")),
        })
    }

    /// Folds in one row's cell (`None` = the cell is null, or the call
    /// is `COUNT(*)` and there is no cell).
    fn update(&mut self, cell: Option<CellNumber>) -> Result<(), QueryError> {
        match (self, cell) {
            (Accumulator::CountStar(count), _) => *count += 1,
            (Accumulator::CountColumn(_), None) => {}
            (Accumulator::CountColumn(count), Some(_)) => *count += 1,
            (_, None) => {}
            (Accumulator::SumF64 { sum, seen }, Some(cell)) => {
                *sum += cell.as_f64();
                *seen = true;
            }
            (Accumulator::SumI64 { sum, seen }, Some(CellNumber::I64(value))) => {
                *sum = sum.checked_add(value).ok_or_else(|| {
                    QueryError::Compute("SUM overflows i64 — refusing a wrong answer".to_owned())
                })?;
                *seen = true;
            }
            (Accumulator::Avg { sum, count }, Some(cell)) => {
                *sum += cell.as_f64();
                *count += 1;
            }
            (Accumulator::MinMaxF64 { value, seen, max }, Some(cell)) => {
                let candidate = cell.as_f64();
                // total_cmp keeps NaN ordered (greater than everything),
                // matching how it will sort and compare downstream.
                let replace = !*seen
                    || (*max && candidate.total_cmp(value).is_gt())
                    || (!*max && candidate.total_cmp(value).is_lt());
                if replace {
                    *value = candidate;
                }
                *seen = true;
            }
            (Accumulator::MinMaxI64 { value, seen, max }, Some(CellNumber::I64(candidate))) => {
                let replace =
                    !*seen || (*max && candidate > *value) || (!*max && candidate < *value);
                if replace {
                    *value = candidate;
                }
                *seen = true;
            }
            _ => unreachable!("accumulator variant chosen from the argument type"),
        }
        Ok(())
    }
}

/// One numeric cell, typed.
#[derive(Clone, Copy)]
enum CellNumber {
    F64(f64),
    I64(i64),
}

impl CellNumber {
    fn as_f64(self) -> f64 {
        match self {
            CellNumber::F64(value) => value,
            CellNumber::I64(value) => value as f64,
        }
    }
}

/// The aggregate projection: group live rows by key columns in the
/// query-lifetime unified key space (decision #6 — codes remap per
/// segment), fold the accumulators, and emit one batch with one row per
/// group, groups in first-seen order (deterministic; callers wanting a
/// specific order say ORDER BY). No GROUP BY keys means one global
/// group — emitted even over zero rows, per SQL.
fn project_aggregate(
    schema: &Schema,
    views: &[&SegmentView],
    keys: &[String],
    items: &[AggItem],
) -> Result<QueryOutput, QueryError> {
    // Resolve keys (must be key columns) and calls (typed accumulators).
    let key_indices: Vec<usize> = keys
        .iter()
        .map(|key| {
            let (index, field) = resolve(schema, key)?;
            if field.column_type() != ColumnType::Key {
                return Err(QueryError::TypeError(format!(
                    "GROUP BY '{key}' must be a key column — grouping is what keys are for"
                )));
            }
            Ok(index)
        })
        .collect::<Result<Vec<usize>, QueryError>>()?;
    let calls: Vec<(&AggCall, Option<usize>, Option<ColumnType>)> = items
        .iter()
        .filter_map(|item| match item {
            AggItem::Call(call) => Some(call),
            AggItem::Key { .. } => None,
        })
        .map(|call| {
            let argument = call
                .argument
                .as_ref()
                .map(|name| resolve(schema, name))
                .transpose()?;
            let index = argument.map(|(index, _)| index);
            let column_type = argument.map(|(_, field)| field.column_type());
            Ok((call, index, column_type))
        })
        .collect::<Result<Vec<_>, QueryError>>()?;
    let template: Vec<Accumulator> = calls
        .iter()
        .map(|(call, _, column_type)| Accumulator::new(call, *column_type))
        .collect::<Result<Vec<Accumulator>, QueryError>>()?;
    // The unified key space, one per key column.
    let mut unified: Vec<HashMap<String, usize>> = vec![HashMap::new(); keys.len()];
    let mut unified_values: Vec<Vec<String>> = vec![Vec::new(); keys.len()];
    // Groups in first-seen order.
    let mut groups: HashMap<Vec<GroupCode>, usize> = HashMap::new();
    let mut group_keys: Vec<Vec<GroupCode>> = Vec::new();
    let mut accumulators: Vec<Vec<Accumulator>> = Vec::new();
    if keys.is_empty() {
        groups.insert(Vec::new(), 0);
        group_keys.push(Vec::new());
        accumulators.push(template.clone());
    }
    for view in views {
        let batch = view.segment.batch();
        // This view's key columns, with per-segment codes remapped into
        // the unified space (decision #6's query-time remap).
        let mut remaps: Vec<(&KeyColumn, Vec<usize>)> = Vec::with_capacity(key_indices.len());
        for (position, &index) in key_indices.iter().enumerate() {
            let Column::Key(column) = &batch.columns()[index] else {
                unreachable!("validated as a key column above")
            };
            let dictionary = column.dictionary();
            let remap: Vec<usize> = (0..dictionary.len() as u32)
                .map(|code| {
                    let value = dictionary.value(code);
                    if let Some(&unified_code) = unified[position].get(value) {
                        unified_code
                    } else {
                        let unified_code = unified_values[position].len();
                        unified[position].insert(value.to_owned(), unified_code);
                        unified_values[position].push(value.to_owned());
                        unified_code
                    }
                })
                .collect();
            remaps.push((column, remap));
        }
        for row in live_rows(view) {
            let group_key: Vec<GroupCode> = remaps
                .iter()
                .map(|(column, remap)| {
                    column
                        .is_valid(row)
                        .then(|| remap[column.codes().as_slice()[row] as usize])
                })
                .collect();
            let group = *groups.entry(group_key.clone()).or_insert_with(|| {
                group_keys.push(group_key.clone());
                accumulators.push(template.clone());
                group_keys.len() - 1
            });
            for ((_, argument_index, _), accumulator) in
                calls.iter().zip(accumulators[group].iter_mut())
            {
                let cell = match argument_index {
                    None => None,
                    Some(index) => match &batch.columns()[*index] {
                        Column::Numeric(NumericData::F64(numeric)) => numeric
                            .is_valid(row)
                            .then(|| CellNumber::F64(numeric.values().as_slice()[row])),
                        Column::Numeric(NumericData::I64(numeric)) => numeric
                            .is_valid(row)
                            .then(|| CellNumber::I64(numeric.values().as_slice()[row])),
                        Column::Key(_) => {
                            return Err(QueryError::TypeError(
                                "aggregates take numeric arguments; keys are labels".to_owned(),
                            ))
                        }
                    },
                };
                accumulator.update(cell)?;
            }
        }
    }
    // Assemble the single output batch, SELECT-list order.
    let group_count = group_keys.len();
    let mut fields = Vec::with_capacity(items.len());
    let mut columns = Vec::with_capacity(items.len());
    let mut next_call = 0usize;
    for item in items {
        match item {
            AggItem::Key { name, alias } => {
                let position = keys.iter().position(|key| key == name).expect("validated");
                let mut dictionary = Dictionary::new();
                let mut codes: Buffer<u32> = Buffer::with_capacity(group_count);
                let mut validity: Vec<bool> = Vec::with_capacity(group_count);
                for group_key in &group_keys {
                    match group_key[position] {
                        Some(code) => {
                            codes.push(dictionary.intern(&unified_values[position][code]));
                            validity.push(true);
                        }
                        None => {
                            codes.push(0);
                            validity.push(false);
                        }
                    }
                }
                let nullable = validity.iter().any(|&valid| !valid);
                let column = if nullable {
                    KeyColumn::new_nullable(
                        codes,
                        Bitmap::from_bools(validity.iter().copied()),
                        dictionary,
                    )
                } else {
                    KeyColumn::new_non_null(codes, dictionary)
                };
                fields.push(Field::new(
                    alias.as_deref().unwrap_or(name),
                    ColumnType::Key,
                    nullable,
                ));
                columns.push(Column::Key(column));
            }
            AggItem::Call(call) => {
                let default_name = match call.function {
                    AggFunction::Count => "count",
                    AggFunction::Sum => "sum",
                    AggFunction::Avg => "avg",
                    AggFunction::Min => "min",
                    AggFunction::Max => "max",
                };
                let name = call.alias.as_deref().unwrap_or(default_name);
                let (field, column) = assemble_aggregate(
                    name,
                    accumulators.iter().map(|group| &group[next_call]),
                    group_count,
                );
                fields.push(field);
                columns.push(column);
                next_call += 1;
            }
        }
    }
    let schema = Schema::new(fields);
    let batches = if group_count == 0 {
        Vec::new()
    } else {
        vec![RecordBatch::new(schema.clone(), columns)]
    };
    Ok(QueryOutput { schema, batches })
}

/// One aggregate output column from its per-group accumulators.
fn assemble_aggregate<'a>(
    name: &str,
    accumulators: impl Iterator<Item = &'a Accumulator>,
    groups: usize,
) -> (Field, Column) {
    let mut f64_values: Vec<Option<f64>> = Vec::with_capacity(groups);
    let mut i64_values: Vec<Option<i64>> = Vec::with_capacity(groups);
    let mut is_i64 = false;
    for accumulator in accumulators {
        match accumulator {
            Accumulator::CountStar(count) | Accumulator::CountColumn(count) => {
                is_i64 = true;
                i64_values.push(Some(*count));
            }
            Accumulator::SumF64 { sum, seen } => f64_values.push(seen.then_some(*sum)),
            Accumulator::SumI64 { sum, seen } => {
                is_i64 = true;
                i64_values.push(seen.then_some(*sum));
            }
            Accumulator::Avg { sum, count } => {
                f64_values.push((*count > 0).then(|| sum / *count as f64))
            }
            Accumulator::MinMaxF64 { value, seen, .. } => f64_values.push(seen.then_some(*value)),
            Accumulator::MinMaxI64 { value, seen, .. } => {
                is_i64 = true;
                i64_values.push(seen.then_some(*value));
            }
        }
    }
    if is_i64 {
        let nullable = i64_values.iter().any(Option::is_none);
        let values: Buffer<i64> = i64_values.iter().map(|value| value.unwrap_or(0)).collect();
        let column = if nullable {
            NumericColumn::new_nullable(
                values,
                Bitmap::from_bools(i64_values.iter().map(Option::is_some)),
            )
        } else {
            NumericColumn::new_non_null(values)
        };
        (
            Field::new(name, ColumnType::I64, nullable),
            Column::Numeric(NumericData::I64(column)),
        )
    } else {
        let nullable = f64_values.iter().any(Option::is_none);
        let values: Buffer<f64> = f64_values
            .iter()
            .map(|value| value.unwrap_or(0.0))
            .collect();
        let column = if nullable {
            NumericColumn::new_nullable(
                values,
                Bitmap::from_bools(f64_values.iter().map(Option::is_some)),
            )
        } else {
            NumericColumn::new_non_null(values)
        };
        (
            Field::new(name, ColumnType::F64, nullable),
            Column::Numeric(NumericData::F64(column)),
        )
    }
}

/// A sortable view of one output cell.
#[derive(Clone, PartialEq, PartialOrd)]
enum SortCell {
    I64(i64),
    F64(f64),
    Text(String),
}

/// Sorts the whole output by one column into a single batch — the
/// materialization ORDER BY inherently asks for. Nulls follow the
/// PostgreSQL (and DuckDB) default: last under ASC, first under DESC;
/// `f64` uses total order, so NaN sorts above every number.
fn sort_output(output: QueryOutput, order_by: &OrderBy) -> Result<QueryOutput, QueryError> {
    let (column_index, _) = resolve(&output.schema, &order_by.column)?;
    let mut picks: Vec<(usize, usize)> = Vec::with_capacity(output.num_rows());
    let mut cells: Vec<Option<SortCell>> = Vec::with_capacity(output.num_rows());
    for (batch_index, batch) in output.batches.iter().enumerate() {
        let column = &batch.columns()[column_index];
        for row in 0..batch.num_rows() {
            picks.push((batch_index, row));
            cells.push(match column {
                Column::Numeric(NumericData::F64(numeric)) => numeric
                    .is_valid(row)
                    .then(|| SortCell::F64(numeric.values().as_slice()[row])),
                Column::Numeric(NumericData::I64(numeric)) => numeric
                    .is_valid(row)
                    .then(|| SortCell::I64(numeric.values().as_slice()[row])),
                Column::Key(keys) => keys
                    .value_at(row)
                    .map(|value| SortCell::Text(value.to_owned())),
            });
        }
    }
    let mut order: Vec<usize> = (0..picks.len()).collect();
    let compare = |left: &Option<SortCell>, right: &Option<SortCell>| match (left, right) {
        (None, None) => std::cmp::Ordering::Equal,
        // Nulls sort as the largest value (ASC last); DESC reversal
        // below then puts them first — the PostgreSQL default.
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (Some(_), None) => std::cmp::Ordering::Less,
        (Some(SortCell::F64(left)), Some(SortCell::F64(right))) => left.total_cmp(right),
        (Some(left), Some(right)) => left.partial_cmp(right).expect("same variant per column"),
    };
    order.sort_by(|&left, &right| {
        let ordering = compare(&cells[left], &cells[right]);
        if order_by.descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
    let picks: Vec<(usize, usize)> = order.into_iter().map(|index| picks[index]).collect();
    let batch = take_rows(&output.schema, &output.batches, &picks);
    Ok(QueryOutput {
        schema: output.schema,
        batches: vec![batch],
    })
}

/// Applies OFFSET/LIMIT across the output's rows (in the output's
/// current order), materializing the kept rows into one batch.
fn limit_output(output: QueryOutput, offset: usize, limit: Option<usize>) -> QueryOutput {
    let keep = limit.unwrap_or(usize::MAX);
    let picks: Vec<(usize, usize)> = output
        .batches
        .iter()
        .enumerate()
        .flat_map(|(batch_index, batch)| (0..batch.num_rows()).map(move |row| (batch_index, row)))
        .skip(offset)
        .take(keep)
        .collect();
    if picks.is_empty() {
        return QueryOutput {
            schema: output.schema,
            batches: Vec::new(),
        };
    }
    let batch = take_rows(&output.schema, &output.batches, &picks);
    QueryOutput {
        schema: output.schema,
        batches: vec![batch],
    }
}

/// Gathers `picks` (batch, row) into one batch. Key columns re-encode
/// into a fresh dictionary — the sources' per-segment dictionaries don't
/// share codes.
fn take_rows(schema: &Schema, batches: &[RecordBatch], picks: &[(usize, usize)]) -> RecordBatch {
    let columns = (0..schema.fields().len())
        .map(|column_index| {
            let cell_column = |batch: usize| &batches[batch].columns()[column_index];
            match cell_column(picks.first().map(|&(batch, _)| batch).unwrap_or(0)) {
                Column::Numeric(NumericData::F64(_)) => {
                    let mut values: Buffer<f64> = Buffer::with_capacity(picks.len());
                    let mut validity: Vec<bool> = Vec::with_capacity(picks.len());
                    for &(batch, row) in picks {
                        let Column::Numeric(NumericData::F64(numeric)) = cell_column(batch) else {
                            unreachable!("batches share a schema")
                        };
                        values.push(numeric.values().as_slice()[row]);
                        validity.push(numeric.is_valid(row));
                    }
                    assemble_numeric_f64(values, validity)
                }
                Column::Numeric(NumericData::I64(_)) => {
                    let mut values: Buffer<i64> = Buffer::with_capacity(picks.len());
                    let mut validity: Vec<bool> = Vec::with_capacity(picks.len());
                    for &(batch, row) in picks {
                        let Column::Numeric(NumericData::I64(numeric)) = cell_column(batch) else {
                            unreachable!("batches share a schema")
                        };
                        values.push(numeric.values().as_slice()[row]);
                        validity.push(numeric.is_valid(row));
                    }
                    assemble_numeric_i64(values, validity)
                }
                Column::Key(_) => {
                    let mut dictionary = Dictionary::new();
                    let mut codes: Buffer<u32> = Buffer::with_capacity(picks.len());
                    let mut validity: Vec<bool> = Vec::with_capacity(picks.len());
                    for &(batch, row) in picks {
                        let Column::Key(keys) = cell_column(batch) else {
                            unreachable!("batches share a schema")
                        };
                        match keys.value_at(row) {
                            Some(value) => {
                                codes.push(dictionary.intern(value));
                                validity.push(true);
                            }
                            None => {
                                codes.push(0);
                                validity.push(false);
                            }
                        }
                    }
                    let column = if validity.iter().any(|&valid| !valid) {
                        KeyColumn::new_nullable(
                            codes,
                            Bitmap::from_bools(validity.iter().copied()),
                            dictionary,
                        )
                    } else {
                        KeyColumn::new_non_null(codes, dictionary)
                    };
                    Column::Key(column)
                }
            }
        })
        .collect();
    RecordBatch::new(schema.clone(), columns)
}

fn assemble_numeric_f64(values: Buffer<f64>, validity: Vec<bool>) -> Column {
    let column = if validity.iter().any(|&valid| !valid) {
        NumericColumn::new_nullable(values, Bitmap::from_bools(validity.iter().copied()))
    } else {
        NumericColumn::new_non_null(values)
    };
    Column::Numeric(NumericData::F64(column))
}

fn assemble_numeric_i64(values: Buffer<i64>, validity: Vec<bool>) -> Column {
    let column = if validity.iter().any(|&valid| !valid) {
        NumericColumn::new_nullable(values, Bitmap::from_bools(validity.iter().copied()))
    } else {
        NumericColumn::new_non_null(values)
    };
    Column::Numeric(NumericData::I64(column))
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

    pub(super) fn registry() -> Registry {
        let mut registry = Registry::new();
        registry.register("mean", Arc::new(Mean));
        registry
    }

    pub(super) fn schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ])
    }

    /// One mask-free view holding `rows`, as the M1 tests built.
    pub(super) fn segment(rows: &[(i64, &str, f64)]) -> Vec<SegmentView> {
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
    pub(super) fn store(rows: &[(i64, &str, f64)], segment_rows: usize) -> Store {
        let mut store = Store::with_segment_rows(schema(), 0, segment_rows).unwrap();
        for &(ts, sym, x) in rows {
            store
                .append(&[RowValue::I64(ts), RowValue::Key(sym), RowValue::F64(x)])
                .unwrap();
        }
        store
    }

    pub(super) fn segmented(rows: &[(i64, &str, f64)], segment_rows: usize) -> Vec<SegmentView> {
        store(rows, segment_rows).snapshot().unwrap()
    }

    pub(super) fn f64_column(batch: &RecordBatch, index: usize) -> &NumericColumn<f64> {
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            panic!("expected f64 column")
        };
        column
    }

    /// Flattens one output column of a multi-batch result into
    /// `Option<f64>` per row, for comparison against a reference.
    pub(super) fn flatten(output: &QueryOutput, index: usize) -> Vec<Option<f64>> {
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

    pub(super) fn run(views: &[SegmentView], sql: &str) -> Result<QueryOutput, QueryError> {
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
                "unknown function",
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

#[cfg(test)]
mod query1_tests {
    use super::tests::{f64_column, flatten, run, schema, segment, segmented, store};
    use super::*;

    #[test]
    fn where_filters_before_everything() {
        let rows: &[(i64, &str, f64)] = &[
            (1, "A", 1.0),
            (2, "B", 2.0),
            (3, "A", 3.0),
            (4, "B", 4.0),
            (5, "A", 5.0),
        ];
        for segment_rows in [2, 100] {
            let views = segmented(rows, segment_rows);
            let output = run(&views, "SELECT x FROM t WHERE sym = 'A' AND ts > 1").unwrap();
            assert_eq!(flatten(&output, 0), [Some(3.0), Some(5.0)]);
            // WHERE applies before windows (standard SQL): the window
            // sees only surviving rows, exactly as if the others were
            // never ingested.
            let filtered = run(
                &views,
                "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
                 FROM t WHERE sym = 'A'",
            )
            .unwrap();
            let reference = run(
                &segment(&[(1, "A", 1.0), (3, "A", 3.0), (5, "A", 5.0)]),
                "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
                 FROM t",
            )
            .unwrap();
            assert_eq!(flatten(&filtered, 0), flatten(&reference, 0));
        }
    }

    #[test]
    fn aggregates_match_hand_computation() {
        let rows: &[(i64, &str, f64)] = &[
            (1, "A", 1.0),
            (2, "B", 10.0),
            (3, "A", 2.0),
            (4, "B", 20.0),
            (5, "A", 6.0),
        ];
        for segment_rows in [2, 100] {
            let views = segmented(rows, segment_rows);
            let output = run(
                &views,
                "SELECT sym, count(*) AS n, sum(x) AS total, avg(x) AS mean_x, \
                 min(x) AS low, max(x) AS high FROM t GROUP BY sym ORDER BY sym",
            )
            .unwrap();
            assert_eq!(output.batches.len(), 1);
            let batch = &output.batches[0];
            let Column::Key(sym) = &batch.columns()[0] else {
                panic!("sym type")
            };
            assert_eq!(sym.value_at(0), Some("A"));
            assert_eq!(sym.value_at(1), Some("B"));
            let Column::Numeric(NumericData::I64(n)) = &batch.columns()[1] else {
                panic!("count type")
            };
            assert_eq!(n.values().as_slice(), &[3, 2]);
            assert_eq!(f64_column(batch, 2).values().as_slice(), &[9.0, 30.0]);
            assert_eq!(f64_column(batch, 3).values().as_slice(), &[3.0, 15.0]);
            assert_eq!(f64_column(batch, 4).values().as_slice(), &[1.0, 10.0]);
            assert_eq!(f64_column(batch, 5).values().as_slice(), &[6.0, 20.0]);
        }
    }

    #[test]
    fn global_aggregates_emit_one_row_even_over_nothing() {
        let views = segment(&[(1, "A", 1.0), (2, "B", 2.0)]);
        let output = run(&views, "SELECT count(*) AS n, sum(x) AS s FROM t").unwrap();
        let batch = &output.batches[0];
        let Column::Numeric(NumericData::I64(n)) = &batch.columns()[0] else {
            panic!("count type")
        };
        assert_eq!(n.values().as_slice(), &[1 + 1]);
        // Over a fully filtered table: COUNT is 0, SUM is NULL — SQL.
        let output = run(
            &views,
            "SELECT count(*) AS n, sum(x) AS s FROM t WHERE ts > 99",
        )
        .unwrap();
        let batch = &output.batches[0];
        let Column::Numeric(NumericData::I64(n)) = &batch.columns()[0] else {
            panic!("count type")
        };
        assert_eq!(n.values().as_slice(), &[0]);
        let s = f64_column(batch, 1);
        assert!(!s.is_valid(0));
        // With GROUP BY and nothing surviving: zero groups, zero batches.
        let output = run(
            &views,
            "SELECT sym, count(*) FROM t WHERE ts > 99 GROUP BY sym",
        )
        .unwrap();
        assert_eq!(output.batches.len(), 0);
    }

    #[test]
    fn sum_of_i64_is_exact_and_overflow_is_loud() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("n", ColumnType::I64, false),
        ]);
        let mut buffer = storage_lite::WriteBuffer::new(schema.clone(), 0).unwrap();
        for (ts, n) in [(1, i64::MAX - 1), (2, 1)] {
            buffer
                .append(&[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::I64(n),
                ])
                .unwrap();
        }
        let views = vec![SegmentView::all_live(Arc::new(buffer.freeze().unwrap()))];
        let output = execute(
            &schema,
            &views,
            &crate::plan::plan("SELECT sum(n) AS s FROM t").unwrap(),
            &Registry::new(),
        )
        .unwrap();
        let Column::Numeric(NumericData::I64(s)) = &output.batches[0].columns()[0] else {
            panic!("sum type")
        };
        assert_eq!(s.values().as_slice(), &[i64::MAX]);
        // One more row overflows: a loud error, never a wrong answer.
        let mut buffer = storage_lite::WriteBuffer::new(schema.clone(), 0).unwrap();
        for (ts, n) in [(1, i64::MAX), (2, 1)] {
            buffer
                .append(&[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::I64(n),
                ])
                .unwrap();
        }
        let views = vec![SegmentView::all_live(Arc::new(buffer.freeze().unwrap()))];
        assert!(matches!(
            execute(
                &schema,
                &views,
                &crate::plan::plan("SELECT sum(n) FROM t").unwrap(),
                &Registry::new(),
            ),
            Err(QueryError::Compute(_))
        ));
    }

    #[test]
    fn order_by_sorts_and_limit_trims() {
        let rows: &[(i64, &str, f64)] = &[
            (1, "B", 3.0),
            (2, "A", 1.0),
            (3, "C", 2.0),
            (4, "A", 5.0),
            (5, "B", 4.0),
        ];
        for segment_rows in [2, 100] {
            let views = segmented(rows, segment_rows);
            let output = run(&views, "SELECT ts, x FROM t ORDER BY x").unwrap();
            assert_eq!(output.batches.len(), 1); // materialized
            assert_eq!(
                flatten(&output, 1),
                [Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
            );
            let output = run(&views, "SELECT ts, x FROM t ORDER BY x DESC LIMIT 2").unwrap();
            assert_eq!(flatten(&output, 1), [Some(5.0), Some(4.0)]);
            let output = run(&views, "SELECT x FROM t ORDER BY x LIMIT 2 OFFSET 1").unwrap();
            assert_eq!(flatten(&output, 0), [Some(2.0), Some(3.0)]);
            // Keys sort by rendered value, not by dictionary code.
            let output = run(&views, "SELECT sym, x FROM t ORDER BY sym").unwrap();
            let Column::Key(sym) = &output.batches[0].columns()[0] else {
                panic!("sym type")
            };
            let values: Vec<&str> = (0..5).map(|row| sym.value_at(row).unwrap()).collect();
            assert_eq!(values, ["A", "A", "B", "B", "C"]);
            // LIMIT without ORDER BY keeps ingest order.
            let output = run(&views, "SELECT ts FROM t LIMIT 3").unwrap();
            let Column::Numeric(NumericData::I64(ts)) = &output.batches[0].columns()[0] else {
                panic!("ts type")
            };
            assert_eq!(ts.values().as_slice(), &[1, 2, 3]);
        }
    }

    #[test]
    fn order_by_nulls_follow_postgres_defaults() {
        let views = segment(&[(1, "A", 1.0), (2, "B", 2.0), (3, "C", 3.0)]);
        // A window column with a NULL first row provides the nulls.
        let sql_asc = "SELECT needs2(x) OVER (ORDER BY ts ROWS BETWEEN 9 PRECEDING AND \
                       CURRENT ROW) AS w FROM t ORDER BY w";
        let sql_desc = "SELECT needs2(x) OVER (ORDER BY ts ROWS BETWEEN 9 PRECEDING AND \
                        CURRENT ROW) AS w FROM t ORDER BY w DESC";
        struct NeedsTwo;
        impl WindowAggregate for NeedsTwo {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok((args[0].len() >= 2).then(|| args[0][args[0].len() - 1]))
            }
        }
        let mut registry = Registry::new();
        registry.register("needs2", Arc::new(NeedsTwo));
        let ascending = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql_asc).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&ascending, 0), [Some(2.0), Some(3.0), None]); // nulls last
        let descending = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql_desc).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&descending, 0), [None, Some(3.0), Some(2.0)]); // nulls first
    }

    #[test]
    fn where_composes_with_mutation_masks() {
        // WHERE ANDs into tombstone masks rather than replacing them.
        let mut store = store(
            &[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)],
            100,
        );
        store.tombstone(&[1]).unwrap(); // ts=2 dies
        let output = run(&store.snapshot().unwrap(), "SELECT ts FROM t WHERE ts <= 3").unwrap();
        let Column::Numeric(NumericData::I64(ts)) = &output.batches[0].columns()[0] else {
            panic!("ts type")
        };
        assert_eq!(ts.values().as_slice(), &[1, 3]);
    }

    #[test]
    fn group_by_multiple_keys_uses_composite_groups() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("a", ColumnType::Key, false),
            Field::new("b", ColumnType::Key, false),
        ]);
        let mut buffer = storage_lite::WriteBuffer::new(schema.clone(), 0).unwrap();
        for (ts, a, b) in [(1, "x", "p"), (2, "x", "q"), (3, "x", "p"), (4, "y", "q")] {
            buffer
                .append(&[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::Key(a),
                    storage_lite::RowValue::Key(b),
                ])
                .unwrap();
        }
        let views = vec![SegmentView::all_live(Arc::new(buffer.freeze().unwrap()))];
        let output = execute(
            &schema,
            &views,
            &crate::plan::plan("SELECT a, b, count(*) AS n FROM t GROUP BY a, b ORDER BY n DESC")
                .unwrap(),
            &Registry::new(),
        )
        .unwrap();
        let batch = &output.batches[0];
        assert_eq!(batch.num_rows(), 3); // (x,p)=2, (x,q)=1, (y,q)=1
        let Column::Numeric(NumericData::I64(n)) = &batch.columns()[2] else {
            panic!("count type")
        };
        assert_eq!(n.values().as_slice()[0], 2);
    }

    #[test]
    fn grouping_by_numeric_is_refused_by_design() {
        let views = segment(&[(1, "A", 1.0)]);
        let error = run(&views, "SELECT x, count(*) FROM t GROUP BY x")
            .unwrap_err()
            .to_string();
        assert!(error.contains("keys"), "{error}");
    }
}
