//! Execution: the plan over a snapshot of segment views.
//!
//! ## The compute seam
//!
//! Window-aggregate *implementations* are not defined here. The embedder
//! (`engine`) registers them in a [`Registry`] — that is how compute
//! (regressions, pair statistics, Lua kernels) reaches SQL while this crate
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
//! a masked view is filter-materialized once per query — an O(rows)
//! gather (proportional to the segment, not bounded by a constant), the
//! price a masked read pays. Windows that *span segments* and
//! `PARTITION BY` gather the same way they did before; for `PARTITION
//! BY` across segments, each segment's dictionary codes are first
//! remapped into a query-lifetime key space (the query-time remap
//! decision #6 accepted).

use crate::plan::{
    AggCall, AggFunction, AggItem, ArithOp, OrderBy, Plan, PlanItem, Projection, QueryError,
    ScalarExpr, ScalarFunction, SEQUENCE_COLUMN,
};
use crate::predicate::{can_match, cmp_f64, evaluate as evaluate_predicate, Predicate};
use arrow_lite::{
    Bitmap, Buffer, Column, ColumnType, Dictionary, Field, KeyColumn, NumericColumn, NumericData,
    RecordBatch, Schema,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use storage_lite::{Segment, SegmentView, SequenceInfo};

/// One window-aggregate implementation, registered by the embedder.
pub trait WindowAggregate: Send + Sync {
    /// Number of column arguments the function takes.
    fn arity(&self) -> usize;

    /// Evaluates one window. `args` holds one slice per argument, all the
    /// same length (the window's rows, oldest first). `Ok(None)` means
    /// the aggregate is undefined for this window and becomes SQL `NULL`;
    /// `Err` aborts the query.
    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String>;

    /// Evaluates every trailing frame over one contiguous run of rows —
    /// position `i`'s frame is rows `i.saturating_sub(preceding) ..= i`
    /// (all rows from the run's start when `preceding` is `None`), the
    /// executor's one frame shape. `columns` holds one slice per
    /// argument (at least one — every aggregate takes arguments), all
    /// the same length; the result holds one entry per position, in
    /// order. The executor always calls this, once per contiguous run
    /// (the whole snapshot, or one partition).
    ///
    /// The default recomputes each frame through [`Self::evaluate`] —
    /// correct for any aggregate, `O(run · window)`. An aggregate whose
    /// consecutive frames share work (running moments) overrides this
    /// with an incremental form; because the executor only ever calls
    /// through here, the override is a pure implementation swap with no
    /// caller change.
    fn evaluate_frames(
        &self,
        columns: &[&[f64]],
        preceding: Option<usize>,
    ) -> Result<Vec<Option<f64>>, String> {
        recompute_frames(self, columns, preceding)
    }

    /// The Arrow type of this window's output column. Computed in `f64`
    /// internally, but a function whose result is logically integral (e.g.
    /// `COUNT`) declares `I64` so its output column matches SQL — the
    /// integer values are cast back exactly at materialization. Defaults
    /// to `F64`; this is also where a script-backed window will declare
    /// its return type (F2, M2.7).
    fn output_type(&self) -> ColumnType {
        ColumnType::F64
    }
}

/// An embedder-registered per-row function (SQL calls these *scalar*
/// functions), evaluated a whole column at a time — one call per view,
/// vectorized, so an interpreter-backed implementation pays its entry
/// cost once per batch instead of once per row. Appears in projection
/// as an ordinary call: `SELECT f(x, y) FROM t`.
pub trait ColumnFunction: Send + Sync {
    /// How many arguments a call must pass.
    fn arity(&self) -> usize;
    /// One call per view: each argument arrives dense over the live
    /// rows as `(values, validity)` (a `false` slot is SQL NULL; its
    /// value is unspecified). Returns one result per row — `None` is
    /// NULL. The output column is nullable `f64`; exact-`i64` and key
    /// outputs are deferred surface (#40's exactness rules would bind
    /// them).
    fn evaluate(&self, args: &[(&[f64], &[bool])]) -> Result<Vec<Option<f64>>, String>;
}

/// The function registry: SQL name → implementation, window aggregates
/// and column functions in separate namespaces (SQL resolves them from
/// different positions — `OVER` calls the first, plain projection
/// calls the second).
#[derive(Clone, Default)]
pub struct Registry {
    aggregates: HashMap<String, Arc<dyn WindowAggregate>>,
    columns: HashMap<String, Arc<dyn ColumnFunction>>,
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

    /// Registers `function` as a column function under `name`
    /// (lower-cased; last one wins — the promotion path).
    pub fn register_column(&mut self, name: &str, function: Arc<dyn ColumnFunction>) {
        self.columns.insert(name.to_lowercase(), function);
    }

    /// Every registered **window aggregate** and its implementation, in
    /// no particular order — what lets an embedding expose them to a
    /// scripting layer under the same names SQL uses.
    ///
    /// Column functions ([`Registry::register_column`]) are a second
    /// namespace and are *not* included: they return a whole column,
    /// while the script-side host-function seam returns one value per
    /// call, so there is no shape to install them under. Closing that
    /// gap means a column-shaped host seam, not a wider iterator.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &Arc<dyn WindowAggregate>)> {
        self.aggregates
            .iter()
            .map(|(name, aggregate)| (name.as_str(), aggregate))
    }

    fn column(&self, name: &str) -> Option<&Arc<dyn ColumnFunction>> {
        self.columns.get(name)
    }

    fn get(&self, name: &str) -> Option<&Arc<dyn WindowAggregate>> {
        self.aggregates.get(name)
    }
}

/// A query's result: the output schema plus its batches. The schema is
/// carried explicitly so an empty result is still well-formed (it has
/// no batches at all).
///
/// **How many batches is not a contract.** The streaming shape — one
/// batch per segment with live rows, in append order — is what a plain
/// scan produces, but any stage that must see all rows at once
/// (`ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT`, `HAVING`, `GROUP BY`)
/// collapses the result to a single materialized batch. A consumer that
/// needs one contiguous batch should say so with [`contiguous`] rather
/// than test `batches.len() == 1`, and a consumer that wants to stream
/// should handle any count.
#[derive(Debug)]
pub struct QueryOutput {
    /// Schema of every batch.
    pub schema: Schema,
    /// The result's batches — see the type's note on how many.
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
    if plan.join.is_some() {
        return Err(QueryError::Unsupported(
            "joins execute through the multi-table doorway (Database), not a single table"
                .to_owned(),
        ));
    }
    execute_single(schema, views, plan, registry)
}

/// Runs a star-schema join plan: `fact_views` joined against
/// `dimension_views` on the plan's key columns, then the ordinary
/// single-table pipeline over the joined intermediate.
///
/// The join is fact-driven: output stays one batch per fact segment;
/// each fact segment's join-key codes are remapped **once per distinct
/// dictionary value** into dimension row lookups (decision #6's
/// pattern), and the dimension's columns are gathered per fact row —
/// the copy a join is: sized by the *fact* table, not the dimension, and
/// **every** dimension attribute is gathered regardless of the `SELECT`
/// (there is no projection pushdown yet — #56). `INNER` drops unmatched
/// fact rows
/// through the live mask (the same mechanism as tombstones and WHERE);
/// `LEFT` keeps them with null dimension cells. Null join keys match
/// nothing, per SQL. Dimension columns join as nullable. The dimension
/// key must be unique among its live rows — a star-schema dimension is
/// a lookup table, and a duplicate key is an error, not a silent row
/// multiplication.
pub fn execute_join(
    fact_schema: &Schema,
    fact_views: &[SegmentView],
    dimension_schema: &Schema,
    dimension_views: &[SegmentView],
    plan: &Plan,
    registry: &Registry,
) -> Result<QueryOutput, QueryError> {
    let Some(join) = &plan.join else {
        return execute(fact_schema, fact_views, plan, registry);
    };
    let (fact_key_index, fact_key_field) = resolve(fact_schema, &join.fact_key)?;
    if fact_key_field.column_type() != ColumnType::Key {
        return Err(QueryError::TypeError(format!(
            "join column '{}' must be a key column — joining is what keys are for",
            join.fact_key
        )));
    }
    let (dimension_key_index, dimension_key_field) =
        resolve(dimension_schema, &join.dimension_key)?;
    if dimension_key_field.column_type() != ColumnType::Key {
        return Err(QueryError::TypeError(format!(
            "join column '{}' must be a key column — joining is what keys are for",
            join.dimension_key
        )));
    }
    // The dimension lookup: key value → (view, row), unique or bust.
    let mut lookup: HashMap<String, (usize, usize)> = HashMap::new();
    for (view_index, view) in dimension_views.iter().enumerate() {
        let Column::Key(keys) = &view.segment.batch().columns()[dimension_key_index] else {
            unreachable!("validated as a key column above")
        };
        for row in live_rows(view) {
            let Some(value) = keys.value_at(row) else {
                continue; // a null dimension key matches nothing
            };
            if lookup.insert(value.to_owned(), (view_index, row)).is_some() {
                return Err(QueryError::TypeError(format!(
                    "dimension key '{}' is not unique (value '{value}'): a star-schema \
                     dimension is a lookup table",
                    join.dimension_key
                )));
            }
        }
    }
    // The joined schema: fact columns, then the dimension columns the
    // query actually reads (#81) minus its key (which duplicates the
    // fact key), all nullable (LEFT produces nulls; INNER's
    // placeholders sit under the dead mask). Gathering an unread
    // attribute would cost a full fact-cardinality column for nothing —
    // dimensions are wide and queries are narrow.
    let referenced = plan.referenced_columns();
    let mut fields: Vec<Field> = fact_schema.fields().to_vec();
    let mut dimension_columns: Vec<usize> = Vec::new();
    for (index, field) in dimension_schema.fields().iter().enumerate() {
        if index == dimension_key_index {
            continue;
        }
        // The name clash is refused whether or not the query reads the
        // column: which query runs must not decide whether a schema
        // pairing is legal.
        if fact_schema
            .fields()
            .iter()
            .any(|fact_field| fact_field.name() == field.name())
        {
            return Err(QueryError::Unsupported(format!(
                "column '{}' exists in both tables — star-schema dimensions need \
                 distinct attribute names",
                field.name()
            )));
        }
        if !referenced.contains(field.name()) {
            continue;
        }
        dimension_columns.push(index);
        fields.push(Field::new(field.name(), field.column_type(), true));
    }
    let joined_schema = Schema::new(fields);
    let mut joined_views = Vec::with_capacity(fact_views.len());
    for view in fact_views {
        let batch = view.segment.batch();
        let Column::Key(keys) = &batch.columns()[fact_key_index] else {
            unreachable!("validated as a key column above")
        };
        let dictionary = keys.dictionary();
        // The once-per-distinct-value lookup.
        let remap: Vec<Option<(usize, usize)>> = (0..dictionary.len() as u32)
            .map(|code| lookup.get(dictionary.value(code)).copied())
            .collect();
        let codes = keys.codes().as_slice();
        let picks: Vec<Option<(usize, usize)>> = (0..batch.num_rows())
            .map(|row| {
                if keys.is_valid(row) {
                    remap[codes[row] as usize]
                } else {
                    None
                }
            })
            .collect();
        let live = if join.left {
            view.live.clone()
        } else {
            // INNER: unmatched rows die exactly like tombstoned ones.
            let matched = Bitmap::from_bools(picks.iter().map(Option::is_some));
            Some(match &view.live {
                None => matched,
                Some(live) => live.and(&matched),
            })
        };
        let mut columns: Vec<Column> = batch.columns().to_vec();
        for &dimension_column in &dimension_columns {
            columns.push(gather_dimension_column(
                dimension_views,
                dimension_column,
                dimension_schema.fields()[dimension_column].column_type(),
                &picks,
            ));
        }
        let joined = RecordBatch::new(joined_schema.clone(), columns);
        // Joining widens rows, it never adds, drops or reorders them
        // (INNER's misses die under the mask, in place) — so the fact
        // segment's birth sequences still describe row i, and `_seq`
        // reads the same coordinate through a join as without one. The
        // virtual form has to be made explicit: it means "sequence ==
        // row id", and this scratch segment's row ids start at 0.
        let sequence = match view.segment.sequence_info() {
            SequenceInfo::RowIds => SequenceInfo::Contiguous {
                base: view.segment.base_row_id(),
            },
            carried => carried.clone(),
        };
        let segment = Segment::from_batch(
            joined,
            view.segment.ordering_key(),
            view.segment.is_ordered(),
        )
        .with_sequence(sequence);
        joined_views.push(SegmentView {
            segment: Arc::new(segment),
            live,
        });
    }
    execute_single(&joined_schema, &joined_views, plan, registry)
}

/// One dimension column, gathered per fact row (`None` pick = no match:
/// a null cell). The output type comes from the dimension *schema*, not
/// from the views — so a join against an **empty** dimension (no views to
/// sniff a type from) still builds a column of the declared type rather
/// than defaulting to `f64` and mismatching the joined schema.
fn gather_dimension_column(
    dimension_views: &[SegmentView],
    column_index: usize,
    column_type: ColumnType,
    picks: &[Option<(usize, usize)>],
) -> Column {
    let cell = |view: usize| &dimension_views[view].segment.batch().columns()[column_index];
    match column_type {
        ColumnType::F64 => {
            let mut values: Buffer<f64> = Buffer::with_capacity(picks.len());
            let mut validity: Vec<bool> = Vec::with_capacity(picks.len());
            for pick in picks {
                match *pick {
                    Some((view, row)) => {
                        let Column::Numeric(NumericData::F64(numeric)) = cell(view) else {
                            unreachable!("batches share a schema")
                        };
                        values.push(numeric.values().as_slice()[row]);
                        validity.push(numeric.is_valid(row));
                    }
                    None => {
                        values.push(0.0);
                        validity.push(false);
                    }
                }
            }
            assemble_numeric_f64(values, validity)
        }
        ColumnType::I64 => {
            let mut values: Buffer<i64> = Buffer::with_capacity(picks.len());
            let mut validity: Vec<bool> = Vec::with_capacity(picks.len());
            for pick in picks {
                match *pick {
                    Some((view, row)) => {
                        let Column::Numeric(NumericData::I64(numeric)) = cell(view) else {
                            unreachable!("batches share a schema")
                        };
                        values.push(numeric.values().as_slice()[row]);
                        validity.push(numeric.is_valid(row));
                    }
                    None => {
                        values.push(0);
                        validity.push(false);
                    }
                }
            }
            assemble_numeric_i64(values, validity)
        }
        ColumnType::Key => {
            let mut dictionary = Dictionary::new();
            let mut codes: Buffer<u32> = Buffer::with_capacity(picks.len());
            let mut validity: Vec<bool> = Vec::with_capacity(picks.len());
            for pick in picks {
                let value = pick.and_then(|(view, row)| {
                    let Column::Key(keys) = cell(view) else {
                        unreachable!("batches share a schema")
                    };
                    keys.value_at(row)
                });
                match value {
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
            Column::Key(assemble_key(codes, validity, dictionary))
        }
    }
}

/// Builds a nullable key column, keeping every code in dictionary range
/// even when every row is null (an empty dictionary gets a placeholder
/// entry that no valid row references).
fn assemble_key(codes: Buffer<u32>, validity: Vec<bool>, mut dictionary: Dictionary) -> KeyColumn {
    if dictionary.is_empty() && !codes.is_empty() {
        dictionary.intern("");
    }
    if validity.iter().any(|&valid| !valid) {
        KeyColumn::new_nullable(
            codes,
            Bitmap::from_bools(validity.iter().copied()),
            dictionary,
        )
    } else {
        KeyColumn::new_non_null(codes, dictionary)
    }
}

/// The single-input pipeline `execute` and `execute_join` share.
fn execute_single(
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
        Projection::Aggregate {
            keys,
            items,
            having,
        } => match having {
            None => project_aggregate(schema, &views, keys, items)?,
            Some(having) => {
                let mut extended = items.to_vec();
                extended.extend(having.items.iter().cloned());
                let output = project_aggregate(schema, &views, keys, &extended)?;
                filter_having(output, &having.predicate, items.len())?
            }
        },
    };
    if plan.distinct {
        output = distinct_output(output);
    }
    if let Some(order_by) = &plan.order_by {
        // Only the rows OFFSET/LIMIT can reach are worth sorting.
        let keep = plan
            .limit
            .map(|limit| limit.saturating_add(plan.offset.unwrap_or(0)));
        output = sort_output(output, order_by, keep)?;
    }
    if plan.limit.is_some() || plan.offset.is_some() {
        output = limit_output(output, plan.offset.unwrap_or(0), plan.limit);
    }
    Ok(output)
}

/// `SELECT DISTINCT`: keeps each projected row's first occurrence.
/// Row identity is by *value*, type-tagged per cell: key cells compare
/// as their dictionary strings (per-segment dictionaries make codes
/// incomparable across batches), `f64` under the one comparison
/// relation (NaN equals itself; `-0.0` merges with `0.0`, DuckDB's
/// behavior), and NULLs equal — SQL's DISTINCT semantics. The result
/// consolidates to one batch.
fn distinct_output(output: QueryOutput) -> QueryOutput {
    let mut seen = HashSet::new();
    let mut picks: Vec<(usize, usize)> = Vec::new();
    for (batch_index, batch) in output.batches.iter().enumerate() {
        for row in 0..batch.num_rows() {
            let mut identity = Vec::new();
            for column in batch.columns() {
                match column {
                    Column::Numeric(NumericData::F64(numeric)) => {
                        if numeric.is_valid(row) {
                            let value = numeric.values().as_slice()[row];
                            let canonical = if value.is_nan() {
                                f64::NAN
                            } else if value == 0.0 {
                                0.0
                            } else {
                                value
                            };
                            identity.push(1u8);
                            identity.extend_from_slice(&canonical.to_bits().to_le_bytes());
                        } else {
                            identity.push(0);
                        }
                    }
                    Column::Numeric(NumericData::I64(numeric)) => {
                        if numeric.is_valid(row) {
                            identity.push(2);
                            identity
                                .extend_from_slice(&numeric.values().as_slice()[row].to_le_bytes());
                        } else {
                            identity.push(0);
                        }
                    }
                    Column::Key(keys) => match keys.value_at(row) {
                        Some(value) => {
                            identity.push(3);
                            identity.extend_from_slice(&(value.len() as u32).to_le_bytes());
                            identity.extend_from_slice(value.as_bytes());
                        }
                        None => identity.push(0),
                    },
                }
            }
            if seen.insert(identity) {
                picks.push((batch_index, row));
            }
        }
    }
    let batch = take_rows(&output.schema, &output.batches, &picks);
    QueryOutput {
        schema: output.schema,
        batches: vec![batch],
    }
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
            PlanItem::Computed { expr, name } => {
                computed_column(schema, views, expr, name, registry)?
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

/// Applies `HAVING`: the filter runs over the aggregate output rows —
/// hidden `__having{i}` columns included — through the same predicate
/// machinery WHERE uses (the output wraps as a query-lifetime segment,
/// so numeric and key comparison semantics cannot diverge), keeps the
/// TRUE rows (UNKNOWN filters, per SQL), and drops the hidden columns.
fn filter_having(
    output: QueryOutput,
    predicate: &Predicate,
    visible: usize,
) -> Result<QueryOutput, QueryError> {
    let full_schema = output.schema.clone();
    let mut picks: Vec<(usize, usize)> = Vec::new();
    for (batch_index, batch) in output.batches.iter().enumerate() {
        // Query-lifetime scratch: the predicate is evaluated row-wise
        // here, never zone-pruned, so computing maps would be waste.
        let view = SegmentView::all_live(Arc::new(Segment::from_batch_unpruned(
            batch.clone(),
            0,
            false,
        )));
        let matched = evaluate_predicate(predicate, &full_schema, &view)?;
        for row in 0..batch.num_rows() {
            if matched.get(row) {
                picks.push((batch_index, row));
            }
        }
    }
    let filtered = take_rows(&full_schema, &output.batches, &picks);
    let schema = Schema::new(full_schema.fields()[..visible].to_vec());
    let columns = filtered.columns()[..visible].to_vec();
    Ok(QueryOutput {
        schema: schema.clone(),
        batches: vec![RecordBatch::new(schema, columns)],
    })
}

/// The computed-projection slot (#49): evaluates a scalar expression
/// per view, vectorized over the live rows, three-valued throughout —
/// the output is a nullable `f64` column per view.
fn computed_column(
    schema: &Schema,
    views: &[&SegmentView],
    expr: &ScalarExpr,
    name: &str,
    registry: &Registry,
) -> Result<(Field, Vec<Column>), QueryError> {
    // A registered kernel must see the query's rows as ONE column:
    // storage segmentation is an internal detail, and a kernel with
    // window semantics (a rolling combinator) would otherwise reset at
    // segment boundaries. Pure expressions stay on the per-view path —
    // elementwise semantics don't care, and it copies nothing.
    //
    // The routing does NOT depend on how many segments the rows happen
    // to occupy: whether a query is accepted must not turn on an
    // internal detail, so a one-segment table takes the same path (and
    // meets the same refusals) as a hundred-segment one.
    if uses_registered(expr) {
        return computed_column_whole(schema, views, expr, name, registry);
    }
    let mut columns = Vec::with_capacity(views.len());
    for view in views {
        let (values, validity) = evaluate_scalar(expr, schema, view, registry)?;
        columns.push(Column::Numeric(NumericData::F64(assemble_f64_values(
            values, validity,
        ))));
    }
    Ok((Field::new(name, ColumnType::F64, true), columns))
}

/// Whether the expression calls a registered kernel anywhere.
fn uses_registered(expr: &ScalarExpr) -> bool {
    match expr {
        ScalarExpr::Registered { .. } => true,
        ScalarExpr::Column(_) | ScalarExpr::Literal(_) => false,
        ScalarExpr::Negate(inner) => uses_registered(inner),
        ScalarExpr::Binary { left, right, .. } => uses_registered(left) || uses_registered(right),
        ScalarExpr::Call { args, .. } => args.iter().any(uses_registered),
        ScalarExpr::Case { whens, otherwise } => {
            whens.iter().any(|(_, value)| uses_registered(value))
                || otherwise.as_deref().is_some_and(uses_registered)
        }
    }
}

/// Column names a registered-kernel expression reads. `CASE` is
/// refused here — its predicates would need their own gather (keys
/// included), which nothing motivates yet.
fn registered_columns(expr: &ScalarExpr, out: &mut Vec<String>) -> Result<(), QueryError> {
    match expr {
        ScalarExpr::Column(name) => {
            out.push(name.clone());
            Ok(())
        }
        ScalarExpr::Literal(_) => Ok(()),
        ScalarExpr::Negate(inner) => registered_columns(inner, out),
        ScalarExpr::Binary { left, right, .. } => {
            registered_columns(left, out)?;
            registered_columns(right, out)
        }
        ScalarExpr::Call { args, .. } | ScalarExpr::Registered { args, .. } => {
            for arg in args {
                registered_columns(arg, out)?;
            }
            Ok(())
        }
        ScalarExpr::Case { .. } => Err(QueryError::Unsupported(
            "CASE combined with a registered function in one expression \
             (lift the function out of the CASE)"
                .to_owned(),
        )),
    }
}

/// The whole-query path: gather the used columns dense (live rows, in
/// view order — a copy proportional to the queried rows, like the
/// cross-segment window gathers, not bounded by a constant), evaluate
/// the expression ONCE over a synthetic single view, and split the
/// result back per view.
fn computed_column_whole(
    schema: &Schema,
    views: &[&SegmentView],
    expr: &ScalarExpr,
    name: &str,
    registry: &Registry,
) -> Result<(Field, Vec<Column>), QueryError> {
    let mut names = Vec::new();
    registered_columns(expr, &mut names)?;
    names.sort();
    names.dedup();
    let total: usize = views.iter().map(|view| view.live_rows()).sum();
    // Column 0 is a synthetic i64 ordering key so the batch satisfies
    // the segment shape; the expression never references it.
    let mut fields = vec![Field::new("__row", ColumnType::I64, false)];
    let row_ids: Buffer<i64> = (0..total as i64).collect();
    let mut gathered: Vec<Column> = vec![Column::Numeric(NumericData::I64(
        NumericColumn::new_non_null(row_ids),
    ))];
    for column_name in &names {
        let mut values = Vec::with_capacity(total);
        let mut validity = Vec::with_capacity(total);
        for view in views {
            let column = ScalarExpr::Column(column_name.clone());
            let (mut v, mut m) = evaluate_scalar(&column, schema, view, registry)?;
            values.append(&mut v);
            validity.append(&mut m);
        }
        fields.push(Field::new(column_name.clone(), ColumnType::F64, true));
        gathered.push(Column::Numeric(NumericData::F64(assemble_f64_values(
            values, validity,
        ))));
    }
    let batch = RecordBatch::new(Schema::new(fields), gathered);
    let synthetic = SegmentView::all_live(Arc::new(Segment::from_batch_unpruned(batch, 0, false)));
    let reduced = synthetic.segment.batch().schema().clone();
    let (values, validity) = evaluate_scalar(expr, &reduced, &synthetic, registry)?;
    let mut columns = Vec::with_capacity(views.len());
    let mut offset = 0;
    for view in views {
        let count = view.live_rows();
        columns.push(Column::Numeric(NumericData::F64(assemble_f64_values(
            values[offset..offset + count].to_vec(),
            validity[offset..offset + count].to_vec(),
        ))));
        offset += count;
    }
    Ok((Field::new(name, ColumnType::F64, true), columns))
}

/// One view's worth of a scalar expression: `(values, validity)` over
/// the live rows, in stored order.
fn evaluate_scalar(
    expr: &ScalarExpr,
    schema: &Schema,
    view: &SegmentView,
    registry: &Registry,
) -> Result<(Vec<f64>, Vec<bool>), QueryError> {
    let rows = view.live_rows();
    match expr {
        ScalarExpr::Column(name) => {
            let (index, field) = resolve(schema, name)?;
            match &view.segment.batch().columns()[index] {
                Column::Numeric(NumericData::F64(numeric)) => {
                    let raw = numeric.values().as_slice();
                    // Bulk path: no tombstone mask means every row is
                    // live in stored order — one memcpy, no per-row
                    // filter loop.
                    if view.live.is_none() {
                        let validity = match numeric.validity() {
                            None => vec![true; rows],
                            Some(bitmap) => (0..rows).map(|row| bitmap.get(row)).collect(),
                        };
                        return Ok((raw.to_vec(), validity));
                    }
                    let mut values = Vec::with_capacity(rows);
                    let mut validity = Vec::with_capacity(rows);
                    for row in live_rows(view) {
                        values.push(raw[row]);
                        validity.push(numeric.is_valid(row));
                    }
                    Ok((values, validity))
                }
                Column::Numeric(NumericData::I64(_)) => Err(QueryError::TypeError(format!(
                    "column '{name}' is i64: exact integer expression arithmetic \
                     is not built (#40); cast at ingest or use an f64 column"
                ))),
                Column::Key(_) => Err(QueryError::TypeError(format!(
                    "column '{}' is a key: expressions are numeric (numeric-or-key)",
                    field.name()
                ))),
            }
        }
        ScalarExpr::Literal(value) => Ok((vec![*value; rows], vec![true; rows])),
        ScalarExpr::Negate(inner) => {
            let (mut values, validity) = evaluate_scalar(inner, schema, view, registry)?;
            for value in &mut values {
                *value = -*value;
            }
            Ok((values, validity))
        }
        ScalarExpr::Binary { op, left, right } => {
            let (lv, lval) = evaluate_scalar(left, schema, view, registry)?;
            let (rv, rval) = evaluate_scalar(right, schema, view, registry)?;
            let values = lv
                .iter()
                .zip(&rv)
                .map(|(&a, &b)| match op {
                    ArithOp::Add => a + b,
                    ArithOp::Sub => a - b,
                    ArithOp::Mul => a * b,
                    ArithOp::Div => a / b,
                    ArithOp::Mod => a % b,
                })
                .collect();
            let validity = lval.iter().zip(&rval).map(|(&a, &b)| a && b).collect();
            Ok((values, validity))
        }
        ScalarExpr::Call { function, args } => {
            let mut evaluated = Vec::with_capacity(args.len());
            for arg in args {
                evaluated.push(evaluate_scalar(arg, schema, view, registry)?);
            }
            let mut values = Vec::with_capacity(rows);
            let mut validity = Vec::with_capacity(rows);
            for row in 0..rows {
                let valid = evaluated.iter().all(|(_, v)| v[row]);
                validity.push(valid);
                let arg = |i: usize| evaluated[i].0[row];
                values.push(match function {
                    ScalarFunction::Abs => arg(0).abs(),
                    ScalarFunction::Round => arg(0).round(),
                    ScalarFunction::Floor => arg(0).floor(),
                    ScalarFunction::Ceil => arg(0).ceil(),
                    ScalarFunction::Sqrt => arg(0).sqrt(),
                    ScalarFunction::Ln => arg(0).ln(),
                    ScalarFunction::Exp => arg(0).exp(),
                    ScalarFunction::Power => arg(0).powf(arg(1)),
                });
            }
            Ok((values, validity))
        }
        ScalarExpr::Case { whens, otherwise } => {
            // Conditions evaluate vectorized, once per view (the WHERE
            // machinery); selection is then per live row, first TRUE arm
            // wins, UNKNOWN falls through like FALSE.
            let mut conditions = Vec::with_capacity(whens.len());
            let mut arms = Vec::with_capacity(whens.len());
            for (predicate, arm) in whens {
                conditions.push(evaluate_predicate(predicate, schema, view)?);
                arms.push(evaluate_scalar(arm, schema, view, registry)?);
            }
            let fallback = otherwise
                .as_ref()
                .map(|expr| evaluate_scalar(expr, schema, view, registry))
                .transpose()?;
            let mut values = vec![0.0f64; rows];
            let mut validity = vec![false; rows];
            for (out, row) in live_rows(view).enumerate() {
                let mut chosen = None;
                for (condition, arm) in conditions.iter().zip(&arms) {
                    if condition.get(row) {
                        chosen = Some((arm.0[out], arm.1[out]));
                        break;
                    }
                }
                let (value, valid) = chosen.unwrap_or_else(|| match &fallback {
                    Some((values, validity)) => (values[out], validity[out]),
                    None => (0.0, false),
                });
                values[out] = value;
                validity[out] = valid;
            }
            Ok((values, validity))
        }
        ScalarExpr::Registered { name, args } => {
            let Some(function) = registry.column(name) else {
                return Err(QueryError::Unsupported(format!(
                    "no registered column function '{name}' on this table \
                     (a window function needs OVER; register column \
                     functions through the table handle)"
                )));
            };
            if args.len() != function.arity() {
                return Err(QueryError::TypeError(format!(
                    "{name} takes {} argument(s), got {}",
                    function.arity(),
                    args.len()
                )));
            }
            // Arguments evaluate through this same machinery, so any
            // scalar expression composes into a registered call; the
            // kernel then runs once for the whole view.
            let mut evaluated = Vec::with_capacity(args.len());
            for arg in args {
                evaluated.push(evaluate_scalar(arg, schema, view, registry)?);
            }
            let dense: Vec<(&[f64], &[bool])> = evaluated
                .iter()
                .map(|(values, validity)| (values.as_slice(), validity.as_slice()))
                .collect();
            let results = function.evaluate(&dense).map_err(QueryError::Compute)?;
            if results.len() != rows {
                return Err(QueryError::Compute(format!(
                    "{name} returned {} results for {rows} rows",
                    results.len()
                )));
            }
            let mut values = Vec::with_capacity(rows);
            let mut validity = Vec::with_capacity(rows);
            for result in results {
                values.push(result.unwrap_or(0.0));
                validity.push(result.is_some());
            }
            Ok((values, validity))
        }
    }
}

/// Builds a numeric column from parallel values/validity — the one
/// values-plus-bitmap assembly every output path shares. The bitmap
/// exists only if some value is actually absent, same as storage.
fn assemble_numeric<T: arrow_lite::Element>(
    values: Buffer<T>,
    validity: Vec<bool>,
) -> NumericColumn<T> {
    if validity.iter().all(|&valid| valid) {
        NumericColumn::new_non_null(values)
    } else {
        NumericColumn::new_nullable(values, Bitmap::from_bools(validity))
    }
}

/// As [`assemble_numeric`], from a plain vector.
fn assemble_f64_values(values: Vec<f64>, validity: Vec<bool>) -> NumericColumn<f64> {
    assemble_numeric(Buffer::from_slice(&values), validity)
}

/// Looks up a column by name in the table schema.
fn resolve<'a>(schema: &'a Schema, name: &str) -> Result<(usize, &'a Field), QueryError> {
    schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == name)
        // Everything reaching here resolves against the *stored*
        // schema, where the pseudocolumn does not exist: projection
        // intercepts it first, so this is always a refusal.
        .ok_or_else(|| crate::plan::no_such_column(name))
}

/// The ingest-sequence pseudocolumn, materialized per view: every row's
/// birth sequence as `BIGINT`, live rows only, in stored order — the
/// same rows and order [`passthrough`] would hand back for a real
/// column. Sequences are `u64` in storage and `i64` here because SQL has
/// no unsigned type; the gap is unreachable (`i64::MAX` appends).
fn sequence_column(views: &[&SegmentView], name: &str) -> (Field, Vec<Column>) {
    let columns = views
        .iter()
        .map(|view| {
            let values: Buffer<i64> = live_rows(view)
                .map(|row| view.segment.sequence_at(row) as i64)
                .collect();
            Column::Numeric(NumericData::I64(NumericColumn::new_non_null(values)))
        })
        .collect();
    (Field::new(name, ColumnType::I64, false), columns)
}

/// Local row indices a reader sees, in stored order.
fn live_rows<'a>(view: &'a SegmentView) -> impl Iterator<Item = usize> + 'a {
    (0..view.segment.batch().num_rows()).filter(move |&row| view.is_live(row))
}

/// Rebuilds `column` with only the mask's live rows — the O(rows) copy a
/// masked segment costs its readers, whether the mask came from
/// tombstones or a `WHERE` predicate (paid even when the predicate keeps
/// every row — there is no "matches everything, skip the copy" fast path).
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
    if name == SEQUENCE_COLUMN {
        return Ok(sequence_column(views, alias.unwrap_or(name)));
    }
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

/// The window slice for row `position` in a run of rows: `preceding`
/// rows back (`None` = from the start of the run) through the current
/// row, ragged at the start of the run.
/// Frame-by-frame recomputation over one contiguous run — the
/// [`WindowAggregate::evaluate_frames`] default, exposed so an
/// incremental override can fall back to it for frame shapes it does
/// not accelerate.
pub fn recompute_frames<A: WindowAggregate + ?Sized>(
    aggregate: &A,
    columns: &[&[f64]],
    preceding: Option<usize>,
) -> Result<Vec<Option<f64>>, String> {
    let rows = columns.first().map_or(0, |column| column.len());
    let mut results = Vec::with_capacity(rows);
    let mut frame: Vec<&[f64]> = Vec::with_capacity(columns.len());
    for position in 0..rows {
        let (start, end) = window_bounds(position, preceding);
        frame.clear();
        frame.extend(columns.iter().map(|column| &column[start..end]));
        results.push(aggregate.evaluate(&frame)?);
    }
    Ok(results)
}

/// Calls the aggregate's frame-sequence evaluation and holds it to the
/// executor's contract: one result per row of the run.
fn evaluate_run(
    aggregate: &dyn WindowAggregate,
    columns: &[&[f64]],
    rows: usize,
    preceding: Option<usize>,
) -> Result<Vec<Option<f64>>, QueryError> {
    debug_assert!(columns.iter().all(|column| column.len() == rows));
    let results = aggregate
        .evaluate_frames(columns, preceding)
        .map_err(QueryError::Compute)?;
    if results.len() != rows {
        return Err(QueryError::Compute(format!(
            "window aggregate returned {} results for {rows} frames",
            results.len()
        )));
    }
    Ok(results)
}

fn window_bounds(position: usize, preceding: Option<usize>) -> (usize, usize) {
    let start = match preceding {
        Some(preceding) => position.saturating_sub(preceding),
        None => 0,
    };
    (start, position + 1)
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
    preceding: Option<usize>,
    alias: Option<&str>,
) -> Result<(Field, Vec<Column>), QueryError> {
    // The embedder's registry wins (explicit beats implicit); the
    // standard aggregates are always available as window functions.
    let builtin;
    let aggregate: &Arc<dyn WindowAggregate> = match registry.get(function) {
        Some(aggregate) => aggregate,
        None => match builtin_window(function) {
            Some(aggregate) => {
                builtin = aggregate;
                &builtin
            }
            None => return Err(QueryError::UnknownFunction(function.to_owned())),
        },
    };
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
    let output_type = aggregate.output_type();
    let columns = results
        .into_iter()
        .map(|result| match output_type {
            // An I64-typed window (COUNT) produces integral f64 values;
            // cast them back exactly (B5). Others stay f64.
            ColumnType::I64 => assemble_i64_from_f64(result),
            _ => assemble_f64(result),
        })
        .collect();
    Ok((Field::new(name, output_type, true), columns))
}

/// Unpartitioned windows run over the snapshot's live rows in append
/// order. A single mask-free view is the pure zero-copy path — every
/// window is a direct slice of the stored buffer. Otherwise each
/// argument is gathered once into contiguous scratch (windows span view
/// boundaries; the stored buffers don't) and the windows slice that.
fn unpartitioned(
    aggregate: &dyn WindowAggregate,
    args: &[Vec<ArgValues<'_>>],
    preceding: Option<usize>,
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
    let rows: usize = results.iter().map(Vec::len).sum();
    let mut outputs = evaluate_run(aggregate, &arg_slices, rows, preceding)?.into_iter();
    for result in results.iter_mut() {
        for slot in result.iter_mut() {
            *slot = outputs.next().expect("length checked by evaluate_run");
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
    preceding: Option<usize>,
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
    for (values, rows) in scratch.iter().zip(&origins) {
        let columns: Vec<&[f64]> = values.iter().map(Vec::as_slice).collect();
        let outputs = evaluate_run(aggregate, &columns, rows.len(), preceding)?;
        for (output, &(view_index, live_position)) in outputs.into_iter().zip(rows) {
            results[view_index][live_position] = output;
        }
    }
    Ok(())
}

/// The standard aggregates as window functions — always available, no
/// registration needed (an embedder registration of the same name
/// wins). Each sees the window's rows as one non-null `f64` slice, the
/// executor's window-argument contract.
struct BuiltinWindow(AggFunction);

impl WindowAggregate for BuiltinWindow {
    fn arity(&self) -> usize {
        1
    }

    fn output_type(&self) -> ColumnType {
        // COUNT is an integer count (SQL/DuckDB return BIGINT); the rest
        // are f64 over f64 arguments. (i64 SUM/MIN/MAX windows are refused
        // upstream — that's the separate #40.)
        match self.0 {
            AggFunction::Count => ColumnType::I64,
            _ => ColumnType::F64,
        }
    }

    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        let window = args[0];
        if window.is_empty() {
            return Ok(None);
        }
        Ok(Some(match self.0 {
            AggFunction::Count => window.len() as f64,
            AggFunction::Sum => window.iter().sum(),
            AggFunction::Avg => window.iter().sum::<f64>() / window.len() as f64,
            AggFunction::Min => window
                .iter()
                .copied()
                .min_by(|left, right| cmp_f64(*left, *right))
                .expect("window is non-empty"),
            AggFunction::Max => window
                .iter()
                .copied()
                .max_by(|left, right| cmp_f64(*left, *right))
                .expect("window is non-empty"),
        }))
    }
}

fn builtin_window(function: &str) -> Option<Arc<dyn WindowAggregate>> {
    let function = match function {
        "count" => AggFunction::Count,
        "sum" => AggFunction::Sum,
        "avg" => AggFunction::Avg,
        "min" => AggFunction::Min,
        "max" => AggFunction::Max,
        _ => return None,
    };
    Some(Arc::new(BuiltinWindow(function)))
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

    /// The Arrow type this accumulator's output column takes — the single
    /// source of truth for the aggregate's result type, read from a
    /// template accumulator so the type comes from the *plan*, never from
    /// how many groups happened to match (B4).
    fn column_type(&self) -> ColumnType {
        match self {
            Accumulator::CountStar(_)
            | Accumulator::CountColumn(_)
            | Accumulator::SumI64 { .. }
            | Accumulator::MinMaxI64 { .. } => ColumnType::I64,
            Accumulator::SumF64 { .. }
            | Accumulator::Avg { .. }
            | Accumulator::MinMaxF64 { .. } => ColumnType::F64,
        }
    }

    /// This accumulator's result as an `i64` (for an `I64`-typed column);
    /// `None` = SQL NULL (no rows seen). Only called on i64 variants.
    fn i64_value(&self) -> Option<i64> {
        match self {
            Accumulator::CountStar(count) | Accumulator::CountColumn(count) => Some(*count),
            Accumulator::SumI64 { sum, seen } => seen.then_some(*sum),
            Accumulator::MinMaxI64 { value, seen, .. } => seen.then_some(*value),
            _ => None,
        }
    }

    /// This accumulator's result as an `f64` (for an `F64`-typed column);
    /// `None` = SQL NULL. Only called on f64 variants.
    fn f64_value(&self) -> Option<f64> {
        match self {
            Accumulator::SumF64 { sum, seen } => seen.then_some(*sum),
            Accumulator::Avg { sum, count } => (*count > 0).then(|| sum / *count as f64),
            Accumulator::MinMaxF64 { value, seen, .. } => seen.then_some(*value),
            _ => None,
        }
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
                // The one f64 relation (cmp_f64): NaN is greatest, matching
                // WHERE and pruning — not total_cmp's bitwise order (B3).
                let replace = !*seen
                    || (*max && cmp_f64(candidate, *value).is_gt())
                    || (!*max && cmp_f64(candidate, *value).is_lt());
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
                // Output type from the plan (the template accumulator), so
                // it is the same whether or not any group matched (B4).
                let output_type = template[next_call].column_type();
                let (field, column) = assemble_aggregate(
                    name,
                    output_type,
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

/// One aggregate output column from its per-group accumulators. The
/// column's `output_type` comes from the plan (a template accumulator),
/// not from the accumulator instances here — so zero groups still yields
/// the right Arrow type (B4).
fn assemble_aggregate<'a>(
    name: &str,
    output_type: ColumnType,
    accumulators: impl Iterator<Item = &'a Accumulator>,
    groups: usize,
) -> (Field, Column) {
    match output_type {
        ColumnType::I64 => {
            let mut cells: Vec<Option<i64>> = Vec::with_capacity(groups);
            for accumulator in accumulators {
                cells.push(accumulator.i64_value());
            }
            let nullable = cells.iter().any(Option::is_none);
            let values: Buffer<i64> = cells.iter().map(|value| value.unwrap_or(0)).collect();
            let column = if nullable {
                NumericColumn::new_nullable(
                    values,
                    Bitmap::from_bools(cells.iter().map(Option::is_some)),
                )
            } else {
                NumericColumn::new_non_null(values)
            };
            (
                Field::new(name, ColumnType::I64, nullable),
                Column::Numeric(NumericData::I64(column)),
            )
        }
        // Aggregates are numeric; a Key output can't arise.
        _ => {
            let mut cells: Vec<Option<f64>> = Vec::with_capacity(groups);
            for accumulator in accumulators {
                cells.push(accumulator.f64_value());
            }
            let nullable = cells.iter().any(Option::is_none);
            let values: Buffer<f64> = cells.iter().map(|value| value.unwrap_or(0.0)).collect();
            let column = if nullable {
                NumericColumn::new_nullable(
                    values,
                    Bitmap::from_bools(cells.iter().map(Option::is_some)),
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
}

/// A sortable view of one output cell. Numbers only: symbol columns
/// are unordered labels (#58, ruled 2026-07-29), so nothing else can
/// reach the sort.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum SortCell {
    I64(i64),
    F64(f64),
}

/// One row's place in the sort: its key cell, where the row lives, and
/// its input position — which breaks ties, so every path here returns
/// exactly the stable order.
type SortEntry = (Option<SortCell>, (usize, usize), usize);

/// Sorts the output by one column into a single batch. Nulls sort
/// **last in both directions**, DuckDB's default (PostgreSQL flips them
/// under DESC; when the two disagree we follow our oracle); `f64` uses
/// total order, so NaN sorts above every number.
///
/// `keep` is how many leading rows the query can actually use —
/// `OFFSET + LIMIT`, or `None` when it asks for all of them. Given a
/// bound, this takes the **top-k** path (#80): one sweep holding a
/// heap of the k best rows, so the work is O(n log k) and the memory
/// O(k), where the full sort costs O(n log n) and materializes a
/// sorted copy of everything — a ten-row answer no longer pays for a
/// million-row sort. Both paths return the same rows in the same
/// order; the tie-breaking position is what makes that exact rather
/// than merely equivalent.
///
/// A symbol column is refused (#58 = B, ruled 2026-07-29): its codes
/// are per-segment first-appearance ranks, so they carry no order to
/// sort by, and the alternative — ranking the labels as text — asks an
/// engine that refuses to *produce* a string to rank strings, in a byte
/// order that is only "alphabetical" for ASCII. Identities are not
/// ordered; the arithmetic refusal and this one are the same rule.
fn sort_output(
    output: QueryOutput,
    order_by: &OrderBy,
    keep: Option<usize>,
) -> Result<QueryOutput, QueryError> {
    let (column_index, field) = resolve(&output.schema, &order_by.column)?;
    if field.column_type() == ColumnType::Key {
        return Err(QueryError::TypeError(format!(
            "ORDER BY '{}': symbol columns are unordered labels, not ordered text \
             — group or filter by them, and order by a number (their codes are \
             per-segment, so they rank nothing)",
            order_by.column
        )));
    }
    let cell = |batch: &RecordBatch, row: usize| -> Option<SortCell> {
        match &batch.columns()[column_index] {
            Column::Numeric(NumericData::F64(numeric)) => numeric
                .is_valid(row)
                .then(|| SortCell::F64(numeric.values().as_slice()[row])),
            Column::Numeric(NumericData::I64(numeric)) => numeric
                .is_valid(row)
                .then(|| SortCell::I64(numeric.values().as_slice()[row])),
            Column::Key(_) => unreachable!("symbol columns are refused above"),
        }
    };
    // Nulls last in both directions unless the query says otherwise
    // (NULLS FIRST/LAST) — placement sits outside the DESC reversal.
    let null_order = if order_by.nulls_first.unwrap_or(false) {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    };
    let compare = |left: &SortEntry, right: &SortEntry| {
        let ordering = match (&left.0, &right.0) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => null_order,
            (Some(_), None) => null_order.reverse(),
            (Some(left), Some(right)) => {
                let ordering = match (left, right) {
                    (SortCell::F64(left), SortCell::F64(right)) => cmp_f64(*left, *right),
                    (left, right) => left.partial_cmp(right).expect("same variant per column"),
                };
                if order_by.descending {
                    ordering.reverse()
                } else {
                    ordering
                }
            }
        };
        // Input position breaks every tie: this is what a stable sort
        // does, and holding to it keeps top-k's answer identical to
        // the full sort's rather than merely as valid.
        ordering.then(left.2.cmp(&right.2))
    };
    let rows = output.num_rows();
    let mut entries: Vec<SortEntry> = Vec::with_capacity(keep.unwrap_or(rows).min(rows));
    let mut position = 0usize;
    for (batch_index, batch) in output.batches.iter().enumerate() {
        for row in 0..batch.num_rows() {
            let entry = (cell(batch, row), (batch_index, row), position);
            position += 1;
            match keep {
                // Bounded: the heap's root is the worst row kept, so a
                // candidate is admitted only by displacing it — and
                // nothing else is ever retained.
                Some(k) if entries.len() == k => {
                    if k > 0 && compare(&entry, &entries[0]) == std::cmp::Ordering::Less {
                        entries[0] = entry;
                        sift_down(&mut entries, 0, &compare);
                    }
                }
                _ => {
                    entries.push(entry);
                    let last = entries.len() - 1;
                    sift_up(&mut entries, last, &compare);
                }
            }
        }
    }
    entries.sort_by(compare);
    let picks: Vec<(usize, usize)> = entries.into_iter().map(|entry| entry.1).collect();
    let batch = take_rows(&output.schema, &output.batches, &picks);
    Ok(QueryOutput {
        schema: output.schema,
        batches: vec![batch],
    })
}

/// Restores the max-heap property upward from `index` — the worst
/// entry (greatest under `compare`) rises toward the root.
fn sift_up(
    heap: &mut [SortEntry],
    mut index: usize,
    compare: &impl Fn(&SortEntry, &SortEntry) -> std::cmp::Ordering,
) {
    while index > 0 {
        let parent = (index - 1) / 2;
        if compare(&heap[index], &heap[parent]) != std::cmp::Ordering::Greater {
            break;
        }
        heap.swap(index, parent);
        index = parent;
    }
}

/// Restores the max-heap property downward from `index`.
fn sift_down(
    heap: &mut [SortEntry],
    mut index: usize,
    compare: &impl Fn(&SortEntry, &SortEntry) -> std::cmp::Ordering,
) {
    loop {
        let (left, right) = (2 * index + 1, 2 * index + 2);
        let mut worst = index;
        if left < heap.len() && compare(&heap[left], &heap[worst]) == std::cmp::Ordering::Greater {
            worst = left;
        }
        if right < heap.len() && compare(&heap[right], &heap[worst]) == std::cmp::Ordering::Greater
        {
            worst = right;
        }
        if worst == index {
            return;
        }
        heap.swap(index, worst);
        index = worst;
    }
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

/// Every row of `output`, in order, as **one** batch — for consumers
/// that need contiguity (the script driver hands result columns to Lua
/// as single views, so it needs exactly this). A result already in one
/// batch is moved out untouched; several batches pay one gather, with
/// key columns re-encoded into a merged dictionary because per-segment
/// dictionaries do not share codes. An empty result still yields
/// correctly-typed empty columns.
///
/// This lives here, beside the row gather it delegates to, so the
/// merge-and-remap rule has exactly one implementation in the
/// workspace.
pub fn contiguous(output: QueryOutput) -> RecordBatch {
    let QueryOutput {
        schema,
        mut batches,
    } = output;
    if batches.len() == 1 {
        return batches.pop().expect("length checked");
    }
    let picks: Vec<(usize, usize)> = batches
        .iter()
        .enumerate()
        .flat_map(|(batch, rows)| (0..rows.num_rows()).map(move |row| (batch, row)))
        .collect();
    take_rows(&schema, &batches, &picks)
}

/// Gathers `picks` (batch, row) into one batch. Key columns re-encode
/// into a fresh dictionary — the sources' per-segment dictionaries don't
/// share codes.
fn take_rows(schema: &Schema, batches: &[RecordBatch], picks: &[(usize, usize)]) -> RecordBatch {
    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(column_index, field)| {
            let cell_column = |batch: usize| &batches[batch].columns()[column_index];
            // The schema, not a sample batch, decides each column's
            // variant: with zero picks there is no batch to sample, and
            // the empty result still needs correctly-typed columns.
            match field.column_type() {
                ColumnType::F64 => {
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
                ColumnType::I64 => {
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
                ColumnType::Key => {
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
                    Column::Key(assemble_key(codes, validity, dictionary))
                }
            }
        })
        .collect();
    RecordBatch::new(schema.clone(), columns)
}

fn assemble_numeric_f64(values: Buffer<f64>, validity: Vec<bool>) -> Column {
    Column::Numeric(NumericData::F64(assemble_numeric(values, validity)))
}

fn assemble_numeric_i64(values: Buffer<i64>, validity: Vec<bool>) -> Column {
    Column::Numeric(NumericData::I64(assemble_numeric(values, validity)))
}

/// One view's output column: nullable f64, bitmap only if a window
/// actually came back undefined.
fn assemble_f64(results: Vec<Option<f64>>) -> Column {
    let validity: Vec<bool> = results.iter().map(Option::is_some).collect();
    let values: Buffer<f64> = results.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    assemble_numeric_f64(values, validity)
}

/// Materializes an `i64` output column from integral `f64` results — the
/// `COUNT`-window path (B5). Each present value is an exact integer count,
/// so the cast is lossless.
fn assemble_i64_from_f64(results: Vec<Option<f64>>) -> Column {
    let validity: Vec<bool> = results.iter().map(Option::is_some).collect();
    let values: Buffer<i64> = results
        .into_iter()
        .map(|v| v.map_or(0, |x| x as i64))
        .collect();
    assemble_numeric_i64(values, validity)
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

    /// The `_seq` values of a result, flattened across batches.
    fn seq_values(output: &QueryOutput, index: usize) -> Vec<i64> {
        output
            .batches
            .iter()
            .flat_map(|batch| {
                let Column::Numeric(NumericData::I64(column)) = &batch.columns()[index] else {
                    panic!("_seq is i64")
                };
                column.values().as_slice().to_vec()
            })
            .collect()
    }

    #[test]
    fn the_sequence_pseudocolumn_reads_each_rows_coordinate() {
        let rows: Vec<(i64, &str, f64)> = (0..7)
            .map(|i| (i as i64, "A", i as f64))
            .collect::<Vec<_>>();
        let views = segmented(&rows, 3);
        let output = run(&views, "SELECT _seq, ts FROM t").unwrap();
        // A table that never diverged: sequence == row id == ts here.
        assert_eq!(seq_values(&output, 0), (0..7).collect::<Vec<i64>>());
        assert_eq!(output.schema.fields()[0].name(), "_seq");
        assert_eq!(output.schema.fields()[0].column_type(), ColumnType::I64);
        assert!(!output.schema.fields()[0].nullable());
        // It is a column like any other downstream — order and page by
        // it, which is how a session reads back its latest coordinate.
        let latest = run(&views, "SELECT _seq FROM t ORDER BY _seq DESC LIMIT 1").unwrap();
        assert_eq!(seq_values(&latest, 0), [6]);
        let aliased = run(&views, "SELECT _seq AS at FROM t ORDER BY at DESC LIMIT 1").unwrap();
        assert_eq!(aliased.schema.fields()[0].name(), "at");
        assert_eq!(seq_values(&aliased, 0), [6]);
    }

    #[test]
    fn the_sequence_pseudocolumn_follows_the_live_mask() {
        // WHERE narrows the rows; the coordinates must stay attached to
        // the rows that survive, not renumber to their output position.
        let rows: Vec<(i64, &str, f64)> = (0..6)
            .map(|i| (i as i64, if i % 2 == 0 { "A" } else { "B" }, i as f64))
            .collect::<Vec<_>>();
        let views = segmented(&rows, 2);
        let output = run(&views, "SELECT _seq, ts FROM t WHERE sym = 'B'").unwrap();
        assert_eq!(seq_values(&output, 0), [1, 3, 5]);
    }

    #[test]
    fn the_sequence_pseudocolumn_is_projection_only() {
        let views = segment(&[(1, "A", 1.0)]);
        for sql in [
            "SELECT ts FROM t WHERE _seq > 0",
            "SELECT count(*) AS n FROM t GROUP BY _seq",
            "SELECT _seq + 1 AS next FROM t",
        ] {
            let error = run(&views, sql).unwrap_err().to_string();
            assert!(error.contains("can be selected"), "{sql}: {error}");
        }
    }

    #[test]
    fn top_k_returns_exactly_the_leading_rows_of_the_full_sort() {
        // The bounded heap must not merely produce *a* valid answer: it
        // must produce the same rows, in the same order, as sorting
        // everything and cutting — ties included, which is where a heap
        // that ignores input position drifts from a stable sort.
        let rows: Vec<(i64, &str, f64)> = (0..40)
            .map(|i| {
                // x repeats every 4 rows: 10 four-way ties, so almost
                // every k lands inside a tie group.
                (i as i64, ["A", "B", "C"][i % 3], (i % 4) as f64)
            })
            .collect();
        let views = segmented(&rows, 7);
        let ts_of = |output: &QueryOutput| -> Vec<i64> {
            output
                .batches
                .iter()
                .flat_map(|batch| {
                    let Column::Numeric(NumericData::I64(column)) = &batch.columns()[0] else {
                        panic!("ts is i64")
                    };
                    column.values().as_slice().to_vec()
                })
                .collect()
        };
        for order in ["ORDER BY x", "ORDER BY x DESC", "ORDER BY ts DESC"] {
            let full = ts_of(&run(&views, &format!("SELECT ts, sym, x FROM t {order}")).unwrap());
            for k in [0usize, 1, 3, 9, 39, 40, 100] {
                let bounded = ts_of(
                    &run(
                        &views,
                        &format!("SELECT ts, sym, x FROM t {order} LIMIT {k}"),
                    )
                    .unwrap(),
                );
                assert_eq!(bounded, full[..k.min(full.len())], "{order} LIMIT {k}");
            }
            // OFFSET rides along: the bound is offset + limit, and the
            // window taken from it must still match the full sort's.
            for (offset, limit) in [(0, 5), (5, 5), (37, 5), (40, 1)] {
                let window = ts_of(
                    &run(
                        &views,
                        &format!("SELECT ts, sym, x FROM t {order} LIMIT {limit} OFFSET {offset}"),
                    )
                    .unwrap(),
                );
                let expected: Vec<i64> = full
                    .iter()
                    .copied()
                    .skip(offset)
                    .take(limit)
                    .collect::<Vec<_>>();
                assert_eq!(window, expected, "{order} LIMIT {limit} OFFSET {offset}");
            }
        }
    }

    #[test]
    fn symbol_columns_cannot_be_ordered_by() {
        // #58 = B: labels are identities, and identities have no order
        // — not through their codes (per-segment first-appearance
        // ranks) and not through their text (an engine that refuses to
        // produce a string does not rank strings).
        let views = segmented(&[(1, "C", 1.0), (2, "A", 2.0), (3, "B", 3.0)], 2);
        for sql in [
            "SELECT sym, x FROM t ORDER BY sym",
            "SELECT sym, x FROM t ORDER BY sym DESC",
            "SELECT sym, x FROM t ORDER BY sym LIMIT 1",
            // Through an alias, and through a grouped projection —
            // wherever the output column is a symbol.
            "SELECT sym AS label, x FROM t ORDER BY label",
            "SELECT sym, count(*) AS n FROM t GROUP BY sym ORDER BY sym",
        ] {
            let error = run(&views, sql).unwrap_err().to_string();
            assert!(error.contains("unordered labels"), "{sql}: {error}");
        }
        // The rest of the surface is untouched: group by them, filter
        // by them, order by a number.
        assert!(run(&views, "SELECT sym, x FROM t ORDER BY x").is_ok());
        assert!(run(&views, "SELECT sym, count(*) AS n FROM t GROUP BY sym").is_ok());
        assert!(run(&views, "SELECT ts FROM t WHERE sym = 'A'").is_ok());
    }

    #[test]
    fn top_k_places_nulls_where_the_full_sort_does() {
        // Null placement is the other thing the heap has to carry: a
        // NULLS FIRST query's k rows are the null ones.
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("y", ColumnType::F64, true),
        ]);
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        for i in 0..12i64 {
            let y = if i % 3 == 0 {
                RowValue::Null
            } else {
                RowValue::F64((12 - i) as f64)
            };
            buffer.append(&[RowValue::I64(i), y]).unwrap();
        }
        let views = vec![SegmentView::all_live(Arc::new(buffer.freeze().unwrap()))];
        let registry = registry();
        let go = |sql: &str| -> Vec<i64> {
            let output =
                execute(&schema, &views, &crate::plan::plan(sql).unwrap(), &registry).unwrap();
            output
                .batches
                .iter()
                .flat_map(|batch| {
                    let Column::Numeric(NumericData::I64(column)) = &batch.columns()[0] else {
                        panic!("ts is i64")
                    };
                    column.values().as_slice().to_vec()
                })
                .collect()
        };
        assert_eq!(
            go("SELECT ts, y FROM t ORDER BY y NULLS FIRST LIMIT 4"),
            [0, 3, 6, 9]
        );
        assert_eq!(
            go("SELECT ts, y FROM t ORDER BY y NULLS FIRST LIMIT 4"),
            go("SELECT ts, y FROM t ORDER BY y NULLS FIRST")[..4]
        );
        assert_eq!(
            go("SELECT ts, y FROM t ORDER BY y DESC LIMIT 3"),
            go("SELECT ts, y FROM t ORDER BY y DESC")[..3]
        );
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
    fn aggregates_and_order_by_use_the_where_relation_for_nan() {
        // B3: MIN/MAX and ORDER BY use the one f64 relation (cmp_f64, NaN
        // greatest) that WHERE and pruning use — not total_cmp, which ranks
        // -NaN below -inf. With a -NaN row: MAX is NaN, MIN is the finite
        // minimum, and ORDER BY DESC puts NaN first.
        let views = segment(&[(1, "A", 1.0), (2, "A", -f64::NAN), (3, "A", 5.0)]);
        let hi = run(&views, "SELECT max(x) FROM t GROUP BY sym").unwrap();
        assert!(
            flatten(&hi, 0)[0].unwrap().is_nan(),
            "MAX must be NaN (greatest), not the finite max"
        );
        let lo = run(&views, "SELECT min(x) FROM t GROUP BY sym").unwrap();
        assert_eq!(flatten(&lo, 0), [Some(1.0)]);
        let sorted = run(&views, "SELECT x FROM t ORDER BY x DESC").unwrap();
        let xs = flatten(&sorted, 0);
        assert!(xs[0].unwrap().is_nan(), "NaN sorts first under DESC");
        assert_eq!((xs[1], xs[2]), (Some(5.0), Some(1.0)));
    }

    #[test]
    fn empty_group_aggregate_types_from_the_plan() {
        // B4: zero groups must still export COUNT as i64 — the type comes
        // from the plan (a template accumulator), not from instances that
        // never got created.
        let out = run(&[], "SELECT sym, count(*) FROM t GROUP BY sym").unwrap();
        assert_eq!(out.batches.len(), 0);
        assert_eq!(out.schema.fields()[1].column_type(), ColumnType::I64);
    }

    #[test]
    fn window_count_output_is_i64() {
        // B5: a COUNT window returns an integer column, like SQL/DuckDB.
        let views = segment(&[(1, "A", 1.0), (2, "A", 2.0)]);
        let out = run(
            &views,
            "SELECT count(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
             FROM t",
        )
        .unwrap();
        assert_eq!(out.schema.fields()[0].column_type(), ColumnType::I64);
        let Column::Numeric(NumericData::I64(n)) = &out.batches[0].columns()[0] else {
            panic!("count window must be i64")
        };
        assert_eq!(n.values().as_slice(), [1, 2]);
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
                 min(x) AS low, max(x) AS high FROM t GROUP BY sym",
            )
            .unwrap();
            assert_eq!(output.batches.len(), 1);
            let batch = &output.batches[0];
            let Column::Key(sym) = &batch.columns()[0] else {
                panic!("sym type")
            };
            // Group order is arbitrary — a symbol column cannot be
            // ordered by (#58) — so the groups are read by label. Here
            // first appearance happens to be A then B.
            let labels: Vec<&str> = (0..batch.num_rows())
                .map(|row| sym.value_at(row).expect("no null group"))
                .collect();
            let a = labels.iter().position(|&label| label == "A").expect("A");
            let b = labels.iter().position(|&label| label == "B").expect("B");
            let Column::Numeric(NumericData::I64(n)) = &batch.columns()[1] else {
                panic!("count type")
            };
            assert_eq!([n.values().as_slice()[a], n.values().as_slice()[b]], [3, 2]);
            let column =
                |index: usize, row: usize| f64_column(batch, index).values().as_slice()[row];
            assert_eq!([column(2, a), column(2, b)], [9.0, 30.0]);
            assert_eq!([column(3, a), column(3, b)], [3.0, 15.0]);
            assert_eq!([column(4, a), column(4, b)], [1.0, 10.0]);
            assert_eq!([column(5, a), column(5, b)], [6.0, 20.0]);
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
            // Symbols are unordered labels: ORDER BY refuses them
            // rather than ranking codes or rendering text (#58).
            let error = run(&views, "SELECT sym, x FROM t ORDER BY sym")
                .unwrap_err()
                .to_string();
            assert!(error.contains("unordered labels"), "{error}");
            // LIMIT without ORDER BY keeps ingest order.
            let output = run(&views, "SELECT ts FROM t LIMIT 3").unwrap();
            let Column::Numeric(NumericData::I64(ts)) = &output.batches[0].columns()[0] else {
                panic!("ts type")
            };
            assert_eq!(ts.values().as_slice(), &[1, 2, 3]);
        }
    }

    #[test]
    fn order_by_nulls_sort_last_in_both_directions() {
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
        // DuckDB's default (our oracle): nulls last under DESC too.
        assert_eq!(flatten(&descending, 0), [Some(3.0), Some(2.0), None]);
    }

    #[test]
    fn executor_routes_windows_through_the_frame_sequence() {
        // An aggregate that only answers through evaluate_frames: if the
        // executor ever took the per-frame path, the query would error.
        // Proves the sequence seam is the one road in.
        struct SequenceOnly;
        impl WindowAggregate for SequenceOnly {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, _: &[&[f64]]) -> Result<Option<f64>, String> {
                Err("per-frame path used".to_owned())
            }
            fn evaluate_frames(
                &self,
                columns: &[&[f64]],
                preceding: Option<usize>,
            ) -> Result<Vec<Option<f64>>, String> {
                // Frame sums, carried incrementally: enough state to
                // prove the whole run arrives in one call.
                let column = columns[0];
                let mut sum = 0.0f64;
                Ok((0..column.len())
                    .map(|position| {
                        sum += column[position];
                        if let Some(preceding) = preceding {
                            if position > preceding {
                                sum -= column[position - preceding - 1];
                            }
                        }
                        Some(sum)
                    })
                    .collect())
            }
        }
        let views = segment(&[(1, "A", 1.0), (2, "B", 2.0), (3, "A", 4.0)]);
        let mut registry = Registry::new();
        registry.register("runsum", Arc::new(SequenceOnly));
        let sql = "SELECT runsum(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND \
                   CURRENT ROW) AS w FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        );
        assert_eq!(
            flatten(&output.unwrap(), 0),
            [Some(1.0), Some(3.0), Some(6.0)]
        );
        // Partitioned: each key's rows form their own run.
        let sql = "SELECT runsum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN 1 \
                   PRECEDING AND CURRENT ROW) AS w FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        );
        assert_eq!(
            flatten(&output.unwrap(), 0),
            [Some(1.0), Some(2.0), Some(5.0)]
        );
    }

    #[test]
    fn a_wrong_length_frame_sequence_is_an_error_not_a_panic() {
        struct ShortChanger;
        impl WindowAggregate for ShortChanger {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, _: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok(Some(0.0))
            }
            fn evaluate_frames(
                &self,
                _: &[&[f64]],
                _: Option<usize>,
            ) -> Result<Vec<Option<f64>>, String> {
                Ok(vec![Some(0.0)]) // one result, however many frames
            }
        }
        let views = segment(&[(1, "A", 1.0), (2, "B", 2.0), (3, "A", 4.0)]);
        let mut registry = Registry::new();
        registry.register("short", Arc::new(ShortChanger));
        let sql = "SELECT short(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND \
                   CURRENT ROW) AS w FROM t";
        let error = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap_err();
        assert!(
            matches!(&error, QueryError::Compute(message) if message.contains("1 results for 3 frames")),
            "{error:?}"
        );
    }

    #[test]
    fn distinct_deduplicates_by_value_across_segments() {
        // Two segments (threshold 2) so 'A' appears under different
        // dictionary codes; DISTINCT must merge them by value.
        let views = segmented(
            &[(1, "A", 1.0), (2, "B", 1.0), (3, "A", 1.0), (4, "A", 2.0)],
            2,
        );
        let registry = Registry::new();
        let sql = "SELECT sym, x FROM t";
        let all = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(all.num_rows(), 4);
        let sql = "SELECT DISTINCT sym, x FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(output.num_rows(), 3); // (A,1) (B,1) (A,2)
        assert_eq!(output.batches.len(), 1, "DISTINCT consolidates");
        let sql = "SELECT DISTINCT x FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&output, 0), [Some(1.0), Some(2.0)]);
    }

    #[test]
    fn computed_projections_evaluate_vectorized() {
        let views = segmented(&[(1, "A", 1.0), (2, "B", 4.0), (3, "A", 9.0)], 2);
        let registry = Registry::new();
        let sql = "SELECT x * 2 + 1 AS y, SQRT(x) AS r FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&output, 0), [Some(3.0), Some(9.0), Some(19.0)]);
        assert_eq!(flatten(&output, 1), [Some(1.0), Some(2.0), Some(3.0)]);
        // Unaliased names render the expression's SQL text.
        let sql = "SELECT x + 1 FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(output.schema.fields()[0].name(), "x + 1");
        // Nested function arguments work.
        let sql = "SELECT ABS(1 - x) AS d FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&output, 0), [Some(0.0), Some(3.0), Some(8.0)]);
        // i64 and key columns are refused loudly (numeric-or-key, #40).
        for sql in ["SELECT ts + 1 FROM t", "SELECT sym * 2 FROM t"] {
            let error = crate::plan::plan(sql)
                .and_then(|plan| execute(&schema(), &views, &plan, &registry));
            assert!(error.is_err(), "{sql} should be refused");
        }
    }

    #[test]
    fn case_selects_per_row_with_three_valued_conditions() {
        let views = segmented(&[(1, "A", 1.0), (2, "B", 2.0), (3, "A", 3.0)], 2);
        let registry = Registry::new();
        let sql = "SELECT CASE WHEN x > 2 THEN 100 WHEN sym = 'A' THEN x ELSE 0 END AS c FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&output, 0), [Some(1.0), Some(0.0), Some(100.0)]);
        // Missing ELSE yields NULL.
        let sql = "SELECT CASE WHEN x > 2 THEN 1 END AS c FROM t";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&output, 0), [None, None, Some(1.0)]);
    }

    #[test]
    fn having_filters_groups_and_hides_its_columns() {
        let views = segmented(
            &[
                (1, "A", 1.0),
                (2, "B", 2.0),
                (3, "A", 3.0),
                (4, "B", 4.0),
                (5, "C", 10.0),
            ],
            2,
        );
        let registry = Registry::new();
        let sql = "SELECT sym, SUM(x) AS s FROM t GROUP BY sym HAVING SUM(x) > 4 ORDER BY s";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(output.schema.fields().len(), 2, "hidden columns dropped");
        assert_eq!(flatten(&output, 1), [Some(6.0), Some(10.0)]); // B=6, C=10... A=4 filtered
                                                                  // HAVING may reference the group key too.
        let sql = "SELECT sym, COUNT(x) AS c FROM t GROUP BY sym HAVING sym <> 'C'";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(output.num_rows(), 2);
        // HAVING without aggregation is refused toward WHERE.
        assert!(crate::plan::plan("SELECT x FROM t HAVING x > 1").is_err());
    }

    #[test]
    fn like_filters_keys_per_distinct_value() {
        let views = segment(&[(1, "AAPL", 1.0), (2, "MSFT", 2.0), (3, "AMZN", 3.0)]);
        let registry = Registry::new();
        let sql = "SELECT x FROM t WHERE sym LIKE 'A%'";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        );
        assert_eq!(flatten(&output.unwrap(), 0), [Some(1.0), Some(3.0)]);
        let sql = "SELECT x FROM t WHERE sym NOT LIKE '_S%'";
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(sql).unwrap(),
            &registry,
        );
        assert_eq!(flatten(&output.unwrap(), 0), [Some(1.0), Some(3.0)]);
    }

    #[test]
    fn nulls_first_overrides_the_default_placement() {
        let views = segment(&[(1, "A", 1.0), (2, "B", 2.0), (3, "C", 3.0)]);
        // needs2-style: the first window produces NULL.
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
        let frame = "OVER (ORDER BY ts ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)";
        let sql = format!("SELECT needs2(x) {frame} AS w FROM t ORDER BY w NULLS FIRST");
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(&sql).unwrap(),
            &registry,
        );
        assert_eq!(flatten(&output.unwrap(), 0), [None, Some(2.0), Some(3.0)]);
        let sql = format!("SELECT needs2(x) {frame} AS w FROM t ORDER BY w DESC NULLS LAST");
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(&sql).unwrap(),
            &registry,
        );
        assert_eq!(flatten(&output.unwrap(), 0), [Some(3.0), Some(2.0), None]);
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
    fn builtin_window_aggregates_match_hand_computation() {
        let views = segment(&[(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0), (4, "A", 4.0)]);
        let output = run(
            &views,
            "SELECT sum(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s, \
             avg(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a, \
             min(x) OVER (ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS lo, \
             max(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS hi, \
             count(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS n FROM t",
        )
        .unwrap();
        assert_eq!(
            flatten(&output, 0),
            [Some(1.0), Some(3.0), Some(5.0), Some(7.0)]
        );
        assert_eq!(
            flatten(&output, 1),
            [Some(1.0), Some(1.5), Some(2.0), Some(2.5)]
        );
        assert_eq!(
            flatten(&output, 2),
            [Some(1.0), Some(1.0), Some(1.0), Some(2.0)]
        );
        assert_eq!(
            flatten(&output, 3),
            [Some(1.0), Some(2.0), Some(3.0), Some(4.0)]
        );
        // count is an integer window (B5): SQL COUNT returns i64, not f64.
        let Column::Numeric(NumericData::I64(n)) = &output.batches[0].columns()[4] else {
            panic!("count window must be i64")
        };
        assert_eq!(n.values().as_slice(), [1, 2, 2, 2]);
        // An embedder registration of the same name wins over the
        // builtin.
        struct AlwaysNine;
        impl WindowAggregate for AlwaysNine {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, _: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok(Some(9.0))
            }
        }
        let mut registry = Registry::new();
        registry.register("sum", Arc::new(AlwaysNine));
        let output = execute(
            &schema(),
            &views,
            &crate::plan::plan(
                "SELECT sum(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
            )
            .unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(flatten(&output, 0), [Some(9.0); 4]);
    }

    #[test]
    fn unbounded_windows_span_partitions_and_segments() {
        let rows: Vec<(i64, &str, f64)> = (0..12)
            .map(|i| (i, ["A", "B"][(i % 2) as usize], i as f64))
            .collect();
        let sql = "SELECT sum(x) OVER (PARTITION BY sym ORDER BY ts \
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t";
        let reference = flatten(&run(&segment(&rows), sql).unwrap(), 0);
        // A's running sums over 0,2,4,..; B's over 1,3,5,..
        assert_eq!(reference[0], Some(0.0));
        assert_eq!(reference[1], Some(1.0));
        assert_eq!(reference[10], Some(0.0 + 2.0 + 4.0 + 6.0 + 8.0 + 10.0));
        for segment_rows in [3, 5] {
            assert_eq!(
                flatten(&run(&segmented(&rows, segment_rows), sql).unwrap(), 0),
                reference
            );
        }
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
