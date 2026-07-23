//! One table: schema in, rows in, SQL out — with compute inside the
//! engine.
//!
//! A [`Table`] owns the whole pipeline for its rows: schema definition
//! (numeric-or-key and the declared `NOT NULL` ordering key, enforced at
//! definition time), one-row-at-a-time ingest through `storage-lite`'s
//! multi-segment [`Store`], SQL through `query-lite`, and the
//! LAPACK-backed rolling regressions registered as the window functions
//! `regr_slope(y, x)` / `regr_intercept(y, x)`. Appends and queries
//! interleave freely: a query runs over a point-in-time snapshot of the
//! store, and appends after it never disturb the result. Results leave as
//! a [`QueryOutput`] — one batch per segment — or as an
//! `ArrowArrayStream` via [`Table::query_stream`].
//!
//! ## Where the copies are (and aren't)
//!
//! Passthrough columns in each result batch share that segment's buffers
//! (copy-on-write handles), and the C Data export hands those same
//! buffers out — asserted by pointer identity in this crate's tests.
//! Windows over a single segment feed the regression as plain sub-slices;
//! windows that span segments and partitioned windows run over a bounded
//! O(rows) gather, the same class of copy as the regression's `[1 | x]`
//! design-matrix gather (the trade recorded in deferred issue #4).

use arrow_lite::{ArrowArrayStream, Schema};
use compute_lapack::{ColMajor, ComputeError, LapackBackend, NativeLapack, Op};
use query_lite::{execute, plan, Plan, QueryError, QueryOutput, Registry, WindowAggregate};
use std::fmt;
use std::sync::Arc;
use storage_lite::{FsBackend, RowValue, StorageBackend, StorageError, Store};

/// Why a table or database operation failed.
#[derive(Debug)]
pub enum EngineError {
    /// Schema definition problems (bad ordering key, and — via storage —
    /// anything that violates numeric-or-key).
    Storage(StorageError),
    /// Query planning or execution problems.
    Query(QueryError),
    /// The query names a table this handle does not hold.
    WrongTable { expected: String, got: String },
    /// The query names a table the database does not hold.
    UnknownTable(String),
    /// A table with this name already exists in the database.
    DuplicateTable(String),
    /// The declared ordering key is not a column of the schema.
    UnknownOrderingKey(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Storage(error) => write!(f, "{error}"),
            EngineError::Query(error) => write!(f, "{error}"),
            EngineError::WrongTable { expected, got } => {
                write!(f, "query names table '{got}', this table is '{expected}'")
            }
            EngineError::UnknownTable(name) => write!(f, "unknown table '{name}'"),
            EngineError::DuplicateTable(name) => write!(f, "table '{name}' already exists"),
            EngineError::UnknownOrderingKey(name) => {
                write!(f, "ordering key '{name}' is not a column")
            }
        }
    }
}

impl std::error::Error for EngineError {}

impl From<StorageError> for EngineError {
    fn from(error: StorageError) -> Self {
        EngineError::Storage(error)
    }
}

impl From<QueryError> for EngineError {
    fn from(error: QueryError) -> Self {
        EngineError::Query(error)
    }
}

/// A single table: ingest one row at a time, query with SQL, freely
/// interleaved.
///
/// ```
/// use arrow_lite::{ColumnType, Field, Schema};
/// use engine::{RowValue, Table};
///
/// let schema = Schema::new(vec![
///     Field::new("ts", ColumnType::I64, false),
///     Field::new("sym", ColumnType::Key, false),
///     Field::new("x", ColumnType::F64, false),
///     Field::new("y", ColumnType::F64, false),
/// ]);
/// let mut table = Table::new("trades", schema, "ts").unwrap();
/// for i in 0..40 {
///     let x = i as f64;
///     table
///         .append(&[
///             RowValue::I64(i),
///             RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
///             RowValue::F64(x),
///             RowValue::F64(3.0 * x + 1.0), // exactly linear per sym
///         ])
///         .unwrap();
/// }
/// let output = table
///     .query(
///         "SELECT regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
///          ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS beta FROM trades",
///     )
///     .unwrap();
/// // Exact data ⇒ exact slope wherever the window has two points.
/// let batch = &output.batches[0];
/// let arrow_lite::Column::Numeric(arrow_lite::NumericData::F64(beta)) = &batch.columns()[0]
/// else {
///     unreachable!()
/// };
/// assert!((beta.values()[39] - 3.0).abs() < 1e-12);
/// // And the table is still open for appends — no write-then-read wall.
/// table
///     .append(&[
///         RowValue::I64(40),
///         RowValue::Key("A"),
///         RowValue::F64(40.0),
///         RowValue::F64(121.0),
///     ])
///     .unwrap();
/// ```
pub struct Table {
    name: String,
    store: Store,
    registry: Registry,
}

impl Table {
    /// Defines a table: `schema` (numeric-or-key by construction — the
    /// column types are a closed enum) with `ordering_key` naming the
    /// `i64 NOT NULL` column ingest arrives roughly sorted on.
    pub fn new(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
    ) -> Result<Table, EngineError> {
        Table::build(name, schema, ordering_key, None)
    }

    /// As [`Table::new`], with an explicit segment-row threshold — how
    /// many rows storage accumulates before freezing a segment. Tests and
    /// benchmarks use small thresholds to exercise many segments.
    pub fn with_segment_rows(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
        segment_rows: usize,
    ) -> Result<Table, EngineError> {
        Table::build(name, schema, ordering_key, Some(segment_rows))
    }

    /// A table stored durably in `dir` (created if absent): opens the
    /// existing table there — verifying the stored schema and every
    /// segment — or creates a fresh one. Durability follows storage's
    /// contract: flushed segments survive a crash, the write buffer does
    /// not; [`Table::flush`] is the boundary.
    pub fn persistent(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<Table, EngineError> {
        let index = ordering_index(&schema, ordering_key)?;
        let backend = fs_backend(dir)?;
        Ok(Table::from_store(
            name,
            Store::persistent(backend, schema, index)?,
        ))
    }

    /// As [`Table::persistent`], with an explicit segment-row threshold.
    pub fn persistent_with_segment_rows(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
        dir: impl AsRef<std::path::Path>,
        segment_rows: usize,
    ) -> Result<Table, EngineError> {
        let index = ordering_index(&schema, ordering_key)?;
        let backend = fs_backend(dir)?;
        Ok(Table::from_store(
            name,
            Store::persistent_with_segment_rows(backend, schema, index, segment_rows)?,
        ))
    }

    fn build(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
        segment_rows: Option<usize>,
    ) -> Result<Table, EngineError> {
        let ordering_index = ordering_index(&schema, ordering_key)?;
        let store = match segment_rows {
            None => Store::new(schema, ordering_index)?,
            Some(rows) => Store::with_segment_rows(schema, ordering_index, rows)?,
        };
        Ok(Table::from_store(name, store))
    }

    fn from_store(name: impl Into<String>, store: Store) -> Table {
        let mut registry = Registry::new();
        let backend = NativeLapack;
        registry.register(
            "regr_slope",
            Arc::new(RollingRegression {
                backend,
                output: RegressionOutput::Slope,
            }),
        );
        registry.register(
            "regr_intercept",
            Arc::new(RollingRegression {
                backend,
                output: RegressionOutput::Intercept,
            }),
        );
        Table {
            name: name.into(),
            store,
            registry,
        }
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The table's schema.
    pub fn schema(&self) -> &Schema {
        self.store.schema()
    }

    /// Appends one row (see [`RowValue`]); every cell is validated
    /// against the schema. Returns the row's internal monotonic row id
    /// (decision #1).
    pub fn append(&mut self, row: &[RowValue<'_>]) -> Result<u64, EngineError> {
        Ok(self.store.append(row)?)
    }

    /// Freezes the current write buffer into a segment now. Storage does
    /// this on its own as rows accumulate; explicit flushes exist for
    /// embedders that want segment boundaries at moments they choose.
    pub fn flush(&mut self) -> Result<(), EngineError> {
        Ok(self.store.flush()?)
    }

    /// Runs one SQL query over a point-in-time snapshot of the table.
    pub fn query(&self, sql: &str) -> Result<QueryOutput, EngineError> {
        let plan = plan(sql)?;
        if plan.table != self.name {
            return Err(EngineError::WrongTable {
                expected: self.name.clone(),
                got: plan.table,
            });
        }
        self.execute_plan(&plan)
    }

    /// Runs an already-planned query (the database handle plans once to
    /// route by table name, then calls this).
    pub(crate) fn execute_plan(&self, plan: &Plan) -> Result<QueryOutput, EngineError> {
        let segments = self.store.snapshot()?;
        Ok(execute(
            self.store.schema(),
            &segments,
            plan,
            &self.registry,
        )?)
    }

    /// Runs one SQL query and exports the result as an
    /// `ArrowArrayStream` — one batch per segment, through the same
    /// doorway `arrow-lite`'s oracle harness proved against arrow-rs and
    /// PyArrow.
    pub fn query_stream(&self, sql: &str) -> Result<ArrowArrayStream, EngineError> {
        let QueryOutput { schema, batches } = self.query(sql)?;
        Ok(arrow_lite::export_stream(schema, batches.into_iter()))
    }
}

/// Resolves the declared ordering key to its column index.
fn ordering_index(schema: &Schema, ordering_key: &str) -> Result<usize, EngineError> {
    schema
        .fields()
        .iter()
        .position(|field| field.name() == ordering_key)
        .ok_or_else(|| EngineError::UnknownOrderingKey(ordering_key.to_owned()))
}

/// The native storage backend: a directory of files.
fn fs_backend(dir: impl AsRef<std::path::Path>) -> Result<Arc<dyn StorageBackend>, EngineError> {
    let backend = FsBackend::new(dir.as_ref()).map_err(StorageError::from)?;
    Ok(Arc::new(backend))
}

/// Which coefficient of the per-window fit `y ≈ intercept + slope · x`
/// an instance returns.
enum RegressionOutput {
    Slope,
    Intercept,
}

/// Rolling least-squares of `y` on `x`, one solve per window through
/// `compute-lapack` (QR via `dgels` today; decision #20 ruled
/// QR-fast-path-plus-SVD-fallback, and the SVD side arrives with the M2
/// work on that op).
struct RollingRegression {
    backend: NativeLapack,
    output: RegressionOutput,
}

impl WindowAggregate for RollingRegression {
    fn arity(&self) -> usize {
        2 // regr_slope(y, x): dependent first, per SQL convention
    }

    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        // Capability negotiation, surfaced as a clean error — on a future
        // partial backend this is how "no LAPACK here yet" reads.
        if !self.backend.supports(Op::LeastSquares) {
            return Err("least-squares is unavailable on this compute backend".to_owned());
        }
        let (y, x) = (args[0], args[1]);
        let rows = y.len();
        if rows < 2 {
            return Ok(None); // a one-point regression is undefined: NULL
        }
        // Zero variance in x makes the regression undefined — SQL NULL,
        // exactly regr_slope's definition. Checked here because QR
        // without pivoting cannot be trusted to flag it: rounding leaves
        // the triangular factor almost-but-not-exactly singular and dgels
        // happily returns garbage coefficients (the QR weakness that
        // decided #20: an SVD fallback joins the op at M2).
        if x.iter().all(|&value| value == x[0]) {
            return Ok(None);
        }
        // The one bounded copy (issue #4): gather the [1 | x] design
        // matrix in the column-major layout LAPACK requires.
        let mut design = Vec::with_capacity(rows * 2);
        design.resize(rows, 1.0);
        design.extend_from_slice(x);
        match self
            .backend
            .least_squares(ColMajor::new(&design, rows, 2), y)
        {
            Ok(coefficients) => Ok(Some(match self.output {
                RegressionOutput::Slope => coefficients[1],
                RegressionOutput::Intercept => coefficients[0],
            })),
            // Rank-deficient window (constant x): the regression is
            // undefined there — SQL NULL, matching regr_slope semantics.
            Err(ComputeError::Lapack { .. }) => Ok(None),
            Err(error) => Err(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_lite::{Column, ColumnType, Field, NumericColumn, NumericData, RecordBatch};

    fn m1_schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, false),
        ])
    }

    fn linear_row(i: i64) -> [RowValue<'static>; 4] {
        let x = i as f64;
        let (sym, y) = if i % 2 == 0 {
            ("A", 2.0 * x + 5.0)
        } else {
            ("B", -1.5 * x + 40.0)
        };
        [
            RowValue::I64(i),
            RowValue::Key(sym),
            RowValue::F64(x),
            RowValue::F64(y),
        ]
    }

    /// Two interleaved symbols with exactly linear (but different)
    /// relationships, so every ≥2-point window recovers them exactly.
    fn linear_table(segment_rows: Option<usize>) -> Table {
        let mut table = match segment_rows {
            None => Table::new("trades", m1_schema(), "ts").unwrap(),
            Some(rows) => Table::with_segment_rows("trades", m1_schema(), "ts", rows).unwrap(),
        };
        for i in 0..30i64 {
            table.append(&linear_row(i)).unwrap();
        }
        table
    }

    fn f64_column(batch: &RecordBatch, index: usize) -> &NumericColumn<f64> {
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            panic!("expected f64")
        };
        column
    }

    /// Flattens one f64 output column across batches.
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

    const REGRESSION_SQL: &str = "SELECT sym, regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS beta, \
         regr_intercept(y, x) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS alpha FROM trades";

    #[test]
    fn rolling_regression_recovers_exact_lines_per_symbol() {
        let table = linear_table(None);
        let output = table.query(REGRESSION_SQL).unwrap();
        let batch = &output.batches[0];
        let beta = f64_column(batch, 1);
        let alpha = f64_column(batch, 2);
        let Column::Key(sym) = &batch.columns()[0] else {
            panic!("sym")
        };
        for row in 0..batch.num_rows() {
            // Each partition's first row has a one-point window: NULL.
            let first_of_partition = row < 2;
            assert_eq!(beta.is_valid(row), !first_of_partition, "row {row}");
            if beta.is_valid(row) {
                let (slope, intercept) = match sym.value_at(row).unwrap() {
                    "A" => (2.0, 5.0),
                    _ => (-1.5, 40.0),
                };
                assert!((beta.values()[row] - slope).abs() < 1e-10, "row {row}");
                assert!((alpha.values()[row] - intercept).abs() < 1e-10, "row {row}");
            }
        }
    }

    #[test]
    fn segmented_table_matches_single_segment_table() {
        // Engine-level golden: the same ingest through a many-segment
        // store computes exactly what the single-segment store computes.
        let reference = linear_table(None).query(REGRESSION_SQL).unwrap();
        for segment_rows in [1, 4, 7, 30] {
            let table = linear_table(Some(segment_rows));
            let output = table.query(REGRESSION_SQL).unwrap();
            assert_eq!(flatten(&output, 1), flatten(&reference, 1), "beta");
            assert_eq!(flatten(&output, 2), flatten(&reference, 2), "alpha");
        }
    }

    #[test]
    fn appends_and_queries_interleave() {
        let mut table = Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap();
        for i in 0..6i64 {
            table.append(&linear_row(i)).unwrap();
        }
        let before = table.query("SELECT x FROM trades").unwrap();
        assert_eq!(before.num_rows(), 6);
        // Appends after a query succeed, and the old result is untouched.
        for i in 6..10i64 {
            table.append(&linear_row(i)).unwrap();
        }
        assert_eq!(before.num_rows(), 6);
        let after = table.query("SELECT x FROM trades").unwrap();
        assert_eq!(after.num_rows(), 10);
        // Ingest interrupted by queries computes exactly what
        // uninterrupted ingest of the same rows computes.
        let mut uninterrupted = Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap();
        for i in 0..10i64 {
            uninterrupted.append(&linear_row(i)).unwrap();
        }
        assert_eq!(
            flatten(&table.query(REGRESSION_SQL).unwrap(), 1),
            flatten(&uninterrupted.query(REGRESSION_SQL).unwrap(), 1)
        );
    }

    #[test]
    fn passthrough_shares_buffers_through_the_whole_engine_path() {
        let table = linear_table(None);
        let first = table.query("SELECT x FROM trades").unwrap();
        let second = table.query("SELECT x AS renamed FROM trades").unwrap();
        // Both results and the stored segment hand out the same
        // allocation — the zero-copy claim at the engine boundary.
        assert_eq!(
            f64_column(&first.batches[0], 0).values().as_ptr(),
            f64_column(&second.batches[0], 0).values().as_ptr()
        );
    }

    #[test]
    fn constant_x_window_is_null_not_garbage() {
        let mut table = Table::new("t", m1_schema(), "ts").unwrap();
        for i in 0..5i64 {
            table
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key("A"),
                    RowValue::F64(7.0), // no variance in x
                    RowValue::F64(i as f64),
                ])
                .unwrap();
        }
        let output = table
            .query(
                "SELECT regr_slope(y, x) OVER (ORDER BY ts \
                 ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) FROM t",
            )
            .unwrap();
        let column = f64_column(&output.batches[0], 0);
        assert_eq!(column.null_count(), output.num_rows());
    }

    #[test]
    fn stream_export_round_trips_through_the_c_interface() {
        // Multiple segments ⇒ multiple batches through the C stream.
        let table = linear_table(Some(8));
        let expected = table.query("SELECT ts, sym, x, y FROM trades").unwrap();
        assert!(expected.batches.len() > 1);
        let stream = table
            .query_stream("SELECT ts, sym, x, y FROM trades")
            .unwrap();
        // SAFETY: a live stream our own engine just exported.
        let reader = unsafe { arrow_lite::StreamReader::new(stream) }.unwrap();
        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches, expected.batches);
    }

    #[test]
    fn empty_table_queries_cleanly() {
        let table = Table::new("t", m1_schema(), "ts").unwrap();
        let output = table.query("SELECT ts, x FROM t").unwrap();
        assert_eq!(output.num_rows(), 0);
        assert_eq!(output.batches.len(), 0);
        assert_eq!(output.schema.fields()[1].name(), "x");
    }

    #[test]
    fn persistent_table_reopens_with_identical_results() {
        let dir =
            std::env::temp_dir().join(format!("tallydb-engine-persist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let reference;
        {
            let mut table =
                Table::persistent_with_segment_rows("trades", m1_schema(), "ts", &dir, 8).unwrap();
            for i in 0..30i64 {
                table.append(&linear_row(i)).unwrap();
            }
            table.flush().unwrap();
            reference = flatten(&table.query(REGRESSION_SQL).unwrap(), 1);
        }
        // A fresh process-equivalent: open the same directory, ask the
        // same question, get bit-identical regression output.
        let reopened =
            Table::persistent_with_segment_rows("trades", m1_schema(), "ts", &dir, 8).unwrap();
        assert_eq!(
            flatten(&reopened.query(REGRESSION_SQL).unwrap(), 1),
            reference
        );
        // And the reopened table keeps ingesting where it left off.
        let mut reopened = reopened;
        assert_eq!(reopened.append(&linear_row(30)).unwrap(), 30);
        // Schema disagreement at open is refused loudly.
        let wrong = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("z", ColumnType::F64, false),
        ]);
        assert!(matches!(
            Table::persistent("trades", wrong, "ts", &dir),
            Err(EngineError::Storage(StorageError::SchemaMismatch { .. }))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn lifecycle_errors_are_specific() {
        let table = linear_table(None);
        assert!(matches!(
            table.query("SELECT x FROM elsewhere"),
            Err(EngineError::WrongTable { .. })
        ));
        assert!(matches!(
            Table::new("t", m1_schema(), "nope"),
            Err(EngineError::UnknownOrderingKey(_))
        ));
        // Ordering-key rules come from storage: f64 ordering key refused.
        assert!(matches!(
            Table::new("t", m1_schema(), "x"),
            Err(EngineError::Storage(StorageError::BadOrderingKey { .. }))
        ));
    }
}
