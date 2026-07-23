//! The M1 table: schema in, rows in, SQL out — the vertical slice that
//! proves compute lives inside the engine.
//!
//! A [`Table`] owns the whole pipeline: schema definition (numeric-or-key
//! and the declared `NOT NULL` ordering key, enforced at definition time),
//! one-row-at-a-time ingest through `storage-lite`, SQL through
//! `query-lite`, and the LAPACK-backed rolling regressions registered as
//! the window functions `regr_slope(y, x)` / `regr_intercept(y, x)`.
//! Results leave as record batches, or as an `ArrowArrayStream` via
//! [`Table::query_stream`].
//!
//! ## M1 shape: write, then read
//!
//! Appends go to the write buffer; the first query freezes it into the
//! single immutable in-memory segment and further appends are refused.
//! Real interleaving (multiple segments, flush policy) arrives with the
//! storage work at M2 — refusing now is honest scope, not a bug.
//!
//! ## Where the copies are (and aren't)
//!
//! Passthrough columns in a query result share the segment's buffers
//! (copy-on-write handles), and the C Data export hands those same
//! buffers out — asserted by pointer identity in this crate's tests.
//! Window slices feed the regression as plain sub-slices. The regression
//! itself performs the one bounded copy this design accepts: assembling
//! the `[1 | x]` design matrix LAPACK needs, O(rows) against an
//! O(rows·k²) solve — the trade recorded in deferred issue #4.

use arrow_lite::{ArrowArrayStream, ColumnType, RecordBatch, Schema};
use compute_lapack::{ColMajor, ComputeError, LapackBackend, NativeLapack, Op};
use query_lite::{execute, plan, QueryError, Registry, WindowAggregate};
use std::fmt;
use std::sync::Arc;
use storage_lite::{RowValue, Segment, StorageError, WriteBuffer};

/// Why a table operation failed.
#[derive(Debug)]
pub enum EngineError {
    /// Schema definition problems (bad ordering key, and — via storage —
    /// anything that violates numeric-or-key).
    Storage(StorageError),
    /// Query planning or execution problems.
    Query(QueryError),
    /// The query names a table this handle does not hold.
    WrongTable { expected: String, got: String },
    /// An append arrived after the first query froze the table (M1 is
    /// write-then-read; interleaving arrives at M2).
    AppendAfterQuery,
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
            EngineError::AppendAfterQuery => {
                write!(f, "append after first query: M1 tables are write-then-read")
            }
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

/// Write buffer until the first query; frozen segment after.
enum TableState {
    Writing(WriteBuffer),
    Frozen(Segment),
}

/// A single M1 table: ingest one row at a time, query with SQL.
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
/// let batch = table
///     .query(
///         "SELECT regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
///          ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS beta FROM trades",
///     )
///     .unwrap();
/// // Exact data ⇒ exact slope wherever the window has two points.
/// let arrow_lite::Column::Numeric(arrow_lite::NumericData::F64(beta)) = &batch.columns()[0]
/// else {
///     unreachable!()
/// };
/// assert!((beta.values()[39] - 3.0).abs() < 1e-12);
/// ```
pub struct Table {
    name: String,
    schema: Schema,
    state: TableState,
    registry: Registry,
}

impl Table {
    /// Defines a table: `schema` (numeric-or-key by construction — the
    /// column types are a closed enum) with `ordering_key` naming the
    /// `i64 NOT NULL` column ingest arrives sorted on.
    pub fn new(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
    ) -> Result<Table, EngineError> {
        let ordering_index = schema
            .fields()
            .iter()
            .position(|field| field.name() == ordering_key)
            .ok_or_else(|| EngineError::UnknownOrderingKey(ordering_key.to_owned()))?;
        let buffer = WriteBuffer::new(schema.clone(), ordering_index)?;
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
        Ok(Table {
            name: name.into(),
            schema,
            state: TableState::Writing(buffer),
            registry,
        })
    }

    /// The table's schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Appends one row (see [`RowValue`]); every cell is validated
    /// against the schema.
    pub fn append(&mut self, row: &[RowValue<'_>]) -> Result<(), EngineError> {
        match &mut self.state {
            TableState::Writing(buffer) => Ok(buffer.append(row)?),
            TableState::Frozen(_) => Err(EngineError::AppendAfterQuery),
        }
    }

    /// Freezes on first use; later calls reuse the segment.
    fn segment(&mut self) -> Result<&Segment, EngineError> {
        if let TableState::Writing(_) = self.state {
            let TableState::Writing(buffer) =
                std::mem::replace(&mut self.state, TableState::Frozen(placeholder_segment()))
            else {
                unreachable!()
            };
            self.state = TableState::Frozen(buffer.freeze()?);
        }
        match &self.state {
            TableState::Frozen(segment) => Ok(segment),
            TableState::Writing(_) => unreachable!(),
        }
    }

    /// Runs one SQL query, returning the result batch.
    pub fn query(&mut self, sql: &str) -> Result<RecordBatch, EngineError> {
        let plan = plan(sql)?;
        if plan.table != self.name {
            return Err(EngineError::WrongTable {
                expected: self.name.clone(),
                got: plan.table,
            });
        }
        let registry = self.registry.clone();
        let segment = self.segment()?;
        Ok(execute(segment, &plan, &registry)?)
    }

    /// Runs one SQL query and exports the result as an
    /// `ArrowArrayStream` — the same doorway `arrow-lite`'s oracle
    /// harness proved against arrow-rs and PyArrow.
    pub fn query_stream(&mut self, sql: &str) -> Result<ArrowArrayStream, EngineError> {
        let batch = self.query(sql)?;
        let schema = batch.schema().clone();
        Ok(arrow_lite::export_stream(schema, std::iter::once(batch)))
    }
}

/// An empty stand-in used only inside [`Table::segment`]'s state swap.
fn placeholder_segment() -> Segment {
    WriteBuffer::new(
        Schema::new(vec![arrow_lite::Field::new("_", ColumnType::I64, false)]),
        0,
    )
    .expect("static schema is valid")
    .freeze()
    .expect("empty freeze succeeds")
}

/// Which coefficient of the per-window fit `y ≈ intercept + slope · x`
/// an instance returns.
enum RegressionOutput {
    Slope,
    Intercept,
}

/// The M1 flagship: rolling least-squares of `y` on `x`, one solve per
/// window through `compute-lapack` (QR via `dgels`, provisional per open
/// decision #20).
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
        // happily returns garbage coefficients (the QR weakness recorded
        // in open decision #20).
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
    use arrow_lite::{Column, Field, NumericColumn, NumericData};

    fn m1_schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, false),
        ])
    }

    /// Two interleaved symbols with exactly linear (but different)
    /// relationships, so every ≥2-point window recovers them exactly.
    fn linear_table() -> Table {
        let mut table = Table::new("trades", m1_schema(), "ts").unwrap();
        for i in 0..30i64 {
            let x = i as f64;
            let (sym, y) = if i % 2 == 0 {
                ("A", 2.0 * x + 5.0)
            } else {
                ("B", -1.5 * x + 40.0)
            };
            table
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key(sym),
                    RowValue::F64(x),
                    RowValue::F64(y),
                ])
                .unwrap();
        }
        table
    }

    fn f64_column(batch: &RecordBatch, index: usize) -> &NumericColumn<f64> {
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            panic!("expected f64")
        };
        column
    }

    #[test]
    fn rolling_regression_recovers_exact_lines_per_symbol() {
        let mut table = linear_table();
        let batch = table
            .query(
                "SELECT sym, regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
                 ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS beta, \
                 regr_intercept(y, x) OVER (PARTITION BY sym ORDER BY ts \
                 ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS alpha FROM trades",
            )
            .unwrap();
        let beta = f64_column(&batch, 1);
        let alpha = f64_column(&batch, 2);
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
        let batch = table
            .query(
                "SELECT regr_slope(y, x) OVER (ORDER BY ts \
                 ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) FROM t",
            )
            .unwrap();
        let column = f64_column(&batch, 0);
        assert_eq!(column.null_count(), batch.num_rows());
    }

    #[test]
    fn passthrough_shares_buffers_through_the_whole_engine_path() {
        let mut table = linear_table();
        let first = table.query("SELECT x FROM trades").unwrap();
        let second = table.query("SELECT x AS renamed FROM trades").unwrap();
        // Both results and the stored segment hand out the same
        // allocation — the zero-copy claim at the engine boundary.
        assert_eq!(
            f64_column(&first, 0).values().as_ptr(),
            f64_column(&second, 0).values().as_ptr()
        );
    }

    #[test]
    fn stream_export_round_trips_through_the_c_interface() {
        let mut table = linear_table();
        let expected = table.query("SELECT ts, sym, x, y FROM trades").unwrap();
        let stream = table
            .query_stream("SELECT ts, sym, x, y FROM trades")
            .unwrap();
        // SAFETY: a live stream our own engine just exported.
        let reader = unsafe { arrow_lite::StreamReader::new(stream) }.unwrap();
        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches, vec![expected]);
    }

    #[test]
    fn lifecycle_errors_are_specific() {
        let mut table = linear_table();
        assert!(matches!(
            table.query("SELECT x FROM elsewhere"),
            Err(EngineError::WrongTable { .. })
        ));
        table.query("SELECT x FROM trades").unwrap();
        assert!(matches!(
            table.append(&[
                RowValue::I64(99),
                RowValue::Key("A"),
                RowValue::F64(0.0),
                RowValue::F64(0.0)
            ]),
            Err(EngineError::AppendAfterQuery)
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
