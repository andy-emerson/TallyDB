//! One table: schema in, rows in, SQL out — with compute inside the
//! engine.
//!
//! A [`Table`] owns the whole pipeline for its rows: schema definition
//! (numeric-or-key and the declared `NOT NULL` ordering key, enforced at
//! definition time), one-row-at-a-time ingest through `storage-lite`'s
//! multi-segment [`Store`], SQL through `query-lite`, the rolling
//! regressions and pair statistics registered as the window functions
//! `regr_slope(y, x)` / `regr_intercept(y, x)`, and application-
//! registered Lua window kernels via `Table::register_lua_window`
//! (behind the `lua` feature)
//! (the `script` module). Appends and queries
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
//! Windows over a single segment feed the aggregates as plain sub-slices;
//! windows that span segments and partitioned windows run over an O(rows)
//! gather — table-proportional, not bounded by a constant (~56 B/row for
//! a two-argument window) — the copy recorded in deferred issue #4
//! (peak-memory accounting in #56).

use arrow_lite::{ArrowArrayStream, Column, ColumnType, Field, NumericData, Schema};
#[cfg(feature = "lua")]
use compute_lua::LogSink;
use query_lite::{
    evaluate_predicate, execute, parse_statement, plan, recompute_frames, ColumnFunction,
    DeletePlan, Number, Plan, QueryError, QueryOutput, Registry, SetValue, Statement, UpdatePlan,
    WindowAggregate,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use storage_lite::{
    FsBackend, RowValue, SegmentView, StorageBackend, StorageError, Store, StoreOptions,
    StoreReader,
};

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
    /// The schema declares a column whose name the engine reserves.
    ReservedColumn(String),
    /// Registering a script-backed function failed (bad kernel syntax,
    /// unusable parameter name, unsupported output type).
    Script(String),
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
            EngineError::ReservedColumn(name) => write!(
                f,
                "column '{name}' is reserved — it is the ingest-sequence \
                 pseudocolumn every table already has"
            ),
            EngineError::Script(message) => write!(f, "script: {message}"),
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
    /// SQL name → window implementation, shared with every reader
    /// handle. The lock is held only to clone or swap the inner `Arc`,
    /// so a snapshot pins the function set of its moment and a later
    /// registration never mutates what a reader already holds.
    registry: Arc<Mutex<Arc<Registry>>>,
    /// Where Lua kernels' `log(...)` goes — `None` (the default) keeps
    /// `log` a no-op, per compute-lua's off-by-default contract. Applies
    /// to kernels registered *after* it is set.
    #[cfg(feature = "lua")]
    lua_log_sink: Option<Arc<dyn LogSink + Sync>>,
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
        let index = validated_ordering_index(&schema, ordering_key)?;
        let backend = fs_backend(dir)?;
        Ok(Table::from_store(
            name,
            Store::persistent(backend, schema, index)?,
        ))
    }

    /// As [`Table::persistent`], with explicit [`StoreOptions`]: the
    /// freeze threshold (rows, or bytes — the memory bound an embedder
    /// actually budgets, #44) and the durability level (#43's ruling:
    /// default `Group(100ms)`, measured ~free; `Full` for a zero loss
    /// window at ~670× per-append cost; `Off` for the flush-boundary
    /// contract and replayable upstreams).
    pub fn persistent_with(
        name: impl Into<String>,
        schema: Schema,
        ordering_key: &str,
        dir: impl AsRef<std::path::Path>,
        options: StoreOptions,
    ) -> Result<Table, EngineError> {
        let index = validated_ordering_index(&schema, ordering_key)?;
        let backend = fs_backend(dir)?;
        Ok(Table::from_store(
            name,
            Store::persistent_with(backend, schema, index, options)?,
        ))
    }

    /// Opens an existing persistent table, taking schema and ordering
    /// key from its manifest — the shell's doorway to a table it did
    /// not create in this process.
    pub fn open(
        name: impl Into<String>,
        dir: impl AsRef<std::path::Path>,
        options: StoreOptions,
    ) -> Result<Table, EngineError> {
        let backend = fs_backend(dir)?;
        Ok(Table::from_store(
            name,
            Store::open_existing(backend, options)?,
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
        let index = validated_ordering_index(&schema, ordering_key)?;
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
        let ordering_index = validated_ordering_index(&schema, ordering_key)?;
        let store = match segment_rows {
            None => Store::new(schema, ordering_index)?,
            Some(rows) => Store::with_segment_rows(schema, ordering_index, rows)?,
        };
        Ok(Table::from_store(name, store))
    }

    fn from_store(name: impl Into<String>, store: Store) -> Table {
        let mut registry = Registry::new();
        registry.register(
            "regr_slope",
            Arc::new(RollingRegression {
                output: RegressionOutput::Slope,
            }),
        );
        registry.register(
            "regr_intercept",
            Arc::new(RollingRegression {
                output: RegressionOutput::Intercept,
            }),
        );
        registry.register(
            "covar_pop",
            Arc::new(PairStatistic {
                kind: PairKind::CovarPop,
            }),
        );
        registry.register(
            "corr",
            Arc::new(PairStatistic {
                kind: PairKind::Corr,
            }),
        );
        registry.register(
            "eigen_max",
            Arc::new(PairStatistic {
                kind: PairKind::EigenMax,
            }),
        );
        Table {
            name: name.into(),
            store,
            registry: Arc::new(Mutex::new(Arc::new(registry))),
            #[cfg(feature = "lua")]
            lua_log_sink: None,
        }
    }

    /// The registry as of now — a cheap `Arc` clone under a brief lock.
    fn current_registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry.lock().expect("registry lock poisoned"))
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The table's schema.
    pub fn schema(&self) -> &Schema {
        self.store.schema()
    }

    /// The ordering-key column's name — the schema alone cannot say
    /// which column ingest is ordered on, so anything reconstructing a
    /// full definition (the console's `.schema`) needs this too.
    pub fn ordering_key(&self) -> &str {
        self.store.schema().fields()[self.store.ordering_key()].name()
    }

    #[cfg(feature = "lua")]
    /// Installs the destination for Lua kernels' `log(...)` output.
    /// Until an embedder calls this, `log` is a no-op (compute-lua's
    /// off-by-default contract); afterwards, every *newly registered*
    /// kernel routes through the sink — install before registering.
    pub fn set_lua_log_sink(&mut self, sink: Arc<dyn LogSink + Sync>) {
        self.lua_log_sink = Some(sink);
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
        let segments = match plan.as_of {
            // Knowledge-time travel: the same executor over an as-of
            // snapshot — the knowledge mask is just a live mask.
            Some(cut) => self.store.knowledge_snapshot()?.as_of(cut),
            None => self.store.snapshot()?,
        };
        Ok(execute(
            self.store.schema(),
            &segments,
            plan,
            &self.current_registry(),
        )?)
    }

    /// The table's ingest-sequence watermark: the sequence the next
    /// appended row will receive. Every knowledge coordinate the table
    /// holds is `<` it, so **`ASOF next_sequence() - 1` reads the
    /// latest state** whatever the table's mutation history — and keeps
    /// reading it, because the coordinate has been spent and no later
    /// arrival can join it. `ASOF next_sequence()` reads the same state
    /// today but addresses a coordinate that is still unspent, so its
    /// answer moves with the next append. Prefer the spent one; record
    /// it before a correction to keep a queryable before/after boundary.
    ///
    /// Zero rows means zero spent coordinates: there is nothing below
    /// the watermark to cut at, and every `AS OF` reads an empty table
    /// anyway.
    pub fn next_sequence(&self) -> u64 {
        self.store.next_sequence()
    }

    /// Runs a join plan with `self` as the fact table (the database
    /// handle resolves the dimension and calls this).
    pub(crate) fn execute_join_plan(
        &self,
        plan: &Plan,
        dimension: &Table,
    ) -> Result<QueryOutput, EngineError> {
        let fact_views = self.store.snapshot()?;
        let dimension_views = dimension.store.snapshot()?;
        Ok(query_lite::execute_join(
            self.store.schema(),
            &fact_views,
            dimension.store.schema(),
            &dimension_views,
            plan,
            &self.current_registry(),
        )?)
    }

    /// A cheap, cloneable handle for reader threads: it mints
    /// point-in-time [`TableSnapshot`]s while this table's single
    /// writer (whoever holds `&mut Table`) appends, mutates, or
    /// compacts. This is the single-writer/concurrent-readers cut made
    /// visible in the types: exactly one `&mut Table` can exist, and
    /// readers can neither block it for longer than a snapshot's
    /// microseconds nor observe a torn state (#51).
    ///
    /// ```
    /// # use arrow_lite::{ColumnType, Field, Schema};
    /// # use engine::{RowValue, Table};
    /// let schema = Schema::new(vec![
    ///     Field::new("ts", ColumnType::I64, false),
    ///     Field::new("x", ColumnType::F64, false),
    /// ]);
    /// let mut table = Table::new("t", schema, "ts").unwrap();
    /// for i in 0..10 {
    ///     table.append(&[RowValue::I64(i), RowValue::F64(i as f64)]).unwrap();
    /// }
    /// let reader = table.reader();
    /// let worker = std::thread::spawn(move || {
    ///     let snapshot = reader.snapshot().unwrap();
    ///     snapshot.query("SELECT x FROM t").unwrap().batches.len()
    /// });
    /// // The writer keeps writing while the reader thread queries.
    /// table.append(&[RowValue::I64(10), RowValue::F64(10.0)]).unwrap();
    /// assert!(worker.join().unwrap() >= 1);
    /// ```
    pub fn reader(&self) -> TableReader {
        TableReader {
            name: self.name.clone(),
            schema: self.store.schema().clone(),
            store: self.store.reader(),
            registry: Arc::clone(&self.registry),
        }
    }

    /// A point-in-time snapshot, directly from the writer's thread —
    /// equivalent to `table.reader().snapshot()`.
    pub fn snapshot(&self) -> Result<TableSnapshot, EngineError> {
        self.reader().snapshot()
    }

    /// Runs one SQL query and exports the result as an
    /// `ArrowArrayStream` — one batch per segment, through the same
    /// doorway `arrow-lite`'s oracle harness proved against arrow-rs and
    /// PyArrow.
    pub fn query_stream(&self, sql: &str) -> Result<ArrowArrayStream, EngineError> {
        let QueryOutput { schema, batches } = self.query(sql)?;
        Ok(arrow_lite::export_stream(schema, batches.into_iter()))
    }

    /// Applies an `INSERT ... VALUES` plan: rows validate and append
    /// through the ordinary append path (WAL included), literals
    /// coercing exactly or loudly — an integer literal fits an `f64`
    /// column exactly or is refused; a float literal never silently
    /// truncates into a `BIGINT` column; strings go to key columns
    /// only. Returns the rows inserted.
    pub(crate) fn insert(&mut self, plan: &query_lite::InsertPlan) -> Result<u64, EngineError> {
        let schema = self.store.schema().clone();
        let order: Vec<usize> = match &plan.columns {
            None => (0..schema.fields().len()).collect(),
            Some(names) => {
                let mut order = Vec::with_capacity(names.len());
                for name in names {
                    let position = schema
                        .fields()
                        .iter()
                        .position(|field| field.name() == name)
                        .ok_or_else(|| {
                            EngineError::Query(QueryError::UnknownColumn(name.clone()))
                        })?;
                    order.push(position);
                }
                order
            }
        };
        let mut inserted = 0u64;
        for row in &plan.rows {
            if row.len() != order.len() {
                return Err(EngineError::Query(QueryError::Unsupported(format!(
                    "INSERT row has {} values for {} columns",
                    row.len(),
                    order.len()
                ))));
            }
            let mut cells: Vec<RowValue<'_>> = vec![RowValue::Null; schema.fields().len()];
            for (value, &position) in row.iter().zip(&order) {
                let field = &schema.fields()[position];
                cells[position] = match value {
                    query_lite::InsertValue::Null => RowValue::Null,
                    query_lite::InsertValue::String(text) => RowValue::Key(text),
                    query_lite::InsertValue::Number(number) => match field.column_type() {
                        ColumnType::F64 => match number {
                            Number::Float(value) => RowValue::F64(*value),
                            Number::Int(value) => {
                                let as_f64 = *value as f64;
                                // Compared in i128: the naive `as_f64 as
                                // i64` saturates, so i64::MAX (rounded
                                // up to 2^63 by the cast) saturates
                                // right back and passes a check it
                                // should fail.
                                if as_f64 as i128 != i128::from(*value) {
                                    return Err(EngineError::Query(QueryError::TypeError(
                                        format!(
                                            "integer {value} does not fit an f64 exactly \
                                             (column '{}')",
                                            field.name()
                                        ),
                                    )));
                                }
                                RowValue::F64(as_f64)
                            }
                        },
                        ColumnType::I64 => match number {
                            Number::Int(value) => RowValue::I64(*value),
                            Number::Float(value) => {
                                return Err(EngineError::Query(QueryError::TypeError(format!(
                                    "float {value} into BIGINT column '{}' would truncate",
                                    field.name()
                                ))))
                            }
                        },
                        ColumnType::Key => {
                            return Err(EngineError::Query(QueryError::TypeError(format!(
                                "number into SYMBOL column '{}' (labels take strings)",
                                field.name()
                            ))))
                        }
                    },
                };
            }
            self.append(&cells)?;
            inserted += 1;
        }
        Ok(inserted)
    }

    /// Runs one SQL mutation (`UPDATE` or `DELETE`), returning the rows
    /// affected. Both are the design's one mutation mechanism: `DELETE`
    /// tombstones the matched rows; `UPDATE` tombstones them and
    /// reappends corrected copies, which get fresh row ids at the tail
    /// of the ingest sequence — so an update whose rows carry old
    /// ordering-key values leaves the table unordered until
    /// [`Table::compact`], and window queries in between refuse loudly
    /// rather than mis-compute. Not the fast path, by design — nor the
    /// lean one: a full-table `UPDATE` peaks at several times the columnar
    /// footprint (a row-major copy, a string per key cell, then reappend;
    /// accounted in #56).
    pub fn mutate(&mut self, sql: &str) -> Result<u64, EngineError> {
        match parse_statement(sql)? {
            Statement::Select(_) => Err(EngineError::Query(QueryError::Unsupported(
                "SELECT runs through query, not mutate".to_owned(),
            ))),
            Statement::CreateTable(_) => Err(EngineError::Query(QueryError::Unsupported(
                "CREATE TABLE runs through the database handle".to_owned(),
            ))),
            Statement::Insert(insert) => {
                self.check_table(&insert.table)?;
                self.insert(&insert)
            }
            Statement::Delete(delete) => self.delete(delete),
            Statement::Update(update) => self.update(update),
        }
    }

    /// Compacts the table's storage: tombstones resolve, order is
    /// restored, row ids become contiguous again (see storage's
    /// compaction contract).
    pub fn compact(&mut self) -> Result<(), EngineError> {
        Ok(self.store.compact()?)
    }

    #[cfg(feature = "lua")]
    /// Registers a Lua kernel as a SQL window function on this table —
    /// the Lua-in-SQL window slot (#41). `parameters` names the frame's
    /// column arguments, positionally, as the globals the kernel reads
    /// them through (zero-copy views, oldest row first); the kernel
    /// returns one number per frame, or `NULL`. `output` declares the
    /// result column's type (F2): `F64` or `I64` — never inferred from
    /// what a call returns. Everything that can fail confusingly at
    /// query time fails loudly here instead: kernel syntax, unusable
    /// parameter names, a key-typed output.
    ///
    /// Kernels can call the curated native ops over the same views —
    /// `dot(x, y)` (compute-linalg), `regr_slope(y, x)` / `regr_intercept(y, x)`,
    /// `covar_pop(y, x)` / `corr(y, x)` / `eigen_max(y, x)` — the very
    /// implementations the SQL windows run, sharing buffers with no
    /// copy; each returns a number, or `NULL` where undefined.
    ///
    /// A registration under a built-in name shadows the built-in; a
    /// second registration under the same name replaces the first.
    ///
    /// ```
    /// use arrow_lite::{ColumnType, Field, Schema};
    /// use engine::{RowValue, Table};
    ///
    /// let schema = Schema::new(vec![
    ///     Field::new("ts", ColumnType::I64, false),
    ///     Field::new("x", ColumnType::F64, false),
    /// ]);
    /// let mut table = Table::new("trades", schema, "ts").unwrap();
    /// for i in 0..40 {
    ///     table
    ///         .append(&[RowValue::I64(i), RowValue::F64(i as f64)])
    ///         .unwrap();
    /// }
    /// // Mean absolute deviation — a loop the built-ins don't cover.
    /// table
    ///     .register_lua_window(
    ///         "mad",
    ///         &["x"],
    ///         "local n = #x\n\
    ///          local mean = 0.0\n\
    ///          for i = 1, n do mean = mean + x[i] end\n\
    ///          mean = mean / n\n\
    ///          local mad = 0.0\n\
    ///          for i = 1, n do mad = mad + math.abs(x[i] - mean) end\n\
    ///          return mad / n",
    ///         ColumnType::F64,
    ///     )
    ///     .unwrap();
    /// let output = table
    ///     .query(
    ///         "SELECT mad(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING \
    ///          AND CURRENT ROW) AS m FROM trades",
    ///     )
    ///     .unwrap();
    /// // A ramp's full 4-row window deviates by exactly 1.0.
    /// let arrow_lite::Column::Numeric(arrow_lite::NumericData::F64(m)) =
    ///     &output.batches[0].columns()[0]
    /// else {
    ///     panic!("expected f64")
    /// };
    /// assert_eq!(m.values().as_slice()[0], 0.0); // one-row window
    /// assert!(m.values().as_slice()[4..].iter().all(|&v| v == 1.0));
    /// ```
    pub fn register_lua_window(
        &mut self,
        name: &str,
        parameters: &[&str],
        chunk: &str,
        output: ColumnType,
    ) -> Result<(), EngineError> {
        if !is_identifier(name) {
            // Checked before compiling the chunk, so the name error is
            // the first thing a bad registration hears.
            return Err(EngineError::Script(format!(
                "function name '{name}' is not callable from SQL"
            )));
        }
        // The kernel's vocabulary is the registry as of this moment —
        // every native and every previously registered kernel, by name.
        let ops = self.current_registry();
        let window = crate::script::LuaWindow::new(
            parameters,
            chunk,
            output,
            self.lua_log_sink.clone(),
            &ops,
        )
        .map_err(EngineError::Script)?;
        self.register_kernel(name, Arc::new(window))
    }

    /// Registers a native window kernel — **the primary extension
    /// path** (DESIGN.md, *the extension model*): implement
    /// [`WindowAggregate`], register it, call it from SQL at native
    /// speed. No interpreter is involved; the kernel is engine code,
    /// compiled into the embedding application. The built-in
    /// statistics (`regr_slope`, `covar_pop`, …) are ordinary
    /// registrations of this same kind.
    ///
    /// Registering a name again **replaces** the previous
    /// implementation — deliberately: that is the promotion path. A
    /// kernel prototyped as a Lua script keeps its SQL name when an
    /// embedder re-registers a native under it; every query and
    /// script continues to work, now compiled. Snapshots taken before
    /// a registration keep the function set they saw
    /// (see [`Table::reader`]).
    ///
    /// ```
    /// use arrow_lite::{ColumnType, Field, Schema};
    /// use engine::{RowValue, Table, WindowAggregate};
    ///
    /// // The whole extension surface: one trait, ~20 lines.
    /// struct WeightedMean;
    ///
    /// impl WindowAggregate for WeightedMean {
    ///     fn arity(&self) -> usize {
    ///         2 // wmean(x, w)
    ///     }
    ///     fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
    ///         let (x, w) = (args[0], args[1]);
    ///         let sw: f64 = w.iter().sum();
    ///         Ok((sw != 0.0)
    ///             .then(|| x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / sw))
    ///     }
    /// }
    ///
    /// let schema = Schema::new(vec![
    ///     Field::new("ts", ColumnType::I64, false),
    ///     Field::new("x", ColumnType::F64, false),
    ///     Field::new("w", ColumnType::F64, false),
    /// ]);
    /// let mut table = Table::new("t", schema, "ts").unwrap();
    /// for i in 0..4i64 {
    ///     table
    ///         .append(&[
    ///             RowValue::I64(i),
    ///             RowValue::F64(i as f64),
    ///             RowValue::F64(1.0),
    ///         ])
    ///         .unwrap();
    /// }
    /// table.register_window("wmean", WeightedMean).unwrap();
    /// let output = table
    ///     .query(
    ///         "SELECT wmean(x, w) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING \
    ///          AND CURRENT ROW) AS m FROM t",
    ///     )
    ///     .unwrap();
    /// let arrow_lite::Column::Numeric(arrow_lite::NumericData::F64(m)) =
    ///     &output.batches[0].columns()[0]
    /// else {
    ///     panic!("expected f64")
    /// };
    /// // Unit weights: each two-row window's plain mean.
    /// assert_eq!(m.values().as_slice(), &[0.0, 0.5, 1.5, 2.5]);
    /// ```
    pub fn register_window(
        &mut self,
        name: &str,
        kernel: impl WindowAggregate + 'static,
    ) -> Result<(), EngineError> {
        self.register_kernel(name, Arc::new(kernel))
    }

    /// Registers a native **column function** — SQL's per-row (scalar)
    /// extension slot, evaluated a whole column at a time: implement
    /// [`ColumnFunction`], register it, and `SELECT f(x, y) FROM t`
    /// calls it once per view over dense argument columns. Same naming
    /// and replacement rules as [`Table::register_window`]; the output
    /// is a nullable `f64` column.
    pub fn register_column_function(
        &mut self,
        name: &str,
        function: impl ColumnFunction + 'static,
    ) -> Result<(), EngineError> {
        self.register_column_kernel(name, Arc::new(function))
    }

    #[cfg(feature = "lua")]
    /// Registers a Lua **column** kernel as a SQL scalar function —
    /// the vectorized whole-column shape (#53). The parameters bind as
    /// whole-column views (NULL is the `NULL` sentinel), the script
    /// fills the preallocated `out` column (`out[i] = ...`; slots never
    /// written come back NULL), and the interpreter is entered **once
    /// per view**, never per row — which is what makes a scripted
    /// per-row function viable in bulk. The compose-don't-loop idiom
    /// still applies inside: prefer the registered vocabulary over
    /// element loops where one fits.
    ///
    /// ```text
    /// .luascalar spread(hi, lo) for i = 1, #hi do out[i] = hi[i] - lo[i] end
    /// SELECT spread(hi, lo) FROM quotes
    /// ```
    pub fn register_lua_scalar(
        &mut self,
        name: &str,
        parameters: &[&str],
        chunk: &str,
    ) -> Result<(), EngineError> {
        if !is_identifier(name) {
            return Err(EngineError::Script(format!(
                "function name '{name}' is not callable from SQL"
            )));
        }
        let ops = self.current_registry();
        let column =
            crate::script::LuaColumn::new(parameters, chunk, self.lua_log_sink.clone(), &ops)
                .map_err(EngineError::Script)?;
        self.register_column_kernel(name, Arc::new(column))
    }

    /// [`Table::register_kernel`]'s column-function twin.
    fn register_column_kernel(
        &mut self,
        name: &str,
        function: Arc<dyn ColumnFunction>,
    ) -> Result<(), EngineError> {
        if !is_identifier(name) {
            return Err(EngineError::Script(format!(
                "function name '{name}' is not callable from SQL"
            )));
        }
        let mut guard = self.registry.lock().expect("registry lock poisoned");
        let mut next = (**guard).clone();
        next.register_column(name, function);
        *guard = Arc::new(next);
        Ok(())
    }

    /// The one registration mechanism both extension paths share: an
    /// identifier check, then a copy-and-swap of the registry `Arc` so
    /// existing snapshots keep the function set they were minted with.
    fn register_kernel(
        &mut self,
        name: &str,
        kernel: Arc<dyn WindowAggregate>,
    ) -> Result<(), EngineError> {
        if !is_identifier(name) {
            return Err(EngineError::Script(format!(
                "function name '{name}' is not callable from SQL"
            )));
        }
        let mut guard = self.registry.lock().expect("registry lock poisoned");
        let mut next = (**guard).clone();
        next.register(name, kernel);
        *guard = Arc::new(next);
        Ok(())
    }

    fn check_table(&self, named: &str) -> Result<(), EngineError> {
        if named != self.name {
            return Err(EngineError::WrongTable {
                expected: self.name.clone(),
                got: named.to_owned(),
            });
        }
        Ok(())
    }

    /// Live row ids matching `predicate` (all live rows when `None`).
    fn matched_row_ids(
        &self,
        views: &[SegmentView],
        predicate: Option<&query_lite::Predicate>,
    ) -> Result<Vec<u64>, EngineError> {
        let schema = self.store.schema();
        let mut ids = Vec::new();
        for view in views {
            let matches = predicate
                .map(|predicate| evaluate_predicate(predicate, schema, view))
                .transpose()?;
            let base = view.segment.base_row_id();
            for row in 0..view.segment.batch().num_rows() {
                let hit = view.is_live(row) && matches.as_ref().is_none_or(|mask| mask.get(row));
                if hit {
                    ids.push(base + row as u64);
                }
            }
        }
        Ok(ids)
    }

    fn delete(&mut self, delete: DeletePlan) -> Result<u64, EngineError> {
        self.check_table(&delete.table)?;
        let views = self.store.snapshot()?;
        let ids = self.matched_row_ids(&views, delete.predicate.as_ref())?;
        Ok(self.store.tombstone(&ids)?)
    }

    fn update(&mut self, update: UpdatePlan) -> Result<u64, EngineError> {
        self.check_table(&update.table)?;
        let schema = self.store.schema().clone();
        // Validate every assignment against the schema before touching
        // anything, so a bad statement mutates nothing.
        let mut assigned: Vec<(usize, OwnedValue)> = Vec::with_capacity(update.assignments.len());
        for assignment in &update.assignments {
            let index = schema
                .fields()
                .iter()
                .position(|field| field.name() == assignment.column)
                .ok_or_else(|| {
                    EngineError::Query(QueryError::UnknownColumn(assignment.column.clone()))
                })?;
            let field = &schema.fields()[index];
            let value = match (&assignment.value, field.column_type()) {
                (SetValue::Number(Number::Float(value)), ColumnType::F64) => {
                    OwnedValue::F64(*value)
                }
                (SetValue::Number(Number::Int(value)), ColumnType::F64) => {
                    OwnedValue::F64(*value as f64)
                }
                (SetValue::Number(Number::Int(value)), ColumnType::I64) => OwnedValue::I64(*value),
                (SetValue::String(value), ColumnType::Key) => OwnedValue::Key(value.clone()),
                (SetValue::Null, _) if field.nullable() => OwnedValue::Null,
                (SetValue::Null, _) => {
                    return Err(EngineError::Query(QueryError::TypeError(format!(
                        "column '{}' is NOT NULL",
                        assignment.column
                    ))))
                }
                _ => {
                    return Err(EngineError::Query(QueryError::TypeError(format!(
                        "SET value does not fit column '{}' ({:?})",
                        assignment.column,
                        field.column_type()
                    ))))
                }
            };
            assigned.push((index, value));
        }
        // Build the corrected copies of every matched live row.
        let views = self.store.snapshot()?;
        let mut matched_ids: Vec<u64> = Vec::new();
        let mut corrected: Vec<Vec<OwnedValue>> = Vec::new();
        for view in &views {
            let matches = update
                .predicate
                .as_ref()
                .map(|predicate| evaluate_predicate(predicate, &schema, view))
                .transpose()?;
            let batch = view.segment.batch();
            let base = view.segment.base_row_id();
            for row in 0..batch.num_rows() {
                let hit = view.is_live(row) && matches.as_ref().is_none_or(|mask| mask.get(row));
                if !hit {
                    continue;
                }
                matched_ids.push(base + row as u64);
                let mut cells: Vec<OwnedValue> = batch
                    .columns()
                    .iter()
                    .map(|column| OwnedValue::from_cell(column, row))
                    .collect();
                for (index, value) in &assigned {
                    cells[*index] = value.clone();
                }
                corrected.push(cells);
            }
        }
        // One knowledge event (issue #73): the replacements and the
        // tombstones land together at a single ingest sequence —
        // old-or-new under `AS OF` and across a crash alike, never a
        // cut that sees both versions or a recovery that keeps half.
        let rows: Vec<Vec<RowValue<'_>>> = corrected
            .iter()
            .map(|cells| cells.iter().map(OwnedValue::as_row_value).collect())
            .collect();
        let affected = self.store.supersede(&rows, &matched_ids)?;
        Ok(affected)
    }
}

/// An owned cell — what `UPDATE` builds its corrected rows from before
/// handing them back to storage as borrowed [`RowValue`]s.
#[derive(Clone)]
enum OwnedValue {
    F64(f64),
    I64(i64),
    Key(String),
    Null,
}

impl OwnedValue {
    fn from_cell(column: &Column, row: usize) -> OwnedValue {
        match column {
            Column::Numeric(NumericData::F64(numeric)) => {
                if numeric.is_valid(row) {
                    OwnedValue::F64(numeric.values().as_slice()[row])
                } else {
                    OwnedValue::Null
                }
            }
            Column::Numeric(NumericData::I64(numeric)) => {
                if numeric.is_valid(row) {
                    OwnedValue::I64(numeric.values().as_slice()[row])
                } else {
                    OwnedValue::Null
                }
            }
            Column::Key(keys) => keys
                .value_at(row)
                .map_or(OwnedValue::Null, |value| OwnedValue::Key(value.to_owned())),
        }
    }

    fn as_row_value(&self) -> RowValue<'_> {
        match self {
            OwnedValue::F64(value) => RowValue::F64(*value),
            OwnedValue::I64(value) => RowValue::I64(*value),
            OwnedValue::Key(value) => RowValue::Key(value),
            OwnedValue::Null => RowValue::Null,
        }
    }
}

/// Whether `name` is a plain identifier (ASCII letter or underscore,
/// then letters, digits, underscores) — required of SQL function names
/// so queries can actually call them, and of Lua parameter names so
/// kernels can actually reference them.
pub(crate) fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Resolves the declared ordering key to its column index.
/// Checks a schema at definition time and locates its ordering key.
/// Beyond the ordering key itself, the one rule is the reserved name:
/// `_seq` is the ingest-sequence pseudocolumn every table already has
/// (#75), and a declared column of that name would shadow it. `CREATE
/// TABLE` refuses it in the planner; this is the same refusal for
/// schemas built through the Rust API.
fn validated_ordering_index(schema: &Schema, ordering_key: &str) -> Result<usize, EngineError> {
    if schema
        .fields()
        .iter()
        .any(|field| field.name() == query_lite::SEQUENCE_COLUMN)
    {
        return Err(EngineError::ReservedColumn(
            query_lite::SEQUENCE_COLUMN.to_owned(),
        ));
    }
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

/// Maps a lowered `CREATE TABLE` (the #49 DDL grammar: `BIGINT`,
/// `DOUBLE`, `SYMBOL`, one `ORDERING KEY` column) onto the
/// engine's schema types. Returns the schema and the ordering-key
/// column name. Shared by every SQL surface — shell, and later the
/// server and workbench — so the mapping cannot fork.
pub fn schema_from_create(
    plan: &query_lite::CreateTablePlan,
) -> Result<(Schema, String), EngineError> {
    let mut fields = Vec::with_capacity(plan.columns.len());
    let mut ordering = None;
    for column in &plan.columns {
        let column_type = match column.type_name.as_str() {
            "BIGINT" => ColumnType::I64,
            "DOUBLE" => ColumnType::F64,
            "SYMBOL" => ColumnType::Key,
            other => {
                return Err(EngineError::Query(QueryError::Unsupported(format!(
                    "column type '{other}'"
                ))))
            }
        };
        if column.ordering_key {
            if column_type != ColumnType::I64 {
                return Err(EngineError::Query(QueryError::Unsupported(
                    "the ORDERING KEY column must be BIGINT (a timestamp, sequence, \
                     or offset — the monotonic-on-ingest key)"
                        .to_owned(),
                )));
            }
            ordering = Some(column.name.clone());
        }
        fields.push(Field::new(&column.name, column_type, !column.not_null));
    }
    let ordering = ordering.expect("the planner requires exactly one ORDERING KEY");
    Ok((Schema::new(fields), ordering))
}

/// The DDL name of a column type — [`schema_from_create`]'s inverse,
/// kept beside it so a renderer (the console's `.schema`) cannot fork
/// its own mapping.
pub fn type_name(column_type: ColumnType) -> &'static str {
    match column_type {
        ColumnType::I64 => "BIGINT",
        ColumnType::F64 => "DOUBLE",
        ColumnType::Key => "SYMBOL",
    }
}

/// A cheap, cloneable, `Send` handle a reader thread holds to mint
/// point-in-time [`TableSnapshot`]s while the table's single writer
/// proceeds. Created by [`Table::reader`]; see there for the
/// concurrency contract. Each [`TableReader::snapshot`] takes the
/// table's brief state lock (bounded by one write-buffer copy — the
/// freeze threshold caps it) and returns fully detached views.
#[derive(Clone)]
pub struct TableReader {
    name: String,
    schema: Schema,
    store: StoreReader,
    registry: Arc<Mutex<Arc<Registry>>>,
}

impl TableReader {
    /// A point-in-time snapshot: the rows and the registered functions
    /// exactly as of this call. Appends, mutations, compactions, and
    /// registrations after it never affect the returned snapshot.
    pub fn snapshot(&self) -> Result<TableSnapshot, EngineError> {
        let knowledge = self.store.knowledge_snapshot()?;
        Ok(TableSnapshot {
            name: self.name.clone(),
            schema: self.schema.clone(),
            knowledge,
            registry: Arc::clone(&self.registry.lock().expect("registry lock poisoned")),
        })
    }
}

/// An immutable point-in-time view of one table: query it as often as
/// desired, from any thread, entirely independent of the writer. Old
/// segments a compaction replaced stay alive for as long as a snapshot
/// references them (`Arc`-backed) — read-copy-update, no coordination.
/// `Send + Sync`: share it or move it freely; Lua-backed window
/// functions serialize internally on their interpreter's mutex.
///
/// Scope: single-table `SELECT`s. Joins resolve dimension tables
/// through a [`crate::Database`], and no cross-table snapshot
/// consistency is promised (by design — see DESIGN.md); mutation is
/// the writer's alone.
pub struct TableSnapshot {
    name: String,
    schema: Schema,
    /// The latest-knowledge views plus what an `AS OF` query needs —
    /// history and pending kill stamps — captured at the same instant.
    knowledge: storage_lite::KnowledgeSnapshot,
    registry: Arc<Registry>,
}

impl TableSnapshot {
    /// Runs one SQL `SELECT` over this frozen view. `ASOF n` works
    /// here too: the snapshot carries the knowledge state of its
    /// moment, so time travel is relative to what was known then.
    pub fn query(&self, sql: &str) -> Result<QueryOutput, EngineError> {
        let plan = plan(sql)?;
        if plan.table != self.name {
            return Err(EngineError::WrongTable {
                expected: self.name.clone(),
                got: plan.table,
            });
        }
        let views;
        let segments = match plan.as_of {
            Some(cut) => {
                views = self.knowledge.as_of(cut);
                &views
            }
            None => &self.knowledge.views,
        };
        Ok(execute(&self.schema, segments, &plan, &self.registry)?)
    }

    /// As [`TableSnapshot::query`], exported as an `ArrowArrayStream`.
    pub fn query_stream(&self, sql: &str) -> Result<ArrowArrayStream, EngineError> {
        let QueryOutput { schema, batches } = self.query(sql)?;
        Ok(arrow_lite::export_stream(schema, batches.into_iter()))
    }

    /// The snapshot's schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

/// Which coefficient of the per-window fit `y ≈ intercept + slope · x`
/// an instance returns.
pub(crate) enum RegressionOutput {
    Slope,
    Intercept,
}

/// Rolling least-squares of `y` on `x`, solved in closed form.
///
/// A two-parameter fit has an exact solution — `slope = Sxy / Sxx` over
/// the centered sums — so it needs no matrix factorization. It is
/// computed by the **corrected two-pass** algorithm (Chan–Golub–LeVeque):
/// the naive form assumes centering leaves `Σ(x − x̄)` exactly zero,
/// which floating point does not honor, and the residual offset walks
/// into the intercept. Measured against the SVD answer
/// (`measure_closed_form`, release, container hardware, 2026-07-27),
/// worst predicted-y drift over the data:
///
/// | design | QR (`dgels`) | corrected | naive |
/// |---|---|---|---|
/// | 64-row window, x offset 1e9 | 1.07e-14 | 2.84e-14 | 8.31e-7 |
/// | 64-row window, x offset 1e12 | 1.42e-14 | 2.49e-14 | 1.01e-3 |
/// | near-degenerate, spread 1e-10 | 2.84e-14 | 1.99e-13 | 6.75e-7 |
///
/// The corrected form tracks QR within a small constant factor — both at
/// the float noise floor against `|y|` of order 10–200 — while the naive
/// form is a real regression at timestamp-scale offsets (bug #45's
/// regime). Worst relative slope error: 1.6e-15 at a 1e12 offset,
/// 7.6e-10 on the pathological near-degenerate designs.
///
/// Why not LAPACK: a general solver's per-call overhead dwarfs a
/// 64 × 2 problem's arithmetic — measured at roughly 2.3µs of the 2.5µs
/// per window. Decision #20 (QR fast path, SVD fallback) still governs
/// the `least_squares` op itself; it no longer governs this window,
/// because this window no longer solves a general system. See DESIGN.md,
/// *Curated compute: what the engine calls, and why*.
pub(crate) struct RollingRegression {
    pub(crate) output: RegressionOutput,
}

impl WindowAggregate for RollingRegression {
    fn arity(&self) -> usize {
        2 // regr_slope(y, x): dependent first, per SQL convention
    }

    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        let (y, x) = (args[0], args[1]);
        let rows = y.len();
        if rows < 2 {
            return Ok(None); // a one-point regression is undefined: NULL
        }
        let count = rows as f64;
        let mean_x = x.iter().sum::<f64>() / count;
        let mean_y = y.iter().sum::<f64>() / count;
        // Second pass about the means. `sum_dx` and `sum_dy` are zero in
        // exact arithmetic and merely small in floating point; carrying
        // them is what makes this the corrected form.
        let (mut sum_dx, mut sum_dy, mut sxy, mut sxx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&xi, &yi) in x.iter().zip(y) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            sum_dx += dx;
            sum_dy += dy;
            sxy += dx * dy;
            sxx += dx * dx;
        }
        let sxx = sxx - sum_dx * sum_dx / count;
        let sxy = sxy - sum_dx * sum_dy / count;
        // Zero variance in x — or negative, which rounding can produce
        // from the correction — leaves the regression undefined: SQL
        // NULL, exactly `regr_slope`'s definition. NaN is tested for
        // explicitly because every comparison against it is false, so it
        // would otherwise slip through as "not ≤ 0".
        if sxx <= 0.0 || sxx.is_nan() {
            return Ok(None);
        }
        let slope = sxy / sxx;
        // The fit is `a + slope·(x − x̄)` with `a` correcting for the
        // leftover offset; the reported intercept is its value at x = 0.
        let centered_intercept = mean_y - slope * (sum_dx / count);
        Ok(Some(match self.output {
            RegressionOutput::Slope => slope,
            RegressionOutput::Intercept => centered_intercept - slope * mean_x,
        }))
    }

    fn evaluate_frames(
        &self,
        columns: &[&[f64]],
        preceding: Option<usize>,
    ) -> Result<Vec<Option<f64>>, String> {
        let Some(preceding) = preceding else {
            // Unbounded frames only grow — no slide to make incremental.
            // Recompute per frame, exactly as before this override.
            return recompute_frames(self, columns, None);
        };
        let (y, x) = (columns[0], columns[1]);
        let mut results = Vec::with_capacity(y.len());
        shifted_sweep(y, x, preceding + 1, |lo, i, moments| {
            results.push(match moments {
                Some(moments) => self.value_from_shifted(moments),
                // A frame containing NaN/±Inf runs the reference
                // arithmetic directly, so the two paths agree exactly
                // where running sums cannot be trusted.
                None => self
                    .evaluate(&[&y[lo..=i], &x[lo..=i]])
                    .expect("the closed-form regression does not error"),
            });
        });
        Ok(results)
    }
}

impl RollingRegression {
    /// The regression's value from shifted moments — mirroring [`Self::evaluate`]'s
    /// semantics exactly: NULL under two rows or non-positive `Sxx` (NaN
    /// checked explicitly), `slope = Sxy / Sxx` over corrected sums, the
    /// intercept extrapolated to `x = 0` from the window means.
    fn value_from_shifted(&self, moments: &ShiftedMoments) -> Option<f64> {
        if moments.rows() < 2 {
            return None;
        }
        let (_, var_x, covar) = moments.population();
        if var_x <= 0.0 || var_x.is_nan() {
            return None;
        }
        let slope = covar / var_x;
        Some(match self.output {
            RegressionOutput::Slope => slope,
            RegressionOutput::Intercept => moments.mean_y() - slope * moments.mean_x(),
        })
    }
}

/// Which pair statistic an instance computes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PairKind {
    /// Population covariance of `(y, x)` — 0 for a single point,
    /// matching `covar_pop`.
    CovarPop,
    /// Pearson correlation; `NULL` when either variance is zero.
    Corr,
    /// The largest eigenvalue of the window's 2 × 2 population
    /// covariance matrix — the first principal component's variance,
    /// solved in closed form. `NULL` under two rows.
    EigenMax,
}

/// Two-column window statistics over `(y, x)`, sharing one accumulation
/// of the population moments.
pub(crate) struct PairStatistic {
    pub(crate) kind: PairKind,
}

impl WindowAggregate for PairStatistic {
    fn arity(&self) -> usize {
        2 // (y, x), same convention as regr_slope
    }

    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        let (y, x) = (args[0], args[1]);
        let n = y.len();
        if n == 0 {
            return Ok(None);
        }
        let count = n as f64;
        let (mean_y, mean_x) = (y.iter().sum::<f64>() / count, x.iter().sum::<f64>() / count);
        // The corrected two-pass (Chan–Golub–LeVeque), same as the
        // rolling regression: `sum_dy`/`sum_dx` are zero in exact
        // arithmetic but only small in floating point, and dropping the
        // correction costs ~4.9e-8 relative error at a 1e12 offset where
        // carrying it holds the noise floor (~1e-14) — measured against
        // the compensated reference in `window_truth`, and guarded by
        // `window_numerics_guard` below.
        let (mut sum_dy, mut sum_dx) = (0.0f64, 0.0f64);
        let (mut var_y, mut var_x, mut covar) = (0.0f64, 0.0f64, 0.0f64);
        for (&yi, &xi) in y.iter().zip(x) {
            let (dy, dx) = (yi - mean_y, xi - mean_x);
            sum_dy += dy;
            sum_dx += dx;
            var_y += dy * dy;
            var_x += dx * dx;
            covar += dy * dx;
        }
        var_y = (var_y - sum_dy * sum_dy / count) / count;
        var_x = (var_x - sum_dx * sum_dx / count) / count;
        covar = (covar - sum_dy * sum_dx / count) / count;
        Ok(self.value_from_moments(n, var_y, var_x, covar))
    }

    fn evaluate_frames(
        &self,
        columns: &[&[f64]],
        preceding: Option<usize>,
    ) -> Result<Vec<Option<f64>>, String> {
        let Some(preceding) = preceding else {
            // Unbounded frames only grow — no slide to make incremental.
            // Recompute per frame, exactly as before this override.
            return recompute_frames(self, columns, None);
        };
        let (y, x) = (columns[0], columns[1]);
        let mut results = Vec::with_capacity(y.len());
        shifted_sweep(y, x, preceding + 1, |lo, i, moments| {
            results.push(match moments {
                Some(moments) => {
                    let (var_y, var_x, covar) = moments.population();
                    self.value_from_moments(moments.rows(), var_y, var_x, covar)
                }
                // Non-finite frame: the reference arithmetic, exactly
                // as the recompute path would run it.
                None => self
                    .evaluate(&[&y[lo..=i], &x[lo..=i]])
                    .expect("the pair statistics do not error"),
            });
        });
        Ok(results)
    }
}

impl PairStatistic {
    /// The statistic's value from finished population moments — one
    /// finalization shared by the per-window and incremental paths, so
    /// the NULL semantics cannot diverge between them.
    fn value_from_moments(&self, rows: usize, var_y: f64, var_x: f64, covar: f64) -> Option<f64> {
        match self.kind {
            PairKind::CovarPop => Some(covar),
            PairKind::Corr => {
                if var_y <= 0.0 || var_x <= 0.0 {
                    return None; // undefined, per corr's definition
                }
                Some(covar / (var_y * var_x).sqrt())
            }
            PairKind::EigenMax => {
                if rows < 2 {
                    return None;
                }
                // The largest eigenvalue of a symmetric 2 × 2 in closed
                // form: λ_max = t + r for half-trace t = (var_y + var_x)/2
                // and radius r = √(((var_y − var_x)/2)² + covar²). Both
                // terms are non-negative (variances are), so the sum
                // carries no cancellation — this is the well-conditioned
                // half of the quadratic (λ_min, the differenced one, is
                // not computed here). A general eigensolver on a 2 × 2 is
                // dominated by its own call overhead; see DESIGN.md, the
                // curated-op cost record.
                let half_trace = (var_y + var_x) / 2.0;
                let half_gap = (var_y - var_x) / 2.0;
                let radius = half_gap.hypot(covar);
                Some(half_trace + radius)
            }
        }
    }
}

/// Running moments about a shift taken from the data — the
/// shifted-incremental window algorithm (3b-C) behind the
/// `evaluate_frames` overrides above. Values enter and leave as
/// deviations from `(ky, kx)`, so the accumulated sums stay at the
/// window's own scale even when the data sits at a 1e12 offset; the
/// E[d²] − E[d]² form is safe here for exactly that reason (about the
/// *raw* values it is bug #45's catastrophic form — see
/// `measure_incremental_windows`, variant B, rejected permanently).
#[derive(Default, Clone, Copy)]
struct ShiftedMoments {
    n: f64,
    sy: f64,
    sx: f64,
    syy: f64,
    sxx: f64,
    sxy: f64,
    ky: f64,
    kx: f64,
}

impl ShiftedMoments {
    fn add(&mut self, yi: f64, xi: f64) {
        let (dy, dx) = (yi - self.ky, xi - self.kx);
        self.n += 1.0;
        self.sy += dy;
        self.sx += dx;
        self.syy += dy * dy;
        self.sxx += dx * dx;
        self.sxy += dy * dx;
    }

    fn remove(&mut self, yi: f64, xi: f64) {
        let (dy, dx) = (yi - self.ky, xi - self.kx);
        self.n -= 1.0;
        self.sy -= dy;
        self.sx -= dx;
        self.syy -= dy * dy;
        self.sxx -= dx * dx;
        self.sxy -= dy * dx;
    }

    /// Population `(var_y, var_x, covar)` about the window means.
    fn population(&self) -> (f64, f64, f64) {
        let (my, mx) = (self.sy / self.n, self.sx / self.n);
        (
            self.syy / self.n - my * my,
            self.sxx / self.n - mx * mx,
            self.sxy / self.n - my * mx,
        )
    }

    fn mean_y(&self) -> f64 {
        self.ky + self.sy / self.n
    }

    fn mean_x(&self) -> f64 {
        self.kx + self.sx / self.n
    }

    fn rows(&self) -> usize {
        self.n as usize
    }
}

/// Sweeps one contiguous run with trailing frames of up to `w` rows,
/// calling `emit` once per position with the frame's bounds and — for
/// frames of finite values — that frame's moments. O(run): each step
/// slides by one `add` and one `remove`, and the accumulator is rebuilt
/// about a fresh shift every `w` steps so rounding cannot accumulate
/// across the column. Measured (`measure_3b`, 2026-07-27): ~7× the
/// per-window recompute at 20k rows / window 64, worst relative error
/// 5e-15–1.1e-14 against the compensated reference — held to 1e-12 in
/// CI by `window_numerics_guard`, which runs this exact path.
///
/// **Non-finite values break the sliding identity**: `NaN − NaN = NaN`,
/// so once a NaN (or ±Inf, whose differences produce NaN) enters the
/// running sums they stay poisoned even after the row leaves the frame
/// — the incremental path would return NaN for windows whose true
/// frames are clean. So the sweep counts the frame's non-finite rows:
/// while any is present it emits `None` (the caller runs the per-frame
/// reference arithmetic — bit-identical to the recompute path), and the
/// first clean frame afterwards rebuilds the accumulator from scratch.
fn shifted_sweep(
    y: &[f64],
    x: &[f64],
    w: usize,
    mut emit: impl FnMut(usize, usize, Option<&ShiftedMoments>),
) {
    debug_assert!(w >= 1 && y.len() == x.len());
    let finite = |j: usize| y[j].is_finite() && x[j].is_finite();
    let mut moments = ShiftedMoments::default();
    let mut dirty = 0usize; // non-finite rows in the current frame
    let mut since_rebuild = usize::MAX; // force a build on the first row
    for i in 0..y.len() {
        let lo = (i + 1).saturating_sub(w);
        if !finite(i) {
            dirty += 1;
        }
        if i >= w && !finite(i - w) {
            dirty -= 1;
        }
        if dirty > 0 {
            since_rebuild = usize::MAX; // rebuild when the frame comes clean
            emit(lo, i, None);
            continue;
        }
        if since_rebuild >= w {
            moments = ShiftedMoments {
                ky: y[i],
                kx: x[i],
                ..Default::default()
            };
            for j in lo..=i {
                moments.add(y[j], x[j]);
            }
            since_rebuild = 0;
        } else {
            moments.add(y[i], x[i]);
            if i >= w {
                moments.remove(y[i - w], x[i - w]);
            }
            since_rebuild += 1;
        }
        emit(lo, i, Some(&moments));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_lite::{Column, ColumnType, Field, NumericColumn, NumericData, RecordBatch};

    pub(super) fn m1_schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, false),
        ])
    }

    #[test]
    fn empty_results_survive_the_materializing_stages() {
        // ORDER BY, DISTINCT, and HAVING all gather rows into one
        // batch; each must survive having zero batches to gather.
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 4).unwrap();
        let empty = [
            "SELECT ts FROM t ORDER BY ts",
            "SELECT ts, sym FROM t ORDER BY ts DESC NULLS LAST LIMIT 3",
            "SELECT DISTINCT sym FROM t",
        ];
        for sql in empty {
            assert_eq!(table.query(sql).unwrap().num_rows(), 0, "{sql} (no rows)");
        }
        for i in 0..6 {
            table.append(&linear_row(i)).unwrap();
        }
        // Rows exist, but the predicate (and pruning) removes them all.
        for sql in [
            "SELECT ts FROM t WHERE ts > 100 ORDER BY ts",
            "SELECT DISTINCT sym FROM t WHERE ts > 100",
            "SELECT sym, COUNT(x) AS n FROM t GROUP BY sym HAVING COUNT(x) > 100",
        ] {
            assert_eq!(table.query(sql).unwrap().num_rows(), 0, "{sql} (filtered)");
        }
    }

    pub(super) fn linear_row(i: i64) -> [RowValue<'static>; 4] {
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

    pub(super) fn f64_column(batch: &RecordBatch, index: usize) -> &NumericColumn<f64> {
        let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
            panic!("expected f64")
        };
        column
    }

    /// Flattens one f64 output column across batches.
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

    /// `_seq` (#75) reads back the coordinate `AS OF` addresses — the
    /// point of the pseudocolumn is that a session can learn a cut from
    /// SQL alone, with no Rust-side `next_sequence()` call.
    #[test]
    fn the_sequence_pseudocolumn_addresses_the_knowledge_axis() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, false),
        ]);
        let mut table = Table::with_segment_rows("t", schema, "ts", 2).unwrap();
        for i in 0..4i64 {
            table
                .append(&[RowValue::I64(i), RowValue::F64(i as f64)])
                .unwrap();
        }
        let seqs = |table: &Table, sql: &str| -> Vec<i64> {
            let output = table.query(sql).unwrap();
            output
                .batches
                .iter()
                .flat_map(|batch| {
                    let arrow_lite::Column::Numeric(arrow_lite::NumericData::I64(column)) =
                        &batch.columns()[0]
                    else {
                        panic!("_seq is i64")
                    };
                    column.values().as_slice().to_vec()
                })
                .collect()
        };
        assert_eq!(seqs(&table, "SELECT _seq FROM t"), [0, 1, 2, 3]);
        // A correction: the replacement is born at coordinate 4, and
        // `_seq` reports *birth*, not position — so the corrected row
        // carries 4 while its neighbours keep their original
        // coordinates, and the table is now diverged (row id != seq).
        table.mutate("UPDATE t SET x = 100.0 WHERE ts = 1").unwrap();
        let coordinates = seqs(&table, "SELECT _seq, ts FROM t ORDER BY ts");
        assert_eq!(coordinates, [0, 4, 2, 3]);
        // The readback is a usable cut: the largest coordinate is the
        // latest state, and one below it is the state before the
        // correction — no Rust-side call anywhere in the loop.
        let latest = seqs(&table, "SELECT _seq FROM t ORDER BY _seq DESC LIMIT 1")[0];
        assert_eq!(latest, 4);
        assert_eq!(
            flatten(
                &table
                    .query(&format!("SELECT ts, x FROM t ASOF {latest} ORDER BY ts"))
                    .unwrap(),
                1
            ),
            [Some(0.0), Some(100.0), Some(2.0), Some(3.0)]
        );
        assert_eq!(
            flatten(
                &table
                    .query(&format!(
                        "SELECT ts, x FROM t ASOF {} ORDER BY ts",
                        latest - 1
                    ))
                    .unwrap(),
                1
            ),
            [Some(0.0), Some(1.0), Some(2.0), Some(3.0)]
        );
        // Compaction reassigns row ids; birth coordinates do not move.
        table.compact().unwrap();
        assert_eq!(
            seqs(&table, "SELECT _seq, ts FROM t ORDER BY ts"),
            coordinates
        );
    }

    /// `ASOF next_sequence()` is the latest state in every mutation
    /// shape — and, since a `DELETE` consumes its coordinate (ruled
    /// 2026-07-29), so is `ASOF next_sequence() - 1`. The second form
    /// is the one worth having: it names a coordinate that has actually
    /// been spent, so the cut keeps its meaning as ingest continues,
    /// while the watermark is a moving address. Before the ruling this
    /// test existed to record that `- 1` was *wrong* after a delete.
    #[test]
    fn as_of_at_the_watermark_is_the_latest_state_after_any_mutation() {
        let build = || {
            let schema = Schema::new(vec![
                Field::new("ts", ColumnType::I64, false),
                Field::new("x", ColumnType::F64, false),
            ]);
            let mut table = Table::new("t", schema, "ts").unwrap();
            for i in 0..5i64 {
                table
                    .append(&[RowValue::I64(i), RowValue::F64(i as f64)])
                    .unwrap();
            }
            table
        };
        let cases: Vec<(&str, Vec<&str>)> = vec![
            ("appends only", vec![]),
            ("update", vec!["UPDATE t SET x = 9.0 WHERE ts = 1"]),
            ("delete", vec!["DELETE FROM t WHERE ts = 2"]),
            (
                "delete then update",
                vec![
                    "DELETE FROM t WHERE ts = 2",
                    "UPDATE t SET x = 9.0 WHERE ts = 1",
                ],
            ),
            (
                "update then delete",
                vec![
                    "UPDATE t SET x = 9.0 WHERE ts = 1",
                    "DELETE FROM t WHERE ts = 2",
                ],
            ),
        ];
        for (label, mutations) in cases {
            let mut table = build();
            for sql in &mutations {
                table.mutate(sql).unwrap();
            }
            let latest = table.query("SELECT ts FROM t").unwrap().num_rows();
            let watermark = table.next_sequence();
            let at_watermark = table
                .query(&format!("SELECT ts FROM t ASOF {watermark}"))
                .unwrap()
                .num_rows();
            assert_eq!(
                at_watermark, latest,
                "{label}: ASOF next_sequence() must read the latest state"
            );
            // Every mutation now consumes a coordinate, so the last
            // spent one is the latest state too — the stable idiom.
            let at_last_spent = table
                .query(&format!("SELECT ts FROM t ASOF {}", watermark - 1))
                .unwrap()
                .num_rows();
            assert_eq!(
                at_last_spent, latest,
                "{label}: ASOF next_sequence() - 1 must read the latest state"
            );
            // And it stays that answer: a later arrival is born above
            // the cut, which is exactly what the watermark cannot
            // promise (it addresses the arrival too).
            table
                .append(&[RowValue::I64(99), RowValue::F64(99.0)])
                .unwrap();
            let after_arrival = table
                .query(&format!("SELECT ts FROM t ASOF {}", watermark - 1))
                .unwrap()
                .num_rows();
            assert_eq!(
                after_arrival, latest,
                "{label}: a spent coordinate's answer must not drift"
            );
        }
    }

    #[test]
    fn as_of_reads_the_table_as_it_was_known() {
        // The corrections model end to end: correct a row, then read
        // both the corrected present and the uncorrected past — before
        // and after compaction moves the superseded version to history.
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, false),
        ]);
        let mut table = Table::new("t", schema, "ts").unwrap();
        for i in 0..4i64 {
            table
                .append(&[RowValue::I64(i), RowValue::F64(i as f64)])
                .unwrap();
        }
        // The correction: one knowledge event (issue #73) — the
        // replacement is born and the original killed at the same
        // coordinate, 4, and the mutation consumes exactly one
        // sequence.
        table.mutate("UPDATE t SET x = 100.0 WHERE ts = 1").unwrap();
        assert_eq!(table.next_sequence(), 5);
        let latest = "SELECT ts, x FROM t ORDER BY ts";
        let past = "SELECT ts, x FROM t ASOF 3 ORDER BY ts";
        let corrected = [Some(0.0), Some(100.0), Some(2.0), Some(3.0)];
        let original = [Some(0.0), Some(1.0), Some(2.0), Some(3.0)];
        assert_eq!(flatten(&table.query(latest).unwrap(), 1), corrected);
        assert_eq!(flatten(&table.query(past).unwrap(), 1), original);
        // The SQL:2011 spelling reads identically.
        assert_eq!(
            flatten(
                &table
                    .query("SELECT ts, x FROM t FOR SYSTEM_TIME AS OF 3 ORDER BY ts")
                    .unwrap(),
                1
            ),
            original
        );
        // ASOF 0: only the first row was known.
        assert_eq!(table.query("SELECT x FROM t ASOF 0").unwrap().num_rows(), 1);
        // Old-or-new on the knowledge axis: the cut at the mutation's
        // coordinate is already the corrected state, and no cut
        // anywhere sees both versions at once.
        assert_eq!(
            flatten(
                &table
                    .query("SELECT ts, x FROM t ASOF 4 ORDER BY ts")
                    .unwrap(),
                1
            ),
            corrected
        );
        for cut in 0..=5 {
            let rows = table
                .query(&format!("SELECT x FROM t ASOF {cut}"))
                .unwrap()
                .num_rows();
            assert!(rows <= 4, "cut {cut} saw {rows} rows");
        }
        // Compaction retains the superseded version in history — the
        // past answer does not move, and neither does the present.
        table.compact().unwrap();
        assert_eq!(flatten(&table.query(past).unwrap(), 1), original);
        assert_eq!(flatten(&table.query(latest).unwrap(), 1), corrected);
        // A detached snapshot carries its knowledge state with it.
        let snapshot = table.snapshot().unwrap();
        assert_eq!(flatten(&snapshot.query(past).unwrap(), 1), original);
        assert_eq!(flatten(&snapshot.query(latest).unwrap(), 1), corrected);
        // Aggregation over the past runs through the same executor.
        assert_eq!(
            flatten(&table.query("SELECT SUM(x) AS s FROM t ASOF 3").unwrap(), 0),
            [Some(6.0)]
        );
        assert_eq!(
            flatten(&table.query("SELECT SUM(x) AS s FROM t").unwrap(), 0),
            [Some(105.0)]
        );
    }

    const REGRESSION_SQL: &str = "SELECT sym, regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS beta, \
         regr_intercept(y, x) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS alpha FROM trades";

    /// Bug #45: a regressor with a large offset relative to its in-window
    /// spread (a timestamp-scale x) made the raw [1 | x] design matrix
    /// catastrophically ill-conditioned — the fitted slope was garbage at
    /// offsets ≥ 1e9 while the CI oracle, solving the same raw matrix,
    /// agreed with the garbage. The reference here is the centered
    /// closed form, deliberately a different computational path from the
    /// engine's centered QR solve.
    #[test]
    fn rolling_regression_survives_timestamp_scale_x() {
        let mut table = Table::new("trades", m1_schema(), "ts").unwrap();
        let n = 20usize;
        let offset = 1e9f64;
        let slope = 2e-6f64;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let x = offset + i as f64;
            // Deterministic perturbation so the fit is not exact — the
            // regime where the uncentered solve loses the slope entirely.
            let y = 3.0 + slope * i as f64 + f64::from((i as u32 * 37) % 17) * 1e-5;
            xs.push(x);
            ys.push(y);
            table
                .append(&[
                    RowValue::I64(i as i64),
                    RowValue::Key("A"),
                    RowValue::F64(x),
                    RowValue::F64(y),
                ])
                .unwrap();
        }
        // Centered closed-form reference over the full window.
        let count = n as f64;
        let (mean_x, mean_y) = (
            xs.iter().sum::<f64>() / count,
            ys.iter().sum::<f64>() / count,
        );
        let mut ss_xx = 0.0f64;
        let mut ss_xy = 0.0f64;
        for (&x, &y) in xs.iter().zip(&ys) {
            ss_xx += (x - mean_x) * (x - mean_x);
            ss_xy += (x - mean_x) * (y - mean_y);
        }
        let slope_ref = ss_xy / ss_xx;
        let intercept_ref = mean_y - slope_ref * mean_x;

        let output = table
            .query(
                "SELECT regr_slope(y, x) OVER (ORDER BY ts \
                 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS beta, \
                 regr_intercept(y, x) OVER (ORDER BY ts \
                 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS alpha FROM trades",
            )
            .unwrap();
        let batch = output.batches.last().unwrap();
        let last = batch.num_rows() - 1;
        let beta = f64_column(batch, 0).values().as_slice()[last];
        let alpha = f64_column(batch, 1).values().as_slice()[last];
        let beta_err = ((beta - slope_ref) / slope_ref).abs();
        assert!(
            beta_err < 1e-6,
            "slope {beta} vs reference {slope_ref} (relative error {beta_err:e})"
        );
        let alpha_err = ((alpha - intercept_ref) / intercept_ref).abs();
        assert!(
            alpha_err < 1e-6,
            "intercept {alpha} vs reference {intercept_ref} (relative error {alpha_err:e})"
        );
    }

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
        // Release the directory lock so the next open reaches the
        // schema check (a held lock refuses first, by design).
        drop(reopened);
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
        // The reserved pseudocolumn name, refused through the Rust API
        // exactly as `CREATE TABLE` refuses it.
        let shadowing = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new(query_lite::SEQUENCE_COLUMN, ColumnType::I64, false),
        ]);
        assert!(matches!(
            Table::new("t", shadowing, "ts"),
            Err(EngineError::ReservedColumn(_))
        ));
    }
}

#[cfg(test)]
mod ddl_and_insert {
    //! M3.5's engine half: `INSERT ... VALUES` through the mutation
    //! entry (WAL included, exact-or-loud literal coercion) and the
    //! #49 DDL grammar mapped onto engine types.

    use super::tests::m1_schema;
    use super::*;

    #[test]
    fn insert_appends_through_the_ordinary_path() {
        // As m1_schema, but with y nullable so NULL literals land.
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, true),
        ]);
        let mut table = Table::with_segment_rows("t", schema, "ts", 4).unwrap();
        let n = table
            .mutate("INSERT INTO t VALUES (1, 'A', 1.5, 2.5), (2, 'B', -3, NULL)")
            .unwrap();
        assert_eq!(n, 2);
        let output = table.query("SELECT ts, x, y FROM t ORDER BY ts").unwrap();
        assert_eq!(output.num_rows(), 2);
        // Integer literal -3 landed exactly in the f64 column.
        let batch = &output.batches[0];
        let arrow_lite::Column::Numeric(arrow_lite::NumericData::F64(x)) = &batch.columns()[1]
        else {
            panic!("x is f64")
        };
        assert_eq!(x.values().as_slice()[1], -3.0);
    }

    #[test]
    fn insert_honors_a_named_column_order() {
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 4).unwrap();
        table
            .mutate("INSERT INTO t (x, ts, sym, y) VALUES (9.0, 7, 'C', 1.0)")
            .unwrap();
        let output = table.query("SELECT ts, x FROM t").unwrap();
        let batch = &output.batches[0];
        let arrow_lite::Column::Numeric(arrow_lite::NumericData::I64(ts)) = &batch.columns()[0]
        else {
            panic!("ts is i64")
        };
        assert_eq!(ts.values().as_slice()[0], 7);
    }

    #[test]
    fn insert_coercions_are_exact_or_loud() {
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 4).unwrap();
        // Float into BIGINT would truncate: loud.
        assert!(table
            .mutate("INSERT INTO t VALUES (1.5, 'A', 1.0, 1.0)")
            .is_err());
        // Number into a SYMBOL column: loud.
        assert!(table
            .mutate("INSERT INTO t VALUES (1, 7, 1.0, 1.0)")
            .is_err());
        // String into numeric: loud (storage's own validation).
        assert!(table
            .mutate("INSERT INTO t VALUES (1, 'A', 'oops', 1.0)")
            .is_err());
        // Wrong arity: loud.
        assert!(table.mutate("INSERT INTO t VALUES (1, 'A', 1.0)").is_err());
        // Nothing partial landed from the failures... the first row of a
        // multi-row INSERT that fails midway HAS landed (documented
        // non-atomicity is #56/#40 territory); here every statement
        // failed on its only row, so the table is empty.
        assert_eq!(table.query("SELECT ts FROM t").unwrap().num_rows(), 0);
    }

    #[test]
    fn int_to_double_exactness_holds_at_the_i64_edges() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, false),
        ]);
        let mut table = Table::with_segment_rows("t", schema, "ts", 4).unwrap();
        // i64::MAX rounds up to 2^63 as f64; the saturating cast back
        // lands on i64::MAX again, so a naive round-trip check passes a
        // value it must refuse.
        assert!(table
            .mutate("INSERT INTO t VALUES (1, 9223372036854775807)")
            .is_err());
        // 2^53 + 1 is the first plain integer f64 cannot hold.
        assert!(table
            .mutate("INSERT INTO t VALUES (1, 9007199254740993)")
            .is_err());
        // -2^53 is exactly representable: accepted (and the negative
        // path through the unary minus works).
        table
            .mutate("INSERT INTO t VALUES (1, -9007199254740992)")
            .unwrap();
        assert_eq!(table.query("SELECT x FROM t").unwrap().num_rows(), 1);
    }

    #[test]
    fn the_ddl_grammar_maps_onto_engine_types() {
        let sql = "CREATE TABLE ticks (ts BIGINT ORDERING KEY, sym SYMBOL NOT NULL, px DOUBLE)";
        let Statement::CreateTable(plan) = parse_statement(sql).unwrap() else {
            panic!("parses as CREATE TABLE")
        };
        let (schema, ordering) = schema_from_create(&plan).unwrap();
        assert_eq!(ordering, "ts");
        let fields = schema.fields();
        assert_eq!(fields[0].column_type(), ColumnType::I64);
        assert!(!fields[0].nullable(), "ordering key implies NOT NULL");
        assert_eq!(fields[1].column_type(), ColumnType::Key);
        assert!(!fields[1].nullable());
        assert_eq!(fields[2].column_type(), ColumnType::F64);
        assert!(fields[2].nullable());
        // And the whole thing actually builds a working table.
        let mut table = Table::new("ticks", schema, &ordering).unwrap();
        table
            .mutate("INSERT INTO ticks VALUES (1, 'ES', 5432.25)")
            .unwrap();
        assert_eq!(table.query("SELECT px FROM ticks").unwrap().num_rows(), 1);
    }

    #[test]
    fn varchar_is_refused_with_a_teaching_error() {
        let error =
            parse_statement("CREATE TABLE t (ts BIGINT ORDERING KEY, name VARCHAR)").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("interned labels"), "{message}");
        assert!(message.contains("SYMBOL"), "{message}");
        // And the retired spelling names its replacement rather than
        // reciting the type list.
        let message = parse_statement("CREATE TABLE t (ts BIGINT ORDERING KEY, sym KEY)")
            .unwrap_err()
            .to_string();
        assert!(message.contains("spelled SYMBOL"), "{message}");
        // No ordering key: loud, with guidance.
        let error = parse_statement("CREATE TABLE t (a BIGINT, b DOUBLE)").unwrap_err();
        assert!(error.to_string().contains("ORDERING KEY"), "{}", error);
        // A DOUBLE ordering key: loud.
        let Statement::CreateTable(plan) =
            parse_statement("CREATE TABLE t (a DOUBLE ORDERING KEY)").unwrap()
        else {
            panic!()
        };
        assert!(schema_from_create(&plan).is_err());
    }
}

#[cfg(test)]
mod snapshot_concurrency {
    //! #51's evidence: single writer + concurrent snapshot readers. One
    //! test sequences the interleaving deterministically over channels
    //! (a held snapshot survives a delete, appends, and a compaction —
    //! the generation swap — unchanged); one races freely as a smoke
    //! test; one pins the Send/Sync story at compile time.

    use super::tests::{flatten, linear_row, m1_schema};
    use super::*;

    /// COUNT over the snapshot — one number summarizing what it sees.
    fn count(snapshot: &TableSnapshot) -> f64 {
        let output = snapshot.query("SELECT COUNT(x) AS c FROM t").unwrap();
        let arrow_lite::Column::Numeric(arrow_lite::NumericData::I64(c)) =
            &output.batches[0].columns()[0]
        else {
            panic!("COUNT returns i64");
        };
        c.values().as_slice()[0] as f64
    }

    #[test]
    fn a_held_snapshot_survives_delete_append_and_compaction() {
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 8).unwrap();
        for i in 0..100i64 {
            table.append(&linear_row(i)).unwrap();
        }
        let reader = table.reader();
        let (to_reader, reader_gate) = std::sync::mpsc::channel::<()>();
        let (to_main, main_gate) = std::sync::mpsc::channel::<f64>();
        let worker = std::thread::spawn(move || {
            let held = reader.snapshot().unwrap();
            to_main.send(count(&held)).unwrap();
            // Main deletes, appends, and compacts before releasing us.
            reader_gate.recv().unwrap();
            // Point-in-time stability: the held snapshot's answer is
            // unchanged across the mutation storm and the generation
            // swap — its old segments live on through their Arcs.
            to_main.send(count(&held)).unwrap();
            // A fresh snapshot sees the new world.
            let fresh = reader.snapshot().unwrap();
            to_main.send(count(&fresh)).unwrap();
        });
        assert_eq!(main_gate.recv().unwrap(), 100.0);
        table.mutate("DELETE FROM t WHERE ts < 10").unwrap();
        for i in 100..150i64 {
            table.append(&linear_row(i)).unwrap();
        }
        table.compact().unwrap();
        to_reader.send(()).unwrap();
        assert_eq!(main_gate.recv().unwrap(), 100.0, "held snapshot moved");
        assert_eq!(main_gate.recv().unwrap(), 140.0, "fresh snapshot wrong");
        worker.join().unwrap();
    }

    #[test]
    fn snapshots_race_a_live_writer_safely() {
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 16).unwrap();
        table.append(&linear_row(0)).unwrap();
        let reader = table.reader();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let worker = std::thread::spawn(move || {
            let mut last = 0.0f64;
            let mut snapshots = 0u32;
            while !stopped.load(std::sync::atomic::Ordering::Relaxed) {
                let snapshot = reader.snapshot().unwrap();
                let seen = count(&snapshot);
                // An append-only writer can never make a later snapshot
                // smaller; a torn read would.
                assert!(seen >= last, "count went backwards: {seen} < {last}");
                last = seen;
                snapshots += 1;
            }
            snapshots
        });
        for i in 1..2_000i64 {
            table.append(&linear_row(i)).unwrap();
            if i % 900 == 0 {
                table.compact().unwrap();
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let snapshots = worker.join().unwrap();
        assert!(snapshots > 0, "the reader never got a snapshot in");
    }

    #[test]
    fn reader_and_snapshot_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TableReader>();
        assert_send_sync::<TableSnapshot>();
    }

    #[test]
    fn a_registered_column_function_runs_in_projection() {
        // The native half of the vectorized scalar slot (#53): one
        // call per view, dense arguments, NULL decided by the kernel.
        struct SpreadPct;
        impl ColumnFunction for SpreadPct {
            fn arity(&self) -> usize {
                2
            }
            fn evaluate(&self, args: &[(&[f64], &[bool])]) -> Result<Vec<Option<f64>>, String> {
                let (x, xv) = args[0];
                let (y, yv) = args[1];
                Ok((0..x.len())
                    .map(|i| (xv[i] && yv[i] && y[i] != 0.0).then(|| (x[i] - y[i]) / y[i]))
                    .collect())
            }
        }
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 4).unwrap();
        for i in 0..6i64 {
            table.append(&linear_row(i)).unwrap();
        }
        table
            .register_column_function("spread_pct", SpreadPct)
            .unwrap();
        // Composes with expression arguments through the same machinery.
        let output = table
            .query("SELECT spread_pct(x + 1.0, x + 1.0) AS s FROM t WHERE ts >= 1")
            .unwrap();
        let values = flatten(&output, 0);
        assert_eq!(values, vec![Some(0.0); 5]);
        // Unknown names stay loud, at execution, by name.
        let error = table
            .query("SELECT nope(x) FROM t")
            .unwrap_err()
            .to_string();
        assert!(error.contains("nope"), "{error}");
        // Wrong arity is loud too.
        let error = table
            .query("SELECT spread_pct(x) FROM t")
            .unwrap_err()
            .to_string();
        assert!(error.contains("argument"), "{error}");
    }

    #[test]
    fn snapshots_pin_the_functions_of_their_moment() {
        // A native kernel through the public trait path, so this test
        // runs identically with and without the `lua` feature.
        struct Twice;
        impl WindowAggregate for Twice {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok(Some(2.0 * args[0].iter().map(|v| v * v).sum::<f64>()))
            }
        }
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 8).unwrap();
        for i in 0..4i64 {
            table.append(&linear_row(i)).unwrap();
        }
        let before = table.snapshot().unwrap();
        table.register_window("twice", Twice).unwrap();
        let after = table.snapshot().unwrap();
        let sql = "SELECT twice(x) OVER (ORDER BY ts ROWS BETWEEN 0 PRECEDING                    AND CURRENT ROW) AS d FROM t";
        assert!(
            before.query(sql).is_err(),
            "pre-registration snapshot knows the function"
        );
        assert!(after.query(sql).is_ok());
    }
}

#[cfg(test)]
mod mutation_tests {
    use super::tests::{f64_column, flatten, linear_row, m1_schema};
    use super::*;

    fn small_table() -> Table {
        // ts, sym, x, y — segment size 3 so mutations cross segments.
        let mut table = Table::with_segment_rows("t", m1_schema(), "ts", 3).unwrap();
        for i in 0..10i64 {
            table.append(&linear_row(i)).unwrap();
        }
        table
    }

    #[test]
    fn delete_removes_matched_rows_everywhere() {
        let mut table = small_table();
        let affected = table.mutate("DELETE FROM t WHERE sym = 'B'").unwrap();
        assert_eq!(affected, 5);
        let output = table.query("SELECT ts, x FROM t").unwrap();
        assert_eq!(output.num_rows(), 5);
        // Deleting again affects nothing (idempotent end state).
        assert_eq!(table.mutate("DELETE FROM t WHERE sym = 'B'").unwrap(), 0);
        // Unqualified DELETE clears the table.
        assert_eq!(table.mutate("DELETE FROM t").unwrap(), 5);
        assert_eq!(table.query("SELECT ts FROM t").unwrap().num_rows(), 0);
    }

    #[test]
    fn is_null_selects_rows_for_mutation() {
        // The predicate's other consumer: DELETE/UPDATE reach the same
        // evaluator, and a wrong bitmap here destroys data rather than
        // returning the wrong page. Nulls span segments (size 3).
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("y", ColumnType::F64, true),
        ]);
        let mut table = Table::with_segment_rows("t", schema, "ts", 3).unwrap();
        for i in 0..10i64 {
            let y = if i % 3 == 0 {
                RowValue::Null
            } else {
                RowValue::F64(i as f64)
            };
            table
                .append(&[RowValue::I64(i), RowValue::Key("A"), y])
                .unwrap();
        }
        // ts 0, 3, 6, 9 are null.
        assert_eq!(
            table
                .query("SELECT ts FROM t WHERE y IS NULL ORDER BY ts")
                .unwrap()
                .num_rows(),
            4
        );
        // UPDATE finds exactly those rows, and they stop being null.
        // (0 is a value no other row carries — y is `i` elsewhere.)
        assert_eq!(
            table.mutate("UPDATE t SET y = 0 WHERE y IS NULL").unwrap(),
            4
        );
        assert_eq!(
            table
                .query("SELECT ts FROM t WHERE y IS NULL")
                .unwrap()
                .num_rows(),
            0
        );
        assert_eq!(
            table
                .query("SELECT ts FROM t WHERE y IS NOT NULL")
                .unwrap()
                .num_rows(),
            10
        );
        // DELETE takes the rows the UPDATE just wrote, and only those.
        assert_eq!(table.mutate("DELETE FROM t WHERE y = 0").unwrap(), 4);
        let survivors = table.query("SELECT ts FROM t ORDER BY ts").unwrap();
        assert_eq!(survivors.num_rows(), 6);
    }

    #[test]
    fn update_is_tombstone_plus_reappend() {
        let mut table = small_table();
        let affected = table
            .mutate("UPDATE t SET y = 0 WHERE ts >= 8 AND sym = 'A'")
            .unwrap();
        assert_eq!(affected, 1); // only ts=8 is 'A' in 8..10
        let output = table.query("SELECT ts, y FROM t").unwrap();
        assert_eq!(output.num_rows(), 10); // row count unchanged
        let pairs: Vec<(i64, f64)> = output
            .batches
            .iter()
            .flat_map(|batch| {
                let Column::Numeric(NumericData::I64(ts)) = &batch.columns()[0] else {
                    panic!("ts type")
                };
                let y = f64_column(batch, 1);
                (0..batch.num_rows())
                    .map(|row| (ts.values().as_slice()[row], y.values().as_slice()[row]))
                    .collect::<Vec<_>>()
            })
            .collect();
        // The corrected copy exists with y = 0; the original is gone.
        assert!(pairs.contains(&(8, 0.0)));
        assert_eq!(pairs.iter().filter(|(ts, _)| *ts == 8).count(), 1);
        // Windows before compaction refuse the reappend's disorder…
        let window =
            "SELECT regr_slope(y, x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) \
             FROM t";
        assert!(matches!(
            table.query(window),
            Err(EngineError::Query(QueryError::Unordered(_)))
        ));
        // …and compaction restores order and the query runs.
        table.compact().unwrap();
        table.query(window).unwrap();
    }

    #[test]
    fn update_validates_before_mutating() {
        let mut table = small_table();
        // Type mismatch: string into a numeric column.
        assert!(matches!(
            table.mutate("UPDATE t SET x = 'oops' WHERE ts = 1"),
            Err(EngineError::Query(QueryError::TypeError(_)))
        ));
        // NULL into NOT NULL.
        assert!(matches!(
            table.mutate("UPDATE t SET ts = NULL"),
            Err(EngineError::Query(QueryError::TypeError(_)))
        ));
        // Unknown column.
        assert!(matches!(
            table.mutate("UPDATE t SET nope = 1"),
            Err(EngineError::Query(QueryError::UnknownColumn(_)))
        ));
        // Nothing changed.
        assert_eq!(table.query("SELECT ts FROM t").unwrap().num_rows(), 10);
        let output = table.query("SELECT x FROM t").unwrap();
        assert_eq!(
            flatten(&output, 0).iter().filter(|v| v.is_none()).count(),
            0
        );
    }

    #[test]
    fn update_can_rewrite_keys_and_set_null() {
        let schema = Schema::new(vec![
            arrow_lite::Field::new("ts", ColumnType::I64, false),
            arrow_lite::Field::new("sym", ColumnType::Key, false),
            arrow_lite::Field::new("y", ColumnType::F64, true),
        ]);
        let mut table = Table::with_segment_rows("t", schema, "ts", 2).unwrap();
        for i in 0..4i64 {
            table
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key(if i % 2 == 0 { "OLD" } else { "KEEP" }),
                    RowValue::F64(i as f64),
                ])
                .unwrap();
        }
        assert_eq!(
            table
                .mutate("UPDATE t SET sym = 'NEW', y = NULL WHERE sym = 'OLD'")
                .unwrap(),
            2
        );
        table.compact().unwrap();
        let output = table.query("SELECT sym, y FROM t").unwrap();
        let mut names = Vec::new();
        let mut nulls = 0;
        for batch in &output.batches {
            let Column::Key(sym) = &batch.columns()[0] else {
                panic!("sym type")
            };
            let y = f64_column(batch, 1);
            for row in 0..batch.num_rows() {
                names.push(sym.value_at(row).unwrap().to_owned());
                if !y.is_valid(row) {
                    nulls += 1;
                }
            }
        }
        names.sort();
        assert_eq!(names, ["KEEP", "KEEP", "NEW", "NEW"]);
        assert_eq!(nulls, 2);
    }

    #[test]
    fn mutations_persist_and_survive_reopen() {
        let dir = std::env::temp_dir().join(format!("tallydb-mutate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut table =
                Table::persistent_with_segment_rows("t", m1_schema(), "ts", &dir, 3).unwrap();
            for i in 0..9i64 {
                table.append(&linear_row(i)).unwrap();
            }
            table.flush().unwrap();
            assert_eq!(table.mutate("DELETE FROM t WHERE ts < 3").unwrap(), 3);
        }
        {
            // Tombstones survived without compaction.
            let table =
                Table::persistent_with_segment_rows("t", m1_schema(), "ts", &dir, 3).unwrap();
            assert_eq!(table.query("SELECT ts FROM t").unwrap().num_rows(), 6);
            let mut table = table;
            table.compact().unwrap();
        }
        // And the compacted state reopens identically.
        let table = Table::persistent_with_segment_rows("t", m1_schema(), "ts", &dir, 3).unwrap();
        assert_eq!(table.query("SELECT ts FROM t").unwrap().num_rows(), 6);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn database_routes_mutations_by_table() {
        let mut db = crate::Database::new();
        db.add_table(Table::with_segment_rows("a", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(Table::with_segment_rows("b", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        for i in 0..6i64 {
            let row = linear_row(i);
            db.table_mut("a").unwrap().append(&row).unwrap();
            db.table_mut("b").unwrap().append(&row).unwrap();
        }
        assert_eq!(db.mutate("DELETE FROM a WHERE ts < 2").unwrap(), 2);
        assert_eq!(db.query("SELECT ts FROM a").unwrap().num_rows(), 4);
        assert_eq!(db.query("SELECT ts FROM b").unwrap().num_rows(), 6);
        db.compact("a").unwrap();
        assert_eq!(db.query("SELECT ts FROM a").unwrap().num_rows(), 4);
        assert!(matches!(
            db.mutate("DELETE FROM nope"),
            Err(EngineError::UnknownTable(_))
        ));
    }
}

#[cfg(test)]
mod window_truth {
    //! The accuracy yardstick for every window statistic: a compensated
    //! high-precision computation, far better than plain f64, so "which
    //! algorithm is closer to the true answer" is a measurable question
    //! rather than an assumption. The lesson this module encodes: every
    //! wrong accuracy conclusion this project has drawn came from judging
    //! one f64 algorithm against another — a metric blind to the quantity
    //! under test, or the shipped code standing in for truth. Nothing may
    //! be judged accurate by comparison with itself.

    /// The statistics one window's moments yield. `intercept` needs the
    /// window means as well as the central moments, so only
    /// `high_precision` fills it; `stats_from` leaves it NaN.
    #[derive(Clone, Copy, Default)]
    pub(crate) struct Stats {
        pub(crate) covar: f64,
        pub(crate) corr: f64,
        pub(crate) eigen_max: f64,
        pub(crate) slope: f64,
        pub(crate) intercept: f64,
    }

    /// Derives the statistics from centered moments — the shared tail of
    /// every algorithm under test, so differences are in the moments.
    pub(crate) fn stats_from(var_y: f64, var_x: f64, covar: f64) -> Stats {
        let corr = if var_y > 0.0 && var_x > 0.0 {
            covar / (var_y * var_x).sqrt()
        } else {
            f64::NAN
        };
        let half_trace = (var_y + var_x) / 2.0;
        let radius = ((var_y - var_x) / 2.0).hypot(covar);
        Stats {
            covar,
            corr,
            eigen_max: half_trace + radius,
            slope: if var_x > 0.0 { covar / var_x } else { f64::NAN },
            intercept: f64::NAN,
        }
    }

    /// Compensated (Neumaier) accumulation: keeps the rounding error of
    /// every addition in a second term, so a sum of n values carries
    /// error of order eps² rather than n·eps.
    #[derive(Default, Clone, Copy)]
    pub(crate) struct Compensated {
        hi: f64,
        lo: f64,
    }

    impl Compensated {
        pub(crate) fn add(&mut self, value: f64) {
            // Knuth's two-sum: `sum` plus `error` reproduces the
            // operands exactly, whichever is larger.
            let sum = self.hi + value;
            let shifted = sum - self.hi;
            self.lo += (self.hi - (sum - shifted)) + (value - shifted);
            self.hi = sum;
        }

        /// Adds `a · b`, keeping the product's own rounding error too —
        /// `mul_add` is a fused multiply-add, so it yields that error
        /// exactly.
        pub(crate) fn add_product(&mut self, a: f64, b: f64) {
            let product = a * b;
            self.add(product);
            self.lo += a.mul_add(b, -product);
        }

        pub(crate) fn value(self) -> f64 {
            self.hi + self.lo
        }
    }

    /// The reference: per trailing window, moments about `(x[0], y[0])`
    /// — exact shifts for offset data, removing the cancellation the
    /// means introduce — accumulated with compensation.
    pub(crate) fn high_precision(y: &[f64], x: &[f64], w: usize) -> Vec<Stats> {
        (0..y.len())
            .map(|i| {
                let lo = (i + 1).saturating_sub(w);
                let (wy, wx) = (&y[lo..=i], &x[lo..=i]);
                let n = wy.len() as f64;
                let (x0, y0) = (wx[0], wy[0]);
                let (mut sdx, mut sdy) = (Compensated::default(), Compensated::default());
                let (mut sxx, mut syy, mut sxy) = (
                    Compensated::default(),
                    Compensated::default(),
                    Compensated::default(),
                );
                for (&yi, &xi) in wy.iter().zip(wx) {
                    let (dx, dy) = (xi - x0, yi - y0);
                    sdx.add(dx);
                    sdy.add(dy);
                    sxx.add_product(dx, dx);
                    syy.add_product(dy, dy);
                    sxy.add_product(dx, dy);
                }
                let (sdx, sdy) = (sdx.value(), sdy.value());
                let var_x = (sxx.value() - sdx * sdx / n) / n;
                let var_y = (syy.value() - sdy * sdy / n) / n;
                let covar = (sxy.value() - sdx * sdy / n) / n;
                let mut stats = stats_from(var_y, var_x, covar);
                // The window means recover exactly from the shifts (x0,
                // y0 are data values) plus the compensated residuals.
                stats.intercept = (y0 + sdy / n) - stats.slope * (x0 + sdx / n);
                stats
            })
            .collect()
    }

    /// Deterministic pseudo-random values in [0, 1).
    pub(crate) struct Lcg(pub(crate) u64);

    impl Lcg {
        pub(crate) fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `(name, y, x)` corpora spanning the benign case and the offset
    /// shapes where cancellation bites — the regime bug #45 was about.
    /// Constructed so every statistic sits well away from zero, keeping
    /// plain relative error meaningful.
    pub(crate) fn corpora(rows: usize) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
        let mut sets = Vec::new();
        for (name, offset) in [
            ("benign (x ~ 0..10)", 0.0),
            ("offset 1e6", 1e6),
            ("offset 1e9 (unix seconds)", 1e9),
            ("offset 1e12 (unix millis)", 1e12),
        ] {
            let mut rng = Lcg(0x3B_5EED_1234_5678);
            let mut x = Vec::with_capacity(rows);
            let mut y = Vec::with_capacity(rows);
            for _ in 0..rows {
                let xi = offset + rng.next() * 10.0;
                x.push(xi);
                y.push(2.0 * (xi - offset) + rng.next());
            }
            sets.push((name, y, x));
        }
        // A monotonic ordering key: the timestamp-regressor shape, where
        // the data also drifts away from any fixed shift.
        let mut rng = Lcg(0x99_5EED_8765_4321);
        let mut x = Vec::with_capacity(rows);
        let mut y = Vec::with_capacity(rows);
        for i in 0..rows {
            let xi = 1e9 + i as f64 * 0.05;
            x.push(xi);
            y.push(0.5 * (i as f64 * 0.05) + rng.next());
        }
        sets.push(("monotonic ts (1e9 + 0.05·i)", y, x));
        sets
    }
}

#[cfg(test)]
mod window_numerics_guard {
    //! The permanent accuracy contract, run in CI on every change (not
    //! ignored, not a measurement): every shipped window statistic must
    //! track the compensated reference at the float noise floor over the
    //! adversarial corpora — offsets to 1e12 and a drifting monotonic
    //! ordering key included. This is what makes "improved performance
    //! without sacrificing correctness" a checked property instead of a
    //! claim: any future change to these ops that trades accuracy away
    //! fails here, loudly, before it lands.
    //!
    //! The bound is 1e-12 relative — two orders above the ~1e-14 the
    //! corrected two-pass achieves (headroom for corpus changes), four
    //! orders below the ~4.9e-8 the uncorrected form degraded to at a
    //! 1e12 offset (the defect this guard exists to keep out; disabling
    //! the correction trips this test at the first 1e12 window,
    //! verified by hand 2026-07-27).

    use super::window_truth::{corpora, high_precision, Stats};
    use super::*;

    const BOUND: f64 = 1e-12;

    fn relative(got: f64, reference: f64) -> f64 {
        ((got - reference) / reference).abs()
    }

    #[test]
    fn shipped_window_statistics_track_the_compensated_reference() {
        let rows = 2_000;
        let w = 64;
        for (name, y, x) in corpora(rows) {
            let reference = high_precision(&y, &x, w);
            for i in (w - 1)..rows {
                let lo = (i + 1).saturating_sub(w);
                let window: [&[f64]; 2] = [&y[lo..=i], &x[lo..=i]];
                let truth = reference[i];
                // The guard guards itself: a statistic near zero would
                // make relative error meaningless, so the corpora must
                // keep them all well away from it.
                assert!(
                    truth.covar.abs() > 1e-3
                        && truth.corr.abs() > 1e-3
                        && truth.eigen_max.abs() > 1e-3
                        && truth.slope.abs() > 1e-3,
                    "{name} row {i}: corpus left a statistic near zero"
                );
                for (kind, expected) in [
                    (PairKind::CovarPop, truth.covar),
                    (PairKind::Corr, truth.corr),
                    (PairKind::EigenMax, truth.eigen_max),
                ] {
                    let got = PairStatistic { kind }
                        .evaluate(&window)
                        .unwrap()
                        .expect("defined on these corpora");
                    assert!(
                        relative(got, expected) < BOUND,
                        "{name} row {i} {kind:?}: {got} vs {expected} (relative {:.2e})",
                        relative(got, expected)
                    );
                }
                for (output, expected) in [
                    (RegressionOutput::Slope, truth.slope),
                    (RegressionOutput::Intercept, truth.intercept),
                ] {
                    let got = RollingRegression { output }
                        .evaluate(&window)
                        .unwrap()
                        .expect("defined on these corpora");
                    assert!(
                        relative(got, expected) < BOUND,
                        "{name} row {i} regression: {got} vs {expected} (relative {:.2e})",
                        relative(got, expected)
                    );
                }
            }
        }
    }

    /// NaN and ±Inf are first-class values here, and they break the
    /// sliding identity (`NaN − NaN = NaN` outlives the row): without
    /// the dirty-frame handling in `shifted_sweep`, the incremental
    /// path returns NaN for windows whose true frames are clean, for up
    /// to a full window after the non-finite row leaves. The oracle
    /// corpus contains no NaN, so only this test sees it: the two paths
    /// must agree at every position — same NULLs, same NaNs, and
    /// near-identical finite values.
    #[test]
    fn incremental_and_recompute_paths_agree_on_nan_and_inf() {
        let rows = 60;
        let mut y: Vec<f64> = (0..rows).map(|i| 0.7 * i as f64 + 3.0).collect();
        let mut x: Vec<f64> = (0..rows).map(|i| 1e9 + 1.3 * i as f64).collect();
        y[7] = f64::NAN;
        x[19] = f64::INFINITY;
        y[20] = f64::NEG_INFINITY;
        y[21] = f64::NAN; // adjacent dirty rows: the counter, not a flag
        let columns: [&[f64]; 2] = [&y, &x];
        let aggregates: Vec<(&str, Box<dyn WindowAggregate>)> = vec![
            (
                "regr_slope",
                Box::new(RollingRegression {
                    output: RegressionOutput::Slope,
                }),
            ),
            (
                "regr_intercept",
                Box::new(RollingRegression {
                    output: RegressionOutput::Intercept,
                }),
            ),
            (
                "covar_pop",
                Box::new(PairStatistic {
                    kind: PairKind::CovarPop,
                }),
            ),
            (
                "corr",
                Box::new(PairStatistic {
                    kind: PairKind::Corr,
                }),
            ),
            (
                "eigen_max",
                Box::new(PairStatistic {
                    kind: PairKind::EigenMax,
                }),
            ),
        ];
        for preceding in [1usize, 3, 7] {
            for (name, aggregate) in &aggregates {
                let incremental = aggregate
                    .evaluate_frames(&columns, Some(preceding))
                    .unwrap();
                let reference =
                    query_lite::recompute_frames(aggregate.as_ref(), &columns, Some(preceding))
                        .unwrap();
                assert_eq!(incremental.len(), reference.len());
                for (i, (got, want)) in incremental.iter().zip(&reference).enumerate() {
                    match (got, want) {
                        (None, None) => {}
                        (Some(a), Some(b)) if a.is_nan() && b.is_nan() => {}
                        // Dirty frames run the same arithmetic on both
                        // paths — ±Inf results are bit-equal.
                        (Some(a), Some(b)) if a == b => {}
                        (Some(a), Some(b)) => assert!(
                            ((a - b) / b).abs() < 1e-9,
                            "{name} w={} row {i}: {a} vs {b}",
                            preceding + 1
                        ),
                        other => panic!(
                            "{name} w={} row {i}: paths disagree on definedness: {other:?}",
                            preceding + 1
                        ),
                    }
                }
            }
        }
    }

    /// The incremental path (`evaluate_frames`, the shifted sweep) is
    /// held to the same reference as the per-window path — the executor
    /// routes every SQL window through it, so it inherits the contract.
    #[test]
    fn incremental_frame_sequences_track_the_compensated_reference() {
        let rows = 2_000;
        let w = 64;
        for (name, y, x) in corpora(rows) {
            let reference = high_precision(&y, &x, w);
            let columns: [&[f64]; 2] = [&y, &x];
            let check = |label: &str, results: Vec<Option<f64>>, pick: &dyn Fn(&Stats) -> f64| {
                for (i, result) in results.iter().enumerate().skip(w - 1) {
                    let got = result.expect("defined on these corpora");
                    let expected = pick(&reference[i]);
                    assert!(
                        relative(got, expected) < BOUND,
                        "{name} row {i} {label}: {got} vs {expected} (relative {:.2e})",
                        relative(got, expected)
                    );
                }
            };
            for (kind, pick) in [
                (
                    PairKind::CovarPop,
                    &(|s: &Stats| s.covar) as &dyn Fn(&Stats) -> f64,
                ),
                (PairKind::Corr, &|s: &Stats| s.corr),
                (PairKind::EigenMax, &|s: &Stats| s.eigen_max),
            ] {
                let results = PairStatistic { kind }
                    .evaluate_frames(&columns, Some(w - 1))
                    .unwrap();
                check(&format!("{kind:?}"), results, pick);
            }
            for (output, label, pick) in [
                (
                    RegressionOutput::Slope,
                    "slope",
                    &(|s: &Stats| s.slope) as &dyn Fn(&Stats) -> f64,
                ),
                (RegressionOutput::Intercept, "intercept", &|s: &Stats| {
                    s.intercept
                }),
            ] {
                let results = RollingRegression { output }
                    .evaluate_frames(&columns, Some(w - 1))
                    .unwrap();
                check(label, results, pick);
            }
        }
    }
}

#[cfg(test)]
mod measure_incremental_windows {
    //! The 3b A/B: **incremental** window moments against the shipped
    //! **recompute-per-window** algorithm — speed *and* numerics, every
    //! variant judged against `window_truth`'s compensated reference
    //! (never against another f64 algorithm; see that module's note).
    //!
    //! The three algorithms:
    //!
    //! - **A — recompute** (what ships): corrected two-pass per window
    //!   (Chan–Golub–LeVeque), matching `PairStatistic` and
    //!   `RollingRegression`. O(n·w) work. (The earlier *uncorrected*
    //!   two-pass shipped until 2026-07-27 and degraded to ~4.9e-8
    //!   relative at a 1e12 offset — the defect `window_numerics_guard`
    //!   now keeps out.)
    //! - **B — naive incremental**: raw running `Σx, Σy, Σxx, Σyy, Σxy`,
    //!   variance as `E[x²] − E[x]²`. O(n), fastest possible, and the
    //!   textbook cancellation trap (bug #45's shape) — kept here as the
    //!   permanent cautionary contrast, never to ship.
    //! - **C — shifted incremental with re-baselining**: the same O(1)
    //!   slide, moments kept about a shift near the data, accumulator
    //!   rebuilt every `w` steps so rounding cannot accumulate across
    //!   the column. One extra pass overall; still O(n).
    //!
    //! Run explicitly, in release:
    //!
    //! ```text
    //! cargo test -p engine --release measure_3b -- --ignored --nocapture
    //! ```

    use super::window_truth::{corpora, high_precision, stats_from, Stats};

    /// One algorithm under test: window statistics for a whole column.
    type WindowAlgorithm<'a> = &'a dyn Fn(&[f64], &[f64], usize) -> Vec<Stats>;

    /// A — the shipped arrangement: corrected two-pass per window.
    fn recompute(y: &[f64], x: &[f64], w: usize) -> Vec<Stats> {
        (0..y.len())
            .map(|i| {
                let lo = (i + 1).saturating_sub(w);
                let (wy, wx) = (&y[lo..=i], &x[lo..=i]);
                let n = wy.len() as f64;
                let mean_y = wy.iter().sum::<f64>() / n;
                let mean_x = wx.iter().sum::<f64>() / n;
                let (mut sdy, mut sdx) = (0.0f64, 0.0f64);
                let (mut vy, mut vx, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
                for (&yi, &xi) in wy.iter().zip(wx) {
                    let (dy, dx) = (yi - mean_y, xi - mean_x);
                    sdy += dy;
                    sdx += dx;
                    vy += dy * dy;
                    vx += dx * dx;
                    cxy += dy * dx;
                }
                stats_from(
                    (vy - sdy * sdy / n) / n,
                    (vx - sdx * sdx / n) / n,
                    (cxy - sdy * sdx / n) / n,
                )
            })
            .collect()
    }

    /// Running raw moments; `ky`/`kx` are subtracted from every value as
    /// it enters (zero for the naive variant).
    #[derive(Default, Clone, Copy)]
    struct Moments {
        n: f64,
        sy: f64,
        sx: f64,
        syy: f64,
        sxx: f64,
        sxy: f64,
        ky: f64,
        kx: f64,
    }

    impl Moments {
        fn add(&mut self, yi: f64, xi: f64) {
            let (dy, dx) = (yi - self.ky, xi - self.kx);
            self.n += 1.0;
            self.sy += dy;
            self.sx += dx;
            self.syy += dy * dy;
            self.sxx += dx * dx;
            self.sxy += dy * dx;
        }

        fn remove(&mut self, yi: f64, xi: f64) {
            let (dy, dx) = (yi - self.ky, xi - self.kx);
            self.n -= 1.0;
            self.sy -= dy;
            self.sx -= dx;
            self.syy -= dy * dy;
            self.sxx -= dx * dx;
            self.sxy -= dy * dx;
        }

        fn stats(&self) -> Stats {
            let (my, mx) = (self.sy / self.n, self.sx / self.n);
            stats_from(
                self.syy / self.n - my * my,
                self.sxx / self.n - mx * mx,
                self.sxy / self.n - my * mx,
            )
        }
    }

    /// B — naive incremental: raw moments, no shift, never rebuilt.
    fn incremental_naive(y: &[f64], x: &[f64], w: usize) -> Vec<Stats> {
        let mut moments = Moments::default();
        (0..y.len())
            .map(|i| {
                moments.add(y[i], x[i]);
                if i >= w {
                    moments.remove(y[i - w], x[i - w]);
                }
                moments.stats()
            })
            .collect()
    }

    /// C — shifted incremental, rebuilt every `w` steps about a shift
    /// taken from the current window.
    fn incremental_shifted(y: &[f64], x: &[f64], w: usize) -> Vec<Stats> {
        let mut moments = Moments::default();
        let mut since_rebuild = usize::MAX; // force a build on the first row
        (0..y.len())
            .map(|i| {
                if since_rebuild >= w {
                    let lo = (i + 1).saturating_sub(w);
                    moments = Moments {
                        ky: y[i],
                        kx: x[i],
                        ..Moments::default()
                    };
                    for j in lo..=i {
                        moments.add(y[j], x[j]);
                    }
                    since_rebuild = 0;
                } else {
                    moments.add(y[i], x[i]);
                    if i >= w {
                        moments.remove(y[i - w], x[i - w]);
                    }
                    since_rebuild += 1;
                }
                moments.stats()
            })
            .collect()
    }

    /// Worst relative difference against the high-precision reference,
    /// over full windows only (short leading windows are a different
    /// regime).
    fn worst_relative_error(got: &[Stats], reference: &[Stats], w: usize) -> Stats {
        let mut worst = Stats::default();
        let relative = |a: f64, b: f64| {
            if b == 0.0 || !b.is_finite() || !a.is_finite() {
                if a == b || (a.is_nan() && b.is_nan()) {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                ((a - b) / b).abs()
            }
        };
        for i in w..got.len() {
            worst.covar = worst.covar.max(relative(got[i].covar, reference[i].covar));
            worst.corr = worst.corr.max(relative(got[i].corr, reference[i].corr));
            worst.eigen_max = worst
                .eigen_max
                .max(relative(got[i].eigen_max, reference[i].eigen_max));
            worst.slope = worst.slope.max(relative(got[i].slope, reference[i].slope));
        }
        worst
    }

    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn measure_3b_incremental_vs_recompute() {
        let rows = 20_000;
        let w = 64;
        println!(
            "3b A/B: {rows} rows, window {w}; A = corrected recompute (shipped), \
             B = naive incremental, C = shifted incremental rebuilt every {w}.\n\
             Errors are worst relative deviation from the compensated \
             high-precision reference."
        );
        for (name, y, x) in corpora(rows) {
            // Timing: best of a few passes, black-boxed against DCE.
            let time = |f: WindowAlgorithm<'_>| {
                let mut best = f64::INFINITY;
                let mut last = Vec::new();
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    last = f(&y, &x, w);
                    best = best.min(start.elapsed().as_secs_f64());
                    std::hint::black_box(&last);
                }
                (best, last)
            };
            let (time_a, recomputed) = time(&recompute);
            let (time_b, naive) = time(&incremental_naive);
            let (time_c, shifted) = time(&incremental_shifted);
            let reference = high_precision(&y, &x, w);
            let error_a = worst_relative_error(&recomputed, &reference, w);
            let error_b = worst_relative_error(&naive, &reference, w);
            let error_c = worst_relative_error(&shifted, &reference, w);
            println!("\n  {name}");
            println!(
                "    time    A {:>8.2}ms   B {:>8.2}ms ({:>5.1}x)   C {:>8.2}ms ({:>5.1}x)",
                time_a * 1e3,
                time_b * 1e3,
                time_a / time_b,
                time_c * 1e3,
                time_a / time_c
            );
            for (label, error) in [
                ("A recompute (shipped)", error_a),
                ("B naive incremental  ", error_b),
                ("C shifted incremental", error_c),
            ] {
                println!(
                    "    {label}  covar {:.2e}  corr {:.2e}  eigen {:.2e}  slope {:.2e}",
                    error.covar, error.corr, error.eigen_max, error.slope
                );
            }
        }
    }
}

#[cfg(test)]
mod regression_numerics {
    //! Guards for the closed-form rolling regression — specifically the
    //! part of it that is easy to "simplify" away.
    //!
    //! `RollingRegression` uses the **corrected** two-pass form: it
    //! carries `Σ(x − x̄)` and `Σ(y − ȳ)` rather than assuming the
    //! centering left them exactly zero. The correction looks redundant —
    //! both sums are zero in exact arithmetic — and deleting it is the
    //! obvious cleanup. These tests exist so that cleanup fails loudly.
    //!
    //! The reference is an independent, more accurate computation of the
    //! same statistic: moments taken about `x[0]` and `y[0]` rather than
    //! about the means. For offset data those shifts are exact (the
    //! values share an exponent, so the subtraction is), which removes
    //! the cancellation the means introduce.

    use super::*;

    /// Slope and the fitted value at the last row, computed about
    /// `(x[0], y[0])` — no cancellation, so this is the reference.
    fn reference(x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let (x0, y0) = (x[0], y[0]);
        let (mut sdx, mut sdy, mut sxy, mut sxx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&xi, &yi) in x.iter().zip(y) {
            let (dx, dy) = (xi - x0, yi - y0);
            sdx += dx;
            sdy += dy;
            sxy += dx * dy;
            sxx += dx * dx;
        }
        let slope = (sxy - sdx * sdy / n) / (sxx - sdx * sdx / n);
        // Fit at the last row, expressed relative to (x0, y0) so no huge
        // intercept is ever formed.
        let mean_dx = sdx / n;
        let mean_dy = sdy / n;
        let last = x[x.len() - 1] - x0;
        (slope, y0 + mean_dy + slope * (last - mean_dx))
    }

    /// A 64-row window on an offset ordering key, `y = 3·(x − offset) + 7`
    /// so the slope is 3 with respect to x.
    ///
    /// The x values are deliberately **irregular**. An evenly spaced ramp
    /// lands on exactly-representable values at any offset, which makes
    /// the mean exact, `Σ(x − x̄)` exactly zero, and the correction under
    /// test a no-op — the fixture would pass whether or not the code is
    /// right. Irregular values do not divide evenly into the offset's
    /// ulp, so the centering is inexact and the correction matters.
    fn offset_window(offset: f64) -> (Vec<f64>, Vec<f64>) {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let x: Vec<f64> = (0..64)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                offset + (state >> 11) as f64 / (1u64 << 53) as f64 * 10.0
            })
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * (xi - offset) + 7.0).collect();
        (x, y)
    }

    fn slope_of(y: &[f64], x: &[f64]) -> Option<f64> {
        RollingRegression {
            output: RegressionOutput::Slope,
        }
        .evaluate(&[y, x])
        .unwrap()
    }

    #[test]
    fn slope_survives_timestamp_scale_offsets() {
        // Bug #45's property, now guarded without LAPACK: a regressor
        // carrying a unix-timestamp-scale offset must not lose the slope.
        for offset in [0.0, 1e6, 1e9, 1e12, 1e15] {
            let (x, y) = offset_window(offset);
            let slope = slope_of(&y, &x).expect("a 64-row window with spread is defined");
            let (expected, _) = reference(&x, &y);
            assert!(
                ((slope - expected) / expected).abs() < 1e-12,
                "offset {offset:e}: slope {slope} vs reference {expected}"
            );
            // The data is exactly linear, so the true slope is known.
            assert!(
                (slope - 3.0).abs() < 1e-9,
                "offset {offset:e}: slope {slope} is not 3"
            );
        }
    }

    #[test]
    fn the_centering_correction_is_load_bearing() {
        // The naive form — `a = ȳ`, no `Σ(x − x̄)` correction — against
        // the shipped one, both judged by the fitted value at the last
        // row (predictions, not the x = 0 intercept, which is
        // intrinsically imprecise for offset data whatever the method).
        fn naive_fit_at_last(x: &[f64], y: &[f64]) -> f64 {
            let n = x.len() as f64;
            let mean_x = x.iter().sum::<f64>() / n;
            let mean_y = y.iter().sum::<f64>() / n;
            let (mut sxy, mut sxx) = (0.0f64, 0.0f64);
            for (&xi, &yi) in x.iter().zip(y) {
                let dx = xi - mean_x;
                sxy += dx * (yi - mean_y);
                sxx += dx * dx;
            }
            let slope = sxy / sxx;
            // Naive: intercept taken as ȳ, then un-centered.
            let intercept = mean_y - slope * mean_x;
            intercept + slope * x[x.len() - 1]
        }

        fn shipped_fit_at_last(x: &[f64], y: &[f64]) -> f64 {
            let slope = slope_of(y, x).expect("defined");
            let intercept = RollingRegression {
                output: RegressionOutput::Intercept,
            }
            .evaluate(&[y, x])
            .unwrap()
            .expect("defined");
            intercept + slope * x[x.len() - 1]
        }

        // At a 1e12 offset the naive form's fitted value drifts further
        // than the shipped one. Both are compared to the
        // cancellation-free reference.
        let (x, y) = offset_window(1e12);
        let (slope, expected) = reference(&x, &y);
        let naive_error = (naive_fit_at_last(&x, &y) - expected).abs();
        let shipped_error = (shipped_fit_at_last(&x, &y) - expected).abs();
        assert!(
            shipped_error < naive_error,
            "the correction bought nothing: shipped {shipped_error:e}, naive {naive_error:e}"
        );
        // How close the shipped form can possibly get: reconstructing a
        // fit of order 100 through an x = 0 intercept of order 3e12 is
        // bounded by that intercept's own resolution, whatever the
        // estimator. The bound is the representation floor, not a
        // tolerance chosen to pass.
        let intercept_scale = slope * x[0];
        let floor = f64::EPSILON * intercept_scale.abs();
        assert!(
            shipped_error <= floor,
            "shipped fit drifted {shipped_error:e}, past the {floor:e} representation floor"
        );

        // Where the intercept *is* well resolved, the fit is tight in
        // absolute terms too.
        let (x, y) = offset_window(1e6);
        let (_, expected) = reference(&x, &y);
        let shipped_error = (shipped_fit_at_last(&x, &y) - expected).abs();
        assert!(
            shipped_error < 1e-8,
            "at a 1e6 offset the shipped fit drifted {shipped_error:e}"
        );
    }

    #[test]
    fn degenerate_windows_are_null_not_wrong() {
        // Constant x: no slope exists. One row: undefined. NaN: the
        // regression is undefined rather than silently NaN-valued.
        let constant = vec![5.0f64; 16];
        let y: Vec<f64> = (0..16).map(f64::from).collect();
        assert_eq!(slope_of(&y, &constant), None);
        assert_eq!(slope_of(&[1.0], &[1.0]), None);
        let with_nan = [1.0f64, 2.0, f64::NAN, 4.0];
        let plain = [1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(slope_of(&plain, &with_nan), None);
    }
}
