//! The multi-table handle: what an application embeds.
//!
//! A [`Database`] is a set of named [`Table`]s — and maintained views
//! (#83) — behind a SQL doorway that routes each query to what it
//! names, a view's through the union read. It adds no storage or
//! execution machinery of its own — each table still owns its store and
//! its registered compute — but it is the shape applications program
//! against (`add_table` / `append` / `query` / `mutate`), and it is
//! where star-schema joins resolve their dimension tables.

use crate::table::{EngineError, Table};
use crate::view::MaterializedView;
use arrow_lite::{ArrowArrayStream, Schema};
use query_lite::{parse_statement, plan, QueryError, QueryOutput, Statement};
use std::collections::HashMap;
use storage_lite::RowValue;

/// A set of named tables — and maintained views — behind one SQL
/// doorway.
///
/// ```
/// use arrow_lite::{ColumnType, Field, Schema};
/// use engine::{Database, RowValue};
///
/// let mut db = Database::new();
/// let schema = Schema::new(vec![
///     Field::new("ts", ColumnType::I64, false),
///     Field::new("x", ColumnType::F64, false),
/// ]);
/// db.create_table("trades", schema, "ts").unwrap();
/// db.append("trades", &[RowValue::I64(1), RowValue::F64(0.5)]).unwrap();
/// let output = db.query("SELECT x FROM trades").unwrap();
/// assert_eq!(output.num_rows(), 1);
/// ```
#[derive(Default)]
pub struct Database {
    tables: HashMap<String, Table>,
    /// Maintained views (#83), by name. A view's name shares the
    /// table namespace — `query` routes to either.
    views: HashMap<String, MaterializedView>,
    /// The `log(...)` destination for driver scripts (`run_script`);
    /// kernels use their table's sink.
    #[cfg(feature = "lua")]
    script_log_sink: Option<std::sync::Arc<dyn compute_lua::LogSink + Sync>>,
}

impl Database {
    /// An empty database.
    pub fn new() -> Database {
        Database::default()
    }

    /// Creates a table (see [`Table::new`]); the name must be unused.
    pub fn create_table(
        &mut self,
        name: &str,
        schema: Schema,
        ordering_key: &str,
    ) -> Result<(), EngineError> {
        self.claim_name(name)?;
        let table = Table::new(name, schema, ordering_key)?;
        self.tables.insert(name.to_owned(), table);
        Ok(())
    }

    /// Adds an already-built table (for embedders that configured it —
    /// segment thresholds, for instance); the name must be unused.
    pub fn add_table(&mut self, table: Table) -> Result<(), EngineError> {
        self.claim_name(table.name())?;
        self.tables.insert(table.name().to_owned(), table);
        Ok(())
    }

    /// Creates an in-memory maintained view (#83) over the named
    /// source table — a bucketed, running, or cumulative single-table
    /// aggregate, or a join view (the enriched blotter, aggregates
    /// over the as-of join, or star aggregates over the equi join;
    /// the second table is resolved from the definition's own JOIN
    /// clause) — kept fresh by refresh; see [`MaterializedView`] for
    /// the shapes and what a definition may contain. The name shares
    /// the table namespace.
    pub fn create_materialized_view(&mut self, name: &str, sql: &str) -> Result<(), EngineError> {
        self.claim_name(name)?;
        let lowered = plan(sql)?;
        let source = self
            .tables
            .get(&lowered.table)
            .ok_or_else(|| EngineError::UnknownTable(lowered.table.clone()))?;
        let dimension = match &lowered.join {
            None => None,
            Some(join) => Some(
                self.tables
                    .get(&join.dimension)
                    .ok_or_else(|| EngineError::UnknownTable(join.dimension.clone()))?,
            ),
        };
        let view = MaterializedView::new(name, sql, source, dimension)?;
        self.views.insert(name.to_owned(), view);
        Ok(())
    }

    /// Adds an already-built maintained view (persistent ones — see
    /// [`MaterializedView::persistent`] / [`MaterializedView::open`]);
    /// its source table must already be in the database, and the name
    /// must be unused.
    pub fn add_view(&mut self, view: MaterializedView) -> Result<(), EngineError> {
        self.claim_name(view.name())?;
        if !self.tables.contains_key(view.source()) {
            return Err(EngineError::UnknownTable(view.source().to_owned()));
        }
        if let Some(dimension) = view.dimension() {
            if !self.tables.contains_key(dimension) {
                return Err(EngineError::UnknownTable(dimension.to_owned()));
            }
        }
        self.views.insert(view.name().to_owned(), view);
        Ok(())
    }

    /// The named maintained view, if it exists.
    pub fn view(&self, name: &str) -> Option<&MaterializedView> {
        self.views.get(name)
    }

    /// The maintained views' names, in arbitrary order.
    pub fn view_names(&self) -> Vec<String> {
        self.views.keys().cloned().collect()
    }

    /// Refreshes the named maintained view — folds everything its
    /// stamp does not cover and advances the stamp (see
    /// [`MaterializedView::refresh`]). Returns the number of buckets
    /// re-folded (`u64::MAX` for the rebuild floor). When to call it
    /// is the embedder's choice: TallyDB is
    /// a library and owns no clock, so refresh cadence — after a batch,
    /// on a timer, before a read — belongs to the application.
    pub fn refresh_view(&mut self, name: &str) -> Result<u64, EngineError> {
        let view = self
            .views
            .get_mut(name)
            .ok_or_else(|| EngineError::UnknownTable(name.to_owned()))?;
        match view.dimension().map(str::to_owned) {
            None => {
                let source = self
                    .tables
                    .get_mut(view.source())
                    .ok_or_else(|| EngineError::UnknownTable(view.source().to_owned()))?;
                view.refresh(source, None)
            }
            Some(dimension) => {
                // Two disjoint mutable borrows from one map — the
                // definition door refuses a self-join, so the names
                // always differ.
                let source_name = view.source().to_owned();
                let [source, dimension] = self
                    .tables
                    .get_disjoint_mut([source_name.as_str(), dimension.as_str()]);
                let source =
                    source.ok_or_else(|| EngineError::UnknownTable(source_name.clone()))?;
                let dimension = dimension.ok_or_else(|| {
                    EngineError::UnknownTable(
                        view.dimension().expect("matched Some above").to_owned(),
                    )
                })?;
                view.refresh(source, Some(dimension))
            }
        }
    }

    /// Refreshes every maintained view, in arbitrary order — the
    /// batch-boundary call an embedder makes after landing a batch.
    pub fn refresh_views(&mut self) -> Result<(), EngineError> {
        let names = self.view_names();
        for name in names {
            self.refresh_view(&name)?;
        }
        Ok(())
    }

    /// One namespace across tables and views: a name may be claimed by
    /// at most one of them, or `query` routing would be ambiguous.
    fn claim_name(&self, name: &str) -> Result<(), EngineError> {
        if self.tables.contains_key(name) || self.views.contains_key(name) {
            return Err(EngineError::DuplicateTable(name.to_owned()));
        }
        Ok(())
    }

    /// The named table, if it exists.
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables.get(name)
    }

    /// A reader handle for `table` — see [`Table::reader`]: reader
    /// threads mint point-in-time snapshots from it while this database
    /// handle keeps writing.
    pub fn reader(&self, table: &str) -> Result<crate::TableReader, EngineError> {
        self.table(table)
            .map(Table::reader)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))
    }

    /// The open tables' names, in arbitrary order.
    pub fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    /// The named table, mutably (for appends and registration through
    /// the table handle).
    pub fn table_mut(&mut self, name: &str) -> Option<&mut Table> {
        self.tables.get_mut(name)
    }

    /// Appends one row to the named table; returns its internal row id.
    pub fn append(&mut self, table: &str, row: &[RowValue<'_>]) -> Result<u64, EngineError> {
        if self.views.contains_key(table) {
            return Err(derived_refusal(table, "append to"));
        }
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .append(row)
    }

    /// Runs one SQL query against the table(s) it names — including
    /// star-schema joins, which resolve their dimension table here.
    ///
    /// A query naming a maintained view answers **exactly**, however
    /// stale the materialization: clean materialized buckets unioned
    /// with a live fold of everything the view's stamp does not cover
    /// (the union read; see [`MaterializedView`]). `AS OF` on a view
    /// recomputes the definition over the source as of that cut — the
    /// materialization accelerates current reads, it is never the
    /// authority. (A join view refuses `AS OF`: one coordinate cannot
    /// span two sequence spaces; the two-cut form is #99.)
    pub fn query(&self, sql: &str) -> Result<QueryOutput, EngineError> {
        let plan = plan(sql)?;
        if let Some(join) = &plan.join {
            if self.views.contains_key(&plan.table) || self.views.contains_key(&join.dimension) {
                return Err(EngineError::Query(QueryError::Unsupported(
                    "a maintained view as a join OPERAND — query the view \
                     alone, or join the base tables. (Views OVER joins are \
                     built: define the join inside the view instead.)"
                        .to_owned(),
                )));
            }
        }
        let Some(table) = self.tables.get(&plan.table) else {
            if let Some(view) = self.views.get(&plan.table) {
                let source = self
                    .tables
                    .get(view.source())
                    .ok_or_else(|| EngineError::UnknownTable(view.source().to_owned()))?;
                let dimension = match view.dimension() {
                    None => None,
                    Some(name) => Some(
                        self.tables
                            .get(name)
                            .ok_or_else(|| EngineError::UnknownTable(name.to_owned()))?,
                    ),
                };
                return view.query_union(source, dimension, &plan);
            }
            return Err(EngineError::UnknownTable(plan.table.clone()));
        };
        match &plan.join {
            None => table.execute_plan(&plan),
            Some(join) => {
                let dimension = self
                    .tables
                    .get(&join.dimension)
                    .ok_or_else(|| EngineError::UnknownTable(join.dimension.clone()))?;
                table.execute_join_plan(&plan, dimension)
            }
        }
    }

    /// As [`Database::query`], exported as an `ArrowArrayStream`.
    pub fn query_stream(&self, sql: &str) -> Result<ArrowArrayStream, EngineError> {
        let QueryOutput { schema, batches } = self.query(sql)?;
        Ok(arrow_lite::export_stream(schema, batches.into_iter()))
    }

    /// Runs one SQL mutation (`UPDATE` / `DELETE`) against the table it
    /// names; returns the rows affected.
    pub fn mutate(&mut self, sql: &str) -> Result<u64, EngineError> {
        let table = match parse_statement(sql)? {
            Statement::Update(update) => update.table,
            Statement::Delete(delete) => delete.table,
            Statement::Insert(insert) => insert.table,
            Statement::CreateTable(_) => {
                return Err(EngineError::Query(QueryError::Unsupported(
                    "CREATE TABLE makes a table, it doesn't mutate one — \
                     build it with schema_from_create + a Table constructor \
                     and add_table (the console does exactly this)"
                        .to_owned(),
                )))
            }
            Statement::Select(_) => {
                return Err(EngineError::Query(QueryError::Unsupported(
                    "SELECT runs through query, not mutate".to_owned(),
                )))
            }
        };
        if self.views.contains_key(&table) {
            return Err(derived_refusal(&table, "mutate"));
        }
        self.tables
            .get_mut(&table)
            .ok_or_else(|| EngineError::UnknownTable(table.clone()))?
            .mutate(sql)
    }

    /// Registers a native window kernel on the named table — the
    /// primary extension path (see [`Table::register_window`] for the
    /// full contract and the ~20-line example).
    pub fn register_window(
        &mut self,
        table: &str,
        name: &str,
        kernel: impl query_lite::WindowAggregate + 'static,
    ) -> Result<(), EngineError> {
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .register_window(name, kernel)
    }

    /// Registers a Lua kernel as a SQL window function on the named
    /// table (see [`Table::register_lua_window`]).
    #[cfg(feature = "lua")]
    pub fn register_lua_window(
        &mut self,
        table: &str,
        name: &str,
        parameters: &[&str],
        chunk: &str,
        output: arrow_lite::ColumnType,
    ) -> Result<(), EngineError> {
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .register_lua_window(name, parameters, chunk, output)
    }

    /// Registers a Lua column kernel as a SQL scalar function on the
    /// named table (see [`Table::register_lua_scalar`]).
    #[cfg(feature = "lua")]
    pub fn register_lua_scalar(
        &mut self,
        table: &str,
        name: &str,
        parameters: &[&str],
        chunk: &str,
    ) -> Result<(), EngineError> {
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .register_lua_scalar(name, parameters, chunk)
    }

    /// Runs `source` as a **driver script** — SQL-in-Lua (#70): the
    /// script's `query(sql)` and `append(table, row)` globals reach
    /// this database, so it can issue SQL, receive result columns as
    /// views (a one-batch result passes through untouched; several
    /// batches pay one gather, proportional to the result), and feed
    /// derived rows back exactly. Each call runs in a fresh
    /// interpreter: no state crosses between scripts.
    ///
    /// Every result a script queries stays live until the script
    /// returns — that is what keeps its views valid, and it means a
    /// driver looping one `query` per row holds them all. Write
    /// drivers that query in bulk and compute over columns.
    ///
    /// See the `driver` module docs for what each statement kind means
    /// here.
    #[cfg(feature = "lua")]
    pub fn run_script(&mut self, source: &str) -> Result<(), EngineError> {
        let mut state = compute_lua::LuaState::new().map_err(EngineError::Script)?;
        if let Some(sink) = &self.script_log_sink {
            state.set_log_sink(Box::new(crate::script::SharedSink(std::sync::Arc::clone(
                sink,
            ))));
        }
        let chunk = state.compile(source).map_err(EngineError::Script)?;
        let mut host = crate::driver::DatabaseHost { database: self };
        state
            .run_driver(&chunk, &mut host)
            .map_err(EngineError::Script)
    }

    /// Installs the destination for driver scripts' `log(...)` output
    /// (see [`Database::run_script`]); off (a no-op) until set.
    #[cfg(feature = "lua")]
    pub fn set_script_log_sink(&mut self, sink: std::sync::Arc<dyn compute_lua::LogSink + Sync>) {
        self.script_log_sink = Some(sink);
    }

    /// Compacts the named table or maintained view (see
    /// [`Table::compact`]; a view compacts its materialization, which
    /// accumulates one small segment per refresh).
    pub fn compact(&mut self, table: &str) -> Result<(), EngineError> {
        if let Some(view) = self.views.get_mut(table) {
            return view.compact();
        }
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .compact()
    }
}

/// The refusal every write path gives a maintained view: derived data
/// is corrected by correcting its source.
fn derived_refusal(name: &str, verb: &str) -> EngineError {
    EngineError::Query(QueryError::Unsupported(format!(
        "{verb} '{name}' — it is a maintained view, and a view is derived: \
         correct the base table and the view follows"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_lite::{ColumnType, Field};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, false),
        ])
    }

    #[test]
    fn tables_are_independent_and_routed_by_name() {
        let mut db = Database::new();
        db.create_table("trades", schema(), "ts").unwrap();
        db.create_table("quotes", schema(), "ts").unwrap();
        db.append("trades", &[RowValue::I64(1), RowValue::F64(1.0)])
            .unwrap();
        db.append("quotes", &[RowValue::I64(1), RowValue::F64(10.0)])
            .unwrap();
        db.append("quotes", &[RowValue::I64(2), RowValue::F64(20.0)])
            .unwrap();
        assert_eq!(db.query("SELECT x FROM trades").unwrap().num_rows(), 1);
        assert_eq!(db.query("SELECT x FROM quotes").unwrap().num_rows(), 2);
        // Row ids are per-table sequences.
        let id = db
            .append("trades", &[RowValue::I64(2), RowValue::F64(2.0)])
            .unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn database_errors_are_specific() {
        let mut db = Database::new();
        db.create_table("trades", schema(), "ts").unwrap();
        assert!(matches!(
            db.create_table("trades", schema(), "ts"),
            Err(EngineError::DuplicateTable(_))
        ));
        assert!(matches!(
            db.query("SELECT x FROM nope"),
            Err(EngineError::UnknownTable(_))
        ));
        assert!(matches!(
            db.append("nope", &[RowValue::I64(1), RowValue::F64(0.0)]),
            Err(EngineError::UnknownTable(_))
        ));
    }

    #[test]
    fn add_table_takes_a_configured_table() {
        let mut db = Database::new();
        let table = Table::with_segment_rows("t", schema(), "ts", 2).unwrap();
        db.add_table(table).unwrap();
        for i in 0..5i64 {
            db.append("t", &[RowValue::I64(i), RowValue::F64(i as f64)])
                .unwrap();
        }
        // The configured threshold survives: 5 rows over 2-row segments
        // means a multi-batch result.
        assert_eq!(db.query("SELECT x FROM t").unwrap().batches.len(), 3);
    }
}

#[cfg(test)]
mod join_tests {
    use super::*;
    use arrow_lite::{Column, ColumnType, Field, NumericData};

    fn fact_schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ])
    }

    fn dimension_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("sector", ColumnType::Key, false),
            Field::new("weight", ColumnType::F64, false),
        ])
    }

    /// Fact rows over four symbols; the dimension knows only three of
    /// them (D is the miss), split across segments so dictionary codes
    /// differ per segment on both sides.
    fn database() -> Database {
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", fact_schema(), "ts", 3).unwrap())
            .unwrap();
        db.add_table(Table::with_segment_rows("symbols", dimension_schema(), "id", 2).unwrap())
            .unwrap();
        for (i, sym) in ["A", "B", "C", "D", "B", "A", "D", "C"].iter().enumerate() {
            db.append(
                "trades",
                &[
                    RowValue::I64(i as i64),
                    RowValue::Key(sym),
                    RowValue::F64(i as f64),
                ],
            )
            .unwrap();
        }
        for (i, (sym, sector, weight)) in
            [("C", "tech", 0.5), ("A", "energy", 1.5), ("B", "tech", 2.5)]
                .iter()
                .enumerate()
        {
            db.append(
                "symbols",
                &[
                    RowValue::I64(i as i64),
                    RowValue::Key(sym),
                    RowValue::Key(sector),
                    RowValue::F64(*weight),
                ],
            )
            .unwrap();
        }
        db
    }

    fn f64s(output: &QueryOutput, index: usize) -> Vec<Option<f64>> {
        output
            .batches
            .iter()
            .flat_map(|batch| {
                let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
                    panic!("expected f64")
                };
                (0..column.len())
                    .map(|row| {
                        column
                            .is_valid(row)
                            .then(|| column.values().as_slice()[row])
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn inner_join_looks_up_and_drops_misses() {
        let db = database();
        let output = db
            .query(
                "SELECT ts, sector, weight FROM trades JOIN symbols \
                 ON trades.sym = symbols.sym ORDER BY ts",
            )
            .unwrap();
        // Rows with sym D (ts 3 and 6) drop; six survive.
        assert_eq!(output.num_rows(), 6);
        assert_eq!(
            f64s(&output, 2),
            [1.5, 2.5, 0.5, 2.5, 1.5, 0.5].map(Some).to_vec()
        );
        // The joined sector renders correctly across per-segment codes.
        let Column::Key(sector) = &output.batches[0].columns()[1] else {
            panic!("sector type")
        };
        assert_eq!(sector.value_at(0), Some("energy"));
    }

    #[test]
    fn the_sequence_pseudocolumn_survives_the_join() {
        // Joining widens rows without adding, dropping or reordering
        // them, so a fact row's birth coordinate is the same one it had
        // unjoined. If the joined segment lost its sequences, `_seq`
        // would quietly become the row's position instead.
        let db = database();
        let seqs = |sql: &str| -> Vec<i64> {
            let output = db.query(sql).unwrap();
            output
                .batches
                .iter()
                .flat_map(|batch| {
                    let Column::Numeric(NumericData::I64(column)) = &batch.columns()[0] else {
                        panic!("_seq is i64")
                    };
                    column.values().as_slice().to_vec()
                })
                .collect()
        };
        // INNER drops the two D rows (ingest coordinates 3 and 6).
        assert_eq!(
            seqs(
                "SELECT _seq, ts FROM trades JOIN symbols \
                 ON trades.sym = symbols.sym ORDER BY ts"
            ),
            [0, 1, 2, 4, 5, 7]
        );
        // LEFT keeps them, still carrying their own coordinates.
        assert_eq!(
            seqs(
                "SELECT _seq, ts FROM trades LEFT JOIN symbols \
                 ON trades.sym = symbols.sym ORDER BY ts"
            ),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn hidden_order_by_reaches_a_dimension_column_through_the_pushdown() {
        // ORDER BY on an unprojected dimension attribute: the used-set
        // walker must count the sort column, or the pushdown would
        // drop exactly the column the hidden sort then needs.
        let db = database();
        let output = db
            .query(
                "SELECT ts FROM trades JOIN symbols \
                 ON trades.sym = symbols.sym ORDER BY weight DESC LIMIT 2",
            )
            .unwrap();
        assert_eq!(output.schema.fields().len(), 1);
        let Column::Numeric(NumericData::I64(ts)) = &output.batches[0].columns()[0] else {
            panic!("ts type")
        };
        // weight 2.5 is B: fact rows ts 1 and 4, in input order.
        assert_eq!(ts.values().as_slice(), &[1, 4]);
    }

    #[test]
    fn join_against_empty_dimension_does_not_panic() {
        // B2 regression: an empty dimension has no views to sniff a column
        // type from. The gather must take each column's type from the
        // dimension *schema*, not default to f64 and mismatch the joined
        // schema — which panicked `RecordBatch::new` for a key/i64 column.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", fact_schema(), "ts", 3).unwrap())
            .unwrap();
        // Created, never appended — an empty dimension (zero snapshot views).
        db.add_table(Table::with_segment_rows("symbols", dimension_schema(), "id", 2).unwrap())
            .unwrap();
        for (i, sym) in ["A", "B", "C"].iter().enumerate() {
            db.append(
                "trades",
                &[
                    RowValue::I64(i as i64),
                    RowValue::Key(sym),
                    RowValue::F64(i as f64),
                ],
            )
            .unwrap();
        }
        // INNER: every fact row misses → zero rows, but the gather still
        // runs (before the live mask), so this is the panic path.
        let inner = db
            .query("SELECT ts, sector, weight FROM trades JOIN symbols ON trades.sym = symbols.sym")
            .unwrap();
        assert_eq!(inner.num_rows(), 0);
        // LEFT: all fact rows kept with null dimension cells; the joined
        // `sector` must come back a Key column, not a defaulted f64.
        let left = db
            .query(
                "SELECT ts, sector, weight FROM trades LEFT JOIN symbols \
                 ON trades.sym = symbols.sym ORDER BY ts",
            )
            .unwrap();
        assert_eq!(left.num_rows(), 3);
        let Column::Key(sector) = &left.batches[0].columns()[1] else {
            panic!("sector must stay a key column even against an empty dimension");
        };
        assert!((0..sector.len()).all(|row| !sector.is_valid(row)));
        assert!(f64s(&left, 2).iter().all(Option::is_none));
    }

    #[test]
    fn left_join_keeps_misses_with_null_dimension_cells() {
        let db = database();
        let output = db
            .query(
                "SELECT ts, weight FROM trades LEFT JOIN symbols \
                 ON trades.sym = symbols.sym ORDER BY ts",
            )
            .unwrap();
        assert_eq!(output.num_rows(), 8);
        let weights = f64s(&output, 1);
        assert_eq!(weights[3], None); // sym D
        assert_eq!(weights[6], None);
        assert_eq!(weights[0], Some(1.5));
    }

    #[test]
    fn joined_tables_run_the_whole_query_surface() {
        let db = database();
        // WHERE on a dimension attribute, GROUP BY it, aggregate a fact
        // column — the star-schema query shape.
        // Groups come back in no particular order — a symbol column
        // cannot be ordered by (#58) — so they are read by label.
        let output = db
            .query(
                "SELECT sector, count(*) AS n, sum(x) AS s FROM trades \
                 JOIN symbols ON trades.sym = symbols.sym \
                 WHERE weight > 1 GROUP BY sector",
            )
            .unwrap();
        let batch = &output.batches[0];
        let Column::Key(sector) = &batch.columns()[0] else {
            panic!("sector type")
        };
        let Column::Numeric(NumericData::I64(n)) = &batch.columns()[1] else {
            panic!("count type")
        };
        let sums = f64s(&output, 2);
        let mut groups: Vec<(&str, i64, Option<f64>)> = (0..batch.num_rows())
            .map(|row| {
                (
                    sector.value_at(row).expect("no null sector"),
                    n.values().as_slice()[row],
                    sums[row],
                )
            })
            .collect();
        groups.sort_by_key(|group| group.0);
        // energy is A (ts 0, 5); tech is B only, weight 2.5 (ts 1, 4).
        assert_eq!(groups, [("energy", 2, Some(5.0)), ("tech", 2, Some(5.0))]);
        // Windows run over the joined intermediate too.
        let output = db
            .query(
                "SELECT ts, sum(weight) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING \
                 AND CURRENT ROW) AS running FROM trades JOIN symbols \
                 ON trades.sym = symbols.sym",
            )
            .unwrap();
        assert_eq!(
            f64s(&output, 1).last().copied().flatten(),
            Some(1.5 + 2.5 + 0.5 + 2.5 + 1.5 + 0.5)
        );
    }

    #[test]
    fn join_errors_are_specific() {
        let mut db = database();
        // Unknown dimension table.
        assert!(matches!(
            db.query("SELECT ts FROM trades JOIN nope ON trades.sym = nope.sym"),
            Err(EngineError::UnknownTable(_))
        ));
        // Non-key join column.
        let error = db
            .query("SELECT ts FROM trades JOIN symbols ON trades.x = symbols.sym")
            .unwrap_err()
            .to_string();
        assert!(error.contains("key column"), "{error}");
        // Column collision (both tables have plain 'id'? fact has none —
        // fabricate via colliding attribute): x exists in fact; give the
        // dimension an x by joining trades to itself conceptually —
        // instead check the duplicate-dimension-key error.
        db.append(
            "symbols",
            &[
                RowValue::I64(9),
                RowValue::Key("A"), // duplicate dimension key
                RowValue::Key("tech"),
                RowValue::F64(9.0),
            ],
        )
        .unwrap();
        let error = db
            .query("SELECT ts FROM trades JOIN symbols ON trades.sym = symbols.sym")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not unique"), "{error}");
        // Joins through a bare table handle are refused.
        let table = Table::new("t", fact_schema(), "ts").unwrap();
        let error = table
            .query("SELECT ts FROM t JOIN u ON t.sym = u.sym")
            .unwrap_err()
            .to_string();
        assert!(error.contains("multi-table"), "{error}");
    }

    /// Trades against a quote history: the as-of fixture.
    ///
    /// The quote table's time column is `qts`, not `ts`, because a
    /// dimension attribute sharing a fact column's name is refused
    /// (see `an_asof_join_still_refuses_a_clashing_attribute_name`).
    ///
    /// Deliberate shapes, each of which some plausible wrong
    /// implementation gets wrong: a trade before its symbol's first
    /// quote (nothing to match), two quotes on the same timestamp (the
    /// tie), a trade exactly on a quote (at-or-before versus strictly
    /// before), a symbol with no quotes at all, and quotes for a symbol
    /// that never trades. Segments hold two rows apiece, so both sides
    /// span several with per-segment dictionaries.
    fn asof_database() -> Database {
        let mut db = Database::new();
        db.add_table(
            Table::with_segment_rows("trades", fact_schema(), "ts", 2).expect("fact schema"),
        )
        .unwrap();
        let quote_schema = Schema::new(vec![
            Field::new("qts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("bid", ColumnType::F64, false),
        ]);
        db.add_table(
            Table::with_segment_rows("quotes", quote_schema, "qts", 2).expect("quote schema"),
        )
        .unwrap();
        for (qts, sym, bid) in [
            (10, "A", 1.0),
            (20, "A", 2.0),
            (20, "A", 3.0), // same timestamp: the later row is the match
            (30, "B", 9.0),
            (40, "A", 4.0),
            (40, "Z", 0.0), // a symbol that never trades
        ] {
            db.append(
                "quotes",
                &[RowValue::I64(qts), RowValue::Key(sym), RowValue::F64(bid)],
            )
            .unwrap();
        }
        for (ts, sym) in [
            (5, "A"),  // before A's first quote
            (10, "A"), // exactly on one
            (20, "A"), // exactly on the tied pair
            (25, "A"),
            (30, "C"), // a symbol with no quotes at all
            (35, "B"),
            (50, "A"),
        ] {
            db.append(
                "trades",
                &[
                    RowValue::I64(ts),
                    RowValue::Key(sym),
                    RowValue::F64(ts as f64),
                ],
            )
            .unwrap();
        }
        db
    }

    #[test]
    fn an_asof_join_takes_each_symbols_latest_quote_at_or_before_the_trade() {
        let db = asof_database();
        let bids = |sql: &str| f64s(&db.query(sql).unwrap(), 1);
        // LEFT keeps the unmatched trades, with a null bid: the trade
        // before A's first quote, and the symbol with no quotes.
        assert_eq!(
            bids(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym ORDER BY ts"
            ),
            [
                None,
                Some(1.0),
                Some(3.0),
                Some(3.0),
                None,
                Some(9.0),
                Some(4.0)
            ]
        );
        // INNER drops exactly those two.
        assert_eq!(
            bids(
                "SELECT ts, bid FROM trades ASOF INNER JOIN quotes \
                 ON trades.sym = quotes.sym ORDER BY ts"
            ),
            [Some(1.0), Some(3.0), Some(3.0), Some(9.0), Some(4.0)]
        );
        // The explicit inequality only chooses whether a quote landing
        // exactly on the trade counts. Strictly before: the ts=10 trade
        // loses A's first quote, and ts=20 falls back past the tie.
        assert_eq!(
            bids(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym AND quotes.qts < trades.ts ORDER BY ts"
            ),
            [None, None, Some(1.0), Some(3.0), None, Some(9.0), Some(4.0)]
        );
        // …and written the other way round it says the same thing.
        assert_eq!(
            bids(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym AND trades.ts >= quotes.qts ORDER BY ts"
            ),
            bids(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym ORDER BY ts"
            ),
        );
    }

    #[test]
    fn asof_tie_winners_follow_knowledge_and_survive_compaction() {
        // #83 tranche 3, F8: among quotes sharing a timestamp the match
        // is the latest-KNOWN version — so correcting the LOSER of a
        // tie makes its corrected form win (its rebirth is the newest
        // knowledge), and compaction must never change the answer. The
        // discriminating storage-vs-sequence case lives in query-lite
        // (segments built with the orders disagreeing); this is the
        // end-to-end belt over real ingest, correction, and compaction.
        let mut db = asof_database();
        let winner = |db: &Database| {
            f64s(
                &db.query(
                    "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                     ON trades.sym = quotes.sym WHERE ts = 20",
                )
                .unwrap(),
                1,
            )
        };
        // The fixture's tie at qts = 20: bids 2.0 then 3.0 — the
        // later-ingested 3.0 is the match.
        assert_eq!(winner(&db), [Some(3.0)]);
        // Correct the tie's LOSER: its rebirth is now the newest
        // knowledge at that timestamp, so it takes the match.
        db.mutate("UPDATE quotes SET bid = 7.0 WHERE bid = 2.0")
            .unwrap();
        assert_eq!(winner(&db), [Some(7.0)]);
        db.compact("quotes").unwrap();
        db.compact("trades").unwrap();
        assert_eq!(
            winner(&db),
            [Some(7.0)],
            "compaction changed an as-of tie winner"
        );
    }

    #[test]
    fn an_asof_join_does_not_multiply_rows_the_way_an_equi_join_would() {
        // The rule an as-of join relaxes is the dimension's unique key
        // — a quote table has many rows per symbol, which a plain join
        // refuses outright. What it must NOT relax is the row count:
        // one output row per fact row, still.
        let db = asof_database();
        let plain = db
            .query("SELECT ts FROM trades JOIN quotes ON trades.sym = quotes.sym")
            .unwrap_err()
            .to_string();
        assert!(plain.contains("not unique"), "{plain}");
        let output = db
            .query(
                "SELECT ts FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym",
            )
            .unwrap();
        assert_eq!(output.num_rows(), 7, "one row per trade, as ingested");
    }

    #[test]
    fn an_asof_joins_time_axis_is_the_declared_ordering_key() {
        // The inequality is validated, not obeyed: it may only restate
        // the two tables' declared ordering keys. Naming anything else
        // is a refusal, because obeying it would mean a search where
        // the design promises a walk.
        let db = asof_database();
        let error = db
            .query(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym AND quotes.bid <= trades.ts",
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("declared ordering keys"), "{error}");
        // Backwards is the whole point: asking for the quote *after*
        // each trade is a different question, refused rather than
        // quietly answered in reverse.
        let error = db
            .query(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym AND trades.ts <= quotes.qts",
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("looks backwards"), "{error}");
    }

    #[test]
    fn an_asof_join_still_refuses_a_clashing_attribute_name() {
        // A quote table whose time column is also called `ts` — the
        // natural schema — collides with the fact's, and the join
        // refuses it exactly as a plain join does. Recorded here
        // because it is the shape a desk reaches for first, and the
        // refusal is the current answer, not an oversight.
        let mut db = Database::new();
        db.add_table(Table::new("trades", fact_schema(), "ts").unwrap())
            .unwrap();
        let clashing = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("bid", ColumnType::F64, false),
        ]);
        db.add_table(Table::new("quotes", clashing, "ts").unwrap())
            .unwrap();
        db.append(
            "trades",
            &[RowValue::I64(1), RowValue::Key("A"), RowValue::F64(1.0)],
        )
        .unwrap();
        db.append(
            "quotes",
            &[RowValue::I64(1), RowValue::Key("A"), RowValue::F64(2.0)],
        )
        .unwrap();
        let error = db
            .query(
                "SELECT ts, bid FROM trades ASOF LEFT JOIN quotes \
                 ON trades.sym = quotes.sym",
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("exists in both tables"), "{error}");
    }
}
