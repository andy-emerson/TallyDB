//! The multi-table handle: what an application embeds.
//!
//! A [`Database`] is a set of named [`Table`]s and a SQL doorway that
//! routes each query to the table it names. It adds no storage or
//! execution machinery of its own — each table still owns its store and
//! its registered compute — but it is the shape applications program
//! against (`create_table` / `append` / `query`), and it is where
//! star-schema joins will resolve their dimension tables when they arrive
//! (M2.5).

use crate::table::{EngineError, Table};
use arrow_lite::{ArrowArrayStream, Schema};
use query_lite::{parse_statement, plan, QueryError, QueryOutput, Statement};
use std::collections::HashMap;
use storage_lite::RowValue;

/// A set of named tables behind one SQL doorway.
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
        if self.tables.contains_key(name) {
            return Err(EngineError::DuplicateTable(name.to_owned()));
        }
        let table = Table::new(name, schema, ordering_key)?;
        self.tables.insert(name.to_owned(), table);
        Ok(())
    }

    /// Adds an already-built table (for embedders that configured it —
    /// segment thresholds, for instance); the name must be unused.
    pub fn add_table(&mut self, table: Table) -> Result<(), EngineError> {
        if self.tables.contains_key(table.name()) {
            return Err(EngineError::DuplicateTable(table.name().to_owned()));
        }
        self.tables.insert(table.name().to_owned(), table);
        Ok(())
    }

    /// The named table, if it exists.
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables.get(name)
    }

    /// The named table, mutably (for appends through the table handle).
    pub fn table_mut(&mut self, name: &str) -> Option<&mut Table> {
        self.tables.get_mut(name)
    }

    /// Appends one row to the named table; returns its internal row id.
    pub fn append(&mut self, table: &str, row: &[RowValue<'_>]) -> Result<u64, EngineError> {
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .append(row)
    }

    /// Runs one SQL query against the table(s) it names — including
    /// star-schema joins, which resolve their dimension table here.
    pub fn query(&self, sql: &str) -> Result<QueryOutput, EngineError> {
        let plan = plan(sql)?;
        let table = self
            .tables
            .get(&plan.table)
            .ok_or_else(|| EngineError::UnknownTable(plan.table.clone()))?;
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
            Statement::Select(_) => {
                return Err(EngineError::Query(QueryError::Unsupported(
                    "SELECT runs through query, not mutate".to_owned(),
                )))
            }
        };
        self.tables
            .get_mut(&table)
            .ok_or_else(|| EngineError::UnknownTable(table.clone()))?
            .mutate(sql)
    }

    /// Compacts the named table (see [`Table::compact`]).
    pub fn compact(&mut self, table: &str) -> Result<(), EngineError> {
        self.tables
            .get_mut(table)
            .ok_or_else(|| EngineError::UnknownTable(table.to_owned()))?
            .compact()
    }
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
        let output = db
            .query(
                "SELECT sector, count(*) AS n, sum(x) AS s FROM trades \
                 JOIN symbols ON trades.sym = symbols.sym \
                 WHERE weight > 1 GROUP BY sector ORDER BY sector",
            )
            .unwrap();
        let batch = &output.batches[0];
        let Column::Key(sector) = &batch.columns()[0] else {
            panic!("sector type")
        };
        assert_eq!(sector.value_at(0), Some("energy")); // A: ts 0, 5
        assert_eq!(sector.value_at(1), Some("tech")); // B only (weight 2.5): ts 1, 4
        let Column::Numeric(NumericData::I64(n)) = &batch.columns()[1] else {
            panic!("count type")
        };
        assert_eq!(n.values().as_slice(), &[2, 2]);
        assert_eq!(f64s(&output, 2), [Some(5.0), Some(5.0)]);
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
}
