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
use query_lite::{plan, QueryOutput};
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

    /// Runs one SQL query against the table it names.
    pub fn query(&self, sql: &str) -> Result<QueryOutput, EngineError> {
        let plan = plan(sql)?;
        let table = self
            .tables
            .get(&plan.table)
            .ok_or_else(|| EngineError::UnknownTable(plan.table.clone()))?;
        table.execute_plan(&plan)
    }

    /// As [`Database::query`], exported as an `ArrowArrayStream`.
    pub fn query_stream(&self, sql: &str) -> Result<ArrowArrayStream, EngineError> {
        let QueryOutput { schema, batches } = self.query(sql)?;
        Ok(arrow_lite::export_stream(schema, batches.into_iter()))
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
