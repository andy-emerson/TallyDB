//! SQL-in-Lua over a [`Database`]: the engine half of `compute-lua`'s
//! driver seam (#70). [`Database::run_script`] runs a Lua chunk whose
//! `query(sql)` and `append(table, row)` reach this database — the
//! script issues SQL, receives result columns as views, and feeds
//! derived rows back, completing the bidirectional embed (Role 1,
//! Lua-in-SQL, is the kernel slot in `script`).
//!
//! ## What a script's statements mean here
//!
//! - `SELECT` runs through [`Database::query`] and is made contiguous
//!   ([`query_lite::contiguous`]), because a script sees each result
//!   column as one view. A result that already arrived in one batch is
//!   moved through untouched, so its views point straight at the
//!   buffers the query produced; several batches pay one gather, with
//!   key columns re-encoded into a merged dictionary. That copy is
//!   proportional to the result, not bounded by a constant — the price
//!   of contiguity, paid by whoever asks for it. Note that "one batch"
//!   does not imply "stored buffers": `ORDER BY` and friends already
//!   materialize a fresh batch upstream.
//! - `INSERT`/`UPDATE`/`DELETE` run through [`Database::mutate`].
//! - `CREATE TABLE` creates an **in-memory** table on this handle
//!   (scratch space for a pipeline's derived data). Persistence is the
//!   embedder's business — at the console, durable tables are created
//!   at the prompt before the script runs.
//! - `append` maps the row table onto the schema by name — every
//!   column present, `NULL` spelled explicitly — and runs through
//!   [`Database::append`], so a script feeds computed values back
//!   *exactly* (Lua integers are `i64`; no text round trip).
//!
//! [`Database`]: crate::Database
//! [`Database::query`]: crate::Database::query
//! [`Database::mutate`]: crate::Database::mutate
//! [`Database::append`]: crate::Database::append
//! [`Database::run_script`]: crate::Database::run_script

use crate::database::Database;
use arrow_lite::{Column, NumericData, RecordBatch};
use compute_lua::{ColumnView, ResultColumns, ScriptHost, ScriptValue, SqlOutcome};
use query_lite::{parse_statement, Statement};
use storage_lite::RowValue;

/// The [`ScriptHost`] a driving script reaches: statements resolve
/// against this database.
pub(crate) struct DatabaseHost<'a> {
    pub(crate) database: &'a mut Database,
}

impl ScriptHost for DatabaseHost<'_> {
    fn statement(&mut self, sql: &str) -> Result<SqlOutcome, String> {
        let sql = sql.trim();
        let sql = sql.strip_suffix(';').unwrap_or(sql);
        match parse_statement(sql).map_err(|error| error.to_string())? {
            Statement::Select(_) => {
                let output = self
                    .database
                    .query(sql)
                    .map_err(|error| error.to_string())?;
                Ok(SqlOutcome::Rows(Box::new(HeldOutput {
                    batch: query_lite::contiguous(output),
                })))
            }
            Statement::Insert(_) | Statement::Update(_) | Statement::Delete(_) => {
                Ok(SqlOutcome::Affected(
                    self.database
                        .mutate(sql)
                        .map_err(|error| error.to_string())?,
                ))
            }
            Statement::CreateTable(plan) => {
                let (schema, ordering) =
                    crate::table::schema_from_create(&plan).map_err(|error| error.to_string())?;
                self.database
                    .create_table(&plan.table, schema, &ordering)
                    .map_err(|error| error.to_string())?;
                Ok(SqlOutcome::Done)
            }
        }
    }

    fn append(&mut self, table: &str, row: &[(String, ScriptValue)]) -> Result<u64, String> {
        let schema = self
            .database
            .table(table)
            .ok_or_else(|| format!("append: unknown table '{table}'"))?
            .schema()
            .clone();
        for (name, _) in row {
            if !schema.fields().iter().any(|field| field.name() == name) {
                return Err(format!("append into '{table}': no column '{name}'"));
            }
        }
        let mut values = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let Some((_, value)) = row.iter().find(|(name, _)| name == field.name()) else {
                return Err(format!(
                    "append into '{table}': column '{}' missing (pass NULL explicitly)",
                    field.name()
                ));
            };
            // Type fit (and NOT NULL) is storage's validation — loud
            // there, not duplicated here.
            values.push(match value {
                ScriptValue::F64(value) => RowValue::F64(*value),
                ScriptValue::I64(value) => RowValue::I64(*value),
                ScriptValue::Text(text) => RowValue::Key(text),
                ScriptValue::Null => RowValue::Null,
            });
        }
        self.database
            .append(table, &values)
            .map_err(|error| error.to_string())
    }
}

/// A SELECT's buffers, held for the rest of the driving call — always
/// one batch, because a script sees each result column as a single
/// view.
struct HeldOutput {
    batch: RecordBatch,
}

impl ResultColumns for HeldOutput {
    fn rows(&self) -> usize {
        self.batch.num_rows()
    }

    fn columns(&self) -> Vec<(String, ColumnView<'_>)> {
        self.batch
            .schema()
            .fields()
            .iter()
            .zip(self.batch.columns())
            .map(|(field, column)| {
                let view = match column {
                    Column::Numeric(NumericData::F64(numeric)) => ColumnView::F64 {
                        values: numeric.values().as_slice(),
                        validity: numeric.validity(),
                    },
                    Column::Numeric(NumericData::I64(numeric)) => ColumnView::I64 {
                        values: numeric.values().as_slice(),
                        validity: numeric.validity(),
                    },
                    Column::Key(key) => ColumnView::Key {
                        codes: key.codes().as_slice(),
                        validity: key.validity(),
                        dictionary: key.dictionary(),
                    },
                };
                (field.name().to_owned(), view)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! #70's evidence: the end-to-end scripted pipeline (SQL → Lua →
    //! SQL) cross-checked against the same computation staged by hand
    //! through the Arrow surface.

    use super::*;
    use crate::table::Table;
    use crate::Database;
    use arrow_lite::{ColumnType, Field, Schema};
    use query_lite::QueryOutput;

    /// Column `index` of every batch, flattened — the hand-staged
    /// concatenation the script's contiguous result is checked against.
    fn f64s(output: &QueryOutput, index: usize) -> Vec<f64> {
        output
            .batches
            .iter()
            .flat_map(|batch| match &batch.columns()[index] {
                Column::Numeric(NumericData::F64(numeric)) => numeric.values().as_slice().to_vec(),
                _ => panic!("expected f64 column"),
            })
            .collect()
    }

    fn i64s(output: &QueryOutput, index: usize) -> Vec<i64> {
        output
            .batches
            .iter()
            .flat_map(|batch| match &batch.columns()[index] {
                Column::Numeric(NumericData::I64(numeric)) => numeric.values().as_slice().to_vec(),
                _ => panic!("expected i64 column"),
            })
            .collect()
    }

    fn texts(output: &QueryOutput, index: usize) -> Vec<String> {
        output
            .batches
            .iter()
            .flat_map(|batch| match &batch.columns()[index] {
                Column::Key(key) => key
                    .codes()
                    .as_slice()
                    .iter()
                    .map(|&code| key.dictionary().value(code).to_owned())
                    .collect::<Vec<String>>(),
                _ => panic!("expected key column"),
            })
            .collect()
    }

    #[test]
    fn a_script_pipeline_matches_the_hand_staged_computation() {
        // The source spans several 16-row segments, and the key
        // column's per-segment dictionaries genuinely differ ("C"
        // first appears mid-table), so the contiguous result the
        // script sees exercises the merge-and-remap path.
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, false),
        ]);
        let mut table = Table::with_segment_rows("trades", schema, "ts", 16).unwrap();
        for i in 0..60i64 {
            let sym = match (i >= 32, i % 2 == 0) {
                (false, true) => "A",
                (false, false) => "B",
                (true, true) => "C",
                (true, false) => "A",
            };
            table
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key(sym),
                    RowValue::F64(i as f64 * 0.5 + 3.0),
                    RowValue::F64((i as f64).mul_add(-0.25, 40.0)),
                ])
                .unwrap();
        }
        let mut db = Database::new();
        db.add_table(table).unwrap();

        // The pipeline: SQL out, the vectorized vocabulary over the
        // result views, exact feed-back row by row, SQL again — all
        // driven from inside the script.
        db.run_script(
            r#"
                query("CREATE TABLE derived (ts BIGINT ORDERING KEY, sym KEY NOT NULL, \
                       rel DOUBLE, rdot DOUBLE)")
                local r, n = query("SELECT ts, sym, x, y FROM trades")
                assert(n == 60)
                local rel = (r.x - r.y) / r.y
                local rdot = rolling_dot(r.x, r.y, 7)
                for i = 1, n do
                    append("derived", {
                        ts = r.ts[i], sym = r.sym:text(i), rel = rel[i], rdot = rdot[i],
                    })
                end
                local d, m = query("SELECT rel FROM derived")
                assert(m == 60 and d.rel[1] == rel[1])
            "#,
        )
        .unwrap();

        // The hand-staged half: the same columns through the ordinary
        // Arrow surface, the same arithmetic in Rust.
        let source = db.query("SELECT ts, sym, x, y FROM trades").unwrap();
        let (ts, sym) = (i64s(&source, 0), texts(&source, 1));
        let (x, y) = (f64s(&source, 2), f64s(&source, 3));
        let derived = db.query("SELECT ts, sym, rel, rdot FROM derived").unwrap();
        assert_eq!(derived.num_rows(), 60);
        assert_eq!(i64s(&derived, 0), ts);
        assert_eq!(
            texts(&derived, 1),
            sym,
            "key texts survive the merge and feed-back"
        );
        let rel = f64s(&derived, 2);
        for i in 0..60 {
            assert_eq!(
                rel[i].to_bits(),
                ((x[i] - y[i]) / y[i]).to_bits(),
                "row {i}: the script's arithmetic is the native arithmetic"
            );
        }
        let rdot = f64s(&derived, 3);
        for (i, got) in rdot.iter().enumerate() {
            let lo = (i + 1).saturating_sub(7);
            let expected: f64 = (lo..=i).map(|j| x[j] * y[j]).sum();
            let scale = expected.abs().max(1.0);
            assert!(
                ((got - expected) / scale).abs() < 1e-12,
                "row {i}: {got} vs {expected}"
            );
        }

        // Mutations flow through the same doorway, with counts the
        // script can assert on: rel < 0 exactly while x < y (i < 50).
        db.run_script(r#"assert(query("DELETE FROM derived WHERE rel < 0") == 50)"#)
            .unwrap();
        assert_eq!(db.query("SELECT ts FROM derived").unwrap().num_rows(), 10);

        // And a script error is a loud, specific Err — not a hang or
        // a half-applied pipeline.
        let error = db
            .run_script("query(\"SELECT nope FROM trades\")")
            .unwrap_err();
        assert!(error.to_string().contains("nope"), "got: {error}");
    }
}
