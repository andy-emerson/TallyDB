//! Maintained views (#83, tranche 1): bucketed single-table aggregates
//! kept fresh as ordered data arrives.
//!
//! ## The model, in one paragraph
//!
//! A maintained view is a **fold over the ingest sequence**: a real
//! table (segments, WAL, `AS OF` — all inherited) holding the result of
//! a bucketed aggregate query, plus a **stamp** — the source table's
//! ingest-sequence watermark below which the materialization is
//! complete. Everything at or above the stamp is the view's tail, not
//! yet folded; a refresh (cycle 2) folds it and advances the stamp.
//! Corrections need no bookkeeping of their own: the buckets they touch
//! are **derivable** from the source's knowledge history — buckets
//! touched by any coordinate in `(stamp, now]` — so the only state that
//! must persist correctly is the stamp itself, and a crash anywhere
//! simply leaves it old, which the next refresh heals. Repair is always
//! re-fold-from-base (uniform repair, ruled 2026-08-02 on #83): no
//! accumulator state, no delta arithmetic, no f64 subtraction hazard.
//!
//! ## What tranche 1 admits, and why the line sits there
//!
//! The definition must be a single-table `GROUP BY` over **one bucket
//! of the ordering key** (`ts / 60`, `(ts / 60) * 60`, or bare `ts`),
//! plus any symbol-column keys, with the built aggregates and an
//! optional row-local `WHERE`. That shape is exactly what re-fold
//! repair makes maintainable: every output row belongs to one bucket,
//! so a correction's blast radius is its bucket and repair is the
//! stored query over a restricted range. Shapes outside it are refused
//! **by name** with the tranche that will admit them:
//!
//! - running/cumulative shapes (no bucket) — a correction at `t`
//!   touches every result after `t`; tranche 2's bucket-partials
//!   representation prices that honestly.
//! - joins — tranche 3, q-hierarchical only (the PODS 2017 dichotomy
//!   names exactly which joins can be maintained in O(1)).
//! - `AS OF` / `_seq` in the definition — refused permanently, not
//!   deferred: a view definition must read within one knowledge
//!   snapshot, or `view AS OF s = Q(base AS OF s)` stops being
//!   well-defined (snapshot reducibility).
//!
//! `ORDER BY` / `LIMIT` / `DISTINCT` / `HAVING` are refused because a
//! view is a table: order, limit, and filter at read, where they
//! compose with everything else.

use crate::table::{EngineError, Table};
use arrow_lite::{ColumnType, Field, Schema};
use query_lite::{plan as lower_plan, GroupKey, Plan, Projection, QueryError, SEQUENCE_COLUMN};
use std::path::Path;
use storage_lite::format::crc32c;
use storage_lite::StoreOptions;

/// The definition sidecar's filename inside the view's directory. Its
/// presence is what marks a table directory as a maintained view.
pub const DEFINITION_FILE: &str = "view.def";

/// A maintained view: the materialization table, the definition that
/// fills it, and the stamp saying how much of the source it reflects.
pub struct MaterializedView {
    /// The materialization — a real table whose ordering key is the
    /// view's bucket column.
    table: Table,
    /// The definition, verbatim SQL — the durable form. The lowered
    /// plan is re-derived from it wherever needed (the refresh will
    /// hold one), never persisted.
    sql: String,
    /// The source table's name.
    source: String,
    /// The stamp: the source's ingest-sequence watermark below which
    /// the materialization is complete. `0` = nothing folded yet —
    /// a freshly created view materializes nothing; the first refresh
    /// folds everything below the then-current watermark.
    stamp: u64,
}

impl MaterializedView {
    /// Creates an in-memory maintained view over `source`. The
    /// definition is validated against the source's schema and refused
    /// by name outside tranche 1's shape (see the module doc).
    pub fn new(name: &str, sql: &str, source: &Table) -> Result<MaterializedView, EngineError> {
        let (schema, bucket) = validated_definition(sql, source)?;
        let table = Table::new(name, schema, &bucket)?;
        Ok(MaterializedView {
            table,
            sql: sql.to_owned(),
            source: source.name().to_owned(),
            stamp: 0,
        })
    }

    /// As [`MaterializedView::new`], persisted in `dir`: the
    /// materialization is an ordinary persistent table there, and the
    /// definition and stamp live beside it in [`DEFINITION_FILE`].
    pub fn persistent(
        name: &str,
        sql: &str,
        source: &Table,
        dir: impl AsRef<Path>,
    ) -> Result<MaterializedView, EngineError> {
        let (schema, bucket) = validated_definition(sql, source)?;
        let table = Table::persistent(name, schema, &bucket, dir.as_ref())?;
        let view = MaterializedView {
            table,
            sql: sql.to_owned(),
            source: source.name().to_owned(),
            stamp: 0,
        };
        view.write_definition(dir.as_ref())?;
        Ok(view)
    }

    /// Opens a persisted view: the definition and stamp from
    /// [`DEFINITION_FILE`], the materialization from the table files
    /// beside it. `source` must be the already-open source table — the
    /// definition is re-validated against it, so a source whose schema
    /// no longer fits the view is a loud error at open, not a wrong
    /// answer at read.
    pub fn open(
        name: &str,
        dir: impl AsRef<Path>,
        source: &Table,
        options: StoreOptions,
    ) -> Result<MaterializedView, EngineError> {
        let record = std::fs::read(dir.as_ref().join(DEFINITION_FILE))
            .map_err(|error| definition_error(format!("reading {DEFINITION_FILE}: {error}")))?;
        let (stamp, source_name, sql) = decode_definition(&record)?;
        if source_name != source.name() {
            return Err(definition_error(format!(
                "view '{name}' is over '{source_name}', not '{}'",
                source.name()
            )));
        }
        validated_definition(&sql, source)?;
        let table = Table::open(name, dir.as_ref(), options)?;
        Ok(MaterializedView {
            table,
            sql,
            source: source_name,
            stamp,
        })
    }

    /// The view's name.
    pub fn name(&self) -> &str {
        self.table.name()
    }

    /// The definition, verbatim.
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// The source table's name.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The stamp: the source ingest-sequence watermark below which the
    /// materialization is complete. Everything at or above it is the
    /// view's unfolded tail.
    pub fn stamp(&self) -> u64 {
        self.stamp
    }

    /// The materialization, read-only. What it answers is the view **as
    /// of the stamp**; the always-exact answer is the union read
    /// (cycle 3), which tops this up over the unfolded tail.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Persists the definition record — called at create and after
    /// every stamp advance. The stamp is the one piece of view state
    /// whose durability matters: it only ever advances *after* the
    /// materialization it describes is durable, so a crash between the
    /// two leaves an old stamp and the next refresh re-folds — never a
    /// stamp describing a materialization that does not exist.
    fn write_definition(&self, dir: &Path) -> Result<(), EngineError> {
        let record = encode_definition(self.stamp, &self.source, &self.sql);
        let path = dir.join(DEFINITION_FILE);
        let staging = dir.join(format!("{DEFINITION_FILE}.staging"));
        std::fs::write(&staging, &record)
            .and_then(|()| std::fs::rename(&staging, &path))
            .map_err(|error| definition_error(format!("writing {DEFINITION_FILE}: {error}")))
    }
}

/// Lowers and validates a view definition against its source, returning
/// the view table's schema and its ordering-key (bucket) column name.
fn validated_definition(sql: &str, source: &Table) -> Result<(Schema, String), EngineError> {
    let plan = lower_plan(sql).map_err(EngineError::Query)?;
    if plan.table != source.name() {
        return Err(EngineError::WrongTable {
            expected: source.name().to_owned(),
            got: plan.table,
        });
    }
    let bucket = eligible_shape(&plan, source)?;
    let schema = output_schema(&plan, source)?;
    // The bucket column is the view table's ordering key; the executor
    // may mark aggregate outputs nullable, but a bucket of a NOT NULL
    // ordering key is never null, and Table::new requires NOT NULL.
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if field.name() == bucket {
                Field::new(field.name(), field.column_type(), false)
            } else {
                field.clone()
            }
        })
        .collect();
    Ok((Schema::new(fields), bucket))
}

/// The tranche-1 eligibility check: refuses, by name, every definition
/// shape outside "single-table bucketed aggregate" — naming the tranche
/// that will admit it where one is planned. Returns the bucket term's
/// output column name.
fn eligible_shape(plan: &Plan, source: &Table) -> Result<String, EngineError> {
    let refuse = |what: &str| Err(EngineError::Query(QueryError::Unsupported(what.to_owned())));
    if plan.as_of.is_some() {
        return refuse(
            "ASOF in a view definition — a definition reads one knowledge \
             snapshot, or 'view AS OF s = query(base AS OF s)' stops being \
             well-defined; query the view with ASOF instead",
        );
    }
    if plan.referenced_columns().contains(SEQUENCE_COLUMN) {
        return refuse(
            "'_seq' in a view definition — the ingest sequence is knowledge \
             time, and a definition reads one knowledge snapshot",
        );
    }
    if plan.join.is_some() {
        return refuse(
            "a join in a view definition — maintained joins are tranche 3 \
             of #83 (q-hierarchical only); maintain a view per table and \
             join them at read",
        );
    }
    if plan.distinct {
        return refuse("DISTINCT in a view definition — deduplicate at read");
    }
    if plan.order_by.is_some() || plan.limit.is_some() || plan.offset.is_some() {
        return refuse(
            "ORDER BY / LIMIT / OFFSET in a view definition — a view is a \
             table; order and limit at read, where they compose",
        );
    }
    let Projection::Aggregate {
        keys,
        items,
        having,
    } = &plan.projection
    else {
        return refuse(
            "a row-per-row view — a maintained view maintains aggregates; \
             running and cumulative shapes are tranche 2 of #83",
        );
    };
    if having.is_some() {
        return refuse(
            "HAVING in a view definition — a view stores every group; \
             filter at read",
        );
    }
    let mut bucket_terms = keys.iter().filter(|key| {
        matches!(key, GroupKey::Bucket { .. })
            || matches!(key, GroupKey::Column(column) if column == source.ordering_key())
    });
    let Some(bucket) = bucket_terms.next() else {
        return refuse(
            "a view with no bucket of the ordering key in GROUP BY — \
             without one, a correction's blast radius is unbounded; \
             running shapes are tranche 2 of #83",
        );
    };
    if bucket_terms.next().is_some() {
        return refuse("two buckets of the ordering key in one GROUP BY");
    }
    for key in keys {
        if let GroupKey::Column(column) = key {
            if column != source.ordering_key()
                && source
                    .schema()
                    .fields()
                    .iter()
                    .any(|f| f.name() == column && f.column_type() != ColumnType::Key)
            {
                return refuse(
                    "a non-symbol, non-bucket GROUP BY key in a view \
                     definition — group by symbols and one bucket of the \
                     ordering key",
                );
            }
        }
    }
    // The bucket is the view table's ordering key, so it must be a
    // SELECT output — and its output name is the alias when the query
    // wrote one.
    items
        .iter()
        .find_map(|item| match item {
            query_lite::AggItem::Key { key, alias } if key == bucket => {
                Some(alias.clone().unwrap_or_else(|| key.output_name()))
            }
            _ => None,
        })
        .ok_or_else(|| {
            EngineError::Query(QueryError::Unsupported(
                "a view whose SELECT list omits its bucket — the bucket is \
                 the view's ordering key, so select it (alias it to taste)"
                    .to_owned(),
            ))
        })
}

/// The view table's schema: the definition executed over zero segments
/// — the executor's own output schema, with no rows paid for. This also
/// re-validates every column reference and aggregate against the real
/// source schema at create and open.
fn output_schema(plan: &Plan, source: &Table) -> Result<Schema, EngineError> {
    Ok(source.execute_plan_empty(plan)?.schema)
}

fn definition_error(message: String) -> EngineError {
    EngineError::Query(QueryError::Unsupported(message))
}

/// The definition record: `b"TDBV"`, a format version, the stamp, then
/// length-prefixed source name and SQL, then CRC-32C of everything
/// before it. Little-endian throughout, like the segment format.
fn encode_definition(stamp: u64, source: &str, sql: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 2 + 8 + 8 + source.len() + sql.len() + 4);
    out.extend_from_slice(b"TDBV");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&stamp.to_le_bytes());
    out.extend_from_slice(&(source.len() as u32).to_le_bytes());
    out.extend_from_slice(source.as_bytes());
    out.extend_from_slice(&(sql.len() as u32).to_le_bytes());
    out.extend_from_slice(sql.as_bytes());
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn decode_definition(bytes: &[u8]) -> Result<(u64, String, String), EngineError> {
    let corrupt = |what: &str| definition_error(format!("{DEFINITION_FILE} is corrupt: {what}"));
    if bytes.len() < 4 + 2 + 8 + 4 + 4 + 4 {
        return Err(corrupt("truncated"));
    }
    let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
    let stored = u32::from_le_bytes(crc_bytes.try_into().expect("split at 4"));
    if crc32c(payload) != stored {
        return Err(corrupt("checksum mismatch"));
    }
    if &payload[0..4] != b"TDBV" {
        return Err(corrupt("bad magic"));
    }
    let version = u16::from_le_bytes(payload[4..6].try_into().expect("sized"));
    if version != 1 {
        return Err(corrupt(&format!("unknown version {version}")));
    }
    let stamp = u64::from_le_bytes(payload[6..14].try_into().expect("sized"));
    let mut at = 14usize;
    let mut read_string = |what: &str| -> Result<String, EngineError> {
        let len_end = at.checked_add(4).filter(|&e| e <= payload.len());
        let Some(len_end) = len_end else {
            return Err(corrupt(&format!("truncated {what} length")));
        };
        let len = u32::from_le_bytes(payload[at..len_end].try_into().expect("sized")) as usize;
        let end = len_end.checked_add(len).filter(|&e| e <= payload.len());
        let Some(end) = end else {
            return Err(corrupt(&format!("truncated {what}")));
        };
        at = end;
        String::from_utf8(payload[len_end..end].to_vec())
            .map_err(|_| corrupt(&format!("{what} is not UTF-8")))
    };
    let source = read_string("source name")?;
    let sql = read_string("definition SQL")?;
    Ok((stamp, source, sql))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::tests::{linear_row, m1_schema};
    use crate::Database;

    fn source() -> Table {
        let mut table = Table::new("trades", m1_schema(), "ts").unwrap();
        for i in 0..12 {
            table.append(&linear_row(i)).unwrap();
        }
        table
    }

    const OHLC: &str = "SELECT sym, ts / 4 AS bar, first(x) AS o, max(x) AS h, \
                        min(x) AS l, last(x) AS c FROM trades GROUP BY sym, ts / 4";

    #[test]
    fn a_view_definition_is_validated_and_shapes_its_table() {
        let source = source();
        let view = MaterializedView::new("ohlc", OHLC, &source).unwrap();
        assert_eq!(view.name(), "ohlc");
        assert_eq!(view.source(), "trades");
        assert_eq!(view.sql(), OHLC);
        // Nothing folded yet: the stamp is zero and the
        // materialization empty — create is O(1), the first refresh
        // pays for the backlog.
        assert_eq!(view.stamp(), 0);
        assert_eq!(
            view.table().query("SELECT o FROM ohlc").unwrap().num_rows(),
            0
        );
        // The table's shape came from the executor: the bucket alias
        // is the ordering key, the aggregates are columns.
        let schema = view.table().schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name()).collect();
        assert_eq!(names, ["sym", "bar", "o", "h", "l", "c"]);
        assert_eq!(view.table().ordering_key(), "bar");
        // The bucket start spelling and a bare-ts bucket are accepted
        // too, and an unaliased bucket keeps its arithmetic name.
        MaterializedView::new(
            "bars",
            "SELECT (ts / 4) * 4, sum(x) FROM trades GROUP BY (ts / 4) * 4",
            &source,
        )
        .unwrap();
        MaterializedView::new(
            "instants",
            "SELECT ts, count(*) AS n FROM trades GROUP BY ts",
            &source,
        )
        .unwrap();
    }

    #[test]
    fn ineligible_definitions_are_refused_by_name() {
        let source = source();
        let refused = |sql: &str, needle: &str| {
            let error = MaterializedView::new("v", sql, &source)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{sql}: {error}");
        };
        // The permanent refusals: knowledge time inside a definition.
        refused(
            "SELECT ts / 4 AS b, sum(x) FROM trades ASOF 5 GROUP BY ts / 4",
            "ASOF in a view definition",
        );
        refused(
            "SELECT ts / 4 AS b, sum(_seq) FROM trades GROUP BY ts / 4",
            "'_seq' in a view definition",
        );
        // The deferred refusals, each naming its tranche.
        refused("SELECT x FROM trades", "tranche 2");
        refused(
            "SELECT sym, sum(x) AS s FROM trades GROUP BY sym",
            "no bucket of the ordering key",
        );
        // A view is a table: what composes at read is refused in the
        // definition.
        refused(
            "SELECT ts / 4 AS b, sum(x) AS s FROM trades GROUP BY ts / 4 HAVING sum(x) > 1",
            "HAVING in a view definition",
        );
        refused(
            "SELECT ts / 4 AS b, sum(x) AS s FROM trades GROUP BY ts / 4 ORDER BY s",
            "ORDER BY / LIMIT / OFFSET",
        );
        refused(
            "SELECT ts / 4 AS b, sum(x) AS s FROM trades GROUP BY ts / 4 LIMIT 3",
            "ORDER BY / LIMIT / OFFSET",
        );
        refused(
            "SELECT DISTINCT sym FROM trades",
            "DISTINCT in a view definition",
        );
        // The bucket is the view's ordering key, so it must be output.
        refused(
            "SELECT sum(x) AS s FROM trades GROUP BY ts / 4",
            "SELECT list omits its bucket",
        );
        // Definitions that never planned: a bad column stays a loud
        // planner error, not a view-shaped one.
        let error = MaterializedView::new(
            "v",
            "SELECT ts / 4 AS b, sum(nope) AS s FROM trades GROUP BY ts / 4",
            &source,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("nope"), "{error}");
        // And a definition naming another table meets the table check.
        let error = MaterializedView::new(
            "v",
            "SELECT ts / 4 AS b, sum(x) AS s FROM elsewhere GROUP BY ts / 4",
            &source,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("elsewhere"), "{error}");
    }

    #[test]
    fn a_join_in_a_definition_is_refused() {
        let source = source();
        let error = MaterializedView::new(
            "v",
            "SELECT ts / 4 AS b, sum(w) AS s FROM trades \
             JOIN dim ON trades.sym = dim.sym GROUP BY ts / 4",
            &source,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("tranche 3"), "{error}");
    }

    #[test]
    fn a_persistent_view_reopens_with_its_definition_and_stamp() {
        let dir = std::env::temp_dir().join(format!("tallydb-view-def-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = source();
        {
            let view = MaterializedView::persistent("ohlc", OHLC, &source, &dir).unwrap();
            assert_eq!(view.stamp(), 0);
        }
        let reopened =
            MaterializedView::open("ohlc", &dir, &source, StoreOptions::default()).unwrap();
        assert_eq!(reopened.sql(), OHLC);
        assert_eq!(reopened.source(), "trades");
        assert_eq!(reopened.stamp(), 0);
        // A flipped bit in the record is a loud checksum error, not a
        // silently different definition.
        let path = dir.join(DEFINITION_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let error = MaterializedView::open("ohlc", &dir, &source, StoreOptions::default())
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        // And a view opened against the wrong source is refused by
        // name, not answered wrongly.
        std::fs::write(&path, encode_definition(0, "quotes", OHLC)).unwrap();
        let error = MaterializedView::open("ohlc", &dir, &source, StoreOptions::default())
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("is over 'quotes'"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_database_routes_views_and_refuses_writes_to_them() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..12 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("ohlc", OHLC).unwrap();
        assert_eq!(db.view_names(), ["ohlc"]);
        assert_eq!(db.view("ohlc").unwrap().source(), "trades");
        // Querying the view answers from the materialization — empty
        // until the first refresh, honestly reflecting stamp 0.
        assert_eq!(db.query("SELECT o FROM ohlc").unwrap().num_rows(), 0);
        // One namespace: neither a table nor a second view may take
        // the name, in either direction.
        let error = db.create_table("ohlc", m1_schema(), "ts").unwrap_err();
        assert!(matches!(error, EngineError::DuplicateTable(_)));
        let error = db.create_materialized_view("trades", OHLC).unwrap_err();
        assert!(matches!(error, EngineError::DuplicateTable(_)));
        // Writes to a view are refused with the teaching error.
        let error = db.append("ohlc", &linear_row(99)).unwrap_err().to_string();
        assert!(error.contains("maintained view"), "{error}");
        let error = db.mutate("DELETE FROM ohlc").unwrap_err().to_string();
        assert!(error.contains("maintained view"), "{error}");
        let error = db
            .mutate("UPDATE ohlc SET o = 0 WHERE bar = 1")
            .unwrap_err()
            .to_string();
        assert!(error.contains("maintained view"), "{error}");
        // A view in a join is refused by name on either side.
        db.create_table(
            "dim",
            arrow_lite::Schema::new(vec![
                arrow_lite::Field::new("ts", arrow_lite::ColumnType::I64, false),
                arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                arrow_lite::Field::new("w", arrow_lite::ColumnType::F64, false),
            ]),
            "ts",
        )
        .unwrap();
        let error = db
            .query("SELECT ohlc.o FROM ohlc JOIN dim ON ohlc.sym = dim.sym")
            .unwrap_err()
            .to_string();
        assert!(error.contains("view in a join"), "{error}");
        // A view over a missing source cannot be added.
        let orphan_source = Table::new("orphan", m1_schema(), "ts").unwrap();
        let orphan = MaterializedView::new(
            "v2",
            "SELECT ts, count(*) AS n FROM orphan GROUP BY ts",
            &orphan_source,
        )
        .unwrap();
        assert!(matches!(
            db.add_view(orphan).map(|_| ()).unwrap_err(),
            EngineError::UnknownTable(_)
        ));
    }
}
