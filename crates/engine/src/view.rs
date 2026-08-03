//! Maintained views (#83): single-table aggregates — bucketed,
//! running, and cumulative — and join views over a fact table and a
//! second source — the enriched blotter, aggregates over the as-of
//! join, and star aggregates over the equi join — kept fresh as
//! ordered data arrives.
//!
//! ## The model, in one paragraph
//!
//! A maintained view is a **fold over the ingest sequence**: a real
//! table (segments, WAL, `AS OF` — all inherited) holding a bucketed
//! materialization of the definition, plus a **stamp** — the source
//! table's ingest-sequence watermark below which the materialization
//! is complete. Everything at or above the stamp is the view's
//! unfolded tail; a refresh folds it and advances the stamp, and a
//! query never waits for one: the **union read** answers exactly
//! however stale the materialization — clean materialized buckets plus
//! a live fold of whatever the stamp does not cover — and a past
//! coordinate answers by recompute (`view AS OF s` is the definition
//! over `base AS OF s`; the materialization is never the authority).
//! Corrections need no bookkeeping of their own: the buckets they
//! touch are **derivable** from the source's knowledge history, so the
//! only durable view state is the stamp (plus, for the partials
//! shapes, the chosen bucket width), written strictly after the
//! materialization it describes is flushed — everything a stamp covers
//! therefore survives any crash the source's own WAL contract admits,
//! and a crash elsewhere just leaves the stamp old, which the next
//! refresh heals. Repair is always re-fold-from-base (uniform repair,
//! ruled 2026-08-02 on #83): no accumulator state, no delta
//! arithmetic, no f64 subtraction hazard.
//!
//! ## The single-table shapes
//!
//! **Bucketed** (tranche 1): a `GROUP BY` over one bucket of the
//! ordering key (`ts / 60`, `(ts / 60) * 60`, or bare `ts`), plus any
//! symbol keys, built aggregates, optional row-local `WHERE`. The
//! materialization IS the answer: every output row belongs to one
//! bucket, so a correction's blast radius is its bucket.
//!
//! **Running** (tranche 2): the same aggregates with NO bucket —
//! per-symbol or global totals. The materialization stores per-bucket
//! **partials** under a hidden bucket of the ordering key (width
//! chosen at the first refresh with data, persisted in the definition
//! record), and the read combines them per group and finalizes (`AVG`
//! divides once, after the combine). A correction still re-folds one
//! hidden bucket; the O(suffix) rewrite never exists because no
//! suffix is stored.
//!
//! **Cumulative** (tranche 2): a row-per-row projection of expanding
//! windows (`sum`/`count`/`avg`/`min`/`max` OVER unbounded-preceding
//! frames, ordered by the ordering key, partitioned by symbols; the
//! ordering key and every partition symbol selected; one PARTITION BY
//! list per view). Same partials materialization; the read splits each
//! window at the query predicate's ordering-key lower bound — a
//! **boundary** combine over the partials strictly below that bucket,
//! an **assembly** of the definition over the source from the bucket's
//! low edge, and a per-column adjustment folding the two (`AVG`
//! through hidden sum/count helper windows, never through its
//! quotient). A query with no lower bound wants every output row, so
//! it recomputes — the partials cannot shorten an O(n)-row answer.
//! Because the assembly and recompute run real windows, they inherit
//! the executor's ordered-data requirement: a read that meets
//! uncompacted correction segments refuses exactly as the base's
//! windows do (`compact` heals both), and `view AS OF s` refuses once
//! corrections sit in history segments — refusal parity, not a gap:
//! `view AS OF s = Q(base AS OF s)`, refusals included.
//!
//! ## Join views (tranche 3)
//!
//! A join view has TWO sources: the **fact** table, whose ordering
//! key is the view's axis, and a **dimension** — a quote history for
//! the as-of shapes, a keyed attribute table for the star shape. The
//! durable state grows a pair: the dimension's own stamp and the
//! **ceiling**, the fact-key bound below which materialization is
//! allowed. The ceiling is the dimension frontier's bucket edge:
//! a fact key at or above it could still change matches when the
//! next in-order quote arrives, so refresh materializes strictly
//! below it and the union read answers the rest live. In-order
//! dimension arrivals therefore never dirty the materialization; a
//! LATE quote below the ceiling dirties exactly its **correction
//! interval** — from its key to the same symbol's next quote —
//! derivable from the dimension's knowledge history like every other
//! correction (the interval lemma; proof in DESIGN's tranche-3
//! record). A frontier regression (a correction deletes the newest
//! quote) lowers the ceiling and dematerializes the stranded band.
//! The star shape needs no ceiling: any dimension change rebuilds
//! the materialization whole (F4, ruled on #83), and the read serves
//! the answer live while a rebuild is pending. As-of ties break by
//! birth sequence (`_seq`), engine-wide (F8). Eligibility (F7): the
//! as-of join is q-hierarchical; the equi join is admitted with the
//! join key unique on the dimension side, checked loudly at
//! execution.
//!
//! ## What stays refused, and why
//!
//! - bare projections over a single table — a view must fold or
//!   match something; the blotter is admitted exactly because it
//!   materializes the match.
//! - `AS OF` over a join view — a join view's answer is a function
//!   of TWO knowledge coordinates, so a single `s` is ambiguous; the
//!   honest two-cut form is seated as #99.
//! - `AS OF` / `_seq` in the definition — refused permanently, not
//!   deferred: a view definition must read within one knowledge
//!   snapshot, or `view AS OF s = Q(base AS OF s)` stops being
//!   well-defined (snapshot reducibility).
//! - other window functions, bounded frames, LAG/LEAD, cross-sectional
//!   partitions — refused by name; rolling windows derive at read.
//! - names beginning with `__` in a running/cumulative definition —
//!   the synthesis mints its hidden columns there.
//!
//! `ORDER BY` / `LIMIT` / `OFFSET` / `DISTINCT` / `HAVING` are refused
//! because a view is a table: order, limit, and filter at read, where
//! they compose with everything else. Prose says "maintained view";
//! the API type is [`MaterializedView`] — one concept, two registers.

use crate::table::{EngineError, Table};
use arrow_lite::{ColumnType, Field, Schema};
use query_lite::{
    plan as lower_plan, CmpOp, GroupKey, Number, Plan, Predicate, Projection, QueryError,
    SEQUENCE_COLUMN,
};
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
    /// plan is re-derived from it wherever needed (each refresh and
    /// union read builds a [`Definition`]), never persisted.
    sql: String,
    /// The source table's name.
    source: String,
    /// The stamp: the source's ingest-sequence watermark below which
    /// the materialization is complete. `0` = nothing folded yet —
    /// a freshly created view materializes nothing; the first refresh
    /// folds everything below the then-current watermark.
    stamp: u64,
    /// A running view's hidden-bucket width in ordering-key units;
    /// `0` = not yet chosen (the first refresh with data chooses it
    /// from the observed key span and persists it). Bucketed views
    /// carry it as 0, unused.
    width: u64,
    /// Where the definition record persists; `None` for in-memory
    /// views, whose stamp lives only as long as they do.
    dir: Option<std::path::PathBuf>,
    /// The answer schema — the shape queries against this view
    /// return. The materialization's own for a bucketed view; the
    /// user definition's output for the partials shapes, whose
    /// materialization stores internal columns (`__p{i}`, `__bucket`)
    /// no query ever answers with.
    answers: Schema,
    /// A join view's second-source state (#83 tranche 3): the
    /// dimension's name, its stamp, and the materialization ceiling.
    /// `None` for single-source views, whose record stays format v2.
    join: Option<JoinState>,
    /// Opened via [`MaterializedView::open_read_only`]: refresh
    /// refuses, the union read serves.
    read_only: bool,
}

/// A join view's durable second-source state, carried beside the
/// fact-side stamp in the v3 definition record.
#[derive(Clone, Debug, PartialEq, Eq)]
struct JoinState {
    /// The dimension (reference) table's name — the quote side of an
    /// as-of blotter.
    dimension: String,
    /// The dimension's ingest-sequence watermark below which its
    /// corrections are folded in — the second half of the pair stamp.
    stamp: u64,
    /// The materialization **ceiling**: the fact ordering-key value
    /// below which rows are materialized (exclusive). Held at the
    /// dimension's ordering-key frontier at the last refresh: a fact
    /// row below it can only be changed by a CORRECTION (which the
    /// dimension's knowledge history reports); a row at or above it
    /// would be re-matched by ordinary in-order dimension appends, so
    /// it stays in the union read's live half — staleness, never
    /// wrongness. `i64::MIN` = nothing materialized yet.
    ceiling: i64,
}

impl MaterializedView {
    /// Creates an in-memory maintained view over `source`. The
    /// definition is validated against the source's schema and refused
    /// by name outside tranche 1's shape (see the module doc).
    pub fn new(
        name: &str,
        sql: &str,
        source: &Table,
        dimension: Option<&Table>,
    ) -> Result<MaterializedView, EngineError> {
        let (schema, bucket, answers, join) = validated_definition(sql, source, dimension)?;
        let table = Table::new(name, schema, &bucket)?;
        Ok(MaterializedView {
            table,
            sql: sql.to_owned(),
            source: source.name().to_owned(),
            stamp: 0,
            width: 0,
            dir: None,
            answers,
            join,
            read_only: false,
        })
    }

    /// As [`MaterializedView::new`], persisted in `dir`: the
    /// materialization is an ordinary persistent table there, and the
    /// definition and stamp live beside it in [`DEFINITION_FILE`].
    pub fn persistent(
        name: &str,
        sql: &str,
        source: &Table,
        dimension: Option<&Table>,
        dir: impl AsRef<Path>,
    ) -> Result<MaterializedView, EngineError> {
        let (schema, bucket, answers, join) = validated_definition(sql, source, dimension)?;
        let table = Table::persistent(name, schema, &bucket, dir.as_ref())?;
        let view = MaterializedView {
            table,
            sql: sql.to_owned(),
            source: source.name().to_owned(),
            stamp: 0,
            width: 0,
            dir: Some(dir.as_ref().to_path_buf()),
            answers,
            join,
            read_only: false,
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
        dimension: Option<&Table>,
        options: StoreOptions,
    ) -> Result<MaterializedView, EngineError> {
        let (stamp, width, source_name, sql, answers, join) =
            read_definition(dir.as_ref(), name, source, dimension)?;
        let table = Table::open(name, dir.as_ref(), options)?;
        Ok(MaterializedView {
            table,
            sql,
            source: source_name,
            stamp,
            width,
            dir: Some(dir.as_ref().to_path_buf()),
            answers,
            join,
            read_only: false,
        })
    }

    /// As [`MaterializedView::open`], **read-only** — the cross-process
    /// reader shape (F4 in DESIGN's M5 roadmap): the
    /// cross-process shape — a console or binding watching a directory
    /// another process maintains. The union read needs no writes, so a
    /// read-only view still answers exactly; what it cannot do is
    /// persist repair, and [`MaterializedView::refresh`] refuses
    /// loudly, like every mutation on a read-only table.
    pub fn open_read_only(
        name: &str,
        dir: impl AsRef<Path>,
        source: &Table,
        dimension: Option<&Table>,
    ) -> Result<MaterializedView, EngineError> {
        let (stamp, width, source_name, sql, answers, join) =
            read_definition(dir.as_ref(), name, source, dimension)?;
        let table = Table::open_read_only(name, dir.as_ref())?;
        Ok(MaterializedView {
            table,
            sql,
            source: source_name,
            stamp,
            width,
            // Read-only: the stamp is never advanced, so nothing is
            // ever written back.
            dir: None,
            answers,
            join,
            read_only: true,
        })
    }

    /// The source-table name a persisted view's definition record
    /// names — what a directory scanner (the console) reads to open
    /// the source before the view, without opening the view first.
    pub fn stored_source(dir: impl AsRef<Path>) -> Result<String, EngineError> {
        let record = std::fs::read(dir.as_ref().join(DEFINITION_FILE))
            .map_err(|error| definition_error(format!("reading {DEFINITION_FILE}: {error}")))?;
        let (_, _, source, _, _) = decode_definition(&record)?;
        Ok(source)
    }

    /// The dimension-table name a persisted JOIN view's record names,
    /// `None` for single-source views — the scanner reads it beside
    /// [`MaterializedView::stored_source`] to open both tables before
    /// the view.
    pub fn stored_dimension(dir: impl AsRef<Path>) -> Result<Option<String>, EngineError> {
        let record = std::fs::read(dir.as_ref().join(DEFINITION_FILE))
            .map_err(|error| definition_error(format!("reading {DEFINITION_FILE}: {error}")))?;
        let (_, _, _, _, join) = decode_definition(&record)?;
        Ok(join.map(|join| join.dimension))
    }

    /// The dimension (reference) table's name, `None` for
    /// single-source views.
    pub fn dimension(&self) -> Option<&str> {
        self.join.as_ref().map(|join| join.dimension.as_str())
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

    /// The view's answer schema — the shape queries against it return.
    /// For a running or cumulative view this is the user definition's
    /// output, NOT the hidden partials materialization (whose internal
    /// columns no query ever answers with).
    pub fn schema(&self) -> &Schema {
        &self.answers
    }

    /// Compacts the materialization: each refresh flushes one small
    /// segment (the durability the stamp asserts), and compaction
    /// merges them back into sorted, contiguous ones — the same
    /// maintenance any table wants, reachable for views through
    /// [`Database::compact`].
    ///
    /// [`Database::compact`]: crate::Database::compact
    pub fn compact(&mut self) -> Result<(), EngineError> {
        self.table.compact()
    }

    /// Folds everything the stamp does not cover — new appends and
    /// corrections alike — and advances the stamp. This is the
    /// maintenance pass, and it is deliberately *one* mechanism for
    /// both kinds of change (uniform repair, ruled 2026-08-02 on #83):
    ///
    /// 1. Derive the touched buckets — the buckets of every row born
    ///    or killed by a coordinate in `(stamp, now]` — from the
    ///    source's knowledge history. A correction that moves a row
    ///    across buckets touches both: the kill names the old value,
    ///    the reinsert's birth the new one.
    /// 2. Run the stored definition over exactly those buckets (the
    ///    range predicate is prunable, so untouched segments are
    ///    skipped by their zone maps).
    /// 3. Supersede those buckets' rows in the materialization — one
    ///    knowledge event where buckets are replaced (#73's rule) —
    ///    and advance the stamp to `now`.
    ///
    /// Cost is proportional to what changed, not to the live table —
    /// plus a scan of compacted correction history, whose kill
    /// coordinates live in the segments rather than the metadata (see
    /// `touched_ordering_keys`; an additive manifest field removes the
    /// scan if it ever measures hot). A
    /// crash anywhere leaves the stamp old (it persists only after the
    /// materialization is durable), and the next refresh re-derives
    /// and re-folds: the view self-heals, which is why the dirty list
    /// needs no durability of its own.
    ///
    /// Returns the number of buckets re-folded (`u64::MAX` for the
    /// rebuild floor, which re-folds everything and counts nothing).
    ///
    /// Takes the source mutably because the first thing a refresh does
    /// is **flush it**: the stamp asserts durability ("the view
    /// reflects everything below this coordinate"), so everything it
    /// covers must survive any crash the source's own WAL contract
    /// admits — a stamp covering buffered rows would leave ghost
    /// buckets when a crash rewinds the source (found by the repo-wide
    /// code review; the ghost test replays it). At the intended
    /// cadence — refresh at the freeze boundary — the buffer is empty
    /// and the flush is free; a mid-buffer refresh pays one early
    /// freeze, the price of durably stamping what it folds.
    /// A JOIN view (#83 tranche 3) refreshes through `dimension`: pass
    /// its reference table, which is flushed and stamped by the same
    /// discipline. A single-source view refuses `Some`, a join view
    /// refuses `None` — both by name.
    pub fn refresh(
        &mut self,
        source: &mut Table,
        dimension: Option<&mut Table>,
    ) -> Result<u64, EngineError> {
        if source.name() != self.source {
            return Err(EngineError::WrongTable {
                expected: self.source.clone(),
                got: source.name().to_owned(),
            });
        }
        if self.read_only {
            return Err(EngineError::Query(QueryError::Unsupported(
                "refresh on a read-only view — repair is the maintaining \
                 process's job; this handle serves exact answers via the \
                 union read and never writes"
                    .to_owned(),
            )));
        }
        match (&self.join, dimension) {
            (Some(join), Some(dimension)) => {
                if dimension.name() != join.dimension {
                    return Err(EngineError::WrongTable {
                        expected: join.dimension.clone(),
                        got: dimension.name().to_owned(),
                    });
                }
                return self.refresh_joined(source, dimension);
            }
            (Some(join), None) => {
                return Err(definition_error(format!(
                    "this view joins '{}': refresh with both tables",
                    join.dimension
                )));
            }
            (None, Some(dimension)) => {
                return Err(definition_error(format!(
                    "'{}' passed to a single-source view's refresh",
                    dimension.name()
                )));
            }
            (None, None) => {}
        }
        source.flush()?;
        let now = source.next_sequence();
        let mut definition = Definition::of(&self.sql, source, None, self.width)?;
        if !matches!(definition.read, ReadShape::Direct) && self.width == 0 {
            // A running view's hidden-bucket width is chosen once, at
            // the first refresh that sees data: the observed key span
            // over a target bucket count. Heuristic, internal, and
            // recorded in the definition record — a later re-widthing
            // is a rebuild, not a format question.
            match source_span(source)? {
                None => {
                    // No rows yet: nothing to fold, nothing to size.
                    self.advance_stamp(now)?;
                    return Ok(0);
                }
                Some((low, high)) => {
                    const TARGET_BUCKETS: i128 = 1024;
                    let span = (high as i128 - low as i128 + 1).max(1);
                    let width = ((span + TARGET_BUCKETS - 1) / TARGET_BUCKETS).max(1) as u64;
                    self.width = width;
                    definition = Definition::of(&self.sql, source, None, width)?;
                    // Persist the width before folding under it: a
                    // crash between the two re-folds under the SAME
                    // width, never a different one.
                    if let Some(dir) = self.dir.clone() {
                        self.write_definition(&dir)?;
                    }
                }
            }
        }
        if now < self.stamp {
            // The source's watermark sits BELOW the stamp: with the
            // flush-then-stamp discipline this cannot come from a
            // crash — only from a foreign or tampered pairing (a
            // source directory swapped under the view, a stamp file
            // hand-edited). Nothing the stamp claims can be trusted,
            // so this is the rebuild floor: every materialized row
            // out, one full fold in.
            let replacement = source.execute_plan(&definition.plan)?;
            self.table.replace_matching(None, &replacement)?;
            self.advance_stamp(now)?;
            return Ok(u64::MAX);
        }
        if now == self.stamp {
            return Ok(0);
        }
        let Some(runs) = definition.touched_runs(source, self.stamp)? else {
            // Coordinates were spent (a DELETE that matched nothing,
            // say) but no row changed: nothing to fold, and the stamp
            // still advances past them.
            self.advance_stamp(now)?;
            return Ok(0);
        };
        let folded = runs
            .iter()
            .map(|&(first, last)| last - first + 1)
            .sum::<i64>() as u64;
        // The re-fold: the definition, restricted to the touched
        // buckets, over the source's latest state.
        let replacement = source.execute_plan(&definition.restricted_to(&runs, source))?;
        // The write half: those buckets' materialized rows out, the
        // re-folded ones in.
        let victims = definition.view_ranges(&runs);
        self.table.replace_matching(Some(&victims), &replacement)?;
        self.advance_stamp(now)?;
        Ok(folded)
    }

    /// Answers `user_plan` (a query naming this view) **exactly**: the
    /// materialization's clean buckets unioned with a live fold of
    /// everything the stamp does not cover — dirty buckets and the
    /// unfolded tail alike, one mechanism (the read-side half of the
    /// 2026-08-02 ruling on #83). Repair never changes an answer; a
    /// refresh only shrinks the live part of this union. Read-only:
    /// nothing is persisted, which is what lets a cross-process
    /// read-only reader (F4) serve exact view answers over a directory
    /// another process writes.
    ///
    /// `AS OF` on a view recomputes: `view AS OF s` is *defined* as
    /// the definition over `base AS OF s`, so the materialization —
    /// which reflects only the latest state — is bypassed entirely.
    /// The materialization accelerates current reads; it is never the
    /// authority.
    pub(crate) fn query_union(
        &self,
        source: &Table,
        dimension: Option<&Table>,
        user_plan: &Plan,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        if user_plan.referenced_columns().contains(SEQUENCE_COLUMN) {
            // Found by the repo-wide code review: the union's scratch
            // segments would fabricate sequences, and the fresh path
            // would serve the view table's own — two different wrong
            // answers depending on staleness. The view's knowledge
            // axis is the SOURCE's; ask the source.
            return Err(EngineError::Query(QueryError::Unsupported(
                "'_seq' on a maintained view — a view row has no single \
                 ingest coordinate (it summarizes many); query the base \
                 table's '_seq', or the view with ASOF"
                    .to_owned(),
            )));
        }
        let definition = Definition::of(&self.sql, source, dimension, self.width)?;
        match &definition.read {
            ReadShape::Running(running) => {
                return self.query_running(source, &definition, running, user_plan)
            }
            ReadShape::Cumulative(cumulative) => {
                return self.query_cumulative(source, &definition, cumulative, user_plan)
            }
            ReadShape::Joined => {
                let dimension = dimension.expect("a joined definition validated its dimension");
                return self.query_joined(source, dimension, &definition, user_plan);
            }
            ReadShape::Direct => {}
        }
        if let Some(cut) = user_plan.as_of {
            let mut past = definition.plan.clone();
            past.as_of = Some(cut);
            let folded = source.execute_plan(&past)?;
            let mut current = user_plan.clone();
            current.as_of = None;
            return self.over_scratch(std::iter::empty(), folded, &current);
        }
        let Some(runs) = definition.touched_runs(source, self.stamp)? else {
            return self.table.execute_plan(user_plan);
        };
        // The live half: the definition over exactly the uncovered
        // buckets. The clean half: every materialized row outside them.
        let fresh = source.execute_plan(&definition.restricted_to(&runs, source))?;
        let mut clean = select_everything(&self.table)?;
        clean.predicate = Some(Predicate::Not(Box::new(definition.view_ranges(&runs))));
        let clean = self.table.execute_plan(&clean)?;
        self.over_scratch(clean.batches.into_iter(), fresh, user_plan)
    }

    /// The join-view refresh (#83 tranche 3): both sources flushed,
    /// both watermarks captured, dirty fact-key ranges derived from
    /// BOTH knowledge histories, one restricted join re-fold, both
    /// stamps and the ceiling advanced — strictly after the fold is
    /// durable, per the single-source discipline.
    ///
    /// The dirty set has three parts: fact rows born or killed since
    /// the fact stamp; the **correction intervals** `[t, next
    /// reference row for that key after t)` for every reference row
    /// born or killed since the dimension stamp (the tranche-3 lemma —
    /// the proof lives in DESIGN's tranche-3 section); and the
    /// **ceiling advance** `[old ceiling, new frontier)`, the fact
    /// rows that became materializable because the reference frontier
    /// moved. Everything clips below the new ceiling: rows at or above
    /// it stay in the union read's live half, where ordinary in-order
    /// reference appends can re-match them at no maintenance cost.
    fn refresh_joined(
        &mut self,
        source: &mut Table,
        dimension: &mut Table,
    ) -> Result<u64, EngineError> {
        source.flush()?;
        dimension.flush()?;
        let now_fact = source.next_sequence();
        let now_dim = dimension.next_sequence();
        let definition = Definition::of(&self.sql, source, Some(dimension), self.width)?;
        let join = self.join.clone().expect("routed here on Some");
        let is_asof = definition
            .plan
            .join
            .as_ref()
            .is_some_and(|join| join.as_of.is_some());
        if !is_asof {
            return self.refresh_star(source, dimension, &definition, join);
        }
        let Some(frontier) = table_okey_max(dimension)? else {
            // No reference rows at all: nothing is materializable
            // (every fact row's match could still arrive in order). If
            // corrections emptied the reference table, the
            // materialization must empty with it.
            if join.ceiling > i64::MIN {
                let nothing = source
                    .execute_join_plan(&definition.restricted_to(&[(1, 0)], source), dimension)?;
                self.table.replace_matching(None, &nothing)?;
            }
            if let Some(join) = self.join.as_mut() {
                join.stamp = now_dim;
                join.ceiling = i64::MIN;
            }
            self.advance_stamp(now_fact)?;
            return Ok(0);
        };
        // Materialize strictly below the frontier: an in-order arrival
        // lands at or above it (a tie at the frontier included), so
        // every materialized row can only be changed by a correction —
        // which the dimension's knowledge history reports. For a
        // bucketed aggregate the ceiling rounds DOWN to its bucket's
        // low edge — the bucket containing the frontier is partially
        // unstable, so none of it materializes (identity when
        // divide == 1, the blotter).
        let ceiling = bucket_low(frontier / definition.divide, definition.divide);
        if now_fact < self.stamp || now_dim < join.stamp {
            // The rebuild floor, on either axis: a watermark below its
            // stamp cannot come from a crash under flush-then-stamp —
            // only from a foreign or tampered pairing. Trust nothing.
            // (The range converts to BUCKET runs like every fold —
            // restricted_to speaks bucket indices; the key-space form
            // over-folded past the ceiling for divide > 1, found by
            // the repo-wide code review.)
            let below = key_ranges_to_bucket_runs(
                vec![(i64::MIN, ceiling.saturating_sub(1))],
                definition.divide,
            );
            let replacement =
                source.execute_join_plan(&definition.restricted_to(&below, source), dimension)?;
            self.table.replace_matching(None, &replacement)?;
            if let Some(join) = self.join.as_mut() {
                join.stamp = now_dim;
                join.ceiling = ceiling;
            }
            self.advance_stamp(now_fact)?;
            return Ok(u64::MAX);
        }
        // A RebuildAll from the derivation would mean the shape split
        // drifted; fold everything below the ceiling rather than
        // panicking (defensive — correct either way).
        let mut ranges =
            match joined_touched_ranges(source, dimension, &definition, self.stamp, join.stamp)? {
                JoinedDirty::Ranges(ranges) => ranges,
                JoinedDirty::RebuildAll => vec![(i64::MIN, i64::MAX)],
            };
        // The fold covers dirty rows BELOW the new ceiling; the victim
        // set additionally covers the shrink band when the frontier
        // REGRESSED (a correction deleted or moved the frontier
        // reference row): rows in [new ceiling, old ceiling) leave the
        // materialization — they are live-half territory again — and
        // folding them would be wrong, but forgetting to victimize
        // them left them stranded as "clean" while their knowledge
        // coordinates were swallowed by the stamp advance (the
        // silent-corruption bug the repo-wide code review reproduced).
        ranges.retain(|&(low, _)| low < ceiling);
        for range in &mut ranges {
            range.1 = range.1.min(ceiling.saturating_sub(1));
        }
        let mut victim_ranges = ranges.clone();
        if join.ceiling < ceiling {
            // The frontier advanced: fact rows between the old ceiling
            // and the new one just became stable, and no touched
            // signal names them — the ceiling does.
            ranges.push((join.ceiling, ceiling.saturating_sub(1)));
            victim_ranges.push((join.ceiling, ceiling.saturating_sub(1)));
        }
        if ceiling < join.ceiling {
            victim_ranges.push((ceiling, join.ceiling.saturating_sub(1)));
        }
        if victim_ranges.is_empty() {
            if let Some(join) = self.join.as_mut() {
                join.stamp = now_dim;
                join.ceiling = ceiling;
            }
            self.advance_stamp(now_fact)?;
            return Ok(0);
        }
        let fold_runs = key_ranges_to_bucket_runs(merge_key_ranges(ranges), definition.divide);
        let victim_runs =
            key_ranges_to_bucket_runs(merge_key_ranges(victim_ranges), definition.divide);
        let folded = folded_bucket_count(source, &fold_runs, definition.divide)?;
        // An all-shrink refresh folds nothing; an inverted range keeps
        // the plan well-formed and empty.
        let replacement = if fold_runs.is_empty() {
            source.execute_join_plan(&definition.restricted_to(&[(1, 0)], source), dimension)?
        } else {
            source.execute_join_plan(&definition.restricted_to(&fold_runs, source), dimension)?
        };
        let victims = definition.view_ranges(&victim_runs);
        self.table.replace_matching(Some(&victims), &replacement)?;
        if let Some(join) = self.join.as_mut() {
            join.stamp = now_dim;
            join.ceiling = ceiling;
        }
        self.advance_stamp(now_fact)?;
        Ok(folded)
    }

    /// The star (equi-join) refresh: F4's ruling made concrete. The
    /// dimension is content, not history — its rows have no time
    /// axis a fact key range could name — so any dimension change
    /// re-folds the whole materialization (`u64::MAX`, the rebuild
    /// count), while fact-only changes fold incrementally. No ceiling:
    /// an equi match never depends on a reference frontier, so
    /// everything materializes (the stored ceiling parks at i64::MAX
    /// after the first refresh).
    fn refresh_star(
        &mut self,
        source: &mut Table,
        dimension: &mut Table,
        definition: &Definition,
        join: JoinState,
    ) -> Result<u64, EngineError> {
        let now_fact = source.next_sequence();
        let now_dim = dimension.next_sequence();
        let full = [(i64::MIN, i64::MAX)];
        let rebuild =
            |view: &mut MaterializedView, source: &mut Table| -> Result<u64, EngineError> {
                let replacement = source.execute_join_plan(
                    &definition.restricted_to(
                        &key_ranges_to_bucket_runs(full.to_vec(), definition.divide),
                        source,
                    ),
                    dimension,
                )?;
                view.table.replace_matching(None, &replacement)?;
                if let Some(join) = view.join.as_mut() {
                    join.stamp = now_dim;
                    join.ceiling = i64::MAX;
                }
                view.advance_stamp(now_fact)?;
                Ok(u64::MAX)
            };
        if now_fact < self.stamp || now_dim < join.stamp || join.ceiling < i64::MAX {
            // The rebuild floor on either axis, and the FIRST refresh
            // (ceiling still below MAX): one full fold either way.
            return rebuild(self, source);
        }
        match joined_touched_ranges(source, dimension, definition, self.stamp, join.stamp)? {
            JoinedDirty::RebuildAll => rebuild(self, source),
            JoinedDirty::Ranges(ranges) => {
                if ranges.is_empty() {
                    if let Some(join) = self.join.as_mut() {
                        join.stamp = now_dim;
                    }
                    self.advance_stamp(now_fact)?;
                    return Ok(0);
                }
                let runs = key_ranges_to_bucket_runs(merge_key_ranges(ranges), definition.divide);
                let folded = folded_bucket_count(source, &runs, definition.divide)?;
                let replacement = source
                    .execute_join_plan(&definition.restricted_to(&runs, source), dimension)?;
                let victims = definition.view_ranges(&runs);
                self.table.replace_matching(Some(&victims), &replacement)?;
                if let Some(join) = self.join.as_mut() {
                    join.stamp = now_dim;
                }
                self.advance_stamp(now_fact)?;
                Ok(folded)
            }
        }
    }

    /// The join view's union read: materialized rows below the ceiling
    /// and outside the dirty ranges, plus a live join over everything
    /// else — the dirty ranges and the whole tail at or above the
    /// ceiling (where in-order reference arrivals land). Exact however
    /// stale, like every union read.
    ///
    /// `AS OF` is refused: one coordinate cannot span two tables'
    /// independent sequence spaces, and the base's own AS OF-with-JOIN
    /// is refused for the same reason — refusal parity (the two-cut
    /// form is issue #99).
    fn query_joined(
        &self,
        source: &Table,
        dimension: &Table,
        definition: &Definition,
        user_plan: &Plan,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        if user_plan.as_of.is_some() {
            return Err(EngineError::Query(QueryError::Unsupported(
                "ASOF on a join view — one knowledge coordinate cannot span \
                 two tables' sequence spaces (the base refuses AS OF with a \
                 JOIN for the same reason); a two-coordinate form is issue \
                 #99"
                .to_owned(),
            )));
        }
        let join = self.join.as_ref().expect("routed here on Some");
        let mut ranges =
            match joined_touched_ranges(source, dimension, definition, self.stamp, join.stamp)? {
                // A star view with an unfolded dimension change: the
                // whole answer is live until the rebuild runs.
                JoinedDirty::RebuildAll => vec![(i64::MIN, i64::MAX)],
                JoinedDirty::Ranges(ranges) => ranges,
            };
        // The unmaterialized tail: everything at or above the ceiling
        // lives in the live half (with the ceiling at i64::MIN, that
        // is the whole axis and the read is a plain live join).
        ranges.push((join.ceiling, i64::MAX));
        let runs = key_ranges_to_bucket_runs(merge_key_ranges(ranges), definition.divide);
        let fresh =
            source.execute_join_plan(&definition.restricted_to(&runs, source), dimension)?;
        let mut clean = select_everything(&self.table)?;
        clean.predicate = Some(Predicate::Not(Box::new(definition.view_ranges(&runs))));
        let clean = self.table.execute_plan(&clean)?;
        self.over_scratch(clean.batches.into_iter(), fresh, user_plan)
    }

    /// The running view's read: partials in, answers out.
    ///
    /// `AS OF` and the not-yet-sized view (width 0, nothing ever
    /// folded) recompute the **user definition** directly over the
    /// source — it is ordinary SQL there, and for `AS OF` that IS the
    /// definition of the answer. Otherwise: the partials union (clean
    /// materialized buckets + a live partial fold of everything the
    /// stamp does not cover) runs through the **combine** — a
    /// symbol-keyed aggregate reassembling cross-bucket totals — and
    /// **finalize** turns combined columns into the user-facing row
    /// shape (`AVG` divides here, once). The user's query then runs
    /// over that row set as scratch.
    fn query_running(
        &self,
        source: &Table,
        definition: &Definition,
        running: &RunningRead,
        user_plan: &Plan,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        let mut current = user_plan.clone();
        current.as_of = None;
        if user_plan.as_of.is_some() || self.width == 0 {
            let mut recompute = running.user.clone();
            recompute.as_of = user_plan.as_of;
            let answers = source.execute_plan(&recompute)?;
            return run_over_output(
                &running.output,
                answers.batches,
                &current,
                &self.table.current_registry(),
            );
        }
        let partials = self.partials_union(source, definition)?;
        let combined = self.over_scratch(
            partials.batches.into_iter(),
            query_lite::QueryOutput {
                schema: self.table.schema().clone(),
                batches: Vec::new(),
            },
            &running.combine,
        )?;
        let finalized = finalize_combined(running, &combined)?;
        run_over_output(
            &running.output,
            finalized.into_iter().collect(),
            &current,
            &self.table.current_registry(),
        )
    }

    /// The cumulative view's read: partials in, per-row answers out.
    ///
    /// `AS OF` and the not-yet-sized view recompute the user definition
    /// directly over the source, like the running read. Otherwise the
    /// query's own predicate names an ordering-key **lower bound**; its
    /// bucket `B` splits every expanding window into the boundary
    /// combine (partials strictly below `B`, per partition) plus the
    /// assembly (the user definition over the source from `B`'s low
    /// edge), folded together by the adjustment. A query with no lower
    /// bound wants every output row, so the assembly would cover the
    /// whole source anyway — recompute IS that read, and the partials
    /// cannot shorten an O(n)-row answer.
    fn query_cumulative(
        &self,
        source: &Table,
        definition: &Definition,
        cumulative: &CumulativeRead,
        user_plan: &Plan,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        let mut current = user_plan.clone();
        current.as_of = None;
        let recompute_floor = |floor: Option<i64>| -> Option<i64> {
            if user_plan.as_of.is_some() || self.width == 0 {
                None
            } else {
                floor
            }
        };
        let floor = recompute_floor(
            user_plan
                .predicate
                .as_ref()
                .and_then(|predicate| okey_lower_bound(predicate, &cumulative.okey_output)),
        );
        let Some(floor) = floor else {
            let mut recompute = cumulative.user.clone();
            recompute.as_of = user_plan.as_of;
            let answers = source.execute_plan(&recompute)?;
            return run_over_scratch(
                &cumulative.output,
                cumulative.okey_index,
                answers.batches,
                &current,
                &self.table.current_registry(),
            );
        };
        let width = self.width as i64;
        // The executor's own truncating bucket arithmetic: everything
        // in buckets below this is boundary, everything from its low
        // edge up is assembly — disjoint and complete, because
        // truncating division is monotone.
        let boundary_bucket = floor / width;
        let partials = self.partials_union(source, definition)?;
        let mut combine = cumulative.combine.clone();
        combine.predicate = Some(Predicate::Compare {
            column: crate::partials::HIDDEN_BUCKET.to_owned(),
            op: CmpOp::Lt,
            value: Number::Int(boundary_bucket),
        });
        let combined = self.over_scratch(
            partials.batches.into_iter(),
            query_lite::QueryOutput {
                schema: self.table.schema().clone(),
                batches: Vec::new(),
            },
            &combine,
        )?;
        let boundaries = boundary_rows(cumulative, &combined);
        let mut assembly = cumulative.assembly.clone();
        let low_edge = Predicate::Compare {
            column: source.ordering_key().to_owned(),
            op: CmpOp::Ge,
            value: Number::Int(bucket_low(boundary_bucket, width)),
        };
        assembly.predicate = Some(match assembly.predicate.take() {
            Some(own) => Predicate::And(Box::new(low_edge), Box::new(own)),
            None => low_edge,
        });
        let assembled = source.execute_plan(&assembly)?;
        let adjusted = adjust_batches(cumulative, assembled.batches, &boundaries);
        run_over_scratch(
            &cumulative.output,
            cumulative.okey_index,
            adjusted,
            &current,
            &self.table.current_registry(),
        )
    }

    /// The partials union — the running/cumulative read's first half:
    /// clean materialized buckets plus a live partial fold of
    /// everything the stamp does not cover, as view-schema batches.
    fn partials_union(
        &self,
        source: &Table,
        definition: &Definition,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        match definition.touched_runs(source, self.stamp)? {
            None => self.table.execute_plan(&select_everything(&self.table)?),
            Some(runs) => {
                let fresh = source.execute_plan(&definition.restricted_to(&runs, source))?;
                let mut clean = select_everything(&self.table)?;
                clean.predicate = Some(Predicate::Not(Box::new(definition.view_ranges(&runs))));
                let mut clean = self.table.execute_plan(&clean)?;
                clean.batches.extend(fresh.batches);
                Ok(clean)
            }
        }
    }

    /// Runs `user_plan` over an ad-hoc union of view-shaped batches —
    /// [`run_over_scratch`] with this view's materialization schema
    /// and ordering key (see there for the ordering and registry
    /// stances).
    fn over_scratch(
        &self,
        clean: impl Iterator<Item = arrow_lite::RecordBatch>,
        fresh: query_lite::QueryOutput,
        user_plan: &Plan,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        let okey = self
            .table
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == self.table.ordering_key())
            .expect("the view table validated its ordering key at construction");
        run_over_scratch(
            self.table.schema(),
            okey,
            clean.chain(fresh.batches).collect(),
            user_plan,
            &self.table.current_registry(),
        )
    }

    /// Moves the stamp forward and, for a persistent view, publishes
    /// it atomically (write + rename) — strictly after the view table
    /// is **flushed**, because a cross-process read-only reader sees
    /// only the durable prefix, and a stamp ahead of the flushed
    /// materialization would make it treat never-written buckets as
    /// clean and silently drop them from the union. The record itself
    /// is not fsynced and need not be: a stamp can only be lost
    /// *backward*, and an old stamp merely re-folds. One small segment
    /// per refresh is the cost; `compact` on the view table restores
    /// contiguity, like any table's.
    fn advance_stamp(&mut self, now: u64) -> Result<(), EngineError> {
        self.stamp = now;
        if let Some(dir) = self.dir.clone() {
            self.table.flush()?;
            self.write_definition(&dir)?;
        }
        Ok(())
    }

    /// Persists the definition record — called at create and after
    /// every stamp advance. The stamp is the one piece of view state
    /// whose durability matters: it only ever advances *after* the
    /// materialization it describes is durable, so a crash between the
    /// two leaves an old stamp and the next refresh re-folds — never a
    /// stamp describing a materialization that does not exist.
    fn write_definition(&self, dir: &Path) -> Result<(), EngineError> {
        let record = encode_definition(
            self.stamp,
            self.width,
            &self.source,
            &self.sql,
            self.join.as_ref(),
        );
        let path = dir.join(DEFINITION_FILE);
        let staging = dir.join(format!("{DEFINITION_FILE}.staging"));
        std::fs::write(&staging, &record)
            .and_then(|()| std::fs::rename(&staging, &path))
            .map_err(|error| definition_error(format!("writing {DEFINITION_FILE}: {error}")))
    }
}

/// Reads and validates a persisted definition record against the
/// already-open source — the shared head of both `open` flavors:
/// the record's stamp, source name, and SQL, with the wrong-source
/// pairing and any schema drift refused loudly here rather than
/// answered wrongly later.
#[allow(clippy::type_complexity)]
fn read_definition(
    dir: &Path,
    name: &str,
    source: &Table,
    dimension: Option<&Table>,
) -> Result<(u64, u64, String, String, Schema, Option<JoinState>), EngineError> {
    let record = std::fs::read(dir.join(DEFINITION_FILE))
        .map_err(|error| definition_error(format!("reading {DEFINITION_FILE}: {error}")))?;
    let (stamp, width, source_name, sql, stored_join) = decode_definition(&record)?;
    if source_name != source.name() {
        return Err(definition_error(format!(
            "view '{name}' is over '{source_name}', not '{}'",
            source.name()
        )));
    }
    if let Some(stored) = &stored_join {
        match dimension {
            None => {
                return Err(definition_error(format!(
                    "view '{name}' joins '{}' — open it with that table",
                    stored.dimension
                )))
            }
            Some(dimension) if dimension.name() != stored.dimension => {
                return Err(definition_error(format!(
                    "view '{name}' joins '{}', not '{}'",
                    stored.dimension,
                    dimension.name()
                )));
            }
            Some(_) => {}
        }
    }
    let (_, _, answers, join) = validated_definition(&sql, source, dimension)?;
    // The validated shape says whether this is a join view; the STORED
    // stamps say where its maintenance stands. Marry them: shape from
    // the SQL, state from the record.
    let join = match (join, stored_join) {
        (Some(_), Some(stored)) => Some(stored),
        (None, None) => None,
        _ => {
            return Err(definition_error(format!(
                "view '{name}': the record's source count disagrees with \
                 its own SQL — the record is corrupt or hand-edited"
            )))
        }
    };
    Ok((stamp, width, source_name, sql, answers, join))
}

/// A lowered, validated view definition plus its bucket arithmetic —
/// what both halves of the machinery share: the refresh restricts and
/// folds with it, the union read restricts and tops up with it. For a
/// running or cumulative view, `plan` is the **synthesized partials
/// materialization** (a legal bucketed plan over the hidden bucket),
/// and `read` carries what that shape's read needs to reassemble the
/// user-facing answer.
struct Definition {
    plan: Plan,
    bucket_name: String,
    divide: i64,
    view_scale: i64,
    read: ReadShape,
}

/// How a query against the view turns its materialization into
/// answers.
enum ReadShape {
    /// Tranche 1: the materialization IS the answer; the union read
    /// serves it directly.
    Direct,
    /// A running aggregate: combine the partials per group, finalize,
    /// then the user's query. (Boxed: a read shape is built once per
    /// read, and the plans inside dwarf the discriminant.)
    Running(Box<RunningRead>),
    /// A cumulative window: prefix-combine the partials into per-
    /// partition boundary values, assemble the queried range from the
    /// source, adjust by the boundaries, then the user's query.
    Cumulative(Box<CumulativeRead>),
    /// Tranche 3, cycle 1: a bare as-of join (the enriched blotter).
    /// The materialization IS the answer, keyed by the fact ordering
    /// key; the read tops it up with a live join over the dirty ranges
    /// and the unmaterialized tail at or above the ceiling.
    Joined,
}

/// The read-side half of a running view: how partials become answers.
struct RunningRead {
    /// The user definition, verbatim-lowered — directly executable
    /// over the source, which is what `AS OF` and the rebuild/unsized
    /// paths run.
    user: Plan,
    /// The combine: a symbol-keyed aggregate over the partials, run on
    /// the partials union as scratch. Its output is
    /// `[user keys…, combined partials…]` in user-item order.
    combine: Plan,
    /// Finalize steps, one per user output column, over the combine's
    /// output columns by index.
    finalize: Vec<FinalStep>,
    /// The user-facing output schema (from the user plan), plus the
    /// appended `__row` scratch ordering key as its last field.
    output: Schema,
}

/// One user output column, assembled from combined-partial columns.
enum FinalStep {
    /// The combined column at this index passes through (keys, SUM,
    /// MIN, MAX, FIRST, LAST).
    Pass(usize),
    /// `COUNT`: the combined column with NULL grounded to zero — a
    /// count's combine is a SUM of counts, and SUM over an empty
    /// group is NULL where COUNT is 0.
    CountZero(usize),
    /// `AVG`: combined sum over combined count, NULL where the count
    /// is zero — the division happens once, after the cross-bucket
    /// combine, never per bucket.
    AvgDivide { sum: usize, count: usize },
}

/// The read-side half of a cumulative view: every expanding window is
/// split at one hidden-bucket boundary `B`, derived per query from the
/// user predicate's ordering-key lower bound. The **boundary** is a
/// combine over the partials strictly below `B` (one row per partition
/// combination — everything before the assembled range, folded); the
/// **assembly** runs the user definition over the source from `B`'s
/// low edge; the **adjustment** folds each row's boundary into its
/// assembled window values. Union of the two ranges is exact and
/// disjoint: truncating division is monotone, so `bucket < B` is
/// precisely `key < bucket_low(B)`.
struct CumulativeRead {
    /// The user definition, verbatim-lowered — directly executable
    /// over the source, which is what `AS OF`, the unsized view, and
    /// the no-lower-bound query run.
    user: Plan,
    /// The assembly: the user plan, plus hidden expanding `sum`/`count`
    /// windows for each `AVG` (an average adjusts through its parts,
    /// never through its quotient). Its range restriction is ANDed in
    /// per query.
    assembly: Plan,
    /// The boundary combine: a partition-keyed aggregate over the
    /// partials, `[partition symbols…, combined partials…]`. Its
    /// `__bucket < B` restriction is ANDed in per query.
    combine: Plan,
    /// One step per user output column, indexing into the assembly's
    /// and the combine's output columns.
    adjust: Vec<AdjustStep>,
    /// The partition symbols' column indices in the assembly output —
    /// what keys a row to its boundary row.
    partition_assembly: Vec<usize>,
    /// The ordering-key column's output name — what the query-side
    /// lower bound is extracted against.
    okey_output: String,
    /// The ordering-key column's index in `output` — the scratch
    /// ordering key (a cumulative answer keeps the source's axis, so
    /// no `__row` is fabricated).
    okey_index: usize,
    /// The user-facing output schema, window columns forced nullable.
    output: Schema,
}

/// One user output column of a cumulative view, adjusted by the
/// row's partition boundary.
enum AdjustStep {
    /// A non-window column (the ordering key, a partition symbol):
    /// the assembly column passes through.
    Pass(usize),
    /// `SUM` / `COUNT`: assembled value plus the boundary's, in the
    /// assembly column's own type (`SUM` windows are f64, `COUNT` i64).
    Add { column: usize, boundary: usize },
    /// `MIN`: the smaller of assembled and boundary.
    MinFold { column: usize, boundary: usize },
    /// `MAX`: the larger.
    MaxFold { column: usize, boundary: usize },
    /// `AVG`: (hidden assembled sum + boundary sum) over (hidden
    /// assembled count + boundary count), NULL where the total count
    /// is zero — the division happens once, after the fold.
    AvgAssemble {
        sum: usize,
        count: usize,
        boundary_sum: usize,
        boundary_count: usize,
    },
}

impl Definition {
    /// Builds the definition for `sql` over `source`. `width` is a
    /// running view's hidden-bucket width in ordering-key units — `0`
    /// means not yet chosen (the first refresh with data chooses it),
    /// and the caller must not fold with an unsized definition; a
    /// placeholder width of 1 keeps the synthesized plan well-formed
    /// for schema derivation.
    fn of(
        sql: &str,
        source: &Table,
        dimension: Option<&Table>,
        width: u64,
    ) -> Result<Definition, EngineError> {
        let plan = lower_plan(sql).map_err(EngineError::Query)?;
        match eligible_shape(&plan, source, dimension)? {
            Shape::Bucketed(bucket, bucket_name) => {
                let (divide, view_scale) = bucket_arithmetic(&bucket);
                Ok(Definition {
                    plan,
                    bucket_name,
                    divide,
                    view_scale,
                    read: ReadShape::Direct,
                })
            }
            Shape::Running => synthesize_running(plan, source, width.max(1) as i64),
            Shape::Cumulative => synthesize_cumulative(plan, source, width.max(1) as i64),
            Shape::Joined(okey_name) => Ok(Definition {
                plan,
                bucket_name: okey_name,
                // A blotter row's "bucket" is its own fact ordering-key
                // value: repair granularity is the key itself.
                divide: 1,
                view_scale: 1,
                read: ReadShape::Joined,
            }),
            Shape::JoinedBucketed(bucket, bucket_name) => {
                let (divide, view_scale) = bucket_arithmetic(&bucket);
                Ok(Definition {
                    plan,
                    bucket_name,
                    divide,
                    view_scale,
                    read: ReadShape::Joined,
                })
            }
        }
    }

    /// The touched buckets since `stamp`, as maximal runs of
    /// consecutive indices — `None` when no row changed.
    fn touched_runs(
        &self,
        source: &Table,
        stamp: u64,
    ) -> Result<Option<Vec<(i64, i64)>>, EngineError> {
        let mut buckets = std::collections::BTreeSet::new();
        let divide = self.divide;
        source.touched_ordering_keys(stamp, |value| {
            buckets.insert(value / divide);
        })?;
        if buckets.is_empty() {
            return Ok(None);
        }
        Ok(Some(contiguous_runs(&buckets)))
    }

    /// The definition, restricted to the runs' buckets: its own plan
    /// with the prunable source-side range predicate ANDed in, so zone
    /// maps skip every untouched segment.
    fn restricted_to(&self, runs: &[(i64, i64)], source: &Table) -> Plan {
        let mut restricted = self.plan.clone();
        let ranges = ranges_predicate(
            source.ordering_key(),
            runs.iter().map(|&(first, last)| {
                (
                    bucket_low(first, self.divide),
                    bucket_high(last, self.divide),
                )
            }),
        );
        restricted.predicate = Some(match restricted.predicate.take() {
            Some(own) => Predicate::And(Box::new(ranges), Box::new(own)),
            None => ranges,
        });
        restricted
    }

    /// The same runs on the view side: the view's ordering key stores
    /// the bucket index scaled by the definition's multiplier (or the
    /// raw key for a bare-`ts` bucket), so a run of indices maps to
    /// one scaled range.
    fn view_ranges(&self, runs: &[(i64, i64)]) -> Predicate {
        ranges_predicate(
            &self.bucket_name,
            runs.iter().map(|&(first, last)| {
                (
                    first.saturating_mul(self.view_scale),
                    last.saturating_mul(self.view_scale),
                )
            }),
        )
    }
}

/// Synthesizes a running view's machinery from its user plan: the
/// partials materialization plan (bucketed on the hidden bucket — a
/// legal tranche-1 plan, which is what lets refresh, touched-bucket
/// derivation, and the stamp serve running views unchanged), the
/// combine plan, and the finalize steps.
fn synthesize_running(user: Plan, source: &Table, width: i64) -> Result<Definition, EngineError> {
    use crate::partials::{decompose, HIDDEN_BUCKET};
    use query_lite::{AggItem, Projection as Proj};
    let Proj::Aggregate { keys, items, .. } = &user.projection else {
        unreachable!("classified Running from an aggregate projection")
    };
    // The materialization: user keys + the hidden bucket, user key
    // items + the bucket item + the partial calls.
    let mut internal_keys = keys.clone();
    internal_keys.push(GroupKey::Bucket {
        column: source.ordering_key().to_owned(),
        divide: width,
        multiply: None,
    });
    let mut internal_items: Vec<AggItem> = Vec::new();
    let mut combine_items: Vec<AggItem> = Vec::new();
    let mut finalize: Vec<FinalStep> = Vec::new();
    let mut partial_index = 0usize;
    let mut combined_index = 0usize;
    // The combine runs over the MATERIALIZATION as scratch, where a
    // key column is stored under its selected output name — the user's
    // alias when one was written. The combine's group keys and key
    // items must therefore name the STORED column, not the source one
    // (found by the repo-wide code review: source-named combine keys
    // broke every partials-path read of an aliased-key view).
    let stored_key_name = |key: &GroupKey| {
        items
            .iter()
            .find_map(|item| match item {
                AggItem::Key { key: k, alias } if k == key => {
                    Some(alias.clone().unwrap_or_else(|| k.output_name()))
                }
                _ => None,
            })
            .expect("eligibility required every group key selected")
    };
    let mut combine_keys: Vec<GroupKey> = Vec::with_capacity(keys.len());
    for key in keys {
        combine_keys.push(GroupKey::Column(stored_key_name(key)));
    }
    for item in items {
        match item {
            AggItem::Key { key, alias } => {
                internal_items.push(item.clone());
                let stored = alias.clone().unwrap_or_else(|| key.output_name());
                combine_items.push(AggItem::Key {
                    key: GroupKey::Column(stored.clone()),
                    // Re-alias to the same output name: the combine's
                    // output column keeps the user-facing name.
                    alias: Some(stored),
                });
                finalize.push(FinalStep::Pass(combined_index));
                combined_index += 1;
            }
            AggItem::Call(call) => {
                let decomposition = decompose(call, partial_index);
                partial_index += decomposition.partials.len();
                for partial in decomposition.partials {
                    internal_items.push(AggItem::Call(partial));
                }
                let combined_first = combined_index;
                combined_index += decomposition.combines.len();
                let form = decomposition.form;
                for combine in decomposition.combines {
                    combine_items.push(AggItem::Call(combine));
                }
                finalize.push(match form {
                    crate::partials::PartialForm::SumCount => FinalStep::AvgDivide {
                        sum: combined_first,
                        count: combined_first + 1,
                    },
                    crate::partials::PartialForm::Count => FinalStep::CountZero(combined_first),
                    _ => FinalStep::Pass(combined_first),
                });
            }
        }
    }
    internal_items.push(AggItem::Key {
        key: GroupKey::Bucket {
            column: source.ordering_key().to_owned(),
            divide: width,
            multiply: None,
        },
        alias: Some(HIDDEN_BUCKET.to_owned()),
    });
    let internal = Plan {
        table: user.table.clone(),
        join: None,
        projection: Proj::Aggregate {
            keys: internal_keys,
            items: internal_items,
            having: None,
        },
        distinct: false,
        predicate: user.predicate.clone(),
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    // The combine groups by the user's symbol keys alone — the hidden
    // bucket has done its job once the partials are assembled.
    let combine = Plan {
        table: user.table.clone(),
        join: None,
        projection: Proj::Aggregate {
            keys: combine_keys,
            items: combine_items,
            having: None,
        },
        distinct: false,
        predicate: None,
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    // The user-facing output schema, from the user plan itself — plus
    // `__row`, the finalized scratch's ordering key (a running answer
    // has no natural i64 axis; scratch segments need one).
    let mut output_fields: Vec<Field> = source
        .execute_plan_empty(&user)?
        .schema
        .fields()
        .iter()
        .map(|field| {
            // Every aggregate output can be NULL (the average of an
            // emptied group, say), and finalize always builds a
            // validity bitmap for AVG — so non-key outputs are
            // nullable in the scratch schema regardless of what the
            // executor inferred over zero rows.
            if field.column_type() == ColumnType::Key {
                field.clone()
            } else {
                Field::new(field.name(), field.column_type(), true)
            }
        })
        .collect();
    output_fields.push(Field::new("__row", ColumnType::I64, false));
    Ok(Definition {
        plan: internal,
        bucket_name: crate::partials::HIDDEN_BUCKET.to_owned(),
        divide: width,
        view_scale: 1,
        read: ReadShape::Running(Box::new(RunningRead {
            user,
            combine,
            finalize,
            output: Schema::new(output_fields),
        })),
    })
}

/// Synthesizes a cumulative view's machinery from its user plan. The
/// materialization is the same hidden-bucket partials plan a running
/// view stores — grouped by the windows' partition symbols — so every
/// piece of tranche-1 maintenance (refresh, touched buckets, the
/// stamp, the crash story) serves it unchanged. The read splits each
/// expanding window at a per-query bucket boundary; what this function
/// builds is everything that split needs: the boundary combine, the
/// assembly plan (with hidden `sum`/`count` helper windows for each
/// `AVG` — an average adjusts through its parts, never through its
/// quotient), and the adjustment steps.
fn synthesize_cumulative(
    user: Plan,
    source: &Table,
    width: i64,
) -> Result<Definition, EngineError> {
    use crate::partials::{decompose, expanding_window_function, PartialForm, HIDDEN_BUCKET};
    use query_lite::plan::WindowCall;
    use query_lite::{AggCall, AggItem, PlanItem, Projection as Proj};
    let Proj::Items(items) = &user.projection else {
        unreachable!("classified Cumulative from an Items projection")
    };
    // Partition symbols in first-appearance order (classification
    // proved every window shares one PARTITION BY list).
    let mut partition_columns: Vec<String> = Vec::new();
    for item in items {
        if let PlanItem::Window {
            call: WindowCall::Agg { partition_by, .. },
            ..
        } = item
        {
            for term in partition_by {
                if let GroupKey::Column(column) = term {
                    if !partition_columns.iter().any(|have| have == column) {
                        partition_columns.push(column.clone());
                    }
                }
            }
        }
    }
    let bucket = GroupKey::Bucket {
        column: source.ordering_key().to_owned(),
        divide: width,
        multiply: None,
    };
    let mut internal_keys: Vec<GroupKey> = partition_columns
        .iter()
        .cloned()
        .map(GroupKey::Column)
        .collect();
    internal_keys.push(bucket.clone());
    let mut internal_items: Vec<AggItem> = partition_columns
        .iter()
        .map(|column| AggItem::Key {
            key: GroupKey::Column(column.clone()),
            alias: None,
        })
        .collect();
    let mut combine_items: Vec<AggItem> = internal_items.clone();
    let mut assembly_items: Vec<PlanItem> = items.clone();
    let mut adjust: Vec<AdjustStep> = Vec::new();
    let mut partial_index = 0usize;
    // The combine's output leads with its partition key columns.
    let mut combined_index = partition_columns.len();
    for (index, item) in items.iter().enumerate() {
        let PlanItem::Window { call, .. } = item else {
            adjust.push(AdjustStep::Pass(index));
            continue;
        };
        let WindowCall::Agg {
            function,
            args,
            partition_by,
            order_by,
            frame,
        } = call
        else {
            unreachable!("classification refused positional windows")
        };
        let function =
            expanding_window_function(function).expect("classification admitted the family");
        let decomposition = decompose(
            &AggCall {
                function,
                argument: args.first().cloned(),
                alias: None,
            },
            partial_index,
        );
        partial_index += decomposition.partials.len();
        for partial in decomposition.partials {
            internal_items.push(AggItem::Call(partial));
        }
        let combined_first = combined_index;
        combined_index += decomposition.combines.len();
        for combine in decomposition.combines {
            combine_items.push(AggItem::Call(combine));
        }
        adjust.push(match decomposition.form {
            PartialForm::Sum | PartialForm::Count => AdjustStep::Add {
                column: index,
                boundary: combined_first,
            },
            PartialForm::Min => AdjustStep::MinFold {
                column: index,
                boundary: combined_first,
            },
            PartialForm::Max => AdjustStep::MaxFold {
                column: index,
                boundary: combined_first,
            },
            PartialForm::SumCount => {
                let helper = |function: &str, alias: String| PlanItem::Window {
                    call: WindowCall::Agg {
                        function: function.to_owned(),
                        args: args.clone(),
                        partition_by: partition_by.clone(),
                        order_by: order_by.clone(),
                        frame: *frame,
                    },
                    alias: Some(alias),
                };
                let sum = assembly_items.len();
                assembly_items.push(helper("sum", format!("__w{index}_sum")));
                let count = assembly_items.len();
                assembly_items.push(helper("count", format!("__w{index}_count")));
                AdjustStep::AvgAssemble {
                    sum,
                    count,
                    boundary_sum: combined_first,
                    boundary_count: combined_first + 1,
                }
            }
            PartialForm::First | PartialForm::Last => {
                unreachable!("first/last are not expanding windows")
            }
        });
    }
    internal_items.push(AggItem::Key {
        key: bucket,
        alias: Some(HIDDEN_BUCKET.to_owned()),
    });
    let internal = Plan {
        table: user.table.clone(),
        join: None,
        projection: Proj::Aggregate {
            keys: internal_keys,
            items: internal_items,
            having: None,
        },
        distinct: false,
        predicate: user.predicate.clone(),
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    let combine = Plan {
        table: user.table.clone(),
        join: None,
        projection: Proj::Aggregate {
            keys: partition_columns
                .iter()
                .cloned()
                .map(GroupKey::Column)
                .collect(),
            items: combine_items,
            having: None,
        },
        distinct: false,
        predicate: None,
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    let mut assembly = user.clone();
    assembly.projection = Proj::Items(assembly_items);
    // The ordering key's output position and name (classification
    // required it selected), and each partition symbol's — the
    // adjustment keys assembly rows to boundary rows by these.
    let output_position = |wanted: &str| {
        items
            .iter()
            .position(|item| matches!(item, PlanItem::Column { name, .. } if name == wanted))
    };
    let okey_index = output_position(source.ordering_key())
        .expect("classification required the ordering key selected");
    let okey_output = match &items[okey_index] {
        PlanItem::Column { name, alias } => alias.clone().unwrap_or_else(|| name.clone()),
        _ => unreachable!("found as a Column just above"),
    };
    let partition_assembly: Vec<usize> = partition_columns
        .iter()
        .map(|column| {
            output_position(column).expect("classification required partition symbols selected")
        })
        .collect();
    // The user-facing output schema: window columns forced nullable —
    // the adjustment builds validity bitmaps for them regardless of
    // what the executor inferred over zero rows.
    let output_fields: Vec<Field> = source
        .execute_plan_empty(&user)?
        .schema
        .fields()
        .iter()
        .zip(items)
        .map(|(field, item)| {
            if matches!(item, PlanItem::Window { .. }) {
                Field::new(field.name(), field.column_type(), true)
            } else {
                field.clone()
            }
        })
        .collect();
    Ok(Definition {
        plan: internal,
        bucket_name: HIDDEN_BUCKET.to_owned(),
        divide: width,
        view_scale: 1,
        read: ReadShape::Cumulative(Box::new(CumulativeRead {
            user,
            assembly,
            combine,
            adjust,
            partition_assembly,
            okey_output,
            okey_index,
            output: Schema::new(output_fields),
        })),
    })
}

/// The re-fold count for a run list, clipped to the fact table's
/// actual span in bucket units: a first refresh's range opens at
/// i64::MIN, and "buckets re-folded" should mean buckets that exist,
/// not the width of the axis. One body for both join refresh paths.
fn folded_bucket_count(
    source: &Table,
    runs: &[(i64, i64)],
    divide: i64,
) -> Result<u64, EngineError> {
    Ok(match source_span(source)? {
        None => 0,
        Some((fact_low, fact_high)) => {
            let (fact_low, fact_high) = (fact_low / divide, fact_high / divide);
            runs.iter()
                .map(|&(low, high)| {
                    let low = low.max(fact_low) as i128;
                    let high = high.min(fact_high) as i128;
                    (high - low + 1).max(0) as u128
                })
                .sum::<u128>()
                .min(u64::MAX as u128) as u64
        }
    })
}

/// The first cell of a single-column, single-row i64 aggregate — the
/// shared tail of the structural MIN/MAX probes below.
fn single_i64_cell(output: &query_lite::QueryOutput) -> Option<i64> {
    use arrow_lite::{Column, NumericData};
    let batch = output
        .batches
        .first()
        .filter(|batch| batch.num_rows() > 0)?;
    match &batch.columns()[0] {
        Column::Numeric(NumericData::I64(column)) => {
            column.is_valid(0).then(|| column.values().as_slice()[0])
        }
        _ => None,
    }
}

/// A join view's dirty set, derived from BOTH knowledge histories.
enum JoinedDirty {
    /// Fact-key ranges to re-fold: fact rows born or killed since the
    /// fact stamp (single-key ranges), plus — for the ASOF shape — the
    /// correction interval per touched reference row. Unclipped:
    /// refresh clips below its ceiling, the read unions the tail in.
    Ranges(Vec<(i64, i64)>),
    /// The star shape's answer to ANY dimension change (F4, ruled on
    /// #83): a dimension row's blast radius is every fact bucket
    /// holding its key — symbol-shaped, which the range machinery
    /// cannot express — so the whole materialization re-folds.
    /// O(fact) per rare event, zero new machinery; the per-symbol
    /// index is a held seat.
    RebuildAll,
}

/// The dirty derivation (#83 tranche 3). For the ASOF shape, a
/// touched reference row at `t` contributes the correction interval
/// `[t, next reference row for that key strictly after t, in current
/// state)` — open-ended when no next exists; a NULL-keyed reference
/// row matches nothing and is skipped. For the equi (star) shape, any
/// touched dimension row escalates to [`JoinedDirty::RebuildAll`].
fn joined_touched_ranges(
    source: &Table,
    dimension: &Table,
    definition: &Definition,
    fact_stamp: u64,
    dim_stamp: u64,
) -> Result<JoinedDirty, EngineError> {
    let join = definition
        .plan
        .join
        .as_ref()
        .expect("a joined definition carries its join");
    let mut ranges: Vec<(i64, i64)> = Vec::new();
    source.touched_ordering_keys(fact_stamp, |key| ranges.push((key, key)))?;
    let mut touched_reference: Vec<(i64, Option<String>)> = Vec::new();
    dimension.touched_rows(dim_stamp, &join.dimension_key, |key, value| {
        touched_reference.push((key, value.map(str::to_owned)));
    })?;
    if join.as_of.is_none() {
        if !touched_reference.is_empty() {
            return Ok(JoinedDirty::RebuildAll);
        }
        return Ok(JoinedDirty::Ranges(ranges));
    }
    for (at, value) in touched_reference {
        let Some(value) = value else {
            continue; // a null reference key matches nothing, ever
        };
        let next = next_reference_key(dimension, &join.dimension_key, &value, at)?;
        // The interval's exclusive end is the symbol's next reference
        // row — but "reaches" differs by match mode: under AtOrBefore
        // a fact at exactly `next` already matches `next`, so the
        // interval stops at `next - 1`; under StrictlyBefore that fact
        // still matches the row BEFORE `next`, so the correction at
        // `at` reaches it and the interval includes `next` itself
        // (found by the repo-wide code review — the off-by-one left a
        // fact at exactly `next` silently stale under the strict form).
        let high = match join.as_of.expect("ranges with intervals only for ASOF") {
            query_lite::AsOfMatch::AtOrBefore => next.map_or(i64::MAX, |next| next - 1),
            query_lite::AsOfMatch::StrictlyBefore => next.unwrap_or(i64::MAX),
        };
        if at <= high {
            ranges.push((at, high));
        }
    }
    Ok(JoinedDirty::Ranges(ranges))
}

/// The least reference ordering-key value strictly after `after` for
/// `value`'s rows, in the dimension's CURRENT state — the correction
/// interval's exclusive end (`None` = open-ended). One small prunable
/// aggregate per touched reference row; corrections are rare, and the
/// lookup is what keeps their blast radius an interval instead of a
/// suffix.
fn next_reference_key(
    dimension: &Table,
    key_column: &str,
    value: &str,
    after: i64,
) -> Result<Option<i64>, EngineError> {
    use query_lite::{AggCall, AggFunction, AggItem, Projection as Proj};
    let plan = Plan {
        table: dimension.name().to_owned(),
        join: None,
        projection: Proj::Aggregate {
            keys: Vec::new(),
            items: vec![AggItem::Call(AggCall {
                function: AggFunction::Min,
                argument: Some(dimension.ordering_key().to_owned()),
                alias: Some("__next".to_owned()),
            })],
            having: None,
        },
        distinct: false,
        predicate: Some(Predicate::And(
            Box::new(Predicate::KeyEquals {
                column: key_column.to_owned(),
                value: value.to_owned(),
                negated: false,
            }),
            Box::new(Predicate::Compare {
                column: dimension.ordering_key().to_owned(),
                op: CmpOp::Gt,
                value: Number::Int(after),
            }),
        )),
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    Ok(single_i64_cell(&dimension.execute_plan(&plan)?))
}

/// The dimension's current ordering-key frontier (its greatest value),
/// `None` when empty — what a join view's materialization ceiling
/// advances to.
fn table_okey_max(table: &Table) -> Result<Option<i64>, EngineError> {
    use query_lite::{AggCall, AggFunction, AggItem, Projection as Proj};
    let plan = Plan {
        table: table.name().to_owned(),
        join: None,
        projection: Proj::Aggregate {
            keys: Vec::new(),
            items: vec![AggItem::Call(AggCall {
                function: AggFunction::Max,
                argument: Some(table.ordering_key().to_owned()),
                alias: Some("__hi".to_owned()),
            })],
            having: None,
        },
        distinct: false,
        predicate: None,
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    Ok(single_i64_cell(&table.execute_plan(&plan)?))
}

/// Key-space ranges to bucket-index runs under the definition's
/// truncating bucket width — merged again afterward, since adjacent
/// keys can share a bucket. Identity when `divide == 1` (the blotter,
/// whose repair granularity is the key itself).
fn key_ranges_to_bucket_runs(ranges: Vec<(i64, i64)>, divide: i64) -> Vec<(i64, i64)> {
    if divide == 1 {
        return ranges;
    }
    merge_key_ranges(
        ranges
            .into_iter()
            .map(|(low, high)| (low / divide, high / divide))
            .collect(),
    )
}

/// Sorts and merges overlapping or adjacent key ranges — the join
/// view's dirty set arrives as arbitrary intervals (single keys,
/// correction spans, the ceiling advance), and one merged run list
/// keeps the fold's predicate small.
fn merge_key_ranges(mut ranges: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    ranges.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(ranges.len());
    for (low, high) in ranges {
        match merged.last_mut() {
            Some((_, last_high)) if low <= last_high.saturating_add(1) => {
                *last_high = (*last_high).max(high);
            }
            _ => merged.push((low, high)),
        }
    }
    merged
}

/// The source's ordering-key span `(min, max)`, `None` when empty —
/// what sizes a running view's hidden bucket. Runs as one global
/// aggregate; exactness past 2^53 is irrelevant here because the width
/// is a heuristic, not a semantic.
fn source_span(source: &Table) -> Result<Option<(i64, i64)>, EngineError> {
    use query_lite::{AggCall, AggFunction, AggItem, Projection as Proj};
    let call = |function, alias: &str| {
        AggItem::Call(AggCall {
            function,
            argument: Some(source.ordering_key().to_owned()),
            alias: Some(alias.to_owned()),
        })
    };
    let plan = Plan {
        table: source.name().to_owned(),
        join: None,
        projection: Proj::Aggregate {
            keys: Vec::new(),
            items: vec![
                call(AggFunction::Min, "__lo"),
                call(AggFunction::Max, "__hi"),
            ],
            having: None,
        },
        distinct: false,
        predicate: None,
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    };
    let output = source.execute_plan(&plan)?;
    let Some(batch) = output.batches.first().filter(|batch| batch.num_rows() > 0) else {
        return Ok(None);
    };
    let cell = |index: usize| -> Option<i64> {
        use arrow_lite::{Column, NumericData};
        match &batch.columns()[index] {
            Column::Numeric(NumericData::I64(column)) => {
                column.is_valid(0).then(|| column.values().as_slice()[0])
            }
            Column::Numeric(NumericData::F64(column)) => column
                .is_valid(0)
                .then(|| column.values().as_slice()[0] as i64),
            Column::Key(_) => None,
        }
    };
    Ok(match (cell(0), cell(1)) {
        (Some(low), Some(high)) => Some((low, high)),
        _ => None,
    })
}

/// Turns the combine's output into the user-facing row shape: one
/// column per finalize step, plus the `__row` scratch ordering key.
/// `AVG` divides its combined sum by its combined count here — once,
/// after the cross-bucket combine — and is NULL where the count is
/// zero, standard SQL's average of nothing.
fn finalize_combined(
    running: &RunningRead,
    combined: &query_lite::QueryOutput,
) -> Result<Option<arrow_lite::RecordBatch>, EngineError> {
    use arrow_lite::{Bitmap, Buffer, Column, NumericColumn, NumericData, RecordBatch};
    // Collapsing stages materialize one batch (QueryOutput's contract);
    // an empty result has none, and finalizes to none.
    let Some(batch) = combined.batches.first() else {
        return Ok(None);
    };
    let rows = batch.num_rows();
    let mut columns: Vec<Column> = Vec::with_capacity(running.finalize.len() + 1);
    for step in &running.finalize {
        columns.push(match step {
            FinalStep::Pass(index) => batch.columns()[*index].clone(),
            FinalStep::CountZero(index) => {
                let Column::Numeric(NumericData::I64(counts)) = &batch.columns()[*index] else {
                    unreachable!("a COUNT combine is an exact i64 sum")
                };
                let values: Buffer<i64> = (0..rows)
                    .map(|row| {
                        if counts.is_valid(row) {
                            counts.values().as_slice()[row]
                        } else {
                            0
                        }
                    })
                    .collect();
                Column::Numeric(NumericData::I64(NumericColumn::new_non_null(values)))
            }
            FinalStep::AvgDivide { sum, count } => {
                // The combined sum's type follows the argument column:
                // f64 for an f64 argument, i64 for an i64 one (SUM over
                // i64 stays exact i64). Both divide in f64 — AVG's
                // output type either way.
                let sum_at = |row: usize| -> Option<f64> {
                    match &batch.columns()[*sum] {
                        Column::Numeric(NumericData::F64(sums)) => {
                            sums.is_valid(row).then(|| sums.values().as_slice()[row])
                        }
                        Column::Numeric(NumericData::I64(sums)) => sums
                            .is_valid(row)
                            .then(|| sums.values().as_slice()[row] as f64),
                        Column::Key(_) => unreachable!("an AVG sum partial is numeric"),
                    }
                };
                let Column::Numeric(NumericData::I64(counts)) = &batch.columns()[*count] else {
                    unreachable!("an AVG decomposition's count partial is i64")
                };
                let mut values = Vec::with_capacity(rows);
                let mut validity = Vec::with_capacity(rows);
                for row in 0..rows {
                    let count = if counts.is_valid(row) {
                        counts.values().as_slice()[row]
                    } else {
                        0
                    };
                    let sum = sum_at(row);
                    let defined = count > 0 && sum.is_some();
                    validity.push(defined);
                    values.push(if defined {
                        sum.expect("checked") / count as f64
                    } else {
                        0.0
                    });
                }
                Column::Numeric(NumericData::F64(NumericColumn::new_nullable(
                    values.into_iter().collect(),
                    Bitmap::from_bools(validity),
                )))
            }
        });
    }
    columns.push(Column::Numeric(NumericData::I64(
        NumericColumn::new_non_null((0..rows as i64).collect::<Buffer<i64>>()),
    )));
    Ok(Some(RecordBatch::new(running.output.clone(), columns)))
}

/// Runs `user_plan` over finished output rows as scratch, appending
/// the `__row` axis where the rows arrived without one (the recompute
/// paths, whose batches carry the bare user schema). Every batch here
/// is ordered on `__row` by construction (it counts rows).
fn run_over_output(
    output: &Schema,
    batches: Vec<arrow_lite::RecordBatch>,
    user_plan: &Plan,
    registry: &query_lite::Registry,
) -> Result<query_lite::QueryOutput, EngineError> {
    use arrow_lite::{Column, NumericColumn, NumericData, RecordBatch};
    let okey = output.fields().len() - 1; // __row, by construction
    let batches = batches
        .into_iter()
        .map(|batch| {
            if batch.columns().len() == output.fields().len() {
                batch
            } else {
                let rows = batch.num_rows() as i64;
                let mut columns = batch.columns().to_vec();
                columns.push(Column::Numeric(NumericData::I64(
                    NumericColumn::new_non_null((0..rows).collect::<arrow_lite::Buffer<i64>>()),
                )));
                RecordBatch::new(output.clone(), columns)
            }
        })
        .collect();
    run_over_scratch(output, okey, batches, user_plan, registry)
}

/// A conservative ordering-key **lower bound** from a query predicate:
/// a value `v` such that every row the predicate can accept has
/// `okey >= v` — `None` when no such bound is derivable, which the
/// caller answers by full recompute (correct, just unshortened). The
/// direction of conservatism matters: a bound may be *lower* than the
/// truth (assembling extra rows the query then filters), never higher.
/// `AND` takes the tighter branch, `OR` needs both and takes the
/// looser, `>` weakens to `>=` (one extra value), a float literal
/// floors, and every unhandled shape is `None`.
fn okey_lower_bound(predicate: &Predicate, okey: &str) -> Option<i64> {
    match predicate {
        Predicate::And(left, right) => {
            match (okey_lower_bound(left, okey), okey_lower_bound(right, okey)) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (one, other) => one.or(other),
            }
        }
        Predicate::Or(left, right) => {
            match (okey_lower_bound(left, okey), okey_lower_bound(right, okey)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                _ => None,
            }
        }
        Predicate::Compare { column, op, value } if column == okey => {
            let floor = match value {
                Number::Int(value) => *value,
                Number::Float(value) => value.floor() as i64,
            };
            match op {
                CmpOp::Ge | CmpOp::Eq | CmpOp::Gt => Some(floor),
                _ => None,
            }
        }
        _ => None,
    }
}

/// One combined-boundary value — the two numeric shapes a combine can
/// emit. An adjustment reads it in the assembly column's own domain
/// (a `MIN` window is f64 over an i64 column whose partials stay i64).
#[derive(Clone, Copy)]
enum Cell {
    I64(i64),
    F64(f64),
}

impl Cell {
    fn as_f64(self) -> f64 {
        match self {
            Cell::I64(value) => value as f64,
            Cell::F64(value) => value,
        }
    }

    fn as_i64(self) -> i64 {
        match self {
            Cell::I64(value) => value,
            Cell::F64(value) => value as i64,
        }
    }
}

/// The boundary combine's output as a lookup: partition values (in
/// combine key order) to the full combined row. Absence means no
/// source rows below the boundary for that partition — the adjustment
/// then adds nothing, which is exactly right.
fn boundary_rows(
    cumulative: &CumulativeRead,
    combined: &query_lite::QueryOutput,
) -> std::collections::HashMap<Vec<Option<String>>, Vec<Option<Cell>>> {
    use arrow_lite::{Column, NumericData};
    let mut map = std::collections::HashMap::new();
    // Collapsing stages materialize one batch (QueryOutput's contract);
    // an empty result has none.
    let Some(batch) = combined.batches.first() else {
        return map;
    };
    let partitions = cumulative.partition_assembly.len();
    for row in 0..batch.num_rows() {
        let key: Vec<Option<String>> = (0..partitions)
            .map(|column| match &batch.columns()[column] {
                Column::Key(keys) => keys.value_at(row).map(str::to_owned),
                _ => None,
            })
            .collect();
        let cells: Vec<Option<Cell>> = batch
            .columns()
            .iter()
            .map(|column| match column {
                Column::Numeric(NumericData::I64(values)) => values
                    .is_valid(row)
                    .then(|| Cell::I64(values.values().as_slice()[row])),
                Column::Numeric(NumericData::F64(values)) => values
                    .is_valid(row)
                    .then(|| Cell::F64(values.values().as_slice()[row])),
                Column::Key(_) => None,
            })
            .collect();
        map.insert(key, cells);
    }
    map
}

/// Applies a cumulative view's adjustment steps to the assembly's
/// batches: each row's boundary — looked up by its partition values —
/// folds into its assembled window columns, and the hidden `AVG`
/// helper columns collapse into the one user-facing quotient. The
/// output batches carry the user-facing schema.
fn adjust_batches(
    cumulative: &CumulativeRead,
    batches: Vec<arrow_lite::RecordBatch>,
    boundaries: &std::collections::HashMap<Vec<Option<String>>, Vec<Option<Cell>>>,
) -> Vec<arrow_lite::RecordBatch> {
    use arrow_lite::{Bitmap, Column, NumericColumn, NumericData, RecordBatch};
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let rows = batch.num_rows();
        if rows == 0 {
            continue;
        }
        let row_boundary: Vec<Option<&Vec<Option<Cell>>>> = (0..rows)
            .map(|row| {
                let key: Vec<Option<String>> = cumulative
                    .partition_assembly
                    .iter()
                    .map(|&column| match &batch.columns()[column] {
                        Column::Key(keys) => keys.value_at(row).map(str::to_owned),
                        _ => None,
                    })
                    .collect();
                boundaries.get(&key)
            })
            .collect();
        let boundary = |row: usize, index: usize| -> Option<Cell> {
            row_boundary[row].and_then(|cells| cells[index])
        };
        let columns: Vec<Column> = cumulative
            .adjust
            .iter()
            .map(|step| match step {
                AdjustStep::Pass(index) => batch.columns()[*index].clone(),
                AdjustStep::Add {
                    column,
                    boundary: b,
                } => folded(
                    &batch.columns()[*column],
                    rows,
                    |row| boundary(row, *b),
                    Fold::Add,
                ),
                AdjustStep::MinFold {
                    column,
                    boundary: b,
                } => folded(
                    &batch.columns()[*column],
                    rows,
                    |row| boundary(row, *b),
                    Fold::Min,
                ),
                AdjustStep::MaxFold {
                    column,
                    boundary: b,
                } => folded(
                    &batch.columns()[*column],
                    rows,
                    |row| boundary(row, *b),
                    Fold::Max,
                ),
                AdjustStep::AvgAssemble {
                    sum,
                    count,
                    boundary_sum,
                    boundary_count,
                } => {
                    let Column::Numeric(NumericData::F64(sums)) = &batch.columns()[*sum] else {
                        unreachable!("an expanding SUM window is f64")
                    };
                    let Column::Numeric(NumericData::I64(counts)) = &batch.columns()[*count] else {
                        unreachable!("an expanding COUNT window is i64")
                    };
                    let mut values = Vec::with_capacity(rows);
                    let mut validity = Vec::with_capacity(rows);
                    for row in 0..rows {
                        let assembled_count = if counts.is_valid(row) {
                            counts.values().as_slice()[row]
                        } else {
                            0
                        };
                        let total_count = assembled_count
                            + boundary(row, *boundary_count)
                                .map(Cell::as_i64)
                                .unwrap_or(0);
                        let assembled = sums.is_valid(row).then(|| sums.values().as_slice()[row]);
                        let total_sum =
                            match (assembled, boundary(row, *boundary_sum).map(Cell::as_f64)) {
                                (Some(a), Some(b)) => Some(a + b),
                                (one, other) => one.or(other),
                            };
                        let defined = total_count > 0 && total_sum.is_some();
                        validity.push(defined);
                        values.push(if defined {
                            total_sum.expect("checked") / total_count as f64
                        } else {
                            0.0
                        });
                    }
                    Column::Numeric(NumericData::F64(NumericColumn::new_nullable(
                        values.into_iter().collect(),
                        Bitmap::from_bools(validity),
                    )))
                }
            })
            .collect();
        out.push(RecordBatch::new(cumulative.output.clone(), columns));
    }
    out
}

/// How an assembled window value folds with its boundary.
#[derive(Clone, Copy)]
enum Fold {
    Add,
    Min,
    Max,
}

/// Folds one assembled window column with its per-row boundary cells,
/// in the column's own numeric domain. A row missing either side keeps
/// the other; a row missing both is NULL.
fn folded(
    column: &arrow_lite::Column,
    rows: usize,
    boundary: impl Fn(usize) -> Option<Cell>,
    fold: Fold,
) -> arrow_lite::Column {
    use arrow_lite::{Bitmap, Column, NumericColumn, NumericData};
    match column {
        Column::Numeric(NumericData::F64(assembled)) => {
            let mut values = Vec::with_capacity(rows);
            let mut validity = Vec::with_capacity(rows);
            for row in 0..rows {
                let own = assembled
                    .is_valid(row)
                    .then(|| assembled.values().as_slice()[row]);
                let other = boundary(row).map(Cell::as_f64);
                let combined = match (own, other) {
                    (Some(a), Some(b)) => Some(match fold {
                        Fold::Add => a + b,
                        // The engine's comparison relation places NaN
                        // GREATER than every number (M5.0's total
                        // order), and both reference computations — the
                        // boundary's MAX aggregate and the assembled
                        // MAX window — propagate it. Rust's `f64::max`
                        // silently drops NaN, which would make the
                        // answer depend on the query's WHERE clause;
                        // `min` under the same relation correctly
                        // prefers the number, which Rust's `min` also
                        // does.
                        Fold::Min => a.min(b),
                        Fold::Max => {
                            if a.is_nan() || b.is_nan() {
                                f64::NAN
                            } else {
                                a.max(b)
                            }
                        }
                    }),
                    (one, other) => one.or(other),
                };
                validity.push(combined.is_some());
                values.push(combined.unwrap_or(0.0));
            }
            Column::Numeric(NumericData::F64(NumericColumn::new_nullable(
                values.into_iter().collect(),
                Bitmap::from_bools(validity),
            )))
        }
        Column::Numeric(NumericData::I64(assembled)) => {
            let mut values = Vec::with_capacity(rows);
            let mut validity = Vec::with_capacity(rows);
            for row in 0..rows {
                let own = assembled
                    .is_valid(row)
                    .then(|| assembled.values().as_slice()[row]);
                let other = boundary(row).map(Cell::as_i64);
                let combined = match (own, other) {
                    (Some(a), Some(b)) => Some(match fold {
                        Fold::Add => a + b,
                        Fold::Min => a.min(b),
                        Fold::Max => a.max(b),
                    }),
                    (one, other) => one.or(other),
                };
                validity.push(combined.is_some());
                values.push(combined.unwrap_or(0));
            }
            Column::Numeric(NumericData::I64(NumericColumn::new_nullable(
                values.into_iter().collect(),
                Bitmap::from_bools(validity),
            )))
        }
        Column::Key(_) => unreachable!("window outputs are numeric"),
    }
}

/// The one scratch runner every view read funnels through: `user_plan`
/// over ad-hoc batches as resident scratch segments, ordering key at
/// `okey`. Per-batch orderedness is inspected, never assumed: a stale
/// union can interleave ranges, and the executor's own ordering checks
/// then govern — the same stance every disordered table gets.
///
/// The registry rule, uniform across shapes and staleness: a view read
/// resolves registered functions from the VIEW's own registry, which
/// is always empty — a view has no registration surface
/// (`register_window` targets tables), so a custom kernel in a query
/// against a view is refused identically whether the view is fresh,
/// stale, or recomputing. Register on the base and query the base;
/// a per-view registration surface is a held seat, not an accident.
fn run_over_scratch(
    output: &Schema,
    okey: usize,
    batches: Vec<arrow_lite::RecordBatch>,
    user_plan: &Plan,
    registry: &query_lite::Registry,
) -> Result<query_lite::QueryOutput, EngineError> {
    use storage_lite::{Segment, SegmentHandle};
    let handles: Vec<SegmentHandle> = batches
        .into_iter()
        .filter(|batch| batch.num_rows() > 0)
        .map(|batch| {
            let ordered = is_non_decreasing(&batch, okey);
            SegmentHandle::resident(
                std::sync::Arc::new(Segment::from_batch_unpruned(batch, okey, ordered)),
                None,
            )
        })
        .collect();
    query_lite::execute_with_ordering_key(output, &handles, okey, user_plan, registry)
        .map_err(EngineError::Query)
}

/// A plan projecting every column of `table`, built structurally — an
/// unaliased bucket's column name (`ts / 4`) cannot round-trip through
/// SQL text, where it would parse as arithmetic.
fn select_everything(table: &Table) -> Result<Plan, EngineError> {
    Ok(Plan {
        table: table.name().to_owned(),
        join: None,
        projection: query_lite::Projection::Items(
            table
                .schema()
                .fields()
                .iter()
                .map(|field| query_lite::PlanItem::Column {
                    name: field.name().to_owned(),
                    alias: None,
                })
                .collect(),
        ),
        distinct: false,
        predicate: None,
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
    })
}

/// Whether `batch`'s column `index` is non-decreasing — the honest
/// per-batch orderedness of a scratch segment.
fn is_non_decreasing(batch: &arrow_lite::RecordBatch, index: usize) -> bool {
    use arrow_lite::{Column, NumericData};
    let Column::Numeric(NumericData::I64(column)) = &batch.columns()[index] else {
        return false;
    };
    column.values().as_slice().windows(2).all(|w| w[0] <= w[1])
}

/// Lowers and validates a view definition against its source, returning
/// the **materialization** table's schema, its ordering-key column
/// name — the definition's own bucket for a bucketed view, the hidden
/// bucket of the partials for a running or cumulative one — and the
/// **answer** schema, the shape queries against the view return (the
/// materialization's own for a bucketed view; the user definition's
/// for the partials shapes, whose materialization stores internal
/// columns no query ever answers with).
#[allow(clippy::type_complexity)]
fn validated_definition(
    sql: &str,
    source: &Table,
    dimension: Option<&Table>,
) -> Result<(Schema, String, Schema, Option<JoinState>), EngineError> {
    let plan = lower_plan(sql).map_err(EngineError::Query)?;
    if plan.table != source.name() {
        return Err(EngineError::WrongTable {
            expected: source.name().to_owned(),
            got: plan.table,
        });
    }
    match (&plan.join, dimension) {
        (Some(join), Some(dimension)) if join.dimension != dimension.name() => {
            return Err(EngineError::WrongTable {
                expected: join.dimension.clone(),
                got: dimension.name().to_owned(),
            });
        }
        (None, Some(dimension)) => {
            return Err(definition_error(format!(
                "'{}' passed as a dimension, but the definition joins nothing",
                dimension.name()
            )));
        }
        _ => {} // a joined plan with no dimension is refused by the door
    }
    // A placeholder width of 1 keeps a running synthesis well-formed;
    // the schema does not depend on the width's value.
    let definition = Definition::of(sql, source, dimension, 1)?;
    let bucket = definition.bucket_name.clone();
    let schema = match (&definition.read, dimension) {
        (ReadShape::Joined, Some(dimension)) => {
            source
                .execute_join_plan_empty(&definition.plan, dimension)?
                .schema
        }
        _ => output_schema(&definition.plan, source)?,
    };
    // The bucket column is the view table's ordering key; the executor
    // may mark aggregate outputs nullable, but a bucket of a NOT NULL
    // ordering key is never null, and Table::new requires NOT NULL.
    let fields: Vec<Field> = schema
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
    let materialization = Schema::new(fields);
    let answers = match &definition.read {
        ReadShape::Direct | ReadShape::Joined => materialization.clone(),
        ReadShape::Running(running) => source.execute_plan_empty(&running.user)?.schema,
        ReadShape::Cumulative(cumulative) => source.execute_plan_empty(&cumulative.user)?.schema,
    };
    // A fresh join view starts with nothing folded on either axis:
    // both stamps 0, the ceiling at i64::MIN (the constructors persist
    // this; `open` replaces it with the record's stored state).
    let join = match &definition.read {
        ReadShape::Joined => Some(JoinState {
            dimension: dimension.expect("checked above").name().to_owned(),
            stamp: 0,
            ceiling: i64::MIN,
        }),
        _ => None,
    };
    Ok((materialization, bucket, answers, join))
}

/// What kind of maintainable definition a plan is.
enum Shape {
    /// Tranche 1: a bucketed aggregate — the materialization IS the
    /// answer, keyed on the definition's own bucket.
    Bucketed(GroupKey, String),
    /// Tranche 2: a running aggregate — no bucket in the definition,
    /// so the materialization stores per-hidden-bucket **partials**
    /// and the answer is assembled at read by combining them.
    Running,
    /// Tranche 2's remainder: a cumulative window — one output row per
    /// source row, each carrying an expanding aggregate. The same
    /// partials, read as per-partition **boundary** values.
    Cumulative,
    /// Tranche 3, cycle 1: a bare as-of join, carrying the fact
    /// ordering key's OUTPUT name (the view table's ordering key).
    Joined(String),
    /// Tranche 3, cycle 2: a bucketed aggregate over an as-of join —
    /// the tranche-1 shape whose FROM is a join. Carries the bucket
    /// term and its output name.
    JoinedBucketed(GroupKey, String),
}

/// The eligibility check: classifies a definition as bucketed
/// (tranche 1) or running (tranche 2), and refuses everything else by
/// name — naming the tranche that will admit it where one is planned.
fn eligible_shape(
    plan: &Plan,
    source: &Table,
    dimension: Option<&Table>,
) -> Result<Shape, EngineError> {
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
        return classify_joined(plan, source, dimension);
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
        return classify_cumulative(plan, source);
    };
    if having.is_some() {
        return refuse(
            "HAVING in a view definition — a view stores every group; \
             filter at read",
        );
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
                     definition — group by symbols and at most one bucket \
                     of the ordering key",
                );
            }
        }
    }
    let mut bucket_terms = keys.iter().filter(|key| {
        matches!(key, GroupKey::Bucket { .. })
            || matches!(key, GroupKey::Column(column) if column == source.ordering_key())
    });
    let Some(bucket) = bucket_terms.next() else {
        // No bucket: a RUNNING aggregate — totals per symbol group, or
        // one global row. The materialization stores per-hidden-bucket
        // partials, so a correction's blast radius is one hidden
        // bucket, never the whole answer. The reserved names the
        // synthesis will add must be free.
        reserved_names_free(plan, "running")?;
        // Every group key must be a SELECT output: the combine
        // reassembles groups from the materialization's stored
        // columns, so a key the materialization does not store cannot
        // be grouped on again (found by the repo-wide code review —
        // the old code accepted the shape and every partials-path
        // read then failed on the missing column).
        for key in keys {
            let selected = items
                .iter()
                .any(|item| matches!(item, query_lite::AggItem::Key { key: k, .. } if k == key));
            if !selected {
                return refuse(
                    "a running view whose SELECT list omits a GROUP BY key — \
                     the combine reads keys from the materialization, so \
                     select every key (alias it to taste)",
                );
            }
        }
        return Ok(Shape::Running);
    };
    if bucket_terms.next().is_some() {
        return refuse("two buckets of the ordering key in one GROUP BY");
    }
    if let GroupKey::Bucket {
        divide,
        multiply: Some(multiply),
        ..
    } = bucket
    {
        if multiply != divide {
            // Found by the repo-wide code review: the executor
            // multiplies unchecked, the view's range inverse
            // saturates, and a mismatched multiplier is where the two
            // could disagree at i64's edge. A bucket start multiplies
            // back by its own width; anything else is refused.
            return refuse(
                "a bucket whose multiplier differs from its width in a view \
                 definition — a bucket start is (ts / w) * w, same w",
            );
        }
    }
    // The bucket is the view table's ordering key, so it must be a
    // SELECT output — and its output name is the alias when the query
    // wrote one.
    let name = items
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
        })?;
    Ok(Shape::Bucketed(bucket.clone(), name))
}

/// Classifies a definition whose FROM clause is a join (#83
/// tranche 3). Bare projections admit the **enriched blotter**: one
/// `ASOF LEFT/INNER JOIN`, the fact ordering key selected (it is the
/// view's axis) — a view must fold or match something, and the
/// blotter materializes the match. Aggregate projections route to
/// [`classify_joined_aggregate`] (ASOF and equi both).
fn classify_joined(
    plan: &Plan,
    source: &Table,
    dimension: Option<&Table>,
) -> Result<Shape, EngineError> {
    use query_lite::PlanItem;
    let refuse = |what: &str| Err(EngineError::Query(QueryError::Unsupported(what.to_owned())));
    let join = plan.join.as_ref().expect("routed here on Some");
    let Some(dimension) = dimension else {
        return Err(definition_error(format!(
            "a join view needs its dimension table: pass '{}'",
            join.dimension
        )));
    };
    if plan.distinct {
        return refuse("DISTINCT in a view definition — deduplicate at read");
    }
    if plan.order_by.is_some() || plan.limit.is_some() || plan.offset.is_some() {
        return refuse(
            "ORDER BY / LIMIT / OFFSET in a view definition — a view is a \
             table; order and limit at read, where they compose",
        );
    }
    let Projection::Items(items) = &plan.projection else {
        return classify_joined_aggregate(plan, source, dimension);
    };
    for item in items {
        match item {
            PlanItem::Column { .. } => {}
            PlanItem::Window { .. } => {
                return refuse(
                    "a window over a join in a view definition — windows \
                     compose at read, over the blotter",
                )
            }
            PlanItem::Computed { .. } => {
                return refuse(
                    "a computed expression in a join view definition — \
                     select columns; compose expressions at read",
                )
            }
        }
    }
    // The fact ordering key is the view's axis: it must be selected,
    // and its output name (the alias, if any) is the view table's
    // ordering key.
    let okey = items
        .iter()
        .find_map(|item| match item {
            PlanItem::Column { name, alias } if name == source.ordering_key() => {
                Some(alias.clone().unwrap_or_else(|| name.clone()))
            }
            _ => None,
        })
        .ok_or_else(|| {
            EngineError::Query(QueryError::Unsupported(
                "a join view whose SELECT list omits the fact ordering key — \
                 it is the view's axis, so select it (alias it to taste)"
                    .to_owned(),
            ))
        })?;
    // The dimension must be keyed... for the AS OF shape it is a
    // history (many rows per key) — but its ordering key must exist
    // and differ from the fact's by name (the executor refuses the
    // clash; failing here keeps the error at the definition door).
    if dimension.ordering_key() == source.ordering_key() {
        return refuse(
            "a join view whose two tables share an ordering-key NAME — \
             rename the dimension's (the as-of executor refuses the clash)",
        );
    }
    Ok(Shape::Joined(okey))
}

/// Classifies a bucketed aggregate over an as-of join (#83 tranche 3,
/// cycle 2): the tranche-1 shape — one bucket of the FACT ordering key
/// plus symbol keys, built aggregates, optional row-local WHERE — with
/// the FROM clause a shipped ASOF join. Group keys and aggregate
/// arguments may name either side's columns (a dimension attribute as
/// a group key rides the same interval repair: a quote correction
/// re-folds its buckets whole).
fn classify_joined_aggregate(
    plan: &Plan,
    source: &Table,
    dimension: &Table,
) -> Result<Shape, EngineError> {
    let refuse = |what: &str| Err(EngineError::Query(QueryError::Unsupported(what.to_owned())));
    let Projection::Aggregate {
        keys,
        items,
        having,
    } = &plan.projection
    else {
        unreachable!("routed here from the Aggregate arm")
    };
    if having.is_some() {
        return refuse(
            "HAVING in a view definition — a view stores every group; \
             filter at read",
        );
    }
    let symbol_on_either = |column: &str| {
        source
            .schema()
            .fields()
            .iter()
            .chain(dimension.schema().fields().iter())
            .any(|field| field.name() == column && field.column_type() == ColumnType::Key)
    };
    for key in keys {
        if let GroupKey::Column(column) = key {
            if column != source.ordering_key() && !symbol_on_either(column) {
                return refuse(
                    "a non-symbol, non-bucket GROUP BY key in a join view \
                     definition — group by symbols (either side's) and at \
                     most one bucket of the fact ordering key",
                );
            }
        }
    }
    let mut bucket_terms = keys.iter().filter(|key| {
        matches!(key, GroupKey::Bucket { column, .. } if column == source.ordering_key())
            || matches!(key, GroupKey::Column(column) if column == source.ordering_key())
    });
    let Some(bucket) = bucket_terms.next() else {
        return refuse(
            "an aggregate over a join with no bucket of the fact ordering \
             key — running and cumulative shapes over joins hold their \
             seats; bucket the fact axis, or maintain a single-table view",
        );
    };
    if bucket_terms.next().is_some() {
        return refuse("two buckets of the ordering key in one GROUP BY");
    }
    if keys.iter().any(
        |key| matches!(key, GroupKey::Bucket { column, .. } if column != source.ordering_key()),
    ) {
        return refuse(
            "a bucket of a non-fact column in a join view definition — the \
             view's axis is the fact ordering key",
        );
    }
    if let GroupKey::Bucket {
        divide,
        multiply: Some(multiply),
        ..
    } = bucket
    {
        if multiply != divide {
            return refuse(
                "a bucket whose multiplier differs from its width in a view \
                 definition — a bucket start is (ts / w) * w, same w",
            );
        }
    }
    let name = items
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
        })?;
    Ok(Shape::JoinedBucketed(bucket.clone(), name))
}

/// Refuses a running or cumulative definition that touches the `__`
/// name space — as columns it reads or as aliases it writes. The
/// synthesis mints hidden columns there (`__bucket`, `__p{i}`,
/// `__row`, `__w{i}_sum` / `__w{i}_count`), and a user name shadowing
/// a minted one produces a view that creates and refreshes fine but
/// can never be read (found by the repo-wide code review, which
/// probed exactly that). One prefix rule covers every minted name,
/// present and future; bucketed views mint nothing and keep the wider
/// name space.
fn reserved_names_free(plan: &Plan, shape: &str) -> Result<(), EngineError> {
    let mut names: Vec<String> = plan.referenced_columns().into_iter().collect();
    match &plan.projection {
        Projection::Aggregate { items, .. } => {
            for item in items {
                match item {
                    query_lite::AggItem::Key {
                        alias: Some(alias), ..
                    } => names.push(alias.clone()),
                    query_lite::AggItem::Call(call) => names.extend(call.alias.clone()),
                    query_lite::AggItem::Key { alias: None, .. } => {}
                }
            }
        }
        Projection::Items(items) => {
            for item in items {
                match item {
                    query_lite::PlanItem::Column {
                        alias: Some(alias), ..
                    }
                    | query_lite::PlanItem::Window {
                        alias: Some(alias), ..
                    } => names.push(alias.clone()),
                    query_lite::PlanItem::Computed { name, .. } => names.push(name.clone()),
                    _ => {}
                }
            }
        }
    }
    match names.iter().find(|name| name.starts_with("__")) {
        None => Ok(()),
        Some(name) => Err(EngineError::Query(QueryError::Unsupported(format!(
            "'{name}' in a {shape} view definition — names beginning with \
             '__' are reserved for the materialization's internal columns"
        )))),
    }
}

/// Classifies a row-per-row projection: a **cumulative** view when it
/// carries expanding windows in the admitted family, a refusal by name
/// otherwise. The requirements mirror tranche 1's "select your bucket"
/// rule: the ordering key must be selected (it is the output's axis
/// and the range-read's handle), and so must every partition symbol
/// (the boundary adjustment looks partitions up by their output
/// values).
fn classify_cumulative(plan: &Plan, source: &Table) -> Result<Shape, EngineError> {
    use crate::partials::expanding_window_function;
    use query_lite::plan::WindowCall;
    use query_lite::PlanItem;
    let refuse = |what: &str| Err(EngineError::Query(QueryError::Unsupported(what.to_owned())));
    let Projection::Items(items) = &plan.projection else {
        unreachable!("classified from an Items projection")
    };
    let mut selected_columns: Vec<&str> = Vec::new();
    let mut windows = 0usize;
    let mut partition_columns: Vec<String> = Vec::new();
    for item in items {
        match item {
            PlanItem::Column { name, .. } => selected_columns.push(name),
            PlanItem::Window { call, .. } => {
                windows += 1;
                let WindowCall::Agg {
                    function,
                    partition_by,
                    order_by,
                    frame,
                    ..
                } = call
                else {
                    return refuse(
                        "LAG/LEAD in a view definition — positional lookups \
                         are not running state; query them over the base",
                    );
                };
                if expanding_window_function(function).is_none() {
                    return refuse(
                        "a window outside sum/count/avg/min/max in a view \
                         definition — only those decompose into bucket \
                         partials today",
                    );
                }
                if !matches!(frame, query_lite::Frame::Rows(None)) {
                    return refuse(
                        "a bounded window frame in a view definition — a \
                         maintained view holds running state; rolling \
                         windows derive at read (from the base, or by \
                         differencing a cumulative view)",
                    );
                }
                if order_by.as_deref() != Some(source.ordering_key()) {
                    return refuse(
                        "a cumulative window not ordered by the ordering key \
                         in a view definition",
                    );
                }
                for term in partition_by {
                    match term {
                        GroupKey::Column(column) if column != source.ordering_key() => {
                            if !partition_columns.contains(column) {
                                partition_columns.push(column.clone());
                            }
                        }
                        _ => {
                            return refuse(
                                "a cross-sectional partition in a cumulative \
                                 view definition — partition by symbols; the \
                                 instant direction is not running state",
                            )
                        }
                    }
                }
            }
            PlanItem::Computed { .. } => {
                return refuse(
                    "a computed expression in a cumulative view definition — \
                     select columns and whole windows; compose expressions \
                     at read",
                )
            }
        }
    }
    if windows == 0 {
        return refuse(
            "a row-per-row view with no window — a maintained view \
             maintains aggregates; a plain projection is just a query",
        );
    }
    if !selected_columns
        .iter()
        .any(|name| *name == source.ordering_key())
    {
        return refuse(
            "a cumulative view whose SELECT list omits the ordering key — \
             it is the output's axis, so select it",
        );
    }
    for partition in &partition_columns {
        if !selected_columns.iter().any(|name| name == partition) {
            return refuse(
                "a cumulative view whose SELECT list omits a partition \
                 symbol — the boundary adjustment reads it from the output, \
                 so select it",
            );
        }
    }
    // Every window must agree on its partitioning: the boundary is
    // computed once per partition combination.
    for item in items {
        if let PlanItem::Window {
            call: WindowCall::Agg { partition_by, .. },
            ..
        } = item
        {
            let mut named: Vec<&str> = partition_by
                .iter()
                .filter_map(|term| match term {
                    GroupKey::Column(column) => Some(column.as_str()),
                    _ => None,
                })
                .collect();
            named.sort_unstable();
            let mut expected: Vec<&str> = partition_columns.iter().map(String::as_str).collect();
            expected.sort_unstable();
            if named != expected {
                return refuse(
                    "cumulative windows with different PARTITION BY \
                     lists in one view definition — one partitioning \
                     per view; split the view",
                );
            }
        }
    }
    reserved_names_free(plan, "cumulative")?;
    Ok(Shape::Cumulative)
}

/// The bucket term's arithmetic: `(divide, view_scale)`. `divide` maps
/// an ordering-key value to its bucket index with the executor's own
/// truncating `/`; `view_scale` maps a bucket index to the value the
/// view's ordering-key column stores — the multiplier for a
/// bucket-start definition, `1` for a bucket index, and (with
/// `divide = 1`) the identity for a bare-`ts` bucket.
fn bucket_arithmetic(bucket: &GroupKey) -> (i64, i64) {
    match bucket {
        GroupKey::Column(_) => (1, 1),
        GroupKey::Bucket {
            divide, multiply, ..
        } => (*divide, multiply.unwrap_or(1)),
    }
}

/// The smallest ordering-key value in bucket `index` under truncating
/// division. Truncation makes bucket 0 double-width — every value in
/// `(-divide, divide)` truncates to 0 — so its low edge is negative.
/// Saturating: an edge clamped at i64's end can only widen the range,
/// and a wider range only re-folds more, never wrongly.
fn bucket_low(index: i64, divide: i64) -> i64 {
    match index.cmp(&0) {
        std::cmp::Ordering::Greater => index.saturating_mul(divide),
        std::cmp::Ordering::Equal => -(divide - 1),
        std::cmp::Ordering::Less => index.saturating_mul(divide).saturating_sub(divide - 1),
    }
}

/// The largest ordering-key value in bucket `index` — `bucket_low`'s
/// mirror.
fn bucket_high(index: i64, divide: i64) -> i64 {
    match index.cmp(&0) {
        std::cmp::Ordering::Greater => index.saturating_mul(divide).saturating_add(divide - 1),
        std::cmp::Ordering::Equal => divide - 1,
        std::cmp::Ordering::Less => index.saturating_mul(divide),
    }
}

/// Collapses a sorted bucket set into maximal `(first, last)` runs of
/// consecutive indices, so a burst of appends becomes one range
/// predicate instead of one arm per bucket.
fn contiguous_runs(buckets: &std::collections::BTreeSet<i64>) -> Vec<(i64, i64)> {
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for &bucket in buckets {
        match runs.last_mut() {
            Some((_, last)) if *last + 1 == bucket => *last = bucket,
            _ => runs.push((bucket, bucket)),
        }
    }
    runs
}

/// An OR-chain of closed ranges over one i64 column — the prunable
/// shape, so zone maps skip every segment outside the touched buckets.
fn ranges_predicate(column: &str, ranges: impl Iterator<Item = (i64, i64)>) -> Predicate {
    let arm = |(low, high): (i64, i64)| {
        Predicate::And(
            Box::new(Predicate::Compare {
                column: column.to_owned(),
                op: CmpOp::Ge,
                value: Number::Int(low),
            }),
            Box::new(Predicate::Compare {
                column: column.to_owned(),
                op: CmpOp::Le,
                value: Number::Int(high),
            }),
        )
    };
    let mut arms = ranges.map(arm);
    let first = arms.next().expect("at least one touched bucket");
    arms.fold(first, |or, next| {
        Predicate::Or(Box::new(or), Box::new(next))
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

/// The definition record: `b"TDBV"`, a format version, the stamp, the
/// hidden-bucket width (version 2's addition; 0 = unchosen or unused),
/// then length-prefixed source name and SQL, then CRC-32C of
/// everything before it. Little-endian throughout, like the segment
/// format. Version 1 records (no width field) decode with width 0,
/// which self-heals: a running view's first refresh chooses one.
/// Version 3 (#83 tranche 3) is the JOIN-view form: after the SQL it
/// carries the dimension stamp, the materialization ceiling, and the
/// length-prefixed dimension name. Single-source views keep writing
/// v2 — the version IS the source count, and an old binary meeting a
/// v3 record refuses loudly instead of misreading it.
fn encode_definition(
    stamp: u64,
    width: u64,
    source: &str,
    sql: &str,
    join: Option<&JoinState>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 2 + 8 + 8 + 8 + source.len() + sql.len() + 24 + 4);
    out.extend_from_slice(b"TDBV");
    let version: u16 = if join.is_some() { 3 } else { 2 };
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&stamp.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&(source.len() as u32).to_le_bytes());
    out.extend_from_slice(source.as_bytes());
    out.extend_from_slice(&(sql.len() as u32).to_le_bytes());
    out.extend_from_slice(sql.as_bytes());
    if let Some(join) = join {
        out.extend_from_slice(&join.stamp.to_le_bytes());
        out.extend_from_slice(&join.ceiling.to_le_bytes());
        out.extend_from_slice(&(join.dimension.len() as u32).to_le_bytes());
        out.extend_from_slice(join.dimension.as_bytes());
    }
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

#[allow(clippy::type_complexity)]
fn decode_definition(
    bytes: &[u8],
) -> Result<(u64, u64, String, String, Option<JoinState>), EngineError> {
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
    if !(1..=3).contains(&version) {
        return Err(corrupt(&format!("unknown version {version}")));
    }
    let stamp = u64::from_le_bytes(payload[6..14].try_into().expect("sized"));
    let (width, mut at) = if version >= 2 {
        if payload.len() < 22 {
            return Err(corrupt("truncated width"));
        }
        (
            u64::from_le_bytes(payload[14..22].try_into().expect("sized")),
            22usize,
        )
    } else {
        (0, 14usize)
    };
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
    let join = if version == 3 {
        let fixed_end = at.checked_add(16).filter(|&e| e <= payload.len());
        let Some(fixed_end) = fixed_end else {
            return Err(corrupt("truncated join state"));
        };
        let dim_stamp = u64::from_le_bytes(payload[at..at + 8].try_into().expect("sized"));
        let ceiling = i64::from_le_bytes(payload[at + 8..fixed_end].try_into().expect("sized"));
        at = fixed_end;
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
        let dimension = read_string("dimension name")?;
        Some(JoinState {
            dimension,
            stamp: dim_stamp,
            ceiling,
        })
    } else {
        None
    };
    Ok((stamp, width, source, sql, join))
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
        let view = MaterializedView::new("ohlc", OHLC, &source, None).unwrap();
        assert_eq!(view.name(), "ohlc");
        assert_eq!(view.source(), "trades");
        assert_eq!(view.sql(), OHLC);
        // Nothing folded yet: the stamp is zero and the
        // materialization empty — create is O(1), the first refresh
        // pays for the backlog.
        assert_eq!(view.stamp(), 0);
        assert_eq!(
            view.table.query("SELECT o FROM ohlc").unwrap().num_rows(),
            0
        );
        // The table's shape came from the executor: the bucket alias
        // is the ordering key, the aggregates are columns.
        let schema = view.table.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name()).collect();
        assert_eq!(names, ["sym", "bar", "o", "h", "l", "c"]);
        assert_eq!(view.table.ordering_key(), "bar");
        // The bucket start spelling and a bare-ts bucket are accepted
        // too, and an unaliased bucket keeps its arithmetic name.
        MaterializedView::new(
            "bars",
            "SELECT (ts / 4) * 4, sum(x) FROM trades GROUP BY (ts / 4) * 4",
            &source,
            None,
        )
        .unwrap();
        MaterializedView::new(
            "instants",
            "SELECT ts, count(*) AS n FROM trades GROUP BY ts",
            &source,
            None,
        )
        .unwrap();
    }

    #[test]
    fn ineligible_definitions_are_refused_by_name() {
        let source = source();
        let refused = |sql: &str, needle: &str| {
            let error = MaterializedView::new("v", sql, &source, None)
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
        // A row-per-row projection with no window maintains nothing.
        // (The no-bucket aggregate tranche 1 refused is now the
        // RUNNING shape, and windowed projections the CUMULATIVE one
        // — both accepted, tested in their own batteries.)
        refused("SELECT x FROM trades", "no window");
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
            None,
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
            None,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("elsewhere"), "{error}");
    }

    #[test]
    fn join_definitions_meet_the_tranche_3_door() {
        // The join door (#83 tranche 3, cycle 1): a joined definition
        // without its dimension table is told what to pass; an
        // EQUI-join is refused by name until the star cycle; an
        // aggregate over the ASOF join is refused by name until the
        // next cycle.
        let source = source();
        let dim = Table::new(
            "dim",
            arrow_lite::Schema::new(vec![
                arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                arrow_lite::Field::new("w", arrow_lite::ColumnType::F64, false),
            ]),
            "qts",
        )
        .unwrap();
        let error = MaterializedView::new(
            "v",
            "SELECT ts / 4 AS b, sum(w) AS s FROM trades \
             JOIN dim ON trades.sym = dim.sym GROUP BY ts / 4",
            &source,
            None,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("pass 'dim'"), "{error}");
        // Cycle 3 ADMITS the equi (star) shape under the widened door
        // — its repair is rebuild-on-dim-change, tested in the star
        // battery.
        MaterializedView::new(
            "v0",
            "SELECT ts / 4 AS b, sum(w) AS s FROM trades \
             JOIN dim ON trades.sym = dim.sym GROUP BY ts / 4",
            &source,
            Some(&dim),
        )
        .unwrap();
        // Cycle 2 ADMITS the bucketed aggregate over the ASOF join —
        // a dimension-valued aggregate included.
        MaterializedView::new(
            "v",
            "SELECT ts / 4 AS b, sum(w) AS s FROM trades \
             ASOF LEFT JOIN dim ON trades.sym = dim.sym GROUP BY ts / 4",
            &source,
            Some(&dim),
        )
        .unwrap();
        // What stays refused around it, by name.
        let refused = |sql: &str, needle: &str| {
            let error = MaterializedView::new("v", sql, &source, Some(&dim))
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{sql}: {error}");
        };
        refused(
            "SELECT sym, sum(w) AS s FROM trades \
             ASOF LEFT JOIN dim ON trades.sym = dim.sym GROUP BY sym",
            "no bucket of the fact ordering key",
        );
        refused(
            "SELECT qts / 4 AS b, sum(w) AS s FROM trades \
             ASOF LEFT JOIN dim ON trades.sym = dim.sym GROUP BY qts / 4",
            "no bucket of the fact ordering key",
        );
        refused(
            "SELECT ts / 4 AS b, sum(w) AS s FROM trades \
             ASOF LEFT JOIN dim ON trades.sym = dim.sym GROUP BY ts / 4 \
             HAVING sum(w) > 1",
            "HAVING",
        );
    }

    #[test]
    fn a_persistent_view_reopens_with_its_definition_and_stamp() {
        let dir = std::env::temp_dir().join(format!("tallydb-view-def-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = source();
        {
            let view = MaterializedView::persistent("ohlc", OHLC, &source, None, &dir).unwrap();
            assert_eq!(view.stamp(), 0);
        }
        let reopened =
            MaterializedView::open("ohlc", &dir, &source, None, StoreOptions::default()).unwrap();
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
        let error = MaterializedView::open("ohlc", &dir, &source, None, StoreOptions::default())
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        // And a view opened against the wrong source is refused by
        // name, not answered wrongly.
        std::fs::write(&path, encode_definition(0, 0, "quotes", OHLC, None)).unwrap();
        let error = MaterializedView::open("ohlc", &dir, &source, None, StoreOptions::default())
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
        // Querying the view answers exactly even before any refresh:
        // the union read's live half covers everything the stamp does
        // not, which at stamp 0 is the whole answer.
        assert_eq!(db.query("SELECT o FROM ohlc").unwrap().num_rows(), 6);
        // The all-views doorway folds it (and is otherwise exercised
        // nowhere else — one real call keeps it honest).
        db.refresh_views().unwrap();
        assert!(db.view("ohlc").unwrap().stamp() > 0);
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
        assert!(error.contains("view as a join OPERAND"), "{error}");
        // A view over a missing source cannot be added.
        let orphan_source = Table::new("orphan", m1_schema(), "ts").unwrap();
        let orphan = MaterializedView::new(
            "v2",
            "SELECT ts, count(*) AS n FROM orphan GROUP BY ts",
            &orphan_source,
            None,
        )
        .unwrap();
        assert!(matches!(
            db.add_view(orphan).map(|_| ()).unwrap_err(),
            EngineError::UnknownTable(_)
        ));
    }

    /// Every row of `output`, rendered to a sortable string — the
    /// comparison currency for "view equals recompute": the view table
    /// may hold its rows in refresh order, so equality is up to row
    /// order, never up to values.
    fn sorted_rows(output: &query_lite::QueryOutput) -> Vec<String> {
        use arrow_lite::{Column, NumericData};
        let mut rows = Vec::new();
        for batch in &output.batches {
            for row in 0..batch.num_rows() {
                let mut cells = Vec::new();
                for column in batch.columns() {
                    cells.push(match column {
                        Column::Key(keys) => format!("{:?}", keys.value_at(row)),
                        Column::Numeric(NumericData::F64(values)) => {
                            format!("{:?}", values.is_valid(row).then(|| values.values()[row]))
                        }
                        Column::Numeric(NumericData::I64(values)) => {
                            format!("{:?}", values.is_valid(row).then(|| values.values()[row]))
                        }
                    });
                }
                rows.push(cells.join("|"));
            }
        }
        rows.sort();
        rows
    }

    /// The subsuming check of the refresh: after it, the
    /// materialization holds exactly what recomputing the definition
    /// from the source holds.
    fn assert_matches_recompute(db: &Database, view: &str) {
        let sql = db.view(view).unwrap().sql().to_owned();
        let recomputed = db
            .table(db.view(view).unwrap().source())
            .unwrap()
            .query(&sql)
            .unwrap();
        let columns = db
            .view(view)
            .unwrap()
            .table
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let materialized = db.query(&format!("SELECT {columns} FROM {view}")).unwrap();
        assert_eq!(
            sorted_rows(&materialized),
            sorted_rows(&recomputed),
            "view '{view}' diverged from recompute"
        );
    }

    #[test]
    fn refresh_folds_appends_and_only_what_moved() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..12 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("ohlc", OHLC).unwrap();
        // First fold: everything below the watermark, 3 buckets.
        assert_eq!(db.refresh_view("ohlc").unwrap(), 3);
        assert_matches_recompute(&db, "ohlc");
        // Nothing changed: nothing folds, the stamp already covers it.
        assert_eq!(db.refresh_view("ohlc").unwrap(), 0);
        // New appends land in one new bucket: exactly one bucket folds.
        for i in 12..16 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        assert_eq!(db.refresh_view("ohlc").unwrap(), 1);
        assert_matches_recompute(&db, "ohlc");
    }

    #[test]
    fn refresh_repairs_corrections_by_re_folding_their_buckets() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..16 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("ohlc", OHLC).unwrap();
        assert_eq!(db.refresh_view("ohlc").unwrap(), 4);
        // A correction to an already-folded bucket: one bucket refolds
        // and the view converges — uniform repair, no delta arithmetic.
        db.mutate("UPDATE trades SET x = 100.0 WHERE ts = 5")
            .unwrap();
        assert_eq!(db.refresh_view("ohlc").unwrap(), 1);
        assert_matches_recompute(&db, "ohlc");
        // A DELETE that empties a whole bucket: its view rows go too —
        // no ghost group survives.
        db.mutate("DELETE FROM trades WHERE ts >= 12").unwrap();
        assert_eq!(db.refresh_view("ohlc").unwrap(), 1);
        assert_matches_recompute(&db, "ohlc");
        assert_eq!(
            db.query("SELECT c FROM ohlc").unwrap().num_rows(),
            db.table("trades").unwrap().query(OHLC).unwrap().num_rows()
        );
        // A correction that MOVES a row across buckets dirties both:
        // the kill names the old bucket, the reinsert the new one.
        db.mutate("UPDATE trades SET ts = 9 WHERE ts = 1").unwrap();
        assert_eq!(db.refresh_view("ohlc").unwrap(), 2);
        assert_matches_recompute(&db, "ohlc");
        // A DELETE matching nothing spends a coordinate but touches no
        // bucket: the stamp advances, nothing folds.
        db.mutate("DELETE FROM trades WHERE ts = 9999").unwrap();
        assert_eq!(db.refresh_view("ohlc").unwrap(), 0);
    }

    #[test]
    fn bucket_ranges_are_exact_under_truncating_division() {
        // Truncating `/` makes bucket 0 double-width — every value in
        // (-4, 4) truncates to 0 — and negative buckets sit on the
        // other side of their multiples. The fold must agree with the
        // executor over exactly these edges, so the source carries
        // negative, zero, and positive keys spanning all three cases.
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for ts in [-9i64, -7, -4, -3, -1, 0, 2, 3, 4, 7, 8] {
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::Key("A"),
                    storage_lite::RowValue::F64(ts as f64),
                    storage_lite::RowValue::F64(0.0),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view(
            "bars",
            "SELECT ts / 4 AS bar, count(*) AS n, sum(x) AS s FROM trades GROUP BY ts / 4",
        )
        .unwrap();
        db.refresh_view("bars").unwrap();
        assert_matches_recompute(&db, "bars");
        // Corrections on the edges of bucket 0 and a negative bucket.
        db.mutate("UPDATE trades SET x = 50.0 WHERE ts = -3")
            .unwrap();
        db.mutate("DELETE FROM trades WHERE ts = -7").unwrap();
        db.refresh_view("bars").unwrap();
        assert_matches_recompute(&db, "bars");
        // The bucket-start spelling scales the view-side key by the
        // multiplier; same convergence.
        db.create_materialized_view(
            "starts",
            "SELECT (ts / 4) * 4 AS bar, sum(x) AS s FROM trades GROUP BY (ts / 4) * 4",
        )
        .unwrap();
        db.refresh_view("starts").unwrap();
        assert_matches_recompute(&db, "starts");
        db.mutate("UPDATE trades SET x = -50.0 WHERE ts = 8")
            .unwrap();
        db.refresh_view("starts").unwrap();
        db.refresh_view("bars").unwrap();
        assert_matches_recompute(&db, "starts");
        assert_matches_recompute(&db, "bars");
        // A correction touching ONLY bucket 0, so its run is [0, 0] and
        // both of the double-width bucket's edges are load-bearing —
        // a merged run never consults them, and an edge quietly
        // narrowed to [0, divide) would drop the negative half here.
        db.mutate("UPDATE trades SET x = 9.0 WHERE ts = -1")
            .unwrap();
        assert_eq!(db.refresh_view("bars").unwrap(), 1);
        assert_matches_recompute(&db, "bars");
        // And one touching only a negative bucket, isolating ITS edges.
        db.mutate("UPDATE trades SET x = 9.5 WHERE ts = -9")
            .unwrap();
        assert_eq!(db.refresh_view("bars").unwrap(), 1);
        assert_matches_recompute(&db, "bars");
    }

    #[test]
    fn a_filtered_definition_folds_through_its_own_where() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..12 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view(
            "a_bars",
            "SELECT ts / 4 AS bar, sum(x) AS s FROM trades WHERE sym = 'A' GROUP BY ts / 4",
        )
        .unwrap();
        db.refresh_view("a_bars").unwrap();
        assert_matches_recompute(&db, "a_bars");
        // A correction to a row the WHERE excludes still re-folds its
        // bucket (the touched set is definition-blind), and the re-fold
        // — running the definition — leaves the view right.
        db.mutate("UPDATE trades SET x = 100.0 WHERE ts = 5")
            .unwrap();
        assert_eq!(db.refresh_view("a_bars").unwrap(), 1);
        assert_matches_recompute(&db, "a_bars");
    }

    #[test]
    fn a_stale_stamp_self_heals_across_reopen() {
        // The crash story: the stamp persists only after the
        // materialization it describes, so a crash between a source
        // mutation and the next refresh — or between the refresh's
        // write and its stamp advance — leaves an old stamp, and the
        // next refresh re-derives and re-folds. Simulated exactly:
        // reopen with a stamp rewritten backwards.
        let dir = std::env::temp_dir().join(format!("tallydb-view-heal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source_dir = dir.join("trades");
        let view_dir = dir.join("ohlc");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&view_dir).unwrap();
        let mut source = Table::persistent("trades", m1_schema(), "ts", &source_dir).unwrap();
        for i in 0..12 {
            source.append(&linear_row(i)).unwrap();
        }
        {
            let mut view =
                MaterializedView::persistent("ohlc", OHLC, &source, None, &view_dir).unwrap();
            view.refresh(&mut source, None).unwrap();
            let stamp = view.stamp();
            // Mutate the source; crash before any refresh.
            source
                .mutate("UPDATE trades SET x = 77.0 WHERE ts = 2")
                .unwrap();
            assert!(source.next_sequence() > stamp);
        }
        // Reopen: the stamp is honest about what it covers, and one
        // refresh converges the view.
        let mut view =
            MaterializedView::open("ohlc", &view_dir, &source, None, StoreOptions::default())
                .unwrap();
        assert_eq!(view.refresh(&mut source, None).unwrap(), 1);
        let recomputed = source.query(OHLC).unwrap();
        let view_columns = "sym, bar, o, h, l, c";
        let materialized = view
            .table
            .query(&format!("SELECT {view_columns} FROM ohlc"))
            .unwrap();
        assert_eq!(sorted_rows(&materialized), sorted_rows(&recomputed));
        // And a stamp rewound on disk (the crash-mid-refresh shape) is
        // only ever conservative: the re-fold is idempotent. (Drop
        // first: one writer per store directory.)
        drop(view);
        let record = encode_definition(0, 0, "trades", OHLC, None);
        std::fs::write(view_dir.join(DEFINITION_FILE), record).unwrap();
        let mut view =
            MaterializedView::open("ohlc", &view_dir, &source, None, StoreOptions::default())
                .unwrap();
        assert_eq!(view.stamp(), 0);
        view.refresh(&mut source, None).unwrap();
        let materialized = view
            .table
            .query(&format!("SELECT {view_columns} FROM ohlc"))
            .unwrap();
        assert_eq!(sorted_rows(&materialized), sorted_rows(&recomputed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refresh_walks_frozen_segments_and_compacted_history() {
        // Everything above ran in the write buffer (the default freeze
        // threshold is far larger than these fixtures). This is the
        // segmented shape: 4 rows per segment, so the fold crosses
        // frozen segments, the sequence-end skip has real segments to
        // skip — and, after a compaction lands between the correction
        // and the refresh, the kill lives in a HISTORY segment, which
        // is the history walk's only route to being exercised.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        for i in 0..16 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("ohlc", OHLC).unwrap();
        assert_eq!(db.refresh_view("ohlc").unwrap(), 4);
        assert_matches_recompute(&db, "ohlc");
        // Correction, then compaction, THEN refresh: the pending
        // tombstone is gone — resolved into history — before the
        // refresh ever sees it. The touched bucket must come from the
        // history segment's kill coordinates.
        db.mutate("UPDATE trades SET x = 200.0 WHERE ts = 6")
            .unwrap();
        db.compact("trades").unwrap();
        assert!(db.refresh_view("ohlc").unwrap() >= 1);
        assert_matches_recompute(&db, "ohlc");
        // A pure DELETE compacted before the refresh is the case where
        // the history walk alone is load-bearing: an UPDATE's reinsert
        // touches the same bucket by birth, but a DELETE leaves no
        // reinsert — the kill in the history segment is the ONLY
        // record that its bucket changed.
        db.mutate("DELETE FROM trades WHERE ts = 10").unwrap();
        db.compact("trades").unwrap();
        assert!(db.refresh_view("ohlc").unwrap() >= 1);
        assert_matches_recompute(&db, "ohlc");
        // And the ordinary order — correction, refresh, compaction,
        // refresh — stays converged: the post-compaction refresh
        // re-derives from history what it already folded from the
        // pending map, which over-folds (idempotently) rather than
        // diverging.
        db.mutate("DELETE FROM trades WHERE ts = 11").unwrap();
        assert!(db.refresh_view("ohlc").unwrap() >= 1);
        assert_matches_recompute(&db, "ohlc");
        db.compact("trades").unwrap();
        db.refresh_view("ohlc").unwrap();
        assert_matches_recompute(&db, "ohlc");
        // New appends after all of that land in their own buckets.
        for i in 16..20 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        assert_eq!(db.refresh_view("ohlc").unwrap(), 1);
        assert_matches_recompute(&db, "ohlc");
    }

    #[test]
    fn a_view_read_is_exact_at_every_knowledge_coordinate() {
        // The subsuming property of the union read: whatever
        // interleaving of appends, corrections, deletes, and refreshes
        // has happened, querying the view equals recomputing its
        // definition — at EVERY coordinate, not just after a refresh.
        // The interleaving is pseudo-random but deterministic (a fixed
        // LCG), so a failure replays.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        db.create_materialized_view("ohlc", OHLC).unwrap();
        let mut lcg: u64 = 0xB16B_00B5;
        let mut roll = |sides: u64| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 33) % sides
        };
        let mut next_ts: i64 = 0;
        let mut live: Vec<i64> = Vec::new();
        for step in 0..140 {
            match roll(10) {
                0..=5 => {
                    // Append — mostly forward, occasionally late.
                    let ts = if roll(8) == 0 && next_ts > 4 {
                        next_ts - 3
                    } else {
                        next_ts += 1;
                        next_ts
                    };
                    live.push(ts);
                    db.append("trades", &linear_row(ts)).unwrap();
                }
                6..=7 if !live.is_empty() => {
                    let ts = live[roll(live.len() as u64) as usize];
                    db.mutate(&format!("UPDATE trades SET x = {step}.5 WHERE ts = {ts}"))
                        .unwrap();
                }
                8 if !live.is_empty() => {
                    let index = roll(live.len() as u64) as usize;
                    let ts = live.swap_remove(index);
                    live.retain(|&other| other != ts);
                    db.mutate(&format!("DELETE FROM trades WHERE ts = {ts}"))
                        .unwrap();
                }
                _ => {
                    db.refresh_view("ohlc").unwrap();
                }
            }
            assert_matches_recompute(&db, "ohlc");
        }
        // And one compaction-then-more-churn coda over the same
        // property.
        db.compact("trades").unwrap();
        for step in 0..20 {
            if !live.is_empty() && roll(2) == 0 {
                let ts = live[roll(live.len() as u64) as usize];
                db.mutate(&format!("UPDATE trades SET y = {step}.25 WHERE ts = {ts}"))
                    .unwrap();
            } else {
                next_ts += 1;
                live.push(next_ts);
                db.append("trades", &linear_row(next_ts)).unwrap();
            }
            assert_matches_recompute(&db, "ohlc");
        }
    }

    #[test]
    fn as_of_on_a_view_recomputes_from_the_source() {
        // D2.3 = (i), ruled 2026-08-02: 'view AS OF s' IS the
        // definition over 'base AS OF s'. The materialization reflects
        // only latest knowledge, so it is bypassed — including for
        // cuts the view has never folded, and after corrections whose
        // pre-image no current state holds.
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..8 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        let before = db.table("trades").unwrap().next_sequence() - 1;
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        db.refresh_view("ohlc").map(|_| ()).unwrap_err(); // no view yet
        db.create_materialized_view("ohlc", OHLC).unwrap();
        db.refresh_view("ohlc").unwrap();
        // As of 'before', the correction is unknown: the view answers
        // the ORIGINAL x=3 world, though its materialization holds the
        // corrected one.
        let past_view = db
            .query(&format!(
                "SELECT sym, bar, o, h, l, c FROM ohlc ASOF {before}"
            ))
            .unwrap();
        let past_base = db
            .table("trades")
            .unwrap()
            .query(&format!(
                "SELECT sym, ts / 4 AS bar, first(x) AS o, max(x) AS h, min(x) AS l, \
                 last(x) AS c FROM trades ASOF {before} GROUP BY sym, ts / 4"
            ))
            .unwrap();
        assert_eq!(sorted_rows(&past_view), sorted_rows(&past_base));
        // The corrected world differs from the past one — the cut is
        // real, not vacuous.
        let current = db.query("SELECT sym, bar, o, h, l, c FROM ohlc").unwrap();
        assert_ne!(sorted_rows(&current), sorted_rows(&past_view));
    }

    #[test]
    fn a_read_only_view_answers_exactly_and_refuses_repair() {
        // F4: the union read needs no writes, so a read-only process
        // serves exact view answers over a directory another process
        // maintains — however stale the materialization it finds.
        let dir = std::env::temp_dir().join(format!("tallydb-view-ro-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source_dir = dir.join("trades");
        let view_dir = dir.join("ohlc");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&view_dir).unwrap();
        let mut writer = Table::persistent("trades", m1_schema(), "ts", &source_dir).unwrap();
        for i in 0..12 {
            writer.append(&linear_row(i)).unwrap();
        }
        {
            let mut view =
                MaterializedView::persistent("ohlc", OHLC, &writer, None, &view_dir).unwrap();
            view.refresh(&mut writer, None).unwrap();
        }
        // The writer keeps going — a correction and new rows the
        // reader's materialization has never seen — and flushes.
        writer
            .mutate("UPDATE trades SET x = 300.0 WHERE ts = 2")
            .unwrap();
        for i in 12..14 {
            writer.append(&linear_row(i)).unwrap();
        }
        writer.flush().unwrap();
        // The read-only pair: a stale materialization plus the durable
        // source. The union read converges them without writing a byte.
        let ro_source = Table::open_read_only("trades", &source_dir).unwrap();
        let mut db = Database::new();
        db.add_table(ro_source).unwrap();
        let ro_view =
            MaterializedView::open_read_only("ohlc", &view_dir, db.table("trades").unwrap(), None)
                .unwrap();
        db.add_view(ro_view).unwrap();
        assert_matches_recompute(&db, "ohlc");
        // Repair is the writer's job: a read-only refresh refuses
        // loudly, like every mutation on a read-only table.
        assert!(db.refresh_view("ohlc").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_crash_that_rewinds_the_source_leaves_no_ghost_buckets() {
        // Found by the repo-wide code review. Under WalSync::Off the
        // durability boundary is the flush: a crash loses acknowledged
        // but unflushed appends, rewinding the source's watermark. If
        // a refresh had folded those rows and durably stamped past
        // them, the view would hold buckets whose source rows never
        // durably existed — and the old stamp logic silently ADOPTED
        // the rewound watermark, so no refresh would ever remove the
        // ghosts. The rule now: a persistent view's stamp never
        // exceeds the source's flushed watermark, so what the stamp
        // covers survives any crash the source's own contract admits.
        let dir = std::env::temp_dir().join(format!("tallydb-view-ghost-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source_dir = dir.join("trades");
        let view_dir = dir.join("ohlc");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&view_dir).unwrap();
        let off = storage_lite::StoreOptions {
            wal_sync: storage_lite::WalSync::Off,
            ..Default::default()
        };
        {
            let mut source =
                Table::persistent_with("trades", m1_schema(), "ts", &source_dir, off).unwrap();
            for i in 0..8 {
                source.append(&linear_row(i)).unwrap();
            }
            source.flush().unwrap();
            let mut view =
                MaterializedView::persistent("ohlc", OHLC, &source, None, &view_dir).unwrap();
            view.refresh(&mut source, None).unwrap();
            // Refresh flushes the source, so anything IT saw is
            // durable; the losable tail is what arrives after the
            // last refresh.
            for i in 8..16 {
                source.append(&linear_row(i)).unwrap();
            }
            // The critical refresh: it sees the buffered tail, so its
            // stamp covers it — which is exactly why it must make the
            // tail durable first.
            view.refresh(&mut source, None).unwrap();
            let recomputed = source.query(OHLC).unwrap();
            let via_union = view
                .query_union(
                    &source,
                    None,
                    &lower_plan("SELECT sym, bar, o, h, l, c FROM ohlc").unwrap(),
                )
                .unwrap();
            assert_eq!(sorted_rows(&via_union), sorted_rows(&recomputed));
            // Crash: both handles drop; the source's unflushed tail is
            // gone (WalSync::Off), while the view's materialization
            // and stamp are durable — and cover only the flushed 8.
        }
        let mut source =
            Table::persistent_with("trades", m1_schema(), "ts", &source_dir, off).unwrap();
        assert_eq!(
            source.query("SELECT ts FROM trades").unwrap().num_rows(),
            16,
            "everything a refresh stamped must survive the crash — a \
             loss here means refresh stamped unflushed rows"
        );
        let mut view =
            MaterializedView::open("ohlc", &view_dir, &source, None, StoreOptions::default())
                .unwrap();
        view.refresh(&mut source, None).unwrap();
        let recomputed = source.query(OHLC).unwrap();
        let materialized = view
            .query_union(
                &source,
                None,
                &lower_plan("SELECT sym, bar, o, h, l, c FROM ohlc").unwrap(),
            )
            .unwrap();
        assert_eq!(
            sorted_rows(&materialized),
            sorted_rows(&recomputed),
            "ghost buckets survived the crash"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stamp_ahead_of_the_source_triggers_the_rebuild_floor() {
        // With the flush-then-stamp discipline a crash can never leave
        // the stamp ahead of the source; only a foreign pairing can —
        // a source directory swapped under the view, a hand-edited
        // record. Nothing such a stamp claims is trustworthy, so the
        // refresh answers with the rebuild floor: every materialized
        // row out, one full fold in.
        let dir = std::env::temp_dir().join(format!("tallydb-view-belt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let view_dir = dir.join("ohlc");
        std::fs::create_dir_all(&view_dir).unwrap();
        let mut source = Table::new("trades", m1_schema(), "ts").unwrap();
        for i in 0..8 {
            source.append(&linear_row(i)).unwrap();
        }
        {
            let mut view =
                MaterializedView::persistent("ohlc", OHLC, &source, None, &view_dir).unwrap();
            view.refresh(&mut source, None).unwrap();
        }
        // The tamper: a stamp far past anything the source has spent.
        std::fs::write(
            view_dir.join(DEFINITION_FILE),
            encode_definition(1_000_000, 0, "trades", OHLC, None),
        )
        .unwrap();
        let mut view =
            MaterializedView::open("ohlc", &view_dir, &source, None, StoreOptions::default())
                .unwrap();
        assert_eq!(view.refresh(&mut source, None).unwrap(), u64::MAX);
        let recomputed = source.query(OHLC).unwrap();
        let materialized = view
            .query_union(
                &source,
                None,
                &lower_plan("SELECT sym, bar, o, h, l, c FROM ohlc").unwrap(),
            )
            .unwrap();
        assert_eq!(sorted_rows(&materialized), sorted_rows(&recomputed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn seq_on_a_view_is_refused_and_mismatched_multipliers_too() {
        // Two review findings, both refusals. '_seq' through the union
        // path would fabricate coordinates on scratch segments and
        // serve real ones when fresh — two wrong answers selected by
        // staleness — so it is refused with the pointer to the base.
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..8 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("ohlc", OHLC).unwrap();
        let error = db
            .query("SELECT bar, _seq FROM ohlc")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("'_seq' on a maintained view"), "{error}");
        let error = db
            .query("SELECT bar FROM ohlc WHERE _seq >= 3")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("'_seq' on a maintained view"), "{error}");
        // A bucket start multiplies back by its own width; the executor
        // multiplies unchecked and the view's range inverse saturates,
        // so a mismatched multiplier is refused before the two could
        // disagree at i64's edge.
        let error = db
            .create_materialized_view(
                "bad",
                "SELECT (ts / 4) * 5 AS bar, sum(x) AS s FROM trades GROUP BY (ts / 4) * 5",
            )
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("multiplier differs from its width"),
            "{error}"
        );
    }

    /// The running battery's comparison: exact where the fixture's
    /// values are dyadic (every sum representable), and the module's
    /// stated 1e-12-relative contract is exercised separately by the
    /// non-dyadic case below.
    const RUNNING: &str = "SELECT sym, count(*) AS n, sum(x) AS s, avg(x) AS a, \
                           min(x) AS lo, max(x) AS hi, first(x) AS o, last(x) AS c \
                           FROM trades GROUP BY sym";
    const RUNNING_COLUMNS: &str = "sym, n, s, a, lo, hi, o, c";

    fn assert_running_matches(db: &Database, view: &str, columns: &str) {
        let sql = db.view(view).unwrap().sql().to_owned();
        let recomputed = db
            .table(db.view(view).unwrap().source())
            .unwrap()
            .query(&sql)
            .unwrap();
        let materialized = db.query(&format!("SELECT {columns} FROM {view}")).unwrap();
        assert_eq!(
            sorted_rows(&materialized),
            sorted_rows(&recomputed),
            "running view '{view}' diverged from recompute"
        );
    }

    #[test]
    fn a_running_view_answers_exactly_through_every_state() {
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        for i in 0..24 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("totals", RUNNING).unwrap();
        // Unsized (width 0, nothing folded): answers by recompute.
        assert_eq!(db.view("totals").unwrap().stamp(), 0);
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        // First refresh sizes the hidden bucket and folds partials.
        assert!(db.refresh_view("totals").unwrap() >= 1);
        assert!(db.view("totals").unwrap().stamp() > 0);
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        // Fresh, stale-with-tail, corrected, and deleted states all
        // meet the same equality.
        for i in 24..30 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        assert_running_matches(&db, "totals", RUNNING_COLUMNS); // stale tail
        db.refresh_view("totals").unwrap();
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        assert_running_matches(&db, "totals", RUNNING_COLUMNS); // dirty, unrefreshed
        db.refresh_view("totals").unwrap();
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        // The extremum case a running MIN/MAX cannot fake: deleting
        // the maximum forces the answer DOWN, which no accumulator
        // can produce — only the re-fold of its bucket can.
        db.mutate("DELETE FROM trades WHERE x = 500.0").unwrap();
        db.refresh_view("totals").unwrap();
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        // AS OF recomputes the pre-correction world.
        let before = db.table("trades").unwrap().next_sequence() - 1;
        db.mutate("UPDATE trades SET x = -9.0 WHERE ts = 10")
            .unwrap();
        db.refresh_view("totals").unwrap();
        let past = db
            .query(&format!(
                "SELECT {RUNNING_COLUMNS} FROM totals ASOF {before}"
            ))
            .unwrap();
        let past_base = db
            .table("trades")
            .unwrap()
            .query(&format!(
                "SELECT sym, count(*) AS n, sum(x) AS s, avg(x) AS a, min(x) AS lo, \
                 max(x) AS hi, first(x) AS o, last(x) AS c \
                 FROM trades ASOF {before} GROUP BY sym"
            ))
            .unwrap();
        assert_eq!(sorted_rows(&past), sorted_rows(&past_base));
        // A refresh after a single-bucket correction folds ONE hidden
        // bucket — the pricing the partials exist for.
        db.mutate("UPDATE trades SET x = 7.5 WHERE ts = 20")
            .unwrap();
        assert_eq!(db.refresh_view("totals").unwrap(), 1);
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
    }

    #[test]
    fn a_global_running_view_is_one_row_that_stays_true() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..10 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view(
            "overall",
            "SELECT count(*) AS n, sum(x) AS s, avg(x) AS a FROM trades",
        )
        .unwrap();
        db.refresh_view("overall").unwrap();
        assert_running_matches(&db, "overall", "n, s, a");
        db.mutate("DELETE FROM trades WHERE ts >= 5").unwrap();
        assert_running_matches(&db, "overall", "n, s, a"); // stale
        db.refresh_view("overall").unwrap();
        assert_running_matches(&db, "overall", "n, s, a");
        // Empty it entirely: the global aggregate of nothing is one
        // row of COUNT 0 with NULL sum and average — via the partials
        // path exactly as via recompute.
        db.mutate("DELETE FROM trades WHERE ts >= 0").unwrap();
        db.refresh_view("overall").unwrap();
        assert_running_matches(&db, "overall", "n, s, a");
    }

    #[test]
    fn running_sums_meet_the_stated_tolerance_on_non_dyadic_data() {
        // The combine contract: partial-then-combine association may
        // differ from single-pass in the final ulps, within 1e-12
        // relative. Non-dyadic values (thirds) force the difference to
        // exist if it ever will; the comparison here is the contract's,
        // not exact equality.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        for i in 0..200i64 {
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(i),
                    storage_lite::RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                    storage_lite::RowValue::F64(i as f64 / 3.0),
                    storage_lite::RowValue::F64(0.0),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view(
            "t2",
            "SELECT sym, sum(x) AS s, avg(x) AS a FROM trades GROUP BY sym",
        )
        .unwrap();
        db.refresh_view("t2").unwrap();
        let through_view = db.query("SELECT s, a FROM t2 ORDER BY s").unwrap();
        let recomputed = db
            .table("trades")
            .unwrap()
            .query("SELECT sym, sum(x) AS s, avg(x) AS a FROM trades GROUP BY sym ORDER BY s")
            .unwrap();
        let view_s = crate::table::tests::flatten(&through_view, 0);
        let base_s = crate::table::tests::flatten(&recomputed, 1);
        assert_eq!(view_s.len(), base_s.len());
        for (view, base) in view_s.iter().zip(&base_s) {
            let (view, base) = (view.unwrap(), base.unwrap());
            assert!(
                ((view - base) / base).abs() < 1e-12,
                "combine drifted past the contract: {view} vs {base}"
            );
        }
    }

    #[test]
    fn a_running_view_persists_its_width_and_serves_read_only() {
        let dir = std::env::temp_dir().join(format!("tallydb-view-running-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source_dir = dir.join("trades");
        let view_dir = dir.join("totals");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&view_dir).unwrap();
        let mut source = Table::persistent("trades", m1_schema(), "ts", &source_dir).unwrap();
        for i in 0..16 {
            source.append(&linear_row(i)).unwrap();
        }
        {
            let mut view =
                MaterializedView::persistent("totals", RUNNING, &source, None, &view_dir).unwrap();
            view.refresh(&mut source, None).unwrap();
        }
        // The width survives the record round trip (v2) — read back
        // from the bytes, not inferred — and the reopened view keeps
        // folding under it rather than re-sizing.
        let record = std::fs::read(view_dir.join(DEFINITION_FILE)).unwrap();
        let (_, width, _, _, _) = decode_definition(&record).unwrap();
        assert!(width > 0, "the chosen width was not persisted");
        let mut view =
            MaterializedView::open("totals", &view_dir, &source, None, StoreOptions::default())
                .unwrap();
        source
            .mutate("UPDATE trades SET x = 50.0 WHERE ts = 2")
            .unwrap();
        assert_eq!(view.refresh(&mut source, None).unwrap(), 1);
        drop(view);
        // Read-only: exact answers over a stale materialization, no
        // writes, refresh refused.
        source
            .mutate("UPDATE trades SET x = 60.0 WHERE ts = 9")
            .unwrap();
        source.flush().unwrap();
        let ro_source = Table::open_read_only("trades", &source_dir).unwrap();
        let mut db = Database::new();
        db.add_table(ro_source).unwrap();
        let ro_view = MaterializedView::open_read_only(
            "totals",
            &view_dir,
            db.table("trades").unwrap(),
            None,
        )
        .unwrap();
        db.add_view(ro_view).unwrap();
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        assert!(db.refresh_view("totals").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The cumulative battery's definition: every admitted expanding
    /// window, partitioned by symbol. `y` gives MIN/MAX motion in both
    /// directions (increasing for A, decreasing for B).
    const CUMULATIVE: &str = "SELECT ts, sym, \
         sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cs, \
         count(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cn, \
         avg(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ca, \
         min(y) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS clo, \
         max(y) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS chi \
         FROM trades";
    const CUMULATIVE_COLUMNS: &str = "ts, sym, cs, cn, ca, clo, chi";

    /// The cumulative A/B: the same ranged query through the two code
    /// paths — the boundary + assembly split, against the recompute
    /// path forced via `ASOF` at the current watermark (the definition
    /// over `base AS OF now` *is* the current answer). Returns the row
    /// count so callers can refuse vacuous agreement.
    fn cumulative_ab(db: &Database, view: &str, columns: &str, floor: i64) -> usize {
        let now = db
            .table(db.view(view).unwrap().source())
            .unwrap()
            .next_sequence();
        let ranged = db
            .query(&format!("SELECT {columns} FROM {view} WHERE ts >= {floor}"))
            .unwrap();
        let truth = db
            .query(&format!(
                "SELECT {columns} FROM {view} ASOF {now} WHERE ts >= {floor}"
            ))
            .unwrap();
        assert_eq!(
            sorted_rows(&ranged),
            sorted_rows(&truth),
            "cumulative view '{view}' range read from {floor} diverged from recompute"
        );
        ranged.num_rows()
    }

    fn assert_cumulative_matches(db: &Database, view: &str, columns: &str, floor: i64) {
        assert!(
            cumulative_ab(db, view, columns, floor) > 0,
            "vacuous cumulative check: no rows at or above {floor}"
        );
    }

    #[test]
    fn a_cumulative_view_answers_exactly_through_every_state() {
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        for i in 0..24 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("cum", CUMULATIVE).unwrap();
        // Unsized (width 0, nothing folded): BOTH A/B legs recompute
        // here, so this check certifies routing and the recompute
        // path's own predicate handling — not the partials machinery,
        // which has no width to run under yet.
        assert_eq!(db.view("cum").unwrap().stamp(), 0);
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        // First refresh sizes the hidden bucket and folds partials;
        // range reads from several floors — including 0, where the
        // boundary is empty and assembly covers everything — all agree.
        assert!(db.refresh_view("cum").unwrap() >= 1);
        assert!(db.view("cum").unwrap().stamp() > 0);
        for floor in [0, 5, 12, 23] {
            assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, floor);
        }
        // Stale tail: appended rows the stamp does not cover reach the
        // boundary through the live half of the partials union.
        // The last spent coordinate — ASOF is inclusive.
        let before = db.table("trades").unwrap().next_sequence() - 1;
        for i in 24..30 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        // AS OF the pre-append watermark answers the shorter world —
        // nontrivially, since the materialization already reflects it.
        let past = db
            .query(&format!(
                "SELECT {CUMULATIVE_COLUMNS} FROM cum ASOF {before}"
            ))
            .unwrap();
        let past_base = db
            .table("trades")
            .unwrap()
            .query(&format!("{CUMULATIVE} ASOF {before}"))
            .unwrap();
        assert_eq!(sorted_rows(&past), sorted_rows(&past_base));
        assert_eq!(past.num_rows(), 24);
        db.refresh_view("cum").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        // A correction BELOW the floor: the ranged read never sees the
        // corrected row itself, only its effect on the boundary — the
        // seam this whole read exists for. (Compaction first: windows
        // refuse disordered segments — the same refusal the base gives
        // — and the reinsert lands out of order until compacted; the
        // view stays dirty regardless, since compaction moves kills to
        // history without refreshing anything.)
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        db.compact("trades").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12); // dirty, unrefreshed
        db.refresh_view("cum").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        // Deleting that row pulls the running MAX back down for every
        // later row — the correction no accumulator can produce.
        db.mutate("DELETE FROM trades WHERE ts = 3").unwrap();
        db.refresh_view("cum").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        // A refresh after a single-row correction folds ONE hidden
        // bucket — the same pricing the running battery proves.
        db.mutate("UPDATE trades SET x = 7.5 WHERE ts = 20")
            .unwrap();
        db.compact("trades").unwrap();
        assert_eq!(db.refresh_view("cum").unwrap(), 1);
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        // AS OF once corrections sit in history: the past's rows live
        // in history segments whose key ranges interleave with the
        // live generation's, and windows refuse what they cannot
        // order — on the view exactly as on the base. `view AS OF s =
        // Q(base AS OF s)` includes the refusals.
        let view_error = db
            .query(&format!(
                "SELECT {CUMULATIVE_COLUMNS} FROM cum ASOF {before}"
            ))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        let base_error = db
            .table("trades")
            .unwrap()
            .query(&format!("{CUMULATIVE} ASOF {before}"))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(view_error.contains("not sorted"), "{view_error}");
        assert!(base_error.contains("not sorted"), "{base_error}");
    }

    #[test]
    fn a_global_cumulative_view_has_one_partition_that_stays_true() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..10 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view(
            "gcum",
            "SELECT ts, sum(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cs, \
             avg(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ca FROM trades",
        )
        .unwrap();
        db.refresh_view("gcum").unwrap();
        assert_cumulative_matches(&db, "gcum", "ts, cs, ca", 4);
        db.mutate("DELETE FROM trades WHERE ts < 3").unwrap();
        assert_cumulative_matches(&db, "gcum", "ts, cs, ca", 4); // stale
        db.refresh_view("gcum").unwrap();
        assert_cumulative_matches(&db, "gcum", "ts, cs, ca", 4);
        // A floor past all data: both paths answer empty, not wrongly.
        assert_eq!(cumulative_ab(&db, "gcum", "ts, cs, ca", 1_000), 0);
    }

    #[test]
    fn cumulative_range_reads_cross_bucket_edges_exactly() {
        // A width wide enough that floors land mid-bucket, and data
        // straddling zero so the double-width bucket 0 of truncating
        // division is on the path. Every floor — edges, mid-bucket,
        // negative, at both ends — meets the same A/B.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        for i in 0..31i64 {
            let ts = -1000 + i * 100;
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                    storage_lite::RowValue::F64(i as f64),
                    storage_lite::RowValue::F64(-(i as f64)),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view("cum", CUMULATIVE).unwrap();
        db.refresh_view("cum").unwrap();
        assert!(db.view("cum").unwrap().stamp() > 0);
        for floor in [-1000, -101, -100, -99, -1, 0, 1, 99, 100, 950, 2000] {
            assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, floor);
        }
        // And a correction below a mid-bucket floor still lands in the
        // boundary while unrefreshed.
        db.mutate("UPDATE trades SET x = 400.0 WHERE ts = -500")
            .unwrap();
        db.compact("trades").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 950);
        db.refresh_view("cum").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 950);
    }

    #[test]
    fn cumulative_sums_meet_the_stated_tolerance_on_non_dyadic_data() {
        // The combine contract again, at the boundary seam: folding a
        // boundary sum into an assembled window associates differently
        // than the recompute's single pass. Thirds force the difference
        // to exist if it ever will; agreement is the contract's 1e-12
        // relative, not exact equality.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        for i in 0..200i64 {
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(i),
                    storage_lite::RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                    storage_lite::RowValue::F64(i as f64 / 3.0),
                    storage_lite::RowValue::F64(0.0),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view(
            "c2",
            "SELECT ts, sym, sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cs, \
             avg(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ca FROM trades",
        )
        .unwrap();
        db.refresh_view("c2").unwrap();
        let now = db.table("trades").unwrap().next_sequence();
        for column in ["cs", "ca"] {
            let ranged = db
                .query(&format!(
                    "SELECT {column} FROM c2 WHERE ts >= 150 ORDER BY ts"
                ))
                .unwrap();
            let truth = db
                .query(&format!(
                    "SELECT {column} FROM c2 ASOF {now} WHERE ts >= 150 ORDER BY ts"
                ))
                .unwrap();
            let ranged = crate::table::tests::flatten(&ranged, 0);
            let truth = crate::table::tests::flatten(&truth, 0);
            assert_eq!(ranged.len(), truth.len());
            assert!(!ranged.is_empty());
            for (view, base) in ranged.iter().zip(&truth) {
                let (view, base) = (view.unwrap(), base.unwrap());
                assert!(
                    ((view - base) / base).abs() < 1e-12,
                    "{column} drifted past the contract: {view} vs {base}"
                );
            }
        }
    }

    #[test]
    fn a_cumulative_view_persists_its_width_and_serves_read_only() {
        let dir =
            std::env::temp_dir().join(format!("tallydb-view-cumulative-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source_dir = dir.join("trades");
        let view_dir = dir.join("cum");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&view_dir).unwrap();
        let mut source = Table::persistent("trades", m1_schema(), "ts", &source_dir).unwrap();
        for i in 0..16 {
            source.append(&linear_row(i)).unwrap();
        }
        {
            let mut view =
                MaterializedView::persistent("cum", CUMULATIVE, &source, None, &view_dir).unwrap();
            view.refresh(&mut source, None).unwrap();
        }
        let record = std::fs::read(view_dir.join(DEFINITION_FILE)).unwrap();
        let (_, width, _, _, _) = decode_definition(&record).unwrap();
        assert!(width > 0, "the chosen width was not persisted");
        let mut view =
            MaterializedView::open("cum", &view_dir, &source, None, StoreOptions::default())
                .unwrap();
        source
            .mutate("UPDATE trades SET x = 50.0 WHERE ts = 2")
            .unwrap();
        assert_eq!(view.refresh(&mut source, None).unwrap(), 1);
        drop(view);
        source
            .mutate("UPDATE trades SET x = 60.0 WHERE ts = 9")
            .unwrap();
        // Compaction restores cross-segment order (windows refuse
        // disorder); the view is still stale about ts = 9, which is
        // the read-only staleness this test wants.
        source.compact().unwrap();
        source.flush().unwrap();
        let ro_source = Table::open_read_only("trades", &source_dir).unwrap();
        let mut db = Database::new();
        db.add_table(ro_source).unwrap();
        let ro_view =
            MaterializedView::open_read_only("cum", &view_dir, db.table("trades").unwrap(), None)
                .unwrap();
        db.add_view(ro_view).unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 8);
        // And with the un-refreshed correction (ts = 9) strictly BELOW
        // the floor: the read-only reader's boundary must fold the
        // dirty bucket live — the read-only path has no repair to
        // lean on, only the union.
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
        assert!(db.refresh_view("cum").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cumulative_view_over_an_empty_source_stays_well_defined() {
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        db.create_materialized_view(
            "cum",
            "SELECT ts, sym, sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cs FROM trades",
        )
        .unwrap();
        // No rows: nothing to size, nothing to fold — and both read
        // paths answer empty rather than failing.
        assert_eq!(db.refresh_view("cum").unwrap(), 0);
        assert_eq!(db.query("SELECT cs FROM cum").unwrap().num_rows(), 0);
        assert_eq!(
            db.query("SELECT cs FROM cum WHERE ts >= 5")
                .unwrap()
                .num_rows(),
            0
        );
        // Rows arriving after the empty refresh serve by recompute
        // until the next refresh chooses a width (both A/B legs share
        // that path here — this checks routing, not partials).
        for i in 0..6 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        assert_cumulative_matches(&db, "cum", "ts, sym, cs", 2);
        assert!(db.refresh_view("cum").unwrap() >= 1);
        assert_cumulative_matches(&db, "cum", "ts, sym, cs", 2);
    }

    #[test]
    fn a_cumulative_read_over_uncompacted_corrections_refuses_like_the_base() {
        // Windows require ordered data, and a correction's reinsert
        // lands out of order until compaction. The FULL read recomputes
        // over the whole source, so it meets that segment and refuses
        // LOUDLY — the same refusal the base gives, never a silently
        // wrong answer. A ranged read ABOVE the correction, though,
        // never touches the stray segment (zone maps prune it) and its
        // boundary re-folds with aggregates, which need no order — it
        // keeps answering exactly, dirty and uncompacted alike. That
        // asymmetry is the partials paying rent.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 8).unwrap())
            .unwrap();
        for i in 0..24 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("cum", CUMULATIVE).unwrap();
        db.refresh_view("cum").unwrap();
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        let through_view = db
            .query(&format!("SELECT {CUMULATIVE_COLUMNS} FROM cum"))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        let over_base = db
            .table("trades")
            .unwrap()
            .query(CUMULATIVE)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(through_view.contains("not sorted"), "{through_view}");
        assert!(over_base.contains("not sorted"), "{over_base}");
        let while_dirty = db
            .query(&format!(
                "SELECT {CUMULATIVE_COLUMNS} FROM cum WHERE ts >= 12"
            ))
            .unwrap();
        db.compact("trades").unwrap();
        let after_compact = db
            .query(&format!(
                "SELECT {CUMULATIVE_COLUMNS} FROM cum WHERE ts >= 12"
            ))
            .unwrap();
        assert!(while_dirty.num_rows() > 0);
        assert_eq!(sorted_rows(&while_dirty), sorted_rows(&after_compact));
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 12);
    }

    #[test]
    fn a_stale_union_read_survives_a_numeric_where() {
        // Regression: the resident-handle metadata used to turn a
        // scratch segment's "no zone maps at all" into per-column
        // `None` maps, which `can_match` reads as "no valid values" —
        // silently pruning the union read's entire live half under any
        // numeric WHERE. The stale answer just lost rows.
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..12 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view("ohlc", OHLC).unwrap();
        // Entirely unrefreshed: the whole answer is the live half.
        let stale = db
            .query("SELECT sym, bar, o, c FROM ohlc WHERE bar >= 1")
            .unwrap();
        db.refresh_view("ohlc").unwrap();
        let fresh = db
            .query("SELECT sym, bar, o, c FROM ohlc WHERE bar >= 1")
            .unwrap();
        assert!(fresh.num_rows() > 0);
        assert_eq!(sorted_rows(&stale), sorted_rows(&fresh));
    }

    #[test]
    fn ineligible_cumulative_definitions_are_refused_by_name() {
        let source = source();
        let refused = |sql: &str, needle: &str| {
            let error = MaterializedView::new("v", sql, &source, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{sql}: {error}");
        };
        refused(
            "SELECT ts, sym, lag(x, 1) OVER (PARTITION BY sym ORDER BY ts) AS p FROM trades",
            "LAG/LEAD",
        );
        refused(
            "SELECT ts, sym, first(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS f FROM trades",
            "outside sum/count/avg/min/max",
        );
        refused(
            "SELECT ts, sym, sum(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS r FROM trades",
            "bounded window frame",
        );
        refused(
            "SELECT ts, sym, sum(x) OVER (PARTITION BY sym ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM trades",
            "not ordered by the ordering key",
        );
        refused(
            "SELECT ts, sum(x) OVER (PARTITION BY ts / 4 ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM trades",
            "cross-sectional partition",
        );
        refused(
            "SELECT ts, sym, x + 0.0 AS xx, sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s \
             FROM trades",
            "computed expression",
        );
        refused(
            "SELECT sym, sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM trades",
            "omits the ordering key",
        );
        refused(
            "SELECT ts, sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM trades",
            "omits a partition symbol",
        );
        refused(
            "SELECT ts, sym, sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a, \
             sum(y) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS b FROM trades",
            "different PARTITION BY lists",
        );
    }

    #[test]
    fn a_running_avg_over_an_i64_column_divides_not_panics() {
        // The AVG sum partial follows its argument's type: SUM over an
        // i64 column is exact i64, and finalize used to destructure it
        // as f64 and panic on the first partials-path read (found by
        // the repo-wide code review, reproduced there). The i64 sum
        // divides in f64 like any average.
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..10 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view(
            "iavg",
            "SELECT sym, avg(ts) AS a, min(ts) AS lo FROM trades GROUP BY sym",
        )
        .unwrap();
        db.refresh_view("iavg").unwrap();
        assert_running_matches(&db, "iavg", "sym, a, lo");
    }

    #[test]
    fn a_cumulative_max_propagates_nan_across_the_boundary() {
        // The engine's comparison relation places NaN GREATER than
        // every number, and both reference computations — the boundary
        // MAX aggregate and the assembled MAX window — propagate it.
        // Rust's f64::max silently drops NaN, which made the answer
        // depend on the query's WHERE clause (found by the repo-wide
        // code review, reproduced there): a NaN below the floor
        // reaches later rows only through the boundary fold.
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..12 {
            let x = if i == 3 { f64::NAN } else { i as f64 };
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(i),
                    storage_lite::RowValue::Key("A"),
                    storage_lite::RowValue::F64(x),
                    storage_lite::RowValue::F64(0.0),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view(
            "m",
            "SELECT ts, sym, max(x) OVER (PARTITION BY sym ORDER BY ts \
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS hi FROM trades",
        )
        .unwrap();
        db.refresh_view("m").unwrap();
        assert_cumulative_matches(&db, "m", "ts, sym, hi", 8);
        let ranged = db.query("SELECT hi FROM m WHERE ts >= 8").unwrap();
        let values = crate::table::tests::flatten(&ranged, 0);
        assert!(
            values.iter().all(|v| v.unwrap().is_nan()),
            "the boundary NaN must poison every later running max: {values:?}"
        );
    }

    #[test]
    fn a_running_view_with_aliased_keys_answers_from_partials() {
        // The combine runs over the materialization, where a key
        // column is stored under its selected output name — the
        // user's alias. Source-named combine keys broke every
        // partials-path read of an aliased-key view (found by the
        // repo-wide code review, reproduced there: worked at width 0,
        // failed after the first refresh).
        let mut db = Database::new();
        db.create_table("trades", m1_schema(), "ts").unwrap();
        for i in 0..10 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        db.create_materialized_view(
            "aliased",
            "SELECT sym AS s, sum(x) AS total FROM trades GROUP BY sym",
        )
        .unwrap();
        db.refresh_view("aliased").unwrap();
        assert_running_matches(&db, "aliased", "s, total");
        // And the shape the materialization cannot serve at all — a
        // group key the SELECT list omits — is refused at create, not
        // broken at first refresh.
        let error = db
            .create_materialized_view("keyless", "SELECT sum(x) AS s FROM trades GROUP BY sym")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("omits a GROUP BY key"), "{error}");
    }

    #[test]
    fn reserved_names_are_refused_at_the_definition_door() {
        // The synthesis mints hidden columns in the '__' name space; a
        // user column or alias there produced a view that created and
        // refreshed fine but could never be read (found by the
        // repo-wide code review, which probed a source column named
        // '__p0' selected as a running key). One prefix rule closes
        // the whole family.
        let schema = arrow_lite::Schema::new(vec![
            arrow_lite::Field::new("ts", arrow_lite::ColumnType::I64, false),
            arrow_lite::Field::new("__p0", arrow_lite::ColumnType::Key, false),
            arrow_lite::Field::new("x", arrow_lite::ColumnType::F64, false),
        ]);
        let source = Table::new("t", schema, "ts").unwrap();
        let refused = |sql: &str| {
            let error = MaterializedView::new("v", sql, &source, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains("reserved"), "{sql}: {error}");
        };
        refused("SELECT __p0, sum(x) AS s FROM t GROUP BY __p0");
        refused("SELECT ts, sum(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS __w0_sum FROM t");
        refused("SELECT sum(x) AS __row FROM t");
        // A bucketed view mints no hidden names and keeps the wider
        // name space — the prefix rule is the partials shapes' alone.
        MaterializedView::new(
            "ok",
            "SELECT ts, sum(x) AS s FROM t GROUP BY ts",
            &source,
            None,
        )
        .unwrap();
    }

    #[test]
    fn a_views_schema_names_its_answers_not_its_partials() {
        // The public accessor answers with the shape queries return;
        // for the partials shapes that is the user definition's
        // output, never the internal materialization (whose '__p{i}' /
        // '__bucket' columns no query answers with — found stranded by
        // the repo-wide code review after the ReadShape refactor).
        let source = source();
        let running = MaterializedView::new(
            "r",
            "SELECT sym, avg(x) AS a FROM trades GROUP BY sym",
            &source,
            None,
        )
        .unwrap();
        let names: Vec<&str> = running.schema().fields().iter().map(|f| f.name()).collect();
        assert_eq!(names, ["sym", "a"]);
        let bucketed = MaterializedView::new("b", OHLC, &source, None).unwrap();
        let names: Vec<&str> = bucketed
            .schema()
            .fields()
            .iter()
            .map(|f| f.name())
            .collect();
        assert_eq!(names, ["sym", "bar", "o", "h", "l", "c"]);
    }

    #[test]
    fn okey_lower_bounds_extract_conservatively() {
        // Every arm of the extraction, checked directly: a bound may
        // sit BELOW the truth (extra assembly, filtered later), never
        // above it (missing rows).
        let cmp = |op, value| Predicate::Compare {
            column: "ts".to_owned(),
            op,
            value,
        };
        let bound = |predicate: &Predicate| okey_lower_bound(predicate, "ts");
        assert_eq!(bound(&cmp(CmpOp::Ge, Number::Int(5))), Some(5));
        assert_eq!(bound(&cmp(CmpOp::Gt, Number::Int(5))), Some(5)); // weakened, safe
        assert_eq!(bound(&cmp(CmpOp::Eq, Number::Int(7))), Some(7));
        assert_eq!(bound(&cmp(CmpOp::Ge, Number::Float(4.5))), Some(4)); // floored, safe
        assert_eq!(bound(&cmp(CmpOp::Le, Number::Int(9))), None);
        let other = Predicate::Compare {
            column: "x".to_owned(),
            op: CmpOp::Ge,
            value: Number::Int(100),
        };
        // AND takes the tighter branch and absorbs an unbounded one.
        let and = |a, b| Predicate::And(Box::new(a), Box::new(b));
        let or = |a, b| Predicate::Or(Box::new(a), Box::new(b));
        assert_eq!(
            bound(&and(
                cmp(CmpOp::Ge, Number::Int(5)),
                cmp(CmpOp::Ge, Number::Int(8))
            )),
            Some(8)
        );
        assert_eq!(
            bound(&and(other.clone(), cmp(CmpOp::Ge, Number::Int(5)))),
            Some(5)
        );
        // OR needs both branches bounded and takes the looser.
        assert_eq!(
            bound(&or(
                cmp(CmpOp::Ge, Number::Int(5)),
                cmp(CmpOp::Ge, Number::Int(8))
            )),
            Some(5)
        );
        assert_eq!(bound(&or(cmp(CmpOp::Ge, Number::Int(5)), other)), None);
        // Negation and everything unhandled fall to full recompute.
        assert_eq!(
            bound(&Predicate::Not(Box::new(cmp(CmpOp::Lt, Number::Int(5))))),
            None
        );
    }

    #[test]
    fn multi_row_buckets_fold_and_reassemble_exactly() {
        // Every other fixture's span collapses the width heuristic to
        // 1, leaving each hidden bucket a single row — a partial that
        // mis-folds WITHIN a bucket (FIRST returning the last row,
        // say) passed everything (found by the repo-wide test review).
        // 4096 dense rows force width 4: two rows per symbol per
        // bucket, values varying within the bucket, checked against
        // recompute for both tranche-2 shapes — and the mid-bucket
        // floors have real rows between the bucket's low edge and the
        // floor, so the boundary/assembly seam is exercised INSIDE a
        // bucket.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 512).unwrap())
            .unwrap();
        for i in 0..4096i64 {
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(i),
                    storage_lite::RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                    storage_lite::RowValue::F64((i % 97) as f64),
                    storage_lite::RowValue::F64((i % 13) as f64),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view("totals", RUNNING).unwrap();
        db.create_materialized_view("cum", CUMULATIVE).unwrap();
        db.refresh_views().unwrap();
        assert!(
            db.view("totals").unwrap().width > 1 && db.view("cum").unwrap().width > 1,
            "the fixture no longer forces multi-row buckets — widths {} and {}",
            db.view("totals").unwrap().width,
            db.view("cum").unwrap().width
        );
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        for floor in [0, 2000, 2001, 2003, 4090] {
            assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, floor);
        }
        // A disjunctive predicate reaches the OR arm of the bound
        // extraction end to end (looser branch governs).
        let now = db.table("trades").unwrap().next_sequence();
        let ranged = db
            .query(&format!(
                "SELECT {CUMULATIVE_COLUMNS} FROM cum WHERE ts >= 3000 OR ts >= 2500"
            ))
            .unwrap();
        let truth = db
            .query(&format!(
                "SELECT {CUMULATIVE_COLUMNS} FROM cum ASOF {now} WHERE ts >= 3000 OR ts >= 2500"
            ))
            .unwrap();
        assert!(ranged.num_rows() > 0);
        assert_eq!(sorted_rows(&ranged), sorted_rows(&truth));
        // A correction inside one multi-row bucket repairs that bucket
        // alone and the answers still match.
        db.mutate("UPDATE trades SET x = 777.0 WHERE ts = 1024")
            .unwrap();
        db.compact("trades").unwrap();
        assert_eq!(db.refresh_view("totals").unwrap(), 1);
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        db.refresh_view("cum").unwrap();
        assert_cumulative_matches(&db, "cum", CUMULATIVE_COLUMNS, 2001);
    }

    /// The blotter fixture: a fact table and a quote history whose
    /// streams interleave — facts run AHEAD of quotes at the frontier,
    /// which is exactly the min-frontier case the ceiling exists for.
    fn blotter_db() -> Database {
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "quotes",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
                ]),
                "qts",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        for i in 0..20 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        // Quotes every 4 ticks per symbol, frontier at 12 — trades
        // 13..19 run ahead of every quote.
        for (qts, sym, bid) in [
            (0, "A", 1.0),
            (1, "B", 2.0),
            (4, "A", 1.4),
            (5, "B", 2.5),
            (8, "A", 1.8),
            (9, "B", 2.9),
            (12, "A", 1.12),
        ] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        db
    }

    const BLOTTER: &str = "SELECT ts, sym, x, bid FROM trades \
         ASOF LEFT JOIN quotes ON trades.sym = quotes.sym";

    /// The blotter A/B: the view's answer against the SAME join run
    /// directly over the base tables — an independent leg (the direct
    /// join never touches view machinery), exact at every state.
    fn assert_blotter_matches(db: &Database, view: &str) {
        let through = db
            .query(&format!("SELECT ts, sym, x, bid FROM {view}"))
            .unwrap();
        let direct = db.query(BLOTTER).unwrap();
        assert_eq!(
            sorted_rows(&through),
            sorted_rows(&direct),
            "blotter view '{view}' diverged from the direct join"
        );
        assert!(through.num_rows() > 0, "vacuous blotter check");
    }

    #[test]
    fn a_blotter_view_answers_exactly_through_every_state() {
        let mut db = blotter_db();
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        assert_eq!(db.view("blotter").unwrap().dimension(), Some("quotes"));
        // Stale: nothing folded, the whole answer is the live join.
        assert_blotter_matches(&db, "blotter");
        // First refresh: the ceiling lands at the quote frontier (12),
        // so exactly the trades below it materialize; the tail 12..19
        // stays live — and the answer is identical either way.
        let folded = db.refresh_view("blotter").unwrap();
        assert_eq!(folded, 12, "trades 0..=11 sit below the frontier");
        assert_blotter_matches(&db, "blotter");
        // In-order quote appends land ABOVE the ceiling: no
        // materialized row dirties, the answer stays exact unrefreshed
        // (the min-frontier property).
        for (qts, sym, bid) in [(13, "B", 2.13), (16, "A", 1.16)] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        assert_blotter_matches(&db, "blotter"); // stale, still exact
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        // A LATE quote strictly below the ceiling: a correction. Its
        // blast radius is [6, 8) for B — the interval lemma — and the
        // unrefreshed read must already fold it live.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(6),
                storage_lite::RowValue::Key("B"),
                storage_lite::RowValue::F64(9.9),
            ],
        )
        .unwrap();
        assert_blotter_matches(&db, "blotter"); // dirty, unrefreshed
        let folded = db.refresh_view("blotter").unwrap();
        assert!(folded >= 1, "the late quote dirtied its interval");
        assert_blotter_matches(&db, "blotter");
        // Amend and then delete a quote below the ceiling: the value
        // flips, then falls back to the predecessor — the delete
        // branch of the lemma.
        db.mutate("UPDATE quotes SET bid = 7.7 WHERE qts = 4")
            .unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        db.mutate("DELETE FROM quotes WHERE qts = 4").unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        // Fact-side corrections repair through the fact stamp.
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        db.mutate("DELETE FROM trades WHERE ts = 7").unwrap();
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        // Compaction on either side changes nothing.
        db.compact("trades").unwrap();
        db.compact("quotes").unwrap();
        assert_blotter_matches(&db, "blotter");
    }

    #[test]
    fn a_blotter_correction_repairs_its_interval_not_the_prefix() {
        // The interval lemma, priced: a late quote at t dirties
        // [t, next quote for that symbol) — and nothing else, so the
        // refresh count equals that interval's width (plus nothing).
        let mut db = blotter_db();
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        db.refresh_view("blotter").unwrap();
        // Late quote for A at 2: A's next quote is 4, so the interval
        // is [2, 4) — but the fold rounds to whole keys of BOTH
        // symbols in [2, 3], which is 2 keys.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(2),
                storage_lite::RowValue::Key("A"),
                storage_lite::RowValue::F64(8.8),
            ],
        )
        .unwrap();
        assert_eq!(db.refresh_view("blotter").unwrap(), 2);
        assert_blotter_matches(&db, "blotter");
        // A tie correction (cycle 0's rule, end to end): a second
        // quote at A's qts = 8 — its rebirth is the newest knowledge
        // at that timestamp, and the blotter must serve it.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(8),
                storage_lite::RowValue::Key("A"),
                storage_lite::RowValue::F64(4.4),
            ],
        )
        .unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        db.compact("quotes").unwrap();
        assert_blotter_matches(&db, "blotter");
    }

    #[test]
    fn a_blotter_persists_its_pair_stamp_and_serves_read_only() {
        let dir = std::env::temp_dir().join(format!("tallydb-blotter-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let trades_dir = dir.join("trades");
        let quotes_dir = dir.join("quotes");
        let view_dir = dir.join("blotter");
        for sub in [&trades_dir, &quotes_dir, &view_dir] {
            std::fs::create_dir_all(sub).unwrap();
        }
        let mut trades = Table::persistent("trades", m1_schema(), "ts", &trades_dir).unwrap();
        let mut quotes = Table::persistent(
            "quotes",
            arrow_lite::Schema::new(vec![
                arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
            ]),
            "qts",
            &quotes_dir,
        )
        .unwrap();
        for i in 0..10 {
            trades.append(&linear_row(i)).unwrap();
        }
        for (qts, sym, bid) in [(0, "A", 1.0), (1, "B", 2.0), (6, "A", 1.6), (7, "B", 2.7)] {
            quotes
                .append(&[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ])
                .unwrap();
        }
        {
            let mut view =
                MaterializedView::persistent("blotter", BLOTTER, &trades, Some(&quotes), &view_dir)
                    .unwrap();
            view.refresh(&mut trades, Some(&mut quotes)).unwrap();
        }
        // The record is v3: both stamps and the ceiling round-trip.
        let record = std::fs::read(view_dir.join(DEFINITION_FILE)).unwrap();
        let (stamp, _, source_name, _, join) = decode_definition(&record).unwrap();
        assert!(stamp > 0);
        assert_eq!(source_name, "trades");
        let join = join.expect("a blotter records its dimension");
        assert_eq!(join.dimension, "quotes");
        assert!(join.stamp > 0);
        assert_eq!(join.ceiling, 7, "the ceiling is the quote frontier");
        assert_eq!(
            MaterializedView::stored_dimension(&view_dir)
                .unwrap()
                .as_deref(),
            Some("quotes")
        );
        // Reopen writable: the pair survives; a correction below the
        // ceiling repairs.
        let mut view = MaterializedView::open(
            "blotter",
            &view_dir,
            &trades,
            Some(&quotes),
            StoreOptions::default(),
        )
        .unwrap();
        quotes
            .mutate("UPDATE quotes SET bid = 9.0 WHERE qts = 0")
            .unwrap();
        assert!(view.refresh(&mut trades, Some(&mut quotes)).unwrap() >= 1);
        drop(view);
        // Opening without the dimension is refused by name; so is a
        // wrong pairing.
        let error =
            MaterializedView::open("blotter", &view_dir, &trades, None, StoreOptions::default())
                .map(|_| ())
                .unwrap_err()
                .to_string();
        assert!(error.contains("joins 'quotes'"), "{error}");
        // Read-only over both tables: exact answers, refresh refused —
        // including a stale correction only the union can serve.
        quotes
            .mutate("UPDATE quotes SET bid = 6.0 WHERE qts = 6")
            .unwrap();
        quotes.flush().unwrap();
        trades.flush().unwrap();
        drop(quotes);
        drop(trades);
        let ro_trades = Table::open_read_only("trades", &trades_dir).unwrap();
        let ro_quotes = Table::open_read_only("quotes", &quotes_dir).unwrap();
        let mut db = Database::new();
        db.add_table(ro_trades).unwrap();
        db.add_table(ro_quotes).unwrap();
        let ro_view = MaterializedView::open_read_only(
            "blotter",
            &view_dir,
            db.table("trades").unwrap(),
            Some(db.table("quotes").unwrap()),
        )
        .unwrap();
        db.add_view(ro_view).unwrap();
        assert_blotter_matches(&db, "blotter");
        assert!(db.refresh_view("blotter").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn blotter_refusals_are_by_name() {
        let mut db = blotter_db();
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        // AS OF on a join view: one coordinate cannot span two
        // sequence spaces (refusal parity with the base; #99).
        let error = db
            .query("SELECT ts, bid FROM blotter ASOF 5")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("#99"), "{error}");
        // '_seq' on a view: unchanged refusal, doubly true here.
        let error = db
            .query("SELECT ts, _seq FROM blotter")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("'_seq' on a maintained view"), "{error}");
        // The refresh arity errors, both directions: a join view
        // without its dimension, and a single-source view given one.
        let mut arity = blotter_db();
        arity.create_materialized_view("b2", BLOTTER).unwrap();
        let mut trades = Table::new("trades", m1_schema(), "ts").unwrap();
        let mut quotes_alone = Table::new(
            "quotes",
            arrow_lite::Schema::new(vec![
                arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
            ]),
            "qts",
        )
        .unwrap();
        let mut standalone =
            MaterializedView::new("b3", BLOTTER, &trades, Some(&quotes_alone)).unwrap();
        let error = standalone
            .refresh(&mut trades, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("refresh with both tables"), "{error}");
        let mut single = MaterializedView::new("s", OHLC, &trades, None).unwrap();
        let error = single
            .refresh(&mut trades, Some(&mut quotes_alone))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("single-source view"), "{error}");
        // Definition-door refusals.
        let refused = |sql: &str, needle: &str| {
            let error = {
                let mut db = blotter_db();
                db.create_materialized_view("v", sql).map(|_| ())
            }
            .unwrap_err()
            .to_string();
            assert!(error.contains(needle), "{sql}: {error}");
        };
        refused(
            "SELECT sym, x, bid FROM trades ASOF LEFT JOIN quotes \
             ON trades.sym = quotes.sym",
            "omits the fact ordering key",
        );
        refused(
            "SELECT ts, sym, x + 0.0 AS xx, bid FROM trades ASOF LEFT JOIN quotes \
             ON trades.sym = quotes.sym",
            "computed expression",
        );
        refused(
            "SELECT DISTINCT ts, sym FROM trades ASOF LEFT JOIN quotes \
             ON trades.sym = quotes.sym",
            "DISTINCT",
        );
        refused(
            "SELECT ts, sym, bid FROM trades ASOF LEFT JOIN quotes \
             ON trades.sym = quotes.sym ORDER BY ts",
            "ORDER BY / LIMIT / OFFSET",
        );
        refused(
            "SELECT ts, sym, \
             sum(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s, \
             bid FROM trades ASOF LEFT JOIN quotes ON trades.sym = quotes.sym",
            "window over a join",
        );
    }

    #[test]
    fn a_late_quote_repairs_to_its_own_symbols_next_not_the_global_next() {
        // The seam F5 exists for, made discriminating: after a late
        // quote for A at 2, A's own next quote is 20 — but ANOTHER
        // symbol's quote sits at 10. A symbol-blind endpoint would
        // stop the repair interval at [2, 10) and leave the A-fact at
        // ts = 15 silently matched to the OLD quote; the sound
        // interval [2, 20) re-folds it.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "quotes",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
                ]),
                "qts",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        // Facts: A on even keys through 24 — 15 is B, so use 14/16.
        for i in 0..24 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        for (qts, sym, bid) in [
            (0, "A", 1.0),
            (1, "B", 2.0),
            (10, "B", 3.0),
            (20, "A", 4.0),
            (21, "B", 5.0),
        ] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        db.refresh_view("blotter").unwrap();
        // The late quote: A at 2. Its interval ends at A's OWN next
        // (20), not the global next (10); A-facts at 12..18 sit
        // between the two and flip to bid 8.5.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(2),
                storage_lite::RowValue::Key("A"),
                storage_lite::RowValue::F64(8.5),
            ],
        )
        .unwrap();
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        // Pin the flipped value directly, so this cannot pass by both
        // legs sharing a wrong fold: the A-fact at 14 now carries the
        // late quote's bid.
        let row = db.query("SELECT bid FROM blotter WHERE ts = 14").unwrap();
        let bids = crate::table::tests::flatten(&row, 0);
        assert_eq!(bids, [Some(8.5)]);
    }

    const JOINED_BARS: &str = "SELECT sym, ts / 4 AS bar, count(*) AS n, \
         avg(bid) AS ab, min(x) AS lo FROM trades \
         ASOF LEFT JOIN quotes ON trades.sym = quotes.sym \
         GROUP BY sym, ts / 4";

    /// The cycle-2 A/B: the aggregate view against the SAME aggregate
    /// run directly over the base join — independent leg, exact at
    /// every state.
    fn assert_joined_bars_match(db: &Database, view: &str) {
        let through = db
            .query(&format!("SELECT sym, bar, n, ab, lo FROM {view}"))
            .unwrap();
        let direct = db.query(JOINED_BARS).unwrap();
        assert_eq!(
            sorted_rows(&through),
            sorted_rows(&direct),
            "joined-aggregate view '{view}' diverged from the direct query"
        );
        assert!(through.num_rows() > 0, "vacuous joined-bars check");
    }

    #[test]
    fn a_bucketed_aggregate_over_an_asof_join_answers_exactly() {
        let mut db = blotter_db();
        db.create_materialized_view("bars", JOINED_BARS).unwrap();
        // Stale: the whole answer is a live join-fold.
        assert_joined_bars_match(&db, "bars");
        // First refresh: frontier 12 rounds to its own bucket's low
        // edge (12 = bucket 3's edge under width 4), so buckets 0..=2
        // materialize — 3 buckets folded, everything else live.
        assert_eq!(db.refresh_view("bars").unwrap(), 3);
        assert_joined_bars_match(&db, "bars");
        // In-order quote appends land above the ceiling: exact while
        // stale, no materialized bucket dirtied.
        for (qts, sym, bid) in [(13, "B", 2.13), (16, "A", 1.16)] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        assert_joined_bars_match(&db, "bars");
        db.refresh_view("bars").unwrap();
        assert_joined_bars_match(&db, "bars");
        // A late quote below the ceiling dirties its interval's
        // buckets — whole buckets, the aggregate's repair granularity.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(6),
                storage_lite::RowValue::Key("B"),
                storage_lite::RowValue::F64(9.9),
            ],
        )
        .unwrap();
        assert_joined_bars_match(&db, "bars"); // dirty, unrefreshed
        db.refresh_view("bars").unwrap();
        assert_joined_bars_match(&db, "bars");
        // Quote amend + delete below the ceiling; fact corrections.
        db.mutate("UPDATE quotes SET bid = 7.7 WHERE qts = 4")
            .unwrap();
        db.refresh_view("bars").unwrap();
        assert_joined_bars_match(&db, "bars");
        db.mutate("DELETE FROM quotes WHERE qts = 4").unwrap();
        assert_joined_bars_match(&db, "bars");
        db.refresh_view("bars").unwrap();
        assert_joined_bars_match(&db, "bars");
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        assert_joined_bars_match(&db, "bars");
        db.refresh_view("bars").unwrap();
        assert_joined_bars_match(&db, "bars");
        db.compact("trades").unwrap();
        db.compact("quotes").unwrap();
        assert_joined_bars_match(&db, "bars");
        // AS OF stays refused on join views, aggregate shape included.
        let error = db
            .query("SELECT bar FROM bars ASOF 5")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("#99"), "{error}");
    }

    #[test]
    fn a_dimension_attribute_group_key_rides_the_interval_repair() {
        // Group by a DIMENSION symbol (the quote venue): a quote
        // correction can move rows between groups, and the whole-
        // bucket re-fold must carry them — the "either side's columns"
        // half of the cycle-2 door.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "quotes",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("venue", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
                ]),
                "qts",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        for i in 0..16 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        for (qts, sym, venue, bid) in [
            (0, "A", "X", 1.0),
            (1, "B", "Y", 2.0),
            (6, "A", "Y", 1.6),
            (7, "B", "X", 2.7),
            (12, "A", "X", 1.12),
            (13, "B", "Y", 2.13),
        ] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::Key(venue),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        const BY_VENUE: &str = "SELECT venue, ts / 4 AS bar, sum(bid) AS s \
             FROM trades ASOF LEFT JOIN quotes ON trades.sym = quotes.sym \
             GROUP BY venue, ts / 4";
        db.create_materialized_view("vbars", BY_VENUE).unwrap();
        db.refresh_view("vbars").unwrap();
        let check = |db: &Database| {
            let through = db.query("SELECT venue, bar, s FROM vbars").unwrap();
            let direct = db.query(BY_VENUE).unwrap();
            assert_eq!(sorted_rows(&through), sorted_rows(&direct));
            assert!(through.num_rows() > 0);
        };
        check(&db);
        // A venue correction below the ceiling MOVES rows across
        // groups; the interval re-fold replaces the buckets whole.
        db.mutate("UPDATE quotes SET venue = 'Z' WHERE qts = 6")
            .unwrap();
        check(&db); // dirty, unrefreshed
        assert!(db.refresh_view("vbars").unwrap() >= 1);
        check(&db);
    }

    /// The star fixture: a fact table and a small keyed dimension
    /// (sector per symbol) — the lookup-table shape.
    fn star_db() -> Database {
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "dim",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("id", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("sector", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("weight", arrow_lite::ColumnType::F64, false),
                ]),
                "id",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        for i in 0..16 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        for (id, sym, sector, weight) in [(0, "A", "tech", 2.0), (1, "B", "energy", 3.0)] {
            db.append(
                "dim",
                &[
                    storage_lite::RowValue::I64(id),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::Key(sector),
                    storage_lite::RowValue::F64(weight),
                ],
            )
            .unwrap();
        }
        db
    }

    const STAR_BARS: &str = "SELECT sector, ts / 4 AS bar, sum(x) AS s, \
         count(*) AS n FROM trades JOIN dim ON trades.sym = dim.sym \
         GROUP BY sector, ts / 4";

    fn assert_star_matches(db: &Database, view: &str) {
        let through = db
            .query(&format!("SELECT sector, bar, s, n FROM {view}"))
            .unwrap();
        let direct = db.query(STAR_BARS).unwrap();
        assert_eq!(
            sorted_rows(&through),
            sorted_rows(&direct),
            "star view '{view}' diverged from the direct query"
        );
        assert!(through.num_rows() > 0, "vacuous star check");
    }

    #[test]
    fn a_star_view_folds_facts_incrementally_and_rebuilds_on_dim_change() {
        let mut db = star_db();
        db.create_materialized_view("sbars", STAR_BARS).unwrap();
        assert_star_matches(&db, "sbars"); // stale: all live
                                           // First refresh is the rebuild (u64::MAX, the rebuild count):
                                           // a star view has no ceiling, everything materializes.
        assert_eq!(db.refresh_view("sbars").unwrap(), u64::MAX);
        assert_star_matches(&db, "sbars");
        // Fact appends fold incrementally: 4 new keys = 1 new bucket.
        for i in 16..20 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        assert_star_matches(&db, "sbars"); // stale tail
        assert_eq!(db.refresh_view("sbars").unwrap(), 1);
        assert_star_matches(&db, "sbars");
        // A fact correction folds its bucket alone.
        db.mutate("UPDATE trades SET x = 500.0 WHERE ts = 3")
            .unwrap();
        assert_star_matches(&db, "sbars");
        assert_eq!(db.refresh_view("sbars").unwrap(), 1);
        assert_star_matches(&db, "sbars");
        // ANY dimension change rebuilds (F4): a sector move shifts
        // every bucket of that symbol across groups — and the read
        // must already be exact while the rebuild is pending.
        db.mutate("UPDATE dim SET sector = 'ai' WHERE sym = 'A'")
            .unwrap();
        assert_star_matches(&db, "sbars"); // dirty, unrefreshed: all live
        assert_eq!(db.refresh_view("sbars").unwrap(), u64::MAX);
        assert_star_matches(&db, "sbars");
        // A NEW dimension row (a symbol gaining coverage) is a dim
        // change like any other: rebuild, and exact while stale.
        db.append(
            "trades",
            &[
                storage_lite::RowValue::I64(20),
                storage_lite::RowValue::Key("C"),
                storage_lite::RowValue::F64(7.0),
                storage_lite::RowValue::F64(0.0),
            ],
        )
        .unwrap();
        db.append(
            "dim",
            &[
                storage_lite::RowValue::I64(2),
                storage_lite::RowValue::Key("C"),
                storage_lite::RowValue::Key("bio"),
                storage_lite::RowValue::F64(1.0),
            ],
        )
        .unwrap();
        assert_star_matches(&db, "sbars");
        assert_eq!(db.refresh_view("sbars").unwrap(), u64::MAX);
        assert_star_matches(&db, "sbars");
        db.compact("trades").unwrap();
        db.compact("dim").unwrap();
        assert_star_matches(&db, "sbars");
        // A duplicate dimension key is the executor's loud error, at
        // the view exactly as at the base — the door's data-level
        // precondition (the F7 widening rests on the dim being keyed).
        db.append(
            "dim",
            &[
                storage_lite::RowValue::I64(3),
                storage_lite::RowValue::Key("A"),
                storage_lite::RowValue::Key("dup"),
                storage_lite::RowValue::F64(9.0),
            ],
        )
        .unwrap();
        let error = db
            .query("SELECT sector, bar, s, n FROM sbars")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("not unique"), "{error}");
    }

    #[test]
    fn a_star_blotter_enriches_rows_and_rebuilds_on_dim_change() {
        // The bare-projection star shape: per-row enrichment from the
        // lookup table (the LEFT form keeps uncovered symbols with
        // NULL attributes).
        let mut db = star_db();
        db.append(
            "trades",
            &[
                storage_lite::RowValue::I64(16),
                storage_lite::RowValue::Key("C"),
                storage_lite::RowValue::F64(1.0),
                storage_lite::RowValue::F64(0.0),
            ],
        )
        .unwrap();
        const STAR_BLOTTER: &str = "SELECT ts, sym, x, sector, weight FROM trades \
             LEFT JOIN dim ON trades.sym = dim.sym";
        db.create_materialized_view("enriched", STAR_BLOTTER)
            .unwrap();
        let check = |db: &Database| {
            let through = db
                .query("SELECT ts, sym, x, sector, weight FROM enriched")
                .unwrap();
            let direct = db.query(STAR_BLOTTER).unwrap();
            assert_eq!(sorted_rows(&through), sorted_rows(&direct));
            assert!(through.num_rows() > 0);
        };
        check(&db);
        assert_eq!(db.refresh_view("enriched").unwrap(), u64::MAX);
        check(&db);
        db.mutate("UPDATE dim SET weight = 9.0 WHERE sym = 'B'")
            .unwrap();
        check(&db); // dirty: whole read live until the rebuild
        assert_eq!(db.refresh_view("enriched").unwrap(), u64::MAX);
        check(&db);
        for i in 17..20 {
            db.append("trades", &linear_row(i)).unwrap();
        }
        check(&db);
        assert!(db.refresh_view("enriched").unwrap() < u64::MAX); // incremental
        check(&db);
    }

    #[test]
    fn a_frontier_regression_dematerializes_the_stranded_band() {
        // Found by the repo-wide code review, reproduced there: when a
        // correction LOWERS the reference frontier, the old refresh
        // clipped the correction interval against the new ceiling,
        // swallowed its knowledge coordinate, and left the rows in
        // [new ceiling, old ceiling) marked clean but permanently
        // stale. The refresh must victimize that band — those rows are
        // live-half territory again.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "quotes",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
                ]),
                "qts",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        for ts in [60, 70, 90] {
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::Key("A"),
                    storage_lite::RowValue::F64(ts as f64),
                    storage_lite::RowValue::F64(0.0),
                ],
            )
            .unwrap();
        }
        for (qts, bid) in [(50, 1.0), (100, 2.0)] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key("A"),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        db.refresh_view("blotter").unwrap(); // ceiling 100; ts 60/70/90 materialized
        assert_blotter_matches(&db, "blotter");
        // The frontier REGRESSES: delete the frontier quote.
        db.mutate("DELETE FROM quotes WHERE qts = 100").unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap(); // ceiling shrinks to 50
        assert_blotter_matches(&db, "blotter");
        // An in-order arrival BELOW the old ceiling: under the bug the
        // band [50, 100) was still marked clean and this quote's birth
        // was swallowed — ts = 90 stayed matched to the dead world.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(80),
                storage_lite::RowValue::Key("A"),
                storage_lite::RowValue::F64(3.0),
            ],
        )
        .unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        let bid = db.query("SELECT bid FROM blotter WHERE ts = 90").unwrap();
        assert_eq!(crate::table::tests::flatten(&bid, 0), [Some(3.0)]);
    }

    #[test]
    fn a_strict_asof_blotter_repairs_through_its_inclusive_edge() {
        // Found by the repo-wide code review, reproduced there: under
        // StrictlyBefore, a fact at exactly the symbol's NEXT
        // reference key still matches the corrected row before it, so
        // the correction interval must include `next` itself — the
        // at-or-before endpoint left that fact silently stale.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "quotes",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
                ]),
                "qts",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        db.append(
            "trades",
            &[
                storage_lite::RowValue::I64(20),
                storage_lite::RowValue::Key("A"),
                storage_lite::RowValue::F64(1.0),
                storage_lite::RowValue::F64(0.0),
            ],
        )
        .unwrap();
        for (qts, bid) in [(10, 1.0), (20, 7.0), (100, 9.0)] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key("A"),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        const STRICT: &str = "SELECT ts, sym, x, bid FROM trades \
             ASOF LEFT JOIN quotes \
             ON trades.sym = quotes.sym AND quotes.qts < trades.ts";
        db.create_materialized_view("strict", STRICT).unwrap();
        db.refresh_view("strict").unwrap();
        // Strictly-before: the ts = 20 fact matches the quote at 10,
        // never the one at 20.
        let bid = db.query("SELECT bid FROM strict WHERE ts = 20").unwrap();
        assert_eq!(crate::table::tests::flatten(&bid, 0), [Some(1.0)]);
        // Correct the matched quote: its interval must reach THROUGH
        // qts = 20 (the fact at exactly 20 still matches it).
        db.mutate("UPDATE quotes SET bid = 5.0 WHERE qts = 10")
            .unwrap();
        let check = |db: &Database| {
            let through = db.query("SELECT ts, sym, x, bid FROM strict").unwrap();
            let direct = db.query(STRICT).unwrap();
            assert_eq!(sorted_rows(&through), sorted_rows(&direct));
            let bid = db.query("SELECT bid FROM strict WHERE ts = 20").unwrap();
            assert_eq!(crate::table::tests::flatten(&bid, 0), [Some(5.0)]);
        };
        check(&db); // dirty, unrefreshed
        db.refresh_view("strict").unwrap();
        check(&db);
    }

    #[test]
    fn a_join_views_rebuild_floor_folds_only_below_the_ceiling() {
        // The tamper path, exercised for join views: a stamp no crash
        // can explain meets the rebuild floor, and the rebuild's fold
        // converts its range to BUCKET runs — the key-space form
        // over-folded past the ceiling for widths > 1 (found by both
        // reviewers, previously untested).
        let dir = std::env::temp_dir().join(format!("tallydb-jfloor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let trades_dir = dir.join("trades");
        let quotes_dir = dir.join("quotes");
        let view_dir = dir.join("jb");
        for sub in [&trades_dir, &quotes_dir, &view_dir] {
            std::fs::create_dir_all(sub).unwrap();
        }
        let mut trades = Table::persistent("trades", m1_schema(), "ts", &trades_dir).unwrap();
        let mut quotes = Table::persistent(
            "quotes",
            arrow_lite::Schema::new(vec![
                arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
            ]),
            "qts",
            &quotes_dir,
        )
        .unwrap();
        for i in 0..20 {
            trades.append(&linear_row(i)).unwrap();
        }
        for (qts, sym, bid) in [(0, "A", 1.0), (1, "B", 2.0), (10, "A", 1.5), (11, "B", 2.5)] {
            quotes
                .append(&[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ])
                .unwrap();
        }
        const JB: &str = "SELECT sym, ts / 4 AS bar, count(*) AS n, avg(bid) AS ab \
             FROM trades ASOF LEFT JOIN quotes ON trades.sym = quotes.sym \
             GROUP BY sym, ts / 4";
        {
            let mut view =
                MaterializedView::persistent("jb", JB, &trades, Some(&quotes), &view_dir).unwrap();
            view.refresh(&mut trades, Some(&mut quotes)).unwrap();
        }
        // Tamper: a fact stamp far past anything spent.
        let record = std::fs::read(view_dir.join(DEFINITION_FILE)).unwrap();
        let (_, _, _, _, join) = decode_definition(&record).unwrap();
        let join = join.unwrap();
        std::fs::write(
            view_dir.join(DEFINITION_FILE),
            encode_definition(1_000_000, 0, "trades", JB, Some(&join)),
        )
        .unwrap();
        let mut view = MaterializedView::open(
            "jb",
            &view_dir,
            &trades,
            Some(&quotes),
            StoreOptions::default(),
        )
        .unwrap();
        assert_eq!(
            view.refresh(&mut trades, Some(&mut quotes)).unwrap(),
            u64::MAX
        );
        // The rebuild honored the ceiling (frontier 11 rounds to
        // bucket edge 8 under width 4): only buckets 0..=1 hold rows.
        let max_bar = single_i64_cell(&view.table.query("SELECT max(bar) AS m FROM jb").unwrap())
            .expect("the rebuild materialized no rows");
        assert!(
            max_bar < 8,
            "the rebuild materialized past the ceiling: max bar {max_bar}"
        );
        // And the union read is exact regardless.
        let mut db = Database::new();
        db.add_table(trades).unwrap();
        db.add_table(quotes).unwrap();
        db.add_view(view).unwrap();
        let through = db.query("SELECT sym, bar, n, ab FROM jb").unwrap();
        let direct = db.query(JB).unwrap();
        assert_eq!(sorted_rows(&through), sorted_rows(&direct));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn join_views_hold_on_negative_keys_and_composed_reads() {
        // Negative fact and reference keys put truncating division's
        // double-width bucket 0 and negative ceilings on the path (no
        // oracle covers negatives — this is their in-crate home); the
        // read side composes a WHERE and an aggregate over the view.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("trades", m1_schema(), "ts", 4).unwrap())
            .unwrap();
        db.add_table(
            Table::with_segment_rows(
                "quotes",
                arrow_lite::Schema::new(vec![
                    arrow_lite::Field::new("qts", arrow_lite::ColumnType::I64, false),
                    arrow_lite::Field::new("sym", arrow_lite::ColumnType::Key, false),
                    arrow_lite::Field::new("bid", arrow_lite::ColumnType::F64, false),
                ]),
                "qts",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        for i in 0..20i64 {
            let ts = -30 + i * 3;
            db.append(
                "trades",
                &[
                    storage_lite::RowValue::I64(ts),
                    storage_lite::RowValue::Key(if i % 2 == 0 { "A" } else { "B" }),
                    storage_lite::RowValue::F64(i as f64),
                    storage_lite::RowValue::F64(0.0),
                ],
            )
            .unwrap();
        }
        for (qts, sym, bid) in [
            (-25, "A", 1.0),
            (-24, "B", 2.0),
            (-10, "A", 1.5),
            (-9, "B", 2.5),
            (5, "A", 1.9),
        ] {
            db.append(
                "quotes",
                &[
                    storage_lite::RowValue::I64(qts),
                    storage_lite::RowValue::Key(sym),
                    storage_lite::RowValue::F64(bid),
                ],
            )
            .unwrap();
        }
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        db.refresh_view("blotter").unwrap(); // negative-through-zero ceiling path
        assert_blotter_matches(&db, "blotter");
        // A late quote at a negative key repairs its interval.
        db.append(
            "quotes",
            &[
                storage_lite::RowValue::I64(-20),
                storage_lite::RowValue::Key("B"),
                storage_lite::RowValue::F64(9.9),
            ],
        )
        .unwrap();
        assert_blotter_matches(&db, "blotter");
        db.refresh_view("blotter").unwrap();
        assert_blotter_matches(&db, "blotter");
        // Composed reads over the view: a WHERE crossing zero, and an
        // aggregate over the view's rows — both against the direct
        // join under the same composition.
        let through = db
            .query("SELECT ts, sym, x, bid FROM blotter WHERE ts >= -12")
            .unwrap();
        let direct_rows = db.query(&format!("{BLOTTER} WHERE ts >= -12")).unwrap();
        assert!(through.num_rows() > 0);
        assert_eq!(sorted_rows(&through), sorted_rows(&direct_rows));
        let through = db
            .query("SELECT sym, avg(bid) AS ab FROM blotter GROUP BY sym")
            .unwrap();
        let direct = db
            .query(
                "SELECT sym, avg(bid) AS ab FROM trades \
                 ASOF LEFT JOIN quotes ON trades.sym = quotes.sym GROUP BY sym",
            )
            .unwrap();
        assert_eq!(sorted_rows(&through), sorted_rows(&direct));
    }

    #[test]
    fn a_blotter_repairs_same_symbol_multi_corrections_in_one_window() {
        // The interval lemma's hardest case: several corrections to
        // ONE symbol land in a single refresh window, so a killed
        // quote's current-state next skips other rows killed in the
        // same window and the per-row intervals must union to cover
        // every changed match. Amend A@8 (a kill-then-rebirth chain),
        // kill the rebirth, and kill A@4 — the kill at 4's surviving
        // next is 12, two touched rows away.
        let mut db = blotter_db();
        db.create_materialized_view("blotter", BLOTTER).unwrap();
        db.refresh_view("blotter").unwrap();
        db.mutate("UPDATE quotes SET bid = 8.8 WHERE qts = 8")
            .unwrap();
        db.mutate("DELETE FROM quotes WHERE qts = 8").unwrap();
        db.mutate("DELETE FROM quotes WHERE qts = 4").unwrap();
        assert_blotter_matches(&db, "blotter"); // dirty, unrefreshed
        let folded = db.refresh_view("blotter").unwrap();
        assert!(folded >= 1, "the correction window dirtied its intervals");
        assert_blotter_matches(&db, "blotter");
        // The concrete fallback: with A@4 and both rows at A@8 dead,
        // an A fact between them reaches all the way back to A@0.
        let bid = db.query("SELECT bid FROM blotter WHERE ts = 10").unwrap();
        assert_eq!(crate::table::tests::flatten(&bid, 0), [Some(1.0)]);
    }
}
