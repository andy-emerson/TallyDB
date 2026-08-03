//! Maintained views (#83, tranche 1): bucketed single-table aggregates
//! kept fresh as ordered data arrives.
//!
//! ## The model, in one paragraph
//!
//! A maintained view is a **fold over the ingest sequence**: a real
//! table (segments, WAL, `AS OF` — all inherited) holding the result of
//! a bucketed aggregate query, plus a **stamp** — the source table's
//! ingest-sequence watermark below which the materialization is
//! complete. Everything at or above the stamp is the view's unfolded
//! tail; a refresh folds it and advances the stamp, and a query never
//! waits for one: the **union read** answers exactly however stale the
//! materialization — clean materialized buckets plus a live fold of
//! whatever the stamp does not cover — and a past coordinate answers
//! by recompute (`view AS OF s` is the definition over
//! `base AS OF s`; the materialization is never the authority). Corrections need no
//! bookkeeping of their own: the buckets they touch are **derivable**
//! from the source's knowledge history, so the only durable view state
//! is the stamp, written strictly after the materialization it
//! describes is flushed — everything a stamp covers therefore survives
//! any crash the source's own WAL contract admits, and a crash
//! elsewhere just leaves the stamp old, which the next refresh heals.
//! Repair is always re-fold-from-base (uniform repair, ruled
//! 2026-08-02 on #83): no accumulator state, no delta arithmetic, no
//! f64 subtraction hazard.
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
    /// Opened via [`MaterializedView::open_read_only`]: refresh
    /// refuses, the union read serves.
    read_only: bool,
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
            width: 0,
            dir: None,
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
        dir: impl AsRef<Path>,
    ) -> Result<MaterializedView, EngineError> {
        let (schema, bucket) = validated_definition(sql, source)?;
        let table = Table::persistent(name, schema, &bucket, dir.as_ref())?;
        let view = MaterializedView {
            table,
            sql: sql.to_owned(),
            source: source.name().to_owned(),
            stamp: 0,
            width: 0,
            dir: Some(dir.as_ref().to_path_buf()),
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
        options: StoreOptions,
    ) -> Result<MaterializedView, EngineError> {
        let (stamp, width, source_name, sql) = read_definition(dir.as_ref(), name, source)?;
        let table = Table::open(name, dir.as_ref(), options)?;
        Ok(MaterializedView {
            table,
            sql,
            source: source_name,
            stamp,
            width,
            dir: Some(dir.as_ref().to_path_buf()),
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
    ) -> Result<MaterializedView, EngineError> {
        let (stamp, width, source_name, sql) = read_definition(dir.as_ref(), name, source)?;
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
            read_only: true,
        })
    }

    /// The source-table name a persisted view's definition record
    /// names — what a directory scanner (the console) reads to open
    /// the source before the view, without opening the view first.
    pub fn stored_source(dir: impl AsRef<Path>) -> Result<String, EngineError> {
        let record = std::fs::read(dir.as_ref().join(DEFINITION_FILE))
            .map_err(|error| definition_error(format!("reading {DEFINITION_FILE}: {error}")))?;
        let (_, _, source, _) = decode_definition(&record)?;
        Ok(source)
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

    /// The view's output schema — the shape its rows answer with.
    pub fn schema(&self) -> &Schema {
        self.table.schema()
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
    pub fn refresh(&mut self, source: &mut Table) -> Result<u64, EngineError> {
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
        source.flush()?;
        let now = source.next_sequence();
        let mut definition = Definition::of(&self.sql, source, self.width)?;
        if definition.running.is_some() && self.width == 0 {
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
                    definition = Definition::of(&self.sql, source, width)?;
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
        let definition = Definition::of(&self.sql, source, self.width)?;
        if let Some(running) = &definition.running {
            return self.query_running(source, &definition, running, user_plan);
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
            return run_over_output(&running.output, answers.batches, &current, source);
        }
        let partials = match definition.touched_runs(source, self.stamp)? {
            None => self.table.execute_plan(&select_everything(&self.table)?)?,
            Some(runs) => {
                let fresh = source.execute_plan(&definition.restricted_to(&runs, source))?;
                let mut clean = select_everything(&self.table)?;
                clean.predicate = Some(Predicate::Not(Box::new(definition.view_ranges(&runs))));
                let mut clean = self.table.execute_plan(&clean)?;
                clean.batches.extend(fresh.batches);
                clean
            }
        };
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
            source,
        )
    }

    /// Runs `user_plan` over an ad-hoc union of view-shaped batches, as
    /// scratch segments. Orderedness is inspected per batch, never
    /// assumed: a stale view's union can interleave bucket ranges, and
    /// the executor's own ordering checks then govern — the same
    /// stance every disordered table gets (correct, less optimized,
    /// `refresh` + `compact` restore the fast path).
    fn over_scratch(
        &self,
        clean: impl Iterator<Item = arrow_lite::RecordBatch>,
        fresh: query_lite::QueryOutput,
        user_plan: &Plan,
    ) -> Result<query_lite::QueryOutput, EngineError> {
        use storage_lite::{Segment, SegmentHandle};
        let ordering_key = self
            .table
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == self.table.ordering_key())
            .expect("the view table validated its ordering key at construction");
        let handles: Vec<SegmentHandle> = clean
            .chain(fresh.batches)
            .filter(|batch| batch.num_rows() > 0)
            .map(|batch| {
                let ordered = is_non_decreasing(&batch, ordering_key);
                SegmentHandle::resident(
                    std::sync::Arc::new(Segment::from_batch_unpruned(batch, ordering_key, ordered)),
                    None,
                )
            })
            .collect();
        query_lite::execute_with_ordering_key(
            self.table.schema(),
            &handles,
            ordering_key,
            user_plan,
            &self.table.current_registry(),
        )
        .map_err(EngineError::Query)
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
        let record = encode_definition(self.stamp, self.width, &self.source, &self.sql);
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
fn read_definition(
    dir: &Path,
    name: &str,
    source: &Table,
) -> Result<(u64, u64, String, String), EngineError> {
    let record = std::fs::read(dir.join(DEFINITION_FILE))
        .map_err(|error| definition_error(format!("reading {DEFINITION_FILE}: {error}")))?;
    let (stamp, width, source_name, sql) = decode_definition(&record)?;
    if source_name != source.name() {
        return Err(definition_error(format!(
            "view '{name}' is over '{source_name}', not '{}'",
            source.name()
        )));
    }
    validated_definition(&sql, source)?;
    Ok((stamp, width, source_name, sql))
}

/// A lowered, validated view definition plus its bucket arithmetic —
/// what both halves of the machinery share: the refresh restricts and
/// folds with it, the union read restricts and tops up with it. For a
/// running view, `plan` is the **synthesized partials materialization**
/// (a legal bucketed plan over the hidden bucket), and `running`
/// carries what the read needs to reassemble the user-facing answer.
struct Definition {
    plan: Plan,
    bucket_name: String,
    divide: i64,
    view_scale: i64,
    running: Option<RunningRead>,
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

impl Definition {
    /// Builds the definition for `sql` over `source`. `width` is a
    /// running view's hidden-bucket width in ordering-key units — `0`
    /// means not yet chosen (the first refresh with data chooses it),
    /// and the caller must not fold with an unsized definition; a
    /// placeholder width of 1 keeps the synthesized plan well-formed
    /// for schema derivation.
    fn of(sql: &str, source: &Table, width: u64) -> Result<Definition, EngineError> {
        let plan = lower_plan(sql).map_err(EngineError::Query)?;
        match eligible_shape(&plan, source)? {
            Shape::Bucketed(bucket, bucket_name) => {
                let (divide, view_scale) = bucket_arithmetic(&bucket);
                Ok(Definition {
                    plan,
                    bucket_name,
                    divide,
                    view_scale,
                    running: None,
                })
            }
            Shape::Running => synthesize_running(plan, source, width.max(1) as i64),
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
    for item in items {
        match item {
            AggItem::Key { .. } => {
                internal_items.push(item.clone());
                combine_items.push(item.clone());
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
            keys: keys.clone(),
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
        running: Some(RunningRead {
            user,
            combine,
            finalize,
            output: Schema::new(output_fields),
        }),
    })
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
                let Column::Numeric(NumericData::F64(sums)) = &batch.columns()[*sum] else {
                    unreachable!("an AVG decomposition's sum partial is f64")
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
                    let defined = count > 0 && sums.is_valid(row);
                    validity.push(defined);
                    values.push(if defined {
                        sums.values().as_slice()[row] / count as f64
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
/// paths, whose batches carry the bare user schema).
fn run_over_output(
    output: &Schema,
    batches: Vec<arrow_lite::RecordBatch>,
    user_plan: &Plan,
    source: &Table,
) -> Result<query_lite::QueryOutput, EngineError> {
    use arrow_lite::{Column, NumericColumn, NumericData, RecordBatch};
    use storage_lite::{Segment, SegmentHandle};
    let ordering_key = output.fields().len() - 1; // __row, by construction
    let handles: Vec<SegmentHandle> = batches
        .into_iter()
        .filter(|batch| batch.num_rows() > 0)
        .map(|batch| {
            let batch = if batch.columns().len() == output.fields().len() {
                batch
            } else {
                let rows = batch.num_rows() as i64;
                let mut columns = batch.columns().to_vec();
                columns.push(Column::Numeric(NumericData::I64(
                    NumericColumn::new_non_null((0..rows).collect::<arrow_lite::Buffer<i64>>()),
                )));
                RecordBatch::new(output.clone(), columns)
            };
            SegmentHandle::resident(
                std::sync::Arc::new(Segment::from_batch_unpruned(batch, ordering_key, true)),
                None,
            )
        })
        .collect();
    query_lite::execute_with_ordering_key(
        output,
        &handles,
        ordering_key,
        user_plan,
        &source.current_registry(),
    )
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
/// the **materialization** table's schema and its ordering-key column
/// name — the definition's own bucket for a bucketed view, the hidden
/// bucket of the partials for a running one.
fn validated_definition(sql: &str, source: &Table) -> Result<(Schema, String), EngineError> {
    let plan = lower_plan(sql).map_err(EngineError::Query)?;
    if plan.table != source.name() {
        return Err(EngineError::WrongTable {
            expected: source.name().to_owned(),
            got: plan.table,
        });
    }
    // A placeholder width of 1 keeps a running synthesis well-formed;
    // the schema does not depend on the width's value.
    let definition = Definition::of(sql, source, 1)?;
    let bucket = definition.bucket_name.clone();
    let schema = output_schema(&definition.plan, source)?;
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

/// What kind of maintainable definition a plan is.
enum Shape {
    /// Tranche 1: a bucketed aggregate — the materialization IS the
    /// answer, keyed on the definition's own bucket.
    Bucketed(GroupKey, String),
    /// Tranche 2: a running aggregate — no bucket in the definition,
    /// so the materialization stores per-hidden-bucket **partials**
    /// and the answer is assembled at read by combining them.
    Running,
}

/// The eligibility check: classifies a definition as bucketed
/// (tranche 1) or running (tranche 2), and refuses everything else by
/// name — naming the tranche that will admit it where one is planned.
fn eligible_shape(plan: &Plan, source: &Table) -> Result<Shape, EngineError> {
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
             cumulative window shapes are the remainder of tranche 2 \
             (#83)",
        );
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
        for reserved in [crate::partials::HIDDEN_BUCKET] {
            if source
                .schema()
                .fields()
                .iter()
                .any(|f| f.name() == reserved)
            {
                return refuse(
                    "a running view over a source with a '__bucket' column — \
                     the name is reserved for the partials materialization",
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
fn encode_definition(stamp: u64, width: u64, source: &str, sql: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 2 + 8 + 8 + 8 + source.len() + sql.len() + 4);
    out.extend_from_slice(b"TDBV");
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&stamp.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&(source.len() as u32).to_le_bytes());
    out.extend_from_slice(source.as_bytes());
    out.extend_from_slice(&(sql.len() as u32).to_le_bytes());
    out.extend_from_slice(sql.as_bytes());
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn decode_definition(bytes: &[u8]) -> Result<(u64, u64, String, String), EngineError> {
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
    if version != 1 && version != 2 {
        return Err(corrupt(&format!("unknown version {version}")));
    }
    let stamp = u64::from_le_bytes(payload[6..14].try_into().expect("sized"));
    let (width, mut at) = if version == 2 {
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
    Ok((stamp, width, source, sql))
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
        // The deferred refusal names the tranche remainder. (The
        // no-bucket aggregate tranche 1 refused here is now the
        // RUNNING shape — accepted, tested in the running battery.)
        refused("SELECT x FROM trades", "tranche 2");
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
        std::fs::write(&path, encode_definition(0, 0, "quotes", OHLC)).unwrap();
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
            let mut view = MaterializedView::persistent("ohlc", OHLC, &source, &view_dir).unwrap();
            view.refresh(&mut source).unwrap();
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
            MaterializedView::open("ohlc", &view_dir, &source, StoreOptions::default()).unwrap();
        assert_eq!(view.refresh(&mut source).unwrap(), 1);
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
        let record = encode_definition(0, 0, "trades", OHLC);
        std::fs::write(view_dir.join(DEFINITION_FILE), record).unwrap();
        let mut view =
            MaterializedView::open("ohlc", &view_dir, &source, StoreOptions::default()).unwrap();
        assert_eq!(view.stamp(), 0);
        view.refresh(&mut source).unwrap();
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
            let mut view = MaterializedView::persistent("ohlc", OHLC, &writer, &view_dir).unwrap();
            view.refresh(&mut writer).unwrap();
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
            MaterializedView::open_read_only("ohlc", &view_dir, db.table("trades").unwrap())
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
            let mut view = MaterializedView::persistent("ohlc", OHLC, &source, &view_dir).unwrap();
            view.refresh(&mut source).unwrap();
            // Refresh flushes the source, so anything IT saw is
            // durable; the losable tail is what arrives after the
            // last refresh.
            for i in 8..16 {
                source.append(&linear_row(i)).unwrap();
            }
            // The critical refresh: it sees the buffered tail, so its
            // stamp covers it — which is exactly why it must make the
            // tail durable first.
            view.refresh(&mut source).unwrap();
            let recomputed = source.query(OHLC).unwrap();
            let via_union = view
                .query_union(
                    &source,
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
            MaterializedView::open("ohlc", &view_dir, &source, StoreOptions::default()).unwrap();
        view.refresh(&mut source).unwrap();
        let recomputed = source.query(OHLC).unwrap();
        let materialized = view
            .query_union(
                &source,
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
            let mut view = MaterializedView::persistent("ohlc", OHLC, &source, &view_dir).unwrap();
            view.refresh(&mut source).unwrap();
        }
        // The tamper: a stamp far past anything the source has spent.
        std::fs::write(
            view_dir.join(DEFINITION_FILE),
            encode_definition(1_000_000, 0, "trades", OHLC),
        )
        .unwrap();
        let mut view =
            MaterializedView::open("ohlc", &view_dir, &source, StoreOptions::default()).unwrap();
        assert_eq!(view.refresh(&mut source).unwrap(), u64::MAX);
        let recomputed = source.query(OHLC).unwrap();
        let materialized = view
            .query_union(
                &source,
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
                MaterializedView::persistent("totals", RUNNING, &source, &view_dir).unwrap();
            view.refresh(&mut source).unwrap();
        }
        // The width survives the record round trip (v2) — read back
        // from the bytes, not inferred — and the reopened view keeps
        // folding under it rather than re-sizing.
        let record = std::fs::read(view_dir.join(DEFINITION_FILE)).unwrap();
        let (_, width, _, _) = decode_definition(&record).unwrap();
        assert!(width > 0, "the chosen width was not persisted");
        let mut view =
            MaterializedView::open("totals", &view_dir, &source, StoreOptions::default()).unwrap();
        source
            .mutate("UPDATE trades SET x = 50.0 WHERE ts = 2")
            .unwrap();
        assert_eq!(view.refresh(&mut source).unwrap(), 1);
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
        let ro_view =
            MaterializedView::open_read_only("totals", &view_dir, db.table("trades").unwrap())
                .unwrap();
        db.add_view(ro_view).unwrap();
        assert_running_matches(&db, "totals", RUNNING_COLUMNS);
        assert!(db.refresh_view("totals").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
