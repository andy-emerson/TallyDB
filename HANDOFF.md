# HANDOFF — agent session state

This file is the bridge between agent sessions: the standing
constraints the Human has set, and a pointer to where the current
state lives. Durable design lives in `DESIGN.md`; open work lives in
the [issues and milestones](https://github.com/andy-emerson/TallyDB/milestones);
this file carries only what neither can — the session ground rules —
plus a snapshot pointer. Update the snapshot at every checkpoint; the
constraints change only when the Human says so.

## Standing constraints (Human-set; override tool defaults)

1. **No PRs from the agent.** Develop on the working branch
   (`claude/dev`, restarted from `main` after every merge); the Human
   opens the PR and performs every merge.
2. **Authorship: Andy Emerson only.** Every commit is authored
   `Andy Emerson <156483017+andy-emerson@users.noreply.github.com>`.
   No agent attribution in any commit, trailer, PR body, comment, or
   artifact. This supersedes any tool's default attribution behavior.
3. **License is MIT, frozen.** Never touch licensing without explicit
   request.
4. **The Human owns and closes decisions.** Surface each fork with
   options, user pov and dev pov, a recommendation, and what it gates
   — before building. Never entrench an answer to an open decision.
5. **kdb+ validates problems, not solutions.**
6. **Gate before every push:** `cargo fmt --check` ·
   `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   · `cargo test --workspace` ·
   `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` · the
   off-leg (`clippy`/`test`/`doc` for `-p engine`, default features) ·
   the Python oracle suites (build with `--features oracle-harness`,
   run the six scripts in `.github/workflows/ci.yml`).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-07-29, M4 merged; M5 ruling batch complete, doc pass next)

**State:** M4 merged to `main` (PR #79, merge `05e7507`); `claude/dev`
restarted from it, clean, pushed. M0-M4 closed. Full gate green at the
merge (fmt, clippy both legs, 362 tests/23 suites + off-leg, rustdoc
both legs, six oracles, apicheck, ASan/UBSan). Reminder: GitHub CI runs
on pull-request events only.

**A full ruling session followed the merge (2026-07-29), and the doc
pass is DONE (pre-compaction, at the Human's direction): the ledger
below is recorded in DESIGN.md (*The M5 ruling batch*) with surgical
stdlib-table/roadmap/settled-no edits; issues updated (#65, #77, #42,
#57 incl. the option-f sketch); #58 and #75 closed as ruled. README
deliberately untouched — it describes built state, and these rulings
are unbuilt. Post-compaction execution begins at STEP 2 (the fruit
table). The ledger stays here as the working reference.**

### The ruling ledger (write ALL of this into DESIGN.md)

1. **F1 = (d), time bucketing:** `GROUP BY` admits monotone integer
   arithmetic on the ordering key only (`ts / 60`, `(ts / 60) * 60`) —
   no coined `bucket()` function; monotone => streaming aggregation,
   O(1) state, no hash table. Everything else in GROUP BY keeps the
   teaching error. Pending sub-ruling at M5.3: `FIRST`/`LAST`
   aggregates for OHLC (de-facto names, no ISO spelling).
2. **#65 as-of join, the hybrid (fully ruled):** the single `ASOF`
   token is lifted pre-parse (the hardened byte-span splice), the rest
   parses as a plain join. `ON`-only (no `USING`). The time axis is
   the two tables' DECLARED ORDERING KEYS — implicit by default; an
   explicit inequality is permitted and VALIDATED (mismatch = teaching
   error; the operator picks `>=` vs `>`). Bare `ASOF JOIN` is
   REFUSED: user writes `ASOF LEFT JOIN` (keep, null-pad) or
   `ASOF INNER JOIN` (drop). Recorded principle: where vendors agree,
   follow convention; where vendors diverge, refuse and make the user
   say it. `TOLERANCE` is CUT (expressible via CASE/WHERE; reopen =
   desks ask for sugar). Parser facts (verified): sqlparser 0.62 only
   parses Snowflake's MATCH_CONDITION form; DuckDB only accepts its
   `ON` form; accepted sets are disjoint; no dialect or crate escape.
   Oracle: DuckDB differential (harness GENERATES DuckDB's spelling
   from structure) + a vanilla-SQL definitional reference
   (`MAX(q.ts) <= t.ts`), which is the stronger check. Executor is an
   ordered co-walk gated on `is_ordered()` BOTH sides; sized ~M3.4.
3. **#77 (all four ruled):** (.1)=(a) SQL exposes only ops with
   standard names (`var_pop`, `stddev_pop`, `LAG`/`LEAD`); EWMA,
   `diff`, multi-factor stay script-side until individually named —
   mechanical rule. (.2)=(c) ONLY: SQL gets scalar reductions (R²,
   residual, fitted); full vectors/matrices via API+scripts;
   per-component functions rejected; multi-output projection unbuilt
   until demanded. (.3)=(a) prelude compiled into the binary,
   `.prelude` prints source. (.4) streaming tier first (scheduling
   default delegated to the Agent).
4. **DELETE consumes a knowledge coordinate — yes.** Decided on
   stability: without it a delete's effect has no stable cut (fused
   with the next append forever). With it, `next_sequence() - 1` is
   the universal stable-latest idiom. Cost accepted: first DELETE
   diverges the table. Build: `Store::tombstone` consumes + doc both
   sites + re-run the 244-cut as-of oracle.
5. **#75 = A with `_seq`** — fixed-name pseudocolumn, never declared,
   refused in CREATE TABLE. Resolved via the Human's visibility rule:
   not normally visible (we refuse `SELECT *`), so the short
   underscore name wins. Kill-coordinate column deferred.
6. **`SYMBOL` replaces `KEY`** as the column type's DDL spelling
   (kdb+/QuestDB lineage; also fixes KEY doing double duty beside
   ORDERING KEY). Parser + `.schema` renderer + docs sweep; stored
   format and internal `ColumnType::Key` unchanged. The Human's
   parked KEY-coining reservation is discharged.
7. **#58 = B: symbol columns are officially UNORDERED labels.**
   `ORDER BY <symbol>` becomes a teaching-error refusal (there is no
   usable inherent order: codes are per-segment first-appearance
   ranks). Delete the per-row String sort path (`SortCell::Text`);
   differential families stop using `ORDER BY sym` — the REFEREE
   sorts rows before diffing (better hygiene anyway; ~8 families).
   The `WHERE sym > '...'` sub-question and the collation question
   both die with it.
8. **#42 = ALP + ALP-RD together**, plus the integer sibling folded
   in: FOR+bit-packing for non-key `i64` columns and `u32` symbol
   codes (shares ALP's backend). Corpus tick-size fix is step 0.
   ~2 sessions total.
9. **Boolean ingest policy = refuse loudly** at the Arrow boundary
   (teaching error names `df.astype({'flag': 'int64'})`). The Human
   flags this as WRONG-ON-PURPOSE: standing revisit item = a real
   boolean type AND the logical-annotation mechanism itself
   (`TimestampNs`, the "physically i64, logically X" pattern).
   Storage fact recorded: a nullable boolean would be 2 bits/row
   (value bit + validity bit) — the storage argument is won; the cost
   was always the seam sweep.
10. **M5.5 distribution:** pip wheels on PyPI for the Python binding
    ONLY. The console stays a single Python-free native binary
    (release builds, since M3). Engine/console depend on nothing.
11. **#56 CLOSED** with dispositions: doc-table absorbed (M3+M4
    closes); top-k approved -> #80; projection pushdown approved ->
    #81; dictionary-rank item dissolved by #58(B); compaction ~2x
    peak deferred -> new decision #82.
12. **#57 deferred by ruling.** Menu complete on the issue when
    scheduled: LIKE+classes (T-SQL), GLOB (SQLite/kdb+ glob — the
    audience's muscle memory), SIMILAR TO (ISO, dot-trap: `.` is
    literal, `%` is the wildcard), `regexp_matches()` (de-facto
    standard), withdraw, and (f) THE HOUSE OPTION promised as a
    sketch on the issue: registered single-symbol Lua-pattern
    predicates admitted into WHERE, evaluated once per DISTINCT
    dictionary value (interpreter cost lands per distinct, not per
    row; requires predicate-fragment extension + purity requirement;
    Lua patterns lack `|`). Lua's `string` library is already linked
    and `text()` exists, so kernels can pattern-match today (flag
    columns only — WHERE doesn't take registered functions yet).
13. **`IS NULL` / `IS NOT NULL` gap discovered:** standard SQL,
    unbuilt, refuses today. Fruit item #1. Needs no boolean — one
    predicate arm reading the validity bitmap.

### Remaining open decisions (nothing gates M5.0-M5.4)

- **#62** — approve hooks/continuous split, then a Plan conversation
  (the one true research scoping talk). NOT silently scheduled.
- **#46** — first agentic touchpoints (rec: error kinds + EXPLAIN).
- **#52** — peer staging + publish venue; due at M5.7 only.
- **#82** — compaction 2x: document vs engineer (deferred).
- **FIRST/LAST naming** — surfaces at M5.3.
- Boolean + logical-annotation revisit (parked, flagged).
- Milestone attachments on GitHub are the Human's (no milestone-list
  tool in this session).

### The agreed execution plan (the Human's 10 steps, post-compaction)

1. **Doc pass** — the entire ledger above into DESIGN.md (+ README
   where user-facing: SYMBOL, as-of join when built); update issue
   bodies: #65 (hybrid + corrected free-evidence claim), #57 (menu +
   option-f sketch), #77 (rulings), #58 (B), #75 (closed by _seq
   ruling — close it), #42 (scope). HANDOFF snapshot refresh.
2. **Everything on the fruit table** (order): IS NULL -> `_seq`
   projection -> DELETE-consumes (+oracle re-sweep) -> #80 top-k ->
   #81 pushdown -> SYMBOL rename -> #58(B) enforcement.
3. **Repo-wide code pass** (review + fixes).
4. **PAUSE for the Human.** Agenda to offer: FIRST/LAST naming, #62
   split approval, anything the reviews raised.
5. **Table-2 M5 research items** — Agent-proposed amendment (flagged
   to the Human, accepted-by-silence at the pause): scope = **#42
   ALP+RD(+integer sibling) and F4 cross-process readers only**;
   incremental multi-factor belongs inside M5.4 (step 8), and #62
   only if the pause rules it in.
6. Code pass / doc pass. 7. **Merge** (Human).
8. **M5.0-M5.4** per the recorded table: M5.0 streaming primitives +
   prelude; M5.1 LAG/LEAD + RANGE; M5.2 as-of join as ruled; M5.3
   bucketing (F1(d)) + cross-sectional partitioning; M5.4 matrix tier
   on faer (SQL = scalar reductions only).
9. Code pass / doc pass. 10. **Merge** (Human).
Post-plan remainder of M5 (NOT in these steps): M5.5 bulk Arrow
ingest + Python binding (+ boolean refusal at that boundary), F3
segment-lazy open, M5.7 benchmark suite (#52).

### Standing session facts

- The scratchpad probe crate (path-dep throwaway tests) lives at
  /tmp/.../scratchpad/probe — rebuild it if needed, never in-repo.
- sqlparser 0.62.0 is current; GenericDialect in use.
- The M4-close reviews found the ASOF-comment silent-corruption bug
  class in hand-rolled pre-parse text handling: any new pre-parse
  lift (ASOF-join token, etc.) MUST splice by byte span, skip
  comments whole, and carry adversarial tests.
- Beta shape confirmed with the Human: after M5, feed-writer process
  + console/Python readers via F4 is the release story; no server,
  no browser, app owns the socket.
