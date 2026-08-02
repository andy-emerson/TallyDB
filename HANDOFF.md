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

## Snapshot (2026-07-30: M5.0–M5.4 BUILT; reviews then merge)

**State:** `main` = `8103306`. `claude/dev` rebased onto it, 16
commits, full gate green at every push. M5.0 through M5.4 are built —
the whole ordered-axis dividend plus two gaps found on the way:

| Commit | What |
|---|---|
| `…` M5.0/M5.1 | dispersion windows, prelude, `LAG`/`LEAD`, `RANGE` frames |
| as-of join ×2 | grammar + executor (#65) |
| bucketing | `GROUP BY ts / 60`, `FIRST`/`LAST` (F1 = d) |
| cross-sectional | `PARTITION BY ts` and buckets of it |
| #94 | scalar expressions over window results |
| PARTITION ×N | several partition terms intersect |
| streaming | bucketed grouping holds the open bucket (measured 1.65×) |
| #95 | predicates compare expressions, not only column-vs-literal |
| `regr_r2` | the one M5.4 reduction with a standard name |

**Evidence:** 143 → **146 differential families** vs DuckDB, 29 test
suites, six oracles. Every guard added was shown to trip; two
sabotages this stretch passed *silently* because the pattern matched
nothing, which is now a standing lesson — a sabotage that does not
fail is not a verified guard, it is an unverified edit.

**Issues opened:** #92 (as-of streaming co-walk — clause 2 of the join
constraint is designed, not built), #93 (**open decision for the
Human**: `quotes.ts` beside `trades.ts` is refused; rename today),
#94 (built, closed), #95 (built, closed).

**M5.4 is done on TallyDB's side, and its centre of gravity moved.**
`regr_r2` is ISO-named and shipped. Residual, fitted value and
multi-factor have no standard SQL name, so #77.1 puts them
script-side — and the Human ruled (2026-07-30) that they go to
**MatLua** (github.com/andy-emerson/MatLua), a Lua array + linalg
crate over faer and arrow-rs. SQL keeps only standard names; #77 was
**not** amended. A requirements letter for the MatLua team is drafted
at `scratchpad/matlua-requirements.md` — send it as an issue there.

**Standing ruling from the Human (2026-07-30), important:** decisions
made ad-hoc while building internal tools are *revisitable*; only
decisions that undermine **what TallyDB is** are non-negotiable. Do
not defend a choice by citing that it was ruled — give the reason, or
concede. (Applied: D2's NaN half is ours and revisitable; D2's NULL
half is SQL's and therefore identity-level.)

**Toolchain:** Rust 1.97.1, matching CI.

### The plan (amended by the Human 2026-07-30, after step 1 landed)

The research steps are **done for now**: step 1 shipped, and the
Human deferred steps 3–6 ("enough research-grade stuff for today").
The order is now step 1 → merge → **M5.0–M5.4** → passes → merge.

1. ~~**Tables bigger than memory**~~ — DONE. Ruled 2026-07-30
   (option b; metadata home = manifest section; (c) + budget
   semantics + default deferred to #87) and built the same day —
   see the snapshot above and DESIGN.md *The residency design*.
   Proposed as PR #89; the Human merges.
2. **M5.0–M5.4** ← NEXT, once #89 merges and `claude/dev` restarts
   from the new `main`. Every design decision it needs is closed:
   - **M5.0 streaming primitives** (#77 tranche 2a): rolling
     var/stddev, expanding aggregates, lag/diff/shift, log/simple
     returns, EWMA — O(n) incremental on the re-anchored compensated
     discipline; NumPy oracle + bench row per op. **SQL surface per
     #77.1 = (a)**: only standard-named ops reach SQL (`var_pop`,
     `stddev_pop`); EWMA/`diff`/multi-factor stay script-side until
     individually named. The **prelude ships here, compiled into the
     binary with `.prelude` printing its source** (#77.3 = a).
   - **M5.1** `LAG`/`LEAD` + `RANGE` frames — standard names, so in
     by the #77.1 rule; DuckDB differential.
   - ~~**M5.2** the as-of join~~ — DONE (2026-07-30), as ruled (#65
     hybrid): `ASOF` lifted pre-parse by byte span, `ON` only,
     ordering keys are the time axis (explicit inequality validated,
     not obeyed), bare `ASOF JOIN` refused, no `TOLERANCE`. Evidence
     as ruled: seven differential families against a vanilla-SQL
     **definitional** oracle (a correlated subquery, not DuckDB's own
     `ASOF JOIN`). Three build notes for the Human, all in DESIGN.md
     *M5 ruling batch* item 2: it is **not** the ordered co-walk the
     ruling described — a per-key sorted index plus binary search,
     which needs no `is_ordered()` gate and is correct over late
     arrivals, but materializes the dimension (**#92**: clause 2 of
     the join constraint stays designed, not built); ties on the
     dimension's clock go to the **last row in storage order**; and
     the inequality's sides are assigned by qualifier, not operator,
     so a backwards comparison is refused. **Open decision #93** (not
     gating): both tables are timestamped, and a dimension attribute
     sharing a fact column's name is refused, so `quotes.ts` beside
     `trades.ts` must be renamed today.
   - ~~**M5.3** bucketing + `FIRST`/`LAST` + cross-sectional~~ — DONE
     (2026-07-30). Bucketed `GROUP BY` (`ts / 60`, `(ts / 60) * 60`,
     bare `ts`; `//` accepted; nameable by SELECT alias);
     `FIRST`/`LAST` positional on the time axis; cross-sectional
     `PARTITION BY` including multi-term intersection; and **#94**,
     scalar expressions over window results, without which the
     cross-sectional weight could be computed but not used.
     The streaming dividend is built and **measured** — accumulator
     state is the open bucket, 1.65× less than hashing over 160k
     groups — with unordered data falling back to hashing rather than
     refusing (`compact()` restores it). Two new decisions recorded in
     DESIGN.md's F1 build note: `/` truncates between integers (ISO;
     constrains #40) and the `GROUP BY`/`PARTITION BY` type asymmetry.
     Opened on the way: **#95** (`WHERE x > y` — comparisons between
     expressions, refused since `WHERE` landed).
   - ~~**M5.4** the matrix tier~~ — DONE on TallyDB's side
     (2026-07-30). `regr_r2(y, x)` shipped: ISO-named, two-variable,
     so #77.1 admits it without coining anything; NULL where the fit
     is undefined; both the recompute and incremental paths, each
     guard shown to trip. Residual, fitted value and multi-factor
     have **no** standard SQL name, so they went script-side per
     #77.1 — and by the Human's ruling they go to **MatLua** rather
     than being built here. Nothing further is TallyDB's until
     MatLua answers the requirements letter.
3. **Code pass** (repo-wide review + fixes) ← HERE.
4. **Doc pass** (truth then clarity; README claims at evidence).
5. **Merge** (the Human).

**Deferred, not dropped** (the Human's research steps 3–6, to be
rescheduled): **continuous queries (#83)** — opens with the Plan
conversation the issue requires (which query shapes are maintainable;
what a maintained result does when its input is corrected), never
with code. **Incremental multi-factor** — K > 2 rolling OLS by
sliding-window factorization up/downdating, with per-frame faer
recompute as the shipped-correct safety net; it is M5.4's deep end,
so M5.4 lands without it and gains it later.

### Rulings landed since the M5 ruling batch

- **FIRST/LAST = (a)**: `FIRST(x)`/`LAST(x)`, the de-facto TSDB
  names; well-defined here because the ordering key is declared.
- **#62 split approved**: #62 = ingest hooks (engineering, decisions
  still open); #83 = continuous queries (research, Plan conversation
  required). Neither scheduled by that split — #83 is now step 3.
- **`_seq` stays `_seq`** — with an open revisit (the Human learned
  the "never seen" claim was really "never seen *unbidden*"; users
  do type it).
- **Delete-flush cost = (a)** accept and document (done): persistent
  DELETE seals the buffer; reopen trigger recorded in DESIGN.md.

### Remaining open decisions (none gate steps 1–7 except as noted)

- Step 1's residency design and step 3's #83 scoping — the two
  conversations the plan itself schedules.
- **#82** compaction 2× peak: document vs engineer (deferred).
- **#57** regex menu (deferred by ruling; option-f sketch on issue).
- **#46** agentic touchpoints; **#52** benchmark suite (M5.7).
- Boolean + logical-annotation revisit (parked, flagged
  wrong-on-purpose).
- #62 ingest hooks: unscheduled.

### Standing session facts

- Gate = fmt · clippy both legs `-D warnings` · test both legs ·
  rustdoc both legs · six oracle scripts (pyarrow_roundtrip via
  `-p arrow-lite --features oracle-harness`; the other five via
  `-p engine --features oracle-harness`). 402 tests at the merge.
  Doc-only (.md) pushes have gone without the full gate.
- CI runs on pull-request events only; CI stable moves — if clippy
  fails there and not here, `rustup update stable` and re-run.
- The scratchpad probe crate lives under the session scratchpad
  (path-deps on real crates); rebuild if needed, never in-repo.
- sqlparser 0.62.0, GenericDialect. Any new pre-parse lift must
  splice by byte span, skip comments whole, carry adversarial tests.
- Beta shape (built as of this merge): feed-writer process +
  read-only consoles (`--read-only`, `.refresh`/`.flush`) over one
  directory; readers see the durable prefix, old-or-new per
  mutation.
- `rm -rf crates/engine/tests/__pycache__` after running oracles
  (gitignored now, but keep the tree clean).

### Superseded snapshot (kept for the ledger below)

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

1. ~~**Doc pass**~~ — DONE (`dc0e16f`).
2. ~~**Everything on the fruit table**~~ — DONE, seven commits
   `d41acf8`..`255432f` plus the doc pass that follows them.
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
