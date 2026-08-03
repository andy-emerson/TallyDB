# TallyDB

**A small, embeddable, SQL-native database for numeric data — with numeric compute living inside the engine, not bolted on beside it.**

> **Status:** Under construction, and a first thin engine runs. The columnar
> foundation (`arrow-lite`) is implemented and cross-checked against arrow-rs
> and PyArrow; on top of it, a working vertical slice appends rows one at a
> time into persistent, crash-safe, multi-segment storage and serves a real
> SQL subset — `SELECT`/`WHERE`/`GROUP BY`/`ORDER BY`/`LIMIT`, small-table
> joins, window functions, and `UPDATE`/`DELETE` — with numeric compute
> (regression, covariance, PCA) exposed as SQL. Every query family is born
> cross-checked against DuckDB and NumPy in CI, over data that has
> round-tripped through storage. M2 (feature-complete) is merged: its
> final increment, M2.7, put embedded Lua on the engine's own zero-copy
> buffers — scripted window kernels, the curated ops callable from
> scripts, a NumPy-checked oracle family in CI. M3 (native GA) is
> merged: incremental window evaluation that beats both
> DuckDB+NumPy and NumPy-over-our-own-export in every measured shape
> under a compensated-truth CI guard, single-writer/concurrent-reader
> snapshots, a crash-tested WAL with sync levels, the ruled SQL
> IN-tier (`HAVING`, `DISTINCT`, scalar expressions, `CASE`, `LIKE`,
> DDL), and the `tallydb` console binary with per-platform release
> builds. M4 (extension model) is merged too: the native
> `WindowAggregate` trait as the primary extension path, Lua behind an
> off-by-default feature the console turns on, corrections on a
> knowledge axis (`AS OF` reads the table as it was known), and
> SQL-in-Lua driver scripts. M5 (desk adoption) landed the ordered-axis
> dividends — as-of joins, time bucketing, `RANGE` frames, `LAG`/`LEAD`,
> cross-sectional partitioning, `regr_r2` — and lazy residency, so a
> table need not fit in memory. On the working branch: **maintained
> views** (#83, first tranche) — bucketed aggregates kept fresh
> incrementally as data arrives, exact across corrections via the
> knowledge axis, with refresh cost proportional to what changed
> rather than to the table. The settled
> design and the reasoning behind it live in [`DESIGN.md`](DESIGN.md); open
> work and decisions live in the
> [issues and milestones](https://github.com/andy-emerson/TallyDB/issues).

TallyDB is an HTAP-shaped store: fast, append-heavy ingest (the
write-optimized half) feeding directly into ordered, columnar, analytical
reads (the read-optimized half), with no ETL step between them — and with
curated native and Lua compute that runs *on the engine's own buffers,
in-process, with no copy*. It is being built around three assumptions about
the data it stores:

1. **Append-optimized.** Data arrives as new rows, cheaply and one at a time
   — though corrections are supported (see below), just not the design
   center.
2. **Ordered.** Rows arrive roughly sorted on a declared **ordering key** (a
   timestamp is the common case, but any monotonically-increasing-on-ingest
   key works — a sequence number, an event id, a ledger offset). Storage is
   partitioned on that key.
3. **Numeric-or-key.** Every column is either a **number** (`f64` or `i64`,
   used in arithmetic, aggregation, windows) or a **key** (a
   dictionary-encoded identifier or label, used only for filtering,
   grouping, and joining — never arithmetic).

These three assumptions aren't restrictions bolted on after the fact —
they're the whole design. Relaxing any one of them is what makes
general-purpose databases (Postgres, DuckDB, SQLite) bigger, slower to
start, and harder to embed. Holding all three is what lets TallyDB stay
small, fast, and honest about what it's for — and, crucially, is what makes
fixed-width columns you can hand straight to a math library possible.

> **On "time-series."** Time-series, sensor telemetry, and tick data are the
> motivating **use cases**, not the definition. What's load-bearing is
> *ordered ingest on some key*, not that the key means "time." A monotonic
> sequence id serves the storage engine exactly as well as a nanosecond
> timestamp. So TallyDB is an **append-ordered numeric store**;
> "time-series database" is one hat it wears.

## What it's for — and what it isn't

**For:** workloads that are a big, append-heavy ledger of numbers with some
labels attached, analyzed with SQL — rolling aggregates, joins against small
reference tables, grouping, window functions, and numeric compute
(regression, covariance/PCA, portfolio math) run *in the database*.
Quantitative research, sensor and telemetry pipelines, event/metric streams,
financial ledgers: anything whose shape matches the three assumptions above.

**Not for:** general-purpose relational work. There will be no arbitrary
text columns or blobs, no third column type, and no joins beyond the two
shapes the engine can execute without a cost-based optimizer: equi-joins
where one side is small enough to materialize (the star-schema family —
lookups, dimensions, reference tables), and as-of joins (`ASOF LEFT
JOIN` / `ASOF INNER JOIN`, matched on the tables' declared ordering
keys; relatives planned).
Two large tables joined on an arbitrary key is refused loudly, not
served slowly. If your data doesn't fit the three
assumptions, use Postgres, DuckDB, or SQLite — they're better at being
general. TallyDB is a **specialized component** you reach for alongside a
general store, the way SQLite often is — not the one database that runs your
whole org.

## The SQL surface

TallyDB's SQL surface is designed to be standard SQL over its schema:
`SELECT`/`WHERE`/`GROUP BY`/`ORDER BY`, equi-joins, window functions, and —
yes — `UPDATE`/`DELETE`. Grouping and windowing both run in either
direction over the ordered axis: down one symbol through time
(`PARTITION BY sym`), or across every symbol at one instant
(`PARTITION BY ts`, or a bucket of it — `ts / 60` for minute bars).
Under the hood, both mutations are implemented as
tombstone-plus-reinsert against immutable, append-only storage (the same
mechanism handles ordinary corrections), resolved at the next compaction
rather than in place. They aren't the fast path, and the engine isn't
optimized for frequent use of them — but they're real, correct, and
available, because withholding a SQL verb the storage engine already
supports under a different name would just push the same work into
application code.

**Null and NaN, precisely.** NULL is absence, not a value: it matches no
comparison (three-valued logic), aggregates skip it, and `ORDER BY`
places it after all values in *both* directions. `IS NULL` /
`IS NOT NULL` is the one test that always has an answer — it asks about
presence, so it is never UNKNOWN, which is exactly why SQL needs it.
NaN is a value —
computed, greater than every number, equal to itself — under one
comparison relation shared by sorting, `WHERE`, `MIN`/`MAX`, and
zone-map pruning (see `DESIGN.md`, *Null, NaN, and ordering semantics*).

**Strings, precisely.** The numeric-or-key rule holds across the *entire
pipeline* — stored columns, intermediate results, and query outputs are
always numeric or key; a bare string never exists in the engine. That is
more permissive than it sounds:

- **String *predicates* on key columns are in scope.** `WHERE symbol =
  '...'`, `WHERE symbol IN (...)`, and `WHERE name LIKE '%Bank%'` are
  built today; regex matching is in scope but deferred by ruling
  (2026-07-29) with its design menu open on issue #57 — the engine
  rejects it loudly until then. All
  of them consume the interned strings and emit a *row selection*, not a
  string. Because keys are dictionary-encoded, such a predicate is
  evaluated once per *distinct* value and applied as integer
  set-membership: string filtering is not just allowed, it's cheap.
- **Symbols are labels, not ordered text.** `ORDER BY` on a symbol
  column is refused: the stored codes are per-segment interning ranks,
  so they rank nothing, and ranking the labels as text would ask an
  engine that never produces a string to order strings. Group by them,
  filter by them, order by a number. (The column type is declared
  `SYMBOL`, the spelling kdb+ and QuestDB use.)
- **String *production* is out.** No function may *emit* a string value: no
  `SUBSTRING`/`CONCAT` projection, no `CAST(x AS VARCHAR)`, no
  `GROUP_CONCAT`. A key comes back as its integer code plus the dictionary
  needed to render it; turning that into display text happens in your
  application.

More generally: **any standard SQL function or verb is in scope as long as
it (a) doesn't require a non-numeric, non-key column type and (b) doesn't
require a general-purpose cost-based optimizer.** We don't require ourselves
to imagine a specific use case before including something — real usage
regularly surprises the people who built the tool. The invariants are the
boundary, not our own foresight.

## How you'll use it

- Link it into your application like SQLite or DuckDB — no server process,
  no separate database to administer. One process writes; any number of
  read-only processes (`tallydb DIR --read-only`, or `Table::open_read_only`)
  watch the same directory and see its durable state, refreshed when they
  choose — a feed writer plus analyst consoles, with no server between
  them. (A standalone single-file CLI
  binary per release — the `sqlite3`-shell shape, still no server — ships
  with M3: `tallydb <dir>` opens a console with line editing, `CREATE
  TABLE`/`INSERT`/CSV import, the full query surface, `.lua` kernel
  registration, and `.run FILE` to execute a driver script; see
  `DESIGN.md`, *Deployment shapes*.)
- Query results come back in an Arrow-compatible columnar layout, directly
  usable by NumPy or other Arrow-aware tooling — no conversion step.
- For anything the built-in SQL functions don't cover, drop into embedded
  Lua — called directly from SQL, operating on the same numeric buffers the
  query engine already has in memory. Nothing is copied out to a separate
  scripting process or serialized across a boundary; the script, the query
  engine, and the curated native ops all read and write the same
  buffers in place. This **compute-without-copying** property is the thing
  TallyDB is actually built around, not a bolted-on extra.
- Runs natively (Linux/Mac/Windows) for production and research pipelines.
  (A WASM build is a planned future direction — see `DESIGN.md`.)

## Why it exists — and what's actually novel

None of the individual ingredients is new. Append-optimized columnar
storage, dictionary-encoded keys, in-database compute — each exists
somewhere. The differentiator is the *combination and packaging*: numeric
compute (regression, covariance, factor math) running inside an
**embeddable, SQL-native** engine, over **off-the-shelf** numeric libraries
(canonical Lua, pure-Rust linear algebra) on **zero-copy shared buffers** — rather than a
bespoke array language (kdb+'s q) or a serialization hop (DuckDB ↔ Python).
The honest one-line framing is *"an open, SQL-native, embeddable kdb+ for
teams below kdb+ scale"*: the workload kdb+ proved over 25 years, minus the
q language, minus the license, minus the server.

**Prior art.** TallyDB borrows validated ideas rather than inventing them:

- **InfluxDB** validates the key/numeric split directly — its tags-vs-fields
  model is close in spirit to TallyDB's numeric-or-key rule, and its more
  recent move to real SQL validates SQL-native as the right surface. (Note
  that InfluxDB is actually *more permissive* — it allows string and boolean
  *fields*; TallyDB deliberately takes a strict subset, which is where the
  footprint and performance wins come from.) InfluxDB itself isn't minimal
  or embeddable (it's a distributed server on Arrow/DataFusion/Parquet),
  which is the gap TallyDB fills.
- **kdb+** validates both the workload (25+ years as the quant-finance
  standard) and the "keys as interned integers, keep everything else
  numeric" performance pattern — *and* the idea of compute living inside the
  database. But it's proprietary, licensed, and built around q rather than
  SQL. TallyDB replicates the shape, not the language or the licensing.

## Where things stand

`arrow-lite` is implemented: the shared bitmap, 64-byte-aligned `f64`/`i64`
buffers, `u32`-dictionary key columns, the two-variant column enum with
zero-copy views, logical-type export annotations, and the C Data Interface
including `ArrowArrayStream` — every piece round-trip-tested against
arrow-rs and PyArrow in CI, with the unsafe core additionally checked
under Miri in CI on every change (issue #63) — `arrow-lite` concentrates
the workspace's unsafe, so that is where the interpreter is pointed.

On top of it runs the vertical slice, now past its M1 write-then-read
shape: `storage-lite` appends validated rows into a per-table store —
a write buffer freezing into immutable segments at a row threshold,
each row carrying an internal monotonic row id — and persists them
behind a storage-backend trait (natively, a directory of files) in a
self-describing, CRC-checked, deterministic on-disk format whose bytes
are locked by a committed golden: per-column codec tags with
delta-of-delta on the ordered ordering key (measured on the checked-in
corpus: 2–2.5× vs raw, ahead of plain delta on both corpus families),
ALP for `f64` values — decimals-in-doubles become bit-packed integers,
verified bit-exact per value at encode time, with the ALP-RD and raw
fallbacks so the codec never bloats (penny-priced ticks measure 4.2×
vs raw; continuous sensor reals 1.16×, near the lossless ceiling for
that shape) — and frame-of-reference + bit-packing on symbol codes
(6–10×) and non-clock integers,
zone maps (driving query-time pruning), and reopen that verifies schema,
checksums, and row-id contiguity. Durability is a sidecar write-ahead
log with sync levels (default: group commit every 100ms, measured at
~1µs added per append; `Full` for a zero loss window; `Off` restoring
the flush boundary for replayable upstreams), crash-tested down to
torn-record and stale-generation windows.
A table need not fit in memory: segments fault in lazily under a byte
budget (`tallydb --cache MiB`), zone pruning runs on segment-handle
metadata *before* the fault, so a pruned segment's file is never read,
and a table bigger than its budget answers exactly like a resident one.
Mutation is real: `UPDATE`/`DELETE` run as tombstone + reinsert against
row-id delete logs, each mutation spending exactly one coordinate on the
ingest-sequence axis — so `AS OF` the last spent one is the latest
state, permanently, and `_seq` reads a row's own coordinate back
through SQL. Reads resolve tombstones through live masks, and
crash-safe generational compaction merges live rows back into sorted,
contiguous segments — with end-state semantics validated against DuckDB
in CI. `query-lite` speaks a real query subset via sqlparser-rs: SELECT with
WHERE (the predicate fragment — numeric comparisons, key string
equality, `IN` and `LIKE` evaluated once per distinct dictionary value,
`IS [NOT] NULL`, `AND`/`OR`/`NOT` — with zone-map pruning skipping
segments that cannot match, plus comparisons between whole expressions
such as `x * 2 > y + 1`, which no zone map can rule out: those prune
nothing, and pruning degrades **per conjunct**, so `WHERE ts > 1000 AND
x > y` still skips segments on `ts`), GROUP BY over key columns and
over monotone buckets of the ordering key (`ts / 60`, `(ts / 60) * 60`,
bare `ts`) with `COUNT`/`SUM`/`AVG`/`MIN`/`MAX`/`FIRST`/`LAST` under
SQL null semantics (`SUM` over `i64` stays exact and errors loudly on
overflow rather than silently widening; a bucket grouping streams,
holding only the open bucket's state rather than a hash table over
every group), top-level ORDER BY and LIMIT/OFFSET (bounded, the sort
runs top-k: a ten-row answer holds ten rows, not the whole sorted
result), equi-joins (one large table
against small key-unique dimension tables — the star-schema family —
INNER or LEFT, run fact-driven through the same pipeline as everything
else, gathering only the dimension columns the query reads), as-of
joins (`ASOF LEFT|INNER JOIN ... ON f.sym = q.sym AND f.ts >= q.ts` —
each fact row against the last dimension row at or before it), the
standard aggregates as window functions over `ROWS` and `RANGE` frames
and over the whole partition, `LAG`/`LEAD`, cross-sectional
`PARTITION BY` (one or several terms, including an unordered partition
that ranks peers within a timestamp), scalar expressions over window
results, and `UPDATE`/`DELETE`. It
executes across all segments of a snapshot, returning one Arrow batch
per segment for a plain scan — the collapsing stages (`ORDER BY`,
`LIMIT`, `DISTINCT`, `HAVING`, `GROUP BY`) materialize a single batch
instead — with per-segment key dictionaries remapped at query time
where grouping or partitioning needs them, and a generated
differential harness diffs query families against DuckDB over the
corpus in CI;
`engine` ties them together behind a
multi-table `Database` handle — which also carries **maintained
views** (#83): a bucketed aggregate materialized as a real table plus
a stamp (the source watermark it reflects), refreshed by re-folding
only the buckets the knowledge history says changed, and read through
a union that tops the materialization up with a live fold of whatever
the stamp does not cover, so a view answers exactly however stale its
materialization (a seventh oracle family diffs it against DuckDB
recompute after every scripted step, in CI) — registering `regr_slope` /
`regr_intercept` / `regr_r2`, `covar_pop` / `corr`, `var_pop` /
`stddev_pop`, and `eigen_max` (the window's first
principal-component variance) as SQL window functions — every window
re-derived independently by NumPy and DuckDB in CI, over a fixture that
spans several segments and a storage round trip. **These are solved in
closed form, and the engine links no LAPACK at all**: a two-parameter
regression and a 2 × 2 eigenvalue have exact solutions, while a general
solver is dominated by its own per-call overhead at window scale
(measured: ~2.3µs of `regr_slope`'s 2.5µs per 64-row window). Removing it
moved `regr_slope` from 5× behind DuckDB's equivalent window to 3.3×
ahead, dropped the system-LAPACK build dependency, and took a
LAPACK-in-WASM layer off the WASM build's critical path. A LAPACK-class
backend returns only when an op needs more than two parameters or
dimensions, where no closed form exists — see `DESIGN.md`, *Curated
compute: what the engine calls, and why*. Passthrough results share the stored buffers
(pointer-verified); the design-matrix and cross-segment window gathers
are copies proportional to the rows they cover — not bounded by a
constant, as the crate docs record. `compute-linalg`
provides the multiplication-class kernels behind the same
capability-negotiating trait shape (`dot`, matrix–vector, matrix–matrix
— checked against hand computations; not yet called from query inner
loops, which stays profiling-gated), and it is pure Rust: `dot` is a
source-fixed loop, bit-identical on every CPU and target and measured
fastest at window scale, while the matrix products use faer, measured
3.7–10× ahead of a naive loop (and ahead of reference BLAS) at the Gram
shapes a future multi-parameter op would need. The engine links no
system math library at all — no BLAS, no LAPACK — and the compute stack
compiles for wasm32 as-is. `compute-lua` embeds canonical PUC Lua 5.4 (vendored,
unmodified) behind the frozen value-map contract: nullable columns
cross as zero-copy views (NULL is the `NULL` sentinel, three-valued
through arithmetic; keys read as codes with `text()`/`code_of()`),
results coerce exact-or-loud to a type declared at registration, and
application kernels run as SQL window functions
(`Table::register_lua_window`) with the curated native ops
callable from scripts over the same buffers — every kernel family
re-derived by NumPy in CI over a multi-segment storage round trip, the
C boundary additionally run under `LUA_USE_APICHECK` and ASan/UBSan in
CI, and `log()` routing script diagnostics to an embedder-installed
sink (`print` is gone; stdout is not an embedded library's to own). One
honest set of numbers to hold beside the design: the latency benchmark
(`m2_compute_latency_bench.py`, run 2026-07-27, container hardware, 20k
rows, window 64) now measures **two peers** — vectorized NumPy riding
TallyDB's own ~0.1ms Arrow export (the *marginal* question: given data
in TallyDB, is in-engine compute worth it?) and the competitor stack
entire, the same rows stored in DuckDB with NumPy pulling from DuckDB
(the *product* question: TallyDB, or DuckDB + NumPy?). The curated
statistics now evaluate **incrementally** — running moments about a
data-anchored shift, re-anchored every window-length so rounding cannot
accumulate, through a frame-sequence seam every window function runs
through (per-frame recompute remains the default for everything else).
With that landed, in-engine compute wins every curated statistic in
every measured shape: `regr_slope` by **9.6×** against the competitor
stack, the pair statistics (`covar_pop`, `corr`, `eigen_max`) by
**3–4×** against the competitor stack and **1.2–1.6×** against
vectorized NumPy even when NumPy rides TallyDB's own ~free export, and
the live-query shape — the newest window, now — by **6–9×**, because
pulling even 64 rows out of DuckDB costs ~1ms before any math happens,
while the append-ordered engine serves the whole query in ~120µs. It is
also the only arrangement in the comparison holding 1e-12-to-truth
accuracy at timestamp-scale offsets: the vectorized peer's fast idiom
(rolling cumsum moments) is the catastrophic-cancellation form the
engine rejected for correctness, so the peer buys its speed with wrong
answers exactly where the ordering key lives. That accuracy contract is
enforced in CI on every change by a compensated-reference guard over
adversarial corpora, covering both the per-window and incremental
paths. The Lua story split into tiers and earned its keep: an
*element loop* written in Lua remains interpreter-bound (~14× behind
vectorized NumPy in bulk — the documented code smell), but a kernel
written in the **vectorized vocabulary** — whole-column operators and
rolling combinators running native under one interpreter entry, the
NumPy model — measures **~2–2.5× ahead of the DuckDB+NumPy stack**
(m2_compute_latency_bench, run 2026-07-28, 20k rows: composed
`(a-b)/b` 0.9ms and `rolling_dot(64)` 1.1ms vs 1.8–2.4ms), while the
newest-window latency shape reaches parity with NumPy over the
engine's own export. The vendored interpreter is byte-identical to
upstream PUC Lua v5.4.7 (verified file-by-file; the full official
suite and `ltests` torture harness run in CI). The four statistics
above are native implementations; the promotion ladder (prototype a
kernel, then graduate it to native) is the intended path for future
statistics, not how these four arrived. The extension model is ruled
(2026-07-28): the Rust `WindowAggregate` trait is the primary
extension path, Python composes from outside over the same Arrow
buffers (never embedded), and Lua is an opt-in feature owned by the
console — its sunset clause stood trial on that evidence and
**dissolved (ruled 2026-07-28: Lua stays)**. With the verdict in,
the embed became bidirectional: **SQL-in-Lua** landed — a driver
script runs `query(sql)` and receives result columns as the same
zero-copy views kernels consume, computes with the vectorized
vocabulary, and feeds derived rows back exactly through
`append(table, row)` (`Database::run_script`; `.run FILE` at the
console) — with the end-to-end SQL → Lua → SQL pipeline re-derived
by NumPy in CI. The zero-copy property itself is pointer-verified
and stands.
`lua.wasm` (the one WASM compute
dependency still to come, for later) is a real, working, MIT-licensed
project already in progress by the same author — tracked as a future
dependency, not part of the current native-first build. (Its sibling
`blas.wasm` is no longer needed: with LAPACK removed and system BLAS
replaced by pure Rust, TallyDB's linear algebra compiles for wasm32
directly.)

## How we work

This repository follows the working agreement in [`AGENTS.md`](AGENTS.md)
(v2.4.0) and the craft conventions in
[`CONTRIBUTING.md`](CONTRIBUTING.md), both copied unedited from
[working-agreement](https://github.com/andy-emerson/working-agreement).
The repo-specific half lives here:

- **Durable documents:** four, each answering one question.
  [`AGENTS.md`](AGENTS.md) — *how we work*: phases, claims, reviews,
  merges. [`DESIGN.md`](DESIGN.md) — *what we build and why*: philosophy,
  invariants, crate boundaries, settled decisions, build order, and the
  test plan's skeleton. [`CONTRIBUTING.md`](CONTRIBUTING.md) — *how it is
  written*: the conventions of the craft, including the commit shape.
  This README — *where it is now*, for a user. The first and third are
  project-agnostic and arrive by replacement; the other two are ours.
- **Living status:** [GitHub Issues](https://github.com/andy-emerson/TallyDB/issues).
  Open decisions carry the `decision` label; everything else open is a todo
  or a bug. Settled decisions — including rejected alternatives and their
  reopen triggers — are recorded in the durable documents, not kept as open
  issues.
- **Roadmap:** [GitHub Milestones](https://github.com/andy-emerson/TallyDB/milestones)
  — M0 layout locked · M1 compute proven · M2 feature-complete · M3 native
  GA · M4 extension model · M5 desk adoption · M6 WASM parity · M7 served
  product + workbench.
- **Checks:** GitHub Actions on every pull request and every push to
  `main` — fmt, clippy, build, tests including doctests, rustdoc with
  warnings as errors, the Python oracle suite (PyArrow round trip;
  DuckDB and NumPy differentials, the Lua-window family included), the
  Lua `apicheck` build, Miri over `arrow-lite`, the official Lua 5.4.7
  test suite (`ltests`) over the vendored interpreter, and an
  ASan/UBSan job over the C boundary. Doctests are this repository's
  preferred executable evidence.
- **Audience:** documentation is written for a reader with a BS in applied
  mathematics and a CS minor; code for the CS-minor side — see DESIGN.md,
  *Who we write for*. `CONTRIBUTING.md` leaves that fluency setting to the
  Human; DESIGN.md is where this repository sets it.
