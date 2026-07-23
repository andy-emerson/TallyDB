# TallyDB

**A small, embeddable, SQL-native database for numeric data — with numeric compute living inside the engine, not bolted on beside it.**

> **Status:** TallyDB is under construction, and a first thin engine runs.
> The columnar foundation (`arrow-lite`) is implemented and cross-checked
> against arrow-rs and PyArrow in CI, and a minimal vertical slice now
> works end to end: append rows one at a time — freely interleaved with
> queries, into multi-segment storage with internal row ids that
> persists to disk in a golden-locked columnar format and reopens
> verified — run a SQL rolling least-squares regression solved by
> LAPACK inside the engine, and read the results over an Arrow stream —
> validated row-for-row against NumPy and DuckDB in CI, over data that
> has round-tripped through storage — plus real `UPDATE`/`DELETE` via
> tombstone + reinsert, resolved by crash-safe compaction, with
> end-state semantics diffed against DuckDB in CI. It is still one
> SELECT shape, not a database yet: the general query surface (WHERE,
> GROUP BY, joins) and the Lua/BLAS surface are still ahead. The
> developer-facing design lives in
> [`DESIGN.md`](DESIGN.md); open work and decisions live in the
> repository's
> [issues and milestones](https://github.com/andy-emerson/TallyDB/issues).

TallyDB is an HTAP-shaped store: fast, append-heavy ingest (the
write-optimized half) feeding directly into ordered, columnar, analytical
reads (the read-optimized half), with no ETL step between them — and with
BLAS/LAPACK and Lua compute that runs *on the engine's own buffers,
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
text columns or blobs, no third column type, and no general multi-table
joins outside a star-schema shape. If your data doesn't fit the three
assumptions, use Postgres, DuckDB, or SQLite — they're better at being
general. TallyDB is a **specialized component** you reach for alongside a
general store, the way SQLite often is — not the one database that runs your
whole org.

## The SQL surface

TallyDB's SQL surface is designed to be standard SQL over its schema:
`SELECT`/`WHERE`/`GROUP BY`/`ORDER BY`, equi-joins, window functions, and —
yes — `UPDATE`/`DELETE`. Under the hood, both mutations are implemented as
tombstone-plus-reinsert against immutable, append-only storage (the same
mechanism handles ordinary corrections), resolved at the next compaction
rather than in place. They aren't the fast path, and the engine isn't
optimized for frequent use of them — but they're real, correct, and
available, because withholding a SQL verb the storage engine already
supports under a different name would just push the same work into
application code.

**Strings, precisely.** The numeric-or-key rule holds across the *entire
pipeline* — stored columns, intermediate results, and query outputs are
always numeric or key; a bare string never exists in the engine. That is
more permissive than it sounds:

- **String *predicates* on key columns are in scope.** `WHERE symbol IN
  (...)`, `WHERE name LIKE '%Bank%'`, regex matching — these consume the
  interned strings and emit a *row selection*, not a string. Because keys
  are dictionary-encoded, the predicate is evaluated once per *distinct*
  value and applied as integer set-membership: string filtering is not just
  allowed, it's cheap.
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
  no separate database to administer.
- Query results come back in an Arrow-compatible columnar layout, directly
  usable by NumPy or other Arrow-aware tooling — no conversion step.
- For anything the built-in SQL functions don't cover, drop into embedded
  Lua — called directly from SQL, operating on the same numeric buffers the
  query engine already has in memory. Nothing is copied out to a separate
  scripting process or serialized across a boundary; the script, the query
  engine, and the curated BLAS/LAPACK ops all read and write the same
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
(LuaJIT, BLAS/LAPACK) on **zero-copy shared buffers** — rather than a
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
arrow-rs and PyArrow in CI, with the unsafe core also run under Miri.

On top of it runs the vertical slice, now past its M1 write-then-read
shape: `storage-lite` appends validated rows into a per-table store —
a write buffer freezing into immutable segments at a row threshold,
each row carrying an internal monotonic row id — and persists them
behind a storage-backend trait (natively, a directory of files) in a
self-describing, CRC-checked, deterministic on-disk format whose bytes
are locked by a committed golden: per-column codec tags with
delta-of-delta on the ordered ordering key (measured on the checked-in
corpus: 2–2.5× vs raw, ahead of plain delta on both corpus families),
zone maps awaiting query-time pruning, and reopen that verifies schema,
checksums, and row-id contiguity (durability boundary is the flush).
Mutation is real: `UPDATE`/`DELETE` run as tombstone + reinsert against
row-id delete logs, reads resolve tombstones through live masks, and
crash-safe generational compaction merges live rows back into sorted,
contiguous segments — with end-state semantics validated against DuckDB
in CI. `query-lite` parses one SELECT shape — window aggregates over
`ROWS BETWEEN n PRECEDING AND CURRENT ROW`, optionally per key — plus
`UPDATE`/`DELETE` with a predicate fragment (numeric comparisons, key
string equality and `IN` evaluated once per distinct dictionary value,
`AND`/`OR`/`NOT`) via sqlparser-rs, and executes across all segments of
a snapshot, returning one Arrow batch per segment with per-segment key
dictionaries remapped at query time where partitioning needs them;
`compute-lapack` links system
LAPACK and solves least squares through `dgels` behind a
capability-negotiating trait; and `engine` ties them together behind a
multi-table `Database` handle, registering `regr_slope` /
`regr_intercept` as SQL window functions whose every window is
re-derived independently by `np.linalg.lstsq` and DuckDB in CI — over
a fixture that spans several segments, so cross-segment windows are in
the cross-check. Passthrough results share the stored buffers
(pointer-verified); the design-matrix and cross-segment window gathers
are the bounded copies, as recorded in the crate docs. `compute-blas` and
`compute-lua` remain scaffolds: documented boundaries and settled
contracts, not yet implementations. `blas.wasm` and `lua.wasm` (the WASM
compute dependencies, for later) are real, working, MIT-licensed projects
already in progress by the same author, with LAPACK-in-WASM as their next
planned milestone — tracked as future dependencies, not part of the current
native-first build.

## How we work

This repository follows the working agreement in [`AGENTS.md`](AGENTS.md)
([source](https://github.com/andy-emerson/working-agreement)). The
repo-specific half lives here:

- **Durable documents:** this README (the user-facing current state) and
  [`DESIGN.md`](DESIGN.md) — the design companion: philosophy, invariants,
  crate boundaries, settled decisions, build order, and the test plan's
  skeleton.
- **Living status:** [GitHub Issues](https://github.com/andy-emerson/TallyDB/issues).
  Open decisions carry the `decision` label; everything else open is a todo
  or a bug. Settled decisions — including rejected alternatives and their
  reopen triggers — are recorded in the durable documents, not kept as open
  issues.
- **Roadmap:** [GitHub Milestones](https://github.com/andy-emerson/TallyDB/milestones)
  — M0 layout locked · M1 compute proven · M2 feature-complete · M3 native
  GA · M4 WASM parity.
- **Checks:** GitHub Actions on every pull request — fmt, clippy, build,
  tests including doctests, rustdoc with warnings as errors. Doctests are
  this repository's preferred executable evidence.
- **Audience:** documentation is written for a reader with a BS in applied
  mathematics and a CS minor; code for the CS-minor side — see DESIGN.md,
  *Who we write for*.
