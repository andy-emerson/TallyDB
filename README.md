# TallyDB

**A small, embeddable, SQL-native database for numeric data — with numeric compute living inside the engine, not bolted on beside it.**

> **Status:** TallyDB is in its design phase. The architecture, invariants,
> and crate boundaries are specified, settled decisions are recorded, and the
> open ones are tracked — but no runnable engine exists yet. Everything below
> describes what TallyDB is being built to be; the design is real, the
> software is not yet. The developer-facing design lives in
> [`DESIGN.md`](DESIGN.md); open work and decisions live in the repository's
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

The seven workspace crates are scaffolded: documented boundaries and settled
contracts, not yet implementations. `blas.wasm` and `lua.wasm` (the WASM
compute dependencies, for later) are real, working, MIT-licensed projects
already in progress by the same author, with LAPACK-in-WASM as their next
planned milestone — tracked as future dependencies, not part of the current
native-first build.

- **[`DESIGN.md`](DESIGN.md)** — the forward-looking developer companion to
  this document: positioning, invariants, crate boundaries, settled
  decisions, build order.
- **[`AGENTS.md`](AGENTS.md)** — how work happens in this repository.
- **[Issues & milestones](https://github.com/andy-emerson/TallyDB/issues)**
  — all open work: todos, decisions, and the roadmap.
