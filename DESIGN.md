# TallyDB — Design

This is the **forward-looking developer companion** to `README.md`: what we
are building, why, and which parts are settled. The README describes where
the project is now from the user's point of view; this document describes
where it is going from the developer's. How we work — passes, reviews,
issues, integration — is `AGENTS.md`.

## What this is (positioning, so scope calls stay anchored)

An **append-ordered numeric store**: embeddable, SQL-native, with numeric
compute (Lua + BLAS/LAPACK) running *inside* the engine on its own buffers,
zero-copy. Time-series / sensor / quant are **use cases**, not the
definition — what's load-bearing is *ordered ingest on a key*, not that the
key means "time." The one-line frame: an open, SQL-native, embeddable kdb+
for teams below kdb+ scale. The differentiator is the packaging (embeddable
+ compute-fusion over off-the-shelf libs), not "it only holds numbers"
(that's table stakes for any TSDB) and not "compute inside the DB" (kdb+
already does that). Don't let scope drift toward looking like a general DB
or a general TSDB; the three assumptions are the moat.

## The three assumptions (do not relax these to unblock a feature)

1. **Append-optimized.** Writes are cheap, low-latency, one row at a time.
   The fast path is *append*, not in-place update — keep it that way.
2. **Ordered.** Data arrives roughly sorted on a declared **ordering key**
   (a timestamp is the common case, but any monotonic-on-ingest key works —
   a sequence id, an event id, a ledger offset). Storage is partitioned on
   that key. "Ordered" is load-bearing (it's what makes zone-map pruning and
   delta compression work); "time" is not — don't hardcode a timestamp where
   the declared ordering key belongs.
3. **Numeric-or-key.** Every column is numeric (`f64` or `i64`) or a
   dictionary-encoded key. No third type. Ever. This holds across the whole
   pipeline — stored columns, intermediates, and query results — not just
   storage. If a feature seems to need a third type, the feature is wrong,
   not the invariant.

These assumptions aren't restrictions bolted on after the fact — they're the
whole design. Relaxing any one of them is what makes general-purpose
databases bigger, slower to start, and harder to embed; holding all three is
what makes fixed-width columns you can hand straight to a math library
possible.

### Numbers: `f64` and `i64`, with roles

"Numeric" is not monolithically `f64`. An append-ordered store's most
important column is usually its *ordering key*, and epoch **nanoseconds
don't fit in `f64`** — `f64` has 53 bits of integer precision (exact
integers to ~9.0×10¹⁵), while epoch-nanos are already ~1.8×10¹⁸, so `f64`
timestamps silently cap at microsecond precision. So numeric columns come in
two flavors with distinct roles:

- **`i64` (and fixed-point decimal over `i64`) — the exact / stored / fact
  type.** Nanosecond timestamps, money as scaled integers, volumes, counts.
  Exact, bit-for-bit reproducible, and — bonus — ordered `i64` columns are
  exactly what delta / delta-of-delta compression is built for.
- **`f64` — the analytic / derived type.** Anything BLAS/LAPACK touches.
  Regression coefficients, covariance eigenvalues, correlations, portfolio
  weights are *irrational in general*, so the analytics layer is inherently
  floating-point; this is also what keeps NumPy interop and the DuckDB
  oracle strategy working.

The schema declares which flavor each numeric column is. We considered and
**rejected** making the numeric type a rational (`i64/i64`) and writing our
own integer linear algebra: rational denominators overflow `i64` within a
handful of divisions (a mean of ~4 returns already blows past the ceiling),
a bignum rational is variable-width and kills the fixed-width Arrow-interop
and SIMD story, and — decisively — rationals can't even *represent* the
irrational outputs (√, log, eigenvalues) the analytics produce.
Floating-point *done carefully* is the right tool; where reproducibility
matters, it is handled at the BLAS build level (non-FMA kernels — see
*Numerical consistency*), not by dropping floats.

**Decision record — `f32` (considered and set aside, kept cheap to add).**
A single-precision analytic subtype was rejected for now: 32 bits can never
hold the ordering key or money (`f32`'s exact-integer ceiling is 2²⁴;
`i32` nanoseconds span ±2.1 s), `f32` accumulation quietly loses
million-row sums and variance to cancellation, and the whole oracle
strategy (DuckDB, NumPy) speaks `f64`. What makes the rejection cheap:
the numeric subtype tag is an extensible integer registry, so adding `F32`
later is a new variant and buffer width — never a format migration. Reopen
triggers: a GPU/WebGPU compute backend actually lands on the roadmap (WGSL
has no `f64`, so there `f32` is the entry ticket), or profiling shows
bandwidth-bound, precision-tolerant workloads dominating real usage. The
adoption shape when triggered: per-op downconversion at the compute
boundary or an opt-in stored subtype — never for ordering keys or money.

### Strings: predicates yes, production no

The numeric-or-key rule holds across the *entire pipeline* — stored columns,
intermediate results, and query outputs are always numeric or key; a bare
string never exists in the engine. That is more permissive than it sounds:

- **String *predicates* on key columns are in scope.** `WHERE symbol IN
  (...)`, `WHERE name LIKE '%Bank%'`, regex matching — these consume the
  interned strings and emit a *row selection*, not a string, so they don't
  need a third type. Because keys are dictionary-encoded, the predicate is
  evaluated once per *distinct* value in the small dictionary and then
  applied as integer set-membership: string filtering is not just allowed,
  it's cheap.
- **String *production* is out.** No function may *emit* a string value: no
  `SUBSTRING`/`CONCAT` projection, no `CAST(x AS VARCHAR)`, no
  `GROUP_CONCAT`. A key result comes back as its integer code plus the
  dictionary needed to render it; formatting is the application's job.

### Keys assume repeating labels (low cardinality)

The dictionary is the one variable-width structure in the system, and it's
acceptable because it is *reference data, not row data*: sized by distinct
values, not rows, and never on the per-row scan/compute path. That holds
only while keys are repeating labels (symbols, sensor ids, exchange codes).
A key column fed never-repeating values (a UUID per row) degenerates — the
dictionary grows with row count and `u32` codes exhaust at ~4.3B distinct
values. A never-repeating identifier is a number: declare it `i64` numeric,
not key. (`engine` should eventually warn when distinct/rows approaches 1
on a large table.)

### What "numeric-or-key" means at the engine level

This isn't a naming convention — it's enforced in the type system. A column
is either:

- **Numeric** (`f64` or `i64`): usable in arithmetic, aggregation,
  comparison, and — for `f64` — passed directly into BLAS/LAPACK/Lua as raw
  numeric buffers.
- **Key**: dictionary-encoded to an integer at ingest (string interning,
  similar to kdb+'s symbol type or Arrow's dictionary encoding), usable in
  equality/grouping/joins and string *predicates*, never in arithmetic.

There is no third column type. A column that can't be classified as one or
the other is rejected at schema-definition time, not silently coerced — and
this holds for query results and intermediates, not just stored columns.

The vocabulary is final (issue #7, decided 2026-07-23): the two species are
**numeric** and **key**, chosen because the pair states the invariant and
"key" matches SQL's own usage on a SQL-native surface. A key is a *label*,
not a primary key — repeating values are the point, not a violation. For
readers arriving from the BI/OLAP world: key columns play the *dimension*
role in a star schema, numeric columns are the *measures*; the
Kimball vocabulary was considered and set aside because "dimension" and
"measure" collide with this document's mathematical audience.

## The inclusion principle for the SQL surface

Include a standard SQL function or verb if it (a) doesn't require a
non-numeric, non-key column type, and (b) doesn't require a general-purpose
cost-based optimizer. **"We can't think of a quant use case for it" is
explicitly NOT a valid reason to exclude something otherwise in scope** —
real usage regularly surprises the people who built the tool. The
invariants are the boundary, not our own imagination.

## Null, NaN, and ordering semantics

> **Decided (2026-07-24): NULL is placed, not ordered; NaN is a value,
> greater than every number, everywhere.** The engine's three-valued
> predicate logic already put NULL outside the number line — a null
> matches neither `x > 5` nor `x <= 5`, aggregates skip it, arithmetic
> propagates it — and ordering says the same thing: nulls are not
> compared but *placed*, after all values, in both sort directions.
> Consequently `ORDER BY x DESC` is not the sequence-reversal of `ASC`:
> within the values it is an exact mirror (total order guarantees it);
> only the non-values stay put. That asymmetry is sound here because of
> two premises, which are its reopen tripwire: the executor never
> serves `DESC` by reversing an `ASC` result (each query sorts by its
> own comparator, and there is no optimizer to introduce the shortcut),
> and the one physically-ordered column — the ordering key — is `NOT
> NULL` by schema rule. NaN, by contrast, *is* a value: computed, and
> comparable under one relation used by sort, predicates, MIN/MAX, and
> zone-map pruning alike — NaN is greater than every number and equal
> to itself, while `-0.0 = 0.0` stays true (NaN lifted to the top, not
> bitwise total order). The ascending ladder is *numbers… +∞, NaN,
> then NULL off the end*. Pruning stays sound via a has-NaN bit in the
> `f64` zone map (see `format.rs`). Rejected: nulls-as-largest/smallest
> (they put absence *on* the number line for sorting while predicates
> keep it off — one seam, two answers), and IEEE-strict predicates
> (NaN invisible to every operator but `<>` while sorting as a value —
> the trap this ruling closed). `NULLS FIRST`/`LAST` syntax is an
> additive todo. The choice was made from the numeric-or-key thesis,
> not oracle convenience: where the SQL standard leaves semantics
> implementation-defined, the choice is ours and the differential
> harness normalizes.

## Storage, ordering, corrections, and UPDATE/DELETE

Storage is columnar, partitioned on the declared ordering key, and immutable
once flushed — segments are never rewritten in place. Zone maps (min/max per
column per segment) exploit ordered ingest to prune segments at query time;
delta/delta-of-delta compression exploits it to shrink ordered numeric
columns. **This is why "ordered" is load-bearing and "time" is not:**
without *some* clustering key the data arrives roughly sorted on, both
pruning and compression collapse — you keep columnar *layout* but lose
columnar-*fast-at-scale*.

All mutation — an out-of-order correction, a SQL `UPDATE`, a SQL `DELETE` —
goes through one mechanism: **tombstone + reinsert.** The old row is marked
deleted, a corrected row is appended fresh if there is one, and background
compaction resolves tombstones and merges segments. This means:

- No MVCC, no row versioning, no general in-place update engine — one
  mutation primitive, reused everywhere.
- Optimized for the common case (in-order append), correct-but-unoptimized
  for the rest (corrections, `UPDATE`, `DELETE`) — all fully supported, not
  excluded just because they aren't the fast path.
- Query-time reads resolve "newest version wins" for any tombstoned row — a
  small, well-understood cost, paid only when mutation has actually
  happened.

> **Decided (issues #1 and #6, 2026-07-23), settling `storage-lite`'s
> formats:** (1) **Row identity is kdb+-style pure append** — rows carry an
> internal monotonic row id, duplicates are first-class, `UPDATE`/`DELETE`
> address rows by predicate, and corrections supersede by ingest sequence.
> The rejected InfluxDB-style `(key-set, ordering-key)` primary key
> silently collapses distinct same-tuple events — data loss with no error;
> if user-visible overwrite semantics are ever needed, the path is an
> opt-in declared uniqueness constraint on top, not a reversal. (2) **Key
> dictionaries are per-segment** — segments are fully self-contained, which
> keeps immutability pure, compaction simple, and matches Arrow's per-batch
> dictionary export; with identity resolved by row id, compaction never
> compares key values across segments, which is what made a global
> dictionary attractive. The recorded extension: a process-lifetime
> code-remap cache at query time, added only when profiling shows the
> remap cost is material. (3) **The format carries a per-column codec
> tag** (issue #28) — a one-byte, append-only integer registry
> (`0 = uncompressed`), same pattern as the frozen type-tag registries —
> so every codec is an additive entry, never a format migration.
> **Ordered `i64` columns use delta-of-delta** (issue #29), the TSDB
> standard for clock-like keys, with a confirm-against-plain-delta
> measurement on the corpus at implementation. **`f64` columns ship
> uncompressed behind the tag** — a legitimate answer for hot data, not
> a placeholder. **The general-`f64` codec is decided: ALP** (issue
> #30, closed 2026-07-24 by argument over the published evidence
> rather than an in-house A/B — sound precisely because the codec
> registry makes the choice an additive tag, cheap to reverse). ALP
> converts decimals-in-doubles to integers per vector
> (frame-of-reference + bit-packing, verbatim exceptions, ALP-RD
> fallback for true doubles), with losslessness enforced per value at
> encode time; it leads the field on both of our weighted criteria —
> decode throughput on the read path first, ratio second (encode runs
> at freeze/compaction, off the hot path). Rejected: Gorilla and Chimp
> (the XOR family's bit-serial decode cannot vectorize; Chimp remains
> the named low-effort fallback if ALP's implementation cost vetoes
> it), Elf (near-parity ratio bought with ~215× slower decode and a
> global erase-and-restore correctness obligation), and zstd±byte-split
> (float-blind, and a dependency where a hand-roll fits the registry).
> Implementation is #42, scheduled by footprint need rather than
> milestone; until it lands, uncompressed remains the shipped answer.
> Measurement caveat: before ALP is measured, the corpus ticks family
> must round to a realistic tick size — real prices are decimals, and
> an unrounded random walk misrepresents the target workload.

## Deployment shapes

> **Decided (2026-07-23): library first; a single-file shell binary at
> M3; never a server.** TallyDB ships two ways: as an embeddable
> library (the design center, unchanged), and — from M3 — as a
> standalone single-file binary attached to each release: a CLI shell
> over the same `engine::Database` doorway, the `sqlite3`/`duckdb`
> precedent. Installation is copying one file. This is not a move
> toward general purpose: the shell exposes exactly the library's SQL
> surface, and the three assumptions bound it the same way.
>
> What the shell shape pulls in (all additive, none a refactor): DDL
> (`CREATE TABLE` with the numeric-or-key types and the declared
> ordering key) and ingest (`INSERT`, plus a bulk import) in SQL;
> statically linked compute (which dovetails with the pinned
> from-source OpenBLAS build recorded under *Numerical consistency*);
> a process lock on the storage directory (two processes opening one
> table is undefined until then); and per-platform release builds in
> CI. Rendering key columns as text in the shell is fine — the shell
> *is* an application, exactly where the strings-precisely rule says
> display text belongs.
>
> **The rejected alternative is the engine growing a listener.** A
> server needs a wire protocol, auth, TLS, sessions, backpressure,
> multi-tenancy — general-purpose infrastructure orthogonal to the
> three assumptions — and the differentiator dies at a network
> boundary: compute-without-copying only exists in-process. If a
> served deployment is ever wanted, it is a **separate product that
> embeds TallyDB** (Arrow Flight is the natural seam — SQL in,
> `ArrowArrayStream` out is already the engine's shape), the way
> rqlite wraps SQLite and MotherDuck wraps DuckDB. The engine-side
> obligation that keeps third-party servers viable is only this: stay
> embeddable in a concurrent host — snapshot reads through `&self`,
> single writer, a clean `Send`/`Sync` story. No reopen condition is
> foreseen for the listener; the network-boundary argument is
> structural.

## Current milestone: native only

We are building the **native build first** — Linux/Mac/Windows, linked into
an application. A WASM build (and eventually a WASM compute layer) is a real
future direction, not current scope — do not add WASM-target dependencies
(`lua.wasm`, `blas.wasm`, LAPACK-wasm) or write WASM-specific code paths
yet. What *is* required now: keep I/O and compute behind trait boundaries
from day one (storage backend, scripting backend, math backends), with no
filesystem, threading, or dependency assumptions baked into the core crates
that would block a future `wasm32` target. That discipline is cheap today
and expensive to retrofit — don't skip it, but don't build the WASM side of
it either. WASM matters to this project specifically because the two hardest
WASM pieces — [`blas.wasm`](https://github.com/andy-emerson/blas.wasm) and
`lua.wasm` — already exist, authored by the same author, with a
LAPACK-in-WASM layer as their next milestone.

## Design philosophy

Every architectural choice follows one rule: **take mature, narrow,
well-tested dependencies as-is where they exist; write only the part that's
actually novel.**

### Taken as-is (do not fork, vendor, or reimplement)

- **`sqlparser-rs`** — SQL parsing.
- **PUC Lua 5.4** — embedded scripting; the canonical upstream sources
  compiled into the engine unmodified, which is the embedding model Lua
  is designed and distributed for. (Not LuaJIT, and not via `mlua` — see
  *The Lua layer* below for the decision record.)
- **Native BLAS/LAPACK** (OpenBLAS/MKL/Accelerate) — via FFI.

These are mature, narrow, embedding-oriented dependencies — linking them
whole is safe because their entire purpose is being called into by a host
program. Don't write a SQL parser, a Lua interpreter, or a linear algebra
library from scratch — that work already exists and is already correct.

### Used as a correctness oracle, never linked at runtime

- **DuckDB (primary) and DataFusion (secondary).** Dev-dependency only, used
  to differentially test `query-lite`'s executor: for the portion of SQL
  semantics that overlaps standard behavior (aggregates, joins, window
  functions), run the same query against the oracle and diff the output.
  Oracle criteria are **not** product criteria: the oracle never ships, so
  its size is irrelevant — what matters is authority on analytic-SQL
  semantics (window functions, statistical aggregates) and running
  in-process inside `cargo test`. That's DuckDB. SQLite is too thin exactly
  there (no statistical aggregates, weaker windows); InfluxDB is a server,
  not a linkable library — and its v3 SQL engine *is* DataFusion, which the
  secondary oracle covers directly, as a library. This oracle strategy is
  one more reason the analytic numeric type stays `f64` — the oracle
  computes in `f64`, so an integer/rational compute path would have nothing
  to diff against. We do **not** vendor DataFusion's executor — its useful
  parts are coupled to its own general-purpose planner, and extracting a
  piece drags the planner's scaffolding with it. If you find yourself
  wanting to pull in DataFusion code to solve an execution problem, stop —
  write the narrow thing ourselves and check it against DuckDB/DataFusion's
  output instead.
- **arrow-rs / PyArrow.** Dev-dependency / CI-only, used as the round-trip
  oracle for `arrow-lite`'s hand-rolled layout and C Data Interface export
  (issue #2). Same pattern: the mature implementation validates our bytes in
  tests and is never linked at runtime.

### Genuinely original (no oracle exists — our tests are the spec)

- `storage-lite`'s append/ordered/compaction/tombstone design.
- The numeric-or-key schema invariant itself, enforced in `engine`.

Test these thoroughly and deliberately. There is no reference implementation
to diff against for this part of the project — the tests written here
effectively *are* the specification.

## How decisions are made here (hygiene, 2026-07-24)

Three rules, adopted after a sweep of this project's own decision
history found the same defect twice (a codec fork framed from a 2015
paper and decided in 2026; an interpreter treated as settled because
early drafts named it):

1. **Option spaces carry provenance.** A decision record states how and
   when its options were assembled; a fork bounded by a moving field
   cites a check of current practice at decision time, not framing time.
2. **A tripwire for what must be surfaced.** A choice is a decision —
   not routing — when it freezes an external contract (bytes, API),
   sets user-visible semantics, or sets a product guarantee. These are
   surfaced to the architect even when discovered mid-pass, even when
   one option seems obvious.
3. **Settled requires a record.** A choice inherited from early drafts
   is not settled; settled means a record exists naming the
   alternatives that lost. Absence of a record means open.

Ratified as deliberate under rule 3 (2026-07-24): `SUM(i64)` stays
exact and errors loudly on overflow; query output is one Arrow batch
per segment; window frames are `ROWS`-only for now. Interim states with
their decisions still open: durability boundary is the flush (issue
#43, must close before M3 ships a binary) and segments freeze at a
fixed row count (issue #44) — the two sibling cadence questions.

## Things that are settled "no"s — don't relitigate without a specific trigger

- **Compiled Lua C extensions** (`package.loadlib`). Pure-Lua libraries are
  fine and need no special handling. (See *The Lua layer* below for the full
  reasoning.)
- **A general LAPACK surface.** `compute-lapack` wraps a curated set: a
  least-squares solve, symmetric eigendecomposition, a general linear solve,
  and Cholesky — chosen because specific workflows (regression,
  covariance/PCA, portfolio weights) need exactly these. Don't add routines
  because LAPACK has them; add them because a named workflow needs them.
  (The multiplication-class primitives — dot, gemv, gemm — live in the
  separate `compute-blas` crate; see *The compute split* below. Don't
  conflate the two.)
- **Autodiff / a Torch-style tensor framework.** Different computational
  paradigm than anything the target workload (closed-form / classical
  numerical methods) needs. If a specific, repeated, real need shows up
  later, it gets a narrow scoped addition, not this whole paradigm.
- **Building out a "scientific ecosystem"** (e.g. Julia's
  DifferentialEquations.jl-style breadth) to compensate for Lua's thinner
  ecosystem. Not this project's job — the embedded Lua scripting layer is
  the intended escape hatch for gaps, not something we pre-fill.
- **A general query optimizer / cost-based planner.** Query shapes are
  assumed simple (one fact table + small dimension tables, star-schema
  equi-joins).
- **Arrow IPC / Flight / Parquet in `arrow-lite`.** The interop surface is
  the C Data Interface (including the stream variant), nothing else — IPC
  drags in FlatBuffers and a much larger spec. Parquet in/out is the
  application's job via ecosystem tools that already speak C-Data.

If something on this list seems newly justified, that's a conversation to
have explicitly (update this document and its companions together), not a
decision to make silently inside an implementation PR.

## The compute split: `compute-blas` vs `compute-lapack`

Compute sits behind trait boundaries the `engine` calls through, so native
implementations can eventually be joined by WASM ones without changing
anything above. The compute layer is **two crates**, following the real
library boundary (LAPACK is built on BLAS and calls into it), the consumer
boundary, and the WASM-availability boundary:

- **`compute-blas`** — multiplication-class primitives (dot, gemv, gemm).
  Direct consumers: the executor's window/numeric inner loops and Lua
  scripts through `compute-lua`'s registered functions. Native: OpenBLAS
  BLAS. WASM (future): `blas.wasm`, which exists.
- **`compute-lapack`** — the curated analytical solves/decompositions, each
  justified by a named workflow: least-squares solve (rolling regression),
  symmetric eigendecomposition (covariance/PCA), general linear solve
  (portfolio weights/factor models), Cholesky (positive-definite covariance
  fast path). Native: LAPACK. WASM (future): a LAPACK-in-WASM layer that
  does **not** exist yet and is the next milestone of the `blas.wasm`
  project.

`compute-lapack` does **not** depend on `compute-blas` at the Rust level —
it calls the LAPACK library, which internally calls its own BLAS. Both are
siblings over `arrow-lite`; `engine` depends on both.

The design-critical part (do this now, it's what keeps WASM from being a
rewrite): keep the two behind **distinct traits with independently gated
backends**, and make **capability negotiation** first-class — "this op is
unavailable on this backend" is a returnable answer, not a panic. That's how
a WASM build lands with storage + query + Lua + BLAS-class ops working and
LAPACK-class analytics gracefully degraded until LAPACK-in-WASM ships. The
crate split itself is the honest expression of that boundary; don't hide
LAPACK inside a crate named "blas."

**Decision record — the honest zero-copy claim (column-group arena
considered and set aside).** LAPACK wants column-major matrices in one
allocation with uniform stride; table columns are separate allocations. So
the zero-copy claim is stated precisely: **vector-shaped ops and window
slices are zero-copy into compute; assembling a multi-column design matrix
is one bounded gather** — an O(n·k) copy feeding an O(n·k²) solve, so the
copy is asymptotically invisible exactly where it would matter most. The
rejected alternative was a shared arena allocating a segment's same-length
`NOT NULL` `f64` columns at uniform stride so a table chunk *is* a matrix;
set aside because it couples `arrow-lite`'s allocator to `storage-lite`'s
segment layout and constrains compaction. Reopen trigger: profiling on
target workloads shows design-matrix assembly is a material fraction of
query time.

**Decision record — rolling regression solves a centered factorization
(2026-07-24).** The design matrix is `[1 | x − x̄]`, never raw `[1 | x]`:
a regressor with a large offset relative to its in-window spread (a
timestamp-scale x) makes the raw pair catastrophically ill-conditioned —
measured on a 20-row window (run 2026-07-24, pinned as
`rolling_regression_survives_timestamp_scale_x`): the raw solve loses
the slope entirely from offset 1e9 while the centered solve holds
~3e-11 relative error through 1e15 (bug #45). The rejected default was
streaming sufficient statistics — O(1) per window slide and how DuckDB
computes `regr_slope` — because the running-sums formula squares the
condition number and degrades a thousand-fold earlier (five digits gone
at offset 1e6). It may return later as an explicit opt-in fast path
with its accuracy caveat documented; reopen trigger: profiling shows
per-window factorization dominating a real workload.

## Batch, not per-row, for Lua and BLAS/LAPACK calls

Every call from the query executor into `compute-lua`, `compute-blas`, or
`compute-lapack` should operate on a whole column or window per call, not
element-by-element. Per-row calls throw away the entire performance
rationale for pairing a columnar engine with these compute layers. If an API
makes per-row calls the easy/obvious way to use it, that's a bug in the API
shape.

## The Lua layer

The embedded interpreter is **canonical PUC Lua 5.4**, compiled into the
engine from the unmodified upstream sources — the embedding model Lua is
designed around. Scripts reach the engine's buffers through zero-copy
userdata views: the userdata wraps the live `arrow-lite` buffer pointer
and its accessors are implemented on the Rust side, so no bytes are
copied. Stated precisely, in the same spirit as the compute-split's
zero-copy record above: *access* is zero-copy, but each element read is
a metamethod dispatch rather than a compiled raw load. The curated
`compute-blas`/`compute-lapack` ops are exposed to scripts as registered
functions operating over those same views — sharing buffers, not
copying between them. Lua 5.4's numeric model — one number type with a
64-bit integer subtype and a 64-bit float subtype — is exactly TallyDB's
`i64`/`f64` column pair, so numeric values cross the script boundary
without losing exactness; that alignment is a load-bearing reason for
the 5.4 choice, not a convenience.

The performance story for scripts is a **promotion ladder**, not a JIT:
write the custom kernel in Lua to get it *correct* — immediately,
cross-checkably — and if it proves hot, promote it to a curated native
op to make it *fast*. That is the pattern `regr_slope`, `covar_pop`,
`corr`, and `eigen_max` already followed. Interpreter speed is a
comfort, not a foundation: the engine's speed lives in columnar storage
and pruning, BLAS/LAPACK, and the batch calling convention above, none
of which pass through the interpreter's inner loop.

**Decision record — interpreter and binding (2026-07-24).** Two
alternatives rejected, each with a reopen condition:

- **LuaJIT** (the original plan) — rejected. It is a fork frozen at Lua
  5.1: no native 64-bit integers (only `int64_t` cdata boxes, with
  different equality, hashing, and mixing semantics — a permanent seam
  through the scripting surface of a database that is careful about
  `i64` exactness everywhere else), and a permanent version skew
  against the WASM build's `lua.wasm`, which is Lua 5.4 (a fork of
  lua-aot, whose runtime is stock 5.4). Canonical 5.4 on both targets
  deletes the skew instead of managing it, and canonical-over-fork is
  this project's own thesis applied to a dependency. What LuaJIT
  offered — trace-compiled script loops and `ffi` raw-pointer access —
  is covered by the promotion ladder. Reopen condition: a real workload
  shows ad-hoc kernel performance is unacceptable *and* promotion to a
  native op cannot cover it.
- **`mlua`** (the safe binding wrapper) — rejected, including as a
  dev-only witness. It is neither canonical nor small (five Lua
  versions, serde, async, macro machinery — we would use a sliver), and
  the witness role does not survive inspection: diffing two bindings
  over the same vendored interpreter mostly tests the interpreter
  against itself, while a binding's real failure modes — stack
  imbalance, GC anchoring mistakes, a `longjmp` over Rust frames — are
  memory-safety violations that output diffing cannot see. Reopen
  condition: the C API surface we actually need balloons well past the
  ~two dozen functions the batch convention implies.

What ships instead: **hand-rolled thin bindings** to the 5.4 C API,
with the error discipline built in by construction — every entry into
Lua goes through `lua_pcall`; Rust functions called from Lua never
raise a Lua error across frames with pending destructors; and
`catch_unwind` at the boundary so a Rust panic never unwinds into C.
Verified with no binding dependency at all, using Lua's own enforcement
plus standard tooling: test builds compile the vendored interpreter
with `LUA_USE_APICHECK`, so the interpreter itself asserts on C API
misuse (the real oracle for binding discipline); seam tests run under
the official test suite's GC/allocation-torture infrastructure
(`ltests.c` — full collection on every allocation, injectable
allocation failure); the official Lua test suite runs against the
vendored build in CI; and ASan/UBSan cover the combined artifact. This
is the arrow-lite configuration — a frozen canonical spec *and* an
external oracle — the same pair that decided the hand-roll there (#2).
The AOT compilation path (lua-aot natively) is *not* adopted: our
ad-hoc scripts are unknown at build time, so AOT lands on the one part
of the design that cannot use it; it remains available later, at zero
semantic cost, for any precompiled script library we might ship.

Embedded Lua supports pure-Lua libraries (plain `.lua` source) out of the
box — they run as ordinary Lua code with no extra integration work.
Compiled C extensions (LuaRocks packages with a `.so`/`.dll` component,
loaded via `package.loadlib`) are not supported: allowing arbitrary compiled
code to load inside an embedded database process is a real attack-surface
and stability tradeoff, and it cuts against the curated-not-general instinct
behind everything else in this design. This is also structurally true for
the WASM backend regardless of policy — WASM's sandbox can't do
`dlopen`-style dynamic loading at all — so the two constraints reinforce
each other rather than being separate decisions.

## Numerical consistency

Native and WASM builds won't be bit-identical by default — floating-point
addition isn't associative, and different SIMD widths / FMA usage change
summation order. We're not solving full native/WASM bit-identity now; that's
future work once a WASM build exists. But it's cheap to build toward:
`blas.wasm` already defers fused-FMA variants specifically to preserve
determinism, and native OpenBLAS built from source with `TARGET=SANDYBRIDGE`
forces pre-FMA kernels (AVX, no FMA) while staying fast on essentially any
x86_64 CPU from 2011 onward — meaningfully faster than the more conservative
`TARGET=NEHALEM` (SSE-only) for the same determinism guarantee. There's no
off-the-shelf "non-FMA" package — it's a build-time `TARGET=` decision, not
a switch, and a known low-effort one to make when it's actually needed.

## How we test this repository

The test plan's skeleton, kept here per the working agreement: the plan's
schedule lives in the milestones, its executable detail in the test code
and corpus, and the latest results in CI. Growing enumerations — case
lists, corpus entries — belong with the tests, not in this file.

### What "correct" means here

1. **Agrees with the oracle.** For the SQL semantics that overlap standard
   behavior: same query, same data → same output as DuckDB (primary) /
   DataFusion (secondary). **Every oracle has a declared scope of
   authority** (convention, 2026-07-24): an oracle checks that we compute
   *our chosen* semantics correctly — where the standard leaves a choice
   (null placement, integer overflow), the choice is ours, recorded in
   this document, and the harness normalizes the documented divergence.
   An oracle never chooses semantics, and a diff must never share the
   implementation's computational path (the #45 lesson: an oracle solving
   the same ill-conditioned matrix agreed with the wrong answer).
2. **Round-trips with real Arrow.** Columns exported over the C Data
   Interface import identically in arrow-rs and PyArrow, and vice versa —
   dictionaries, nulls, and logical types intact.
3. **Deterministic where promised.** Same seeded input, same pinned
   compute backend → bit-identical segment bytes and result buffers,
   checked against committed goldens. Storage bytes are promised
   backend-independent; `f64` results are promised per the pinned non-FMA
   OpenBLAS build (see *Numerical consistency*). A change that moves those
   bits is a behavioral change, not a refactor — re-blessing the goldens
   is part of its review.
4. **Meets its own spec** where no reference exists — `storage-lite`'s
   tests are the spec (see the reference map).

### The reference map

| Claim family | Reference | Tier |
|---|---|---|
| `query-lite` SQL semantics | DuckDB (primary) / DataFusion (secondary) | independent oracle |
| `arrow-lite` layout + C Data Interface | arrow-rs / PyArrow round-trips, dev-only | independent oracle |
| compute seam (our calls into BLAS/LAPACK) | NumPy/SciPy on the same inputs | independent oracle |
| determinism (storage bytes; pinned-backend results) | committed goldens | prior output |
| `storage-lite` behavior (append, compaction, tombstones) | its own spec-tests | none — tests are the spec |

`storage-lite` occupies the weakest tier — no independent reference exists
for its behavior. That is why the build order front-loads it and why its
tests deserve the most scrutiny.

### Peers, for measurement claims

**DuckDB** — primary peer, also the oracle and the control group: one
corpus, diffed for correctness and timed for performance, so we never
benchmark a wrong answer. **SQLite** — the floor: what the simplest
embeddable store costs on this workload. **The exported-workflow
pipeline** (DuckDB → pandas/NumPy) — the peer for the headline pair: the
same rolling analytics computed in-engine versus exported-and-computed,
the copy tax made visible. **kdb+** is excluded from *published* numbers
pending a license review (commercial database licenses commonly prohibit
benchmark publication). Below the SQL surface there is no peer;
micro-level work uses self-comparison benches as engineering instruments.

### The corpus

Seeded synthetic generators — ordered `i64` timestamps, low-cardinality
keys, `f64` values, with disorder fraction and null density as
parameters — checked into the repository as the plan's executable detail.
It grows two ways: new capabilities add case families, and every closed
bug adds the case that would have caught it.

### Blast radius (where evidence lands earliest and heaviest)

1. **Storage bytes** — silent corruption; entrenches at format freeze.
   Golden-locked *before* the first real data exists in the format.
2. **C Data Interface unsafe export** — silent corruption in *other
   processes'* memory.
3. **Oracle-visible SQL semantics** — wrong answers, but loud under
   differential testing.
4. Everything else.

## Workspace layout

TallyDB is a single Cargo workspace — each crate has a clean boundary and
can be reasoned about and tested in isolation, but they share one version
history and one build.

```
tallydb/
  crates/
    arrow-lite/     # hand-rolled Arrow-compatible columnar format (f64/i64
                    #   buffers, u32-dictionary keys, C Data Interface export;
                    #   arrow-rs/PyArrow as dev-only round-trip oracles)
    storage-lite/   # append-optimized segments partitioned on the ordering
                    #   key; compaction; zone maps; I/O behind a backend
                    #   trait (native = a directory of files; mmap/ranged
                    #   reads when pruning or profiling asks; OPFS/WASM
                    #   backend can be added later)
    query-lite/     # scoped SQL parser (via sqlparser-rs) + our own executor;
                    #   validated against DuckDB/DataFusion as an oracle
    engine/         # ties storage + query + compute together; enforces
                    #   numeric-or-key as a hard schema rule
    compute-lua/    # Lua scripting behind a trait; vendored PUC Lua 5.4,
                    #   hand-rolled bindings (lua.wasm, also 5.4, later)
    compute-blas/   # multiplication-class BLAS behind a trait; OpenBLAS via
                    #   FFI for now (blas.wasm later)
    compute-lapack/ # curated LAPACK solves/decompositions behind a trait;
                    #   native LAPACK via FFI for now (LAPACK-wasm later)
    corpus/         # dev-only: the seeded synthetic generators of "The
                    #   corpus" above; measurement and differential-test
                    #   data, never linked by the engine
  Cargo.toml
```

## Build order (recommended, not mandatory)

The dependency graph is shallow and wide, not a deep chain: everything
depends on `arrow-lite`, almost nothing else depends on anything else. So
the only *order-critical* thing is locking `arrow-lite`'s layout first;
after that the rest is a wide front, and the ordering below is a
**risk**-ordering (front-load the unoracled crates), not a dependency chain.

1. `arrow-lite` — smallest, clearest spec (Arrow's public layout), no
   internal dependencies. Lock its two interfaces early: the raw-pointer/FFI
   view (for compute) and the serialize-to-segment view (for storage).
   **Resolved (issue #2):** hand-rolled, no runtime arrow-rs dependency;
   `u32` dictionary codes; optional validity bitmaps (`NOT NULL` columns
   have none; the ordering key is always `NOT NULL`); logical-type export
   annotations (`Timestamp(ns)`, `Decimal64(scale)`); C Data Interface
   only, including the batch-stream variant. Round-trip test against
   arrow-rs/PyArrow (dev-only). Get this right before anything else.
2. `storage-lite` — the highest-risk, most original crate. Deserves the
   most scrutiny and the most tests, precisely because there's no oracle.
   (Its two format-gating decisions are settled — row identity by internal
   row id, per-segment dictionaries; see *Storage* above.)
3. `query-lite` — can lean on DuckDB/DataFusion as a differential oracle
   once `storage-lite` is stable enough to query.
4. `compute-lua` / `compute-blas` / `compute-lapack` — native backends
   (vendored Lua 5.4, OpenBLAS, LAPACK); can be developed in parallel with
   `query-lite` once `arrow-lite`'s buffer format is stable, since they
   consume it directly.
5. `engine` — last, since it's the integration point for everything above.

**The one sequencing constraint that matters most:** the differentiator is
compute-fusion (zero-copy numeric ops on stored buffers), and that's the
riskiest, least-trodden part. Reach a thin end-to-end proof of it *early* —
ingest numeric+key rows → a windowed query that calls a `compute-lapack` op
on stored buffers with no copy → Arrow out — rather than leaving it for
last. Building the storage engine beautifully while the compute story slips
just yields "another embeddable TSDB" and misses the point.

Don't try to scaffold all seven crates' real implementations in one pass.

## Who we write for

The imagined reader holds a **BS in Applied Mathematics with a minor in
Computer Science** — which is also a fair description of the target user.
Concretely:

- **Documentation is written for the math-major side.** It may assume
  mathematical fluency — "positive semi-definite," "least squares," "QR
  decomposition" need no apology — but must not assume systems fluency:
  terms like *mmap*, *tombstone*, *cache line*, or *FFI* are defined at
  first use.
- **Code is written for the CS-minor side.** Standard idioms, clear
  structure, no cleverness for its own sake. Where performance demands a
  non-obvious idiom (unsafe pointer work, bitmap tricks, SIMD-shaped
  loops), the accompanying comment carries the naive equivalent or an
  explanation, so the reader can verify the clever version against it.
- **Performance wins every conflict with this constraint** — it is a
  nice-to-have, never a reason to ship slower code. But each win is
  documented as a deliberate bend, which keeps the constraint honest.

Where documentation can carry executable evidence, prefer it: Rust doctests
compile and run in `cargo test`, so a documented claim with a doctest fails
loudly when it stops being true (see `AGENTS.md` on executable
documentation).
