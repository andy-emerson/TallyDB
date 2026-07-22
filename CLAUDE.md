# Working in this repository

Read `README.md` first — it has the full user-facing and developer-facing
picture. This file is operational guidance for whoever (human or agent) is
actually writing code here: the guardrails, not the pitch.

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

### Numbers are `f64` *or* `i64` — with roles

- **`i64`** (and fixed-point decimal over it) is the exact/stored/fact type:
  nanosecond timestamps (which do **not** fit in `f64` — 53-bit mantissa caps
  it at microseconds), money as scaled integers, volumes, counts. Ordered
  `i64` columns are also what delta/delta-of-delta compression is built for.
- **`f64`** is the analytic/derived type: anything BLAS/LAPACK touches
  (regression, covariance eigenvalues, correlations, weights — irrational in
  general), and what keeps NumPy interop and the DuckDB oracle working.

Rationals-as-the-numeric-type (`i64/i64`) plus a homegrown integer linear
algebra were considered and **rejected**: denominators overflow `i64` within
a few divisions, bignum rationals are variable-width (killing Arrow interop
and SIMD), and rationals can't even represent the irrational analytics
outputs. Floating-point *done carefully* is the tool; reproducibility is
handled at the BLAS build level (non-FMA kernels), not by dropping floats.

### Strings: predicates yes, production no

The numeric-or-key rule is not "no strings anywhere." Key columns are
dictionary-encoded interned strings, so string **predicates** on keys
(`=`, `IN`, `LIKE`, regex) are in scope — they emit a row selection, not a
string, and run once per *distinct* dictionary value (cheap). What's out is
string **production**: no function may *emit* a string value (`SUBSTRING`
projection, `CONCAT`, `CAST AS VARCHAR`, `GROUP_CONCAT`). A key result comes
back as its integer code plus the dictionary to render it; formatting is the
application's job.

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

## Current milestone: native only

We are building the **native build first**. A WASM build is a real future
direction, not current scope — do not add WASM-target dependencies
(`lua.wasm`, `blas.wasm`, LAPACK-wasm) or write WASM-specific code paths
yet. What *is* required now: keep I/O and compute behind trait boundaries
so a WASM backend can slot in later without a rewrite. That discipline is
cheap today and expensive to retrofit — don't skip it, but don't build the
WASM side of it either.

## What's taken as-is (do not fork, vendor, or reimplement)

- **`sqlparser-rs`** — SQL parsing.
- **LuaJIT** — native scripting, via FFI.
- **Native BLAS/LAPACK** (OpenBLAS/MKL/Accelerate) — via FFI.

These are mature, narrow, embedding-oriented dependencies. Link them.
Don't write a SQL parser, a Lua interpreter, or a linear algebra library
from scratch — that work already exists and is already correct.

## What's used as a correctness oracle, never linked at runtime

- **DuckDB (primary) and DataFusion (secondary).** Dev-dependency only, used
  to differentially test `query-lite`'s executor (run the same query against
  both, diff the output). Oracle criteria are **not** product criteria: the
  oracle never ships, so its size is irrelevant — what matters is authority
  on analytic-SQL semantics (window functions, statistical aggregates) and
  running in-process inside `cargo test`. That's DuckDB. SQLite is too thin
  exactly there (no statistical aggregates, weaker windows); InfluxDB is a
  server, not a linkable library — and its v3 SQL engine *is* DataFusion,
  which the secondary oracle covers directly, as a library. This oracle strategy is one more reason the analytic numeric type
  stays `f64` — the oracle computes in `f64`, so an integer/rational compute
  path would have nothing to diff against. We do **not** vendor DataFusion's
  executor — its useful parts
  are coupled to its own general-purpose planner, and extracting a piece
  drags the planner's scaffolding with it. If you find yourself wanting to
  pull in DataFusion code to solve an execution problem, stop — write the
  narrow thing ourselves and check it against DuckDB/DataFusion's output
  instead.
- **arrow-rs / PyArrow.** Dev-dependency / CI-only, used as the round-trip
  oracle for `arrow-lite`'s hand-rolled layout and C Data Interface export
  (issue #2). Same pattern: the mature implementation validates our bytes
  in tests and is never linked at runtime.

## What's genuinely original here (no oracle exists — our tests are the spec)

- `storage-lite`'s append/ordered/compaction/tombstone design.
- The numeric-or-key schema invariant itself, enforced in `engine`.

Test these thoroughly and deliberately. There is no reference
implementation to diff against for this part of the project — the tests
you write here effectively *are* the specification.

## The inclusion principle for the SQL surface

Include a standard SQL function or verb if it (a) doesn't require a
non-numeric, non-key column type, and (b) doesn't require a general-purpose
cost-based optimizer. **"We can't think of a quant use case for it" is
explicitly NOT a valid reason to exclude something otherwise in scope** —
real usage regularly surprises the people who built the tool. The
invariants are the boundary, not our own imagination.

`UPDATE` and `DELETE` are in scope, implemented as tombstone + reinsert
against `storage-lite` (the same mechanism ordinary corrections use) — not
a separate mutation path, and not excluded just because they aren't the
fast path. **Open decision (tracked in issues):** the row-identity rule that
makes "newest version wins" well-defined — InfluxDB-style `(key-set,
ordering-key)` primary key with overwrite-on-collision vs. kdb+-style pure
append with an internal row id and predicate-scoped deletes — is not yet
settled and gates `storage-lite`'s tombstone format. Don't hardcode either
one until it's decided.

## Things that are settled "no"s — don't relitigate without a specific trigger

- **Compiled Lua C extensions** (`package.loadlib`). Pure-Lua libraries
  are fine and need no special handling.
- **A general LAPACK surface.** `compute-lapack` wraps a curated set: a
  least-squares solve, symmetric eigendecomposition, a general linear
  solve, and Cholesky — chosen because specific workflows
  (regression, covariance/PCA, portfolio weights) need exactly these.
  Don't add routines because LAPACK has them; add them because a named
  workflow needs them. (The multiplication-class primitives — dot, gemv,
  gemm — live in the separate `compute-blas` crate; see "compute split"
  below. Don't conflate the two.)
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
have explicitly (update this file and the README together), not a decision
to make silently inside an implementation PR.

## The compute split: `compute-blas` vs `compute-lapack`

Two crates, following the real library boundary (LAPACK is built on BLAS and
calls into it), the consumer boundary, and the WASM-availability boundary:

- **`compute-blas`** — multiplication-class primitives (dot, gemv, gemm).
  Direct consumers: the executor's window/numeric inner loops and Lua-over-
  FFI. Native: OpenBLAS BLAS. WASM (future): `blas.wasm`, which exists.
- **`compute-lapack`** — the curated analytical solves/decompositions (the
  four above). Native: LAPACK. WASM (future): a LAPACK-in-WASM layer that
  does **not** exist yet.

`compute-lapack` does **not** depend on `compute-blas` at the Rust level — it
calls the LAPACK library, which internally calls its own BLAS. Both are
siblings over `arrow-lite`; `engine` depends on both.

The design-critical part (do this now, it's what keeps WASM from being a
rewrite): keep the two behind **distinct traits with independently gated
backends**, and make **capability negotiation** first-class — "this op is
unavailable on this backend" is a returnable answer, not a panic. That's how
a WASM build lands with BLAS-class ops working and LAPACK-class ops
gracefully degraded until LAPACK-in-WASM ships. The crate split itself is the
honest expression of that boundary; don't hide LAPACK inside a crate named
"blas."

## Batch, not per-row, for Lua and BLAS/LAPACK calls

Every call from the query executor into `compute-lua`, `compute-blas`, or
`compute-lapack` should operate on a whole column or window per call, not
element-by-element. Per-row calls throw away the entire performance rationale
for pairing a columnar engine with these compute layers. If an API makes
per-row calls the easy/obvious way to use it, that's a bug in the API shape.

## Build order (recommended, not mandatory)

The dependency graph is shallow and wide, not a deep chain: everything
depends on `arrow-lite`, almost nothing else depends on anything else. So the
only *order-critical* thing is locking `arrow-lite`'s layout first; after
that the rest is a wide front, and the ordering below is a **risk**-ordering
(front-load the unoracled crates), not a dependency chain.

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
   (Gated on two tracked decisions: row identity — above — for its
   tombstone format, and per-segment vs. global dictionary for its segment
   format.)
3. `query-lite` — can lean on DuckDB/DataFusion as a differential oracle
   once `storage-lite` is stable enough to query.
4. `compute-lua` / `compute-blas` / `compute-lapack` — native backends
   (LuaJIT, OpenBLAS, LAPACK) via FFI; can be developed in parallel with
   `query-lite` once `arrow-lite`'s buffer format is stable, since they
   consume it directly.
5. `engine` — last, since it's the integration point for everything above.

**The one sequencing constraint that matters most:** the differentiator is
compute-fusion (zero-copy numeric ops on stored buffers), and that's the
riskiest, least-trodden part. Reach a thin end-to-end proof of it *early* —
ingest numeric+key rows → a windowed query that calls a `compute-lapack` op
on stored buffers with no copy → Arrow out — rather than leaving it for last.
Building the storage engine beautifully while the compute story slips just
yields "another embeddable TSDB" and misses the point.

Don't try to scaffold all seven crates' real implementations in one pass.
