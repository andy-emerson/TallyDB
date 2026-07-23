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

## The inclusion principle for the SQL surface

Include a standard SQL function or verb if it (a) doesn't require a
non-numeric, non-key column type, and (b) doesn't require a general-purpose
cost-based optimizer. **"We can't think of a quant use case for it" is
explicitly NOT a valid reason to exclude something otherwise in scope** —
real usage regularly surprises the people who built the tool. The
invariants are the boundary, not our own imagination.

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

> **Open decisions (tracked in issues, both gating `storage-lite`'s
> formats):** (1) the *row-identity* rule that makes "newest version wins"
> well-defined — an InfluxDB-style `(key-set, ordering-key)` primary key
> with overwrite-on-collision, versus a kdb+-style pure-append model with an
> internal row id and predicate-scoped deletes — gates the tombstone format;
> (2) *per-segment vs. global dictionary* for key columns gates the segment
> format. Don't hardcode either until decided.

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
- **LuaJIT** — native scripting, via FFI.
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
  Direct consumers: the executor's window/numeric inner loops and
  Lua-over-FFI. Native: OpenBLAS BLAS. WASM (future): `blas.wasm`, which
  exists.
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

## Batch, not per-row, for Lua and BLAS/LAPACK calls

Every call from the query executor into `compute-lua`, `compute-blas`, or
`compute-lapack` should operate on a whole column or window per call, not
element-by-element. Per-row calls throw away the entire performance
rationale for pairing a columnar engine with these compute layers. If an API
makes per-row calls the easy/obvious way to use it, that's a bug in the API
shape.

## The Lua layer

LuaJIT and native BLAS/LAPACK aren't just linked side by side — they're
called directly from Lua via LuaJIT's FFI, which lets Lua declare a C
function's signature and call straight into a linked library with near-zero
overhead, no hand-written binding layer, numeric arrays passed as raw
pointers into the same memory the query engine already holds. This is how
the original Torch (pre-PyTorch) worked: LuaJIT + BLAS-backed tensors over
FFI. TallyDB's `compute-lua`/`compute-blas`/`compute-lapack` crates use the
same mechanism, scoped to a curated set of operations rather than a general
tensor/autodiff library.

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
   DataFusion (secondary).
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
                    #   key; compaction; zone maps; native backend = mmap
                    #   today, behind a trait so an OPFS/WASM backend can be
                    #   added later
    query-lite/     # scoped SQL parser (via sqlparser-rs) + our own executor;
                    #   validated against DuckDB/DataFusion as an oracle
    engine/         # ties storage + query + compute together; enforces
                    #   numeric-or-key as a hard schema rule
    compute-lua/    # Lua scripting behind a trait; LuaJIT via FFI for now
    compute-blas/   # multiplication-class BLAS behind a trait; OpenBLAS via
                    #   FFI for now (blas.wasm later)
    compute-lapack/ # curated LAPACK solves/decompositions behind a trait;
                    #   native LAPACK via FFI for now (LAPACK-wasm later)
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
   (Gated on two tracked decisions: row identity for its tombstone format,
   and per-segment vs. global dictionary for its segment format.)
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
