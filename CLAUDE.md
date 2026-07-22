# Working in this repository

Read `README.md` first — it has the full user-facing and developer-facing
picture. This file is operational guidance for whoever (human or agent) is
actually writing code here: the guardrails, not the pitch.

## The three assumptions (do not relax these to unblock a feature)

1. **Append-optimized.** Writes are cheap, low-latency, one row at a time.
2. **Ordered.** Data arrives roughly in time order; storage is time-partitioned.
3. **Numeric-or-key.** Every column is numeric (`f64` by default) or a
   dictionary-encoded key. No third type. Ever. If a feature seems to need
   one, the feature is wrong, not the invariant.

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

- **DuckDB and/or DataFusion.** Dev-dependency only, used to differentially
  test `query-lite`'s executor (run the same query against both, diff the
  output). We do **not** vendor DataFusion's executor — its useful parts
  are coupled to its own general-purpose planner, and extracting a piece
  drags the planner's scaffolding with it. If you find yourself wanting to
  pull in DataFusion code to solve an execution problem, stop — write the
  narrow thing ourselves and check it against DuckDB/DataFusion's output
  instead.

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
fast path.

## Things that are settled "no"s — don't relitigate without a specific trigger

- **Compiled Lua C extensions** (`package.loadlib`). Pure-Lua libraries
  are fine and need no special handling.
- **A general LAPACK surface.** `compute-blas` wraps a curated set: a
  least-squares solve, symmetric eigendecomposition, a general linear
  solve, and Cholesky — chosen because specific workflows
  (regression, covariance/PCA, portfolio weights) need exactly these.
  Don't add routines because LAPACK has them; add them because a named
  workflow needs them.
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

If something on this list seems newly justified, that's a conversation to
have explicitly (update this file and the README together), not a decision
to make silently inside an implementation PR.

## Batch, not per-row, for Lua and BLAS/LAPACK calls

Every call from the query executor into `compute-lua` or `compute-blas`
should operate on a whole column or window per call, not element-by-element.
Per-row calls throw away the entire performance rationale for pairing a
columnar engine with these compute layers. If an API makes per-row calls
the easy/obvious way to use it, that's a bug in the API shape.

## Build order (recommended, not mandatory)

1. `arrow-lite` — smallest, clearest spec (Arrow's public layout), no
   internal dependencies. Get this right and tested before anything else.
2. `storage-lite` — the highest-risk, most original crate. Deserves the
   most scrutiny and the most tests, precisely because there's no oracle.
3. `query-lite` — can lean on DuckDB/DataFusion as a differential oracle
   once `storage-lite` is stable enough to query.
4. `compute-lua` / `compute-blas` — native backends (LuaJIT, OpenBLAS)
   via FFI; can be developed in parallel with `query-lite` once
   `arrow-lite`'s buffer format is stable, since both consume it directly.
5. `engine` — last, since it's the integration point for everything above.

Don't try to scaffold all six crates' real implementations in one pass.
