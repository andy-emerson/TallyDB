# TallyDB

**A small, embeddable, SQL-native database for numeric data — with numeric compute living inside the engine, not bolted on beside it.**

TallyDB is an HTAP-shaped store: fast, append-heavy ingest (the write-optimized half) feeding directly into ordered, columnar, analytical reads (the read-optimized half), with no ETL step between them — and with BLAS/LAPACK and Lua compute that runs *on the engine's own buffers, in-process, with no copy*. It's built around three assumptions about the data it stores:

1. **Append-optimized.** Data arrives as new rows, cheaply and one at a time — though corrections are supported (see below), just not the design center.
2. **Ordered.** Rows arrive roughly sorted on a declared **ordering key** (a timestamp is the common case, but any monotonically-increasing-on-ingest key works — a sequence number, an event id, a ledger offset). Storage is partitioned on that key.
3. **Numeric-or-key.** Every column is either a **number** (`f64` or `i64`, used in arithmetic, aggregation, windows) or a **key** (a dictionary-encoded identifier or label, used only for filtering, grouping, and joining — never arithmetic).

These three assumptions aren't restrictions bolted on after the fact — they're the whole design. Relaxing any one of them is what makes general-purpose databases (Postgres, DuckDB, SQLite) bigger, slower to start, and harder to embed. Holding all three is what lets TallyDB stay small, fast, and honest about what it's for — and, crucially, is what makes fixed-width columns you can hand straight to a math library possible.

> **On "time-series."** Time-series, sensor telemetry, and tick data are the motivating **use cases**, not the definition. What's load-bearing is *ordered ingest on some key*, not that the key means "time." A monotonic sequence id serves the storage engine exactly as well as a nanosecond timestamp. So TallyDB is an **append-ordered numeric store**; "time-series database" is one hat it wears.

---

## For users: what TallyDB is and isn't

**What it's for.** Workloads that are a big, append-heavy ledger of numbers with some labels attached, analyzed with SQL — rolling aggregates, joins against small reference tables, grouping, window functions, and numeric compute (regression, covariance/PCA, portfolio math) run *in the database*. Quantitative research, sensor and telemetry pipelines, event/metric streams, financial ledgers: anything whose shape matches the three assumptions above.

**What it's not for.** It is not a general-purpose relational database. There are no arbitrary text columns or blobs, no third column type, and no general multi-table joins outside a star-schema shape. If your data doesn't fit the three assumptions, use Postgres, DuckDB, or SQLite — they're better at being general. TallyDB is a **specialized component** you reach for alongside a general store, the way SQLite often is — not the one database that runs your whole org.

**The SQL surface.** TallyDB supports standard SQL over its schema: `SELECT`/`WHERE`/`GROUP BY`/`ORDER BY`, equi-joins, window functions, and — yes — `UPDATE`/`DELETE`. Under the hood, both mutations are implemented as tombstone-plus-reinsert against immutable, append-only storage (the same mechanism handles ordinary corrections), resolved at the next compaction rather than in place. They aren't the fast path, and the engine isn't optimized for frequent use of them — but they're real, correct, and available, because withholding a SQL verb the storage engine already supports under a different name would just push the same work into application code.

**Strings, precisely.** The numeric-or-key rule holds across the *entire pipeline* — stored columns, intermediate results, and query outputs are always numeric or key; a bare string never exists in the engine. That has one nuance worth stating clearly, because it's more permissive than it sounds:

- **String *predicates* on key columns are in scope.** `WHERE symbol IN (...)`, `WHERE name LIKE '%Bank%'`, regex matching — these consume the interned strings and emit a *row selection*, not a string, so they don't need a third type. Because keys are dictionary-encoded, the predicate is evaluated once per *distinct* value in the small dictionary and then applied as integer set-membership: string filtering is not just allowed, it's cheap.
- **String *production* is out.** No function may *emit* a string value: no `SUBSTRING`/`CONCAT` projection, no `CAST(x AS VARCHAR)`, no `GROUP_CONCAT`. A key comes back as its integer code plus the dictionary needed to render it; turning that into display text happens in your application.

More generally: **any standard SQL function or verb is in scope as long as it (a) doesn't require a non-numeric, non-key column type and (b) doesn't require a general-purpose cost-based optimizer.** We don't require ourselves to imagine a specific use case before including something — real usage regularly surprises the people who built the tool. The invariants are the boundary, not our own foresight.

**How you'd use it.**
- Link it into your application like SQLite or DuckDB — no server process, no separate database to administer.
- Query results come back in an Arrow-compatible columnar layout, directly usable by NumPy or other Arrow-aware tooling — no conversion step.
- For anything the built-in SQL functions don't cover, drop into embedded Lua — called directly from SQL, operating on the same numeric buffers the query engine already has in memory. Nothing is copied out to a separate scripting process or serialized across a boundary; the script, the query engine, and the curated BLAS/LAPACK ops all read and write the same buffers in place. This **compute-without-copying** property is the thing TallyDB is actually built around, not a bolted-on extra.
- Runs natively (Linux/Mac/Windows) for production and research pipelines.

**Why it exists — and what's actually novel.** None of the individual ingredients is new. Append-optimized columnar storage, dictionary-encoded keys, in-database compute — each exists somewhere. The differentiator is the *combination and packaging*: numeric compute (regression, covariance, factor math) running inside an **embeddable, SQL-native** engine, over **off-the-shelf** numeric libraries (LuaJIT, BLAS/LAPACK) on **zero-copy shared buffers** — rather than a bespoke array language (kdb+'s q) or a serialization hop (DuckDB ↔ Python). The honest one-line framing is *"an open, SQL-native, embeddable kdb+ for teams below kdb+ scale"*: the workload kdb+ proved over 25 years, minus the q language, minus the license, minus the server.

**Prior art.** TallyDB borrows validated ideas rather than inventing them:
- **InfluxDB** validates the key/numeric split directly — its tags-vs-fields model is close in spirit to TallyDB's key-vs-numeric rule, and its more recent move to real SQL validates SQL-native as the right surface. (Note that InfluxDB is actually *more permissive* — it allows string and boolean *fields*; TallyDB deliberately takes a strict subset, which is where the footprint and performance wins come from.) InfluxDB itself isn't minimal or embeddable (it's a distributed server on Arrow/DataFusion/Parquet), which is the gap TallyDB fills.
- **kdb+** validates both the workload (25+ years as the quant-finance standard) and the "keys as interned integers, keep everything else numeric" performance pattern — *and* the idea of compute living inside the database. But it's proprietary, licensed, and built around q rather than SQL. TallyDB replicates the shape, not the language or the licensing.

---

## For developers: how TallyDB is built

### Current focus: native-first, WASM-ready

Development right now targets the **native build only** — Linux/Mac/Windows, linked into an application. A WASM build (and eventually a WASM compute layer) is a real future direction, not a current deliverable, and isn't being built yet. What *is* happening now is making sure nothing in the native design forecloses it: I/O and compute both sit behind trait boundaries from day one (storage backend, scripting backend, math backends), with no filesystem, threading, or dependency assumptions baked into the core crates that would block a future `wasm32` target. The cost of this discipline now is low; retrofitting it later would not be. WASM matters to us specifically because the two hardest WASM pieces — [`blas.wasm`](https://github.com/andy-emerson/blas.wasm) and `lua.wasm` — already exist, authored by the same author, with a LAPACK-in-WASM layer as their next milestone.

### Design philosophy

Every architectural choice follows one rule: **take mature, narrow, well-tested dependencies as-is where they exist; write only the part that's actually novel.**

- **Taken as-is, no modification:** `sqlparser-rs` (SQL parsing), LuaJIT (scripting), native BLAS/LAPACK (OpenBLAS/MKL/Accelerate). Stable, narrow, embedding-oriented libraries — linking them whole is safe because their entire purpose is being called into by a host program.
- **Used as a differential correctness oracle, not vendored:** DuckDB and/or DataFusion. For the portion of SQL semantics that overlaps standard behavior (aggregates, joins, window functions), we run the same query against an oracle and diff the output. We do **not** vendor DataFusion's executor — its useful parts are coupled to its own general-purpose planner, and extracting a piece drags the planner in. We write our own small, scoped executor and check it against DuckDB/DataFusion's *output*.
- **Original, unoracled work:** the storage/compaction layer and the numeric-or-key schema invariant. No existing project enforces this specific rule or this specific append/ordered/tombstone-correction design — validated by our own test suite, not a diff against someone else's behavior.

### Numbers: `f64` and `i64`, with roles

"Numeric" is not monolithically `f64`. A time-series/ledger store's most important column is usually its *ordering key*, and epoch **nanoseconds don't fit in `f64`** — `f64` has 53 bits of integer precision (exact integers to ~9.0×10¹⁵), while epoch-nanos are already ~1.8×10¹⁸, so `f64` timestamps silently cap at microsecond precision. So numeric columns come in two flavors with distinct roles:

- **`i64` (and fixed-point decimal over `i64`) — the exact / stored / fact type.** Nanosecond timestamps, money as scaled integers, volumes, counts. Exact, bit-for-bit reproducible, and — bonus — ordered `i64` columns are exactly what delta / delta-of-delta compression is built for.
- **`f64` — the analytic / derived type.** Anything BLAS/LAPACK touches. Regression coefficients, covariance eigenvalues, correlations, portfolio weights are *irrational in general*, so the analytics layer is inherently floating-point; this is also what keeps NumPy interop and the DuckDB oracle strategy working.

The schema declares which flavor each numeric column is. We considered and **rejected** making the numeric type a rational (`i64/i64`) and writing our own integer linear algebra: rational denominators overflow `i64` within a handful of divisions (a mean of ~4 returns already blows past the ceiling), a bignum rational is variable-width and kills the fixed-width Arrow-interop and SIMD story, and — decisively — rationals can't even *represent* the irrational outputs (√, log, eigenvalues) the analytics produce. Floating-point *done carefully* is the right tool; where reproducibility matters, we address it the way BLAS does (see below), not by abandoning floats.

### What "numeric-or-key" means at the engine level

This isn't a naming convention — it's enforced in the type system. A column is either:
- **Numeric** (`f64` or `i64`): usable in arithmetic, aggregation, comparison, and — for `f64` — passed directly into BLAS/LAPACK/Lua as raw numeric buffers.
- **Key**: dictionary-encoded to an integer at ingest (string interning, similar to kdb+'s symbol type or Arrow's dictionary encoding), usable in equality/grouping/joins and string *predicates*, never in arithmetic.

There is no third column type. A column that can't be classified as one or the other is rejected at schema-definition time, not silently coerced — and this holds for query results and intermediates, not just stored columns.

### Storage, ordering, corrections, and UPDATE/DELETE

Storage is columnar, partitioned on the declared ordering key, and immutable once flushed — segments are never rewritten in place. Zone maps (min/max per column per segment) exploit ordered ingest to prune segments at query time; delta/delta-of-delta compression exploits it to shrink ordered numeric columns. **This is why "ordered" is load-bearing and "time" is not:** without *some* clustering key the data arrives roughly sorted on, both pruning and compression collapse — you keep columnar *layout* but lose columnar-*fast-at-scale*.

All mutation — an out-of-order correction, a SQL `UPDATE`, a SQL `DELETE` — goes through one mechanism: **tombstone + reinsert.** The old row is marked deleted, a corrected row is appended fresh if there is one, and background compaction resolves tombstones and merges segments. This means:

- No MVCC, no row versioning, no general in-place update engine — one mutation primitive, reused everywhere.
- Optimized for the common case (in-order append), correct-but-unoptimized for the rest (corrections, `UPDATE`, `DELETE`).
- Query-time reads resolve "newest version wins" for any tombstoned row — a small, well-understood cost, paid only when mutation has actually happened.

> **Open design decision (tracked in issues):** the precise *row-identity* rule that makes "newest version wins" well-defined — an InfluxDB-style `(key-set, ordering-key)` primary key with overwrite-on-collision, versus a kdb+-style pure-append model with an internal row id and predicate-scoped deletes — is not yet settled. The two produce different tombstone records, so it's decided before `storage-lite`'s format is fixed.

### Compute backends: BLAS *and* LAPACK, split honestly

Compute sits behind trait boundaries the `engine` calls through, so native implementations can eventually be joined by WASM ones without changing anything above. The compute layer is **two crates**, following the real library boundary — BLAS (multiplication-class) and LAPACK (solves/decompositions) are different libraries, have different consumers, and reach WASM on different timelines:

- **`compute-blas`** — the multiplication-class primitives (dot, matrix-vector, matrix-matrix). Consumed directly by the executor's window/numeric inner loops and by Lua over FFI. Native backend: OpenBLAS BLAS. WASM backend (future): `blas.wasm`, which already exists.
- **`compute-lapack`** — a *curated* set of analytical routines, each justified by a named workflow: least-squares solve (rolling regression), symmetric eigendecomposition (covariance/PCA), general linear solve (portfolio weights/factor models), Cholesky (positive-definite covariance fast path). Native backend: LAPACK. WASM backend (future): a LAPACK-in-WASM layer that does not yet exist and is the next milestone of the `blas.wasm` project.

Because the two backends arrive on different timelines, each op is exposed through a trait that supports **per-backend capability negotiation** — "this operation is unavailable on this backend" is a first-class, queryable answer, not a panic. A future WASM build lands with storage + query + Lua + BLAS-class ops working and the LAPACK-class analytics gracefully degraded until LAPACK-in-WASM ships.

The bar for adding a routine to `compute-lapack` is a real, specific, repeated workflow need — not "LAPACK has it." No autodiff, no general tensor framework, no "as much of LAPACK as we can wrap."

### Numerical consistency

Native and WASM builds won't be bit-identical by default — floating-point addition isn't associative, and different SIMD widths / FMA usage change summation order. We're not solving full native/WASM bit-identity now; that's future work once a WASM build exists. But it's cheap to build toward: `blas.wasm` already defers fused-FMA variants specifically to preserve determinism, and native OpenBLAS built from source with `TARGET=SANDYBRIDGE` forces pre-FMA kernels (AVX, no FMA) while staying fast on essentially any x86_64 CPU from 2011 onward — meaningfully faster than the more conservative `TARGET=NEHALEM` (SSE-only) for the same determinism guarantee. It's a build-time `TARGET=` decision, not a switch, and a known low-effort one to make when it's actually needed.

LuaJIT and native BLAS/LAPACK are called directly from Lua via LuaJIT's FFI — Lua declares a C function's signature and calls straight into a linked library with near-zero overhead, no hand-written binding layer, numeric arrays passed as raw pointers into the same memory the query engine already holds. This is how the original Torch (pre-PyTorch) worked: LuaJIT + BLAS-backed tensors over FFI. TallyDB's `compute-lua`/`compute-blas`/`compute-lapack` crates use the same mechanism, scoped to a curated set of operations rather than a general tensor/autodiff library.

TallyDB's embedded Lua supports pure-Lua libraries (plain `.lua` source) out of the box. Compiled C extensions (LuaRocks packages with a `.so`/`.dll`, loaded via `package.loadlib`) are not supported: real attack-surface and stability cost for an embedded database process, cutting against the curated-not-general instinct. This is also structurally true for the WASM backend regardless of policy — WASM's sandbox can't do `dlopen`-style loading — so the two constraints reinforce each other.

### Workspace layout

TallyDB is a single Cargo workspace — each crate has a clean boundary and can be reasoned about and tested in isolation, but they share one version history and one build.

```
tallydb/
  crates/
    arrow-lite/     # Arrow-layout-compatible columnar in-memory format
                    #   (f64 + i64 numeric buffers, key/dictionary columns)
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

---

## Status

Early-stage design and planning; the storage/query/engine/compute crates are scaffolded (documented boundaries, not yet implemented). `blas.wasm` and `lua.wasm` (the WASM compute dependencies, for later) are real, working, MIT-licensed projects already in progress by the same author, with LAPACK-in-WASM as their next planned milestone — tracked as future dependencies, not part of the current native-first build. See the repository issues for open design decisions still to be resolved.
