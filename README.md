# TallyDB

**A small, embeddable, SQL-native database for numeric time-series data.**

TallyDB is an HTAP-shaped database: fast, append-heavy ingest (the OLTP-like half) feeding directly into ordered, numeric, analytical reads (the OLAP-like half), with no ETL step between them. It's built around three assumptions about the data it stores:

1. **Append-optimized.** Data arrives as new rows, not updates — though corrections are supported (see below), just not the design center.
2. **Ordered.** Rows arrive roughly in time order.
3. **Numeric-or-key.** Every column is either a number (used in arithmetic, aggregation, windows) or a key (an identifier or label, used only for filtering and grouping — never arithmetic).

These three assumptions aren't restrictions bolted on after the fact — they're the whole design. Relaxing any one of them is what makes general-purpose databases (Postgres, DuckDB, SQLite) bigger, slower to start, and harder to embed. Holding all three lets TallyDB stay small, fast, and honest about what it's for.

---

## For users: what TallyDB is and isn't

**What it's for.** TallyDB is aimed at quantitative research and similar workloads: tick data, sensor feeds, time-ordered numeric records that get analyzed with SQL — rolling aggregates, joins against small reference tables, grouping, window functions. If your data looks like a big append-heavy ledger of numbers with some labels attached, TallyDB is built for exactly that shape.

**What it's not for.** It is not a general-purpose relational database. There's no support for arbitrary text columns or blobs, and no general multi-table joins outside a star-schema shape. If your data doesn't fit the three assumptions above, use Postgres, DuckDB, or SQLite instead — they're better at being general.

**The SQL surface.** TallyDB supports standard SQL over its schema: `SELECT`/`WHERE`/`GROUP BY`/`ORDER BY`, equi-joins, window functions, and — yes — `UPDATE`/`DELETE`. Under the hood, both are implemented as tombstone-plus-reinsert against immutable, append-only storage (the same mechanism handles ordinary data corrections), resolved at the next compaction rather than in place. They aren't the fast path, and the engine isn't optimized for frequent use of them — but they're real, correct, and available, because withholding a SQL verb that the storage engine already supports under a different name would just push the same work into application code. More generally: **any standard SQL function or verb is in scope as long as it doesn't violate the numeric-or-key invariant and doesn't require a general-purpose query optimizer or type system.** We don't require ourselves to imagine a specific quant-research use case before including something — that's a poor filter, since real usage regularly surprises the people who built the tool. The invariants are the actual boundary, not our own foresight.

**How you'd use it.**
- Link it into your application like SQLite or DuckDB — no server process, no separate database to administer.
- Query results come back in a layout that's directly usable by NumPy or similar numeric tooling — no conversion step.
- For anything the built-in SQL functions don't cover, drop into embedded Lua — called directly from SQL, operating on the same numeric buffers the query engine already has in memory. Nothing gets copied out to a separate scripting process or serialized across a boundary; the script, the query engine, and the curated BLAS/LAPACK ops all read and write the same data in place. This "compute without copying" property is one of the main things TallyDB is built around, not a bolted-on extra.
- Runs natively (Linux/Mac/Windows) for production and research pipelines.

**Why it exists.** Teams below the scale that justifies kdb+'s cost and learning curve currently choose between building their own infrastructure on pandas/DuckDB (accepting real friction between "database" and "numeric compute") or paying for tooling built for a much larger workload than they have. TallyDB is built for the gap in between: SQL-native (not a bespoke language like kdb+'s q), embeddable (not a server you have to run), and structured around the exact shape quantitative time-series data actually has.

**Prior art.** TallyDB's design borrows validated ideas from existing systems rather than inventing all of them from scratch:
- **InfluxDB** validates the core key/numeric split directly — its tags-vs-fields model is close to identical in spirit to TallyDB's numeric-or-key rule, and its more recent versions moving to real SQL (dropping the older Flux language) validates SQL-native as the right query surface for this kind of data. InfluxDB itself isn't minimal or embeddable (it's a distributed server built on Arrow/DataFusion/Parquet-on-object-storage), which is exactly the gap TallyDB is built to fill.
- **kdb+** validates both the workload (25+ years as the standard in quant finance) and the "keys as interned integers, keep everything else numeric" performance pattern — but it's proprietary, licensed, and built around q rather than SQL, which is the specific combination TallyDB doesn't replicate.

---

## For developers: how TallyDB is built

### Current focus: native-first, WASM-ready

Development right now targets the **native build only** — Linux/Mac/Windows, linked into an application like SQLite or DuckDB. A WASM build (and eventually a WASM-native compute layer) is a real future direction, not a current deliverable, and isn't being built yet. What *is* happening now is making sure nothing in the native design forecloses it later: I/O and compute both sit behind trait boundaries from day one (storage backend, scripting backend, math backend), with no filesystem, threading, or dependency assumptions baked into the core crates that would block a future `wasm32` target. The cost of this discipline now is low; the cost of not doing it and having to retrofit it later would not be.

### Design philosophy

Every architectural choice in TallyDB follows the same rule: **take mature, narrow, well-tested dependencies as-is where they exist; write only the part that's actually novel.** Concretely:

- **Taken as-is, no modification:** `sqlparser-rs` (SQL parsing), LuaJIT (scripting), native BLAS/LAPACK (OpenBLAS/MKL/Accelerate). These are stable, narrow, embedding-oriented libraries — linking them whole is safe because their entire design purpose is being called into by a host program.
- **Used as a differential correctness oracle, not vendored:** DuckDB and/or DataFusion. For the portion of SQL semantics that overlaps standard behavior (aggregates, joins, window functions), we run the same query against an oracle and diff the output — the same validation strategy `blas.wasm` uses against reference BLAS, and `lua.wasm` uses against Lua's official test suite. We do **not** vendor DataFusion's executor: its useful parts are coupled to its own general-purpose planner, and extracting a piece drags the planner's scaffolding in. Writing our own small, scoped executor and checking it against DuckDB/DataFusion's *output* gets the correctness benefit without the generality cost.
- **Original, unoracled work:** the storage/compaction layer and the numeric-or-key schema invariant. No existing project enforces this specific rule or this specific append/ordered/tombstone-correction design — this is genuinely new design, validated by our own test suite, not a diff against someone else's behavior.

### Workspace layout

TallyDB is a single Cargo workspace, not a monolith and not separate repos — each crate has a clean boundary and can be reasoned about (and tested) in isolation, but they share one version history and one build.

```
tallydb/
  crates/
    arrow-lite/    # Arrow-layout-compatible columnar in-memory format
    storage-lite/  # append-optimized, ordered, time-partitioned segments;
                   #   compaction; zone maps; native backend = mmap today,
                   #   behind a trait so an OPFS/WASM backend can be added later
    query-lite/    # scoped SQL parser (via sqlparser-rs) + our own executor;
                   #   validated against DuckDB/DataFusion as an oracle
    engine/        # ties storage + query together; enforces numeric-or-key
                   #   as a hard schema rule
    compute-lua/   # Lua scripting behind a trait; LuaJIT for now — a WASM
                   #   backend (lua.wasm) can slot in later without a rewrite
    compute-blas/  # curated BLAS/LAPACK ops behind the same trait; native
                   #   BLAS/LAPACK for now — WASM backends (blas.wasm,
                   #   LAPACK-wasm) are a future swap-in
  Cargo.toml
```

### What "numeric-or-key" means at the engine level

This isn't a naming convention — it's enforced in the type system. A column is either:
- **Numeric** (`f64` by default): usable in arithmetic, aggregation, comparison, and passed directly into BLAS/LAPACK/Lua as raw numeric buffers.
- **Key**: dictionary-encoded to an integer at ingest (string interning, similar to kdb+'s symbol type or Arrow's dictionary encoding), usable in equality/grouping/joins, never in arithmetic.

There is no third column type. This is a hard schema constraint, not a convention — a column that can't be classified as one or the other is rejected at schema-definition time, not silently coerced.

### Storage, corrections, and UPDATE/DELETE

Storage is columnar, time-partitioned, and immutable once flushed — segments are never rewritten in place. All mutation — an out-of-order correction, a SQL `UPDATE`, a SQL `DELETE` — goes through the same mechanism: **tombstone + reinsert**. The old row is marked deleted, a corrected row is appended fresh if there is one, and a background compaction pass resolves tombstones and merges segments. This means:

- No MVCC, no row versioning, no general in-place update engine — one mutation primitive, reused for every case that needs it.
- The engine is optimized for the common case (in-order append) and correct-but-unoptimized for the rest (corrections, `UPDATE`, `DELETE`) — all fully supported, none of them the fast path.
- Query-time reads resolve "newest version wins" for any tombstoned key — a small, well-understood cost, paid only when mutation has actually happened.

### Compute backends and numerical consistency

Compute (Lua scripting, BLAS/LAPACK) sits behind a trait the `engine` calls through, so the native implementation (LuaJIT + native BLAS/LAPACK, linked as-is) can eventually be joined by a WASM implementation (`lua.wasm`, `blas.wasm`, and a future LAPACK-in-WASM layer) without changing anything above that boundary.

Native and WASM builds won't produce bit-identical results by default — floating-point addition isn't associative, and different SIMD widths/FMA usage change summation order. We're not solving full native/WASM bit-identity now; that's future work once a WASM build actually exists. But it's worth building toward it cheaply where possible: `blas.wasm` already defers fused-FMA variants specifically to preserve determinism, and native BLAS builds can match that today at low cost — OpenBLAS built from source with `TARGET=SANDYBRIDGE` forces it onto pre-FMA kernels (AVX, no FMA) while still being safe on essentially any x86_64 CPU from 2011 onward, meaningfully faster than the more conservative `TARGET=NEHALEM` (SSE-only) option for the same determinism guarantee. There's no off-the-shelf "non-FMA" package — this is a build-time `TARGET=` decision, not a switch — but it's a known, low-effort one, not new engineering.

LuaJIT and native BLAS/LAPACK aren't just linked side by side — they're called directly from Lua via LuaJIT's FFI, which lets Lua declare a C function's signature and call straight into a linked library with near-zero overhead, no hand-written binding layer, and numeric arrays passed as raw pointers into the same memory the query engine already holds. This isn't a novel combination: the original Torch (pre-PyTorch) was built the same way — LuaJIT plus BLAS-backed tensors over FFI — for years before PyTorch replaced it. TallyDB's `compute-lua`/`compute-blas` crates use the same mechanism, scoped to a curated set of operations rather than a general tensor/autodiff library.

TallyDB's embedded Lua supports pure-Lua libraries (plain `.lua` source, no compiled component) out of the box — they run as ordinary Lua code with no extra integration work. Compiled C extensions (LuaRocks packages with a `.so`/`.dll` component) are not supported: allowing arbitrary compiled code to load inside an embedded database process is a real attack-surface and stability tradeoff, and it cuts against the same curated-not-general instinct behind everything else in this design. This also isn't a native-only restriction that WASM happens to share — WASM's sandboxed execution model structurally can't do `dlopen`-style dynamic loading regardless of policy, so the two constraints reinforce each other rather than being separate decisions.

---

## Status

Early-stage design and planning; storage/query/engine crates not yet built. `blas.wasm` and `lua.wasm` (the WASM compute dependencies, for later) are real, working, MIT-licensed projects already in progress by the same author, with LAPACK-in-WASM as their next planned milestone — tracked as future dependencies, not part of the current native-first build.
