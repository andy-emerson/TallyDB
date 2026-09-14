# DESIGN.md

What TallyDB is, what it refuses, and how it is put together. This is the
developer's companion to `README.md`, which describes the project from the
user's side.

It is deliberately short. *Why* each thing is the way it is — every settled
ruling, its rejected alternatives, and what would reopen it — lives in
[DECISIONS.md](DECISIONS.md). How we work — passes, reviews, issues,
integration — is [AGENTS.md](AGENTS.md), and the craft conventions are
[CONTRIBUTING.md](CONTRIBUTING.md).

## Scope & non-goals

An **append-ordered numeric store**: embeddable, SQL-native, with numeric
compute running *inside* the engine on its own buffers, zero-copy.
Time-series, sensor, and quant work are **use cases**, not the definition —
what is load-bearing is *ordered ingest on a key*, not that the key means
"time". The one-line frame: an open, SQL-native, embeddable kdb+ for teams
below kdb+ scale.

The differentiator is the packaging — embeddable, plus compute-fusion over
off-the-shelf libraries. Not "it only holds numbers" (table stakes for any
TSDB) and not "compute inside the DB" (kdb+ already does that). Scope must
not drift toward looking like a general database or a general TSDB; the
three assumptions below are the moat.

**In scope.** Broad standard SQL over the schema below; window functions and
the curated statistics; joins whose strategy a structural fact fixes;
corrections on a knowledge axis; maintained views; numeric compute in the
engine and in embedded Lua; one machine, one process, one writer per table.

**Not in scope, and not by omission.** A third column type. String
production. A cost-based optimizer. A network listener. Distribution across
machines. Each is refused because an assumption forces it, and each refusal
has a record in DECISIONS.md rather than being left to look like an oversight.

### The three assumptions

Do not relax these to unblock a feature.

1. **Append-optimized.** Writes are cheap, low-latency, one row at a time.
   The fast path is *append*, not in-place update.
2. **Ordered.** Data arrives roughly sorted on a declared **ordering key** —
   a timestamp is the common case, but any key monotonic on ingest works: a
   sequence id, an event id, a ledger offset. Storage is partitioned on it.
   "Ordered" is what makes zone-map pruning and delta compression work;
   "time" is not, so never hardcode a timestamp where the declared ordering
   key belongs.
3. **Numeric-or-key.** Every column is numeric (`f64` or `i64`) or a
   dictionary-encoded key. No third type, ever. This holds across the whole
   pipeline — stored columns, intermediates, and query results. If a feature
   seems to need a third type, the feature is wrong, not the invariant.

These are not restrictions bolted on afterward — they are the whole design.
Relaxing any one is what makes general-purpose databases bigger, slower to
start, and harder to embed; holding all three is what makes fixed-width
columns you can hand straight to a math library possible.

### Numbers have roles

"Numeric" is not monolithically `f64`, because epoch nanoseconds do not fit
in one: `f64` holds exact integers to ~9.0×10¹⁵ and epoch-nanos are already
~1.8×10¹⁸, so `f64` timestamps silently cap at microsecond precision. The
two flavors carry distinct roles, declared per column in the schema:

- **`i64` — the exact, stored, fact type.** Nanosecond timestamps, money as
  scaled integers, volumes, counts. Bit-for-bit reproducible, and ordered
  `i64` columns are exactly what delta-of-delta compression is built for.
- **`f64` — the analytic, derived type.** Anything the numeric ops touch.
  Regression coefficients, eigenvalues, correlations and portfolio weights
  are irrational in general, so the analytics layer is inherently
  floating-point — which is also what keeps NumPy interop and the DuckDB
  oracle strategy working.

### What numeric-or-key means at the engine level

Enforced in the type system, not a naming convention. A column is either
**numeric** — usable in arithmetic, aggregation and comparison, and passed
directly into the numeric ops and Lua as raw buffers — or a **key**,
dictionary-encoded to an integer at ingest, usable in equality, grouping,
joins and string *predicates*, never in arithmetic. A column that is neither
is rejected at schema-definition time, not silently coerced.

A key is a *label*, not a primary key: repeating values are the point. For
readers arriving from BI/OLAP, key columns play the *dimension* role and
numeric columns are the *measures*.

### Null and NaN

NULL is absence: it matches no comparison, aggregates skip it, arithmetic
propagates it, and in ordering it is not compared but **placed**, after all
values, in both directions. NaN is a value — computed, and comparable under
one relation shared by sorting, predicates, `MIN`/`MAX` and zone-map pruning
alike: greater than every number and equal to itself, while `-0.0 = 0.0`
stays true. The ascending ladder is *numbers… +∞, NaN, then NULL off the
end*. Pruning stays sound via a has-NaN bit in the `f64` zone map.

## The inclusion principle

One principle governs both the SQL and Lua surfaces: **a capability is in
scope by default, and excluded only where it would violate a named
invariant.** We do not hand-pick a feature list, and we do not require a use
case to admit something — "we can't think of a quant use case for it" is
explicitly **not** a valid reason to exclude something otherwise in scope.
The invariants are the boundary, not our imagination.

**SQL is bounded by** (a) numeric-or-key and (b) no general-purpose
cost-based optimizer. **Lua is bounded by** (a) the sandbox — no filesystem,
process, network, native code, or escape hazard — (b) determinism unless the
author opts out, and (c) the same numeric-or-key rule on what may cross into
the engine.

| Lua stdlib | In / Out | Bounding invariant |
|---|---|---|
| `math`, `table`, `string`, `utf8`, curated `base` | **in** | — |
| `math.random` / `randomseed` | **in**, documented | (b) with opt-out — forfeits reproducibility |
| `io`, `os` | **out** | (a): filesystem / process |
| `debug`, raw metatable & `raw*` functions | **out** | (a): sandbox escape (shared metatables) |
| `load` / `loadstring` / `loadfile` / `dofile` | **out** | (a): code injection / memory safety |
| `package.loadlib`, native `require` | **out** | (a): native code (`package` curated to a pure-Lua searcher) |
| `coroutine` | **out** (deferred) | *not an invariant* — it fights the `pcall`/`longjmp` discipline, and the principle admits it once a binding can host it |

**The moat test.** The inclusion principle is a negative filter: it says what
is *admissible*, and cannot order the backlog. The companion positive filter
does: **build first the things the three assumptions make cheaper for this
engine than for a general database.** Of each admissible-but-unbuilt
candidate, ask *"does DuckDB have to work harder than us here?"* If yes,
building it cashes a dividend the cuts already paid for. If no, it is
generality wearing a feature's clothes.

## The axes: cuts, refusals, and reversal classes

A specialized engine's deletions are **forced by positive workload
assumptions**, never selected from a menu; each cut's **reversal class** sets
how much scrutiny its licensing assumption deserves, because the assumption
is what fails, not the cut. Every absence has a name here, so a reader can
tell refusal from oversight.

| Axis | Our position | Licensing assumption | Reversal class |
|---|---|---|---|
| Mutation | Cut to the endpoint: append-only storage, tombstone+reinsert | Data is appended, not revised (assumption 1) | Foundational |
| Working set | Cut: the *queried working set* fits in memory — the table need not. Opens read manifest metadata only; the executor prunes before any decode; segments fault in on first touch under a byte budget | The rows a query touches fit in memory | Foundational |
| Query (planner) | Cut: a **fixed-strategy planner** — `plan()` exists; search, costing and choice do not | One access path ⇒ nothing to choose between | Additive |
| Query (surface) | **Refused**: broad standard SQL, bounded by the inclusion principle | — | Additive |
| Access path | Cut totally: no secondary indexes; ordering-key clustering + scan is the one path (zone maps are pruning metadata, not a path) | Ordered ingest on the declared key (assumption 2) | Invasive |
| Write | **Refused**: cheap online single-row append is the design center | — | n/a |
| Transaction | Cut: submitted units only — no `BEGIN`/`COMMIT`/`ROLLBACK`, no session state | Work arrives as single statements | Invasive |
| Isolation | Fixed: snapshot isolation at statement granularity | One guarantee suffices | Invasive |
| Concurrency | Single writer *per table*, concurrent snapshot readers; writers scale by table. Cross-process readers use writer-exclusive / reader-shared locks, POSIX-first — `tallydb DIR --read-only`, with `.refresh` and `.flush`, is the console half | One writer per table is enough | Additive to correct |
| Distribution | Cut totally: one machine — and **compute-without-copying exists only because no network boundary exists anywhere** | Data and load fit one node | Foundational |
| Deployment | Cut: library, never a server. Live in-process ingest+compute is in; networked subscriber fan-out is out | One application owns the data | Additive |
| Schema | The hardest cut: numeric-or-key, enforced in the type system (assumption 3) | Every column is a number or a label | Foundational |
| Durability | Not cut: publish is atomic **and synced**, and the write buffer sits behind a sidecar WAL with sync levels | — | — |

**Transaction contract.** Work reaches the engine only as submitted units —
one `append`, one `query`, one `mutate`. There is no interactive transaction,
no session state, and no verb for either; adding a session object would be a
*reversal on the transaction axis* and must be treated as one.

**Isolation contract.** A read sees a snapshot taken at statement start:
`Store::snapshot` returns owned segment views, and appends or mutations after
the call are invisible to it (test:
`snapshot_is_isolated_from_later_appends`). One guarantee — snapshot isolation
at statement granularity — no isolation-level menu. A statement reading more than one
table takes multiple `snapshot()` calls at distinct instants; through a
`Database` handle this is sound, because the writer needs `&mut` on the same
handle. Detached reader handles are single-table by scope, so no shipped path
can take a cross-table torn read, and no cross-table snapshot epoch is
promised.

**Batch, not per-row.** Every call from the query executor into the Lua or
linear-algebra layers operates on a whole column or window per call, never
element-by-element. Per-row calls throw away the entire performance rationale
for pairing a columnar engine with these compute layers. If an API makes
per-row calls the easy way to use it, that is a bug in the API shape.

## The SQL surface

One table, so the in/out line is a deliberate record rather than an
accumulation of implementation accidents. Every IN row is born with a DuckDB
differential family; the console's help cites this table.

| Construct | Status | Note |
|---|---|---|
| `SELECT` projection, aliases | in, built | |
| `WHERE` (numeric compares, key `=`/`IN`/`LIKE`, `AND`/`OR`/`NOT`) | in, built | NaN-aware; zone-map pruning; LIKE per distinct value |
| `WHERE` comparing two expressions (`x > y`, `x * 2 > y + 1`) | in, built (#95) | a zone map knows a column's range, not an expression's over several, so this leaf **prunes nothing**. Pruning degrades **per conjunct** — `WHERE ts > 1000 AND x > y` still skips segments on `ts`. `40 < x` is mirrored to `x > 40` before lowering. A registered kernel is refused here by name |
| regex on keys | deferred by ruling | menu incl. the Lua-pattern house option on #57 |
| `IS NULL` / `IS NOT NULL` | in, built | one predicate arm over the validity bitmap; the only *total* leaf (never UNKNOWN); `IS NOT NULL` prunes an all-null segment |
| `GROUP BY` + `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` | in, built | exact-loud `SUM(i64)` |
| `GROUP BY` monotone ordering-key arithmetic (`ts / 60`) | in, built | bucket index, `(ts / 60) * 60` for the bucket start, bare `ts` for the finest bucket. `/` truncates (ISO); `//` accepted (DuckDB's spelling). May be named by its SELECT alias. Over ordered data the grouping **streams** — accumulator state is the open bucket, not the result; unordered data falls back to hashing with the same answers, and `compact()` restores the fast path |
| `FIRST` / `LAST` aggregates | in, built | positional on the **time axis**, not row order, so a late-arriving row cannot become "last". Ties on the clock go to the last row in storage order; nulls skipped. Also available as window functions, where — being positional — they **refuse an unordered window** |
| cross-sectional `PARTITION BY` (the ordering key, or a bucket of it) | in, built | the transpose of `PARTITION BY sym` — one partition per instant, across every symbol. Unordered windows take the whole partition, per standard SQL; several terms intersect (`sym, ts / 60` = per symbol per bar); any `BIGINT` partitions, `DOUBLE` never |
| scalar expressions over window results | in, built | `x / sum(x) OVER (PARTITION BY ts)`, `x - lag(x) OVER (ORDER BY ts)`, the rolling z-score. Window calls hoist out of the scalar and compute first — standard SQL's evaluation order |
| `HAVING` | in, built | hidden-column lowering; WHERE grammar over the group row |
| `DISTINCT` | in, built | by value; NaN=NaN, −0=0, NULLs equal; `DISTINCT ON` out |
| scalar expressions (`+ − * / %`, `ABS ROUND FLOOR CEIL SQRT LN EXP POWER`) | in, built | f64, three-valued; IEEE division — NaN is a value; i64 refused loudly (#40) |
| `CASE WHEN` | in, built | conditions are WHERE grammar; UNKNOWN falls through |
| `ORDER BY` one column, `NULLS FIRST/LAST` | in, built | default nulls-last both directions; refused on symbol columns (unordered labels); bounded by `LIMIT` it runs top-k, O(k) memory |
| multi-column `ORDER BY` | in, later | additive lowering |
| `LIMIT`/`OFFSET` | in, built | |
| window functions over `ROWS` frames | in, built | curated + Lua kernels; incremental sweep |
| `var_pop` / `stddev_pop` as windows | in, built | one column; variance *is* self-covariance, so they share `covar_pop`'s corrected two-pass and incremental sweep. Population forms only, matching the `_pop` family; sample forms and the group-level surface are additive and unbuilt |
| `regr_r2` as a window | in, built | the squared correlation of the simple fit, `covar² / (var_x·var_y)`. Undefined — SQL NULL, not a fabricated 1.0 — where either column is flat, and clamped to [0, 1] |
| `LAG` / `LEAD` | in, built | positional, not aggregates: they copy a neighbouring row, so the output keeps the **source column's type** — a lagged `BIGINT` stays `BIGINT`, because a nanosecond stamp is past 2^53. Frameless; optional offset defaults to 1; the third `default` argument and symbol columns are refused by name |
| `RANGE` frames | in, built | bounded by ordering-key **value**, in the key's own units (no `INTERVAL` type: five minutes over ns stamps is `300000000000`). Ends at the current row's **last peer**, per standard SQL. Answers via per-frame recompute today |
| star-schema equi-joins (`INNER`/`LEFT`) | in, built | structural-fact rule; gathers only the dimension columns the query reads |
| `ASOF LEFT` / `ASOF INNER JOIN` | in, built | each fact row takes the most recent of its key's dimension rows on the two **declared ordering keys**; an explicit inequality is validated, not obeyed. Ties on the dimension's clock go to the last row in storage order. The dimension side is indexed in memory, not co-walked |
| ordered-merge relatives beyond `ASOF` (`LT`, `SPLICE`, …) | in, later | nothing coined; the same lift mechanism serves them |
| `UPDATE`/`DELETE` | in, built | tombstone + reinsert |
| DDL (`CREATE TABLE`), `INSERT`, bulk import | in, built | `BIGINT`/`DOUBLE`/`SYMBOL`, `ORDERING KEY` constraint; `VARCHAR`, `PRIMARY KEY` and the retired `KEY` spelling refused with teaching errors |
| non-correlated subqueries / CTEs | in, later | named subplans |
| `UNION ALL` (then `UNION`) | in, later | low priority |
| correlated subqueries | **out** | the road to a cost-based optimizer — settled no |
| string production (`CONCAT`, `CAST AS VARCHAR`, …) | **out** | numeric-or-key invariant |
| `DISTINCT` over window/aggregate projections | out until asked | refused loudly today |

Any standard SQL function or verb is in scope as long as it needs neither a
third column type nor a cost-based optimizer.

## How it works

### Storage

Columnar, partitioned on the declared ordering key, and immutable once
flushed — segments are never rewritten in place. Zone maps (min/max per
column per segment) exploit ordered ingest to prune segments at query time;
delta-of-delta compression exploits it to shrink ordered numeric columns.
**This is why "ordered" is load-bearing and "time" is not:** without some
clustering key the data arrives roughly sorted on, both pruning and
compression collapse — you keep columnar *layout* but lose
columnar-*fast-at-scale*.

All mutation — an out-of-order correction, a SQL `UPDATE`, a SQL `DELETE` —
goes through one mechanism: **tombstone + reinsert.** The old row is marked
deleted, a corrected row is appended fresh if there is one, and background
compaction resolves tombstones and merges segments. So there is no MVCC, no
row versioning, and no general in-place update engine: one mutation
primitive, reused everywhere, optimized for the common case and
correct-but-unoptimized for the rest.

A table larger than memory is served by segment-granular lazy residency: an
open reads manifest metadata only, segments decode on first data access, and
decoded segments are retained under a byte budget, evicted least-recently-
used. The budget is **advisory over retention, never over correctness** — a
segment an `Arc` still pins is never evicted.

### Query

A scoped SQL parser over `sqlparser-rs` plus our own executor. The planner is
fixed-strategy: it lowers, it does not search. A join is supported when a
structural property of its inputs fixes the execution strategy — one side
small enough to materialize and key-unique, or both sides ordered on the join
key — never by cost. A join with neither guarding fact is **refused loudly,
naming the missing structure**, rather than served slowly.

Results leave as Arrow record batches through the C Data Interface, including
`ArrowArrayStream`. Passthrough columns share the stored buffers.

### Corrections and maintained views

`UPDATE` and `DELETE` each spend exactly one coordinate on a hidden
**ingest-sequence axis**, which is the engine's knowledge-time dimension:
`ASOF s` reads a table as it was known at a coordinate, and `_seq` reads a
row's own coordinate back through SQL.

A **maintained view** is a fold over that sequence, stamped with the source
watermark below which the materialization is complete. Reads are a union —
materialized clean buckets plus a live fold of everything the stamp does not
cover — so a view is semantically always exact, and refresh only shrinks the
live half. Bucketed, running, cumulative and join shapes are built.

### Compute

Compute sits behind trait boundaries the engine calls through, so native
implementations can be joined by WASM ones without changing anything above.

- **`compute_linalg`** — multiplication-class primitives (dot,
  matrix–vector, matrix–matrix) behind `LinalgBackend`. Pure Rust: a
  source-fixed loop for `dot`, faer for the matrix products, so one
  implementation serves native and `wasm32`.
- **`compute_lua`** — the scripting tier: canonical PUC Lua 5.4 compiled from
  unmodified upstream sources, behind hand-rolled thin bindings.

The curated statistics — `regr_slope`, `regr_intercept`, `regr_r2`,
`covar_pop`, `corr`, `var_pop`, `stddev_pop`, `eigen_max` — are solved in
closed form, with no LAPACK linked at all, and evaluate incrementally over
the engine's own buffers. Above two parameters there is no closed form, so
the rolling multi-factor fit maintains anchored moments across the slide and
re-solves per frame.

Scripts reach the engine's buffers through zero-copy userdata views: the
userdata wraps the live buffer pointer and its accessors are implemented on
the Rust side, so no bytes are copied. Stated precisely: *access* is
zero-copy, but each element read is a metamethod dispatch rather than a
compiled raw load. Lua 5.4's numeric model — one number type with a 64-bit
integer subtype and a 64-bit float subtype — is exactly TallyDB's `i64`/`f64`
pair, so numeric values cross without losing exactness.

**The two directions, and only two.** Lua and the engine meet in exactly two
directions, named for which language encloses the other: **Lua-in-SQL** — the
engine calls a Lua kernel mid-query (`my_kernel(x) OVER (…)`) — and
**SQL-in-Lua** — a Lua program drives the engine and runs SQL. A third, the
**data-only baseline** (staged `SQL → columns → Lua → columns → SQL`, neither
side calling the other), falls out for free from being an embeddable library.
These exhaust the in-process embed: direction is the only axis, so there is
no third role to invent.

SQL-in-Lua is the driver seam: `query(sql)` returns result columns as the
same zero-copy views kernels consume (several segments concatenate — the
bounded copy — with per-segment key dictionaries merged), and
`append(table, row)` feeds derived rows back exactly. Both globals are live
only inside a driving call (`LuaState::run_driver`, `Database::run_script`,
the console's `.run`), so a kernel can never re-enter the engine mid-query.
The Rust-side extension seam is `WindowAggregate` with `Table::register_window`
and `Database::register_window`; the `ScriptHost` trait is what a scripting
backend implements behind it.

**The idiom: compose, don't loop.** Lua's cost model has three tiers, and the
documentation teaches the discipline NumPy's culture teaches Python: (1) an
element loop written in Lua pays an interpreted dispatch per element — the
code smell; (2) a kernel that *composes registered ops* (`return 2 *
sumsq(x)`) runs compiled arithmetic with one interpreter entry per call;
(3) the engine-driven paths (`evaluate_frames`, `eval_column`) enter the
interpreter once per run or view. Performance is a **promotion ladder**, not
a JIT: write the kernel in Lua to get it correct, and promote it to a curated
native op if it proves hot.

### Numerical consistency

Native and WASM builds will not be bit-identical by default — floating-point
addition is not associative, and different SIMD widths and FMA usage change
summation order. The portability standard is set when the WASM milestone
starts. The ground is mostly already held: every closed-form window statistic
and the `dot` kernel fix their operation order in source, so their results are
bit-identical across CPUs and targets by construction. Two known holes are
tracked rather than solved — faer's matrix kernels dispatch SIMD at runtime
(today on no query path), and `eigen_max` calls `hypot`, whose last bit is
libm-implementation-specific — switching to `sqrt(a² + b²)` is a one-line fix
to make when the standard is set.

## Dependencies

Every architectural choice follows one rule: **take mature, narrow,
well-tested dependencies as-is where they exist; write only the part that is
actually novel.**

**Taken as-is** — do not fork, vendor, or reimplement: `sqlparser-rs` for SQL
parsing; **PUC Lua 5.4** for embedded scripting, the canonical upstream
sources compiled in unmodified; **faer** for pure-Rust linear-algebra kernels
(slim feature set: no thread pool, no RNG). Linking them whole is safe
because their entire purpose is being called into by a host program.

**Used as a correctness oracle, never linked at runtime**: **DuckDB**
(primary) and **DataFusion** (secondary) differentially test the executor;
**arrow-rs / PyArrow** round-trip the hand-rolled Arrow layout and C Data
Interface export. Oracle criteria are not product criteria — the oracle never
ships, so its size is irrelevant; what matters is authority on analytic-SQL
semantics and running in-process inside `cargo test`.

**Genuinely original** — no oracle exists, so our tests are the spec: the
append/ordered/compaction/tombstone design, and the numeric-or-key schema
invariant itself.

## Verification bar

### What "correct" means here

1. **Agrees with the oracle.** For SQL semantics that overlap standard
   behavior: same query, same data → same output as DuckDB (primary) /
   DataFusion (secondary). **Every oracle has a declared scope of
   authority**: an oracle checks that we compute *our chosen* semantics
   correctly — where the standard leaves a choice, the choice is ours,
   recorded in DECISIONS.md, and the harness normalizes the documented
   divergence. An oracle never chooses semantics, and a diff must never share
   the implementation's computational path.
2. **Round-trips with real Arrow.** Columns exported over the C Data
   Interface import identically in arrow-rs and PyArrow, and vice versa —
   dictionaries, nulls, and logical types intact.
3. **Deterministic where promised.** Same seeded input, same pinned compute
   backend → bit-identical segment bytes and result buffers, checked against
   committed goldens. Storage bytes are promised backend-independent; `f64`
   results are promised for the source-fixed paths. A change that moves those
   bits is a behavioral change, not a refactor.
4. **Meets its own spec** where no reference exists.

### The reference map

| Claim family | Reference | Tier |
|---|---|---|
| SQL semantics | DuckDB (primary) / DataFusion (secondary) | independent oracle |
| Arrow layout + C Data Interface | arrow-rs / PyArrow round-trips, dev-only | independent oracle |
| compute seam (curated ops and kernels) | NumPy/SciPy on the same inputs | independent oracle |
| determinism (storage bytes; pinned-backend results) | committed goldens | prior output |
| storage behavior (append, compaction, tombstones) | its own spec-tests | none — tests are the spec |

Storage occupies the weakest tier — no independent reference exists for its
behavior. That is why the build order front-loaded it and why its tests
deserve the most scrutiny.

### Peers, for measurement claims

**DuckDB** — primary peer, also the oracle and the control group: one corpus,
diffed for correctness and timed for performance, so we never benchmark a
wrong answer. **SQLite** — the floor. **The exported-workflow pipeline**
(DuckDB → pandas/NumPy) — the peer for the headline pair, making the copy tax
visible. **kdb+** is excluded from *published* numbers pending a license
review. Below the SQL surface there is no peer; micro-level work uses
self-comparison benches as engineering instruments.

### The corpus

Seeded synthetic generators — ordered `i64` timestamps, low-cardinality keys,
`f64` values, with disorder fraction and null density as parameters — checked
into the repository. It grows two ways: new capabilities add case families,
and every closed bug adds the case that would have caught it.

### Blast radius

Where evidence lands earliest and heaviest:

1. **Storage bytes** — silent corruption; entrenches at format freeze.
   Golden-locked *before* the first real data exists in the format.
2. **C Data Interface unsafe export** — silent corruption in *other
   processes'* memory.
3. **Oracle-visible SQL semantics** — wrong answers, but loud under
   differential testing.
4. Everything else.

## Layout

TallyDB is **one crate**, `tallydb`, with a library target and a binary
target. The library is what an application links, like SQLite or DuckDB. The
binary is the console. The parts of the system are modules of that crate,
each keeping the boundary it had as a workspace crate.

```
tallydb/
  Cargo.toml
  build.rs             # compiles the vendored Lua when the `lua` feature is on
  src/
    lib.rs             # the crate root: what an embedder reaches, and the
                       #   numeric-or-key invariant. The engine proper is the
                       #   root's own files — table.rs, database.rs, view.rs,
                       #   multifactor.rs, partials.rs, script.rs, driver.rs
    arrow_lite/        # hand-rolled Arrow-compatible columnar format (f64/i64
                       #   buffers, u32-dictionary keys, C Data Interface
                       #   export; arrow-rs/PyArrow as dev-only oracles).
                       #   A SEAM: the standalone arrow-lite crate — a
                       #   broader design, specified in its own repository —
                       #   replaces this module when it exists.
    storage_lite/      # append-optimized segments partitioned on the ordering
                       #   key; compaction; zone maps; the WAL; I/O behind a
                       #   backend trait; lazy fault-in under a byte budget
    query_lite/        # scoped SQL parser (via sqlparser-rs) + our own
                       #   executor; DuckDB/DataFusion as differential oracles
    compute_lua/       # Lua scripting behind a trait; vendored PUC Lua 5.4
                       #   under compute_lua/vendor, hand-rolled bindings, the
                       #   upstream test suite under compute_lua/upstream-tests.
                       #   Feature `lua`, off by default; the console turns
                       #   it on.
    compute_linalg/    # multiplication-class kernels behind `LinalgBackend`
                       #   (faer). A SEAM: the interim solve the MatLua
                       #   ruling names; MatLua takes this module's place
                       #   when its endpoints land.
    corpus.rs          # the seeded synthetic generators of "The corpus"
                       #   above: `cfg(test)` and feature `oracle-harness`
                       #   only — compiled for the unit tests and the Python
                       #   oracles, never in the default build
    harness.rs         # the `extern "C"` oracle hooks the Python scripts
                       #   drive through the shared library (the PyArrow
                       #   round trip's sit in arrow_lite/harness.rs).
                       #   Feature `oracle-harness`
    shell.rs           # the console's `Console`. Feature `cli`
    bin/tallydb.rs     # the console binary, a thin skin over `shell`
  tests/               # integration tests, golden files, and the ten Python
                       #   oracle and benchmark scripts
```

**Features.** `cli` (default) enables the console and its two dependencies,
rustyline and csv, and implies `lua`. `lua` enables the scripting layer and
the build of the vendored interpreter. An embedder that wants neither writes
`default-features = false` and links a library whose only dependencies are
sqlparser and faer. `apicheck` builds the interpreter with `LUA_USE_APICHECK`
(test builds and CI). `oracle-harness` compiles the corpus and the C hooks for
CI's differential oracles; it is never on in a published build.

**One shared library.** The crate has one `cdylib` target, `libtallydb`,
carrying both the Arrow C Data Interface export and every oracle hook: one
build with `--features oracle-harness` serves all nine oracle scripts. There
used to be two — `libarrow_lite` and `libengine` — and a script had to know
which.

**The public surface.** The API is exactly what `lib.rs` re-exports; the
modules are `pub(crate)`, so no path into them is nameable from outside and
semver applies to that list and nothing else. Their rustdoc roots still
render under `--document-private-items`.

**What the crate boundaries used to guarantee, and what guarantees it now:**

| Guarantee | Then | Now |
|---|---|---|
| The engine carries no console dependency | `shell` was the only crate depending on rustyline and csv | both are optional, behind `cli`; the `--no-default-features` CI leg builds, tests, and documents the library without them |
| The corpus is never linked by the engine | `corpus` was `publish = false` and a separate crate | a module under `cfg(test)` and `oracle-harness`, absent from the default build |
| Miri runs over the columnar layer; sanitizers and `LUA_USE_APICHECK` over the Lua boundary | per-crate CI jobs | the same jobs: Miri filtered to `arrow_lite::` in the library's test binary, without the default features; the sanitizer job filtered to `compute_lua::` with the vendored C compiled sanitized |
| A region can be read and tested on its own | a crate | a `pub(crate)` module with the same seam; its tests run under its path prefix (`cargo test storage_lite::`) |

**The one sequencing constraint that still matters:** the differentiator is
compute-fusion — zero-copy numeric ops on stored buffers — and every change to
the layout is judged by whether that path stays a straight line from stored
buffer to curated op to Arrow out, with no copy. Building the storage engine
beautifully while the compute story slips would yield "another embeddable
TSDB" and miss the point.

## Roadmap

Milestones M0 through M5 are merged, taking the engine from a locked layout
to desk adoption: storage and its formats, the query surface, the console,
the extension model and corrections, then the desk features chosen by the
moat test — multi-factor compute, the ordered-axis dividends, segment-lazy
open, cross-process readers, and maintained views.

**M6 (WASM parity)** adds *embed in a browser*: the compute stack already
compiles for `wasm32`, so the remaining work is a browser `StorageBackend`
(OPFS/IndexedDB behind the existing trait), the JS bindings, and `lua.wasm`
behind the same feature flag.

**M7** adds *embed in a server*: a served product and a workbench UI, both
**separate artifacts embedding the engine** — the never-a-server guardrail's
sanctioned form — with the console module reused as the server's shell. The
load-bearing observation for M7's sync story: **segments are immutable,
self-describing, CRC'd objects committed by a generation manifest, so the
storage format is already the replication format.** A read-only browser or
client replica fetches the manifest and pulls segments lazily, verified by the
same checks reopen runs; the single writer stays wherever the WAL is. Nothing
earlier may foreclose this.

Until then we build the **native build first** — Linux, macOS, Windows,
linked into an application. Do not add WASM-target dependencies or write
WASM-specific code paths yet. What *is* required now: keep I/O and compute
behind trait boundaries from day one, with no filesystem, threading, or
dependency assumptions baked into the core that would block a future `wasm32`
target. That discipline is cheap today and expensive to retrofit.

## Who we write for

The imagined reader holds a **BS in Applied Mathematics with a minor in
Computer Science** — which is also a fair description of the target user.

- **Documentation is written for the math-major side.** It may assume
  mathematical fluency — "positive semi-definite", "least squares", "QR
  decomposition" need no apology — but must not assume systems fluency: terms
  like *mmap*, *tombstone*, *cache line*, or *FFI* are defined at first use.
- **Code is written for the CS-minor side.** Standard idioms, clear
  structure, no cleverness for its own sake. Where performance demands a
  non-obvious idiom, the accompanying comment carries the naive equivalent,
  so the reader can verify the clever version against it.
- **Performance wins every conflict with this constraint** — it is a
  nice-to-have, never a reason to ship slower code. But each win is
  documented as a deliberate bend, which keeps the constraint honest.

Where documentation can carry executable evidence, prefer it: Rust doctests
compile and run in `cargo test`, so a documented claim with a doctest fails
loudly when it stops being true.

## Operating

**The push gate.** `scripts/gate.sh` runs every check CI runs on stable, in
CI's order, judged by exit code: fmt; clippy with all features and with none;
build; the test suite with default features and with none; rustdoc with
warnings as errors, both legs; the shared library with the oracle hooks and
the nine Python oracle scripts against it; and the interpreter built with
`LUA_USE_APICHECK`. Green here is the evidence a push claims. The Miri,
Lua-suite, and sanitizer jobs need nightly or a C toolchain and stay in CI.

**Releasing.** Two workflows, deliberately separate. `.github/workflows/
release.yml` fires on a `v*` tag and attaches the console binaries for Linux,
macOS and Windows to the GitHub Release — it ships the *program*.
`.github/workflows/publish.yml` is `workflow_dispatch` only, refuses to run
from anywhere but `main`, and uploads to crates.io — it ships the *library*.
Publishing is manual because a version number is spent permanently the moment
it succeeds: yanking hides a release but never frees its number. Both build
`--locked`, so what ships is built from the dependency set CI tested.

A release is: land the work on `main`, bump the version in `Cargo.toml` (and
`Cargo.lock`), date the `[Unreleased]` section of `CHANGELOG.md`, tag the
merge commit, then run Publish from the Actions tab.
