# TallyDB — Design

This is the **developer companion** to `README.md`: what we are building,
why, and which parts are settled. The README describes where the project is
now from the user's point of view; this document describes where it is going
from the developer's — and, as decisions settle, **why it is the way it is**.
It is forward-looking today, but written to endure: when the project is
complete this becomes the durable record of its design decisions and the
principles behind them — a power-user-level inventory of *why*, not just
*what*. How we work — passes, reviews, issues, integration — is `AGENTS.md`.

## What this is (positioning, so scope calls stay anchored)

An **append-ordered numeric store**: embeddable, SQL-native, with numeric
compute (Lua + curated native ops) running *inside* the engine on its own buffers,
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
- **`f64` — the analytic / derived type.** Anything the numeric ops touch.
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
matters, it is handled by fixing the operation order in source (see
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

- **String *predicates* on key columns are in scope.** `WHERE symbol =
  '...'` / `IN (...)` / `WHERE name LIKE '%Bank%'` are built; regex
  matching is in scope but not yet implemented (rejected loudly until
  then — a todo, not a silent gap). All consume the interned strings and
  emit a *row selection*, not a string, so they don't need a third type.
  Because keys are dictionary-encoded, such a predicate is evaluated once
  per *distinct* value in the small dictionary and then applied as
  integer set-membership: string filtering is not just allowed, it's
  cheap.
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
  comparison, and — for `f64` — passed directly into the numeric ops and
  Lua as raw numeric buffers.
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

## The inclusion principle (SQL and Lua)

One principle governs both surfaces: **a capability — a SQL verb or
function, or a Lua stdlib facility — is in scope by default, and excluded
only where it would violate a named invariant.** We do not hand-pick a
feature list, and we do not require a use case to admit something:
**"we can't think of a quant use case for it" is explicitly NOT a valid
reason to exclude something otherwise in scope** — real usage regularly
surprises the people who built the tool. The invariants are the boundary,
not our imagination. The two surfaces share this *method* and differ only
in *which* invariants apply.

**The moat test (adopted from external review, ruled 2026-07-28).** The
inclusion principle is a negative filter: it says what is *admissible*.
It cannot order the backlog. The companion positive filter does: **build
first the things the three assumptions make cheaper for this engine than
for a general database.** Of each admissible-but-unbuilt candidate, ask
*"does DuckDB have to work harder than us here?"* If yes, building it
cashes a dividend the cuts already paid for — ordered ingest turning a
hash aggregate into a streaming sweep, contiguity turning a partition
into a slice. If no, it is generality wearing a feature's clothes, and
it spends the very thing the cuts purchased. The inclusion principle
decides *in or out*; the moat test decides *what's next*.

**SQL is bounded by** (a) numeric-or-key — no non-numeric, non-key column
type — and (b) no general-purpose cost-based optimizer.

| SQL capability | In / Out | Bounding invariant |
|---|---|---|
| `SELECT`/`WHERE`/`GROUP BY`/`ORDER BY`/`LIMIT`, equi-joins against small key-unique tables (the star-schema family), window functions, `UPDATE`/`DELETE`, scalar math, `CASE`, `HAVING`, `DISTINCT`, `LIKE` on keys, `NULLS FIRST`/`LAST`, `CREATE TABLE`/`INSERT` | **in** (built) | — |
| `RANGE` frames, `ASOF JOIN` (#65) | **in** (built, M5.1–M5.2) | — |
| regex on keys (#57), ordered-merge relatives beyond `ASOF JOIN` | **in** (not yet built) | — |
| `SUBSTRING`/`CONCAT`/`CAST AS VARCHAR`/`GROUP_CONCAT` — string *production* | **out** | (a): would need a text column |
| joins no structural fact licenses — neither side small enough to materialize, inputs not co-ordered on the join key, join-*order* search | **out** | (b): would need a cost-based optimizer (see *the join constraint, completed*) |
| a third column type (text, blob, boolean) | **out** | (a): numeric-or-key |

**The oracle-set rule for built-in functions (decided 2026-07-25).**
The inclusion principle bounds which *verbs* are in scope; this rule
bounds which *built-in functions we ship* on the SQL surface: **a
built-in joins the SQL surface only if the differential-oracle set
implements it** — DuckDB today, DataFusion when wired as the secondary
(which, since DataFusion is InfluxDB v3's SQL engine, also covers the
modern surface of our closest use-case neighbor). The rule does two
jobs at once: every admitted built-in is *born diffable* (it rides the
differential harness like everything else, rather than needing a
hand-built check), and every admitted built-in is *guessable* (users
find functions by knowing standard analytical SQL, which the oracle
set curates). Everything else TallyDB can compute — decompositions,
solves, anything matrix-shaped — is reached through the Lua
surface, where results need not fit SQL's scalar-per-cell type system.
The rule governs what *we* ship built-in; a user's own registered
functions are their code, named as they please. Applied at adoption:
`regr_slope` / `regr_intercept` / `covar_pop` / `corr` stay (all in
the oracle set); `eigen_max` leaves the SQL surface when the SQL-in-Lua
scripting API lands (#41) — it was an eigendecomposition amputated to
a scalar so SQL could return it, and it migrates to a SQL-in-Lua example
whose NumPy check becomes that example's differential test. Reopen
condition per function: a function that later becomes standard in the
oracle set becomes eligible here.

**Lua is bounded by** (a) the sandbox — no filesystem, process, network,
native code, memory-safety, or escape hazard — (b) determinism unless the
author opts out, and (c) the same numeric-or-key rule on what may *cross
into the engine*.

| Lua stdlib | In / Out | Bounding invariant |
|---|---|---|
| `math`, `table`, `string`, `utf8`, curated `base` | **in** | — |
| `math.random` / `randomseed` | **in**, documented | (b) with opt-out — forfeits reproducibility |
| `io`, `os` | **out** | (a): filesystem / process |
| `debug`, raw metatable & `raw*` functions | **out** | (a): sandbox escape (shared metatables) |
| `load` / `loadstring` / `loadfile` / `dofile` | **out** | (a): code injection / memory safety |
| `package.loadlib`, native `require` | **out** | (a): native code (`package` curated to a pure-Lua searcher) |
| `coroutine` | **out** (deferred) | *not an invariant* — see exceptions |

### Apparent exceptions (named, so they don't read as drift)

- **`string` is cut in SQL but open in Lua** — not a contradiction. The
  numeric-or-key invariant governs *what crosses a boundary* (a stored,
  intermediate, or output value), not *local scratch*. SQL has no scratch:
  every value is a column, so a string function *is* a text column, and it
  is out. Lua has scratch (locals), so string manipulation is transient,
  and the invariant is enforced at the Lua→engine boundary — a returned
  string interns into a **key**; a bare string column cannot cross. Same
  invariant, opposite-looking result. (The one real guard is on the
  *output*: a script synthesizing a unique label per row would blow the
  low-cardinality key assumption — capped at the boundary, not by
  crippling `string`.)
- **`math.random` is admitted despite nondeterminism.** The determinism
  invariant carries an explicit opt-out: a script may be nondeterministic
  if the author chooses, at the documented cost of query reproducibility.
  We can afford this because we are not replicated — unlike Redis before
  7.0, whose *script* replication forced determinism (Redis 7 relaxed it
  once it replicated *effects* instead).
- **`coroutine` is a genuine exception** — excluded for an *implementation*
  reason, not a principled one. It breaks no invariant; it fights the
  `pcall`/`longjmp` discipline at the Rust↔C boundary. It is a *deferral*
  pending a binding that can host it safely, not an invariant-based cut —
  when the binding can, the principle admits it.

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
> the trap this ruling closed). `NULLS FIRST`/`LAST` syntax is built
> (M3.4). The choice was made from the numeric-or-key thesis,
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
> **Built (#42, 2026-07-29)** — `storage-lite/src/alp.rs`, registry
> tags 2–4: ALP with per-vector RD and raw fallbacks for `f64`
> (the encoder computes all candidates and keeps the smallest, so it
> can never bloat), and the integer sibling — frame-of-reference +
> bit-packing — for non-clock `i64` columns and `u32` symbol codes.
> The corpus ticks family rounds to pennies first, per the caveat.
> Measured (release, seed 42, 1M rows/family, 2026-07-29,
> `measure_42`): ticks prices **4.18×** vs raw through ALP; telemetry
> continuous reals 1.16× through RD (lossless real doubles are
> near-incompressible — ~1.2× is the published family); symbol codes
> 6.4–10.5×; integer cents 4.0–4.2×. Decode 31–93M values/s, the
> shipped delta-of-delta's band, paid once per segment at open. The
> writer-policy change was the first deliberate encoder revision:
> `segment_v1.bin` became the decode-compat golden (old bytes decode
> forever), `segment_v2.bin` locks the new encoder.

## Deployment shapes

> **Decided (2026-07-23): library first; a single-file shell binary at
> M3; never a server.** TallyDB ships two ways: as an embeddable
> library (the design center, unchanged), and — from M3 — as a
> standalone single-file binary attached to each release: a CLI shell
> over the same `engine::Database` doorway, the `sqlite3`/`duckdb`
> precedent. Installation is copying one file. (Ruled 2026-07-29: a
> third channel arrives with M5.5 — the Python binding as wheels on
> PyPI. Python-specific by design; the engine and console never
> depend on it.) This is not a move
> toward general purpose: the shell exposes exactly the library's SQL
> surface, and the three assumptions bound it the same way.
>
> What the shell shape pulls in (all additive, none a refactor): DDL
> (`CREATE TABLE` with the numeric-or-key types and the declared
> ordering key) and ingest (`INSERT`, plus a bulk import) in SQL;
> statically linked compute (already the default: the compute stack is
> pure Rust plus the vendored Lua sources);
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
> single writer, a clean `Send`/`Sync` story. *Satisfied 2026-07-27
> (#51):* `Table::reader()` hands any thread a cloneable `Send + Sync`
> handle minting point-in-time `TableSnapshot`s while the one
> `&mut Table` writer appends, mutates, or compacts — the shared state
> sits behind a per-table lock held only for reads and swaps (bounded
> by one write-buffer copy), compaction is read-copy-update through the
> segment `Arc`s, and the single-writer cut stays a compile-time fact.
> No reopen condition is foreseen for the listener; the
> network-boundary argument is structural.

## The console, and the deployment roadmap beyond it

**Decision record — the M3.5 console (#39, ruled 2026-07-27).** The
shell / security / systems separation is the architecture: the engine
(systems) stays dependency-clean; `tallydb-shell`'s `Console` is a
reusable module a future served product embeds; `main.rs` is a thin
skin. Rulings, each with its losing alternatives: **dependencies** —
rustyline and csv only, confined to the shell crate (zero-dep
hand-rolling rejected as reinvention; a CLI framework rejected as
surface without need). **DDL grammar** — `BIGINT` / `DOUBLE` / the
coined `KEY`, one `ORDERING KEY` column constraint; `VARCHAR`/`TEXT`
refused with the keys-are-interned-labels teaching error rather than
aliased (an alias teaches the wrong model); user-typed `PRIMARY KEY`
refused with its own teaching error (the ordering key is not a
uniqueness constraint — duplicates are first-class), while serving
internally as the parser's carrier for the `ORDERING KEY` phrase.
**Import** — CSV in the shell layer feeding the ordinary append path;
the engine never parses CSV. **Code registration** — explicit
dot-commands only, and there are three: `.lua` and `.luascalar`
register kernels typed at the prompt, and `.run FILE` executes a
driver script from a path the user names. What they share is the
property that matters: code enters through a channel the user typed
deliberately, never through data. `CREATE FUNCTION ... LANGUAGE LUA`
is deliberately *not* SQL, so a SQL string — which may be built from
user input — is never a code-injection vector; the SQL form is a
recorded decision for the served product's threat model, not before.
(`.run` reads a file, so it inherits the console's trust in the local
filesystem, the same trust `.import` already needs.) Local security posture: an OS file lock
(released by the OS on death — no stale locks) admits one process per
directory; table names stay identifiers (they become directory names).

**The roadmap beyond native GA (recorded 2026-07-27; reordered
2026-07-28, twice — first the desk before the browser, then the
extension model before the desk: the 2026-07-28 review rulings touched
M0–M3 design, and the back-end must settle before anything user-facing
is built on it).** M3 ships *embed in your application* plus the
console. **M4 (the extension model + corrections)** makes the
2026-07-28 rulings real — the back-end settles before anything
user-facing builds on it. The plan of record, approved 2026-07-28:

- **M4.0 Trait exposure** — `WindowAggregate` and `Registry`
  re-exported, `Table::register_window` + `Database::register_window`
  public, the ~20-line embedder kernel as a doctest.
- **M4.1 The feature gate** — compute-lua becomes a non-default
  feature the console enables; CI builds and tests both legs;
  sanitizer/apicheck jobs run in the on-leg only.
- **M4.2 The Lua front-end** — Lua as a thin front-end over compiled
  ops, on the architecture NumPy proved (a slow interpreter is fine
  when the loops live in compiled code and scripts only compose): the
  vocabulary invariant (every registered *window aggregate* is callable
  from Lua by its SQL name — registry-driven, so future natives flow in
  for free; column functions are a second namespace and do not cross,
  see below), the vectorized
  whole-column kernel slot wired (`eval_column`, built in M2.7 and
  never connected; likely closes #53), the compose-don't-loop idiom
  documented, promotion made mechanical (one registry name, Lua
  implementation swappable for a trait implementation with no query
  change).
- **M4.3 The corrections design cycle — ruled 2026-07-28, closed.**
  F2 is **(a) whole**, and its three sub-decisions are settled: the
  ingest-sequence column is **default-on** for every table (one
  solution for arrival order, the `AS OF` coordinate, and the
  ready-at-hand stable id; the virtual-until-divergence design makes
  it nearly free — store nothing while sequence == row id, materialize
  delta-coded from the first divergence); the retention horizon is
  **unbounded by default** with a per-table bound available; and **one
  keyword — `ASOF` — with structure dispatching** (amended by ruling,
  2026-07-28): followed by `JOIN` it is the event-time nearest-match
  join (the DuckDB/ClickHouse/QuestDB/Snowflake spelling, unchanged);
  followed by a sequence it is knowledge-time travel
  (`FROM trades ASOF 41520`). Why one word: the join side has a
  universal convention worth honoring while the travel side has none
  (Oracle `AS OF SCN`, Delta `VERSION AS OF`, Snowflake `AT` — no
  consensus to diverge from), and two-word `AS OF` collides with
  SQL's alias grammar (`trades AS OF` = "trades renamed OF"). Riders:
  the SQL:2011 long form `FOR SYSTEM_TIME AS OF n` stays accepted
  (sqlparser parses it natively; it is the internal carrier), and a
  textual teaching error catches two-word `AS OF <n>` before the
  parser garbles it — the error message itself teaches the two axes
  (join = event time on the ordering key; travel = knowledge time on
  the ingest sequence). Mix-ups cannot yield wrong semantics: the two
  uses need different surrounding syntax, so confusion is a loud
  error, never a silently wrong answer.
- **M4.4 The corrections build** — the hidden ingest-sequence column
  (the permanent knowledge axis; delta-coded to almost nothing while
  uncorrected), retaining compaction (history segments), the knowledge
  mask (the live mask's analog), the `AS OF` predicate. Oracle: DuckDB
  re-deriving as-of answers over an explicit history table — emulation
  in the referee only, never in the product. Format additions ride the
  one manifest revision shared with F3's zone-map lift (sections
  reserved now, filled by whichever lands first).
- **M4.5 The correctness batch** — #73 (the atomic mutation commit
  record: old-or-new for crashes and readers, recovery
  auto-completes), #63 (Miri in CI), #69 (the upstream Lua test
  suite), the review-noted redundancies.
- **The Lua trial** — ruled 2026-07-28, **pass** (see *The Lua
  layer*): the Agent brought the evidence brief (#76), the Human
  ruled Lua stays. The sunset clause dissolved.
- **M4.6 SQL-in-Lua (#70)** — built on the pass: driver scripts
  (`query`/`append` through the `ScriptHost` seam, the console's
  `.run`), evidenced by an end-to-end SQL → Lua → SQL differential in
  the CI Lua oracle.
- **M4-close reviews (2026-07-28)** — three independent repo-wide code
  reviewers, every finding reproduced before its fix. What they found
  is worth recording, because it says where this milestone's risk
  actually sat: two of the three correctness bugs were in *text
  handling and routing*, not in the storage machinery the milestone
  was about. The `ASOF` pre-pass reassembled statements by joining
  tokens with spaces, so a `--` comment silently swallowed the rest of
  a query — the precise failure the clause's own ruling said could not
  happen; whether a query was accepted depended on how many segments
  its rows occupied; two embedder-facing paths could abort the process
  (an unreserved Lua stack push, and embedder code called without
  `catch_unwind`); a supersession at coordinate 0 wrote commit evidence
  indistinguishable from a plain delete; and absent zone maps
  *falsified* pruning against the invariant stated in two other files.
  The reviews also caught this document and several crate docs claiming
  a vocabulary invariant wider than the code holds. Lesson taken:
  hand-rolled pre-parse text manipulation deserves the same adversarial
  testing as the format code, and a "sound over-approximation" needs a
  test that a *missing* input cannot flip it.

**M5 (desk adoption)** then builds what the target user needs, chosen
by the moat test: multi-factor curated compute (K > 2 — the recorded
LAPACK-class-returns trigger firing; **re-ruled 2026-07-30**, this now
goes to MatLua in the Lua tier rather than to a faer dependency of our
own — see the decision record *where non-standard compute lives*,
below), the ordered-axis
dividends (cross-sectional partitioning, time bucketing — F1 ruled (d)
2026-07-29, monotone ordering-key arithmetic in `GROUP BY`,
`LAG`/`LEAD`, `RANGE` frames, the `ASOF` join — **all built
2026-08-01**, see *The M5 ruling batch* for the rulings and the
stdlib table for what shipped), segment-lazy open (F3), cross-process
readers (F4 —
**built 2026-07-29**: read-only opens over a live writer's directory
see the durable prefix consistently, old-or-new per mutation;
`tallydb DIR --read-only` with `.refresh`/`.flush` is the console
half; #42's codecs landed the same day),
and reach (bulk Arrow ingest, a Python binding with host-callback
NumPy kernels — distribution ruled 2026-07-29: wheels on PyPI for the
binding; the console stays a Python-free single binary). **M6 (WASM parity)** adds *embed
in a browser*: the compute stack already compiles for wasm32; the
remaining work is a browser `StorageBackend` (OPFS/IndexedDB behind
the existing trait — written knowing an HTTP-fetch sibling comes
later), the JS bindings, and `lua.wasm` behind the same feature flag.
**M7** adds *embed in a server*: a Servette-shaped served product and
a workbench UI, both **separate artifacts embedding the engine** (the
never-a-server guardrail's sanctioned form), the console module reused
as the server's shell. The load-bearing observation for M7's sync
story: **segments are immutable, self-describing, CRC'd objects
committed by a generation manifest — the storage format is already
the replication format.** A read-only browser or client replica
fetches the manifest and pulls segments lazily (zone maps prune the
fetch), verified by the same checks reopen runs; the single writer
stays wherever the WAL is. Nothing earlier may foreclose this.

## Current milestone: native only

We are building the **native build first** — Linux/Mac/Windows, linked into
an application. A WASM build (and eventually a WASM compute layer) is a real
future direction, not current scope — do not add WASM-target dependencies
(`lua.wasm`) or write WASM-specific code paths
yet. What *is* required now: keep I/O and compute behind trait boundaries
from day one (storage backend, scripting backend, math backends), with no
filesystem, threading, or dependency assumptions baked into the core crates
that would block a future `wasm32` target. That discipline is cheap today
and expensive to retrofit — don't skip it, but don't build the WASM side of
it either. WASM matters to this project specifically because its hardest pieces are
already in hand: `lua.wasm` (also Lua 5.4, same author) exists, and the
linear-algebra layer needs no WASM port at all — `compute-linalg` is pure
Rust and compiles for `wasm32-unknown-unknown` as-is (verified
2026-07-27; see *Curated compute: what the engine calls, and why*). A
LAPACK-in-WASM layer left the critical path when LAPACK left the query
path; `blas.wasm` left it with the system-BLAS removal.

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
- **faer** — pure-Rust linear-algebra kernels (slim feature set: no
  thread pool, no RNG), consumed by `compute-linalg` for the matrix
  products.

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

## The axes: cuts, refusals, and reversal classes (adopted 2026-07-25)

This section adopts the review vocabulary of *A Taxonomy of Principled
Database Simplifications* (2026): a specialized engine's deletions are
**forced by positive workload assumptions**, never selected from a
menu; each cut's **reversal class** (additive / invasive /
foundational) sets how much scrutiny its licensing assumption
deserves, because the assumption is what fails, not the cut. The
external axis assessment of 2026-07-24 audited this project against
that method; this section records the outcome so every absence has a
name and every refusal is visibly deliberate.

**Two defaults govern every cut, each overridden only by a stated
use-case assumption (the cut-depth principle, 2026-07-25).**

1. *General-purpose by default.* The engine keeps a subsystem until an
   assumption forces its deletion — no cut without a licensing
   assumption.
2. *Clean (endpoint) by default.* Once an assumption licenses a cut,
   take it to its endpoint — the maximal deletion — unless a further
   assumption justifies stopping short. A partial cut or a kept surface
   is itself a decision that must name the assumption permitting it:
   keeping `UPDATE`/`DELETE` (corrections happen), keeping broad SQL
   (analysts want it), tolerating out-of-order ingest (reinserts arrive
   out of order) are all such justified deviations.

Two qualifiers keep this honest. *Tidiness is not a default to trade
against — it is hygiene:* whatever depth a cut lands at, it carries no
residual machinery from the deleted subsystem and leaks the deleted
concern nowhere. A partial cut is allowed; a leaky one never is. And
*scrutiny scales with reversal cost:* prefer the endpoint, but hold a
foundational cut's licensing assumption to a far higher standard than
an additive one's — being wrong about the foundational cut costs a
rebuild, being wrong about the additive one costs a later layer.

| Axis | Our position | Licensing assumption | Reversal class |
|---|---|---|---|
| Mutation | Cut to the endpoint: append-only storage, tombstone+reinsert | Data is appended, not revised (assumption 1) | Foundational |
| Working set | **Cut, built 2026-07-30** (the residency design, below): the *queried working set* fits in memory — the table itself need not. Opens read manifest metadata only; the executor prunes on it before any decode; segments fault in on first touch and are retained under a byte budget. (Whole-object reads and the backend contract are untouched — this is not the retired mmap/ranged-read path.) An unpruned full-table scan's working set is still the whole table (#88). | The rows a query touches fit in memory (see below) | Foundational |
| Query (planner) | Cut: a **fixed-strategy planner** — `plan()` exists; search, costing, and choice do not | One access path ⇒ nothing to choose between | Additive |
| Query (surface) | **Refused**: broad standard SQL, bounded by the inclusion principle | — | Additive |
| Access path | Cut totally: no secondary indexes; ordering-key clustering + scan is the one path (zone maps are pruning metadata, not a path) | Ordered ingest on the declared key (assumption 2) | Invasive |
| Write | **Refused**: cheap online single-row append is the design center | — | n/a |
| Transaction | Cut: submitted units only — no `BEGIN`/`COMMIT`/`ROLLBACK`, no session state (contract below) | Work arrives as single statements | Invasive |
| Isolation | Fixed: snapshot isolation at statement granularity (contract below) | One guarantee suffices | Invasive |
| Concurrency | Single writer *per table*, concurrent snapshot readers — shipped at the facade in M3.1 (#51): `Table::reader()` mints `Send + Sync` snapshot handles while the one writer proceeds. Writers scale by table (one owner each — a thread, or a whole process: directory locks are per table). **Cross-process readers ruled in 2026-07-28 (M5):** writer-exclusive / reader-shared locks; POSIX-first — unlink-keeps-open-files-alive gives reader safety for free on Linux/macOS, Windows requires deferred deletes and is documented as the lagging platform until its cleanup pass | One writer per table is enough | Additive to correct |
| Distribution | Cut totally: one machine | Data and load fit one node — and **compute-without-copying exists only because no network boundary exists anywhere**, the deployment argument generalized | Foundational |
| Deployment | Cut: library, never a server (see *Deployment shapes*; live in-process ingest+compute is in, networked subscriber fan-out is out — *Live data* below) | One application owns the data | Additive |
| Schema | The hardest cut: numeric-or-key, enforced in the type system (assumption 3) | Every column is a number or a label | Foundational |
| Durability | Not cut: publish is atomic **and synced**, and the write buffer sits behind a sidecar WAL with sync levels (#43, ruled on measurement: default group commit ≤ 100ms, `Full` for a zero loss window, `Off` restoring the flush boundary) | — | — |

**`RANGE` frames, and what they cost.** Every `ROWS` frame is a
trailing row count, uniform across the column, which is what lets
`WindowAggregate::evaluate_frames` slide one add and one remove per
step. A `RANGE` frame is bounded by ordering-key *value*, so its width
varies row to row — and, because standard SQL ends such a frame at the
current row's **last peer**, it is not even trailing in row-index
terms: a frame can extend forward over rows sharing the current row's
key. The executor therefore computes explicit `(start, end)` bounds per
row (one O(rows) pass, both pointers monotone) and hands them to
`WindowAggregate::evaluate_bounded_frames`, whose default recomputes
each frame. That default is why `RANGE` is correct for every
aggregate, including embedders' and Lua kernels', the day it ships.
What it is not yet is *incremental*: an aggregate with sliding state
can override that method with a two-pointer sweep, and until it does,
a wide `RANGE` frame costs O(rows x window) where the equivalent
`ROWS` frame costs O(rows). Tracked, with the safety net that a
statistic whose override cannot hold its accuracy simply keeps the
default and stays correct.

Refusals are design decisions too: the write axis and the query
surface are kept deliberately, and a reader should be able to tell
refusal from oversight.

**Live data, precisely.** The Distribution and Deployment cuts draw the
line through the middle of the phrase "live feed," so a reader comparing
TallyDB to a tickerplant must read them together. *Freshness is not
sacrificed:* a query snapshot includes the live write buffer, not only the
frozen segments (`Store::snapshot` appends the buffer's rows to the
segment sequence; the contract is that a snapshot covers exactly the rows
appended before the call), so a row appended microseconds ago is visible
to the very next query. Freeze/flush is the *durability and layout*
boundary for segments; power-loss durability of the newest rows is the
WAL's, at the configured sync level (#43) — neither is a visibility gate. So an application that ingests
a live feed and recomputes over it in the same process — real-time risk,
live P&L, a moving regression on the newest window — is squarely in scope,
and is the compute-without-copying sweet spot: socket → storage → SQL →
curated compute with no serialization hop. What is *out* is being the tick **server**:
one process streaming ticks over the network to a farm of subscriber
processes, which the never-a-server (Deployment) and no-network-boundary
(Distribution) cuts forbid outright. Stated as a single rule: "live feed"
as *in-process analytics over freshly-landed data* is in; "live feed" as
*server-side publish/subscribe fan-out* is out. The reactive-compute shape
over that fresh data — batch ingest hooks and continuous queries, per-row
invocation excluded — is the *feed-reactive compute* decision record
below.

**The access-path cut licenses the planner cut.** A planner exists to
choose among access paths; with exactly one path there is nothing for
a cost model to decide, so the absence of an optimizer is structural,
not a bet about workload simplicity. Corollary, to prevent an
over-broad reading of the settled "no optimizer": **predicate
reordering is not forbidden** — evaluating the cheapest, most
selective predicate first is a heuristic needing no statistics beyond
existing zone maps, and remains available. Likewise, if within-segment
scan acceleration is ever wanted, the sanctioned shape is block-level
min/max summaries at the codec's block granularity (small materialized
aggregates — decades old, patent-safe); per-value order-preserving
lossy codes should be patent-checked before any implementation, and
their value lies in unclustered data, which assumption 2 removes.

**Working set — decided (2026-07-25): the cut is owned.** Version 1
opens a table by decoding it into memory, and this is now a stated
commitment, not drift: the licensing assumption is that *a table fits
in memory* at the scale this engine targets. The banked simplification
is real (no buffer pool, no page cache, no partial-read machinery; the
mmap/ranged-read follow-up notes are retired). Reopen trigger, stated
because this cut is foundational-class: the M3 benchmark suite's
startup-time and footprint results embarrassing open-time on
realistic tables. The recorded escape is a zero-copy-open format
version — additive under the append-only version and codec registries.

**Transaction contract.** Work reaches the engine only as submitted
units — one `append`, one `query`, one `mutate`. There is no
interactive transaction, no session state, and no verb for either;
adding a session object would be a *reversal on the transaction axis*
and must be treated as one.

**Isolation contract.** A read sees a snapshot taken at statement
start: `Store::snapshot` returns owned segment views, and appends or
mutations after the call are invisible to it (test:
`snapshot_is_isolated_from_later_appends`). One guarantee — snapshot
isolation at statement granularity — no isolation-level menu.

*Cross-table consistency, as shipped (#51, M3.1):* a statement that
reads more than one table — today only a join, which snapshots the
fact table and each dimension separately — takes multiple `snapshot()`
calls at distinct instants. Through a `Database` handle this is sound:
the writer needs `&mut` on the same handle, so no write interleaves
mid-statement. The detached reader handles are **single-table by
scope** (`TableSnapshot` serves `SELECT` over its one table; joins
resolve only through a `Database`), so no shipped path can take a
cross-table torn read — and no cross-table snapshot epoch is promised.
#51 kept the single writer; concurrent *writers* never arrived. If a
future surface lets detached readers span tables, the one-epoch
obligation recorded here revives with it.

**Truth values — decided (2026-07-25).** There is no boolean type and
none is coming. A flag column is `i64` in {0, 1} — which is the right
answer, not a workaround: `SUM(flag)` is a count and `AVG(flag)` is a
duty cycle. When computed expressions land in projection, a projected
comparison yields `i64` in {0, 1}. `bool_and`/`bool_or` are not
offered (`MIN`/`MAX`/`SUM` over the flag serve). Recorded now so a
third type cannot arrive as an implementation detail of the
arithmetic-projection commit.

**The join constraint is a size invariant, not a modelling shape
(restated 2026-07-25).** What execution requires is that the build
side be small enough to materialize; "star schema" was the use case
wearing the invariant's clothes — the same correction as
time-vs-ordered. The rule is: **one large table joined against tables
small enough to materialize.** This preserves every current behavior,
and it *licenses* (as ordinary future todos, not scope fights) shapes
the modelling name wrongly excluded — join chains against several
small tables, snowflakes, self-joins against small aggregates — while
naming the real hazard: a nominal "dimension" grown too large to
materialize. A stated threshold belongs in the contract when the
executor generalizes.

**Decision record — the join constraint, completed: strategy fixed by
structure (Human-ruled 2026-07-26).** The size invariant above is one
clause of a two-clause principle: **a join is supported when a structural
property of its inputs fixes the execution strategy — never by cost.**
Each admitted strategy is guarded by the structural fact that makes it
safe at any scale without estimation:

| Strategy | Guarding structural fact | What it protects against |
|---|---|---|
| Broadcast/hash lookup | one side small enough to materialize, key-unique | unbounded memory |
| Ordered merge (`ASOF JOIN` and relatives, #65) | both sides ordered on the join key — their declared ordering key | unbounded memory *and* sorting |

**Build note (2026-07-30, M5.2): what shipped is clause 1's shape, not
clause 2's.** The as-of join as built indexes the dimension side in
memory — one ascending `(clock, row)` list per key — and binary-searches
it per fact row. That is correct at any ordering (the index is stably
sorted, so a late-arriving quote still matches), and it needs no
`is_ordered()` gate; but it materializes the dimension, so it is
guarded by clause 1's size invariant, not clause 2's streaming
property. The co-walk clause 2 describes — a cursor per side, memory
independent of both inputs, licensing large ⋈ large — is **not built**:
`execute_join` materializes both sides up front, and making the
dimension streaming is the same work as streaming scans generally
(#88). Until then, the ordered-merge row above states a *design*, and
the as-of join's actual reach is "dimension fits in memory". Tracked as
#92; the trigger to close it is a quote history that does not fit.

Clause 1 is the size invariant, unchanged, guarding the strategy that
materializes a build side. Clause 2 needs no size bound as a *property,
not an exemption*: a merge over inputs already clustered on the join key
is a streaming co-walk — a cursor per side plus the current match window
— so its memory does not scale with input size; that is exactly why it
may admit large ⋈ large, and why only on the ordering key, where the
storage layout guarantees the clustering (assumption 2 plays for clause
2 the role the size invariant plays for clause 1). Its runtime guard
would be `Segment::is_ordered` — the check the window executor already
relies on; a transiently disordered table (UPDATE reappends before
compaction) would refuse the merge loudly rather than serve a wrong
answer. (Design, not code: per the build note above, the co-walk is
unbuilt, and what shipped needs no such gate.) Dispatch
never estimates: an embedded single-machine snapshot knows every table's
**exact** row count and every dictionary's exact cardinality, so "small
enough" *can* be a measurement rather than a cost model — the
fixed-strategy planner stays fixed. That measurement is **not yet
enforced**: `execute_join` materializes both sides unconditionally and
no size check refuses an oversized dimension, so the size invariant is
today stated, not checked. A threshold belongs in the contract when the
executor generalizes; the as-of join riding clause 1 (#92) is what
makes it start to matter. A join with *neither* guarding
fact — two large tables on a non-ordering key, or join-*order* search —
is **refused loudly, naming the missing structure**: serving it needs
spilling, partitioning, or an optimizer, i.e. a different product.
*Precedent (validates the workload, no vote on the how):* kdb+ dispatches
by user-named join verbs trusting declared structure (`lj` keyed lookup,
`aj` sorted as-of) with no optimizer anywhere; QuestDB keys `ASOF`/`LT`/
`SPLICE` off the schema-declared designated timestamp — the closest
living relative of this rule; ClickHouse ran years of the world's largest
analytics on exactly clause 1 (hash join, right side fits memory, join
order = syntax order) before generalizing into CBO — the reopen path,
visible; DuckDB implements `ASOF JOIN` as first-class syntax over a CBO,
which we refuse, but it standardizes the surface and serves as the
differential oracle, so the ordered-merge family is born cross-checked
(the oracle-set rule holds).

**Load-bearing note on dictionaries:** the low-cardinality assumption
carries query performance, not just storage size — cross-segment
grouping and joins pay segments × distinct-values remapping, and the
recorded remap-cache mitigation bounds but does not remove it.

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
per segment; window frames are `ROWS`-only for now. Two of the three
have since been superseded by later work rather than reopened: M5.1
added `RANGE` and whole-partition frames, so the frame shape is no
longer `ROWS`-only; and batch count is not a contract — a plain scan
still yields one batch per segment, but the collapsing stages
(`ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT`, `HAVING`, `GROUP BY`)
materialize a single batch, as `QueryOutput`'s own documentation says.
The `SUM(i64)` half stands. The two sibling cadence
questions closed together, both ruled by the Human 2026-07-27 on a
measurement (recorded in #43/#44 and built in M3.2/M3.3):

**Decision record — durability: WAL with sync levels (#43).** A
sidecar write-ahead log (the segment format untouched), three levels:
`Group(interval)` — the default, 100 ms — logs every append and
group-commits with an in-thread sync, bounding the loss window at the
interval for +0.4–1µs on a ~1µs append (measured; the in-repo
`measure_wal_regimes` re-earns the number: off 0.99µs, group-100ms
2.06µs, full 728µs per append, run 2026-07-27, container fs); `Full`
syncs every append — zero window at ~700× per-append cost, shipped
documented, never default; `Off` writes no log and restores the
flush-boundary contract for replayable upstreams. Replay recovers the
per-record-CRC clean prefix, skips segment-covered rows, and ignores
wrong-generation logs (compaction reassigns row ids). *Rejected:*
flush-boundary-only as the GA contract (strangers assume a database
keeps what it acknowledged) and per-table dual contracts with no
default answer. *Reopen trigger:* tail-latency complaints from the
unlucky append paying the in-thread sync (10–46 ms worst on the
measured disk) — the fix is a background sync thread, which is the
may-the-library-own-a-thread question shared with #44's deferred
time-aligned freezing.

**Decision record — freeze threshold in bytes (#44).** The knob
speaks bytes (the buffer bound an embedder budgets); numeric-or-key
makes rows fixed-width, so bytes convert exactly to a per-schema row
count at construction (8 per number, 4 per key code; dictionaries —
bounded by distinct values — sit outside the bound, documented).
Setting rows and bytes together is refused loudly. *Deferred with
triggers:* time-aligned hybrid freezing — pruning-profile evidence
from the end-to-end suite (#52), and the library-thread question
above.

## The SQL stdlib — the surface, tabulated (#49, ruled 2026-07-27)

One table, so the in/out line is a deliberate record instead of an
accumulation of implementation accidents. Every IN row is born with a
DuckDB differential family; the shell's help cites this table.

| Construct | Status | Note |
|---|---|---|
| `SELECT` projection, aliases | in, built | |
| `WHERE` (numeric compares, key `=`/`IN`/`LIKE`, `AND`/`OR`/`NOT`) | in, built | NaN-aware; zone-map pruning; LIKE per distinct value |
| `WHERE` comparing two expressions (`x > y`, `x * 2 > y + 1`) | in, built (#95) | a zone map knows a column's range, not an expression's over several of them, so this leaf **prunes nothing**. Pruning therefore degrades **per conjunct** — `WHERE ts > 1000 AND x > y` still skips segments on `ts` — which is why it is a separate predicate variant, not a generalisation of the prunable one. `40 < x` is mirrored to `x > 40` before lowering, so which side the column sits on changes nothing. A registered kernel is refused here by name: it must see the query's rows as one column, and a filter is evaluated per segment |
| regex on keys | deferred by ruling (2026-07-29) | menu incl. the Lua-pattern house option on #57 |
| `IS NULL` / `IS NOT NULL` | in, built | one predicate arm over the validity bitmap; the only *total* leaf (never UNKNOWN); `IS NOT NULL` prunes an all-null segment |
| `GROUP BY` + `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` | in, built | exact-loud `SUM(i64)` |
| `GROUP BY` monotone ordering-key arithmetic (`ts / 60`) | in, built (M5.3) | bucket index, `(ts / 60) * 60` for the bucket start, bare `ts` for the finest bucket. `/` truncates (ISO); `//` accepted (DuckDB's spelling). May be named by its SELECT alias. Over ordered data the grouping **streams** — accumulator state is the open bucket, not the result (measured 1.65× less than the hash path over 160k groups); unordered data falls back to hashing with the same answers, and `compact()` restores the fast path |
| `FIRST` / `LAST` aggregates | in, built (M5.3) | the de-facto TSDB names (ruled (a)); positional on the **time axis**, not row order, so a late-arriving row cannot become "last". Ties on the clock go to the last row in storage order; nulls skipped, which coincides with DuckDB's `arg_min`/`arg_max`. Also available as window functions, where — being positional — they **refuse an unordered window**: with no order there is no first |
| cross-sectional `PARTITION BY` (the ordering key, or a bucket of it) | in, built (M5.3) | the transpose of `PARTITION BY sym` — one partition per instant, across every symbol. Unordered windows (`OVER (PARTITION BY ts)`) take the whole partition, per standard SQL; several terms intersect (`sym, ts / 60` = per symbol per bar); any `BIGINT` partitions, `DOUBLE` never |
| scalar expressions over window results | in, built (#94) | `x / sum(x) OVER (PARTITION BY ts)`, `x - lag(x) OVER (ORDER BY ts)`, the rolling z-score. Window calls hoist out of the scalar and compute first — standard SQL's evaluation order |
| `HAVING` | in, built | hidden-column lowering; WHERE grammar over the group row |
| `DISTINCT` | in, built | by value; NaN=NaN, −0=0, NULLs equal; `DISTINCT ON` out |
| scalar expressions (`+ − * / %`, `ABS ROUND FLOOR CEIL SQRT LN EXP POWER`) | in, built | f64, three-valued; IEEE division — NaN is a value; i64 refused loudly (#40) |
| `CASE WHEN` | in, built | conditions are WHERE grammar; UNKNOWN falls through |
| `ORDER BY` one column, `NULLS FIRST/LAST` | in, built | default nulls-last both directions (oracle's convention); refused on symbol columns (#58 = B, unordered labels); bounded by `LIMIT` it runs top-k, O(k) memory |
| multi-column `ORDER BY` | in, later | additive lowering |
| `LIMIT`/`OFFSET` | in, built | |
| window functions over `ROWS` frames | in, built | curated + Lua kernels; incremental sweep |
| `var_pop` / `stddev_pop` as windows | in, built (M5.0) | one column; variance *is* self-covariance, so they share `covar_pop`'s corrected two-pass and incremental sweep. Population forms only, matching the `_pop` family; sample forms and the group-level (non-window) surface are additive and unbuilt — as they are for `covar_pop`/`corr` |
| `regr_r2` as a window | in, built (M5.4) | the squared correlation of the simple fit, `covar² / (var_x·var_y)`, riding the same corrected two-pass and incremental sweep as `regr_slope`. Undefined — SQL NULL, not a fabricated 1.0 — where either column is flat, and clamped to [0, 1] because the correction can push a perfect fit a rounding step past 1. The only member of #77.2's scalar-reduction list with a standard SQL name; residual and fitted value have none, so under #77.1 they go to the Lua tier |
| `LAG` / `LEAD` | in, built (M5.1) | positional, not aggregates: they copy a neighbouring row, so the output keeps the **source column's type** — a lagged `BIGINT` stays `BIGINT`, because a nanosecond stamp is past 2^53 where `f64` stops being exact. Frameless (standard SQL gives them no frame; a frame clause is refused, not ignored); optional offset defaults to 1; the third `default` argument and symbol columns are refused by name |
| `RANGE` frames | in, built (M5.1) | bounded by ordering-key **value**, in the key's own units (no `INTERVAL` type: a 5-minute span over ns stamps is `300000000000`). Ends at the current row's **last peer**, per standard SQL, so tied rows share one window. Answers via per-frame recompute today; the incremental sweep over these bounds is the tracked follow-up |
| star-schema equi-joins (`INNER`/`LEFT`) | in, built | structural-fact rule; gathers only the dimension columns the query reads |
| `ASOF LEFT` / `ASOF INNER JOIN` | in, built (M5.2) | the hybrid — see *The M5 ruling batch*, item 2, and its build note. Each fact row takes the most recent of its key's dimension rows on the two **declared ordering keys**; an explicit inequality is validated, not obeyed. Ties on the dimension's clock go to the last row in storage order. The dimension side is indexed in memory, not co-walked — see the join-constraint note |
| ordered-merge relatives beyond `ASOF` (`LT`, `SPLICE`, …) | in, later | nothing coined; the same lift mechanism serves them |
| `UPDATE`/`DELETE` | in, built | tombstone + reinsert |
| DDL (`CREATE TABLE`), `INSERT`, bulk import | in, built | #39; `BIGINT`/`DOUBLE`/`SYMBOL`, `ORDERING KEY` constraint; `VARCHAR`, `PRIMARY KEY` and the retired `KEY` spelling refused with teaching errors |
| non-correlated subqueries / CTEs | in, later | named subplans |
| `UNION ALL` (then `UNION`) | in, later | low priority |
| correlated subqueries | **out** | the road to a cost-based optimizer — settled no |
| string production (`CONCAT`, `CAST AS VARCHAR`, …) | **out** | numeric-or-key invariant |
| `DISTINCT` over window/aggregate projections | out until asked | refused loudly today |

## The M5 ruling batch (all ruled by the Human, 2026-07-29)

Thirteen decisions closed in one sitting, recorded here as one dated
block so none of them lives only in a conversation. Each is design;
none is built unless its row in the stdlib table says so.

1. **Time bucketing (F1) = monotone integer arithmetic on the ordering
   key.** `GROUP BY ts / 60` (bucket index) and `GROUP BY (ts / 60) * 60`
   (bucket start) are admitted — the planner proves monotonicity
   structurally (ordering key, positive integer literal, `/` or `*`),
   so grouping streams with O(1) state and no hash table. No `bucket()`
   function is coined; unit sugar, if ever, is a later question.
   Everything else in `GROUP BY` keeps the teaching error. Rejected:
   general expressions (kills the streaming dividend, drags in
   float-equality group identity); refusal (concedes the workload's
   most common query). `FIRST`/`LAST` aggregates (OHLC's open/close)
   surface a naming sub-ruling when M5.3 builds this.

   **Build note (M5.3, 2026-07-30).** Built, and the streaming dividend
   with it — but the claim above needed narrowing to what the code
   earns:

   - **"O(1) state, no hash table" is now "the open bucket's state".**
     Once a bucket is left it cannot come back, so its groups close and
     reduce to cells there and then; what stays live is the groups
     *inside one bucket* (the symbols trading in this minute), not
     every group the query will produce. The result itself, and the
     keys labelling it, are one per group either way — so the honest
     claim is that the accumulator state stops scaling, not that
     memory does. Measured on a 200,000-row fixture over 160,000
     groups: **40.4 MB streaming vs 66.7 MB hashing, a ratio of 1.65**
     (`bucket_grouping_memory`, 2026-07-30). Truly constant state
     would need streaming *output* too, which is #88.
   - **Unordered data falls back rather than refusing.** Whether the
     grouping can stream is read from segment metadata before a byte
     is touched, so dispatch stays structural — but a table whose
     order an `UPDATE` disturbed takes the hash path and gets the same
     answer more expensively, with `compact()` restoring the fast
     path. Correct always, fast when the data behaves: the same
     bargain tombstones make. Refusing was considered and rejected —
     it would make a correction change which queries *run*, not just
     how fast they run.
   - **`/` between integers truncates**, which is ISO and PostgreSQL;
     `//` is accepted as a synonym because DuckDB spells it that way.
     In this position only one meaning is available (a `DOUBLE` cannot
     key a group), so accepting both costs nothing. It does constrain
     #40: when exact integer arithmetic reaches projection, `ts / 60`
     must truncate there too, or one text means two things.
   - **`GROUP BY` may name a bucket by its SELECT alias**, as
     PostgreSQL and DuckDB both allow. Narrow by design: only aliases
     of buckets substitute.
   - **`FIRST`/`LAST` = the de-facto TSDB names** (the pending
     sub-ruling, closed (a) 2026-07-29). Positional on the time axis,
     not on row order — the group-level counterpart of `LAG`/`LEAD` —
     so a late-arriving row cannot become "last" by arriving last.
     Ties on the clock go to the last row in storage order, the rule
     the as-of join follows; nulls are skipped, as every other
     aggregate here skips them, which coincides exactly with DuckDB's
     `arg_min`/`arg_max` and is how the differential checks them.

   **Cross-sectional partitioning — a later ruling (2026-07-30), built
   in the same milestone and recorded here because it completes the
   same axis.** The transpose of `PARTITION BY sym`: partitioning on the ordering
   key, or a monotone bucket of it, gives each row its own *instant*
   across every symbol instead of its own symbol across time. An
   unordered window (`OVER (PARTITION BY ts)`, no `ORDER BY`) takes its
   whole partition, which is standard SQL and is exactly a
   cross-section; a frame clause beside it is refused as the
   contradiction it is. Several terms intersect — `PARTITION BY sym,
   ts / 60` is one partition per symbol per bar.

   `PARTITION BY` admits any `BIGINT`, not only symbols; `DOUBLE` is
   refused because float equality is not partition identity (the same
   reason F1 rejected general expressions), and bucket arithmetic stays
   ordering-key-only, where monotonicity means something. Which column
   is named decides the direction, and which path runs is decided by
   declared structure plus segment metadata — never by cost.

   *Note the asymmetry, deliberate and flagged:* `GROUP BY` stays
   restricted to the ordering key (F1's ruling), while `PARTITION BY`
   admits any `BIGINT`. Different clauses, different rulings; revisit
   F1 if the difference ever bites.

   **The idiom this exists for needed a second thing.** A
   cross-section is only useful if a row can be compared *to* it, and
   that meant scalar expressions over window results (#94, built the
   same day): `x / sum(x) OVER (PARTITION BY ts)` is the weight, `x -
   avg(x) OVER (...)` the demeaning. Window calls are hoisted out of
   the scalar at lowering and computed first — standard SQL's own
   evaluation order, and forced rather than chosen, since a partition
   spans segments while a scalar walks one at a time. The same change
   made `x - lag(x) OVER (ORDER BY ts)` and the rolling z-score
   expressible.
2. **The as-of join (#65) — the hybrid.** Grammar from ClickHouse,
   authority from the schema: the single `ASOF` token is lifted
   pre-parse (byte-span splice, comments skipped — the hardened
   mechanism), and the remainder parses as a plain join. `ON` only, no
   `USING`. The time axis is the two tables' **declared ordering
   keys** — implicit by default; an explicit inequality is permitted
   and **validated, not obeyed** (naming anything else is a teaching
   error; the operator selects `>=` vs `>`). **Bare `ASOF JOIN` is
   refused**: write `ASOF LEFT JOIN` (keep unmatched facts,
   null-padded) or `ASOF INNER JOIN` (drop them). Recorded principle,
   reusable: *where vendors agree, follow convention; where vendors
   genuinely diverge (bare as-of semantics do), refuse and make the
   user say it.* `TOLERANCE` is cut — expressible today via
   `CASE`/`WHERE` arithmetic; reopens as sugar if desks ask. Parser
   facts (verified 2026-07-29): sqlparser 0.62 parses only Snowflake's
   `MATCH_CONDITION` form; DuckDB accepts only its `ON` form; the sets
   are disjoint — hence the lift-and-plain-join design. Evidence when
   built: the DuckDB differential (the harness *generates* DuckDB's
   spelling from structure) plus a vanilla-SQL definitional reference
   (the row with `MAX(ts) <=` the fact's), which checks the definition
   rather than another vendor's implementation. The executor is an
   ordered co-walk gated on `is_ordered()` for **both** sides.

   **Build note (M5.2, 2026-07-30).** Built, and the evidence landed
   as ruled: seven differential families whose oracle side is a
   correlated scalar subquery in vanilla SQL — the definition of "the
   latest quote at or before" — rather than DuckDB's own `ASOF JOIN`.
   Three things about the build differ from or extend the ruling, and
   are recorded here rather than left in a conversation:

   - **Not a co-walk.** The executor indexes the dimension side per key
     (an ascending `(clock, row)` list, stably sorted) and binary-
     searches it per fact row. That is correct whatever order the data
     arrived in — a late-arriving quote still matches — so it needs no
     `is_ordered()` gate on either side, where a co-walk would have to
     refuse a transiently disordered table. The cost is that the
     dimension materializes: the streaming property clause 2 of the
     join constraint claims is **designed, not built** (#92, and see
     the build note there).
   - **Ties on the dimension's clock go to the last row in storage
     order** — the same "newest version wins" rule corrections follow.
     The alternative (refusing a duplicate `(key, clock)`) was rejected:
     a quote table legitimately prints twice on one stamp, and kdb+'s
     `aj` takes the last such row too. The differential covers this on
     purpose: the fixture injects per-symbol tied timestamps and the
     oracle counts them before trusting the families.
   - **The inequality's sides are assigned by qualifier, not by
     operator.** `t.ts <= q.ts` and `q.ts <= t.ts` are the same
     operator and opposite questions; reading the operator alone would
     have answered the first (the quote *after* each trade) with the
     one before it. Written backwards, it is refused.

   One limitation the build meets rather than creates: both tables are
   timestamped, and a dimension attribute sharing a fact column's name
   is refused, so `quotes.ts` beside `trades.ts` must be renamed. That
   is the pre-existing equi-join rule, not a new choice — the open
   decision about whether to change it is #93.
3. **Library naming (#77.1): SQL exposes only operations bearing
   standard names.** `var_pop`, `stddev_pop`, `LAG`/`LEAD` pass into
   SQL freely; EWMA, `diff`, multi-factor regression have no standard
   SQL spelling and stay script-side until individually named by the
   Human. The rule is mechanical — no per-op judgment.
4. **Matrix-valued results (#77.2): scalar reductions only in SQL**
   (R², residual, fitted value); full vectors/matrices flow through
   the API and scripts, which receive them from one evaluation.
   Per-component SQL functions rejected; multi-output projection
   plumbing deliberately unbuilt until demanded.

   **What that meant at build (M5.4).** Item 3's mechanical rule
   decides which of the three reach SQL, and only one does: `regr_r2`
   has a standard SQL name and shipped; residual and fitted value have
   none, so they are script-side. Read items 3 and 4 in that order —
   item 4 says a scalar reduction *may* enter SQL, item 3 says only a
   standard name *does*. As first written item 4 read as a promise of
   all three, which item 3 forbids.
5. **The prelude (#77.3): compiled into the binary**, `.prelude`
   prints the source — single-file deployment holds; read-copy-modify
   is preserved by printing, not by an editable side file.
6. **Library build order (#77.4): streaming tier first**, matrix tier
   second (scheduling default, delegated).
7. **`DELETE` consumes a knowledge coordinate.** Decided on
   *stability*: an unconsumed kill coordinate is shared with the next
   append, so the delete's effect has no cut of its own — a recorded
   boundary drifts as data arrives. Consuming makes every knowledge
   event own one coordinate and makes `ASOF next_sequence() - 1` the
   universally stable "latest" idiom. Cost accepted: a table's first
   `DELETE` diverges it (the sequence column materializes;
   delta-codes to almost nothing). A second cost surfaced in the
   build and was ruled accepted too (2026-07-29, option (a) of the
   recorded three): a persistent `DELETE` flushes the write buffer
   first — recovery must not renumber rows across the consumed
   coordinate — so interleaved delete/append workloads seal small
   segments until compaction merges them. Reopen trigger: a real
   workload shows delete-driven fragmentation that compaction
   cadence cannot absorb; the fix on the shelf is a WAL
   consumption marker, rejected for now because it grows the most
   safety-critical code we have to optimize a verb the design says
   not to lean on.
8. **The sequence column's SQL surface (#75): a fixed-name
   pseudocolumn, `_seq`.** Never declared, refused in `CREATE TABLE`.
   Chosen by the
   visibility rule: the engine refuses `SELECT *`, so the column is
   never seen unbidden — the short system-side name wins over the
   spelled-out one. Kill-coordinate exposure deferred until asked.
9. **`SYMBOL` replaces `KEY` as the column type's DDL spelling.**
   kdb+/QuestDB lineage the audience reads on sight, and it ends the
   word KEY serving two grammatical roles in one statement (`ts BIGINT
   ORDERING KEY, sym KEY`). Spelling only: the stored format and the
   internal type are unchanged.
10. **Symbol columns are officially unordered labels (#58 = B).**
    `ORDER BY` on a symbol column becomes a teaching-error refusal.
    The deciding facts: codes are per-segment first-appearance ranks
    (no usable inherent order exists), and byte-order "alphabetical"
    is honest only for ASCII — an engine that refuses to produce a
    string does not rank them. Identities are never ordered: the
    arithmetic refusal and this one are the same rule. Differential
    families stop using `ORDER BY sym`; the referee sorts rows before
    diffing. The `WHERE sym > '…'` question and the collation
    question both dissolve.
11. **Compression (#42): ALP and ALP-RD together**, plus the integer
    sibling sharing the same backend — frame-of-reference +
    bit-packing for non-key `i64` columns and `u32` symbol codes.
    Corpus tick-size realism is step 0.
12. **Arrow-boundary booleans: refused loudly at ingest** (the
    teaching error names `df.astype({'flag': 'int64'})`). Recorded
    with the Human's explicit flag: *ruled wrong on purpose* — a
    standing revisit covers a real boolean type **and** the
    logical-annotation mechanism itself (`TimestampNs`-style
    "physically i64, logically X"). Storage fact settled for that
    revisit: a nullable boolean is 2 bits/row (value bit + validity
    bit) — the cost was never storage, it is the seam sweep.
13. **Python distribution (M5.5): wheels on PyPI** — for the Python
    binding only. The console remains a single Python-free native
    binary; the engine and console depend on nothing.

Also recorded from the same sitting: regex on symbols (#57) is
**deferred by ruling** — the menu on the issue now includes the house
option (registered single-symbol Lua-pattern predicates in `WHERE`,
evaluated once per distinct dictionary value); and `IS NULL` /
`IS NOT NULL` was found missing from the predicate fragment — standard
SQL, in scope, and built immediately after.

**Built since (2026-07-29).** Items 7, 8, 9 and 10 are code, not
plans: a delete consumes its coordinate and reopen recovers the spent
one from the delete logs; `_seq` reads a row's birth coordinate back
through SQL; `SYMBOL` is the DDL spelling and `KEY` is refused with a
pointer to it; `ORDER BY` on a symbol column is a teaching error and
the differential families that leaned on it now diff as sets. Two
scope notes belong with them. `_seq` is **projection-only** — it can
be selected, aliased, ordered and paged by, but not filtered or
grouped on, because `AS OF` is how a coordinate filters and a second,
weaker spelling of that would be worse than none; whether predicates
should reach it is a live question, not a settled no. And the
kill-coordinate column stays deferred, as ruled.

## The residency design (ruled by the Human, 2026-07-30; built the same day)

**The ruling.** Tables bigger than memory are handled by **segment-granular
lazy residency (option b)**: an open reads metadata only, segments decode
on first data access, and decoded segments are retained as a cache under a
byte budget, evicted least-recently-used. The prune-metadata lives in **a
manifest section (tag 1)** — one record per live segment carrying its
name, row span, ordering flag, sequence summary, and zone maps, written by
every flush (segment file → manifest → WAL reset) and by compaction's
commit. The manifest is thereby the authoritative segment list; a stray
segment file (a crash inside the flush window) is never adopted, its rows
recovered from the WAL. Legacy manifests fall back to the backend scan and
earn the section at the next writer open.

**Rejected alternatives.** *Document the ceiling* (nothing built): the
append-heavy ledger is exactly the shape that outgrows RAM — disqualifying.
*Query-scoped streaming with no cache*: every query re-pays decode
(31–93M values/s per #42's run), turning the ~120µs hot-window query into
~10ms+ — wrong trade for an access pattern predictably skewed to recent
data, which is precisely what LRU serves. *Column-granular residency*:
a refinement, not an alternative — deferred with its reopen trigger to
#87. *Metadata via ranged reads + segment-header parse*: forces a
per-section checksum redesign now (the whole-file CRC cannot verify a
partial read), and in a browser costs one range request per segment just
to plan, where the manifest is a single small fetch — the WASM future
argues *for* the manifest section, not against it. Ranged reads travel
with column granularity in #87, when partial-segment fetch earns its
checksum revision.

**The budget's contract.** `StoreOptions::cache_bytes` (engine
`open_read_only_with_cache`, console `--cache MiB`), surviving a reader's
refresh. It is **advisory over retention, never over correctness**: a
segment an `Arc` still pins (a snapshot, a running query) is never
evicted, so peak memory is the budget plus the largest concurrent working
set. The interim default is unbounded — today's behavior exactly; the
default's final value, and a strict-refusal mode, are deferred to #87.
The bound's sharp edge is recorded as #88: the executor materializes
every surviving segment for a query's lifetime and zero-copy outputs pin
inputs, so an unpruned full-table scan's working set is the table itself;
streaming aggregation is the recorded follow-up. Compaction likewise
materializes everything by construction (#82's territory).

**What a refresh keeps.** The F4 reader's refresh reuses the previous
open's slot context wholesale — same cache, same decoded segments — for
every name the new manifest still carries (files are immutable within a
generation; history files always). Resident stays resident,
pointer-equal, zero re-reads.

**Reversal class.** Additive: the eager path is the lazy path with every
fault taken at once, the format change is one skippable manifest section,
and old binaries scan as before.

**Evidence.** Counting-backend tests pin each claim (a recorded open
reads no segment files; a pruned segment's file is never read; the budget
evicts cold, never pinned; refresh keeps decoded state); a forty-segment
table under a four-segment budget answers six query shapes
batch-identical to an unbounded open; the crash window between segment
and manifest writes is pinned end-to-end by injected failure; all six
differential oracles then in the gate pass over the faulting path. Corruption moved with
the design: a bad segment file is loud at first fault, not at open.

## Maintained views (#83, ruled by the Human 2026-08-02; tranche 1 built the same day)

The first continuous-query build — derived data kept fresh on ordered
append, correct across corrections. The research survey, the option
menu, and the full ruling set live on issue #83; this section records
what was ruled, what shipped, and what the evidence earned. The scoping
principle behind all of it: **the field's three answers to "what does a
maintained result do when its input is corrected" — retraction deltas,
invalidation + repair, refuse/rebuild — compose here because the
knowledge axis already gives base and view one shared version
coordinate.** (Prose says "maintained view"; the API type is
`MaterializedView`. The view's **stamp**, used throughout below, is
the source-table ingest-sequence watermark below which the
materialization is complete.)

**The ruling set** (each with its rejected alternatives recorded on the
issue): eligibility is **(c) the full reach, taken piecemeal** —
tranche 1 is bucketed single-table views, tranche 2 running/cumulative
shapes via bucket-partials, tranche 3 joins (q-hierarchical only, per
the PODS 2017 dichotomy); correction semantics is **the versioned view
with uniform repair** — every correction marks its bucket, repair is
always re-fold-from-base, the class-split delta fast path rejected for
v1 (a second code path plus an f64 subtraction hazard for a path
corrections are too rare to need); reads are **the union read** —
materialized clean buckets plus a live fold of everything the stamp
does not cover, so the view is semantically always exact and repair
only shrinks the live half; **`AS OF s` on a view recomputes**
`Q(base AS OF s)` — the materialization accelerates current reads and
is never the authority; the surface is **engine API first**
(`create_materialized_view` / `refresh_view`, the `register_window`
pattern), SQL DDL after behavior is proven; storage is **a real table**
plus a CRC'd definition record carrying the stamp.

**The model as built.** A view is a fold over the ingest sequence,
stamped with the source watermark below which the materialization is
complete. The dirty list is derivable state — buckets touched by any
coordinate the stamp does not cover, re-derived from the knowledge
history machinery M4.4 built — so the stamp is the only durable view
state, and it is written strictly after the materialization it
describes is flushed. Refresh flushes the source first: the stamp
asserts durability, so everything it covers must survive any crash the
source's own WAL contract admits (the alternative — stamping buffered
rows — left permanent ghost buckets when a crash rewound the source;
the repo-wide code review found it, and the ghost test replays it). A
stamp found *ahead* of the source (a swapped directory, a tampered
record — impossible from a crash under this discipline) meets the
rebuild floor: every materialized row out, one full fold in. A
read-only process (F4) serves exact view answers with no writes: the
union read needs none, and an older (stamp, materialization) pair only
means more live work, never a wrong answer.

Two permanent restrictions, both definitional: a view definition may
not read across knowledge time (`AS OF` / `_seq` in the definition
breaks `view AS OF s = Q(base AS OF s)` — snapshot reducibility), and
`_seq` *of* a view is refused (a view row summarizes many source rows
and has no single ingest coordinate). `ORDER BY` / `LIMIT` /
`DISTINCT` / `HAVING` are refused in definitions because a view is a
table — they compose at read.

**Evidence at the build** (2026-08-02, this container): the subsuming
property — view equals recompute at the current knowledge coordinate,
whatever the history (past coordinates are exact by construction: they
recompute) — holds at each of 160 states along one seeded
pseudo-random interleaving of append, update, delete, and refresh,
with one mid-run compaction; plus the seventh oracle family
(`m5_view_oracle.py`, in CI): every statement mirrored into DuckDB and
the view's answer diffed against from-scratch recompute at eleven
scripted checkpoints spanning stale, fresh, corrected, compacted, and
reopened states.
The scaling claim is measured: at 4× the table with a fixed 2,000-row
batch, refresh cost is flat (0.78ms vs 0.85ms, ratio 1.09, guarded
< 2.5 in `perf_sanity`) while full recompute scales 32ms → 122ms; the
union read's staleness premium is dominated by the live fold of the
tail (1.0–1.7ms vs 0.24–0.31ms fresh across the two table sizes — the
staleness-premium check the read-semantics ruling asked for, affirming
it).

**Costs and seats, stated plainly**: each refresh flushes one small
segment on the view (and possibly one early freeze on the source when
called mid-buffer; at the freeze-boundary cadence the flush is free);
refresh also scans compacted correction *history* unconditionally —
its kill coordinates live in the segments, not the metadata, and an
additive manifest field removes the scan if it ever measures hot (the
flat-refresh measurement above is a correction-free run) —
`compact` on the view restores contiguity; the console opens existing
views correctly (both scan sites route on the definition marker) but
has no verbs to create or refresh them yet — the API-first ruling's
deliberate gap.

### Tranche 2: running and cumulative shapes via bucket partials (built 2026-08-03)

The shapes tranche 1 refused because their blast radius under
correction is unbounded — a correction at `t` changes every result
after `t` — are admitted by changing what is stored. The
materialization holds, per **hidden bucket** of the ordering key, not
the answer but the **partials** the answer recombines from: a bucket's
sum, count, (sum, count) for `AVG`, min, max, or edge value. The
load-bearing fact of the whole representation: **partials and their
combines are themselves built aggregates**, so the synthesized
materialization is a legal tranche-1 bucketed plan and every piece of
tranche-1 machinery — refresh, touched-bucket derivation, the stamp,
the crash story, the rebuild floor — serves both new shapes unchanged.
A correction re-folds one hidden bucket; the O(suffix) rewrite never
exists because no suffix is stored.

**The hidden bucket width** is a heuristic, not a semantic: chosen at
the first refresh that sees data (observed key span over a target 1024
buckets, clamped to at least 1), persisted in the definition record
(format v2, additive; v1 records decode as width-unchosen and
self-heal) *before* folding under it, so a crash re-folds under the
same width. A re-widthing is a rebuild, deliberately not a format
question.

**The running read**: partials union (clean materialized buckets + a
live partial fold of everything the stamp does not cover) → a
symbol-keyed **combine** reassembling cross-bucket totals → finalize
into the user row shape — `AVG` divides once, after the combine (an
average of averages weights buckets, not rows, and is simply wrong);
`COUNT`'s NULL sum-of-counts grounds to 0.

**The cumulative read** splits every expanding window at the query
predicate's ordering-key lower bound (conservatively extracted: `AND`
takes the tighter branch, `OR` needs both and takes the looser,
unhandled shapes fall to recompute — a bound may sit below the truth,
never above it): a **boundary** combine over the partials strictly
below that bucket, an **assembly** of the user definition over the
source from the bucket's low edge (truncating division is monotone, so
the two ranges partition exactly), and a per-column adjustment folding
boundary into assembly — `AVG` through hidden sum/count helper
windows, never through its quotient; `MAX` propagates NaN under the
engine's NaN-greatest relation. A query with no lower bound wants
every output row and recomputes: the partials cannot shorten an
O(n)-row answer.

**The combine contract, stated** (2026-08-03, revisitable): combining
per-bucket f64 sums associates differently than a single pass, so
`SUM`/`AVG` through partials agree with recompute within **1e-12
relative** — the tolerance every DuckDB oracle family applies; both
folds use compensated summation. `COUNT`/`MIN`/`MAX`/`FIRST`/`LAST`
combine exactly. Exact single-pass equality is impossible under any
partials representation.

**Refusal parity, inherited**: cumulative reads run real windows, and
windows refuse disordered data — so a full read over uncompacted
correction segments refuses exactly as the base's windows do
(`compact` heals both), and `view AS OF s` refuses once corrections
sit in history segments (their key ranges interleave with the live
generation's). `view AS OF s = Q(base AS OF s)` includes the
refusals. A *ranged* read above an uncompacted correction keeps
answering exactly — zone maps prune the stray segment and the boundary
re-folds with aggregates, which need no order.

**Evidence at the build** (2026-08-03, this container): the m5 oracle
grew to three views over one source — bucketed, running, cumulative —
diffed against DuckDB recompute at all eleven checkpoints (the
cumulative full read's refusals are themselves asserted, by reason);
a 4096-row dense battery forces width 4 so multi-row hidden buckets
are value-checked (FIRST/LAST inside a bucket, mid-bucket range
floors, an OR-predicate bound, one-bucket repair at width 4);
bucket-edge crossings cover negative keys and truncation's
double-width bucket 0. Pricing, measured as same-run ratios
(`perf_sanity`, 2026-08-03 run): a one-row correction on a running
view over 1M rows repairs in 1.6ms vs 140.8ms full recompute (ratio
0.011, guarded < 0.1); a cumulative ranged read of the last 10k of 1M
rows costs 33.4ms vs 156.5s for the full read (ratio ~0.0002, guarded
< 0.05) — the full read pays the executor's quadratic expanding-window
sweep, which the ranged read never touches.

**Tranche-2 costs and seats**: the executor's expanding-window sweep
is O(n²) in the frame lengths — an incremental sweep in `query-lite`
would fix the cumulative full read (and every plain expanding-window
query) and holds a seat; a view read resolves registered functions
from the view's own always-empty registry (register on the base,
query the base — a per-view registration surface is a held seat);
names beginning with `__` are reserved in running/cumulative
definitions for the minted hidden columns; tranche 3 (q-hierarchical
joins) holds its seat with the teaching refusal naming it.

## Things that are settled "no"s — don't relitigate without a specific trigger

- **A boolean column type — no for now, with the Human's explicit
  revisit flag (2026-07-29: "ruled wrong on purpose").** Flags are
  `BIGINT` 0/1; predicates never materialize; Arrow booleans are
  refused at ingest with a teaching error naming the one-line cast.
  Not a performance or WASM question — a type multiplies against every
  seam (formats, value map, predicates, aggregates, oracles) and buys
  nothing 0/1 lacks. The standing revisit covers the type AND the
  logical-annotation mechanism (`TimestampNs`-style); storage is
  pre-settled for it: a nullable boolean is 2 bits/row.
- **Compiled Lua C extensions** (`package.loadlib`). Pure-Lua libraries are
  fine and need no special handling. (See *The Lua layer* below for the full
  reasoning.)
- **A LAPACK dependency, at all, until an op needs more than two
  parameters or two dimensions.** Not "a general LAPACK surface" — any
  LAPACK surface. Every statistic the engine computes today has an exact
  closed form at the size it needs, and the removal is what frees the WASM
  build from a LAPACK-in-WASM layer that does not exist. When a wider op
  is committed, the rule that governed the old curated set still governs
  its replacement: don't add routines because LAPACK has them; add them
  because a named workflow needs them. See *Curated compute: what the
  engine calls, and why*.
- **Autodiff / a Torch-style tensor framework.** Different computational
  paradigm than anything the target workload (closed-form / classical
  numerical methods) needs. If a specific, repeated, real need shows up
  later, it gets a narrow scoped addition, not this whole paradigm.
- **Building out a "scientific ecosystem"** (e.g. Julia's
  DifferentialEquations.jl-style breadth) to compensate for Lua's thinner
  ecosystem. Not this project's job — the embedded Lua scripting layer is
  the intended escape hatch for gaps, not something we pre-fill.
- **A general query optimizer / cost-based planner.** Join strategy is
  fixed by input structure — a small materializable side, or co-ordering
  on the join key — never chosen by cost (see *the join constraint,
  completed*).
- **Arrow IPC / Flight / Parquet in `arrow-lite`.** The interop surface is
  the C Data Interface (including the stream variant), nothing else — IPC
  drags in FlatBuffers and a much larger spec. Parquet in/out is the
  application's job via ecosystem tools that already speak C-Data.

If something on this list seems newly justified, that's a conversation to
have explicitly (update this document and its companions together), not a
decision to make silently inside an implementation PR.

## Curated compute: what the engine calls, and why

Compute sits behind trait boundaries the `engine` calls through, so native
implementations can eventually be joined by WASM ones without changing
anything above. Today there is **one** compute library crate plus the
script layer:

- **`compute-linalg`** — multiplication-class primitives (dot,
  matrix–vector, matrix–matrix). Direct consumers: Lua scripts through
  `compute-lua`'s registered host functions, and eventually the
  executor's numeric inner loops (still profiling-gated). Pure Rust — a
  source-fixed loop for `dot`, faer for the matrix products — so one
  implementation serves native and wasm32 (see the decision record
  below).
- **`compute-lua`** — the scripting tier and the promotion ladder's first
  rung (see *The Lua layer*).

**Decision record — no LAPACK on the query path (2026-07-27).** The
`compute-lapack` crate is removed and the engine links no LAPACK routine.
The reason is a measurement, not a preference: **LAPACK's value scales
with parameter count, not data size**, and every statistic the engine
currently exposes is two-dimensional. A two-parameter least squares and a
2 × 2 symmetric eigenvalue both have exact closed forms costing a handful
of flops, while a general solver is dominated by its own per-call
overhead at window scale — measured at roughly 2.3µs of `regr_slope`'s
2.5µs per 64-row window, and 0.68µs per window for `dsyev` on a 2 × 2.
Replacing both with closed forms left `regr_slope` costing the same as
the other two-pass window statistics, where it had cost ~11× more, and
moved it from 5× behind DuckDB's `regr_slope` window to **3.3× ahead**
(`m2_compute_latency_bench.py`, run 2026-07-27, release, container
hardware, 20k rows, window 64).

*What this bought beyond speed:* the engine no longer requires a system
LAPACK to build or embed, which repairs the link-it-in-like-SQLite
property, and **the WASM milestone (M6) no longer waits on a LAPACK-in-WASM layer** — the
current feature set can reach WASM parity without one.

*The closed forms are the corrected ones, and that distinction is
load-bearing.* The rolling regression uses the corrected two-pass form
(Chan–Golub–LeVeque), which carries `Σ(x − x̄)` rather than assuming
centering left it exactly zero. Measured against the SVD answer before
the removal (`measure_closed_form`, release, container hardware,
2026-07-27), worst predicted-y drift over the data:

| design | QR (`dgels`) | corrected | naive |
|---|---|---|---|
| 64-row window, x offset 1e9 | 1.07e-14 | 2.84e-14 | 8.31e-7 |
| 64-row window, x offset 1e12 | 1.42e-14 | 2.49e-14 | 1.01e-3 |
| near-degenerate, spread 1e-10 | 2.84e-14 | 1.99e-13 | 6.75e-7 |

The corrected form tracks QR within a small constant factor — both at the
float noise floor against `|y|` of order 10–200 — while the **naive** form
is a real regression at timestamp-scale offsets, bug #45's regime. That
comparison needed LAPACK as its reference and cannot be re-run now that
the dependency is gone; what guards the property going forward is
`regression_numerics` in `engine`, which checks the shipped form against
a cancellation-free reference computed about `x[0]` and asserts the
correction beats the naive form on irregular offset data.

*Reopen trigger:* the first committed op needing **more than two
parameters or two dimensions** — multi-regressor regression, PCA beyond
2 × 2, portfolio solves, Cholesky. No closed form exists there, and a
solver-class backend comes back behind the same capability-negotiating
trait shape — the measured candidate is faer's solver family, which beat
reference LAPACK's `dgels` at k = 2–4 in the same-run three-way
measurement (see the kernel decision record below) and compiles to
wasm32. The engine's ops should then dispatch on parameter count:
closed form at two, a solver above it.

**Decision record — where non-standard compute lives: the Lua tier,
via MatLua (Human-ruled 2026-07-30).** The reopen trigger above fired
at M5.4: multi-factor regression is the first committed op needing more
than two parameters. The answer is **not** a faer dependency of our
own, and **not** new SQL names. SQL stays standard — item 3 of the M5
ruling batch, unchanged — and a user who wants more than the standard
spells turns to the Lua tier, where [MatLua](https://github.com/andy-emerson/MatLua)
is the matrix and linear-algebra vehicle. Two things follow. TallyDB
does not coin `regr_multi`, `pca`, or their relatives into SQL: had we
built the solver ourselves, the pressure to expose it would have been
immediate, and item 3 would have had to bend. And TallyDB does not take
faer directly: MatLua already depends on it, and Cargo unifies
semver-compatible versions, so reaching linear algebra *through* MatLua
costs no second copy.

*Status: ruled, not built.* Nothing in the tree depends on MatLua
today. A requirements letter is out — what would break TallyDB if
MatLua chose otherwise (a Lua face that works against a host-owned
interpreter, no `Drop` value live across a `longjmp`, no panic across
the C boundary, `i64` exactness with no implicit widening at the
boundary, a documented contract for absence, and an Arrow **C Data
Interface** path so neither side links the other's Arrow stack) versus
what is theirs to decide (NaN or mask internally, indexing, which
factorization backs `lstsq`, behaviour on singular input, error
taxonomy, dtype order). The split follows the Human's standing
principle: **a decision made ad hoc to an emerging need while building
our own tools is revisitable; only a decision that would undermine what
TallyDB *is* is not.** MatLua are the linear-algebra experts; we are
the time-series-database ones, and where the two overlap we take their
design.

*What could reopen this:* MatLua declining the embedding or exactness
requirements, at which point the choice returns to a faer dependency of
our own with the SQL surface still frozen at standard names.

The design-critical part survives the removal and must keep surviving:
compute stays behind **distinct traits with independently gated
backends**, and **capability negotiation** stays first-class — "this op is
unavailable on this backend" is a returnable answer, not a panic. That is
what keeps a WASM build from being a rewrite, and what lets a solver-class
backend return later without touching a caller.

**Decision record — system BLAS replaced by pure-Rust kernels
(2026-07-27).** `compute-blas` (system BLAS via FFI) is now
`compute-linalg`: the same trait seam, implemented in pure Rust — a
strict left-to-right loop for `dot`, faer (slim features: no thread
pool, no RNG, no file formats) for the matrix products. A three-way
measurement (plain Rust vs system reference BLAS/LAPACK vs faer,
release, container hardware, run 2026-07-27) drove the split: at window
scale (≤ 64 elements) the plain loop beats both libraries — there is
nothing to amortize — while faer wins long dots by 2.4–4.7×
(256–4096 elements) and Gram-shaped products by 3.7–10× (k = 4–16 over
64 rows), shapes where reference `dgemm` also loses to it. Accuracy is
a wash: identical least-squares residuals, eigenpair residuals at the
1e-16 floor on both sides. What the swap buys is packaging: no system
math library to install, link, or version (CI drops `libblas-dev`;
`ldd` on the engine library shows none), and the whole compute stack
compiles for `wasm32-unknown-unknown` — verified — so `blas.wasm` is no
longer a TallyDB dependency at all. The `dot` loop is the one kernel
whose result is bit-identical on every CPU and target, and deliberately
the only kernel on a per-window path today.

*Rejected alternatives:* keeping system BLAS — wins no measured shape
the engine runs and costs the system dependency; reopen if a platform
BLAS (Accelerate, MKL) is measured materially ahead at a shape that has
reached a query path. OpenBLAS pinned from source with
`TARGET=SANDYBRIDGE` — the old determinism plan; heavier to build,
still a C dependency, and source-fixed Rust loops achieve the property
more directly for the paths that promise it. *Reopen trigger for the
dot split itself:* profiling showing a long-vector dot on a hot path —
then the loop yields to faer above a measured length threshold, trading
bit-portability knowingly.

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

*Updated 2026-07-27:* the factorization is gone — the window solves in
closed form (see *Curated compute*) — and the centering it required
remains, now in **corrected** two-pass form in every window statistic
(`RollingRegression` and `PairStatistic` alike; the uncorrected form
carried up to 4.9e-8 relative error at a 1e12 offset). Accuracy is
judged against a compensated high-precision reference and enforced in
CI by `window_numerics_guard`: every shipped window statistic must
track that reference within 1e-12 relative over corpora spanning
offsets to 1e12 and a drifting monotonic ordering key — the shipped
form measures 1–2e-15. The streaming alternative was measured against
the same reference (`measure_3b`, release, container hardware): naive
running sums are ~8× faster and reproduce exactly the failure this
record predicted (`eigen_max` off by 9.6e7 at a 1e12 offset,
`corr`/`slope` undefined where the recompute has an answer) — rejected
permanently. A **shifted** variant — moments kept about a value near
the data, accumulator rebuilt every window-length — keeps ~7× and sits
at 5e-15–1.1e-14, marginally less accurate than the corrected
recompute but still at the noise floor.

*Shipped 2026-07-27 (#72).* The sequence seam exists — a defaulted
`evaluate_frames` on `WindowAggregate`: the executor hands each
aggregate one contiguous run (the snapshot, or one partition) and the
default recomputes per frame, so only overriders change behavior. The
rejected seam shapes, for the record: a separate sequence trait
(needless registry duplication) and executor special-casing of known
op names (breaks the trait boundary and duplicates the math — rejected
on sight). `PairStatistic` and `RollingRegression` override with the
shifted sweep for bounded frames; unbounded frames recompute as
before; one shared finalization keeps the NULL semantics identical on
both paths. The guard extended before the speed landed:
`window_numerics_guard` holds the incremental path — the one every SQL
window now runs — to the same 1e-12 bound on every corpus, intercept
included, and was verified to trip (1.07e-12, drifting-timestamp
corpus) with the re-anchoring rebuild disabled. Arrival numbers
(`m2_compute_latency_bench.py`, run 2026-07-27, release, container
hardware, 20k rows, window 64): `regr_slope` 0.6ms — 9.6× ahead of the
DuckDB+NumPy stack; `covar_pop`/`corr`/`eigen_max` 0.7–1.1ms —
1.2–1.6× ahead of vectorized NumPy riding TallyDB's own export, 3–4×
ahead of the DuckDB+NumPy stack. The in-engine path is now the fastest
measured arrangement for every curated statistic *and* the only one
holding 1e-12-to-truth at timestamp-scale offsets — the vectorized
peer's cumsum form is exactly this record's rejected streaming
algorithm.

## Batch, not per-row, for Lua and linear-algebra calls

Every call from the query executor into `compute-lua` or
`compute-linalg` should operate on a whole column or window per call, not
element-by-element. Per-row calls throw away the entire performance
rationale for pairing a columnar engine with these compute layers. If an API
makes per-row calls the easy/obvious way to use it, that's a bug in the API
shape.

## The Lua layer

**Decision record — the extension model (ruled 2026-07-28, from
external review).** User compute reaches the engine through **one
mechanism per host**, and the embedded interpreter serves only the
hosts that have no language of their own:

1. **Rust host → the trait.** `WindowAggregate` is the extension API:
   an embedder implements it (~20 lines) and registers the kernel on
   the table — native speed, full type safety, no interpreter. This is
   the *primary* extension path; the trait and a `register_window`
   entry are public engine surface (correcting the M2.7 state, which
   shipped only the interpreter path publicly).
2. **Python host → callbacks through the binding (M5).** Python is
   **never embedded in the engine** — ruled out on structure, not
   taste: NumPy (the thing users actually know — the familiarity is
   the library, not the syntax) is welded to CPython; CPython brings
   the process-global GIL, no viable sandbox, tens of megabytes, and —
   decisive — *circularity*, since the primary host process already is
   Python, and a library importing a second interpreter into it fights
   the first. (RustPython/MicroPython rejected: no NumPy, so the
   familiarity argument evaporates.) Instead the binding registers a
   host-side callable as a window kernel: the engine calls back into
   the host's own interpreter through the `evaluate_frames` seam —
   whole columns per call, zero-copy views in, vectorized NumPy
   inside, an array out. In-query compute in real NumPy, with no
   interpreter shipped.
3. **No host language (console; browser at M6) → embedded Lua.** The
   one territory where an embedded interpreter is non-substitutable —
   a console user cannot compile Rust at a prompt. Lua becomes a
   **non-default feature** the console (and later the browser bundle)
   turns on; library embedders opt in or never carry the C boundary,
   its sanitizer CI, or the interpreter at all. Its honest value:
   interactive kernel registration, and the measured low-latency
   niche (parity with NumPy-on-export at the newest-window shape,
   where fixed costs dominate). It is **not** the extensibility story;
   the trait is.

**The sunset clause — ruled 2026-07-28: the trial passed, the clause
dissolved.** The clause (restructured earlier the same day) held that
~32k lines of vendored C + bindings were not yet justified by Lua's
niche, so Lua would stand trial at the end of M4, once its best case
existed — the trait beneath it, the vocabulary invariant, the
vectorized whole-column slot, the compose-don't-loop idiom, and the
upstream test suite all built. The Agent brought the evidence brief
(#76): the vendored sources byte-identical to upstream v5.4.7 with
the full official suite + `ltests` torture harness green in CI, and —
after the vectorized vocabulary (option A) landed — composed column
kernels measured **ahead of the DuckDB+NumPy competitor stack**
(~2–2.5× at 20k rows after the dense fast paths), with the honest
behinds stated: element loops ~14× behind vectorized NumPy, composed
shapes still behind NumPy riding the engine's own export. **The Human
ruled Lua stays.** Consequences, all taken: SQL-in-Lua (#70) built as
M4's closing increment — the second direction earned by the first —
and the no-coined-SQL-names reopen trigger does **not** fire (the
scripting layer remains the home for novel compute names). The
rejected fail-branch stays recorded for its reopen value: removal
whole (crate, feature flag, console `.lua`), tier 3 query-only, SQL
naming re-decided. Interpreter swaps were examined for the element-
loop gap and rejected on invariants, not taste: LuaJIT/Luau descend
from Lua 5.1 — **no 64-bit integer subtype**, so the `i64` exactness
contract breaks precisely where the ordering key lives (nanosecond
timestamps exceed 2^53) — and neither targets wasm32; the recorded
element-loop answer is vocabulary completeness (more registered ops,
more rolling combinators), not a faster interpreter. The
runaway-kernel guard (#61) keeps its scoping: required before Lua
ships in any surface serving untrusted input (the M7 served product),
optional for a local console.

**The idiom: compose, don't loop (M4.2).** Lua's cost model has three
tiers, and the documentation teaches the same discipline NumPy's
culture teaches Python: (1) an element loop written in Lua pays an
interpreted dispatch per element — the 13× tier, the code smell;
(2) a kernel that *composes registered ops* (`return 2 * sumsq(x)`)
runs compiled arithmetic with one interpreter entry per call; (3) the
engine-driven paths (`evaluate_frames` for windows, `eval_column` for
column functions) enter the interpreter once per *run or view*. The
vocabulary invariant makes tier 2 grow for free — every registered
window aggregate is callable from a kernel by its SQL name,
registry-driven — and the vectorized column slot puts scripted per-row
work in tier 3.

**The invariant's edge, stated precisely (found by the M4-close code
review, 2026-07-28).** It covers window aggregates, not *column*
functions: the host-function seam a script calls through returns one
value per call, while a column function returns a whole column, so
there is no shape to install one under — `register_column_function`
and the console's `.luascalar` are SQL-callable but resolve to nil
inside a kernel. This is a real edge, not a wiring slip, and the docs
claimed it closed until the review caught them. A script wanting
whole-column work has the vectorized vocabulary (operators,
`rolling_*`) instead; widening the invariant needs a column-shaped
host seam, which is where the tranche-2 primitives (#77) would land.

Promotion is mechanical:
one registry name, a Lua implementation swappable for a trait
implementation with no query change (both pinned by contract tests).

**A history correction (same review).** The four curated statistics
were *not* produced by promoting Lua prototypes — the regressions
predate the Lua layer by two milestones. The promotion ladder is the
intended path for future kernels, not the origin story of the shipped
ones; documentation must not claim otherwise.

The embedded interpreter is **canonical PUC Lua 5.4**, compiled into the
engine from the unmodified upstream sources — the embedding model Lua is
designed around. Scripts reach the engine's buffers through zero-copy
userdata views: the userdata wraps the live `arrow-lite` buffer pointer
and its accessors are implemented on the Rust side, so no bytes are
copied. Stated precisely, in the same spirit as the compute-split's
zero-copy record above: *access* is zero-copy, but each element read is
a metamethod dispatch rather than a compiled raw load. The curated
`compute-linalg` and engine ops are exposed to scripts as registered
functions operating over those same views — sharing buffers, not
copying between them. Lua 5.4's numeric model — one number type with a
64-bit integer subtype and a 64-bit float subtype — is exactly TallyDB's
`i64`/`f64` column pair, so numeric values cross the script boundary
without losing exactness; that alignment is a load-bearing reason for
the 5.4 choice, not a convenience.

**The two directions (and only two).** Lua and the engine meet in exactly
two directions, named for which language encloses the other:
**Lua-in-SQL** — the engine calls a Lua kernel mid-query
(`my_kernel(x) OVER (…)`) — and **SQL-in-Lua** — a Lua program drives the
engine and runs SQL. A third, the **data-only baseline** (staged
`SQL → columns → Lua → columns → SQL`, neither side calling the other),
falls out for free from being an embeddable library. These *exhaust* the
in-process embed: direction — who calls whom — is the only axis, so there
is no third role to invent. SQL-only queries and Lua-only computation are
the degenerate cases (no crossing), not prohibitions. Both directions are
built: **Lua-in-SQL** shipped its window slot in M2.7 (#47, #53) and its
vectorized column slot in M4.2; **SQL-in-Lua** (#70, M4.6) is the driver
seam — `query(sql)` returns result columns as the same zero-copy views
kernels consume (several segments concatenate — the bounded copy — with
per-segment key dictionaries merged), `append(table, row)` feeds derived
rows back exactly, and both globals are live only inside a driving call
(`LuaState::run_driver` / `Database::run_script` / the console's
`.run`), so a kernel can never re-enter the engine mid-query. The
remaining Lua-in-SQL slots — table-valued, predicate — are deferred
sub-scopes of that direction, not new roles. (These replace the earlier
"Role 1 / Role 2" labels: Lua-in-SQL was Role 1, SQL-in-Lua was Role 2.)

**Decision record — NULL across the script boundary (2026-07-26).** NULL
crosses to Lua as a distinct **sentinel value** — a `pd.NA`-style
singleton — not as Lua `nil` and not as NaN. *The principle that decided
it:* a non-editing round trip across any of the three boundaries
(DB↔SQL, DB↔Lua, Lua↔SQL) must preserve everything, which requires each
value mapping to be a total, invertible function. `nil` is not total —
inside a Lua table a `nil` *deletes* the slot, destroying both the value
and the row's structure, and that failure cannot be prevented inside
arbitrary user scripts. A distinct sentinel is a real value that survives
in a table, so the mapping stays total and faithful; it is kept distinct
from NaN (a computed value — see *Null, NaN, and ordering*) and
propagates over both numeric subtypes, so `i64` exactness holds. This was
chosen by a bake-off spike against the `nil` alternative, not by
argument.

*The cost we do not pay:* the prior art that reached the same conclusion
(Tarantool `box.NULL`, OpenResty `ngx.null`, pandas `pd.NA`) pays a real
price — the sentinel is truthy (`if x` is true for a null) and `x == nil`
silently misses it. Those systems carry data *as language values*, so
every field access meets the sentinel and its footguns. TallyDB does not:
columns cross as zero-copy views over engine buffers, and compute is
batch, not per-row. Null-aware batch ops (`v:sum()`), an out-of-band
validity view (`v:mask()`), and the curated compute ops consume
`(buffer, validity)` engine-side and never materialize the sentinel. The
footguns are real but confined to the discouraged manual per-element
path; the common path never meets them. A sentinel is the faithful
representation you rarely have to handle — an advantage that follows
directly from being a zero-copy, compute-in-engine store rather than a
value-shaped one. (One honest limit: relational `<`/`<=` against the
sentinel is a loud error, because Lua forces those operators to a
boolean and three-valued logic cannot propagate through them; arithmetic
propagates to the sentinel.)

**Decision record — the value map: return types, coercion, keys
(2026-07-26).** Three conventions govern how a script's results cross the
typed boundary, each chosen from TallyDB's own invariants (numeric-or-key,
exact-or-loud, zero-copy, the fixed-strategy planner), not from any one
precedent — the outside systems that solve the same problem validate the
*workload*, they do not get a vote on the *how*.

- **Return type is declared at registration and resolved at plan time** —
  never inferred from the value a call happens to return. A Lua-backed
  function names its result type (`f64` / `i64` / `key`) when registered;
  the planner fixes the output column's type from that, so a query yields
  the same Arrow schema on every run. Inferring per call would make the
  output type *data-dependent* (Lua silently floats integers) — the
  dynamic-typing property the fixed-strategy planner exists to exclude,
  and the root of bugs B4/B5 (#54). Every statically-typed peer declares
  it; the choice follows from our own static schema, not from theirs.
- **Coercion is exact-or-loud.** A Lua `integer` fills an `i64` and a
  `float` fills an `f64`; a `float` may fill an `i64` *only if it is
  losslessly integral*, otherwise it is a loud error — never a silent
  truncation. A Lua `boolean` maps to `i64 {0, 1}`: Booleans are a
  transient value, never a third column type (the numeric-or-key
  invariant holds). `nil` is NULL (the sentinel above). A `string` is
  interned into the output key dictionary — the only way a script
  produces a key. This closes B6 (#54).
- **Keys read as codes, with lazy text.** A key element reads as its
  integer dictionary code, so equality, grouping, and membership stay
  integer-cheap — which is *why* keys are dictionary-encoded, and what
  keeps the read zero-copy (the code is in the buffer; a string is not).
  `v:text(i)` decodes on demand; `v:code_of(literal)` resolves a literal
  once (the once-per-distinct-value pattern `WHERE` already uses), so
  `key == literal` is an integer compare. Codes are per-segment (#6); the
  engine guarantees a script sees one consistent code space per call
  (per-call or query-lifetime-remapped), so a raw code is never compared
  across segments.

Together with the NULL sentinel above, this is the frozen value-map
contract for the Lua boundary. The ergonomics layered on top — batch
reductions, `v:mask()`, `v:get(i, default)` — are additive and do not
change it.

**Decision record — the calling convention (Option A, 2026-07-26;
Observed).** A Lua-backed function is a **vectorized UDF**: the engine
calls it **once per segment**, handing whole columns as zero-copy input
views and — for the scalar/elementwise slot — a preallocated zero-copy
output view; the script loops the column *inside Lua* and writes the
output column. The window slot is the same shape one level in: the engine
drives the framing and the script reduces one frame to one scalar, exactly
the `regr_slope` / `eigen_max` pattern already shipped. Arguments are
**positional, with each argument's kind (column vs scalar constant)
declared at registration** alongside the return type (the value map
above), so the engine binds columns as views and constants as plain Lua
numbers. This is the batch-not-per-row rule made concrete — one boundary
crossing per *(function, segment)*, never per row. *Rejected:* **inline
Lua in the SQL text** (code in query strings has no registration to
declare a return type on, fights app-registered kernels, and recompiles
per call) and **a single uniform batch object** (it hands frame control to
the script — the one thing the measurement says to keep engine-side — and
buys only the table-valued / multi-output slots #47 already defers).
Evidence (`values_map_spike`, release, 4,096 rows): the vectorized call
produces its output column in ~518µs against ~12.7ms for per-row
invocation of the same kernel — a **25× penalty avoided**, the same
crossing tax the feed-reactive ruling excludes. The ~120× over a native
Rust loop is the interpreter / metamethod cost the promotion ladder
closes, not a property of the convention.

**Decision record — script observability: `log()` (2026-07-26).** Scripts
get one host-routed diagnostic function, `log(...)`, the replacement for
Lua's `print`. `print` is removed *not* because of the string invariant —
its text never becomes a column — but because its **destination**, the
process's stdout, is not an embeddable library's to own and is
uncapturable. `log(...)` routes instead to an **embedder-installed sink**:
a trait the host implements — the shell wires it to stderr, a library
embedder to its own logger, a headless embedding to a no-op (off by
default). It is a **pure side-channel**: no return that feeds results, it
cannot change query output — observational only (a diagnostic that alters
the answer is not a diagnostic). It logs scalars and short text; a **view
logs as a summary** (`f64 view, len 4096`), never its contents — a
diagnostic, not a buffer dump or an exfiltration path. Surface and sink
are both **flat** (`fn log(&self, msg)`; Lua `log(...)`), single severity:
the sink is script-only, so a level parameter would be permanently
degenerate (every message an `info`), and TallyDB carries no speculative
machinery. Severity, if a real need appears, is added *additively later* —
a defaulted `log_at(level, msg)` trait method breaks no existing embedder
(and at 0.0.1 nothing is frozen) — or the sink is deliberately widened to
an engine-wide diagnostic channel, a named scope expansion rather than a
default. Runaway volume (a kernel logging per row) is bounded by the
instruction-count hook (#61) and the batch-not-per-row doctrine: log per
batch, not per row. Because the sink is host-captured, an agentic harness
driving the engine reads a script's log as its observable output — the
"sight" half of #46.

**Decision record — feed-reactive compute (settled direction, 2026-07-26;
implementation M3+).** Reacting to new ordered data with compute is *in
scope* — it is a TSDB-native pattern, not a general-DB frill, and kdb+
(the reference workload) is built on it. The admitted shapes are **ingest
hooks** — a kernel invoked at a *batch* boundary inside `append()`
(segment freeze / batch land), compute-only, app-registered — and
**continuous queries** — derived data kept fresh on ordered append,
reduction via a kernel, with a restricted append-friendly incremental
scheme (details at implementation). Per-row *semantics* are available by
looping a batch inside one kernel call; **per-row hook *invocation* is
out** — measured at ~27× the batched loop (near-pure `pcall` crossing
tax, `values_map_spike`), and against the batch-not-per-row rule, the
append fast path, and columnar execution. Also out: **catalog-persisted
stored procedures** (never-a-server; code in the catalog is not
numeric-or-key — app-registered named kernels, which *are* the Lua-in-SQL
model, stay in) and **network delivery / push** (the app delivers; the
engine detects). Recompute-on-demand is not a reactive feature — it folds
into plain invocation. This is the standing **kdb+-as-floor** principle in
action: kdb+ sets a soft feature/perf floor at the *user* POV
(meet-or-exceed unless it conflicts with an invariant), and its own model
is exactly the batch ingest hook (`upd`), app-registered, no per-row
triggers, no catalog stored-procs — the same shape, delivered in-process
rather than via a multi-process server. Interacts with #44 (segment
freeze = the hook point), #49 (continuous-query SQL surface), and the
#41 / #47 kernel contract.

The performance story for scripts is a **promotion ladder**, not a JIT:
write the custom kernel in Lua to get it *correct* — immediately,
cross-checkably — and if it proves hot, promote it to a curated native
op to make it *fast*. That is the pattern `regr_slope`, `covar_pop`,
`corr`, and `eigen_max` already followed. Interpreter speed is a
comfort, not a foundation: the engine's speed lives in columnar storage
and pruning, the curated native ops, and the batch calling convention
above, none
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
summation order. We're not solving full native/WASM bit-identity now; the
portability *standard* (bit-exact, or bounded-difference, and over which
ops) is set when the WASM milestone (M6) starts. But the ground is mostly already held: every
closed-form window statistic and the `dot` kernel fix their operation
order in source — plain Rust loops, no runtime dispatch — so their
results are bit-identical across CPUs and targets by construction. The
two known holes, tracked rather than solved: faer's matrix kernels
dispatch SIMD at runtime and may round differently across CPU
generations (today they sit on no query path — the exposure begins if a
multi-parameter op adopts them); and `eigen_max` calls `hypot`, whose
last bit is libm-implementation-specific — switching to
`sqrt(a² + b²)` is a one-line fix to make when the standard is set.

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
   backend-independent; `f64` results are promised for the source-fixed
   paths — the closed-form window statistics and `dot` — and not yet for
   faer's matrix kernels (see *Numerical consistency*). A change that
   moves those bits is a behavioral change, not a refactor — re-blessing
   the goldens is part of its review.
4. **Meets its own spec** where no reference exists — `storage-lite`'s
   tests are the spec (see the reference map).

### The reference map

| Claim family | Reference | Tier |
|---|---|---|
| `query-lite` SQL semantics | DuckDB (primary) / DataFusion (secondary) | independent oracle |
| `arrow-lite` layout + C Data Interface | arrow-rs / PyArrow round-trips, dev-only | independent oracle |
| compute seam (curated ops and kernels) | NumPy/SciPy on the same inputs | independent oracle |
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
                    #   key; compaction; zone maps; the WAL; I/O behind a
                    #   backend trait (native = a directory of files;
                    #   lazy fault-in under a byte budget is the working-set
                    #   cut — see *The residency design*; OPFS/WASM later)
    query-lite/     # scoped SQL parser (via sqlparser-rs) + our own executor;
                    #   validated against DuckDB/DataFusion as an oracle
    engine/         # ties storage + query + compute together; enforces
                    #   numeric-or-key as a hard schema rule
    compute-lua/    # Lua scripting behind a trait; vendored PUC Lua 5.4,
                    #   hand-rolled bindings (lua.wasm, also 5.4, later)
    compute-linalg/ # multiplication-class kernels behind a trait; pure
                    #   Rust (faer + a source-fixed dot), wasm32-ready
    shell/          # the tallydb console binary: rustyline + csv live here,
                    #   the engine stays dependency-clean (#39's separation)
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
4. `compute-lua` / `compute-linalg` — compute backends
   (vendored Lua 5.4, faer); can be developed in parallel with
   `query-lite` once `arrow-lite`'s buffer format is stable, since they
   consume it directly.
5. `engine` — last, since it's the integration point for everything above.

**The one sequencing constraint that matters most:** the differentiator is
compute-fusion (zero-copy numeric ops on stored buffers), and that's the
riskiest, least-trodden part. Reach a thin end-to-end proof of it *early* —
ingest numeric+key rows → a windowed query that calls a curated numeric op
on stored buffers with no copy → Arrow out — rather than leaving it for
last. Building the storage engine beautifully while the compute story slips
just yields "another embeddable TSDB" and misses the point.

Don't try to scaffold all eight crates' real implementations in one pass.

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
