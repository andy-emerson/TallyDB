# Changelog

Notable changes to TallyDB, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html): before 1.0
a minor bump may break the API and a patch bump may not.

The reasoning behind any entry here lives in [DESIGN.md](DESIGN.md), which
records decisions, their rejected alternatives, and what would reopen them.

## [0.1.1] — 2026-09-14

### Fixed

- A leading `--` comment no longer hides `CREATE TABLE` from the rewrite that
  carries `ORDERING KEY` through the parser, so a commented DDL statement is
  accepted rather than rejected as unsupported syntax (#116).
- `tallydb DIR -c "sql"` runs its statements and exits instead of going on to
  read stdin, which hung any script whose stdin was an open pipe — forever, and
  holding the writer lock the whole time (#117).
- A multi-line paste into the interactive console is handled line by line, as
  the piped path always was. Pasting a statement followed by a dot command left
  the dot command stranded in the SQL buffer, and pasting `.lua` followed by a
  query compiled the query into the Lua chunk (#118).
- The continuation prompt appears only when a statement is actually pending.
  A leftover newline in the buffer used to show `  ...>` after every statement
  (#118).

## [0.1.0] — 2026-09-14

First published release. Everything below was built before it, through
milestones M0 to M5, and arrives together.

### Storage

- Append one row at a time into a per-table store: a write buffer freezing
  into immutable segments at a row threshold, each row carrying an internal
  monotonic row id.
- A self-describing, CRC-checked, deterministic on-disk format whose bytes are
  locked by a committed golden file, behind a storage-backend trait.
- Per-column codecs: delta-of-delta on the ordered key, ALP with its
  ALP-RD and raw fallbacks for `f64` values, and frame-of-reference with
  bit-packing for symbol codes and non-clock integers. Measured on the
  checked-in corpus, penny-priced ticks compress 4.2× against raw and symbol
  codes 6–10×.
- Zone maps per column per segment, driving query-time pruning.
- A sidecar write-ahead log with sync levels: group commit every 100ms by
  default, measured at roughly 1µs added per append, `Full` for a zero loss
  window, `Off` for replayable upstreams. Crash-tested down to torn-record and
  stale-generation windows.
- Lazy residency: segments fault in under a byte budget, pruning runs on
  segment metadata before the fault, and a table larger than its budget
  answers exactly like a resident one.
- Single-writer, concurrent-reader snapshots. Read-only opens over a live
  writer's directory see its durable state and refresh when they choose.
- Crash-safe generational compaction that resolves tombstones, restores order,
  and reassigns contiguous row ids.

### Query

- `SELECT` with `WHERE`, `GROUP BY` with `HAVING`, `ORDER BY` with
  `NULLS FIRST` and `NULLS LAST`, `LIMIT` and `OFFSET`, `DISTINCT`, scalar
  expressions and `CASE`, `CREATE TABLE`, `INSERT`, `UPDATE`, and `DELETE`.
- Zone-map pruning that degrades per conjunct, so a predicate no zone map can
  rule out does not cost pruning on the ones it can.
- `GROUP BY` over key columns and over monotone buckets of the ordering key.
  A bucket grouping streams, holding only the open bucket rather than a hash
  table over every group.
- `ORDER BY` with `LIMIT` runs top-k: a ten-row answer holds ten rows, not a
  sorted copy of the whole result.
- Equi-joins against small key-unique dimension tables, `INNER` or `LEFT`,
  gathering only the dimension columns the query reads.
- As-of joins, `ASOF LEFT JOIN` and `ASOF INNER JOIN`, matching each row to the
  most recent other-table row at or before it, with ties broken by ingest
  sequence.
- Window functions over `ROWS` and `RANGE` frames and whole partitions,
  `LAG` and `LEAD`, and cross-sectional `PARTITION BY` including an unordered
  partition that ranks peers within one timestamp.
- Results leave as Arrow record batches through the C Data Interface,
  including `ArrowArrayStream`. Passthrough columns share the stored buffers.

### Corrections

- `UPDATE` and `DELETE` as tombstone plus reinsert against append-only
  storage, each mutation spending exactly one coordinate on the
  ingest-sequence axis.
- `AS OF` reads a table as it was known at a coordinate, and `_seq` reads a
  row's own coordinate back through SQL.

### Maintained views

- Bucketed, running, and cumulative aggregates materialized as real tables
  plus the source watermark they reflect, refreshed by re-folding only the
  buckets the knowledge history says changed.
- Join views: the enriched blotter carrying each fact's as-of match,
  aggregates over the as-of join, and star aggregates over an equi-joined
  dimension. A late correction repairs exactly its interval; an in-order
  arrival costs nothing.
- A view read tops the materialization up with a live fold of whatever the
  watermark does not cover, so a view answers exactly however stale it is.

### Compute

- Curated statistics as SQL window functions: `regr_slope`, `regr_intercept`,
  `regr_r2`, `covar_pop`, `corr`, `var_pop`, `stddev_pop`, and `eigen_max`.
  Solved in closed form, with no LAPACK linked at all.
- Incremental evaluation: running moments about a data-anchored shift,
  re-anchored every window length so rounding cannot accumulate. Against the
  same rows in DuckDB with NumPy pulling from DuckDB, `regr_slope` measures
  9.6× faster, the pair statistics 3–4× faster, and the newest-window shape
  6–9× faster.
- Rolling multi-factor regression above two parameters, maintaining the
  window's moments across the slide and re-solving the small system per frame,
  4×–38× faster than per-frame recompute.
- Multiplication-class linear algebra in pure Rust behind a
  capability-negotiating trait. No BLAS, no LAPACK, and the compute stack
  compiles for wasm32 as is.
- Embedded Lua 5.4, vendored byte-identical to upstream v5.4.7, behind a
  frozen value-map contract: nullable columns cross as zero-copy views,
  results coerce exact-or-loud to a declared type, and kernels register as SQL
  window functions. A runaway kernel can be bounded by an instruction budget.
- SQL-in-Lua driver scripts: a script issues `query(sql)`, receives result
  columns as the same zero-copy views kernels consume, and feeds derived rows
  back through `append(table, row)`.

### The console

- `tallydb DIR` opens a console with line editing, `CREATE TABLE`, `INSERT`,
  CSV import, the full query surface, `.lua` kernel registration, and
  `.run FILE` for driver scripts.
- `--read-only` opens over a live writer's directory, with `.refresh` and
  `.flush`; `--cache MiB` sets the residency budget.
- Binaries for Linux, macOS, and Windows attach to each GitHub release.

### Correctness

- Query families are diffed against DuckDB, and compute against NumPy, over a
  seeded synthetic corpus that has round-tripped through storage, on every
  change in CI.
- The accuracy contract is 1e-12 against a compensated reference, guarded on
  every change over adversarial corpora at timestamp-scale offsets.
- The unsafe columnar core runs under Miri; the C boundary runs under
  `LUA_USE_APICHECK` and under ASan and UBSan; the official Lua 5.4.7 test
  suite and its `ltests` torture harness run against the vendored interpreter.

[Unreleased]: https://github.com/andy-emerson/TallyDB/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/andy-emerson/TallyDB/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/andy-emerson/TallyDB/releases/tag/v0.1.0
