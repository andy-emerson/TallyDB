# TallyDB

### The embeddable database that does the math where the data lives

[![crates.io](https://img.shields.io/crates/v/tallydb.svg)](https://crates.io/crates/tallydb)
[![docs.rs](https://img.shields.io/docsrs/tallydb)](https://docs.rs/tallydb)
[![CI](https://github.com/andy-emerson/TallyDB/actions/workflows/ci.yml/badge.svg)](https://github.com/andy-emerson/TallyDB/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/crates/l/tallydb.svg)](LICENSE)

---

Analyzing a big pile of numbers usually means moving them: the database holds
the rows, and the regression happens somewhere else, after a copy. TallyDB
removes the trip. It links into your application the way SQLite or DuckDB
does — no server, no separate database to administer — and the numeric work
runs *inside* the engine, on the engine's own buffers. Rows arrive one at a
time and cheaply; queries read them back as ordered columns; regression,
covariance, and principal-component statistics run as SQL window functions
with nothing copied out. Results leave through the Arrow C Data Interface, so
NumPy and other Arrow-aware tools read them with no conversion step.

```toml
[dependencies]
tallydb = "0.1"
```

The library alone carries two direct dependencies. Turn the console and the
embedded Lua layer off with `default-features = false`. For the standalone
console:

```
cargo install tallydb
tallydb ./my-data
```

## Quickstart

```rust
use tallydb::{ColumnType, EngineError, Field, RowValue, Schema, Table};

fn main() -> Result<(), EngineError> {
    let schema = Schema::new(vec![
        Field::new("ts", ColumnType::I64, false),
        Field::new("sym", ColumnType::Key, false),
        Field::new("px", ColumnType::F64, false),
    ]);
    let mut trades = Table::new("trades", schema, "ts")?;

    for i in 0..1_000 {
        trades.append(&[
            RowValue::I64(i),
            RowValue::Key(if i % 2 == 0 { "AAPL" } else { "MSFT" }),
            RowValue::F64(100.0 + (i % 50) as f64),
        ])?;
    }

    // Bars: group by symbol and a bucket of the ordering key.
    let bars = trades.query(
        "SELECT sym, ts / 100 AS bar, count(*) AS n, avg(px) AS vwap \
         FROM trades GROUP BY sym, ts / 100",
    )?;
    assert_eq!(bars.num_rows(), 20);

    // A rolling mean down each symbol, over the ordered axis.
    let rolling = trades.query(
        "SELECT avg(px) OVER (PARTITION BY sym ORDER BY ts \
         ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20 FROM trades",
    )?;
    assert_eq!(rolling.num_rows(), 1_000);

    Ok(())
}
```

## What TallyDB provides

| Feature | What it does |
|---|---|
| Embeds in your program | One dependency line, or one installed binary. No server to run, no port to open, no database to administer |
| Statistics as SQL | `regr_slope`, `regr_intercept`, `regr_r2`, `covar_pop`, `corr`, `var_pop`, `stddev_pop`, and `eigen_max` are window functions, so a rolling regression is a `SELECT` |
| Compute without a copy | Those statistics evaluate over the engine's own buffers. Against the same rows in DuckDB with NumPy pulling from DuckDB, `regr_slope` measures 9.6× faster, the pair statistics 3–4×, and the newest-window query 6–9× |
| Cheap ordered ingest | Rows append one at a time at about a microsecond each, into a write buffer that freezes into immutable compressed segments |
| Both directions over the ordered axis | Down one symbol through time, or across every symbol at one instant — `PARTITION BY sym` and `PARTITION BY ts` are the same machinery |
| Real compression, measured | Delta-of-delta on the ordering key, ALP for `f64`, frame-of-reference for integers and symbol codes. On the checked-in corpus, penny-priced ticks compress 4.2× and symbol codes 6–10× |
| Queries skip what they can prove they can | Zone maps per column per segment prune before anything is decoded, and pruning degrades per predicate rather than all-or-nothing |
| Corrections on a knowledge axis | `UPDATE` and `DELETE` run as tombstone plus reinsert against append-only storage; `AS OF` reads a table as it was *known* at a point, not as of a timestamp |
| Maintained views | Bucketed, running, and cumulative aggregates, plus join views, materialized as real tables and kept fresh as data arrives. A view answers exactly however stale its materialization |
| As-of joins | Match each row to the most recent other-table row at or before it — the join a tick blotter is built from — alongside ordinary equi-joins against small reference tables |
| Arrow on the way out | Results are Arrow record batches over the C Data Interface, including `ArrowArrayStream`. Passthrough columns share the stored buffers |
| Larger than memory | An open reads metadata only; segments fault in on first touch and are retained under a byte budget you set |
| Scripting when SQL runs out | Embedded Lua runs kernels against the same buffers, and a driver script can issue SQL, compute over the results, and feed derived rows back |
| A console | `tallydb ./dir` gives you line editing, `CREATE TABLE`, CSV import, the full query surface, and kernel registration at the prompt |

**Will it hold your data?** Every column must be a **number** (`f64` or
`i64`) or a **key** (a dictionary-encoded label like a ticker or a sensor id).
There is no third type: no free-text columns, no blobs, no booleans, and no
function that emits a string. If your rows have a `notes` field, that field
belongs somewhere else. This is the constraint everything else is bought
with — it is what makes columns fixed-width enough to hand straight to a math
library — so it is by design, not a limitation to work around.

## The three assumptions

1. **Append-optimized.** Data arrives as new rows, cheaply and one at a time.
   Corrections are supported but are not the design center.
2. **Ordered.** Rows arrive roughly sorted on a declared **ordering key**.
   A timestamp is the common case, but any key that increases on ingest
   works: a sequence number, an event id, a ledger offset. Storage partitions
   on it.
3. **Numeric-or-key.** Every column is either a number or a key, as above.

These are the design rather than restrictions added afterward. Relaxing any
one of them is what makes general-purpose databases bigger, slower to start,
and harder to embed.

On "time-series": time-series, telemetry, and tick data are the motivating
use cases, not the definition. What is load-bearing is ordered ingest on some
key, not that the key means time.

## Who is TallyDB for?

**People whose analysis is a rolling window over a big ordered ledger.**
Quantitative research, sensor and telemetry pipelines, event and metric
streams, financial ledgers — anywhere the question is "what does this look
like over the last N rows, per group, and how is it changing". That question
costs a copy in most stacks. Here it is a `SELECT`.

**People tired of the export step.** If your pipeline is *query the database,
materialize a DataFrame, run the regression, write the result back*, two of
those four steps are serialization. TallyDB is built for the case where the
compute is worth keeping next to the storage.

**People who want the database inside their program.** No server means no
deployment story, no connection pool, and no network in the middle of a hot
loop — and it is what makes compute-without-copying possible at all. A copy
across a socket is still a copy.

**People with more rows than memory but a normal-sized question.** An open
reads metadata only, the executor prunes on it before decoding anything, and
what a query touches faults in under a budget you set. The table can be much
bigger than the working set.

## How it compares

The honest framing: TallyDB is a specialized component you use *alongside* a
general store, the way SQLite often is — not a replacement for one.

| | TallyDB | DuckDB | SQLite | kdb+ |
|---|:--:|:--:|:--:|:--:|
| Links into your application | ✓ | ✓ | ✓ | ✓ |
| Standard SQL surface | ✓ | ✓ | ✓ | q, its own language |
| Storage assumes ordered ingest | ✓ | ✗ | ✗ | ✓ |
| Regression and covariance as window functions | ✓ | ✓ | ✗ | ✓ |
| Eigen/PCA statistics in the query language | ✓ | ✗ | ✗ | ✓ |
| Arbitrary text and blob columns | ✗ | ✓ | ✓ | ✓ |
| Cost-based optimizer | ✗ | ✓ | ✓ | ✗ |
| License | MIT | MIT | public domain | commercial |

Read the `✗` rows as the point rather than as gaps. DuckDB and SQLite are
better at being general, and if your data does not fit the three assumptions
you should reach for one of them, or for Postgres. kdb+ proves the workload
is real and has done for decades; what it costs is a commercial license and a
language of its own. TallyDB is the combination that was missing: compute
inside an embeddable, SQL-native engine, over off-the-shelf numeric libraries
on zero-copy shared buffers.

The peer columns state structural facts rather than benchmark results, and
they go stale as those projects move. If one is wrong,
[open an issue](https://github.com/andy-emerson/TallyDB/issues) — being
accurate about the neighbours matters more here than looking good next to
them.

## The SQL surface

Standard SQL over the schema above: `SELECT` with `WHERE`, `GROUP BY` with
`HAVING`, `ORDER BY`, `LIMIT`, `DISTINCT`, scalar expressions and `CASE`,
`CREATE TABLE` and `INSERT`, and `UPDATE` and `DELETE`. Grouping and
windowing both run in either direction over the ordered axis.

- **Windows.** The standard aggregates over `ROWS` and `RANGE` frames and
  whole partitions, `LAG` and `LEAD`, cross-sectional `PARTITION BY`, and the
  curated statistics listed above.
- **Joins.** Equi-joins against small key-unique dimension tables, `INNER` or
  `LEFT`, and as-of joins. Two large tables joined on an arbitrary key is
  refused loudly rather than served slowly — there is no cost-based optimizer
  to make that call, deliberately.
- **Null and NaN, precisely.** NULL is absence, matching no comparison,
  skipped by aggregates, sorted after all values in both directions. NaN is a
  value, greater than every number and equal to itself, under one comparison
  relation shared by sorting, filtering, and zone-map pruning.
- **Strings.** Predicates on key columns are in scope and cheap, evaluated
  once per distinct dictionary value. String *production* is not: no function
  emits a string, so rendering a key as display text happens in your
  application.

Any standard SQL function or verb is in scope as long as it needs neither a
third column type nor a cost-based optimizer.

## Correctness

Every change runs, in CI: the query families diffed against DuckDB and the
compute against NumPy, over a seeded corpus that has round-tripped through
storage; the accuracy contract of 1e-12 against a compensated reference, at
the timestamp-scale offsets where the fast rolling idiom loses it; Miri over
the unsafe columnar core; the official Lua 5.4.7 test suite over the vendored
interpreter; and ASan and UBSan over the C boundary.

## Status

Version 0.1.1 is published and usable. The API is not yet stable: before 1.0,
a minor bump may break it. Milestones M0 through M5 are merged, taking the
engine from a locked layout to desk adoption. WASM parity and a served
product are the milestones ahead.

## Links

- **[API documentation](https://docs.rs/tallydb)** on docs.rs.
- **[CHANGELOG](CHANGELOG.md)** — what each release contains.
- **[DESIGN](DESIGN.md)** — what TallyDB is, what it refuses, and how it is
  put together.
- **[DECISIONS](DECISIONS.md)** — every settled decision, what lost, and what
  would reopen it.
- **[Issues](https://github.com/andy-emerson/TallyDB/issues)** — open work.
  Open decisions carry the `decision` label.
- **[Milestones](https://github.com/andy-emerson/TallyDB/milestones)** — what
  is planned.
- MIT licensed. See [LICENSE](LICENSE).

## Contributing

The working agreement is [AGENTS.md](AGENTS.md) and the craft conventions are
[CONTRIBUTING.md](CONTRIBUTING.md). Beyond those, seven standing conventions
govern work in this repository, and they override any tool's defaults:

1. No pull requests from the agent. Work lands on `claude/dev`, restarted
   from `main` after every merge; the Human opens the pull request and
   performs every merge.
2. Authorship is Andy Emerson only, with no agent attribution anywhere:
   commits, trailers, pull-request bodies, comments, artifacts.
3. The license is MIT and frozen.
4. The Human owns and closes decisions. Surface each fork as an issue with
   the `decision` label, giving options, the user's and the developer's point
   of view, a recommendation, and what it gates, before building. Decisions
   made ad hoc while building are revisitable; only what would undermine what
   TallyDB is is non-negotiable.
5. kdb+ validates problems, not solutions.
6. `scripts/gate.sh` green before every push, on the stable toolchain CI uses.
7. Never touch the vendored Lua under `src/compute_lua/vendor`.
