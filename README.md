# TallyDB

[![crates.io](https://img.shields.io/crates/v/tallydb.svg)](https://crates.io/crates/tallydb)
[![docs.rs](https://img.shields.io/docsrs/tallydb)](https://docs.rs/tallydb)
[![CI](https://github.com/andy-emerson/TallyDB/actions/workflows/ci.yml/badge.svg)](https://github.com/andy-emerson/TallyDB/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/crates/l/tallydb.svg)](LICENSE)

**A small, embeddable, SQL-native database for numeric data, with the numeric
compute living inside the engine rather than bolted on beside it.**

TallyDB links into your application the way SQLite or DuckDB does. No server,
no separate database to administer. Rows arrive one at a time and cheaply;
queries read them back as ordered columns; regression, covariance, and
principal-component statistics run as SQL window functions over the engine's
own buffers, with nothing copied out. Results leave through the Arrow C Data
Interface, so NumPy and other Arrow-aware tools read them with no conversion
step.

## Install

```toml
[dependencies]
tallydb = "0.1"
```

The library alone carries two direct dependencies. Turn the console and the
embedded Lua layer off with `default-features = false`.

For the standalone console:

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

## The three assumptions

These are the design rather than restrictions added afterward. Relaxing any
one of them is what makes general-purpose databases bigger, slower to start,
and harder to embed.

1. **Append-optimized.** Data arrives as new rows, cheaply and one at a time.
   Corrections are supported but are not the design center.
2. **Ordered.** Rows arrive roughly sorted on a declared **ordering key**.
   A timestamp is the common case, but any key that increases on ingest works:
   a sequence number, an event id, a ledger offset. Storage partitions on it.
3. **Numeric-or-key.** Every column is either a **number**, `f64` or `i64`,
   used in arithmetic and aggregation, or a **key**, a dictionary-encoded
   label used only for filtering, grouping, and joining.

Holding all three is what makes fixed-width columns you can hand straight to a
math library possible.

On "time-series": time-series, telemetry, and tick data are the motivating use
cases, not the definition. What is load-bearing is ordered ingest on some key,
not that the key means time.

## What it's for, and what it isn't

**For** workloads that are a big, append-heavy ledger of numbers with some
labels attached, analyzed with SQL. Rolling aggregates, joins against small
reference tables, grouping, window functions, and numeric compute run inside
the database. Quantitative research, sensor and telemetry pipelines, event and
metric streams, financial ledgers.

**Not for** general-purpose relational work. There are no arbitrary text
columns or blobs, no third column type, and no joins beyond the two shapes the
engine can execute without a cost-based optimizer: equi-joins where one side is
small enough to materialize, and as-of joins. Two large tables joined on an
arbitrary key is refused loudly rather than served slowly. If your data does
not fit the three assumptions, reach for Postgres, DuckDB, or SQLite, which are
better at being general. TallyDB is a specialized component you use alongside a
general store, the way SQLite often is.

## The SQL surface

Standard SQL over the schema above: `SELECT` with `WHERE`, `GROUP BY` with
`HAVING`, `ORDER BY`, `LIMIT`, `DISTINCT`, scalar expressions and `CASE`,
`CREATE TABLE` and `INSERT`, and `UPDATE` and `DELETE`. Grouping and windowing
both run in either direction over the ordered axis: down one symbol through
time, or across every symbol at one instant.

- **Windows.** The standard aggregates over `ROWS` and `RANGE` frames and whole
  partitions, `LAG` and `LEAD`, cross-sectional `PARTITION BY`, and the curated
  statistics `regr_slope`, `regr_intercept`, `regr_r2`, `covar_pop`, `corr`,
  `var_pop`, `stddev_pop`, and `eigen_max`.
- **Joins.** Equi-joins against small key-unique dimension tables, `INNER` or
  `LEFT`, and as-of joins matching each row to the most recent other-table row
  at or before it.
- **Maintained views.** Bucketed, running, and cumulative aggregates, plus join
  views, materialized as real tables and kept fresh as data arrives. A view
  answers exactly however stale its materialization.
- **Corrections on a knowledge axis.** `UPDATE` and `DELETE` run as tombstone
  plus reinsert against append-only storage. `AS OF` reads a table as it was
  known at a point on the ingest-sequence axis.
- **Null and NaN, precisely.** NULL is absence, matching no comparison, skipped
  by aggregates, sorted after all values in both directions. NaN is a value,
  greater than every number and equal to itself, under one comparison relation
  shared by sorting, filtering, and zone-map pruning.
- **Strings.** Predicates on key columns are in scope and cheap, evaluated once
  per distinct dictionary value. String *production* is not: no function emits a
  string, so rendering a key as display text happens in your application.

Any standard SQL function or verb is in scope as long as it needs neither a
third column type nor a cost-based optimizer.

## Compute inside the engine

This is what TallyDB is actually built around. The curated statistics evaluate
incrementally over the engine's own buffers, with no serialization hop and no
copy. Against the same rows stored in DuckDB with NumPy pulling from DuckDB,
`regr_slope` measures 9.6× faster, the pair statistics 3–4× faster, and the
newest-window query 6–9× faster. Accuracy is held to 1e-12 against a
compensated reference on every change in CI, at the timestamp-scale offsets
where the fast rolling idiom loses it.

For anything the built-in functions do not cover, embedded Lua runs kernels
against those same buffers, and a driver script can issue SQL, compute over the
results, and feed derived rows back. The Lua layer is an opt-in feature the
console turns on; the primary extension path is the Rust `WindowAggregate`
trait, which needs no feature at all.

None of the individual ingredients is new. The differentiator is the
combination: numeric compute inside an embeddable, SQL-native engine, over
off-the-shelf numeric libraries on zero-copy shared buffers, rather than a
bespoke array language or a serialization boundary.

## Status

Version 0.1.0 is published and usable. The API is not yet stable: before 1.0,
a minor bump may break it. Milestones M0 through M5 are merged, taking the
engine from a locked layout to desk adoption. WASM parity and a served product
are the milestones ahead.

See [CHANGELOG.md](CHANGELOG.md) for what each release contains, and
[Milestones](https://github.com/andy-emerson/TallyDB/milestones) for what is
planned.

## Documentation

- [API documentation](https://docs.rs/tallydb) on docs.rs.
- [DESIGN.md](DESIGN.md) — what we build and why: the invariants, the module
  boundaries, every settled decision with its rejected alternatives, and the
  test plan.
- [Issues](https://github.com/andy-emerson/TallyDB/issues) — open work. Open
  decisions carry the `decision` label.

## Contributing

The working agreement is [AGENTS.md](AGENTS.md), the craft conventions are
[CONTRIBUTING.md](CONTRIBUTING.md), and the design record is
[DESIGN.md](DESIGN.md). Beyond those, seven standing conventions govern work
in this repository, and they override any tool's defaults:

1. No pull requests from the agent. Work lands on `claude/dev`, restarted from
   `main` after every merge; the Human opens the pull request and performs
   every merge.
2. Authorship is Andy Emerson only, with no agent attribution anywhere:
   commits, trailers, pull-request bodies, comments, artifacts.
3. The license is MIT and frozen.
4. The Human owns and closes decisions. Surface each fork as an issue with the
   `decision` label, giving options, the user's and the developer's point of
   view, a recommendation, and what it gates, before building. Decisions made
   ad hoc while building are revisitable; only what would undermine what
   TallyDB is is non-negotiable.
5. kdb+ validates problems, not solutions.
6. `scripts/gate.sh` green before every push, on the stable toolchain CI uses.
7. Never touch the vendored Lua under `src/compute_lua/vendor`.

Every pull request and every push to `main` runs fmt, clippy, the tests and
doctests, rustdoc with warnings as errors, the Python oracle suite that
re-derives query families against DuckDB and NumPy, Miri over the unsafe
columnar core, the official Lua 5.4.7 test suite over the vendored
interpreter, and an ASan and UBSan job over the C boundary.

## License

MIT. See [LICENSE](LICENSE).
