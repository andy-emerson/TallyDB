#!/usr/bin/env python3
"""Differential oracle: generated query families vs DuckDB (M2.4+).

The generated side of the differential harness: this script owns query
generation (there is no second list to keep in sync in Rust — the SQL
travels over the C ABI), runs every query against both TallyDB's corpus
fixture and a DuckDB replica of the same rows, and diffs the results.
Ordered families carry ORDER BY over the unique `ts`, so both engines
agree on a total order and the diff is row-for-row. Grouped and
DISTINCT results carry no such order — a symbol column cannot be
ordered by (#58) — so the REFEREE sorts both sides before diffing.

Known, deliberate divergences the generator avoids:
  - SUM over an i64 column: DuckDB promotes to HUGEINT; TallyDB keeps
    exact i64 and errors loudly on overflow. The families sum f64
    columns only.
  - DuckDB encodes undefined regressions as NaN where TallyDB (and
    NumPy) use NULL; window comparisons normalize NaN to None.
  - `ORDER BY <symbol column>`: DuckDB sorts, TallyDB refuses (#58 = B).
    Checked as a refusal rather than avoided.
  - Division by zero: TallyDB is IEEE (NaN, a value — decision D2),
    DuckDB returns NULL. Only reachable where a family divides by a
    window result, so IEEE_DIVISION_FAMILIES normalizes both sides and
    says so; every other family keeps the strict comparison.

Usage: m2_differential_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
from pyarrow.cffi import ffi


def load_library() -> ctypes.CDLL:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        repo = Path(__file__).resolve().parents[3]
        path = repo / "target" / "debug" / "libengine.so"
    if not path.exists():
        sys.exit(
            f"{path} not found - build it with "
            "`cargo build -p engine --features oracle-harness`"
        )
    return ctypes.CDLL(str(path))


def read_stream_hook(lib, symbol: str) -> pa.Table:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    getattr(lib, symbol)(ctypes.c_void_p(ptr))
    return pa.RecordBatchReader._import_from_c(ptr).read_all()


def tallydb_query(lib, sql: str) -> pa.Table:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    status = lib.tallydb_corpus_query_stream(
        ctypes.c_char_p(sql.encode()), ctypes.c_void_p(ptr)
    )
    if status != 0:
        sys.exit(f"FAIL engine rejected: {sql}")
    return pa.RecordBatchReader._import_from_c(ptr).read_all()


def tallydb_refuses(lib, sql: str) -> bool:
    """Whether the engine rejects `sql` — a refusal is an answer too."""
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    return (
        lib.tallydb_corpus_query_stream(
            ctypes.c_char_p(sql.encode()), ctypes.c_void_p(ptr)
        )
        != 0
    )


def sorted_rows(table: pa.Table, columns: list[str] | None = None) -> list[tuple]:
    """A table's rows under a total python-side order — the referee's
    own sort, for results whose row order neither engine promises.
    None ranks after every value; pyarrow cannot sort dictionary
    columns, which is why this is done in python."""
    names = table.column_names
    values = [table[name].to_pylist() for name in names]
    rows = list(zip(*values))
    order = [names.index(name) for name in (columns or names)]

    def rank(cell):
        return (1, 0) if cell is None else (0, cell)

    return sorted(rows, key=lambda row: tuple(rank(row[index]) for index in order))


def close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def nan_to_none(values):
    return [
        None if isinstance(v, float) and math.isnan(v) else v for v in values
    ]


def families() -> list[str]:
    """Query families with a deterministic total order (unique ts, or a
    grouped key). Grows with the SQL surface."""
    queries = []
    # Passthrough with ordering and paging — including ORDER BY on a
    # column the query does not project (standard SQL: carried hidden,
    # sorted by, dropped), alone and under LIMIT/OFFSET, and where an
    # alias shadows the stored name (the alias wins, per standard
    # output-name precedence).
    queries += [
        "SELECT ts, sym, x, y FROM corpus ORDER BY ts",
        "SELECT ts, x FROM corpus ORDER BY ts DESC LIMIT 100",
        "SELECT ts, x FROM corpus ORDER BY x LIMIT 50 OFFSET 25",
        "SELECT ts, y FROM corpus ORDER BY y DESC LIMIT 40",
        "SELECT sym, x FROM corpus ORDER BY ts",
        "SELECT x FROM corpus ORDER BY ts DESC LIMIT 60",
        "SELECT ts FROM corpus ORDER BY x LIMIT 30 OFFSET 10",
        "SELECT x * 2 AS d FROM corpus ORDER BY ts LIMIT 25",
        "SELECT ts, x AS y FROM corpus ORDER BY y LIMIT 35",
        "SELECT ts, x FROM corpus WHERE sym = 'K003' ORDER BY y DESC LIMIT 20",
    ]
    # WHERE: numeric boundaries, key membership, boolean structure.
    for predicate in [
        "x > 100",
        "x <= 99.25",
        "y > 140",
        "ts >= 1700000000000000000 AND x < 101",
        "sym = 'K003'",
        "sym IN ('K000', 'K005', 'K007')",
        "sym NOT IN ('K001', 'K002', 'K003', 'K004')",
        "sym <> 'K006' AND (x > 100 OR y < 130)",
        "NOT (x > 100)",
        "x > 99 AND x < 100.5 AND sym IN ('K000', 'K001')",
    ]:
        queries.append(f"SELECT ts, sym, x, y FROM corpus WHERE {predicate} ORDER BY ts")
    # Star-schema joins: lookup, misses under INNER vs LEFT, the full
    # query surface over the joined shape. K007 is missing from sensors.
    queries += [
        "SELECT ts, site, calib FROM corpus JOIN sensors "
        "ON corpus.sym = sensors.sym ORDER BY ts",
        "SELECT ts, corpus.sym, calib FROM corpus LEFT JOIN sensors "
        "ON corpus.sym = sensors.sym ORDER BY ts",
        "SELECT ts, x, calib FROM corpus JOIN sensors ON corpus.sym = sensors.sym "
        "WHERE calib > 1 AND x < 101 ORDER BY ts",
        "SELECT ts, sum(calib) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING "
        "AND CURRENT ROW) AS w FROM corpus JOIN sensors "
        "ON corpus.sym = sensors.sym ORDER BY ts",
    ]
    # Join projection pushdown (#81): only the dimension columns a
    # query reads are gathered, so the queries that matter are the ones
    # reading a dimension column WITHOUT projecting it — a wrong
    # used-set computation shows up here and nowhere else.
    queries += [
        "SELECT ts, x FROM corpus JOIN sensors ON corpus.sym = sensors.sym "
        "WHERE site = 'north' ORDER BY ts",
        "SELECT ts, x FROM corpus JOIN sensors ON corpus.sym = sensors.sym "
        "WHERE calib > 1.0 AND site <> 'east' ORDER BY ts",
        "SELECT ts, x * calib AS scaled FROM corpus JOIN sensors "
        "ON corpus.sym = sensors.sym ORDER BY ts",
        "SELECT ts, CASE WHEN site = 'north' THEN x ELSE 0 END AS northern "
        "FROM corpus JOIN sensors ON corpus.sym = sensors.sym ORDER BY ts",
        "SELECT ts, sum(x) OVER (PARTITION BY site ORDER BY ts "
        "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS w FROM corpus JOIN sensors "
        "ON corpus.sym = sensors.sym ORDER BY ts",
        "SELECT ts, x FROM corpus LEFT JOIN sensors ON corpus.sym = sensors.sym "
        "WHERE calib IS NULL ORDER BY ts",
    ]
    # The full window surface: standard aggregates as windows, mixed
    # frames, several windows in one query.
    queries += [
        "SELECT ts, sum(x) OVER (PARTITION BY sym ORDER BY ts "
        "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
        "SELECT ts, avg(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING "
        "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
        "SELECT ts, min(x) OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN 4 PRECEDING "
        "AND CURRENT ROW) AS lo, max(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED "
        "PRECEDING AND CURRENT ROW) AS hi FROM corpus ORDER BY ts",
        "SELECT ts, count(x) OVER (ORDER BY ts ROWS BETWEEN 99 PRECEDING AND "
        "CURRENT ROW) AS n FROM corpus ORDER BY ts",
    ]
    # Aggregates: global and grouped, nulls exercised through y.
    queries += [
        "SELECT count(*) AS n FROM corpus",
        "SELECT count(y) AS n FROM corpus",
        "SELECT sum(x) AS s, avg(x) AS a, min(x) AS lo, max(x) AS hi FROM corpus",
        "SELECT avg(y) AS a, min(y) AS lo, max(y) AS hi FROM corpus",
        "SELECT min(ts) AS lo, max(ts) AS hi, count(*) AS n FROM corpus",
        "SELECT count(*) AS n FROM corpus WHERE x > 1e12",
    ]
    # The M3.4 IN-tier (#49): computed projections, CASE, LIKE on keys,
    # DISTINCT, HAVING, NULLS FIRST/LAST. Everything computes as DOUBLE
    # (the engine's expression type); sqrt keeps its argument
    # non-negative so both engines stay in IEEE territory.
    queries += [
        "SELECT ts, x * 2 + 1 AS a, x - y AS b, x / 7 AS c FROM corpus ORDER BY ts",
        "SELECT ts, abs(100.0 - x) AS d, sqrt(abs(y - 100)) AS r FROM corpus ORDER BY ts",
        "SELECT ts, floor(x) AS f, ceil(y) AS c, round(x) AS rn FROM corpus ORDER BY ts",
        "SELECT ts, power(x / 100, 3) AS p, exp((100 - x) / 50) AS e FROM corpus ORDER BY ts",
        "SELECT ts, CASE WHEN x > 100 THEN 1 WHEN x > 99.5 THEN 0.5 ELSE 0 END AS tier "
        "FROM corpus ORDER BY ts",
        "SELECT ts, CASE WHEN sym = 'K003' THEN x ELSE 0 - x END AS signed_x "
        "FROM corpus ORDER BY ts",
        "SELECT ts, CASE WHEN y > 140 THEN y END AS high_y FROM corpus ORDER BY ts",
        "SELECT ts, sym, x FROM corpus WHERE sym LIKE 'K00%' ORDER BY ts",
        "SELECT ts, sym, x FROM corpus WHERE sym LIKE '_00_' ORDER BY ts",
        "SELECT ts, sym, x FROM corpus WHERE sym NOT LIKE '%3' ORDER BY ts",
    ]
    # IS [NOT] NULL: the total test, which no value comparison can
    # stand in for. The corpus's y column carries the nulls; ts, x and
    # sym are NOT NULL, so their arms are the constant answers.
    queries += [
        "SELECT ts, y FROM corpus WHERE y IS NULL ORDER BY ts",
        "SELECT ts, y FROM corpus WHERE y IS NOT NULL ORDER BY ts",
        "SELECT ts, y FROM corpus WHERE NOT (y IS NULL) ORDER BY ts",
        "SELECT ts, sym, x, y FROM corpus WHERE y IS NULL AND x > 100 ORDER BY ts",
        "SELECT ts, y FROM corpus WHERE y IS NULL OR y > 140 ORDER BY ts",
        "SELECT ts, x FROM corpus WHERE ts IS NOT NULL AND sym IS NOT NULL ORDER BY ts",
        "SELECT ts, x FROM corpus WHERE x IS NULL ORDER BY ts",
        "SELECT ts, CASE WHEN y IS NULL THEN 0 ELSE y END AS filled FROM corpus ORDER BY ts",
    ]
    return queries


# Queries whose rows come back in no engine-guaranteed order, because
# the only column that could have ordered them is a symbol — and symbol
# columns are unordered labels (#58 = B, ruled 2026-07-29). The REFEREE
# sorts both sides before diffing, which is better hygiene anyway: a
# grouped result is a set, and asking the engine to order it was always
# borrowing a total order from an ORDER BY the query did not need.
UNORDERED_FAMILIES = [
    "SELECT sym, count(*) AS n FROM corpus GROUP BY sym",
    "SELECT sym, count(y) AS n, avg(y) AS a FROM corpus GROUP BY sym",
    "SELECT sym, sum(x) AS s, min(x) AS lo, max(x) AS hi FROM corpus GROUP BY sym",
    "SELECT sym, count(*) AS n FROM corpus WHERE x > 100 GROUP BY sym",
    "SELECT sym, avg(x) AS a FROM corpus WHERE sym IN ('K000', 'K002', 'K004') "
    "GROUP BY sym",
    "SELECT DISTINCT sym FROM corpus",
    "SELECT sym, sum(x) AS s FROM corpus GROUP BY sym HAVING sum(x) > 100",
    "SELECT sym, count(y) AS n FROM corpus GROUP BY sym "
    "HAVING count(y) >= 1 AND sym <> 'K001'",
    "SELECT sym, avg(x) AS a FROM corpus GROUP BY sym HAVING NOT (avg(x) > 100)",
    "SELECT sym, count(*) AS n FROM corpus WHERE y IS NULL GROUP BY sym",
    "SELECT sym, count(*) AS n FROM corpus GROUP BY sym HAVING avg(y) IS NOT NULL",
    "SELECT site, count(*) AS n, avg(x) AS a FROM corpus JOIN sensors "
    "ON corpus.sym = sensors.sym GROUP BY site",
    "SELECT site, count(*) AS n FROM corpus JOIN sensors "
    "ON corpus.sym = sensors.sym WHERE calib > 1.0 GROUP BY site",
    "SELECT site, avg(x) AS a FROM corpus JOIN sensors "
    "ON corpus.sym = sensors.sym GROUP BY site HAVING sum(calib) > 100",
]

# Deliberate divergence: DuckDB sorts these, TallyDB refuses them.
# Symbol codes are per-segment first-appearance ranks, so they rank
# nothing, and ranking the labels as text would ask an engine that
# refuses to *produce* a string to order strings. The refusal is part
# of the surface, so it is checked like any other answer.
REFUSED_QUERIES = [
    "SELECT ts, sym FROM corpus ORDER BY sym",
    "SELECT ts, sym, x FROM corpus WHERE x > 100 ORDER BY sym DESC",
    "SELECT sym, count(*) AS n FROM corpus GROUP BY sym ORDER BY sym",
    "SELECT DISTINCT sym FROM corpus ORDER BY sym",
]


# (sql, canonical sort columns): ORDER BY columns with ties — verified
# by checking the sort column's sequence, then diffing under a total
# python-side re-sort, because tie order is engine-arbitrary.
TIE_QUERIES = [
    # NULLS FIRST/LAST: the null rows tie on the sort key, so their
    # internal order is engine-arbitrary; placement is what's checked.
    ("SELECT ts, y FROM corpus ORDER BY y NULLS FIRST", ["y", "ts"]),
    ("SELECT ts, y FROM corpus ORDER BY y DESC NULLS FIRST", ["y", "ts"]),
    ("SELECT ts, y FROM corpus ORDER BY y NULLS LAST", ["y", "ts"]),
]

WINDOW_QUERIES = [
    # Windows ride the compute path; DuckDB's regr_* are the oracle. The
    # WHERE strips the corpus's null y rows first (nullable window
    # arguments are a recorded limitation; null comparisons are false in
    # both engines, so both see identical rows).
    "SELECT ts, regr_slope(y, x) OVER (PARTITION BY sym ORDER BY ts "
    "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    "SELECT ts, regr_intercept(y, x) OVER (ORDER BY ts "
    "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    # M2.6: the pair statistics DuckDB also implements.
    "SELECT ts, covar_pop(y, x) OVER (PARTITION BY sym ORDER BY ts "
    "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    "SELECT ts, corr(y, x) OVER (ORDER BY ts "
    "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    "SELECT ts, covar_pop(y, x) OVER (ORDER BY ts "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    # M5.0: the one-column dispersion pair, which DuckDB also
    # implements. `x` is the ordering key's own scale (no offset in the
    # corpus, but a wide range), and the unbounded frame exercises the
    # recompute path while the trailing frames exercise the incremental
    # sweep — both must match the same oracle.
    "SELECT ts, var_pop(x) OVER (PARTITION BY sym ORDER BY ts "
    "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "ORDER BY ts",
    "SELECT ts, stddev_pop(x) OVER (ORDER BY ts "
    "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "ORDER BY ts",
    "SELECT ts, var_pop(y) OVER (ORDER BY ts "
    "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    "SELECT ts, stddev_pop(y) OVER (PARTITION BY sym ORDER BY ts "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "WHERE y > -100000 ORDER BY ts",
    # A single-row frame: population spread of one point is 0, not NULL
    # — the edge where a "needs two rows" reading would diverge.
    "SELECT ts, var_pop(x) OVER (ORDER BY ts "
    "ROWS BETWEEN 0 PRECEDING AND CURRENT ROW) AS w FROM corpus "
    "ORDER BY ts",
    # M5.1: LAG/LEAD, positional and frameless. The head/tail rows a
    # lookup cannot define must be NULL in both engines, and the
    # partitioned forms must not read across a partition boundary.
    "SELECT ts, lag(x, 1) OVER (ORDER BY ts) AS w FROM corpus ORDER BY ts",
    "SELECT ts, lead(x, 1) OVER (ORDER BY ts) AS w FROM corpus ORDER BY ts",
    "SELECT ts, lag(x, 5) OVER (ORDER BY ts) AS w FROM corpus ORDER BY ts",
    "SELECT ts, lag(x, 1) OVER (PARTITION BY sym ORDER BY ts) AS w FROM corpus "
    "ORDER BY ts",
    "SELECT ts, lead(x, 3) OVER (PARTITION BY sym ORDER BY ts) AS w FROM corpus "
    "ORDER BY ts",
    # The default offset is 1, in both engines.
    "SELECT ts, lag(x) OVER (ORDER BY ts) AS w FROM corpus ORDER BY ts",
    # M5.1: RANGE frames — bounded by ordering-key VALUE, not row count.
    # NOTE ON COVERAGE: this corpus has 5000 rows and 5000 DISTINCT
    # timestamps, so these families do NOT exercise the peer rule
    # (standard SQL ends a RANGE frame at the current row's last peer,
    # so tied rows all share one frame). What they do cover is the
    # value-span arithmetic against irregular spacing, which the row
    # cadence's jitter provides. The peer rule is covered instead by
    # `range_frames_share_one_window_across_tied_timestamps` in engine,
    # whose expected values were taken from DuckDB directly.
    "SELECT ts, sum(x) OVER (ORDER BY ts RANGE BETWEEN 500 PRECEDING "
    "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    "SELECT ts, avg(x) OVER (ORDER BY ts RANGE BETWEEN 0 PRECEDING "
    "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    "SELECT ts, count(x) OVER (ORDER BY ts RANGE BETWEEN 100 PRECEDING "
    "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    "SELECT ts, min(x) OVER (PARTITION BY sym ORDER BY ts "
    "RANGE BETWEEN 2000 PRECEDING AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    "SELECT ts, var_pop(x) OVER (ORDER BY ts RANGE BETWEEN 1000 PRECEDING "
    "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    # A span wider than the whole corpus: every frame starts at row 1.
    "SELECT ts, sum(x) OVER (ORDER BY ts RANGE BETWEEN 100000000 PRECEDING "
    "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    # M5.3: cross-sectional windows over the raw time axis — PARTITION
    # BY the instant rather than the symbol, so each row sees its own
    # instant across every symbol. Standard SQL gives an ORDER BY-less
    # window its whole partition, and DuckDB agrees, so these run the
    # same text on both sides.
    #
    # The corpus has one row per distinct ts, so `PARTITION BY ts` is a
    # partition per row — a real edge (every partition a singleton), not
    # an interesting cross-section. The bucketed forms, which do put
    # several rows in a partition, need DuckDB's `//` and so live in
    # CROSS_SECTIONAL_FAMILIES below.
    "SELECT ts, sum(x) OVER (PARTITION BY ts) AS w FROM corpus ORDER BY ts",
    # The whole snapshot as one partition — `OVER ()`, the grand total
    # beside every row.
    "SELECT ts, sum(x) OVER () AS w FROM corpus ORDER BY ts",
    # #94: scalar expressions OVER window results. These are the idioms
    # the window surface exists to serve — a row against its own frame —
    # and each is one expression, not a second query.
    "SELECT ts, x - lag(x) OVER (ORDER BY ts) AS w FROM corpus ORDER BY ts",
    "SELECT ts, x - avg(x) OVER (ORDER BY ts ROWS BETWEEN 9 PRECEDING "
    "AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    "SELECT ts, x / sum(x) OVER () AS w FROM corpus ORDER BY ts",
    # Two windows in one expression, computed independently.
    "SELECT ts, lead(x) OVER (ORDER BY ts) - lag(x) OVER (ORDER BY ts) AS w "
    "FROM corpus ORDER BY ts",
    # A window inside a scalar function call, and partitioned.
    "SELECT ts, abs(x - avg(x) OVER (PARTITION BY sym ORDER BY ts "
    "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)) AS w FROM corpus ORDER BY ts",
]


# M5.2 (#65): as-of joins, checked against the DEFINITION rather than
# against DuckDB's own ASOF JOIN.
#
# Every other family here runs identical SQL on both sides, so DuckDB's
# implementation is the reference. That is the wrong reference for this
# one: two implementations of "the latest quote at or before" can agree
# with each other and both be wrong about ties, about the empty prefix,
# or about which side the inequality binds. So the oracle side is a
# correlated scalar subquery — the textbook definition, in vanilla SQL,
# with no ASOF anything in it — and agreement means the engine computes
# what the phrase means.
#
# Ties matter and are not avoided: the fixture repeats timestamps
# deliberately, and the definition breaks the tie the way the engine
# does, by taking the last of them in storage order. `seq` is that
# order, attached by the referee when it replicates the table.
ASOF_MATCH = """
    (SELECT r.q FROM quotes r
      WHERE r.sym = t.sym AND r.qts {operator} t.ts
      ORDER BY r.qts DESC, r.seq DESC LIMIT 1)
"""

# (engine sql, oracle sql). Both must project the same column names.
ASOF_FAMILIES = [
    # The plain shapes: LEFT keeps the corpus rows whose symbol has no
    # quote yet, INNER drops them.
    (
        "SELECT ts, corpus.sym, q FROM corpus ASOF LEFT JOIN quotes "
        "ON corpus.sym = quotes.sym ORDER BY ts",
        "SELECT t.ts AS ts, t.sym AS sym, " + ASOF_MATCH.format(operator="<=") + " AS q "
        "FROM corpus t ORDER BY t.ts",
    ),
    (
        "SELECT ts, corpus.sym, q FROM corpus ASOF INNER JOIN quotes "
        "ON corpus.sym = quotes.sym ORDER BY ts",
        "SELECT * FROM (SELECT t.ts AS ts, t.sym AS sym, "
        + ASOF_MATCH.format(operator="<=")
        + " AS q FROM corpus t) WHERE q IS NOT NULL ORDER BY ts",
    ),
    # An explicit inequality restating the ordering keys, both ways
    # round: it selects the comparison and nothing else.
    (
        "SELECT ts, q FROM corpus ASOF LEFT JOIN quotes "
        "ON corpus.sym = quotes.sym AND quotes.qts < corpus.ts ORDER BY ts",
        "SELECT t.ts AS ts, " + ASOF_MATCH.format(operator="<") + " AS q "
        "FROM corpus t ORDER BY t.ts",
    ),
    (
        "SELECT ts, q FROM corpus ASOF LEFT JOIN quotes "
        "ON corpus.sym = quotes.sym AND corpus.ts >= quotes.qts ORDER BY ts",
        "SELECT t.ts AS ts, " + ASOF_MATCH.format(operator="<=") + " AS q "
        "FROM corpus t ORDER BY t.ts",
    ),
    # The joined rows are ordinary rows: filters, arithmetic, aggregates
    # and windows run over them exactly as they would over one table.
    (
        "SELECT ts, x * q AS scaled FROM corpus ASOF INNER JOIN quotes "
        "ON corpus.sym = quotes.sym WHERE x > 5 ORDER BY ts",
        "SELECT ts, x * q AS scaled FROM (SELECT t.ts AS ts, t.x AS x, "
        + ASOF_MATCH.format(operator="<=")
        + " AS q FROM corpus t) WHERE q IS NOT NULL AND x > 5 ORDER BY ts",
    ),
    (
        "SELECT ts, sum(q) OVER (PARTITION BY corpus.sym ORDER BY ts "
        "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS w "
        "FROM corpus ASOF INNER JOIN quotes ON corpus.sym = quotes.sym ORDER BY ts",
        "SELECT ts, sum(q) OVER (PARTITION BY sym ORDER BY ts "
        "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS w FROM "
        "(SELECT t.ts AS ts, t.sym AS sym, "
        + ASOF_MATCH.format(operator="<=")
        + " AS q FROM corpus t) WHERE q IS NOT NULL ORDER BY ts",
    ),
    # A join whose dimension column the SELECT never names still has to
    # match, because the WHERE reads it (#81's pushdown must not drop
    # the column the as-of match produced).
    (
        "SELECT ts FROM corpus ASOF INNER JOIN quotes "
        "ON corpus.sym = quotes.sym WHERE q > 0 ORDER BY ts",
        "SELECT ts FROM (SELECT t.ts AS ts, "
        + ASOF_MATCH.format(operator="<=")
        + " AS q FROM corpus t) WHERE q > 0 ORDER BY ts",
    ),
]

# M5.3 (F1 = d): bucketed aggregation, and FIRST/LAST.
#
# Two spellings differ from DuckDB's and both are deliberate, so these
# families carry their own oracle SQL rather than running the same text
# on both sides:
#
#  - Integer division. TallyDB's `/` between integers truncates, which
#    is ISO and PostgreSQL; DuckDB's `/` returns a DOUBLE and its `//`
#    truncates. TallyDB accepts `//` as a synonym, so the *engine* side
#    could be written either way — the ORACLE side must say `//`.
#  - FIRST/LAST have no ISO spelling. DuckDB's own `first`/`last` are
#    arrival-order aggregates, which is a different question. The
#    definition wanted here is "the value at the group's earliest/latest
#    ordering key", and DuckDB spells that `arg_min(x, ts)` /
#    `arg_max(x, ts)` — a definitional reference, not a name match.
#
# Row order is engine-arbitrary for grouped results, so the referee
# re-sorts both sides (as UNORDERED_FAMILIES does).
BUCKET_FAMILIES = [
    # A minute of nanoseconds. The corpus's 1s cadence puts ~60 rows in
    # each bucket, so the groups are neither singletons nor one blob.
    (
        "SELECT ts / 60000000000 AS bar, count(*) AS n, sum(x) AS s FROM corpus "
        "GROUP BY ts / 60000000000",
        "SELECT ts // 60000000000 AS bar, count(*) AS n, sum(x) AS s FROM corpus "
        "GROUP BY ts // 60000000000",
    ),
    # The bucket's START value — the same groups, relabelled.
    (
        "SELECT (ts / 60000000000) * 60000000000 AS bar, count(*) AS n FROM corpus "
        "GROUP BY (ts / 60000000000) * 60000000000",
        "SELECT (ts // 60000000000) * 60000000000 AS bar, count(*) AS n FROM corpus "
        "GROUP BY (ts // 60000000000) * 60000000000",
    ),
    # The OHLC shape: per symbol, per bar, open/high/low/close.
    (
        "SELECT sym, ts / 300000000000 AS bar, first(x) AS o, max(x) AS h, "
        "min(x) AS l, last(x) AS c FROM corpus GROUP BY sym, ts / 300000000000",
        "SELECT sym, ts // 300000000000 AS bar, arg_min(x, ts) AS o, max(x) AS h, "
        "min(x) AS l, arg_max(x, ts) AS c FROM corpus GROUP BY sym, ts // 300000000000",
    ),
    # FIRST/LAST over a nullable column: both sides skip nulls, so the
    # answer is the earliest/latest row that HAS a value — not NULL
    # because the earliest row happened to be missing one.
    (
        "SELECT sym, first(y) AS o, last(y) AS c FROM corpus GROUP BY sym",
        "SELECT sym, arg_min(y, ts) AS o, arg_max(y, ts) AS c FROM corpus GROUP BY sym",
    ),
    # A bare ordering key is the finest bucket there is.
    (
        "SELECT ts, count(*) AS n FROM corpus GROUP BY ts",
        "SELECT ts, count(*) AS n FROM corpus GROUP BY ts",
    ),
    # Buckets compose with everything grouping already did: WHERE
    # before, HAVING after.
    (
        "SELECT ts / 60000000000 AS bar, avg(x) AS a FROM corpus WHERE x > 100 "
        "GROUP BY ts / 60000000000 HAVING count(*) > 20",
        "SELECT ts // 60000000000 AS bar, avg(x) AS a FROM corpus WHERE x > 100 "
        "GROUP BY ts // 60000000000 HAVING count(*) > 20",
    ),
]

# The cross-sectional families whose partition is a BUCKET of the time
# axis. Paired SQL for the same reason BUCKET_FAMILIES is paired:
# TallyDB's `/` between integers truncates, DuckDB's returns a DOUBLE.
# That difference is not cosmetic here — a float bucket of a nanosecond
# stamp keeps its fractional part, so DuckDB would partition per ROW
# rather than per bar, and the two engines would be answering different
# questions. The oracle side says `//`.
CROSS_SECTIONAL_FAMILIES = [
    (
        "SELECT ts, avg(x) OVER (PARTITION BY ts / 60000000000) AS w FROM corpus "
        "ORDER BY ts",
        "SELECT ts, avg(x) OVER (PARTITION BY ts // 60000000000) AS w FROM corpus "
        "ORDER BY ts",
    ),
    (
        "SELECT ts, count(x) OVER (PARTITION BY ts / 60000000000) AS w FROM corpus "
        "ORDER BY ts",
        "SELECT ts, count(x) OVER (PARTITION BY ts // 60000000000) AS w FROM corpus "
        "ORDER BY ts",
    ),
    (
        "SELECT ts, max(x) OVER (PARTITION BY ts / 300000000000) AS w FROM corpus "
        "ORDER BY ts",
        "SELECT ts, max(x) OVER (PARTITION BY ts // 300000000000) AS w FROM corpus "
        "ORDER BY ts",
    ),
    (
        "SELECT ts, var_pop(x) OVER (PARTITION BY ts / 300000000000) AS w FROM corpus "
        "ORDER BY ts",
        "SELECT ts, var_pop(x) OVER (PARTITION BY ts // 300000000000) AS w FROM corpus "
        "ORDER BY ts",
    ),
    # The cross-sectional weight: each row's share of its own bar.
    # This needs the partition (M5.3) and the composition (#94) at once.
    (
        "SELECT ts, x / sum(x) OVER (PARTITION BY ts / 60000000000) AS w FROM corpus "
        "ORDER BY ts",
        "SELECT ts, x / sum(x) OVER (PARTITION BY ts // 60000000000) AS w FROM corpus "
        "ORDER BY ts",
    ),
    # Cross-sectional demeaning, the other half of a z-score.
    (
        "SELECT ts, x - avg(x) OVER (PARTITION BY ts / 60000000000) AS w FROM corpus "
        "ORDER BY ts",
        "SELECT ts, x - avg(x) OVER (PARTITION BY ts // 60000000000) AS w FROM corpus "
        "ORDER BY ts",
    ),
    # A bucket partition is not restricted to the cross-sectional
    # reading: ordered inside the bucket, it is a frame that resets at
    # each bar boundary.
    (
        "SELECT ts, sum(x) OVER (PARTITION BY ts / 60000000000 ORDER BY ts "
        "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
        "SELECT ts, sum(x) OVER (PARTITION BY ts // 60000000000 ORDER BY ts "
        "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
    ),
]

# Families that DIVIDE by a window result, where the two engines'
# division-by-zero rules differ and the difference is by ruling, not by
# accident. TallyDB's arithmetic is IEEE — `x/0` is ±inf or NaN, and NaN
# is a value (decision D2, DESIGN.md *Null, NaN, and ordering
# semantics*). DuckDB returns NULL for division by zero. A rolling
# z-score hits this on every partition's first row, where the frame is
# one row, the deviation is 0 and the spread is 0.
#
# So the referee normalizes NaN to NULL on BOTH sides here, and only
# here: that compares the numbers both engines agree are numbers,
# without either pretending the other's zero-division rule is its own.
# Every other window family keeps the one-sided normalization, so a
# stray engine NaN still fails them.
IEEE_DIVISION_FAMILIES = [
    # The rolling z-score, expressible in SQL at all only because M5.0
    # added stddev_pop and #94 added the composition.
    "SELECT ts, (x - avg(x) OVER (PARTITION BY sym ORDER BY ts "
    "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)) / stddev_pop(x) OVER "
    "(PARTITION BY sym ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) "
    "AS w FROM corpus ORDER BY ts",
    # The same shape unpartitioned, over a wider frame.
    "SELECT ts, (x - avg(x) OVER (ORDER BY ts ROWS BETWEEN 49 PRECEDING "
    "AND CURRENT ROW)) / stddev_pop(x) OVER (ORDER BY ts "
    "ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS w FROM corpus ORDER BY ts",
]

EIGEN_PRECEDING = 19


def numpy_eigen_check(lib, inputs) -> None:
    """eigen_max has no DuckDB counterpart: recompute every window's
    largest 2x2 population-covariance eigenvalue with NumPy instead."""
    import numpy as np

    sql = (
        "SELECT ts, eigen_max(y, x) OVER (PARTITION BY sym ORDER BY ts "
        f"ROWS BETWEEN {EIGEN_PRECEDING} PRECEDING AND CURRENT ROW) AS w "
        "FROM corpus WHERE y > -100000 ORDER BY ts"
    )
    engine = tallydb_query(lib, sql)
    rows = sorted(
        (
            (ts, sym, x, y)
            for ts, sym, x, y in zip(
                inputs["ts"].to_pylist(),
                inputs["sym"].to_pylist(),
                inputs["x"].to_pylist(),
                inputs["y"].to_pylist(),
            )
            if y is not None
        ),
        key=lambda row: row[0],
    )
    per_sym: dict[str, list[tuple]] = {}
    expected_by_ts: dict[int, float | None] = {}
    for ts, sym, x, y in rows:
        history = per_sym.setdefault(sym, [])
        history.append((x, y))
        window = history[-(EIGEN_PRECEDING + 1) :]
        if len(window) < 2:
            expected_by_ts[ts] = None
            continue
        wx = np.array([w[0] for w in window])
        wy = np.array([w[1] for w in window])
        covariance = np.array(
            [
                [np.mean((wy - wy.mean()) ** 2), np.mean((wy - wy.mean()) * (wx - wx.mean()))],
                [np.mean((wy - wy.mean()) * (wx - wx.mean())), np.mean((wx - wx.mean()) ** 2)],
            ]
        )
        expected_by_ts[ts] = float(np.linalg.eigvalsh(covariance)[-1])
    engine_ts = engine["ts"].to_pylist()
    engine_w = engine["w"].to_pylist()
    for row, (ts, value) in enumerate(zip(engine_ts, engine_w)):
        expected = expected_by_ts[ts]
        if not close(value, expected):
            sys.exit(
                f"FAIL eigen_max vs numpy: row {row} engine {value!r} "
                f"vs numpy {expected!r}"
            )
    print(f"PASS eigen_max vs numpy ({len(engine_ts)} rows)")


def compare_tables(sql: str, engine: pa.Table, oracle: pa.Table, window: bool) -> None:
    if engine.num_rows != oracle.num_rows:
        sys.exit(
            f"FAIL {sql}\n  row count: engine {engine.num_rows} "
            f"vs duckdb {oracle.num_rows}"
        )
    if engine.column_names != oracle.column_names:
        sys.exit(
            f"FAIL {sql}\n  columns: engine {engine.column_names} "
            f"vs duckdb {oracle.column_names}"
        )
    for column in engine.column_names:
        engine_values = engine[column].to_pylist()
        oracle_values = oracle[column].to_pylist()
        if window and column == "w":
            oracle_values = nan_to_none(oracle_values)
        for row, (engine_value, oracle_value) in enumerate(
            zip(engine_values, oracle_values)
        ):
            if isinstance(engine_value, float) or isinstance(oracle_value, float):
                equal = close(engine_value, oracle_value)
            else:
                equal = engine_value == oracle_value
            if not equal:
                sys.exit(
                    f"FAIL {sql}\n  {column} row {row}: engine "
                    f"{engine_value!r} vs duckdb {oracle_value!r}"
                )


def main() -> None:
    lib = load_library()
    lib.tallydb_corpus_query_stream.restype = ctypes.c_int32
    inputs = read_stream_hook(lib, "tallydb_corpus_inputs_stream")
    dimension = read_stream_hook(lib, "tallydb_corpus_dimension_stream")
    connection = duckdb.connect()
    connection.register("corpus_input", inputs)
    connection.execute("CREATE TABLE corpus AS SELECT * FROM corpus_input")
    connection.register("sensors_input", dimension)
    connection.execute("CREATE TABLE sensors AS SELECT * FROM sensors_input")
    # The quote history, numbered in the order the engine stores it —
    # `seq` is what breaks a tie between quotes sharing a timestamp, so
    # it has to be attached here, before DuckDB is free to reorder.
    quotes = read_stream_hook(lib, "tallydb_corpus_quotes_stream")
    quotes = quotes.append_column(
        "seq", pa.array(range(quotes.num_rows), type=pa.int64())
    )
    connection.register("quotes_input", quotes)
    connection.execute("CREATE TABLE quotes AS SELECT * FROM quotes_input")

    passed = 0
    for sql in families():
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        compare_tables(sql, engine, oracle, window=False)
        passed += 1
    for sql in UNORDERED_FAMILIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        if engine.column_names != oracle.column_names:
            sys.exit(
                f"FAIL {sql}\n  columns: engine {engine.column_names} "
                f"vs duckdb {oracle.column_names}"
            )
        # Floats are compared exactly here; the aggregates in these
        # families are sums and counts over the same rows in the same
        # per-group order, so the two engines produce identical bits.
        if sorted_rows(engine) != sorted_rows(oracle):
            sys.exit(f"FAIL {sql}\n  row multisets differ")
        passed += 1
    for sql in REFUSED_QUERIES:
        if not tallydb_refuses(lib, sql):
            sys.exit(f"FAIL {sql}\n  ORDER BY on a symbol column must be refused")
        passed += 1
    for sql, canonical in TIE_QUERIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        # The ORDER BY column itself must come back correctly ordered...
        tail = sql.split("ORDER BY ")[1]
        order_column = tail.split()[0]
        descending = " DESC" in tail
        nulls_first = "NULLS FIRST" in tail
        sequence = engine[order_column].to_pylist()
        values = sorted(
            (v for v in sequence if v is not None), reverse=descending
        )
        nones = [None] * (len(sequence) - len(values))
        expected = nones + values if nulls_first else values + nones
        if sequence != expected:
            sys.exit(f"FAIL {sql}\n  engine '{order_column}' not in order")
        # ...and the row multisets must agree, under the referee's own
        # total re-sort.
        if sorted_rows(engine, canonical) != sorted_rows(oracle, canonical):
            sys.exit(f"FAIL {sql}\n  row multisets differ")
        passed += 1
    for sql in WINDOW_QUERIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        compare_tables(sql, engine, oracle, window=True)
        passed += 1
    # The as-of families claim to cover the tie rule; check that before
    # trusting them. (The M5.1 lesson: a corpus with 5000 distinct
    # timestamps cannot exercise a rule about equal ones, and a comment
    # saying otherwise is just a comment.)
    ties = connection.execute(
        "SELECT count(*) FROM (SELECT sym, qts FROM quotes "
        "GROUP BY sym, qts HAVING count(*) > 1)"
    ).fetchone()[0]
    if ties < 50:
        sys.exit(
            f"FAIL the quote history has only {ties} tied (sym, qts) "
            "timestamps — the as-of families cannot cover the tie rule"
        )
    for sql in IEEE_DIVISION_FAMILIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        engine_w = nan_to_none(engine["w"].to_pylist())
        oracle_w = nan_to_none(oracle["w"].to_pylist())
        if len(engine_w) != len(oracle_w):
            sys.exit(f"FAIL {sql}\n  row counts differ")
        # A sanity floor: if the normalization swallowed everything the
        # family would pass vacuously, so require most rows to be real
        # numbers that actually got compared.
        compared = sum(1 for value in oracle_w if value is not None)
        if compared < len(oracle_w) // 2:
            sys.exit(
                f"FAIL {sql}\n  only {compared}/{len(oracle_w)} rows are "
                "non-NULL — the NaN normalization would hide disagreement"
            )
        for row, (mine, theirs) in enumerate(zip(engine_w, oracle_w)):
            if not close(mine, theirs):
                sys.exit(f"FAIL {sql}\n  w row {row}: engine {mine!r} vs duckdb {theirs!r}")
        passed += 1
    for sql, definition in CROSS_SECTIONAL_FAMILIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(definition).to_arrow_table()
        compare_tables(sql, engine, oracle, window=True)
        passed += 1
    for sql, definition in BUCKET_FAMILIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(definition).to_arrow_table()
        if engine.column_names != oracle.column_names:
            sys.exit(
                f"FAIL {sql}\n  columns: engine {engine.column_names} "
                f"vs duckdb {oracle.column_names}"
            )
        engine_rows = sorted_rows(engine)
        oracle_rows = sorted_rows(oracle)
        if len(engine_rows) != len(oracle_rows):
            sys.exit(
                f"FAIL {sql}\n  group count: engine {len(engine_rows)} "
                f"vs duckdb {len(oracle_rows)}"
            )
        for row, (mine, theirs) in enumerate(zip(engine_rows, oracle_rows)):
            for column, (a, b) in enumerate(zip(mine, theirs)):
                equal = close(a, b) if isinstance(a, float) or isinstance(b, float) else a == b
                if not equal:
                    sys.exit(
                        f"FAIL {sql}\n  row {row} column "
                        f"{engine.column_names[column]}: engine {a!r} vs duckdb {b!r}"
                    )
        passed += 1
    for sql, definition in ASOF_FAMILIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(definition).to_arrow_table()
        compare_tables(sql, engine, oracle, window=True)
        passed += 1
    numpy_eigen_check(lib, inputs)
    print(
        f"Differential: {passed} generated queries agree with DuckDB "
        f"{duckdb.__version__} over {inputs.num_rows} corpus rows "
        f"and {quotes.num_rows} quotes"
    )


if __name__ == "__main__":
    main()
