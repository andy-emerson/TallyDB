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
    numpy_eigen_check(lib, inputs)
    print(
        f"Differential: {passed} generated queries agree with DuckDB "
        f"{duckdb.__version__} over {inputs.num_rows} corpus rows"
    )


if __name__ == "__main__":
    main()
