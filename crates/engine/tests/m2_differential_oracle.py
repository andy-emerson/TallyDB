#!/usr/bin/env python3
"""M2.4 differential oracle: generated query families vs DuckDB.

The generated side of the differential harness: this script owns query
generation (there is no second list to keep in sync in Rust — the SQL
travels over the C ABI), runs every query against both TallyDB's corpus
fixture and a DuckDB replica of the same rows, and diffs the results.
Every query carries ORDER BY over the unique `ts` (or over the grouped
key), so both engines agree on a total order and the diff is
row-for-row.

Known, deliberate divergences the generator avoids:
  - SUM over an i64 column: DuckDB promotes to HUGEINT; TallyDB keeps
    exact i64 and errors loudly on overflow. The families sum f64
    columns only.
  - DuckDB encodes undefined regressions as NaN where TallyDB (and
    NumPy) use NULL; window comparisons normalize NaN to None.

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
    # Passthrough with ordering and paging.
    queries += [
        "SELECT ts, sym, x, y FROM corpus ORDER BY ts",
        "SELECT ts, x FROM corpus ORDER BY ts DESC LIMIT 100",
        "SELECT ts, x FROM corpus ORDER BY x LIMIT 50 OFFSET 25",
        "SELECT ts, y FROM corpus ORDER BY y DESC LIMIT 40",
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
    # Aggregates: global and grouped, nulls exercised through y.
    queries += [
        "SELECT count(*) AS n FROM corpus",
        "SELECT count(y) AS n FROM corpus",
        "SELECT sum(x) AS s, avg(x) AS a, min(x) AS lo, max(x) AS hi FROM corpus",
        "SELECT avg(y) AS a, min(y) AS lo, max(y) AS hi FROM corpus",
        "SELECT min(ts) AS lo, max(ts) AS hi, count(*) AS n FROM corpus",
        "SELECT sym, count(*) AS n FROM corpus GROUP BY sym ORDER BY sym",
        "SELECT sym, count(y) AS n, avg(y) AS a FROM corpus GROUP BY sym ORDER BY sym",
        "SELECT sym, sum(x) AS s, min(x) AS lo, max(x) AS hi FROM corpus "
        "GROUP BY sym ORDER BY sym",
        "SELECT sym, count(*) AS n FROM corpus WHERE x > 100 GROUP BY sym ORDER BY sym",
        "SELECT sym, avg(x) AS a FROM corpus WHERE sym IN ('K000', 'K002', 'K004') "
        "GROUP BY sym ORDER BY sym",
        "SELECT count(*) AS n FROM corpus WHERE x > 1e12",
    ]
    return queries


# (sql, canonical sort columns): ORDER BY columns with ties — verified
# by checking the sort column's sequence, then diffing under a total
# python-side re-sort, because tie order is engine-arbitrary.
TIE_QUERIES = [
    ("SELECT ts, sym FROM corpus ORDER BY sym", ["sym", "ts"]),
    ("SELECT ts, sym, x FROM corpus WHERE x > 100 ORDER BY sym DESC", ["sym", "ts"]),
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
]


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
    connection = duckdb.connect()
    connection.register("corpus_input", inputs)
    connection.execute("CREATE TABLE corpus AS SELECT * FROM corpus_input")

    passed = 0
    for sql in families():
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        compare_tables(sql, engine, oracle, window=False)
        passed += 1
    for sql, canonical in TIE_QUERIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        # The ORDER BY column itself must come back correctly ordered...
        order_column = sql.split("ORDER BY ")[1].split()[0]
        descending = sql.rstrip().endswith("DESC")
        sequence = engine[order_column].to_pylist()
        expected = sorted(sequence, reverse=descending)
        if sequence != expected:
            sys.exit(f"FAIL {sql}\n  engine '{order_column}' not in order")
        # ...and the row multisets must agree, under a total re-sort
        # (python-side: pyarrow cannot sort dictionary columns).
        def rows_of(table: pa.Table) -> list[tuple]:
            columns = [table[c].to_pylist() for c in table.column_names]
            rows = list(zip(*columns))
            order = [table.column_names.index(c) for c in canonical]
            return sorted(
                rows, key=lambda row: tuple(row[i] for i in order)
            )
        if rows_of(engine) != rows_of(oracle):
            sys.exit(f"FAIL {sql}\n  row multisets differ")
        passed += 1
    for sql in WINDOW_QUERIES:
        engine = tallydb_query(lib, sql)
        oracle = connection.execute(sql).to_arrow_table()
        compare_tables(sql, engine, oracle, window=True)
        passed += 1
    print(
        f"M2.4 differential: {passed} generated queries agree with DuckDB "
        f"{duckdb.__version__} over {inputs.num_rows} corpus rows"
    )


if __name__ == "__main__":
    main()
