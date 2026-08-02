#!/usr/bin/env python3
"""M5/#83 maintained-view oracle: the view's answer vs DuckDB recompute.

Drives the `oracle-harness` view context in libengine: one persistent
source table and one maintained bucketed view over it. Every statement
this script sends to the engine is mirrored into DuckDB; after EVERY
step — whether or not a refresh has run — the engine's answer to
`SELECT ... FROM bars` (the union read: materialized clean buckets plus
a live fold of whatever the view's stamp does not cover) is diffed
against DuckDB running the definition from scratch over the mirrored
rows. Refreshes, a compaction (kills move to history — the derivation
branch nothing else reaches end-to-end), and a full close-and-reopen of
both directories are interleaved, so stale, fresh, corrected,
compacted, and reopened states all meet the same check: the view is
exact at every knowledge coordinate.

Usage: m5_view_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
from pyarrow.cffi import ffi

# The view's SELECT list, in definition order (the engine's definition
# is fetched from the harness, so the two cannot drift; this is only
# the read-back projection and the DuckDB mirror of the fold).
VIEW_COLUMNS = "sym, bar, n, s, lo, hi, o, c"
DUCKDB_DEFINITION = (
    "SELECT sym, ts // 5 AS bar, count(*) AS n, sum(x) AS s, min(x) AS lo, "
    "max(x) AS hi, arg_min(x, ts) AS o, arg_max(x, ts) AS c "
    "FROM trades GROUP BY sym, ts // 5"
)

# The scripted drive: (kind, payload). Timestamps are unique per symbol
# by construction, so FIRST/LAST vs arg_min/arg_max never meet a tie —
# the tie rule has its own families in m2_differential_oracle.py.
SCRIPT = [
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES "
            + ", ".join(f"({t}, 'AAPL', {t}.5, 0.0)" for t in range(0, 40, 2))),
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES "
            + ", ".join(f"({t}, 'MSFT', {t}.25, 1.0)" for t in range(1, 40, 2))),
    ("check", None),           # stale: stamp 0, whole answer is the live fold
    ("refresh", None),
    ("check", None),           # fresh
    ("sql", "UPDATE trades SET x = 999.0 WHERE ts = 7"),
    ("check", None),           # dirty bucket, unrefreshed
    ("sql", "DELETE FROM trades WHERE ts >= 35"),
    ("check", None),
    ("refresh", None),
    ("check", None),
    ("sql", "UPDATE trades SET ts = 12 WHERE ts = 3"),   # cross-bucket move
    ("check", None),
    ("compact", None),         # the pending kill becomes history
    ("sql", "DELETE FROM trades WHERE ts = 20"),
    ("compact", None),         # a kill compacted BEFORE any refresh
    ("check", None),
    ("refresh", None),
    ("check", None),
    ("reopen", None),          # the storage round trip, mid-life
    ("check", None),
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES (40, 'AAPL', 40.5, 0.0), "
            "(41, 'MSFT', 41.25, 1.0)"),
    ("check", None),           # stale again after reopen
    ("refresh", None),
    ("check", None),
]


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
    lib = ctypes.CDLL(str(path))
    lib.tallydb_view_open.restype = ctypes.c_void_p
    lib.tallydb_view_definition.restype = ctypes.c_char_p
    lib.tallydb_view_statement.restype = ctypes.c_int64
    lib.tallydb_view_statement.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tallydb_view_refresh.restype = ctypes.c_int64
    lib.tallydb_view_refresh.argtypes = [ctypes.c_void_p]
    lib.tallydb_view_query_stream.restype = ctypes.c_int32
    lib.tallydb_view_query_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.tallydb_view_compact.restype = ctypes.c_int32
    lib.tallydb_view_compact.argtypes = [ctypes.c_void_p]
    lib.tallydb_view_reopen.restype = ctypes.c_int32
    lib.tallydb_view_reopen.argtypes = [ctypes.c_void_p]
    lib.tallydb_view_close.argtypes = [ctypes.c_void_p]
    return lib


def engine_view_rows(lib, context) -> list:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    sql = f"SELECT {VIEW_COLUMNS} FROM bars".encode()
    if lib.tallydb_view_query_stream(context, sql, ctypes.c_void_p(ptr)) != 0:
        sys.exit("engine view query failed")
    table = pa.RecordBatchReader._import_from_c(ptr).read_all()
    rows = []
    for batch in table.to_batches():
        columns = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
        rows.extend(zip(*columns) if columns else [])
    return sorted(rows, key=repr)


def duckdb_rows(oracle) -> list:
    rows = oracle.execute(DUCKDB_DEFINITION).fetchall()
    return sorted(rows, key=repr)


def rows_equal(left, right) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, float) and isinstance(y, float):
                if not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12):
                    return False
            elif x != y:
                return False
    return True


def main() -> None:
    lib = load_library()
    definition = lib.tallydb_view_definition().decode()
    # The engine-side definition names the same shape this script
    # mirrors; a drift fails here, loudly, before any diff runs.
    for fragment in ("ts / 5", "first(x)", "last(x)", "GROUP BY sym"):
        if fragment not in definition:
            sys.exit(f"definition drift: {fragment!r} not in {definition!r}")
    context = ctypes.c_void_p(lib.tallydb_view_open())
    oracle = duckdb.connect()
    oracle.execute("CREATE TABLE trades (ts BIGINT, sym VARCHAR, x DOUBLE, y DOUBLE)")
    checks = 0
    try:
        for kind, payload in SCRIPT:
            if kind == "sql":
                changed = lib.tallydb_view_statement(context, payload.encode())
                if changed < 0:
                    sys.exit(f"engine refused: {payload}")
                oracle.execute(payload)
            elif kind == "refresh":
                if lib.tallydb_view_refresh(context) < 0:
                    sys.exit("refresh failed")
            elif kind == "compact":
                if lib.tallydb_view_compact(context) != 0:
                    sys.exit("compact failed")
            elif kind == "reopen":
                if lib.tallydb_view_reopen(context) != 0:
                    sys.exit("reopen failed")
            else:
                engine = engine_view_rows(lib, context)
                expected = duckdb_rows(oracle)
                if not engine:
                    sys.exit("vacuous check: the view answered no rows")
                if not rows_equal(engine, expected):
                    sys.exit(
                        "view diverged from DuckDB recompute at check "
                        f"{checks}:\n  engine: {engine}\n  duckdb: {expected}"
                    )
                checks += 1
                print(f"PASS check {checks} ({len(engine)} groups)")
    finally:
        lib.tallydb_view_close(context)
    print(
        f"Maintained-view family validated end-to-end: {checks} checks "
        f"across stale/fresh/corrected/compacted/reopened states "
        f"(duckdb {duckdb.__version__})"
    )


if __name__ == "__main__":
    main()
