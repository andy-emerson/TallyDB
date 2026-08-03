#!/usr/bin/env python3
"""#83 tranche-3 join-view oracle: each join view's answer vs DuckDB.

Drives the `oracle-harness` join-view context in libengine: a fact
table, a quote history, a small keyed dimension, and three maintained
join views — the ASOF blotter, bucketed aggregates over the ASOF join,
and star bars over the equi join. Every statement is mirrored into
DuckDB, whose NATIVE ASOF JOIN independently recomputes the two
as-of views (the star view diffs against DuckDB's ordinary join); at
each scripted checkpoint every view is diffed against DuckDB running
the definition from scratch. The script walks the states the tranche-3
design bets on: facts running ahead of quotes (the ceiling), in-order
quote appends while stale (must stay exact, unrefreshed), late quotes
below the ceiling (the correction interval), quote amends and deletes,
a dimension change (the star rebuild), fact corrections, compaction,
and a full reopen (the v3 record round trip).

Quote timestamps stay unique per symbol so DuckDB's tie rule never
meets ours — ties are pinned in-crate against the ruled `_seq` rule.

Usage: m5_join_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
from pyarrow.cffi import ffi

BLOTTER_COLUMNS = "ts, sym, x, bid"
BLOTTER_MIRROR = (
    "SELECT ts, trades.sym AS sym, x, bid FROM trades ASOF LEFT JOIN quotes "
    "ON trades.sym = quotes.sym AND trades.ts >= quotes.qts"
)
JBARS_COLUMNS = "sym, bar, n, ab, s"
JBARS_MIRROR = (
    "SELECT trades.sym AS sym, ts // 5 AS bar, count(*) AS n, avg(bid) AS ab, "
    "sum(x) AS s FROM trades ASOF LEFT JOIN quotes "
    "ON trades.sym = quotes.sym AND trades.ts >= quotes.qts "
    "GROUP BY trades.sym, ts // 5"
)
SBARS_COLUMNS = "sector, bar, s, n"
SBARS_MIRROR = (
    "SELECT sector, ts // 5 AS bar, sum(x) AS s, count(*) AS n "
    "FROM trades JOIN dim ON trades.sym = dim.sym GROUP BY sector, ts // 5"
)

# The scripted drive. Trades arrive in ts order; quotes in qts order
# except the deliberate late arrivals; every checkpoint diffs all
# three views.
SCRIPT = [
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES "
            + ", ".join(f"({t}, 'A', {t}.5, 0.0)" if t % 2 == 0
                        else f"({t}, 'B', {t}.25, 1.0)"
                        for t in range(0, 40))),
    ("sql", "INSERT INTO quotes (qts, sym, bid) VALUES "
            + ", ".join(f"({q}, 'A', {q}.1)" if q % 2 == 0
                        else f"({q}, 'B', {q}.2)"
                        for q in range(0, 31, 3))),
    ("sql", "INSERT INTO dim (id, sym, sector) VALUES "
            "(0, 'A', 'tech'), (1, 'B', 'energy')"),
    ("check", None),           # stale: every answer is a live join
    ("refresh", None),
    ("check", None),           # fresh; facts 31..39 above the ceiling
    ("sql", "INSERT INTO quotes (qts, sym, bid) VALUES (32, 'A', 32.1), "
            "(35, 'B', 35.2)"),
    ("check", None),           # in-order arrivals: exact while stale
    ("refresh", None),
    ("check", None),
    ("sql", "INSERT INTO quotes (qts, sym, bid) VALUES (7, 'A', 99.9)"),
    ("check", None),           # late quote below the ceiling, unrefreshed
    ("refresh", None),
    ("check", None),
    ("sql", "UPDATE quotes SET bid = 55.5 WHERE qts = 9"),
    ("check", None),           # quote amend, unrefreshed
    ("sql", "DELETE FROM quotes WHERE qts = 12"),
    ("check", None),           # quote delete: matches fall back
    ("refresh", None),
    ("check", None),
    ("sql", "UPDATE dim SET sector = 'ai' WHERE sym = 'A'"),
    ("check", None),           # dim change: the star answer moves whole
    ("refresh", None),
    ("check", None),
    ("sql", "UPDATE trades SET x = 500.0 WHERE ts = 8"),
    ("sql", "DELETE FROM trades WHERE ts >= 38"),
    ("check", None),           # fact corrections, unrefreshed
    ("refresh", None),
    ("check", None),
    ("compact", None),
    ("check", None),
    ("reopen", None),          # the v3 record round trip, mid-life
    ("check", None),
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES (40, 'A', 40.5, 0.0), "
            "(41, 'B', 41.25, 1.0)"),
    ("check", None),
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
    lib.tallydb_join_view_open.restype = ctypes.c_void_p
    lib.tallydb_join_view_definitions.restype = ctypes.c_char_p
    lib.tallydb_join_view_statement.restype = ctypes.c_int64
    lib.tallydb_join_view_statement.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tallydb_join_view_refresh.restype = ctypes.c_int64
    lib.tallydb_join_view_refresh.argtypes = [ctypes.c_void_p]
    lib.tallydb_join_view_query_stream.restype = ctypes.c_int32
    lib.tallydb_join_view_query_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.tallydb_join_view_compact.restype = ctypes.c_int32
    lib.tallydb_join_view_compact.argtypes = [ctypes.c_void_p]
    lib.tallydb_join_view_reopen.restype = ctypes.c_int32
    lib.tallydb_join_view_reopen.argtypes = [ctypes.c_void_p]
    lib.tallydb_join_view_close.argtypes = [ctypes.c_void_p]
    return lib


def engine_rows(lib, context, sql: str) -> list:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    if lib.tallydb_join_view_query_stream(context, sql.encode(), ctypes.c_void_p(ptr)) != 0:
        sys.exit(f"engine query failed: {sql}")
    table = pa.RecordBatchReader._import_from_c(ptr).read_all()
    rows = []
    for batch in table.to_batches():
        columns = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
        rows.extend(zip(*columns) if columns else [])
    return sorted(rows, key=repr)


def duckdb_rows(oracle, sql: str) -> list:
    return sorted(oracle.execute(sql).fetchall(), key=repr)


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


def diff(name, checks, engine, expected):
    if not engine:
        sys.exit(f"vacuous check: {name} answered no rows at check {checks}")
    if not rows_equal(engine, expected):
        sys.exit(
            f"{name} diverged from DuckDB recompute at check {checks}:\n"
            f"  engine: {engine}\n  duckdb: {expected}"
        )


def main() -> None:
    lib = load_library()
    definitions = lib.tallydb_join_view_definitions().decode()
    for fragment in ("ASOF LEFT JOIN quotes", "GROUP BY sym, ts / 5",
                     "JOIN dim", "GROUP BY sector"):
        if fragment not in definitions:
            sys.exit(f"definition drift: {fragment!r} not in {definitions!r}")
    context = ctypes.c_void_p(lib.tallydb_join_view_open())
    oracle = duckdb.connect()
    oracle.execute("CREATE TABLE trades (ts BIGINT, sym VARCHAR, x DOUBLE, y DOUBLE)")
    oracle.execute("CREATE TABLE quotes (qts BIGINT, sym VARCHAR, bid DOUBLE)")
    oracle.execute("CREATE TABLE dim (id BIGINT, sym VARCHAR, sector VARCHAR)")
    checks = 0
    try:
        for kind, payload in SCRIPT:
            if kind == "sql":
                changed = lib.tallydb_join_view_statement(context, payload.encode())
                if changed < 0:
                    sys.exit(f"engine refused: {payload}")
                oracle.execute(payload)
            elif kind == "refresh":
                if lib.tallydb_join_view_refresh(context) < 0:
                    sys.exit("refresh failed")
            elif kind == "compact":
                if lib.tallydb_join_view_compact(context) != 0:
                    sys.exit("compact failed")
            elif kind == "reopen":
                if lib.tallydb_join_view_reopen(context) != 0:
                    sys.exit("reopen failed")
            else:
                diff("blotter", checks,
                     engine_rows(lib, context, f"SELECT {BLOTTER_COLUMNS} FROM blotter"),
                     duckdb_rows(oracle, BLOTTER_MIRROR))
                diff("jbars", checks,
                     engine_rows(lib, context, f"SELECT {JBARS_COLUMNS} FROM jbars"),
                     duckdb_rows(oracle, JBARS_MIRROR))
                diff("sbars", checks,
                     engine_rows(lib, context, f"SELECT {SBARS_COLUMNS} FROM sbars"),
                     duckdb_rows(oracle, SBARS_MIRROR))
                checks += 1
                print(f"PASS check {checks}")
    finally:
        lib.tallydb_join_view_close(context)
    print(
        f"Maintained join-view families validated end-to-end: {checks} checks "
        f"x {{blotter, asof-bars, star-bars}} against DuckDB recompute — "
        f"native ASOF JOIN for the as-of shapes, ordinary join for the star "
        f"— across stale/late-quote/dim-changed/compacted/reopened states "
        f"(duckdb {duckdb.__version__})"
    )


if __name__ == "__main__":
    main()
