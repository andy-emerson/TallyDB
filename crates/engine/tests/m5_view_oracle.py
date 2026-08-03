#!/usr/bin/env python3
"""M5/#83 maintained-view oracle: each view's answer vs DuckDB recompute.

Drives the `oracle-harness` view context in libengine: one persistent
source table and three maintained views over it — bucketed (tranche 1),
running, and cumulative (tranche 2). Every statement this script sends
to the engine is mirrored into DuckDB; at eleven checkpoints — placed
after the states that matter, refreshed or not — each view's answer
(the union read: materialized clean buckets plus a live fold of
whatever the view's stamp does not cover; for tranche 2, partials
recombined at read) is diffed against DuckDB running the definition
from scratch over the mirrored rows. Refreshes, a compaction (kills
move to history — the derivation branch nothing else reaches
end-to-end), and a full close-and-reopen of all directories are
interleaved, so stale, fresh, corrected, compacted, and reopened
states all meet the same check: the view equals recompute, whatever
the history.

The cumulative view runs two probes. The RANGED read (`WHERE ts >= 25`,
above every correction the script makes) exercises the boundary +
assembly split and holds at every checkpoint — a correction below the
floor reaches the answer only through the boundary, and the stray
uncompacted segment is never read. The FULL read recomputes the
windows over the whole source, so it holds only where the source is
window-ordered; at the checkpoints between a correction and its
compaction the script instead asserts the engine REFUSES loudly (the
same refusal the base's windows give) rather than answering wrongly.

Usage: m5_view_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import os
import sys
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
from pyarrow.cffi import ffi

# Each view's read-back projection, in definition order (the engine's
# definitions are fetched from the harness, so the two cannot drift;
# these are only the SELECT lists and the DuckDB mirrors of the folds).
BARS_COLUMNS = "sym, bar, n, s, lo, hi, o, c"
BARS_MIRROR = (
    "SELECT sym, ts // 5 AS bar, count(*) AS n, sum(x) AS s, min(x) AS lo, "
    "max(x) AS hi, arg_min(x, ts) AS o, arg_max(x, ts) AS c "
    "FROM trades GROUP BY sym, ts // 5"
)
TOTALS_COLUMNS = "sym, n, s, a, lo, hi, o, c"
TOTALS_MIRROR = (
    "SELECT sym, count(*) AS n, sum(x) AS s, avg(x) AS a, min(x) AS lo, "
    "max(x) AS hi, arg_min(x, ts) AS o, arg_max(x, ts) AS c "
    "FROM trades GROUP BY sym"
)
CUM_COLUMNS = "ts, sym, cs, cn, ca, clo, chi"
# The cumulative definition is standard SQL: DuckDB runs it verbatim
# (fetched from the harness at startup). The ranged floor sits above
# every correction the script makes.
CUM_FLOOR = 25

# The scripted drive: (kind, payload). Timestamps are unique per symbol
# by construction, so FIRST/LAST vs arg_min/arg_max never meet a tie —
# the tie rule has its own families in m2_differential_oracle.py. The
# ingest arrives in ts order (symbols interleaved) so the source is
# window-ordered from the start; only corrections disorder it, and only
# until the next compaction. Checks are ("check", full_cum) where
# full_cum says whether the whole-table cumulative read must answer
# (True), or must refuse as loudly as the base would (False).
SCRIPT = [
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES "
            + ", ".join(f"({t}, 'AAPL', {t}.5, 0.0)" if t % 2 == 0
                        else f"({t}, 'MSFT', {t}.25, 1.0)"
                        for t in range(0, 40))),
    ("check", True),           # stale: stamp 0, whole answer is the live fold
    ("refresh", None),
    ("check", True),           # fresh
    ("sql", "UPDATE trades SET x = 999.0 WHERE ts = 7"),
    ("check", False),          # dirty bucket, unrefreshed, disordered
    ("sql", "DELETE FROM trades WHERE ts >= 35"),
    ("check", False),
    ("refresh", None),
    ("check", False),
    ("sql", "UPDATE trades SET ts = 12 WHERE ts = 3"),   # cross-bucket move
    ("check", False),
    ("compact", None),         # the pending kill becomes history; order heals
    ("sql", "DELETE FROM trades WHERE ts = 20"),
    ("compact", None),         # a kill compacted BEFORE any refresh
    ("check", True),
    ("refresh", None),
    ("check", True),
    ("reopen", None),          # the storage round trip, mid-life
    ("check", True),
    ("sql", "INSERT INTO trades (ts, sym, x, y) VALUES (40, 'AAPL', 40.5, 0.0), "
            "(41, 'MSFT', 41.25, 1.0)"),
    ("check", True),           # stale again after reopen
    ("refresh", None),
    ("check", True),
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
    lib.tallydb_view_running_definition.restype = ctypes.c_char_p
    lib.tallydb_view_cumulative_definition.restype = ctypes.c_char_p
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


def engine_rows(lib, context, sql: str):
    """The engine's answer to `sql`, as sorted row tuples — or, on
    refusal, the engine's own stderr line (a str), so a caller
    expecting a refusal can check it is the RIGHT refusal and not an
    unrelated failure wearing its clothes."""
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    # The engine reports refusals on the C library's stderr; capture
    # fd 2 around the call so the reason is inspectable.
    captured = tempfile.TemporaryFile()
    saved = os.dup(2)
    sys.stderr.flush()
    os.dup2(captured.fileno(), 2)
    try:
        code = lib.tallydb_view_query_stream(context, sql.encode(), ctypes.c_void_p(ptr))
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    captured.seek(0)
    message = captured.read().decode(errors="replace")
    captured.close()
    if code != 0:
        return message
    sys.stderr.write(message)  # pass through anything non-fatal
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
    if isinstance(engine, str):
        sys.exit(f"{name} refused at check {checks} where it must answer: {engine}")
    if not engine:
        sys.exit(f"vacuous check: {name} answered no rows at check {checks}")
    if not rows_equal(engine, expected):
        sys.exit(
            f"{name} diverged from DuckDB recompute at check {checks}:\n"
            f"  engine: {engine}\n  duckdb: {expected}"
        )


def main() -> None:
    lib = load_library()
    definition = lib.tallydb_view_definition().decode()
    running = lib.tallydb_view_running_definition().decode()
    cumulative = lib.tallydb_view_cumulative_definition().decode()
    # The engine-side definitions name the shapes this script mirrors;
    # a drift fails here, loudly, before any diff runs.
    for fragment in ("ts / 5", "first(x)", "last(x)", "GROUP BY sym"):
        if fragment not in definition:
            sys.exit(f"definition drift: {fragment!r} not in {definition!r}")
    for fragment in ("GROUP BY sym", "avg(x)"):
        if fragment not in running:
            sys.exit(f"running drift: {fragment!r} not in {running!r}")
    for fragment in ("PARTITION BY sym", "UNBOUNDED PRECEDING"):
        if fragment not in cumulative:
            sys.exit(f"cumulative drift: {fragment!r} not in {cumulative!r}")
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
                diff("bars", checks,
                     engine_rows(lib, context, f"SELECT {BARS_COLUMNS} FROM bars"),
                     duckdb_rows(oracle, BARS_MIRROR))
                diff("totals", checks,
                     engine_rows(lib, context, f"SELECT {TOTALS_COLUMNS} FROM totals"),
                     duckdb_rows(oracle, TOTALS_MIRROR))
                diff("cum ranged", checks,
                     engine_rows(lib, context,
                                 f"SELECT {CUM_COLUMNS} FROM cum WHERE ts >= {CUM_FLOOR}"),
                     duckdb_rows(oracle,
                                 f"SELECT * FROM ({cumulative}) WHERE ts >= {CUM_FLOOR}"))
                full = engine_rows(lib, context, f"SELECT {CUM_COLUMNS} FROM cum")
                if payload:
                    diff("cum full", checks, full, duckdb_rows(oracle, cumulative))
                elif not isinstance(full, str):
                    sys.exit(
                        f"cum full ANSWERED at check {checks} over a "
                        "correction-disordered source — the base's windows "
                        "refuse there, and the view must refuse with them"
                    )
                elif "not sorted" not in full:
                    sys.exit(
                        f"cum full refused at check {checks} for the WRONG "
                        f"reason — expected the executor's ordering refusal, "
                        f"got: {full}"
                    )
                checks += 1
                print(f"PASS check {checks}"
                      + ("" if payload else " (cum full refused, as the base does)"))
    finally:
        lib.tallydb_view_close(context)
    print(
        f"Maintained-view families validated end-to-end: {checks} checks "
        f"x {{bucketed, running, cumulative-ranged, cumulative-full}} across "
        f"stale/fresh/corrected/compacted/reopened states "
        f"(duckdb {duckdb.__version__})"
    )


if __name__ == "__main__":
    main()
