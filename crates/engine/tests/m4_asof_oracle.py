#!/usr/bin/env python3
"""M4.4 as-of oracle: TallyDB's `ASOF` answers vs DuckDB over an
explicit history table.

Drives the `oracle-harness` as-of hooks in libengine statement by
statement: the engine ingests its deterministic fixture into persistent
storage, then this script applies a scripted mutation sequence through
the real mutation path, reading the ingest-sequence watermark around
every statement. From those watermarks it maintains an explicit
version table in DuckDB — one row per version with `[birth, kill)`
interval columns — and for EVERY cut from 0 to the final watermark it
asks both sides "what was known at n": the engine through `ASOF n`
(knowledge masks over live segments, history segments, pending
tombstone stamps), DuckDB through a plain interval predicate over the
version table. Compactions are interleaved (moving superseded rows
into history segments) and the store is closed and reopened mid-run —
the answers must not move. The emulation lives in the referee only.

Interval model (must match the engine's knowledge axis):
- an appended row's birth is the watermark at its append (the fixture's
  row i has birth i);
- an UPDATE is ONE knowledge event (issue #73): every replacement is
  born at the pre-mutation watermark w0 and every victim killed at w0,
  and the mutation consumes exactly one sequence — no cut can see both
  versions;
- a DELETE kills its matches at the current watermark without consuming
  a sequence.

Usage: m4_asof_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
from pyarrow.cffi import ffi

# The scripted history: ("delete", predicate) | ("update", predicate,
# column, value) | ("compact",) | ("reopen",).
SCRIPT = [
    ("delete", "sym = 'TSLA'"),
    ("update", "ts = 40", "y", "0"),
    ("compact",),
    ("update", "ts < 30 AND sym = 'AAPL'", "x", "5.5"),
    ("delete", "ts >= 220"),
    ("compact",),
    ("reopen",),
    ("update", "ts = 100", "y", "99"),
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
    lib.tallydb_asof_open.restype = ctypes.c_void_p
    lib.tallydb_asof_close.argtypes = [ctypes.c_void_p]
    lib.tallydb_asof_next_sequence.restype = ctypes.c_uint64
    lib.tallydb_asof_next_sequence.argtypes = [ctypes.c_void_p]
    lib.tallydb_asof_mutate.restype = ctypes.c_int64
    lib.tallydb_asof_mutate.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tallydb_asof_compact.argtypes = [ctypes.c_void_p]
    lib.tallydb_asof_reopen.argtypes = [ctypes.c_void_p]
    lib.tallydb_asof_query.restype = ctypes.c_int32
    lib.tallydb_asof_query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    return lib


def query(lib, context, sql: str) -> pa.Table:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    if lib.tallydb_asof_query(context, sql.encode(), ctypes.c_void_p(ptr)) != 0:
        sys.exit(f"FAIL engine query errored: {sql}")
    return pa.RecordBatchReader._import_from_c(ptr).read_all()


def close(a, b) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def rows_of(table: pa.Table) -> list:
    """Rows as sorted tuples — a canonical multiset (torn cuts hold two
    versions of one ts, so no single column is a total order)."""
    columns = [table[name].to_pylist() for name in ("ts", "sym", "x", "y")]
    return sorted(zip(*columns))


def rows_equal(engine_rows: list, oracle_rows: list) -> bool:
    if len(engine_rows) != len(oracle_rows):
        return False
    for (ets, esym, ex, ey), (ots, osym, ox, oy) in zip(engine_rows, oracle_rows):
        if ets != ots or esym != osym or not close(ex, ox) or not close(ey, oy):
            return False
    return True


def oracle_state(connection, cut) -> list:
    predicate = "kill IS NULL" if cut is None else (
        f"birth <= {cut} AND (kill IS NULL OR kill > {cut})"
    )
    return sorted(
        connection.execute(
            f"SELECT ts, sym, x, y FROM versions WHERE {predicate}"
        ).fetchall()
    )


def check_cut(lib, context, connection, cut: int, cache: dict, phase: str) -> None:
    engine = rows_of(query(lib, context, f"SELECT ts, sym, x, y FROM trades ASOF {cut}"))
    if cut not in cache:
        cache[cut] = oracle_state(connection, cut)
    if not rows_equal(engine, cache[cut]):
        sys.exit(
            f"FAIL cut {cut} ({phase}): engine {len(engine)} rows vs "
            f"duckdb {len(cache[cut])} rows, or values differ"
        )
    # Every tenth cut, the mask composed with a predicate and with
    # grouping — the pruning and aggregate paths over the same views.
    if cut % 10 == 0:
        filtered = query(
            lib, context, f"SELECT ts, x FROM trades ASOF {cut} WHERE x > 5"
        )
        engine_filtered = sorted(
            zip(filtered["ts"].to_pylist(), filtered["x"].to_pylist())
        )
        oracle_filtered = sorted(
            (ts, x) for (ts, _, x, _) in cache[cut] if x > 5
        )
        if len(engine_filtered) != len(oracle_filtered) or any(
            ets != ots or not close(ex, ox)
            for (ets, ex), (ots, ox) in zip(engine_filtered, oracle_filtered)
        ):
            sys.exit(f"FAIL cut {cut} ({phase}): filtered scan differs")
        grouped = query(
            lib,
            context,
            f"SELECT sym, COUNT(x) AS n, SUM(x) AS s FROM trades ASOF {cut} GROUP BY sym",
        )
        engine_groups = {
            sym: (n, s)
            for sym, n, s in zip(
                grouped["sym"].to_pylist(),
                grouped["n"].to_pylist(),
                grouped["s"].to_pylist(),
            )
        }
        oracle_groups = {}
        for _, sym, x, _ in cache[cut]:
            n, s = oracle_groups.get(sym, (0, 0.0))
            oracle_groups[sym] = (n + 1, s + x)
        # Empty groups don't exist on either side; compare keys + values
        # (SUM under a different addition order gets a tolerance).
        if set(engine_groups) != set(oracle_groups) or any(
            engine_groups[sym][0] != oracle_groups[sym][0]
            or not math.isclose(
                engine_groups[sym][1], oracle_groups[sym][1], rel_tol=1e-9
            )
            for sym in oracle_groups
        ):
            sys.exit(f"FAIL cut {cut} ({phase}): grouped aggregate differs")


def main() -> None:
    lib = load_library()
    context = ctypes.c_void_p(lib.tallydb_asof_open())

    # Seed the version table from the fixture itself: row i was born at
    # sequence i (virtual until divergence — sequence == row id).
    initial = query(lib, context, "SELECT ts, sym, x, y FROM trades")
    connection = duckdb.connect()
    connection.register("fixture", initial)
    connection.execute(
        "CREATE TABLE versions AS "
        "SELECT ts, sym, x, y, ts AS birth, CAST(NULL AS BIGINT) AS kill "
        "FROM fixture"
    )

    for step in SCRIPT:
        w0 = lib.tallydb_asof_next_sequence(context)
        if step[0] == "compact":
            if lib.tallydb_asof_compact(context) != 0:
                sys.exit("FAIL compaction errored")
        elif step[0] == "reopen":
            if lib.tallydb_asof_reopen(context) != 0:
                sys.exit("FAIL reopen errored")
        elif step[0] == "delete":
            (_, predicate) = step
            changed = lib.tallydb_asof_mutate(
                context, f"DELETE FROM trades WHERE {predicate}".encode()
            )
            # Count before stamping: kill stamps may legitimately
            # collide across statements (they mark, not consume, the
            # watermark), so counting by stamp would over-count.
            oracle_changed = connection.execute(
                f"SELECT COUNT(*) FROM versions WHERE kill IS NULL AND ({predicate})"
            ).fetchone()[0]
            connection.execute(
                f"UPDATE versions SET kill = {w0} WHERE kill IS NULL AND ({predicate})"
            )
            if changed != oracle_changed:
                sys.exit(
                    f"FAIL delete '{predicate}': engine changed {changed}, "
                    f"oracle {oracle_changed}"
                )
            # A delete stamps the watermark without consuming it.
            if lib.tallydb_asof_next_sequence(context) != w0:
                sys.exit("FAIL delete moved the watermark")
        else:
            (_, predicate, column, value) = step
            matched = connection.execute(
                "SELECT ts, sym, x, y FROM versions "
                f"WHERE kill IS NULL AND ({predicate}) ORDER BY ts"
            ).fetchall()
            changed = lib.tallydb_asof_mutate(
                context,
                f"UPDATE trades SET {column} = {value} WHERE {predicate}".encode(),
            )
            if changed != len(matched):
                sys.exit(
                    f"FAIL update '{predicate}': engine changed {changed}, "
                    f"oracle matched {len(matched)}"
                )
            for ts, sym, x, y in matched:
                new = dict(ts=ts, sym=sym, x=x, y=y)
                new[column] = float(value)
                connection.execute(
                    "INSERT INTO versions VALUES (?, ?, ?, ?, ?, NULL)",
                    [new["ts"], new["sym"], new["x"], new["y"], w0],
                )
            connection.execute(
                f"UPDATE versions SET kill = {w0} "
                f"WHERE kill IS NULL AND ({predicate}) AND birth < {w0}"
            )
            if matched and lib.tallydb_asof_next_sequence(context) != w0 + 1:
                sys.exit("FAIL update did not consume exactly one sequence")
        # After every step, the engine's latest state must equal the
        # open versions — the same invariant the M2.3 oracle proves,
        # here holding at every point of the history.
        latest = rows_of(query(lib, context, "SELECT ts, sym, x, y FROM trades"))
        if not rows_equal(latest, oracle_state(connection, None)):
            sys.exit(f"FAIL latest state diverged after {step}")

    final_watermark = lib.tallydb_asof_next_sequence(context)
    cache: dict = {}
    for cut in range(final_watermark + 1):
        check_cut(lib, context, connection, cut, cache, "before reopen")
    # The full round trip: everything above must come back from bytes.
    if lib.tallydb_asof_reopen(context) != 0:
        sys.exit("FAIL final reopen errored")
    for cut in range(final_watermark + 1):
        check_cut(lib, context, connection, cut, cache, "after reopen")

    lib.tallydb_asof_close(context)
    print(
        f"M4.4 as-of semantics validated against DuckDB {duckdb.__version__} "
        f"over an explicit history table ({len(SCRIPT)} scripted steps, "
        f"{final_watermark + 1} cuts swept twice across a reopen)"
    )


if __name__ == "__main__":
    main()
