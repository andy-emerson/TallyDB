#!/usr/bin/env python3
"""M2.3 mutation oracle: TallyDB's UPDATE/DELETE end state vs DuckDB.

Drives the `oracle-harness` hooks in libengine: the engine ingests its
deterministic fixture (through a persist-and-reopen cycle), applies a
scripted UPDATE/DELETE sequence through the real mutation path
(tombstone + reinsert), compacts, and exports the end state. This script
replays the same statements in DuckDB against the same inputs and diffs
the surviving rows — the differential check of mutation *semantics*,
not implementation.

Usage: m2_mutation_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
from pyarrow.cffi import ffi

# KEEP IN SYNC with MUTATIONS in src/harness.rs — a mismatch fails this
# oracle loudly, it cannot pass silently.
MUTATIONS = [
    "DELETE FROM trades WHERE sym = 'TSLA'",
    "DELETE FROM trades WHERE ts >= 220",
    "UPDATE trades SET y = 0 WHERE x < 2 AND sym IN ('AAPL', 'MSFT')",
    "UPDATE trades SET x = 5.5 WHERE ts < 30 AND sym <> 'MSFT'",
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
    return ctypes.CDLL(str(path))


def read_stream(lib, symbol: str) -> pa.Table:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    getattr(lib, symbol)(ctypes.c_void_p(ptr))
    return pa.RecordBatchReader._import_from_c(ptr).read_all()


def close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    lib = load_library()
    inputs = read_stream(lib, "tallydb_m1_inputs_stream")
    engine_state = read_stream(lib, "tallydb_m2_mutated_stream")

    connection = duckdb.connect()
    connection.register("trades_input", inputs)
    connection.execute("CREATE TABLE trades AS SELECT * FROM trades_input")
    for statement in MUTATIONS:
        connection.execute(statement)
    oracle_state = connection.execute(
        "SELECT ts, sym, x, y FROM trades ORDER BY ts"
    ).to_arrow_table()

    if engine_state.num_rows != oracle_state.num_rows:
        sys.exit(
            f"FAIL row count: engine {engine_state.num_rows} "
            f"vs duckdb {oracle_state.num_rows}"
        )
    # The engine's compacted table is sorted by the ordering key, and the
    # fixture's ts values are unique — both sides share a total order.
    for column in ["ts", "sym", "x", "y"]:
        engine_values = engine_state[column].to_pylist()
        oracle_values = oracle_state[column].to_pylist()
        for row, (engine_value, oracle_value) in enumerate(
            zip(engine_values, oracle_values)
        ):
            equal = (
                engine_value == oracle_value
                if column in ("ts", "sym")
                else close(engine_value, oracle_value)
            )
            if not equal:
                sys.exit(
                    f"FAIL {column}: row {row} engine {engine_value!r} "
                    f"vs duckdb {oracle_value!r}"
                )
        print(f"PASS {column} ({len(engine_values)} rows)")
    print(
        f"M2.3 mutation semantics validated against DuckDB "
        f"{duckdb.__version__} ({len(MUTATIONS)} statements, "
        f"{inputs.num_rows} -> {engine_state.num_rows} rows)"
    )


if __name__ == "__main__":
    main()
