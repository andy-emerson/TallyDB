#!/usr/bin/env python3
"""M1 end-to-end oracle: the engine's rolling regression vs NumPy (and DuckDB).

Drives the whole vertical slice through the `oracle-harness` hooks in
libengine: the engine ingests its fixture row by row, runs the SQL rolling
regression through compute-lapack, and exports both the raw inputs and the
computed coefficients over the Arrow C stream interface. This script then
recomputes every window independently:

  - np.linalg.lstsq (SVD-based) over each trailing per-symbol window —
    the compute-seam oracle;
  - DuckDB's regr_slope/regr_intercept window aggregates over the same
    frame, when duckdb is importable — the SQL-semantics oracle.

Usage: m1_slice_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow.cffi import ffi

ABS_TOL = 1e-9
REL_TOL = 1e-9


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


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def numpy_rolling(inputs: pa.Table, preceding: int):
    """Recompute every window with np.linalg.lstsq; returns per-row
    (slope, intercept) with None where the regression is undefined."""
    ts = inputs["ts"].to_pylist()
    sym = inputs["sym"].to_pylist()
    x = inputs["x"].to_pylist()
    y = inputs["y"].to_pylist()
    n = len(ts)
    per_sym_rows: dict[str, list[int]] = {}
    for row in range(n):
        per_sym_rows.setdefault(sym[row], []).append(row)
    slopes: list[float | None] = [None] * n
    intercepts: list[float | None] = [None] * n
    for rows in per_sym_rows.values():
        for position, row in enumerate(rows):
            window = rows[max(0, position - preceding) : position + 1]
            if len(window) < 2:
                continue  # undefined -> stays None
            wx = np.array([x[r] for r in window])
            wy = np.array([y[r] for r in window])
            if np.ptp(wx) == 0.0:
                continue  # rank-deficient -> stays None
            design = np.column_stack([np.ones(len(window)), wx])
            coef, *_ = np.linalg.lstsq(design, wy, rcond=None)
            intercepts[row] = float(coef[0])
            slopes[row] = float(coef[1])
    return slopes, intercepts


def compare(name, engine_values, oracle_values):
    for row, (engine_value, oracle_value) in enumerate(
        zip(engine_values, oracle_values)
    ):
        if (engine_value is None) != (oracle_value is None):
            sys.exit(
                f"FAIL {name}: row {row} nullness differs "
                f"(engine {engine_value!r}, oracle {oracle_value!r})"
            )
        if engine_value is not None and not close(engine_value, oracle_value):
            sys.exit(
                f"FAIL {name}: row {row} engine {engine_value!r} "
                f"vs oracle {oracle_value!r}"
            )
    print(f"PASS {name} ({len(engine_values)} rows)")


def duckdb_check(inputs: pa.Table, regression: pa.Table, preceding: int) -> None:
    try:
        import duckdb
    except ImportError:
        print("SKIP duckdb oracle (module not installed)")
        return
    connection = duckdb.connect()
    connection.register("trades", inputs)
    result = connection.execute(
        f"""
        SELECT regr_slope(y, x) OVER w AS slope,
               regr_intercept(y, x) OVER w AS intercept
        FROM trades
        WINDOW w AS (PARTITION BY sym ORDER BY ts
                     ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW)
        ORDER BY ts
        """
    ).to_arrow_table()

    # DuckDB reports an undefined regression (one-point window, zero
    # variance) as NaN, where the engine and NumPy use NULL — normalize
    # to compare the semantics, not the encoding of "undefined".
    def nan_to_none(values):
        return [None if v is not None and math.isnan(v) else v for v in values]

    compare(
        "engine vs duckdb slope",
        regression["slope"].to_pylist(),
        nan_to_none(result["slope"].to_pylist()),
    )
    compare(
        "engine vs duckdb intercept",
        regression["intercept"].to_pylist(),
        nan_to_none(result["intercept"].to_pylist()),
    )
    print(f"PASS duckdb oracle (duckdb {duckdb.__version__})")


def main() -> None:
    lib = load_library()
    lib.tallydb_m1_window_preceding.restype = ctypes.c_uint64
    preceding = int(lib.tallydb_m1_window_preceding())

    inputs = read_stream(lib, "tallydb_m1_inputs_stream")
    regression = read_stream(lib, "tallydb_m1_regression_stream")
    assert inputs.num_rows == regression.num_rows

    slopes, intercepts = numpy_rolling(inputs, preceding)
    compare("engine vs numpy slope", regression["slope"].to_pylist(), slopes)
    compare(
        "engine vs numpy intercept", regression["intercept"].to_pylist(), intercepts
    )
    duckdb_check(inputs, regression, preceding)
    print(
        f"M1 slice validated end-to-end (numpy {np.__version__}, "
        f"pyarrow {pa.__version__}, window {preceding + 1} rows, "
        f"{inputs.num_rows} rows)"
    )


if __name__ == "__main__":
    main()
