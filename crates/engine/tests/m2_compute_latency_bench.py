#!/usr/bin/env python3
"""M2.7 latency benchmark: in-engine compute vs a DuckDB+NumPy round trip.

Measurement, not a check — run by hand, never in CI; a number cites its
run. The point is to earn (or honestly bound) the compute-without-
copying latency claim: how does running a window statistic *inside* the
engine compare against the architecture TallyDB exists to avoid — keep
the data in the store, export it over Arrow, compute outside (DuckDB
for SQL statistics, vectorized NumPy for custom kernels), and land the
result back in application memory?

Three configurations, precisely:
  in-engine   one SQL query against a prebuilt table (fixture built
              once, outside all timing): window function runs inside
              TallyDB (native curated op, or a registered Lua kernel),
              result exported over ArrowArrayStream and materialized in
              Python.
  tdb peer    per iteration, the round trip an external-compute design
              pays on data it stores in TallyDB: export the raw columns
              over ArrowArrayStream, materialize, compute the same
              statistic outside (DuckDB's window executor, or
              vectorized NumPy). This isolates the marginal value of
              in-engine compute — the peer rides TallyDB's own export.
  ddb peer    per iteration, the competitor stack entire: the same rows
              live in a DuckDB table (in-memory, loaded once outside
              timing, mirroring the in-engine fixture), and NumPy pulls
              from DuckDB (`ORDER BY` the ordering key, fetchnumpy) and
              computes; for regr_slope, DuckDB's own window executor
              runs directly on its own table. This is "DuckDB + NumPy"
              as a user would actually deploy it.

Both paths end with the answer in application memory; both run on the
same data, same process, same hardware, back to back. The two paths'
results are cross-checked (full windows, tolerance-based) — a benchmark
that computes wrong answers measures nothing.

The spread runs cheap -> heavy, plus the interpreter-only case:
  lua_dot     Lua kernel calling the native dot host function (cheap op,
              script dispatch)         peer: NumPy rolling dot (cumsum)
  lua_mad     pure-Lua kernel (no native op — the promotion ladder's
              first rung)              peer: vectorized NumPy MAD
  regr_slope  closed-form least squares peer: DuckDB's regr_slope window
              per window                     (incremental running moments)
  eigen_max   native closed-form 2x2   peer: NumPy closed-form 2x2 from
              eigenvalue per window         rolling moments (same math,
                                            vectorized over the column)
  covar_pop   window moments, two-pass peer: NumPy rolling moments
  corr        window moments, two-pass peer: NumPy rolling moments

The last two are the pure recompute-vs-vectorized comparison: identical
arithmetic, so the gap is entirely O(n·w) per-window recompute against
the peer's O(n) sweep.

Usage: m2_compute_latency_bench.py [libengine.so] [rows] [iters]
Defaults: target/debug/libengine.so, 20000 rows, 3 iterations (min
taken). Exits nonzero only if the paths disagree.
"""

import ctypes
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
from numpy.lib.stride_tricks import sliding_window_view
from pyarrow.cffi import ffi

WINDOW = 64  # 63 PRECEDING + CURRENT ROW
FRAME = "OVER (ORDER BY ts ROWS BETWEEN 63 PRECEDING AND CURRENT ROW)"


def load_library(path_arg):
    if path_arg:
        path = Path(path_arg)
    else:
        repo = Path(__file__).resolve().parents[3]
        path = repo / "target" / "debug" / "libengine.so"
    if not path.exists():
        sys.exit(
            f"{path} not found - build it with "
            "`cargo build -p engine --features oracle-harness` "
            "(release recommended for real numbers)"
        )
    lib = ctypes.CDLL(str(path))
    lib.tallydb_bench_open.argtypes = [ctypes.c_uint64]
    lib.tallydb_bench_open.restype = ctypes.c_void_p
    lib.tallydb_bench_query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.tallydb_bench_query.restype = ctypes.c_int
    lib.tallydb_bench_close.argtypes = [ctypes.c_void_p]
    return lib


def bench_query(lib, context, sql: str) -> pa.Table:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    status = lib.tallydb_bench_query(
        context, ctypes.c_char_p(sql.encode()), ctypes.c_void_p(ptr)
    )
    if status != 0:
        sys.exit(f"bench query failed: {sql}")
    return pa.RecordBatchReader._import_from_c(ptr).read_all()


def timed(callable_, iters):
    """Minimum wall time over `iters` runs, plus the last result."""
    best = float("inf")
    result = None
    for _ in range(iters):
        start = time.perf_counter()
        result = callable_()
        best = min(best, time.perf_counter() - start)
    return best, result


def column(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window sums for every row (head windows are shorter)."""
    sums = np.concatenate(([0.0], np.cumsum(values)))
    start = np.maximum(0, np.arange(1, len(values) + 1) - window)
    return sums[1:] - sums[start]


def peer_dot(x, y):
    return rolling_sum(x * y, WINDOW)


def peer_mad(x):
    result = np.empty(len(x))
    # Head windows (shorter than WINDOW) looped; the body vectorized.
    for i in range(min(WINDOW - 1, len(x))):
        window = x[: i + 1]
        result[i] = np.abs(window - window.mean()).mean()
    if len(x) >= WINDOW:
        windows = sliding_window_view(x, WINDOW)
        means = windows.mean(axis=1)
        result[WINDOW - 1 :] = np.abs(windows - means[:, None]).mean(axis=1)
    return result


def rolling_moments(x, y):
    """Rolling (var_x, var_y, cov_xy) by the cumsum trick — the peers'
    shared O(n) core."""
    counts = np.minimum(np.arange(1, len(x) + 1), WINDOW).astype(float)
    mean_x = rolling_sum(x, WINDOW) / counts
    mean_y = rolling_sum(y, WINDOW) / counts
    var_x = rolling_sum(x * x, WINDOW) / counts - mean_x * mean_x
    var_y = rolling_sum(y * y, WINDOW) / counts - mean_y * mean_y
    cov_xy = rolling_sum(x * y, WINDOW) / counts - mean_x * mean_y
    return var_x, var_y, cov_xy


def peer_eigen(x, y):
    var_x, var_y, cov_xy = rolling_moments(x, y)
    half_trace = (var_x + var_y) / 2.0
    radius = np.sqrt(((var_y - var_x) / 2.0) ** 2 + cov_xy**2)
    return half_trace + radius


def peer_covar(x, y):
    return rolling_moments(x, y)[2]


def peer_corr(x, y):
    var_x, var_y, cov_xy = rolling_moments(x, y)
    # Leading windows can be degenerate (a one-row window has zero
    # variance); those rows are outside the compared range.
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov_xy / np.sqrt(var_y * var_x)


def check(name, engine_values, peer_values, rtol, atol):
    """Full windows must agree between the paths, or nothing here means
    anything."""
    body = slice(WINDOW, None)
    if not np.allclose(engine_values[body], peer_values[body], rtol=rtol, atol=atol):
        worst = np.nanmax(np.abs(engine_values[body] - peer_values[body]))
        sys.exit(f"{name}: paths disagree (worst absolute difference {worst})")


def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    lib = load_library(path_arg)
    context = lib.tallydb_bench_open(rows)

    # The bare round trip, for context: export + materialize, no compute.
    export_time, raw = timed(
        lambda: bench_query(lib, context, "SELECT ts, x, y FROM bench"), iters
    )
    x = column(raw, "x")
    y = column(raw, "y")
    assert len(x) == rows

    connection = duckdb.connect()

    def engine(sql, name):
        elapsed, table = timed(lambda: bench_query(lib, context, sql), iters)
        return elapsed, column(table, name)

    # The competitor's store, loaded once outside timing, like the
    # in-engine fixture.
    connection.register("raw_export", raw)
    connection.execute("CREATE TABLE ddb_bench AS SELECT * FROM raw_export")

    def peer(compute):
        def run():
            exported = bench_query(lib, context, "SELECT ts, x, y FROM bench")
            px = column(exported, "x")
            py = column(exported, "y")
            return compute(px, py)

        return timed(run, iters)

    def ddb_peer(compute):
        def run():
            fetched = connection.execute(
                "SELECT x, y FROM ddb_bench ORDER BY ts"
            ).fetchnumpy()
            return compute(fetched["x"], fetched["y"])

        return timed(run, iters)

    def duckdb_peer():
        def run():
            exported = bench_query(lib, context, "SELECT ts, x, y FROM bench")
            connection.register("t", exported)
            result = connection.execute(
                f"SELECT regr_slope(y, x) {FRAME} AS r FROM t ORDER BY ts"
            ).arrow()
            if hasattr(result, "read_all"):  # newer duckdb returns a reader
                result = result.read_all()
            return result.column("r").to_numpy(zero_copy_only=False)

        return timed(run, iters)

    def duckdb_native_peer():
        def run():
            result = connection.execute(
                f"SELECT regr_slope(y, x) {FRAME} AS r FROM ddb_bench ORDER BY ts"
            ).arrow()
            if hasattr(result, "read_all"):
                result = result.read_all()
            return result.column("r").to_numpy(zero_copy_only=False)

        return timed(run, iters)

    results = []

    def family(name, sql, compute, rtol, atol, label, peer_label):
        elapsed, values = engine(sql, "r")
        peer_time, peer_values = peer(compute)
        check(name, values, peer_values, rtol, atol)
        ddb_time, ddb_values = ddb_peer(compute)
        check(f"{name} (ddb)", values, ddb_values, rtol, atol)
        results.append((label, peer_label, elapsed, peer_time, ddb_time))

    family(
        "lua_dot",
        f"SELECT lua_dot(y, x) {FRAME} AS r FROM bench",
        lambda px, py: peer_dot(px, py),
        1e-9,
        1e-9,
        "lua_dot (Lua->native)",
        "NumPy rolling dot",
    )
    family(
        "lua_mad",
        f"SELECT lua_mad(x) {FRAME} AS r FROM bench",
        lambda px, py: peer_mad(px),
        1e-9,
        1e-12,
        "lua_mad (pure Lua)",
        "NumPy vectorized MAD",
    )
    # The column-kernel shape (#53): per-row math, but the interpreter
    # is entered once per column view, never per row — the amortization
    # the front-end architecture promises. The native scalar expression
    # below it is the same arithmetic in the engine's own projection
    # slot: the floor a promoted kernel reaches.
    family(
        "lua_spread",
        "SELECT lua_spread(x, y) AS r FROM bench",
        lambda px, py: (px - py) / py,
        1e-12,
        1e-12,
        "lua_spread (column)",
        "NumPy (x - y) / y",
    )
    family(
        "native_expr",
        "SELECT (x - y) / y AS r FROM bench",
        lambda px, py: (px - py) / py,
        1e-12,
        1e-12,
        "(x - y) / y (native)",
        "NumPy (x - y) / y",
    )

    elapsed, values = engine(f"SELECT regr_slope(y, x) {FRAME} AS r FROM bench", "r")
    peer_time, peer_values = duckdb_peer()
    check("regr_slope", values, peer_values, 1e-6, 1e-9)
    ddb_time, ddb_values = duckdb_native_peer()
    check("regr_slope (ddb)", values, ddb_values, 1e-6, 1e-9)
    results.append(("regr_slope (closed form)", "DuckDB window", elapsed, peer_time, ddb_time))

    family(
        "eigen_max",
        f"SELECT eigen_max(y, x) {FRAME} AS r FROM bench",
        lambda px, py: peer_eigen(px, py),
        1e-6,
        1e-9,
        "eigen_max (closed form)",
        "NumPy closed-form",
    )
    family(
        "covar_pop",
        f"SELECT covar_pop(y, x) {FRAME} AS r FROM bench",
        lambda px, py: peer_covar(px, py),
        1e-6,
        1e-9,
        "covar_pop (moments)",
        "NumPy rolling moments",
    )
    family(
        "corr",
        f"SELECT corr(y, x) {FRAME} AS r FROM bench",
        lambda px, py: peer_corr(px, py),
        1e-6,
        1e-9,
        "corr (moments)",
        "NumPy rolling moments",
    )

    # ---- the latency shape: one window over the latest rows ----
    # The bulk sweep above amortizes the round trip over n windows; a
    # live query ("the current statistic, now") cannot. Both paths fetch
    # only the last WINDOW rows (TallyDB's zone maps prune either way)
    # and produce one number.
    low = rows - WINDOW
    latency_iters = max(iters * 10, 30)

    def engine_last(sql):
        def run():
            return column(bench_query(lib, context, sql), "r")[-1]

        return timed(run, latency_iters)

    def peer_last(compute):
        def run():
            exported = bench_query(
                lib, context, f"SELECT ts, x, y FROM bench WHERE ts >= {low}"
            )
            return compute(column(exported, "x"), column(exported, "y"))

        return timed(run, latency_iters)

    def ddb_peer_last(compute):
        def run():
            fetched = connection.execute(
                f"SELECT x, y FROM ddb_bench WHERE ts >= {low} ORDER BY ts"
            ).fetchnumpy()
            return compute(fetched["x"], fetched["y"])

        return timed(run, latency_iters)

    def closed_form_slope(px, py):
        dx = px - px.mean()
        return (dx * (py - py.mean())).sum() / (dx * dx).sum()

    latency = []
    elapsed, value = engine_last(
        f"SELECT lua_dot(y, x) {FRAME} AS r FROM bench WHERE ts >= {low}"
    )
    peer_time, peer_value = peer_last(lambda px, py: float(np.dot(px, py)))
    if not np.isclose(value, peer_value, rtol=1e-9):
        sys.exit(f"latency lua_dot: paths disagree ({value} vs {peer_value})")
    ddb_time, ddb_value = ddb_peer_last(lambda px, py: float(np.dot(px, py)))
    if not np.isclose(value, ddb_value, rtol=1e-9):
        sys.exit(f"latency lua_dot (ddb): paths disagree ({value} vs {ddb_value})")
    latency.append(("last-window lua_dot", "NumPy dot", elapsed, peer_time, ddb_time))

    elapsed, value = engine_last(
        f"SELECT regr_slope(y, x) {FRAME} AS r FROM bench WHERE ts >= {low}"
    )
    peer_time, peer_value = peer_last(closed_form_slope)
    if not np.isclose(value, peer_value, rtol=1e-6):
        sys.exit(f"latency regr: paths disagree ({value} vs {peer_value})")
    ddb_time, ddb_value = ddb_peer_last(closed_form_slope)
    if not np.isclose(value, ddb_value, rtol=1e-6):
        sys.exit(f"latency regr (ddb): paths disagree ({value} vs {ddb_value})")
    latency.append(("last-window regr", "NumPy closed form", elapsed, peer_time, ddb_time))

    lib.tallydb_bench_close(context)

    print(
        f"m2_compute_latency_bench: {rows} rows, window {WINDOW}, "
        f"min of {iters} runs; bare export (round trip, no compute) "
        f"{export_time * 1e3:.1f} ms"
    )
    print(
        f"{'op':<22}{'peer':<22}{'engine':>8}{'tdb peer':>9}{'ddb peer':>9}"
        f"{'tdb/eng':>9}{'ddb/eng':>9}"
    )
    for name, peer_name, engine_time, peer_time, ddb_time in results:
        print(
            f"{name:<22}{peer_name:<22}{engine_time * 1e3:>8.1f}"
            f"{peer_time * 1e3:>9.1f}{ddb_time * 1e3:>9.1f}"
            f"{peer_time / engine_time:>9.2f}{ddb_time / engine_time:>9.2f}"
        )
    print("(columns are milliseconds)")
    print(
        f"latency shape ({WINDOW} rows, one window, min of {latency_iters} runs):"
    )
    for name, peer_name, engine_time, peer_time, ddb_time in latency:
        print(
            f"{name:<22}{peer_name:<22}{engine_time * 1e6:>8.0f}"
            f"{peer_time * 1e6:>9.0f}{ddb_time * 1e6:>9.0f}"
            f"{peer_time / engine_time:>9.2f}{ddb_time / engine_time:>9.2f}"
        )
    print("(latency columns are microseconds)")
    print(
        "tdb peer: NumPy/DuckDB fed by TallyDB's own Arrow export (the "
        "marginal value of in-engine compute); ddb peer: the same rows "
        "stored in DuckDB, NumPy pulling from DuckDB (the competitor "
        "stack entire). Ratios > 1 favor in-engine compute. Report both "
        "directions honestly."
    )


if __name__ == "__main__":
    main()
