#!/usr/bin/env python3
"""Lua-window oracle (M2.7): scripted kernels re-derived by NumPy.

The engine registers four Lua kernels and runs them as partitioned SQL
window functions over the M1 fixture — which is persistent, reopened
from disk, and multi-segment, so every value here has survived the full
storage round trip and the cross-segment window gather. This script
recomputes every window independently in NumPy and diffs:

  lua_mad     pure-Lua mean absolute deviation (interpreter arithmetic)
  lua_wdot    a kernel calling the native `dot` host function over the
              same zero-copy views (the curated-op spread, end to end)
  lua_npos    an I64-declared count — the declared-return-type path:
              the exported Arrow column must BE int64 (F2/B5)
  lua_spread  max - min, NULL under three rows — the kernel-NULL path
              (the sentinel crossing back out as SQL NULL)

Two composed COLUMN kernels (option A) ride the same stream, checked
unpartitioned over the whole exported column (storage order = ts order
in this fixture):

  lua_rel     return (a - b) / b — vectorized operators; must equal the
              NumPy expression bit-for-bit (same IEEE ops, same order)
  lua_rdot    return rolling_dot(a, b, 10) — the native rolling
              combinator, continuous across the fixture's 64-row
              segment boundaries

Usage: m2_lua_window_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np
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


def reference_windows(x, y, window):
    """Every kernel, recomputed per trailing window, one partition."""
    n = len(x)
    mad = np.empty(n)
    wdot = np.empty(n)
    npos = np.empty(n, dtype=np.int64)
    spread = [None] * n
    for i in range(n):
        wx = x[max(0, i - window + 1) : i + 1]
        wy = y[max(0, i - window + 1) : i + 1]
        mad[i] = np.abs(wx - wx.mean()).mean()
        wdot[i] = float(np.dot(wy, wx))
        npos[i] = int((wx > 5.0).sum())
        if len(wx) >= 3:
            spread[i] = float(wx.max() - wx.min())
    return mad, wdot, npos, spread


def main():
    lib = load_library()
    window = int(lib.tallydb_lua_window_preceding()) + 1
    table = read_stream_hook(lib, "tallydb_lua_window_stream")

    # The declared-type path, end to end: lua_npos declared I64 must
    # export as an Arrow int64 column, never a float that "happens to be
    # integral".
    npos_type = table.schema.field("npos").type
    if npos_type != pa.int64():
        sys.exit(f"npos exported as {npos_type}, expected int64 (F2/B5)")

    symbols = table.column("sym").to_pylist()
    ts = table.column("ts").to_numpy(zero_copy_only=False)
    x = table.column("x").to_numpy(zero_copy_only=False)
    y = table.column("y").to_numpy(zero_copy_only=False)
    got_mad = table.column("mad").to_numpy(zero_copy_only=False)
    got_wdot = table.column("wdot").to_numpy(zero_copy_only=False)
    got_npos = table.column("npos").to_pylist()
    got_spread = table.column("spread").to_pylist()

    # The composed column kernels: whole-column, unpartitioned.
    got_rel = table.column("rel").to_numpy(zero_copy_only=False)
    got_rdot = table.column("rdot").to_numpy(zero_copy_only=False)
    if not np.array_equal(got_rel, (x - y) / y):
        worst = np.nanmax(np.abs(got_rel - (x - y) / y))
        sys.exit(f"lua_rel differs from (x - y) / y (worst {worst})")
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        expected = float(np.dot(x[lo : i + 1], y[lo : i + 1]))
        if not np.isclose(got_rdot[i], expected, rtol=1e-12, atol=1e-9):
            sys.exit(f"lua_rdot row {i}: {got_rdot[i]} != {expected}")

    checked = 0
    for symbol in sorted(set(symbols)):
        rows = [i for i, s in enumerate(symbols) if s == symbol]
        # Window frames follow ORDER BY ts within the partition.
        rows.sort(key=lambda i: ts[i])
        ref_mad, ref_wdot, ref_npos, ref_spread = reference_windows(
            x[rows], y[rows], window
        )
        for position, row in enumerate(rows):
            if not np.isclose(got_mad[row], ref_mad[position], rtol=1e-12, atol=1e-12):
                sys.exit(
                    f"{symbol} row {position}: mad {got_mad[row]} "
                    f"!= {ref_mad[position]}"
                )
            if not np.isclose(
                got_wdot[row], ref_wdot[position], rtol=1e-12, atol=1e-12
            ):
                sys.exit(
                    f"{symbol} row {position}: wdot {got_wdot[row]} "
                    f"!= {ref_wdot[position]}"
                )
            if got_npos[row] != ref_npos[position]:
                sys.exit(
                    f"{symbol} row {position}: npos {got_npos[row]} "
                    f"!= {ref_npos[position]}"
                )
            if got_spread[row] is None or ref_spread[position] is None:
                if got_spread[row] != ref_spread[position]:
                    sys.exit(
                        f"{symbol} row {position}: spread null mismatch "
                        f"({got_spread[row]} vs {ref_spread[position]})"
                    )
            elif got_spread[row] != ref_spread[position]:
                sys.exit(
                    f"{symbol} row {position}: spread {got_spread[row]} "
                    f"!= {ref_spread[position]}"
                )
            checked += 1
    print(f"PASS lua windows vs numpy ({checked} windows x 4 kernels)")
    print(f"PASS composed column kernels vs numpy ({len(x)} rows x 2 kernels)")

    # The driver pipeline (SQL-in-Lua, #70): a script SELECTed the same
    # fixture, computed rel/rdot over the result views, and fed every
    # row back through append into a scratch table; the export here is
    # that derived table. The inputs must round-trip bit-for-bit
    # (SELECT -> view -> Lua number -> append -> storage -> export),
    # the key texts must survive the per-segment dictionary merge, and
    # the derived columns must re-derive in NumPy exactly like the
    # kernel-slot versions above.
    derived = read_stream_hook(lib, "tallydb_driver_pipeline_stream")
    d_ts = derived.column("ts").to_numpy(zero_copy_only=False)
    if not np.array_equal(d_ts, np.sort(ts)):
        sys.exit("driver pipeline: ts set changed through the round trip")
    order = np.argsort(ts, kind="stable")
    if derived.column("sym").to_pylist() != [symbols[i] for i in order]:
        sys.exit("driver pipeline: key texts changed through the merge/feed-back")
    d_x = derived.column("x").to_numpy(zero_copy_only=False)
    d_y = derived.column("y").to_numpy(zero_copy_only=False)
    if not (np.array_equal(d_x, x[order]) and np.array_equal(d_y, y[order])):
        sys.exit("driver pipeline: numeric inputs changed through the round trip")
    d_rel = derived.column("rel").to_numpy(zero_copy_only=False)
    if not np.array_equal(d_rel, (d_x - d_y) / d_y):
        sys.exit("driver pipeline: rel differs from (x - y) / y")
    d_rdot = derived.column("rdot").to_numpy(zero_copy_only=False)
    for i in range(len(d_x)):
        lo = max(0, i - window + 1)
        expected = float(np.dot(d_x[lo : i + 1], d_y[lo : i + 1]))
        if not np.isclose(d_rdot[i], expected, rtol=1e-12, atol=1e-9):
            sys.exit(f"driver pipeline: rdot row {i}: {d_rdot[i]} != {expected}")
    print(f"PASS driver pipeline (SQL -> Lua -> SQL) vs numpy ({len(d_x)} rows)")
    print(
        f"Lua-window family validated end-to-end over the storage round trip "
        f"(numpy {np.__version__}, pyarrow {pa.__version__}, "
        f"window {window} rows, {len(symbols)} rows)"
    )


if __name__ == "__main__":
    main()
