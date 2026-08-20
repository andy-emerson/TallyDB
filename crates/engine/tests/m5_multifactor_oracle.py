#!/usr/bin/env python3
"""#90 multi-factor oracle: the engine's rolling K-factor fit vs NumPy.

Drives the `oracle-harness` multi-factor hooks in libengine: the engine
ingests a fixture whose design matrix is wide enough for a K > 2 fit,
runs the rolling regression through the incremental sweep the executor
picks for `ROWS` frames, and exports both the raw inputs and the fitted
coefficients over the Arrow C stream interface. This script then rebuilds
every trailing window's design matrix and solves it independently with
`np.linalg.lstsq`.

The independence is the point, and it is structural rather than
incidental. `lstsq` is QR/SVD-class: it never forms a Gram matrix and
never solves a normal equation, which is exactly the route the engine
takes. A diff must never share the implementation's computational path
(the #45 lesson — an oracle solving the same ill-conditioned matrix
agreed with the wrong answer).

Two consequences of that independence are handled explicitly rather than
papered over:

  - Where the window is rank-deficient, the engine REFUSES (NULL, the
    ruled semantic) while `lstsq` happily returns a minimum-norm
    solution. Those windows are not diffed; instead the script asserts
    the engine refused exactly where the design is singular, and that it
    refused nowhere else. The fixture holds a stretch where one factor
    goes flat precisely so this path is exercised, not assumed.
  - A normal-equations solve carries a kappa^2 forward error where QR
    carries kappa, so the tolerance is conditioned on the window rather
    than fixed: well-conditioned windows are held tight, and the few
    that are merely usable get room proportional to their conditioning.

Usage: m5_multifactor_oracle.py [path/to/libengine.so]
Exits nonzero on the first disagreement.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow.cffi import ffi

# Windows the engine must refuse: the pivot floor draws the line near
# kappa(X) ~ 1e6, where a normal-equations solve has under six correct
# digits (see SPD_PIVOT_FLOOR in multifactor.rs).
REFUSAL_CONDITION = 1e6
# Held-tight tolerance for well-conditioned windows.
BASE_TOL = 1e-9


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
    lib.tallydb_multifactor_preceding.restype = ctypes.c_uint64
    return lib


def read_stream(lib, symbol: str) -> pa.Table:
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr = int(ffi.cast("uintptr_t", c_stream))
    getattr(lib, symbol)(ctypes.c_void_p(ptr))
    return pa.RecordBatchReader._import_from_c(ptr).read_all()


def main() -> None:
    lib = load_library()
    preceding = int(lib.tallydb_multifactor_preceding())
    inputs = read_stream(lib, "tallydb_multifactor_inputs_stream")
    fits = read_stream(lib, "tallydb_multifactor_fits_stream")

    y = np.array(inputs["y"].to_pylist(), dtype=float)
    factors = np.column_stack(
        [np.array(inputs[name].to_pylist(), dtype=float) for name in ("a", "b", "c")]
    )
    rows = len(y)
    if rows == 0 or fits.num_rows != rows:
        sys.exit(f"shape mismatch: {rows} input rows, {fits.num_rows} fitted rows")

    engine = {
        name: fits[name].to_pylist()
        for name in ("intercept", "beta_a", "beta_b", "beta_c", "r2")
    }
    # Which engine column holds which coefficient of [b0, ba, bb, bc].
    # Every coefficient position is checked. Leaving one out would let a
    # systematic mis-mapping of that slot pass unnoticed.
    slots = {"intercept": 0, "beta_a": 1, "beta_b": 2, "beta_c": 3}

    compared = 0
    refused = 0
    worst = 0.0
    worst_where = None
    for row in range(rows):
        lo = max(0, row - preceding)
        window = slice(lo, row + 1)
        design = np.column_stack(
            [np.ones(row + 1 - lo), factors[window]]
        )
        # Rank and conditioning decide what this row can assert.
        singular = np.linalg.matrix_rank(design) < design.shape[1]
        # Fewer rows than parameters is underdetermined and refused.
        # Exactly as many is determined, not under-determined, and the
        # engine fits it — the same convention regr_slope already uses,
        # where two points determine a line.
        too_short = design.shape[0] < design.shape[1]
        condition = np.inf if singular else np.linalg.cond(design)
        engine_refused = engine["beta_a"][row] is None

        if too_short or singular or condition > REFUSAL_CONDITION:
            if not engine_refused:
                sys.exit(
                    f"FAIL row {row}: the engine fitted a window it should have "
                    f"refused (rank-deficient={singular}, short={too_short}, "
                    f"cond={condition:.3e})"
                )
            refused += 1
            continue

        if engine_refused:
            sys.exit(
                f"FAIL row {row}: the engine refused a usable window "
                f"(cond={condition:.3e}, rows={design.shape[0]})"
            )

        coefficients, *_ = np.linalg.lstsq(design, y[window], rcond=None)
        # A Gram route pays kappa^2 where QR pays kappa; give the diff
        # room proportional to that, floored at the tight bound.
        tolerance = max(BASE_TOL, BASE_TOL * (condition / 1e3) ** 2)
        for name, slot in slots.items():
            got = engine[name][row]
            expected = coefficients[slot]
            error = abs(got - expected) / max(abs(expected), 1.0)
            if error > tolerance:
                sys.exit(
                    f"FAIL row {row} {name}: engine {got!r} vs lstsq "
                    f"{expected!r} (rel {error:.3e} > {tolerance:.3e}, "
                    f"cond {condition:.3e})"
                )
            if error > worst:
                worst, worst_where = error, f"row {row} {name}"

        # R^2 from the residuals lstsq itself leaves behind.
        residual = y[window] - design @ coefficients
        centered = y[window] - y[window].mean()
        total = float(centered @ centered)
        if total > 0.0:
            expected_r2 = 1.0 - float(residual @ residual) / total
            got_r2 = engine["r2"][row]
            if got_r2 is None:
                sys.exit(f"FAIL row {row}: engine reported no R2 for a fitted window")
            if abs(got_r2 - expected_r2) > max(1e-9, tolerance):
                sys.exit(
                    f"FAIL row {row} r2: engine {got_r2!r} vs numpy "
                    f"{expected_r2!r} (cond {condition:.3e})"
                )
        compared += 1

    if compared == 0:
        sys.exit("vacuous run: no window was comparable")
    if refused == 0:
        sys.exit(
            "vacuous refusal check: no window was rank-deficient, so the "
            "NULL semantic was never exercised — the fixture's flat stretch "
            "has stopped being flat"
        )
    print(
        f"Multi-factor rolling fit validated end-to-end: {compared} windows "
        f"x {{intercept, beta_a, beta_b, beta_c, R2}} against np.linalg.lstsq "
        f"(QR/SVD, not the engine's normal-equations path), plus {refused} "
        f"rank-deficient windows the engine refused exactly as ruled "
        f"(numpy {np.__version__})"
    )
    print(f"worst relative disagreement {worst:.3e} at {worst_where}")


if __name__ == "__main__":
    main()
