#!/usr/bin/env python3
"""PyArrow round-trip oracle for arrow-lite's C Data Interface (issue #15).

Drives the `oracle-harness` hooks in libarrow_lite as a shared library and
checks both directions against fixtures defined independently here and in
`src/harness.rs`:

  1. arrow-lite exports the canonical batch -> PyArrow imports and compares.
  2. PyArrow exports the same batch          -> arrow-lite imports and compares.
  3. arrow-lite exports a 3-batch stream     -> PyArrow reads and compares.
  4. PyArrow exports the same stream         -> arrow-lite reads and compares.

Usage: pyarrow_roundtrip.py [path/to/libarrow_lite.so]
Exits nonzero on the first failure.
"""

import ctypes
import decimal
import sys
from pathlib import Path

import pyarrow as pa
from pyarrow.cffi import ffi


def load_library() -> ctypes.CDLL:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        repo = Path(__file__).resolve().parents[3]
        path = repo / "target" / "debug" / "libarrow_lite.so"
    if not path.exists():
        sys.exit(
            f"{path} not found - build it with "
            "`cargo build -p arrow-lite --features oracle-harness`"
        )
    return ctypes.CDLL(str(path))


def canonical_batch() -> pa.RecordBatch:
    """The canonical batch; must agree with harness.rs, field for field."""
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("ns"), nullable=False),
            pa.field(
                "sym",
                pa.dictionary(pa.uint32(), pa.string()),
                nullable=False,
            ),
            pa.field("px", pa.float64(), nullable=True),
            pa.field("qty", pa.int64(), nullable=False),
            pa.field("amt", pa.decimal64(18, 2), nullable=False),
        ]
    )
    return pa.record_batch(
        [
            pa.array([1_000, 2_000, 3_000, 4_000], pa.timestamp("ns")),
            pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 0, 2], pa.uint32()),
                pa.array(["AAPL", "MSFT", "TSLA"], pa.string()),
            ),
            pa.array([101.5, None, 99.25, None], pa.float64()),
            pa.array([10, 20, 30, 40], pa.int64()),
            pa.array(
                [decimal.Decimal(v) for v in ("199.99", "2.50", "-0.75", "0.00")],
                pa.decimal64(18, 2),
            ),
        ],
        schema=schema,
    )


def slice_batches() -> list[pa.RecordBatch]:
    """The stream fixture; must agree with harness.rs."""
    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("n", pa.int64(), nullable=False),
        ]
    )
    return [
        pa.record_batch(
            [
                pa.array([float(i), i + 0.5], pa.float64()),
                pa.array([i * 10, i * 10 + 1], pa.int64()),
            ],
            schema=schema,
        )
        for i in range(3)
    ]


def new_ptrs():
    c_schema = ffi.new("struct ArrowSchema*")
    c_array = ffi.new("struct ArrowArray*")
    return (
        c_schema,
        c_array,
        int(ffi.cast("uintptr_t", c_schema)),
        int(ffi.cast("uintptr_t", c_array)),
    )


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        sys.exit(f"FAIL {name}{': ' + detail if detail else ''}")
    print(f"PASS {name}")


def main() -> None:
    lib = load_library()
    lib.tallydb_oracle_verify_batch.restype = ctypes.c_int32
    lib.tallydb_oracle_verify_stream.restype = ctypes.c_int32

    # 1. arrow-lite exports, PyArrow imports.
    c_schema, c_array, schema_ptr, array_ptr = new_ptrs()
    lib.tallydb_oracle_export_batch(
        ctypes.c_void_p(schema_ptr), ctypes.c_void_p(array_ptr)
    )
    imported = pa.RecordBatch._import_from_c(array_ptr, schema_ptr)
    imported.validate(full=True)
    expected = canonical_batch()
    check(
        "arrow-lite -> pyarrow batch",
        imported.equals(expected) and imported.schema == expected.schema,
        f"imported {imported!r}, expected {expected!r}",
    )

    # 2. PyArrow exports, arrow-lite imports and verifies.
    c_schema, c_array, schema_ptr, array_ptr = new_ptrs()
    expected._export_to_c(array_ptr, schema_ptr)
    rc = lib.tallydb_oracle_verify_batch(
        ctypes.c_void_p(schema_ptr), ctypes.c_void_p(array_ptr)
    )
    check("pyarrow -> arrow-lite batch", rc == 0, f"verify returned {rc}")

    # 3. arrow-lite exports a stream, PyArrow reads it.
    c_stream = ffi.new("struct ArrowArrayStream*")
    stream_ptr = int(ffi.cast("uintptr_t", c_stream))
    lib.tallydb_oracle_export_stream(ctypes.c_void_p(stream_ptr))
    reader = pa.RecordBatchReader._import_from_c(stream_ptr)
    read = list(reader)
    expected_slices = slice_batches()
    check(
        "arrow-lite -> pyarrow stream",
        len(read) == len(expected_slices)
        and all(a.equals(e) for a, e in zip(read, expected_slices)),
        f"read {read!r}",
    )

    # 4. PyArrow exports a stream, arrow-lite reads and verifies.
    c_stream = ffi.new("struct ArrowArrayStream*")
    stream_ptr = int(ffi.cast("uintptr_t", c_stream))
    reader = pa.RecordBatchReader.from_batches(
        expected_slices[0].schema, expected_slices
    )
    reader._export_to_c(stream_ptr)
    rc = lib.tallydb_oracle_verify_stream(ctypes.c_void_p(stream_ptr))
    check("pyarrow -> arrow-lite stream", rc == 0, f"verify returned {rc}")

    print(f"all pyarrow round-trips passed (pyarrow {pa.__version__})")


if __name__ == "__main__":
    main()
