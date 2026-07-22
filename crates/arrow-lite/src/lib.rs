//! `arrow-lite` — Arrow-layout-compatible columnar in-memory format.
//!
//! ## Why this crate exists
//! TallyDB's numeric-or-key invariant means every value column is a flat,
//! fixed-width numeric buffer. If that buffer's memory layout matches
//! Apache Arrow's columnar spec, query results are directly usable by
//! NumPy/Arrow-aware tooling with zero copying — no conversion step between
//! "database result" and "numeric array." That's the property this crate
//! exists to guarantee.
//!
//! ## Scope (deliberately narrow)
//! - Numeric column types — **`f64` and `i64`** — each stored as a contiguous
//!   fixed-width buffer plus a validity/null bitmap, per Arrow's layout.
//!   `f64` is the analytic type (what BLAS/LAPACK consume as raw buffers);
//!   `i64` is the exact/stored type (nanosecond timestamps, money as scaled
//!   integers, counts) — nanosecond epochs do **not** fit in `f64`, so `i64`
//!   is not optional. Both are Arrow primitive layouts, so the zero-copy
//!   NumPy/Arrow interop property holds for either.
//! - A key column type: dictionary-encoded to `u32`/`u64` indices plus a
//!   string-interning table, per Arrow's dictionary encoding.
//! - NOT a general Arrow implementation. No variable-length lists, no
//!   nested/struct types, no non-numeric primitive types beyond `f64`/`i64`.
//!   If a type isn't numeric or key, it doesn't belong here — see the root
//!   README.
//!
//! ## Lock these two views early (they're the whole contract)
//! Everything downstream inherits this crate's layout, so pin two interfaces
//! before adding features: (1) the **raw-pointer / FFI view** — how a
//! contiguous numeric buffer is handed to `compute-*` as a pointer for
//! zero-copy compute; and (2) the **serialize-to-segment view** — how a
//! column (including a key's dictionary) is written into a `storage-lite`
//! segment. The load-bearing decisions are the boring ones: dictionary index
//! width (`u32` vs `u64`), null-bitmap presence, and the `f64`/`i64` subtype
//! tag. Get those right; defer anything fancier.
//!
//! ## Open question for whoever implements this (see conversation history
//! / README "Design philosophy" section)
//! Two viable paths, either is acceptable, pick one and document why:
//! 1. Hand-roll the buffer/bitmap structs ourselves, matching Arrow's
//!    layout spec exactly but with zero external dependency.
//! 2. Take a thin dependency on the real `arrow-array`/`arrow-buffer`
//!    crates and wrap them.
//! Either way: the crate's own tests should validate real interop (e.g.
//! round-tripping through `arrow-rs` or PyArrow in a test) — memory layout
//! bugs here are silent and expensive to find later, so don't skip this.
//!
//! ## What NOT to pull in
//! Do not depend on `arrow-compute` or `datafusion` here. This crate is
//! the *data format* only. Compute lives in `compute-lua` / `compute-blas`,
//! and query execution lives in `query-lite`.

// TODO: numeric column buffer types (f64 and i64, each with validity bitmap)
// TODO: key column type (dictionary-encoded, string interning table)
// TODO: column enum wrapping numeric (f64/i64) + key variants, nothing else
// TODO: round-trip test against a real Arrow implementation (both numeric
//       subtypes and dictionary columns)
