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
//! - A numeric column type (default `f64`), stored as a contiguous buffer
//!   plus a validity/null bitmap, per Arrow's layout.
//! - A key column type: dictionary-encoded to `u32`/`u64` indices plus a
//!   string-interning table, per Arrow's dictionary encoding.
//! - NOT a general Arrow implementation. No variable-length lists, no
//!   nested/struct types, no non-numeric primitive types. If a type isn't
//!   numeric or key, it doesn't belong here — see the root README.
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

// TODO: numeric column buffer type (f64 default, validity bitmap)
// TODO: key column type (dictionary-encoded, string interning table)
// TODO: column enum wrapping the above two variants and nothing else
// TODO: round-trip test against a real Arrow implementation
