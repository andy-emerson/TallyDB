//! `arrow-lite` — hand-rolled, Arrow-layout-compatible columnar in-memory
//! format.
//!
//! ## Why this crate exists
//! TallyDB's numeric-or-key invariant means every value column is a flat,
//! fixed-width buffer. If that layout matches Apache Arrow's columnar spec,
//! the *same bytes* serve three consumers with zero copying: compute (raw
//! pointers into BLAS/LAPACK/Lua), storage (serialize/mmap into segments),
//! and the outside world (query results handed to NumPy/Polars/DuckDB via
//! the Arrow C Data Interface with no conversion step). That third boundary
//! is the reason to be *Arrow*-shaped specifically — the first two only need
//! "contiguous." Being Arrow internally is what makes "no conversion step"
//! true by construction rather than by an export routine.
//!
//! ## Decision (issue #2, resolved): hand-rolled, oracle-tested
//! We implement the layout ourselves rather than wrapping `arrow-array`/
//! `arrow-buffer`. Reasons: the subset we need is tiny and Arrow's layout
//! spec is frozen; arrow-rs is a large, fast-churning dependency sitting
//! under every other crate; and the two-variant wrapper layer must exist
//! either way (arrow-rs happily builds string columns, so it could never be
//! the public type). The risk this buys — hand-written unsafe C Data
//! Interface export (release callbacks, LSB bitmaps, buffer counts, format
//! strings) — is narrow and cold, and is covered by **round-trip tests
//! against arrow-rs and PyArrow as dev-only oracles** (build → export →
//! import with the real implementation → diff, and the reverse). They play
//! the same role DuckDB plays for `query-lite`: validate our output in
//! tests, never linked at runtime.
//!
//! ## The pieces
//! - **Numeric columns:** `f64` and `i64` contiguous buffers (64-byte
//!   aligned), one entry per row. The subtype tag is an extensible enum so
//!   a future `F32` (GPU/bandwidth path — issue #3) is additive, not a
//!   format migration.
//! - **Validity:** an optional side `Bitmap`, present only for columns the
//!   schema declares nullable. A `NOT NULL` column has no bitmap and is
//!   BLAS-ready by construction. The ordering key is always `NOT NULL`.
//! - **`Bitmap` as a first-class shared type:** LSB-ordered per Arrow, with
//!   and/or/not, popcount, and set-bit iteration. Used for validity here,
//!   for row selections in `query-lite` (WHERE, dictionary-LIKE bitmaps),
//!   and plausibly for tombstone masks in `storage-lite` — one
//!   implementation, shared by all three.
//! - **Key columns:** `u32` dictionary codes per row (u32 only — no u64
//!   variant) plus an interning table of distinct values.
//! - **Views:** zero-copy slices (offset + length) over any column —
//!   Arrow-native offsets. This is what lets window functions feed compute
//!   with pointer arithmetic instead of a copy per window.
//! - **Logical-type annotations:** an optional tag (`Timestamp(ns)`,
//!   `Decimal64(scale)`) over the same physical `i64` buffer, consulted
//!   only at export, so ecosystem consumers see datetimes and decimals
//!   instead of opaque integers. No effect on in-engine representation.
//! - **Export:** the Arrow **C Data Interface only**, including the
//!   `ArrowArrayStream` variant — results leave as a stream of record
//!   batches, matching segment-at-a-time execution with no final
//!   concatenation copy.
//!
//! ## The one variable-width structure (and why it's fine)
//! The dictionary's values are strings — the single variable-width buffer in
//! the system. It is *reference data, not row data*: sized by distinct
//! values, not rows; touched at intern time and once-per-distinct-value
//! predicate evaluation, never on the per-row scan/compute path. Corollary
//! (see DESIGN.md): keys assume repeating, low-cardinality labels. A
//! never-repeating identifier is a number — it belongs in an `i64` column,
//! not a key.
//!
//! ## What NOT to pull in or build
//! - No `arrow-compute`, no `datafusion` — this crate is the data format
//!   only. Compute lives in `compute-*`; execution lives in `query-lite`.
//! - No Arrow IPC / Flight / Parquet — the C Data Interface is the entire
//!   interop surface. File formats are the application's job via ecosystem
//!   tools that already speak C-Data.
//! - No matrix/column-group arena for LAPACK-shaped allocations —
//!   considered and deferred (issue #4): the design-matrix gather is
//!   O(n·k) against an O(n·k²) solve. Revisit only with profiling evidence.

pub mod bitmap;
pub mod buffer;
pub mod key;

pub use bitmap::Bitmap;
pub use buffer::{Buffer, Element, NumericColumn, BUFFER_ALIGN};
pub use key::{Dictionary, KeyColumn};

// TODO: column enum wrapping numeric + key variants, nothing else
// TODO: zero-copy views (offset + len) over columns
// TODO: logical-type tags (Timestamp(ns), Decimal64(scale)) on i64 columns,
//       consulted at export only
// TODO: C Data Interface export/import, incl. ArrowArrayStream
// TODO: round-trip tests against arrow-rs and PyArrow (dev-only oracles),
//       covering both numeric subtypes, dictionaries, nulls, logical-type
//       annotations, and batch streams
