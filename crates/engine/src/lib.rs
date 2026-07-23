//! `engine` — ties storage, query, and compute together; owns the
//! numeric-or-key schema invariant.
//!
//! ## This crate's one non-negotiable job
//! Enforce numeric-or-key as a **hard** schema constraint. A column is
//! either numeric (`f64` or `i64`) or a dictionary-encoded key; anything
//! that can't be classified as one of those is rejected at schema-definition
//! time, not silently coerced, not stored as a third type "just this
//! once." Every other crate in this workspace assumes this invariant
//! already holds by the time data reaches them — this is the one place
//! that's actually responsible for making that true. Do not weaken this
//! to unblock a feature; if something seems to need a third column type,
//! that's a signal to stop and reconsider the feature, not the invariant.
//! The invariant holds across the whole pipeline (results and intermediates,
//! not just stored columns): no operation may produce a value that is neither
//! numeric nor key — in particular, nothing here emits a bare string.
//!
//! ## The ordering key
//! The schema also declares the **ordering key** — the column ingest arrives
//! roughly sorted on, that `storage-lite` partitions and builds zone maps on.
//! It is usually a timestamp but need not be (any monotonic-on-ingest numeric
//! key works). Treat it as a declared property of the schema, not a hardcoded
//! "time" column.
//!
//! ## What this crate is
//! The public entry point: schema definition/validation, wiring
//! `storage-lite` + `query-lite` together, and exposing the compute
//! backends (`compute-lua`, `compute-blas`, `compute-lapack`) to SQL as
//! callable functions. Application code depends on this crate, not on the
//! lower-level crates directly.
//!
//! ## Compute backend selection
//! `compute-lua`, `compute-blas`, and `compute-lapack` are consumed here
//! through their trait interfaces (see those crates), not through concrete
//! types. Right now that means the native implementations (LuaJIT, native
//! BLAS, native LAPACK) are the only ones that exist — but this crate should
//! never hardcode that assumption. Select the concrete implementation with
//! `cfg(target_arch = "wasm32")` / a Cargo feature at the point where a
//! concrete type is actually needed, not throughout this crate's logic. Route
//! compute calls so that a backend reporting an op as unavailable (e.g. a
//! future WASM build with BLAS but not yet LAPACK) surfaces as a clean
//! "unsupported here" error, not a panic — the compute crates expose that
//! capability signal on their traits.
//!
//! ## Current milestone: native only
//! Nothing here should assume a filesystem, threading model, or blocking
//! I/O that would foreclose a future wasm32 build — but building that
//! WASM target is explicitly not the current goal. Don't gold-plate the
//! WASM path prematurely; do keep the trait boundaries clean so it isn't
//! a rewrite later.

#[cfg(feature = "oracle-harness")]
pub mod harness;
pub mod table;

pub use storage_lite::RowValue;
pub use table::{EngineError, Table};

// TODO: multi-table handle (connection type) once storage grows past the
//       single write-then-read M1 table
// TODO: expose compute-blas (multiplication-class) ops and the remaining
//       compute-lapack ops as callable SQL functions, with
//       backend-capability errors surfaced cleanly (not panics)
// TODO: expose compute-lua as a callable-from-SQL scripting layer
//       (batch calling convention — whole column/window per call, not
//       per-row; see compute-lua's docs)
