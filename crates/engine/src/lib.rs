//! `engine` — ties storage, query, and compute together; owns the
//! numeric-or-key schema invariant.
//!
//! ## This crate's one non-negotiable job
//! Enforce numeric-or-key as a **hard** schema constraint. A column that
//! can't be classified as numeric or key is rejected at schema-definition
//! time, not silently coerced, not stored as a third type "just this
//! once." Every other crate in this workspace assumes this invariant
//! already holds by the time data reaches them — this is the one place
//! that's actually responsible for making that true. Do not weaken this
//! to unblock a feature; if something seems to need a third column type,
//! that's a signal to stop and reconsider the feature, not the invariant.
//!
//! ## What this crate is
//! The public entry point: schema definition/validation, wiring
//! `storage-lite` + `query-lite` together, and exposing the compute
//! backends (`compute-lua`, `compute-blas`) to SQL as callable functions.
//! Application code depends on this crate, not on the lower-level crates
//! directly.
//!
//! ## Compute backend selection
//! `compute-lua` and `compute-blas` are consumed here through their trait
//! interfaces (see those crates), not through concrete types. Right now
//! that means the native implementations (LuaJIT, native BLAS/LAPACK) are
//! the only ones that exist — but this crate should never hardcode that
//! assumption. Select the concrete implementation with
//! `cfg(target_arch = "wasm32")` / a Cargo feature at the point where a
//! concrete type is actually needed, not throughout this crate's logic.
//!
//! ## Current milestone: native only
//! Nothing here should assume a filesystem, threading model, or blocking
//! I/O that would foreclose a future wasm32 build — but building that
//! WASM target is explicitly not the current goal. Don't gold-plate the
//! WASM path prematurely; do keep the trait boundaries clean so it isn't
//! a rewrite later.

// TODO: schema definition API (numeric vs. key column declaration,
//       rejecting anything else at definition time)
// TODO: wire storage-lite + query-lite into a single connection/handle type
// TODO: expose curated compute-blas ops as callable SQL functions
// TODO: expose compute-lua as a callable-from-SQL scripting layer
//       (batch calling convention — whole column/window per call, not
//       per-row; see compute-lua's docs)
