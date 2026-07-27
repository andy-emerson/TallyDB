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
//! backends (`compute-lua`, `compute-linalg`) to SQL as callable
//! functions. Application code depends on this crate, not on the
//! lower-level crates directly.
//!
//! ## Compute backend selection
//! `compute-linalg` is consumed here through its trait interface (see that
//! crate), not through a concrete type; `compute-lua` is currently
//! consumed as its concrete native state (its backend trait is extracted
//! when the WASM backend starts — see that crate's docs). Right now the
//! implementations (vendored Lua 5.4, pure-Rust linear algebra) are the only
//! ones that exist — but this crate should never hardcode that
//! assumption. Select the concrete implementation with
//! `cfg(target_arch = "wasm32")` / a Cargo feature at the point where a
//! concrete type is actually needed, not throughout this crate's logic.
//! Route compute calls so that a backend reporting an op as unavailable
//! surfaces as a clean "unsupported here" error, not a panic — the
//! compute crates expose that capability signal on their traits.
//!
//! ## No LAPACK on the query path
//! Every window function this crate registers is solved in closed form:
//! a two-parameter regression and a 2 × 2 symmetric eigenvalue both have
//! exact solutions, and a general solver's per-call overhead dwarfs their
//! arithmetic at window scale. The engine therefore links no LAPACK
//! routine at all, which is what lets a WASM build reach parity with the
//! current feature set without a LAPACK-in-WASM layer. A
//! LAPACK-class dependency returns only when an op needs more than two
//! parameters or dimensions — where no closed form exists. See DESIGN.md,
//! *Curated compute: what the engine calls, and why*.
//!
//! ## Concurrency: single writer, concurrent snapshot readers
//! Exactly one `&mut Table` writer exists at a time — a compile-time
//! fact, not a runtime policy — while any number of threads hold
//! [`TableReader`]s and query point-in-time [`TableSnapshot`]s. A
//! snapshot never blocks the writer for longer than one write-buffer
//! copy under a brief per-table lock, and a compaction never
//! invalidates a held snapshot (its segments stay alive through their
//! `Arc`s). Cross-table snapshot consistency is deliberately not
//! promised.
//!
//! ## Current milestone: native only
//! Nothing here should assume a filesystem, threading model, or blocking
//! I/O that would foreclose a future wasm32 build — but building that
//! WASM target is explicitly not the current goal. Don't gold-plate the
//! WASM path prematurely; do keep the trait boundaries clean so it isn't
//! a rewrite later.

pub mod database;
#[cfg(feature = "oracle-harness")]
pub mod harness;
mod script;
pub mod table;

pub use database::Database;
pub use query_lite::QueryOutput;
pub use storage_lite::{RowValue, StoreOptions, WalSync};
pub use table::{schema_from_create, EngineError, Table, TableReader, TableSnapshot};

// The Lua-in-SQL window slot is built: `Table::register_lua_window`
// runs application-registered Lua kernels as SQL window functions
// through the same seam as the curated native windows (the `script`
// module; whole window per call, never per-row), and kernels call the
// curated ops — dot, regr_slope/intercept, covar_pop/corr/eigen_max —
// over the same views, no copy.
//
// TODO: expose the remaining compute-linalg (multiplication-class) ops as
//       callable SQL functions, with backend-capability errors surfaced
//       cleanly (not panics)
// TODO: the scalar-projection Lua slot (#53, post-M2: PlanItem is
//       Column|WindowAgg only; computed projections aren't built)
