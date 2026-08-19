//! `compute-linalg` — multiplication-class linear algebra, callable from
//! the query executor and from `compute-lua`. Pure Rust, no system
//! library.
//!
//! ## Scope: multiplication-class primitives only
//! This crate wraps dot products, matrix–vector, and matrix–matrix
//! products, and nothing else. The analytical solves and decompositions —
//! least squares, symmetric eigendecomposition, general solve, Cholesky —
//! are solver-class, and TallyDB does not build them: every statistic the
//! engine currently computes has an exact closed form at the two
//! parameters or two dimensions it needs, and a general solver's per-call
//! overhead dwarfs that arithmetic at window scale. A solver-class
//! dependency returns only when an op needs more than that, where no
//! closed form exists — and the measured candidate for that day is faer's
//! own solvers, already this crate's kernel source. See DESIGN.md,
//! *Curated compute: what the engine calls, and why*.
//!
//! Keeping this crate's scope narrow is still the point: having fast
//! products is necessary but not sufficient, and "we have the kernels"
//! is not "we're done."
//!
//! ## One pure-Rust implementation, both targets
//! The kernels come from [faer](https://crates.io/crates/faer) (slim
//! features: no thread pool, no RNG, no file formats), except `dot`,
//! which is a strict left-to-right loop — measured fastest at window
//! scale *and* bit-identical on every CPU and target, which no
//! runtime-dispatched SIMD kernel guarantees (see `backend`). Because
//! all of it is Rust, the same implementation compiles for native and
//! wasm32 — there is no separate WASM backend to build, and no system
//! BLAS to install, link, or version. (The predecessor of this crate
//! linked system BLAS via FFI; the swap was measured, not assumed —
//! faer met or beat reference BLAS at every multiplication shape the
//! engine could plausibly run, and the decision record lives in
//! DESIGN.md.)
//!
//! ## Numeric type: `f64`
//! The kernels operate on `f64` contiguous buffers, and the analytics
//! that consume these primitives are floating-point anyway. `i64`
//! columns (timestamps, money, counts) are the exact/stored type — they
//! are converted to `f64` before being handed to a kernel, not passed in
//! raw. See the numeric-type discussion in DESIGN.md.
//!
//! ## Capability negotiation
//! Ops are exposed through [`backend::LinalgBackend`], which answers
//! "unavailable on this backend" as a first-class result rather than
//! panicking — the seam that lets a backend be swapped or reported
//! missing without a caller changing.
//!
//! ## Batch, not per-row
//! Every entry point takes whole columns / windows per call, per the
//! batch rule in DESIGN.md. If the API makes per-row calls easy, that's
//! a bug in the API shape.
//!
//! ## Explicitly NOT in scope
//! No solver-class routines *here*. No autodiff. No general tensor
//! operations.
//!
//! This crate stays multiplication-class. The one solve TallyDB does
//! own — a fixed-size symmetric `K × K` Cholesky for the rolling
//! multi-factor fit (#90, ruled 2026-08-03) — deliberately does **not**
//! live behind this trait: it is a few dozen lines operating on
//! moments the window layer already maintains, it needs no backend
//! negotiation, and putting it here would turn a private detail into a
//! general solver surface. See `engine::multifactor`.

pub mod backend;

pub use backend::{LinalgBackend, LinalgError, LinalgOp, RustLinalg};

// TODO: wire into the executor's numeric inner loops WHEN (and only
//       when) profiling produces a number that asks for it — the crate
//       docs' discipline
