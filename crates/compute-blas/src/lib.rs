//! `compute-blas` — multiplication-class BLAS operations, callable from the
//! query executor and from `compute-lua`.
//!
//! ## Scope: multiplication-class primitives only
//! This crate wraps dot products, matrix-vector (gemv), and matrix-matrix
//! (gemm), and nothing else. The analytical solves and decompositions —
//! least squares, symmetric eigendecomposition, general solve, Cholesky —
//! are LAPACK-class, and TallyDB does not build them: every statistic the
//! engine currently computes has an exact closed form at the two
//! parameters or two dimensions it needs, and a general solver's per-call
//! overhead dwarfs that arithmetic at window scale. A LAPACK-class
//! dependency returns only when an op needs more than that, where no
//! closed form exists. See DESIGN.md, *Curated compute: what the engine
//! calls, and why*.
//!
//! Keeping this crate's scope narrow is still the point: BLAS is
//! necessary but not sufficient, and "we have BLAS" is not "we're done."
//!
//! ## Native backend: link as-is
//! OpenBLAS (or MKL/Accelerate) via FFI, no fork, no rebuild — mature,
//! narrow, embedding-oriented, exactly the kind of dependency this project
//! takes whole. Where numerical determinism matters (eventual native/WASM
//! consistency), build OpenBLAS from source with `TARGET=SANDYBRIDGE` — this
//! forces pre-FMA kernels (AVX, no FMA) while staying fast on essentially any
//! x86_64 CPU from 2011 onward; prefer this over the more conservative
//! `TARGET=NEHALEM` (SSE-only) unless there's a specific reason to support
//! pre-2011 hardware. There is no off-the-shelf "non-FMA" package — this is a
//! build-time decision, make it deliberately when it's actually needed.
//!
//! ## Numeric type: `f64`
//! BLAS operates on `f64`/`f32` contiguous buffers, and the analytics that
//! consume these primitives are floating-point anyway. `i64` columns
//! (timestamps, money, counts) are the exact/stored type — they are converted
//! to `f64` before being handed to a BLAS routine, not passed in raw. See the
//! numeric-type discussion in DESIGN.md.
//!
//! ## Capability negotiation
//! Ops are exposed through [`backend::BlasBackend`], which answers
//! "unavailable on this backend" as a first-class result rather than
//! panicking — the same seam `compute-lapack` uses, so `engine` treats
//! both compute crates uniformly.
//!
//! ## WASM backend: future, not current milestone
//! `blas.wasm` (github.com/andy-emerson/blas.wasm) already exists,
//! SIMD-tuned and bit-identical by design, deferring FMA specifically to
//! preserve determinism. Do not add this dependency until the WASM milestone
//! actually starts.
//!
//! ## Batch, not per-row
//! Every entry point takes whole columns / windows per call, per the batch
//! rule in DESIGN.md. If the API makes per-row calls easy, that's a bug
//! in the API shape.
//!
//! ## Explicitly NOT in scope
//! No LAPACK-class routines (see `compute-lapack`). No autodiff. No general
//! tensor operations.

pub mod backend;

pub use backend::{BlasBackend, BlasError, BlasOp, NativeBlas};

// TODO: wire into the executor's numeric inner loops WHEN (and only
//       when) profiling produces a number that asks for it — the crate
//       docs' discipline
// TODO: expose to `compute-lua` over shared arrow-lite buffers (M2.7)
