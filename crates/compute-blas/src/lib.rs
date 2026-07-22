//! `compute-blas` — multiplication-class BLAS operations, callable from the
//! query executor and from `compute-lua`.
//!
//! ## Scope: BLAS only. LAPACK lives next door.
//! This crate wraps the *multiplication-class* primitives — dot products,
//! matrix-vector (gemv), matrix-matrix (gemm) — and nothing else. The
//! analytical solves and decompositions (least-squares, symmetric
//! eigendecomposition, general solve, Cholesky) are LAPACK-class and live in
//! the separate `compute-lapack` crate. Don't conflate them: BLAS is
//! necessary but not sufficient, and "we have BLAS" is not "we're done."
//!
//! The split follows a real boundary — different libraries (LAPACK is built
//! on BLAS), different consumers (the primitives here are called directly by
//! the executor's window/numeric inner loops and by Lua-over-FFI, whereas the
//! LAPACK curated set is the higher-level analytics surface), and different
//! WASM-availability timelines (`blas.wasm` exists today; a LAPACK-in-WASM
//! layer does not).
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
//! numeric-type discussion in the root docs.
//!
//! ## Capability negotiation
//! Expose ops through a trait that can answer "unavailable on this backend"
//! as a first-class result rather than panicking — the same seam
//! `compute-lapack` needs. It matters less here (BLAS has a WASM backend
//! already) but keeping both compute crates on the same trait shape is what
//! lets `engine` treat them uniformly.
//!
//! ## WASM backend: future, not current milestone
//! `blas.wasm` (github.com/andy-emerson/blas.wasm) already exists,
//! SIMD-tuned and bit-identical by design, deferring FMA specifically to
//! preserve determinism. Do not add this dependency until the WASM milestone
//! actually starts.
//!
//! ## Batch, not per-row
//! Every entry point takes whole columns / windows per call, per the batch
//! rule in the root docs. If the API makes per-row calls easy, that's a bug
//! in the API shape.
//!
//! ## Explicitly NOT in scope
//! No LAPACK-class routines (see `compute-lapack`). No autodiff. No general
//! tensor operations.

// TODO: BLAS backend trait (native-OpenBLAS-via-FFI implementation first),
//       with capability negotiation on the same trait shape as compute-lapack
// TODO: dot product
// TODO: matrix-vector multiply (gemv)
// TODO: matrix-matrix multiply (gemm)
// TODO: expose the above as functions callable from `compute-lua` (shared
//       arrow-lite buffers, no copy) and, via `engine`, from SQL
