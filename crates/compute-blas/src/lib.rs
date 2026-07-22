//! `compute-blas` — curated BLAS/LAPACK operations, callable from SQL
//! and from `compute-lua`.
//!
//! ## Curated, not general — this is the actual scope filter
//! We do not expose "all of BLAS/LAPACK." We expose the specific
//! operations the target workflows (quant research on time-series data)
//! actually need:
//! - Rolling regression -> least-squares solve (QR or SVD-based)
//! - Covariance / PCA -> symmetric eigendecomposition
//! - Portfolio weights / factor models -> general linear solve
//! - Fast path for positive-definite covariance work -> Cholesky
//! That's the list. Do not add routines speculatively "in case someone
//! wants them" — see the root README on the general inclusion principle,
//! which applies to SQL syntax, not to which LAPACK routines get wrapped.
//! The bar for adding a new op here is a real, specific workflow need,
//! not "LAPACK has it so we could."
//!
//! ## BLAS vs. LAPACK — don't conflate them
//! BLAS (multiplication-class: dot products, matrix-vector,
//! matrix-matrix) is necessary but not sufficient. The actual analytics
//! above are LAPACK-class (solves, eigendecomposition, SVD). Both are in
//! scope; "we have BLAS" is not "we're done."
//!
//! ## Native backend: link as-is
//! OpenBLAS (or MKL/Accelerate) via FFI, no fork, no rebuild — mature,
//! narrow, embedding-oriented, exactly the kind of dependency this
//! project takes whole. Where numerical determinism matters (eventual
//! native/WASM consistency), build OpenBLAS from source with
//! `TARGET=SANDYBRIDGE` — this forces pre-FMA kernels (AVX, no FMA) while
//! staying fast on essentially any x86_64 CPU from 2011 onward; prefer
//! this over the more conservative `TARGET=NEHALEM` (SSE-only) unless
//! there's a specific reason to support pre-2011 hardware. There is no
//! off-the-shelf "non-FMA" package — this is a build-time decision, make
//! it deliberately when it's actually needed, not preemptively.
//!
//! ## WASM backend: future, not current milestone
//! `blas.wasm` (github.com/andy-emerson/blas.wasm) already exists,
//! SIMD-tuned and bit-identical by design, deferring FMA specifically to
//! preserve determinism. A LAPACK-in-WASM layer does not yet exist as of
//! this writing (per that project's own README) and is its likely next
//! milestone. Do not block native-milestone work on this.
//!
//! ## Explicitly NOT in scope
//! No autodiff. No general tensor operations. No "as much of LAPACK as
//! we can wrap" — see the curated list above. If this crate's op list is
//! growing without a specific, named workflow driving each addition,
//! that's scope drift — stop and check against the root README.

// TODO: FFI bindings to system BLAS/LAPACK (bindgen or hand-written,
//       scoped to the routines actually used below)
// TODO: least-squares solve (regression)
// TODO: symmetric eigendecomposition (covariance / PCA)
// TODO: general linear solve (portfolio weights / factor models)
// TODO: Cholesky (fast path for positive-definite covariance)
// TODO: expose all of the above as SQL-callable functions via `engine`,
//       and as Lua-callable functions via `compute-lua`, sharing
//       arrow-lite buffers with no copy in either direction
