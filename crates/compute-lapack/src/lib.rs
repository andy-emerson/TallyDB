//! `compute-lapack` — curated LAPACK-class solves and decompositions,
//! callable from SQL (via `engine`) and from `compute-lua`.
//!
//! ## Why this is a separate crate from `compute-blas`
//! BLAS (multiplication-class: dot, matrix-vector, matrix-matrix) and LAPACK
//! (solves, eigendecomposition, SVD) are different libraries with different
//! consumers and — importantly for us — different WASM-availability
//! timelines. LAPACK is *built on top of* BLAS and calls into it, but the two
//! surfaces are distinct. Splitting them into two crates follows that real
//! boundary honestly, instead of hiding LAPACK-class routines inside a crate
//! named "blas." This crate does **not** depend on `compute-blas` at the Rust
//! level: it calls the LAPACK library, which internally calls its own BLAS.
//!
//! ## Curated, not general — this is the actual scope filter
//! We do not expose "all of LAPACK." We expose the specific operations the
//! target workflows (quant research / numeric analytics on ordered data)
//! actually need:
//! - Rolling regression -> least-squares solve (QR or SVD-based)
//! - Covariance / PCA -> symmetric eigendecomposition
//! - Portfolio weights / factor models -> general linear solve
//! - Fast path for positive-definite covariance work -> Cholesky
//! That's the list. Do not add routines speculatively "in case someone wants
//! them." The bar for adding an op is a real, specific, repeated workflow
//! need, not "LAPACK has it so we could." If this crate's op list is growing
//! without a named workflow driving each addition, that's scope drift — stop
//! and check against DESIGN.md.
//!
//! ## `f64`, not integers or rationals
//! These routines produce irrational results in general (eigenvalues, √ in
//! correlations, least-squares coefficients), so the analytic numeric type is
//! `f64` — see the numeric-type discussion in DESIGN.md. `i64` columns
//! (timestamps, money, counts) are the exact/stored type and are converted to
//! `f64` on the way into these routines, not fed in raw.
//!
//! ## Native backend: link as-is
//! LAPACK (from OpenBLAS/MKL/Accelerate) via FFI, no fork, no rebuild —
//! mature, narrow, embedding-oriented, exactly the kind of dependency this
//! project takes whole. Where numerical determinism matters (eventual
//! native/WASM consistency), the same `TARGET=SANDYBRIDGE` non-FMA build note
//! from `compute-blas` applies to the BLAS that LAPACK calls underneath.
//!
//! ## Capability negotiation (this is the design-critical seam)
//! The WASM LAPACK backend does not exist yet, while the WASM BLAS backend
//! (`blas.wasm`) does. So every op here must be reachable through a trait
//! that can answer **"unavailable on this backend"** as a first-class,
//! queryable result — never a panic. A future WASM build should come up with
//! BLAS-class ops working and these LAPACK-class ops gracefully degraded
//! until a LAPACK-in-WASM layer ships. Build that negotiation into the trait
//! now; retrofitting it later is a rewrite.
//!
//! ## Batch, not per-row
//! Every entry point takes whole columns / windows (matrices) per call, per
//! the batch rule in DESIGN.md. There is no per-element calling
//! convention here.
//!
//! ## Explicitly NOT in scope
//! No autodiff. No general tensor operations. No "as much of LAPACK as we can
//! wrap" — see the curated list above.

// TODO: LAPACK-class backend trait (native-LAPACK-via-FFI implementation
//       first), with capability negotiation baked in from the start
// TODO: least-squares solve (regression)
// TODO: symmetric eigendecomposition (covariance / PCA)
// TODO: general linear solve (portfolio weights / factor models)
// TODO: Cholesky (fast path for positive-definite covariance)
// TODO: expose all of the above as SQL-callable functions via `engine`, and
//       as Lua-callable functions via `compute-lua`, sharing arrow-lite
//       buffers with no copy in either direction
