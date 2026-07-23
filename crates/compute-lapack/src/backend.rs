//! The LAPACK-class backend seam: capability negotiation plus the curated
//! operations, native implementation first.
//!
//! Every operation is reachable only through [`LapackBackend`], and every
//! backend must answer [`LapackBackend::supports`] honestly — "unavailable
//! on this backend" is a first-class, queryable result, never a panic.
//! That negotiation exists for the planned WASM build, which will come up
//! with BLAS-class ops working and LAPACK-class ops degraded until a
//! LAPACK-in-WASM layer ships; `engine` routes an unsupported op to a
//! clean error.
//!
//! ## Calling convention
//!
//! Batch, never per-row: whole windows and matrices per call, in the
//! column-major layout LAPACK expects. Inputs are borrowed slices over
//! `arrow-lite` buffers or gathered design matrices — the backend copies
//! what LAPACK would overwrite (its routines work in place), so callers'
//! buffers are never clobbered.

use std::fmt;

/// The curated LAPACK-class operations (see the crate docs for why this
/// list is closed): each entry exists because a named workflow needs it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// Least-squares solve — rolling regression. Implemented (M1).
    LeastSquares,
    /// Symmetric eigendecomposition — covariance / PCA. M2.
    SymmetricEigen,
    /// General linear solve — portfolio weights / factor models. M2.
    LinearSolve,
    /// Cholesky — positive-definite covariance fast path. M2.
    Cholesky,
}

/// Why a compute call failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ComputeError {
    /// The backend cannot run this operation (capability negotiation —
    /// expected on partial backends, e.g. a future WASM build).
    Unsupported(Op),
    /// The inputs do not describe a valid problem (dimension mismatch,
    /// empty system).
    InvalidInput(String),
    /// LAPACK itself reported failure (e.g. a rank-deficient system under
    /// QR); carries the routine's `info` code.
    Lapack { routine: &'static str, info: i32 },
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeError::Unsupported(op) => {
                write!(f, "operation {op:?} is unavailable on this backend")
            }
            ComputeError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            ComputeError::Lapack { routine, info } => {
                write!(f, "LAPACK {routine} failed with info = {info}")
            }
        }
    }
}

impl std::error::Error for ComputeError {}

/// A column-major matrix view: `m` rows, `n` columns, values in one
/// contiguous slice with column `j` at `values[j * m .. (j + 1) * m]` —
/// exactly LAPACK's layout, so a gathered design matrix passes straight
/// through.
#[derive(Clone, Copy, Debug)]
pub struct ColMajor<'a> {
    values: &'a [f64],
    m: usize,
    n: usize,
}

impl<'a> ColMajor<'a> {
    /// Wraps a column-major slice.
    ///
    /// # Panics
    /// If `values.len() != m * n` (a programmer error, not a data error).
    pub fn new(values: &'a [f64], m: usize, n: usize) -> Self {
        assert_eq!(
            values.len(),
            m.checked_mul(n).expect("matrix size overflow"),
            "column-major slice length must be m * n"
        );
        ColMajor { values, m, n }
    }

    /// Row count.
    pub fn num_rows(&self) -> usize {
        self.m
    }

    /// Column count.
    pub fn num_cols(&self) -> usize {
        self.n
    }

    /// The backing slice.
    pub fn values(&self) -> &'a [f64] {
        self.values
    }
}

/// The backend seam: capability negotiation + the curated operations.
pub trait LapackBackend {
    /// Whether this backend can run `op`. Callers route an unsupported op
    /// to a clean error; they never find out by panicking.
    fn supports(&self, op: Op) -> bool;

    /// Solves the least-squares problem `min ‖A x − b‖₂` for full-rank
    /// `A` (m ≥ n), returning the n coefficients.
    ///
    /// The provisional inner routine is QR (`dgels`) per open decision
    /// #20 — swappable behind this seam until the op's numerical behavior
    /// is golden-locked at M2.
    fn least_squares(&self, a: ColMajor<'_>, b: &[f64]) -> Result<Vec<f64>, ComputeError>;
}

/// The native backend: system LAPACK via FFI, linked as-is.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeLapack;

// LAPACK's Fortran ABI: every scalar by pointer; matrices column-major.
extern "C" {
    fn dgels_(
        trans: *const u8,
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

impl LapackBackend for NativeLapack {
    fn supports(&self, op: Op) -> bool {
        // Only what is actually implemented; the M2 ops flip to true when
        // their routines land.
        matches!(op, Op::LeastSquares)
    }

    fn least_squares(&self, a: ColMajor<'_>, b: &[f64]) -> Result<Vec<f64>, ComputeError> {
        let (m, n) = (a.num_rows(), a.num_cols());
        if n == 0 || m < n {
            return Err(ComputeError::InvalidInput(format!(
                "least squares needs m >= n >= 1, got m = {m}, n = {n}"
            )));
        }
        if b.len() != m {
            return Err(ComputeError::InvalidInput(format!(
                "b has {} rows, A has {m}",
                b.len()
            )));
        }
        let (m_i, n_i) = (as_lapack_dim(m)?, as_lapack_dim(n)?);
        // dgels overwrites both A (with its QR factorization) and b (with
        // the solution): work on copies so callers' buffers survive.
        let mut a_work = a.values().to_vec();
        let mut b_work = b.to_vec();
        let mut info = 0i32;
        // Workspace query: lwork = -1 asks dgels for its optimal size.
        let mut work_query = [0.0f64];
        // SAFETY: all pointers are valid for the extents dgels reads or
        // writes (a: m*n, b: ldb = m >= n, work: 1 during the query);
        // dimensions are positive i32s checked above.
        unsafe {
            dgels_(
                b"N".as_ptr(),
                &m_i,
                &n_i,
                &1,
                a_work.as_mut_ptr(),
                &m_i,
                b_work.as_mut_ptr(),
                &m_i,
                work_query.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dgels",
                info,
            });
        }
        let lwork = work_query[0] as i32;
        let mut work = vec![0.0f64; lwork.max(1) as usize];
        // SAFETY: as above, with the queried workspace size.
        unsafe {
            dgels_(
                b"N".as_ptr(),
                &m_i,
                &n_i,
                &1,
                a_work.as_mut_ptr(),
                &m_i,
                b_work.as_mut_ptr(),
                &m_i,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dgels",
                info,
            });
        }
        b_work.truncate(n);
        Ok(b_work)
    }
}

/// Converts a dimension to LAPACK's i32, rejecting overflow.
fn as_lapack_dim(value: usize) -> Result<i32, ComputeError> {
    i32::try_from(value)
        .map_err(|_| ComputeError::InvalidInput(format!("dimension {value} exceeds LAPACK's i32")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// `max |aᵢ|` scaled tolerance for float comparisons.
    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "coefficient {index}: {a} vs {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn capability_negotiation_is_honest() {
        let backend = NativeLapack;
        assert!(backend.supports(Op::LeastSquares));
        // The M2 ops are declared but not implemented — a backend must say
        // so, not pretend.
        assert!(!backend.supports(Op::SymmetricEigen));
        assert!(!backend.supports(Op::LinearSolve));
        assert!(!backend.supports(Op::Cholesky));
    }

    #[test]
    fn exact_fit_is_recovered() {
        // y = 1 + 2x through three collinear points: residual is zero, so
        // the solution is exact.
        let a = vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0]; // [1 | x], column-major
        let b = vec![1.0, 3.0, 5.0];
        let x = NativeLapack
            .least_squares(ColMajor::new(&a, 3, 2), &b)
            .expect("solves");
        assert_close(&x, &[1.0, 2.0], 1e-12);
    }

    #[test]
    fn matches_numpy_golden_two_regressors() {
        // Golden from np.linalg.lstsq (NumPy 2.4.6, 2026-07-23, this
        // container): A = [1 | x1 | x2] with x1 = 0..5, x2 as below.
        let a = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // intercept
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // x1
            1.0, 0.0, 2.0, 1.0, 3.0, 2.0, // x2
        ];
        let b = vec![1.1, 3.2, 5.3, 6.9, 9.4, 11.1];
        let x = NativeLapack
            .least_squares(ColMajor::new(&a, 6, 3), &b)
            .expect("solves");
        assert_close(
            &x,
            &[1.0666666666666633, 1.9500000000000008, 0.15000000000000094],
            1e-10,
        );
    }

    #[test]
    fn matches_numpy_golden_no_intercept() {
        // Golden from np.linalg.lstsq (NumPy 2.4.6, same run as above).
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.1, 3.9, 6.2, 7.8];
        let x = NativeLapack
            .least_squares(ColMajor::new(&a, 4, 1), &b)
            .expect("solves");
        assert_close(&x, &[1.9899999999999998], 1e-12);
    }

    #[test]
    fn rejects_bad_shapes() {
        let backend = NativeLapack;
        // Underdetermined: m < n.
        let err = backend
            .least_squares(ColMajor::new(&[1.0, 2.0], 1, 2), &[1.0])
            .expect_err("m < n");
        assert!(matches!(err, ComputeError::InvalidInput(_)));
        // b length disagrees with A's rows.
        let err = backend
            .least_squares(ColMajor::new(&[1.0, 2.0], 2, 1), &[1.0])
            .expect_err("b mismatch");
        assert!(matches!(err, ComputeError::InvalidInput(_)));
    }

    #[test]
    fn rank_deficient_reports_lapack_error_not_garbage() {
        // Two identical columns: QR's triangular factor is singular. dgels
        // must report it (info > 0), not hand back arbitrary numbers.
        let a = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = NativeLapack.least_squares(ColMajor::new(&a, 3, 2), &b);
        assert!(matches!(
            result,
            Err(ComputeError::Lapack {
                routine: "dgels",
                ..
            })
        ));
    }

    #[test]
    fn callers_buffers_survive_the_call() {
        let a = vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let b = vec![1.0, 3.0, 5.0];
        let (a_before, b_before) = (a.clone(), b.clone());
        NativeLapack
            .least_squares(ColMajor::new(&a, 3, 2), &b)
            .expect("solves");
        assert_eq!(a, a_before);
        assert_eq!(b, b_before);
    }

    proptest! {
        /// The normal-equations invariant: at the least-squares optimum
        /// the residual is orthogonal to A's column space, Aᵀ(b − Ax̂) ≈ 0.
        /// Holds for every full-rank system, no oracle needed.
        #[test]
        fn residual_is_orthogonal_to_columns(
            m in 3usize..24,
            n in 1usize..3,
            seed_values in prop::collection::vec(-100.0f64..100.0, 24 * 2 + 24),
        ) {
            let n = n.min(m - 1);
            // Column-major A: an intercept column plus (n - 1) generated
            // columns, diagonally boosted to keep the system full-rank
            // and well-conditioned.
            let mut a = vec![1.0; m];
            for j in 1..n {
                for i in 0..m {
                    let mut v = seed_values[(j - 1) * m + i];
                    if i == j {
                        v += 1_000.0;
                    }
                    a.push(v);
                }
            }
            let b = &seed_values[seed_values.len() - m..];
            let x = NativeLapack
                .least_squares(ColMajor::new(&a, m, n), b)
                .expect("well-conditioned system solves");
            // r = b − A x̂; check max |Aᵀ r| against a scale-aware bound.
            let residual: Vec<f64> = (0..m)
                .map(|i| b[i] - (0..n).map(|j| a[j * m + i] * x[j]).sum::<f64>())
                .collect();
            let scale: f64 = a.iter().fold(1.0f64, |acc, v| acc.max(v.abs()))
                * b.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
            for j in 0..n {
                let dot: f64 = (0..m).map(|i| a[j * m + i] * residual[i]).sum();
                prop_assert!(
                    dot.abs() <= scale * (m as f64) * 1e-10,
                    "column {j} not orthogonal to residual: {dot}"
                );
            }
        }
    }
}
