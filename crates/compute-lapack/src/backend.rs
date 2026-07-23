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
    /// Decision #20 (2026-07-23): this op runs **both routines behind one
    /// seam** — QR (`dgels`) as the fast path, SVD (`dgelsd`) as the
    /// fallback, because QR cannot be trusted to flag near-singular
    /// designs (observed at M1: garbage coefficients with `info = 0` on a
    /// constant-x window). The switch criterion, set by the M2.6 A/B: the
    /// QR result is kept when `dgels` succeeds **and** every coefficient
    /// is finite; a `dgels` failure or a non-finite coefficient retries
    /// through SVD, whose minimum-norm answer is defined even
    /// rank-deficient. Exactly-degenerate windows that QR cannot detect
    /// at all (constant x, `info = 0`, finite garbage) remain the
    /// caller's guard — `engine`'s zero-variance check. The A/B behind
    /// the criterion (Observed 2026-07-23, release, `measure_20`, this
    /// machine): SVD costs 2.01× QR over 2,855 corpus windows, and the
    /// worst QR-vs-SVD fitted-value drift on near-degenerate designs
    /// (x-spread down to 1e-10) is 9.5e-7 — QR's *fits* stay sound even
    /// where its coefficients wobble, so the cheap path stays the
    /// default and the fallback fires only on provable trouble.
    fn least_squares(&self, a: ColMajor<'_>, b: &[f64]) -> Result<Vec<f64>, ComputeError>;

    /// Eigenvalues (ascending) and column-major eigenvectors of a
    /// symmetric `n × n` matrix (`dsyev`; the lower triangle is read).
    fn symmetric_eigen(&self, a: ColMajor<'_>) -> Result<(Vec<f64>, Vec<f64>), ComputeError> {
        let _ = a;
        Err(ComputeError::Unsupported(Op::SymmetricEigen))
    }

    /// Solves the square system `A x = b` (`dgesv`, partial pivoting).
    fn linear_solve(&self, a: ColMajor<'_>, b: &[f64]) -> Result<Vec<f64>, ComputeError> {
        let _ = (a, b);
        Err(ComputeError::Unsupported(Op::LinearSolve))
    }

    /// The lower Cholesky factor `L` (column-major, upper triangle
    /// zeroed) of a symmetric positive-definite matrix (`dpotrf`). A
    /// non-positive-definite input surfaces as the routine's `info > 0`.
    fn cholesky(&self, a: ColMajor<'_>) -> Result<Vec<f64>, ComputeError> {
        let _ = a;
        Err(ComputeError::Unsupported(Op::Cholesky))
    }
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
    fn dgelsd_(
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        s: *mut f64,
        rcond: *const f64,
        rank: *mut i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
    fn dsyev_(
        jobz: *const u8,
        uplo: *const u8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgesv_(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dpotrf_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
}

/// Checks that `a` is square and returns its side as a LAPACK dimension.
fn square_side(a: &ColMajor<'_>, what: &str) -> Result<(usize, i32), ComputeError> {
    let n = a.num_rows();
    if n == 0 || a.num_cols() != n {
        return Err(ComputeError::InvalidInput(format!(
            "{what} needs a square non-empty matrix, got {} x {}",
            a.num_rows(),
            a.num_cols()
        )));
    }
    Ok((n, as_lapack_dim(n)?))
}

impl LapackBackend for NativeLapack {
    fn supports(&self, op: Op) -> bool {
        matches!(
            op,
            Op::LeastSquares | Op::SymmetricEigen | Op::LinearSolve | Op::Cholesky
        )
    }

    fn symmetric_eigen(&self, a: ColMajor<'_>) -> Result<(Vec<f64>, Vec<f64>), ComputeError> {
        let (n, n_i) = square_side(&a, "symmetric eigen")?;
        // dsyev overwrites A with the eigenvectors: work on a copy.
        let mut a_work = a.values().to_vec();
        let mut eigenvalues = vec![0.0f64; n];
        let mut info = 0i32;
        let mut work_query = [0.0f64];
        // SAFETY: pointers valid for the extents dsyev touches (a: n*n,
        // w: n, work: 1 during the query); dimensions checked above.
        unsafe {
            dsyev_(
                b"V".as_ptr(),
                b"L".as_ptr(),
                &n_i,
                a_work.as_mut_ptr(),
                &n_i,
                eigenvalues.as_mut_ptr(),
                work_query.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        let lwork = if info == 0 { work_query[0] as i32 } else { 0 };
        let mut work = vec![0.0f64; lwork.max(1) as usize];
        if info == 0 {
            // SAFETY: as above, with the queried workspace.
            unsafe {
                dsyev_(
                    b"V".as_ptr(),
                    b"L".as_ptr(),
                    &n_i,
                    a_work.as_mut_ptr(),
                    &n_i,
                    eigenvalues.as_mut_ptr(),
                    work.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );
            }
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dsyev",
                info,
            });
        }
        Ok((eigenvalues, a_work))
    }

    fn linear_solve(&self, a: ColMajor<'_>, b: &[f64]) -> Result<Vec<f64>, ComputeError> {
        let (n, n_i) = square_side(&a, "linear solve")?;
        if b.len() != n {
            return Err(ComputeError::InvalidInput(format!(
                "b has {} rows, A has {n}",
                b.len()
            )));
        }
        let mut a_work = a.values().to_vec();
        let mut b_work = b.to_vec();
        let mut pivots = vec![0i32; n];
        let mut info = 0i32;
        // SAFETY: pointers valid for dgesv's extents (a: n*n, ipiv: n,
        // b: n); dimensions checked above.
        unsafe {
            dgesv_(
                &n_i,
                &1,
                a_work.as_mut_ptr(),
                &n_i,
                pivots.as_mut_ptr(),
                b_work.as_mut_ptr(),
                &n_i,
                &mut info,
            );
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dgesv",
                info,
            });
        }
        Ok(b_work)
    }

    fn cholesky(&self, a: ColMajor<'_>) -> Result<Vec<f64>, ComputeError> {
        let (n, n_i) = square_side(&a, "Cholesky")?;
        let mut a_work = a.values().to_vec();
        let mut info = 0i32;
        // SAFETY: pointers valid for dpotrf's extents (a: n*n).
        unsafe {
            dpotrf_(b"L".as_ptr(), &n_i, a_work.as_mut_ptr(), &n_i, &mut info);
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dpotrf",
                info,
            });
        }
        // dpotrf leaves the strict upper triangle untouched; zero it so
        // the result is exactly L.
        for column in 1..n {
            for row in 0..column {
                a_work[column * n + row] = 0.0;
            }
        }
        Ok(a_work)
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
            // QR refused (rank deficiency it happened to detect): the
            // SVD path is defined even there.
            return self.least_squares_svd(a, b);
        }
        b_work.truncate(n);
        // The #20 switch criterion: keep QR only when every coefficient
        // is finite; otherwise retry through SVD.
        if b_work.iter().any(|value| !value.is_finite()) {
            return self.least_squares_svd(a, b);
        }
        Ok(b_work)
    }
}

impl NativeLapack {
    /// The SVD half of the one least-squares op (#20): `dgelsd`, minimum
    /// norm, defined for any rank. Slower than QR; reached through the
    /// documented switch criterion (or a QR failure).
    fn least_squares_svd(&self, a: ColMajor<'_>, b: &[f64]) -> Result<Vec<f64>, ComputeError> {
        let (m, n) = (a.num_rows(), a.num_cols());
        let (m_i, n_i) = (as_lapack_dim(m)?, as_lapack_dim(n)?);
        let ldb = m.max(n);
        let ldb_i = as_lapack_dim(ldb)?;
        let mut a_work = a.values().to_vec();
        let mut b_work = vec![0.0f64; ldb];
        b_work[..m].copy_from_slice(b);
        let mut singular_values = vec![0.0f64; m.min(n)];
        let rcond = -1.0f64; // machine-precision rank threshold
        let mut rank = 0i32;
        let mut info = 0i32;
        let mut work_query = [0.0f64];
        let mut iwork_query = [0i32];
        // SAFETY: pointers valid for dgelsd's extents (a: m*n, b: ldb,
        // s: min(m,n), work: 1 and iwork: 1 during the query).
        unsafe {
            dgelsd_(
                &m_i,
                &n_i,
                &1,
                a_work.as_mut_ptr(),
                &m_i,
                b_work.as_mut_ptr(),
                &ldb_i,
                singular_values.as_mut_ptr(),
                &rcond,
                &mut rank,
                work_query.as_mut_ptr(),
                &-1,
                iwork_query.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dgelsd",
                info,
            });
        }
        let lwork = work_query[0] as i32;
        let mut work = vec![0.0f64; lwork.max(1) as usize];
        let mut iwork = vec![0i32; (iwork_query[0].max(1)) as usize];
        // SAFETY: as above, with the queried workspace sizes.
        unsafe {
            dgelsd_(
                &m_i,
                &n_i,
                &1,
                a_work.as_mut_ptr(),
                &m_i,
                b_work.as_mut_ptr(),
                &ldb_i,
                singular_values.as_mut_ptr(),
                &rcond,
                &mut rank,
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(ComputeError::Lapack {
                routine: "dgelsd",
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
        // The native backend implements the full curated set (M2.6);
        // a backend that doesn't must say so, not pretend — pinned by
        // the trait's Unsupported defaults.
        struct Partial;
        impl LapackBackend for Partial {
            fn supports(&self, op: Op) -> bool {
                op == Op::LeastSquares
            }
            fn least_squares(&self, _: ColMajor<'_>, _: &[f64]) -> Result<Vec<f64>, ComputeError> {
                unimplemented!("not under test")
            }
        }
        let identity = [1.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            Partial.symmetric_eigen(ColMajor::new(&identity, 2, 2)),
            Err(ComputeError::Unsupported(Op::SymmetricEigen))
        ));
        assert!(!Partial.supports(Op::Cholesky));
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
    fn rank_deficient_least_squares_is_finite_and_residual_optimal() {
        // Two identical columns. Before #20's fallback this errored (or,
        // worse, QR could hand back finite garbage with info = 0); the
        // one op now always returns a finite solution whose residual is
        // the least-squares optimum — for this fit of b on [1 | 1], the
        // best constant is mean(b), so the coefficients sum to 2.
        let a = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let solution = NativeLapack
            .least_squares(ColMajor::new(&a, 3, 2), &b)
            .expect("defined via the SVD path");
        assert!(solution.iter().all(|v| v.is_finite()));
        assert!(
            (solution[0] + solution[1] - 2.0).abs() < 1e-9,
            "{solution:?}"
        );
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

#[cfg(test)]
mod m26_tests {
    use super::*;

    /// `A x` for column-major `A` (n × n) — the test-side reference.
    fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
        (0..n)
            .map(|row| (0..n).map(|col| a[col * n + row] * x[col]).sum())
            .collect()
    }

    #[test]
    fn eigen_satisfies_its_defining_identity() {
        // A symmetric 4x4 with distinct eigenvalues; the test verifies
        // the *identity* ‖A v − λ v‖ ≈ 0 — self-verifying, no fixture.
        let n = 4;
        let a = vec![
            4.0, 1.0, 0.5, 0.25, //
            1.0, 3.0, 1.0, 0.5, //
            0.5, 1.0, 2.0, 1.0, //
            0.25, 0.5, 1.0, 1.0,
        ];
        let backend = NativeLapack;
        let (eigenvalues, eigenvectors) = backend.symmetric_eigen(ColMajor::new(&a, n, n)).unwrap();
        assert!(eigenvalues.windows(2).all(|pair| pair[0] <= pair[1]));
        // Trace identity.
        let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
        assert!((eigenvalues.iter().sum::<f64>() - trace).abs() < 1e-10);
        for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
            let vector = &eigenvectors[index * n..(index + 1) * n];
            let av = matvec(&a, n, vector);
            for (row, &value) in av.iter().enumerate() {
                assert!(
                    (value - eigenvalue * vector[row]).abs() < 1e-10,
                    "eigenpair {index} violates A v = λ v at row {row}"
                );
            }
        }
    }

    #[test]
    fn linear_solve_satisfies_its_residual() {
        let n = 5;
        // Diagonally dominant (hence nonsingular), deterministic.
        let mut a = vec![0.0f64; n * n];
        for col in 0..n {
            for row in 0..n {
                a[col * n + row] = if row == col {
                    10.0 + row as f64
                } else {
                    ((row * 3 + col * 7) % 5) as f64 * 0.25 - 0.5
                };
            }
        }
        let b: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 - 2.0).collect();
        let backend = NativeLapack;
        let x = backend.linear_solve(ColMajor::new(&a, n, n), &b).unwrap();
        let ax = matvec(&a, n, &x);
        for (row, (&lhs, &rhs)) in ax.iter().zip(&b).enumerate() {
            assert!((lhs - rhs).abs() < 1e-10, "residual at row {row}");
        }
        // Singular system: loud, never garbage.
        let singular = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
        assert!(matches!(
            backend.linear_solve(ColMajor::new(&singular, 2, 2), &[1.0, 2.0]),
            Err(ComputeError::Lapack { routine: "dgesv", info }) if info > 0
        ));
    }

    #[test]
    fn cholesky_reconstructs_its_input() {
        // A = M Mᵀ + I is symmetric positive definite by construction.
        let n = 4;
        let m_values = vec![
            1.0, 0.5, -0.25, 2.0, //
            0.0, 1.5, 0.75, -1.0, //
            0.5, 0.0, 2.0, 0.25, //
            1.0, 1.0, 0.0, 1.5,
        ];
        let mut a = vec![0.0f64; n * n];
        for col in 0..n {
            for row in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += m_values[k * n + row] * m_values[k * n + col];
                }
                a[col * n + row] = sum + if row == col { 1.0 } else { 0.0 };
            }
        }
        let backend = NativeLapack;
        let l = backend.cholesky(ColMajor::new(&a, n, n)).unwrap();
        // Upper triangle is exactly zero.
        for col in 1..n {
            for row in 0..col {
                assert_eq!(l[col * n + row], 0.0);
            }
        }
        // ‖L Lᵀ − A‖∞ ≈ 0 — the defining identity.
        for row in 0..n {
            for col in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[k * n + row] * l[k * n + col];
                }
                assert!(
                    (sum - a[col * n + row]).abs() < 1e-10,
                    "L Lᵀ differs from A at ({row}, {col})"
                );
            }
        }
        // Not positive definite: loud.
        let indefinite = vec![1.0, 2.0, 2.0, 1.0];
        assert!(matches!(
            backend.cholesky(ColMajor::new(&indefinite, 2, 2)),
            Err(ComputeError::Lapack { routine: "dpotrf", info }) if info > 0
        ));
    }

    #[test]
    fn least_squares_qr_and_svd_agree_when_both_are_defined() {
        // Well-conditioned: the public op (QR path) and the SVD path
        // must produce the same answer.
        let m = 12;
        let mut design = Vec::with_capacity(m * 2);
        design.resize(m, 1.0);
        design.extend((0..m).map(|i| i as f64 * 0.5));
        let b: Vec<f64> = (0..m).map(|i| 3.0 * (i as f64 * 0.5) + 7.0).collect();
        let backend = NativeLapack;
        let qr = backend
            .least_squares(ColMajor::new(&design, m, 2), &b)
            .unwrap();
        let svd = backend
            .least_squares_svd(ColMajor::new(&design, m, 2), &b)
            .unwrap();
        assert!((qr[0] - 7.0).abs() < 1e-9 && (qr[1] - 3.0).abs() < 1e-9);
        assert!((qr[0] - svd[0]).abs() < 1e-9 && (qr[1] - svd[1]).abs() < 1e-9);
    }

    #[test]
    fn rank_deficient_least_squares_stays_valid_and_svd_is_minimum_norm() {
        // Two identical columns, exactly fittable. QR without pivoting
        // may return a finite *valid* particular solution here (residual
        // optimal, info = 0) — the criterion rightly leaves it alone;
        // what the one op guarantees is a finite, residual-optimal
        // answer. The SVD path itself must return the minimum-norm
        // member: the true slope split evenly.
        let m = 8;
        let x: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let mut design = Vec::with_capacity(m * 2);
        design.extend_from_slice(&x);
        design.extend_from_slice(&x);
        let b: Vec<f64> = x.iter().map(|&v| 4.0 * v).collect();
        let backend = NativeLapack;
        let solution = backend
            .least_squares(ColMajor::new(&design, m, 2), &b)
            .unwrap();
        assert!(solution.iter().all(|v| v.is_finite()));
        // Residual-optimal: the combined slope reproduces b exactly.
        assert!(
            (solution[0] + solution[1] - 4.0).abs() < 1e-9,
            "{solution:?}"
        );
        let minimum_norm = backend
            .least_squares_svd(ColMajor::new(&design, m, 2), &b)
            .unwrap();
        assert!((minimum_norm[0] - 2.0).abs() < 1e-9, "{minimum_norm:?}");
        assert!((minimum_norm[1] - 2.0).abs() < 1e-9, "{minimum_norm:?}");
    }

    #[test]
    fn every_m2_op_reports_supported() {
        let backend = NativeLapack;
        for op in [
            Op::LeastSquares,
            Op::SymmetricEigen,
            Op::LinearSolve,
            Op::Cholesky,
        ] {
            assert!(backend.supports(op), "{op:?}");
        }
    }
}

#[cfg(test)]
mod measure_20 {
    use super::*;

    fn design(x: &[f64]) -> Vec<f64> {
        let mut design = Vec::with_capacity(x.len() * 2);
        design.resize(x.len(), 1.0);
        design.extend_from_slice(x);
        design
    }

    /// The #20 A/B (a measurement recorded for the ruling's switch
    /// criterion, not a decision). Run explicitly, in release mode:
    ///
    /// ```text
    /// cargo test -p compute-lapack --release measure_20 -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn qr_vs_svd_on_corpus_windows() {
        let backend = NativeLapack;
        let rows = corpus::Spec::telemetry(20_000, 20).generate();
        let windows: Vec<(Vec<f64>, Vec<f64>)> = rows
            .windows(20)
            .step_by(7)
            .filter_map(|window| {
                let pairs: Vec<(f64, f64)> = window
                    .iter()
                    .filter_map(|row| row.aux.map(|aux| (row.value, aux)))
                    .collect();
                (pairs.len() >= 4).then(|| {
                    (
                        pairs.iter().map(|(x, _)| *x).collect(),
                        pairs.iter().map(|(_, y)| *y).collect(),
                    )
                })
            })
            .collect();
        // Timing: the public op (QR path) vs the SVD path, same windows.
        let start = std::time::Instant::now();
        for (x, y) in &windows {
            backend
                .least_squares(ColMajor::new(&design(x), x.len(), 2), y)
                .unwrap();
        }
        let qr_time = start.elapsed();
        let start = std::time::Instant::now();
        for (x, y) in &windows {
            backend
                .least_squares_svd(ColMajor::new(&design(x), x.len(), 2), y)
                .unwrap();
        }
        let svd_time = start.elapsed();
        // Accuracy where it matters: near-degenerate designs (x almost
        // constant), where QR without pivoting is the untrusted party.
        // The reference is the SVD answer; report the worst QR drift.
        let mut worst_drift = 0.0f64;
        for scale in [1e-6, 1e-8, 1e-10] {
            for seed in 0..50u64 {
                let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) + 1;
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                };
                let x: Vec<f64> = (0..20).map(|_| 100.0 + next() * scale).collect();
                let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0 + next() * 0.01).collect();
                let a = design(&x);
                let qr = backend.least_squares(ColMajor::new(&a, 20, 2), &y);
                let svd = backend.least_squares_svd(ColMajor::new(&a, 20, 2), &y);
                if let (Ok(qr), Ok(svd)) = (qr, svd) {
                    // Compare predictions at the data mean (coefficient
                    // drift alone overstates: collinear designs make
                    // coefficients unstable while fits agree).
                    let mean = x.iter().sum::<f64>() / 20.0;
                    let drift = ((qr[0] + qr[1] * mean) - (svd[0] + svd[1] * mean)).abs();
                    worst_drift = worst_drift.max(drift);
                }
            }
        }
        println!(
            "#20 A/B: {} corpus windows; QR {qr_time:.2?}, SVD {svd_time:.2?} \
             (SVD/QR = {:.2}x); worst QR-vs-SVD fitted-value drift on \
             near-degenerate designs (spread 1e-6..1e-10): {worst_drift:.3e}",
            windows.len(),
            svd_time.as_secs_f64() / qr_time.as_secs_f64(),
        );
    }
}
