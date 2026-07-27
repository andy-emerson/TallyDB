//! The BLAS backend seam: capability negotiation plus the
//! multiplication-class primitives, native implementation first.
//!
//! The shape every compute backend here follows — `supports` answers
//! honestly, inputs are borrowed column-major slices, and it never
//! clobbers a caller's buffer (these routines write only their output
//! argument). System BLAS is linked as-is; the executor does **not**
//! call these yet — wiring BLAS into query inner loops is
//! profiling-gated, per the crate docs, and lands only with a number
//! that asks for it.

use std::fmt;

/// The multiplication-class operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlasOp {
    /// `xᵀ y` — dot product (`ddot`).
    Dot,
    /// `A x` — matrix–vector multiply (`dgemv`).
    MatVec,
    /// `A B` — matrix–matrix multiply (`dgemm`).
    MatMat,
}

/// Why a BLAS call failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BlasError {
    /// The backend cannot run this operation.
    Unsupported(BlasOp),
    /// The inputs do not describe a valid problem.
    InvalidInput(String),
}

impl fmt::Display for BlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlasError::Unsupported(op) => {
                write!(f, "operation {op:?} is unavailable on this backend")
            }
            BlasError::InvalidInput(message) => write!(f, "invalid input: {message}"),
        }
    }
}

impl std::error::Error for BlasError {}

/// The backend seam.
pub trait BlasBackend {
    /// Whether this backend can run `op`.
    fn supports(&self, op: BlasOp) -> bool;

    /// `xᵀ y` over equal-length slices.
    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, BlasError>;

    /// `A x` for column-major `A` (`m × n` in one slice, column `j` at
    /// `a[j*m .. (j+1)*m]`) and `x` of length `n`; returns `m` values.
    fn matvec(&self, a: &[f64], m: usize, n: usize, x: &[f64]) -> Result<Vec<f64>, BlasError>;

    /// `A B` for column-major `A` (`m × k`) and `B` (`k × n`); returns
    /// the column-major `m × n` product.
    #[allow(clippy::many_single_char_names)]
    fn matmat(
        &self,
        a: &[f64],
        m: usize,
        k: usize,
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>, BlasError>;
}

/// The native backend: system BLAS via FFI, linked as-is.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBlas;

// BLAS's Fortran ABI: scalars by pointer, matrices column-major.
extern "C" {
    fn ddot_(
        n: *const i32,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
    ) -> f64;
    #[allow(clippy::too_many_arguments)]
    fn dgemv_(
        trans: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn dgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

fn as_blas_dim(value: usize) -> Result<i32, BlasError> {
    i32::try_from(value)
        .map_err(|_| BlasError::InvalidInput(format!("dimension {value} exceeds BLAS's i32")))
}

impl BlasBackend for NativeBlas {
    fn supports(&self, op: BlasOp) -> bool {
        matches!(op, BlasOp::Dot | BlasOp::MatVec | BlasOp::MatMat)
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, BlasError> {
        if x.len() != y.len() {
            return Err(BlasError::InvalidInput(format!(
                "dot needs equal lengths, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        let n = as_blas_dim(x.len())?;
        // SAFETY: both slices are valid for n reads at stride 1.
        Ok(unsafe { ddot_(&n, x.as_ptr(), &1, y.as_ptr(), &1) })
    }

    fn matvec(&self, a: &[f64], m: usize, n: usize, x: &[f64]) -> Result<Vec<f64>, BlasError> {
        if a.len() != m * n || x.len() != n || m == 0 || n == 0 {
            return Err(BlasError::InvalidInput(format!(
                "matvec needs a: m*n = {}*{} and x: {n}, got a: {} and x: {}",
                m,
                n,
                a.len(),
                x.len()
            )));
        }
        let (m_i, n_i) = (as_blas_dim(m)?, as_blas_dim(n)?);
        let mut result = vec![0.0f64; m];
        // SAFETY: a is m*n, x is n, result is m — the extents dgemv
        // reads and writes for trans = 'N', alpha = 1, beta = 0.
        unsafe {
            dgemv_(
                b"N".as_ptr(),
                &m_i,
                &n_i,
                &1.0,
                a.as_ptr(),
                &m_i,
                x.as_ptr(),
                &1,
                &0.0,
                result.as_mut_ptr(),
                &1,
            );
        }
        Ok(result)
    }

    fn matmat(
        &self,
        a: &[f64],
        m: usize,
        k: usize,
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>, BlasError> {
        if a.len() != m * k || b.len() != k * n || m == 0 || k == 0 || n == 0 {
            return Err(BlasError::InvalidInput(format!(
                "matmat needs a: {m}*{k} and b: {k}*{n}, got a: {} and b: {}",
                a.len(),
                b.len()
            )));
        }
        let (m_i, k_i, n_i) = (as_blas_dim(m)?, as_blas_dim(k)?, as_blas_dim(n)?);
        let mut result = vec![0.0f64; m * n];
        // SAFETY: a is m*k, b is k*n, result is m*n — the extents dgemm
        // touches for 'N','N', alpha = 1, beta = 0.
        unsafe {
            dgemm_(
                b"N".as_ptr(),
                b"N".as_ptr(),
                &m_i,
                &n_i,
                &k_i,
                &1.0,
                a.as_ptr(),
                &m_i,
                b.as_ptr(),
                &k_i,
                &0.0,
                result.as_mut_ptr(),
                &m_i,
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_hand_computation() {
        let backend = NativeBlas;
        assert_eq!(
            backend.dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]).unwrap(),
            1.0 * 4.0 - 2.0 * 5.0 + 3.0 * 6.0
        );
        assert!(backend.dot(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn matvec_matches_hand_computation() {
        // A = [[1, 3], [2, 4]] column-major; A [5, 6]ᵀ = [23, 34].
        let backend = NativeBlas;
        let result = backend
            .matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0])
            .unwrap();
        assert_eq!(result, [23.0, 34.0]);
    }

    #[test]
    fn matmat_matches_matvec_column_by_column() {
        // The defining identity: (A B).column(j) = A · B.column(j).
        let backend = NativeBlas;
        let (m, k, n) = (3, 4, 2);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.5 - 2.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * -0.25 + 1.0).collect();
        let product = backend.matmat(&a, m, k, &b, n).unwrap();
        for column in 0..n {
            let expected = backend
                .matvec(&a, m, k, &b[column * k..(column + 1) * k])
                .unwrap();
            assert_eq!(&product[column * m..(column + 1) * m], expected.as_slice());
        }
    }

    #[test]
    fn capability_negotiation_is_honest() {
        let backend = NativeBlas;
        assert!(backend.supports(BlasOp::Dot));
        assert!(backend.supports(BlasOp::MatVec));
        assert!(backend.supports(BlasOp::MatMat));
    }
}

#[cfg(test)]
mod measure_dot {
    //! Is the FFI call to BLAS `ddot` worth it at TallyDB's shapes?
    //!
    //! The engine's `dot` is called once per window — 64 elements is the
    //! benchmark's shape, and even a large window is thousands, not
    //! millions. At that size a BLAS call is mostly call: argument
    //! marshalling by pointer, the library's own dispatch, and no chance
    //! to amortize any of it. A plain Rust loop has none of that, and
    //! LLVM vectorizes it.
    //!
    //! Two things are measured, because both matter for the decision:
    //! **time** across window sizes, and **bit-agreement** — a Rust loop
    //! sums strictly left to right, while a BLAS kernel may block and
    //! reassociate, so the two need not agree to the last bit. Whether
    //! they do bears directly on native/WASM reproducibility.
    //!
    //! Read the bit-agreement result with care: it is a statement about
    //! *the BLAS that happens to be linked*, not about BLAS. Reference
    //! netlib BLAS accumulates in a plain loop and matches; a blocked,
    //! SIMD-kernel implementation (OpenBLAS, MKL) reassociates and need
    //! not. That variability across implementations is itself the point
    //! when the goal is a reproducible answer on every target.
    //!
    //! Run explicitly, in release:
    //!
    //! ```text
    //! cargo test -p compute-blas --release measure_dot -- --ignored --nocapture
    //! ```

    use super::*;

    /// The obvious Rust dot product: strict left-to-right accumulation.
    fn rust_dot(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y).map(|(&a, &b)| a * b).sum()
    }

    /// Deterministic values in [-1, 1).
    fn sample(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn measure_dot_rust_vs_blas() {
        use std::hint::black_box;
        let backend = NativeBlas;
        println!(
            "dot product: Rust loop vs BLAS ddot through FFI \
             (per-call time; ratio > 1 favors Rust)"
        );
        for len in [16usize, 64, 256, 1_024, 4_096, 65_536] {
            let x = sample(len, 0x2545_F491_4F6C_DD1D);
            let y = sample(len, 0x9E37_79B9_7F4A_7C15);
            // Enough repetitions that the timer resolution is irrelevant
            // at the small sizes the engine actually uses.
            let rounds = (1 << 22) / len.max(1);

            let start = std::time::Instant::now();
            let mut rust_result = 0.0;
            for _ in 0..rounds {
                rust_result = black_box(rust_dot(black_box(&x), black_box(&y)));
            }
            let rust_time = start.elapsed().as_secs_f64() / rounds as f64;

            let start = std::time::Instant::now();
            let mut blas_result = 0.0;
            for _ in 0..rounds {
                blas_result = black_box(backend.dot(black_box(&x), black_box(&y)).unwrap());
            }
            let blas_time = start.elapsed().as_secs_f64() / rounds as f64;

            let agree = rust_result.to_bits() == blas_result.to_bits();
            let relative = if blas_result == 0.0 {
                0.0
            } else {
                ((rust_result - blas_result) / blas_result).abs()
            };
            println!(
                "  len {len:>6}:  Rust {:>9.1}ns   BLAS {:>9.1}ns   \
                 BLAS/Rust {:>5.2}x   bit-identical {agree:<5}  rel {relative:.1e}",
                rust_time * 1e9,
                blas_time * 1e9,
                blas_time / rust_time,
            );
        }
    }
}
