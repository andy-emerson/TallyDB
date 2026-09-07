//! The linear-algebra backend seam: capability negotiation plus the
//! multiplication-class primitives, pure Rust throughout.
//!
//! The shape every compute backend here follows — `supports` answers
//! honestly, inputs are borrowed column-major slices, and it never
//! clobbers a caller's buffer (these routines write only their output
//! argument). The executor does **not** call these yet — wiring them
//! into query inner loops is profiling-gated, per the crate docs, and
//! lands only with a number that asks for it.
//!
//! ## Why `dot` is a plain loop and the matrix ops are faer
//! Measured at TallyDB's shapes (see `measure_kernels` below): at window
//! scale (≤ 64 elements) a strict left-to-right Rust loop beats every
//! library kernel — there is nothing to amortize — and from a few
//! hundred elements up faer's SIMD wins the dot by 2–4×. But the loop
//! has a property no SIMD kernel offers: its summation order is fixed by
//! the source, so the result is bit-identical on every CPU and every
//! target, native or WASM. `dot` is the one op scripts call per window
//! today, where the loop is also the *fastest* option — so it takes the
//! portable form. The matrix ops have no closed-form rival, are not on
//! any per-window path, and win 4–6× from faer's blocked kernels — so
//! they take faer, accepting that a runtime-dispatched kernel's rounding
//! may differ across CPU generations (the same trade every optimized
//! BLAS makes; the M4 portability standard will decide how much that
//! matters, and the seam localizes the answer here).

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use std::fmt;

/// The multiplication-class operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinalgOp {
    /// `xᵀ y` — dot product.
    Dot,
    /// `A x` — matrix–vector multiply.
    MatVec,
    /// `A B` — matrix–matrix multiply.
    MatMat,
}

/// Why a linear-algebra call failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LinalgError {
    /// The backend cannot run this operation.
    Unsupported(LinalgOp),
    /// The inputs do not describe a valid problem.
    InvalidInput(String),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::Unsupported(op) => {
                write!(f, "operation {op:?} is unavailable on this backend")
            }
            LinalgError::InvalidInput(message) => write!(f, "invalid input: {message}"),
        }
    }
}

impl std::error::Error for LinalgError {}

/// The backend seam.
pub trait LinalgBackend {
    /// Whether this backend can run `op`.
    fn supports(&self, op: LinalgOp) -> bool;

    /// `xᵀ y` over equal-length slices.
    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, LinalgError>;

    /// `A x` for column-major `A` (`m × n` in one slice, column `j` at
    /// `a[j*m .. (j+1)*m]`) and `x` of length `n`; returns `m` values.
    fn matvec(&self, a: &[f64], m: usize, n: usize, x: &[f64]) -> Result<Vec<f64>, LinalgError>;

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
    ) -> Result<Vec<f64>, LinalgError>;
}

/// The pure-Rust backend: a strict left-to-right loop for `dot`
/// (bit-identical on every target), faer's blocked kernels for the
/// matrix ops. See the module docs for why the split.
#[derive(Clone, Copy, Debug, Default)]
pub struct RustLinalg;

impl LinalgBackend for RustLinalg {
    fn supports(&self, op: LinalgOp) -> bool {
        matches!(op, LinalgOp::Dot | LinalgOp::MatVec | LinalgOp::MatMat)
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, LinalgError> {
        if x.len() != y.len() {
            return Err(LinalgError::InvalidInput(format!(
                "dot needs equal lengths, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        // Strict left-to-right accumulation: the order is fixed by the
        // source, so the answer is the same on every CPU and target.
        let mut acc = 0.0f64;
        for (&a, &b) in x.iter().zip(y) {
            acc += a * b;
        }
        Ok(acc)
    }

    fn matvec(&self, a: &[f64], m: usize, n: usize, x: &[f64]) -> Result<Vec<f64>, LinalgError> {
        if a.len() != m * n || x.len() != n || m == 0 || n == 0 {
            return Err(LinalgError::InvalidInput(format!(
                "matvec needs a: m*n = {}*{} and x: {n}, got a: {} and x: {}",
                m,
                n,
                a.len(),
                x.len()
            )));
        }
        // A vector is an n × 1 matrix; one product routine serves both.
        self.matmat(a, m, n, x, 1)
    }

    fn matmat(
        &self,
        a: &[f64],
        m: usize,
        k: usize,
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>, LinalgError> {
        if a.len() != m * k || b.len() != k * n || m == 0 || k == 0 || n == 0 {
            return Err(LinalgError::InvalidInput(format!(
                "matmat needs a: {m}*{k} and b: {k}*{n}, got a: {} and b: {}",
                a.len(),
                b.len()
            )));
        }
        let mut result = vec![0.0f64; m * n];
        matmul(
            MatMut::from_column_major_slice_mut(&mut result, m, n),
            Accum::Replace,
            MatRef::from_column_major_slice(a, m, k),
            MatRef::from_column_major_slice(b, k, n),
            1.0,
            Par::Seq, // per-call work this small never wants a thread pool
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_hand_computation() {
        let backend = RustLinalg;
        assert_eq!(
            backend.dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]).unwrap(),
            1.0 * 4.0 - 2.0 * 5.0 + 3.0 * 6.0
        );
        assert!(backend.dot(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn matvec_matches_hand_computation() {
        // A = [[1, 3], [2, 4]] column-major; A [5, 6]ᵀ = [23, 34].
        let backend = RustLinalg;
        let result = backend
            .matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0])
            .unwrap();
        assert_eq!(result, [23.0, 34.0]);
    }

    #[test]
    fn matmat_matches_matvec_column_by_column() {
        // The defining identity: (A B).column(j) = A · B.column(j).
        let backend = RustLinalg;
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
    fn matmat_matches_the_naive_triple_loop() {
        // faer may block and reassociate, but every entry is a sum of at
        // most k rounded products; with values this benign any summation
        // order lands within a few ulps of the naive triple loop, which
        // is an independent reference, not shared machinery.
        let backend = RustLinalg;
        let (m, k, n) = (5, 7, 4);
        let a: Vec<f64> = (0..m * k).map(|i| ((i * 37 % 23) as f64) - 11.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| ((i * 61 % 19) as f64) * 0.5).collect();
        let product = backend.matmat(&a, m, k, &b, n).unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f64;
                for t in 0..k {
                    expected += a[t * m + i] * b[j * k + t];
                }
                let got = product[j * m + i];
                assert!(
                    (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn capability_negotiation_is_honest() {
        let backend = RustLinalg;
        assert!(backend.supports(LinalgOp::Dot));
        assert!(backend.supports(LinalgOp::MatVec));
        assert!(backend.supports(LinalgOp::MatMat));
    }
}

#[cfg(test)]
mod measure_kernels {
    //! The executable evidence for this backend's split: a plain loop
    //! for `dot`, faer for the matrix ops.
    //!
    //! The engine's `dot` is called once per window — 64 elements is the
    //! benchmark's shape, and even a large window is thousands, not
    //! millions. At that size any library kernel is mostly dispatch;
    //! the loop has none, and LLVM vectorizes it. faer's SIMD only pulls
    //! ahead from a few hundred elements up. For the Gram-shaped matrix
    //! products the blocked kernels win outright at every size measured.
    //!
    //! Run explicitly, in release:
    //!
    //! ```text
    //! cargo test -p compute-linalg --release measure_kernels -- --ignored --nocapture
    //! ```

    use super::*;
    use std::hint::black_box;

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

    fn per_call<F: FnMut()>(mut f: F, rounds: usize) -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            f();
        }
        start.elapsed().as_secs_f64() / rounds as f64
    }

    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn measure_dot_loop_vs_faer() {
        use faer::col::ColRef;
        let backend = RustLinalg;
        println!("dot: strict loop (shipped) vs faer SIMD (ratio > 1 favors the loop)");
        for len in [16usize, 64, 256, 1_024, 4_096, 65_536] {
            let x = sample(len, 0x2545_F491_4F6C_DD1D);
            let y = sample(len, 0x9E37_79B9_7F4A_7C15);
            let rounds = (1 << 22) / len.max(1);

            let loop_time = per_call(
                || {
                    black_box(backend.dot(black_box(&x), black_box(&y)).unwrap());
                },
                rounds,
            );
            let faer_time = per_call(
                || {
                    let xr = ColRef::from_slice(black_box(x.as_slice()));
                    let yr = ColRef::from_slice(black_box(y.as_slice()));
                    black_box(xr.transpose() * yr);
                },
                rounds,
            );
            println!(
                "  len {len:>6}:  loop {:>9.1}ns   faer {:>9.1}ns   faer/loop {:>5.2}x",
                loop_time * 1e9,
                faer_time * 1e9,
                faer_time / loop_time,
            );
        }
    }

    #[test]
    #[ignore = "measurement — run explicitly in release mode"]
    fn measure_matmat_naive_vs_faer() {
        let backend = RustLinalg;
        println!("Gram AᵀA-shaped product, 64 rows (ratio > 1 favors faer)");
        for k in [2usize, 4, 8, 16] {
            let rows = 64usize;
            // Aᵀ (k × rows) times A (rows × k), both from one sample.
            let a = sample(rows * k, 0x2545_F491_4F6C_DD1D);
            let mut a_t = vec![0.0f64; k * rows];
            for i in 0..rows {
                for j in 0..k {
                    a_t[i * k + j] = a[j * rows + i];
                }
            }
            let rounds = 1 << 14;

            let naive_time = per_call(
                || {
                    let mut out = vec![0.0f64; k * k];
                    for i in 0..k {
                        for j in 0..k {
                            let mut acc = 0.0f64;
                            for t in 0..rows {
                                acc += a[i * rows + t] * a[j * rows + t];
                            }
                            out[j * k + i] = acc;
                        }
                    }
                    black_box(out);
                },
                rounds,
            );
            let faer_time = per_call(
                || {
                    black_box(
                        backend
                            .matmat(black_box(&a_t), k, rows, black_box(&a), k)
                            .unwrap(),
                    );
                },
                rounds,
            );
            println!(
                "  k {k:>2}:  naive {:>9.1}ns   faer {:>9.1}ns   naive/faer {:>5.2}x",
                naive_time * 1e9,
                faer_time * 1e9,
                naive_time / faer_time,
            );
        }
    }
}
