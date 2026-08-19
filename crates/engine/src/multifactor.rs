//! Multi-factor regression state (#90): the anchored moment carrier
//! generalized to K factors, and the symmetric solve behind it.
//!
//! ## Why moments and not a factorization
//!
//! A sliding window's fit can be maintained two ways: carry a
//! factorization of the design matrix and up/downdate it as rows enter
//! and leave, or carry the **moments** (`XᵀX`, `Xᵀy`) and re-solve the
//! small system per frame. The research on #90 (2026-08-03) closed
//! this: downdating a Cholesky factor is an **ill-posed problem**, not
//! merely a delicate algorithm — when the departing row leaves the
//! window near-rank-deficient the digits needed were rounded away when
//! the row was added, and no algorithm holding only the factor and the
//! row can recover them (Stewart 1979; Pan 1993; Eldén–Park 1996).
//! Measurement agreed: factor downdating was the worst of six
//! candidates in every case tried, even with a periodic rebuild to
//! rescue it. Moments re-solved per frame cost `O(K²)` to maintain and
//! `O(K³)` to solve, which at these K is nothing.
//!
//! ## The three disciplines this inherits from the K ≤ 2 kernels
//!
//! Naive moment maintenance is itself a known trap — subtracting an
//! evicted row's outer product loses digits when the window mean
//! dwarfs its spread (Chan–Golub–LeVeque 1983), and the error grows
//! with *stream* length, not window length. PostgreSQL refuses float
//! inverse transition functions for exactly this reason. What makes it
//! sound is the discipline [`ShiftedMoments`](crate::table) already
//! uses at K ≤ 2, generalized here:
//!
//! 1. **Anchoring.** Every value enters and leaves as a deviation from
//!    an anchor taken from a *data row*, so the accumulated sums stay
//!    at the window's own scale even when the values sit at 1e12.
//!    Subtracting nearby representable values is near-exact; the
//!    measured effect is large — at offset 1e12 the anchored
//!    incremental path beat a per-frame solve centered on the
//!    computed mean by nine orders of magnitude, because that mean is
//!    itself a length-W summation carrying its own error.
//! 2. **Periodic rebuild.** The caller re-anchors and refolds the
//!    whole window every `w` steps, so add/remove rounding drift is
//!    bounded by one period and cannot accumulate down a column.
//! 3. **One solve for both schedules.** Per-frame recompute and the
//!    incremental sweep call the same [`FactorMoments::fit`],
//!    so the two paths cannot disagree about NULL semantics.
//!
//! ## The solve seam
//!
//! [`solve_spd`] is deliberately the only place a linear system is
//! solved. The Human ruled (F2(c), 2026-08-03) that TallyDB builds its
//! own `K × K` solve now rather than wait for MatLua, with the reopen
//! trigger recorded on #90: when MatLua lands, whichever
//! implementation measures better wins and the other adapts, then
//! MatLua is adopted. Keeping the solve behind one function is what
//! makes that swap a one-line change.

/// The relative pivot floor below which a window is declared singular.
///
/// Cholesky's pivot going non-positive catches an *exactly* degenerate
/// window — a factor that is constant across the frame, so its
/// centered column is all zeros. It does not catch two factors that
/// are near-duplicates, where the pivot stays positive but tiny and
/// the fit it produces is noise. Rejecting at `1e-12` of the pivot's
/// own starting scale draws the line at roughly `κ(X) ≈ 1e6`, beyond
/// which a normal-equations solve has under six correct digits and
/// should not be reported as a fit at all.
const SPD_PIVOT_FLOOR: f64 = 1e-12;

/// Solves `matrix · x = rhs` for a symmetric positive-definite `k × k`
/// `matrix`, writing the solution into `rhs`.
///
/// `matrix` is row-major and is **consumed**: the factorization is
/// written over its lower triangle. Only the lower triangle is read,
/// so a caller that filled both halves gets the same answer.
///
/// Returns `false` when the matrix is not usefully positive definite
/// (see [`SPD_PIVOT_FLOOR`]) — the singular-window signal, which the
/// window layer turns into a NULL for that frame rather than an error
/// for the whole query.
fn solve_spd(matrix: &mut [f64], rhs: &mut [f64], k: usize) -> bool {
    debug_assert_eq!(matrix.len(), k * k);
    debug_assert_eq!(rhs.len(), k);
    // The pivot floor is relative to each column's starting scale, so
    // the test means the same thing whether the factors are priced in
    // dollars or basis points.
    for j in 0..k {
        let floor = matrix[j * k + j] * SPD_PIVOT_FLOOR;
        let mut diagonal = matrix[j * k + j];
        for p in 0..j {
            diagonal -= matrix[j * k + p] * matrix[j * k + p];
        }
        let usable = diagonal.is_finite() && diagonal > 0.0 && diagonal > floor;
        if !usable {
            return false;
        }
        let pivot = diagonal.sqrt();
        matrix[j * k + j] = pivot;
        for i in (j + 1)..k {
            let mut off = matrix[i * k + j];
            for p in 0..j {
                off -= matrix[i * k + p] * matrix[j * k + p];
            }
            matrix[i * k + j] = off / pivot;
        }
    }
    // Forward substitution: L y = rhs.
    for i in 0..k {
        let mut value = rhs[i];
        for p in 0..i {
            value -= matrix[i * k + p] * rhs[p];
        }
        rhs[i] = value / matrix[i * k + i];
    }
    // Back substitution: Lᵀ x = y.
    for i in (0..k).rev() {
        let mut value = rhs[i];
        for p in (i + 1)..k {
            value -= matrix[p * k + i] * rhs[p];
        }
        rhs[i] = value / matrix[i * k + i];
    }
    rhs.iter().all(|value| value.is_finite())
}

/// The anchored moments of one window over `k` factors and a response.
///
/// Every sum is of deviations from `(anchor_x, anchor_y)`, a point
/// taken from a row of the data itself — see the module docs for why
/// that is load-bearing rather than a micro-optimization.
pub(crate) struct FactorMoments {
    k: usize,
    rows: f64,
    anchor_x: Vec<f64>,
    anchor_y: f64,
    /// `Σ d` where `d = x − anchor_x`.
    sum_x: Vec<f64>,
    /// `Σ e` where `e = y − anchor_y`.
    sum_y: f64,
    /// `Σ d dᵀ`, row-major `k × k`. Held whole rather than packed: at
    /// these K the extra half is a handful of flops, and the solve
    /// wants a full matrix anyway.
    cross_xx: Vec<f64>,
    /// `Σ d e`.
    cross_xy: Vec<f64>,
    /// `Σ e²` — carried only so the fit can report `R²` without a
    /// second pass over the window.
    cross_yy: f64,
    /// Reused across frames so a per-frame solve allocates nothing.
    scratch_matrix: Vec<f64>,
    deviation: Vec<f64>,
}

impl FactorMoments {
    /// An empty carrier for `k` factors, anchored at the origin until
    /// the first [`FactorMoments::refold`].
    pub(crate) fn new(k: usize) -> FactorMoments {
        FactorMoments {
            k,
            rows: 0.0,
            anchor_x: vec![0.0; k],
            anchor_y: 0.0,
            sum_x: vec![0.0; k],
            sum_y: 0.0,
            cross_xx: vec![0.0; k * k],
            cross_xy: vec![0.0; k],
            cross_yy: 0.0,
            scratch_matrix: vec![0.0; k * k],
            deviation: vec![0.0; k],
        }
    }

    /// Re-anchors on the window's last row and folds `lo..hi` fresh —
    /// the periodic rebuild that bounds add/remove drift to one
    /// period, and the only way the anchor ever moves.
    pub(crate) fn refold(&mut self, columns: &[&[f64]], y: &[f64], lo: usize, hi: usize) {
        debug_assert_eq!(columns.len(), self.k);
        self.rows = 0.0;
        self.sum_y = 0.0;
        self.cross_yy = 0.0;
        self.sum_x.iter_mut().for_each(|slot| *slot = 0.0);
        self.cross_xy.iter_mut().for_each(|slot| *slot = 0.0);
        self.cross_xx.iter_mut().for_each(|slot| *slot = 0.0);
        if hi <= lo {
            return;
        }
        let anchor_row = hi - 1;
        for (factor, column) in columns.iter().enumerate() {
            self.anchor_x[factor] = column[anchor_row];
        }
        self.anchor_y = y[anchor_row];
        for row in lo..hi {
            self.accumulate(columns, y, row, 1.0);
        }
    }

    /// Folds one row in — the entering observation of a slide.
    pub(crate) fn add(&mut self, columns: &[&[f64]], y: &[f64], row: usize) {
        self.accumulate(columns, y, row, 1.0);
    }

    /// Folds one row out — the leaving observation of a slide. Sound
    /// because every term is a deviation from the anchor, so the
    /// subtraction happens at the window's scale, not the data's.
    pub(crate) fn remove(&mut self, columns: &[&[f64]], y: &[f64], row: usize) {
        self.accumulate(columns, y, row, -1.0);
    }

    /// The shared body of [`add`](Self::add) and
    /// [`remove`](Self::remove): `sign` is `+1` entering, `−1`
    /// leaving.
    fn accumulate(&mut self, columns: &[&[f64]], y: &[f64], row: usize, sign: f64) {
        for (factor, column) in columns.iter().enumerate() {
            self.deviation[factor] = column[row] - self.anchor_x[factor];
        }
        let response = y[row] - self.anchor_y;
        self.rows += sign;
        self.sum_y += sign * response;
        self.cross_yy += sign * response * response;
        for i in 0..self.k {
            let d_i = self.deviation[i];
            self.sum_x[i] += sign * d_i;
            self.cross_xy[i] += sign * d_i * response;
            let base = i * self.k;
            for j in 0..self.k {
                self.cross_xx[base + j] += sign * d_i * self.deviation[j];
            }
        }
    }

    /// Writes `[intercept, β₁ … β_k]` into `out` and returns the fit's
    /// `R²`, or `None` when the window is singular (see [`solve_spd`])
    /// or holds too few rows to identify the fit.
    ///
    /// The system solved is the *centered* one — `Σdd' − n·m_d m_dᵀ`
    /// against `Σde − n·m_d m_e` — which is the anchored form of the
    /// normal equations and is what keeps a large common level out of
    /// the conditioning.
    ///
    /// `R²` comes from the same solve rather than a second pass: for
    /// an ordinary least-squares fit the explained sum of squares is
    /// `βᵀc` over the centered cross-products, so `R² = βᵀc / Σ(y−ȳ)²`.
    /// It is `NaN` when the response is constant across the window —
    /// there is no variance to explain, and the caller reports NULL
    /// rather than inventing a 0 or a 1.
    pub(crate) fn fit(&mut self, out: &mut [f64]) -> Option<f64> {
        debug_assert_eq!(out.len(), self.k + 1);
        // k + 1 parameters (intercept included) need k + 1 rows before
        // the fit is even determined.
        if self.rows < self.k as f64 + 1.0 {
            return None;
        }
        let rows = self.rows;
        let mean_y = self.sum_y / rows;
        for i in 0..self.k {
            let mean_i = self.sum_x[i] / rows;
            for j in 0..self.k {
                let mean_j = self.sum_x[j] / rows;
                self.scratch_matrix[i * self.k + j] =
                    self.cross_xx[i * self.k + j] - rows * mean_i * mean_j;
            }
            out[i + 1] = self.cross_xy[i] - rows * mean_i * mean_y;
        }
        let (intercept_slot, betas) = out.split_at_mut(1);
        // The centered right-hand side is consumed by the solve, so the
        // explained sum of squares is accumulated against a copy.
        let cross = self.deviation.as_mut_slice();
        cross.copy_from_slice(betas);
        if !solve_spd(&mut self.scratch_matrix, betas, self.k) {
            return None;
        }
        let explained: f64 = betas.iter().zip(cross.iter()).map(|(b, c)| b * c).sum();
        let total = self.cross_yy - rows * mean_y * mean_y;
        // ŷ = b₀ + x·β through the window means, undoing the anchor.
        let mut intercept = self.anchor_y + mean_y;
        for ((anchor, sum), beta) in self.anchor_x.iter().zip(&self.sum_x).zip(betas.iter()) {
            intercept -= (anchor + sum / rows) * beta;
        }
        intercept_slot[0] = intercept;
        if !intercept.is_finite() {
            return None;
        }
        Some(if total > 0.0 {
            explained / total
        } else {
            f64::NAN
        })
    }
}

/// Which scalar a [`MultiFactorRegression`] window reports.
///
/// A window function returns one `f64` per row, so a K-factor fit —
/// which produces a vector — has to name what it wants. Coefficients
/// are addressed by position: `0` is the intercept, `1..=k` the
/// factors in the order they were passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiFactorOutput {
    /// One coefficient by position (`0` = intercept).
    Coefficient(usize),
    /// The fit's explained fraction of the response's variance.
    R2,
}

/// Rolling least squares of a response on **K factors** (#90).
///
/// The K ≤ 2 kernels (`regr_slope` and friends) solve in closed form;
/// above two parameters there is no closed form, so this maintains the
/// window's anchored moments and solves the small symmetric system per
/// frame. Arguments follow the SQL convention the two-parameter
/// kernels already use — the **response first**, then the factors:
/// `fit(y, x1, x2, x3)`.
///
/// This has **no SQL name** (#77.1, ruled): multi-factor regression has
/// no standard spelling, so it reaches users through registration
/// rather than through the dialect.
///
/// A frame reports NULL when the fit is not identified — fewer rows
/// than parameters, a factor that is constant across the frame, or two
/// factors that are near-duplicates within it. One degenerate window
/// does not fail the query.
///
/// ```
/// use engine::{MultiFactorOutput, MultiFactorRegression, Table};
/// use arrow_lite::{ColumnType, Field, Schema};
/// use storage_lite::RowValue;
///
/// let schema = Schema::new(vec![
///     Field::new("ts", ColumnType::I64, false),
///     Field::new("y", ColumnType::F64, false),
///     Field::new("a", ColumnType::F64, false),
///     Field::new("b", ColumnType::F64, false),
/// ]);
/// let mut table = Table::new("t", schema, "ts").unwrap();
/// table.register_window(
///     "fit_b",
///     MultiFactorRegression::new(2, MultiFactorOutput::Coefficient(2)),
/// ).unwrap();
/// // y = 1 + 2a + 3b exactly, so the fit recovers b's coefficient.
/// for i in 0..8i64 {
///     let (a, b) = (i as f64, (i * i) as f64);
///     table.append(&[
///         RowValue::I64(i),
///         RowValue::F64(1.0 + 2.0 * a + 3.0 * b),
///         RowValue::F64(a),
///         RowValue::F64(b),
///     ]).unwrap();
/// }
/// let out = table.query(
///     "SELECT fit_b(y, a, b) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS c \
///      FROM t",
/// ).unwrap();
/// let last = out.batches.last().unwrap();
/// let arrow_lite::Column::Numeric(arrow_lite::NumericData::F64(column)) = &last.columns()[0]
/// else { panic!("f64") };
/// let final_row = column.values().as_slice()[last.num_rows() - 1];
/// assert!((final_row - 3.0).abs() < 1e-6, "{final_row}");
/// ```
pub struct MultiFactorRegression {
    factors: usize,
    output: MultiFactorOutput,
}

impl MultiFactorRegression {
    /// A kernel over `factors` factors reporting `output`.
    ///
    /// `factors` must be at least 1 and `output` must address a
    /// coefficient the fit actually has; both are checked here so a
    /// misconfigured kernel cannot be registered and then fail
    /// mid-query.
    pub fn new(factors: usize, output: MultiFactorOutput) -> MultiFactorRegression {
        assert!(factors >= 1, "a multi-factor fit needs at least one factor");
        if let MultiFactorOutput::Coefficient(slot) = output {
            assert!(
                slot <= factors,
                "coefficient {slot} is out of range for {factors} factors \
                 (0 is the intercept, 1..={factors} the factors)"
            );
        }
        MultiFactorRegression { factors, output }
    }
}

impl query_lite::WindowAggregate for MultiFactorRegression {
    fn arity(&self) -> usize {
        self.factors + 1 // the response, then the factors
    }

    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        let (y, columns) = args.split_first().ok_or("no arguments")?;
        let mut moments = FactorMoments::new(self.factors);
        moments.refold(columns, y, 0, y.len());
        let mut coefficients = vec![0.0; self.factors + 1];
        let Some(r_squared) = moments.fit(&mut coefficients) else {
            return Ok(None); // the fit is not identified: NULL
        };
        Ok(match self.output {
            MultiFactorOutput::Coefficient(slot) => Some(coefficients[slot]),
            MultiFactorOutput::R2 => r_squared.is_finite().then_some(r_squared),
        })
    }

    /// The incremental sweep — the point of #90.
    ///
    /// Consecutive `ROWS` frames differ by one row in and one row out,
    /// so the window's moments are maintained across the slide instead
    /// of refolded per frame: `O(K²)` per row rather than `O(W·K²)`.
    /// The accumulator is re-anchored and refolded every `window`
    /// steps, bounding add/remove drift to one period.
    ///
    /// Non-finite values break the sliding identity — `NaN − NaN` is
    /// `NaN`, so a poisoned row outlives its departure — so frames
    /// holding one fall back to the exact per-frame arithmetic of
    /// [`MultiFactorRegression::evaluate`](query_lite::WindowAggregate::evaluate), which is bit-identical to recompute, and the
    /// first clean frame afterwards rebuilds from scratch.
    ///
    /// Unbounded frames have no row leaving and nothing to gain, so
    /// they take the shared recompute path.
    fn evaluate_frames(
        &self,
        columns: &[&[f64]],
        preceding: Option<usize>,
    ) -> Result<Vec<Option<f64>>, String> {
        let Some(preceding) = preceding else {
            return query_lite::recompute_frames(self, columns, preceding);
        };
        let (y, factors) = columns.split_first().ok_or("no arguments")?;
        let rows = y.len();
        let window = preceding + 1;
        let poisoned: Vec<bool> = (0..rows)
            .map(|row| !y[row].is_finite() || factors.iter().any(|c| !c[row].is_finite()))
            .collect();

        let mut moments = FactorMoments::new(self.factors);
        let mut coefficients = vec![0.0; self.factors + 1];
        let mut out = Vec::with_capacity(rows);
        let mut frame = Vec::with_capacity(self.factors + 1);
        let mut low = 0usize;
        let mut since_rebuild = usize::MAX; // force the first fold
        let mut dirty = 0usize;

        for row in 0..rows {
            let lo = row.saturating_sub(preceding);
            let hi = row + 1;
            if since_rebuild >= window {
                moments.refold(factors, y, lo, hi);
                since_rebuild = 0;
                dirty = (lo..hi).filter(|&r| poisoned[r]).count();
            } else {
                moments.add(factors, y, row);
                if poisoned[row] {
                    dirty += 1;
                }
                for (stale, &bad) in (low..lo).zip(&poisoned[low..lo]) {
                    moments.remove(factors, y, stale);
                    if bad {
                        dirty -= 1;
                    }
                }
                since_rebuild += 1;
            }
            low = lo;

            if dirty > 0 {
                // Exact per-frame arithmetic; the accumulator is
                // poisoned, so the next clean frame must rebuild.
                frame.clear();
                frame.push(&y[lo..hi]);
                frame.extend(factors.iter().map(|column| &column[lo..hi]));
                out.push(self.evaluate(&frame)?);
                since_rebuild = usize::MAX;
                continue;
            }
            out.push(match moments.fit(&mut coefficients) {
                None => None, // the fit is not identified: NULL
                Some(r_squared) => match self.output {
                    MultiFactorOutput::Coefficient(slot) => Some(coefficients[slot]),
                    MultiFactorOutput::R2 => r_squared.is_finite().then_some(r_squared),
                },
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Columns of a design whose relationship to `y` is exact, so any
    /// correct solve recovers the coefficients to rounding.
    fn exact_design(rows: usize, offset: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let truth = vec![0.5, -1.25, 3.0];
        let mut columns: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(rows)).collect();
        let mut y = Vec::with_capacity(rows);
        for row in 0..rows {
            let t = row as f64;
            let values = [
                offset + t,
                offset + (t * 0.37).sin() * 4.0,
                offset + (t % 7.0) - 3.0,
            ];
            let mut response = 11.0;
            for (value, coefficient) in values.iter().zip(&truth) {
                response += value * coefficient;
            }
            for (column, value) in columns.iter_mut().zip(values) {
                column.push(value);
            }
            y.push(response);
        }
        (columns, y, truth)
    }

    fn borrow(columns: &[Vec<f64>]) -> Vec<&[f64]> {
        columns.iter().map(|column| column.as_slice()).collect()
    }

    #[test]
    fn the_solve_answers_a_known_system_and_refuses_a_singular_one() {
        // [[4,1,0],[1,3,1],[0,1,2]] x = [1,2,3] — SPD, hand-checkable
        // by substitution back into the original rows.
        let mut matrix = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let original = matrix.clone();
        let mut rhs = vec![1.0, 2.0, 3.0];
        assert!(solve_spd(&mut matrix, &mut rhs, 3));
        for (row, expected) in [1.0, 2.0, 3.0].iter().enumerate() {
            let product: f64 = (0..3).map(|c| original[row * 3 + c] * rhs[c]).sum();
            assert!(
                (product - expected).abs() < 1e-12,
                "row {row}: {product} vs {expected}"
            );
        }
        // The guard trips: a duplicated factor makes the Gram matrix
        // singular, and the pivot floor must catch it rather than
        // return noise. (Sabotage leg — without SPD_PIVOT_FLOOR the
        // second pivot is a rounding artifact, not a refusal.)
        let mut singular = vec![2.0, 2.0, 2.0, 2.0];
        let mut rhs = vec![1.0, 1.0];
        assert!(!solve_spd(&mut singular, &mut rhs, 2));
        // And a negative diagonal is refused outright.
        let mut indefinite = vec![-1.0, 0.0, 0.0, 1.0];
        let mut rhs = vec![1.0, 1.0];
        assert!(!solve_spd(&mut indefinite, &mut rhs, 2));
    }

    #[test]
    fn the_moments_recover_an_exact_relationship() {
        let (columns, y, truth) = exact_design(64, 0.0);
        let columns = borrow(&columns);
        let mut moments = FactorMoments::new(3);
        moments.refold(&columns, &y, 0, 64);
        let mut out = vec![0.0; 4];
        assert!(moments.fit(&mut out).is_some());
        assert!((out[0] - 11.0).abs() < 1e-9, "intercept {}", out[0]);
        for (slot, expected) in truth.iter().enumerate() {
            assert!(
                (out[slot + 1] - expected).abs() < 1e-9,
                "beta {slot}: {} vs {expected}",
                out[slot + 1]
            );
        }
    }

    #[test]
    fn a_slide_agrees_with_refolding_the_same_window() {
        let (columns, y, _) = exact_design(200, 0.0);
        let columns = borrow(&columns);
        let window = 32;
        let mut slid = FactorMoments::new(3);
        slid.refold(&columns, &y, 0, window);
        let mut fresh = FactorMoments::new(3);
        let (mut a, mut b) = (vec![0.0; 4], vec![0.0; 4]);
        for start in 1..(200 - window) {
            slid.add(&columns, &y, start + window - 1);
            slid.remove(&columns, &y, start - 1);
            fresh.refold(&columns, &y, start, start + window);
            assert!(slid.fit(&mut a).is_some());
            assert!(fresh.fit(&mut b).is_some());
            for (slot, (slid_value, fresh_value)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (slid_value - fresh_value).abs() < 1e-6 * fresh_value.abs().max(1.0),
                    "slide {start}, slot {slot}: {slid_value} vs {fresh_value}"
                );
            }
        }
    }

    #[test]
    fn a_removed_row_leaves_no_trace() {
        let (columns, y, _) = exact_design(64, 0.0);
        let columns = borrow(&columns);
        let mut moments = FactorMoments::new(3);
        moments.refold(&columns, &y, 0, 32);
        let mut before = vec![0.0; 4];
        assert!(moments.fit(&mut before).is_some());
        moments.add(&columns, &y, 40);
        moments.remove(&columns, &y, 40);
        let mut after = vec![0.0; 4];
        assert!(moments.fit(&mut after).is_some());
        for (slot, (b, a)) in before.iter().zip(&after).enumerate() {
            assert!((b - a).abs() < 1e-9, "slot {slot}: {b} vs {a}");
        }
    }

    /// The textbook alternative this module exists to avoid: build the
    /// uncentered Gram of `[1, x₁ … x_k]` and solve it directly. No
    /// anchor, no centering — the form Chan–Golub–LeVeque showed
    /// carries a κ² cancellation once the values sit far from zero.
    fn naive_uncentered(columns: &[&[f64]], y: &[f64], rows: usize) -> Option<Vec<f64>> {
        let width = columns.len() + 1;
        let mut gram = vec![0.0; width * width];
        let mut rhs = vec![0.0; width];
        let mut row_values = vec![1.0; width];
        for row in 0..rows {
            for (slot, column) in columns.iter().enumerate() {
                row_values[slot + 1] = column[row];
            }
            for i in 0..width {
                rhs[i] += row_values[i] * y[row];
                for j in 0..width {
                    gram[i * width + j] += row_values[i] * row_values[j];
                }
            }
        }
        solve_spd(&mut gram, &mut rhs, width).then_some(rhs)
    }

    #[test]
    fn anchoring_is_what_survives_a_huge_common_level() {
        // Every factor rides an offset of 1e12. The response then lands
        // near 2.25e12, where an f64 ulp is ~5e-4 against roughly 60 of
        // total signal in y — so the synthetic data itself holds only
        // about five digits about the relationship, and no solve can
        // beat that. What this test discriminates is not absolute
        // accuracy but WHOSE arithmetic survives: the anchored moments
        // land within the data's own precision, while the textbook
        // uncentered Gram on the identical inputs either refuses or is
        // wrong by orders of magnitude. Remove the anchoring and this
        // test fails alone.
        let (columns, y, truth) = exact_design(64, 1e12);
        let columns = borrow(&columns);
        let mut moments = FactorMoments::new(3);
        moments.refold(&columns, &y, 0, 64);
        let mut out = vec![0.0; 4];
        assert!(moments.fit(&mut out).is_some());
        let mut anchored_worst: f64 = 0.0;
        for (slot, expected) in truth.iter().enumerate() {
            let error = (out[slot + 1] - expected).abs() / expected.abs();
            anchored_worst = anchored_worst.max(error);
        }
        assert!(
            anchored_worst < 1e-4,
            "anchored moments lost the data's own precision: {anchored_worst:e}"
        );
        match naive_uncentered(&columns, &y, 64) {
            None => {} // refused outright — the honest failure
            Some(naive) => {
                let mut naive_worst: f64 = 0.0;
                for (slot, expected) in truth.iter().enumerate() {
                    let error = (naive[slot + 1] - expected).abs() / expected.abs();
                    naive_worst = naive_worst.max(error);
                }
                assert!(
                    naive_worst > anchored_worst * 100.0,
                    "the uncentered solve was supposed to lose badly here, \
                     but managed {naive_worst:e} against the anchored {anchored_worst:e} \
                     — the discriminating case has stopped discriminating"
                );
            }
        }
        // At no offset the two agree closely: the anchoring costs
        // nothing when there is nothing to protect against.
        let (columns, y, truth) = exact_design(64, 0.0);
        let columns = borrow(&columns);
        let naive = naive_uncentered(&columns, &y, 64).expect("well-conditioned at offset 0");
        for (slot, expected) in truth.iter().enumerate() {
            assert!((naive[slot + 1] - expected).abs() < 1e-6);
        }
    }

    /// The A/B the whole issue rests on: the incremental sweep against
    /// the same kernel's per-frame recompute, frame for frame. They run
    /// different arithmetic — one slides moments across the window, the
    /// other refolds each frame from scratch — so agreement here is
    /// evidence, not tautology.
    #[test]
    fn the_incremental_sweep_tracks_per_frame_recompute() {
        use query_lite::WindowAggregate;
        let (columns, y, _) = exact_design(400, 1e6);
        let mut args: Vec<&[f64]> = vec![&y];
        args.extend(columns.iter().map(|column| column.as_slice()));
        // Windows with real degrees of freedom. A window holding no
        // more rows than parameters is interpolation, not regression:
        // the system is square and maximally sensitive, so the two
        // paths' agreement there is bounded by the problem's own
        // conditioning rather than by either method. Measured worst
        // relative disagreement on this offset-1e6 corpus, by degrees
        // of freedom (2026-08-03, this container): 0 dof 1.4e-5,
        // 1 dof 1.2e-6, 5 dof 1.2e-7, 13 dof 8.0e-10, 60 dof 1.3e-9 —
        // and at offset 0, 8.3e-14 at 60 dof. The bound below sits an
        // order above the worst case it covers and an order below the
        // near-square case it excludes, so it still discriminates.
        for preceding in [8usize, 16, 63] {
            for output in [
                MultiFactorOutput::Coefficient(0),
                MultiFactorOutput::Coefficient(2),
                MultiFactorOutput::R2,
            ] {
                let kernel = MultiFactorRegression::new(3, output);
                let swept = kernel.evaluate_frames(&args, Some(preceding)).unwrap();
                let recomputed = query_lite::recompute_frames(&kernel, &args, Some(preceding))
                    .expect("recompute is the reference");
                assert_eq!(swept.len(), recomputed.len());
                for (row, (a, b)) in swept.iter().zip(&recomputed).enumerate() {
                    match (a, b) {
                        (None, None) => {}
                        (Some(a), Some(b)) => assert!(
                            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                            "preceding {preceding}, {output:?}, row {row}: {a} vs {b}"
                        ),
                        _ => panic!(
                            "preceding {preceding}, {output:?}, row {row}: \
                             definedness diverged — {a:?} vs {b:?}"
                        ),
                    }
                }
            }
        }
    }

    /// Non-finite rows break the sliding identity, so those frames must
    /// fall back to exact per-frame arithmetic and the accumulator must
    /// recover once the poison leaves the window.
    #[test]
    fn non_finite_rows_agree_with_recompute_and_the_sweep_recovers() {
        use query_lite::WindowAggregate;
        let (mut columns, mut y, _) = exact_design(120, 0.0);
        y[40] = f64::NAN;
        columns[1][41] = f64::INFINITY;
        columns[2][42] = f64::NEG_INFINITY;
        let mut args: Vec<&[f64]> = vec![&y];
        args.extend(columns.iter().map(|column| column.as_slice()));
        let kernel = MultiFactorRegression::new(3, MultiFactorOutput::Coefficient(1));
        for preceding in [3usize, 7] {
            let swept = kernel.evaluate_frames(&args, Some(preceding)).unwrap();
            let recomputed = query_lite::recompute_frames(&kernel, &args, Some(preceding)).unwrap();
            for (row, (a, b)) in swept.iter().zip(&recomputed).enumerate() {
                assert_eq!(
                    a.is_some(),
                    b.is_some(),
                    "preceding {preceding}, row {row}: definedness diverged"
                );
                if let (Some(a), Some(b)) = (a, b) {
                    assert!(
                        (a - b).abs() <= 1e-9 * b.abs().max(1.0) || (a.is_nan() && b.is_nan()),
                        "preceding {preceding}, row {row}: {a} vs {b}"
                    );
                }
            }
            // Well past the poison the sweep is exact again — the
            // rebuild cleared it rather than carrying NaN forward.
            let tail = swept[60..].iter().flatten().count();
            assert!(tail > 0, "the sweep never recovered after the poison");
        }
    }

    #[test]
    fn a_degenerate_window_is_refused_not_guessed() {
        // A factor that is constant across the frame: its centered
        // column is all zeros, so the fit is not identified and the
        // frame must decline rather than report noise.
        let rows = 32;
        let flat = vec![7.0; rows];
        let moving: Vec<f64> = (0..rows).map(|row| row as f64).collect();
        let y: Vec<f64> = (0..rows).map(|row| 3.0 + 2.0 * row as f64).collect();
        let columns: Vec<&[f64]> = vec![&moving, &flat];
        let mut moments = FactorMoments::new(2);
        moments.refold(&columns, &y, 0, rows);
        let mut out = vec![0.0; 3];
        assert!(moments.fit(&mut out).is_none());
        // Too few rows is likewise a refusal, not a guess.
        let mut sparse = FactorMoments::new(2);
        sparse.refold(&columns, &y, 0, 2);
        assert!(sparse.fit(&mut out).is_none());
    }
}
