//! `corpus` — the seeded synthetic data the test plan measures and
//! diffs against (DESIGN.md, *The corpus*).
//!
//! Generators produce the workload shape TallyDB is built for — ordered
//! `i64` timestamps, low-cardinality keys, `f64` values — with the two
//! knobs the design names as parameters: **disorder fraction** (late
//! arrivals that break perfect ordering) and **null density**. Every
//! generator is a pure function of its [`Spec`]: same spec, same rows,
//! on every platform — which is what lets measurements cite a corpus
//! instead of a lost one-off dataset.
//!
//! This crate is dev-only infrastructure (`publish = false`, no
//! dependencies): the engine never links it. It grows two ways, per the
//! design: new capabilities add case families, and every closed bug adds
//! the case that would have caught it.
//!
//! The named presets are the corpus's case families so far:
//!
//! - [`Spec::ticks`] — irregular trade-like arrivals: bursty gaps,
//!   per-key price walks, no nulls.
//! - [`Spec::telemetry`] — regular sensor cadence with jitter, slowly
//!   drifting values, occasional nulls and late arrivals.

/// One generated row. Keys are small integers; [`key_label`] renders
/// the display form consumers intern or hand to an oracle.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Row {
    /// The ordering key: roughly sorted across the generated sequence.
    pub ts: i64,
    /// Which key this row belongs to, in `0..spec.key_cardinality`.
    pub key: u32,
    /// The primary value: a per-key random walk.
    pub value: f64,
    /// A second, nullable value (null with probability
    /// `spec.null_density`), linearly related to `value` plus noise —
    /// so regressions over the corpus have signal to recover.
    pub aux: Option<f64>,
}

/// The display label for key `key` (`"K000"`, `"K001"`, …).
pub fn key_label(key: u32) -> String {
    format!("K{key:03}")
}

/// A generator specification: the corpus is a pure function of this.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Spec {
    /// Rows to generate.
    pub rows: usize,
    /// RNG seed; every stream derives from it deterministically.
    pub seed: u64,
    /// Distinct keys (the low-cardinality assumption made concrete).
    pub key_cardinality: u32,
    /// First timestamp.
    pub start: i64,
    /// Mean inter-arrival gap between consecutive rows.
    pub cadence: i64,
    /// Uniform jitter half-width applied to each gap (`gap ∈ cadence ±
    /// jitter`, floored at 0 — ties are legal, "roughly sorted" allows
    /// them).
    pub jitter: i64,
    /// Fraction of rows delivered late: each such row swaps behind the
    /// row that follows it, the local disorder of real ingest.
    pub disorder_fraction: f64,
    /// Probability that `aux` is null on any given row.
    pub null_density: f64,
}

impl Spec {
    /// Trade-tick shape: irregular bursty gaps, 32 symbols, fully
    /// ordered, no nulls.
    pub fn ticks(rows: usize, seed: u64) -> Spec {
        Spec {
            rows,
            seed,
            key_cardinality: 32,
            start: 1_700_000_000_000_000_000, // a plausible epoch-ns anchor
            cadence: 1_000_000,               // ~1ms mean spacing
            jitter: 999_999,                  // near-fully irregular
            disorder_fraction: 0.0,
            null_density: 0.0,
        }
    }

    /// Sensor-telemetry shape: steady 1s cadence with small jitter,
    /// 8 sensors, occasional nulls, occasional late arrivals.
    pub fn telemetry(rows: usize, seed: u64) -> Spec {
        Spec {
            rows,
            seed,
            key_cardinality: 8,
            start: 1_700_000_000_000_000_000,
            cadence: 1_000_000_000, // 1s
            jitter: 5_000_000,      // ±5ms
            disorder_fraction: 0.01,
            null_density: 0.02,
        }
    }

    /// Generates this spec's rows.
    pub fn generate(&self) -> Vec<Row> {
        assert!(self.key_cardinality > 0, "at least one key");
        let mut rng = SplitMix64(self.seed);
        // Per-key walk state, seeded apart so keys are decorrelated.
        let mut walks: Vec<f64> = (0..self.key_cardinality)
            .map(|_| 100.0 * (rng.next_f64() + 0.5))
            .collect();
        let mut ts = self.start;
        let mut rows = Vec::with_capacity(self.rows);
        for index in 0..self.rows {
            if index > 0 {
                let jitter = if self.jitter > 0 {
                    rng.next_range(2 * self.jitter as u64 + 1) as i64 - self.jitter
                } else {
                    0
                };
                ts += (self.cadence + jitter).max(0);
            }
            let key = rng.next_range(u64::from(self.key_cardinality)) as u32;
            let walk = &mut walks[key as usize];
            *walk += (rng.next_f64() - 0.5) * 2.0;
            let value = *walk;
            let aux = if rng.next_f64() < self.null_density {
                None
            } else {
                // aux ≈ 1.5·value − 20, with noise: recoverable signal.
                Some(1.5 * value - 20.0 + (rng.next_f64() - 0.5) * 4.0)
            };
            rows.push(Row {
                ts,
                key,
                value,
                aux,
            });
        }
        // Late arrivals: swap a row behind its successor. Applied in a
        // deterministic pass so the disorder itself is reproducible.
        if self.disorder_fraction > 0.0 {
            for index in 0..rows.len().saturating_sub(1) {
                if rng.next_f64() < self.disorder_fraction {
                    rows.swap(index, index + 1);
                }
            }
        }
        rows
    }
}

/// SplitMix64: tiny, well-studied, and exactly reproducible everywhere —
/// the reason this crate needs no RNG dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[0, bound)`; bias is negligible for the small bounds
    /// used here and determinism is what matters.
    fn next_range(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let spec = Spec::telemetry(500, 42);
        assert_eq!(spec.generate(), spec.generate());
        // A different seed is a different corpus.
        assert_ne!(spec.generate(), Spec::telemetry(500, 43).generate());
    }

    #[test]
    fn ticks_are_ordered_and_null_free() {
        let rows = Spec::ticks(2_000, 7).generate();
        assert_eq!(rows.len(), 2_000);
        assert!(rows.windows(2).all(|pair| pair[0].ts <= pair[1].ts));
        assert!(rows.iter().all(|row| row.aux.is_some()));
        assert!(rows.iter().all(|row| row.key < 32));
    }

    #[test]
    fn telemetry_has_the_advertised_imperfections() {
        let rows = Spec::telemetry(5_000, 7).generate();
        let disordered = rows
            .windows(2)
            .filter(|pair| pair[0].ts > pair[1].ts)
            .count();
        let nulls = rows.iter().filter(|row| row.aux.is_none()).count();
        // Around 1% disorder and 2% nulls, loosely bounded — the point
        // is presence, not an exact count.
        assert!(disordered > 10, "{disordered}");
        assert!((50..500).contains(&nulls), "{nulls}");
    }

    #[test]
    fn aux_carries_recoverable_signal() {
        // Least-squares over (value, aux) pairs should sit near the
        // generating line aux = 1.5·value − 20.
        let rows = Spec::ticks(10_000, 11).generate();
        let pairs: Vec<(f64, f64)> = rows
            .iter()
            .filter_map(|row| row.aux.map(|aux| (row.value, aux)))
            .collect();
        let n = pairs.len() as f64;
        let (sx, sy): (f64, f64) = pairs
            .iter()
            .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
        let (mx, my) = (sx / n, sy / n);
        let (sxx, sxy): (f64, f64) = pairs.iter().fold((0.0, 0.0), |(sxx, sxy), (x, y)| {
            (sxx + (x - mx) * (x - mx), sxy + (x - mx) * (y - my))
        });
        let slope = sxy / sxx;
        assert!((slope - 1.5).abs() < 0.1, "{slope}");
    }
}
