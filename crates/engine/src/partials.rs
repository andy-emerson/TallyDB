//! Bucket partials (#83, tranche 2): how an aggregate decomposes into
//! per-bucket partial columns and recombines across buckets.
//!
//! Tranche 1 refused running and cumulative shapes because their blast
//! radius under correction is unbounded — a correction at `t` changes
//! every result after `t`. The partials representation removes the
//! problem from storage: the materialization holds, per hidden bucket
//! of the ordering key, not the answer but the **partials** the answer
//! recombines from — a bucket's sum, count, min, max, or edge value.
//! A correction re-folds ONE bucket's partials (uniform repair,
//! unchanged); the running answer is assembled at read by combining
//! partials in bucket order. The O(suffix) rewrite never exists.
//!
//! ## The combine contract, stated
//!
//! Combining per-bucket f64 sums associates differently than a single
//! pass over the rows, so a running view's `SUM`/`AVG` may differ from
//! a from-scratch recompute in the final ulps. The partials fold and
//! the cross-bucket combine both use compensated (Neumaier) summation
//! — the M5.0 discipline — and the contract is agreement with
//! recompute within 1e-12 relative, the same tolerance every DuckDB
//! oracle family already applies. `COUNT`, `MIN`, `MAX`, `FIRST`, and
//! `LAST` combine exactly. (Stated 2026-08-03 with the tranche-2 plan;
//! revisitable, but exact single-pass equality is impossible under any
//! partials representation.)

use query_lite::AggFunction;

/// The per-bucket partial columns one aggregate call needs, and how
/// they recombine across buckets. One aggregate maps to one form; a
/// form may serve several aggregates (`AVG` rides `SumCount`).
// Committed ahead of its production caller (tranche 2, cycle 1 of the
// recorded plan — task ledger and #83): the running-aggregate
// materialization of cycle 2 calls `of` and `columns`, and removing
// this allow is part of that cycle's definition of done.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PartialForm {
    /// One column: the bucket's compensated sum. Combines by addition.
    Sum,
    /// One column: the bucket's non-null count. Combines by addition,
    /// exactly (i64).
    Count,
    /// Two columns: sum and count — `AVG` divides after the combine,
    /// never per bucket (an average of averages weights buckets, not
    /// rows, and is simply wrong).
    SumCount,
    /// One column: the bucket's minimum. Combines by `min`, exactly.
    Min,
    /// One column: the bucket's maximum. Combines by `max`, exactly.
    Max,
    /// One column: the value at the bucket's earliest ordering-key
    /// position. Combines positionally: the first non-empty bucket
    /// wins.
    First,
    /// One column: the value at the bucket's latest position. The last
    /// non-empty bucket wins.
    Last,
}

#[allow(dead_code)] // as on the enum: cycle 2's caller removes both.
impl PartialForm {
    /// The form for an aggregate, if tranche 2 can decompose it.
    pub(crate) fn of(function: AggFunction) -> PartialForm {
        match function {
            AggFunction::Sum => PartialForm::Sum,
            AggFunction::Count => PartialForm::Count,
            AggFunction::Avg => PartialForm::SumCount,
            AggFunction::Min => PartialForm::Min,
            AggFunction::Max => PartialForm::Max,
            AggFunction::First => PartialForm::First,
            AggFunction::Last => PartialForm::Last,
        }
    }

    /// How many partial columns the form materializes.
    pub(crate) fn columns(self) -> usize {
        match self {
            PartialForm::SumCount => 2,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_aggregate_has_a_partial_form() {
        // The eligibility door for tranche 2: every built aggregate
        // decomposes, so "no partial form" can never be a silent
        // execution error — the map is total by construction, and this
        // test exists to break loudly when a new aggregate lands
        // without a decomposition decision.
        for (function, form, columns) in [
            (AggFunction::Sum, PartialForm::Sum, 1),
            (AggFunction::Count, PartialForm::Count, 1),
            (AggFunction::Avg, PartialForm::SumCount, 2),
            (AggFunction::Min, PartialForm::Min, 1),
            (AggFunction::Max, PartialForm::Max, 1),
            (AggFunction::First, PartialForm::First, 1),
            (AggFunction::Last, PartialForm::Last, 1),
        ] {
            assert_eq!(PartialForm::of(function), form);
            assert_eq!(form.columns(), columns);
        }
    }
}
