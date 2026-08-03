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
//! a from-scratch recompute in the final ulps. Both folds run the
//! executor's ordinary aggregates (plain f64 accumulation — the
//! Neumaier reference in the M5.0 numerics guard is a test yardstick,
//! not shipped summation), and the contract is agreement with
//! recompute within 1e-12 relative — the tolerance the mutation,
//! as-of, and view oracle families apply (the slice and differential
//! families run at 1e-9). `COUNT`, `MIN`, `MAX`, `FIRST`, and `LAST`
//! combine exactly. (Stated 2026-08-03 with the tranche-2 plan;
//! revisitable, but exact single-pass equality is impossible under any
//! partials representation.)

use query_lite::{AggCall, AggFunction};

/// The expanding-window functions a cumulative view can maintain,
/// mapped to the aggregate whose decomposition serves them: the
/// per-bucket partial of `sum(x) OVER (UNBOUNDED PRECEDING..)` is a
/// plain per-bucket `SUM(x)`, and so on down the family. Anything
/// else — `var_pop` and friends, `first`/`last` as windows, LAG/LEAD —
/// is refused by name at the definition door.
pub(crate) fn expanding_window_function(name: &str) -> Option<AggFunction> {
    match name {
        "sum" => Some(AggFunction::Sum),
        "count" => Some(AggFunction::Count),
        "avg" => Some(AggFunction::Avg),
        "min" => Some(AggFunction::Min),
        "max" => Some(AggFunction::Max),
        _ => None,
    }
}

/// The hidden bucket column's name in a running or cumulative view's
/// materialization. The whole `__` prefix is reserved at the view
/// layer's definition door: a running/cumulative definition that
/// references or aliases any `__`-prefixed name is refused at create
/// (the synthesis mints `__bucket`, `__p{i}`, `__row`, and
/// `__w{i}_sum`/`__w{i}_count` there, and a user name shadowing a
/// minted one would produce a view that creates fine and can never be
/// read).
pub(crate) const HIDDEN_BUCKET: &str = "__bucket";

/// How one user-facing aggregate of a running view decomposes: the
/// partial calls its materialization stores per hidden bucket, the
/// combine calls that reassemble them across buckets, and how the
/// combined columns finalize into the one user-facing output column.
pub(crate) struct Decomposition {
    /// The aggregate's [`PartialForm`].
    pub(crate) form: PartialForm,
    /// Materialization calls, aliased `__p{i}`, `__p{i+1}`, …
    pub(crate) partials: Vec<AggCall>,
    /// Combine calls over the partial columns, same aliases — combined
    /// column `j` of this decomposition reassembles partial column `j`.
    pub(crate) combines: Vec<AggCall>,
}

/// Decomposes `call` starting at partial-column index `next_index`.
///
/// The combine of a partial is itself a built aggregate, which is the
/// load-bearing fact of the whole representation: the materialization
/// plan and the combine plan are both ordinary bucketed/grouped
/// queries, so every piece of tranche-1 machinery — refresh, touched
/// buckets, the stamp, the crash story — serves running views
/// unchanged. Sums and counts combine by `SUM` over the partials;
/// extrema by `MIN`/`MAX`; `FIRST`/`LAST` by `FIRST`/`LAST` over the
/// partials ordered by the hidden bucket, which agrees with the
/// whole-table answer because bucket index is monotone in the ordering
/// key. `AVG` is the two-column case: `(sum, count)` partials, summed
/// separately, divided only at finalize — an average of averages
/// weights buckets, not rows, and is simply wrong.
pub(crate) fn decompose(call: &AggCall, next_index: usize) -> Decomposition {
    let form = PartialForm::of(call.function);
    let name = |offset: usize| Some(format!("__p{}", next_index + offset));
    let partial = |function: AggFunction, offset: usize| AggCall {
        function,
        argument: call.argument.clone(),
        alias: name(offset),
    };
    let combine = |function: AggFunction, offset: usize| AggCall {
        function,
        argument: Some(format!("__p{}", next_index + offset)),
        alias: name(offset),
    };
    let (partials, combines) = match form {
        PartialForm::Sum => (
            vec![partial(AggFunction::Sum, 0)],
            vec![combine(AggFunction::Sum, 0)],
        ),
        // A bucket's COUNT combines by SUM: counts add.
        PartialForm::Count => (
            vec![partial(AggFunction::Count, 0)],
            vec![combine(AggFunction::Sum, 0)],
        ),
        PartialForm::SumCount => (
            vec![partial(AggFunction::Sum, 0), partial(AggFunction::Count, 1)],
            vec![combine(AggFunction::Sum, 0), combine(AggFunction::Sum, 1)],
        ),
        PartialForm::Min => (
            vec![partial(AggFunction::Min, 0)],
            vec![combine(AggFunction::Min, 0)],
        ),
        PartialForm::Max => (
            vec![partial(AggFunction::Max, 0)],
            vec![combine(AggFunction::Max, 0)],
        ),
        PartialForm::First => (
            vec![partial(AggFunction::First, 0)],
            vec![combine(AggFunction::First, 0)],
        ),
        PartialForm::Last => (
            vec![partial(AggFunction::Last, 0)],
            vec![combine(AggFunction::Last, 0)],
        ),
    };
    Decomposition {
        form,
        partials,
        combines,
    }
}

/// The per-bucket partial columns one aggregate call needs, and how
/// they recombine across buckets. One aggregate maps to one form; a
/// form may serve several aggregates (`AVG` rides `SumCount`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PartialForm {
    /// One column: the bucket's f64 sum. Combines by addition.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompositions_reassemble_with_built_aggregates_only() {
        // The whole representation stands on this: partials AND
        // combines are ordinary built aggregates, so both plans run
        // through the existing executor. COUNT's combine is SUM
        // (counts add), AVG's two columns sum separately, and the
        // positional pair combine positionally over the hidden bucket.
        let call = |function| AggCall {
            function,
            argument: Some("x".to_owned()),
            alias: None,
        };
        for (function, combine_functions) in [
            (AggFunction::Sum, vec![AggFunction::Sum]),
            (AggFunction::Count, vec![AggFunction::Sum]),
            (AggFunction::Avg, vec![AggFunction::Sum, AggFunction::Sum]),
            (AggFunction::Min, vec![AggFunction::Min]),
            (AggFunction::Max, vec![AggFunction::Max]),
            (AggFunction::First, vec![AggFunction::First]),
            (AggFunction::Last, vec![AggFunction::Last]),
        ] {
            let decomposition = decompose(&call(function), 3);
            assert_eq!(decomposition.partials.len(), decomposition.combines.len());
            assert_eq!(
                decomposition.partials.len(),
                if decomposition.form == PartialForm::SumCount {
                    2
                } else {
                    1
                }
            );
            assert_eq!(
                decomposition
                    .combines
                    .iter()
                    .map(|combine| combine.function)
                    .collect::<Vec<_>>(),
                combine_functions
            );
            // The index discipline: partial j and combine j share one
            // alias, offset by next_index, and the combine reads the
            // partial's column.
            for (j, (partial, combine)) in decomposition
                .partials
                .iter()
                .zip(&decomposition.combines)
                .enumerate()
            {
                let alias = format!("__p{}", 3 + j);
                assert_eq!(partial.alias.as_deref(), Some(alias.as_str()));
                assert_eq!(combine.alias.as_deref(), Some(alias.as_str()));
                assert_eq!(combine.argument.as_deref(), Some(alias.as_str()));
            }
        }
    }

    #[test]
    fn every_aggregate_has_a_partial_form() {
        // The eligibility door for tranche 2: every built aggregate
        // decomposes, so "no partial form" can never be a silent
        // execution error — the map is total by construction, and this
        // test exists to break loudly when a new aggregate lands
        // without a decomposition decision.
        for (function, form) in [
            (AggFunction::Sum, PartialForm::Sum),
            (AggFunction::Count, PartialForm::Count),
            (AggFunction::Avg, PartialForm::SumCount),
            (AggFunction::Min, PartialForm::Min),
            (AggFunction::Max, PartialForm::Max),
            (AggFunction::First, PartialForm::First),
            (AggFunction::Last, PartialForm::Last),
        ] {
            assert_eq!(PartialForm::of(function), form);
        }
    }
}
