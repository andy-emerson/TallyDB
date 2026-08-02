//! Row predicates: the WHERE substrate.
//!
//! Built for `UPDATE`/`DELETE` first (M2.3) and deliberately shaped as
//! the layer `SELECT ... WHERE` will reuse (M2.4): a small predicate
//! tree — numeric comparisons, key string equality, `IN` and `LIKE`,
//! `IS [NOT] NULL`, `AND` / `OR` / `NOT` — evaluated per segment into a
//! row bitmap.
//!
//! One leaf is different in kind: [`Predicate::CompareExpr`] compares
//! two whole scalar expressions (`x > y`, `x * 2 > y + 1`), and no zone
//! map can rule it out, so it prunes nothing. Pruning therefore
//! degrades **per conjunct** rather than per query — see that variant's
//! documentation.
//!
//! String predicates follow the design's rule for keys: the string test
//! runs **once per distinct dictionary value**, producing a set of
//! allowed codes; rows are then matched by integer set-membership, never
//! by per-row string comparison. A null cell makes its comparison
//! UNKNOWN, and the tree composes in three-valued (Kleene) logic — so
//! `NOT (a AND b)` is TRUE where a is FALSE even if b is UNKNOWN — with
//! `WHERE` keeping only the rows that come out TRUE.

use crate::plan::QueryError;
use arrow_lite::{Bitmap, Column, ColumnType, NumericData, Schema};
use sqlparser::ast;
use std::cmp::Ordering;
use storage_lite::{SegmentHandle, SegmentView};

/// A numeric literal, kept as written: integers stay exact `i64`, so an
/// `i64` column never round-trips through `f64` precision.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Number {
    /// An integer literal.
    Int(i64),
    /// A floating-point literal.
    Float(f64),
}

impl Number {
    fn as_f64(self) -> f64 {
        match self {
            Number::Int(value) => value as f64,
            Number::Float(value) => value,
        }
    }
}

/// A comparison operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    /// `=`
    Eq,
    /// `<>` / `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl CmpOp {
    fn holds<T: PartialOrd>(self, left: T, right: T) -> bool {
        match self {
            CmpOp::Eq => left == right,
            CmpOp::Ne => left != right,
            CmpOp::Lt => left < right,
            CmpOp::Le => left <= right,
            CmpOp::Gt => left > right,
            CmpOp::Ge => left >= right,
        }
    }

    fn holds_f64(self, left: f64, right: f64) -> bool {
        self.holds_ordering(cmp_f64(left, right))
    }

    /// Whether the operator holds given a precomputed ordering — shared by
    /// the f64 relation and the exact i64-vs-float relation.
    fn holds_ordering(self, ordering: Ordering) -> bool {
        match self {
            CmpOp::Eq => ordering == Ordering::Equal,
            CmpOp::Ne => ordering != Ordering::Equal,
            CmpOp::Lt => ordering == Ordering::Less,
            CmpOp::Le => ordering != Ordering::Greater,
            CmpOp::Gt => ordering == Ordering::Greater,
            CmpOp::Ge => ordering != Ordering::Less,
        }
    }
}

/// The engine's `f64` comparison relation (D2 ruling, 2026-07-24): NaN
/// is a *value*, greater than every number and equal to itself, so a
/// NaN row matches `x > 5` and sorts, filters, and prunes under one
/// consistent order. Ordinary values keep IEEE comparison, so
/// `-0.0 = 0.0` stays true — this is "NaN lifted to the top", not
/// bitwise total order. NULL is not handled here: null is not a value
/// and never reaches a comparison (three-valued logic masks it out).
pub(crate) fn cmp_f64(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(&right).expect("both are non-NaN"),
    }
}

/// Exact ordering of an `i64` against an `f64` literal under the same
/// relation as [`cmp_f64`] — but **without** the lossy `as f64` cast, so
/// an `i64` beyond 2^53 compares correctly (B6). `t` is a finite SQL
/// float literal in practice; a NaN target is treated as greatest, for
/// consistency with `cmp_f64`.
pub(crate) fn cmp_i64_f64(v: i64, t: f64) -> Ordering {
    if t.is_nan() {
        return Ordering::Less; // every number is below NaN
    }
    // Out of i64 range: the sign of `t` alone decides.
    if t >= 9_223_372_036_854_775_808.0 {
        return Ordering::Less; // v <= i64::MAX < 2^63 <= t
    }
    if t < -9_223_372_036_854_775_808.0 {
        return Ordering::Greater; // v >= i64::MIN = -2^63 > t
    }
    // `t` is in [-2^63, 2^63): its floor is integral and fits i64 exactly.
    let floor = t.floor();
    let floor_i = floor as i64;
    match v.cmp(&floor_i) {
        // v == floor(t); a fractional part makes t strictly greater.
        Ordering::Equal if t > floor => Ordering::Less,
        other => other,
    }
}

/// The predicate tree.
#[derive(Clone, PartialEq, Debug)]
pub enum Predicate {
    /// `column <op> number` on a numeric column.
    Compare {
        /// The numeric column.
        column: String,
        /// The operator.
        op: CmpOp,
        /// The literal.
        value: Number,
    },
    /// A comparison between two **expressions** — `x > y`,
    /// `x * 2 > y + 1`, or (inside `CASE`) a window result.
    ///
    /// Unlike every other variant this one cannot be pruned: a zone map
    /// knows a column's range, not the range of an expression over
    /// several of them, so [`can_match`] answers "might match" and the
    /// rows are read. Pruning therefore degrades **per conjunct** —
    /// `WHERE ts > 1000 AND x > y` still skips segments on `ts` — which
    /// is why this is a separate variant rather than a generalisation
    /// of `Compare`.
    CompareExpr {
        /// Left operand.
        left: crate::plan::ScalarExpr,
        /// The operator.
        op: CmpOp,
        /// Right operand.
        right: crate::plan::ScalarExpr,
    },
    /// `column = 'v'` / `column <> 'v'` on a key column.
    KeyEquals {
        /// The key column.
        column: String,
        /// The literal.
        value: String,
        /// `true` for `<>`.
        negated: bool,
    },
    /// `column [NOT] LIKE 'pattern'` on a key column: `%` matches any
    /// run (including empty), `_` any single character; evaluated once
    /// per *distinct* dictionary value, applied as integer
    /// set-membership — the same cheap shape as `IN` (#57's LIKE half;
    /// regex stays tracked there, pending a dependency ruling).
    KeyLike {
        /// The key column.
        column: String,
        /// The LIKE pattern, verbatim.
        pattern: String,
        /// `true` for `NOT LIKE`.
        negated: bool,
    },
    /// `column IS [NOT] NULL`, on a column of either kind. Alone among
    /// the leaves this one is **total**: it asks about presence, not
    /// about a value, so every row comes back TRUE or FALSE and none is
    /// UNKNOWN — which is exactly why SQL needs it.
    IsNull {
        /// The column, numeric or key.
        column: String,
        /// `true` for `IS NOT NULL`.
        negated: bool,
    },
    /// `column [NOT] IN ('a', 'b', ...)` on a key column.
    KeyIn {
        /// The key column.
        column: String,
        /// The literals.
        values: Vec<String>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
    /// Both sides hold.
    And(Box<Predicate>, Box<Predicate>),
    /// Either side holds.
    Or(Box<Predicate>, Box<Predicate>),
    /// The side does not hold (null cells still match nothing).
    Not(Box<Predicate>),
}

/// Lowers a parsed WHERE expression into a [`Predicate`], rejecting —
/// by name — anything outside the supported fragment.
pub fn lower_predicate(
    expr: &ast::Expr,
    windows: &mut Vec<crate::plan::WindowCall>,
) -> Result<Predicate, QueryError> {
    match expr {
        ast::Expr::Nested(inner) => lower_predicate(inner, windows),
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Not,
            expr,
        } => Ok(Predicate::Not(Box::new(lower_predicate(expr, windows)?))),
        ast::Expr::IsNull(inner) => Ok(Predicate::IsNull {
            column: null_test_column(inner)?,
            negated: false,
        }),
        ast::Expr::IsNotNull(inner) => Ok(Predicate::IsNull {
            column: null_test_column(inner)?,
            negated: true,
        }),
        ast::Expr::Like {
            negated,
            expr,
            pattern,
            escape_char,
            any: _,
        } => {
            if escape_char.is_some() {
                return Err(QueryError::Unsupported(
                    "LIKE ... ESCAPE (use the default escaping)".to_owned(),
                ));
            }
            let ast::Expr::Identifier(column) = expr.as_ref() else {
                return Err(QueryError::Unsupported(
                    "LIKE applies to a key column".to_owned(),
                ));
            };
            let ast::Expr::Value(value) = pattern.as_ref() else {
                return Err(QueryError::Unsupported(
                    "LIKE takes a string literal pattern".to_owned(),
                ));
            };
            let ast::Value::SingleQuotedString(pattern) = &value.value else {
                return Err(QueryError::Unsupported(
                    "LIKE takes a string literal pattern".to_owned(),
                ));
            };
            Ok(Predicate::KeyLike {
                column: column.value.clone(),
                pattern: pattern.clone(),
                negated: *negated,
            })
        }
        ast::Expr::BinaryOp { left, op, right } => match op {
            ast::BinaryOperator::And => Ok(Predicate::And(
                Box::new(lower_predicate(left, windows)?),
                Box::new(lower_predicate(right, windows)?),
            )),
            ast::BinaryOperator::Or => Ok(Predicate::Or(
                Box::new(lower_predicate(left, windows)?),
                Box::new(lower_predicate(right, windows)?),
            )),
            _ => lower_comparison(left, op, right, windows),
        },
        ast::Expr::InList {
            expr,
            list,
            negated,
        } => {
            let ast::Expr::Identifier(column) = expr.as_ref() else {
                return Err(QueryError::Unsupported(
                    "IN requires a plain column on the left".to_owned(),
                ));
            };
            let values = list
                .iter()
                .map(|item| match item {
                    ast::Expr::Value(value) => match &value.value {
                        ast::Value::SingleQuotedString(text) => Ok(text.clone()),
                        other => Err(QueryError::Unsupported(format!(
                            "IN list item '{other}' (string literals only)"
                        ))),
                    },
                    other => Err(QueryError::Unsupported(format!(
                        "IN list item '{other}' (string literals only)"
                    ))),
                })
                .collect::<Result<Vec<String>, QueryError>>()?;
            Ok(Predicate::KeyIn {
                column: column.value.clone(),
                values,
                negated: *negated,
            })
        }
        other => Err(QueryError::Unsupported(format!(
            "predicate '{other}' (comparisons, IN, LIKE, IS NULL, AND/OR/NOT only)"
        ))),
    }
}

/// The operand of `IS [NOT] NULL`: a plain column. Nulls are a property
/// of stored cells, so there is nothing else to ask the question of —
/// an expression's nullness is the nullness of the columns it reads.
fn null_test_column(expr: &ast::Expr) -> Result<String, QueryError> {
    match expr {
        ast::Expr::Nested(inner) => null_test_column(inner),
        ast::Expr::Identifier(column) => Ok(column.value.clone()),
        other => Err(QueryError::Unsupported(format!(
            "IS NULL on '{other}' (a plain column only)"
        ))),
    }
}

fn lower_comparison(
    left: &ast::Expr,
    op: &ast::BinaryOperator,
    right: &ast::Expr,
    windows: &mut Vec<crate::plan::WindowCall>,
) -> Result<Predicate, QueryError> {
    // `40 < x` is `x > 40` written backwards. Mirror it before anything
    // else, so it reaches the prunable shape below: otherwise an
    // identical predicate would prune or not depending on which side
    // the user put the column, and `40 < ts` would meet #40's
    // integer-arithmetic refusal while `ts > 40` answered.
    if is_literal_operand(left) && matches!(right, ast::Expr::Identifier(_)) {
        return lower_comparison(right, &flip(op), left, windows);
    }
    let op = match op {
        ast::BinaryOperator::Eq => CmpOp::Eq,
        ast::BinaryOperator::NotEq => CmpOp::Ne,
        ast::BinaryOperator::Lt => CmpOp::Lt,
        ast::BinaryOperator::LtEq => CmpOp::Le,
        ast::BinaryOperator::Gt => CmpOp::Gt,
        ast::BinaryOperator::GtEq => CmpOp::Ge,
        other => {
            return Err(QueryError::Unsupported(format!(
                "operator '{other}' in a predicate"
            )))
        }
    };
    // The prunable shape — a bare column against a literal is what a
    // zone map can rule out, so it keeps its own variant. Any other
    // operand shape falls through to the general comparison below,
    // which is correct but unprunable (#95).
    let ast::Expr::Identifier(column) = left else {
        return compare_expressions(left, op, right, windows);
    };
    // A negative number parses as unary minus over a literal. Keep the
    // original: the fallback below must see `-x`, not `x`.
    let whole_right = right;
    let (negated_literal, right) = match right {
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Minus,
            expr,
        } => (true, expr.as_ref()),
        other => (false, other),
    };
    let ast::Expr::Value(value) = right else {
        return compare_expressions(left, op, whole_right, windows);
    };
    match &value.value {
        ast::Value::Number(text, _) => {
            let mut value = parse_number(text)?;
            if negated_literal {
                value = match value {
                    Number::Int(value) => Number::Int(-value),
                    Number::Float(value) => Number::Float(-value),
                };
            }
            Ok(Predicate::Compare {
                column: column.value.clone(),
                op,
                value,
            })
        }
        // A minus sign belongs to a number. On a string literal it is
        // not a negative key — there is no such thing — so it is
        // refused rather than quietly dropped, which would read
        // `sym = -'A'` as `sym = 'A'`.
        ast::Value::SingleQuotedString(_) if negated_literal => Err(QueryError::TypeError(
            "unary minus on a string literal (keys are labels, not numbers)".to_owned(),
        )),
        ast::Value::SingleQuotedString(text) => match op {
            CmpOp::Eq | CmpOp::Ne => Ok(Predicate::KeyEquals {
                column: column.value.clone(),
                value: text.clone(),
                negated: op == CmpOp::Ne,
            }),
            _ => Err(QueryError::Unsupported(
                "string comparisons other than = / <> (keys are labels, not ordered text)"
                    .to_owned(),
            )),
        },
        _ => compare_expressions(left, op, whole_right, windows),
    }
}

/// Whether an operand is a literal, including the unary minus a
/// negative number parses as. Only used to spot a backwards
/// comparison; the mirrored call puts an identifier on the left, so
/// the recursion is one deep.
fn is_literal_operand(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Value(_) => true,
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Minus,
            expr,
        } => matches!(expr.as_ref(), ast::Expr::Value(_)),
        _ => false,
    }
}

/// The mirror of a comparison operator, for reading `40 < x` as
/// `x > 40`. Equality and inequality are their own mirrors.
fn flip(op: &ast::BinaryOperator) -> ast::BinaryOperator {
    match op {
        ast::BinaryOperator::Lt => ast::BinaryOperator::Gt,
        ast::BinaryOperator::LtEq => ast::BinaryOperator::GtEq,
        ast::BinaryOperator::Gt => ast::BinaryOperator::Lt,
        ast::BinaryOperator::GtEq => ast::BinaryOperator::LtEq,
        other => other.clone(),
    }
}

/// The general comparison: both sides lowered as scalar expressions.
fn compare_expressions(
    left: &ast::Expr,
    op: CmpOp,
    right: &ast::Expr,
    windows: &mut Vec<crate::plan::WindowCall>,
) -> Result<Predicate, QueryError> {
    Ok(Predicate::CompareExpr {
        left: crate::plan::lower_scalar_expr(left, windows)?,
        op,
        right: crate::plan::lower_scalar_expr(right, windows)?,
    })
}

/// Parses a SQL number literal, preserving integer exactness.
pub(crate) fn parse_number(text: &str) -> Result<Number, QueryError> {
    if let Ok(value) = text.parse::<i64>() {
        return Ok(Number::Int(value));
    }
    text.parse::<f64>()
        .map(Number::Float)
        .map_err(|_| QueryError::Parse(format!("bad number literal '{text}'")))
}

/// A three-valued (Kleene) predicate result over a segment's rows:
/// `truth` marks rows where the predicate is TRUE and `falsity` rows
/// where it is FALSE; a row in neither is UNKNOWN (a NULL was involved).
/// `WHERE`/`UPDATE`/`DELETE` keep only TRUE rows, but `AND`/`OR`/`NOT`
/// must distinguish FALSE from UNKNOWN to compose correctly — e.g.
/// `NOT (a AND b)` is TRUE when a is FALSE even if b is UNKNOWN, which a
/// single match/no-match bitmap cannot express.
struct ThreeValued {
    truth: Bitmap,
    falsity: Bitmap,
}

impl ThreeValued {
    fn and(self, other: ThreeValued) -> ThreeValued {
        // TRUE iff both TRUE; FALSE iff either FALSE — FALSE dominates
        // UNKNOWN, which is exactly what the old blanket null-mask missed.
        ThreeValued {
            truth: self.truth.and(&other.truth),
            falsity: self.falsity.or(&other.falsity),
        }
    }

    fn or(self, other: ThreeValued) -> ThreeValued {
        // TRUE iff either TRUE — TRUE dominates UNKNOWN; FALSE iff both FALSE.
        ThreeValued {
            truth: self.truth.or(&other.truth),
            falsity: self.falsity.and(&other.falsity),
        }
    }

    fn not(self) -> ThreeValued {
        // NOT TRUE = FALSE, NOT FALSE = TRUE, NOT UNKNOWN = UNKNOWN.
        ThreeValued {
            truth: self.falsity,
            falsity: self.truth,
        }
    }
}

/// Evaluates `predicate` over one segment view, returning the bitmap of
/// rows for which it is TRUE (`true` = matched); UNKNOWN and FALSE rows
/// are both excluded — SQL's three-valued `WHERE`. Tombstoned rows are
/// evaluated like any other — callers combine with the live mask; this
/// keeps the result independent of mutation state.
pub fn evaluate(
    predicate: &Predicate,
    schema: &Schema,
    view: &SegmentView,
    scalars: &dyn ScalarEval,
) -> Result<Bitmap, QueryError> {
    Ok(evaluate_3vl(predicate, schema, view, scalars)?.truth)
}

/// Evaluates a scalar expression over one view's rows.
///
/// Supplied by the caller rather than implemented here, because a
/// predicate tree knows nothing about the compute registry or about
/// window results — and must not have to, or `WHERE` would depend on
/// the projection machinery.
pub trait ScalarEval {
    /// `(values, validity)` over the view's rows, in stored order.
    fn eval(&self, expr: &crate::plan::ScalarExpr) -> Result<(Vec<f64>, Vec<bool>), QueryError>;
}

/// A `ScalarEval` for predicates that cannot contain expressions —
/// used where a caller has no expression evaluator to offer, and any
/// `CompareExpr` reaching it is a bug rather than a user error.
pub struct NoScalars;

impl ScalarEval for NoScalars {
    fn eval(&self, _expr: &crate::plan::ScalarExpr) -> Result<(Vec<f64>, Vec<bool>), QueryError> {
        Err(QueryError::Unsupported(
            "an expression comparison in a position that cannot evaluate one".to_owned(),
        ))
    }
}

/// The three-valued recursion beneath [`evaluate`]: only the composition
/// operators need the FALSE/UNKNOWN distinction, so the tree is walked in
/// Kleene logic and just the final TRUE set is handed back.
fn evaluate_3vl(
    predicate: &Predicate,
    schema: &Schema,
    view: &SegmentView,
    scalars: &dyn ScalarEval,
) -> Result<ThreeValued, QueryError> {
    let batch = view.segment.batch();
    let rows = batch.num_rows();
    match predicate {
        Predicate::CompareExpr { left, op, right } => {
            let (left, left_valid) = scalars.eval(left)?;
            let (right, right_valid) = scalars.eval(right)?;
            // The two row spaces this leaf straddles, and the reason it
            // is the only leaf that has to. Every other one reads a
            // stored column, indexed by STORED row. A scalar expression
            // is evaluated over the view's LIVE rows — it must be, that
            // is what the projection pipeline hands kernels — so its
            // results are indexed by live position and have to be
            // scattered back before joining a predicate tree that
            // speaks stored rows.
            //
            // Getting this wrong is not a wrong answer but an index
            // panic, and only over a view carrying tombstones: on a
            // compacted table live == stored and it hides completely.
            let mut live_position = 0usize;
            let mut verdict = vec![(false, false); rows];
            for (row, slot) in verdict.iter_mut().enumerate() {
                if !view.is_live(row) {
                    continue; // a dead row's verdict is never read
                }
                // NULL on either side is UNKNOWN, not false — the same
                // three-valued rule every other leaf follows. NaN
                // compares under `cmp_f64`, the one relation sorting
                // and pruning share.
                *slot = (
                    left_valid[live_position] && right_valid[live_position],
                    op.holds_ordering(cmp_f64(left[live_position], right[live_position])),
                );
                live_position += 1;
            }
            debug_assert_eq!(live_position, left.len(), "one verdict per live row");
            Ok(leaf_result(rows, |row| verdict[row]))
        }
        Predicate::Compare { column, op, value } => {
            let index = column_index(schema, column)?;
            match &batch.columns()[index] {
                Column::Numeric(NumericData::F64(numeric)) => {
                    let values = numeric.values().as_slice();
                    let target = value.as_f64();
                    Ok(leaf_result(rows, |row| {
                        (numeric.is_valid(row), op.holds_f64(values[row], target))
                    }))
                }
                Column::Numeric(NumericData::I64(numeric)) => {
                    let values = numeric.values().as_slice();
                    Ok(leaf_result(rows, |row| {
                        let holds = match value {
                            // Exact integer comparison — no f64 round trip.
                            Number::Int(target) => op.holds(values[row], *target),
                            // Also exact against a float literal (B6): no
                            // lossy `as f64` cast, so i64 past 2^53 is right.
                            Number::Float(target) => {
                                op.holds_ordering(cmp_i64_f64(values[row], *target))
                            }
                        };
                        (numeric.is_valid(row), holds)
                    }))
                }
                Column::Key(_) => Err(QueryError::TypeError(format!(
                    "column '{column}' is a key; compare it to a string"
                ))),
            }
        }
        Predicate::IsNull { column, negated } => {
            let index = column_index(schema, column)?;
            let values = &batch.columns()[index];
            // Total, not three-valued: `valid` is a fact about every
            // row, so the verdict is never UNKNOWN. Passing `true` as
            // the validity here is not a shortcut — it says the test
            // itself always has an answer.
            Ok(leaf_result(rows, |row| {
                (true, values.is_valid(row) == *negated)
            }))
        }
        Predicate::KeyEquals {
            column,
            value,
            negated,
        } => key_membership(schema, view, column, std::slice::from_ref(value), *negated),
        Predicate::KeyIn {
            column,
            values,
            negated,
        } => key_membership(schema, view, column, values, *negated),
        Predicate::KeyLike {
            column,
            pattern,
            negated,
        } => key_predicate(schema, view, column, *negated, |value| {
            like_match(pattern, value)
        }),
        Predicate::And(left, right) => Ok(evaluate_3vl(left, schema, view, scalars)?
            .and(evaluate_3vl(right, schema, view, scalars)?)),
        Predicate::Or(left, right) => Ok(evaluate_3vl(left, schema, view, scalars)?
            .or(evaluate_3vl(right, schema, view, scalars)?)),
        Predicate::Not(inner) => Ok(evaluate_3vl(inner, schema, view, scalars)?.not()),
    }
}

/// Builds a leaf result from a per-row `(valid, holds)` verdict: a valid
/// row is TRUE when the comparison holds and FALSE when it does not; a
/// null row (`valid == false`) is UNKNOWN — in neither bitmap.
fn leaf_result(rows: usize, verdict: impl Fn(usize) -> (bool, bool)) -> ThreeValued {
    let mut truth = Vec::with_capacity(rows);
    let mut falsity = Vec::with_capacity(rows);
    for row in 0..rows {
        let (valid, holds) = verdict(row);
        truth.push(valid && holds);
        falsity.push(valid && !holds);
    }
    ThreeValued {
        truth: Bitmap::from_bools(truth),
        falsity: Bitmap::from_bools(falsity),
    }
}

/// Whether any row of `view`'s segment could satisfy `predicate`, judged
/// from zone maps alone — the segment-pruning test. This is a sound
/// over-approximation: `true` means "maybe" (evaluate to find out),
/// `false` means "provably no row matches" and the segment can be
/// skipped without reading its columns. Only numeric comparisons prune
/// (key predicates and NOT are always "maybe"); a numeric column with no
/// zone map holds no valid, comparable values, so a comparison on it
/// matches nothing.
///
/// A segment carrying no zone maps at all (a query-lifetime scratch
/// segment — [`Segment::from_batch_unpruned`]) prunes nothing: absent
/// maps mean nothing is known, which is the opposite of an absent map
/// for one column of a segment that has them.
///
/// [`Segment::from_batch_unpruned`]: storage_lite::Segment::from_batch_unpruned
pub fn can_match(predicate: &Predicate, schema: &Schema, view: &SegmentHandle) -> bool {
    if !view.zone_maps_present() {
        return true;
    }
    match predicate {
        // A zone map bounds a column, not an expression over several of
        // them, so this conjunct prunes nothing — and only this one:
        // its siblings in an AND still do (#95).
        Predicate::CompareExpr { .. } => true,
        Predicate::Compare { column, op, value } => {
            let Some(index) = schema
                .fields()
                .iter()
                .position(|field| field.name() == column)
            else {
                return true; // let evaluate report the unknown column
            };
            if schema.fields()[index].column_type() == ColumnType::Key {
                return true; // let evaluate report the type error
            }
            let Some(zone_map) = view.zone_map(index) else {
                return false; // no valid values: a comparison matches nothing
            };
            fn interval_may_hold<T: PartialOrd>(op: CmpOp, min: T, max: T, target: T) -> bool {
                match op {
                    CmpOp::Eq => min <= target && target <= max,
                    CmpOp::Ne => !(min == target && max == target),
                    CmpOp::Lt => min < target,
                    CmpOp::Le => min <= target,
                    CmpOp::Gt => max > target,
                    CmpOp::Ge => max >= target,
                }
            }
            match (zone_map, value) {
                // f64 zones exclude NaN from min/max, but under the
                // engine's comparison relation NaN is greater than every
                // number — so `>`-side pruning (and `<>`) must also ask
                // whether the segment holds a NaN row. An all-NaN zone
                // stores NaN bounds, and cmp_f64 makes every bound test
                // below answer soundly for it.
                (storage_lite::ZoneMap::F64 { min, max, has_nan }, value) => {
                    let target = value.as_f64();
                    if target.is_nan() {
                        // No SQL literal produces NaN today; if one ever
                        // arrives, never prune on it.
                        return true;
                    }
                    match op {
                        CmpOp::Eq => {
                            cmp_f64(*min, target) != Ordering::Greater
                                && cmp_f64(target, *max) != Ordering::Greater
                        }
                        CmpOp::Ne => {
                            *has_nan
                                || !(cmp_f64(*min, target) == Ordering::Equal
                                    && cmp_f64(*max, target) == Ordering::Equal)
                        }
                        CmpOp::Lt => cmp_f64(*min, target) == Ordering::Less,
                        CmpOp::Le => cmp_f64(*min, target) != Ordering::Greater,
                        CmpOp::Gt => *has_nan || cmp_f64(*max, target) == Ordering::Greater,
                        CmpOp::Ge => *has_nan || cmp_f64(*max, target) != Ordering::Less,
                    }
                }
                (storage_lite::ZoneMap::I64 { min, max }, Number::Int(target)) => {
                    interval_may_hold(*op, *min, *max, *target)
                }
                // i64 bounds vs a float literal: widening to f64 rounds,
                // and a rounded bound could prune a matching segment —
                // soundness beats the optimization, so don't prune.
                (storage_lite::ZoneMap::I64 { .. }, Number::Float(_)) => true,
            }
        }
        Predicate::And(left, right) => {
            can_match(left, schema, view) && can_match(right, schema, view)
        }
        Predicate::Or(left, right) => {
            can_match(left, schema, view) || can_match(right, schema, view)
        }
        // `IS NOT NULL` prunes on the one null fact a zone map carries:
        // a numeric column with no map holds no valid value at all, so
        // nothing in the segment is non-null. `IS NULL` never prunes —
        // maps count values, not their absences.
        Predicate::IsNull {
            column,
            negated: true,
        } => {
            let Some(index) = schema
                .fields()
                .iter()
                .position(|field| field.name() == column)
            else {
                return true; // let evaluate report the unknown column
            };
            if schema.fields()[index].column_type() == ColumnType::Key {
                return true; // keys have no zone map either way
            }
            view.zone_map(index).is_some()
        }
        // Key membership and NOT don't prune: dictionaries aren't ranges,
        // and negating an interval fact soundly needs exact bounds
        // semantics this test deliberately doesn't attempt.
        Predicate::IsNull { .. }
        | Predicate::KeyEquals { .. }
        | Predicate::KeyIn { .. }
        | Predicate::KeyLike { .. }
        | Predicate::Not(_) => true,
    }
}

/// The string test run once per distinct dictionary value, applied to
/// rows as integer set-membership. A null key cell is UNKNOWN (in neither
/// bitmap), like any other null.
fn key_membership(
    schema: &Schema,
    view: &SegmentView,
    column: &str,
    values: &[String],
    negated: bool,
) -> Result<ThreeValued, QueryError> {
    key_predicate(schema, view, column, negated, |value| {
        values.iter().any(|allowed| allowed == value)
    })
}

/// Evaluates a string predicate on a key column the dictionary way:
/// once per *distinct* value, then integer set-membership per row.
fn key_predicate(
    schema: &Schema,
    view: &SegmentView,
    column: &str,
    negated: bool,
    matches: impl Fn(&str) -> bool,
) -> Result<ThreeValued, QueryError> {
    let index = column_index(schema, column)?;
    let Column::Key(keys) = &view.segment.batch().columns()[index] else {
        return Err(QueryError::TypeError(format!(
            "column '{column}' is numeric; compare it to a number"
        )));
    };
    let dictionary = keys.dictionary();
    let allowed: Vec<bool> = (0..dictionary.len() as u32)
        .map(|code| matches(dictionary.value(code)) != negated)
        .collect();
    let codes = keys.codes().as_slice();
    Ok(leaf_result(keys.len(), |row| {
        (keys.is_valid(row), allowed[codes[row] as usize])
    }))
}

/// The SQL LIKE matcher: `%` any run (including empty), `_` exactly one
/// character, everything else literal — over characters, not bytes, so
/// `_` honors multi-byte keys. Iterative with `%`-backtracking.
pub(crate) fn like_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut s) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None; // (pattern pos after %, text pos)
    while s < text.len() {
        if p < pattern.len() && (pattern[p] == '_' || pattern[p] == text[s]) {
            p += 1;
            s += 1;
        } else if p < pattern.len() && pattern[p] == '%' {
            star = Some((p + 1, s));
            p += 1;
        } else if let Some((star_p, star_s)) = star {
            // Backtrack: let the last % swallow one more character.
            p = star_p;
            s = star_s + 1;
            star = Some((star_p, star_s + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|&c| c == '%')
}

fn column_index(schema: &Schema, name: &str) -> Result<usize, QueryError> {
    schema
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .ok_or_else(|| crate::plan::no_such_column(name))
}

#[cfg(test)]
mod like_tests {
    use super::like_match;

    #[test]
    fn the_matcher_is_sql_like() {
        for (pattern, text, expected) in [
            ("%", "", true),
            ("%", "anything", true),
            ("", "", true),
            ("", "a", false),
            ("abc", "abc", true),
            ("abc", "abd", false),
            ("a%", "a", true),
            ("a%", "abc", true),
            ("%c", "abc", true),
            ("%b%", "abc", true),
            ("a_c", "abc", true),
            ("a_c", "ac", false),
            ("a__", "abc", true),
            ("%Bank%", "First Bank of Testing", true),
            ("%Bank%", "First Bink of Testing", false),
            ("a%b%c", "axxbyyc", true),
            ("a%b%c", "axxcyyb", false),
            ("_", "é", true), // characters, not bytes
            ("%%%", "x", true),
        ] {
            assert_eq!(
                like_match(pattern, text),
                expected,
                "LIKE '{pattern}' on '{text}'"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use storage_lite::SegmentHandle;

    /// The pruning probe: `can_match` reads metadata through a handle.
    fn handle_of(view: &SegmentView) -> SegmentHandle {
        SegmentHandle::resident(view.segment.clone(), view.live.clone())
    }

    use super::*;
    use arrow_lite::{ColumnType, Field};
    use storage_lite::{RowValue, WriteBuffer};

    /// A segment mixing NaN with finite values, plus one all-NaN
    /// segment — the D2 ruling's edge cases: NaN is a value, greater
    /// than every number, equal to itself, in predicates and pruning.
    fn nan_view(values: &[f64]) -> (Schema, SegmentView) {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, false),
        ]);
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        for (ts, &x) in values.iter().enumerate() {
            buffer
                .append(&[RowValue::I64(ts as i64), RowValue::F64(x)])
                .unwrap();
        }
        let segment = std::sync::Arc::new(buffer.freeze().unwrap());
        (schema, SegmentView::all_live(segment))
    }

    #[test]
    fn nan_is_a_value_greater_than_every_number() {
        let (schema, view) = nan_view(&[1.0, f64::NAN, 5.0]);
        let cases: &[(CmpOp, f64, [bool; 3])] = &[
            (CmpOp::Gt, 2.0, [false, true, true]),   // NaN > 2
            (CmpOp::Ge, 5.0, [false, true, true]),   // NaN >= 5
            (CmpOp::Lt, 2.0, [true, false, false]),  // NaN is not < 2
            (CmpOp::Le, 1e308, [true, false, true]), // NaN above every number
            (CmpOp::Ne, 5.0, [true, true, false]),   // NaN <> 5
            (CmpOp::Eq, 1.0, [true, false, false]),  // NaN != finite
        ];
        for &(op, target, expected) in cases {
            let predicate = Predicate::Compare {
                column: "x".to_owned(),
                op,
                value: Number::Float(target),
            };
            let bitmap = evaluate(&predicate, &schema, &view, &NoScalars).unwrap();
            for (row, &want) in expected.iter().enumerate() {
                assert_eq!(bitmap.get(row), want, "{op:?} {target} row {row}");
            }
            // Pruning stays sound: any op that matches a row must also
            // report the segment as maybe-matching.
            if expected.iter().any(|&matched| matched) {
                assert!(
                    can_match(&predicate, &schema, &handle_of(&view)),
                    "{op:?} {target}"
                );
            }
        }
    }

    #[test]
    fn nan_rows_defeat_upper_bound_pruning_only() {
        // Finite max is 5.0; the NaN row must keep > / >= / <> alive.
        let (schema, view) = nan_view(&[1.0, f64::NAN, 5.0]);
        let compare = |op, target| Predicate::Compare {
            column: "x".to_owned(),
            op,
            value: Number::Float(target),
        };
        assert!(can_match(
            &compare(CmpOp::Gt, 100.0),
            &schema,
            &handle_of(&view)
        ));
        assert!(can_match(
            &compare(CmpOp::Ge, 100.0),
            &schema,
            &handle_of(&view)
        ));
        assert!(can_match(
            &compare(CmpOp::Ne, 100.0),
            &schema,
            &handle_of(&view)
        ));
        // NaN is not below anything: < / <= / = still prune by bounds.
        assert!(!can_match(
            &compare(CmpOp::Lt, 0.5),
            &schema,
            &handle_of(&view)
        ));
        assert!(!can_match(
            &compare(CmpOp::Le, 0.5),
            &schema,
            &handle_of(&view)
        ));
        assert!(!can_match(
            &compare(CmpOp::Eq, 100.0),
            &schema,
            &handle_of(&view)
        ));
        // Without NaN, the upper bound prunes as before.
        let (schema, clean) = nan_view(&[1.0, 5.0]);
        assert!(!can_match(
            &compare(CmpOp::Gt, 100.0),
            &schema,
            &handle_of(&clean)
        ));
    }

    #[test]
    fn all_nan_segment_prunes_soundly() {
        let (schema, view) = nan_view(&[f64::NAN, f64::NAN]);
        let compare = |op, target| Predicate::Compare {
            column: "x".to_owned(),
            op,
            value: Number::Float(target),
        };
        // NaN matches only the >-side and <>.
        assert!(can_match(
            &compare(CmpOp::Gt, 100.0),
            &schema,
            &handle_of(&view)
        ));
        assert!(can_match(
            &compare(CmpOp::Ne, 3.0),
            &schema,
            &handle_of(&view)
        ));
        assert!(!can_match(
            &compare(CmpOp::Lt, 100.0),
            &schema,
            &handle_of(&view)
        ));
        assert!(!can_match(
            &compare(CmpOp::Eq, 3.0),
            &schema,
            &handle_of(&view)
        ));
        let matched = evaluate(&compare(CmpOp::Gt, 100.0), &schema, &view, &NoScalars).unwrap();
        assert!(matched.get(0) && matched.get(1));
    }

    fn view() -> (Schema, SegmentView) {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, true),
        ]);
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        let rows: &[(i64, &str, f64, Option<f64>)] = &[
            (1, "AAPL", 1.0, Some(10.0)),
            (2, "MSFT", 2.5, None),
            (3, "AAPL", -1.0, Some(30.0)),
            (4, "TSLA", 4.0, Some(-40.0)),
        ];
        for &(ts, sym, x, y) in rows {
            buffer
                .append(&[
                    RowValue::I64(ts),
                    RowValue::Key(sym),
                    RowValue::F64(x),
                    y.map_or(RowValue::Null, RowValue::F64),
                ])
                .unwrap();
        }
        let segment = std::sync::Arc::new(buffer.freeze().unwrap());
        (schema, SegmentView::all_live(segment))
    }

    fn matched(sql_where: &str) -> Vec<usize> {
        let (schema, view) = view();
        let sql = format!("SELECT ts FROM t WHERE {sql_where}");
        let statements =
            sqlparser::parser::Parser::parse_sql(&sqlparser::dialect::GenericDialect {}, &sql)
                .unwrap();
        let sqlparser::ast::Statement::Query(query) = &statements[0] else {
            panic!("not a query")
        };
        let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() else {
            panic!("not a select")
        };
        let predicate =
            lower_predicate(select.selection.as_ref().unwrap(), &mut Vec::new()).unwrap();
        let bitmap = evaluate(&predicate, &schema, &view, &NoScalars).unwrap();
        (0..4).filter(|&row| bitmap.get(row)).collect()
    }

    #[test]
    fn numeric_comparisons_match_by_value() {
        assert_eq!(matched("ts >= 3"), [2, 3]);
        assert_eq!(matched("x < 0"), [2]);
        assert_eq!(matched("x = 2.5"), [1]);
        assert_eq!(matched("ts <> 2"), [0, 2, 3]);
        // Negative literals parse as unary minus over a number — the
        // gap the differential harness found.
        assert_eq!(matched("x >= -1"), [0, 1, 2, 3]);
        assert_eq!(matched("x = -1"), [2]);
        assert_eq!(matched("y < -0.5"), [3]);
    }

    #[test]
    fn key_predicates_run_on_the_dictionary() {
        assert_eq!(matched("sym = 'AAPL'"), [0, 2]);
        assert_eq!(matched("sym <> 'AAPL'"), [1, 3]);
        assert_eq!(matched("sym IN ('MSFT', 'TSLA')"), [1, 3]);
        assert_eq!(matched("sym NOT IN ('MSFT', 'TSLA')"), [0, 2]);
        assert_eq!(matched("sym = 'UNKNOWN'"), Vec::<usize>::new());
    }

    #[test]
    fn a_minus_sign_on_a_string_literal_is_refused_not_dropped() {
        // Found by the repo-wide code review. `-'AAPL'` parses as unary
        // minus over a string; the arm that strips the minus for
        // negative numbers passed the bare text through, so this read
        // as `sym = 'AAPL'` and quietly matched rows.
        let sql = "SELECT ts FROM t WHERE sym = -'AAPL'";
        let statements =
            sqlparser::parser::Parser::parse_sql(&sqlparser::dialect::GenericDialect {}, sql)
                .unwrap();
        let sqlparser::ast::Statement::Query(query) = &statements[0] else {
            panic!("not a query")
        };
        let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() else {
            panic!("not a select")
        };
        let error = lower_predicate(select.selection.as_ref().unwrap(), &mut Vec::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("unary minus on a string literal"), "{error}");
        // The unsigned form still works, and still matches.
        assert_eq!(matched("sym = 'AAPL'"), [0, 2]);
    }

    #[test]
    fn boolean_algebra_composes() {
        assert_eq!(matched("ts > 1 AND sym = 'AAPL'"), [2]);
        assert_eq!(matched("x < 0 OR sym = 'TSLA'"), [2, 3]);
        assert_eq!(matched("NOT (sym = 'AAPL')"), [1, 3]);
        assert_eq!(matched("(ts = 1 OR ts = 4) AND x > 0"), [0, 3]);
    }

    #[test]
    fn nulls_match_nothing_even_under_not() {
        assert_eq!(matched("y > 0"), [0, 2]);
        assert_eq!(matched("y <= 0"), [3]);
        // Row 1's y is NULL: neither `y > 0` nor its negation matches it.
        assert_eq!(matched("NOT (y > 0)"), [3]);
    }

    #[test]
    fn is_null_asks_about_presence_not_value() {
        // Row 1 is the only null y. IS NULL is total: the two arms
        // partition the rows, which no value comparison does.
        assert_eq!(matched("y IS NULL"), [1]);
        assert_eq!(matched("y IS NOT NULL"), [0, 2, 3]);
        // NOT over a total test stays total — no row falls through the
        // UNKNOWN gap that `NOT (y > 0)` leaves.
        assert_eq!(matched("NOT (y IS NULL)"), [0, 2, 3]);
        assert_eq!(matched("NOT (y IS NOT NULL)"), [1]);
        // Composition with an ordinary comparison, both ways round.
        assert_eq!(matched("y IS NULL OR y > 20"), [1, 2]);
        assert_eq!(matched("y IS NOT NULL AND ts <= 3"), [0, 2]);
        // A NOT NULL column and a key column both answer it.
        assert_eq!(matched("ts IS NOT NULL"), [0, 1, 2, 3]);
        assert_eq!(matched("x IS NULL"), Vec::<usize>::new());
        assert_eq!(matched("sym IS NOT NULL"), [0, 1, 2, 3]);
        // The paren-wrapped and qualifier-free operand rule.
        assert_eq!(matched("(y) IS NULL"), [1]);
    }

    #[test]
    fn is_null_rejects_what_it_cannot_answer() {
        let sql = "SELECT ts FROM t WHERE x + 1 IS NULL";
        let statements =
            sqlparser::parser::Parser::parse_sql(&sqlparser::dialect::GenericDialect {}, sql)
                .unwrap();
        let sqlparser::ast::Statement::Query(query) = &statements[0] else {
            panic!("not a query")
        };
        let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() else {
            panic!("not a select")
        };
        let error = lower_predicate(select.selection.as_ref().unwrap(), &mut Vec::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("IS NULL on"), "{error}");
        assert!(error.contains("a plain column only"), "{error}");
    }

    #[test]
    fn not_composes_in_three_valued_logic() {
        // B1 regression. `NOT (a AND b)` with a FALSE and b UNKNOWN:
        // FALSE AND UNKNOWN = FALSE, NOT FALSE = TRUE — the row matches.
        // Row 1 (x=2.5, y=NULL): x>100 is FALSE, so the AND is FALSE
        // regardless of y and the NOT is TRUE. The old blanket null-mask
        // wrongly dropped it; all four rows match.
        assert_eq!(matched("NOT (x > 100 AND y > 0)"), [0, 1, 2, 3]);
        // `NOT (a OR b)`: row 1's UNKNOWN stays UNKNOWN under NOT and is
        // excluded; only row 3 (both operands FALSE) survives.
        assert_eq!(matched("NOT (x > 100 OR y > 0)"), [3]);
        // AND with a NULL operand: TRUE AND UNKNOWN = UNKNOWN (row 1 out),
        // FALSE AND UNKNOWN = FALSE (rows 2, 3 out).
        assert_eq!(matched("x > 0 AND y > 0"), [0]);
    }

    #[test]
    fn type_and_scope_errors_are_specific() {
        let (schema, view) = view();
        let check = |predicate: Predicate, needle: &str| {
            let error = evaluate(&predicate, &schema, &view, &NoScalars)
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{error}");
        };
        check(
            Predicate::Compare {
                column: "sym".into(),
                op: CmpOp::Eq,
                value: Number::Int(1),
            },
            "is a key",
        );
        check(
            Predicate::KeyEquals {
                column: "x".into(),
                value: "A".into(),
                negated: false,
            },
            "is numeric",
        );
        check(
            Predicate::Compare {
                column: "nope".into(),
                op: CmpOp::Eq,
                value: Number::Int(1),
            },
            "unknown column",
        );
    }

    #[test]
    fn exact_i64_comparison_survives_beyond_f64_precision() {
        let schema = Schema::new(vec![Field::new("ts", ColumnType::I64, false)]);
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        let big = (1i64 << 53) + 1; // not representable in f64
        buffer.append(&[RowValue::I64(big)]).unwrap();
        buffer.append(&[RowValue::I64(big + 1)]).unwrap();
        let view = SegmentView::all_live(std::sync::Arc::new(buffer.freeze().unwrap()));
        let predicate = Predicate::Compare {
            column: "ts".into(),
            op: CmpOp::Eq,
            value: Number::Int(big),
        };
        let bitmap = evaluate(&predicate, &schema, &view, &NoScalars).unwrap();
        assert!(bitmap.get(0));
        assert!(!bitmap.get(1)); // an f64 round trip would match both
    }

    #[test]
    fn i64_vs_float_literal_is_exact_beyond_f64_precision() {
        // B6: comparing an i64 column to a *float* literal must not cast
        // through f64. 2^53 + 1 and 2^53 - 1 both collapse to 2^53 under
        // `as f64`; the exact relation keeps them on opposite sides of a
        // 2^53 float literal.
        let schema = Schema::new(vec![Field::new("ts", ColumnType::I64, false)]);
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        let two_pow_53 = 1i64 << 53;
        buffer.append(&[RowValue::I64(two_pow_53 + 1)]).unwrap();
        buffer.append(&[RowValue::I64(two_pow_53 - 1)]).unwrap();
        let view = SegmentView::all_live(std::sync::Arc::new(buffer.freeze().unwrap()));
        let gt = Predicate::Compare {
            column: "ts".into(),
            op: CmpOp::Gt,
            value: Number::Float(9_007_199_254_740_992.0), // exactly 2^53
        };
        let bitmap = evaluate(&gt, &schema, &view, &NoScalars).unwrap();
        assert!(bitmap.get(0)); // 2^53 + 1 > 2^53
        assert!(!bitmap.get(1)); // 2^53 - 1 is not > 2^53 (a cast says it is)
    }
}

#[cfg(test)]
mod pruning_tests {
    use storage_lite::SegmentHandle;

    fn handle_of(view: &SegmentView) -> SegmentHandle {
        SegmentHandle::resident(view.segment.clone(), view.live.clone())
    }

    use super::*;
    use arrow_lite::{ColumnType, Field};
    use storage_lite::{RowValue, SegmentView, WriteBuffer};

    fn view(ts: &[i64], x: &[f64]) -> (Schema, SegmentView) {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("sym", ColumnType::Key, false),
        ]);
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        for (&ts, &x) in ts.iter().zip(x) {
            buffer
                .append(&[RowValue::I64(ts), RowValue::F64(x), RowValue::Key("A")])
                .unwrap();
        }
        let segment = std::sync::Arc::new(buffer.freeze().unwrap());
        (schema, SegmentView::all_live(segment))
    }

    fn compare(column: &str, op: CmpOp, value: Number) -> Predicate {
        Predicate::Compare {
            column: column.into(),
            op,
            value,
        }
    }

    #[test]
    fn interval_logic_prunes_exactly_the_impossible() {
        // ts ∈ [10, 20], x ∈ [1.5, 3.5].
        let (schema, view) = view(&[10, 15, 20], &[1.5, 3.5, 2.0]);
        let cases = [
            (compare("ts", CmpOp::Eq, Number::Int(15)), true),
            (compare("ts", CmpOp::Eq, Number::Int(21)), false),
            (compare("ts", CmpOp::Eq, Number::Int(9)), false),
            (compare("ts", CmpOp::Lt, Number::Int(10)), false),
            (compare("ts", CmpOp::Lt, Number::Int(11)), true),
            (compare("ts", CmpOp::Le, Number::Int(10)), true),
            (compare("ts", CmpOp::Gt, Number::Int(20)), false),
            (compare("ts", CmpOp::Ge, Number::Int(20)), true),
            (compare("ts", CmpOp::Ne, Number::Int(15)), true),
            (compare("x", CmpOp::Gt, Number::Float(3.5)), false),
            (compare("x", CmpOp::Ge, Number::Float(3.5)), true),
            (compare("x", CmpOp::Lt, Number::Float(1.5)), false),
        ];
        for (predicate, expected) in cases {
            assert_eq!(
                can_match(&predicate, &schema, &handle_of(&view)),
                expected,
                "{predicate:?}"
            );
        }
        // Boolean structure: AND prunes if either side prunes; OR only
        // if both do; NOT and key predicates never prune.
        let hit = compare("ts", CmpOp::Eq, Number::Int(15));
        let miss = compare("ts", CmpOp::Eq, Number::Int(99));
        let and = Predicate::And(Box::new(hit.clone()), Box::new(miss.clone()));
        assert!(!can_match(&and, &schema, &handle_of(&view)));
        let or = Predicate::Or(Box::new(hit.clone()), Box::new(miss.clone()));
        assert!(can_match(&or, &schema, &handle_of(&view)));
        let both_miss = Predicate::Or(Box::new(miss.clone()), Box::new(miss.clone()));
        assert!(!can_match(&both_miss, &schema, &handle_of(&view)));
        assert!(can_match(
            &Predicate::Not(Box::new(miss)),
            &schema,
            &handle_of(&view)
        ));
        assert!(can_match(
            &Predicate::KeyEquals {
                column: "sym".into(),
                value: "ZZZ".into(),
                negated: false
            },
            &schema,
            &handle_of(&view)
        ));
        // i64 bounds vs a float literal never prune (soundness first).
        assert!(can_match(
            &compare("ts", CmpOp::Eq, Number::Float(9.5)),
            &schema,
            &handle_of(&view)
        ));
    }

    #[test]
    fn null_only_columns_prune_and_nan_rows_do_not() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("y", ColumnType::F64, true),
        ]);
        // Only NULLs: null is not a value, so no comparison can match.
        let mut nulls = WriteBuffer::new(schema.clone(), 0).unwrap();
        nulls.append(&[RowValue::I64(1), RowValue::Null]).unwrap();
        let null_view = SegmentView::all_live(std::sync::Arc::new(nulls.freeze().unwrap()));
        assert!(!can_match(
            &compare("y", CmpOp::Ge, Number::Float(0.0)),
            &schema,
            &handle_of(&null_view)
        ));
        // NULL plus NaN: NaN *is* a value (greater than every number,
        // D2 ruling), so `>=` may match — and does, on the NaN row.
        let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
        buffer.append(&[RowValue::I64(1), RowValue::Null]).unwrap();
        buffer
            .append(&[RowValue::I64(2), RowValue::F64(f64::NAN)])
            .unwrap();
        let view = SegmentView::all_live(std::sync::Arc::new(buffer.freeze().unwrap()));
        let ge = compare("y", CmpOp::Ge, Number::Float(0.0));
        assert!(can_match(&ge, &schema, &handle_of(&view)));
        let matched = evaluate(&ge, &schema, &view, &NoScalars).unwrap();
        assert!(!matched.get(0) && matched.get(1));
        // The <-side still prunes: NaN is never below a number.
        assert!(!can_match(
            &compare("y", CmpOp::Lt, Number::Float(0.0)),
            &schema,
            &handle_of(&view)
        ));
    }

    #[test]
    fn is_not_null_prunes_the_all_null_segment() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("y", ColumnType::F64, true),
            Field::new("sym", ColumnType::Key, false),
        ]);
        let segment = |values: &[Option<f64>]| {
            let mut buffer = WriteBuffer::new(schema.clone(), 0).unwrap();
            for (ts, &y) in values.iter().enumerate() {
                buffer
                    .append(&[
                        RowValue::I64(ts as i64),
                        y.map_or(RowValue::Null, RowValue::F64),
                        RowValue::Key("A"),
                    ])
                    .unwrap();
            }
            SegmentView::all_live(std::sync::Arc::new(buffer.freeze().unwrap()))
        };
        let is_null = |negated| Predicate::IsNull {
            column: "y".into(),
            negated,
        };
        // No valid value anywhere: nothing can be non-null.
        let all_null = segment(&[None, None]);
        assert!(!can_match(&is_null(true), &schema, &handle_of(&all_null)));
        // ... but nulls are what the segment is made of.
        assert!(can_match(&is_null(false), &schema, &handle_of(&all_null)));
        // One value is enough to keep IS NOT NULL alive, and IS NULL
        // never prunes: zone maps count values, not absences.
        let mixed = segment(&[None, Some(1.0)]);
        assert!(can_match(&is_null(true), &schema, &handle_of(&mixed)));
        assert!(can_match(&is_null(false), &schema, &handle_of(&mixed)));
        // Key columns carry no zone map, so neither arm prunes them.
        let keys = Predicate::IsNull {
            column: "sym".into(),
            negated: true,
        };
        assert!(can_match(&keys, &schema, &handle_of(&all_null)));
    }
}
