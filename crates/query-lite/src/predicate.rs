//! Row predicates: the WHERE substrate.
//!
//! Built for `UPDATE`/`DELETE` first (M2.3) and deliberately shaped as
//! the layer `SELECT ... WHERE` will reuse (M2.4): a small predicate
//! tree — numeric comparisons, key string equality and `IN`, `AND` /
//! `OR` / `NOT` — evaluated per segment into a row bitmap.
//!
//! String predicates follow the design's rule for keys: the string test
//! runs **once per distinct dictionary value**, producing a set of
//! allowed codes; rows are then matched by integer set-membership, never
//! by per-row string comparison. Null cells match no predicate (and
//! `NOT` of no-match is still no-match for them), the standard SQL
//! three-valued outcome for the fragment this tree can express.

use crate::plan::QueryError;
use arrow_lite::{Bitmap, Column, NumericData, Schema};
use sqlparser::ast;
use storage_lite::SegmentView;

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
    /// `column = 'v'` / `column <> 'v'` on a key column.
    KeyEquals {
        /// The key column.
        column: String,
        /// The literal.
        value: String,
        /// `true` for `<>`.
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
pub fn lower_predicate(expr: &ast::Expr) -> Result<Predicate, QueryError> {
    match expr {
        ast::Expr::Nested(inner) => lower_predicate(inner),
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Not,
            expr,
        } => Ok(Predicate::Not(Box::new(lower_predicate(expr)?))),
        ast::Expr::BinaryOp { left, op, right } => match op {
            ast::BinaryOperator::And => Ok(Predicate::And(
                Box::new(lower_predicate(left)?),
                Box::new(lower_predicate(right)?),
            )),
            ast::BinaryOperator::Or => Ok(Predicate::Or(
                Box::new(lower_predicate(left)?),
                Box::new(lower_predicate(right)?),
            )),
            _ => lower_comparison(left, op, right),
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
            "predicate '{other}' (comparisons, IN, AND/OR/NOT only)"
        ))),
    }
}

fn lower_comparison(
    left: &ast::Expr,
    op: &ast::BinaryOperator,
    right: &ast::Expr,
) -> Result<Predicate, QueryError> {
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
    let ast::Expr::Identifier(column) = left else {
        return Err(QueryError::Unsupported(
            "predicate must compare a plain column to a literal".to_owned(),
        ));
    };
    let ast::Expr::Value(value) = right else {
        return Err(QueryError::Unsupported(
            "predicate must compare a plain column to a literal".to_owned(),
        ));
    };
    match &value.value {
        ast::Value::Number(text, _) => {
            let value = parse_number(text)?;
            Ok(Predicate::Compare {
                column: column.value.clone(),
                op,
                value,
            })
        }
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
        other => Err(QueryError::Unsupported(format!(
            "literal '{other}' in a predicate"
        ))),
    }
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

/// Evaluates `predicate` over one segment view, returning a bitmap over
/// the segment's rows (`true` = matched). Tombstoned rows are evaluated
/// like any other — callers combine with the live mask; this keeps the
/// result independent of mutation state.
pub fn evaluate(
    predicate: &Predicate,
    schema: &Schema,
    view: &SegmentView,
) -> Result<Bitmap, QueryError> {
    let batch = view.segment.batch();
    let rows = batch.num_rows();
    match predicate {
        Predicate::Compare { column, op, value } => {
            let index = column_index(schema, column)?;
            match &batch.columns()[index] {
                Column::Numeric(NumericData::F64(numeric)) => {
                    let values = numeric.values().as_slice();
                    let target = value.as_f64();
                    Ok(Bitmap::from_bools((0..rows).map(|row| {
                        numeric.is_valid(row) && op.holds(values[row], target)
                    })))
                }
                Column::Numeric(NumericData::I64(numeric)) => {
                    let values = numeric.values().as_slice();
                    Ok(match value {
                        // Exact integer comparison — no f64 round trip.
                        Number::Int(target) => Bitmap::from_bools(
                            (0..rows)
                                .map(|row| numeric.is_valid(row) && op.holds(values[row], *target)),
                        ),
                        Number::Float(target) => Bitmap::from_bools((0..rows).map(|row| {
                            numeric.is_valid(row) && op.holds(values[row] as f64, *target)
                        })),
                    })
                }
                Column::Key(_) => Err(QueryError::TypeError(format!(
                    "column '{column}' is a key; compare it to a string"
                ))),
            }
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
        Predicate::And(left, right) => {
            Ok(evaluate(left, schema, view)?.and(&evaluate(right, schema, view)?))
        }
        Predicate::Or(left, right) => {
            Ok(evaluate(left, schema, view)?.or(&evaluate(right, schema, view)?))
        }
        Predicate::Not(inner) => {
            // NOT flips matched rows, but a null cell matches nothing on
            // either side — mask nulls back out afterward.
            let flipped = evaluate(inner, schema, view)?.not();
            Ok(mask_out_nulls(inner, schema, view, flipped)?)
        }
    }
}

/// The string test run once per distinct dictionary value, applied to
/// rows as integer set-membership.
fn key_membership(
    schema: &Schema,
    view: &SegmentView,
    column: &str,
    values: &[String],
    negated: bool,
) -> Result<Bitmap, QueryError> {
    let index = column_index(schema, column)?;
    let Column::Key(keys) = &view.segment.batch().columns()[index] else {
        return Err(QueryError::TypeError(format!(
            "column '{column}' is numeric; compare it to a number"
        )));
    };
    let dictionary = keys.dictionary();
    let allowed: Vec<bool> = (0..dictionary.len() as u32)
        .map(|code| {
            let hit = values.iter().any(|value| value == dictionary.value(code));
            hit != negated
        })
        .collect();
    let codes = keys.codes().as_slice();
    Ok(Bitmap::from_bools((0..keys.len()).map(|row| {
        keys.is_valid(row) && allowed[codes[row] as usize]
    })))
}

/// Clears bits for rows where any column `inner` touches is null — the
/// three-valued-logic repair for `NOT`.
fn mask_out_nulls(
    inner: &Predicate,
    schema: &Schema,
    view: &SegmentView,
    mut bitmap: Bitmap,
) -> Result<Bitmap, QueryError> {
    let mut columns = Vec::new();
    collect_columns(inner, &mut columns);
    for column in columns {
        let index = column_index(schema, &column)?;
        let stored = &view.segment.batch().columns()[index];
        let validity = match stored {
            Column::Numeric(numeric) => numeric.validity(),
            Column::Key(keys) => keys.validity(),
        };
        if let Some(validity) = validity {
            bitmap = bitmap.and(validity);
        }
    }
    Ok(bitmap)
}

fn collect_columns(predicate: &Predicate, out: &mut Vec<String>) {
    match predicate {
        Predicate::Compare { column, .. }
        | Predicate::KeyEquals { column, .. }
        | Predicate::KeyIn { column, .. } => out.push(column.clone()),
        Predicate::And(left, right) | Predicate::Or(left, right) => {
            collect_columns(left, out);
            collect_columns(right, out);
        }
        Predicate::Not(inner) => collect_columns(inner, out),
    }
}

fn column_index(schema: &Schema, name: &str) -> Result<usize, QueryError> {
    schema
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .ok_or_else(|| QueryError::UnknownColumn(name.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_lite::{ColumnType, Field};
    use storage_lite::{RowValue, WriteBuffer};

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
        let predicate = lower_predicate(select.selection.as_ref().unwrap()).unwrap();
        let bitmap = evaluate(&predicate, &schema, &view).unwrap();
        (0..4).filter(|&row| bitmap.get(row)).collect()
    }

    #[test]
    fn numeric_comparisons_match_by_value() {
        assert_eq!(matched("ts >= 3"), [2, 3]);
        assert_eq!(matched("x < 0"), [2]);
        assert_eq!(matched("x = 2.5"), [1]);
        assert_eq!(matched("ts <> 2"), [0, 2, 3]);
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
    fn type_and_scope_errors_are_specific() {
        let (schema, view) = view();
        let check = |predicate: Predicate, needle: &str| {
            let error = evaluate(&predicate, &schema, &view)
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
        let bitmap = evaluate(&predicate, &schema, &view).unwrap();
        assert!(bitmap.get(0));
        assert!(!bitmap.get(1)); // an f64 round trip would match both
    }
}
