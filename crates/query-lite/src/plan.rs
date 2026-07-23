//! Parsing and lowering: SQL text → the M1 logical plan.
//!
//! sqlparser-rs parses (taken as-is, pinned); the subsetting happens
//! here, in what this lowering accepts. The M1 shape is exactly:
//!
//! ```sql
//! SELECT <column | fn(args) OVER (
//!            [PARTITION BY key] ORDER BY ordering_key
//!            ROWS BETWEEN n PRECEDING AND CURRENT ROW)>, ...
//! FROM table
//! ```
//!
//! Everything else — WHERE, GROUP BY, joins, subqueries, other frame
//! shapes — is rejected with a message naming what was rejected. The
//! rejection is scope honesty, not a parser limit: those features arrive
//! at M2 through this same lowering.

use crate::predicate::{lower_predicate, parse_number, Number, Predicate};
use sqlparser::ast;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::fmt;

/// Why a query could not be planned or executed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum QueryError {
    /// The SQL text did not parse.
    Parse(String),
    /// Parsed, but outside the supported subset; names the construct.
    Unsupported(String),
    /// A referenced column does not exist.
    UnknownColumn(String),
    /// A window function nobody registered.
    UnknownFunction(String),
    /// A column has the wrong type for its role.
    TypeError(String),
    /// The data is not ordered on the window's ORDER BY column.
    Unordered(String),
    /// A registered aggregate failed.
    Compute(String),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::Parse(message) => write!(f, "parse error: {message}"),
            QueryError::Unsupported(what) => write!(f, "unsupported SQL: {what}"),
            QueryError::UnknownColumn(name) => write!(f, "unknown column '{name}'"),
            QueryError::UnknownFunction(name) => write!(f, "unknown window function '{name}'"),
            QueryError::TypeError(message) => write!(f, "type error: {message}"),
            QueryError::Unordered(message) => write!(f, "data not ordered: {message}"),
            QueryError::Compute(message) => write!(f, "compute error: {message}"),
        }
    }
}

impl std::error::Error for QueryError {}

/// One item of the SELECT list.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PlanItem {
    /// A stored column, passed through.
    Column {
        /// Column name in the schema.
        name: String,
        /// Output name, if aliased.
        alias: Option<String>,
    },
    /// A window aggregate over a trailing frame.
    WindowAgg {
        /// Function name, lower-cased (resolved against the registry).
        function: String,
        /// Argument column names, in call order.
        args: Vec<String>,
        /// PARTITION BY column (a key column), if present.
        partition_by: Option<String>,
        /// ORDER BY column — must be the data's ordering key.
        order_by: String,
        /// Frame: this many rows preceding, through the current row.
        preceding: usize,
        /// Output name, if aliased.
        alias: Option<String>,
    },
}

/// The SELECT plan: one table, a list of items.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Plan {
    /// The FROM table's name (resolved to a segment by the embedder).
    pub table: String,
    /// The SELECT list.
    pub items: Vec<PlanItem>,
}

/// A value the right side of `SET column = ...` may hold.
#[derive(Clone, PartialEq, Debug)]
pub enum SetValue {
    /// A numeric literal (for `f64`/`i64` columns).
    Number(Number),
    /// A string literal (for key columns).
    String(String),
    /// `NULL` (for nullable columns).
    Null,
}

/// One `SET column = literal` assignment.
#[derive(Clone, PartialEq, Debug)]
pub struct Assignment {
    /// The column being assigned.
    pub column: String,
    /// The literal assigned to it.
    pub value: SetValue,
}

/// The `UPDATE` plan: tombstone the matched rows, reappend corrected
/// copies (the one mutation mechanism, per the design).
#[derive(Clone, PartialEq, Debug)]
pub struct UpdatePlan {
    /// The table being updated.
    pub table: String,
    /// The assignments, in statement order.
    pub assignments: Vec<Assignment>,
    /// The WHERE predicate; `None` means every row.
    pub predicate: Option<Predicate>,
}

/// The `DELETE` plan: tombstone the matched rows.
#[derive(Clone, PartialEq, Debug)]
pub struct DeletePlan {
    /// The table being deleted from.
    pub table: String,
    /// The WHERE predicate; `None` means every row.
    pub predicate: Option<Predicate>,
}

/// One supported SQL statement, lowered.
#[derive(Clone, PartialEq, Debug)]
pub enum Statement {
    /// A `SELECT`.
    Select(Plan),
    /// An `UPDATE ... SET ... [WHERE ...]`.
    Update(UpdatePlan),
    /// A `DELETE FROM ... [WHERE ...]`.
    Delete(DeletePlan),
}

/// Parses and lowers one SQL statement.
pub fn parse_statement(sql: &str) -> Result<Statement, QueryError> {
    let statements =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| QueryError::Parse(e.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(QueryError::Unsupported(format!(
            "expected exactly one statement, got {}",
            statements.len()
        )));
    };
    match statement {
        ast::Statement::Query(query) => Ok(Statement::Select(lower_query(query)?)),
        ast::Statement::Update(update) => Ok(Statement::Update(lower_update(update)?)),
        ast::Statement::Delete(delete) => Ok(Statement::Delete(lower_delete(delete)?)),
        _ => Err(QueryError::Unsupported(
            "only SELECT, UPDATE, and DELETE statements are supported".to_owned(),
        )),
    }
}

/// Parses and lowers one SELECT statement (mutations go through
/// [`parse_statement`]).
pub fn plan(sql: &str) -> Result<Plan, QueryError> {
    match parse_statement(sql)? {
        Statement::Select(plan) => Ok(plan),
        Statement::Update(_) | Statement::Delete(_) => Err(QueryError::Unsupported(
            "mutations run through the mutation entry point, not query".to_owned(),
        )),
    }
}

fn lower_update(update: &ast::Update) -> Result<UpdatePlan, QueryError> {
    if update.from.is_some() || !update.table.joins.is_empty() {
        return Err(QueryError::Unsupported(
            "UPDATE with FROM or JOIN".to_owned(),
        ));
    }
    let ast::TableFactor::Table { name, .. } = &update.table.relation else {
        return Err(QueryError::Unsupported(
            "UPDATE target must be a plain table".to_owned(),
        ));
    };
    let table = object_name(name)?;
    let assignments = update
        .assignments
        .iter()
        .map(lower_assignment)
        .collect::<Result<Vec<Assignment>, QueryError>>()?;
    if assignments.is_empty() {
        return Err(QueryError::Unsupported("UPDATE without SET".to_owned()));
    }
    let predicate = update.selection.as_ref().map(lower_predicate).transpose()?;
    Ok(UpdatePlan {
        table,
        assignments,
        predicate,
    })
}

fn lower_assignment(assignment: &ast::Assignment) -> Result<Assignment, QueryError> {
    let ast::AssignmentTarget::ColumnName(name) = &assignment.target else {
        return Err(QueryError::Unsupported(
            "SET target must be a plain column".to_owned(),
        ));
    };
    let column = object_name(name)?;
    let ast::Expr::Value(value) = &assignment.value else {
        return Err(QueryError::Unsupported(format!(
            "SET {column} = '{}' — literals only for now",
            assignment.value
        )));
    };
    let value = match &value.value {
        ast::Value::Number(text, _) => SetValue::Number(parse_number(text)?),
        ast::Value::SingleQuotedString(text) => SetValue::String(text.clone()),
        ast::Value::Null => SetValue::Null,
        other => {
            return Err(QueryError::Unsupported(format!(
                "SET {column} = {other} — numbers, strings, and NULL only"
            )))
        }
    };
    Ok(Assignment { column, value })
}

fn lower_delete(delete: &ast::Delete) -> Result<DeletePlan, QueryError> {
    if !delete.tables.is_empty() || delete.using.is_some() {
        return Err(QueryError::Unsupported(
            "multi-table DELETE / USING".to_owned(),
        ));
    }
    let from = match &delete.from {
        ast::FromTable::WithFromKeyword(from) | ast::FromTable::WithoutKeyword(from) => from,
    };
    let [table] = from.as_slice() else {
        return Err(QueryError::Unsupported(
            "DELETE FROM exactly one table".to_owned(),
        ));
    };
    if !table.joins.is_empty() {
        return Err(QueryError::Unsupported("DELETE with JOIN".to_owned()));
    }
    let ast::TableFactor::Table { name, .. } = &table.relation else {
        return Err(QueryError::Unsupported(
            "DELETE target must be a plain table".to_owned(),
        ));
    };
    let predicate = delete.selection.as_ref().map(lower_predicate).transpose()?;
    Ok(DeletePlan {
        table: object_name(name)?,
        predicate,
    })
}

fn lower_query(query: &ast::Query) -> Result<Plan, QueryError> {
    if query.with.is_some() {
        return Err(QueryError::Unsupported("WITH / CTEs".to_owned()));
    }
    if !query.order_by.as_ref().is_none_or(order_by_is_empty) {
        return Err(QueryError::Unsupported(
            "top-level ORDER BY (results keep ingest order in M1)".to_owned(),
        ));
    }
    if query.limit_clause.is_some() {
        return Err(QueryError::Unsupported("LIMIT / OFFSET".to_owned()));
    }
    let ast::SetExpr::Select(select) = query.body.as_ref() else {
        return Err(QueryError::Unsupported(
            "set operations / VALUES".to_owned(),
        ));
    };
    lower_select(select)
}

fn order_by_is_empty(order_by: &ast::OrderBy) -> bool {
    matches!(&order_by.kind, ast::OrderByKind::Expressions(exprs) if exprs.is_empty())
}

fn lower_select(select: &ast::Select) -> Result<Plan, QueryError> {
    if select.selection.is_some() {
        return Err(QueryError::Unsupported("WHERE (arrives at M2)".to_owned()));
    }
    if !matches!(&select.group_by, ast::GroupByExpr::Expressions(exprs, mods) if exprs.is_empty() && mods.is_empty())
    {
        return Err(QueryError::Unsupported(
            "GROUP BY (arrives at M2)".to_owned(),
        ));
    }
    if select.having.is_some() || select.distinct.is_some() {
        return Err(QueryError::Unsupported("HAVING / DISTINCT".to_owned()));
    }
    let [table] = select.from.as_slice() else {
        return Err(QueryError::Unsupported(format!(
            "exactly one FROM table, got {}",
            select.from.len()
        )));
    };
    if !table.joins.is_empty() {
        return Err(QueryError::Unsupported("JOIN (arrives at M2)".to_owned()));
    }
    let ast::TableFactor::Table { name, .. } = &table.relation else {
        return Err(QueryError::Unsupported(
            "derived tables / table functions".to_owned(),
        ));
    };
    let table = object_name(name)?;
    let mut items = Vec::with_capacity(select.projection.len());
    for projection in &select.projection {
        items.push(lower_item(projection)?);
    }
    Ok(Plan { table, items })
}

fn lower_item(item: &ast::SelectItem) -> Result<PlanItem, QueryError> {
    let (expr, alias) = match item {
        ast::SelectItem::UnnamedExpr(expr) => (expr, None),
        ast::SelectItem::ExprWithAlias { expr, alias } => (expr, Some(ident(alias))),
        _ => {
            return Err(QueryError::Unsupported(
                "wildcard projection (name the columns)".to_owned(),
            ))
        }
    };
    match expr {
        ast::Expr::Identifier(name) => Ok(PlanItem::Column {
            name: ident(name),
            alias,
        }),
        ast::Expr::Function(function) => lower_window_call(function, alias),
        other => Err(QueryError::Unsupported(format!(
            "expression '{other}' (columns and window calls only)"
        ))),
    }
}

fn lower_window_call(
    function: &ast::Function,
    alias: Option<String>,
) -> Result<PlanItem, QueryError> {
    let name = object_name(&function.name)?.to_lowercase();
    let Some(over) = &function.over else {
        return Err(QueryError::Unsupported(format!(
            "plain aggregate '{name}' (only window calls with OVER in M1)"
        )));
    };
    let ast::WindowType::WindowSpec(spec) = over else {
        return Err(QueryError::Unsupported("named WINDOW clauses".to_owned()));
    };
    let args = lower_args(&function.args)?;
    let partition_by = match spec.partition_by.as_slice() {
        [] => None,
        [ast::Expr::Identifier(column)] => Some(ident(column)),
        _ => {
            return Err(QueryError::Unsupported(
                "PARTITION BY must be a single column".to_owned(),
            ))
        }
    };
    let [order] = spec.order_by.as_slice() else {
        return Err(QueryError::Unsupported(
            "ORDER BY must be a single column".to_owned(),
        ));
    };
    let ast::Expr::Identifier(order_column) = &order.expr else {
        return Err(QueryError::Unsupported(
            "ORDER BY must be a plain column".to_owned(),
        ));
    };
    if order.options.asc == Some(false) {
        return Err(QueryError::Unsupported("ORDER BY ... DESC".to_owned()));
    }
    let preceding = lower_frame(spec.window_frame.as_ref())?;
    Ok(PlanItem::WindowAgg {
        function: name,
        args,
        partition_by,
        order_by: ident(order_column),
        preceding,
        alias,
    })
}

/// Accepts exactly `ROWS BETWEEN <n> PRECEDING AND CURRENT ROW`.
fn lower_frame(frame: Option<&ast::WindowFrame>) -> Result<usize, QueryError> {
    let Some(frame) = frame else {
        return Err(QueryError::Unsupported(
            "window without a frame (write ROWS BETWEEN n PRECEDING AND CURRENT ROW)".to_owned(),
        ));
    };
    if frame.units != ast::WindowFrameUnits::Rows {
        return Err(QueryError::Unsupported(
            "RANGE / GROUPS frames (ROWS only)".to_owned(),
        ));
    }
    let ast::WindowFrameBound::Preceding(Some(preceding)) = &frame.start_bound else {
        return Err(QueryError::Unsupported(
            "frame must start at n PRECEDING".to_owned(),
        ));
    };
    let ast::Expr::Value(value) = preceding.as_ref() else {
        return Err(QueryError::Unsupported(
            "frame bound must be a literal number".to_owned(),
        ));
    };
    let ast::Value::Number(number, _) = &value.value else {
        return Err(QueryError::Unsupported(
            "frame bound must be a literal number".to_owned(),
        ));
    };
    let preceding: usize = number
        .parse()
        .map_err(|_| QueryError::Unsupported(format!("frame bound '{number}'")))?;
    match &frame.end_bound {
        Some(ast::WindowFrameBound::CurrentRow) => Ok(preceding),
        _ => Err(QueryError::Unsupported(
            "frame must end at CURRENT ROW".to_owned(),
        )),
    }
}

fn lower_args(args: &ast::FunctionArguments) -> Result<Vec<String>, QueryError> {
    let ast::FunctionArguments::List(list) = args else {
        return Err(QueryError::Unsupported(
            "window call without an argument list".to_owned(),
        ));
    };
    list.args
        .iter()
        .map(|arg| match arg {
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(ast::Expr::Identifier(
                column,
            ))) => Ok(ident(column)),
            other => Err(QueryError::Unsupported(format!(
                "argument '{other}' (plain columns only)"
            ))),
        })
        .collect()
}

fn ident(identifier: &ast::Ident) -> String {
    identifier.value.clone()
}

fn object_name(name: &ast::ObjectName) -> Result<String, QueryError> {
    match name.0.as_slice() {
        [part] => part
            .as_ident()
            .map(ident)
            .ok_or_else(|| QueryError::Unsupported(format!("name '{name}'"))),
        _ => Err(QueryError::Unsupported(format!(
            "qualified name '{name}' (single-part names only)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_the_m1_shape() {
        let plan = plan(
            "SELECT ts, sym, regr_slope(y, x) OVER \
             (PARTITION BY sym ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS beta \
             FROM trades",
        )
        .expect("plans");
        assert_eq!(plan.table, "trades");
        assert_eq!(
            plan.items,
            vec![
                PlanItem::Column {
                    name: "ts".into(),
                    alias: None
                },
                PlanItem::Column {
                    name: "sym".into(),
                    alias: None
                },
                PlanItem::WindowAgg {
                    function: "regr_slope".into(),
                    args: vec!["y".into(), "x".into()],
                    partition_by: Some("sym".into()),
                    order_by: "ts".into(),
                    preceding: 19,
                    alias: Some("beta".into()),
                },
            ]
        );
    }

    #[test]
    fn plans_without_partition() {
        let plan = plan(
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
        )
        .expect("plans");
        assert_eq!(
            plan.items,
            vec![PlanItem::WindowAgg {
                function: "mean".into(),
                args: vec!["x".into()],
                partition_by: None,
                order_by: "ts".into(),
                preceding: 2,
                alias: None,
            }]
        );
    }

    #[test]
    fn rejections_name_the_construct() {
        for (sql, needle) in [
            ("SELECT x FROM t WHERE x > 1", "WHERE"),
            ("SELECT x FROM t GROUP BY x", "GROUP BY"),
            ("SELECT * FROM t", "wildcard"),
            ("SELECT x FROM t JOIN u ON t.a = u.a", "JOIN"),
            ("SELECT x FROM t LIMIT 5", "LIMIT"),
            ("SELECT x FROM t ORDER BY x", "top-level ORDER BY"),
            ("SELECT sum(x) FROM t", "OVER"),
            (
                "SELECT sum(x) OVER (ORDER BY ts RANGE BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
                "ROWS only",
            ),
            (
                "SELECT sum(x) OVER (ORDER BY ts) FROM t",
                "without a frame",
            ),
            (
                "SELECT sum(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
                "n PRECEDING",
            ),
            ("INSERT INTO t VALUES (1)", "SELECT"),
        ] {
            let error = plan(sql).expect_err(sql);
            let message = error.to_string();
            assert!(
                message.contains(needle),
                "{sql}: expected '{needle}' in '{message}'"
            );
        }
    }

    #[test]
    fn parse_errors_surface() {
        assert!(matches!(plan("SELEKT nope"), Err(QueryError::Parse(_))));
    }
}
