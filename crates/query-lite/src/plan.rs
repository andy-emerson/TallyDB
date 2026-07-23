//! Parsing and lowering: SQL text → logical plans.
//!
//! sqlparser-rs parses (taken as-is, pinned); the subsetting happens
//! here, in what this lowering accepts. Three statements lower today:
//!
//! ```sql
//! SELECT <columns | window calls | GROUP BY keys + aggregates>
//! FROM table [WHERE predicate] [GROUP BY keys]
//! [ORDER BY column [DESC]] [LIMIT n] [OFFSET n];
//! UPDATE table SET column = literal, ... [WHERE predicate];
//! DELETE FROM table [WHERE predicate];
//! ```
//!
//! (the predicate fragment is documented in [`crate::predicate`];
//! window calls are `fn(args) OVER ([PARTITION BY key] ORDER BY
//! ordering_key ROWS BETWEEN n PRECEDING AND CURRENT ROW)`; aggregates
//! are `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` over plain columns). Everything
//! else — joins, subqueries, HAVING, DISTINCT, other frame shapes, SET
//! or projection expressions beyond literals and columns — is rejected
//! with a message naming what was rejected. The rejection is scope
//! honesty, not a parser limit: those features arrive through this same
//! lowering as M2 proceeds.

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
    /// A function that is neither a standard aggregate nor a registered
    /// window function.
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
            QueryError::UnknownFunction(name) => write!(f, "unknown function '{name}'"),
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
        /// Frame start: this many rows preceding (`None` = UNBOUNDED
        /// PRECEDING), through the current row.
        preceding: Option<usize>,
        /// Output name, if aliased.
        alias: Option<String>,
    },
}

/// A standard SQL aggregate function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AggFunction {
    /// `COUNT(*)` / `COUNT(col)`.
    Count,
    /// `SUM(col)`.
    Sum,
    /// `AVG(col)`.
    Avg,
    /// `MIN(col)`.
    Min,
    /// `MAX(col)`.
    Max,
}

impl AggFunction {
    fn from_name(name: &str) -> Option<AggFunction> {
        match name {
            "count" => Some(AggFunction::Count),
            "sum" => Some(AggFunction::Sum),
            "avg" => Some(AggFunction::Avg),
            "min" => Some(AggFunction::Min),
            "max" => Some(AggFunction::Max),
            _ => None,
        }
    }
}

/// One plain (non-window) aggregate call in an aggregate projection.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AggCall {
    /// The function.
    pub function: AggFunction,
    /// The argument column; `None` is `COUNT(*)`.
    pub argument: Option<String>,
    /// Output name, if aliased.
    pub alias: Option<String>,
}

/// One output column of an aggregate projection, in SELECT-list order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AggItem {
    /// A GROUP BY key, passed through (must appear in the GROUP BY
    /// list).
    Key {
        /// The key column.
        name: String,
        /// Output name, if aliased.
        alias: Option<String>,
    },
    /// An aggregate call.
    Call(AggCall),
}

/// What the SELECT list computes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Projection {
    /// Plain columns and window calls, one output row per input row.
    Items(Vec<PlanItem>),
    /// `GROUP BY` keys and aggregate calls, one output row per group.
    Aggregate {
        /// The GROUP BY key columns (empty = one global group).
        keys: Vec<String>,
        /// The SELECT list.
        items: Vec<AggItem>,
    },
}

/// Top-level `ORDER BY`: one output column, a direction.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OrderBy {
    /// The output column name (after aliasing).
    pub column: String,
    /// `true` for `DESC`.
    pub descending: bool,
}

/// A star-schema equi-join: the fact table joined to one small
/// dimension table on a key column.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JoinPlan {
    /// The dimension table's name.
    pub dimension: String,
    /// The fact-side join column (a key column).
    pub fact_key: String,
    /// The dimension-side join column (a key column, unique per row).
    pub dimension_key: String,
    /// `true` for LEFT (unmatched fact rows keep null dimension cells);
    /// `false` for INNER (unmatched fact rows drop).
    pub left: bool,
}

/// The SELECT plan.
#[derive(Clone, PartialEq, Debug)]
pub struct Plan {
    /// The FROM table's name (resolved to a snapshot by the embedder).
    pub table: String,
    /// The star-schema join, if the query has one.
    pub join: Option<JoinPlan>,
    /// What the SELECT list computes.
    pub projection: Projection,
    /// The WHERE predicate, applied before everything else.
    pub predicate: Option<Predicate>,
    /// Top-level ORDER BY, applied to the projected output.
    pub order_by: Option<OrderBy>,
    /// `LIMIT`, applied last (with `offset`).
    pub limit: Option<usize>,
    /// `OFFSET`.
    pub offset: Option<usize>,
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
    let order_by = lower_order_by(query.order_by.as_ref())?;
    let (limit, offset) = lower_limit(query.limit_clause.as_ref())?;
    let ast::SetExpr::Select(select) = query.body.as_ref() else {
        return Err(QueryError::Unsupported(
            "set operations / VALUES".to_owned(),
        ));
    };
    let mut plan = lower_select(select)?;
    plan.order_by = order_by;
    plan.limit = limit;
    plan.offset = offset;
    Ok(plan)
}

fn lower_order_by(order_by: Option<&ast::OrderBy>) -> Result<Option<OrderBy>, QueryError> {
    let Some(order_by) = order_by else {
        return Ok(None);
    };
    let ast::OrderByKind::Expressions(exprs) = &order_by.kind else {
        return Err(QueryError::Unsupported("ORDER BY ALL".to_owned()));
    };
    match exprs.as_slice() {
        [] => Ok(None),
        [order] => {
            let ast::Expr::Identifier(column) = &order.expr else {
                return Err(QueryError::Unsupported(
                    "ORDER BY must name an output column".to_owned(),
                ));
            };
            Ok(Some(OrderBy {
                column: ident(column),
                descending: order.options.asc == Some(false),
            }))
        }
        _ => Err(QueryError::Unsupported(
            "ORDER BY one column (multi-column ordering not yet lowered)".to_owned(),
        )),
    }
}

fn lower_limit(
    limit_clause: Option<&ast::LimitClause>,
) -> Result<(Option<usize>, Option<usize>), QueryError> {
    let Some(clause) = limit_clause else {
        return Ok((None, None));
    };
    let ast::LimitClause::LimitOffset {
        limit,
        offset,
        limit_by,
    } = clause
    else {
        return Err(QueryError::Unsupported("OFFSET ... FETCH".to_owned()));
    };
    if !limit_by.is_empty() {
        return Err(QueryError::Unsupported("LIMIT ... BY".to_owned()));
    }
    let number = |expr: &ast::Expr, what: &str| -> Result<usize, QueryError> {
        if let ast::Expr::Value(value) = expr {
            if let ast::Value::Number(text, _) = &value.value {
                if let Ok(value) = text.parse::<usize>() {
                    return Ok(value);
                }
            }
        }
        Err(QueryError::Unsupported(format!(
            "{what} must be a non-negative integer literal"
        )))
    };
    let limit = limit
        .as_ref()
        .map(|expr| number(expr, "LIMIT"))
        .transpose()?;
    let offset = offset
        .as_ref()
        .map(|offset| number(&offset.value, "OFFSET"))
        .transpose()?;
    Ok((limit, offset))
}

fn lower_select(select: &ast::Select) -> Result<Plan, QueryError> {
    if select.having.is_some() || select.distinct.is_some() {
        return Err(QueryError::Unsupported("HAVING / DISTINCT".to_owned()));
    }
    let [table] = select.from.as_slice() else {
        return Err(QueryError::Unsupported(format!(
            "exactly one FROM table, got {}",
            select.from.len()
        )));
    };
    let ast::TableFactor::Table { name, alias, .. } = &table.relation else {
        return Err(QueryError::Unsupported(
            "derived tables / table functions".to_owned(),
        ));
    };
    let fact_alias = alias.as_ref().map(|alias| ident(&alias.name));
    let joins = &table.joins;
    let table = object_name(name)?;
    // With a join in play, qualified names (t.col) are stripped up
    // front after validating their qualifier, so the rest of the
    // lowering — and the executor's joined schema — see plain names.
    let (join, projection_exprs, selection_expr) =
        match lower_join(&table, fact_alias.as_deref(), joins)? {
            Some((plan, dimension_alias)) => {
                let mut known: Vec<&str> = vec![&table, &plan.dimension];
                if let Some(alias) = &fact_alias {
                    known.push(alias);
                }
                if let Some(alias) = &dimension_alias {
                    known.push(alias);
                }
                let projection = select
                    .projection
                    .iter()
                    .map(|item| strip_item_qualifiers(item, &known))
                    .collect::<Result<Vec<ast::SelectItem>, QueryError>>()?;
                let selection = select
                    .selection
                    .as_ref()
                    .map(|expr| strip_qualifiers(expr, &known))
                    .transpose()?;
                (Some(plan), projection, selection)
            }
            None => (None, select.projection.clone(), select.selection.clone()),
        };
    let select_projection = &projection_exprs;
    let predicate = selection_expr.as_ref().map(lower_predicate).transpose()?;
    let keys = lower_group_by(&select.group_by)?;
    // An aggregate projection is signaled by GROUP BY or by any plain
    // (no OVER) call to a standard aggregate in the SELECT list.
    let aggregate_shaped = !keys.is_empty()
        || select_projection.iter().any(|item| {
            let expr = match item {
                ast::SelectItem::UnnamedExpr(expr) => expr,
                ast::SelectItem::ExprWithAlias { expr, .. } => expr,
                _ => return false,
            };
            matches!(expr, ast::Expr::Function(function) if function.over.is_none())
        });
    let projection = if aggregate_shaped {
        let mut items = Vec::with_capacity(select_projection.len());
        for item in select_projection {
            items.push(lower_agg_item(item, &keys)?);
        }
        Projection::Aggregate { keys, items }
    } else {
        let mut items = Vec::with_capacity(select_projection.len());
        for item in select_projection {
            items.push(lower_item(item)?);
        }
        Projection::Items(items)
    };
    Ok(Plan {
        table,
        join,
        projection,
        predicate,
        order_by: None,
        limit: None,
        offset: None,
    })
}

/// Rewrites `qualifier.column` to `column` when the qualifier names a
/// table in scope; unknown qualifiers are errors. (Column-name
/// collisions between the two tables are caught when the executor
/// builds the joined schema.)
fn strip_qualifiers(expr: &ast::Expr, known: &[&str]) -> Result<ast::Expr, QueryError> {
    let recurse = |expr: &ast::Expr| strip_qualifiers(expr, known);
    Ok(match expr {
        ast::Expr::CompoundIdentifier(parts) => match parts.as_slice() {
            [qualifier, column] if known.contains(&qualifier.value.as_str()) => {
                ast::Expr::Identifier(column.clone())
            }
            [qualifier, _] => {
                return Err(QueryError::Unsupported(format!(
                    "qualifier '{}' names no table in this query",
                    qualifier.value
                )))
            }
            _ => {
                return Err(QueryError::Unsupported(
                    "column names may carry one table qualifier".to_owned(),
                ))
            }
        },
        ast::Expr::Nested(inner) => ast::Expr::Nested(Box::new(recurse(inner)?)),
        ast::Expr::UnaryOp { op, expr } => ast::Expr::UnaryOp {
            op: *op,
            expr: Box::new(recurse(expr)?),
        },
        ast::Expr::BinaryOp { left, op, right } => ast::Expr::BinaryOp {
            left: Box::new(recurse(left)?),
            op: op.clone(),
            right: Box::new(recurse(right)?),
        },
        ast::Expr::InList {
            expr,
            list,
            negated,
        } => ast::Expr::InList {
            expr: Box::new(recurse(expr)?),
            list: list.iter().map(recurse).collect::<Result<_, _>>()?,
            negated: *negated,
        },
        ast::Expr::Function(function) => {
            let mut function = function.clone();
            if let ast::FunctionArguments::List(list) = &mut function.args {
                for argument in &mut list.args {
                    if let ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(expr)) = argument {
                        *expr = strip_qualifiers(expr, known)?;
                    }
                }
            }
            if let Some(ast::WindowType::WindowSpec(spec)) = &mut function.over {
                for expr in &mut spec.partition_by {
                    *expr = strip_qualifiers(expr, known)?;
                }
                for order in &mut spec.order_by {
                    order.expr = strip_qualifiers(&order.expr, known)?;
                }
            }
            ast::Expr::Function(function)
        }
        other => other.clone(),
    })
}

fn strip_item_qualifiers(
    item: &ast::SelectItem,
    known: &[&str],
) -> Result<ast::SelectItem, QueryError> {
    Ok(match item {
        ast::SelectItem::UnnamedExpr(expr) => {
            ast::SelectItem::UnnamedExpr(strip_qualifiers(expr, known)?)
        }
        ast::SelectItem::ExprWithAlias { expr, alias } => ast::SelectItem::ExprWithAlias {
            expr: strip_qualifiers(expr, known)?,
            alias: alias.clone(),
        },
        other => other.clone(),
    })
}

/// Lowers the optional star-schema join clause; returns the plan and
/// the dimension's alias (for qualified-name resolution).
fn lower_join(
    fact: &str,
    fact_alias: Option<&str>,
    joins: &[ast::Join],
) -> Result<Option<(JoinPlan, Option<String>)>, QueryError> {
    match joins {
        [] => Ok(None),
        [join] => {
            let constraint = match &join.join_operator {
                ast::JoinOperator::Inner(constraint) | ast::JoinOperator::Join(constraint) => {
                    (constraint, false)
                }
                ast::JoinOperator::LeftOuter(constraint) | ast::JoinOperator::Left(constraint) => {
                    (constraint, true)
                }
                other => {
                    return Err(QueryError::Unsupported(format!(
                        "join type {other:?} (INNER and LEFT only)"
                    )))
                }
            };
            let (constraint, left) = constraint;
            let ast::JoinConstraint::On(on) = constraint else {
                return Err(QueryError::Unsupported(
                    "JOIN must use ON fact.key = dim.key".to_owned(),
                ));
            };
            let ast::TableFactor::Table { name, alias, .. } = &join.relation else {
                return Err(QueryError::Unsupported(
                    "JOIN target must be a plain table".to_owned(),
                ));
            };
            let dimension = object_name(name)?;
            let dimension_alias = alias.as_ref().map(|alias| ident(&alias.name));
            // ON: an equality of two (possibly qualified) columns, one
            // per side, in either order.
            let ast::Expr::BinaryOp {
                left: on_left,
                op: ast::BinaryOperator::Eq,
                right: on_right,
            } = on
            else {
                return Err(QueryError::Unsupported(
                    "JOIN ON must be a single equality".to_owned(),
                ));
            };
            let side = |expr: &ast::Expr| -> Result<(Option<String>, String), QueryError> {
                match expr {
                    ast::Expr::Identifier(column) => Ok((None, ident(column))),
                    ast::Expr::CompoundIdentifier(parts) => match parts.as_slice() {
                        [table, column] => Ok((Some(ident(table)), ident(column))),
                        _ => Err(QueryError::Unsupported(
                            "ON columns may carry one table qualifier".to_owned(),
                        )),
                    },
                    other => Err(QueryError::Unsupported(format!(
                        "ON side '{other}' (plain columns only)"
                    ))),
                }
            };
            let (left_side, right_side) = (side(on_left)?, side(on_right)?);
            let is_fact = |qualifier: &Option<String>| {
                qualifier
                    .as_ref()
                    .map(|name| name == fact || fact_alias.is_some_and(|alias| name == alias))
            };
            // Assign sides: qualified names decide; two unqualified
            // names are ambiguous only if they can't be told apart —
            // require at least one qualifier.
            let (fact_key, dimension_key) = match (is_fact(&left_side.0), is_fact(&right_side.0)) {
                (Some(true), Some(false)) | (Some(true), None) | (None, Some(false)) => {
                    (left_side.1, right_side.1)
                }
                (Some(false), Some(true)) | (None, Some(true)) | (Some(false), None) => {
                    (right_side.1, left_side.1)
                }
                _ => {
                    return Err(QueryError::Unsupported(
                        "qualify at least one ON column (fact.key = dim.key)".to_owned(),
                    ))
                }
            };
            Ok(Some((
                JoinPlan {
                    dimension,
                    fact_key,
                    dimension_key,
                    left,
                },
                dimension_alias,
            )))
        }
        _ => Err(QueryError::Unsupported(
            "one JOIN per query (star schema: fact times one dimension at a time)".to_owned(),
        )),
    }
}

fn lower_group_by(group_by: &ast::GroupByExpr) -> Result<Vec<String>, QueryError> {
    let ast::GroupByExpr::Expressions(exprs, modifiers) = group_by else {
        return Err(QueryError::Unsupported("GROUP BY ALL".to_owned()));
    };
    if !modifiers.is_empty() {
        return Err(QueryError::Unsupported(
            "GROUP BY ROLLUP / CUBE / GROUPING SETS".to_owned(),
        ));
    }
    exprs
        .iter()
        .map(|expr| match expr {
            ast::Expr::Identifier(column) => Ok(ident(column)),
            other => Err(QueryError::Unsupported(format!(
                "GROUP BY '{other}' (plain key columns only)"
            ))),
        })
        .collect()
}

fn lower_agg_item(item: &ast::SelectItem, keys: &[String]) -> Result<AggItem, QueryError> {
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
        ast::Expr::Identifier(name) => {
            let name = ident(name);
            if !keys.contains(&name) {
                return Err(QueryError::Unsupported(format!(
                    "column '{name}' must appear in GROUP BY or an aggregate"
                )));
            }
            Ok(AggItem::Key { name, alias })
        }
        ast::Expr::Function(function) if function.over.is_none() => {
            let name = object_name(&function.name)?.to_lowercase();
            let Some(agg) = AggFunction::from_name(&name) else {
                return Err(QueryError::UnknownFunction(name));
            };
            let argument = lower_agg_argument(&function.args, agg)?;
            Ok(AggItem::Call(AggCall {
                function: agg,
                argument,
                alias,
            }))
        }
        other => Err(QueryError::Unsupported(format!(
            "expression '{other}' in an aggregate SELECT list"
        ))),
    }
}

/// `COUNT(*)` has no argument column; everything else takes exactly one
/// plain column.
fn lower_agg_argument(
    args: &ast::FunctionArguments,
    function: AggFunction,
) -> Result<Option<String>, QueryError> {
    let ast::FunctionArguments::List(list) = args else {
        return Err(QueryError::Unsupported(
            "aggregate call without an argument list".to_owned(),
        ));
    };
    match list.args.as_slice() {
        [ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Wildcard)] => {
            if function == AggFunction::Count {
                Ok(None)
            } else {
                Err(QueryError::Unsupported("only COUNT takes '*'".to_owned()))
            }
        }
        [ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(ast::Expr::Identifier(column)))] => {
            Ok(Some(ident(column)))
        }
        other => Err(QueryError::Unsupported(format!(
            "aggregate arguments {other:?} (one plain column, or * for COUNT)"
        ))),
    }
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
            "plain call '{name}' outside an aggregate projection"
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

/// Accepts `ROWS BETWEEN <n | UNBOUNDED> PRECEDING AND CURRENT ROW`;
/// `None` is the unbounded start.
fn lower_frame(frame: Option<&ast::WindowFrame>) -> Result<Option<usize>, QueryError> {
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
    let preceding = match &frame.start_bound {
        ast::WindowFrameBound::Preceding(None) => None, // UNBOUNDED
        ast::WindowFrameBound::Preceding(Some(preceding)) => {
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
            Some(
                number
                    .parse::<usize>()
                    .map_err(|_| QueryError::Unsupported(format!("frame bound '{number}'")))?,
            )
        }
        _ => {
            return Err(QueryError::Unsupported(
                "frame must start at n PRECEDING or UNBOUNDED PRECEDING".to_owned(),
            ))
        }
    };
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
            plan.projection,
            Projection::Items(vec![
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
                    preceding: Some(19),
                    alias: Some("beta".into()),
                },
            ])
        );
    }

    #[test]
    fn plans_without_partition() {
        let plan = plan(
            "SELECT mean(x) OVER (ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
        )
        .expect("plans");
        assert_eq!(
            plan.projection,
            Projection::Items(vec![PlanItem::WindowAgg {
                function: "mean".into(),
                args: vec!["x".into()],
                partition_by: None,
                order_by: "ts".into(),
                preceding: Some(2),
                alias: None,
            }])
        );
    }

    #[test]
    fn rejections_name_the_construct() {
        for (sql, needle) in [
            ("SELECT * FROM t", "wildcard"),
            (
                "SELECT x FROM t JOIN u ON t.a = u.a JOIN v ON t.b = v.b",
                "one JOIN",
            ),
            (
                "SELECT x FROM t RIGHT JOIN u ON t.a = u.a",
                "INNER and LEFT",
            ),
            ("SELECT x FROM t JOIN u ON a = b", "qualify at least one"),
            ("SELECT w.x FROM t JOIN u ON t.a = u.a", "names no table"),
            ("SELECT x FROM t ORDER BY x, y", "one column"),
            ("SELECT x FROM t WHERE x > 1 HAVING x > 2", "HAVING"),
            ("SELECT DISTINCT x FROM t", "DISTINCT"),
            (
                "SELECT x, sum(y) FROM t GROUP BY x, x + 1",
                "plain key columns",
            ),
            ("SELECT x FROM t GROUP BY x LIMIT x", "LIMIT"),
            ("SELECT nope_agg(x) FROM t", "nope_agg"),
            ("SELECT y FROM t GROUP BY x", "must appear in GROUP BY"),
            (
                "SELECT sum(x) OVER (ORDER BY ts RANGE BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
                "ROWS only",
            ),
            ("SELECT sum(x) OVER (ORDER BY ts) FROM t", "without a frame"),
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
