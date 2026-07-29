//! Parsing and lowering: SQL text → logical plans.
//!
//! sqlparser-rs parses (taken as-is, pinned); the subsetting happens
//! here, in what this lowering accepts. Five statements lower today:
//!
//! ```sql
//! SELECT [DISTINCT] <columns | scalar expressions | CASE | window calls
//!                    | GROUP BY keys + aggregates>
//! FROM fact [[LEFT] JOIN dim ON fact.key = dim.key]
//! [WHERE predicate] [GROUP BY keys [HAVING predicate]]
//! [ORDER BY column [DESC] [NULLS FIRST|LAST]] [LIMIT n] [OFFSET n];
//! CREATE TABLE t (col BIGINT|DOUBLE|KEY [NOT NULL|ORDERING KEY], ...);
//! INSERT INTO t [(columns)] VALUES (literals), ...;
//! UPDATE table SET column = literal, ... [WHERE predicate];
//! DELETE FROM table [WHERE predicate];
//! ```
//!
//! (the predicate fragment is documented in [`crate::predicate`];
//! window calls are `fn(args) OVER ([PARTITION BY key] ORDER BY
//! ordering_key ROWS BETWEEN n PRECEDING AND CURRENT ROW)`; aggregates
//! are `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` over plain columns). Everything
//! else — extra joins, subqueries, CTEs, other frame shapes — is
//! rejected with a message naming what was rejected. The rejection is
//! scope honesty, not a parser limit: what the inclusion principle
//! admits arrives through this same lowering.

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
#[derive(Clone, PartialEq, Debug)]
pub enum PlanItem {
    /// A stored column, passed through.
    Column {
        /// Column name in the schema.
        name: String,
        /// Output name, if aliased.
        alias: Option<String>,
    },
    /// A computed scalar projection (`x + 1`, `ABS(x)`, `CASE ...`).
    Computed {
        /// The expression.
        expr: ScalarExpr,
        /// The output column name: the alias, or the expression's SQL
        /// text when unaliased.
        name: String,
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

/// A scalar arithmetic operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (IEEE: `x/0` is ±inf or NaN, never an error — NaN is a value)
    Div,
    /// `%` (f64 remainder)
    Mod,
}

/// A built-in scalar function of the projection slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarFunction {
    /// `ABS(x)`
    Abs,
    /// `ROUND(x)` — half away from zero
    Round,
    /// `FLOOR(x)`
    Floor,
    /// `CEIL(x)` / `CEILING(x)`
    Ceil,
    /// `SQRT(x)` — IEEE: negative input yields NaN
    Sqrt,
    /// `LN(x)`
    Ln,
    /// `EXP(x)`
    Exp,
    /// `POWER(x, y)`
    Power,
}

impl ScalarFunction {
    fn from_name(name: &str) -> Option<(ScalarFunction, usize)> {
        Some(match name {
            "abs" => (ScalarFunction::Abs, 1),
            "round" => (ScalarFunction::Round, 1),
            "floor" => (ScalarFunction::Floor, 1),
            "ceil" | "ceiling" => (ScalarFunction::Ceil, 1),
            "sqrt" => (ScalarFunction::Sqrt, 1),
            "ln" => (ScalarFunction::Ln, 1),
            "exp" => (ScalarFunction::Exp, 1),
            "power" | "pow" => (ScalarFunction::Power, 2),
            _ => return None,
        })
    }
}

/// A scalar expression over one row — the computed-projection slot
/// (#49; also the seam #53's Lua scalar functions will plug into).
/// Everything computes in `f64` under three-valued logic: a NULL
/// operand makes the result NULL. `i64` columns are refused loudly for
/// now (exact integer expression arithmetic is #40's territory); key
/// columns are refused by numeric-or-key (no string production).
#[derive(Clone, PartialEq, Debug)]
pub enum ScalarExpr {
    /// A stored `f64` column's value.
    Column(String),
    /// A numeric literal.
    Literal(f64),
    /// Unary minus.
    Negate(Box<ScalarExpr>),
    /// A binary arithmetic operation.
    Binary {
        /// The operator.
        op: ArithOp,
        /// Left operand.
        left: Box<ScalarExpr>,
        /// Right operand.
        right: Box<ScalarExpr>,
    },
    /// A built-in scalar function call.
    Call {
        /// The function.
        function: ScalarFunction,
        /// Arguments, in call order.
        args: Vec<ScalarExpr>,
    },
    /// `CASE WHEN p THEN e ... [ELSE e] END` — conditions are the WHERE
    /// grammar (three-valued: an UNKNOWN condition falls through), a
    /// missing ELSE yields NULL.
    Case {
        /// The WHEN arms, in order.
        whens: Vec<(crate::Predicate, ScalarExpr)>,
        /// The ELSE arm.
        otherwise: Option<Box<ScalarExpr>>,
    },
    /// A call to an embedder-registered column function (the vectorized
    /// per-row extension slot, #53). The name resolves against the
    /// registry at execution — a name registered on one table is loudly
    /// unknown on another — so plan time carries it verbatim.
    Registered {
        /// The registered name, lower-cased.
        name: String,
        /// Arguments, in call order — any scalar expressions.
        args: Vec<ScalarExpr>,
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
#[derive(Clone, PartialEq, Debug)]
pub enum Projection {
    /// Plain columns and window calls, one output row per input row.
    Items(Vec<PlanItem>),
    /// `GROUP BY` keys and aggregate calls, one output row per group.
    Aggregate {
        /// The GROUP BY key columns (empty = one global group).
        keys: Vec<String>,
        /// The SELECT list.
        items: Vec<AggItem>,
        /// The `HAVING` filter, if present.
        having: Option<Having>,
    },
}

/// A lowered `HAVING` clause: every aggregate call it references is
/// computed as a hidden output column (aliased `__having{i}`, dropped
/// after filtering), so the filter itself is ordinary WHERE grammar
/// over the aggregate output row — group keys included. Standard SQL
/// semantics: a group survives only where the predicate is TRUE
/// (UNKNOWN filters, like WHERE).
#[derive(Clone, PartialEq, Debug)]
pub struct Having {
    /// The hidden aggregate columns.
    pub items: Vec<AggItem>,
    /// The filter, referencing output names (hidden ones included).
    pub predicate: Predicate,
}

/// Top-level `ORDER BY`: one output column, a direction.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OrderBy {
    /// The output column name (after aliasing).
    pub column: String,
    /// `true` for `DESC`.
    pub descending: bool,
    /// Explicit `NULLS FIRST` / `NULLS LAST`; `None` keeps the default
    /// (nulls last in both directions, DuckDB's convention — D2).
    pub nulls_first: Option<bool>,
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
    /// `SELECT DISTINCT`: deduplicate the projected rows (plain column
    /// projections only; keys compare by value across segments, `f64`
    /// under the one comparison relation — NaN equals itself — and
    /// NULLs equal, per SQL DISTINCT).
    pub distinct: bool,
    /// The WHERE predicate, applied before everything else.
    pub predicate: Option<Predicate>,
    /// Top-level ORDER BY, applied to the projected output.
    pub order_by: Option<OrderBy>,
    /// `LIMIT`, applied last (with `offset`).
    pub limit: Option<usize>,
    /// `OFFSET`.
    pub offset: Option<usize>,
    /// Knowledge-time travel: `ASOF n` / `FOR SYSTEM_TIME AS OF n` —
    /// read the table as it was known at ingest sequence `n`. The
    /// embedder resolves it to an as-of snapshot instead of the latest
    /// one; the executor itself never looks at this field.
    pub as_of: Option<u64>,
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
    /// A `SELECT`. Boxed: a `Plan` (with its projection expressions)
    /// dwarfs the mutation variants, and statements are moved around.
    Select(Box<Plan>),
    /// A `CREATE TABLE`.
    CreateTable(CreateTablePlan),
    /// An `INSERT INTO ... VALUES ...`.
    Insert(InsertPlan),
    /// An `UPDATE ... SET ... [WHERE ...]`.
    Update(UpdatePlan),
    /// A `DELETE FROM ... [WHERE ...]`.
    Delete(DeletePlan),
}

/// One column of a `CREATE TABLE` plan.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ColumnSpec {
    /// The column name.
    pub name: String,
    /// `"BIGINT"`, `"DOUBLE"`, or `"KEY"` — resolved to the engine's
    /// column types by the embedder (query-lite stays schema-agnostic).
    pub type_name: String,
    /// `NOT NULL` present (the ordering key implies it).
    pub not_null: bool,
    /// `ORDERING KEY` present.
    pub ordering_key: bool,
}

/// A lowered `CREATE TABLE`: the DDL surface of the stdlib table (#49,
/// ruled 2026-07-27) — standard names where standard exists (`BIGINT`,
/// `DOUBLE`), the coined `KEY` for dictionary keys, the ordering key
/// declared like a constraint. `VARCHAR`/`TEXT` are refused with a
/// teaching error: keys are interned labels, not text values.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CreateTablePlan {
    /// The table name.
    pub table: String,
    /// The columns, in declared order.
    pub columns: Vec<ColumnSpec>,
}

/// A cell literal of an `INSERT ... VALUES` row.
#[derive(Clone, PartialEq, Debug)]
pub enum InsertValue {
    /// A numeric literal.
    Number(Number),
    /// A string literal (a key value).
    String(String),
    /// `NULL`.
    Null,
}

/// A lowered `INSERT INTO ... [(columns)] VALUES (...), (...)`.
#[derive(Clone, PartialEq, Debug)]
pub struct InsertPlan {
    /// The table name.
    pub table: String,
    /// The named column order, if the statement gave one; `None`
    /// means the schema's declared order.
    pub columns: Option<Vec<String>>,
    /// The literal rows.
    pub rows: Vec<Vec<InsertValue>>,
}

fn lower_create_table(create: &ast::CreateTable) -> Result<CreateTablePlan, QueryError> {
    let table = object_name(&create.name)?;
    if create.if_not_exists || create.or_replace {
        return Err(QueryError::Unsupported(
            "IF NOT EXISTS / OR REPLACE".to_owned(),
        ));
    }
    if let Some(constraint) = create.constraints.first() {
        // Out-of-scope constructs are refused, never dropped; silence
        // here would let `UNIQUE (x)` vanish without a trace.
        return Err(QueryError::Unsupported(format!(
            "table-level constraint '{constraint}' (columns carry the only \
             constraints here: NOT NULL and ORDERING KEY)"
        )));
    }
    let mut columns = Vec::with_capacity(create.columns.len());
    for column in &create.columns {
        let type_name = match &column.data_type {
            ast::DataType::BigInt(_) | ast::DataType::Int8(_) => "BIGINT",
            ast::DataType::Double(_) | ast::DataType::DoublePrecision | ast::DataType::Float8 => {
                "DOUBLE"
            }
            ast::DataType::Custom(name, _) if object_name(name)?.eq_ignore_ascii_case("key") => {
                "KEY"
            }
            ast::DataType::Varchar(_)
            | ast::DataType::Text
            | ast::DataType::Char(_)
            | ast::DataType::String(_) => {
                return Err(QueryError::Unsupported(format!(
                    "column '{}': strings are not a column type here — keys are \
                     interned labels used for filtering, grouping, and joining; \
                     declare it KEY",
                    ident(&column.name)
                )))
            }
            other => {
                return Err(QueryError::Unsupported(format!(
                    "column type '{other}' (BIGINT, DOUBLE, or KEY)"
                )))
            }
        };
        let mut not_null = false;
        let mut ordering_key = false;
        for option in &column.options {
            match &option.option {
                ast::ColumnOption::NotNull => not_null = true,
                ast::ColumnOption::Null => {}
                // The rewrite carries ORDERING KEY through the parser as
                // PRIMARY KEY (user-typed PRIMARY KEY was refused at the
                // door); map the carrier back.
                ast::ColumnOption::PrimaryKey(_) => {
                    ordering_key = true;
                    not_null = true; // the ordering key is NOT NULL
                }
                other => {
                    return Err(QueryError::Unsupported(format!("column option '{other}'")));
                }
            }
        }
        columns.push(ColumnSpec {
            name: ident(&column.name),
            type_name: type_name.to_owned(),
            not_null,
            ordering_key,
        });
    }
    for (index, column) in columns.iter().enumerate() {
        // A duplicated name would make the later column silently
        // unreachable (every resolver takes the first match).
        if columns[..index]
            .iter()
            .any(|other| other.name == column.name)
        {
            return Err(QueryError::Unsupported(format!(
                "column '{}' is declared twice",
                column.name
            )));
        }
        // The pseudocolumn is the engine's to define; a declared one
        // would shadow it and mean something else entirely.
        if column.name == SEQUENCE_COLUMN {
            return Err(QueryError::Unsupported(format!(
                "column '{SEQUENCE_COLUMN}' is reserved — it is the ingest-sequence \
                 pseudocolumn every table already has"
            )));
        }
    }
    match columns.iter().filter(|column| column.ordering_key).count() {
        1 => Ok(CreateTablePlan { table, columns }),
        0 => Err(QueryError::Unsupported(
            "declare exactly one ORDERING KEY column (the BIGINT column \
             ingest arrives roughly sorted on)"
                .to_owned(),
        )),
        _ => Err(QueryError::Unsupported(
            "more than one ORDERING KEY column".to_owned(),
        )),
    }
}

/// The ingest-sequence pseudocolumn's fixed name (#75, ruled
/// 2026-07-29). Every row has a birth sequence — the coordinate `AS OF`
/// addresses — and this is how SQL reads it back. It is never declared:
/// `CREATE TABLE` refuses a column of this name, so the pseudocolumn
/// can never be shadowed, and since `SELECT *` is refused nobody meets
/// it unbidden.
///
/// The short underscore spelling (over a spelled-out
/// `ingest_sequence`) follows from that invisibility: a name users do
/// not normally see is better short and unlikely to collide than
/// long and self-explaining.
pub const SEQUENCE_COLUMN: &str = "_seq";

/// The error for a name that is not a column of the schema — with the
/// pseudocolumn's own answer, since every resolver that reaches here has
/// already passed the one place `_seq` exists (projection).
pub(crate) fn no_such_column(name: &str) -> QueryError {
    if name == SEQUENCE_COLUMN {
        QueryError::Unsupported(format!(
            "'{SEQUENCE_COLUMN}' can be selected, not filtered or grouped on \
             (project it, then order or page by it; `AS OF` is how a coordinate \
             filters)"
        ))
    } else {
        QueryError::UnknownColumn(name.to_owned())
    }
}

fn lower_insert(insert: &ast::Insert) -> Result<InsertPlan, QueryError> {
    let ast::TableObject::TableName(name) = &insert.table else {
        return Err(QueryError::Unsupported(
            "INSERT into something other than a table".to_owned(),
        ));
    };
    let table = object_name(name)?;
    let columns = if insert.columns.is_empty() {
        None
    } else {
        Some(
            insert
                .columns
                .iter()
                .map(object_name)
                .collect::<Result<Vec<String>, QueryError>>()?,
        )
    };
    let Some(source) = &insert.source else {
        return Err(QueryError::Unsupported("INSERT without VALUES".to_owned()));
    };
    let ast::SetExpr::Values(values) = source.body.as_ref() else {
        return Err(QueryError::Unsupported(
            "INSERT ... SELECT (VALUES only)".to_owned(),
        ));
    };
    let mut rows = Vec::with_capacity(values.rows.len());
    for row in &values.rows {
        let mut cells = Vec::with_capacity(row.content.len());
        for cell in &row.content {
            cells.push(lower_insert_value(cell)?);
        }
        rows.push(cells);
    }
    if rows.is_empty() {
        return Err(QueryError::Unsupported("INSERT of zero rows".to_owned()));
    }
    Ok(InsertPlan {
        table,
        columns,
        rows,
    })
}

fn lower_insert_value(expr: &ast::Expr) -> Result<InsertValue, QueryError> {
    match expr {
        ast::Expr::Value(value) => match &value.value {
            ast::Value::Number(text, _) => {
                Ok(InsertValue::Number(crate::predicate::parse_number(text)?))
            }
            ast::Value::SingleQuotedString(text) => Ok(InsertValue::String(text.clone())),
            ast::Value::Null => Ok(InsertValue::Null),
            other => Err(QueryError::Unsupported(format!("INSERT literal '{other}'"))),
        },
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Minus,
            expr,
        } => match lower_insert_value(expr)? {
            InsertValue::Number(Number::Int(value)) => Ok(InsertValue::Number(Number::Int(-value))),
            InsertValue::Number(Number::Float(value)) => {
                Ok(InsertValue::Number(Number::Float(-value)))
            }
            _ => Err(QueryError::Unsupported(
                "negation of a non-number".to_owned(),
            )),
        },
        other => Err(QueryError::Unsupported(format!(
            "INSERT expression '{other}' (literals only)"
        ))),
    }
}

/// Carries the ruled `ORDERING KEY` syntax through a parser that does
/// not know the phrase: outside quotes, user-typed `PRIMARY KEY` is
/// refused with a teaching error (the ordering key is *not* a
/// uniqueness constraint — duplicate ordering-key values are
/// first-class here), then `ORDERING KEY` rewrites to `PRIMARY KEY`
/// as the internal carrier the parser accepts; the lowering maps the
/// carrier back. Only `CREATE TABLE` statements are rewritten.
fn rewrite_ordering_key(sql: &str) -> Result<String, QueryError> {
    let chars: Vec<char> = sql.chars().collect();
    let mut out = String::with_capacity(sql.len());
    let mut quote: Option<char> = None;
    let mut index = 0;
    // Matches `word_a <whitespace> word_b` at a char position, entirely
    // in chars (never byte offsets — the two must not mix), returning
    // the phrase's char length. Both words must sit on word boundaries.
    let phrase_at = |index: usize, word_a: &str, word_b: &str| -> Option<usize> {
        let word = |at: usize, word: &str| -> bool {
            word.chars().enumerate().all(|(offset, w)| {
                chars
                    .get(at + offset)
                    .is_some_and(|c| c.eq_ignore_ascii_case(&w))
            })
        };
        if !word(index, word_a) {
            return None;
        }
        let mut spaces = 0;
        while chars
            .get(index + word_a.len() + spaces)
            .is_some_and(|c| c.is_whitespace())
        {
            spaces += 1;
        }
        if spaces == 0 || !word(index + word_a.len() + spaces, word_b) {
            return None;
        }
        let end = word_a.len() + spaces + word_b.len();
        let boundary = |position: usize| {
            chars
                .get(index + position)
                .is_none_or(|c| !c.is_alphanumeric() && *c != '_')
        };
        let start_ok = index == 0 || !chars[index - 1].is_alphanumeric() && chars[index - 1] != '_';
        // A column *definition* also starts `word KEY` — `ordering KEY,`
        // declares a key column named `ordering`. A constraint never
        // opens a definition, so the phrase only counts when the
        // preceding token is not `(` or `,` (nor the statement start,
        // where no column list is open yet).
        let previous = chars[..index].iter().rev().find(|c| !c.is_whitespace());
        let constraint_position = !matches!(previous, None | Some('(') | Some(','));
        (start_ok && boundary(end) && constraint_position).then_some(end)
    };
    while index < chars.len() {
        let c = chars[index];
        match quote {
            Some(open) => {
                if c == open {
                    quote = None;
                }
                out.push(c);
                index += 1;
            }
            None => {
                if c == '\'' || c == '"' {
                    quote = Some(c);
                    out.push(c);
                    index += 1;
                } else if phrase_at(index, "primary", "key").is_some() {
                    return Err(QueryError::Unsupported(
                        "PRIMARY KEY: TallyDB's ordering key is not a uniqueness \
                         constraint (duplicate ordering-key values are first-class) — \
                         declare the ingest-order column with ORDERING KEY"
                            .to_owned(),
                    ));
                } else if let Some(end) = phrase_at(index, "ordering", "key") {
                    out.push_str("PRIMARY KEY");
                    index += end;
                } else {
                    out.push(c);
                    index += 1;
                }
            }
        }
    }
    Ok(out)
}

/// Whether the statement's first two words are `CREATE TABLE` — the
/// rewrite gate. Word-wise, not a byte prefix: `CREATE  TABLE` and
/// `CREATE\tTABLE` are the same statement and must hit the same gate,
/// or user-typed `PRIMARY KEY` would slip past its refusal.
fn is_create_table(sql: &str) -> bool {
    let mut words = sql.split_whitespace();
    matches!(
        (words.next(), words.next()),
        (Some(create), Some(table))
            if create.eq_ignore_ascii_case("create") && table.eq_ignore_ascii_case("table")
    )
}

/// Splits the knowledge-time clause out of the SQL text before parsing:
/// `ASOF <n>` (the engine's one-word spelling — `ASOF JOIN` is the same
/// keyword followed by `JOIN` and is left alone) and the SQL:2011
/// `FOR SYSTEM_TIME AS OF <n>`, both accepted, both meaning "the table
/// as known at ingest sequence n". Returns the SQL with the clause
/// spliced out (`None` when the text held no clause — untouched input is
/// never rewritten) and the cut it named. The two-word near-miss
/// `AS OF <n>` collides with SQL's alias grammar (`AS OF` parses as an
/// alias named OF), so it gets a teaching error instead of a puzzle.
fn extract_as_of(sql: &str) -> Result<(Option<String>, Option<u64>), QueryError> {
    // Shallow tokenization, recording each token's byte span: quoted runs
    // ('…' with '' escapes, "…") stay single tokens so nothing inside a
    // string literal can look like a clause; comments are skipped whole so
    // nothing inside one does either; hugging punctuation splits off so
    // `ASOF 5,` scans. Spans matter: the clause is spliced out of the
    // ORIGINAL text (below), never reassembled from tokens — reassembly
    // would collapse the newline that terminates a `--` comment and
    // silently comment out the rest of the statement.
    let mut tokens: Vec<&str> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut chars = sql.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if ch == '-' && chars.peek().is_some_and(|&(_, next)| next == '-') {
            for (_, c) in chars.by_ref() {
                if c == '\n' {
                    break;
                }
            }
            continue;
        }
        if ch == '/' && chars.peek().is_some_and(|&(_, next)| next == '*') {
            chars.next();
            let mut previous = '\0';
            for (_, c) in chars.by_ref() {
                if previous == '*' && c == '/' {
                    break;
                }
                previous = c;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            let mut end = sql.len();
            while let Some((i, c)) = chars.next() {
                if c == ch {
                    if ch == '\'' && chars.peek().is_some_and(|&(_, next)| next == '\'') {
                        chars.next();
                        continue;
                    }
                    end = i + c.len_utf8();
                    break;
                }
            }
            tokens.push(&sql[start..end]);
            spans.push((start, end));
            continue;
        }
        if "(),;".contains(ch) {
            let end = start + ch.len_utf8();
            tokens.push(&sql[start..end]);
            spans.push((start, end));
            continue;
        }
        let mut end = sql.len();
        while let Some(&(i, c)) = chars.peek() {
            if c.is_whitespace() || c == '\'' || c == '"' || "(),;".contains(c) {
                end = i;
                break;
            }
            chars.next();
        }
        tokens.push(&sql[start..end]);
        spans.push((start, end));
    }
    let lower: Vec<String> = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect();
    let parse_cut = |token: &str| -> Result<u64, QueryError> {
        token.parse::<u64>().map_err(|_| {
            QueryError::Unsupported(format!(
                "ASOF expects a non-negative integer ingest-sequence literal, got '{token}' \
                 (ASOF is a clause keyword here, so it cannot also name a column)"
            ))
        })
    };
    let mut cut: Option<u64> = None;
    let mut removed: Option<(usize, usize)> = None;
    let mut index = 0;
    while index < tokens.len() {
        let matched = if index + 4 < tokens.len()
            && lower[index] == "for"
            && lower[index + 1] == "system_time"
            && lower[index + 2] == "as"
            && lower[index + 3] == "of"
        {
            Some((parse_cut(tokens[index + 4])?, 5))
        } else if lower[index] == "asof" && lower.get(index + 1).map(String::as_str) != Some("join")
        {
            let Some(argument) = tokens.get(index + 1) else {
                return Err(QueryError::Unsupported(
                    "ASOF at the end of the statement — it takes an \
                     ingest-sequence literal: ASOF <n>"
                        .to_owned(),
                ));
            };
            Some((parse_cut(argument)?, 2))
        } else {
            if index + 2 < tokens.len()
                && lower[index] == "as"
                && lower[index + 1] == "of"
                && tokens[index + 2].parse::<u64>().is_ok()
                && (index == 0 || lower[index - 1] != "system_time")
            {
                return Err(QueryError::Unsupported(
                    "AS OF <n> — SQL's alias grammar claims the two-word form; \
                     write ASOF <n> (one word), or the standard \
                     FOR SYSTEM_TIME AS OF <n>"
                        .to_owned(),
                ));
            }
            None
        };
        match matched {
            Some((value, width)) => {
                if cut.is_some() {
                    return Err(QueryError::Unsupported(
                        "one AS OF per statement".to_owned(),
                    ));
                }
                cut = Some(value);
                removed = Some((spans[index].0, spans[index + width - 1].1));
                index += width;
            }
            None => index += 1,
        }
    }
    // Splice the clause out of the original text: every other byte —
    // newlines, comments, string literals, spacing — reaches the parser
    // exactly as the caller wrote it.
    let Some((start, end)) = removed else {
        return Ok((None, None));
    };
    let mut cleaned = String::with_capacity(sql.len() - (end - start));
    cleaned.push_str(&sql[..start]);
    cleaned.push_str(&sql[end..]);
    Ok((Some(cleaned), cut))
}

pub fn parse_statement(sql: &str) -> Result<Statement, QueryError> {
    let stripped;
    let (sql, as_of) = match extract_as_of(sql)? {
        (Some(cleaned), cut) => {
            stripped = cleaned;
            (stripped.as_str(), cut)
        }
        (None, cut) => (sql, cut),
    };
    let rewritten;
    let sql = if is_create_table(sql) {
        rewritten = rewrite_ordering_key(sql)?;
        &rewritten
    } else {
        sql
    };
    let statements =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| QueryError::Parse(e.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(QueryError::Unsupported(format!(
            "expected exactly one statement, got {}",
            statements.len()
        )));
    };
    if as_of.is_some() && !matches!(statement, ast::Statement::Query(_)) {
        return Err(QueryError::Unsupported(
            "AS OF applies to SELECT — mutations and DDL always act on latest knowledge".to_owned(),
        ));
    }
    match statement {
        ast::Statement::Query(query) => {
            let mut plan = lower_query(query)?;
            if as_of.is_some() {
                if plan.join.is_some() {
                    return Err(QueryError::Unsupported(
                        "AS OF with JOIN — the clause binds to one table's sequence \
                         space and the join lowering does not carry it yet"
                            .to_owned(),
                    ));
                }
                plan.as_of = as_of;
            }
            Ok(Statement::Select(Box::new(plan)))
        }
        ast::Statement::Update(update) => Ok(Statement::Update(lower_update(update)?)),
        ast::Statement::Delete(delete) => Ok(Statement::Delete(lower_delete(delete)?)),
        ast::Statement::CreateTable(create) => {
            Ok(Statement::CreateTable(lower_create_table(create)?))
        }
        ast::Statement::Insert(insert) => Ok(Statement::Insert(lower_insert(insert)?)),
        _ => Err(QueryError::Unsupported(
            "only SELECT, INSERT, UPDATE, DELETE, and CREATE TABLE are supported".to_owned(),
        )),
    }
}

/// Parses and lowers one SELECT statement (mutations go through
/// [`parse_statement`]).
pub fn plan(sql: &str) -> Result<Plan, QueryError> {
    match parse_statement(sql)? {
        Statement::Select(plan) => Ok(*plan),
        Statement::Update(_)
        | Statement::Delete(_)
        | Statement::CreateTable(_)
        | Statement::Insert(_) => Err(QueryError::Unsupported(
            "mutations and DDL run through their entry points, not query".to_owned(),
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
                nulls_first: order.options.nulls_first,
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
    let distinct = match &select.distinct {
        None | Some(ast::Distinct::All) => false,
        Some(ast::Distinct::Distinct) => true,
        Some(ast::Distinct::On(_)) => {
            return Err(QueryError::Unsupported("DISTINCT ON".to_owned()));
        }
    };
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
            matches!(expr, ast::Expr::Function(function) if function.over.is_none()
                && object_name(&function.name)
                    .map(|name| AggFunction::from_name(&name.to_lowercase()).is_some())
                    .unwrap_or(false))
        });
    let projection = if aggregate_shaped {
        let mut items = Vec::with_capacity(select_projection.len());
        for item in select_projection {
            items.push(lower_agg_item(item, &keys)?);
        }
        let having = select
            .having
            .as_ref()
            .map(|expr| {
                let mut hidden = Vec::new();
                let rewritten = extract_having_calls(expr, &mut hidden)?;
                Ok::<Having, QueryError>(Having {
                    items: hidden,
                    predicate: crate::predicate::lower_predicate(&rewritten)?,
                })
            })
            .transpose()?;
        if having.is_some() {
            // The hidden columns share the output row with the visible
            // ones; a visible column occupying a `__having` name would
            // shadow the filter's target and filter on the wrong value.
            let output_name = |item: &AggItem| match item {
                AggItem::Key { name, alias } => alias.clone().unwrap_or_else(|| name.clone()),
                AggItem::Call(call) => call.alias.clone().unwrap_or_default(),
            };
            if let Some(taken) = items
                .iter()
                .map(output_name)
                .find(|name| name.starts_with("__having"))
            {
                return Err(QueryError::Unsupported(format!(
                    "output name '{taken}' with HAVING (the __having prefix is \
                     reserved for its hidden columns)"
                )));
            }
        }
        Projection::Aggregate {
            keys,
            items,
            having,
        }
    } else {
        if select.having.is_some() {
            return Err(QueryError::Unsupported(
                "HAVING without aggregation (use WHERE)".to_owned(),
            ));
        }
        let mut items = Vec::with_capacity(select_projection.len());
        for item in select_projection {
            items.push(lower_item(item)?);
        }
        Projection::Items(items)
    };
    if distinct {
        match &projection {
            Projection::Items(items)
                if items
                    .iter()
                    .all(|item| matches!(item, PlanItem::Column { .. })) => {}
            _ => {
                return Err(QueryError::Unsupported(
                    "DISTINCT over window or aggregate projections".to_owned(),
                ))
            }
        }
    }
    Ok(Plan {
        table,
        join,
        projection,
        distinct,
        predicate,
        order_by: None,
        limit: None,
        offset: None,
        as_of: None,
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
        ast::Expr::IsNull(inner) => ast::Expr::IsNull(Box::new(recurse(inner)?)),
        ast::Expr::IsNotNull(inner) => ast::Expr::IsNotNull(Box::new(recurse(inner)?)),
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

/// Rewrites a HAVING expression: every aggregate call becomes a
/// reference to a hidden output column (`__having{i}`), collected into
/// `hidden` for the executor to compute alongside the SELECT list.
fn extract_having_calls(
    expr: &ast::Expr,
    hidden: &mut Vec<AggItem>,
) -> Result<ast::Expr, QueryError> {
    Ok(match expr {
        ast::Expr::Nested(inner) => {
            ast::Expr::Nested(Box::new(extract_having_calls(inner, hidden)?))
        }
        ast::Expr::IsNull(inner) => {
            ast::Expr::IsNull(Box::new(extract_having_calls(inner, hidden)?))
        }
        ast::Expr::IsNotNull(inner) => {
            ast::Expr::IsNotNull(Box::new(extract_having_calls(inner, hidden)?))
        }
        ast::Expr::UnaryOp { op, expr } => ast::Expr::UnaryOp {
            op: *op,
            expr: Box::new(extract_having_calls(expr, hidden)?),
        },
        ast::Expr::BinaryOp { left, op, right } => ast::Expr::BinaryOp {
            left: Box::new(extract_having_calls(left, hidden)?),
            op: op.clone(),
            right: Box::new(extract_having_calls(right, hidden)?),
        },
        ast::Expr::Function(function) if function.over.is_none() => {
            let name = format!("__having{}", hidden.len());
            let item = lower_agg_item(
                &ast::SelectItem::ExprWithAlias {
                    expr: expr.clone(),
                    alias: ast::Ident::new(name.clone()),
                },
                &[],
            )?;
            hidden.push(item);
            ast::Expr::Identifier(ast::Ident::new(name))
        }
        other => other.clone(),
    })
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
        ast::Expr::Function(function) if function.over.is_some() => {
            lower_window_call(function, alias)
        }
        other => {
            let scalar = lower_scalar_expr(other)?;
            Ok(PlanItem::Computed {
                expr: scalar,
                name: alias.unwrap_or_else(|| other.to_string()),
            })
        }
    }
}

/// Lowers a scalar expression for the computed-projection slot (#49):
/// arithmetic, the built-in scalar functions, and `CASE` with WHERE
/// grammar conditions. Anything else is refused loudly.
fn lower_scalar_expr(expr: &ast::Expr) -> Result<ScalarExpr, QueryError> {
    match expr {
        ast::Expr::Nested(inner) => lower_scalar_expr(inner),
        ast::Expr::Identifier(name) => Ok(ScalarExpr::Column(ident(name))),
        ast::Expr::Value(value) => match &value.value {
            ast::Value::Number(text, _) => {
                let number = crate::predicate::parse_number(text)?;
                Ok(ScalarExpr::Literal(match number {
                    Number::Int(value) => value as f64,
                    Number::Float(value) => value,
                }))
            }
            other => Err(QueryError::Unsupported(format!(
                "literal '{other}' in a scalar expression (numbers only)"
            ))),
        },
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Minus,
            expr,
        } => Ok(ScalarExpr::Negate(Box::new(lower_scalar_expr(expr)?))),
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Plus,
            expr,
        } => lower_scalar_expr(expr),
        ast::Expr::BinaryOp { left, op, right } => {
            let op = match op {
                ast::BinaryOperator::Plus => ArithOp::Add,
                ast::BinaryOperator::Minus => ArithOp::Sub,
                ast::BinaryOperator::Multiply => ArithOp::Mul,
                ast::BinaryOperator::Divide => ArithOp::Div,
                ast::BinaryOperator::Modulo => ArithOp::Mod,
                other => {
                    return Err(QueryError::Unsupported(format!(
                        "operator '{other}' in a scalar expression"
                    )))
                }
            };
            Ok(ScalarExpr::Binary {
                op,
                left: Box::new(lower_scalar_expr(left)?),
                right: Box::new(lower_scalar_expr(right)?),
            })
        }
        ast::Expr::Function(function) => {
            let name = object_name(&function.name)?.to_lowercase();
            if function.over.is_some() {
                return Err(QueryError::Unsupported(
                    "a window call inside a scalar expression".to_owned(),
                ));
            }
            let ast::FunctionArguments::List(list) = &function.args else {
                return Err(QueryError::Unsupported(format!(
                    "{name} without an argument list"
                )));
            };
            let mut args = Vec::with_capacity(list.args.len());
            for arg in &list.args {
                let ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(expr)) = arg else {
                    return Err(QueryError::Unsupported(format!(
                        "argument '{arg}' in {name}"
                    )));
                };
                args.push(lower_scalar_expr(expr)?);
            }
            let Some((scalar, arity)) = ScalarFunction::from_name(&name) else {
                if AggFunction::from_name(&name).is_some() {
                    return Err(QueryError::Unsupported(format!(
                        "aggregate {name} inside a scalar expression"
                    )));
                }
                // Not a built-in: an embedder-registered column
                // function, resolved against the registry at execution
                // (which is where the registry lives) — unknown names
                // stay loud, just later.
                return Ok(ScalarExpr::Registered { name, args });
            };
            if args.len() != arity {
                return Err(QueryError::Unsupported(format!(
                    "{name} takes {arity} argument(s), got {}",
                    args.len()
                )));
            }
            Ok(ScalarExpr::Call {
                function: scalar,
                args,
            })
        }
        // sqlparser gives FLOOR and CEIL dedicated AST nodes.
        floor_or_ceil @ (ast::Expr::Floor { expr, field } | ast::Expr::Ceil { expr, field }) => {
            let ast::CeilFloorKind::DateTimeField(ast::DateTimeField::NoDateTime) = field else {
                return Err(QueryError::Unsupported(
                    "FLOOR/CEIL with a scale or datetime field".to_owned(),
                ));
            };
            let function = if matches!(floor_or_ceil, ast::Expr::Floor { .. }) {
                ScalarFunction::Floor
            } else {
                ScalarFunction::Ceil
            };
            Ok(ScalarExpr::Call {
                function,
                args: vec![lower_scalar_expr(expr)?],
            })
        }
        ast::Expr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if operand.is_some() {
                return Err(QueryError::Unsupported(
                    "CASE <operand> WHEN (use CASE WHEN <condition>)".to_owned(),
                ));
            }
            let mut whens = Vec::with_capacity(conditions.len());
            for case_when in conditions {
                whens.push((
                    crate::predicate::lower_predicate(&case_when.condition)?,
                    lower_scalar_expr(&case_when.result)?,
                ));
            }
            let otherwise = else_result
                .as_ref()
                .map(|expr| lower_scalar_expr(expr).map(Box::new))
                .transpose()?;
            Ok(ScalarExpr::Case { whens, otherwise })
        }
        other => Err(QueryError::Unsupported(format!(
            "expression '{other}' in a projection"
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
    fn the_knowledge_time_clause_is_extracted_in_both_spellings() {
        // One-word ASOF — the engine's spelling.
        let plan_asof = plan("SELECT x FROM t ASOF 41520 WHERE x > 1").expect("plans");
        assert_eq!(plan_asof.as_of, Some(41_520));
        assert!(plan_asof.predicate.is_some(), "the rest of the query holds");
        // The SQL:2011 carrier, same meaning.
        let standard = plan("SELECT x FROM t FOR SYSTEM_TIME AS OF 41520 WHERE x > 1").unwrap();
        assert_eq!(standard.as_of, Some(41_520));
        // Without a clause, nothing is touched.
        assert_eq!(plan("SELECT x FROM t WHERE x > 1").unwrap().as_of, None);
        // Inside a string literal the words are inert.
        let inert = plan("SELECT x FROM t WHERE sym = 'ASOF 5'").unwrap();
        assert_eq!(inert.as_of, None);
    }

    /// Extraction splices the clause out of the original text rather
    /// than reassembling tokens. Reassembly collapsed every newline,
    /// which let a `--` comment swallow the rest of the statement — a
    /// silently wrong answer, exactly what the clause's ruling forbids.
    #[test]
    fn extraction_preserves_the_rest_of_the_statement_verbatim() {
        // A line comment still ends at its newline.
        let commented =
            plan("SELECT x FROM t ASOF 5 -- pick the tail\nWHERE x > 1").expect("plans");
        assert_eq!(commented.as_of, Some(5));
        assert!(
            commented.predicate.is_some(),
            "the WHERE after a comment must survive extraction"
        );
        // And the clause is inert *inside* a comment: comments are
        // skipped whole, so nothing in one is ever read as a clause.
        let in_comment = plan("SELECT x FROM t -- ASOF 5\nWHERE x > 1").expect("plans");
        assert_eq!(in_comment.as_of, None);
        let block = plan("SELECT x FROM t /* ASOF 5 */ WHERE x > 1").expect("plans");
        assert_eq!(block.as_of, None);
        // A clause elsewhere in a commented statement still extracts.
        let both = plan("SELECT x FROM t ASOF 7 /* note */ WHERE x > 1").expect("plans");
        assert_eq!(both.as_of, Some(7));
        assert!(both.predicate.is_some());
    }

    #[test]
    fn knowledge_time_misuses_are_taught() {
        for (sql, needle) in [
            // The two-word form collides with SQL's alias grammar.
            ("SELECT x FROM t AS OF 3", "ASOF <n> (one word)"),
            ("SELECT x FROM t ASOF now", "ingest-sequence literal"),
            ("SELECT x FROM t ASOF", "takes an ingest-sequence"),
            ("SELECT x FROM t ASOF 1 WHERE x > 0 ASOF 2", "one AS OF"),
            (
                "SELECT t.x FROM t ASOF 1 JOIN d ON t.k = d.k",
                "AS OF with JOIN",
            ),
            ("DELETE FROM t ASOF 1 WHERE x > 0", "latest knowledge"),
            ("UPDATE t ASOF 1 SET x = 0", "latest knowledge"),
        ] {
            let error = format!("{}", parse_statement(sql).unwrap_err());
            assert!(error.contains(needle), "{sql}: {error}");
        }
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
            ("SELECT DISTINCT ON (x) x FROM t", "DISTINCT ON"),
            (
                "SELECT DISTINCT count(x) FROM t",
                "DISTINCT over window or aggregate projections",
            ),
            (
                "SELECT x, sum(y) FROM t GROUP BY x, x + 1",
                "plain key columns",
            ),
            ("SELECT x FROM t GROUP BY x LIMIT x", "LIMIT"),
            // (an unknown plain call like nope_agg(x) now lowers to a
            // Registered column function and is refused by name at
            // execution, where the registry lives — tested in engine)
            ("SELECT y FROM t GROUP BY x", "must appear in GROUP BY"),
            (
                "SELECT sum(x) OVER (ORDER BY ts RANGE BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
                "ROWS only",
            ),
            ("SELECT sum(x) OVER (ORDER BY ts) FROM t", "without a frame"),
            ("INSERT INTO t VALUES (1)", "entry points"), // supported, elsewhere
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

    #[test]
    fn the_ddl_gate_is_word_wise_not_a_byte_prefix() {
        // Any whitespace between CREATE and TABLE hits the same gate:
        // PRIMARY KEY is refused, ORDERING KEY parses.
        for sql in [
            "CREATE  TABLE t (ts BIGINT PRIMARY KEY)",
            "CREATE\tTABLE t (ts BIGINT PRIMARY KEY)",
            "create   table t (ts BIGINT PRIMARY KEY)",
        ] {
            let error = parse_statement(sql).expect_err(sql).to_string();
            assert!(error.contains("ORDERING KEY"), "{sql}: {error}");
        }
        for sql in [
            "CREATE  TABLE t (ts BIGINT ORDERING KEY)",
            "CREATE\tTABLE t (ts BIGINT ORDERING  KEY)",
        ] {
            assert!(
                matches!(parse_statement(sql), Ok(Statement::CreateTable(_))),
                "{sql}"
            );
        }
    }

    #[test]
    fn a_column_named_ordering_is_not_the_constraint() {
        // `ordering KEY` after `(` or `,` declares a key column named
        // `ordering`; only the constraint position rewrites.
        let Ok(Statement::CreateTable(plan)) = parse_statement(
            "CREATE TABLE t (ts BIGINT ORDERING KEY, ordering KEY, primary_ish DOUBLE)",
        ) else {
            panic!("parses as CREATE TABLE")
        };
        assert_eq!(plan.columns.len(), 3);
        assert_eq!(plan.columns[1].name, "ordering");
        assert_eq!(plan.columns[1].type_name, "KEY");
        assert!(!plan.columns[1].ordering_key);
        assert!(plan.columns[0].ordering_key);
    }

    #[test]
    fn duplicate_column_names_are_refused_not_shadowed() {
        // Two columns named `x`: every resolver takes the first match,
        // so the second would be silently unreachable forever.
        let error = parse_statement("CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE, x DOUBLE)")
            .expect_err("refused")
            .to_string();
        assert!(error.contains("declared twice"), "{error}");
    }

    #[test]
    fn the_sequence_pseudocolumn_cannot_be_declared() {
        // Declaring `_seq` would shadow the pseudocolumn with something
        // else entirely — the name is the engine's.
        for ddl in [
            "CREATE TABLE t (ts BIGINT ORDERING KEY, _seq BIGINT)",
            "CREATE TABLE t (_seq BIGINT ORDERING KEY, x DOUBLE)",
        ] {
            let error = parse_statement(ddl).expect_err("refused").to_string();
            assert!(error.contains("reserved"), "{error}");
            assert!(error.contains("ingest-sequence"), "{error}");
        }
    }

    #[test]
    fn table_level_constraints_are_refused_not_dropped() {
        let error =
            parse_statement("CREATE TABLE t (ts BIGINT ORDERING KEY, x DOUBLE, UNIQUE (x))")
                .expect_err("refused")
                .to_string();
        assert!(error.contains("table-level constraint"), "{error}");
    }

    #[test]
    fn the_having_prefix_is_reserved_when_having_is_present() {
        // A visible column on a __having name would shadow the hidden
        // filter column and filter on the wrong aggregate.
        let error = plan("SELECT sym, COUNT(x) AS __having0 FROM t GROUP BY sym HAVING SUM(x) > 4")
            .expect_err("refused")
            .to_string();
        assert!(error.contains("__having"), "{error}");
        // Without HAVING the name is just a name.
        assert!(plan("SELECT sym, COUNT(x) AS __having0 FROM t GROUP BY sym").is_ok());
    }
}
