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
//! CREATE TABLE t (col BIGINT|DOUBLE|SYMBOL [NOT NULL|ORDERING KEY], ...);
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
    /// Storage failed to materialize a segment the query needs — a
    /// fault-in error under the residency design (I/O, checksum, or a
    /// reader racing a compaction).
    Storage(storage_lite::StorageError),
}

impl From<storage_lite::StorageError> for QueryError {
    fn from(error: storage_lite::StorageError) -> QueryError {
        QueryError::Storage(error)
    }
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
            QueryError::Storage(error) => write!(f, "storage error: {error}"),
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
        /// The window calls hoisted out of `expr`, in the order
        /// [`ScalarExpr::Window`] indexes them.
        windows: Vec<WindowCall>,
        /// The output column name: the alias, or the expression's SQL
        /// text when unaliased.
        name: String,
    },
    /// A window call, whole: `sum(x) OVER (...)` as a SELECT item.
    Window {
        /// What to compute.
        call: WindowCall,
        /// Output name, if aliased.
        alias: Option<String>,
    },
}

/// One window call — an aggregate over a frame, or a positional
/// lookup. Shared by the whole-item form ([`PlanItem::Window`]) and by
/// calls hoisted out of a scalar expression, so composing a window into
/// arithmetic reuses the call rather than restating it.
#[derive(Clone, PartialEq, Debug)]
pub enum WindowCall {
    /// An aggregate over the frame: `sum(x) OVER (...)`.
    Agg {
        /// Function name, lower-cased (resolved against the registry).
        function: String,
        /// Argument column names, in call order.
        args: Vec<String>,
        /// PARTITION BY terms, in order (empty = one partition over
        /// the whole snapshot). Each is a symbol column (the
        /// time-series direction, one partition per symbol) or the
        /// ordering key / a bucket of it (the cross-sectional
        /// direction, one partition per instant); several together give
        /// the intersection — `PARTITION BY sym, ts / 60` is one
        /// partition per symbol per bar.
        partition_by: Vec<GroupKey>,
        /// ORDER BY column — must be the data's ordering key. `None`
        /// for a cross-sectional window, which has no order *within*
        /// an instant and therefore takes the whole partition as its
        /// frame.
        order_by: Option<String>,
        /// What rows the frame covers (see [`Frame`]).
        frame: Frame,
    },
    /// `LAG(x, k)` / `LEAD(x, k)` — a positional lookup, **not** an
    /// aggregate: it reads another *row* rather than reducing a frame,
    /// which is why standard SQL gives it no frame clause and why it
    /// carries the source column's type instead of computing in `f64`.
    Value {
        /// `true` for `LEAD` (look forward), `false` for `LAG`.
        lead: bool,
        /// The column read.
        column: String,
        /// How many rows away, `>= 1`.
        offset: usize,
        /// PARTITION BY terms (see [`WindowCall::Agg`]).
        partition_by: Vec<GroupKey>,
        /// ORDER BY column — must be the data's ordering key. Required
        /// here: a positional lookup with no order has no meaning.
        order_by: String,
    },
}

impl WindowCall {
    /// The output name this call takes when the query writes no alias.
    pub fn default_name(&self) -> &str {
        match self {
            WindowCall::Agg { function, .. } => function,
            WindowCall::Value { lead: true, .. } => "lead",
            WindowCall::Value { lead: false, .. } => "lag",
        }
    }

    /// Every stored column this call reads.
    pub fn columns(&self) -> Vec<String> {
        let mut names = Vec::new();
        match self {
            WindowCall::Agg {
                args,
                partition_by,
                order_by,
                ..
            } => {
                names.extend(args.iter().cloned());
                names.extend(order_by.iter().cloned());
                names.extend(partition_by.iter().map(|key| key.column().to_owned()));
            }
            WindowCall::Value {
                column,
                partition_by,
                order_by,
                ..
            } => {
                names.push(column.clone());
                names.push(order_by.clone());
                names.extend(partition_by.iter().map(|key| key.column().to_owned()));
            }
        }
        names
    }
}

/// A window's frame — what rows one output row sees.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Frame {
    /// `ROWS BETWEEN n PRECEDING AND CURRENT ROW`, `None` = UNBOUNDED:
    /// a **row count**, uniform across the column.
    Rows(Option<usize>),
    /// `RANGE BETWEEN v PRECEDING AND CURRENT ROW`: every row whose
    /// ordering key is within `v` of the current row's — a **value**
    /// span, so the row count varies from row to row.
    ///
    /// The bound is in the ordering key's own units (there is no
    /// `INTERVAL` type — a five-minute span over nanosecond stamps is
    /// `300000000000`), and it is unsigned because a frame extends
    /// backward from the current row.
    Range(u64),
    /// The **whole partition** — what standard SQL gives a window with
    /// no `ORDER BY`, and what a cross-sectional statistic wants: every
    /// row of the instant sees every other row of it, so there is no
    /// "before" and no frame arithmetic to do.
    Partition,
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
    /// One of this item's window calls, by position in its `windows`
    /// list — the placeholder a window call leaves behind when it is
    /// hoisted out of a scalar expression (#94).
    ///
    /// Hoisting is how standard SQL's evaluation order is honoured
    /// rather than reimplemented: windows compute over the whole
    /// partition first, and the SELECT list's arithmetic then runs over
    /// their results, one row at a time. A window cannot be evaluated
    /// row-by-row from inside a scalar, because its partition spans
    /// segments the scalar walks one at a time.
    Window(usize),
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
    /// `FIRST(col)` — the value at the group's earliest ordering key.
    First,
    /// `LAST(col)` — the value at the group's latest ordering key.
    Last,
}

impl AggFunction {
    fn from_name(name: &str) -> Option<AggFunction> {
        match name {
            "count" => Some(AggFunction::Count),
            "sum" => Some(AggFunction::Sum),
            "avg" => Some(AggFunction::Avg),
            "min" => Some(AggFunction::Min),
            "max" => Some(AggFunction::Max),
            // The de-facto TSDB names (ruled (a) 2026-07-29). There is
            // no ISO spelling, but every time-series engine that has
            // this concept calls it this, and it is well defined here
            // for the reason it is ill defined in general SQL: the
            // ordering key is declared, so "first" is not a question
            // about physical row order.
            "first" => Some(AggFunction::First),
            "last" => Some(AggFunction::Last),
            _ => None,
        }
    }

    /// Whether this aggregate reads the ordering key as well as its
    /// argument — `FIRST`/`LAST` are positional on the time axis, the
    /// group-level counterpart of `LAG`/`LEAD`.
    pub fn is_positional(self) -> bool {
        matches!(self, AggFunction::First | AggFunction::Last)
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
        /// What the group is keyed on.
        key: GroupKey,
        /// Output name, if aliased.
        alias: Option<String>,
    },
    /// An aggregate call.
    Call(AggCall),
}

/// One `GROUP BY` term.
///
/// The planner has no schema, so it cannot tell a symbol column from
/// the ordering key — both are bare identifiers. It records the
/// *shape* and the executor, which knows the column types and which
/// column ingest is ordered on, decides what the shape means.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GroupKey {
    /// A bare column: a symbol column (grouped by value) or the
    /// declared ordering key (grouped by its integer value).
    Column(String),
    /// A time bucket — monotone integer arithmetic on the ordering key
    /// (F1 = d, ruled 2026-07-29): `ts / 60` is the bucket index and
    /// `(ts / 60) * 60` the bucket's start. Because the arithmetic is
    /// monotone on a column the data is already clustered by, the
    /// buckets come out in order and grouping streams.
    ///
    /// The executor refuses this on any column but the ordering key:
    /// on anything else the arithmetic proves nothing about order.
    Bucket {
        /// The column the arithmetic reads (must be the ordering key).
        column: String,
        /// The divisor — the bucket width, in the key's own units.
        /// Positive; `1` for a bare `ts` read as a bucket.
        divide: i64,
        /// The multiplier that turns a bucket index back into the
        /// bucket's start value, when the query wrote one.
        multiply: Option<i64>,
    },
}

impl GroupKey {
    /// The column this term reads.
    pub fn column(&self) -> &str {
        match self {
            GroupKey::Column(name) | GroupKey::Bucket { column: name, .. } => name,
        }
    }

    /// The output name when the query writes no alias: the column's own
    /// name, or the bucket arithmetic in canonical form (`ts / 60`,
    /// `(ts / 60) * 60`) — whatever spacing the query used. Alias a
    /// bucket to name it anything else.
    pub fn output_name(&self) -> String {
        match self {
            GroupKey::Column(name) => name.clone(),
            GroupKey::Bucket {
                column,
                divide,
                multiply: None,
            } => format!("{column} / {divide}"),
            GroupKey::Bucket {
                column,
                divide,
                multiply: Some(multiply),
            } => format!("({column} / {divide}) * {multiply}"),
        }
    }
}

/// What the SELECT list computes.
#[derive(Clone, PartialEq, Debug)]
pub enum Projection {
    /// Plain columns and window calls, one output row per input row.
    Items(Vec<PlanItem>),
    /// `GROUP BY` keys and aggregate calls, one output row per group.
    Aggregate {
        /// The GROUP BY terms (empty = one global group).
        keys: Vec<GroupKey>,
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
    /// `Some` when the query wrote `ASOF LEFT JOIN` / `ASOF INNER
    /// JOIN` (#65): match each fact row to the dimension's most recent
    /// row at-or-before it on the two tables' declared ordering keys,
    /// within the `ON` equality's partition.
    pub as_of: Option<AsOfMatch>,
    /// The two columns an explicit `ASOF` inequality named, if the
    /// query wrote one: `(fact side, dimension side)`. The planner has
    /// no schemas, so it only checks their *shape*; the executor —
    /// which knows the declared ordering keys — is what checks they
    /// name the time axis, and refuses with a teaching error if not.
    pub as_of_named: Option<(String, String)>,
}

/// One side of a join's `ON` comparison: its table qualifier, if the
/// query wrote one, and its column name.
type OnSide = (Option<String>, String);

/// How an as-of join compares the two ordering keys (#65: the operator
/// is what the user's explicit inequality selects, `>=` by default).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AsOfMatch {
    /// The dimension row's key is **at or before** the fact's — the
    /// default, and what `q.ts <= t.ts` spells.
    AtOrBefore,
    /// Strictly before, which `q.ts < t.ts` spells.
    StrictlyBefore,
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

impl Plan {
    /// Every stored-column name this plan reads, by any route: the
    /// projection (plain, computed, window), `GROUP BY`, `HAVING`, and
    /// the `WHERE` predicate. Deliberately over-inclusive — output
    /// aliases and hidden `__having` names come along, and a name that
    /// matches no column simply never matches anything.
    ///
    /// The join uses it to gather only the dimension columns a query
    /// actually needs (#81). That makes completeness load-bearing:
    /// omitting a route here would drop a column the query then cannot
    /// resolve. It fails loudly rather than silently — the column is
    /// absent, not null-filled — but it fails, so every variant above
    /// must be walked, and a new one has to be added here too.
    pub fn referenced_columns(&self) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        if let Some(predicate) = &self.predicate {
            predicate_columns(predicate, &mut names);
        }
        if let Some(order_by) = &self.order_by {
            // Resolved against the *output* schema, so this is only
            // ever an alias — harmless, and free insurance for the
            // unaliased case where the two names coincide.
            names.insert(order_by.column.clone());
        }
        match &self.projection {
            Projection::Items(items) => {
                for item in items {
                    match item {
                        PlanItem::Column { name, .. } => {
                            names.insert(name.clone());
                        }
                        PlanItem::Computed { expr, windows, .. } => {
                            scalar_columns(expr, &mut names);
                            names.extend(windows.iter().flat_map(WindowCall::columns));
                        }
                        PlanItem::Window { call, .. } => {
                            names.extend(call.columns());
                        }
                    }
                }
            }
            Projection::Aggregate {
                keys,
                items,
                having,
            } => {
                names.extend(keys.iter().map(|key| key.column().to_owned()));
                let agg_item =
                    |item: &AggItem, names: &mut std::collections::HashSet<String>| match item {
                        AggItem::Key { key, .. } => {
                            names.insert(key.column().to_owned());
                        }
                        AggItem::Call(call) => names.extend(call.argument.iter().cloned()),
                    };
                for item in items {
                    agg_item(item, &mut names);
                }
                if let Some(having) = having {
                    for item in &having.items {
                        agg_item(item, &mut names);
                    }
                    predicate_columns(&having.predicate, &mut names);
                }
            }
        }
        names
    }
}

/// Every column name a predicate tests.
fn predicate_columns(predicate: &Predicate, names: &mut std::collections::HashSet<String>) {
    match predicate {
        Predicate::Compare { column, .. }
        | Predicate::KeyEquals { column, .. }
        | Predicate::KeyLike { column, .. }
        | Predicate::KeyIn { column, .. }
        | Predicate::IsNull { column, .. } => {
            names.insert(column.clone());
        }
        Predicate::And(left, right) | Predicate::Or(left, right) => {
            predicate_columns(left, names);
            predicate_columns(right, names);
        }
        Predicate::Not(inner) => predicate_columns(inner, names),
    }
}

/// Every column name a scalar expression reads, `CASE` conditions
/// included.
fn scalar_columns(expr: &ScalarExpr, names: &mut std::collections::HashSet<String>) {
    match expr {
        ScalarExpr::Column(name) => {
            names.insert(name.clone());
        }
        // A hoisted window's own columns are collected from the
        // item's `windows` list, not from here.
        ScalarExpr::Literal(_) | ScalarExpr::Window(_) => {}
        ScalarExpr::Negate(inner) => scalar_columns(inner, names),
        ScalarExpr::Binary { left, right, .. } => {
            scalar_columns(left, names);
            scalar_columns(right, names);
        }
        ScalarExpr::Call { args, .. } | ScalarExpr::Registered { args, .. } => {
            for argument in args {
                scalar_columns(argument, names);
            }
        }
        ScalarExpr::Case { whens, otherwise } => {
            for (condition, value) in whens {
                predicate_columns(condition, names);
                scalar_columns(value, names);
            }
            if let Some(otherwise) = otherwise {
                scalar_columns(otherwise, names);
            }
        }
    }
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
    /// `"BIGINT"`, `"DOUBLE"`, or `"SYMBOL"` — resolved to the engine's
    /// column types by the embedder (query-lite stays schema-agnostic).
    pub type_name: String,
    /// `NOT NULL` present (the ordering key implies it).
    pub not_null: bool,
    /// `ORDERING KEY` present.
    pub ordering_key: bool,
}

/// A lowered `CREATE TABLE`: the DDL surface of the stdlib table (#49,
/// ruled 2026-07-27) — standard names where standard exists (`BIGINT`,
/// `DOUBLE`), `SYMBOL` for dictionary-encoded labels (kdb+ and
/// QuestDB spell it that way; TallyDB adopted it 2026-07-29, when the
/// older `KEY` proved to name two different things beside `ORDERING
/// KEY`), the ordering key
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
            ast::DataType::Custom(name, _) if object_name(name)?.eq_ignore_ascii_case("symbol") => {
                "SYMBOL"
            }
            // The type was spelled KEY until 2026-07-29. Name the new
            // spelling rather than listing the type set: a reader who
            // typed KEY knows what they meant.
            ast::DataType::Custom(name, _) if object_name(name)?.eq_ignore_ascii_case("key") => {
                return Err(QueryError::Unsupported(format!(
                    "column '{}': the label type is spelled SYMBOL (KEY named two \
                     different things beside ORDERING KEY)",
                    ident(&column.name)
                )))
            }
            ast::DataType::Varchar(_)
            | ast::DataType::Text
            | ast::DataType::Char(_)
            | ast::DataType::String(_) => {
                return Err(QueryError::Unsupported(format!(
                    "column '{}': strings are not a column type here — keys are \
                     interned labels used for filtering, grouping, and joining; \
                     declare it SYMBOL",
                    ident(&column.name)
                )))
            }
            other => {
                return Err(QueryError::Unsupported(format!(
                    "column type '{other}' (BIGINT, DOUBLE, or SYMBOL)"
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
        // A column *definition* could also start `word KEY` — when the
        // label type was spelled KEY, `ordering KEY,` declared a column
        // named `ordering`. `SYMBOL` retired that reading, but the
        // guard stays: without it the same text rewrites to
        // `(PRIMARY KEY,` and the reader gets a parse error instead of
        // the refusal that names the new spelling. A constraint never
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
/// Shallow tokenization with byte spans, shared by every pre-parse
/// lift. Quoted runs (`'…'` with `''` escapes, `"…"`) stay single
/// tokens so nothing inside a string literal can look like a clause;
/// comments are skipped whole so nothing inside one does either;
/// hugging punctuation splits off so `ASOF 5,` scans.
///
/// **Spans are the point.** A lift splices its clause out of the
/// ORIGINAL text using these spans, never reassembling the statement
/// from tokens — reassembly collapses the newline that terminates a
/// `--` comment and silently comments out the rest of the statement.
/// That was a real bug (found at the M4 close), and it is why every
/// new lift must come through here.
fn tokenize_with_spans(sql: &str) -> (Vec<&str>, Vec<(usize, usize)>, Vec<String>) {
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
    (tokens, spans, lower)
}

/// Lifts the `ASOF` token that *precedes a join* (#65's hybrid: the
/// grammar is ClickHouse's, the authority is our schema). The token is
/// spliced out by byte span and the remainder parses as an ordinary
/// join, which is why no fork of sqlparser is needed — verified
/// 2026-07-29: sqlparser 0.62 parses only Snowflake's `MATCH_CONDITION`
/// form and DuckDB accepts only its own `ON` form, and the two accepted
/// sets are disjoint, so neither spelling could be borrowed.
///
/// Runs **before** [`extract_as_of`], because `ASOF LEFT JOIN` would
/// otherwise look like the time-travel clause with `LEFT` as its cut.
///
/// Bare `ASOF JOIN` is refused: bare as-of semantics are a genuine
/// vendor divergence, so the user says which they mean. (Standing
/// revisit flagged by the Human 2026-07-30 — see #65.)
fn extract_asof_join(sql: &str) -> Result<(Option<String>, bool), QueryError> {
    let (_, spans, lower) = tokenize_with_spans(sql);
    let mut found: Option<(usize, usize)> = None;
    for index in 0..lower.len() {
        if lower[index] != "asof" {
            continue;
        }
        match lower.get(index + 1).map(String::as_str) {
            Some("join") => {
                return Err(QueryError::Unsupported(
                    "bare ASOF JOIN — write ASOF LEFT JOIN (keep unmatched fact rows, \
                     null-padded) or ASOF INNER JOIN (drop them). Vendors disagree on \
                     what a bare as-of join means, so this engine makes you say it"
                        .to_owned(),
                ));
            }
            Some("left") | Some("inner") => {
                // `ASOF LEFT JOIN` / `ASOF INNER JOIN` — and only those:
                // `ASOF LEFT` with no JOIN is not a join at all, and
                // falls through to the time-travel lift's own error.
                if lower.get(index + 2).map(String::as_str) != Some("join") {
                    continue;
                }
                if found.is_some() {
                    return Err(QueryError::Unsupported(
                        "one ASOF join per query".to_owned(),
                    ));
                }
                found = Some(spans[index]);
            }
            _ => continue,
        }
    }
    let Some((start, end)) = found else {
        return Ok((None, false));
    };
    let mut rewritten = String::with_capacity(sql.len());
    rewritten.push_str(&sql[..start]);
    rewritten.push_str(&sql[end..]);
    Ok((Some(rewritten), true))
}

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
    let (tokens, spans, lower) = tokenize_with_spans(sql);

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
    // The as-of JOIN lift runs first: `ASOF LEFT JOIN` would otherwise
    // read as the time-travel clause with `LEFT` as its cut.
    let joined;
    let (sql, asof_join) = match extract_asof_join(sql)? {
        (Some(rewritten), flag) => {
            joined = rewritten;
            (joined.as_str(), flag)
        }
        (None, flag) => (sql, flag),
    };
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
            let mut plan = lower_query(query, asof_join)?;
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
    // A negative number parses as unary minus over a literal — the same
    // unwrap the WHERE grammar does, so `SET x = -1` and `WHERE x = -1`
    // accept exactly the same spellings.
    let (negated, rhs) = match &assignment.value {
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Minus,
            expr,
        } => (true, expr.as_ref()),
        other => (false, other),
    };
    let ast::Expr::Value(value) = rhs else {
        return Err(QueryError::Unsupported(format!(
            "SET {column} = '{}' — literals only for now",
            assignment.value
        )));
    };
    let value = match (&value.value, negated) {
        (ast::Value::Number(text, _), negated) => {
            let number = match (parse_number(text)?, negated) {
                (number, false) => number,
                (Number::Int(value), true) => Number::Int(-value),
                (Number::Float(value), true) => Number::Float(-value),
            };
            SetValue::Number(number)
        }
        (ast::Value::SingleQuotedString(text), false) => SetValue::String(text.clone()),
        (ast::Value::Null, false) => SetValue::Null,
        (other, _) => {
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

fn lower_query(query: &ast::Query, asof_join: bool) -> Result<Plan, QueryError> {
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
    let mut plan = lower_select(select, asof_join)?;
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

fn lower_select(select: &ast::Select, asof_join: bool) -> Result<Plan, QueryError> {
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
        match lower_join(&table, fact_alias.as_deref(), joins, asof_join)? {
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
    let keys = resolve_group_aliases(lower_group_by(&select.group_by)?, select_projection);
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
                AggItem::Key { key, alias } => alias.clone().unwrap_or_else(|| key.output_name()),
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
    asof_join: bool,
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
            // ON: the equality of two (possibly qualified) columns —
            // and, for an as-of join only, an optional second conjunct
            // naming the time axis explicitly.
            let (equality, inequality) = match on {
                ast::Expr::BinaryOp {
                    left,
                    op: ast::BinaryOperator::And,
                    right,
                } if asof_join => (left.as_ref(), Some(right.as_ref())),
                other => (other, None),
            };
            let ast::Expr::BinaryOp {
                left: on_left,
                op: ast::BinaryOperator::Eq,
                right: on_right,
            } = equality
            else {
                return Err(QueryError::Unsupported(if asof_join {
                    "ASOF JOIN ON must start with the partition equality                      (fact.key = dim.key), optionally AND an inequality on the                      ordering keys"
                        .to_owned()
                } else {
                    "JOIN ON must be a single equality".to_owned()
                }));
            };
            let side = |expr: &ast::Expr| -> Result<OnSide, QueryError> {
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
            let is_fact = |qualifier: &Option<String>| {
                qualifier
                    .as_ref()
                    .map(|name| name == fact || fact_alias.is_some_and(|alias| name == alias))
            };
            // The time axis: implicit (at-or-before) unless the query
            // spells an inequality, which is VALIDATED rather than
            // obeyed — it must name the two tables' ordering keys, and
            // all it selects is >= versus >.
            let (as_of, as_of_named) = match (asof_join, inequality) {
                (false, _) => (None, None),
                (true, None) => (Some(AsOfMatch::AtOrBefore), None),
                (true, Some(expr)) => {
                    let (matching, named) = lower_asof_inequality(expr, &side, &is_fact)?;
                    (Some(matching), Some(named))
                }
            };
            let (left_side, right_side) = (side(on_left)?, side(on_right)?);
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
                    as_of,
                    as_of_named,
                },
                dimension_alias,
            )))
        }
        _ => Err(QueryError::Unsupported(
            "one JOIN per query (star schema: fact times one dimension at a time)".to_owned(),
        )),
    }
}

/// Validates an as-of join's explicit time-axis inequality. The
/// engine already knows the axis — it is the two tables' declared
/// ordering keys — so this neither chooses columns nor reorders
/// anything: it checks that what the user wrote agrees with the
/// schema, and reads off which comparison they meant.
///
/// Written dimension-first (`q.ts <= t.ts`) or fact-first
/// (`t.ts >= q.ts`); both say the same thing. Which side is the fact is
/// read from the qualifiers, never from the operator — otherwise
/// `t.ts <= q.ts`, which asks for the quote *after* each trade, would
/// be silently answered with the one before it.
///
/// Returns the comparison and the two column names it used, fact side
/// first, for the executor to check against the declared ordering keys.
fn lower_asof_inequality(
    expr: &ast::Expr,
    side: &dyn Fn(&ast::Expr) -> Result<OnSide, QueryError>,
    is_fact: &dyn Fn(&Option<String>) -> Option<bool>,
) -> Result<(AsOfMatch, (String, String)), QueryError> {
    let ast::Expr::BinaryOp { left, op, right } = expr else {
        return Err(QueryError::Unsupported(
            "ASOF JOIN's second ON conjunct must compare the two ordering keys".to_owned(),
        ));
    };
    let (left_side, right_side) = (side(left)?, side(right)?);
    let fact_on_left = match (is_fact(&left_side.0), is_fact(&right_side.0)) {
        (Some(true), Some(false)) | (Some(true), None) | (None, Some(false)) => true,
        (Some(false), Some(true)) | (None, Some(true)) | (Some(false), None) => false,
        _ => {
            return Err(QueryError::Unsupported(
                "qualify at least one side of ASOF JOIN's time comparison, so it \
                 says which table each ordering key belongs to (fact.ts >= dim.ts)"
                    .to_owned(),
            ))
        }
    };
    // The operator says two things: which way the comparison points,
    // and whether an exactly-equal timestamp counts. Only the second is
    // ours to obey.
    let (left_is_later, matching) = match op {
        ast::BinaryOperator::GtEq => (true, AsOfMatch::AtOrBefore),
        ast::BinaryOperator::Gt => (true, AsOfMatch::StrictlyBefore),
        ast::BinaryOperator::LtEq => (false, AsOfMatch::AtOrBefore),
        ast::BinaryOperator::Lt => (false, AsOfMatch::StrictlyBefore),
        other => {
            return Err(QueryError::Unsupported(format!(
                "ASOF JOIN's time comparison is '{other}' — it must be one of \
                 <=, <, >=, > (which one only selects whether an exactly-equal \
                 timestamp matches)"
            )))
        }
    };
    // An as-of join looks *backwards*: the fact's clock is the later
    // one. Written the other way round, the query is asking for the
    // dimension row that comes after — a different question, and one
    // worth refusing rather than quietly answering the reverse of.
    if left_is_later != fact_on_left {
        return Err(QueryError::Unsupported(format!(
            "ASOF JOIN's time comparison puts the {} row at or after the {} one \
             — an as-of join looks backwards, so write it the other way round \
             (fact.ts >= dim.ts)",
            if fact_on_left { "dimension" } else { "fact" },
            if fact_on_left { "fact" } else { "dimension" },
        )));
    }
    let named = if fact_on_left {
        (left_side.1, right_side.1)
    } else {
        (right_side.1, left_side.1)
    };
    Ok((matching, named))
}

fn lower_group_by(group_by: &ast::GroupByExpr) -> Result<Vec<GroupKey>, QueryError> {
    let ast::GroupByExpr::Expressions(exprs, modifiers) = group_by else {
        return Err(QueryError::Unsupported("GROUP BY ALL".to_owned()));
    };
    if !modifiers.is_empty() {
        return Err(QueryError::Unsupported(
            "GROUP BY ROLLUP / CUBE / GROUPING SETS".to_owned(),
        ));
    }
    exprs.iter().map(lower_group_key).collect()
}

/// Lets `GROUP BY` name a bucket by the alias the SELECT list gave it:
/// `SELECT ts / 60 AS bar … GROUP BY bar` rather than repeating the
/// arithmetic. PostgreSQL and DuckDB both accept the output name here,
/// so this follows convention rather than coining.
///
/// Deliberately narrow: only an alias whose expression is a **bucket**
/// is substituted. Aliases of plain columns are left alone, because
/// there the alias and the column mean the same thing anyway and
/// substituting could only introduce a way to disagree. If a stored
/// column shares a bucket alias's name the alias wins — which is what
/// PostgreSQL does, and the query said the name after writing it.
fn resolve_group_aliases(keys: Vec<GroupKey>, projection: &[ast::SelectItem]) -> Vec<GroupKey> {
    let aliased = |name: &str| -> Option<GroupKey> {
        projection.iter().find_map(|item| {
            let ast::SelectItem::ExprWithAlias { expr, alias } = item else {
                return None;
            };
            if ident(alias) != name {
                return None;
            }
            match lower_group_key(expr) {
                Ok(key @ GroupKey::Bucket { .. }) => Some(key),
                _ => None,
            }
        })
    };
    keys.into_iter()
        .map(|key| match &key {
            GroupKey::Column(name) => aliased(name).unwrap_or(key),
            GroupKey::Bucket { .. } => key,
        })
        .collect()
}

/// Lowers one `GROUP BY` term: a bare column, or the monotone integer
/// arithmetic on the ordering key that F1 = d admits (`ts / 60`,
/// `(ts / 60) * 60`) and nothing else.
///
/// "Nothing else" is the point, not a shortcut. A general expression
/// would have to be evaluated per row into a hash table, which is the
/// cost this whole shape exists to avoid; and a general expression over
/// floats would make group identity float equality. So the admitted
/// forms are recognised **structurally** — a column, integer literals,
/// `/` then optionally `*` — and everything else keeps the teaching
/// error.
fn lower_group_key(expr: &ast::Expr) -> Result<GroupKey, QueryError> {
    // `(ts / 60) * 60` — the bucket's start value. The parenthesised
    // left side arrives as `Nested`; a query that omits the parens gets
    // the same tree by precedence, so both spellings land here.
    if let ast::Expr::BinaryOp {
        left,
        op: ast::BinaryOperator::Multiply,
        right,
    } = expr
    {
        let inner = match left.as_ref() {
            ast::Expr::Nested(inner) => inner.as_ref(),
            other => other,
        };
        if let GroupKey::Bucket {
            column,
            divide,
            multiply: None,
        } = lower_group_key(inner)?
        {
            return Ok(GroupKey::Bucket {
                column,
                divide,
                multiply: Some(bucket_literal(right)?),
            });
        }
        return Err(QueryError::Unsupported(format!(
            "GROUP BY '{expr}' — the only arithmetic GROUP BY admits is a \
             bucket of the ordering key: ts / <width>, or (ts / <width>) * \
             <width> for the bucket's start"
        )));
    }
    match expr {
        ast::Expr::Identifier(column) => Ok(GroupKey::Column(ident(column))),
        ast::Expr::Nested(inner) => lower_group_key(inner),
        // `/` and `//` both mean truncating integer division here, and
        // that is not a fudge: a bucket divides an integer by an
        // integer, and a DOUBLE cannot key a group, so there is exactly
        // one meaning available in this position. Accepting both
        // spellings means the ISO/PostgreSQL habit (`/` truncates) and
        // the DuckDB habit (`/` is float, `//` truncates) each write
        // what they mean and get it.
        //
        // It does constrain #40: when exact integer expression
        // arithmetic reaches projection, `ts / 60` there must truncate
        // too, or the same text would mean two things in two clauses.
        // ISO says truncate, so that is where #40 should land anyway.
        ast::Expr::BinaryOp {
            left,
            op: ast::BinaryOperator::Divide | ast::BinaryOperator::DuckIntegerDivide,
            right,
        } => match left.as_ref() {
            ast::Expr::Identifier(column) => Ok(GroupKey::Bucket {
                column: ident(column),
                divide: bucket_literal(right)?,
                multiply: None,
            }),
            other => Err(QueryError::Unsupported(format!(
                "GROUP BY '{other} / …' — a bucket divides the ordering key \
                 itself, not an expression"
            ))),
        },
        other => Err(QueryError::Unsupported(format!(
            "GROUP BY '{other}' (a column, or a bucket of the ordering key: \
             ts / <width>)"
        ))),
    }
}

/// A bucket width or multiplier: a **positive** integer literal.
///
/// Positive because that is what makes the arithmetic monotone, which
/// is the entire licence for streaming the grouping; zero would divide
/// by zero and a negative would reverse the order the buckets come out
/// in. In the key's own units — there is no `INTERVAL` type, so a
/// minute over nanosecond stamps is `60000000000`.
fn bucket_literal(expr: &ast::Expr) -> Result<i64, QueryError> {
    let ast::Expr::Value(value) = expr else {
        return Err(QueryError::Unsupported(format!(
            "bucket width '{expr}' must be an integer literal"
        )));
    };
    let ast::Value::Number(number, _) = &value.value else {
        return Err(QueryError::Unsupported(format!(
            "bucket width '{expr}' must be an integer literal"
        )));
    };
    match number.parse::<i64>() {
        Ok(width) if width > 0 => Ok(width),
        Ok(_) => Err(QueryError::Unsupported(format!(
            "bucket width '{number}' must be positive — a bucket's width is \
             what makes the arithmetic monotone"
        ))),
        Err(_) => Err(QueryError::Unsupported(format!(
            "bucket width '{number}' must be an integer (in the ordering \
             key's own units — no INTERVAL type)"
        ))),
    }
}

fn lower_agg_item(item: &ast::SelectItem, keys: &[GroupKey]) -> Result<AggItem, QueryError> {
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
        // A bare column or a bucket: either way it must be one of the
        // GROUP BY terms, matched as a whole — `SELECT ts / 60 … GROUP
        // BY ts / 300` names two different buckets, not one.
        ast::Expr::Identifier(_) | ast::Expr::BinaryOp { .. } | ast::Expr::Nested(_) => {
            let key = lower_group_key(expr)?;
            if !keys.contains(&key) {
                return Err(QueryError::Unsupported(format!(
                    "'{expr}' must appear in GROUP BY or an aggregate"
                )));
            }
            Ok(AggItem::Key { key, alias })
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
        ast::Expr::Function(function) if function.over.is_some() => Ok(PlanItem::Window {
            call: lower_window_call(function)?,
            alias,
        }),
        other => {
            let mut windows = Vec::new();
            let scalar = lower_scalar_expr(other, &mut windows)?;
            Ok(PlanItem::Computed {
                expr: scalar,
                windows,
                name: alias.unwrap_or_else(|| other.to_string()),
            })
        }
    }
}

/// Lowers a scalar expression for the computed-projection slot (#49):
/// arithmetic, the built-in scalar functions, and `CASE` with WHERE
/// grammar conditions. Anything else is refused loudly.
fn lower_scalar_expr(
    expr: &ast::Expr,
    windows: &mut Vec<WindowCall>,
) -> Result<ScalarExpr, QueryError> {
    match expr {
        ast::Expr::Nested(inner) => lower_scalar_expr(inner, windows),
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
        } => Ok(ScalarExpr::Negate(Box::new(lower_scalar_expr(
            expr, windows,
        )?))),
        ast::Expr::UnaryOp {
            op: ast::UnaryOperator::Plus,
            expr,
        } => lower_scalar_expr(expr, windows),
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
                left: Box::new(lower_scalar_expr(left, windows)?),
                right: Box::new(lower_scalar_expr(right, windows)?),
            })
        }
        ast::Expr::Function(function) => {
            let name = object_name(&function.name)?.to_lowercase();
            // A window call inside arithmetic: hoist it, leaving a
            // placeholder. Standard SQL computes windows first and runs
            // the SELECT list's expressions over their results, so this
            // honours the existing order rather than inventing one.
            if function.over.is_some() {
                let call = lower_window_call(function)?;
                windows.push(call);
                return Ok(ScalarExpr::Window(windows.len() - 1));
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
                args.push(lower_scalar_expr(expr, windows)?);
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
                args: vec![lower_scalar_expr(expr, windows)?],
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
                    lower_scalar_expr(&case_when.result, windows)?,
                ));
            }
            let otherwise = else_result
                .as_ref()
                .map(|expr| lower_scalar_expr(expr, windows).map(Box::new))
                .transpose()?;
            Ok(ScalarExpr::Case { whens, otherwise })
        }
        other => Err(QueryError::Unsupported(format!(
            "expression '{other}' in a projection"
        ))),
    }
}

fn lower_window_call(function: &ast::Function) -> Result<WindowCall, QueryError> {
    let name = object_name(&function.name)?.to_lowercase();
    let Some(over) = &function.over else {
        return Err(QueryError::Unsupported(format!(
            "plain call '{name}' outside an aggregate projection"
        )));
    };
    let ast::WindowType::WindowSpec(spec) = over else {
        return Err(QueryError::Unsupported("named WINDOW clauses".to_owned()));
    };
    // The partition term takes the same shapes GROUP BY does: a column,
    // or a bucket of the ordering key. Which direction the window runs
    // in is decided by which column it names — down one symbol through
    // time, or across every symbol at one instant.
    let partition_by = spec
        .partition_by
        .iter()
        .map(|expr| {
            lower_group_key(expr).map_err(|_| {
                QueryError::Unsupported(format!(
                    "PARTITION BY '{expr}' — a column, or a bucket of the ordering \
                     key (ts / <width>) for a cross-sectional window"
                ))
            })
        })
        .collect::<Result<Vec<GroupKey>, QueryError>>()?;
    // ORDER BY is optional, and its absence is meaningful rather than
    // sloppy: standard SQL gives an unordered window the whole
    // partition as its frame, which is exactly a cross-sectional
    // statistic — every row of the instant against every other.
    let order_column = match spec.order_by.as_slice() {
        [] => None,
        [order] => {
            let ast::Expr::Identifier(column) = &order.expr else {
                return Err(QueryError::Unsupported(
                    "ORDER BY must be a plain column".to_owned(),
                ));
            };
            if order.options.asc == Some(false) {
                return Err(QueryError::Unsupported("ORDER BY ... DESC".to_owned()));
            }
            Some(ident(column))
        }
        _ => {
            return Err(QueryError::Unsupported(
                "ORDER BY must be a single column".to_owned(),
            ))
        }
    };
    if let Some(lead) = positional_window(&name) {
        // A positional lookup needs somewhere to look: without an
        // order, "the previous row" names nothing.
        let Some(order_column) = order_column else {
            return Err(QueryError::Unsupported(format!(
                "{name} needs an ORDER BY — it reads the previous or next row, \
                 and an unordered window has neither"
            )));
        };
        // A frame clause on `LAG`/`LEAD` is meaningless — the function
        // reads one specific row, not a range of them — and standard
        // SQL accordingly gives them none. Refuse a frame rather than
        // silently ignoring what the user wrote.
        if spec.window_frame.is_some() {
            return Err(QueryError::Unsupported(format!(
                "{name} reads one row, so it takes no frame — drop the \
                 ROWS/RANGE clause"
            )));
        }
        let (column, offset) = lower_offset_args(&name, &function.args)?;
        return Ok(WindowCall::Value {
            lead,
            column,
            offset,
            partition_by,
            order_by: order_column,
        });
    }
    let args = lower_args(&function.args)?;
    // With no order there is nothing for a frame to be relative to, so
    // standard SQL's answer — the whole partition — is the only one
    // available, and a frame clause beside it is a contradiction rather
    // than an extra.
    let frame = match &order_column {
        Some(_) => lower_frame(spec.window_frame.as_ref())?,
        None if spec.window_frame.is_some() => {
            return Err(QueryError::Unsupported(
                "a frame needs an ORDER BY to be relative to — an unordered                  window already covers its whole partition"
                    .to_owned(),
            ))
        }
        None => Frame::Partition,
    };
    Ok(WindowCall::Agg {
        function: name,
        args,
        partition_by,
        order_by: order_column,
        frame,
    })
}

/// Whether `name` is a positional window function, and which way it
/// looks. These are the only window calls whose arguments are not all
/// columns, so they are recognized before the columns-only rule runs.
fn positional_window(name: &str) -> Option<bool> {
    match name {
        "lag" => Some(false),
        "lead" => Some(true),
        _ => None,
    }
}

/// `LAG`/`LEAD`'s arguments: the column, and an optional positive
/// offset defaulting to 1 (SQL's own default). A third `default`
/// argument is standard but unbuilt — refused by name rather than
/// silently dropped, because dropping it would change answers.
fn lower_offset_args(
    name: &str,
    args: &ast::FunctionArguments,
) -> Result<(String, usize), QueryError> {
    let ast::FunctionArguments::List(list) = args else {
        return Err(QueryError::Unsupported(format!(
            "{name} without an argument list"
        )));
    };
    let mut column: Option<String> = None;
    let mut offset: Option<usize> = None;
    for (position, argument) in list.args.iter().enumerate() {
        let ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(expr)) = argument else {
            return Err(QueryError::Unsupported(format!(
                "{name}'s arguments must be a column and an optional offset"
            )));
        };
        match (position, expr) {
            (0, ast::Expr::Identifier(name)) => column = Some(ident(name)),
            (1, ast::Expr::Value(value)) => {
                let ast::Value::Number(number, _) = &value.value else {
                    return Err(QueryError::Unsupported(format!(
                        "{name}'s offset must be a literal positive integer"
                    )));
                };
                let parsed = number.parse::<usize>().map_err(|_| {
                    QueryError::Unsupported(format!(
                        "{name}'s offset must be a literal positive integer, got '{number}'"
                    ))
                })?;
                if parsed == 0 {
                    return Err(QueryError::Unsupported(format!(
                        "{name}'s offset must be at least 1 (offset 0 is the row itself)"
                    )));
                }
                offset = Some(parsed);
            }
            (2, _) => {
                return Err(QueryError::Unsupported(format!(
                    "{name}'s third (default) argument"
                )))
            }
            _ => {
                return Err(QueryError::Unsupported(format!(
                    "{name}'s arguments must be a column and an optional offset"
                )))
            }
        }
    }
    let Some(column) = column else {
        return Err(QueryError::Unsupported(format!("{name} needs a column")));
    };
    Ok((column, offset.unwrap_or(1)))
}

/// Accepts `ROWS BETWEEN <n | UNBOUNDED> PRECEDING AND CURRENT ROW` and
/// `RANGE BETWEEN <v> PRECEDING AND CURRENT ROW`. `GROUPS` is refused —
/// it needs peer-group semantics nothing here has.
fn lower_frame(frame: Option<&ast::WindowFrame>) -> Result<Frame, QueryError> {
    let Some(frame) = frame else {
        return Err(QueryError::Unsupported(
            "window without a frame (write ROWS BETWEEN n PRECEDING AND CURRENT ROW)".to_owned(),
        ));
    };
    if frame.units == ast::WindowFrameUnits::Groups {
        return Err(QueryError::Unsupported(
            "GROUPS frames (ROWS or RANGE)".to_owned(),
        ));
    }
    let range = frame.units == ast::WindowFrameUnits::Range;
    let bound = match &frame.start_bound {
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
            Some(number.clone())
        }
        _ => {
            return Err(QueryError::Unsupported(
                "frame must start at n PRECEDING or UNBOUNDED PRECEDING".to_owned(),
            ))
        }
    };
    if !matches!(frame.end_bound, Some(ast::WindowFrameBound::CurrentRow)) {
        return Err(QueryError::Unsupported(
            "frame must end at CURRENT ROW".to_owned(),
        ));
    }
    if range {
        // UNBOUNDED PRECEDING under RANGE is the whole run either way —
        // the row count and the value span agree — so it lowers to the
        // ROWS form and keeps that path's incremental sweep.
        let Some(number) = bound else {
            return Ok(Frame::Rows(None));
        };
        let span = number.parse::<u64>().map_err(|_| {
            QueryError::Unsupported(format!(
                "RANGE bound '{number}' must be a non-negative integer in the \
                 ordering key's own units (there is no INTERVAL type)"
            ))
        })?;
        return Ok(Frame::Range(span));
    }
    match bound {
        None => Ok(Frame::Rows(None)),
        Some(number) => Ok(Frame::Rows(Some(number.parse::<usize>().map_err(
            |_| QueryError::Unsupported(format!("frame bound '{number}'")),
        )?))),
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
                PlanItem::Window {
                    call: WindowCall::Agg {
                        function: "regr_slope".into(),
                        args: vec!["y".into(), "x".into()],
                        partition_by: vec![GroupKey::Column("sym".into())],
                        order_by: Some("ts".into()),
                        frame: Frame::Rows(Some(19)),
                    },
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
            Projection::Items(vec![PlanItem::Window {
                call: WindowCall::Agg {
                    function: "mean".into(),
                    args: vec!["x".into()],
                    partition_by: Vec::new(),
                    order_by: Some("ts".into()),
                    frame: Frame::Rows(Some(2)),
                },
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
    fn an_asof_join_lifts_its_keyword_and_reads_the_time_axis() {
        // #65's hybrid: the ASOF token is spliced out by byte span and
        // the remainder parses as an ordinary join, so no fork of
        // sqlparser is needed.
        let lifted = plan("SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym").unwrap();
        let join = lifted.join.expect("joined");
        assert_eq!(join.as_of, Some(AsOfMatch::AtOrBefore), "implicit axis");
        assert!(join.left, "ASOF LEFT JOIN keeps unmatched fact rows");
        assert_eq!(join.fact_key, "sym");
        assert_eq!(join.dimension_key, "sym");
        let inner = plan("SELECT t.x FROM t ASOF INNER JOIN q ON t.sym = q.sym").unwrap();
        assert!(!inner.join.expect("joined").left, "INNER drops them");
        // An explicit inequality is permitted and only selects the
        // comparison; either side order says the same thing.
        for (sql, expected) in [
            (
                "SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND q.ts <= t.ts",
                AsOfMatch::AtOrBefore,
            ),
            (
                "SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND t.ts >= q.ts",
                AsOfMatch::AtOrBefore,
            ),
            (
                "SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND q.ts < t.ts",
                AsOfMatch::StrictlyBefore,
            ),
        ] {
            let join = plan(sql).unwrap().join.expect("joined");
            assert_eq!(join.as_of, Some(expected), "{sql}");
            // The columns travel fact-side first whichever way round
            // the query wrote them; only the executor, which has the
            // schemas, can say whether they name the ordering keys.
            assert_eq!(
                join.as_of_named,
                Some(("ts".to_owned(), "ts".to_owned())),
                "{sql}"
            );
        }
        // Written backwards, the inequality asks for the quote *after*
        // each trade — a different question, refused rather than
        // silently answered in reverse. The operator alone cannot tell:
        // both of these are `<=`, and only the qualifiers separate them.
        let error = plan("SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND t.ts <= q.ts")
            .unwrap_err()
            .to_string();
        assert!(error.contains("looks backwards"), "{error}");
        // Unqualified on both sides, nothing says which table is which.
        let error = plan("SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND ts <= ts")
            .unwrap_err()
            .to_string();
        assert!(error.contains("qualify at least one side"), "{error}");
        // An equality is not an ordering.
        let error = plan("SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND q.ts = t.ts")
            .unwrap_err()
            .to_string();
        assert!(error.contains("must be one of"), "{error}");
        // A plain join is untouched — no as-of anything.
        let plain = plan("SELECT t.x FROM t LEFT JOIN q ON t.sym = q.sym").unwrap();
        let plain = plain.join.expect("joined");
        assert_eq!(plain.as_of, None);
        assert_eq!(plain.as_of_named, None);
    }

    #[test]
    fn the_asof_join_lift_survives_comments_and_string_literals() {
        // The M4-close lesson: a pre-parse lift that reassembles from
        // tokens collapses the newline ending a `--` comment and
        // silently comments out the rest of the statement. This one
        // splices by byte span, so the comment stays a comment.
        let commented_join = plan(
            "SELECT t.x FROM t ASOF LEFT JOIN q -- pick the prior quote\n             ON t.sym = q.sym WHERE t.x > 1",
        )
        .expect("plans");
        assert!(commented_join.join.is_some(), "the join survived");
        assert!(commented_join.predicate.is_some(), "and so did the WHERE");
        // ASOF inside a comment or a string is inert.
        let inert = plan("SELECT x FROM t WHERE sym = 'ASOF LEFT JOIN'").unwrap();
        assert!(inert.join.is_none() && inert.as_of.is_none());
        let commented = plan("SELECT x FROM t /* ASOF LEFT JOIN q */ WHERE x > 1").unwrap();
        assert!(commented.join.is_none());
        // One token, two clauses, told apart by what follows it — the
        // lift separates them correctly. Combining them is then refused
        // for an unrelated and pre-existing reason (M4.4: a knowledge
        // cut binds to one table's sequence space, and the join
        // lowering does not carry it), which is the error the user
        // should see rather than a parse puzzle.
        let error = format!(
            "{}",
            plan("SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym ASOF 7").unwrap_err()
        );
        assert!(error.contains("AS OF with JOIN"), "{error}");
    }

    #[test]
    fn bare_asof_join_is_refused_by_name() {
        // Vendors genuinely diverge on bare as-of semantics, so the
        // user says which they mean (#65; standing revisit flagged
        // 2026-07-30).
        let error = format!(
            "{}",
            plan("SELECT t.x FROM t ASOF JOIN q ON t.sym = q.sym").unwrap_err()
        );
        assert!(error.contains("bare ASOF JOIN"), "{error}");
        assert!(error.contains("ASOF LEFT JOIN"), "{error}");
        assert!(error.contains("ASOF INNER JOIN"), "{error}");
        // And a comparison that is not an inequality is named, not
        // quietly ignored.
        let error = format!(
            "{}",
            plan("SELECT t.x FROM t ASOF LEFT JOIN q ON t.sym = q.sym AND q.ts <> t.ts")
                .unwrap_err()
        );
        assert!(error.contains("time comparison"), "{error}");
    }

    #[test]
    fn a_range_frame_parses_into_its_own_frame_kind() {
        // RANGE lowers to a value span rather than a row count. The
        // executor refuses it for now (its frames are not trailing —
        // standard SQL ends a RANGE frame at the current row's last
        // peer), but the planner must carry the distinction so that
        // refusal is not silently reinterpreted as ROWS.
        let ranged = plan(
            "SELECT sum(x) OVER (ORDER BY ts RANGE BETWEEN 300 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        let Projection::Items(items) = &ranged.projection else {
            panic!("items")
        };
        let PlanItem::Window {
            call: WindowCall::Agg { frame, .. },
            ..
        } = &items[0]
        else {
            panic!("window")
        };
        assert_eq!(*frame, Frame::Range(300));
        // UNBOUNDED PRECEDING means the whole run either way, so it
        // lowers to the ROWS form and keeps that path's sweep.
        let unbounded = plan(
            "SELECT sum(x) OVER (ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
             FROM t",
        )
        .unwrap();
        let Projection::Items(items) = &unbounded.projection else {
            panic!("items")
        };
        let PlanItem::Window {
            call: WindowCall::Agg { frame, .. },
            ..
        } = &items[0]
        else {
            panic!("window")
        };
        assert_eq!(*frame, Frame::Rows(None));
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
            // `x + 1` is arithmetic, but not the one monotone shape
            // GROUP BY admits (a bucket of the ordering key).
            (
                "SELECT x, sum(y) FROM t GROUP BY x, x + 1",
                "a bucket of the ordering key",
            ),
            ("SELECT x FROM t GROUP BY x LIMIT x", "LIMIT"),
            // (an unknown plain call like nope_agg(x) now lowers to a
            // Registered column function and is refused by name at
            // execution, where the registry lives — tested in engine)
            ("SELECT y FROM t GROUP BY x", "must appear in GROUP BY"),
            (
                "SELECT sum(x) OVER (ORDER BY ts GROUPS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t",
                "GROUPS frames",
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
        // Only the constraint position rewrites: a column may still be
        // named `ordering`, and a name that merely starts like the
        // phrase is not it.
        let Ok(Statement::CreateTable(plan)) = parse_statement(
            "CREATE TABLE t (ts BIGINT ORDERING KEY, ordering SYMBOL, primary_ish DOUBLE)",
        ) else {
            panic!("parses as CREATE TABLE")
        };
        assert_eq!(plan.columns.len(), 3);
        assert_eq!(plan.columns[1].name, "ordering");
        assert_eq!(plan.columns[1].type_name, "SYMBOL");
        assert!(!plan.columns[1].ordering_key);
        assert!(plan.columns[0].ordering_key);
        // And `ordering KEY` — a column definition under the retired
        // spelling — still reaches the refusal that names the new one,
        // rather than being rewritten into a parse error.
        let error = parse_statement("CREATE TABLE t (ts BIGINT ORDERING KEY, ordering KEY)")
            .expect_err("refused")
            .to_string();
        assert!(error.contains("spelled SYMBOL"), "{error}");
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
