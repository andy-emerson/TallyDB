//! `query-lite` — scoped SQL parsing and execution over `storage-lite`.
//!
//! ## Parsing: taken as-is
//! Use `sqlparser-rs` for parsing. Do not write a parser from scratch and
//! do not fork/vendor `sqlparser-rs` — it's a mature, MIT-licensed,
//! narrow-purpose dependency, exactly the kind of thing this project takes
//! whole rather than reimplementing (see DESIGN.md, "Design philosophy").
//! We use a *subset* of what it can parse; the subsetting happens in what
//! AST nodes this crate's executor handles, not in the parser itself.
//!
//! ## Execution: original work, validated against an oracle
//! The executor (turning a parsed AST into results over `storage-lite`
//! data) is our own code — DataFusion's executor is deliberately NOT
//! vendored, because its useful parts are coupled to its own general
//! planner (see DESIGN.md). Instead, DuckDB is used as a **differential
//! correctness oracle**: the same query runs against the oracle and
//! against this executor, and the outputs are diffed. DuckDB earns
//! primary because it has the broadest standard analytic-SQL semantics
//! (windows, statistical aggregates).
//!
//! That differential does **not** live in this crate. It runs from
//! Python against the `engine` cdylib — `crates/engine/tests/
//! m2_differential_oracle.py` and its siblings — because an oracle
//! should exercise the whole vertical slice (storage round trip, Arrow
//! export) rather than this crate's internals, and because keeping
//! DuckDB out of the dependency graph entirely is a deliberate choice,
//! not an oversight. This crate's own tests cover planning and
//! execution units; the cross-checks are external, and that split is
//! the correctness strategy.
//!
//! ## SQL surface — inclusion principle
//! Any standard SQL function or verb is in scope as long as it (a) doesn't
//! require a non-numeric, non-key column type, and (b) doesn't require a
//! general-purpose cost-based optimizer. That's the actual filter — "can
//! we think of a quant use case for it" is NOT the filter, and is not a
//! reason to exclude something otherwise in scope. Concretely in scope
//! (and built): `SELECT` / `WHERE` / `GROUP BY` (`HAVING` included) /
//! `ORDER BY` (`NULLS FIRST`/`LAST` included) / `LIMIT`, `DISTINCT`,
//! scalar expressions and `CASE` in projection, equi-joins and as-of
//! joins, window functions, `CREATE TABLE` / `INSERT` (lowered here,
//! executed by the
//! embedder), and `UPDATE` / `DELETE` (implemented as tombstone +
//! reinsert against `storage-lite`, not a separate mutation path — see
//! that crate's docs). Concretely out of scope for now: general
//! subqueries/CTEs, string-*producing* functions (`SUBSTRING`, `CONCAT`,
//! `CAST AS VARCHAR`, `GROUP_CONCAT` — a produced string is a value that is
//! neither numeric nor key), and a cost-based join planner beyond
//! star-schema equi-joins and the as-of family.
//!
//! ## Strings: predicates in, production out
//! numeric-or-key holds across the whole pipeline (results and intermediates,
//! not just stored columns), but that does NOT mean "no string operations."
//! Key columns are dictionary-encoded interned strings, so string
//! **predicates** on keys — `=`, `IN`, `LIKE`, regex — are in scope: they
//! emit a row selection, not a string. (`=`, `IN`, and `LIKE` are built;
//! regex is in scope but not yet implemented, rejected loudly until
//! then — #57.) Implement them efficiently: evaluate
//! the predicate once per *distinct* value in the small dictionary to get a
//! bitmap over dictionary indices, then filter rows by integer set
//! membership — never re-run the string match per row. What's out is any
//! function that *emits* a string value (see the out-of-scope list above); a
//! key result leaves the engine as its integer code plus the dictionary
//! needed to render it.
//!
//! ## Window functions
//! These are the highest-value part of the SQL surface for the target
//! workload (rolling aggregates over ordered numeric data) and deserve
//! first-class, hand-written implementations here — not a generic,
//! bolted-on afterthought. Where a window function's inner loop is
//! numeric-heavy, this is exactly the shape of work `compute-linalg` is
//! built to accelerate; keep that seam in mind rather than reimplementing
//! matrix-shaped math by hand.

pub mod exec;
pub mod plan;
pub mod predicate;

pub use exec::{
    contiguous, execute, execute_join, recompute_frames, ColumnFunction, JoinSide, QueryOutput,
    Registry, ViewScalars, WindowAggregate,
};
pub use plan::{
    parse_statement, plan, AggCall, AggFunction, AggItem, AsOfMatch, Assignment, ColumnSpec,
    CreateTablePlan, DeletePlan, Frame, InsertPlan, InsertValue, JoinPlan, OrderBy, Plan, PlanItem,
    Projection, QueryError, SetValue, Statement, UpdatePlan, SEQUENCE_COLUMN,
};
pub use predicate::{
    can_match, evaluate as evaluate_predicate, CmpOp, NoScalars, Number, Predicate, ScalarEval,
};

// TODO: DataFusion as a secondary differential oracle beside DuckDB
// TODO: window ORDER BY beyond the ordering key; DISTINCT aggregates
//       — as the inclusion principle admits them
