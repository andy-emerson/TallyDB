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
//! planner (see DESIGN.md). Instead: DuckDB (primary) and DataFusion
//! (secondary) are used as a **differential correctness oracle** in this
//! crate's test suite — run the same query against the oracle and against
//! this executor, diff the output. DuckDB is primary because it has the
//! broadest standard analytic-SQL semantics (windows, statistical
//! aggregates) and runs in-process inside `cargo test`; DataFusion as
//! secondary also covers InfluxDB-compatible semantics directly, since
//! Influx v3's SQL engine is DataFusion. That's the primary correctness
//! strategy for this crate. Write tests this way from the start, not as an
//! afterthought.
//!
//! ## SQL surface — inclusion principle
//! Any standard SQL function or verb is in scope as long as it (a) doesn't
//! require a non-numeric, non-key column type, and (b) doesn't require a
//! general-purpose cost-based optimizer. That's the actual filter — "can
//! we think of a quant use case for it" is NOT the filter, and is not a
//! reason to exclude something otherwise in scope. Concretely in scope:
//! `SELECT` / `WHERE` / `GROUP BY` / `ORDER BY`, equi-joins, window
//! functions, and `UPDATE` / `DELETE` (implemented as tombstone +
//! reinsert against `storage-lite`, not a separate mutation path — see
//! that crate's docs). Concretely out of scope for now: general
//! subqueries/CTEs, string-*producing* functions (`SUBSTRING`, `CONCAT`,
//! `CAST AS VARCHAR`, `GROUP_CONCAT` — a produced string is a value that is
//! neither numeric nor key), and a cost-based join planner beyond
//! star-schema equi-joins.
//!
//! ## Strings: predicates in, production out
//! numeric-or-key holds across the whole pipeline (results and intermediates,
//! not just stored columns), but that does NOT mean "no string operations."
//! Key columns are dictionary-encoded interned strings, so string
//! **predicates** on keys — `=`, `IN`, `LIKE`, regex — are in scope: they
//! emit a row selection, not a string. Implement them efficiently: evaluate
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
//! numeric-heavy, this is exactly the shape of work `compute-blas` is
//! built to accelerate; keep that seam in mind rather than reimplementing
//! matrix-shaped math by hand.

pub mod exec;
pub mod plan;

pub use exec::{execute, QueryOutput, Registry, WindowAggregate};
pub use plan::{plan, Plan, PlanItem, QueryError};

// TODO: executor: filter (WHERE) / group-by / aggregate over
//       storage-lite segments
// TODO: executor: equi-join (star-schema shape: one fact table, small
//       dimension tables)
// TODO: executor: window functions over ordered numeric columns
// TODO: UPDATE / DELETE -> storage-lite tombstone + reinsert
// TODO: differential test harness against DuckDB/DataFusion (dev-only
//       dependency, never a runtime one)
