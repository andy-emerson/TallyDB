//! Script-backed SQL functions: the Lua-in-SQL window slot (#41).
//!
//! A [`LuaWindow`] adapts an application-registered Lua kernel to
//! `query-lite`'s `WindowAggregate` seam — the same seam the curated
//! LAPACK windows (`regr_slope`, `eigen_max`, …) ship through. The
//! engine drives the framing; the kernel reduces one frame to one
//! scalar, reading its arguments as zero-copy column views and
//! returning a number or `NULL` — the window half of the vectorized
//! calling convention (DESIGN.md, *The Lua layer*).
//!
//! This is the promotion ladder's first rung made concrete: write the
//! kernel in Lua to get it correct — against the same executor, frames,
//! and null semantics the native ops use — and if it proves hot,
//! promote it to a curated native op. Same registry, same SQL surface,
//! so the promotion is invisible to queries.
//!
//! ## Concurrency
//!
//! `WindowAggregate` requires `Send + Sync`; a Lua interpreter is
//! single-threaded. The adapter holds its `LuaState` (which is `Send`,
//! with the safety argument written at the impl) behind a `Mutex`, so
//! frames evaluate one at a time per registered kernel — exactly the
//! interpreter's own constraint, made visible in the type.
//!
//! ## Statefulness
//!
//! One interpreter serves every frame of every query on its table, so
//! globals a kernel writes persist across calls. A kernel should be a
//! pure function of its frame; evaluation order beyond append order
//! within a partition is not a contract, and cross-frame state is
//! unsupported (it will not be preserved by future parallel execution).

use crate::table::{PairKind, PairStatistic, RegressionOutput, RollingRegression};
use arrow_lite::ColumnType;
use compute_blas::{BlasBackend, NativeBlas};
use compute_lapack::NativeLapack;
use compute_lua::{Chunk, ColumnView, HostFunction, LuaState, ReturnType, ScalarValue};
use query_lite::WindowAggregate;
use std::ffi::{CStr, CString};
use std::sync::Mutex;

/// Words that cannot serve as Lua parameter names: the language's
/// keywords, plus the `NULL` sentinel global a parameter must not
/// shadow.
const RESERVED: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if", "in",
    "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while", "NULL",
];

/// Whether `name` is a plain identifier (ASCII letter or underscore,
/// then letters, digits, underscores) — required of Lua parameter names
/// so kernels can actually reference them, and of SQL function names so
/// queries can actually call them.
pub(crate) fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `dot(x, y)` — BLAS `ddot` as a script-callable op, the cheap end of
/// the curated spread.
struct DotOp(NativeBlas);

impl HostFunction for DotOp {
    fn arity(&self) -> usize {
        2
    }
    fn call(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        self.0
            .dot(args[0], args[1])
            .map(Some)
            .map_err(|error| error.to_string())
    }
}

/// Adapts a native window statistic to the host-function seam — the
/// same argument shape (dense `f64` slices) and the same
/// undefined-is-NULL convention, so one implementation serves both the
/// SQL window registry and scripts.
struct CuratedOp<A>(A);

impl<A: WindowAggregate> HostFunction for CuratedOp<A> {
    fn arity(&self) -> usize {
        self.0.arity()
    }
    fn call(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        self.0.evaluate(args)
    }
}

/// Installs the curated compute spread into a kernel's state: `dot`
/// (BLAS), `regr_slope` / `regr_intercept` (least squares), and
/// `covar_pop` / `corr` / `eigen_max` (pair statistics) — the very
/// implementations the SQL windows run, reading the same view buffers
/// with no copy. This is the compute-without-copying surface inside a
/// script: engine buffers, curated native ops, and the interpreter all
/// share memory.
fn install_curated_ops(state: &mut LuaState) -> Result<(), String> {
    let lapack = NativeLapack;
    state.register_host_function("dot", Box::new(DotOp(NativeBlas)))?;
    for (name, output) in [
        ("regr_slope", RegressionOutput::Slope),
        ("regr_intercept", RegressionOutput::Intercept),
    ] {
        state.register_host_function(
            name,
            Box::new(CuratedOp(RollingRegression {
                backend: lapack,
                output,
            })),
        )?;
    }
    for (name, kind) in [
        ("covar_pop", PairKind::CovarPop),
        ("corr", PairKind::Corr),
        ("eigen_max", PairKind::EigenMax),
    ] {
        state.register_host_function(name, Box::new(CuratedOp(PairStatistic { kind })))?;
    }
    Ok(())
}

/// An application-registered Lua window kernel behind the
/// `WindowAggregate` seam.
pub(crate) struct LuaWindow {
    /// The interpreter and its compiled kernel, serialized per the
    /// concurrency note above. The chunk is compiled once at
    /// registration and called per window — parsing per frame is the
    /// dominant cost of a script kernel, and it belongs to neither.
    state: Mutex<(LuaState, Chunk)>,
    /// Positional argument names, bound as globals for each call.
    parameters: Vec<CString>,
    /// The declared output type (F2): `F64` or `I64`, fixed at
    /// registration.
    output: ColumnType,
}

impl LuaWindow {
    /// Builds the adapter, failing loudly at registration time on
    /// anything that would otherwise fail confusingly at query time: a
    /// key-typed output, an unusable parameter name, or a kernel that
    /// does not compile.
    pub(crate) fn new(
        parameters: &[&str],
        chunk: &str,
        output: ColumnType,
    ) -> Result<LuaWindow, String> {
        if output == ColumnType::Key {
            return Err(
                "a window's output is numeric (f64 or i64); key-typed windows are not supported"
                    .to_owned(),
            );
        }
        if parameters.is_empty() {
            return Err("a lua window takes at least one column argument".to_owned());
        }
        let mut names: Vec<CString> = Vec::with_capacity(parameters.len());
        for &parameter in parameters {
            if !is_identifier(parameter) {
                return Err(format!(
                    "parameter '{parameter}' is not a usable Lua identifier"
                ));
            }
            if RESERVED.contains(&parameter) {
                return Err(format!(
                    "parameter '{parameter}' is reserved in Lua kernels"
                ));
            }
            let name = CString::new(parameter).expect("identifier has no interior NUL");
            if names.contains(&name) {
                return Err(format!("parameter '{parameter}' appears twice"));
            }
            names.push(name);
        }
        let mut state = LuaState::new()?;
        install_curated_ops(&mut state)?;
        // Compiling here is both the loud-early syntax check and the
        // per-window saving: queries call the compiled function.
        let compiled = state.compile(chunk)?;
        Ok(LuaWindow {
            state: Mutex::new((state, compiled)),
            parameters: names,
            output,
        })
    }
}

impl WindowAggregate for LuaWindow {
    fn arity(&self) -> usize {
        self.parameters.len()
    }

    fn output_type(&self) -> ColumnType {
        self.output
    }

    fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| "lua window interpreter poisoned".to_owned())?;
        let (state, chunk) = &mut *guard;
        let views: Vec<(&CStr, ColumnView<'_>)> = self
            .parameters
            .iter()
            .zip(args)
            .map(|(name, &values)| {
                (
                    name.as_c_str(),
                    // The executor's window contract: non-null f64 rows.
                    ColumnView::F64 {
                        values,
                        validity: None,
                    },
                )
            })
            .collect();
        let declared = match self.output {
            ColumnType::I64 => ReturnType::I64,
            _ => ReturnType::F64, // Key refused at registration
        };
        match state.eval_scalar(chunk, &views, declared)? {
            ScalarValue::F64(value) => Ok(Some(value)),
            ScalarValue::I64(value) => {
                // The executor carries window results as f64 and casts
                // I64-typed columns back exactly at materialization
                // (B5). Exact-or-loud holds on the way in too (F3).
                let carrier = value as f64;
                if carrier as i128 == i128::from(value) {
                    Ok(Some(carrier))
                } else {
                    Err(format!(
                        "lua window result {value} does not fit the f64 result carrier exactly"
                    ))
                }
            }
            ScalarValue::Null => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    //! The B evidence: a Lua kernel run end-to-end through SQL matches a
    //! hand-computed reference — over multi-segment storage, through the
    //! same executor path as the native windows — plus the loud edges of
    //! registration and execution.

    use crate::{Database, RowValue, Table};
    use arrow_lite::{Column, ColumnType, Field, NumericData, Schema};

    /// The mean-absolute-deviation kernel — the crate's running example
    /// of a loop the built-ins don't cover.
    const MAD: &str = "local n = #x\n\
                       local mean = 0.0\n\
                       for i = 1, n do mean = mean + x[i] end\n\
                       mean = mean / n\n\
                       local mad = 0.0\n\
                       for i = 1, n do mad = mad + math.abs(x[i] - mean) end\n\
                       return mad / n";

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
        ])
    }

    /// Flattens an output column of f64 windows (None = SQL NULL).
    fn f64s(output: &query_lite::QueryOutput, index: usize) -> Vec<Option<f64>> {
        output
            .batches
            .iter()
            .flat_map(|batch| {
                let Column::Numeric(NumericData::F64(column)) = &batch.columns()[index] else {
                    panic!("expected an f64 column")
                };
                (0..column.len())
                    .map(|row| {
                        column
                            .is_valid(row)
                            .then(|| column.values().as_slice()[row])
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn lua_window_matches_the_reference_end_to_end() {
        // 12 rows over 4-row segments: windows span segment boundaries,
        // so the cross-segment gather path runs too.
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0];
        for (i, &x) in data.iter().enumerate() {
            table
                .append(&[
                    RowValue::I64(i as i64),
                    RowValue::Key("A"),
                    RowValue::F64(x),
                ])
                .unwrap();
        }
        table
            .register_lua_window("mad", &["x"], MAD, ColumnType::F64)
            .unwrap();
        let output = table
            .query(
                "SELECT mad(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) \
                 AS m FROM t",
            )
            .unwrap();
        let results = f64s(&output, 0);
        assert_eq!(results.len(), data.len());

        // The hand-computed spots: a one-row window deviates by zero;
        // the first full window {3,1,4,1} has mean 2.25 and deviations
        // {0.75, 1.25, 1.75, 1.25} — MAD exactly 1.25.
        assert_eq!(results[0], Some(0.0));
        assert_eq!(results[3], Some(1.25));

        // And every row against the same arithmetic in Rust — identical
        // fold order, so agreement is bit-exact, not approximate.
        for (row, result) in results.iter().enumerate() {
            let start = row.saturating_sub(3);
            let window = &data[start..=row];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let reference =
                window.iter().map(|v| (v - mean).abs()).sum::<f64>() / window.len() as f64;
            assert_eq!(
                result.map(f64::to_bits),
                Some(reference.to_bits()),
                "row {row}"
            );
        }
    }

    #[test]
    fn lua_window_agrees_with_the_builtin_oracle_partitioned() {
        // Two implementations of the same statistic in one query — the
        // Lua mean against the built-in AVG — over partitions that span
        // segments (per-segment dictionaries included). Same fold order
        // on both sides: bit-exact agreement expected.
        let mut db = Database::new();
        db.add_table(Table::with_segment_rows("t", schema(), "ts", 3).unwrap())
            .unwrap();
        for i in 0..14i64 {
            let sym = if i % 2 == 0 { "A" } else { "B" };
            db.append(
                "t",
                &[
                    RowValue::I64(i),
                    RowValue::Key(sym),
                    RowValue::F64((i * i % 17) as f64 - 5.0),
                ],
            )
            .unwrap();
        }
        db.table_mut("t")
            .unwrap()
            .register_lua_window(
                "lua_mean",
                &["x"],
                "local s = 0.0\nfor i = 1, #x do s = s + x[i] end\nreturn s / #x",
                ColumnType::F64,
            )
            .unwrap();
        let frame = "OVER (PARTITION BY sym ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)";
        let output = db
            .query(&format!(
                "SELECT lua_mean(x) {frame} AS ours, avg(x) {frame} AS theirs FROM t"
            ))
            .unwrap();
        let ours = f64s(&output, 0);
        let theirs = f64s(&output, 1);
        assert_eq!(ours.len(), 14);
        for row in 0..ours.len() {
            assert_eq!(
                ours[row].map(f64::to_bits),
                theirs[row].map(f64::to_bits),
                "row {row}"
            );
        }
    }

    #[test]
    fn i64_declared_kernel_yields_an_integer_column() {
        // The F2 hook end-to-end: the declared type, not the returned
        // value, fixes the output column's Arrow type (B5's cast-back).
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        for (i, x) in [1.5, -2.0, 3.0, -1.0, 4.0].iter().enumerate() {
            table
                .append(&[
                    RowValue::I64(i as i64),
                    RowValue::Key("A"),
                    RowValue::F64(*x),
                ])
                .unwrap();
        }
        table
            .register_lua_window(
                "n_pos",
                &["x"],
                "local n = 0\nfor i = 1, #x do if x[i] > 0 then n = n + 1 end end\nreturn n",
                ColumnType::I64,
            )
            .unwrap();
        let output = table
            .query(
                "SELECT n_pos(x) OVER (ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) \
                 AS n FROM t",
            )
            .unwrap();
        assert_eq!(output.schema.fields()[0].column_type(), ColumnType::I64);
        let values: Vec<i64> = output
            .batches
            .iter()
            .flat_map(|batch| {
                let Column::Numeric(NumericData::I64(column)) = &batch.columns()[0] else {
                    panic!("declared i64, got something else")
                };
                column.values().as_slice().to_vec()
            })
            .collect();
        assert_eq!(values, [1, 1, 2, 1, 2]);
    }

    #[test]
    fn kernel_null_becomes_sql_null() {
        // A kernel that declines a window (too few rows) returns NULL,
        // and the output column is nullable there — regr_slope's shape.
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        for i in 0..3i64 {
            table
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key("A"),
                    RowValue::F64(i as f64),
                ])
                .unwrap();
        }
        table
            .register_lua_window(
                "needs_two",
                &["x"],
                "if #x < 2 then return NULL end\nreturn x[#x] - x[1]",
                ColumnType::F64,
            )
            .unwrap();
        let output = table
            .query(
                "SELECT needs_two(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
                 AS d FROM t",
            )
            .unwrap();
        assert_eq!(f64s(&output, 0), [None, Some(1.0), Some(1.0)]);
    }

    #[test]
    fn kernel_runtime_errors_abort_the_query_loudly() {
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        table
            .append(&[RowValue::I64(0), RowValue::Key("A"), RowValue::F64(1.0)])
            .unwrap();
        table
            .register_lua_window("boom", &["x"], "error('kernel boom')", ColumnType::F64)
            .unwrap();
        let error = table
            .query(
                "SELECT boom(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
                 FROM t",
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("kernel boom"), "{error}");
    }

    #[test]
    fn registration_is_loud_about_everything_wrong() {
        let mut table = Table::new("t", schema(), "ts").unwrap();
        // A kernel that does not compile fails at registration, not at
        // first query.
        let error = table
            .register_lua_window("bad", &["x"], "return ((", ColumnType::F64)
            .unwrap_err()
            .to_string();
        assert!(error.contains("load"), "{error}");
        // Key-typed windows are refused.
        let error = table
            .register_lua_window("k", &["x"], "return 0", ColumnType::Key)
            .unwrap_err()
            .to_string();
        assert!(error.contains("key-typed windows"), "{error}");
        // Unusable parameter names: not an identifier, a Lua keyword,
        // the NULL global, a duplicate.
        for parameters in [&["2x"][..], &["end"][..], &["NULL"][..], &["x", "x"][..]] {
            assert!(
                table
                    .register_lua_window("f", parameters, "return 0", ColumnType::F64)
                    .is_err(),
                "{parameters:?} should be refused"
            );
        }
        // A function name SQL cannot call back is refused too.
        let error = table
            .register_lua_window("my func", &["x"], "return 0", ColumnType::F64)
            .unwrap_err()
            .to_string();
        assert!(error.contains("function name"), "{error}");
        // No argument columns: refused (the executor feeds columns).
        let error = table
            .register_lua_window("f", &[], "return 0", ColumnType::F64)
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least one"), "{error}");
    }

    // ---- the curated-op spread (increment D) ----

    fn pair_schema() -> Schema {
        Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("sym", ColumnType::Key, false),
            Field::new("x", ColumnType::F64, false),
            Field::new("y", ColumnType::F64, false),
        ])
    }

    #[test]
    fn curated_ops_from_lua_agree_with_their_sql_window_selves() {
        // The same native implementations serve the SQL window registry
        // and the script surface; running both paths over one dataset in
        // one query must agree bit-for-bit, NULLs included (short
        // windows are undefined on both sides). Data spans segments.
        let mut table = Table::with_segment_rows("t", pair_schema(), "ts", 4).unwrap();
        for i in 0..14i64 {
            let x = (i as f64) * 0.5 + ((i % 3) as f64);
            let y = 2.5 * x + ((i % 5) as f64) * 0.25 - 1.0;
            table
                .append(&[
                    RowValue::I64(i),
                    RowValue::Key("A"),
                    RowValue::F64(x),
                    RowValue::F64(y),
                ])
                .unwrap();
        }
        for (name, op) in [
            ("lua_regr", "regr_slope"),
            ("lua_intercept", "regr_intercept"),
            ("lua_covar", "covar_pop"),
            ("lua_corr", "corr"),
            ("lua_eigen", "eigen_max"),
        ] {
            table
                .register_lua_window(
                    name,
                    &["y", "x"],
                    &format!("return {op}(y, x)"),
                    ColumnType::F64,
                )
                .unwrap();
        }
        let frame = "OVER (ORDER BY ts ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)";
        for (lua_name, native_name) in [
            ("lua_regr", "regr_slope"),
            ("lua_intercept", "regr_intercept"),
            ("lua_covar", "covar_pop"),
            ("lua_corr", "corr"),
            ("lua_eigen", "eigen_max"),
        ] {
            let output = table
                .query(&format!(
                    "SELECT {native_name}(y, x) {frame} AS native, \
                     {lua_name}(y, x) {frame} AS scripted FROM t"
                ))
                .unwrap();
            let native = f64s(&output, 0);
            let scripted = f64s(&output, 1);
            assert_eq!(native.len(), 14);
            for row in 0..native.len() {
                assert_eq!(
                    native[row].map(f64::to_bits),
                    scripted[row].map(f64::to_bits),
                    "{native_name} row {row}"
                );
            }
        }
    }

    #[test]
    fn dot_from_lua_matches_the_blas_backend() {
        use compute_blas::{BlasBackend, NativeBlas};
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        let data: Vec<f64> = (0..10).map(|i| (i as f64) * 0.75 - 3.0).collect();
        for (i, &x) in data.iter().enumerate() {
            table
                .append(&[
                    RowValue::I64(i as i64),
                    RowValue::Key("A"),
                    RowValue::F64(x),
                ])
                .unwrap();
        }
        table
            .register_lua_window("sumsq", &["x"], "return dot(x, x)", ColumnType::F64)
            .unwrap();
        let output = table
            .query(
                "SELECT sumsq(x) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) \
                 AS s FROM t",
            )
            .unwrap();
        let results = f64s(&output, 0);
        // The same ddot on the same window slices: identical bits.
        let backend = NativeBlas;
        for (row, result) in results.iter().enumerate() {
            let window = &data[row.saturating_sub(3)..=row];
            let reference = backend.dot(window, window).unwrap();
            assert_eq!(
                result.map(f64::to_bits),
                Some(reference.to_bits()),
                "row {row}"
            );
        }
    }

    #[test]
    fn database_routes_registration_to_its_table() {
        let mut db = Database::new();
        db.create_table("t", schema(), "ts").unwrap();
        db.append(
            "t",
            &[RowValue::I64(0), RowValue::Key("A"), RowValue::F64(2.0)],
        )
        .unwrap();
        db.register_lua_window(
            "t",
            "double_last",
            &["x"],
            "return 2 * x[#x]",
            ColumnType::F64,
        )
        .unwrap();
        let output = db
            .query(
                "SELECT double_last(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING \
                 AND CURRENT ROW) AS d FROM t",
            )
            .unwrap();
        assert_eq!(f64s(&output, 0), [Some(4.0)]);
        // Unknown table: the database-level error, not a panic.
        assert!(db
            .register_lua_window("nope", "f", &["x"], "return 0", ColumnType::F64)
            .is_err());
    }
}
