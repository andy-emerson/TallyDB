//! Script-backed SQL functions: the Lua-in-SQL window slot (#41).
//!
//! A [`LuaWindow`] adapts an application-registered Lua kernel to
//! `query-lite`'s `WindowAggregate` seam — the same seam the curated
//! native windows (`regr_slope`, `eigen_max`, …) ship through. The
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

use arrow_lite::{Bitmap, ColumnType};
use compute_linalg::{LinalgBackend, RustLinalg};
use compute_lua::{
    Chunk, ColumnView, HostFunction, LogSink, LuaState, OutputColumn, ReturnType, ScalarValue,
};
use query_lite::{Registry, WindowAggregate};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

/// One embedder-installed sink shared by every kernel of a table: each
/// `LuaState` owns its sink box, so a shared destination crosses as an
/// `Arc` behind this forwarding shim.
struct SharedSink(Arc<dyn LogSink + Sync>);

impl LogSink for SharedSink {
    fn log(&self, message: &str) {
        self.0.log(message);
    }
}

/// Words that cannot serve as Lua parameter names: the language's
/// keywords, plus the `NULL` sentinel global a parameter must not
/// shadow.
const RESERVED: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if", "in",
    "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while", "NULL",
];

use crate::table::is_identifier;

/// `dot(x, y)` — the backend dot product as a script-callable op, the
/// cheap end of the curated spread.
struct DotOp(RustLinalg);

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

/// Adapts a registered window implementation to the host-function seam
/// — the same argument shape (dense `f64` slices) and the same
/// undefined-is-NULL convention, so one implementation serves both the
/// SQL window registry and scripts.
struct RegistryOp(Arc<dyn WindowAggregate>);

impl HostFunction for RegistryOp {
    fn arity(&self) -> usize {
        self.0.arity()
    }
    fn call(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
        self.0.evaluate(args)
    }
}

/// Installs the whole registered vocabulary into a kernel's state —
/// **the vocabulary invariant: anything SQL can call, a Lua kernel can
/// call**, by the same name, over the same view buffers with no copy.
/// Registry-driven rather than a hardcoded list, so every future
/// native (and every embedder-registered kernel that exists at this
/// registration) flows into scripts for free. `dot` (compute-linalg)
/// rides along as the one op that lives outside the registry. This is
/// the compute-without-copying surface inside a script: engine
/// buffers, native ops, and the interpreter all share memory.
fn install_vocabulary(state: &mut LuaState, ops: &Registry) -> Result<(), String> {
    state.register_host_function("dot", Box::new(DotOp(RustLinalg)))?;
    for (name, aggregate) in ops.entries() {
        state.register_host_function(name, Box::new(RegistryOp(Arc::clone(aggregate))))?;
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
        log_sink: Option<Arc<dyn LogSink + Sync>>,
        ops: &Registry,
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
        install_vocabulary(&mut state, ops)?;
        if let Some(sink) = log_sink {
            state.set_log_sink(Box::new(SharedSink(sink)));
        }
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

/// An application-registered Lua *column* kernel behind the
/// [`query_lite::ColumnFunction`] seam — the vectorized whole-column
/// shape (#53): the arguments bind as whole-column views, the script
/// fills the preallocated `out` column, and the interpreter is entered
/// **once per view**, never per row. This is what makes a scripted
/// per-row function viable in bulk: the loops the script writes run
/// over columns already in memory, and the boundary is crossed once.
pub(crate) struct LuaColumn {
    /// The interpreter and its compiled kernel, serialized exactly as
    /// [`LuaWindow`]'s (see the module's concurrency note).
    state: Mutex<(LuaState, Chunk)>,
    /// Positional argument names, bound as globals for each call.
    parameters: Vec<CString>,
}

impl LuaColumn {
    /// Builds the adapter with [`LuaWindow::new`]'s validation posture:
    /// everything that can fail confusingly at query time fails loudly
    /// here instead. The output is nullable `f64` (slots the script
    /// never writes come back NULL); exact-`i64` and key outputs are
    /// deferred surface.
    pub(crate) fn new(
        parameters: &[&str],
        chunk: &str,
        log_sink: Option<Arc<dyn LogSink + Sync>>,
        ops: &Registry,
    ) -> Result<LuaColumn, String> {
        if parameters.is_empty() {
            return Err("a lua column function takes at least one column argument".to_owned());
        }
        let mut names: Vec<CString> = Vec::with_capacity(parameters.len());
        for &parameter in parameters {
            if !is_identifier(parameter) {
                return Err(format!(
                    "parameter '{parameter}' is not a usable Lua identifier"
                ));
            }
            if RESERVED.contains(&parameter) || parameter == "out" {
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
        install_vocabulary(&mut state, ops)?;
        if let Some(sink) = log_sink {
            state.set_log_sink(Box::new(SharedSink(sink)));
        }
        let compiled = state.compile(chunk)?;
        Ok(LuaColumn {
            state: Mutex::new((state, compiled)),
            parameters: names,
        })
    }
}

impl query_lite::ColumnFunction for LuaColumn {
    fn arity(&self) -> usize {
        self.parameters.len()
    }

    fn evaluate(&self, args: &[(&[f64], &[bool])]) -> Result<Vec<Option<f64>>, String> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| "lua column interpreter poisoned".to_owned())?;
        let (state, chunk) = &mut *guard;
        let rows = args.first().map_or(0, |(values, _)| values.len());
        // NULL crosses as the sentinel (F1), carried by a validity
        // bitmap built only where nulls actually exist.
        let bitmaps: Vec<Option<Bitmap>> = args
            .iter()
            .map(|(_, validity)| {
                if validity.iter().all(|&valid| valid) {
                    None
                } else {
                    Some(Bitmap::from_bools(validity.iter().copied()))
                }
            })
            .collect();
        let inputs: Vec<(&CStr, ColumnView<'_>)> = self
            .parameters
            .iter()
            .zip(args.iter().zip(&bitmaps))
            .map(|(name, (&(values, _), bitmap))| {
                (
                    name.as_c_str(),
                    ColumnView::F64 {
                        values,
                        validity: bitmap.as_ref(),
                    },
                )
            })
            .collect();
        let mut values = vec![0.0f64; rows];
        let mut validity = Bitmap::from_bools(std::iter::repeat_n(false, rows));
        state.eval_column(
            chunk,
            &inputs,
            OutputColumn::F64 {
                values: &mut values,
                validity: &mut validity,
            },
        )?;
        Ok((0..rows)
            .map(|row| validity.get(row).then(|| values[row]))
            .collect())
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
    fn log_routes_to_the_installed_table_sink() {
        use std::sync::{Arc, Mutex};
        struct Capture(Arc<Mutex<Vec<String>>>);
        impl compute_lua::LogSink for Capture {
            fn log(&self, message: &str) {
                self.0.lock().unwrap().push(message.to_owned());
            }
        }
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        for i in 0..3 {
            table
                .append(&[RowValue::I64(i), RowValue::Key("A"), RowValue::F64(1.0)])
                .unwrap();
        }
        let messages = Arc::new(Mutex::new(Vec::new()));
        table.set_lua_log_sink(Arc::new(Capture(Arc::clone(&messages))));
        table
            .register_lua_window(
                "noisy",
                &["x"],
                "log('rows', #x)\nreturn #x",
                ColumnType::I64,
            )
            .unwrap();
        table
            .query(
                "SELECT noisy(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
                 AS n FROM t",
            )
            .unwrap();
        let messages = messages.lock().unwrap();
        assert_eq!(messages.len(), 3, "one log per window");
        assert_eq!(messages[0], "rows\t1");
        assert_eq!(messages[1], "rows\t2");
    }

    #[test]
    fn a_lua_scalar_kernel_runs_whole_columns_per_call() {
        // The vectorized shape (#53): one interpreter entry per view,
        // NULL in as the sentinel, NULL out by never writing the slot.
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false),
            Field::new("x", ColumnType::F64, true),
        ]);
        let mut table = Table::with_segment_rows("t", schema, "ts", 3).unwrap();
        for i in 0..7i64 {
            let x = if i == 2 {
                RowValue::Null
            } else {
                RowValue::F64(i as f64)
            };
            table.append(&[RowValue::I64(i), x]).unwrap();
        }
        table
            .register_lua_scalar(
                "double_or_skip",
                &["x"],
                "for i = 1, #x do\n\
                 if x[i] ~= NULL then out[i] = 2 * x[i] end\n\
                 end",
            )
            .unwrap();
        let output = table.query("SELECT double_or_skip(x) AS d FROM t").unwrap();
        let results: Vec<Option<f64>> = output
            .batches
            .iter()
            .flat_map(|batch| {
                let Column::Numeric(NumericData::F64(column)) = &batch.columns()[0] else {
                    panic!("expected f64")
                };
                (0..column.len())
                    .map(|row| {
                        column
                            .is_valid(row)
                            .then(|| column.values().as_slice()[row])
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            results,
            vec![
                Some(0.0),
                Some(2.0),
                None, // NULL in, slot never written, NULL out
                Some(6.0),
                Some(8.0),
                Some(10.0),
                Some(12.0),
            ]
        );
    }

    #[test]
    fn anything_sql_can_call_lua_can_call() {
        // The vocabulary invariant, registry-driven: a native
        // registered through the public trait path is immediately
        // callable from a Lua kernel by its SQL name — including
        // natives that did not exist when the vocabulary was designed.
        struct SumSq;
        impl query_lite::WindowAggregate for SumSq {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok(Some(args[0].iter().map(|v| v * v).sum()))
            }
        }
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        for i in 0..6i64 {
            table
                .append(&[RowValue::I64(i), RowValue::Key("A"), RowValue::F64(2.0)])
                .unwrap();
        }
        table.register_window("sumsq", SumSq).unwrap();
        table
            .register_lua_window(
                "twice_sumsq",
                &["x"],
                "return 2 * sumsq(x)",
                ColumnType::F64,
            )
            .unwrap();
        let output = table
            .query(
                "SELECT twice_sumsq(x) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING \
                 AND CURRENT ROW) AS s FROM t",
            )
            .unwrap();
        // Full two-row frames of 2.0: 2 * (4 + 4) = 16.
        assert_eq!(f64s(&output, 0)[1..], [Some(16.0); 5]);
        // And every built-in statistic is reachable from a kernel by
        // its SQL name — the registry is the single source of names.
        for name in [
            "regr_slope",
            "regr_intercept",
            "covar_pop",
            "corr",
            "eigen_max",
        ] {
            table
                .register_lua_window(
                    &format!("via_{name}"),
                    &["y", "x"],
                    &format!("return {name}(y, x)"),
                    ColumnType::F64,
                )
                .unwrap();
        }
    }

    #[test]
    fn promotion_swaps_a_lua_kernel_for_a_native_under_its_name() {
        // The promotion path made mechanical: a kernel prototyped in
        // Lua keeps its SQL name when a native lands under it — the
        // query text never changes, and matching fold order gives
        // matching results.
        let mut table = Table::with_segment_rows("t", schema(), "ts", 4).unwrap();
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        for (i, &x) in data.iter().enumerate() {
            table
                .append(&[
                    RowValue::I64(i as i64),
                    RowValue::Key("A"),
                    RowValue::F64(x),
                ])
                .unwrap();
        }
        let sql = "SELECT mean2(x) OVER (ORDER BY ts ROWS BETWEEN 2 PRECEDING \
                   AND CURRENT ROW) AS m FROM t";
        table
            .register_lua_window(
                "mean2",
                &["x"],
                "local s = 0.0\nfor i = 1, #x do s = s + x[i] end\nreturn s / #x",
                ColumnType::F64,
            )
            .unwrap();
        let prototype = f64s(&table.query(sql).unwrap(), 0);
        struct Mean;
        impl query_lite::WindowAggregate for Mean {
            fn arity(&self) -> usize {
                1
            }
            fn evaluate(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
                let mut sum = 0.0;
                for &value in args[0] {
                    sum += value;
                }
                Ok(Some(sum / args[0].len() as f64))
            }
        }
        table.register_window("mean2", Mean).unwrap();
        let promoted = f64s(&table.query(sql).unwrap(), 0);
        // Same fold order on both sides: bit-identical, not approximate.
        assert_eq!(prototype, promoted);
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
                // Same finalization semantics: defined-ness must agree
                // exactly. The values agree to the noise floor but not
                // bitwise — the SQL window runs the incremental sweep
                // (evaluate_frames) while the host op computes its one
                // window directly; both are held to the compensated
                // reference within 1e-12 by window_numerics_guard, so
                // their mutual agreement is bounded by twice that.
                match (native[row], scripted[row]) {
                    (None, None) => {}
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() <= 2e-12 * a.abs().max(b.abs()).max(1.0),
                        "{native_name} row {row}: {a} vs {b}"
                    ),
                    (a, b) => panic!("{native_name} row {row}: {a:?} vs {b:?}"),
                }
            }
        }
    }

    #[test]
    fn dot_from_lua_matches_the_linalg_backend() {
        use compute_linalg::{LinalgBackend, RustLinalg};
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
        // The same dot on the same window slices: identical bits.
        let backend = RustLinalg;
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
