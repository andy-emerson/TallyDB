//! The safe wrapper over the thin bindings: an embedded interpreter
//! whose scripts see engine columns through the value map (`values`).
//! The module discipline is fixed and does not change as the crate
//! grows:
//!
//! 1. Every entry into Lua goes through `lua_pcall` — nothing runs
//!    unprotected.
//! 2. A Rust function called *from* Lua never raises a Lua error
//!    across a frame with pending destructors: view accessors keep
//!    only `Copy` locals, and `lua_error` is the tail call.
//! 3. The boundary never propagates a Rust panic into C.
//!
//! ## The views (the crate's reason to exist)
//!
//! A view userdata holds pointers into live engine buffers — a handful
//! of bytes of handle; **zero bytes of data move**. Scripts index views
//! `v[i]` (1-based, like Lua) and read `f64` elements as Lua floats,
//! `i64` elements as Lua 5.4 integers — exactly, the alignment that
//! decided the interpreter — and key elements as integer dictionary
//! codes. A NULL element reads as the `NULL` sentinel, never `nil` (the
//! F1 decision; see `values`). Out-of-range access is a loud error
//! (this engine refuses wrong answers; a silent `nil` would turn into
//! one), and input views are read-only. A view is valid for the
//! duration of one protected call: every call bumps the state's
//! generation counter on exit, so a handle smuggled out — stashed in a
//! global or a closure — errors later, never a dangling read.
//!
//! Interpreter cost, Observed (run 2026-07-27, release, container
//! hardware, `measure_41_interpreter_kernel_cost`): a 4,096-row mean-
//! absolute-deviation kernel — two full passes over the view — takes
//! ~890µs per window (~1,100 windows/s, ~9M element reads/s through
//! the metamethod accessor, generation and validity checks included).
//! The same run's `measure_vectorized_udf_vs_per_row` puts per-row
//! invocation at 16× the one-call-per-column convention. Absolute
//! numbers are hardware-bound and not comparable to earlier runs on
//! other machines; the ratios are the durable part. That cost is the
//! price of the ad-hoc layer; per the promotion ladder (DESIGN.md,
//! *The Lua layer*), a kernel that proves hot graduates to a curated
//! native op rather than the interpreter getting a JIT.

use crate::ffi;
use crate::host::{self, HostFunction, HostSlot};
use crate::log::{self, LogSink, SinkSlot};
use crate::values::{self, ColumnView, OutputColumn, ReturnType, ScalarValue};
use std::ffi::{CStr, CString};

/// A compiled kernel, held in its interpreter's registry — the unit
/// scripts are *run* as. Compiling is the expensive half of a call
/// (parse, code-generate, allocate a prototype); a kernel that runs
/// once per window must therefore be compiled once at registration and
/// called thereafter, which is why the run methods take one of these
/// rather than source text. Compiling is also where a syntax error
/// surfaces, so registration fails loudly instead of the first query.
#[derive(Debug)]
pub struct Chunk {
    /// Registry key holding the compiled function.
    key: CString,
    /// The interpreter this belongs to; a chunk is not portable between
    /// states (their registries are separate).
    state_id: u64,
}

/// Source of the per-state identity stamped into [`Chunk`].
static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// An embedded Lua 5.4 interpreter with the curated library set
/// (base, math, string, table — no io, no os, no debug; the package
/// library is not even linked, per the ANSI build), the view
/// metatables, and the `NULL` sentinel installed. Base opens minus
/// `print` and `warn` (process streams are not this library's to own);
/// `log(...)` is the diagnostic, routed to the embedder's [`LogSink`]
/// via [`LuaState::set_log_sink`].
pub struct LuaState {
    raw: *mut ffi::lua_State,
    /// The registry-anchored generation counter view lifetimes are
    /// checked against; bumped on every `eval_*` exit.
    generation: *mut u64,
    /// The `log()` sink slot — boxed so the `log` closure's upvalue
    /// pointer survives moves of this struct; freed in `Drop`.
    sink: *mut SinkSlot,
    /// Registered host functions, one stable box each (the closures'
    /// upvalues point at them); freed in `Drop`.
    host_functions: Vec<*mut HostSlot>,
    /// This interpreter's identity, stamped into every [`Chunk`] it
    /// compiles so a chunk cannot be run against another state.
    id: u64,
    /// Serial for registry keys of compiled chunks.
    next_chunk: u64,
}

impl LuaState {
    /// Creates a state with the curated libraries, the view
    /// metatables, and the `NULL` sentinel installed.
    pub fn new() -> Result<LuaState, String> {
        unsafe {
            let raw = ffi::luaL_newstate();
            if raw.is_null() {
                return Err("lua state allocation failed".to_owned());
            }
            ffi::luaL_requiref(raw, c"_G".as_ptr(), ffi::luaopen_base, 1);
            ffi::luaL_requiref(raw, c"math".as_ptr(), ffi::luaopen_math, 1);
            ffi::luaL_requiref(raw, c"string".as_ptr(), ffi::luaopen_string, 1);
            ffi::luaL_requiref(raw, c"table".as_ptr(), ffi::luaopen_table, 1);
            ffi::lua_settop(raw, 0);
            let generation = values::install(raw);
            crate::vector::install(raw);
            let sink = Box::into_raw(Box::new(SinkSlot(None)));
            log::install(raw, sink);
            Ok(LuaState {
                raw,
                generation,
                sink,
                host_functions: Vec::new(),
                id: NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                next_chunk: 0,
            })
        }
    }

    /// Compiles `chunk` (text only) and keeps the compiled function in
    /// this interpreter's registry, returning the handle the run methods
    /// take. Compile once — at registration — and call the result per
    /// window; recompiling per call is the dominant cost of a script
    /// kernel, which is why source text is not a runnable input.
    pub fn compile(&mut self, chunk: &str) -> Result<Chunk, String> {
        unsafe {
            debug_assert_eq!(ffi::lua_gettop(self.raw), 0);
            let status = ffi::luaL_loadbufferx(
                self.raw,
                chunk.as_ptr().cast(),
                chunk.len(),
                c"script".as_ptr(),
                c"t".as_ptr(), // text only: no binary chunks, ever
            );
            if status != ffi::LUA_OK {
                let message = self.pop_error("load");
                debug_assert_eq!(ffi::lua_gettop(self.raw), 0);
                return Err(message);
            }
            // Park the compiled function in the registry under a key
            // owned by the handle (built once, so calls allocate none).
            let key = CString::new(format!("tallydb.chunk.{}", self.next_chunk))
                .expect("generated key has no interior NUL");
            self.next_chunk += 1;
            ffi::lua_setfield(self.raw, ffi::LUA_REGISTRYINDEX, key.as_ptr());
            debug_assert_eq!(ffi::lua_gettop(self.raw), 0);
            Ok(Chunk {
                key,
                state_id: self.id,
            })
        }
    }

    /// Registers an engine-side operation as the global `name`,
    /// callable from scripts over zero-copy views — the seam the
    /// curated `compute-linalg` and engine ops are exposed through. See
    /// [`HostFunction`]. A second registration under the
    /// same name replaces the first (the old function's storage is
    /// retained until the state drops).
    pub fn register_host_function(
        &mut self,
        name: &str,
        function: Box<dyn HostFunction>,
    ) -> Result<(), String> {
        if function.arity() == 0 || function.arity() > host::MAX_ARGS {
            return Err(format!(
                "host function '{name}' takes {} arguments; supported range is 1..={}",
                function.arity(),
                host::MAX_ARGS
            ));
        }
        let global = CString::new(name)
            .map_err(|_| format!("function name '{name}' contains an interior NUL"))?;
        let slot = Box::into_raw(Box::new(HostSlot(function)));
        unsafe { host::install(self.raw, &global, slot) };
        self.host_functions.push(slot);
        Ok(())
    }

    /// Installs the destination for scripts' `log(...)` output —
    /// off (a no-op) until the embedder installs one. See [`LogSink`].
    pub fn set_log_sink(&mut self, sink: Box<dyn LogSink>) {
        unsafe {
            (*self.sink).0 = Some(sink);
        }
    }

    /// Runs `chunk` (text only) with `inputs` bound to global names and
    /// returns its single result coerced to `declared` — the window /
    /// reduction shape: whole columns in, one typed scalar out. A
    /// script that returns `NULL` (or nothing) yields
    /// [`ScalarValue::Null`]. Views are valid only inside this call.
    /// Every failure — load error, runtime error, a result the declared
    /// type cannot hold exactly — is a loud `Err`.
    pub fn eval_scalar(
        &mut self,
        chunk: &Chunk,
        inputs: &[(&CStr, ColumnView<'_>)],
        declared: ReturnType,
    ) -> Result<ScalarValue, String> {
        unsafe {
            debug_assert_eq!(ffi::lua_gettop(self.raw), 0);
            let result = self
                .bind_inputs(inputs)
                .and_then(|()| self.run(chunk, 1))
                .and_then(|()| values::read_result(self.raw, declared));
            self.end_call();
            result
        }
    }

    /// Runs `chunk` (text only) with `inputs` bound to global names and
    /// `output` bound as the global `out` — the vectorized-UDF shape:
    /// whole columns in, one whole column out, one boundary crossing
    /// per call. The script writes `out[i]`; writes coerce exact-or-
    /// loud to the output's type, and slots never written come back
    /// NULL. Views are valid only inside this call.
    pub fn eval_column(
        &mut self,
        chunk: &Chunk,
        inputs: &[(&CStr, ColumnView<'_>)],
        mut output: OutputColumn<'_>,
    ) -> Result<(), String> {
        unsafe {
            debug_assert_eq!(ffi::lua_gettop(self.raw), 0);
            let result = self
                .bind_inputs(inputs)
                .and_then(|()| {
                    values::bind_output(self.raw, self.generation, &mut output);
                    self.run(chunk, 1)
                })
                .and_then(|()| crate::vector::read_column_result(self.raw));
            let result = result.and_then(|column| write_column_result(column, &mut output));
            self.end_call();
            result
        }
    }

    unsafe fn bind_inputs(&mut self, inputs: &[(&CStr, ColumnView<'_>)]) -> Result<(), String> {
        for (name, view) in inputs {
            unsafe { values::bind_input(self.raw, self.generation, name, view)? };
        }
        Ok(())
    }

    /// Bumps the generation — poisoning every view bound or created
    /// during the call, on success and error alike — and clears the
    /// stack.
    unsafe fn end_call(&mut self) {
        unsafe {
            *self.generation = (*self.generation).wrapping_add(1);
            ffi::lua_settop(self.raw, 0);
        }
    }

    /// Pushes the compiled function from the registry and calls it — no
    /// parse, no code generation, no allocation on the call path.
    unsafe fn run(&mut self, chunk: &Chunk, results: std::ffi::c_int) -> Result<(), String> {
        unsafe {
            if chunk.state_id != self.id {
                return Err("chunk belongs to a different interpreter".to_owned());
            }
            ffi::lua_getfield(self.raw, ffi::LUA_REGISTRYINDEX, chunk.key.as_ptr());
            if ffi::lua_pcall(self.raw, 0, results, 0) != ffi::LUA_OK {
                return Err(self.pop_error("run"));
            }
            Ok(())
        }
    }

    unsafe fn pop_error(&mut self, stage: &str) -> String {
        unsafe {
            let mut len = 0usize;
            let text = ffi::lua_tolstring(self.raw, -1, &mut len);
            let message = if text.is_null() {
                format!("{stage}: error object is not a string")
            } else {
                let bytes = std::slice::from_raw_parts(text.cast(), len);
                format!("{stage}: {}", String::from_utf8_lossy(bytes))
            };
            ffi::lua_settop(self.raw, -2);
            message
        }
    }

    /// The data pointer a view userdata carries — the zero-copy proof
    /// hook, compared against the source buffer's pointer in tests
    /// exactly like the engine's passthrough pointer checks.
    #[doc(hidden)]
    pub fn view_data_pointer(&mut self, view_global: &CStr) -> Option<*const u8> {
        unsafe { values::view_data_pointer(self.raw, view_global) }
    }
}

/// Applies a `return <column>` result (the composed-kernel shape) to
/// the output: the returned column replaces the output wholesale, with
/// the F3 exact-or-loud coercion into a declared `i64` output. With no
/// returned value, the script's `out[i]` writes stand.
fn write_column_result(
    column: crate::vector::ColumnResult,
    output: &mut OutputColumn<'_>,
) -> Result<(), String> {
    let crate::vector::ColumnResult::Elements(elements) = column else {
        return Ok(());
    };
    match output {
        OutputColumn::F64 { values, validity } => {
            if elements.len() != values.len() {
                return Err(format!(
                    "result: returned column has {} elements for {} output rows",
                    elements.len(),
                    values.len()
                ));
            }
            for (offset, element) in elements.into_iter().enumerate() {
                match element {
                    Some(value) => {
                        values[offset] = value;
                        validity.set(offset, true);
                    }
                    None => validity.set(offset, false),
                }
            }
            Ok(())
        }
        OutputColumn::I64 { values, validity } => {
            if elements.len() != values.len() {
                return Err(format!(
                    "result: returned column has {} elements for {} output rows",
                    elements.len(),
                    values.len()
                ));
            }
            for (offset, element) in elements.into_iter().enumerate() {
                match element {
                    Some(value) => match crate::values::float_as_i64_exact(value) {
                        Some(integer) => {
                            values[offset] = integer;
                            validity.set(offset, true);
                        }
                        None => {
                            return Err("result: float element does not fit i64 exactly".to_owned())
                        }
                    },
                    None => validity.set(offset, false),
                }
            }
            Ok(())
        }
        OutputColumn::Key { .. } => {
            Err("result: a returned column cannot fill a key output".to_owned())
        }
    }
}

// SAFETY (the Send argument, written once and load-bearing):
//
// 1. `LuaState` uniquely owns its interpreter: the raw `lua_State` is
//    created in `new`, closed in `Drop`, and the pointer is never
//    copied out of the struct (`view_data_pointer` returns buffer
//    pointers, not the state). `generation` points into that same
//    interpreter's registry-anchored allocation; `sink` and each
//    `host_functions` entry point into Boxes this struct alone owns
//    (their `LogSink` / `HostFunction` contents are themselves `Send`
//    by the traits' bounds), so all of it moves with it.
// 2. Every operation takes `&mut self`, so after a move to another
//    thread exactly one thread touches the interpreter at a time.
// 3. Vendored PUC Lua 5.4, compiled unmodified with the ANSI config,
//    keeps all interpreter state inside the `lua_State`/`global_State`
//    allocation: no thread-locals, no mutable process globals in the
//    linked subset (base/math/string/table; no io, no os), and the
//    default allocator is C `realloc`/`free`, which is thread-safe.
// 4. No borrow outlives a call: views over engine buffers are
//    generation-poisoned when their `eval_*` returns, so a moved state
//    carries no live aliases into another thread.
//
// `Sync` is NOT implied and not implemented: two threads sharing
// `&LuaState` would race the interpreter. A holder that needs `Sync`
// wraps the state in a `Mutex` (the engine's Lua-backed window does).
unsafe impl Send for LuaState {}

impl Drop for LuaState {
    fn drop(&mut self) {
        unsafe {
            ffi::lua_close(self.raw);
            // After close nothing can reach the closures' upvalues; the
            // slots are safe to free.
            drop(Box::from_raw(self.sink));
            for slot in self.host_functions.drain(..) {
                drop(Box::from_raw(slot));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The value-map contract, unit-proven: round-trip fidelity across
    //! all three column types, three-valued logic through the sentinel,
    //! loud coercion at every boundary the contract closes, and the
    //! zero-copy and lifetime properties the crate exists for.

    use super::*;
    use arrow_lite::{Bitmap, Dictionary};

    /// Compile-and-run, for tests that exercise one chunk once. The
    /// production path compiles at registration and calls per window.
    fn eval(
        state: &mut LuaState,
        source: &str,
        inputs: &[(&CStr, ColumnView<'_>)],
        declared: ReturnType,
    ) -> Result<ScalarValue, String> {
        let chunk = state.compile(source)?;
        state.eval_scalar(&chunk, inputs, declared)
    }

    /// As [`eval`], for the column-output shape.
    fn eval_col(
        state: &mut LuaState,
        source: &str,
        inputs: &[(&CStr, ColumnView<'_>)],
        output: OutputColumn<'_>,
    ) -> Result<(), String> {
        let chunk = state.compile(source)?;
        state.eval_column(&chunk, inputs, output)
    }

    fn f64s(values: &[f64]) -> ColumnView<'_> {
        ColumnView::F64 {
            values,
            validity: None,
        }
    }

    // ---- zero-copy and exactness (the #41 spike, kept green) ----

    #[test]
    fn f64_view_is_zero_copy_and_reads_exactly() {
        let values: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.25 - 100.0).collect();
        let mut state = LuaState::new().unwrap();
        let sum = eval(
            &mut state,
            "local s = 0.0\nfor i = 1, #v do s = s + v[i] end\nreturn s",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap();
        // Same order, same arithmetic: bit-exact agreement, not approximate.
        let expected: f64 = values.iter().sum();
        assert_eq!(sum, ScalarValue::F64(expected));
        // The zero-copy proof: the userdata carried the buffer's own
        // pointer, not a copy's.
        let pointer = state.view_data_pointer(c"v").expect("view global");
        assert_eq!(pointer, values.as_ptr().cast());
    }

    #[test]
    fn i64_view_crosses_exactly_beyond_2_pow_53() {
        // 2^53 + 1 cannot survive an f64 hop; only an exact integer
        // path returns difference 1.
        let values: Vec<i64> = vec![9_007_199_254_740_993, -9_007_199_254_740_993];
        let mut state = LuaState::new().unwrap();
        let difference = eval(
            &mut state,
            "return v[1] - 9007199254740992",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &values,
                    validity: None,
                },
            )],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(difference, ScalarValue::I64(1));
        let pointer = state.view_data_pointer(c"v").expect("view global");
        assert_eq!(pointer, values.as_ptr().cast());
    }

    #[test]
    fn output_writes_land_in_the_callers_buffer() {
        let values = [1.0f64, 2.0, 3.0];
        let mut out = [0.0f64; 3];
        let mut validity = Bitmap::new_unset(3);
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "for i = 1, #v do out[i] = v[i] * 2 end",
            &[(c"v", f64s(&values))],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap();
        // The zero-copy proof for the output side: the view carried the
        // caller's buffer, and the writes are already in it.
        let pointer = state.view_data_pointer(c"out").expect("out global");
        assert_eq!(pointer, out.as_ptr().cast());
        assert_eq!(out, [2.0, 4.0, 6.0]);
        assert_eq!(validity.count_set(), 3);
    }

    // ---- F1: the NULL sentinel and three-valued logic ----

    #[test]
    fn null_reads_as_sentinel_and_propagates_through_arithmetic() {
        let values = [1.0, 2.0, 3.0];
        let validity = Bitmap::from_bools([true, false, true]); // middle is NULL
        let view = || ColumnView::F64 {
            values: &values,
            validity: Some(&validity),
        };
        let mut state = LuaState::new().unwrap();

        // The naive sum does not crash and does not skip: NULL poisons
        // the whole result (soft 3VL, the SQL/pd.NA behavior).
        let naive = eval(
            &mut state,
            "local s = 0.0\nfor i = 1, #v do s = s + v[i] end\nreturn s",
            &[(c"v", view())],
            ReturnType::F64,
        )
        .unwrap();
        assert_eq!(naive, ScalarValue::Null);

        // The guard idiom: identity comparison against the sentinel.
        let guarded = eval(
            &mut state,
            "local s = 0.0\n\
                 for i = 1, #v do if v[i] ~= NULL then s = s + v[i] end end\n\
                 return s",
            &[(c"v", view())],
            ReturnType::F64,
        )
        .unwrap();
        assert_eq!(guarded, ScalarValue::F64(4.0)); // 1 + 3, null skipped
    }

    #[test]
    fn nan_and_null_stay_distinct() {
        // NaN is a computed value (self-unequal, in the buffer); NULL is
        // absence (the sentinel). They must not collapse.
        let values = [f64::NAN, 5.0];
        let validity = Bitmap::from_bools([true, false]);
        let mut state = LuaState::new().unwrap();
        let code = eval(
            &mut state,
            "local nulls, nans = 0, 0\n\
                 for i = 1, #v do\n\
                   if v[i] == NULL then nulls = nulls + 1\n\
                   elseif v[i] ~= v[i] then nans = nans + 1 end\n\
                 end\n\
                 return nulls * 10 + nans",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(code, ScalarValue::I64(11), "one NULL and one NaN");
    }

    #[test]
    fn sentinel_propagates_over_the_integer_subtype() {
        // The i64-exactness constraint holds around NULL too: an
        // integer beyond 2^53 combined with NULL yields NULL — no crash,
        // no float coercion.
        let mut state = LuaState::new().unwrap();
        let result = eval(
            &mut state,
            "return 9007199254740993 + NULL",
            &[],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::Null);
    }

    #[test]
    fn sentinel_is_truthy_the_documented_wart() {
        // Only nil and false are falsy in Lua, so any non-nil sentinel
        // is truthy: `if v[i]` is true for a NULL. Documented, not
        // fixable — the guard idiom is `~= NULL`, not truthiness.
        let values = [0.0];
        let validity = Bitmap::from_bools([false]);
        let mut state = LuaState::new().unwrap();
        let result = eval(
            &mut state,
            "if v[1] then return 1 else return 0 end",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::I64(1));
    }

    #[test]
    fn comparison_with_null_is_loud() {
        // Lua forces `<`/`<=` to a boolean, so 3VL cannot propagate
        // through them; the honest behavior is a loud error.
        let values = [0.0];
        let validity = Bitmap::from_bools([false]);
        let mut state = LuaState::new().unwrap();
        let error = eval(
            &mut state,
            "return v[1] < 5",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::I64,
        )
        .unwrap_err();
        assert!(
            error.contains("compare") || error.contains("attempt"),
            "{error}"
        );
    }

    // ---- F2: the declared type decides, never the returned value ----

    #[test]
    fn declared_type_is_enforced_not_inferred() {
        let mut state = LuaState::new().unwrap();
        // The same chunk lands as either type when exact...
        assert_eq!(
            eval(&mut state, "return 3", &[], ReturnType::F64).unwrap(),
            ScalarValue::F64(3.0)
        );
        assert_eq!(
            eval(&mut state, "return 3", &[], ReturnType::I64).unwrap(),
            ScalarValue::I64(3)
        );
        // ...and is refused loudly when it cannot be exact.
        let error = eval(&mut state, "return 2.5", &[], ReturnType::I64).unwrap_err();
        assert!(error.contains("does not fit i64 exactly"), "{error}");
        let error = eval(&mut state, "return 9007199254740993", &[], ReturnType::F64).unwrap_err();
        assert!(error.contains("does not fit f64 exactly"), "{error}");
        // A lossless float fills i64 (2.0 is integral), per F3.
        assert_eq!(
            eval(&mut state, "return 4.0 / 2.0", &[], ReturnType::I64).unwrap(),
            ScalarValue::I64(2)
        );
    }

    #[test]
    fn scalar_key_results_are_refused() {
        let mut state = LuaState::new().unwrap();
        let error = eval(&mut state, "return 'AAPL'", &[], ReturnType::Key).unwrap_err();
        assert!(error.contains("output column"), "{error}");
    }

    // ---- F3: exact-or-loud coercion at the output boundary ----

    #[test]
    fn output_coercion_is_exact_or_loud() {
        let mut state = LuaState::new().unwrap();
        let mut i64_out = [0i64; 1];
        let mut validity = Bitmap::new_unset(1);

        // Lossless float → i64 fills; boolean maps to {0, 1}.
        eval_col(
            &mut state,
            "out[1] = 2.0",
            &[],
            OutputColumn::I64 {
                values: &mut i64_out,
                validity: &mut validity,
            },
        )
        .unwrap();
        assert_eq!((i64_out[0], validity.get(0)), (2, true));
        eval_col(
            &mut state,
            "out[1] = true",
            &[],
            OutputColumn::I64 {
                values: &mut i64_out,
                validity: &mut validity,
            },
        )
        .unwrap();
        assert_eq!((i64_out[0], validity.get(0)), (1, true));

        // Non-integral float → i64 is a loud error, never truncation.
        let error = eval_col(
            &mut state,
            "out[1] = 2.5",
            &[],
            OutputColumn::I64 {
                values: &mut i64_out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("does not fit i64 exactly"), "{error}");

        let mut f64_out = [0.0f64; 1];
        // Boolean → f64 has no defined mapping: loud.
        let error = eval_col(
            &mut state,
            "out[1] = true",
            &[],
            OutputColumn::F64 {
                values: &mut f64_out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("boolean maps to i64"), "{error}");
        // Strings produce keys, nothing else: loud into f64...
        let error = eval_col(
            &mut state,
            "out[1] = 'x'",
            &[],
            OutputColumn::F64 {
                values: &mut f64_out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("produces a key"), "{error}");
        // ...and an integer beyond 2^53 refuses to round into f64.
        let error = eval_col(
            &mut state,
            "out[1] = 9007199254740993",
            &[],
            OutputColumn::F64 {
                values: &mut f64_out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("does not fit f64 exactly"), "{error}");

        // Numbers never become keys — codes are per-segment, so writing
        // one through would be meaningless at best.
        let mut codes = [0u32; 1];
        let mut dictionary = Dictionary::new();
        let error = eval_col(
            &mut state,
            "out[1] = 1",
            &[],
            OutputColumn::Key {
                codes: &mut codes,
                validity: &mut validity,
                dictionary: &mut dictionary,
            },
        )
        .unwrap_err();
        assert!(error.contains("only a string produces a key"), "{error}");
    }

    #[test]
    fn output_nulls_unwritten_slots_and_readback() {
        let mut out = [0.0f64; 4];
        let mut validity = Bitmap::new_set(4); // stale bits: must be reset
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            // Slot 1 written; slot 2 NULLed explicitly; slot 3 via
            // nil; slot 4 never touched. Readback: out[2] reads as
            // NULL mid-script, so the guarded rewrite fires.
            "out[1] = 1.5\n\
                 out[2] = NULL\n\
                 out[3] = nil\n\
                 if out[2] == NULL then out[1] = out[1] + 40.5 end",
            &[],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap();
        assert_eq!(out[0], 42.0);
        assert_eq!(
            (0..4).map(|i| validity.get(i)).collect::<Vec<_>>(),
            [true, false, false, false],
            "explicit NULL, nil, and unwritten slots all come back NULL"
        );
    }

    // ---- F3/F4: keys — codes in, strings out, text on demand ----

    #[test]
    fn key_elements_read_as_codes_with_lazy_text() {
        let codes = [0u32, 1, 0];
        let mut dictionary = Dictionary::new();
        dictionary.intern("AAPL");
        dictionary.intern("MSFT");
        let view = || ColumnView::Key {
            codes: &codes,
            validity: None,
            dictionary: &dictionary,
        };
        let mut state = LuaState::new().unwrap();
        // The element read is the integer code (F4) — integer-cheap.
        assert_eq!(
            eval(
                &mut state,
                "return v[2]",
                &[(c"v", view())],
                ReturnType::I64
            )
            .unwrap(),
            ScalarValue::I64(1)
        );
        // text(i) decodes on demand.
        assert_eq!(
            eval(
                &mut state,
                "if v:text(2) == 'MSFT' then return 1 else return 0 end",
                &[(c"v", view())],
                ReturnType::I64
            )
            .unwrap(),
            ScalarValue::I64(1)
        );
        // code_of resolves a literal once; an absent literal is nil
        // (absence, not SQL NULL), which the typed result maps to Null.
        assert_eq!(
            eval(
                &mut state,
                "return v:code_of('MSFT')",
                &[(c"v", view())],
                ReturnType::I64
            )
            .unwrap(),
            ScalarValue::I64(1)
        );
        assert_eq!(
            eval(
                &mut state,
                "return v:code_of('TSLA')",
                &[(c"v", view())],
                ReturnType::I64
            )
            .unwrap(),
            ScalarValue::Null
        );
    }

    #[test]
    fn key_column_round_trips_through_text() {
        // The identity kernel for keys: decode with text, re-intern on
        // write. Codes may renumber (the output dictionary is its own
        // code space); the text and the nulls must survive exactly.
        let codes = [0u32, 1, 0, 0];
        let validity = Bitmap::from_bools([true, false, true, true]);
        let mut dictionary = Dictionary::new();
        dictionary.intern("AAPL");
        dictionary.intern("MSFT");

        let mut out_codes = [99u32; 4];
        let mut out_validity = Bitmap::new_unset(4);
        let mut out_dictionary = Dictionary::new();
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "for i = 1, #v do out[i] = v:text(i) end",
            &[(
                c"v",
                ColumnView::Key {
                    codes: &codes,
                    validity: Some(&validity),
                    dictionary: &dictionary,
                },
            )],
            OutputColumn::Key {
                codes: &mut out_codes,
                validity: &mut out_validity,
                dictionary: &mut out_dictionary,
            },
        )
        .unwrap();
        for i in 0..4 {
            assert_eq!(out_validity.get(i), validity.get(i), "slot {i}");
            if validity.get(i) {
                assert_eq!(
                    out_dictionary.value(out_codes[i]),
                    dictionary.value(codes[i]),
                    "slot {i}"
                );
            }
        }
        // Interning is idempotent: three AAPLs, one MSFT slot (null),
        // so the output dictionary holds exactly one distinct value.
        assert_eq!(out_dictionary.len(), 1);
    }

    // ---- the batch ergonomics: sum() and mask() ----

    #[test]
    fn sum_skips_nulls_and_matches_sql_semantics() {
        let mut state = LuaState::new().unwrap();

        let values = [10.0, 20.0, 30.0, 40.0];
        let validity = Bitmap::from_bools([true, false, true, false]);
        let sum = eval(
            &mut state,
            "return v:sum()",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::F64,
        )
        .unwrap();
        assert_eq!(sum, ScalarValue::F64(40.0)); // 10 + 30, nulls skipped

        // SUM over nothing is NULL, exactly as in SQL.
        let all_null = Bitmap::new_unset(4);
        let sum = eval(
            &mut state,
            "return v:sum()",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&all_null),
                },
            )],
            ReturnType::F64,
        )
        .unwrap();
        assert_eq!(sum, ScalarValue::Null);

        // i64 sums stay exact beyond 2^53 and overflow loudly, matching
        // the engine's SUM semantics (no silent widening).
        let big = [9_007_199_254_740_993i64, 2];
        let sum = eval(
            &mut state,
            "return v:sum()",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &big,
                    validity: None,
                },
            )],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(sum, ScalarValue::I64(9_007_199_254_740_995));
        let overflowing = [i64::MAX, 1];
        let error = eval(
            &mut state,
            "return v:sum()",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &overflowing,
                    validity: None,
                },
            )],
            ReturnType::I64,
        )
        .unwrap_err();
        assert!(error.contains("overflows"), "{error}");

        // Keys are not arithmetic: no sum, loudly.
        let codes = [0u32];
        let mut dictionary = Dictionary::new();
        dictionary.intern("AAPL");
        let error = eval(
            &mut state,
            "return v:sum()",
            &[(
                c"v",
                ColumnView::Key {
                    codes: &codes,
                    validity: None,
                    dictionary: &dictionary,
                },
            )],
            ReturnType::I64,
        )
        .unwrap_err();
        assert!(error.contains("not arithmetic"), "{error}");
    }

    #[test]
    fn mask_reads_validity_out_of_band() {
        let values = [1.0, 2.0, 3.0];
        let validity = Bitmap::from_bools([true, false, true]);
        let mut state = LuaState::new().unwrap();
        // Validity as its own boolean view: the value stream stays
        // purely numeric while the script counts nulls separately.
        let count = eval(
            &mut state,
            "local m = v:mask()\n\
                 local n = 0\n\
                 for i = 1, #m do if m[i] then n = n + 1 end end\n\
                 return n",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(count, ScalarValue::I64(2));
        // A column with no validity sidecar masks to all-true.
        let count = eval(
            &mut state,
            "local m = v:mask()\n\
                 local n = 0\n\
                 for i = 1, #m do if m[i] then n = n + 1 end end\n\
                 return n",
            &[(c"v", f64s(&values))],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(count, ScalarValue::I64(3));
    }

    // ---- round-trip fidelity through an identity kernel ----

    #[test]
    fn f64_identity_preserves_values_nan_and_null() {
        let values = [f64::NAN, 1.5, 0.0];
        let validity = Bitmap::from_bools([true, true, false]);
        let mut out = [0.0f64; 3];
        let mut out_validity = Bitmap::new_unset(3);
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "for i = 1, #v do out[i] = v[i] end",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut out_validity,
            },
        )
        .unwrap();
        assert!(out[0].is_nan(), "NaN is a value and survives");
        assert_eq!(out[1], 1.5);
        assert!(!out_validity.get(2), "NULL survives as NULL");
        assert_eq!(out_validity.count_set(), 2);
    }

    #[test]
    fn i64_identity_is_exact_beyond_2_pow_53() {
        let values = [9_007_199_254_740_993i64, i64::MIN, 7];
        let validity = Bitmap::from_bools([true, true, false]);
        let mut out = [0i64; 3];
        let mut out_validity = Bitmap::new_unset(3);
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "for i = 1, #v do out[i] = v[i] end",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            OutputColumn::I64 {
                values: &mut out,
                validity: &mut out_validity,
            },
        )
        .unwrap();
        assert_eq!(out[..2], values[..2], "bit-exact, no float hop");
        assert!(!out_validity.get(2));
    }

    // ---- lifetimes and misuse ----

    #[test]
    fn views_and_masks_poison_after_their_call() {
        let values = [1.0f64, 2.0];
        let validity = Bitmap::from_bools([true, true]);
        let mut state = LuaState::new().unwrap();
        eval(
            &mut state,
            "stash = function() return v[1] end\n\
                 stashed_mask = v:mask()\n\
                 return 0",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::I64,
        )
        .unwrap();
        // The borrows have ended; both the stashed closure and the
        // stashed mask must find poisoned views, never a dangling read.
        let error = eval(&mut state, "return stash()", &[], ReturnType::F64).unwrap_err();
        assert!(error.contains("outside its call"), "{error}");
        let error = eval(&mut state, "return stashed_mask[1]", &[], ReturnType::I64).unwrap_err();
        assert!(error.contains("outside its call"), "{error}");
    }

    #[test]
    fn a_detached_method_rechecks_the_view_type() {
        // Method closures are detachable (`local f = v.sum`), so the
        // dispatch-time type check alone is not enough: calling a
        // numeric method with a key view must be loud, never a read of
        // u32 codes as i64s.
        let codes = [0u32, 1];
        let mut dictionary = Dictionary::new();
        dictionary.intern("AAPL");
        dictionary.intern("MSFT");
        let values = [1.0f64, 2.0];
        let mut state = LuaState::new().unwrap();
        let inputs = [
            (c"v", f64s(&values)),
            (
                c"k",
                ColumnView::Key {
                    codes: &codes,
                    validity: None,
                    dictionary: &dictionary,
                },
            ),
        ];
        let error = eval(
            &mut state,
            "local f = v.sum\nreturn f(k)",
            &inputs,
            ReturnType::I64,
        )
        .unwrap_err();
        assert!(error.contains("not arithmetic"), "{error}");
        // The mirror image: a key method detached and fed a numeric view.
        let error = eval(
            &mut state,
            "local f = k.text\nreturn f(v, 1)",
            &inputs,
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("key-view methods"), "{error}");
    }

    #[test]
    fn view_misuse_fails_loudly() {
        let values = [1.0f64, 2.0];
        let mut state = LuaState::new().unwrap();
        // Out of range — never nil, always an error.
        for chunk in ["return v[3]", "return v[0]"] {
            let error =
                eval(&mut state, chunk, &[(c"v", f64s(&values))], ReturnType::F64).unwrap_err();
            assert!(error.contains("out of range"), "{error}");
        }
        // Unknown method.
        let error = eval(
            &mut state,
            "return v:median()",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("no such view method"), "{error}");
        // Key methods on a numeric view.
        let error = eval(
            &mut state,
            "return v:text(1)",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("key-view methods"), "{error}");
        // Input views are read-only.
        let error = eval(
            &mut state,
            "v[1] = 9\nreturn 0",
            &[(c"v", f64s(&values))],
            ReturnType::I64,
        )
        .unwrap_err();
        assert!(error.contains("read-only"), "{error}");
        // Output indices are integers.
        let mut out = [0.0f64; 2];
        let mut validity = Bitmap::new_unset(2);
        let error = eval_col(
            &mut state,
            "out['x'] = 1.0",
            &[],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("must be an integer"), "{error}");
    }

    #[test]
    fn validity_length_mismatch_is_loud() {
        let values = [1.0f64, 2.0, 3.0];
        let validity = Bitmap::from_bools([true, false]); // one bit short
        let mut state = LuaState::new().unwrap();
        let error = eval(
            &mut state,
            "return 0",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::I64,
        )
        .unwrap_err();
        assert!(error.contains("2 bits for 3 values"), "{error}");
    }

    // ---- log(): the host-routed diagnostic side-channel ----

    /// A capture sink for tests.
    struct Capture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    impl LogSink for Capture {
        fn log(&self, message: &str) {
            self.0.lock().unwrap().push(message.to_owned());
        }
    }

    #[test]
    fn log_routes_to_the_installed_sink_and_summarizes_views() {
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut state = LuaState::new().unwrap();
        state.set_log_sink(Box::new(Capture(messages.clone())));
        let values = [0.25f64, 0.5, 0.75];
        let result = eval(
            &mut state,
            "log('x:', 42, 1.5, nil, NULL, v, v:mask())\nreturn v:sum()",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::F64(1.5));
        let captured = messages.lock().unwrap();
        assert_eq!(captured.len(), 1);
        // Arguments tab-joined; scalars as text, the sentinel by name —
        // and views as summaries, never their contents.
        assert_eq!(
            captured[0],
            "x:\t42\t1.5\tnil\tNULL\tf64 view, len 3\tmask, len 3"
        );
        assert!(!captured[0].contains("0.25"), "no buffer dump");
    }

    #[test]
    fn log_is_a_pure_side_channel() {
        // No sink installed: log is a no-op, not an error.
        let mut state = LuaState::new().unwrap();
        let result = eval(
            &mut state,
            "log('into the void')\nreturn 7",
            &[],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::I64(7));
        // log returns nothing, so it cannot feed a result.
        let result = eval(
            &mut state,
            "local r = log('x')\nif r == nil then return 1 else return 0 end",
            &[],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::I64(1));
        // The answer is identical with and without a sink.
        let quiet = eval(&mut state, "return 2 + 2", &[], ReturnType::I64).unwrap();
        state.set_log_sink(Box::new(Capture(Default::default())));
        let logged = eval(
            &mut state,
            "log('computing')\nreturn 2 + 2",
            &[],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(quiet, logged);
    }

    #[test]
    fn print_and_warn_are_removed() {
        // Their destinations (stdout, stderr) are process streams an
        // embedded library does not own; both fail loudly, pointing the
        // kernel author at log().
        let mut state = LuaState::new().unwrap();
        for chunk in ["print(1)", "warn('w')"] {
            let error = eval(&mut state, chunk, &[], ReturnType::I64).unwrap_err();
            assert!(error.contains("nil"), "{chunk}: {error}");
        }
    }

    // ---- host functions: engine compute over shared views ----

    use crate::host::HostFunction;

    /// Sums its one argument and records the slice pointer it was
    /// handed — the zero-copy probe.
    struct PointerProbe(std::sync::Arc<std::sync::Mutex<usize>>);

    impl HostFunction for PointerProbe {
        fn arity(&self) -> usize {
            1
        }
        fn call(&self, args: &[&[f64]]) -> Result<Option<f64>, String> {
            *self.0.lock().unwrap() = args[0].as_ptr() as usize;
            Ok(Some(args[0].iter().sum()))
        }
    }

    #[test]
    fn host_functions_see_the_engine_buffer_itself() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let mut state = LuaState::new().unwrap();
        state
            .register_host_function("probe", Box::new(PointerProbe(seen.clone())))
            .unwrap();
        let values: Vec<f64> = (0..64).map(|i| f64::from(i) * 0.5).collect();
        let result = eval(
            &mut state,
            "return probe(v)",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::F64(values.iter().sum()));
        // The pointer-verified no-copy claim: the host function received
        // the bound buffer itself, through Lua, with no bytes moved.
        assert_eq!(*seen.lock().unwrap(), values.as_ptr() as usize);
    }

    /// A function that reports undefined (None), errors, or panics on
    /// demand — the trampoline's three non-value outcomes.
    struct Moody(u8);

    impl HostFunction for Moody {
        fn arity(&self) -> usize {
            1
        }
        fn call(&self, _args: &[&[f64]]) -> Result<Option<f64>, String> {
            match self.0 {
                0 => Ok(None),
                1 => Err("op declined".to_owned()),
                _ => panic!("embedder bug"),
            }
        }
    }

    #[test]
    fn host_function_outcomes_map_to_null_error_and_contained_panic() {
        let values = [1.0f64];
        let mut state = LuaState::new().unwrap();
        state
            .register_host_function("undefined", Box::new(Moody(0)))
            .unwrap();
        state
            .register_host_function("failing", Box::new(Moody(1)))
            .unwrap();
        state
            .register_host_function("panicking", Box::new(Moody(2)))
            .unwrap();
        // Undefined → the NULL sentinel, exactly like a SQL window.
        let result = eval(
            &mut state,
            "if undefined(v) == NULL then return 1 else return 0 end",
            &[(c"v", f64s(&values))],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(result, ScalarValue::I64(1));
        // An op error is the script's error, loudly.
        let error = eval(
            &mut state,
            "return failing(v)",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("op declined"), "{error}");
        // A panicking embedder function is contained — a loud Lua error,
        // never an unwind into C — and the state survives.
        let error = eval(
            &mut state,
            "return panicking(v)",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("panicked"), "{error}");
        let alive = eval(&mut state, "return 1", &[], ReturnType::I64).unwrap();
        assert_eq!(alive, ScalarValue::I64(1));
    }

    #[test]
    fn host_function_misuse_fails_loudly() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let mut state = LuaState::new().unwrap();
        state
            .register_host_function("probe", Box::new(PointerProbe(seen)))
            .unwrap();
        let values = [1.0f64, 2.0];
        // Wrong argument count.
        let error = eval(
            &mut state,
            "return probe(v, v)",
            &[(c"v", f64s(&values))],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("wrong number"), "{error}");
        // A plain number is not a view.
        let error = eval(&mut state, "return probe(3)", &[], ReturnType::F64).unwrap_err();
        assert!(error.contains("column views"), "{error}");
        // An i64 view is the wrong element type for the f64 ops.
        let i64s = [1i64, 2];
        let error = eval(
            &mut state,
            "return probe(v)",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &i64s,
                    validity: None,
                },
            )],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("f64 views"), "{error}");
        // A null-bearing view is refused: the ops take dense input.
        let validity = Bitmap::from_bools([true, false]);
        let error = eval(
            &mut state,
            "return probe(v)",
            &[(
                c"v",
                ColumnView::F64 {
                    values: &values,
                    validity: Some(&validity),
                },
            )],
            ReturnType::F64,
        )
        .unwrap_err();
        assert!(error.contains("NULL"), "{error}");
        // Registration bounds the arity.
        struct TooWide;
        impl HostFunction for TooWide {
            fn arity(&self) -> usize {
                99
            }
            fn call(&self, _: &[&[f64]]) -> Result<Option<f64>, String> {
                Ok(None)
            }
        }
        assert!(state
            .register_host_function("wide", Box::new(TooWide))
            .is_err());
    }

    #[test]
    fn lua_state_moves_across_threads() {
        // The Send impl, exercised: build state on one thread, run it on
        // another, with interpreter state (a global) carried across the
        // move. &mut discipline means one thread at a time — which is
        // exactly what std::thread::spawn's move closure proves.
        let mut state = LuaState::new().unwrap();
        eval(&mut state, "g = 7\nreturn 0", &[], ReturnType::I64).unwrap();
        let result = std::thread::spawn(move || {
            let mut state = state;
            eval(&mut state, "return g + 35", &[], ReturnType::I64)
        })
        .join()
        .unwrap()
        .unwrap();
        assert_eq!(result, ScalarValue::I64(42));
    }

    #[test]
    fn compiling_does_not_run_and_a_chunk_runs_many_times() {
        let mut state = LuaState::new().unwrap();
        // Compiling produces a callable without executing it: the
        // chunk's side effect has not happened yet.
        let chunk = state.compile("ran = (ran or 0) + 1\nreturn ran").unwrap();
        let ran = eval(
            &mut state,
            "if ran then return 1 else return 0 end",
            &[],
            ReturnType::I64,
        )
        .unwrap();
        assert_eq!(ran, ScalarValue::I64(0));
        // One compiled chunk runs repeatedly — the registration-time
        // compile, per-window call shape the window slot uses.
        for expected in 1..=3 {
            assert_eq!(
                state.eval_scalar(&chunk, &[], ReturnType::I64).unwrap(),
                ScalarValue::I64(expected)
            );
        }
        // A syntax error is loud at compile time, and the state survives.
        let error = state.compile("return ((").unwrap_err();
        assert!(error.contains("load"), "{error}");
        assert_eq!(
            eval(&mut state, "return 1", &[], ReturnType::I64).unwrap(),
            ScalarValue::I64(1)
        );
    }

    #[test]
    fn a_chunk_from_another_interpreter_is_refused() {
        // Registries are per-state; running a chunk against the wrong
        // one must be loud, never a silent call of some other function.
        let mut first = LuaState::new().unwrap();
        let mut second = LuaState::new().unwrap();
        let chunk = first.compile("return 1").unwrap();
        let error = second
            .eval_scalar(&chunk, &[], ReturnType::I64)
            .unwrap_err();
        assert!(error.contains("different interpreter"), "{error}");
    }

    #[test]
    fn script_errors_return_as_values_and_state_survives() {
        let mut state = LuaState::new().unwrap();
        let error = eval(&mut state, "error('deliberate')", &[], ReturnType::F64).unwrap_err();
        assert!(error.contains("deliberate"), "{error}");
        // The same state keeps working after a script error.
        let value = eval(&mut state, "return 40 + 2", &[], ReturnType::I64).unwrap();
        assert_eq!(value, ScalarValue::I64(42));
    }

    // ---- measurements (Observed numbers cited in module docs) ----

    /// The #41 benchmark: interpreter cost for a representative ad-hoc
    /// kernel (mean absolute deviation, a loop the built-ins don't
    /// cover) over a 4,096-row window — the Observed number feeding
    /// future promote-to-native-op decisions. Run with
    /// `cargo test -p compute-lua --release -- --ignored measure_41`.
    #[test]
    #[ignore = "measurement, not a check: run explicitly in release"]
    fn measure_41_interpreter_kernel_cost() {
        let values: Vec<f64> = (0..4096)
            .map(|i| f64::from(i % 97).mul_add(0.5, f64::from(i % 13)))
            .collect();
        let chunk = "local n = #v\nlocal mean = 0.0\nfor i = 1, n do mean = mean + v[i] end\n\
                     mean = mean / n\nlocal mad = 0.0\n\
                     for i = 1, n do mad = mad + math.abs(v[i] - mean) end\nreturn mad / n";
        let mut state = LuaState::new().unwrap();
        let reference = {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean).abs()).sum::<f64>() / values.len() as f64
        };
        let rounds = 200;
        let start = std::time::Instant::now();
        let mut result = ScalarValue::Null;
        for _ in 0..rounds {
            result = eval(&mut state, chunk, &[(c"v", f64s(&values))], ReturnType::F64).unwrap();
        }
        let elapsed = start.elapsed();
        let ScalarValue::F64(mad) = result else {
            panic!("expected a number, got {result:?}");
        };
        assert!((mad - reference).abs() < 1e-9);
        let per_window = elapsed / rounds;
        let windows_per_second = 1.0 / per_window.as_secs_f64();
        println!(
            "measure_41: 4096-row MAD kernel {per_window:?}/window \
             ({windows_per_second:.0} windows/s), {rounds} rounds"
        );
    }

    /// The calling-convention decision's perf evidence, re-hosted from
    /// the deleted `values_map_spike`: a vectorized scalar UDF producing
    /// a full output column in ONE call, against the per-row
    /// anti-pattern producing the same column in N calls, with a
    /// native-Rust floor for context. Run:
    ///   `cargo test -p compute-lua --release -- --ignored measure_vectorized`
    #[test]
    #[ignore = "measurement, not a check: run explicitly in release"]
    fn measure_vectorized_udf_vs_per_row() {
        use std::hint::black_box;
        let n = 4096usize;
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 1000.0).collect();
        let mut state = LuaState::new().unwrap();

        // Native floor: the same elementwise kernel in Rust.
        let native_rounds = 1000;
        let mut native_out = vec![0.0f64; n];
        let start = std::time::Instant::now();
        for _ in 0..native_rounds {
            for i in 0..n {
                let x = black_box(values[i]);
                native_out[i] = black_box(x * x * 0.5 + x);
            }
        }
        let native = start.elapsed() / native_rounds;

        // Option A: one call; Lua loops the view and writes the column.
        let vec_rounds = 100;
        let vec_chunk = "for i = 1, #v do local x = v[i]; out[i] = x*x*0.5 + x end";
        let mut vec_out = vec![0.0f64; n];
        let mut vec_validity = Bitmap::new_unset(n);
        let start = std::time::Instant::now();
        for _ in 0..vec_rounds {
            eval_col(
                &mut state,
                vec_chunk,
                &[(c"v", f64s(&values))],
                OutputColumn::F64 {
                    values: &mut vec_out,
                    validity: &mut vec_validity,
                },
            )
            .unwrap();
        }
        let vectorized = start.elapsed() / vec_rounds;

        // Per-row anti-pattern: N calls, one element each, stitched back.
        let per_row_rounds = 20;
        let row_chunk = "out[1] = v[1]*v[1]*0.5 + v[1]";
        let mut row_out = vec![0.0f64; n];
        let start = std::time::Instant::now();
        for _ in 0..per_row_rounds {
            for i in 0..n {
                let mut slot = [0.0f64];
                let mut slot_validity = Bitmap::new_unset(1);
                eval_col(
                    &mut state,
                    row_chunk,
                    &[(c"v", f64s(&values[i..i + 1]))],
                    OutputColumn::F64 {
                        values: &mut slot,
                        validity: &mut slot_validity,
                    },
                )
                .unwrap();
                row_out[i] = slot[0];
            }
        }
        let per_row = start.elapsed() / per_row_rounds;

        // All three agree bit-for-bit: no dead-code elimination, honest run.
        for i in 0..n {
            assert_eq!(vec_out[i].to_bits(), native_out[i].to_bits());
            assert_eq!(row_out[i].to_bits(), native_out[i].to_bits());
        }
        println!(
            "measure_vectorized: {n} rows/pass — native {native:?}, vectorized {vectorized:?}, \
             per-row {per_row:?} | per-row/vectorized {:.0}x, vectorized/native {:.0}x",
            per_row.as_secs_f64() / vectorized.as_secs_f64(),
            vectorized.as_secs_f64() / native.as_secs_f64(),
        );
    }

    // ---- the vectorized vocabulary (M4, option A) ----

    /// A composed kernel: one interpreter entry, native loops, and the
    /// same IEEE arithmetic as the native expression slot — bit-exact.
    #[test]
    fn composed_operators_match_native_arithmetic_bit_for_bit() {
        let a: Vec<f64> = (0..500).map(|i| f64::from(i) * 0.5 + 3.0).collect();
        let b: Vec<f64> = (0..500)
            .map(|i| f64::from(i).mul_add(-0.25, 40.0))
            .collect();
        let mut out = vec![0.0f64; 500];
        let mut validity = Bitmap::new_unset(500);
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "return (a - b) / b",
            &[(c"a", f64s(&a)), (c"b", f64s(&b))],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap();
        for i in 0..500 {
            assert!(validity.get(i));
            assert_eq!(out[i].to_bits(), ((a[i] - b[i]) / b[i]).to_bits());
        }
    }

    #[test]
    fn scalars_broadcast_and_i64_views_convert_exactly() {
        let ticks: Vec<i64> = (0..8).map(|i| 1_000_000 + i).collect();
        let mut out = vec![0.0f64; 8];
        let mut validity = Bitmap::new_unset(8);
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "return (v - 1000000) * 2.5",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &ticks,
                    validity: None,
                },
            )],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap();
        for (i, value) in out.iter().enumerate() {
            assert_eq!(*value, i as f64 * 2.5);
        }
        // An i64 element beyond 2^53 cannot cross into f64 arithmetic
        // silently (F3).
        let big: Vec<i64> = vec![(1 << 53) + 1];
        let mut out = [0.0f64];
        let mut validity = Bitmap::new_unset(1);
        let error = eval_col(
            &mut state,
            "return v + 0.0",
            &[(
                c"v",
                ColumnView::I64 {
                    values: &big,
                    validity: None,
                },
            )],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("does not fit f64 exactly"), "{error}");
    }

    #[test]
    fn null_propagates_elementwise_through_operators() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [10.0f64, 20.0, 30.0];
        let a_validity = Bitmap::from_bools([true, false, true]);
        let mut out = [0.0f64; 3];
        let mut out_validity = Bitmap::new_unset(3);
        let mut state = LuaState::new().unwrap();
        eval_col(
            &mut state,
            "return a + b",
            &[
                (
                    c"a",
                    ColumnView::F64 {
                        values: &a,
                        validity: Some(&a_validity),
                    },
                ),
                (c"b", f64s(&b)),
            ],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut out_validity,
            },
        )
        .unwrap();
        assert!(out_validity.get(0) && !out_validity.get(1) && out_validity.get(2));
        assert_eq!(out[0], 11.0);
        assert_eq!(out[2], 33.0);
        // The sentinel itself as an operand: every element NULL, in
        // both operand orders (3VL, elementwise).
        for chunk in ["return a * NULL", "return NULL * a"] {
            let mut out = [0.0f64; 3];
            let mut out_validity = Bitmap::new_unset(3);
            eval_col(
                &mut state,
                chunk,
                &[(c"a", f64s(&b))],
                OutputColumn::F64 {
                    values: &mut out,
                    validity: &mut out_validity,
                },
            )
            .unwrap();
            assert!((0..3).all(|i| !out_validity.get(i)), "{chunk}");
        }
    }

    #[test]
    fn vector_misuse_is_loud() {
        let a = [1.0f64, 2.0];
        let b = [1.0f64, 2.0, 3.0];
        let mut state = LuaState::new().unwrap();
        let mut out = [0.0f64; 2];
        let mut validity = Bitmap::new_unset(2);
        let error = eval_col(
            &mut state,
            "return a + b",
            &[(c"a", f64s(&a)), (c"b", f64s(&b))],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("different lengths"), "{error}");
        // A returned column must fill the whole output.
        let mut validity = Bitmap::new_unset(2);
        let error = eval_col(
            &mut state,
            "return b + 0.0",
            &[(c"a", f64s(&a)), (c"b", f64s(&b))],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("3 elements for 2 output rows"), "{error}");
    }

    #[test]
    fn rolling_combinators_match_per_window_recompute() {
        let n = 300;
        let window = 7;
        // Timestamp-scale offsets: the adversarial shape that breaks
        // the cumsum idiom; the compensated sweep must hold ~1e-12
        // relative to per-window recompute.
        let x: Vec<f64> = (0..n)
            .map(|i| 1.0e12 + f64::from(i as u32).sin() * 3.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0e11 + f64::from(i as u32).cos() * 5.0)
            .collect();
        let mut state = LuaState::new().unwrap();
        for (chunk, reference) in [
            (
                "return rolling_sum(x, 7)",
                Box::new(|lo: usize, hi: usize| x[lo..hi].iter().sum::<f64>())
                    as Box<dyn Fn(usize, usize) -> f64>,
            ),
            (
                "return rolling_mean(x, 7)",
                Box::new(|lo: usize, hi: usize| x[lo..hi].iter().sum::<f64>() / (hi - lo) as f64),
            ),
            (
                "return rolling_dot(x, y, 7)",
                Box::new(|lo: usize, hi: usize| (lo..hi).map(|i| x[i] * y[i]).sum::<f64>()),
            ),
        ] {
            let mut out = vec![0.0f64; n];
            let mut validity = Bitmap::new_unset(n);
            eval_col(
                &mut state,
                chunk,
                &[(c"x", f64s(&x)), (c"y", f64s(&y))],
                OutputColumn::F64 {
                    values: &mut out,
                    validity: &mut validity,
                },
            )
            .unwrap();
            for (i, got) in out.iter().enumerate() {
                let lo = (i + 1).saturating_sub(window);
                let expected = reference(lo, i + 1);
                let scale = expected.abs().max(1.0);
                assert!(
                    ((got - expected) / scale).abs() < 1e-12,
                    "{chunk}: row {i}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn rolling_combinators_refuse_nulls_and_bad_windows() {
        let x = [1.0f64, 2.0, 3.0];
        let x_validity = Bitmap::from_bools([true, false, true]);
        let mut state = LuaState::new().unwrap();
        let mut out = [0.0f64; 3];
        let mut validity = Bitmap::new_unset(3);
        let error = eval_col(
            &mut state,
            "return rolling_sum(x, 2)",
            &[(
                c"x",
                ColumnView::F64 {
                    values: &x,
                    validity: Some(&x_validity),
                },
            )],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("non-null input"), "{error}");
        let mut validity = Bitmap::new_unset(3);
        let error = eval_col(
            &mut state,
            "return rolling_sum(x, 0)",
            &[(c"x", f64s(&x))],
            OutputColumn::F64 {
                values: &mut out,
                validity: &mut validity,
            },
        )
        .unwrap_err();
        assert!(error.contains("positive integer"), "{error}");
    }

    #[test]
    fn vectors_survive_calls_but_views_still_do_not() {
        // Vectors own their data (GC-managed), so stashing one across
        // calls is safe; views stay generation-poisoned — the lifetime
        // discipline is unchanged.
        let a = [5.0f64, 6.0];
        let mut state = LuaState::new().unwrap();
        let stash = state
            .compile("stashed_vec = a + 0.0\nstashed_view = a\nreturn 0")
            .unwrap();
        state
            .eval_scalar(&stash, &[(c"a", f64s(&a))], ReturnType::F64)
            .unwrap();
        let read_vector = state.compile("return stashed_vec[2]").unwrap();
        assert_eq!(
            state
                .eval_scalar(&read_vector, &[], ReturnType::F64)
                .unwrap(),
            ScalarValue::F64(6.0)
        );
        let read_view = state.compile("return stashed_view[2]").unwrap();
        let error = state
            .eval_scalar(&read_view, &[], ReturnType::F64)
            .unwrap_err();
        assert!(error.contains("outside its call"), "{error}");
    }
}
