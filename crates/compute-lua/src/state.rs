//! The safe wrapper over the thin bindings — currently the #41
//! confirmation spike: zero-copy column views handed to Lua chunks,
//! pointer-verified. This module grows into the crate's backend; its
//! discipline is fixed now and does not change as it grows:
//!
//! 1. Every entry into Lua goes through `lua_pcall` — nothing runs
//!    unprotected.
//! 2. A Rust function called *from* Lua never raises a Lua error
//!    across a frame with pending destructors: view accessors keep
//!    only `Copy` locals, and `lua_error` is the tail call.
//! 3. The boundary never propagates a Rust panic into C.
//!
//! ## The view (the crate's reason to exist)
//!
//! A view userdata holds `(pointer, length)` into an engine buffer —
//! 16 bytes of handle; **zero bytes of data move**. Scripts index it
//! `v[i]` (1-based, like Lua), take `#v`, and read `f64` elements as
//! Lua floats and `i64` elements as Lua 5.4 integers — exactly, the
//! alignment that decided the interpreter. Out-of-range access is a
//! loud error (this engine refuses wrong answers; a silent `nil`
//! would turn into one), and views are read-only. A view is valid for
//! the duration of one protected call; the caller poisons it after
//! (length zeroed), so a script that smuggles the handle out gets
//! errors later, never a dangling read.
//!
//! Interpreter cost, Observed (run 2026-07-24, release,
//! `measure_41_interpreter_kernel_cost`): a 4,096-row mean-absolute-
//! deviation kernel — two full passes over the view — takes ~484µs per
//! window (~2,066 windows/s, ~17M element reads/s through the
//! metamethod accessor). That is the price of the ad-hoc layer; per
//! the promotion ladder (DESIGN.md, *The Lua layer*), a kernel that
//! proves hot graduates to a curated native op rather than the
//! interpreter getting a JIT.

use crate::ffi;
use std::ffi::{c_char, c_int, CStr};

/// Payload of a view userdata: a borrowed engine buffer. `tag`
/// selects the element type; `len == 0` marks a poisoned view.
#[repr(C)]
#[derive(Clone, Copy)]
struct ViewPayload {
    ptr: *const u8,
    len: usize,
    tag: u8,
}

const TAG_F64: u8 = 0;
const TAG_I64: u8 = 1;

/// Metatable name in the registry.
const VIEW_METATABLE: &CStr = c"tallydb.view";

/// An embedded Lua 5.4 interpreter with the curated library set
/// (base, math, string, table — no io, no os, no debug; the package
/// library is not even linked, per the ANSI build).
pub struct LuaState {
    raw: *mut ffi::lua_State,
}

/// A borrowed column argument for one script call.
pub enum ViewArg<'a> {
    /// An `f64` buffer, read by scripts as Lua floats.
    F64(&'a [f64]),
    /// An `i64` buffer, read by scripts as Lua integers — exactly.
    I64(&'a [i64]),
}

impl LuaState {
    /// Creates a state with the curated libraries and the view
    /// metatable installed.
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
            // The one shared view metatable: __index (element reads),
            // __len, and a read-only __newindex.
            ffi::luaL_newmetatable(raw, VIEW_METATABLE.as_ptr());
            ffi::lua_pushcclosure(raw, view_index, 0);
            ffi::lua_setfield(raw, -2, c"__index".as_ptr());
            ffi::lua_pushcclosure(raw, view_len, 0);
            ffi::lua_setfield(raw, -2, c"__len".as_ptr());
            ffi::lua_pushcclosure(raw, view_newindex, 0);
            ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
            ffi::lua_settop(raw, 0);
            Ok(LuaState { raw })
        }
    }

    /// Runs `chunk` (text only) with `views` bound to global names,
    /// returning the chunk's single numeric result as `f64`. Views are
    /// valid only inside this call — each is poisoned before return,
    /// so the borrow in `ViewArg` is never outlived. Every failure —
    /// load error, runtime error, non-numeric result — is a loud `Err`.
    pub fn eval_scalar(
        &mut self,
        chunk: &str,
        views: &[(&CStr, ViewArg<'_>)],
    ) -> Result<f64, String> {
        unsafe {
            debug_assert_eq!(ffi::lua_gettop(self.raw), 0);
            let mut handles = Vec::with_capacity(views.len());
            for (name, view) in views {
                let payload = match view {
                    ViewArg::F64(values) => ViewPayload {
                        ptr: values.as_ptr().cast(),
                        len: values.len(),
                        tag: TAG_F64,
                    },
                    ViewArg::I64(values) => ViewPayload {
                        ptr: values.as_ptr().cast(),
                        len: values.len(),
                        tag: TAG_I64,
                    },
                };
                let slot = ffi::lua_newuserdatauv(self.raw, std::mem::size_of::<ViewPayload>(), 0)
                    .cast::<ViewPayload>();
                slot.write(payload);
                handles.push(slot);
                ffi::luaL_setmetatable(self.raw, VIEW_METATABLE.as_ptr());
                ffi::lua_setglobal(self.raw, name.as_ptr());
            }
            let result = self.run_scalar(chunk);
            // Poison every view before the borrows end: a handle kept
            // by the script past this call errors instead of dangling.
            // Only the length is zeroed — the accessor checks it before
            // any dereference, and the retained pointer lets tests
            // verify zero-copy after the call (it is never read again).
            for slot in handles {
                (*slot).len = 0;
            }
            ffi::lua_settop(self.raw, 0);
            result
        }
    }

    unsafe fn run_scalar(&mut self, chunk: &str) -> Result<f64, String> {
        unsafe {
            let status = ffi::luaL_loadbufferx(
                self.raw,
                chunk.as_ptr().cast(),
                chunk.len(),
                c"script".as_ptr(),
                c"t".as_ptr(),
            );
            if status != ffi::LUA_OK {
                return Err(self.pop_error("load"));
            }
            if ffi::lua_pcall(self.raw, 0, 1, 0) != ffi::LUA_OK {
                return Err(self.pop_error("run"));
            }
            if ffi::lua_type(self.raw, -1) != ffi::LUA_TNUMBER {
                ffi::lua_settop(self.raw, -2);
                return Err("script did not return a number".to_owned());
            }
            let mut ok = 0;
            let value = ffi::lua_tonumberx(self.raw, -1, &mut ok);
            ffi::lua_settop(self.raw, -2);
            Ok(value)
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
        unsafe {
            ffi::lua_getglobal(self.raw, view_global.as_ptr());
            let payload = ffi::luaL_testudata(self.raw, -1, VIEW_METATABLE.as_ptr());
            let pointer = (!payload.is_null()).then(|| (*payload.cast::<ViewPayload>()).ptr);
            ffi::lua_settop(self.raw, -2);
            pointer
        }
    }
}

impl Drop for LuaState {
    fn drop(&mut self) {
        unsafe { ffi::lua_close(self.raw) }
    }
}

/// `__index`: `v[i]` — bounds-checked element read. Discipline note:
/// every local here is `Copy`; the error paths call `lua_error` as the
/// tail, so the `longjmp` unwinds no Rust destructor.
unsafe extern "C" fn view_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, VIEW_METATABLE.as_ptr()).cast::<ViewPayload>();
        if payload.is_null() {
            return raise(state, c"view accessor on a non-view");
        }
        let view = *payload;
        let mut is_integer = 0;
        let index = ffi::lua_tointegerx(state, 2, &mut is_integer);
        if is_integer == 0 {
            return raise(state, c"view index must be an integer");
        }
        if view.len == 0 {
            return raise(state, c"view used outside its call");
        }
        if index < 1 || index as usize > view.len {
            return raise(state, c"view index out of range");
        }
        let offset = (index - 1) as usize;
        match view.tag {
            TAG_F64 => ffi::lua_pushnumber(state, *view.ptr.cast::<f64>().add(offset)),
            _ => ffi::lua_pushinteger(state, *view.ptr.cast::<i64>().add(offset)),
        }
        1
    }
}

/// `__len`: `#v`.
unsafe extern "C" fn view_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, VIEW_METATABLE.as_ptr()).cast::<ViewPayload>();
        if payload.is_null() {
            return raise(state, c"view accessor on a non-view");
        }
        ffi::lua_pushinteger(state, (*payload).len as i64);
        1
    }
}

/// `__newindex`: views are read-only.
unsafe extern "C" fn view_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe { raise(state, c"views are read-only") }
}

/// Pushes `message` and raises it — tail position only, `Copy` state
/// only (rule 2 of the module discipline).
unsafe fn raise(state: *mut ffi::lua_State, message: &CStr) -> c_int {
    unsafe {
        let bytes = message.to_bytes();
        ffi::lua_pushlstring(state, bytes.as_ptr().cast::<c_char>(), bytes.len());
        ffi::lua_error(state)
    }
}

#[cfg(test)]
mod spike_tests {
    //! The #41 confirmation spike, as ruled: the zero-copy hand-off
    //! proven by pointer comparison (the passthrough-test pattern),
    //! `i64` exactness across the boundary, and the loud failure of
    //! every misuse the view API can express.

    use super::*;

    #[test]
    fn f64_view_is_zero_copy_and_reads_exactly() {
        let values: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.25 - 100.0).collect();
        let mut state = LuaState::new().unwrap();
        let sum = state
            .eval_scalar(
                "local s = 0.0\nfor i = 1, #v do s = s + v[i] end\nreturn s",
                &[(c"v", ViewArg::F64(&values))],
            )
            .unwrap();
        // Same order, same arithmetic: bit-exact agreement, not approximate.
        let expected: f64 = values.iter().sum();
        assert_eq!(sum.to_bits(), expected.to_bits());
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
        let difference = state
            .eval_scalar(
                "return v[1] - 9007199254740992",
                &[(c"v", ViewArg::I64(&values))],
            )
            .unwrap();
        assert_eq!(difference, 1.0);
        let pointer = state.view_data_pointer(c"v").expect("view global");
        assert_eq!(pointer, values.as_ptr().cast());
    }

    #[test]
    fn view_misuse_fails_loudly() {
        let values = [1.0f64, 2.0];
        let mut state = LuaState::new().unwrap();
        // Out of range — never nil, always an error.
        let error = state
            .eval_scalar("return v[3]", &[(c"v", ViewArg::F64(&values))])
            .unwrap_err();
        assert!(error.contains("out of range"), "{error}");
        // Zero is out of range too (views are 1-based like Lua).
        let error = state
            .eval_scalar("return v[0]", &[(c"v", ViewArg::F64(&values))])
            .unwrap_err();
        assert!(error.contains("out of range"), "{error}");
        // Non-integer index.
        let error = state
            .eval_scalar("return v['x']", &[(c"v", ViewArg::F64(&values))])
            .unwrap_err();
        assert!(error.contains("must be an integer"), "{error}");
        // Read-only.
        let error = state
            .eval_scalar("v[1] = 9\nreturn 0", &[(c"v", ViewArg::F64(&values))])
            .unwrap_err();
        assert!(error.contains("read-only"), "{error}");
    }

    #[test]
    fn a_view_smuggled_past_its_call_is_poisoned() {
        let values = [1.0f64, 2.0];
        let mut state = LuaState::new().unwrap();
        state
            .eval_scalar(
                "stash = function() return v[1] end\nreturn 0",
                &[(c"v", ViewArg::F64(&values))],
            )
            .unwrap();
        // The buffer's borrow has ended; the stashed closure must find
        // a poisoned view, never a dangling read.
        let error = state.eval_scalar("return stash()", &[]).unwrap_err();
        assert!(error.contains("outside its call"), "{error}");
    }

    #[test]
    fn script_errors_return_as_values_and_state_survives() {
        let mut state = LuaState::new().unwrap();
        let error = state.eval_scalar("error('deliberate')", &[]).unwrap_err();
        assert!(error.contains("deliberate"), "{error}");
        // The same state keeps working after a script error.
        let value = state.eval_scalar("return 40 + 2", &[]).unwrap();
        assert_eq!(value, 42.0);
    }

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
        let mut result = 0.0;
        for _ in 0..rounds {
            result = state
                .eval_scalar(chunk, &[(c"v", ViewArg::F64(&values))])
                .unwrap();
        }
        let elapsed = start.elapsed();
        assert!((result - reference).abs() < 1e-9);
        let per_window = elapsed / rounds;
        let windows_per_second = 1.0 / per_window.as_secs_f64();
        println!(
            "measure_41: 4096-row MAD kernel {per_window:?}/window \
             ({windows_per_second:.0} windows/s), {rounds} rounds"
        );
    }
}
