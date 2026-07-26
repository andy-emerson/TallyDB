//! Throwaway spike (F1 values-map decision, 2026-07-26) — NOT the real
//! implementation. Its only job is to earn *Observed* evidence for the
//! ruling "columns are views, rows are tables, nil is NULL" before that
//! contract freezes. It adds the two pieces `state.rs`'s #41 spike lacks
//! — a **nullable** input view (null reads as `nil`) and an **output**
//! view (`out[i] = nil` marks the validity bit, never deletes the slot)
//! — and points adversarial kernels at them.
//!
//! Probes (R4 zero-copy and R5 i64-exactness are already green in
//! `state.rs`, so they are not repeated here):
//! - **R1** — does `nil`-as-NULL make naive kernels crash, and is the
//!   guarded form livable?
//! - **R1b** — are NaN (a value) and NULL (`nil`) distinguishable?
//! - **R2** — does `out[i] = nil` mark validity rather than delete?
//! - **R3** — is a real rolling kernel expressible view-only?
//!
//! Discipline is `state.rs`'s: `Copy`-only locals in the C accessors,
//! `lua_error` in tail position, every entry through `lua_pcall`.

use crate::ffi;
use std::ffi::{c_char, c_int, CStr};

#[repr(C)]
#[derive(Clone, Copy)]
struct NullableView {
    ptr: *const f64,
    valid: *const u8, // one byte per element, 0 = null
    len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OutputView {
    ptr: *mut f64,
    valid: *mut u8,
    len: usize,
}

const META_NULLABLE: &CStr = c"tallydb.spike.nullable";
const META_OUTPUT: &CStr = c"tallydb.spike.output";

struct SpikeState {
    raw: *mut ffi::lua_State,
}

impl Drop for SpikeState {
    fn drop(&mut self) {
        unsafe { ffi::lua_close(self.raw) }
    }
}

impl SpikeState {
    fn new() -> SpikeState {
        unsafe {
            let raw = ffi::luaL_newstate();
            assert!(!raw.is_null());
            ffi::luaL_requiref(raw, c"_G".as_ptr(), ffi::luaopen_base, 1);
            ffi::luaL_requiref(raw, c"math".as_ptr(), ffi::luaopen_math, 1);
            ffi::lua_settop(raw, 0);

            // Nullable input view: __index returns nil for null slots.
            ffi::luaL_newmetatable(raw, META_NULLABLE.as_ptr());
            ffi::lua_pushcclosure(raw, nullable_index, 0);
            ffi::lua_setfield(raw, -2, c"__index".as_ptr());
            ffi::lua_pushcclosure(raw, nullable_len, 0);
            ffi::lua_setfield(raw, -2, c"__len".as_ptr());
            ffi::lua_pushcclosure(raw, readonly_newindex, 0);
            ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
            ffi::lua_settop(raw, 0);

            // Output view: __newindex accepts a number (set + mark valid)
            // or nil (mark invalid); __index reads back.
            ffi::luaL_newmetatable(raw, META_OUTPUT.as_ptr());
            ffi::lua_pushcclosure(raw, output_index, 0);
            ffi::lua_setfield(raw, -2, c"__index".as_ptr());
            ffi::lua_pushcclosure(raw, output_len, 0);
            ffi::lua_setfield(raw, -2, c"__len".as_ptr());
            ffi::lua_pushcclosure(raw, output_newindex, 0);
            ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
            ffi::lua_settop(raw, 0);

            SpikeState { raw }
        }
    }

    /// Bind `v` as a nullable view over `(values, valid)` and return the
    /// chunk's single numeric result. `valid[i] == false` → null.
    fn eval_scalar_nullable(
        &mut self,
        chunk: &str,
        values: &[f64],
        valid: &[bool],
    ) -> Result<f64, String> {
        assert_eq!(values.len(), valid.len());
        unsafe {
            let payload = NullableView {
                ptr: values.as_ptr(),
                valid: valid.as_ptr().cast::<u8>(),
                len: values.len(),
            };
            let slot = ffi::lua_newuserdatauv(self.raw, std::mem::size_of::<NullableView>(), 0)
                .cast::<NullableView>();
            slot.write(payload);
            ffi::luaL_setmetatable(self.raw, META_NULLABLE.as_ptr());
            ffi::lua_setglobal(self.raw, c"v".as_ptr());
            let result = self.run(chunk, 1).and_then(|()| self.pop_number());
            ffi::lua_settop(self.raw, 0);
            result
        }
    }

    /// Bind `v` (nullable input) and `out` (output of `out_len`), run the
    /// chunk for effect, and return the output buffers.
    fn eval_into_output(
        &mut self,
        chunk: &str,
        values: &[f64],
        valid: &[bool],
        out_len: usize,
    ) -> Result<(Vec<f64>, Vec<bool>), String> {
        assert_eq!(values.len(), valid.len());
        let mut out_values = vec![0.0f64; out_len];
        let mut out_valid = vec![0u8; out_len];
        unsafe {
            let input = NullableView {
                ptr: values.as_ptr(),
                valid: valid.as_ptr().cast::<u8>(),
                len: values.len(),
            };
            let islot = ffi::lua_newuserdatauv(self.raw, std::mem::size_of::<NullableView>(), 0)
                .cast::<NullableView>();
            islot.write(input);
            ffi::luaL_setmetatable(self.raw, META_NULLABLE.as_ptr());
            ffi::lua_setglobal(self.raw, c"v".as_ptr());

            let output = OutputView {
                ptr: out_values.as_mut_ptr(),
                valid: out_valid.as_mut_ptr(),
                len: out_len,
            };
            let oslot = ffi::lua_newuserdatauv(self.raw, std::mem::size_of::<OutputView>(), 0)
                .cast::<OutputView>();
            oslot.write(output);
            ffi::luaL_setmetatable(self.raw, META_OUTPUT.as_ptr());
            ffi::lua_setglobal(self.raw, c"out".as_ptr());

            let result = self.run(chunk, 0);
            ffi::lua_settop(self.raw, 0);
            result.map(|()| (out_values, out_valid.into_iter().map(|b| b != 0).collect()))
        }
    }

    unsafe fn run(&mut self, chunk: &str, results: c_int) -> Result<(), String> {
        unsafe {
            let status = ffi::luaL_loadbufferx(
                self.raw,
                chunk.as_ptr().cast(),
                chunk.len(),
                c"spike".as_ptr(),
                c"t".as_ptr(),
            );
            if status != ffi::LUA_OK {
                return Err(self.pop_error());
            }
            if ffi::lua_pcall(self.raw, 0, results, 0) != ffi::LUA_OK {
                return Err(self.pop_error());
            }
            Ok(())
        }
    }

    unsafe fn pop_number(&mut self) -> Result<f64, String> {
        unsafe {
            if ffi::lua_type(self.raw, -1) != ffi::LUA_TNUMBER {
                return Err("script did not return a number".to_owned());
            }
            let mut ok = 0;
            Ok(ffi::lua_tonumberx(self.raw, -1, &mut ok))
        }
    }

    unsafe fn pop_error(&mut self) -> String {
        unsafe {
            let mut len = 0usize;
            let text = ffi::lua_tolstring(self.raw, -1, &mut len);
            let message = if text.is_null() {
                "error object is not a string".to_owned()
            } else {
                let bytes = std::slice::from_raw_parts(text.cast(), len);
                String::from_utf8_lossy(bytes).into_owned()
            };
            ffi::lua_settop(self.raw, -2);
            message
        }
    }
}

unsafe extern "C" fn nullable_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_NULLABLE.as_ptr()).cast::<NullableView>();
        if payload.is_null() {
            return raise(state, c"view accessor on a non-view");
        }
        let view = *payload;
        let mut is_integer = 0;
        let index = ffi::lua_tointegerx(state, 2, &mut is_integer);
        if is_integer == 0 {
            return raise(state, c"view index must be an integer");
        }
        if index < 1 || index as usize > view.len {
            return raise(state, c"view index out of range");
        }
        let offset = (index - 1) as usize;
        if *view.valid.add(offset) == 0 {
            ffi::lua_pushnil(state); // NULL crosses as nil
        } else {
            ffi::lua_pushnumber(state, *view.ptr.add(offset));
        }
        1
    }
}

unsafe extern "C" fn nullable_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_NULLABLE.as_ptr()).cast::<NullableView>();
        if payload.is_null() {
            return raise(state, c"view accessor on a non-view");
        }
        ffi::lua_pushinteger(state, (*payload).len as i64);
        1
    }
}

unsafe extern "C" fn output_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_OUTPUT.as_ptr()).cast::<OutputView>();
        if payload.is_null() {
            return raise(state, c"output accessor on a non-output");
        }
        let view = *payload;
        let mut is_integer = 0;
        let index = ffi::lua_tointegerx(state, 2, &mut is_integer);
        if is_integer == 0 {
            return raise(state, c"output index must be an integer");
        }
        if index < 1 || index as usize > view.len {
            return raise(state, c"output index out of range");
        }
        let offset = (index - 1) as usize;
        if *view.valid.add(offset) == 0 {
            ffi::lua_pushnil(state);
        } else {
            ffi::lua_pushnumber(state, *view.ptr.add(offset));
        }
        1
    }
}

unsafe extern "C" fn output_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_OUTPUT.as_ptr()).cast::<OutputView>();
        if payload.is_null() {
            return raise(state, c"output accessor on a non-output");
        }
        ffi::lua_pushinteger(state, (*payload).len as i64);
        1
    }
}

/// `out[i] = x`: a number sets the slot and marks it valid; `nil` marks
/// the slot null (it does NOT delete it — the slot and the length are
/// preserved, which is the whole point a Lua table can't do).
unsafe extern "C" fn output_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload = ffi::luaL_testudata(state, 1, META_OUTPUT.as_ptr()).cast::<OutputView>();
        if payload.is_null() {
            return raise(state, c"output accessor on a non-output");
        }
        let view = *payload;
        let mut is_integer = 0;
        let index = ffi::lua_tointegerx(state, 2, &mut is_integer);
        if is_integer == 0 {
            return raise(state, c"output index must be an integer");
        }
        if index < 1 || index as usize > view.len {
            return raise(state, c"output index out of range");
        }
        let offset = (index - 1) as usize;
        match ffi::lua_type(state, 3) {
            ffi::LUA_TNIL => {
                *view.valid.add(offset) = 0;
            }
            ffi::LUA_TNUMBER => {
                let mut ok = 0;
                let value = ffi::lua_tonumberx(state, 3, &mut ok);
                *view.ptr.add(offset) = value;
                *view.valid.add(offset) = 1;
            }
            _ => return raise(state, c"output expects a number or nil"),
        }
        0
    }
}

unsafe extern "C" fn readonly_newindex(state: *mut ffi::lua_State) -> c_int {
    unsafe { raise(state, c"input views are read-only") }
}

unsafe fn raise(state: *mut ffi::lua_State, message: &CStr) -> c_int {
    unsafe {
        let bytes = message.to_bytes();
        ffi::lua_pushlstring(state, bytes.as_ptr().cast::<c_char>(), bytes.len());
        ffi::lua_error(state)
    }
}

// ---------------------------------------------------------------------
// Config C — the pd.NA-style smart sentinel (2026-07-26 bake-off).
//
// A singleton `NULL` userdata, stashed in the registry and exposed as a
// global. Its metatable propagates through arithmetic (3VL: any operation
// touching NULL yields NULL, so a naive sum does NOT crash — it poisons
// to NULL), keeps identity so `x == NULL` works, and is distinct from
// both `nil` and NaN. The Lua-language wart it cannot escape: it is
// userdata, hence truthy, and relational operators (`<`,`<=`) are forced
// by Lua to return a boolean, so 3VL cannot propagate through them.
// ---------------------------------------------------------------------

const META_NULLCONST: &CStr = c"tallydb.spike.nullconst";
const REG_NULL_SINGLETON: &CStr = c"tallydb.spike.NULL";

#[repr(C)]
#[derive(Clone, Copy)]
struct NullableViewC {
    ptr: *const f64,
    valid: *const u8,
    len: usize,
}

struct SpikeStateC {
    raw: *mut ffi::lua_State,
}

impl Drop for SpikeStateC {
    fn drop(&mut self) {
        unsafe { ffi::lua_close(self.raw) }
    }
}

/// What a Config-C scalar kernel returned: a number, or NULL propagated
/// out.
#[derive(Debug, PartialEq)]
enum CScalar {
    Num(f64),
    Null,
}

impl SpikeStateC {
    fn new() -> SpikeStateC {
        unsafe {
            let raw = ffi::luaL_newstate();
            assert!(!raw.is_null());
            ffi::luaL_requiref(raw, c"_G".as_ptr(), ffi::luaopen_base, 1);
            ffi::luaL_requiref(raw, c"math".as_ptr(), ffi::luaopen_math, 1);
            ffi::lua_settop(raw, 0);

            // The NULL sentinel's metatable: arithmetic propagates, no
            // __eq (identity equality, so `x == NULL` works), __tostring
            // for diagnostics. No __lt/__le — comparing NULL is a loud
            // error, which is the honest limit of 3VL-through-Lua.
            ffi::luaL_newmetatable(raw, META_NULLCONST.as_ptr());
            for op in [
                c"__add".as_ptr(),
                c"__sub".as_ptr(),
                c"__mul".as_ptr(),
                c"__div".as_ptr(),
                c"__mod".as_ptr(),
                c"__pow".as_ptr(),
                c"__unm".as_ptr(),
            ] {
                ffi::lua_pushcclosure(raw, null_propagate, 0);
                ffi::lua_setfield(raw, -2, op);
            }
            ffi::lua_pushcclosure(raw, null_tostring, 0);
            ffi::lua_setfield(raw, -2, c"__tostring".as_ptr());
            ffi::lua_settop(raw, 0);

            // The singleton: a zero-payload userdata with that metatable,
            // stashed in the registry and set as global `NULL`.
            ffi::lua_newuserdatauv(raw, 1, 0);
            ffi::luaL_setmetatable(raw, META_NULLCONST.as_ptr());
            ffi::lua_setfield(raw, ffi::LUA_REGISTRYINDEX, REG_NULL_SINGLETON.as_ptr());
            ffi::lua_getfield(raw, ffi::LUA_REGISTRYINDEX, REG_NULL_SINGLETON.as_ptr());
            ffi::lua_setglobal(raw, c"NULL".as_ptr());

            // Nullable view whose __index returns the singleton for nulls.
            ffi::luaL_newmetatable(raw, META_NULLABLE_C.as_ptr());
            ffi::lua_pushcclosure(raw, nullable_c_index, 0);
            ffi::lua_setfield(raw, -2, c"__index".as_ptr());
            ffi::lua_pushcclosure(raw, nullable_c_len, 0);
            ffi::lua_setfield(raw, -2, c"__len".as_ptr());
            ffi::lua_pushcclosure(raw, readonly_newindex, 0);
            ffi::lua_setfield(raw, -2, c"__newindex".as_ptr());
            ffi::lua_settop(raw, 0);

            SpikeStateC { raw }
        }
    }

    fn eval_scalar_c(
        &mut self,
        chunk: &str,
        values: &[f64],
        valid: &[bool],
    ) -> Result<CScalar, String> {
        assert_eq!(values.len(), valid.len());
        unsafe {
            let payload = NullableViewC {
                ptr: values.as_ptr(),
                valid: valid.as_ptr().cast::<u8>(),
                len: values.len(),
            };
            let slot = ffi::lua_newuserdatauv(self.raw, std::mem::size_of::<NullableViewC>(), 0)
                .cast::<NullableViewC>();
            slot.write(payload);
            ffi::luaL_setmetatable(self.raw, META_NULLABLE_C.as_ptr());
            ffi::lua_setglobal(self.raw, c"v".as_ptr());
            let result = self.run(chunk);
            ffi::lua_settop(self.raw, 0);
            result
        }
    }

    unsafe fn run(&mut self, chunk: &str) -> Result<CScalar, String> {
        unsafe {
            let status = ffi::luaL_loadbufferx(
                self.raw,
                chunk.as_ptr().cast(),
                chunk.len(),
                c"spikec".as_ptr(),
                c"t".as_ptr(),
            );
            if status != ffi::LUA_OK {
                return Err(self.pop_error());
            }
            if ffi::lua_pcall(self.raw, 0, 1, 0) != ffi::LUA_OK {
                return Err(self.pop_error());
            }
            // Is the single result the NULL singleton, a number, or wrong?
            if !ffi::luaL_testudata(self.raw, -1, META_NULLCONST.as_ptr()).is_null() {
                Ok(CScalar::Null)
            } else if ffi::lua_type(self.raw, -1) == ffi::LUA_TNUMBER {
                let mut ok = 0;
                Ok(CScalar::Num(ffi::lua_tonumberx(self.raw, -1, &mut ok)))
            } else {
                Err("script returned neither a number nor NULL".to_owned())
            }
        }
    }

    unsafe fn pop_error(&mut self) -> String {
        unsafe {
            let mut len = 0usize;
            let text = ffi::lua_tolstring(self.raw, -1, &mut len);
            let message = if text.is_null() {
                "error object is not a string".to_owned()
            } else {
                let bytes = std::slice::from_raw_parts(text.cast(), len);
                String::from_utf8_lossy(bytes).into_owned()
            };
            ffi::lua_settop(self.raw, -2);
            message
        }
    }
}

const META_NULLABLE_C: &CStr = c"tallydb.spike.nullable_c";

/// Arithmetic metamethod: return the NULL singleton (3VL propagation).
unsafe extern "C" fn null_propagate(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        ffi::lua_getfield(state, ffi::LUA_REGISTRYINDEX, REG_NULL_SINGLETON.as_ptr());
        1
    }
}

unsafe extern "C" fn null_tostring(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let s = b"NULL";
        ffi::lua_pushlstring(state, s.as_ptr().cast::<c_char>(), s.len());
        1
    }
}

/// Config-C nullable view: null slots read as the NULL singleton.
unsafe extern "C" fn nullable_c_index(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload =
            ffi::luaL_testudata(state, 1, META_NULLABLE_C.as_ptr()).cast::<NullableViewC>();
        if payload.is_null() {
            return raise(state, c"view accessor on a non-view");
        }
        let view = *payload;
        let mut is_integer = 0;
        let index = ffi::lua_tointegerx(state, 2, &mut is_integer);
        if is_integer == 0 {
            return raise(state, c"view index must be an integer");
        }
        if index < 1 || index as usize > view.len {
            return raise(state, c"view index out of range");
        }
        let offset = (index - 1) as usize;
        if *view.valid.add(offset) == 0 {
            ffi::lua_getfield(state, ffi::LUA_REGISTRYINDEX, REG_NULL_SINGLETON.as_ptr());
        } else {
            ffi::lua_pushnumber(state, *view.ptr.add(offset));
        }
        1
    }
}

unsafe extern "C" fn nullable_c_len(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let payload =
            ffi::luaL_testudata(state, 1, META_NULLABLE_C.as_ptr()).cast::<NullableViewC>();
        if payload.is_null() {
            return raise(state, c"view accessor on a non-view");
        }
        ffi::lua_pushinteger(state, (*payload).len as i64);
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // R1 — nil-as-NULL: a naive sum crashes LOUDLY on a null (not a
    // silent wrong answer), and the guarded form reads naturally and
    // skips it. The finding is whether "loud crash, guard required" is an
    // acceptable ergonomics tax.
    #[test]
    fn r1_naive_sum_errors_on_null_guarded_sum_works() {
        let values = [1.0, 2.0, 3.0];
        let valid = [true, false, true]; // middle is NULL
        let mut state = SpikeState::new();

        let naive = state.eval_scalar_nullable(
            "local s = 0.0\nfor i = 1, #v do s = s + v[i] end\nreturn s",
            &values,
            &valid,
        );
        let err = naive.expect_err("arithmetic on nil must be a loud error");
        assert!(
            err.contains("nil"),
            "naive sum should fail loudly on the null, got: {err}"
        );

        let guarded = state
            .eval_scalar_nullable(
                "local s = 0.0\nfor i = 1, #v do if v[i] ~= nil then s = s + v[i] end end\nreturn s",
                &values,
                &valid,
            )
            .expect("guarded sum runs");
        assert_eq!(guarded, 4.0); // 1 + 3, null skipped
    }

    // R1b — NaN is a VALUE (self-unequal, in-band), NULL is nil. A script
    // can tell them apart; they don't collapse.
    #[test]
    fn r1b_nan_and_null_are_distinguishable() {
        let values = [f64::NAN, 5.0];
        let valid = [true, false]; // index 1 = NaN value; index 2 = NULL
        let mut state = SpikeState::new();
        let code = state
            .eval_scalar_nullable(
                "local nils, nans = 0, 0\n\
                 for i = 1, #v do\n\
                   if v[i] == nil then nils = nils + 1\n\
                   elseif v[i] ~= v[i] then nans = nans + 1 end\n\
                 end\n\
                 return nils * 10 + nans",
                &values,
                &valid,
            )
            .expect("runs");
        assert_eq!(code, 11.0, "exactly one nil and one NaN, kept distinct");
    }

    // R2 — out[i] = nil marks the validity bit; the slot survives and the
    // length is unchanged. This is the thing a Lua table cannot do.
    #[test]
    fn r2_output_nil_marks_null_not_delete() {
        let values = [10.0, 20.0, 30.0];
        let valid = [true, true, true];
        let mut state = SpikeState::new();
        let (out_values, out_valid) = state
            .eval_into_output(
                "out[1] = v[1]\nout[2] = nil\nout[3] = v[3]",
                &values,
                &valid,
                3,
            )
            .expect("runs");
        assert_eq!(out_valid, vec![true, false, true], "middle marked null");
        assert_eq!(out_values[0], 10.0);
        assert_eq!(out_values[2], 30.0);
        // Length preserved: three slots, not a two-element sequence.
        assert_eq!(out_values.len(), 3);
    }

    // R3 — a real rolling kernel (trailing mean of a size-2 window,
    // skipping nulls; null out where the window is entirely null) is
    // expressible view-only, in and out.
    #[test]
    fn r3_trailing_mean_skipping_nulls_is_expressible() {
        let values = [2.0, 4.0, 6.0, 8.0];
        let valid = [true, false, true, true]; // index 2 null
        let mut state = SpikeState::new();
        let chunk = "for i = 1, #v do\n\
                       local sum, n = 0.0, 0\n\
                       for j = math.max(1, i - 1), i do\n\
                         if v[j] ~= nil then sum = sum + v[j]; n = n + 1 end\n\
                       end\n\
                       if n == 0 then out[i] = nil else out[i] = sum / n end\n\
                     end";
        let (out_values, out_valid) = state
            .eval_into_output(chunk, &values, &valid, values.len())
            .expect("runs");

        // Rust reference over the same window rule.
        let mut ref_values = vec![0.0; values.len()];
        let mut ref_valid = vec![false; values.len()];
        for i in 0..values.len() {
            let lo = i.saturating_sub(1);
            let mut sum = 0.0;
            let mut n = 0;
            for j in lo..=i {
                if valid[j] {
                    sum += values[j];
                    n += 1;
                }
            }
            if n > 0 {
                ref_values[i] = sum / f64::from(n);
                ref_valid[i] = true;
            }
        }
        assert_eq!(out_valid, ref_valid);
        for i in 0..values.len() {
            if ref_valid[i] {
                assert!(
                    (out_values[i] - ref_values[i]).abs() < 1e-12,
                    "row {i}: {} vs {}",
                    out_values[i],
                    ref_values[i]
                );
            }
        }
    }

    // ---- Config C (smart sentinel) — the A-vs-C contrast ----

    // C-R1 — the headline difference from A: a naive sum over a null does
    // NOT crash; NULL propagates through the arithmetic and the whole sum
    // comes out NULL (soft poison, SQL/pd.NA-like). The guarded form uses
    // `~= NULL` and skips it. Contrast A, where the same naive sum errors.
    #[test]
    fn c_r1_naive_sum_propagates_to_null_guarded_skips() {
        let values = [1.0, 2.0, 3.0];
        let valid = [true, false, true];
        let mut state = SpikeStateC::new();

        let naive = state
            .eval_scalar_c(
                "local s = 0.0\nfor i = 1, #v do s = s + v[i] end\nreturn s",
                &values,
                &valid,
            )
            .expect("propagates, does not error");
        assert_eq!(naive, CScalar::Null, "null propagates through the sum");

        let guarded = state
            .eval_scalar_c(
                "local s = 0.0\nfor i = 1, #v do if v[i] ~= NULL then s = s + v[i] end end\nreturn s",
                &values,
                &valid,
            )
            .expect("runs");
        assert_eq!(guarded, CScalar::Num(4.0));
    }

    // C-R1b — NULL (sentinel, identity-comparable) and NaN (self-unequal
    // value) stay distinct under C too.
    #[test]
    fn c_r1b_nan_and_null_distinguishable() {
        let values = [f64::NAN, 5.0];
        let valid = [true, false];
        let mut state = SpikeStateC::new();
        let code = state
            .eval_scalar_c(
                "local nulls, nans = 0, 0\n\
                 for i = 1, #v do\n\
                   if v[i] == NULL then nulls = nulls + 1\n\
                   elseif v[i] ~= v[i] then nans = nans + 1 end\n\
                 end\n\
                 return nulls * 10 + nans",
                &values,
                &valid,
            )
            .expect("runs");
        assert_eq!(code, CScalar::Num(11.0));
    }

    // C — the sentinel propagates over the INTEGER subtype too, so the
    // i64-exactness constraint that decided Lua 5.4 is respected: a Lua
    // integer beyond 2^53 combined with NULL yields NULL, not a crash and
    // not a float coercion.
    #[test]
    fn c_sentinel_propagates_over_integers() {
        let mut state = SpikeStateC::new();
        let r = state
            .eval_scalar_c("return 9007199254740993 + NULL", &[], &[])
            .expect("propagates");
        assert_eq!(r, CScalar::Null);
    }

    // C-WART 1 — the irreducible Lua cost: the sentinel is userdata, hence
    // TRUTHY. `if v[i]` is TRUE for a NULL. This is the footgun no non-nil
    // sentinel escapes (only nil/false are falsy in Lua).
    #[test]
    fn c_wart_null_is_truthy() {
        let values = [0.0];
        let valid = [false]; // the one element is NULL
        let mut state = SpikeStateC::new();
        let r = state
            .eval_scalar_c("if v[1] then return 1 else return 0 end", &values, &valid)
            .expect("runs");
        assert_eq!(r, CScalar::Num(1.0), "NULL is truthy — the sentinel footgun");
    }

    // C-WART 2 — 3VL cannot propagate through relational operators: Lua
    // forces `<`/`<=` to a boolean, so we make comparing NULL a loud
    // error. Arithmetic propagates; comparison does not.
    #[test]
    fn c_wart_comparison_with_null_is_loud() {
        let values = [0.0];
        let valid = [false];
        let mut state = SpikeStateC::new();
        let err = state
            .eval_scalar_c("return v[1] < 5", &values, &valid)
            .expect_err("comparison with NULL cannot 3VL-propagate");
        assert!(
            err.contains("compare") || err.contains("attempt"),
            "expected a loud compare error, got: {err}"
        );
    }
}
