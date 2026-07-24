//! Hand-rolled declarations for the slice of the Lua 5.4 C API this
//! crate uses (issue #5's ruling: no binding dependency; the spec is
//! the Lua 5.4 reference manual, and the vendored interpreter built
//! with `LUA_USE_APICHECK` asserts on misuse in test builds).
//!
//! Everything here is `unsafe` and `pub(crate)`: the safety story —
//! every entry through `lua_pcall`, no Lua error raised across Rust
//! frames with pending destructors, `catch_unwind` at the boundary —
//! lives in the safe wrapper, not here. Keep this module declarations-
//! only; the count of functions below is the "surface balloons" reopen
//! signal recorded in the decision (~two dozen is the budget).

#![allow(non_camel_case_types)]
// Declarations land ahead of the safe wrapper that consumes them; the
// allow comes off when the wrapper does.
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void};

/// An opaque Lua state.
#[repr(C)]
pub(crate) struct lua_State {
    _opaque: [u8; 0],
}

/// `lua_Number`: Lua 5.4's float subtype (`double`).
pub(crate) type lua_Number = f64;
/// `lua_Integer`: Lua 5.4's integer subtype (`long long`) — exactly
/// the engine's `i64`, the alignment that decided the interpreter.
pub(crate) type lua_Integer = i64;
/// A C function callable from Lua.
pub(crate) type lua_CFunction = unsafe extern "C" fn(*mut lua_State) -> c_int;

/// `lua_pcall` status: success.
pub(crate) const LUA_OK: c_int = 0;
/// Pseudo-index of the registry table.
pub(crate) const LUA_REGISTRYINDEX: c_int = -1_001_000;
/// `lua_type` tags used by the wrapper.
pub(crate) const LUA_TNIL: c_int = 0;
pub(crate) const LUA_TNUMBER: c_int = 3;

unsafe extern "C" {
    // State lifecycle.
    pub(crate) fn luaL_newstate() -> *mut lua_State;
    pub(crate) fn lua_close(L: *mut lua_State);

    // Loading and protected calls — the only way anything runs.
    pub(crate) fn luaL_loadbufferx(
        L: *mut lua_State,
        buff: *const c_char,
        sz: usize,
        name: *const c_char,
        mode: *const c_char,
    ) -> c_int;
    pub(crate) fn lua_pcallk(
        L: *mut lua_State,
        nargs: c_int,
        nresults: c_int,
        errfunc: c_int,
        ctx: isize,
        k: *const c_void,
    ) -> c_int;

    // Stack discipline.
    pub(crate) fn lua_gettop(L: *mut lua_State) -> c_int;
    pub(crate) fn lua_settop(L: *mut lua_State, idx: c_int);
    pub(crate) fn lua_type(L: *mut lua_State, idx: c_int) -> c_int;

    // Reads.
    pub(crate) fn lua_tonumberx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Number;
    pub(crate) fn lua_tointegerx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Integer;
    pub(crate) fn lua_isinteger(L: *mut lua_State, idx: c_int) -> c_int;
    pub(crate) fn lua_tolstring(L: *mut lua_State, idx: c_int, len: *mut usize) -> *const c_char;
    pub(crate) fn lua_touserdata(L: *mut lua_State, idx: c_int) -> *mut c_void;

    // Pushes.
    pub(crate) fn lua_pushnil(L: *mut lua_State);
    pub(crate) fn lua_pushnumber(L: *mut lua_State, n: lua_Number);
    pub(crate) fn lua_pushinteger(L: *mut lua_State, n: lua_Integer);
    pub(crate) fn lua_pushlstring(L: *mut lua_State, s: *const c_char, len: usize)
        -> *const c_char;
    pub(crate) fn lua_pushcclosure(L: *mut lua_State, f: lua_CFunction, n: c_int);
    pub(crate) fn lua_error(L: *mut lua_State) -> c_int;

    // Userdata and metatables — the zero-copy view mechanism.
    pub(crate) fn lua_newuserdatauv(L: *mut lua_State, size: usize, nuvalue: c_int) -> *mut c_void;
    pub(crate) fn luaL_newmetatable(L: *mut lua_State, tname: *const c_char) -> c_int;
    pub(crate) fn luaL_setmetatable(L: *mut lua_State, tname: *const c_char);
    pub(crate) fn luaL_testudata(L: *mut lua_State, ud: c_int, tname: *const c_char)
        -> *mut c_void;
    pub(crate) fn lua_setfield(L: *mut lua_State, idx: c_int, k: *const c_char);
    pub(crate) fn lua_getfield(L: *mut lua_State, idx: c_int, k: *const c_char) -> c_int;
    pub(crate) fn lua_setglobal(L: *mut lua_State, name: *const c_char);

    // Curated standard libraries (opened individually — there is no
    // luaL_openlibs call anywhere in this crate, by policy).
    pub(crate) fn luaL_requiref(
        L: *mut lua_State,
        modname: *const c_char,
        openf: lua_CFunction,
        glb: c_int,
    );
    pub(crate) fn luaopen_base(L: *mut lua_State) -> c_int;
    pub(crate) fn luaopen_math(L: *mut lua_State) -> c_int;
    pub(crate) fn luaopen_string(L: *mut lua_State) -> c_int;
    pub(crate) fn luaopen_table(L: *mut lua_State) -> c_int;
}

/// `lua_pcall` as the 5.4 macro expands it.
///
/// # Safety
/// Same contract as `lua_pcallk` with no continuation.
pub(crate) unsafe fn lua_pcall(
    L: *mut lua_State,
    nargs: c_int,
    nresults: c_int,
    errfunc: c_int,
) -> c_int {
    unsafe { lua_pcallk(L, nargs, nresults, errfunc, 0, std::ptr::null()) }
}
