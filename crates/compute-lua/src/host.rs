//! Host functions: engine-side compute callable from scripts over the
//! same zero-copy views.
//!
//! This is the seam the curated `compute-blas` and engine ops reach
//! scripts through (the engine registers them into every kernel
//! state): a [`HostFunction`] receives its column arguments as plain
//! `&[f64]` slices pointing **directly at the engine buffers the views
//! wrap** — sharing buffers, not copying between them — and returns one
//! value, or `None` where the operation is undefined (which scripts see
//! as `NULL`, the same convention SQL windows use).
//!
//! This crate stays backend-agnostic: it defines the seam and the
//! trampoline; *which* functions exist is the embedder's business
//! (`engine` ties compute to SQL and installs the curated set).
//!
//! ## Boundary discipline
//!
//! The trampoline follows the crate's C-boundary rules: argument
//! checks raise with static messages and `Copy` locals only; the
//! embedder's Rust code runs under `catch_unwind` so a panic never
//! unwinds into C; a dynamic error message is copied into Lua and
//! dropped *before* the raise, so `lua_error`'s `longjmp` crosses no
//! live destructor.

use crate::ffi;
use crate::values;
use std::ffi::{c_char, c_int, c_void, CStr};

/// The most view arguments a host function may take — a trampoline
/// stack-buffer bound, far above any curated op's arity.
pub(crate) const MAX_ARGS: usize = 8;

/// One engine-side operation callable from scripts. Arguments arrive as
/// dense (non-null) `f64` slices over the live engine buffers; the
/// result is one value, or `None` where the operation is undefined
/// (scripts see `NULL`). `Send` because the function moves with its
/// (Send) `LuaState`.
pub trait HostFunction: Send {
    /// Number of column-view arguments the function takes.
    fn arity(&self) -> usize;

    /// Evaluates the operation. `args.len()` equals `arity()`.
    fn call(&self, args: &[&[f64]]) -> Result<Option<f64>, String>;
}

/// The stable box a registered function lives in; the trampoline
/// reaches it through a lightuserdata upvalue, so its address must
/// survive moves of the `LuaState` struct. Freed by `LuaState::drop`.
pub(crate) struct HostSlot(pub(crate) Box<dyn HostFunction>);

/// Installs `slot`'s function as the global `name`.
///
/// # Safety
/// `raw` must be a valid state with an empty stack; `slot` must stay
/// valid for the state's whole life.
pub(crate) unsafe fn install(raw: *mut ffi::lua_State, name: &CStr, slot: *mut HostSlot) {
    unsafe {
        ffi::lua_pushlightuserdata(raw, slot.cast::<c_void>());
        ffi::lua_pushcclosure(raw, host_call, 1);
        ffi::lua_setglobal(raw, name.as_ptr());
    }
}

/// The trampoline every host function runs through.
unsafe extern "C" fn host_call(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let slot = ffi::lua_touserdata(state, ffi::lua_upvalueindex(1)).cast::<HostSlot>();
        let argc = ffi::lua_gettop(state);
        let arity = (*slot).0.arity();
        if argc as usize != arity {
            return values::raise(state, c"wrong number of view arguments");
        }
        // Collect the raw slices first — every raise happens here, with
        // only Copy locals live.
        let mut raw_args = [(std::ptr::null::<f64>(), 0usize); MAX_ARGS];
        for (index, raw_arg) in raw_args.iter_mut().enumerate().take(arity) {
            match values::f64_view_slice(state, (index + 1) as c_int) {
                Ok(pair) => *raw_arg = pair,
                Err(message) => return values::raise(state, message),
            }
        }
        let mut slices: [&[f64]; MAX_ARGS] = [&[]; MAX_ARGS];
        for (slice, &(pointer, len)) in slices.iter_mut().zip(&raw_args).take(arity) {
            // A zero-length view carries its buffer's (possibly dangling
            // but aligned) pointer; from_raw_parts is fine either way.
            *slice = std::slice::from_raw_parts(pointer, len);
        }
        // The embedder's code: contain panics, never unwind into C.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (*slot).0.call(&slices[..arity])
        }));
        match outcome {
            Ok(Ok(Some(value))) => {
                ffi::lua_pushnumber(state, value);
                1
            }
            Ok(Ok(None)) => {
                values::push_null(state);
                1
            }
            Ok(Err(message)) => {
                // Copy the dynamic message into Lua, drop it, then
                // raise: the longjmp crosses no live destructor. (The
                // push's only failure mode is OOM, which would leak the
                // String — bounded, never unsound.)
                ffi::lua_pushlstring(state, message.as_ptr().cast::<c_char>(), message.len());
                drop(message);
                ffi::lua_error(state)
            }
            Err(payload) => {
                // Drop the caught panic payload before raising — the
                // longjmp must cross no live destructor (ASan caught
                // the Err(_) form leaking exactly this box).
                drop(payload);
                values::raise(state, c"host function panicked")
            }
        }
    }
}
