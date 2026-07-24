//! `compute-lua` — embedded Lua scripting, callable from SQL.
//!
//! ## The one rule that matters most in this crate: batch, not per-row
//! Scripts must be handed a whole column (or window) as a single array
//! per call, and iterate over it themselves inside Lua. Do NOT design an
//! API that calls into Lua once per row — that throws away interpreter
//! call overhead savings and defeats the entire point of pairing Lua with
//! a columnar, vectorized storage engine. If the public API of this crate
//! makes it easy to accidentally call per-row, that's a design bug, fix
//! the API shape, not the caller's usage.
//!
//! ## Zero-copy buffer views: this is real, not hand-waved
//! Scripts reach the engine's buffers through userdata views: the
//! userdata wraps the live arrow-lite buffer pointer and its accessors
//! are implemented on the Rust side, so no bytes are copied — access is
//! zero-copy, though each element read is a metamethod dispatch, not a
//! compiled raw load. The ops in `compute-blas` (multiplication-class)
//! and `compute-lapack` (curated solves/decompositions) are exposed to
//! scripts as registered functions over those same views, sharing
//! buffers, not copying between them. Lua 5.4's integer/float number
//! subtypes match the engine's `i64`/`f64` column pair exactly, so
//! scalars cross the boundary without losing exactness. Kernels that
//! prove hot get *promoted to curated native ops* (the `regr_slope` /
//! `covar_pop` / `corr` / `eigen_max` pattern) — interpreter speed is a
//! comfort here, not a foundation.
//!
//! ## Backend split
//! - **Native (current milestone):** canonical PUC Lua 5.4, vendored —
//!   the unmodified upstream sources compiled into the engine, with
//!   hand-rolled thin bindings to the 5.4 C API (settled 2026-07-24; the
//!   full decision record, including LuaJIT and `mlua` as rejected
//!   alternatives with reopen conditions, lives in DESIGN.md, *The Lua
//!   layer*). Binding discipline is verified with Lua's own enforcement:
//!   `LUA_USE_APICHECK` test builds, `ltests.c` GC/allocation torture,
//!   the official Lua test suite over the vendored build, and
//!   ASan/UBSan — no binding dependency, shipped or dev.
//! - **WASM (future, not current milestone):** `lua.wasm`
//!   (github.com/andy-emerson/lua-wasi) — also Lua 5.4, so both targets
//!   share one language semantics and one C API. Both backends sit
//!   behind the same trait; nothing above this crate should need to know
//!   which one is active.
//!
//! ## Pure-Lua libraries: yes. Compiled C extensions: no.
//! Plain `.lua` source libraries work with no special handling — they're
//! just more Lua code the interpreter runs. Compiled C extensions
//! (LuaRocks packages with a `.so`/`.dll` component, loaded via
//! `package.loadlib`) are explicitly NOT supported: real attack-surface
//! and stability cost for an embedded database process, and it cuts
//! against the curated-not-general philosophy of this whole project. This
//! is also structurally true for the WASM backend regardless of policy —
//! WASM's sandbox can't do `dlopen`-style dynamic loading at all. Do not
//! add `package.loadlib`/C-extension support later without revisiting
//! this decision explicitly and deliberately.
//!
//! ## Do NOT build here
//! No autodiff, no general tensor/NN framework (i.e. no "build our own
//! Torch"). If a real, specific, repeatedly-requested need shows up later,
//! it gets a narrow, scoped addition — not a new paradigm bolted on. See
//! DESIGN.md for the reasoning.

mod ffi;
mod state;

pub use state::{LuaState, ViewArg};

// TODO: Lua backend trait (native vendored-5.4 implementation first;
//       first act is the zero-copy userdata-view spike, pointer-verified)
// TODO: surface the standard-library set as a decision before this
//       crate's public API lands: base/math/string/table are opened
//       today; io/os/debug/package are not (package is unlinkable in
//       the ANSI build). Expanding the set is user-visible script
//       semantics — the hygiene tripwire applies.
// TODO: batch calling convention: hand a whole column/window buffer to
//       a Lua chunk in one call
// TODO: expose compute-blas and compute-lapack ops as callable Lua
//       functions, sharing buffers (no copy)
// TODO: pure-Lua module loader (package.path-style); explicitly do NOT
//       wire up package.loadlib / C extension loading

#[cfg(test)]
mod smoke_tests {
    //! Link-level proof that the vendored interpreter runs under the
    //! binding discipline: every entry through `lua_pcall`, stack
    //! balance asserted around every interaction, and Lua 5.4's
    //! integer subtype carrying an `i64` beyond 2^53 exactly — the
    //! alignment that decided the interpreter (issue #5).

    use crate::ffi;
    use std::ffi::c_int;

    /// Runs `chunk`, expecting `results` values back; returns Ok(()) or
    /// the Lua error message. Stack is balanced on both paths.
    unsafe fn run(state: *mut ffi::lua_State, chunk: &str, results: c_int) -> Result<(), String> {
        unsafe {
            let status = ffi::luaL_loadbufferx(
                state,
                chunk.as_ptr().cast(),
                chunk.len(),
                c"chunk".as_ptr(),
                c"t".as_ptr(), // text only: no binary chunks, ever
            );
            if status != ffi::LUA_OK {
                let message = pop_error(state);
                return Err(message);
            }
            if ffi::lua_pcall(state, 0, results, 0) != ffi::LUA_OK {
                return Err(pop_error(state));
            }
            Ok(())
        }
    }

    unsafe fn pop_error(state: *mut ffi::lua_State) -> String {
        unsafe {
            let mut len = 0usize;
            let text = ffi::lua_tolstring(state, -1, &mut len);
            let message = if text.is_null() {
                "error object is not a string".to_owned()
            } else {
                String::from_utf8_lossy(std::slice::from_raw_parts(text.cast(), len)).into_owned()
            };
            ffi::lua_settop(state, -2); // pop the error object
            message
        }
    }

    #[test]
    fn interpreter_runs_and_i64_survives_beyond_2_pow_53() {
        unsafe {
            let state = ffi::luaL_newstate();
            assert!(!state.is_null());
            ffi::luaL_requiref(state, c"_G".as_ptr(), ffi::luaopen_base, 1);
            ffi::luaL_requiref(state, c"math".as_ptr(), ffi::luaopen_math, 1);
            ffi::lua_settop(state, 0);
            assert_eq!(ffi::lua_gettop(state), 0);

            // 2^53 + 1 is unrepresentable in f64; only a true integer
            // subtype can return it unchanged.
            run(state, "return 9007199254740993 + 0", 1).unwrap();
            assert_eq!(ffi::lua_isinteger(state, -1), 1);
            let mut ok = 0;
            assert_eq!(
                ffi::lua_tointegerx(state, -1, &mut ok),
                9_007_199_254_740_993_i64
            );
            assert_eq!(ok, 1);
            ffi::lua_settop(state, 0);

            // Floats stay floats: math.type distinguishes the subtypes.
            run(state, "return math.type(1.5), math.type(1)", 2).unwrap();
            let mut len = 0usize;
            let float_tag = ffi::lua_tolstring(state, -2, &mut len);
            assert_eq!(
                std::slice::from_raw_parts(float_tag.cast::<u8>(), len),
                b"float"
            );
            ffi::lua_settop(state, 0);

            // Errors come back through pcall as values, never a longjmp
            // across this frame; the stack stays balanced.
            let error = run(state, "error('deliberate')", 0).unwrap_err();
            assert!(error.contains("deliberate"), "{error}");
            assert_eq!(ffi::lua_gettop(state), 0);

            // A syntax error is caught at load, same discipline.
            let error = run(state, "return ((", 0).unwrap_err();
            assert!(!error.is_empty());
            assert_eq!(ffi::lua_gettop(state), 0);

            ffi::lua_close(state);
        }
    }
}
