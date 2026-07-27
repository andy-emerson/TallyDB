//! Script observability: `log(...)`, the one host-routed diagnostic
//! (decision record 2026-07-26; DESIGN.md, *The Lua layer*).
//!
//! `log(...)` replaces Lua's `print` — removed not for the string
//! invariant (its text never becomes a column) but because its
//! destination, the process's stdout, is not an embeddable library's to
//! own and is uncapturable. (`warn` goes with it, for its stderr, by
//! the same principle.) `log` routes instead to an embedder-installed
//! [`LogSink`]: a shell wires it to stderr, a library embedder to its
//! own logger, a headless embedding leaves it out — **off by default**,
//! a no-op until a sink is installed.
//!
//! It is a **pure side-channel**: `log` returns nothing, so it cannot
//! feed a result — a diagnostic that alters the answer is not a
//! diagnostic. Arguments are flattened to text and tab-joined; a view
//! logs as a summary (`f64 view, len 4096`), never its contents — a
//! diagnostic, not a buffer dump. Surface and sink are both flat, one
//! severity; a level parameter would be permanently degenerate for a
//! script-only sink (the additive escape hatches are recorded in the
//! decision).

use crate::ffi;
use crate::values;
use std::ffi::{c_char, c_int, c_void};

/// The embedder-installed destination for scripts' `log(...)` output.
/// Flat and single-severity by decision; implement it over a logger,
/// a capture buffer, or stderr as the embedding demands. `Send` because
/// the sink moves with its (Send) `LuaState`.
pub trait LogSink: Send {
    /// One message, already flattened to text (arguments tab-joined).
    fn log(&self, message: &str);
}

/// The slot the `log` closure reads through. It is boxed by `LuaState`
/// so its address is stable across moves of the state struct — the
/// closure's upvalue pointer must survive them.
pub(crate) struct SinkSlot(pub(crate) Option<Box<dyn LogSink>>);

/// Installs the `log` global (closing over `slot`) and removes `print`
/// and `warn` from the base library.
///
/// # Safety
/// `raw` must be a valid state with the base library open and an empty
/// stack; `slot` must stay valid for the state's whole life.
pub(crate) unsafe fn install(raw: *mut ffi::lua_State, slot: *mut SinkSlot) {
    unsafe {
        ffi::lua_pushlightuserdata(raw, slot.cast::<c_void>());
        ffi::lua_pushcclosure(raw, lua_log, 1);
        ffi::lua_setglobal(raw, c"log".as_ptr());
        // print writes to stdout, warn to stderr — process streams an
        // embedded library does not own. log() is the replacement.
        ffi::lua_pushnil(raw);
        ffi::lua_setglobal(raw, c"print".as_ptr());
        ffi::lua_pushnil(raw);
        ffi::lua_setglobal(raw, c"warn".as_ptr());
    }
}

/// The `log(...)` C function. Two phases keep the module discipline:
/// phase 1 does every call that can raise (string conversions) with
/// only `Copy` locals live; phase 2 raises nothing while Rust
/// allocations (the assembled message) are alive, and the embedder's
/// sink runs under `catch_unwind` so a panicking sink never crosses
/// into C.
unsafe extern "C" fn lua_log(state: *mut ffi::lua_State) -> c_int {
    unsafe {
        let slot = ffi::lua_touserdata(state, ffi::lua_upvalueindex(1)).cast::<SinkSlot>();
        let argc = ffi::lua_gettop(state);
        // Phase 1: push each argument's string form above the arguments
        // — views as summaries, everything else via __tostring.
        for index in 1..=argc {
            match values::view_summary(state, index) {
                Some((buffer, len)) => {
                    ffi::lua_pushlstring(state, buffer.as_ptr().cast::<c_char>(), len);
                }
                None => {
                    let mut len = 0usize;
                    ffi::luaL_tolstring(state, index, &mut len);
                }
            }
        }
        // Phase 2: assemble (the pushed values are all strings already,
        // so lua_tolstring converts nothing and raises nothing).
        let mut message = String::new();
        for position in 0..argc {
            let mut len = 0usize;
            let text = ffi::lua_tolstring(state, argc + 1 + position, &mut len);
            if position > 0 {
                message.push('\t');
            }
            if !text.is_null() {
                let bytes = std::slice::from_raw_parts(text.cast::<u8>(), len);
                message.push_str(&String::from_utf8_lossy(bytes));
            }
        }
        if let Some(sink) = &(*slot).0 {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink.log(&message)));
        }
        0
    }
}
