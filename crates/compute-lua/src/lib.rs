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

// TODO: Lua backend trait (native vendored-5.4 implementation first;
//       first act is the zero-copy userdata-view spike, pointer-verified)
// TODO: batch calling convention: hand a whole column/window buffer to
//       a Lua chunk in one call
// TODO: expose compute-blas and compute-lapack ops as callable Lua
//       functions, sharing buffers (no copy)
// TODO: pure-Lua module loader (package.path-style); explicitly do NOT
//       wire up package.loadlib / C extension loading
