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
//! ## LuaJIT + BLAS/LAPACK: this is real, not hand-waved
//! LuaJIT's FFI lets Lua declare a C function's signature and call
//! directly into a linked C library with near-zero overhead — no
//! hand-written binding layer, numeric arrays passed as raw pointers into
//! the *same memory* the query engine already holds (arrow-lite buffers).
//! This is exactly how the original Torch (pre-PyTorch) worked: LuaJIT +
//! BLAS-backed tensors over FFI. The ops in `compute-blas`
//! (multiplication-class) and `compute-lapack` (curated solves/
//! decompositions) should be callable from Lua through this same mechanism,
//! sharing buffers, not copying between them.
//!
//! ## Backend split
//! - **Native (current milestone):** LuaJIT, linked as-is via FFI. No
//!   fork, no rebuild — this is a mature, narrow, embedding-oriented
//!   dependency, take it whole. *How* Rust binds to it — the `mlua` crate
//!   vs. hand-rolled bindings to the (frozen) Lua 5.1 C API — is a
//!   deferred decision tracked in issues; the load-bearing criterion is
//!   zero-copy buffer hand-off, and the arrow-lite hand-roll precedent
//!   could legitimately cut either way here (the safety burden — longjmp
//!   across FFI, GC interaction — is heavier than a memory layout).
//!   Prototype the buffer hand-off in mlua before choosing. The trait
//!   shape, batch convention, and module loader are binding-agnostic and
//!   don't wait on this.
//! - **WASM (future, not current milestone):** `lua.wasm`
//!   (github.com/andy-emerson/lua-wasi) — LuaJIT cannot run in WASM at
//!   all (no runtime codegen in the sandbox); `lua.wasm` provides a stock
//!   interpreter plus an AOT path for code known at build time. Both
//!   backends sit behind the same trait; nothing above this crate should
//!   need to know which one is active.
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
//! root README / project history for the reasoning.

// TODO: Lua backend trait (native LuaJIT-via-FFI implementation first)
// TODO: batch calling convention: hand a whole column/window buffer to
//       a Lua chunk in one call
// TODO: expose compute-blas and compute-lapack ops as callable Lua
//       functions, sharing buffers (no copy)
// TODO: pure-Lua module loader (package.path-style); explicitly do NOT
//       wire up package.loadlib / C extension loading
