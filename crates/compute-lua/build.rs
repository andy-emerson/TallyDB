//! Compiles the vendored canonical PUC Lua 5.4 sources into this crate
//! (issue #5's ruling: unmodified upstream, no fork, hand-rolled
//! bindings). Compiled as C — Lua's error handling is `longjmp`-based
//! in a C build, and the binding's `lua_pcall` discipline is written
//! against exactly that; do not switch this to a C++ build.
//!
//! No platform feature macros (`LUA_USE_LINUX`/`LUA_USE_POSIX`) are
//! defined: the ANSI build has no dynamic loading, which enforces the
//! settled no-C-extensions policy (`package.loadlib`) at compile time
//! rather than by configuration.
//!
//! The `apicheck` feature compiles the interpreter with
//! `LUA_USE_APICHECK`, turning C API misuse into assertions — the
//! binding-discipline oracle used by test builds and CI (DESIGN.md,
//! *The Lua layer*).

fn main() {
    let sources = std::fs::read_dir("vendor/lua-5.4.7")
        .expect("vendored Lua sources present")
        .filter_map(|entry| {
            let path = entry.expect("readable dir entry").path();
            (path.extension().is_some_and(|ext| ext == "c")).then_some(path)
        })
        .collect::<Vec<_>>();
    assert!(
        sources.len() > 30,
        "vendored Lua source set looks incomplete: {} files",
        sources.len()
    );
    let mut build = cc::Build::new();
    build.files(&sources).include("vendor/lua-5.4.7");
    if std::env::var_os("CARGO_FEATURE_APICHECK").is_some() {
        build.define("LUA_USE_APICHECK", None);
    }
    build.compile("lua54");
    println!("cargo:rerun-if-changed=vendor/lua-5.4.7");
}
