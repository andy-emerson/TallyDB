# Vendored Lua sources

`lua-5.4.7/` is **canonical PUC Lua 5.4.7**, unmodified — the MIT
license is embedded at the end of `lua.h` (Copyright 1994-2024 Lua.org,
PUC-Rio). Per the interpreter ruling (issue #5, DESIGN.md *The Lua
layer*): the canonical sources compiled into the engine, no fork, no
patches. Do not edit anything under `lua-5.4.7/`; a version bump
replaces the directory whole.

Obtained 2026-07-24 from the `lua-src` crate (v547.1.0, MIT), which
packages the pristine upstream release tree minus the standalone
binaries' entry points (`lua.c`, `luac.c`) — exactly the embedding
set. Direct download from lua.org was unavailable under this
environment's network policy; the sources verify as 5.4.7 via
`LUA_VERSION_RELEASE` and carry the upstream copyright notice intact.
The `lua-src` crate itself is **not** a dependency — only these files
are, compiled by this crate's `build.rs`.
