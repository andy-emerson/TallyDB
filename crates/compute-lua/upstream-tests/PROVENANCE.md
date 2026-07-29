# Vendored Lua test suite

`testes/`, `ltests.c`, `ltests.h`, and `lua.c` are the official **Lua
5.4.7 test suite and test scaffolding**, unmodified, from the upstream
release. Do not edit anything here except `run.sh` (ours); a version
bump replaces the upstream files whole, together with
`../vendor/lua-5.4.7`.

Obtained 2026-07-28 via the Go module proxy (`proxy.golang.org`, the
one mirror reachable under this environment's network policy), which
serves the upstream repository at its release tag:

- module zip: `github.com/lua/lua@v5.4.7+incompatible`
- origin recorded by the proxy: VCS `git`,
  URL `https://github.com/lua/lua`, ref `refs/tags/v5.4.7`,
  commit `1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3`
- integrity: the zip's dirhash was recomputed locally and matches the
  Go checksum database entry
  (`sum.golang.org`: `h1:fXCw+ooWUPsOCN2hPNjzpb6eRUi7gXsKZo/Cz87ICJo=`)

The binding to our interpreter is stronger than the download chain:
at extraction, **all 59 `.c`/`.h` files under `../vendor/lua-5.4.7`
were byte-identical** to the tagged tree this suite came from — the
tests provably belong to exactly the sources the engine compiles.

`run.sh` builds a standalone test interpreter from the vendored
sources plus `ltests.c` (compiled in via `LUA_USER_H`, enabling the
suite's internal torture checks: a tracing allocator with controlled
failure injection, stack/GC consistency assertions, the `T` library)
and runs `testes/all.lua` in a scratch copy. It mirrors upstream's
`make linux-readline` test configuration; readline is required because
`main.lua`'s interactive-session transcripts assume readline's input
echo. CI runs this on every change (the `lua-suite` job).

License: MIT, the same notice embedded in `lua.h` (Copyright
1994-2024 Lua.org, PUC-Rio).
