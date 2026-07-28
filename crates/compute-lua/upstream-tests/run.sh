#!/bin/sh
# The official Lua 5.4.7 test suite over OUR vendored interpreter
# (issue #69) — the upstream conformance evidence for the Lua trial.
#
# Builds a standalone test interpreter from the exact sources the
# engine embeds (../vendor/lua-5.4.7, byte-identical to the upstream
# v5.4.7 tag — see PROVENANCE.md) plus the release's own test
# instrumentation: `ltests.c` compiled in via LUA_USER_H, which turns
# on the suite's internal torture checks (a tracing allocator with
# controlled failure injection, stack/GC consistency assertions, the
# `T` test library). Then runs `all.lua` — internal tests included —
# in a scratch copy of `testes/`, so the vendored tree stays pristine.
#
# Usage: run.sh [build-dir]   (default: ./build, gitignored)
set -e
here=$(cd "$(dirname "$0")" && pwd)
vendor=$here/../vendor/lua-5.4.7
build=${1:-$here/build}
: "${CC:=cc}"

mkdir -p "$build"
# -I"$here" first so LUA_USER_H='"ltests.h"' resolves to the suite's
# header; the interpreter sources come from the vendored tree only.
# The flags mirror upstream's `make linux-readline` test build:
# -Wl,-E exports the interpreter's symbols so the suite's dynamic
# libraries (testes/libs) resolve the Lua API against the host binary,
# and readline is required — main.lua's interactive-session transcripts
# assume readline's input echo.
$CC -O1 -Wall -DLUA_USER_H='"ltests.h"' -DLUA_USE_LINUX -DLUA_USE_READLINE \
    -I"$here" -I"$vendor" \
    -o "$build/luatest" "$vendor"/*.c "$here/ltests.c" "$here/lua.c" \
    -Wl,-E -lm -ldl -lreadline

# The suite runs in place and writes artifacts (compiled libs, temp
# files): give it a disposable copy.
rm -rf "$build/testes"
cp -R "$here/testes" "$build/testes"
cd "$build/testes/libs"
# The libs makefile's rules, against the vendored headers.
$CC -Wall -std=gnu99 -O2 -I"$vendor" -fPIC -shared -o lib1.so lib1.c
$CC -Wall -std=gnu99 -O2 -I"$vendor" -fPIC -shared -o lib11.so lib11.c
$CC -Wall -std=gnu99 -O2 -I"$vendor" -fPIC -shared -o lib2.so lib2.c
$CC -Wall -std=gnu99 -O2 -I"$vendor" -fPIC -shared -o lib21.so lib21.c
$CC -Wall -std=gnu99 -O2 -I"$vendor" -fPIC -shared -o lib2-v2.so lib22.c
touch all

cd "$build/testes"
exec "$build/luatest" all.lua
