#!/bin/sh
# The push gate: every check CI runs on stable, in CI's order, judged by
# exit code. Green here is the evidence a push claims.
#
#   scripts/gate.sh            run every leg, stop at the first failure
#   scripts/gate.sh --keep-going   run every leg, report all failures
#
# Legs: fmt; clippy with all features and with none; build; the test
# suite with default features and with none (the embedder's library);
# rustdoc with warnings as errors, both legs; the shared library with the
# oracle hooks and the nine Python oracle scripts against it (PyArrow,
# DuckDB, and NumPy recompute independently and diff); the interpreter
# built with LUA_USE_APICHECK. The Miri, Lua-suite, and sanitizer jobs
# need nightly or a C toolchain and stay in .github/workflows/ci.yml.
#
# Needs: the stable toolchain CI uses (pick one with RUSTUP_TOOLCHAIN=…
# when the default differs), and Python 3 with pyarrow, duckdb, and
# numpy importable. Per-leg logs land in target/gate/.
set -u
cd "$(dirname "$0")/.." || exit 99
target=${CARGO_TARGET_DIR:-target}
logs=$target/gate
mkdir -p "$logs"
export CARGO_TERM_COLOR=never
keep_going=0
[ "${1:-}" = "--keep-going" ] && keep_going=1
failed=0
n=0

run() {
    n=$((n + 1))
    name=$1
    shift
    log=$logs/$(printf '%02d' "$n").log
    if "$@" >"$log" 2>&1; then
        echo "PASS $name"
    else
        rc=$?
        echo "FAIL $name (exit $rc) — $log"
        tail -40 "$log"
        failed=1
        [ "$keep_going" = 1 ] || finish
    fi
}

finish() {
    rm -rf tests/__pycache__
    if [ "$failed" = 0 ]; then
        echo "GATE PASSED"
        exit 0
    fi
    echo "GATE FAILED"
    exit 1
}

run "fmt" cargo fmt --all --check
run "clippy, all features" cargo clippy --all-targets --all-features -- -D warnings
run "clippy, no default features" cargo clippy --all-targets --no-default-features -- -D warnings
run "build" cargo build
run "test" cargo test
run "test, no default features" cargo test --no-default-features
run "doc" env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
run "doc, no default features" env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --no-default-features
run "python oracles importable" python3 -c "import pyarrow, duckdb, numpy"
run "build the oracle harness" cargo build --features oracle-harness
lib=$target/debug/libtallydb.so
for script in pyarrow_roundtrip m1_slice_oracle m2_mutation_oracle \
    m2_differential_oracle m2_lua_window_oracle m4_asof_oracle \
    m5_view_oracle m5_join_oracle m5_multifactor_oracle; do
    run "oracle $script" python3 "tests/$script.py" "$lib"
done
run "test with LUA_USE_APICHECK" cargo test --features apicheck
finish
