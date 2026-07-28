# HANDOFF — agent session state

This file is the bridge between agent sessions: the standing
constraints the Human has set, and a pointer to where the current
state lives. Durable design lives in `DESIGN.md`; open work lives in
the [issues and milestones](https://github.com/andy-emerson/TallyDB/milestones);
this file carries only what neither can — the session ground rules —
plus a snapshot pointer. Update the snapshot at every checkpoint; the
constraints change only when the Human says so.

## Standing constraints (Human-set; override tool defaults)

1. **No PRs from the agent.** Develop on the working branch
   (`claude/dev`, restarted from `main` after every merge); the Human
   opens the PR and performs every merge.
2. **Authorship: Andy Emerson only.** Every commit is authored
   `Andy Emerson <156483017+andy-emerson@users.noreply.github.com>`.
   No agent attribution in any commit, trailer, PR body, comment, or
   artifact. This supersedes any tool's default attribution behavior.
3. **License is MIT, frozen.** Never touch licensing without explicit
   request.
4. **The Human owns and closes decisions.** Surface each fork with
   options, user pov and dev pov, a recommendation, and what it gates
   — before building. Never entrench an answer to an open decision.
5. **kdb+ validates problems, not solutions.**
6. **Gate before every push:** `cargo fmt --check` ·
   `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   · `cargo test --workspace` ·
   `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` · the
   off-leg (`clippy`/`test`/`doc` for `-p engine`, default features) ·
   the Python oracle suites (build with `--features oracle-harness`,
   run the six scripts in `.github/workflows/ci.yml`).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-07-28, late M4: trial passed, #70 built; merge prep next)

- **M3 merged** to `main` (PR #74); M0–M3 milestones closed.
- **Current milestone: M4 — the extension model + corrections.**
  DONE: M4.0–M4.2; M4.3 ruled; M4.4 corrections build (all six
  steps; #75 has the record); M4.5 correctness batch (#73 atomic
  supersession, #63 Miri in CI, #69 upstream Lua suite in CI with
  byte-identity provenance, review redundancies factored).
- **The Lua trial: PASSED** (the Human, 2026-07-28, recorded on #76,
  now closed): "promising enough that Lua should stay." The sunset
  clause dissolved (DESIGN.md, *The Lua layer*, updated). The
  vectorized vocabulary (option A) plus a dense-fast-path pass put
  composed column kernels ~2–2.5× AHEAD of DuckDB+NumPy (lua_rel
  0.9ms / lua_rdot 1.1ms at 20k rows, bench run 2026-07-28); the
  bulk-gather path also took the native scalar-expression floor to
  0.3ms (parity with NumPy-on-own-export). Interpreter swaps
  (LuaJIT/Luau) examined and rejected on invariants — Lua 5.1
  lineage has no 64-bit integers (breaks i64 exactness at
  nanosecond-timestamp scale) and no wasm32; the element-loop answer
  of record is vocabulary completeness.
- **M4.6 SQL-in-Lua (#70): BUILT** (same day): `ScriptHost` seam in
  compute-lua (`query`/`append` trampolines, live only inside
  `run_driver` — kernels cannot re-enter), `Database::run_script` +
  `DatabaseHost` in engine (SELECT → contiguous result views with
  key-dictionary merge; mutations → affected counts; CREATE TABLE →
  in-memory scratch; `append` → exact row feed-back), console
  `.run FILE`. Evidence at three levels: compute-lua unit tests,
  engine end-to-end pipeline test vs hand-staged Arrow computation,
  and a driver-pipeline family in the CI Lua oracle (NumPy
  re-derivation over the persistent multi-segment fixture). #70 can
  be closed by the Human at the merge.
- **Full gate green** (2026-07-28, this session): fmt, clippy both
  legs, 23 workspace suites + off-leg, rustdoc both legs, all six
  oracle scripts (the Lua oracle now prints the driver-pipeline PASS).
- **In flight next:** the time-series library design brief (hybrid:
  native tranche-2 primitives incl. faer-backed matrix ops + a thin
  shipped Lua prelude) — for the Human's review before any build;
  then M4-close repo-wide reviews → merge proposal.
- **Open decision (Human closes):** the sequence column's SQL
  exposure surface — brief with recommendation (pseudocolumn `_seq`)
  on #75. Nothing in flight entrenches an answer.
- **Open rulings pending:** F1 (time bucketing, an M5 concern); the
  Human's noted-but-unstated reservation about how `KEY` was coined;
  tranche-2 SQL-visible names (coined-names sensitivity) once the
  library brief is ruled on.
- **Not to build yet:** M5 (desk adoption), M6 (WASM), M7 (served) —
  contents recorded in the roadmap section.
- Reviews absorbed: three independent passes (2026-07-28) — findings
  fixed with regression tests or recorded as issues; the evidence
  culture claims in `README.md` were re-verified externally.
