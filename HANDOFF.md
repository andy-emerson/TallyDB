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
   Python oracle suites (build with `--features oracle-harness`, run
   the five scripts in `.github/workflows/ci.yml`).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-07-28, mid-M4: M4.0–M4.3 done, M4.4 next)

- **M3 merged** to `main` (PR #74); M0–M3 milestones closed.
- **Current milestone: M4 — the extension model + corrections.** Plan
  of record in `DESIGN.md`, *The roadmap beyond native GA*. DONE:
  M4.0 (trait public, register_window, doctest), M4.1 (lua feature
  gate, CI both legs incl. off-leg rustdoc), M4.2 (vocabulary
  invariant registry-driven + tested; promotion test; the vectorized
  column-function slot — ColumnFunction trait + register_lua_scalar +
  .luascalar — closed #53; compose-don't-loop idiom recorded). M4.3
  ruled and recorded (sequence column default-on, unbounded horizon,
  bare AS OF carried on FOR SYSTEM_TIME AS OF, one-word ASOF JOIN).
  NEXT: **M4.4 the corrections build** (sequence column
  virtual-until-divergence, retaining compaction with history
  segments, knowledge mask, AS OF clause; manifest revision designed
  with F3's zone-map sections reserved; oracle = DuckDB re-deriving
  as-of answers over an explicit history table) → M4.5 correctness
  batch (#73 atomic mutation commit record, #63 Miri CI, #69 upstream
  Lua suite, review-noted redundancies) → **the Lua trial** (Agent
  brings the evidence brief; the Human alone rules; see the sunset
  clause) → M4.6 SQL-in-Lua (#70) only on a pass.
- **Open rulings pending:** F1 (time bucketing, an M5 concern); the
  Human's noted-but-unstated reservation about how `KEY` was coined.
  Issue #75 tracks the corrections build; #73/#63/#69 are M4's
  correctness batch; #70 is conditional on the Lua trial.
- **Not to build yet:** M5 (desk adoption), M6 (WASM), M7 (served) —
  contents recorded in the same roadmap section.
- Reviews absorbed: three independent passes (2026-07-28) — findings
  fixed with regression tests or recorded as issues; the evidence
  culture claims in `README.md` were re-verified externally.
