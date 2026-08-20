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
   run the nine oracle scripts under `crates/*/tests/`
   — `m2_compute_latency_bench.py` is a benchmark, not an oracle;
   eight are wired into `.github/workflows/ci.yml`, see the snapshot
   for the ninth).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-08-20: #90 BUILT; reviews done, awaiting merge)

**State:** `main` = `24688cb`. `claude/dev` restarted from it, **6
commits**, full gate green at every push. #90 — rolling multi-factor
regression, the last research-grade item — is built, assessed,
reviewed, and documented:

| Commit | What |
|---|---|
| 0a052b5 | the anchored `FactorMoments` carrier, `solve_spd` (the single solve seam), and `MultiFactorRegression` with the incremental `evaluate_frames` sweep |
| cd08da8 | Assess: the accuracy contract against an in-tree Householder QR reference over eight adversarial corpora, and the sliding-vs-per-frame A/B |
| 251e0a7 | the ninth oracle — 228 windows diffed against NumPy `lstsq` through the C ABI, plus 12 rank-deficient windows refused as ruled |
| 688507c | repo-wide code-review fixes: the fallback-ordering defect (below), the pivot floor's real semantics, the missing middle coefficient, a dedicated solve scratch |
| cf58f6a | doc pass: the DESIGN record, and two decision records the code outgrew |
| 43a80b2 | documentation-review fixes: three spike-imported claims corrected, five stale absolutes elsewhere in the tree |

**The review's severe finding, worth remembering:** the sweep refolded
the window *before* checking whether the frame was poisoned, so every
non-finite frame paid for a fold it discarded and then refolded the
same window again. Answers stayed correct, but the speedup inverted —
roughly **half** the speed of the recompute it replaces on 1%-NaN data,
which is ordinary factor-panel data. `shifted_sweep` has always had the
ordering right; the K > 2 copy diverged. Fixed, and the perf test now
carries a NaN leg (0.54x → 1.4x) because clean-data timing cannot see
this class of failure at all.

**Evidence:** 9 unit tests plus the numerics guard and an ignored A/B
measurement (and a doctest, and a perf-sanity leg); nine oracle
scripts. Accuracy is judged against two references sharing no
computational path with the shipped route — Householder QR in-tree and
NumPy `lstsq` through the C ABI. Speed: 4.2x–37.6x over per-frame
recompute across K in {4,8,16} and windows {32,64,256}, with the
sweep's cost **flat** in the window exactly as O(K^2)-per-row predicts.

**Two things the Human needs to see before merging:**

1. **The ninth oracle is not in CI.** The step is written and passes
   locally, but the push was rejected — this token lacks `workflow`
   scope, so `.github/workflows/ci.yml` cannot be modified. Until it
   lands the claim sits at *observed*, not *tested*. The one-line step
   is in the merge proposal.
2. **The MatLua ruling STANDS — F2(c) is an interim bridge, not a
   supersession** (Human, 2026-08-20). The in-engine `K x K` Cholesky
   is a temporary workaround until MatLua's endpoints land; then the
   two are compared, the better one wins, the other adapts, and MatLua
   is adopted — and the comparison improves MatLua either way. The
   records are written in that framing.

**Open follow-ups** (living status, none blocking): adopt MatLua's
solver once its endpoints land (the recorded reopen trigger); a shared
sweep skeleton between `shifted_sweep` and the K > 2 sweep, since their
divergence is what let the ordering bug in; Jacobi equilibration at
solve time if a scale-spread workload needs it; `RANGE` and unbounded
frames still take per-frame recompute; "poisoned" is overloaded — it
means a held mutex elsewhere in `table.rs` and a non-finite row in the
window sweeps, and the documentation review would rather the sweeps
said "non-finite" (a rename touching both copies, so it waits for the
shared-skeleton pass).

**Standing ruling from the Human (2026-07-30), important:** decisions
made ad-hoc while building are *revisitable*; only decisions that
undermine **what TallyDB is** are non-negotiable. Give the reason,
never cite the ruling.

**Toolchain:** Rust 1.97.1, matching CI.

### What comes next (after the #90 merge)

The Human's D6 rationale orders the roadmap: research-grade items
retire before M5's engineering tail. With #90 built, **that queue is
empty — nothing research-grade remains.** What is left is the M5 tail:
**M5.5 distribution**
(Python binding + wheels), **bulk Arrow ingest**, **M5.7 benchmark
suite** (#52). Also parked: #62 ingest hooks; MatLua's answer to the
requirements letter (drafted at `scratchpad/matlua-requirements.md`).

### Rulings landed since the M5 ruling batch

- **FIRST/LAST = (a)**: `FIRST(x)`/`LAST(x)`, the de-facto TSDB
  names; well-defined here because the ordering key is declared.
- **#83 ruling set (2026-08-02, on the issue)**: eligibility (c) full
  reach piecemeal; uniform repair; union read; AS-OF-recomputes;
  API-first; view-as-table; D6 build now.
- **#90 rulings (2026-08-03; framing confirmed 2026-08-20)**: F1(b)
  build now; F2(c) an INTERIM in-engine K x K solve — the 2026-07-30
  MatLua ruling stands, the bridge lasts until MatLua's endpoints land,
  then compare, better wins, other adapts, MatLua adopted; singular
  windows return NULL for that frame (agent recommendation, taken).
- **#83 tranche-3 ruling set (2026-08-03, on the issue)**: F1 pair
  stamp; F2 no `AS OF` on join views in v1 (#99 seats two-cut); F3
  the blotter admitted ("a view must fold or match something"); F4
  dim churn = full rebuild; F5 the touched walk yields the join key;
  F6 as-of first, star follows; F7 eligibility widened (acyclic +
  dim-side key); F8 as-of ties break by `_seq`, engine-wide.
- **`_seq` stays `_seq`** — with an open revisit.
- **Delete-flush cost = (a)** accept and document (done).

### Remaining open decisions (none gate the #90 merge)

- **#82** compaction 2x peak: document vs engineer (deferred).
- **#57** regex menu (deferred by ruling; option-f sketch on issue).
- **#46** agentic touchpoints; **#52** benchmark suite (M5.7).
- Boolean + logical-annotation revisit (parked, flagged
  wrong-on-purpose).
- #62 ingest hooks: unscheduled.

### Standing session facts

- Gate = fmt | clippy both legs `-D warnings` | test both legs |
  rustdoc both legs | nine oracle scripts (pyarrow_roundtrip via
  `-p arrow-lite --features oracle-harness`; the other eight via
  `-p engine --features oracle-harness`). Check every leg BY EXIT
  CODE, not by grepping counts. Re-run the counts at each close rather
  than carrying arithmetic. Doc-only (.md) pushes have gone without
  the full gate.
- CI runs on every pull request **and** on every push to `main`; the
  jobs are check, miri (`arrow-lite`), lua-suite (official 5.4.7 +
  `ltests`), sanitize. CI stable moves — if clippy fails there and
  not here, `rustup update stable` and re-run.
- The scratchpad probe crate lives under the session scratchpad
  (path-deps on real crates); rebuild if needed, never in-repo.
- sqlparser 0.62.0, GenericDialect. Any new pre-parse lift must
  splice by byte span, skip comments whole, carry adversarial tests.
- Beta shape: feed-writer process + read-only consoles (`--read-only`,
  `.refresh`/`.flush`) over one directory; readers see the durable
  prefix, old-or-new per mutation. Read-only view handles serve exact
  answers over stale materializations, including dirty boundary
  buckets.
- `rm -rf crates/*/tests/__pycache__` after running oracles
  (gitignored now, but keep the tree clean).
