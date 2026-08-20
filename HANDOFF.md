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
   run the nine oracle scripts in `.github/workflows/ci.yml` —
   `m2_compute_latency_bench.py` is a benchmark, not an oracle).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-08-20: #90 MERGED; research-grade queue EMPTY)

**State:** `main` = `585fe31` (the #90 merge, PR #104, which also wired
the ninth oracle into CI — every oracle claim is now *tested*, re-earned
on every change). `claude/dev` restarted from it, clean. Nothing is
awaiting review or merge.

#90 shipped rolling multi-factor regression: the anchored
`FactorMoments` carrier, `solve_spd` (a single-function interim solve —
the MatLua ruling STANDS, see the rulings ledger), and
`MultiFactorRegression` with the incremental sweep. Accuracy is held
against Householder QR in-tree and NumPy `lstsq` through the C ABI;
speed measured 4.2x–37.6x over per-frame recompute with the sweep's
cost flat in the window. The build's memorable lesson: the sweep
refolded before checking for non-finite frames, paying for folds it
discarded — answers stayed right while the speedup inverted to 0.54x
on 1%-NaN data. Clean-data timing cannot see that failure class; the
perf test now carries a NaN leg.

**Open follow-ups** (living status, none blocking): adopt MatLua's
solver once its endpoints land (compare, better wins, other adapts —
the comparison improves MatLua either way); a shared sweep skeleton
between `shifted_sweep` and the K > 2 sweep (their divergence caused
the ordering bug); Jacobi equilibration if a scale-spread workload
needs it; `RANGE`/unbounded frames still take per-frame recompute;
rename the sweeps' "poisoned" to "non-finite" (waits for the shared
skeleton).

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

### Remaining open decisions (nothing gates current work)

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
