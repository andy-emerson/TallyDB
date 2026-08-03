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
   run the seven scripts in `.github/workflows/ci.yml`).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-08-03: #83 TRANCHE 2 BUILT; reviews done, awaiting merge)

**State:** `main` = `a63bf3a` (the #83 tranche-1 merge, PR #97).
`claude/dev` restarted from it, **5 commits**, full gate green at
every push. Tranche 2 — running and cumulative maintained views via
bucket partials — is built, assessed, reviewed, and documented:

| Commit | What |
|---|---|
| cycle 1 | the partial-decomposition core: `PartialForm`, `decompose`, the combine-contract statement |
| cycle 2 | running views: hidden-bucket partials materialization, synthesized combine + finalize, width heuristic (span/1024, persisted, record v2), read via partials union |
| cycle 3 | cumulative views: expanding-window classification, boundary + assembly + adjustment read with conservative lower-bound extraction, AS-OF/no-bound recompute — plus the zone-map metadata fix (a shipped tranche-1 bug: scratch segments silently pruned under any numeric WHERE) |
| Assess | the m5 oracle grew to three views x 11 checkpoints (refusals asserted by reason); pricing measured: one-row repair 1.6ms vs 140.8ms recompute (0.011); ranged read 33.4ms vs 156.5s full (~0.0002) |
| review fixes | three reproduced bugs (AVG-over-i64 read panic; cumulative MAX dropping NaN against the NaN-greatest relation; source-named combine keys breaking aliased/unselected running keys) + `__` prefix reservation, answer-shape `schema()`, console view-namespace check, one scratch runner with a stated registry rule, multi-row-bucket battery (width 4), `okey_lower_bound` unit coverage, tightened perf guards |

**Evidence:** 493 tests default leg / 107 off-leg (re-run at this
commit), seven oracle scripts. The tranche-2 exactness claim — view
equals recompute through partials, whatever the history — holds at 11
DuckDB-diffed checkpoints x {bucketed, running, cumulative-ranged,
cumulative-full} through the C ABI, and in-crate across dense
multi-row buckets, negative keys, truncation's double-width bucket 0,
NaN, and i64 arguments. The combine contract (1e-12 relative for
SUM/AVG, exact otherwise) is exercised on non-dyadic data at both the
running combine and the cumulative boundary seam.

**Refusal parity, the tranche's one inherited limitation:** cumulative
reads run real windows, so a full read over uncompacted correction
segments refuses exactly as the base's windows do (compact heals
both), and `view AS OF s` refuses once corrections sit in history
segments. Ranged reads above the correction keep answering (zone maps
prune; the boundary re-folds with aggregates). The executor's
expanding-window sweep is quadratic — the 156.5s full read — and an
incremental sweep in query-lite is the named seat.

**Open follow-ups from the reviews** (living status, none blocking):
incremental expanding-window sweep in query-lite (fixes the
cumulative full read AND plain expanding-window queries); a per-view
registration surface if anyone wants custom kernels over views
(today: register on the base, query the base — uniform refusal);
console verbs for creating/refreshing views; `CREATE MATERIALIZED
VIEW` SQL once behavior is proven; a Definition cache if view-read
latency measures hot; an additive manifest field for history kill
coordinates if refresh-over-history measures hot; tranche 3
(q-hierarchical joins) holds its seat with the teaching refusal.

**Standing ruling from the Human (2026-07-30), important:** decisions
made ad-hoc while building are *revisitable*; only decisions that
undermine **what TallyDB is** are non-negotiable. Give the reason,
never cite the ruling.

**Toolchain:** Rust 1.97.1, matching CI.

### What comes next (after the #83 tranche-2 merge)

The Human's D6 rationale orders the roadmap: research-grade items
retire before M5's engineering tail. Remaining research-grade:
**tranche 3, q-hierarchical maintained joins** (the last of #83's
piecemeal reach), and **incremental multi-factor (#90)** (deferred by
the Human 2026-07-30; per-frame recompute via MatLua is the
shipped-correct path). Then the M5 tail: **M5.5 distribution**
(Python binding + wheels), **bulk Arrow ingest**, **M5.7 benchmark
suite** (#52). Also parked: #62 ingest hooks; MatLua's answer to the
requirements letter (drafted at `scratchpad/matlua-requirements.md`).

### Rulings landed since the M5 ruling batch

- **FIRST/LAST = (a)**: `FIRST(x)`/`LAST(x)`, the de-facto TSDB
  names; well-defined here because the ordering key is declared.
- **#83 ruling set (2026-08-02, on the issue)**: eligibility (c) full
  reach piecemeal; uniform repair; union read; AS-OF-recomputes;
  API-first; view-as-table; D6 build now.
- **`_seq` stays `_seq`** — with an open revisit.
- **Delete-flush cost = (a)** accept and document (done).

### Remaining open decisions (none gate the tranche-2 merge)

- **#82** compaction 2x peak: document vs engineer (deferred).
- **#57** regex menu (deferred by ruling; option-f sketch on issue).
- **#46** agentic touchpoints; **#52** benchmark suite (M5.7).
- Boolean + logical-annotation revisit (parked, flagged
  wrong-on-purpose).
- #62 ingest hooks: unscheduled.

### Standing session facts

- Gate = fmt | clippy both legs `-D warnings` | test both legs |
  rustdoc both legs | seven oracle scripts (pyarrow_roundtrip via
  `-p arrow-lite --features oracle-harness`; the other six via
  `-p engine --features oracle-harness`). Check every leg BY EXIT
  CODE, not by grepping counts. **493 tests pass on the default leg
  and 107 on the off-leg** at the tranche-2 close (re-run 2026-08-03,
  not carried arithmetic). Doc-only (.md) pushes have gone without
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
