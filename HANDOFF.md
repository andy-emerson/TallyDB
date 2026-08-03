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
   run the eight scripts in `.github/workflows/ci.yml`).
7. **Never touch the vendored Lua** under `crates/compute-lua/vendor`.
8. The process is `AGENTS.md`: Plan → Develop → Assess → Review;
   claims at their evidence; code passes and doc passes never mix;
   repo-wide code review then documentation review before every merge
   proposal.

## Snapshot (2026-08-03: #83 TRANCHE 3 BUILT; reviews done, awaiting merge)

**State:** `main` = `cdbc36d` (the #83 tranche-2 merge, PR #98).
`claude/dev` restarted from it, **7 commits**, full gate green at
every push. Tranche 3 — maintained join views — is built, assessed,
reviewed, and documented. The ruling set (F1–F8, 2026-08-03) is on
#83; the design record with the interval lemma's proof is DESIGN's
tranche-3 section:

| Commit | What |
|---|---|
| cycle 0 | prereqs: as-of ties break by birth sequence engine-wide (F8, with compaction-preserves-winners test); the touched walk yields (ordering key, join key) pairs (F5) |
| cycle 1 | the enriched blotter: bare ASOF join views — `JoinState` (dimension stamp + ceiling, record v3), materialization strictly below the dimension frontier (the ceiling), correction intervals from the lemma, union read with a joined live half |
| cycle 2 | bucketed aggregates over the as-of join (group keys and arguments from either side) |
| cycle 3 | star equi-join views under the widened door (F7): any dimension touch → whole rebuild, ceiling parked, read serves live while pending |
| Assess | the eighth oracle (`m5_join_oracle.py`, in CI): 17 checkpoints x {blotter, as-of bars, star bars} vs DuckDB — its NATIVE ASOF JOIN for the as-of shapes, its ordinary join for the star; pricing: refresh ratio 2.73 at 4x facts (guard < 3.0, honestly O(batch + touched quotes)); late-quote correction on 1M facts re-folds 399 keys in 41.6ms |
| review fixes | three reproduced bugs: a frontier REGRESSION stranded materialized rows and nothing healed (fix: dematerialize the shrink band, always store the ceiling); StrictlyBefore correction intervals stopped one key short of their inclusive edge; the tampered-stamp rebuild floor fed key-space ranges where bucket runs were expected. Plus four regression tests (regression, strict edge, floor-vs-ceiling, negative keys) |
| doc pass | DESIGN's tranche-3 record (rulings, the ceiling, the lemma + proof, evidence, seats), module headers, README, this file |

**Evidence:** 509 tests default leg / 121 off-leg (re-run at this
commit), eight oracle scripts. The join-view exactness claim — view
equals recompute at the current coordinate, both sources' histories
included — holds through the C ABI at 17 checkpoints x 3 shapes
against DuckDB recompute (its native ASOF JOIN, an independent
implementation of the matching rule, for the two as-of shapes; its
ordinary join for the star), and in-crate
across multi-correction windows, kill-then-rebirth chains, the
strict-mode edge, frontier regression, negative keys, and the
symbol-seam discriminating case (a symbol-blind interval endpoint is
provably unsound; the test wedges a foreign quote between a
correction and its true endpoint). As-of ties are pinned in-crate
against the F8 `_seq` rule (the oracle keeps quote keys unique per
symbol so DuckDB's tie rule never meets ours).

**The honest costs:** join-view refresh is O(changed facts + touched
dimension history) — the 2.73 ratio, not tranche 1's flat 1.09 —
until #92-style dimension pruning lands; correcting a symbol's LAST
quote legitimately dirties that symbol's fact tail (the lemma's
`[t, frontier)` edge case — bounded by data, not a constant).

**Open follow-ups from the reviews** (living status, none blocking):
#99 two-cut `AS OF (s_fact, s_dim)` over join views; #92
dimension-side pruning for refresh's touched walk; a per-symbol
touched index; running/cumulative shapes over a join (the partials
compose in principle); incremental expanding-window sweep in
query-lite; per-view registration; console view verbs; `CREATE
MATERIALIZED VIEW` SQL once behavior is proven.

**Standing ruling from the Human (2026-07-30), important:** decisions
made ad-hoc while building are *revisitable*; only decisions that
undermine **what TallyDB is** are non-negotiable. Give the reason,
never cite the ruling.

**Toolchain:** Rust 1.97.1, matching CI.

### What comes next (after the #83 tranche-3 merge)

The Human's D6 rationale orders the roadmap: research-grade items
retire before M5's engineering tail. Tranche 3 closes #83's piecemeal
reach; the one remaining research-grade item is **incremental
multi-factor (#90)** (deferred by
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
- **#83 tranche-3 ruling set (2026-08-03, on the issue)**: F1 pair
  stamp; F2 no `AS OF` on join views in v1 (#99 seats two-cut); F3
  the blotter admitted ("a view must fold or match something"); F4
  dim churn = full rebuild; F5 the touched walk yields the join key;
  F6 as-of first, star follows; F7 eligibility widened (acyclic +
  dim-side key); F8 as-of ties break by `_seq`, engine-wide.
- **`_seq` stays `_seq`** — with an open revisit.
- **Delete-flush cost = (a)** accept and document (done).

### Remaining open decisions (none gate the tranche-3 merge)

- **#82** compaction 2x peak: document vs engineer (deferred).
- **#57** regex menu (deferred by ruling; option-f sketch on issue).
- **#46** agentic touchpoints; **#52** benchmark suite (M5.7).
- Boolean + logical-annotation revisit (parked, flagged
  wrong-on-purpose).
- #62 ingest hooks: unscheduled.

### Standing session facts

- Gate = fmt | clippy both legs `-D warnings` | test both legs |
  rustdoc both legs | eight oracle scripts (pyarrow_roundtrip via
  `-p arrow-lite --features oracle-harness`; the other seven via
  `-p engine --features oracle-harness`). Check every leg BY EXIT
  CODE, not by grepping counts. **509 tests pass on the default leg
  and 121 on the off-leg** at the tranche-3 close (re-run 2026-08-03,
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
