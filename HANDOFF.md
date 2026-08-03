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

## Snapshot (2026-08-02 evening: #83 TRANCHE 1 BUILT; reviews done, awaiting merge)

**State:** `main` = `6cf6440` (the M5.0–M5.4 merge, PR #96, landed
this morning). `claude/dev` restarted from it, **6 commits**, full
gate green at every push. The Human closed #83's scoping decisions in
the Plan conversation (recorded on the issue: eligibility (c)
piecemeal, uniform repair, union read, AS-OF-recomputes, API-first,
view-as-table, and D6 "build now — research-grade risk retires before
M5's engineering tail"). Tranche 1 — bucketed single-table maintained
views — is built, reviewed, and documented:

| Commit | What |
|---|---|
| cycle 1 | definitions, eligibility refusals by name, CRC'd stamp record |
| cycle 2 | the maintenance pass: derivable dirty buckets, restricted re-fold, stamp advance |
| cycle 3 | the union read (exact at every coordinate), AS-OF-recomputes, F4 read-only |
| Assess | the seventh oracle (view vs DuckDB recompute after every step, in CI) + the scaling measurement |
| review fixes | seven findings: the durability-alignment bug (flush-then-stamp + rebuild floor), the console misopening view dirs, read-only refusal, _seq-on-view, multiplier mismatch, v1 kill sentinel, dedup/vestigial cleanups |
| doc pass | DESIGN's #83 section, README, console help, session-speak purge |

**Evidence:** 471 tests default leg / 85 off-leg (re-run at this
commit, not carried), 29 suites, **seven** oracle scripts (the gate
list grew: m5_view_oracle.py joins the six). The subsuming property —
`view == recompute at every knowledge coordinate` — holds over 160
seeded interleavings in-crate and 11 DuckDB-diffed scripted states
through the C ABI. Refresh cost measured flat at 4× the table (ratio
1.09; full recompute scales 32→122ms); union-read staleness premium
bounded by the tail (1.0–1.7ms vs 0.24–0.31ms fresh). Three sabotages
initially passed SILENTLY (merged runs hid bucket-0 edges; an
UPDATE's reinsert shadowed the history walk; a dropped flush was
masked by the rebuild belt) — each test strengthened until its guard
tripped alone. The standing lesson held its ground twice more.

**The one deep bug this stretch** (found by the repo-wide review, now
the ghost test): refresh folded source-buffer rows and durably
stamped past them; a crash under `WalSync::Off`/`Group` rewound the
source below the stamp, leaving permanent ghost buckets the old code
silently adopted. The rule now: **refresh flushes the source first —
a stamp asserts durability, so everything it covers must survive any
crash the source's WAL contract admits** — with the rebuild floor
behind it for stamps no crash can explain.

**Open follow-ups from the reviews** (living status, none blocking):
console verbs for creating/refreshing views (API-first ruling's
deliberate gap); tranche 2 (running/cumulative via bucket-partials)
and tranche 3 (q-hierarchical joins) hold seats with teaching
refusals; a Definition cache in MaterializedView if view-read latency
ever measures hot; an additive manifest field for a history segment's
largest kill if refresh-over-corrected-history ever measures hot.

**M5.4 is done on TallyDB's side, and its centre of gravity moved.**
`regr_r2` is ISO-named and shipped. Residual, fitted value and
multi-factor have no standard SQL name, so #77.1 puts them
script-side — and the Human ruled (2026-07-30) that they go to
**MatLua** (github.com/andy-emerson/MatLua), a Lua array + linalg
crate over faer and arrow-rs. SQL keeps only standard names; #77 was
**not** amended. A requirements letter for the MatLua team is drafted
at `scratchpad/matlua-requirements.md` — send it as an issue there.

**Standing ruling from the Human (2026-07-30), important:** decisions
made ad-hoc while building internal tools are *revisitable*; only
decisions that undermine **what TallyDB is** are non-negotiable. Do
not defend a choice by citing that it was ruled — give the reason, or
concede. (Applied: D2's NaN half is ours and revisitable; D2's NULL
half is SQL's and therefore identity-level.)

**Toolchain:** Rust 1.97.1, matching CI.

### What comes next (after the #83 tranche-1 merge)

The Human's D6 rationale orders the roadmap: **research-grade items
retire before M5's engineering tail** — "so long as research grade
stuff is in front of us, this whole project could in theory be a
pipedream." The remaining research-grade item is **incremental
multi-factor (#90)** — known-hard numerics (downdating), not
open-design; per-frame recompute via MatLua is the shipped-correct
path when it schedules. Then the M5 tail: **M5.5 distribution**
(Python binding + wheels, ruled 2026-07-29), **bulk Arrow ingest**,
**M5.7 benchmark suite** (#52, peer/venue decision open). #83's own
follow-ups: tranche 2 (running/cumulative via bucket-partials),
tranche 3 (q-hierarchical joins), console view verbs, `CREATE
MATERIALIZED VIEW` SQL once behavior is proven. Also parked: #62
ingest hooks (shares the freeze-boundary trigger the views now use),
MatLua's answer to the requirements letter.

### Rulings landed since the M5 ruling batch

- **FIRST/LAST = (a)**: `FIRST(x)`/`LAST(x)`, the de-facto TSDB
  names; well-defined here because the ordering key is declared.
- **#62 split approved**: #62 = ingest hooks (engineering, decisions
  still open); #83 = continuous queries — whose Plan conversation has
  now happened and whose tranche 1 is built (see the snapshot).
- **`_seq` stays `_seq`** — with an open revisit (the Human learned
  the "never seen" claim was really "never seen *unbidden*"; users
  do type it).
- **Delete-flush cost = (a)** accept and document (done): persistent
  DELETE seals the buffer; reopen trigger recorded in DESIGN.md.

### Remaining open decisions (none gate the #83 merge)

- **#82** compaction 2× peak: document vs engineer (deferred).
- **#57** regex menu (deferred by ruling; option-f sketch on issue).
- **#46** agentic touchpoints; **#52** benchmark suite (M5.7).
- Boolean + logical-annotation revisit (parked, flagged
  wrong-on-purpose).
- #62 ingest hooks: unscheduled.

### Standing session facts

- Gate = fmt · clippy both legs `-D warnings` · test both legs ·
  rustdoc both legs · seven oracle scripts (pyarrow_roundtrip via
  `-p arrow-lite --features oracle-harness`; the other six via
  `-p engine --features oracle-harness`). **471 tests pass on the
  default leg and 85 on the off-leg** at the #83 merge (re-run
  2026-08-02 evening, not carried arithmetic).
  Doc-only (.md) pushes have gone without the full gate.
- CI runs on every pull request **and** on every push to `main`; the
  jobs are check, miri (`arrow-lite`), lua-suite (official 5.4.7 +
  `ltests`), sanitize. CI stable moves — if clippy
  fails there and not here, `rustup update stable` and re-run.
- The scratchpad probe crate lives under the session scratchpad
  (path-deps on real crates); rebuild if needed, never in-repo.
- sqlparser 0.62.0, GenericDialect. Any new pre-parse lift must
  splice by byte span, skip comments whole, carry adversarial tests.
- Beta shape (built as of this merge): feed-writer process +
  read-only consoles (`--read-only`, `.refresh`/`.flush`) over one
  directory; readers see the durable prefix, old-or-new per
  mutation.
- `rm -rf crates/engine/tests/__pycache__` after running oracles
  (gitignored now, but keep the tree clean).
