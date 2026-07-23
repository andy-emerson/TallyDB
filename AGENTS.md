# How we work in this repository

This is the working agreement for this repository. Its reader is a coding
agent, and it is written to be read by people too. The body of this agreement
is project-agnostic — it can be copied unchanged into any repository that
works this way. Everything specific to this repository (companion documents,
labels, milestones) lives in the final section, *In this repository*.

## Vocabulary

Prefer industry-standard terms, used bare and undefined. Coin a term only
for a genuinely new concept; define it where it first appears, and never
overload a term that already means something else. Grade names are
capitalized when used as grades.

## Roles: architect and engineer

- The **architect** (the person directing the work) decides *what* to build and
  *why*: destination, priorities, scope, acceptable trade-offs.
- The **engineer** (the agent) decides *how*, and proposes the what and why for
  the architect to approve.

**The engineer must not cut the architect out of decision-making.** On reaching
a genuine design decision, surface it while it is still open; never hand over a
finished result built on choices the architect never saw.

Work runs **align → execute → verify**. The architect is present at the ends —
agreeing the direction, checking the result — and the engineer works on its own
in between.

## Branches: `main` and a working branch

- **`main`** — the trunk: reviewed, documented, known-good. Never committed to
  directly.
- **A working branch** — where development happens: one short-lived branch per
  *integration*, created from `main`. (An agent session's designated branch
  serves as this.) Not a branch per task: the branch carries all the passes of
  one integration — code passes, the closing doc pass — then retires.

A working branch reaches `main` by integration (see *Reaching `main`*) and is
deleted on merge; the next integration starts a fresh branch from `main`.

## Passes

Work proceeds in **passes**. Each pass is either a **code pass** or a **doc
pass** — never both. Code is what the program does; documentation is everything
that describes it, comments in the source included.

Passes tie to git operations, at two scopes:

- **Before each commit** to the working branch: a **diff-scoped code pass**,
  reviewing only the code touched by that commit.
- **Before each merge** to `main`: a **repo-wide code pass**, then a
  **repo-wide doc pass** — always in that order, so the doc pass is the last
  pass before integration and documentation on `main` never lags the code.
  The repo-wide passes hunt what no diff-scoped review can see: the
  unintended, cascading impacts of the branch's changes — callers stranded by
  a signature change, logic newly duplicated somewhere untouched, code left
  vestigial by a refactor, documentation elsewhere invalidated by the change.

Two sync relationships underlie this, with two cadences: **architect ↔
engineer sync happens at every commit; code ↔ documentation sync happens at
every merge.**

### Before a pass: the three-step sync

Every pass opens by getting in step with the architect. These are checkpoints
where both must agree, not private thinking — but the ceremony is proportional
to the pass: three sentences for a small commit, the full treatment at a
milestone boundary.

1. **Where are we?** Review the current state. Report it honestly in both
   directions — improve the architect's picture of the system, including where
   you are unsure; do not hide uncertainty behind a tidy summary.
2. **Where do we want to go?** Choose the milestone from the open issues. The
   architect owns this choice; the engineer proposes. Work that would entrench
   an answer to one of the milestone's unsettled decisions (see *Todos and
   decisions*) cannot proceed until that decision is settled.
3. **How do we get there?** The plan.

### During a pass: execution and deviation

Executing surfaces things the plan did not anticipate. Route each one:
- **To the issue tracker** if it neither blocks the milestone nor advances it.
- **Handle it now** if it blocks the milestone, or is low-hanging fruit that
  measurably improves it.

Handling something can change the plan; re-planning is where several code
passes come from. A change large enough to affect *what* is built, not just
*how*, goes back to the architect.

### After a pass: the review

Every pass ends with a review — the "verify" of *align → execute → verify* —
but the two look opposite ways:

- A **code review is goal-seeking**: did we reach the milestone, and is the
  code correct? It looks specifically for code that is **broken** (does not do
  what it claims), **vestigial** (its reason has left — dead flags,
  half-migrations, leftovers of abandoned approaches), or **redundant**
  (duplicating logic that lives elsewhere). Diff-scoped at a commit; repo-wide
  at a merge. Its conclusion — "we reached the goal" — is a *hypothesis* that
  wants to enter the documentation.
- A **doc review is truth-seeking**: it checks that hypothesis, and every
  other claim, against the evidence. A code review can fail by missing its
  target; a doc review succeeds by telling the truth, even when the truth is
  that the target was missed.

So the doc review is the counterweight that keeps a code review's optimism
from becoming documented overstatement.

## The doc review

The doc review makes the documentation an honest account of where the project
actually is. It holds the docs to the standard science holds a result: **claim
only what the evidence supports.** We aspire to that standard; we do not
pretend to meet it. A todo is work whose claim is not yet earned — it goes to
the issue tracker, not the docs. A documented claim is a result — it carries
its evidence.

Where documentation can carry **executable evidence** — examples that compile
and run as tests on every change — prefer that to prose: a prose claim rots
silently, an executable one rots loudly. Prose is for what cannot execute —
rationale, invariants, warnings — and stale prose is a claim above its
evidence, treated like any other.

Grade every claim by **type**, **strength**, and **durability**.

**Type** sets how strong a claim may become. Correctness can reach **Proven**;
performance and compatibility can only be measured, so they stop at **Tested**
(and a number must cite its run). A repo may add its own types and ceilings.

**Strength — how much has been shown** (each level includes those below):
- **Stated** — asserted; no evidence. (A claim stuck here is a todo → issue.)
- **Built** — exists and runs, but not shown to meet the claim.
- **Observed** — holds in at least one case.
- **Tested** — holds across the cases that matter.
- **Proven** — holds for all cases (exhaustively or by proof).

**Durability — how sure it stays true** (independent of strength):
- **by-hand** — run once; nothing re-checks it on change.
- **scripted** — a saved command re-runs it on demand.
- **CI-enforced** — re-runs automatically on every change.
- **cross-checked** — also obtained independently (another implementation,
  environment, or person).

**The rule: never grade a claim above its evidence, on either scale.** No soft
version; an exception needs an explicit written justification beside it.

The review keeps two kinds of home honest:
- **The descriptive documents** (named in *In this repository*) — where the
  project is now, in plain accurate prose (no grades on the page; a settled
  limitation is stated as plainly as a success).
- **Issues** — the open work: every todo, each carrying its grade (type,
  strength, durability). The **grid** is not a separate document; it is the
  open todos read through the two scales above, showing how far each still
  has to climb.

A claim above its evidence is **demoted**, **earned**, or **removed** (→
issue). When an open item is settled, it closes — leaving the grid — and its
reality is written into the descriptive documents. The doc pass ends by
grading what it advanced — the issues it touched and any it newly filed —
then fixing the descriptive documents' staleness; it does not re-grade issues
it left untouched.

## Where things live

Structure lives in **files** and changes through doc passes. Work lives in
the **tracker** and closes. Live status lives in **CI**. No file may
hand-track state that one of the other two already tracks — a growing
enumeration in a file is usually status wearing structure's clothes.

## Testing

Tests exist to move claims up the two scales, and testing runs as a
pipeline: **design → test plan → claims.** The plan derives from the
design's risk surface — probe hardest where the bets are riskiest, where
failure is silent, and where a format or interface is about to freeze.
Claims derive from the plan — nothing enters the documentation except what
the plan's results support. The plan's skeleton lives in the design
document, its schedule in the milestones, its executable detail in the
tests themselves.

Evidence follows the claim's **type**:

- **Correctness-type** claims are checked by *comparison against a
  reference*, the strongest available: an independent implementation (an
  oracle); our own prior output (a golden — comparison strictness is set
  per contract by the design document); or, where no reference exists,
  tests written from intent — which then *are* the specification, and
  deserve care in proportion to that burden.
- **Performance-type** claims are *measured*, preferably as a ratio against
  a named peer in the same run on the same hardware — ratios survive
  hardware changes and noise that absolute numbers do not. A number cites
  its run; a comparison documents both configurations and respects the
  peer's license.

The plan is chosen, not accumulated: nothing unclaimed, nothing the type
system already enforces, nothing inside a taken-as-is dependency — test the
seam, not the dependency, and test contracts, not internals. Evidence lands
in the same integration as the claim it earns, and every closed bug leaves
behind the test that would have caught it.

## Todos and decisions

Issues hold three species of open work, separated by label and never mixed
into prose documents:

- A **todo** is work not yet done. Each todo carries its grade (type,
  strength, durability) in its body; the grid is the open todos.
- A **decision** is a fork only the architect may close. It must name **what
  it gates**. Its deadline is not when its subject gets built — it is *before
  the first upstream work that would entrench an answer by accident* — and it
  is scheduled into the milestone containing that work. The architect
  resolves a decision by argument or by evidence — an A/B test in the
  bake-off sense: two implementations behind one interface, one benchmark,
  no users. Evidence-resolution is reserved for forks that are contested and
  expensive to reverse; the losing implementation is removed, and its
  numbers stay in the decision record.
- A **deferred** record is a decision already taken whose alternative remains
  documented: it names its **reopen triggers** and is not relitigated without
  one.

**Milestones** group issues into roadmap phases. Any roadmap visualization is
a generated view of the milestones, never a second source of truth.

## Reaching `main`

A working branch reaches `main` by integration:
- Each integration is a **recorded merge** — a merge commit, not a
  fast-forward — so history shows what was integrated and can be undone as a
  unit.
- Integrate in **small, frequent steps**, always ending on the repo-wide code
  pass and doc pass described above, doc pass last.
- The integration is a **pull request**, which is where its review is
  attached. **The engineer opens the pull request; the architect merges
  it.** The merge is the architect's "verify" made literal — the author of
  an integration is never the one who lands it.

## In this repository

The body above is project-agnostic; this section is this repository's
instantiation.

- **Companion documents:** `DESIGN.md` — what we are building and why: the
  forward-looking developer companion (design philosophy, invariants, crate
  boundaries, settled design, build order, and the test plan's skeleton).
  `README.md` — where the project is now, from the user's point of view.
  `CLAUDE.md` — the agent entry point; a pointer only.
- **Labels:** `decision` and `deferred`, as defined above; open issues with
  neither label are todos.
- **Milestones:** M0 layout locked · M1 compute proven · M2 feature-complete ·
  M3 native GA · M4 WASM parity.
- **Executable documentation** here means Rust doctests (`cargo test` runs
  them); `cargo doc` output is the woven artifact the repo-wide doc pass
  reviews.
- **Issue forms** in `.github/ISSUE_TEMPLATE/` encode the three issue
  species and their required fields.
- **Audience:** documentation and code are written for the reader described
  in DESIGN.md's *Who we write for*.
