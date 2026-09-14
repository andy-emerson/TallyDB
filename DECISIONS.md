# Decisions

Every settled decision in TallyDB, newest first: what was ruled, what lost,
and what would reopen it. [DESIGN.md](DESIGN.md) describes the system as it
stands and is short enough to read in one sitting; this file is the record
behind it, and is meant to be searched rather than read straight through.

A decision is settled only when a record exists naming the alternatives that
lost — absence of a record means open. That rule is itself one of these; see
*Decisions carry provenance, a tripwire, and a record*. Open decisions live
on the [issue tracker](https://github.com/andy-emerson/TallyDB/issues) under
the `decision` label.

Dates are the day the Human ruled. Where a later build narrowed a ruling to
what the code actually earns, the narrowing is recorded under the same
heading rather than quietly replacing the claim.

## The public API is what the crate root re-exports, and nothing else (#107)

**The public surface (#107, ruled by the Human 2026-09-07: root
re-exports only).** The API is what `lib.rs` re-exports, and nothing
else: the engine (`Database`, `Table`, `TableReader`, `TableSnapshot`,
`MaterializedView`, the multi-factor types, `EngineError`); the data
model (`Schema`, `Field`, `ColumnType`, `RecordBatch`, `Column` and its
parts, `Bitmap`, `Buffer`, `Dictionary`, the logical types) and the
Arrow C Data Interface (`ArrowSchema`, `ArrowArray`,
`ArrowArrayStream`, the export and import functions); storage's
configuration and errors (`RowValue`, `StoreOptions`, `WalSync`,
`DEFAULT_SEGMENT_ROWS`, `MANIFEST`, `StorageError`, `IoError`,
`FormatError`, `CodecError`); the extension seams (`WindowAggregate`,
`ColumnFunction`, `Registry`, `QueryOutput`, `QueryError`,
`recompute_frames`; `LinalgBackend` with its op and error types and
`RustLinalg`); the Lua embedder's `LogSink` and `PRELUDE` behind
`lua`; and the console's `Console` and `Outcome` with their two line
helpers behind `cli`. The modules are `pub(crate)`: no path into them
is nameable from outside, so semver applies to that list and to
nothing else.

*Rejected:* leaving the modules `pub` — every internal signature
change a semver event; hiding internals item by item — the same audit
with no line to hold it; `#[doc(hidden)]` modules — a promise nobody
enforces. *Accepted consequences:* the store's behavioral tests moved
inside the module (`storage_lite/tests`), since nothing outside the
crate reaches `Store`; the memory tests measure through `Table` and
`Database`, which is the surface their claims are about; items only
tests reached are `#[cfg(test)]`, and items nothing reached are gone;
the rustdoc legs pass `--document-private-items`, so the modules'
essays stay link-checked and render for a maintainer. *Reopen
trigger:* an embedder with a concrete need for something a root
re-export cannot serve — which is also the crate-boundary trigger
below.

## The published package excludes what runs the project (#108)

**The published package (#108, ruled 2026-09-07).** `exclude` names
what runs the project rather than builds or tests the crate: the Lua
upstream suite, `.github`, `scripts`, and the process documents
(AGENTS.md, CONTRIBUTING.md, CLAUDE.md). What ships is the sources
with the vendored interpreter, `tests/` with its goldens and oracle
scripts, DESIGN.md, README.md, and LICENSE. *Rejected:* shipping
everything — a third larger for a suite no installer runs; excluding
`tests/` too — packagers run them.

## The first version is 0.1.0: usable, unstable (#109)

**The first version (#109, ruled 2026-09-07).** 0.1.0: usable,
unstable — within 0.1.x a minor bump is breaking and a patch bump is
compatible, so the API can still move at 0.2.0. The narrowing above is
what earns the 1. *Rejected:* 0.0.1 — honest only for the unnarrowed
surface; a pre-release tag — a second publish to reach the same place;
1.0.0 — a promise the API has not earned.

## Cargo.lock is committed, and CI builds `--locked` (#110)

**Cargo.lock (#110, ruled 2026-09-07).** Committed, and CI builds with
`--locked`, so the build CI tested is the build a checkout reproduces
and a dependency move is a visible diff; `cargo update` is a
deliberate commit. *Rejected:* ignoring it — breakage found by whoever
builds next; a second CI lane on fresh resolution — red through
nobody's fault.

## TallyDB is one published crate, not eight

**Decision record — one published crate (Human, 2026-09-07).** The
workspace's eight crates had earned their keep as *development* boundaries
while the layout was being locked — each could be built, oracled and
reasoned about alone, and the build order front-loaded the unoracled ones.
That work is done. As *distribution* units they cost more than they
return: publishing to crates.io would mean eight crates in dependency
order at every release, six registry pages for plumbing no user names,
and generic names (`engine`, `storage-lite`) that a global registry
cannot take. The user-facing unit is one — `cargo install tallydb`, or
one dependency line — and the crate matches that.

*Rejected:* publish all eight under a `tallydb-` prefix. It works, and it
is the smaller change today; every later release pays for it, and it
freezes the internal decomposition as public surface. *Also rejected:*
literate sources tangled into a single `.rs` — orthogonal to this
question (Rust modules already give organization without new crates),
and costly in Rust specifically, where the compiler's diagnostics and
rust-analyzer are the development loop and would point at generated
code. The literate reading TallyDB wants is rustdoc's — prose in `//!`
and `///` woven from the source, with the code as the single source of
truth.

*Accepted cost:* one compilation unit. A workspace compiles crates in
parallel; a crate does not. The clean developer build is slower for it
— unmeasured: the before and after were never timed on one machine — and
the product is not.

*Reopen trigger:* a second downstream consumer — something other than
TallyDB itself — wanting to depend on one internal module without the
rest. That is the one thing a crate boundary provides and a module
cannot. The planned extraction of `arrow-lite` is **not** this trigger: it
leaves as its own design in its own repository, and TallyDB takes it back
as an external dependency at the `arrow_lite` seam.

## Non-standard compute lives in the Lua tier, via MatLua

**Decision record — where non-standard compute lives: the Lua tier,
via MatLua (Human-ruled 2026-07-30).** The reopen trigger above fired
at M5.4: multi-factor regression is the first committed op needing more
than two parameters. The answer is **not** a faer dependency of our
own, and **not** new SQL names. SQL stays standard — item 3 of the M5
ruling batch, unchanged — and a user who wants more than the standard
spells turns to the Lua tier, where [MatLua](https://github.com/andy-emerson/MatLua)
is the matrix and linear-algebra vehicle. Two things follow. TallyDB
does not coin `regr_multi`, `pca`, or their relatives into SQL: had we
built the solver ourselves, the pressure to expose it would have been
immediate, and item 3 would have had to bend. And TallyDB does not take
faer directly: MatLua already depends on it, and Cargo unifies
semver-compatible versions, so reaching linear algebra *through* MatLua
costs no second copy.

*Interim workaround inside this ruling, 2026-08-03 (F2(c) on #90;
framing confirmed by the Human 2026-08-20: **this ruling stands — it is
not superseded**).* When #90 came to be built, MatLua's Rust API had
grown a working solver family but its host-integration endpoints had
not landed, and the destination could not wait on them. So TallyDB
carries its own `K × K` solve *for the meantime* — a hand-rolled
Cholesky in `engine::multifactor::solve_spd`, behind one function
precisely so the swap is a one-line change — and MatLua remains where
this compute lives once its endpoints land: whichever implementation
measures better wins, the other adapts, then MatLua is adopted. The
Human's stated rationale for the detour cuts both ways on purpose — if
ours turns out better, MatLua improves too. The sentence above, "the
answer is not a faer dependency of our own", still holds (the interim
solve is hand-rolled and calls no faer routine; #90 added no
dependency), and the part of this record that mattered most survives
untouched: the solver was kept out of SQL because of the pressure it
would put on item 3, and that pressure was answered by giving the op
**no SQL name at all** — reachable only by registration, so the
dialect never grew a `regr_multi`.

*Status: ruled and standing; bridged in the engine until MatLua's
endpoints land.* The adoption comparison is the recorded trigger on
#90.
A requirements letter is out — what would break TallyDB if
MatLua chose otherwise (a Lua face that works against a host-owned
interpreter, no `Drop` value live across a `longjmp`, no panic across
the C boundary, `i64` exactness with no implicit widening at the
boundary, a documented contract for absence, and an Arrow **C Data
Interface** path so neither side links the other's Arrow stack) versus
what is theirs to decide (NaN or mask internally, indexing, which
factorization backs `lstsq`, behaviour on singular input, error
taxonomy, dtype order). The split follows the Human's standing
principle: **a decision made ad hoc to an emerging need while building
our own tools is revisitable; only a decision that would undermine what
TallyDB *is* is not.** MatLua are the linear-algebra experts; we are
the time-series-database ones, and where the two overlap we take their
design.

*What could reopen this:* MatLua declining the embedding or exactness
requirements, at which point the choice returns to a faer dependency of
our own with the SQL surface still frozen at standard names.

## Rolling multi-factor regression carries anchored moments, not a factorization (#90)

A rolling least-squares fit of a response on **K factors**, maintained
across the window's slide rather than re-solved per frame. `K > 2` is
what motivated it — that is where the closed form runs out — but the
kernel is general in K.
Above two parameters there is no closed form, so this is the first op
that needed a solve at all — the reopen trigger the no-LAPACK record
set, fired at last.

**Why moments, not a factorization.** The issue proposed rank-one
up/downdating of a Cholesky factor. Research closed that: downdating is
an **ill-posed problem**, not merely a delicate algorithm — when the
departing row leaves the window near-rank-deficient, the digits needed
were rounded away when the row was added, and no algorithm holding only
the factor and the row can recover them (Stewart 1979; Pan 1993;
Eldén–Park 1996). Signal processing reached the same conclusion decades
ago and standardised on periodic restart or exponential forgetting;
PostgreSQL refuses float inverse transition functions for the adjacent
reason, and DuckDB uses subtraction-free segment trees. Measurement
agreed with the literature: factor downdating lost to every
anchored-moment carrier on the shapes that decide it — 1.7e-13 on
*benign* data, where the others sit at 1e-14 — and won none of them,
even with a periodic rebuild to rescue it (four candidates against two
baselines; spike table on #90, 2026-08-03). What ships instead is the **anchored moment carrier** — *the sweep*,
below — `XᵀX` and
`Xᵀy` as sums of deviations from a data-row anchor — re-solved per
frame at `O(K³)`, which at these K is nothing, and maintained at
`O(K²)` per row.

**The three disciplines, inherited from K ≤ 2.** Anchoring (every value
enters and leaves as a deviation from a point taken from the data, so
sums stay at the window's scale); periodic rebuild (re-anchor and
refold every `w` steps, bounding drift to one period); and one shared
`fit()` for both evaluation schedules, so per-frame and incremental
apply one NULL rule — which the numerics guard asserts frame for frame
that they never split on. A fourth rule, inherited from the same place but restated because this
copy got it wrong: non-finite rows
break the sliding identity, so those frames fall back to exact
per-frame arithmetic — and the fallback must be decided **before** the
accumulator is touched, or the frame pays for a fold it discards. The
repo-wide review caught exactly that inversion, which had made the
"incremental" path roughly half the speed of the recompute it replaces
on 1%-NaN data while answering correctly throughout.

**What a frame refuses.** Fewer rows than parameters, or a design whose
factors are near-dependent, yields NULL for that frame — never an error
for the query, the convention the non-finite path already set. The
pivot floor tests near-linear-**dependence**, not conditioning: it
rejects when a factor's variance, after the earlier factors are
projected out, falls below `1e-12` of its own. For collinearity that
lands near `κ(X) ≈ 1e6` (measured 2026-08-03, this container: accepted at
3.1e5, refused by 3.1e6; the test pins a looser bracket either side); a
merely badly *scaled* design passes at any κ, which is right — scale is an
equilibration question, not a rank one.

**Surface.** No SQL name (#77.1 stands): the kernel is reached by
registration, so the dialect never grew `regr_multi`. The solve is an
**interim workaround** under F2(c) — a hand-rolled `K × K` Cholesky
behind a single function; the 2026-07-30 MatLua ruling stands, and
once MatLua's endpoints land the two implementations are compared,
the better one wins, the other adapts, and MatLua is adopted (Human,
2026-08-20: the comparison improves MatLua either way).

**Evidence** (2026-08-03, this container). Accuracy is judged against
two references that share no *solve* with the shipped route:
Householder QR in-tree — which anchors identically on purpose, so the
comparison isolates the solve rather than the centering policy — and
NumPy `lstsq` through the C ABI, which shares neither. Against
QR over eight adversarial corpora at 64-row windows, the sweep holds
1.3e-14 benign, 2.6e-14 / 9.1e-15 / 3.7e-15 at offsets 1e6 / 1e9 /
1e12, 1.7e-13 on a drifting level (where per-frame recompute is
3.8e-12 — the sweep is *better*), 1.2e-14 on scale spread, and 3.3e-5
on near-collinearity, with windows collinear to 1e-6 refused outright.
The ninth oracle diffs 228 windows × {intercept, three coefficients,
R²} against `lstsq`, worst 4.7e-11, plus 12 rank-deficient windows
refused as ruled — in CI since the #90 merge (PR #104), so the leg is
*tested*, re-earned on every change. Speed: 4.2×–37.6× over per-frame recompute across
K ∈ {4, 8, 16} and windows {32, 64, 256}, and the shape confirms the
cost model — the sweep's time is **flat** in the window (13.9 / 13.8 /
14.0 ms at K = 8) where per-frame scales with it (77.1 / 139.2 / 499.4
ms). On 1%-non-finite data the sweep is 1.4×, near parity, because
those frames delegate to the recompute they are timed against.

**Costs and seats.** The per-frame baseline allocates a fresh carrier
per call, so the published ratios carry that allocation in the
*baseline* — unmeasured, and flattering nothing about the sweep.
Compensated (Neumaier) accumulation was measured and did **not** ship:
the spike put it 2–9x ahead in the hardest offset cells for roughly
twice the accumulator cost, while plain accumulation already clears the
guard by two orders at every corpus — so the cost bought nothing the
contract needed. Scale
spread is handled by anchoring but not equilibrated — a Jacobi
diagonal scaling at solve time is the seat if a workload needs it.
`ROWS` frames get the sweep; `RANGE` and unbounded frames still take
per-frame recompute, the same gap the K ≤ 2 kernels have. And the
sweep skeleton now exists twice — here and in `shifted_sweep` — in
subtly different forms; the review noted that divergence is exactly
what let the fallback-ordering bug in, so a shared helper is worth
considering.

## Maintained join views are bounded by the ceiling and repaired by the interval lemma

Views whose FROM clause is a join over two tables: the **enriched
blotter** (a bare projection over `fact ASOF LEFT/INNER JOIN
reference` — every fact row carrying its matched quote), bucketed
**aggregates over the as-of join**, and **star aggregates over the
equi join** (fact rows enriched with keyed dimension attributes,
folded into bars). The full ruling set (F1–F8, each with its rejected
alternatives) is recorded on #83; the load-bearing rulings:

- **F1, the pair stamp**: a join view's durable state grows the
  dimension's own stamp next to the fact stamp (record format v3,
  additive; single-source views keep v2) — the live-dimension variant,
  which would have re-derived the dimension's state per read, was
  rejected as an always-paid cost for a rarely-taken freshness gain.
- **F3, the blotter admitted**: a bare projection is admissible only
  over a join — a view must fold or match something, and the blotter
  materializes the match. Bare single-table projections stay refused.
- **F7, the widened door**: eligibility is "q-hierarchical, OR acyclic
  with the join key a key of the small side." The as-of join is
  q-hierarchical; the equi join rides the second clause with its
  precondition — the join key unique on the dimension side — checked
  loudly at execution, not assumed.
- **F8, the tie rule**: among as-of candidates sharing an ordering
  key, the match is the row with the greatest birth sequence
  (`_seq`) — knowledge order, engine-wide, in the executor itself.
  Storage order had been the tie-breaker by unguarded coincidence;
  a test pins that compaction preserves tie winners.
- **F2, `AS OF` over a join view refused**: the answer is a function
  of two knowledge coordinates, so a single `s` is ambiguous; the
  honest two-cut form (`AS OF (s_fact, s_dim)`) is seated as #99.

**The ceiling.** The one genuinely new maintenance idea. An as-of
match is not bucket-local: the newest quote matches every fact after
it, so a fact materialized against today's last quote is invalidated
by tomorrow's perfectly ordinary in-order quote append — maintenance
cost where the design promises none. The fix inverts the intuition
that fresher is better: refresh materializes fact keys strictly below
the **ceiling** — the low edge of the bucket holding the dimension
frontier, `bucket_low(frontier / divide, divide)` in the code's terms
(truncating division, so negative frontiers land in the double-width
bucket around zero and the edge is still correct) — and leaves
everything at or above it to the union read's live half. An in-order quote arrival
lands at or above the frontier by definition, touching only live
territory: zero materialized damage, no refresh owed. A **frontier
regression** (a correction deletes the newest quote) lowers the
ceiling, stranding materialized rows built against knowledge the
dimension no longer holds; refresh dematerializes the stranded band
`[new ceiling, old ceiling)` and always stores the new ceiling. (The
repo-wide code review caught this as the build's one severe bug —
refresh itself corrupted and nothing healed; the regression test
replays it.) The star shape needs no ceiling: any dimension change
rebuilds the materialization whole (F4 — a dimension row's blast
radius is every fact bucket holding its key, so surgical repair
degenerates), and the union read serves the whole answer live while a
rebuild is pending, so the parked state is never wrong, only slower.

**The interval lemma.** A dimension correction touching row `(s, t)`
changes the join result only for fact rows of symbol `s` with keys in
`[t, next(s, t))`, where `next(s, t)` is the key of `s`'s next
dimension row after `t` in the corrected state. Proof, from the
max-before definition: a fact `(s, u)` matches the dimension row for
`s` with the greatest key `≤ u`; if `u < t`, the set of keys `≤ u` is
unchanged by any edit at `t`, so the match is unchanged; if
`u ≥ next(s, t)`, a surviving row at `next > t` bounds the max from
below, so a row present or absent at `t` is never the max and the
match is again unchanged; what remains is exactly
`t ≤ u < next(s, t)`, one contiguous interval per corrected row
(under `StrictlyBefore` the exact set is `(t, next]`; the
implementation conservatively re-folds `[t, next]`, one key wider).
We found no statement of this in the literature — it appears as
folklore embodied in Feldera's ASOF operator and Flink's
versioned-table state cleanup —
so the proof lives here, and executably: a test prices a late quote's
refresh at exactly its interval width. The lemma is why corrections
need no bookkeeping: the dirty set stays derivable from the knowledge
history, now read through a seam that yields each touched row's
`(ordering key, join key)` pair (F5 — a symbol-blind endpoint is
provably unsound: the *global* next quote truncates another symbol's
interval, silently under-repairing; the discriminating test wedges a
foreign quote between a correction and its true endpoint). Edge case,
named honestly: correcting a symbol's **last** quote makes the
interval `[t, frontier)` — contiguous still, but O(that symbol's
facts); bounded by data, not by a constant.

**Evidence at the build** (2026-08-03, this container): the eighth
oracle family (`m5_join_oracle.py`, in CI) drives all three shapes
through facts-ahead-of-quotes, in-order appends while stale, late
quotes below the ceiling, quote amends and deletes, a dimension
change, fact corrections, compaction, and a reopen — 17 checkpoints,
each diffing the two as-of views against **DuckDB's native ASOF
JOIN** recomputing from scratch (an independent implementation of the
matching rule itself, not just of folding) and the star view against
DuckDB's ordinary join (quote keys stay unique per
symbol so DuckDB's tie rule never meets ours — ties are pinned
in-crate against the F8 rule instead). In-crate batteries cover the
lemma under multi-correction windows (kill-then-rebirth chains, a
kill whose current-state next is itself touched), the strict-mode
inclusive edge, frontier regression, negative keys through bucket
zero, the tampered-stamp rebuild floor honoring the ceiling, and the
symbol seam's discriminating case. Pricing, measured as same-run
ratios (`perf_sanity`, 2026-08-03 run): blotter refresh of a fixed
2,000-row batch at 4× the fact table costs ratio 2.73 (guarded
< 3.0) — **not** tranche 1's flat 1.09, and honestly so: each
refresh's dirty derivation walks the dimension's touched history,
O(Q) until #92-style pruning, and the guard's headroom says "scales
with the batch and the quote book, not the fact table"; a late-quote
correction against 1M facts re-folded 399 keys in 41.6ms — the
interval, not the suffix.

**Tranche-3 costs and seats**: refresh is O(changed facts + touched
dimension history) — the dimension-side scan holds the #92 seat
(manifest pruning would cut it) and a per-symbol touched index holds
another; `AS OF` over join views waits on #99's two-cut form;
running/cumulative shapes over a join compose in principle (the
partials are tranche-1 plans) but hold their seat unbuilt; the
refusal strings for the single-table and joined doors repeat some
phrasing — noted, not worth a shared constant yet; console verbs and
SQL DDL for views remain the API-first ruling's deliberate gap.

## Running and cumulative views store bucket partials, not answers

The shapes tranche 1 refused because their blast radius under
correction is unbounded — a correction at `t` changes every result
after `t` — are admitted by changing what is stored. The
materialization holds, per **hidden bucket** of the ordering key, not
the answer but the **partials** the answer recombines from: a bucket's
sum, count, (sum, count) for `AVG`, min, max, or edge value. The
load-bearing fact of the whole representation: **partials and their
combines are themselves built aggregates**, so the synthesized
materialization is a legal tranche-1 bucketed plan and every piece of
tranche-1 machinery — refresh, touched-bucket derivation, the stamp,
the crash story, the rebuild floor — serves both new shapes unchanged.
A correction re-folds one hidden bucket; the O(suffix) rewrite never
exists because no suffix is stored.

**The hidden bucket width** is a heuristic, not a semantic: chosen at
the first refresh that sees data (observed key span over a target 1024
buckets, clamped to at least 1), persisted in the definition record
(format v2, additive; v1 records decode as width-unchosen and
self-heal) *before* folding under it, so a crash re-folds under the
same width. A re-widthing is a rebuild, deliberately not a format
question.

**The running read**: partials union (clean materialized buckets + a
live partial fold of everything the stamp does not cover) → a
symbol-keyed **combine** reassembling cross-bucket totals → finalize
into the user row shape — `AVG` divides once, after the combine (an
average of averages weights buckets, not rows, and is simply wrong);
`COUNT`'s NULL sum-of-counts grounds to 0.

**The cumulative read** splits every expanding window at the query
predicate's ordering-key lower bound (conservatively extracted: `AND`
takes the tighter branch, `OR` needs both and takes the looser,
unhandled shapes fall to recompute — a bound may sit below the truth,
never above it): a **boundary** combine over the partials strictly
below that bucket, an **assembly** of the user definition over the
source from the bucket's low edge (truncating division is monotone, so
the two ranges partition exactly), and a per-column adjustment folding
boundary into assembly — `AVG` through hidden sum/count helper
windows, never through its quotient; `MAX` propagates NaN under the
engine's NaN-greatest relation. A query with no lower bound wants
every output row and recomputes: the partials cannot shorten an
O(n)-row answer.

**The combine contract, stated** (2026-08-03, revisitable): combining
per-bucket f64 sums associates differently than a single pass, so
`SUM`/`AVG` through partials agree with recompute within **1e-12
relative** — the tolerance the mutation, as-of, and view oracle
families apply (the slice and differential families run at 1e-9).
Both folds run the executor's ordinary aggregates — plain f64
accumulation; the Neumaier reference in the M5.0 numerics guard is a
test yardstick, not shipped summation.
`COUNT`/`MIN`/`MAX`/`FIRST`/`LAST` combine exactly. Exact single-pass
equality is impossible under any partials representation.

**Refusal parity, inherited**: cumulative reads run real windows, and
windows refuse disordered data — so a full read over uncompacted
correction segments refuses exactly as the base's windows do
(`compact` heals both), and `view AS OF s` refuses once corrections
sit in history segments (their key ranges interleave with the live
generation's). `view AS OF s = Q(base AS OF s)` includes the
refusals. A *ranged* read above an uncompacted correction keeps
answering exactly — zone maps prune the stray segment and the boundary
re-folds with aggregates, which need no order.

**Evidence at the build** (2026-08-03, this container): the m5 oracle
grew to three views over one source — bucketed, running, cumulative —
diffed against DuckDB recompute at all eleven checkpoints (the
cumulative full read's refusals are themselves asserted, by reason);
a 4096-row dense battery forces width 4 so multi-row hidden buckets
are value-checked (FIRST/LAST inside a bucket, mid-bucket range
floors, an OR-predicate bound, one-bucket repair at width 4);
bucket-edge crossings cover negative keys and truncation's
double-width bucket 0. Pricing, measured as same-run ratios
(`perf_sanity`, 2026-08-03 run): a one-row correction on a running
view over 1M rows repairs in 1.6ms vs 140.8ms full recompute (ratio
0.011, guarded < 0.1); a cumulative ranged read of the last 10k of 1M
rows costs 33.4ms vs 156.5s for the full read (ratio ~0.0002, guarded
< 0.05) — the full read pays the executor's quadratic expanding-window
sweep, which the ranged read never touches.

**Tranche-2 costs and seats**: the executor's expanding-window sweep
is O(n²) in the frame lengths — an incremental sweep in `query-lite`
would fix the cumulative full read (and every plain expanding-window
query) and holds a seat; a view read resolves registered functions
from the view's own always-empty registry (register on the base,
query the base — a per-view registration surface is a held seat);
names beginning with `__` are reserved in running/cumulative
definitions for the minted hidden columns.

## A maintained view is versioned, repaired uniformly, and read as a union (#83)

The first continuous-query build — derived data kept fresh on ordered
append, correct across corrections. The research survey, the option
menu, and the full ruling set live on issue #83; this section records
what was ruled, what shipped, and what the evidence earned. The scoping
principle behind all of it: **the field's three answers to "what does a
maintained result do when its input is corrected" — retraction deltas,
invalidation + repair, refuse/rebuild — compose here because the
knowledge axis already gives base and view one shared version
coordinate.** (Prose says "maintained view"; the API type is
`MaterializedView`. The view's **stamp**, used throughout below, is
the source-table ingest-sequence watermark below which the
materialization is complete.)

**The ruling set** (each with its rejected alternatives recorded on the
issue): eligibility is **(c) the full reach, taken piecemeal** —
tranche 1 is bucketed single-table views, tranche 2 running/cumulative
shapes via bucket-partials, tranche 3 join views (ruled q-hierarchical
only at first, per the PODS 2017 dichotomy; the door was widened by F7
on 2026-08-03 — see the tranche-3 record below); correction semantics
is **the versioned view
with uniform repair** — every correction marks its bucket, repair is
always re-fold-from-base, the class-split delta fast path rejected for
v1 (a second code path plus an f64 subtraction hazard for a path
corrections are too rare to need); reads are **the union read** —
materialized clean buckets plus a live fold of everything the stamp
does not cover, so the view is semantically always exact and repair
only shrinks the live half; **`AS OF s` on a view recomputes**
`Q(base AS OF s)` — the materialization accelerates current reads and
is never the authority; the surface is **engine API first**
(`create_materialized_view` / `refresh_view`, the `register_window`
pattern), SQL DDL after behavior is proven; storage is **a real table**
plus a CRC'd definition record carrying the stamp.

**The model as built.** A view is a fold over the ingest sequence,
stamped with the source watermark below which the materialization is
complete. The dirty list is derivable state — buckets touched by any
coordinate the stamp does not cover, re-derived from the knowledge
history machinery M4.4 built — so the stamp is the only durable view
state, and it is written strictly after the materialization it
describes is flushed. Refresh flushes the source first: the stamp
asserts durability, so everything it covers must survive any crash the
source's own WAL contract admits (the alternative — stamping buffered
rows — left permanent ghost buckets when a crash rewound the source;
the repo-wide code review found it, and the ghost test replays it). A
stamp found *ahead* of the source (a swapped directory, a tampered
record — impossible from a crash under this discipline) meets the
rebuild floor: every materialized row out, one full fold in. A
read-only process (F4) serves exact view answers with no writes: the
union read needs none, and an older (stamp, materialization) pair only
means more live work, never a wrong answer.

Two permanent restrictions, both definitional: a view definition may
not read across knowledge time (`AS OF` / `_seq` in the definition
breaks `view AS OF s = Q(base AS OF s)` — snapshot reducibility), and
`_seq` *of* a view is refused (a view row summarizes many source rows
and has no single ingest coordinate). `ORDER BY` / `LIMIT` /
`DISTINCT` / `HAVING` are refused in definitions because a view is a
table — they compose at read.

**Evidence at the build** (2026-08-02, this container): the subsuming
property — view equals recompute at the current knowledge coordinate,
whatever the history (past coordinates are exact by construction: they
recompute) — holds at each of 160 states along one seeded
pseudo-random interleaving of append, update, delete, and refresh,
with one mid-run compaction; plus the seventh oracle family
(`m5_view_oracle.py`, in CI): every statement mirrored into DuckDB and
the view's answer diffed against from-scratch recompute at eleven
scripted checkpoints spanning stale, fresh, corrected, compacted, and
reopened states.
The scaling claim is measured: at 4× the table with a fixed 2,000-row
batch, refresh cost is flat (0.78ms vs 0.85ms, ratio 1.09, guarded
< 2.5 in `perf_sanity`) while full recompute scales 32ms → 122ms; the
union read's staleness premium is dominated by the live fold of the
tail (1.0–1.7ms vs 0.24–0.31ms fresh across the two table sizes — the
staleness-premium check the read-semantics ruling asked for, affirming
it).

**Costs and seats, stated plainly**: each refresh flushes one small
segment on the view (and possibly one early freeze on the source when
called mid-buffer; at the freeze-boundary cadence the flush is free);
refresh also scans compacted correction *history* unconditionally —
its kill coordinates live in the segments, not the metadata, and an
additive manifest field removes the scan if it ever measures hot (the
flat-refresh measurement above is a correction-free run) —
`compact` on the view restores contiguity; the console opens existing
views correctly (both scan sites route on the definition marker) but
has no verbs to create or refresh them yet — the API-first ruling's
deliberate gap.

## A table bigger than memory gets segment-granular lazy residency

**The ruling.** Tables bigger than memory are handled by **segment-granular
lazy residency (option b)**: an open reads metadata only, segments decode
on first data access, and decoded segments are retained as a cache under a
byte budget, evicted least-recently-used. The prune-metadata lives in **a
manifest section (tag 1)** — one record per live segment carrying its
name, row span, ordering flag, sequence summary, and zone maps, written by
every flush (segment file → manifest → WAL reset) and by compaction's
commit. The manifest is thereby the authoritative segment list; a stray
segment file (a crash inside the flush window) is never adopted, its rows
recovered from the WAL. Legacy manifests fall back to the backend scan and
earn the section at the next writer open.

**Rejected alternatives.** *Document the ceiling* (nothing built): the
append-heavy ledger is exactly the shape that outgrows RAM — disqualifying.
*Query-scoped streaming with no cache*: every query re-pays decode
(31–93M values/s per #42's run), turning the ~120µs hot-window query into
~10ms+ — wrong trade for an access pattern predictably skewed to recent
data, which is precisely what LRU serves. *Column-granular residency*:
a refinement, not an alternative — deferred with its reopen trigger to
#87. *Metadata via ranged reads + segment-header parse*: forces a
per-section checksum redesign now (the whole-file CRC cannot verify a
partial read), and in a browser costs one range request per segment just
to plan, where the manifest is a single small fetch — the WASM future
argues *for* the manifest section, not against it. Ranged reads travel
with column granularity in #87, when partial-segment fetch earns its
checksum revision.

**The budget's contract.** `StoreOptions::cache_bytes` (engine
`open_read_only_with_cache`, console `--cache MiB`), surviving a reader's
refresh. It is **advisory over retention, never over correctness**: a
segment an `Arc` still pins (a snapshot, a running query) is never
evicted, so peak memory is the budget plus the largest concurrent working
set. The interim default is unbounded — today's behavior exactly; the
default's final value, and a strict-refusal mode, are deferred to #87.
The bound's sharp edge is recorded as #88: the executor materializes
every surviving segment for a query's lifetime and zero-copy outputs pin
inputs, so an unpruned full-table scan's working set is the table itself;
streaming aggregation is the recorded follow-up. Compaction likewise
materializes everything by construction (#82's territory).

**What a refresh keeps.** The F4 reader's refresh reuses the previous
open's slot context wholesale — same cache, same decoded segments — for
every name the new manifest still carries (files are immutable within a
generation; history files always). Resident stays resident,
pointer-equal, zero re-reads.

**Reversal class.** Additive: the eager path is the lazy path with every
fault taken at once, the format change is one skippable manifest section,
and old binaries scan as before.

**Evidence.** Counting-backend tests pin each claim (a recorded open
reads no segment files; a pruned segment's file is never read; the budget
evicts cold, never pinned; refresh keeps decoded state); a forty-segment
table under a four-segment budget answers six query shapes
batch-identical to an unbounded open; the crash window between segment
and manifest writes is pinned end-to-end by injected failure; all six
differential oracles then in the gate pass over the faulting path. Corruption moved with
the design: a bad segment file is loud at first fault, not at open.

## Cross-sectional partitioning is the transpose of `PARTITION BY sym`

**Cross-sectional partitioning — a later ruling (2026-07-30), built
in the same milestone and recorded here because it completes the
same axis.** The transpose of `PARTITION BY sym`: partitioning on the ordering
key, or a monotone bucket of it, gives each row its own *instant*
across every symbol instead of its own symbol across time. An
unordered window (`OVER (PARTITION BY ts)`, no `ORDER BY`) takes its
whole partition, which is standard SQL and is exactly a
cross-section; a frame clause beside it is refused as the
contradiction it is. Several terms intersect — `PARTITION BY sym,
ts / 60` is one partition per symbol per bar.

`PARTITION BY` admits any `BIGINT`, not only symbols; `DOUBLE` is
refused because float equality is not partition identity (the same
reason F1 rejected general expressions), and bucket arithmetic stays
ordering-key-only, where monotonicity means something. Which column
is named decides the direction, and which path runs is decided by
declared structure plus segment metadata — never by cost.

*Note the asymmetry, deliberate and flagged:* `GROUP BY` stays
restricted to the ordering key (F1's ruling), while `PARTITION BY`
admits any `BIGINT`. Different clauses, different rulings; revisit
F1 if the difference ever bites.

**The idiom this exists for needed a second thing.** A
cross-section is only useful if a row can be compared *to* it, and
that meant scalar expressions over window results (#94, built the
same day): `x / sum(x) OVER (PARTITION BY ts)` is the weight, `x -
avg(x) OVER (...)` the demeaning. Window calls are hoisted out of
the scalar at lowering and computed first — standard SQL's own
evaluation order, and forced rather than chosen, since a partition
spans segments while a scalar walks one at a time. The same change
made `x - lag(x) OVER (ORDER BY ts)` and the rolling z-score
expressible.

## Time bucketing is monotone integer arithmetic on the ordering key (F1)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Time bucketing (F1) = monotone integer arithmetic on the ordering
key.** `GROUP BY ts / 60` (bucket index) and `GROUP BY (ts / 60) * 60`
(bucket start) are admitted — the planner proves monotonicity
structurally (ordering key, positive integer literal, `/` or `*`),
so grouping streams with O(1) state and no hash table. No `bucket()`
function is coined; unit sugar, if ever, is a later question.
Everything else in `GROUP BY` keeps the teaching error. Rejected:
general expressions (kills the streaming dividend, drags in
float-equality group identity); refusal (concedes the workload's
most common query). `FIRST`/`LAST` aggregates (OHLC's open/close)
surface a naming sub-ruling when M5.3 builds this.

**Build note (M5.3, 2026-07-30).** Built, and the streaming dividend
with it — but the claim above needed narrowing to what the code
earns:

- **"O(1) state, no hash table" is now "the open bucket's state".**
  Once a bucket is left it cannot come back, so its groups close and
  reduce to cells there and then; what stays live is the groups
  *inside one bucket* (the symbols trading in this minute), not
  every group the query will produce. The result itself, and the
  keys labelling it, are one per group either way — so the honest
  claim is that the accumulator state stops scaling, not that
  memory does. Measured on a 200,000-row fixture over 160,000
  groups: **40.4 MB streaming vs 66.7 MB hashing, a ratio of 1.65**
  (`bucket_grouping_memory`, 2026-07-30). Truly constant state
  would need streaming *output* too, which is #88.
- **Unordered data falls back rather than refusing.** Whether the
  grouping can stream is read from segment metadata before a byte
  is touched, so dispatch stays structural — but a table whose
  order an `UPDATE` disturbed takes the hash path and gets the same
  answer more expensively, with `compact()` restoring the fast
  path. Correct always, fast when the data behaves: the same
  bargain tombstones make. Refusing was considered and rejected —
  it would make a correction change which queries *run*, not just
  how fast they run.
- **`/` between integers truncates**, which is ISO and PostgreSQL;
  `//` is accepted as a synonym because DuckDB spells it that way.
  In this position only one meaning is available (a `DOUBLE` cannot
  key a group), so accepting both costs nothing. It does constrain
  #40: when exact integer arithmetic reaches projection, `ts / 60`
  must truncate there too, or one text means two things.
- **`GROUP BY` may name a bucket by its SELECT alias**, as
  PostgreSQL and DuckDB both allow. Narrow by design: only aliases
  of buckets substitute.
- **`FIRST`/`LAST` = the de-facto TSDB names** (the pending
  sub-ruling, closed (a) 2026-07-29). Positional on the time axis,
  not on row order — the group-level counterpart of `LAG`/`LEAD` —
  so a late-arriving row cannot become "last" by arriving last.
  Ties on the clock go to the last row in storage order, the rule
  the as-of join follows; nulls are skipped, as every other
  aggregate here skips them, which coincides exactly with DuckDB's
  `arg_min`/`arg_max` and is how the differential checks them.

## The as-of join is a hybrid: ClickHouse grammar, schema authority (#65)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**The as-of join (#65) — the hybrid.** Grammar from ClickHouse,
authority from the schema: the single `ASOF` token is lifted
pre-parse (byte-span splice, comments skipped — the hardened
mechanism), and the remainder parses as a plain join. `ON` only, no
`USING`. The time axis is the two tables' **declared ordering
keys** — implicit by default; an explicit inequality is permitted
and **validated, not obeyed** (naming anything else is a teaching
error; the operator selects `>=` vs `>`). **Bare `ASOF JOIN` is
refused**: write `ASOF LEFT JOIN` (keep unmatched facts,
null-padded) or `ASOF INNER JOIN` (drop them). Recorded principle,
reusable: *where vendors agree, follow convention; where vendors
genuinely diverge (bare as-of semantics do), refuse and make the
user say it.* `TOLERANCE` is cut — expressible today via
`CASE`/`WHERE` arithmetic; reopens as sugar if desks ask. Parser
facts (verified 2026-07-29): sqlparser 0.62 parses only Snowflake's
`MATCH_CONDITION` form; DuckDB accepts only its `ON` form; the sets
are disjoint — hence the lift-and-plain-join design. Evidence when
built: the DuckDB differential (the harness *generates* DuckDB's
spelling from structure) plus a vanilla-SQL definitional reference
(the row with `MAX(ts) <=` the fact's), which checks the definition
rather than another vendor's implementation. The executor is an
ordered co-walk gated on `is_ordered()` for **both** sides.

**Build note (M5.2, 2026-07-30).** Built, and the evidence landed
as ruled: seven differential families whose oracle side is a
correlated scalar subquery in vanilla SQL — the definition of "the
latest quote at or before" — rather than DuckDB's own `ASOF JOIN`.
Three things about the build differ from or extend the ruling, and
are recorded here rather than left in a conversation:

- **Not a co-walk.** The executor indexes the dimension side per key
  (an ascending `(clock, row)` list, stably sorted) and binary-
  searches it per fact row. That is correct whatever order the data
  arrived in — a late-arriving quote still matches — so it needs no
  `is_ordered()` gate on either side, where a co-walk would have to
  refuse a transiently disordered table. The cost is that the
  dimension materializes: the streaming property clause 2 of the
  join constraint claims is **designed, not built** (#92, and see
  the build note there).
- **Ties on the dimension's clock go to the last row in storage
  order** — the same "newest version wins" rule corrections follow.
  The alternative (refusing a duplicate `(key, clock)`) was rejected:
  a quote table legitimately prints twice on one stamp, and kdb+'s
  `aj` takes the last such row too. The differential covers this on
  purpose: the fixture injects per-symbol tied timestamps and the
  oracle counts them before trusting the families.
- **The inequality's sides are assigned by qualifier, not by
  operator.** `t.ts <= q.ts` and `q.ts <= t.ts` are the same
  operator and opposite questions; reading the operator alone would
  have answered the first (the quote *after* each trade) with the
  one before it. Written backwards, it is refused.

One limitation the build meets rather than creates: both tables are
timestamped, and a dimension attribute sharing a fact column's name
is refused, so `quotes.ts` beside `trades.ts` must be renamed. That
is the pre-existing equi-join rule, not a new choice — the open
decision about whether to change it is #93.

## SQL exposes only operations bearing standard names (#77.1)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Library naming (#77.1): SQL exposes only operations bearing
standard names.** `var_pop`, `stddev_pop`, `LAG`/`LEAD` pass into
SQL freely; EWMA, `diff`, multi-factor regression have no standard
SQL spelling and stay out of the dialect — reachable through the
registration API and scripts — until individually named by the
Human. The rule is mechanical — no per-op judgment.

## Matrix-valued results reach SQL only as scalar reductions (#77.2)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Matrix-valued results (#77.2): scalar reductions only in SQL**
(R², residual, fitted value); full vectors/matrices flow through
the API and scripts, which receive them from one evaluation.
Per-component SQL functions rejected; multi-output projection
plumbing deliberately unbuilt until demanded.

**What that meant at build (M5.4).** Item 3's mechanical rule
decides which of the three reach SQL, and only one does: `regr_r2`
has a standard SQL name and shipped; residual and fitted value have
none, so they are script-side. Read items 3 and 4 in that order —
item 4 says a scalar reduction *may* enter SQL, item 3 says only a
standard name *does*. As first written item 4 read as a promise of
all three, which item 3 forbids.

## The prelude is compiled into the binary (#77.3)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**The prelude (#77.3): compiled into the binary**, `.prelude`
prints the source — single-file deployment holds; read-copy-modify
is preserved by printing, not by an editable side file.

## The streaming tier is built before the matrix tier (#77.4)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Library build order (#77.4): streaming tier first**, matrix tier
second (scheduling default, delegated).

## `DELETE` consumes a knowledge coordinate

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**`DELETE` consumes a knowledge coordinate.** Decided on
*stability*: an unconsumed kill coordinate is shared with the next
append, so the delete's effect has no cut of its own — a recorded
boundary drifts as data arrives. Consuming makes every knowledge
event own one coordinate and makes `ASOF next_sequence() - 1` the
universally stable "latest" idiom. Cost accepted: a table's first
`DELETE` diverges it (the sequence column materializes;
delta-codes to almost nothing). A second cost surfaced in the
build and was ruled accepted too (2026-07-29, option (a) of the
recorded three): a persistent `DELETE` flushes the write buffer
first — recovery must not renumber rows across the consumed
coordinate — so interleaved delete/append workloads seal small
segments until compaction merges them. Reopen trigger: a real
workload shows delete-driven fragmentation that compaction
cadence cannot absorb; the fix on the shelf is a WAL
consumption marker, rejected for now because it grows the most
safety-critical code we have to optimize a verb the design says
not to lean on.

## The sequence column's SQL surface is a fixed-name pseudocolumn, `_seq` (#75)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**The sequence column's SQL surface (#75): a fixed-name
pseudocolumn, `_seq`.** Never declared, refused in `CREATE TABLE`.
Chosen by the
visibility rule: the engine refuses `SELECT *`, so the column is
never seen unbidden — the short system-side name wins over the
spelled-out one. Kill-coordinate exposure deferred until asked.

## `SYMBOL` replaces `KEY` as the column type's DDL spelling

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**`SYMBOL` replaces `KEY` as the column type's DDL spelling.**
kdb+/QuestDB lineage the audience reads on sight, and it ends the
word KEY serving two grammatical roles in one statement (`ts BIGINT
ORDERING KEY, sym KEY`). Spelling only: the stored format and the
internal type are unchanged.

## Symbol columns are unordered labels (#58)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Symbol columns are officially unordered labels (#58 = B).**
`ORDER BY` on a symbol column becomes a teaching-error refusal.
The deciding facts: codes are per-segment first-appearance ranks
(no usable inherent order exists), and byte-order "alphabetical"
is honest only for ASCII — an engine that refuses to produce a
string does not rank them. Identities are never ordered: the
arithmetic refusal and this one are the same rule. Differential
families stop using `ORDER BY sym`; the referee sorts rows before
diffing. The `WHERE sym > '…'` question and the collation
question both dissolve.

## Compression is ALP and ALP-RD together, plus the integer sibling (#42)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Compression (#42): ALP and ALP-RD together**, plus the integer
sibling sharing the same backend — frame-of-reference +
bit-packing for non-key `i64` columns and `u32` symbol codes.
Corpus tick-size realism is step 0.

## Arrow-boundary booleans are refused loudly at ingest

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Arrow-boundary booleans: refused loudly at ingest** (the
teaching error names `df.astype({'flag': 'int64'})`). Recorded
with the Human's explicit flag: *ruled wrong on purpose* — a
standing revisit covers a real boolean type **and** the
logical-annotation mechanism itself (`TimestampNs`-style
"physically i64, logically X"). Storage fact settled for that
revisit: a nullable boolean is 2 bits/row (value bit + validity
bit) — the cost was never storage, it is the seam sweep.

## The Python binding is distributed as wheels on PyPI (M5.5)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

**Python distribution (M5.5): wheels on PyPI** — for the Python
binding only. The console remains a single Python-free native
binary; the engine and console depend on nothing.

## Regex on symbols is deferred by ruling (#57)

*From the M5 ruling batch — thirteen decisions closed in one sitting by the
Human, 2026-07-29. Each is design; none is built unless the SQL surface
table in [DESIGN.md](DESIGN.md) says so.*

Also recorded from the same sitting: regex on symbols (#57) is
**deferred by ruling** — the menu on the issue now includes the house
option (registered single-symbol Lua-pattern predicates in `WHERE`,
evaluated once per distinct dictionary value); and `IS NULL` /
`IS NOT NULL` was found missing from the predicate fragment — standard
SQL, in scope, and built immediately after.

**Built since (2026-07-29).** Items 7, 8, 9 and 10 are code, not
plans: a delete consumes its coordinate and reopen recovers the spent
one from the delete logs; `_seq` reads a row's birth coordinate back
through SQL; `SYMBOL` is the DDL spelling and `KEY` is refused with a
pointer to it; `ORDER BY` on a symbol column is a teaching error and
the differential families that leaned on it now diff as sets. Two
scope notes belong with them. `_seq` is **projection-only** — it can
be selected, aliased, ordered and paged by, but not filtered or
grouped on, because `AS OF` is how a coordinate filters and a second,
weaker spelling of that would be worse than none; whether predicates
should reach it is a live question, not a settled no. And the
kill-coordinate column stays deferred, as ruled.

## User compute reaches the engine through one mechanism per host

**Decision record — the extension model (ruled 2026-07-28, from
external review).** User compute reaches the engine through **one
mechanism per host**, and the embedded interpreter serves only the
hosts that have no language of their own:

1. **Rust host → the trait.** `WindowAggregate` is the extension API:
   an embedder implements it (~20 lines) and registers the kernel on
   the table — native speed, full type safety, no interpreter. This is
   the *primary* extension path; the trait and a `register_window`
   entry are public engine surface (correcting the M2.7 state, which
   shipped only the interpreter path publicly).
2. **Python host → callbacks through the binding (M5).** Python is
   **never embedded in the engine** — ruled out on structure, not
   taste: NumPy (the thing users actually know — the familiarity is
   the library, not the syntax) is welded to CPython; CPython brings
   the process-global GIL, no viable sandbox, tens of megabytes, and —
   decisive — *circularity*, since the primary host process already is
   Python, and a library importing a second interpreter into it fights
   the first. (RustPython/MicroPython rejected: no NumPy, so the
   familiarity argument evaporates.) Instead the binding registers a
   host-side callable as a window kernel: the engine calls back into
   the host's own interpreter through the `evaluate_frames` seam —
   whole columns per call, zero-copy views in, vectorized NumPy
   inside, an array out. In-query compute in real NumPy, with no
   interpreter shipped.
3. **No host language (console; browser at M6) → embedded Lua.** The
   one territory where an embedded interpreter is non-substitutable —
   a console user cannot compile Rust at a prompt. Lua becomes a
   **non-default feature** the console (and later the browser bundle)
   turns on; library embedders opt in or never carry the C boundary,
   its sanitizer CI, or the interpreter at all. Its honest value:
   interactive kernel registration, and the measured low-latency
   niche (parity with NumPy-on-export at the newest-window shape,
   where fixed costs dominate). It is **not** the extensibility story;
   the trait is.

## Lua stays: the trial passed and the sunset clause dissolved

**The sunset clause — ruled 2026-07-28: the trial passed, the clause
dissolved.** The clause (restructured earlier the same day) held that
~32k lines of vendored C + bindings were not yet justified by Lua's
niche, so Lua would stand trial at the end of M4, once its best case
existed — the trait beneath it, the vocabulary invariant, the
vectorized whole-column slot, the compose-don't-loop idiom, and the
upstream test suite all built. The Agent brought the evidence brief
(#76): the vendored sources byte-identical to upstream v5.4.7 with
the full official suite + `ltests` torture harness green in CI, and —
after the vectorized vocabulary (option A) landed — composed column
kernels measured **ahead of the DuckDB+NumPy competitor stack**
(~2–2.5× at 20k rows after the dense fast paths), with the honest
behinds stated: element loops ~14× behind vectorized NumPy, composed
shapes still behind NumPy riding the engine's own export. **The Human
ruled Lua stays.** Consequences, all taken: SQL-in-Lua (#70) built as
M4's closing increment — the second direction earned by the first —
and the no-coined-SQL-names reopen trigger does **not** fire (the
scripting layer remains the home for novel compute names). The
rejected fail-branch stays recorded for its reopen value: removal
whole (module, feature flag, console `.lua`), tier 3 query-only, SQL
naming re-decided. Interpreter swaps were examined for the element-
loop gap and rejected on invariants, not taste: LuaJIT/Luau descend
from Lua 5.1 — **no 64-bit integer subtype**, so the `i64` exactness
contract breaks precisely where the ordering key lives (nanosecond
timestamps exceed 2^53) — and neither targets wasm32; the recorded
element-loop answer is vocabulary completeness (more registered ops,
more rolling combinators), not a faster interpreter. The
runaway-kernel guard (#61) is built — `LuaState::set_instruction_budget`
arms an instruction-count hook per protected call, and a kernel that
spends the budget fails with a loud error and leaves the state usable —
and keeps its scoping for who turns it on: required before Lua ships
in any surface serving untrusted input (the M7 served product); off by
default, and optional, for a local console, whose runaway kernel hangs
only its author.

## The vocabulary invariant covers window aggregates, not column functions

**The invariant's edge, stated precisely (found by the M4-close code
review, 2026-07-28).** It covers window aggregates, not *column*
functions: the host-function seam a script calls through returns one
value per call, while a column function returns a whole column, so
there is no shape to install one under — `register_column_function`
and the console's `.luascalar` are SQL-callable but resolve to nil
inside a kernel. This is a real edge, not a wiring slip, and the docs
claimed it closed until the review caught them. A script wanting
whole-column work has the vectorized vocabulary (operators,
`rolling_*`) instead; widening the invariant needs a column-shaped
host seam, which is where #77's script-side primitives would land.
(That is #77's own second tier — unrelated to #83's "tranche 2".)

Promotion is mechanical:
one registry name, a Lua implementation swappable for a trait
implementation with no query change (both pinned by contract tests).

**A history correction (same review).** The four curated statistics
were *not* produced by promoting Lua prototypes — the regressions
predate the Lua layer by two milestones. The promotion ladder is the
intended path for future kernels, not the origin story of the shipped
ones; documentation must not claim otherwise.

## Corrections get a default-on ingest-sequence column and one `ASOF` keyword

**M4.3 The corrections design cycle — ruled 2026-07-28, closed.**
F2 is **(a) whole**, and its three sub-decisions are settled: the
ingest-sequence column is **default-on** for every table (one
solution for arrival order, the `AS OF` coordinate, and the
ready-at-hand stable id; the virtual-until-divergence design makes
it nearly free — store nothing while sequence == row id, materialize
delta-coded from the first divergence); the retention horizon is
**unbounded by default** with a per-table bound available; and **one
keyword — `ASOF` — with structure dispatching** (amended by ruling,
2026-07-28): followed by `JOIN` it is the event-time nearest-match
join (the DuckDB/ClickHouse/QuestDB/Snowflake spelling, unchanged);
followed by a sequence it is knowledge-time travel
(`FROM trades ASOF 41520`). Why one word: the join side has a
universal convention worth honoring while the travel side has none
(Oracle `AS OF SCN`, Delta `VERSION AS OF`, Snowflake `AT` — no
consensus to diverge from), and two-word `AS OF` collides with
SQL's alias grammar (`trades AS OF` = "trades renamed OF"). Riders:
the SQL:2011 long form `FOR SYSTEM_TIME AS OF n` stays accepted
(sqlparser parses it natively; it is the internal carrier), and a
textual teaching error catches two-word `AS OF <n>` before the
parser garbles it — the error message itself teaches the two axes
(join = event time on the ordering key; travel = knowledge time on
the ingest sequence). Mix-ups cannot yield wrong semantics: the two
uses need different surrounding syntax, so confusion is a loud
error, never a silently wrong answer.

## What the M4-close reviews found, and the lesson taken

**M4-close reviews (2026-07-28)** — three independent repo-wide code
reviewers, every finding reproduced before its fix. What they found
is worth recording, because it says where this milestone's risk
actually sat: two of the three correctness bugs were in *text
handling and routing*, not in the storage machinery the milestone
was about. The `ASOF` pre-pass reassembled statements by joining
tokens with spaces, so a `--` comment silently swallowed the rest of
a query — the precise failure the clause's own ruling said could not
happen; whether a query was accepted depended on how many segments
its rows occupied; two embedder-facing paths could abort the process
(an unreserved Lua stack push, and embedder code called without
`catch_unwind`); a supersession at coordinate 0 wrote commit evidence
indistinguishable from a plain delete; and absent zone maps
*falsified* pruning against the invariant stated in two other files.
The reviews also caught this document and several crate docs claiming
a vocabulary invariant wider than the code holds. Lesson taken:
hand-rolled pre-parse text manipulation deserves the same adversarial
testing as the format code, and a "sound over-approximation" needs a
test that a *missing* input cannot flip it.

## The roadmap beyond native GA, and the order it was built in

**The roadmap beyond native GA (recorded 2026-07-27; reordered
2026-07-28, twice — first the desk before the browser, then the
extension model before the desk: the 2026-07-28 review rulings touched
M0–M3 design, and the back-end must settle before anything user-facing
is built on it).** M3 ships *embed in your application* plus the
console. **M4 (the extension model + corrections)** makes the
2026-07-28 rulings real — the back-end settles before anything
user-facing builds on it. The plan of record, approved 2026-07-28:

- **M4.0 Trait exposure** — `WindowAggregate` and `Registry`
  re-exported, `Table::register_window` + `Database::register_window`
  public, the ~20-line embedder kernel as a doctest.
- **M4.1 The feature gate** — compute-lua becomes a non-default
  feature the console enables; CI builds and tests both legs;
  sanitizer/apicheck jobs run in the on-leg only.
- **M4.2 The Lua front-end** — Lua as a thin front-end over compiled
  ops, on the architecture NumPy proved (a slow interpreter is fine
  when the loops live in compiled code and scripts only compose): the
  vocabulary invariant (every registered *window aggregate* is callable
  from Lua by its SQL name — registry-driven, so future natives flow in
  for free; column functions are a second namespace and do not cross,
  see below), the vectorized
  whole-column kernel slot wired (`eval_column`, built in M2.7 and
  never connected; likely closes #53), the compose-don't-loop idiom
  documented, promotion made mechanical (one registry name, Lua
  implementation swappable for a trait implementation with no query
  change).
- **M4.3 The corrections design cycle** — see *Corrections get a
  default-on ingest-sequence column and one `ASOF` keyword*.
- **M4.4 The corrections build** — the hidden ingest-sequence column
  (the permanent knowledge axis; delta-coded to almost nothing while
  uncorrected), retaining compaction (history segments), the knowledge
  mask (the live mask's analog), the `AS OF` predicate. Oracle: DuckDB
  re-deriving as-of answers over an explicit history table — emulation
  in the referee only, never in the product. Format additions ride the
  one manifest revision shared with F3's zone-map lift (sections
  reserved now, filled by whichever lands first).
- **M4.5 The correctness batch** — #73 (the atomic mutation commit
  record: old-or-new for crashes and readers, recovery
  auto-completes), #63 (Miri in CI), #69 (the upstream Lua test
  suite), the review-noted redundancies.
- **The Lua trial** — ruled 2026-07-28, **pass** (see *The Lua
  layer*): the Agent brought the evidence brief (#76), the Human
  ruled Lua stays. The sunset clause dissolved.
- **M4.6 SQL-in-Lua (#70)** — built on the pass: driver scripts
  (`query`/`append` through the `ScriptHost` seam, the console's
  `.run`), evidenced by an end-to-end SQL → Lua → SQL differential in
  the CI Lua oracle.
- **M4-close reviews (2026-07-28)** — see *What the M4-close reviews
  found, and the lesson taken*.

**M5 (desk adoption)** then builds what the target user needs, chosen
by the moat test: multi-factor curated compute (K > 2 — the recorded
LAPACK-class-returns trigger firing; **re-ruled 2026-07-30**: this
goes to MatLua in the Lua tier rather than to a faer dependency of our
own, a ruling that **stands** — bridged 2026-08-03 by an interim
in-engine solve (F2(c) on #90) until MatLua's endpoints land — see the
decision record *where non-standard compute lives*, below), the ordered-axis
dividends (cross-sectional partitioning, time bucketing — F1 ruled (d)
2026-07-29, monotone ordering-key arithmetic in `GROUP BY`,
`LAG`/`LEAD`, `RANGE` frames, the `ASOF` join — **all built
2026-08-01**, see *The M5 ruling batch* for the rulings and the
stdlib table for what shipped), segment-lazy open (F3), cross-process
readers (F4 —
**built 2026-07-29**: read-only opens over a live writer's directory
see the durable prefix consistently, old-or-new per mutation;
`tallydb DIR --read-only` with `.refresh`/`.flush` is the console
half; #42's codecs landed the same day),
and reach (bulk Arrow ingest, a Python binding with host-callback
NumPy kernels — distribution ruled 2026-07-29: wheels on PyPI for the
binding; the console stays a Python-free single binary). **M6 (WASM parity)** adds *embed
in a browser*: the compute stack already compiles for wasm32; the
remaining work is a browser `StorageBackend` (OPFS/IndexedDB behind
the existing trait — written knowing an HTTP-fetch sibling comes
later), the JS bindings, and `lua.wasm` behind the same feature flag.
**M7** adds *embed in a server*: a Servette-shaped served product and
a workbench UI, both **separate artifacts embedding the engine** (the
never-a-server guardrail's sanctioned form), the console module reused
as the server's shell. The load-bearing observation for M7's sync
story: **segments are immutable, self-describing, CRC'd objects
committed by a generation manifest — the storage format is already
the replication format.** A read-only browser or client replica
fetches the manifest and pulls segments lazily (zone maps prune the
fetch), verified by the same checks reopen runs; the single writer
stays wherever the WAL is. Nothing earlier may foreclose this.

## The console is a thin skin over a reusable `Console` (#39)

**Decision record — the M3.5 console (#39, ruled 2026-07-27).** The
shell / security / systems separation is the architecture: the engine
(systems) stays dependency-clean; the `shell` module's `Console` is
reusable — a future served product embeds it; `bin/tallydb.rs` is a
thin skin. Rulings, each with its losing alternatives: **dependencies** —
rustyline and csv only, confined to the console's `cli` feature (zero-dep
hand-rolling rejected as reinvention; a CLI framework rejected as
surface without need). **DDL grammar** — `BIGINT` / `DOUBLE` / the
coined `KEY`, one `ORDERING KEY` column constraint; `VARCHAR`/`TEXT`
refused with the keys-are-interned-labels teaching error rather than
aliased (an alias teaches the wrong model); user-typed `PRIMARY KEY`
refused with its own teaching error (the ordering key is not a
uniqueness constraint — duplicates are first-class), while serving
internally as the parser's carrier for the `ORDERING KEY` phrase.
**Import** — CSV in the shell layer feeding the ordinary append path;
the engine never parses CSV. **Code registration** — explicit
dot-commands only, and there are three: `.lua` and `.luascalar`
register kernels typed at the prompt, and `.run FILE` executes a
driver script from a path the user names. What they share is the
property that matters: code enters through a channel the user typed
deliberately, never through data. `CREATE FUNCTION ... LANGUAGE LUA`
is deliberately *not* SQL, so a SQL string — which may be built from
user input — is never a code-injection vector; the SQL form is a
recorded decision for the served product's threat model, not before.
(`.run` reads a file, so it inherits the console's trust in the local
filesystem, the same trust `.import` already needs.) Local security posture: an OS file lock
(released by the OS on death — no stale locks) admits one process per
directory; table names stay identifiers (they become directory names).

## No LAPACK on the query path

**Decision record — no LAPACK on the query path (2026-07-27).** The
`compute-lapack` crate is removed and the engine links no LAPACK routine.
The reason is a measurement, not a preference: **LAPACK's value scales
with parameter count, not data size**, and every statistic the engine
currently exposes is two-dimensional. A two-parameter least squares and a
2 × 2 symmetric eigenvalue both have exact closed forms costing a handful
of flops, while a general solver is dominated by its own per-call
overhead at window scale — measured at roughly 2.3µs of `regr_slope`'s
2.5µs per 64-row window, and 0.68µs per window for `dsyev` on a 2 × 2.
Replacing both with closed forms left `regr_slope` costing the same as
the other two-pass window statistics, where it had cost ~11× more, and
moved it from 5× behind DuckDB's `regr_slope` window to **3.3× ahead**
(`m2_compute_latency_bench.py`, run 2026-07-27, release, container
hardware, 20k rows, window 64).

*What this bought beyond speed:* the engine no longer requires a system
LAPACK to build or embed, which repairs the link-it-in-like-SQLite
property, and **the WASM milestone (M6) no longer waits on a LAPACK-in-WASM layer** — the
current feature set can reach WASM parity without one.

*The closed forms are the corrected ones, and that distinction is
load-bearing.* The rolling regression uses the corrected two-pass form
(Chan–Golub–LeVeque), which carries `Σ(x − x̄)` rather than assuming
centering left it exactly zero. Measured against the SVD answer before
the removal (`measure_closed_form`, release, container hardware,
2026-07-27), worst predicted-y drift over the data:

| design | QR (`dgels`) | corrected | naive |
|---|---|---|---|
| 64-row window, x offset 1e9 | 1.07e-14 | 2.84e-14 | 8.31e-7 |
| 64-row window, x offset 1e12 | 1.42e-14 | 2.49e-14 | 1.01e-3 |
| near-degenerate, spread 1e-10 | 2.84e-14 | 1.99e-13 | 6.75e-7 |

The corrected form tracks QR within a small constant factor — both at the
float noise floor against `|y|` of order 10–200 — while the **naive** form
is a real regression at timestamp-scale offsets, bug #45's regime. That
comparison needed LAPACK as its reference and cannot be re-run now that
the dependency is gone; what guards the property going forward is
`regression_numerics` in `engine`, which checks the shipped form against
a cancellation-free reference computed about `x[0]` and asserts the
correction beats the naive form on irregular offset data.

*Reopen trigger:* the first committed op needing **more than two
parameters or two dimensions** — multi-regressor regression, PCA beyond
2 × 2, portfolio solves, Cholesky. No closed form exists there, and a
solver-class backend comes back behind the same capability-negotiating
trait shape — the measured candidate is faer's solver family, which beat
reference LAPACK's `dgels` at k = 2–4 in the same-run three-way
measurement (see the kernel decision record below) and compiles to
wasm32. The engine's ops should then dispatch on parameter count:
closed form at two, a solver above it.

## System BLAS is replaced by pure-Rust kernels

**Decision record — system BLAS replaced by pure-Rust kernels
(2026-07-27).** `compute-blas` (system BLAS via FFI) is now
`compute-linalg`: the same trait seam, implemented in pure Rust — a
strict left-to-right loop for `dot`, faer (slim features: no thread
pool, no RNG, no file formats) for the matrix products. A three-way
measurement (plain Rust vs system reference BLAS/LAPACK vs faer,
release, container hardware, run 2026-07-27) drove the split: at window
scale (≤ 64 elements) the plain loop beats both libraries — there is
nothing to amortize — while faer wins long dots by 2.4–4.7×
(256–4096 elements) and Gram-shaped products by 3.7–10× (k = 4–16 over
64 rows), shapes where reference `dgemm` also loses to it. Accuracy is
a wash: identical least-squares residuals, eigenpair residuals at the
1e-16 floor on both sides. What the swap buys is packaging: no system
math library to install, link, or version (CI drops `libblas-dev`;
`ldd` on the engine library shows none), and the whole compute stack
compiles for `wasm32-unknown-unknown` — verified — so `blas.wasm` is no
longer a TallyDB dependency at all. The `dot` loop is the one kernel
whose result is bit-identical on every CPU and target, and deliberately
the only kernel on a per-window path today.

*Rejected alternatives:* keeping system BLAS — wins no measured shape
the engine runs and costs the system dependency; reopen if a platform
BLAS (Accelerate, MKL) is measured materially ahead at a shape that has
reached a query path. OpenBLAS pinned from source with
`TARGET=SANDYBRIDGE` — the old determinism plan; heavier to build,
still a C dependency, and source-fixed Rust loops achieve the property
more directly for the paths that promise it. *Reopen trigger for the
dot split itself:* profiling showing a long-vector dot on a hot path —
then the loop yields to faer above a measured length threshold, trading
bit-portability knowingly.

## Durability is a sidecar WAL with sync levels (#43)

**Decision record — durability: WAL with sync levels (#43).** A
sidecar write-ahead log (the segment format untouched), three levels:
`Group(interval)` — the default, 100 ms — logs every append and
group-commits with an in-thread sync, bounding the loss window at the
interval for +0.4–1µs on a ~1µs append (measured; the in-repo
`measure_wal_regimes` re-earns the number: off 0.99µs, group-100ms
2.06µs, full 728µs per append, run 2026-07-27, container fs); `Full`
syncs every append — zero window at ~700× per-append cost, shipped
documented, never default; `Off` writes no log and restores the
flush-boundary contract for replayable upstreams. Replay recovers the
per-record-CRC clean prefix, skips segment-covered rows, and ignores
wrong-generation logs (compaction reassigns row ids). *Rejected:*
flush-boundary-only as the GA contract (strangers assume a database
keeps what it acknowledged) and per-table dual contracts with no
default answer. *Reopen trigger:* tail-latency complaints from the
unlucky append paying the in-thread sync (10–46 ms worst on the
measured disk) — the fix is a background sync thread, which is the
may-the-library-own-a-thread question shared with #44's deferred
time-aligned freezing.

## The freeze threshold speaks bytes (#44)

**Decision record — freeze threshold in bytes (#44).** The knob
speaks bytes (the buffer bound an embedder budgets); numeric-or-key
makes rows fixed-width, so bytes convert exactly to a per-schema row
count at construction (8 per number, 4 per key code; dictionaries —
bounded by distinct values — sit outside the bound, documented).
Setting rows and bytes together is refused loudly. *Deferred with
triggers:* time-aligned hybrid freezing — pruning-profile evidence
from the end-to-end suite (#52), and the library-thread question
above.

## A built-in ships only if the differential-oracle set implements it

**The oracle-set rule for built-in functions (decided 2026-07-25).**
The inclusion principle bounds which *verbs* are in scope; this rule
bounds which *built-in functions we ship* on the SQL surface: **a
built-in joins the SQL surface only if the differential-oracle set
implements it** — DuckDB today, DataFusion when wired as the secondary
(which, since DataFusion is InfluxDB v3's SQL engine, also covers the
modern surface of our closest use-case neighbor). The rule does two
jobs at once: every admitted built-in is *born diffable* (it rides the
differential harness like everything else, rather than needing a
hand-built check), and every admitted built-in is *guessable* (users
find functions by knowing standard analytical SQL, which the oracle
set curates). Everything else TallyDB can compute — decompositions,
solves, anything matrix-shaped — is reached through the Lua
surface, where results need not fit SQL's scalar-per-cell type system.
The rule governs what *we* ship built-in; a user's own registered
functions are their code, named as they please. Applied at adoption:
`regr_slope` / `regr_intercept` / `covar_pop` / `corr` stay (all in
the oracle set); `eigen_max` leaves the SQL surface when the SQL-in-Lua
scripting API lands (#41) — it was an eigendecomposition amputated to
a scalar so SQL could return it, and it migrates to a SQL-in-Lua example
whose NumPy check becomes that example's differential test. Reopen
condition per function: a function that later becomes standard in the
oracle set becomes eligible here.

## The join constraint is completed: strategy is fixed by structure

**Decision record — the join constraint, completed: strategy fixed by
structure (Human-ruled 2026-07-26).** The size invariant above is one
clause of a two-clause principle: **a join is supported when a structural
property of its inputs fixes the execution strategy — never by cost.**
Each admitted strategy is guarded by the structural fact that makes it
safe at any scale without estimation:

| Strategy | Guarding structural fact | What it protects against |
|---|---|---|
| Broadcast/hash lookup | one side small enough to materialize, key-unique | unbounded memory |
| Ordered merge (`ASOF JOIN` and relatives, #65) | both sides ordered on the join key — their declared ordering key | unbounded memory *and* sorting |

**Build note (2026-07-30, M5.2): what shipped is clause 1's shape, not
clause 2's.** The as-of join as built indexes the dimension side in
memory — one ascending `(clock, row)` list per key — and binary-searches
it per fact row. That is correct at any ordering (the index is stably
sorted, so a late-arriving quote still matches), and it needs no
`is_ordered()` gate; but it materializes the dimension, so it is
guarded by clause 1's size invariant, not clause 2's streaming
property. The co-walk clause 2 describes — a cursor per side, memory
independent of both inputs, licensing large ⋈ large — is **not built**:
`execute_join` materializes both sides up front, and making the
dimension streaming is the same work as streaming scans generally
(#88). Until then, the ordered-merge row above states a *design*, and
the as-of join's actual reach is "dimension fits in memory". Tracked as
#92; the trigger to close it is a quote history that does not fit.

Clause 1 is the size invariant, unchanged, guarding the strategy that
materializes a build side. Clause 2 needs no size bound as a *property,
not an exemption*: a merge over inputs already clustered on the join key
is a streaming co-walk — a cursor per side plus the current match window
— so its memory does not scale with input size; that is exactly why it
may admit large ⋈ large, and why only on the ordering key, where the
storage layout guarantees the clustering (assumption 2 plays for clause
2 the role the size invariant plays for clause 1). Its runtime guard
would be `Segment::is_ordered` — the check the window executor already
relies on; a transiently disordered table (UPDATE reappends before
compaction) would refuse the merge loudly rather than serve a wrong
answer. (Design, not code: per the build note above, the co-walk is
unbuilt, and what shipped needs no such gate.) Dispatch
never estimates: an embedded single-machine snapshot knows every table's
**exact** row count and every dictionary's exact cardinality, so "small
enough" *can* be a measurement rather than a cost model — the
fixed-strategy planner stays fixed. That measurement is **not yet
enforced**: `execute_join` materializes both sides unconditionally and
no size check refuses an oversized dimension, so the size invariant is
today stated, not checked. A threshold belongs in the contract when the
executor generalizes; the as-of join riding clause 1 (#92) is what
makes it start to matter. A join with *neither* guarding
fact — two large tables on a non-ordering key, or join-*order* search —
is **refused loudly, naming the missing structure**: serving it needs
spilling, partitioning, or an optimizer, i.e. a different product.
*Precedent (validates the workload, no vote on the how):* kdb+ dispatches
by user-named join verbs trusting declared structure (`lj` keyed lookup,
`aj` sorted as-of) with no optimizer anywhere; QuestDB keys `ASOF`/`LT`/
`SPLICE` off the schema-declared designated timestamp — the closest
living relative of this rule; ClickHouse ran years of the world's largest
analytics on exactly clause 1 (hash join, right side fits memory, join
order = syntax order) before generalizing into CBO — the reopen path,
visible; DuckDB implements `ASOF JOIN` as first-class syntax over a CBO,
which we refuse, but it standardizes the surface and serves as the
differential oracle, so the ordered-merge family is born cross-checked
(the oracle-set rule holds).

## NULL crosses to Lua as a sentinel, not `nil` and not NaN

**Decision record — NULL across the script boundary (2026-07-26).** NULL
crosses to Lua as a distinct **sentinel value** — a `pd.NA`-style
singleton — not as Lua `nil` and not as NaN. *The principle that decided
it:* a non-editing round trip across any of the three boundaries
(DB↔SQL, DB↔Lua, Lua↔SQL) must preserve everything, which requires each
value mapping to be a total, invertible function. `nil` is not total —
inside a Lua table a `nil` *deletes* the slot, destroying both the value
and the row's structure, and that failure cannot be prevented inside
arbitrary user scripts. A distinct sentinel is a real value that survives
in a table, so the mapping stays total and faithful; it is kept distinct
from NaN (a computed value — see *Null, NaN, and ordering*) and
propagates over both numeric subtypes, so `i64` exactness holds. This was
chosen by a bake-off spike against the `nil` alternative, not by
argument.

*The cost we do not pay:* the prior art that reached the same conclusion
(Tarantool `box.NULL`, OpenResty `ngx.null`, pandas `pd.NA`) pays a real
price — the sentinel is truthy (`if x` is true for a null) and `x == nil`
silently misses it. Those systems carry data *as language values*, so
every field access meets the sentinel and its footguns. TallyDB does not:
columns cross as zero-copy views over engine buffers, and compute is
batch, not per-row. Null-aware batch ops (`v:sum()`), an out-of-band
validity view (`v:mask()`), and the curated compute ops consume
`(buffer, validity)` engine-side and never materialize the sentinel. The
footguns are real but confined to the discouraged manual per-element
path; the common path never meets them. A sentinel is the faithful
representation you rarely have to handle — an advantage that follows
directly from being a zero-copy, compute-in-engine store rather than a
value-shaped one. (One honest limit: relational `<`/`<=` against the
sentinel is a loud error, because Lua forces those operators to a
boolean and three-valued logic cannot propagate through them; arithmetic
propagates to the sentinel.)

## The value map: declared return types, exact-or-loud coercion, keys as codes

**Decision record — the value map: return types, coercion, keys
(2026-07-26).** Three conventions govern how a script's results cross the
typed boundary, each chosen from TallyDB's own invariants (numeric-or-key,
exact-or-loud, zero-copy, the fixed-strategy planner), not from any one
precedent — the outside systems that solve the same problem validate the
*workload*, they do not get a vote on the *how*.

- **Return type is declared at registration and resolved at plan time** —
  never inferred from the value a call happens to return. A Lua-backed
  function names its result type (`f64` / `i64` / `key`) when registered;
  the planner fixes the output column's type from that, so a query yields
  the same Arrow schema on every run. Inferring per call would make the
  output type *data-dependent* (Lua silently floats integers) — the
  dynamic-typing property the fixed-strategy planner exists to exclude,
  and the root of bugs B4/B5 (#54). Every statically-typed peer declares
  it; the choice follows from our own static schema, not from theirs.
- **Coercion is exact-or-loud.** A Lua `integer` fills an `i64` and a
  `float` fills an `f64`; a `float` may fill an `i64` *only if it is
  losslessly integral*, otherwise it is a loud error — never a silent
  truncation. A Lua `boolean` maps to `i64 {0, 1}`: Booleans are a
  transient value, never a third column type (the numeric-or-key
  invariant holds). `nil` is NULL (the sentinel above). A `string` is
  interned into the output key dictionary — the only way a script
  produces a key. This closes B6 (#54).
- **Keys read as codes, with lazy text.** A key element reads as its
  integer dictionary code, so equality, grouping, and membership stay
  integer-cheap — which is *why* keys are dictionary-encoded, and what
  keeps the read zero-copy (the code is in the buffer; a string is not).
  `v:text(i)` decodes on demand; `v:code_of(literal)` resolves a literal
  once (the once-per-distinct-value pattern `WHERE` already uses), so
  `key == literal` is an integer compare. Codes are per-segment (#6); the
  engine guarantees a script sees one consistent code space per call
  (per-call or query-lifetime-remapped), so a raw code is never compared
  across segments.

Together with the NULL sentinel above, this is the frozen value-map
contract for the Lua boundary. The ergonomics layered on top — batch
reductions, `v:mask()`, `v:get(i, default)` — are additive and do not
change it.

## A Lua-backed function is a vectorized UDF called once per segment

**Decision record — the calling convention (Option A, 2026-07-26;
Observed).** A Lua-backed function is a **vectorized UDF**: the engine
calls it **once per segment**, handing whole columns as zero-copy input
views and — for the scalar/elementwise slot — a preallocated zero-copy
output view; the script loops the column *inside Lua* and writes the
output column. The window slot is the same shape one level in: the engine
drives the framing and the script reduces one frame to one scalar, exactly
the `regr_slope` / `eigen_max` pattern already shipped. Arguments are
**positional, with each argument's kind (column vs scalar constant)
declared at registration** alongside the return type (the value map
above), so the engine binds columns as views and constants as plain Lua
numbers. This is the batch-not-per-row rule made concrete — one boundary
crossing per *(function, segment)*, never per row. *Rejected:* **inline
Lua in the SQL text** (code in query strings has no registration to
declare a return type on, fights app-registered kernels, and recompiles
per call) and **a single uniform batch object** (it hands frame control to
the script — the one thing the measurement says to keep engine-side — and
buys only the table-valued / multi-output slots #47 already defers).
Evidence (`values_map_spike`, release, 4,096 rows): the vectorized call
produces its output column in ~518µs against ~12.7ms for per-row
invocation of the same kernel — a **25× penalty avoided**, the same
crossing tax the feed-reactive ruling excludes. The ~120× over a native
Rust loop is the interpreter / metamethod cost the promotion ladder
closes, not a property of the convention.

## Scripts get one host-routed diagnostic function, `log()`

**Decision record — script observability: `log()` (2026-07-26).** Scripts
get one host-routed diagnostic function, `log(...)`, the replacement for
Lua's `print`. `print` is removed *not* because of the string invariant —
its text never becomes a column — but because its **destination**, the
process's stdout, is not an embeddable library's to own and is
uncapturable. `log(...)` routes instead to an **embedder-installed sink**:
a trait the host implements — the shell wires it to stderr, a library
embedder to its own logger, a headless embedding to a no-op (off by
default). It is a **pure side-channel**: no return that feeds results, it
cannot change query output — observational only (a diagnostic that alters
the answer is not a diagnostic). It logs scalars and short text; a **view
logs as a summary** (`f64 view, len 4096`), never its contents — a
diagnostic, not a buffer dump or an exfiltration path. Surface and sink
are both **flat** (`fn log(&self, msg)`; Lua `log(...)`), single severity:
the sink is script-only, so a level parameter would be permanently
degenerate (every message an `info`), and TallyDB carries no speculative
machinery. Severity, if a real need appears, is added *additively later* —
a defaulted `log_at(level, msg)` trait method breaks no existing embedder
(and before 1.0 nothing is frozen) — or the sink is deliberately widened to
an engine-wide diagnostic channel, a named scope expansion rather than a
default. Runaway volume (a kernel logging per row) is bounded by the
instruction-count hook (#61) and the batch-not-per-row doctrine: log per
batch, not per row. Because the sink is host-captured, an agentic harness
driving the engine reads a script's log as its observable output — the
"sight" half of #46.

## Feed-reactive compute is in scope at batch boundaries, never per row

**Decision record — feed-reactive compute (settled direction, 2026-07-26;
implementation M3+).** Reacting to new ordered data with compute is *in
scope* — it is a TSDB-native pattern, not a general-DB frill, and kdb+
(the reference workload) is built on it. The admitted shapes are **ingest
hooks** — a kernel invoked at a *batch* boundary inside `append()`
(segment freeze / batch land), compute-only, app-registered — and
**continuous queries** — derived data kept fresh on ordered append,
reduction via a kernel, with a restricted append-friendly incremental
scheme (details at implementation). Per-row *semantics* are available by
looping a batch inside one kernel call; **per-row hook *invocation* is
out** — measured at ~27× the batched loop (near-pure `pcall` crossing
tax, `values_map_spike`), and against the batch-not-per-row rule, the
append fast path, and columnar execution. Also out: **catalog-persisted
stored procedures** (never-a-server; code in the catalog is not
numeric-or-key — app-registered named kernels, which *are* the Lua-in-SQL
model, stay in) and **network delivery / push** (the app delivers; the
engine detects). Recompute-on-demand is not a reactive feature — it folds
into plain invocation. This is the standing **kdb+-as-floor** principle in
action: kdb+ sets a soft feature/perf floor at the *user* POV
(meet-or-exceed unless it conflicts with an invariant), and its own model
is exactly the batch ingest hook (`upd`), app-registered, no per-row
triggers, no catalog stored-procs — the same shape, delivered in-process
rather than via a multi-process server. Interacts with #44 (segment
freeze = the hook point), #49 (continuous-query SQL surface), and the
#41 / #47 kernel contract.

## A cut is forced by an assumption, and taken to its endpoint

**Two defaults govern every cut, each overridden only by a stated
use-case assumption (the cut-depth principle, 2026-07-25).**

1. *General-purpose by default.* The engine keeps a subsystem until an
   assumption forces its deletion — no cut without a licensing
   assumption.
2. *Clean (endpoint) by default.* Once an assumption licenses a cut,
   take it to its endpoint — the maximal deletion — unless a further
   assumption justifies stopping short. A partial cut or a kept surface
   is itself a decision that must name the assumption permitting it:
   keeping `UPDATE`/`DELETE` (corrections happen), keeping broad SQL
   (analysts want it), tolerating out-of-order ingest (reinserts arrive
   out of order) are all such justified deviations.

Two qualifiers keep this honest. *Tidiness is not a default to trade
against — it is hygiene:* whatever depth a cut lands at, it carries no
residual machinery from the deleted subsystem and leaks the deleted
concern nowhere. A partial cut is allowed; a leaky one never is. And
*scrutiny scales with reversal cost:* prefer the endpoint, but hold a
foundational cut's licensing assumption to a far higher standard than
an additive one's — being wrong about the foundational cut costs a
rebuild, being wrong about the additive one costs a later layer.

## The working set cut is owned: the queried rows fit in memory

**Working set — decided (2026-07-25): the cut is owned.** Version 1
opens a table by decoding it into memory, and this is now a stated
commitment, not drift: the licensing assumption is that *a table fits
in memory* at the scale this engine targets. The banked simplification
is real (no buffer pool, no page cache, no partial-read machinery; the
mmap/ranged-read follow-up notes are retired). Reopen trigger, stated
because this cut is foundational-class: the M3 benchmark suite's
startup-time and footprint results embarrassing open-time on
realistic tables. The recorded escape is a zero-copy-open format
version — additive under the append-only version and codec registries.

## There is no boolean type, and none is coming

**Truth values — decided (2026-07-25).** There is no boolean type and
none is coming. A flag column is `i64` in {0, 1} — which is the right
answer, not a workaround: `SUM(flag)` is a count and `AVG(flag)` is a
duty cycle. When computed expressions land in projection, a projected
comparison yields `i64` in {0, 1}. `bool_and`/`bool_or` are not
offered (`MIN`/`MAX`/`SUM` over the flag serve). Recorded now so a
third type cannot arrive as an implementation detail of the
arithmetic-projection commit.

## The join constraint is a size invariant, not a modelling shape

**The join constraint is a size invariant, not a modelling shape
(restated 2026-07-25).** What execution requires is that the build
side be small enough to materialize; "star schema" was the use case
wearing the invariant's clothes — the same correction as
time-vs-ordered. The rule is: **one large table joined against tables
small enough to materialize.** This preserves every current behavior,
and it *licenses* (as ordinary future todos, not scope fights) shapes
the modelling name wrongly excluded — join chains against several
small tables, snowflakes, self-joins against small aggregates — while
naming the real hazard: a nominal "dimension" grown too large to
materialize. A stated threshold belongs in the contract when the
executor generalizes.

## Predicate reordering is not forbidden by the no-optimizer cut

**The access-path cut licenses the planner cut.** A planner exists to
choose among access paths; with exactly one path there is nothing for
a cost model to decide, so the absence of an optimizer is structural,
not a bet about workload simplicity. Corollary, to prevent an
over-broad reading of the settled "no optimizer": **predicate
reordering is not forbidden** — evaluating the cheapest, most
selective predicate first is a heuristic needing no statistics beyond
existing zone maps, and remains available. Likewise, if within-segment
scan acceleration is ever wanted, the sanctioned shape is block-level
min/max summaries at the codec's block granularity (small materialized
aggregates — decades old, patent-safe); per-value order-preserving
lossy codes should be patent-checked before any implementation, and
their value lies in unclustered data, which assumption 2 removes.

## Live data means in-process analytics, never server-side fan-out

**Live data, precisely.** The Distribution and Deployment cuts draw the
line through the middle of the phrase "live feed," so a reader comparing
TallyDB to a tickerplant must read them together. *Freshness is not
sacrificed:* a query snapshot includes the live write buffer, not only the
frozen segments (`Store::snapshot` appends the buffer's rows to the
segment sequence; the contract is that a snapshot covers exactly the rows
appended before the call), so a row appended microseconds ago is visible
to the very next query. Freeze/flush is the *durability and layout*
boundary for segments; power-loss durability of the newest rows is the
WAL's, at the configured sync level (#43) — neither is a visibility gate. So an application that ingests
a live feed and recomputes over it in the same process — real-time risk,
live P&L, a moving regression on the newest window — is squarely in scope,
and is the compute-without-copying sweet spot: socket → storage → SQL →
curated compute with no serialization hop. What is *out* is being the tick **server**:
one process streaming ticks over the network to a farm of subscriber
processes, which the never-a-server (Deployment) and no-network-boundary
(Distribution) cuts forbid outright. Stated as a single rule: "live feed"
as *in-process analytics over freshly-landed data* is in; "live feed" as
*server-side publish/subscribe fan-out* is out. The reactive-compute shape
over that fresh data — batch ingest hooks and continuous queries, per-row
invocation excluded — is the *feed-reactive compute* decision record
below.

## `RANGE` frames are correct before they are incremental

**`RANGE` frames, and what they cost.** Every `ROWS` frame is a
trailing row count, uniform across the column, which is what lets
`WindowAggregate::evaluate_frames` slide one add and one remove per
step. A `RANGE` frame is bounded by ordering-key *value*, so its width
varies row to row — and, because standard SQL ends such a frame at the
current row's **last peer**, it is not even trailing in row-index
terms: a frame can extend forward over rows sharing the current row's
key. The executor therefore computes explicit `(start, end)` bounds per
row (one O(rows) pass, both pointers monotone) and hands them to
`WindowAggregate::evaluate_bounded_frames`, whose default recomputes
each frame. That default is why `RANGE` is correct for every
aggregate, including embedders' and Lua kernels', the day it ships.
What it is not yet is *incremental*: an aggregate with sliding state
can override that method with a two-pointer sweep, and until it does,
a wide `RANGE` frame costs O(rows x window) where the equivalent
`ROWS` frame costs O(rows). Tracked, with the safety net that a
statistic whose override cannot hold its accuracy simply keeps the
default and stays correct.

## NULL is placed, not ordered; NaN is a value greater than every number

**Decided (2026-07-24): NULL is placed, not ordered; NaN is a value,
greater than every number, everywhere.** The engine's three-valued
predicate logic already put NULL outside the number line — a null
matches neither `x > 5` nor `x <= 5`, aggregates skip it, arithmetic
propagates it — and ordering says the same thing: nulls are not
compared but *placed*, after all values, in both sort directions.
Consequently `ORDER BY x DESC` is not the sequence-reversal of `ASC`:
within the values it is an exact mirror (total order guarantees it);
only the non-values stay put. That asymmetry is sound here because of
two premises, which are its reopen tripwire: the executor never
serves `DESC` by reversing an `ASC` result (each query sorts by its
own comparator, and there is no optimizer to introduce the shortcut),
and the one physically-ordered column — the ordering key — is `NOT
NULL` by schema rule. NaN, by contrast, *is* a value: computed, and
comparable under one relation used by sort, predicates, MIN/MAX, and
zone-map pruning alike — NaN is greater than every number and equal
to itself, while `-0.0 = 0.0` stays true (NaN lifted to the top, not
bitwise total order). The ascending ladder is *numbers… +∞, NaN,
then NULL off the end*. Pruning stays sound via a has-NaN bit in the
`f64` zone map (see `format.rs`). Rejected: nulls-as-largest/smallest
(they put absence *on* the number line for sorting while predicates
keep it off — one seam, two answers), and IEEE-strict predicates
(NaN invisible to every operator but `<>` while sorting as a value —
the trap this ruling closed). `NULLS FIRST`/`LAST` syntax is built
(M3.4). The choice was made from the numeric-or-key thesis,
not oracle convenience: where the SQL standard leaves semantics
implementation-defined, the choice is ours and the differential
harness normalizes.

## Decisions carry provenance, a tripwire, and a record

Three rules, adopted after a sweep of this project's own decision
history found the same defect twice (a codec fork framed from a 2015
paper and decided in 2026; an interpreter treated as settled because
early drafts named it):

1. **Option spaces carry provenance.** A decision record states how and
   when its options were assembled; a fork bounded by a moving field
   cites a check of current practice at decision time, not framing time.
2. **A tripwire for what must be surfaced.** A choice is a decision —
   not routing — when it freezes an external contract (bytes, API),
   sets user-visible semantics, or sets a product guarantee. These are
   surfaced to the architect even when discovered mid-pass, even when
   one option seems obvious.
3. **Settled requires a record.** A choice inherited from early drafts
   is not settled; settled means a record exists naming the
   alternatives that lost. Absence of a record means open.

Ratified as deliberate under rule 3 (2026-07-24): `SUM(i64)` stays
exact and errors loudly on overflow; query output is one Arrow batch
per segment; window frames are `ROWS`-only for now. Two of the three
have since been superseded by later work rather than reopened: M5.1
added `RANGE` and whole-partition frames, so the frame shape is no
longer `ROWS`-only; and batch count is not a contract — a plain scan
still yields one batch per segment, but the collapsing stages
(`ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT`, `HAVING`, `GROUP BY`)
materialize a single batch, as `QueryOutput`'s own documentation says.
The `SUM(i64)` half stands. The two sibling cadence
questions closed together, both ruled by the Human 2026-07-27 on a
measurement (recorded in #43/#44 and built in M3.2/M3.3):

## The interpreter is canonical PUC Lua 5.4, with hand-rolled bindings

**Decision record — interpreter and binding (2026-07-24).** Two
alternatives rejected, each with a reopen condition:

- **LuaJIT** (the original plan) — rejected. It is a fork frozen at Lua
  5.1: no native 64-bit integers (only `int64_t` cdata boxes, with
  different equality, hashing, and mixing semantics — a permanent seam
  through the scripting surface of a database that is careful about
  `i64` exactness everywhere else), and a permanent version skew
  against the WASM build's `lua.wasm`, which is Lua 5.4 (a fork of
  lua-aot, whose runtime is stock 5.4). Canonical 5.4 on both targets
  deletes the skew instead of managing it, and canonical-over-fork is
  this project's own thesis applied to a dependency. What LuaJIT
  offered — trace-compiled script loops and `ffi` raw-pointer access —
  is covered by the promotion ladder. Reopen condition: a real workload
  shows ad-hoc kernel performance is unacceptable *and* promotion to a
  native op cannot cover it.
- **`mlua`** (the safe binding wrapper) — rejected, including as a
  dev-only witness. It is neither canonical nor small (five Lua
  versions, serde, async, macro machinery — we would use a sliver), and
  the witness role does not survive inspection: diffing two bindings
  over the same vendored interpreter mostly tests the interpreter
  against itself, while a binding's real failure modes — stack
  imbalance, GC anchoring mistakes, a `longjmp` over Rust frames — are
  memory-safety violations that output diffing cannot see. Reopen
  condition: the C API surface we actually need balloons well past the
  ~two dozen functions the batch convention implies.

What ships instead: **hand-rolled thin bindings** to the 5.4 C API,
with the error discipline built in by construction — every entry into
Lua goes through `lua_pcall`; Rust functions called from Lua never
raise a Lua error across frames with pending destructors; and
`catch_unwind` at the boundary so a Rust panic never unwinds into C.
Verified with no binding dependency at all, using Lua's own enforcement
plus standard tooling: test builds compile the vendored interpreter
with `LUA_USE_APICHECK`, so the interpreter itself asserts on C API
misuse (the real oracle for binding discipline); seam tests run under
the official test suite's GC/allocation-torture infrastructure
(`ltests.c` — full collection on every allocation, injectable
allocation failure); the official Lua test suite runs against the
vendored build in CI; and ASan/UBSan cover the combined artifact. This
is the arrow-lite configuration — a frozen canonical spec *and* an
external oracle — the same pair that decided the hand-roll there (#2).
The AOT compilation path (lua-aot natively) is *not* adopted: our
ad-hoc scripts are unknown at build time, so AOT lands on the one part
of the design that cannot use it; it remains available later, at zero
semantic cost, for any precompiled script library we might ship.

## Rolling regression solves a centered factorization

**Decision record — rolling regression solves a centered factorization
(2026-07-24).** The design matrix is `[1 | x − x̄]`, never raw `[1 | x]`:
a regressor with a large offset relative to its in-window spread (a
timestamp-scale x) makes the raw pair catastrophically ill-conditioned —
measured on a 20-row window (run 2026-07-24, pinned as
`rolling_regression_survives_timestamp_scale_x`): the raw solve loses
the slope entirely from offset 1e9 while the centered solve holds
~3e-11 relative error through 1e15 (bug #45). The rejected default was
streaming sufficient statistics — O(1) per window slide and how DuckDB
computes `regr_slope` — because the running-sums formula squares the
condition number and degrades a thousand-fold earlier (five digits gone
at offset 1e6). It may return later as an explicit opt-in fast path
with its accuracy caveat documented; reopen trigger: profiling shows
per-window factorization dominating a real workload.

*The reopen clause fired, 2026-08-03 (#90), and the rejection needs
splitting in two to stay true.* What was rejected in 2026-07-24 was
**raw** running sums, and the κ² blowup that condemned them is the
LARGE-OFFSET case — which anchoring removes outright: the K > 2 build
maintains sums of deviations from a data-row anchor and lands 3.7e-15
at offset 1e12. Against a per-frame solve centered on a *computed
mean* the gap reaches nine orders of magnitude on long windows (spike
table on #90, 2026-08-03, K = 16 / W = 1024) — that mean being itself
a length-W summation carrying its own error — though at the 64-row
windows the guard runs the shipped per-frame path row-anchors too, so
the two sit together. So the old objection does not transfer to the anchored form.
What does transfer is COLLINEARITY: any normal-equations route pays κ²
there, anchored or not, and the K > 2 path measures 3.3e-5 on a design
whose third factor is nearly the sum of the other two. That is the
accuracy caveat this record asked to see documented, and it is
enforced rather than merely written down — near-dependent windows are
refused outright (see the tranche below). At K ≤ 2 nothing changed: the
closed form still ships, because a two-parameter fit solves exactly off
the same shifted moments — there is no linear system to form.

*Updated 2026-07-27:* the factorization is gone — the window solves in
closed form (see *Curated compute*) — and the centering it required
remains, now in **corrected** two-pass form in every window statistic
(`RollingRegression` and `PairStatistic` alike; the uncorrected form
carried up to 4.9e-8 relative error at a 1e12 offset). Accuracy is
judged against a compensated high-precision reference and enforced in
CI by `window_numerics_guard`: every shipped window statistic must
track that reference within 1e-12 relative over corpora spanning
offsets to 1e12 and a drifting monotonic ordering key — the shipped
form measures 1–2e-15. The streaming alternative was measured against
the same reference (`measure_3b`, release, container hardware): naive
running sums are ~8× faster and reproduce exactly the failure this
record predicted (`eigen_max` off by 9.6e7 at a 1e12 offset,
`corr`/`slope` undefined where the recompute has an answer) — rejected
permanently. A **shifted** variant — moments kept about a value near
the data, accumulator rebuilt every window-length — keeps ~7× and sits
at 5e-15–1.1e-14, marginally less accurate than the corrected
recompute but still at the noise floor.

*Shipped 2026-07-27 (#72).* The sequence seam exists — a defaulted
`evaluate_frames` on `WindowAggregate`: the executor hands each
aggregate one contiguous run (the snapshot, or one partition) and the
default recomputes per frame, so only overriders change behavior. The
rejected seam shapes, for the record: a separate sequence trait
(needless registry duplication) and executor special-casing of known
op names (breaks the trait boundary and duplicates the math — rejected
on sight). `PairStatistic` and `RollingRegression` override with the
shifted sweep for bounded frames; unbounded frames recompute as
before; one shared finalization keeps the NULL semantics identical on
both paths. The guard extended before the speed landed:
`window_numerics_guard` holds the incremental path — the one every SQL
window now runs — to the same 1e-12 bound on every corpus, intercept
included, and was verified to trip (1.07e-12, drifting-timestamp
corpus) with the re-anchoring rebuild disabled. Arrival numbers
(`m2_compute_latency_bench.py`, run 2026-07-27, release, container
hardware, 20k rows, window 64): `regr_slope` 0.6ms — 9.6× ahead of the
DuckDB+NumPy stack; `covar_pop`/`corr`/`eigen_max` 0.7–1.1ms —
1.2–1.6× ahead of vectorized NumPy riding TallyDB's own export, 3–4×
ahead of the DuckDB+NumPy stack. The in-engine path is now the fastest
measured arrangement for every curated statistic *and* the only one
holding 1e-12-to-truth at timestamp-scale offsets — the vectorized
peer's cumsum form is exactly this record's rejected streaming
algorithm.

## The honest zero-copy claim: vectors are free, a design matrix is one gather

**Decision record — the honest zero-copy claim (column-group arena
considered and set aside).** LAPACK wants column-major matrices in one
allocation with uniform stride; table columns are separate allocations. So
the zero-copy claim is stated precisely: **vector-shaped ops and window
slices are zero-copy into compute; assembling a multi-column design matrix
is one bounded gather** — an O(n·k) copy feeding an O(n·k²) solve, so the
copy is asymptotically invisible exactly where it would matter most. The
rejected alternative was a shared arena allocating a segment's same-length
`NOT NULL` `f64` columns at uniform stride so a table chunk *is* a matrix;
set aside because it couples `arrow-lite`'s allocator to `storage-lite`'s
segment layout and constrains compaction. Reopen trigger: profiling on
target workloads shows design-matrix assembly is a material fraction of
query time.

## `f32` is set aside, and kept cheap to add

**Decision record — `f32` (considered and set aside, kept cheap to add).**
A single-precision analytic subtype was rejected for now: 32 bits can never
hold the ordering key or money (`f32`'s exact-integer ceiling is 2²⁴;
`i32` nanoseconds span ±2.1 s), `f32` accumulation quietly loses
million-row sums and variance to cancellation, and the whole oracle
strategy (DuckDB, NumPy) speaks `f64`. What makes the rejection cheap:
the numeric subtype tag is an extensible integer registry, so adding `F32`
later is a new variant and buffer width — never a format migration. Reopen
triggers: a GPU/WebGPU compute backend actually lands on the roadmap (WGSL
has no `f64`, so there `f32` is the entry ticket), or profiling shows
bandwidth-bound, precision-tolerant workloads dominating real usage. The
adoption shape when triggered: per-op downconversion at the compute
boundary or an opt-in stored subtype — never for ordering keys or money.

## The numeric type is not a rational

The schema declares which flavor each numeric column is. We considered and
**rejected** making the numeric type a rational (`i64/i64`) and writing our
own integer linear algebra: rational denominators overflow `i64` within a
handful of divisions (a mean of ~4 returns already blows past the ceiling),
a bignum rational is variable-width and kills the fixed-width Arrow-interop
and SIMD story, and — decisively — rationals can't even *represent* the
irrational outputs (√, log, eigenvalues) the analytics produce.
Floating-point *done carefully* is the right tool; where reproducibility
matters, it is handled by fixing the operation order in source (see
*Numerical consistency*), not by dropping floats.

## Storage formats: row identity, key dictionaries, and the codec registry (#1, #6, #28, #30)

**Decided (issues #1 and #6, 2026-07-23), settling `storage-lite`'s
formats:** (1) **Row identity is kdb+-style pure append** — rows carry an
internal monotonic row id, duplicates are first-class, `UPDATE`/`DELETE`
address rows by predicate, and corrections supersede by ingest sequence.
The rejected InfluxDB-style `(key-set, ordering-key)` primary key
silently collapses distinct same-tuple events — data loss with no error;
if user-visible overwrite semantics are ever needed, the path is an
opt-in declared uniqueness constraint on top, not a reversal. (2) **Key
dictionaries are per-segment** — segments are fully self-contained, which
keeps immutability pure, compaction simple, and matches Arrow's per-batch
dictionary export; with identity resolved by row id, compaction never
compares key values across segments, which is what made a global
dictionary attractive. The recorded extension: a process-lifetime
code-remap cache at query time, added only when profiling shows the
remap cost is material. (3) **The format carries a per-column codec
tag** (issue #28) — a one-byte, append-only integer registry
(`0 = uncompressed`), same pattern as the frozen type-tag registries —
so every codec is an additive entry, never a format migration.
**Ordered `i64` columns use delta-of-delta** (issue #29), the TSDB
standard for clock-like keys, with a confirm-against-plain-delta
measurement on the corpus at implementation. **`f64` columns ship
uncompressed behind the tag** — a legitimate answer for hot data, not
a placeholder. **The general-`f64` codec is decided: ALP** (issue
#30, closed 2026-07-24 by argument over the published evidence
rather than an in-house A/B — sound precisely because the codec
registry makes the choice an additive tag, cheap to reverse). ALP
converts decimals-in-doubles to integers per vector
(frame-of-reference + bit-packing, verbatim exceptions, ALP-RD
fallback for true doubles), with losslessness enforced per value at
encode time; it leads the field on both of our weighted criteria —
decode throughput on the read path first, ratio second (encode runs
at freeze/compaction, off the hot path). Rejected: Gorilla and Chimp
(the XOR family's bit-serial decode cannot vectorize; Chimp remains
the named low-effort fallback if ALP's implementation cost vetoes
it), Elf (near-parity ratio bought with ~215× slower decode and a
global erase-and-restore correctness obligation), and zstd±byte-split
(float-blind, and a dependency where a hand-roll fits the registry).
**Built (#42, 2026-07-29)** — `storage-lite/src/alp.rs`, registry
tags 2–4: ALP with per-vector RD and raw fallbacks for `f64`
(the encoder computes all candidates and keeps the smallest, so it
can never bloat), and the integer sibling — frame-of-reference +
bit-packing — for non-clock `i64` columns and `u32` symbol codes.
The corpus ticks family rounds to pennies first, per the caveat.
Measured (release, seed 42, 1M rows/family, 2026-07-29,
`measure_42`): ticks prices **4.18×** vs raw through ALP; telemetry
continuous reals 1.16× through RD (lossless real doubles are
near-incompressible — ~1.2× is the published family); symbol codes
6.4–10.5×; integer cents 4.0–4.2×. Decode 31–93M values/s, the
shipped delta-of-delta's band, paid once per segment at open. The
writer-policy change was the first deliberate encoder revision:
`segment_v1.bin` became the decode-compat golden (old bytes decode
forever), `segment_v2.bin` locks the new encoder.

## TallyDB ships as a library and a console binary, never a server

**Decided (2026-07-23): library first; a single-file shell binary at
M3; never a server.** TallyDB ships two ways: as an embeddable
library (the design center, unchanged), and — from M3 — as a
standalone single-file binary attached to each release: a CLI shell
over the same `engine::Database` doorway, the `sqlite3`/`duckdb`
precedent. Installation is copying one file. (Ruled 2026-07-29: a
third channel arrives with M5.5 — the Python binding as wheels on
PyPI. Python-specific by design; the engine and console never
depend on it.) This is not a move
toward general purpose: the shell exposes exactly the library's SQL
surface, and the three assumptions bound it the same way.

What the shell shape pulls in (all additive, none a refactor): DDL
(`CREATE TABLE` with the numeric-or-key types and the declared
ordering key) and ingest (`INSERT`, plus a bulk import) in SQL;
statically linked compute (already the default: the compute stack is
pure Rust plus the vendored Lua sources);
a process lock on the storage directory (two processes opening one
table is undefined until then); and per-platform release builds in
CI. Rendering key columns as text in the shell is fine — the shell
*is* an application, exactly where the strings-precisely rule says
display text belongs.

**The rejected alternative is the engine growing a listener.** A
server needs a wire protocol, auth, TLS, sessions, backpressure,
multi-tenancy — general-purpose infrastructure orthogonal to the
three assumptions — and the differentiator dies at a network
boundary: compute-without-copying only exists in-process. If a
served deployment is ever wanted, it is a **separate product that
embeds TallyDB** (Arrow Flight is the natural seam — SQL in,
`ArrowArrayStream` out is already the engine's shape), the way
rqlite wraps SQLite and MotherDuck wraps DuckDB. The engine-side
obligation that keeps third-party servers viable is only this: stay
embeddable in a concurrent host — snapshot reads through `&self`,
single writer, a clean `Send`/`Sync` story. *Satisfied 2026-07-27
(#51):* `Table::reader()` hands any thread a cloneable `Send + Sync`
handle minting point-in-time `TableSnapshot`s while the one
`&mut Table` writer appends, mutates, or compacts — the shared state
sits behind a per-table lock held only for reads and swaps (bounded
by one write-buffer copy), compaction is read-copy-update through the
segment `Arc`s, and the single-writer cut stays a compile-time fact.
No reopen condition is foreseen for the listener; the
network-boundary argument is structural.

## The two column species are numeric and key (#7)

The vocabulary is final (issue #7, decided 2026-07-23): the two species are
**numeric** and **key**, chosen because the pair states the invariant and
"key" matches SQL's own usage on a SQL-native surface. A key is a *label*,
not a primary key — repeating values are the point, not a violation. For
readers arriving from the BI/OLAP world: key columns play the *dimension*
role in a star schema, numeric columns are the *measures*; the
Kimball vocabulary was considered and set aside because "dimension" and
"measure" collide with this document's mathematical audience.

## Things that are settled “no”s

- **A boolean column type — no for now, with the Human's explicit
  revisit flag (2026-07-29: "ruled wrong on purpose").** Flags are
  `BIGINT` 0/1; predicates never materialize; Arrow booleans are
  refused at ingest with a teaching error naming the one-line cast.
  Not a performance or WASM question — a type multiplies against every
  seam (formats, value map, predicates, aggregates, oracles) and buys
  nothing 0/1 lacks. The standing revisit covers the type AND the
  logical-annotation mechanism (`TimestampNs`-style); storage is
  pre-settled for it: a nullable boolean is 2 bits/row.
- **Compiled Lua C extensions** (`package.loadlib`). Pure-Lua libraries are
  fine and need no special handling. (See *The Lua layer* below for the full
  reasoning.)
- **A LAPACK dependency, at all, until an op needs more than two
  parameters or two dimensions.** Not "a general LAPACK surface" — any
  LAPACK surface. Every statistic the engine exposes *in SQL* has an exact
  closed form at the size it needs; the one op that does not — #90's
  K-factor fit, which carries no SQL name — is served by a hand-rolled
  `K × K` Cholesky rather than a LAPACK surface, so the trigger fired
  and this "no" held, and the removal is what frees the WASM
  build from a LAPACK-in-WASM layer that does not exist. When a wider op
  is committed, the rule that governed the old curated set still governs
  its replacement: don't add routines because LAPACK has them; add them
  because a named workflow needs them. See *Curated compute: what the
  engine calls, and why*.
- **Autodiff / a Torch-style tensor framework.** Different computational
  paradigm than anything the target workload (closed-form / classical
  numerical methods) needs. If a specific, repeated, real need shows up
  later, it gets a narrow scoped addition, not this whole paradigm.
- **Building out a "scientific ecosystem"** (e.g. Julia's
  DifferentialEquations.jl-style breadth) to compensate for Lua's thinner
  ecosystem. Not this project's job — the embedded Lua scripting layer is
  the intended escape hatch for gaps, not something we pre-fill.
- **A general query optimizer / cost-based planner.** Join strategy is
  fixed by input structure — a small materializable side, or co-ordering
  on the join key — never chosen by cost (see *the join constraint,
  completed*).
- **Arrow IPC / Flight / Parquet in `arrow-lite`.** The interop surface is
  the C Data Interface (including the stream variant), nothing else — IPC
  drags in FlatBuffers and a much larger spec. Parquet in/out is the
  application's job via ecosystem tools that already speak C-Data.

If something on this list seems newly justified, that's a conversation to
have explicitly (update this document and its companions together), not a
decision to make silently inside an implementation PR.

## Compiled Lua C extensions are out

Embedded Lua supports pure-Lua libraries (plain `.lua` source) out of the
box — they run as ordinary Lua code with no extra integration work.
Compiled C extensions (LuaRocks packages with a `.so`/`.dll` component,
loaded via `package.loadlib`) are not supported: allowing arbitrary compiled
code to load inside an embedded database process is a real attack-surface
and stability tradeoff, and it cuts against the curated-not-general instinct
behind everything else in this design. This is also structurally true for
the WASM backend regardless of policy — WASM's sandbox can't do
`dlopen`-style dynamic loading at all — so the two constraints reinforce
each other rather than being separate decisions.

## String predicates are in scope; string production is not

The numeric-or-key rule holds across the *entire pipeline* — stored columns,
intermediate results, and query outputs are always numeric or key; a bare
string never exists in the engine. That is more permissive than it sounds:

- **String *predicates* on key columns are in scope.** `WHERE symbol =
  '...'` / `IN (...)` / `WHERE name LIKE '%Bank%'` are built; regex
  matching is in scope but not yet implemented (rejected loudly until
  then — a todo, not a silent gap). All consume the interned strings and
  emit a *row selection*, not a string, so they don't need a third type.
  Because keys are dictionary-encoded, such a predicate is evaluated once
  per *distinct* value in the small dictionary and then applied as
  integer set-membership: string filtering is not just allowed, it's
  cheap.
- **String *production* is out.** No function may *emit* a string value: no
  `SUBSTRING`/`CONCAT` projection, no `CAST(x AS VARCHAR)`, no
  `GROUP_CONCAT`. A key result comes back as its integer code plus the
  dictionary needed to render it; formatting is the application's job.

- **`string` is cut in SQL but open in Lua** — not a contradiction. The
  numeric-or-key invariant governs *what crosses a boundary* (a stored,
  intermediate, or output value), not *local scratch*. SQL has no scratch:
  every value is a column, so a string function *is* a text column, and it
  is out. Lua has scratch (locals), so string manipulation is transient,
  and the invariant is enforced at the Lua→engine boundary — a returned
  string interns into a **key**; a bare string column cannot cross. Same
  invariant, opposite-looking result. (The one real guard is on the
  *output*: a script synthesizing a unique label per row would blow the
  low-cardinality key assumption — capped at the boundary, not by
  crippling `string`.)
- **`math.random` is admitted despite nondeterminism.** The determinism
  invariant carries an explicit opt-out: a script may be nondeterministic
  if the author chooses, at the documented cost of query reproducibility.
  We can afford this because we are not replicated — unlike Redis before
  7.0, whose *script* replication forced determinism (Redis 7 relaxed it
  once it replicated *effects* instead).
- **`coroutine` is a genuine exception** — excluded for an *implementation*
  reason, not a principled one. It breaks no invariant; it fights the
  `pcall`/`longjmp` discipline at the Rust↔C boundary. It is a *deferral*
  pending a binding that can host it safely, not an invariant-based cut —
  when the binding can, the principle admits it.

## Keys assume repeating labels

The dictionary is the one variable-width structure in the system, and it's
acceptable because it is *reference data, not row data*: sized by distinct
values, not rows, and never on the per-row scan/compute path. That holds
only while keys are repeating labels (symbols, sensor ids, exchange codes).
A key column fed never-repeating values (a UUID per row) degenerates — the
dictionary grows with row count and `u32` codes exhaust at ~4.3B distinct
values. A never-repeating identifier is a number: declare it `i64` numeric,
not key. (`engine` should eventually warn when distinct/rows approaches 1
on a large table.)
