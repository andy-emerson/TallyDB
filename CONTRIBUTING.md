# Contributing

This file is the craft companion to the working agreement in
[`AGENTS.md`](AGENTS.md): how code, tests, and prose are *written*. Like
that document it is **project- and language-agnostic** — it names no
repository, no framework, and no toolchain, and its examples are
pseudocode. The one exception is Markdown, which is assumed because it is
the documentation language of the host.

It sits third in a set of four:

| Document | Answers | Changes |
|---|---|---|
| `AGENTS.md` | *How we work* — phases, claims, reviews, merges | Never (replaced whole) |
| The design document | *What we build and why* — invariants, decisions | When a decision settles |
| **`CONTRIBUTING.md`** | *How it is written* — the conventions of the craft | When a convention settles |
| The README | *Where it is now*, for a user | Every merge |

Whether a feature is in scope is the design document's question. Whether
your change needs a plan is `AGENTS.md`'s. This file assumes both and
answers only: given that I am about to write something, what shape does
it take?

---

## 1. Who we are writing for

### 1.1 Assume the minimum, in both directions

Fluency is not one dial. A reader may be deep in the repository's subject
matter and shallow in its implementation, or the reverse; both are
common, and you cannot see where either dial is set.

So set both low. **Assume no more than the minimum fluency the subject
requires — in the domain and in the code — until the Human says
otherwise.** Writing above that minimum quietly excludes readers who
belong here. Writing below it costs a little of the time of readers who
did not need the help. The second error is cheaper, and only the Human
can move the setting.

### 1.2 Two vocabularies, opposite treatment

Documentation leans on the vocabulary of the *subject*; code leans on the
vocabulary of the *craft*. Sort every term you are about to use by which
side it comes from:

| Term is from | Treatment |
|---|---|
| **The repository's own subject** | Use bare. No gloss. |
| **Anywhere else** | Define at first use, in the same sentence. |

Which side a word falls on depends on what the repository is *about*, and
inverts between projects: in a numerical library, the mathematics is the
subject and the storage machinery is borrowed; in a storage engine, the
reverse. Decide once, per repository, and hold it.

Getting this backwards fails both ways — glossing the subject's own
terms wastes the reader's time and reads as condescension, while an
unglossed borrowed term loses them entirely. The test is not "is this
word hard?" but "which side does it come from?"

A borrowed term defined in a module's own documentation may be used bare
within that module thereafter. Across documents, define it again.

### 1.3 Where cleverness is required, carry the naive version

Code uses standard idioms and no cleverness for its own sake. But
sometimes performance demands a non-obvious one — hand-managed memory,
bit tricks, an incremental algorithm replacing an obvious one.

**Then the comment carries the naive equivalent** — not a description of
it, the thing itself, in checkable form: the formula, three lines of
pseudocode, or the straightforward computation the fast path replaced.
The reader must be able to verify the clever version *against* something.

This converts "trust me" into "check me," and it keeps optimized code
reviewable a year later by someone who has forgotten why it is shaped
that way. The strongest form of this rule keeps the naive version *in the
code* — as the default path, or as the reference an accuracy test
compares against — so the check runs rather than being available.

### 1.4 Performance wins, but the bend is documented

    correctness  >  performance  >  readability

Readability is a nice-to-have, never a reason to ship slower code. But
correctness outranking performance has a price, and the price is
sometimes large: an algorithm that measures several times faster is still
rejected if it is wrong where the data actually lives.

Each time performance beats readability, say so and say what was given
up. An undocumented bend reads as carelessness; a documented one is a
decision the next reader can evaluate.

### 1.5 Literate programming, in the agentic era

Knuth proposed in 1984 that we stop treating a program as instructions
for a machine:

> "Let us change our traditional attitude to the construction of
> programs: Instead of imagining that our main task is to instruct a
> *computer* what to do, let us concentrate rather on explaining to
> *human beings* what we want a computer to do."

His `WEB` system interleaved prose and code in one source and split it
two ways — `TANGLE` into compiler order, `WEAVE` into reader order —
because those orders differ, and the compiler's is the wrong one for a
human. We owe him the framing. What follows is the adaptation, because
three things have changed.

**The author is now often an agent, and prose has become cheap.** An
agent can produce fluent, plausible, wrong documentation at any volume.
So the value of documentation has moved from how well it reads to whether
it can be checked. **Documentation that can run, runs** — an example
executed by the test suite cannot drift from the code, while an example
in prose drifts silently. Knuth's guarantee was that prose and code sat
adjacent; adjacency was never enough. An explanation that *could* be
executable and is written as prose instead is a defect, not a style
choice. Prose is for what cannot execute: rationale, invariants,
warnings, rejected alternatives.

**The reader is now often an agent, and its memory ends with the
session.** What survives is the repository. So the record of *why* — the
decision, the alternative that lost, the condition that would reopen
it — is not archival courtesy; it is the only thing preventing the next
session from relitigating a settled question or silently reversing it.
Write the durable documents as the memory they are.

**Structure is now navigation for both.** Human and agent alike find
their way by names and headings rather than by reading start to finish.
Naming a thing is how a complex thing becomes a simple thing plus a name;
if you cannot name it, you do not yet understand it. This is where the
practice improves the program and not merely its documentation.

Two of Knuth's mechanics we leave behind. There is **no tangle step**:
source order stays compilable order, and reader order lives in the prose
above it — a source file that is not the file the compiler reads was
literate programming's documented adoption killer. And there is **no
single top-down narrative**: write every module's documentation for a
reader who arrived there directly, because units are read bottom-up as
often as top-down, and a narrative that starts at the entry point serves
the author's exposition rather than the reader's need to understand one
layer and check one result.

---

## 2. Writing code

### 2.1 The five layers

Each answers a different question. Do not answer a *why* at the item
level or a *what* at the module level.

**1. Package documentation — why this unit exists, and what would be a
mistake here.** An essay: the unit's share of the project's invariants,
its local decisions, its warnings. The most valuable thing such a
document can say is often the thing a newcomer would otherwise learn the
hard way — *this component has no external reference to check it against,
so correctness rests entirely on its own tests.*

Its first line is usually reused as the one-line summary everywhere.
Make it a sentence that lets a reader decide whether this is what they
want, with no technical detail. This is also where §1.2's borrowed
vocabulary gets defined for everything below.

**2. Module documentation — the local model.** What abstraction does this
file maintain, and where are its seams? Use headed sections. Name the
exceptions to your own guarantees: a "no copies" claim that does not say
where copying *does* happen is an overstatement awaiting an audit.

Module documentation is **broad**; types document themselves **fully**. A
little duplication is fine. The failure mode is the type whose
documentation says "see the module documentation."

**3. Item documentation — the contract.** What it does, returns, refuses.

- **Summary sentence first**, one line, third-person present indicative:
  "Returns", "Defines" — never "Return" or "This function returns."
- **Do not restate the signature.** The types already say what goes in
  and out. Document what the types cannot.
- **Standard sections** when present, in order: examples, panics/aborts,
  errors, safety.
- Push reasoning up to the module. An item whose model was established
  above needs three lines, not thirty.

**4. Invariant and safety comments — the proof obligation.** Wherever the
language stops guaranteeing something and you take over — unchecked
memory, a lock ordering, an assumption about alignment — state the
invariant that makes it sound, at the point of use, not in a distant
document.

Where the language distinguishes the caller's obligation from the call
site's, both are required: the item documentation says what a caller must
uphold, the inline comment says why *this* caller does.

**5. Inline comments — the non-obvious mechanism.** For what the code
cannot say about itself, and for §1.3. A concurrency property, a reason
this lock is held only briefly, why an operation is ordered as it is —
things no reader would infer from the types.

### 2.2 Examples that execute are the preferred evidence

**When a contract can be demonstrated, demonstrate it in the item's own
documentation** — and where the language runs examples found in
documentation, that is the mechanism to reach for. Where it does not, put
the example in a test and link to it from the documentation.

- **An example shows *why*, not only *how*.** The reader knows how to
  call a function. What they cannot guess is why they would want to.
- **Choose data that makes the assertion exact.** Inputs with a known
  closed-form answer let the assertion be tight rather than a shrug. A
  tight numerical bound is itself an explanation: it says what accuracy
  to expect.
- **Assert the claim the prose makes.** If the paragraph says the
  structure is still usable afterwards, the example uses it afterwards.
- **Prefer a documentation example to a unit test** for public contracts;
  keep unit tests for interior mechanics a user never sees.

Two mechanics worth knowing wherever they exist:

**Untagged code blocks may be compiled and run.** In toolchains that
extract examples from documentation, an untagged block is a test — so any
block that is *not* an example in the host language must be tagged, or
the build breaks on prose.

**Hidden setup lines.** Where the toolchain can compile a line without
displaying it, use that for setup that would bury the point, so the
visible example stays about the claim. It is the closest thing most
languages give us to Knuth's named chunk: the reader sees the idea, the
compiler sees the whole program.

**On error handling in examples.** Examples a reader would paste as a
starting point should handle errors the way real code does. Examples that
exist to demonstrate a claim may assert instead — an unchecked call there
*is* the assertion, stating that it cannot fail under the documented
conditions, with the test suite enforcing it. Softening it would weaken
the claim to "this compiles."

### 2.3 Link, don't repeat

When you name a type, function, or concept that is defined elsewhere,
link it. **A fact lives in exactly one place and everywhere else points
at it** — a repeated fact gets updated in one place and not the other,
which is how documentation starts lying. Prefer reference-style links,
which keep the prose readable for whoever is reading the source.

### 2.4 Errors teach

Anything the system refuses by design is a chance to teach the model
rather than deny the request. An error message is usually the
documentation with the highest read rate in a project, and often a
reader's first encounter with an invariant.

- **Name the supported set**, so the reader learns the boundary rather
  than only that they crossed it.
- **Name the missing structure**, so they know what would make the
  request serviceable.
- **Refuse rather than silently accommodate** where accommodating would
  teach the wrong model. An alias that quietly maps a foreign concept
  onto a local one is worse than a refusal that explains the difference.

### 2.5 Naming

Prefer industry-standard terms, bare and undefined. Coin only for a
genuinely new concept, define it where it first appears, and never
overload a term that already means something else in the field. Every
coinage is recorded in the design document with the condition that would
retire it.

---

## 3. Writing Markdown

### 3.1 Know which of four things you are writing

| Mode | Serves | Typically |
|---|---|---|
| **Reference** | A reader *at work*, who needs facts | Generated API documentation; specification tables |
| **Explanation** | A reader *at study*, who wants to understand | Decision records; package essays |
| **How-to** | A competent reader with a goal | Runbooks; the commit and gate conventions |
| **Tutorial** | A learner being taken by the hand | A first path from nothing to a working result |

**Explanation sprinkled into reference damages both.** The reference
becomes unscannable, and the explanation never develops because it is
trapped in a table. Rule of thumb: lists and tables are reference;
anything you could read in the bath is explanation. Keep them in separate
sections, cross-linked.

The tutorial is the mode most often missing, and its absence is quiet: a
project can have complete reference and no path for someone trying it for
the first time. If nothing takes a new user from nothing to a working
result, that is a gap worth naming as a todo.

### 3.2 Conventions

- **Reference-style links**, single bracket pair.
- **Tables for reference, prose for explanation.**
- **Headings are navigation** — a reader arriving from a link should be
  able to tell whether this is the section they want.
- **Full sentences, punctuated**, including list items.
- **Every acronym, coined term, and issue reference expands somewhere
  reachable** (§1.2).

### 3.3 Where a fact goes

Route by rate of change. Each fact has one home:

| The fact | Home |
|---|---|
| A settled design decision, with its rejected alternative and reopen trigger | The design document |
| A local decision governing one module | That module's own documentation, pointing at the design document |
| What the project can do today, for a user | The README |
| A convention you have now explained twice | This file |
| Planned but unbuilt | An issue (todo) |
| Built but broken | An issue (bug) |
| A fork deliberately left open | An issue (decision) |

The tell for a violation: **an enumeration inside a durable document is
living status wearing a durable document's clothes.** An enumeration does
not need bullets — a sentence carrying six clauses joined by semicolons
is a list, and the prose form is the one that gets missed.

The three species of living status each have a form, and the form is what
makes them distinguishable at a glance:

- A **todo** names the claim it will earn *and* the evidence that will
  earn it. Without the second half it is a wish.
- A **bug** names the wrong behavior and, when closed, the test it left
  behind.
- An **open decision** names the fork, the options, and **what it
  gates** — which work cannot proceed until it is closed.

Write these in the same register as a durable document. Living status is
short-lived, not second-class: it is where the next session starts.

### 3.4 Description, not accumulation

A durable document describes the present. It is not a changelog, and the
commonest way one fails is not error but **accretion**: each cycle
appends what it just achieved, every addition is true and locally
justified, nobody removes anything, and after thirty cycles the document
is accurate and unreadable. Nothing was violated to produce that. It is
what following "record what this cycle achieved" produces when nothing
pulls the other way.

You cannot detect this by noticing growth. The reader who would notice —
someone who saw last month's version — is exactly the reader §1.5 says we
no longer have. So the tests must work in a single reading, with no
memory of what came before:

- **Chronological order.** A section arranged in the order things
  happened, rather than the order a reader needs them, was appended to.
  Description has no natural chronology.
- **News that has stopped being news.** A clause that exists because
  something was once recent — *now*, *finally*, *already*, *no longer* —
  is a changelog entry that wandered in.
- **Sentences that answer "what did we do?"** rather than "what is
  true?" A status section is a snapshot, and a snapshot has no memory of
  how it came to look like that.
- **The screen test.** A section describing current state that no longer
  fits on one screen has already failed, whatever its contents.

Two rules follow.

**A doc pass that only adds is incomplete.** Every addition to a durable
document is paired with a removal or a compression somewhere in it. This
is the counterweight; without it, the arithmetic only ever runs one way.
What comes out is usually not deleted but *moved* — detail to the design
document, open work to issues, history to the commit log that already
holds it.

**Past a threshold, rewrite instead of editing.** An accreted section
cannot be repaired by editing, because every edit preserves the structure
that produced it — you end up with a shorter accretion. Delete it and
describe the current state from scratch, consulting the old text only for
facts, never for shape. Rewriting a section is a normal doc pass, not an
extraordinary act.

Length is only half of it. Text that accretes also arrives *unshaped*,
because an addition inherits whatever structure was there — and appending
to a paragraph requires none. **Give a section the structure its length
has earned:**

- **One paragraph, one job.** A paragraph answering two questions is two
  paragraphs.
- **Past a screen, headings.** The headings are the section's outline, so
  the diagnostic is direct: if you cannot write a table of contents for a
  section, it does not have one — and neither will its reader.
- **Parallel content, parallel form.** Items of the same kind get the
  same shape — all rows of one table, or all bullets of one list, never
  one of each. Where items are genuinely alike, a table beats prose:
  it makes a missing entry visible, which prose never does.
- **Reader's order, not history's.** Arrange by what the reader needs
  first. Chronology is the default an accreted section falls into
  precisely because it requires no decision.
- **A sentence carrying more than about three clauses is a list that has
  not been broken out.**

Structure is not decoration on top of the content; it is the part of the
explanation that survives being skimmed, which is how both of §1.5's
readers arrive.

### 3.5 The decision-record form

Five parts:

1. **The ruling**, with date and issue number.
2. **The rejected alternative**, described well enough to show it was
   understood before it was declined — and carrying its numbers, so the
   comparison survives as the explanation. A record presenting only the
   winner has thrown the explanation away.
3. **The reopen trigger** — specific and observable. "If it turns out to
   be slow" is not one; "profiling shows this assembly step is a material
   fraction of total time" is.
4. **The reversal class** where the project defines one — how much a
   wrong bet here would cost to undo.
5. **The adoption shape** if the trigger fires, so reopening does not
   restart the analysis.

A record with no rejected alternative is a preference; one with no reopen
trigger is a dead end.

### 3.6 The documentation review is a separate pass

A pass is either a code pass or a doc pass, **never both**
(`AGENTS.md`). The two reviews look opposite ways: the code review is
goal-seeking ("did we get there?"), the documentation review
truth-seeking ("is what we wrote true?"). Mixing them lets the first
review's optimism edit the second's evidence. Land the code, then the
documentation describing it, as two commits.

`AGENTS.md` gives the documentation review two checks — **truth** (no
claim above its evidence, nothing the project has outgrown) and
**clarity** (standard terms, coined only when necessary). Truth is the
standing rules below; clarity is §2.5. This file adds a third that
neither covers.

**Shape.** Is this document still organized? Truth is a property of
sentences and clarity a property of words, but organization is a property
of the *whole*, and a document can pass both checks in every paragraph
while its structure decays underneath them. A section can be entirely
true, use only standard terms, and still be unreadable.

Truth and clarity are checked against what this cycle touched. **Shape
can only be checked at repo scale**, because it is not visible from
inside a diff — the vantage that shows a section has lost its outline is
the one that reads the section whole. So at the repo-wide documentation
review before a merge, the document *is* what the cycle touched. That
review is the scheduled moment for §3.4's rewrite, and the only one.

Three standing rules:

- **State the limitation as plainly as the success.** Weaken the claim
  and give it an issue number in the same sentence: *checked by hand, not
  yet wired into CI — issue #N*.
- **Say where the exceptions are.** Any "no copies," "bounded," or
  "constant time" claim names its exceptions in the same breath, or it is
  an overstatement.
- **A doc pass whose diff is all additions has done half the job.** Each
  addition is paired with a removal or a move (§3.4). A document whose
  every revision is longer than the last is not being reviewed; it is
  being appended to.

---

## 4. Writing tests

### 4.1 The test name is the claim

Test names are declarative sentences, and read as a specification:

```
compaction_drops_deleted_rows_and_restores_contiguity
crashed_compaction_before_commit_is_invisible
cleanup_failure_does_not_strand_the_generation
nulls_survive_compaction
```

Not `test_compaction`, not `compaction_works`. Present tense, including
the adverse case. If you cannot name the claim, you do not yet know what
you are testing — §1.5's naming rule, applied to tests.

When a claim generalizes into a standing property, give the family its
own module named for the property, with its own documentation explaining
the reference it checks against. A module name is a claim too.

### 4.2 The oracle ladder

1. **An independent implementation.** Where a mature project computes the
   same answer, diff against it. One generated differential family is
   worth more than many hand-written assertions.
2. **A golden.** Committed fixtures locking bytes — for things whose
   *bytes* are the contract.
3. **Tests written from intent.** Where no reference exists, the tests
   *are* the specification. Write them with the care that implies, and
   say so in the module's documentation.

A differential harness records the **known, deliberate divergences**
between the two implementations and how the comparison normalizes them.
**Never normalize a divergence without recording why it is legitimate** —
that comment is the difference between an oracle and a fudge.

### 4.3 A closed bug leaves its test behind

Closing a bug means landing the test that would have caught it, named for
the claim it now guards.

When you add a guard, **verify it trips.** Break the fix, watch the guard
fail, restore the fix — and record the number it failed with. A guard
never observed failing is an untested test.

---

## 5. Writing measurements

Numbers live as tests, disabled by default, so they never enter a
correctness gate. Three parts, all load-bearing:

- **A distinguishing prefix** — `measure_` — so measurements are
  greppable and never confused with checks.
- **A skip marker with its reason**, so the suite stays deterministic. A
  benchmark inside a correctness gate is a flaky test waiting to happen.
- **Optimized build, stated.** A debug-build number is not a number.

Where a measurement decided something, the record cites its run: date,
build profile, hardware, and the shape of the input. **A measurement
claim is never stronger than its latest run** — a number without a
citation is unearned; one whose run predates the code it describes is
stale. Prefer ratios against a named peer measured in the same run on the
same hardware, and keep the losing variant's numbers (§3.5).

---

## 6. Commits

```
type: lowercase subject naming the thing, not the activity (#issue)
```

Types: `feat`, `fix`, `doc`, `test`, `perf`, `measure`, and compounds
where a change genuinely spans. `measure` is first-class and distinct
from `perf`, which changes behavior.

Write the subject as the thing that is now true — *durability claims
match the log that now exists* — not `updated docs` or `fixed bug`.

The body explains the *why* and the consequences, and closes with an
**`Evidence:`** line stating what was run and what it showed:

```
Evidence: full workspace gate green; every differential oracle passes
over the new path; the new guard test verified to fail if the
optimization degrades.
```

A commit claiming a behavior change and citing no evidence is
incomplete — and note the last clause, which records that the *guard
itself* was verified.

Run every gate the project defines before proposing a merge: formatting,
lints, the full test suite including documentation examples,
documentation build with warnings fatal, and any differential oracles.
Every convention above that can be enforced by a gate should be, and a
convention that could be gated and is not will eventually be violated
silently.
