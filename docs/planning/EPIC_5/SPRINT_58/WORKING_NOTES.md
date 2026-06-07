# Sprint 58 Working Notes

## Day 1

**Objective:** Turn the Sprint 58 project-plan scope plus the Sprint 57
validated close state into a concrete documentation/examples/benchmark
simplification starting point by confirming the preserved reviewed baseline,
naming the Sprint 58 public-surface cleanup workstreams explicitly, and
defining the authoritative docs, headers, examples, and benchmark hotspots
before any wording or example changes begin.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 58 project-plan source and the new sprint plan:
   - `sed -n '281,308p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_58/PLAN.md`
3. Re-read the strongest inherited Sprint 57 closeout sources:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_57/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_57/RETROSPECTIVE.md`
4. Re-read the Epic 5 review/todo guidance for the public-surface cleanup
   queue:
   - `rg -n "Documentation, Examples|benchmark story|tutorial|header narrative|examples/README|benchmarks/README|sprint-history|workflow story" docs/planning/EPIC_5/reviews docs/planning/EPIC_5/PROJECT_PLAN.md docs/planning/EPIC_5/SPRINT_57/RETROSPECTIVE.md`
   - `sed -n '207,240p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '159,171p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Measure the main Sprint 58 public docs, header, example, and benchmark
   hotspot surfaces:
   - `wc -l README.md docs/tutorial.md include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h examples/README.md examples/example_analysis.c examples/example_iterative.c examples/example_ic_minres.c examples/example_eigs.c examples/example_svd_lowrank.c benchmarks/README.md benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c`

### Day 1 Findings

#### 1. Sprint 58 starts from a validated public-surface baseline, not from renewed API, lifecycle, or solver-support design work

The inherited starting state is already explicit and stable:

- Sprint 57 closed with:
  - bounded giant-test maintainability improvement landed where the clean proof
    seams existed
  - stronger public direct repeated-run lifecycle proof landed
  - stronger factor-many / one-shot compatibility proof landed
  - no public header/API redesign
  - no solver-family support-boundary drift
  - no benchmark/example workflow drift
- Sprint 57 also closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited caller-facing contract remains unchanged:
  - one-shot APIs remain first-class supported entry points
  - repeated direct-solver lifecycle support remains the validated Sprint 50-53
    shape
  - repeated-run iterative/eigensolver handles remain the validated Sprint 54
    shape

Interpretation:

- Sprint 58 is not a public design sprint
- Sprint 58 is not a validation-recovery sprint
- Sprint 58 is a bounded product-surface simplification sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible throughout public-surface cleanup work

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 58 should keep using the exact `strongest local reviewed baseline`
  phrasing
- docs-only audit and wording days do not need to rerun the code gates
- later example or public-header code-touch days should continue treating the
  reviewed CMake count and Makefile/CMake parity contract as the main
  truthfulness anchors

#### 3. The Epic 5 review queue is now concentrated in public wording and workflow framing rather than in implementation ownership

The project plan and Epic 5 review/todo notes already fixed the next cleanup
problem:

- remove stale sprint-history framing from permanent public headers and README
  sections
- keep planning chronology in `docs/planning/` rather than on stable public
  surfaces
- normalize lifecycle guidance across:
  - README
  - tutorial
  - examples
  - benchmark docs
  - public headers
- keep the one-shot-first story where appropriate, but make the advanced
  lifecycle story equally clear

The inherited review guidance remains concrete:

- `README.md` still reads as strong but overly large and detailed for a final
  stable product surface
- `include/sparse_eigs.h` was called out directly as carrying stale
  sprint-history framing
- `benchmarks/README.md` and parts of `README.md` still describe benchmark
  surfaces in sprint-local terms rather than product-level terms
- `examples/README.md` remains a high-value public entry surface for workflow
  alignment

Interpretation:

- Sprint 58 should treat the Epic 5 docs/examples/benchmark cleanup queue as
  still live, not historical
- the main remaining maintainability pressure is now caller-facing wording and
  workflow framing, not solver implementation behavior

#### 4. Sprint 58 reduces cleanly to six bounded work classes

The Sprint 58 project-plan items reduce to six bounded work classes:

1. public docs audit
2. README/tutorial reduction
3. public-header narrative cleanup
4. example modernization
5. benchmark taxonomy cleanup
6. sanity sweep and closeout

The strongest architectural narrowing is:

- keep the work centered on stable workflow guidance first
- prefer reduction and simplification over broader explanatory expansion
- preserve the Sprint 50-57 public and lifecycle fence exactly
- do not broaden into public API redesign, solver-family expansion, or
  benchmark/framework redesign

Interpretation:

- Sprint 58 is about making the final product story easier to scan, not about
  reopening how the product works
- the right output shape is shorter and more stable public wording, not richer
  sprint chronology

#### 5. The authoritative Sprint 58 public-surface hotspots are now fixed directly from the live repo

The strongest Sprint 58 public docs, header, example, and benchmark surfaces
are now explicit:

- top-level docs:
  - `README.md` = `987`
  - `docs/tutorial.md` = `415`
- public headers:
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `687`
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_lu.h` = `337`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_cholesky.h` = `204`
- example docs and examples:
  - `examples/README.md` = `134`
  - `examples/example_eigs.c` = `285`
  - `examples/example_ic_minres.c` = `232`
  - `examples/example_analysis.c` = `210`
  - `examples/example_iterative.c` = `144`
  - `examples/example_svd_lowrank.c` = `120`
- benchmark docs and benchmark surfaces:
  - `benchmarks/README.md` = `235`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_eigs_reuse.c` = `253`

Interpretation:

- the strongest top-level docs reduction pressure is still in `README.md`
  first, then `docs/tutorial.md`
- the strongest public-header narrative cleanup pressure is in
  `include/sparse_iterative.h` and `include/sparse_eigs.h`, with
  `include/sparse_analysis.h` still large enough to matter
- the example and benchmark docs are small enough to stay as workflow-shaping
  surfaces rather than as raw size hotspots, but they remain high-value
  because they directly teach the final caller story

#### 6. The inherited public compatibility fence gives Sprint 58 a clean non-goal boundary

The inherited fence remains:

- no public API redesign
- no reopening the direct-solver lifecycle contract
- no reopening the repeated-run iterative/eigensolver support boundary
- no solver-family expansion disguised as docs or example work
- preserve reviewed validation and truthfulness anchors

Interpretation:

- Sprint 58 should reduce and align wording underneath the already-validated
  public surfaces
- simplification, terminology cleanup, and example-story modernization are the
  success criteria, not new user-visible capability

#### 7. Benchmark and example work should now be treated as workflow-story alignment work, not as new proof or performance campaigns

The inherited Sprint 52-57 work already established:

- the direct analyze-once / factor-many public story
- the repeated-run iterative/eigensolver support boundaries
- the benchmark reuse drivers that exercise those caller stories

Interpretation:

- Sprint 58 should focus on the highest-signal workflow descriptions already
  taught by:
  - `example_analysis`
  - iterative/eigensolver examples
  - `bench_refactor*`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- the right cleanup shape is categorization and wording alignment, not broad
  benchmark or example proliferation

## Day 1 Close

Sprint 58 now has an explicit starting point:

- preserved reviewed baseline
- inherited validated public-contract fence from Sprint 57
- named public docs, public-header, example, and benchmark hotspots
- clear simplification-first workstreams
- explicit non-goal fence against public API or feature expansion

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 50-57 public contract decisions.

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 58 must preserve, then define the smallest authoritative
validation boundary for the later docs/header/example cleanup days and the
high-signal example/benchmark rerun set those code-touch batches should use.

### Commands Run

1. Re-read the Sprint 58 Day 2 plan item and the current sprint notes:
   - `sed -n '79,132p' docs/planning/EPIC_5/SPRINT_58/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_58/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the main Sprint 58 follow-on example and benchmark binaries
   already present in the build tree:
   - `ls build/example_analysis build/example_iterative build/example_ic_minres build/example_eigs build/example_svd_lowrank build/bench_refactor build/bench_refactor_csc build/bench_iterative_reuse build/bench_eigs_reuse`

### Day 2 Findings

#### 1. The strongest local reviewed baseline and truthfulness anchors remain exact

The maintained Sprint 58 baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The authority split also remains unchanged:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 58 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the authoritative
  truthfulness anchors for the sprint

#### 2. The code-day validation boundary is now explicit for public-header or example-touching batches

The mandatory gate for later `*.c` / `*.h` public-header or example days
remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial shipped-surface batches remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- any batch that touches public headers or shipped example source should still
  run the required code-day gate
- substantial public-surface changes should continue to run the stronger
  reviewed baseline path too

#### 3. The quality-contract wording is already aligned and does not need reopening on Day 2

The quality-contract wording remains aligned across:

- `README.md`
  - strongest local reviewed baseline command map
  - explicit `deadcode-check` completeness-gate meaning
- `docs/maintainer_guide.md`
  - maintainer-facing authority framing
  - reviewed CMake parity anchor
  - dead-code interpretation boundary
- `Makefile`
  - executable reviewed-target authority
  - rerun guidance
  - test-count parity checks
- GitHub workflows
  - reviewed CMake execution path
  - deadcode-check execution path

Interpretation:

- Sprint 58 does not need to reopen any quality-contract documentation work on
  Day 2
- the maintained baseline language is already stable enough to carry forward
  unchanged

#### 4. The authoritative Sprint 58 rerun set is now fixed around touched examples and benchmark caller-story surfaces

The main Sprint 58 follow-on binaries already present in `build/` are:

- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

These are the default high-signal reruns for Sprint 58 implementation days.

Interpretation:

- the sprint can keep its validation focus on the examples and benchmark
  surfaces that directly teach or summarize the final public workflow story
- no broader default rerun set is required on Day 2

#### 5. Sprint 58’s Day 2 validation contract is now small and explicit enough to keep docs-first momentum

The controlling Day 2 boundary is:

- docs-only audit/design/summary days:
  - no `make format`
  - no `make lint`
  - no `make test`
- public-header or example code-touch days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial shipped-surface changes:
  - `make quality-review-full`

Interpretation:

- Sprint 58 can move through its docs-first early days without fake validation
  churn
- later code-touch days still have an explicit and defensible gate

## Day 2 Close

Day 2 leaves Sprint 58 with an explicit validation and rerun contract:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative example/benchmark rerun set from the live build tree

That is enough to move to the Day 3 public docs audit without validation
ambiguity.

## Day 3

**Objective:** Reduce the strongest caller-facing docs to a concrete cleanup
map by separating their live drift classes, ranking the real bounded
simplification targets, and fixing the strongest first documentation landing
boundary before any permanent wording changes land.

### Commands Run

1. Re-read the Sprint 58 Day 3 plan item and the current sprint notes:
   - `sed -n '133,188p' docs/planning/EPIC_5/SPRINT_58/PLAN.md`
   - `sed -n '1,320p' docs/planning/EPIC_5/SPRINT_58/WORKING_NOTES.md`
2. Re-read the previous sprint’s audit artifact shape as a formatting sanity
   check:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_57/artifacts/day3-direct-solver-giant-test-seam-audit.md`
3. Scan the strongest caller-facing docs for live drift markers:
   - `rg -n "Sprint|planned for Sprint|future sprint|workflow|repeated-run|factor-many|benchmark|example_analysis|example_ic_minres|bench_refactor|bench_iterative_reuse|bench_eigs_reuse" README.md docs/tutorial.md examples/README.md benchmarks/README.md`
4. Read the highest-signal doc surfaces directly:
   - `sed -n '1,120p' README.md`
   - `sed -n '1,260p' docs/tutorial.md`
   - `sed -n '1,180p' examples/README.md`
   - `sed -n '1,220p' benchmarks/README.md`

### Day 3 Findings

#### 1. The public-docs problem is concentrated in a few high-signal surfaces rather than spread evenly across all docs

The main Sprint 58 doc surfaces do not all need the same treatment:

- `README.md`
  - biggest public-surface wording target
  - still carries the heaviest sprint-local chronology
  - still mixes stable workflow guidance with historical implementation notes
- `benchmarks/README.md`
  - strongest secondary target
  - already useful, but still mixes stable workflow groups with
    sprint-stamped benchmark taxonomy
- `examples/README.md`
  - smaller, but still a high-value public entry surface
  - contains a few explicit sprint-history references and support-boundary
    statements that should be made more product-level
- `docs/tutorial.md`
  - comparatively stable already
  - stronger candidate for bounded reduction and terminology alignment than for
    broad structural rewrite

Interpretation:

- Sprint 58 should not distribute effort evenly across the docs set
- the first landing should focus on the top-level README and the tutorial
  boundary, with benchmark/examples docs following behind

#### 2. `README.md` carries all five named drift classes from the Sprint 58 plan

Live README drift classes:

- stale sprint chronology
  - feature bullets still explain stable capabilities through Sprint-day
    history, especially CSC direct solvers, eigensolvers, and SVD
- repeated-run workflow ambiguity
  - the final public repeated-run story exists, but it is not the dominant
    organizing principle
- one-shot versus advanced-path imbalance
  - many features are described in deep implementation detail before the
    simpler caller story is made clear
- example coverage mismatch
  - the strongest shipped examples are referenced, but the README still reads
    more like a feature ledger than a product-level example map
- benchmark taxonomy mismatch
  - benchmark surfaces are visible, but the benchmark story is still partly
    framed in implementation-history language

Interpretation:

- `README.md` is the strongest Day 4-6 target because it is both the highest
  visibility surface and the place where multiple drift classes overlap
- the first cleanup should emphasize reduction, not more explanation

#### 3. `docs/tutorial.md` is relatively healthy, but still needs bounded alignment with the final public workflow story

The tutorial is not carrying the same heavy sprint-local burden:

- no strong stale sprint chronology surfaced in the first-pass tutorial audit
- the main remaining risk is repeated-run workflow ambiguity
- the tutorial still leans naturally toward one-shot examples and API sampling
- it should be tightened to stay aligned with the final one-shot-first story
  while still pointing clearly at:
  - repeated direct lifecycle
  - iterative/eigensolver repeated-run handles
  - example and header follow-through surfaces

Interpretation:

- `docs/tutorial.md` should be treated as a bounded alignment target, not a
  major narrative-rewrite target
- it pairs naturally with the README cleanup because it can absorb concise
  workflow guidance without repeating implementation chronology

#### 4. `examples/README.md` has lower mass but still carries visible support-boundary and chronology drift

Live examples-doc drift classes:

- stale sprint chronology
  - explicit wording like `Sprint 54 intentionally does not broaden...`
  - sprint-stamped eigensolver references in example descriptions
- repeated-run workflow ambiguity
  - the repeated-run handle and direct-lifecycle story is present, but still
    reads partly as inherited sprint framing
- one-shot versus advanced-path imbalance
  - this surface is actually close to the intended one-shot-first posture, but
    some support-boundary explanation is more detailed than necessary
- example coverage mismatch
  - low mismatch risk; the listed examples already broadly match the shipped
    surfaces

Interpretation:

- `examples/README.md` is a worthwhile later target because it is small,
  caller-facing, and likely cheap to simplify once README/tutorial wording is
  settled
- it should follow, not precede, the top-level docs reduction

#### 5. `benchmarks/README.md` is already valuable, but still reflects benchmark-history accretion more than a final product taxonomy

Live benchmark-doc drift classes:

- stale sprint chronology
  - explicit sprint-stamped benchmark references remain in section titles and
    notes
- benchmark taxonomy mismatch
  - stable workflow categories are present, but they still coexist with older
    benchmark-by-sprint framing
- repeated-run workflow ambiguity
  - the direct repeated-run and handle-reuse categories are good, but the file
    still carries more benchmark-local historical detail than the final public
    taxonomy likely needs
- one-shot versus advanced-path imbalance
  - less severe than in README, but still present where workflow grouping and
    historical benchmark notes compete for space

Interpretation:

- `benchmarks/README.md` is the strongest Day 11 target because it needs
  reorganization around stable workflow groups rather than a generic wording
  trim
- it depends on the README/example wording being simplified first so the
  taxonomy can reuse the same final terminology

#### 6. The first landing should be `README.md` plus bounded tutorial reduction, not benchmark or example cleanup first

The ranked Day 3 cleanup order is now:

1. `README.md`
2. `docs/tutorial.md`
3. `benchmarks/README.md`
4. `examples/README.md`

Why this order is strongest:

- caller visibility is highest in the README
- confusion risk is highest where sprint chronology overlaps with stable
  workflow guidance
- truthful simplification is easiest in the README/tutorial pair because the
  final workflow contract is already validated
- benchmark and example docs will be easier to align once the final top-level
  terminology is fixed

Rejected as the first Sprint 58 landing:

- benchmark taxonomy cleanup first
  - too dependent on final top-level terminology
- example README cleanup first
  - lower visibility and smaller payoff before README/tutorial reduction
- broad tutorial rewrite
  - too expansion-prone for a sprint explicitly about simplification

## Day 3 Close

The public-docs problem is now concrete:

- `README.md` is the strongest first target
- `docs/tutorial.md` is the strongest paired alignment target
- `benchmarks/README.md` is the strongest later taxonomy target
- `examples/README.md` is a smaller but still meaningful later cleanup target

That gives Day 4 a clear starting point for the first bounded README/tutorial
reduction design.

## Day 4

**Objective:** Freeze the first bounded README/tutorial simplification
boundary by selecting the exact top-level workflow sections Sprint 58 should
reduce first, defining the wording invariants those edits must preserve, and
recording the non-goal fence before any permanent prose changes land.

### Commands Run

1. Re-read the Sprint 58 Day 4 plan item and the Day 3 audit:
   - `sed -n '167,240p' docs/planning/EPIC_5/SPRINT_58/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_58/artifacts/day3-public-docs-drift-audit.md`
2. Scan the highest-signal README/tutorial workflow markers and sprint-history
   hotspots:
   - `rg -n "## Features|## Building|repeated-run|factor-many|examples|benchmarks|Sprint|planned for Sprint|future sprint|BiCGSTAB|block iterative|LOBPCG|example_analysis|bench_refactor" README.md docs/tutorial.md`
3. Re-read the strongest touched-seam sections directly:
   - `sed -n '1,120p' README.md`
   - `sed -n '1,260p' docs/tutorial.md`

### Day 4 Findings

#### 1. The first landing should stay on top-level workflow framing, not on the deep feature ledger or benchmark-history sections

The README drift audit shows two different cleanup scales:

- top-level workflow framing:
  - high caller visibility
  - lower risk
  - strongest overlap with the tutorial
- deep feature ledger and historical performance sections:
  - higher mass
  - more coupled to later benchmark and header wording
  - higher risk of accidental truthfulness drift if touched too early

Interpretation:

- Day 5 should stay on the highest-signal top-level public guidance first
- it should not try to collapse the deeper CSC, eigensolver, and benchmark
  historical sections in the same batch

#### 2. The selected Day 5 README boundary is now exact

The first README landing should cover:

- top-level feature and workflow summaries near the front of the file
- repeated-run versus one-shot positioning where the public caller story is
  summarized
- the brief benchmark/example summary wording that points users toward the
  product surfaces rather than the sprint history
- explicit exclusions or bounded support statements that should remain visible
  but read more like stable product guidance

The first README landing should intentionally defer:

- deep CSC Cholesky / LDL^T historical performance narratives
- deep eigensolver backend chronology
- long benchmark-history sections
- large test-history inventories

Interpretation:

- the correct first batch is a reduction of the public front door, not a
  repo-wide README rewrite

#### 3. The selected Day 5 tutorial boundary is now exact

The first tutorial landing should cover:

- concise workflow alignment around one-shot-first guidance
- clearer pointers to the repeated direct lifecycle, iterative-handle, and
  eigensolver-handle opt-in paths
- wording that keeps the tutorial aligned with the final shipped example and
  header story

The first tutorial landing should intentionally defer:

- broad structural reordering
- large new sections
- feature-deep expansion that duplicates the README or examples

Interpretation:

- the tutorial is a paired alignment target, not the main size-reduction
  target
- Day 5 should keep the tutorial changes smaller than the README changes

#### 4. The preserved wording invariants are now explicit

The Day 5 docs reduction must preserve:

- truthful workflow claims
  - one-shot APIs remain first-class
  - repeated-run paths remain bounded opt-in workflows
  - supported exclusions stay visible where they matter
- alignment with validated example and benchmark behavior
  - `example_analysis` remains the strongest direct repeated-run example
  - iterative-handle support remains `CG`, `GMRES`, `MINRES`
  - eigensolver-handle support remains grow-m, thick-restart, and explicit
    `LOBPCG`
  - benchmark workflow groupings remain anchored in the current drivers
- stable top-level navigability
  - the README must still function as the top-level product map
  - the tutorial must still function as the practical getting-started guide

Interpretation:

- simplification is allowed
- truthfulness loss is not

#### 5. The cleanup policy and non-goal fence are now fixed for Day 5

Cleanup policy for the first docs batch:

- remove stale sprint-history narrative
- keep product-level guidance
- keep concise support-boundary caveats that matter to callers
- prefer shorter workflow wording over richer implementation commentary

Explicit non-goals for Day 5:

- no broad tutorial rewrite
- no benchmark taxonomy rewrite yet
- no example README cleanup yet
- no public-header cleanup yet
- no attempt to normalize every historical README section in one pass

Interpretation:

- Day 5 can now land a bounded patch with a clear stop line
- later Sprint 58 days still own benchmark, examples, and header cleanup

## Day 4 Close

The first docs reduction boundary is now concrete:

- primary target:
  - top-level `README.md` workflow framing
- paired alignment target:
  - bounded `docs/tutorial.md` workflow wording
- preserved invariants:
  - truthful workflow claims
  - example/benchmark alignment
  - stable top-level navigability
- explicit non-goals:
  - no broad rewrite
  - no benchmark/examples/header work yet

That gives Day 5 a clear starting point for the first bounded README/tutorial
reduction batch.

## Day 5

**Objective:** Land the first bounded top-level docs simplification patch by
reducing the README front-door workflow story, aligning the tutorial to the
same one-shot-first and repeated-run-support boundary, and preserving the
validated example/benchmark truthfulness fence without widening the touched
scope into benchmark, header, or example-doc cleanup yet.

### Commands Run

1. Re-read the Sprint 58 Day 5 design boundary and the touched README/tutorial
   seams:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_58/artifacts/day4-readme-tutorial-reduction-design.md`
   - `sed -n '1,140p' README.md`
   - `sed -n '400,485p' README.md`
   - `sed -n '1,260p' docs/tutorial.md`
2. Re-read the strongest example and benchmark truthfulness anchors for the
   touched wording:
   - `sed -n '1,220p' examples/example_analysis.c`
   - `sed -n '1,200p' benchmarks/README.md`
3. Apply the bounded docs reduction patch:
   - `README.md`
   - `docs/tutorial.md`
4. Run targeted docs sanity checks:
   - `git diff -- README.md docs/tutorial.md`
   - `rg -n "Choose a Workflow|example_analysis|bench_refactor|bench_iterative_reuse|bench_eigs_reuse|BiCGSTAB|block iterative|CG|GMRES|MINRES|LOBPCG" README.md docs/tutorial.md`
   - `wc -l README.md docs/tutorial.md`

### Day 5 Findings

#### 1. The README front door is now materially more workflow-first and less sprint-local

The top of `README.md` changed in the intended direction:

- the heaviest top-level feature ledger bullets were reduced into stable
  product-level summaries for:
  - direct solvers
  - SVD
  - iterative solvers
- the new `Choose a Workflow` section now makes the final public story visible
  near the top of the file:
  - one-shot direct solves first
  - explicit repeated direct lifecycle for stable-pattern reuse
  - repeated iterative handles only for `CG`, `GMRES`, `MINRES`
  - repeated eigensolver handle for grow-m, thick-restart, and explicit
    `LOBPCG`
  - workflow-local benchmark proof surfaces

Interpretation:

- the README now reads more like a product map and less like a sprint-by-sprint
  feature ledger
- the touched front-door story is shorter without hiding the important support
  boundary

#### 2. The tutorial stayed within the bounded alignment target rather than turning into a broader rewrite

The tutorial changes stayed intentionally small:

- added a `Choose a Workflow First` section near the top
- made the repeated-run support boundary explicit
- added a direct repeated-run pointer near the Cholesky section
- added a repeated iterative-handle pointer near the GMRES section

Interpretation:

- `docs/tutorial.md` now matches the final workflow story better
- the tutorial did not widen into a broad structural rewrite or a duplicate of
  the README

#### 3. The touched wording preserved the intended non-goal fence

The Day 5 patch intentionally did not touch:

- deep CSC historical performance sections
- deep eigensolver chronology
- benchmark taxonomy organization
- `examples/README.md`
- public headers

Interpretation:

- Day 5 stayed inside the Day 4 design boundary
- later Sprint 58 days still own the benchmark, examples, and header cleanup
  queue

#### 4. The touched workflow wording stayed aligned with the current example and benchmark truthfulness anchors

The Day 5 wording stays consistent with the live repo:

- `example_analysis` remains the strongest direct repeated-run example
- repeated iterative handles remain bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated symmetric eigensolver handle wording remains bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- benchmark pointers remain aligned to:
  - `bench_refactor`
  - `bench_refactor_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`

Interpretation:

- the docs batch simplified wording without changing the public truthfulness
  contract
- no caller-facing contradiction was introduced against the existing examples
  or benchmark docs

#### 5. The size/result shape is now concrete from the landed patch

Measured touched-surface result:

- `README.md`: `987 -> 973`
- `docs/tutorial.md`: `415 -> 453`

Interpretation:

- the README reduction is real
- the tutorial grew modestly, but only because the repeated-run workflow
  boundary is now explicit in the right place
- that tradeoff is acceptable because Day 5 was about front-door clarity, not
  raw line-count minimization in every touched file

## Day 5 Close

Day 5 landed the first bounded top-level docs simplification patch:

- `README.md` front-door workflow story is shorter and more product-level
- `docs/tutorial.md` now aligns more clearly to the final repeated-run support
  boundary
- example and benchmark truthfulness anchors remained intact
- benchmark/example/header cleanup remains explicitly deferred

That is enough to move to the Day 6 follow-through pass without reopening the
Day 4 design boundary.

## Day 6

**Objective:** Re-audit the landed README/tutorial state after Day 5, then
finish the strongest remaining top-level drift by normalizing the most visible
README summary sections and product-structure framing without widening the
touched scope into benchmark, examples, or public-header cleanup yet.

### Commands Run

1. Re-read the Sprint 58 Day 6 plan item and the Day 5 landed batch:
   - `sed -n '205,275p' docs/planning/EPIC_5/SPRINT_58/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_58/artifacts/day5-readme-and-tutorial-reduction-batch1.md`
2. Re-audit the strongest remaining README summary seams:
   - `rg -n "Sprint|planned for Sprint|future sprint|workflow|repeated-run|factor-many|examples|benchmarks|BiCGSTAB|block iterative|LOBPCG|example_analysis|bench_refactor" README.md docs/tutorial.md`
   - `sed -n '30,120p' README.md`
   - `sed -n '390,470p' README.md`
   - `sed -n '900,950p' README.md`
   - `sed -n '1,80p' docs/tutorial.md`
3. Apply the bounded Day 6 follow-through patch:
   - `README.md`
4. Run targeted docs sanity checks:
   - `git diff -- README.md`
   - `rg -n "Sparse Symmetric Eigensolver|bench_eigs|public repeated-run iterative handle support|Project Structure|planning/|CG|GMRES|MINRES|BiCGSTAB|LOBPCG" README.md`
   - `wc -l README.md docs/tutorial.md`

### Day 6 Findings

#### 1. The remaining top-level drift after Day 5 was mostly concentrated in README summary sections, not in the tutorial

Post-Day-5 re-audit showed:

- `docs/tutorial.md` already sits close to the intended bounded alignment state
- the strongest residual high-signal drift was in `README.md` summary sections:
  - sparse symmetric eigensolver overview
  - repeated iterative-handle support summary wording
  - project-structure/example/benchmark framing

Interpretation:

- Day 6 was correctly a README-only follow-through pass
- expanding the tutorial further would have added more text than clarity

#### 2. The high-signal eigensolver summary is now more product-level and less sprint-local

The touched `README.md` eigensolver overview now:

- removes the sprint-stamped section heading
- keeps the three concrete backends explicit:
  - grow-m Lanczos
  - thick-restart Lanczos
  - `LOBPCG`
- keeps the AUTO dispatch story explicit
- keeps shift-invert, refinement, and benchmark-driver guidance visible
- removes the most visible sprint-day framing from that high-signal summary

Interpretation:

- the eigensolver overview is still informative
- it now reads more like a stable capability summary than a sprint-local
  chronology

#### 3. The repeated-run support boundary wording is now more stable in the README summary layers

The touched repeated-run iterative wording now:

- drops the `Sprint 54` framing in the visible iterative-support summary
- keeps the real boundary intact:
  - repeated-run handles for `CG`, `GMRES`, `MINRES`
  - `BiCGSTAB` and block iterative workflows remain one-shot compatibility
    surfaces

Interpretation:

- the support boundary is still explicit to callers
- the wording is now less tied to implementation history

#### 4. The project-structure and entry-point framing is now less brittle and less count-heavy

The touched `Project Structure` summary now:

- removes brittle counts from:
  - `include/`
  - `src/`
  - `tests/`
- keeps examples and benchmarks visible as product surfaces
- updates the planning subtree wording to the broader current planning layout

Interpretation:

- the top-level structure summary is now less likely to drift as files move
- it reads more like a stable repository map and less like a frozen snapshot

#### 5. The remaining docs queue is now smaller and more concrete after the README follow-through pass

After Day 6, the strongest intentionally deferred docs queue is:

- benchmark taxonomy cleanup:
  - `benchmarks/README.md`
- example-doc alignment:
  - `examples/README.md`
- public-header narrative cleanup:
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

Interpretation:

- the top-level docs are now in a better state for the later benchmark,
  examples, and header work
- the remaining queue is smaller and more explicitly separated by surface type

## Day 6 Close

Day 6 landed the bounded top-level docs follow-through patch:

- `README.md` summary layers are more product-level and less sprint-local
- the tutorial intentionally stayed untouched because its post-Day-5 state was
  already close to the target
- the remaining queue is now clearly benchmark, examples, and headers

That is enough to move to the Day 7 public-header audit/design from a cleaner
caller-facing docs baseline.

## Day 7

**Objective:** Reduce the public-header cleanup problem to a bounded offender
list by auditing the strongest API-adjacent narrative hotspots directly,
separating the real cleanup classes, ranking the touched headers by caller
visibility and risk, and fixing the exact Day 8 header set plus wording
invariants before any permanent header edits land.

### Commands Run

1. Re-read the Sprint 58 Day 7 plan item and confirm current branch state:
   - `sed -n '241,320p' docs/planning/EPIC_5/SPRINT_58/PLAN.md`
   - `git status --short --branch`
2. Scan the strongest public-header offenders for live drift markers:
   - `rg -n "Sprint|planned for Sprint|future sprint|repeated-run|handle|workflow|BiCGSTAB|block iterative|LOBPCG|factor-many|analyze once|analyze-once" include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h`
3. Measure the live header sizes:
   - `wc -l include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h`
4. Read the highest-signal offending sections directly:
   - `sed -n '1,260p' include/sparse_eigs.h`
   - `sed -n '200,380p' include/sparse_iterative.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,220p' include/sparse_analysis.h`

### Day 7 Findings

#### 1. The public-header problem is concentrated in a small number of strong offenders rather than spread evenly across all public headers

The live public-header surfaces do not all need the same treatment:

- `include/sparse_eigs.h` = `687`
  - strongest first target
  - carries the heaviest stale sprint chronology
  - also carries the heaviest future-work and tuning-local commentary
- `include/sparse_iterative.h` = `765`
  - strongest second target
  - repeated-run handle wording is mostly good, but still has a few visible
    support-boundary and narrative normalization seams
- `include/sparse_analysis.h` = `375`
  - meaningful third target
  - mostly stable, but still carries overlong repeated-run explanatory mass at
    the public-header layer
- direct-family headers:
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_lu.h` = `337`
  - these show smaller, more local cleanup seams rather than the main public
    narrative burden

Interpretation:

- Sprint 58 should not try to touch every public header equally
- Day 8 should stay focused on the strongest public narrative offenders first

#### 2. `include/sparse_eigs.h` carries all four named Day 7 cleanup classes

Live `sparse_eigs.h` drift classes:

- stale sprint chronology
  - section headings, overview text, enum docs, and option comments still
    explain stable behavior through Sprint-day history
- stale future-work wording
  - phrases like `planned for Sprint 21` and `future sprints`
- overlong lifecycle explanation
  - the top-of-file overview and several option/result comments repeat internal
    rationale at a depth better suited to planning docs or the algorithm docs
- terminology mismatch with the current README/tutorial wording
  - the stable repeated-run handle and AUTO routing story exist, but the
    header still mixes them with historical bench-corpus and sprint-local
    rationale

Interpretation:

- `include/sparse_eigs.h` is the strongest Day 8 target by both visibility and
  cleanup payoff
- it needs narrative reduction more than API redesign

#### 3. `include/sparse_iterative.h` is the strongest repeated-run summary companion surface

Live `sparse_iterative.h` cleanup classes:

- stale sprint chronology
  - much lighter than `sparse_eigs.h`, but still present in support-boundary
    wording
- overlong lifecycle explanation
  - explicit repeated-run handle comments are useful, but some visible summary
    wording can be tightened now that README/tutorial already carry the stable
    public story
- terminology mismatch
  - the support boundary around `CG`, `GMRES`, `MINRES`, `BiCGSTAB`, and block
    workflows should read exactly like the current README/tutorial wording

Interpretation:

- `include/sparse_iterative.h` is a good Day 8 companion target because it
  carries a caller-visible repeated-run boundary that should now align tightly
  with the simplified top-level docs
- it does not require the same mass reduction as `sparse_eigs.h`

#### 4. `include/sparse_analysis.h` is a plausible third target, but it is lower-risk and more deferrable than the two stronger surfaces

Live `sparse_analysis.h` cleanup classes:

- overlong lifecycle explanation
  - the explicit repeated-run direct path is correctly documented, but the
    top-of-file overview remains more verbose than the current top-level docs
    now need
- terminology mismatch
  - small opportunity to align wording around analyze-once / factor-many
    language with the README/tutorial cleanup

Interpretation:

- `include/sparse_analysis.h` is the strongest optional third header for Day 8
- it should only be touched if the batch can stay bounded and docs-aligned

#### 5. The direct-family headers should mostly stay deferred unless the Day 8 cleanup exposes a real contradiction

The direct-family headers now read comparatively better:

- `include/sparse_cholesky.h`
  - one-shot-first posture is already explicit
  - repeated-run direct path is already pointed to the shared lifecycle API
- `include/sparse_lu.h`
  - not a main public-sprint-history hotspot in this audit
- `include/sparse_ldlt.h`
  - not a main public-sprint-history hotspot in this audit

Interpretation:

- Day 8 should not widen into broad direct-family header cleanup by default
- these headers remain secondary follow-ons only if the touched Day 8 wording
  reveals a contradiction

#### 6. The exact bounded Day 8 set and invariants are now explicit

Selected Day 8 touched-header set:

1. `include/sparse_eigs.h`
2. `include/sparse_iterative.h`
3. `include/sparse_analysis.h` only if the landed batch remains tight and
   clearly aligned to the README/tutorial wording

Preserved Day 8 invariants:

- preserve API semantics
- preserve ownership truth
- keep concise behavioral comments that still help callers use the API safely
- remove stale sprint-history and future-sprint narrative where it is no longer
  needed to understand the contract
- keep support-boundary wording aligned with the current README/tutorial story

## Day 7 Close

The public-header cleanup problem is now concrete:

- `include/sparse_eigs.h` is the strongest first target
- `include/sparse_iterative.h` is the strongest repeated-run companion target
- `include/sparse_analysis.h` is an optional third target if the batch stays
  bounded
- the direct-family headers remain intentionally deferred unless the Day 8
  batch exposes a real contradiction

That gives Day 8 a clear bounded header set and wording invariant map before
any API-adjacent text changes land.
