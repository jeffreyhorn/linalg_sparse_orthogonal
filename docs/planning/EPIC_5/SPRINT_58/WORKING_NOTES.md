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
