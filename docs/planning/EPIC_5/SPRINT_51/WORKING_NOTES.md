# Sprint 51 Working Notes

## Day 1

**Objective:** Turn the Sprint 51 project-plan scope plus the Sprint 50
closeout package into a concrete public direct-solver lifecycle implementation
starting point by confirming the preserved reviewed baseline, naming the Sprint
51 implementation workstreams explicitly, and defining the authoritative
header, implementation, example, benchmark, regression, and validation inputs
before header and source edits begin.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 51 project-plan source and the new sprint plan:
   - `sed -n '57,86p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
3. Re-read the strongest Sprint 50 closeout and handoff artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_50/RETROSPECTIVE.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/artifacts/day8-public-direct-solver-lifecycle-api-design-batch2.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/artifacts/day10-validation-and-landing-plan.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the live direct-solver public-header, implementation, example,
   benchmark, and regression hotspot sizes:
   - `wc -l include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h src/sparse_analysis.c src/sparse_chol_csc.c src/sparse_ldlt_csc.c examples/example_analysis.c benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c tests/test_cholesky.c tests/test_ldlt.c tests/test_etree.c tests/test_chol_csc.c tests/test_ldlt_csc.c README.md`
7. Re-read the live public analysis/refactor precedent and the direct family
   surfaces Sprint 51 is most likely to touch first:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
   - `sed -n '1,220p' examples/example_analysis.c`

### Day 1 Findings

#### 1. Sprint 51 starts from a preserved Sprint 50 implementation contract, not from renewed API-design ambiguity

The inherited starting contract is now explicit and stable:

- Sprint 50 already fixed the direct repeated-run contract around:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - analyze once
  - factor / solve
  - refactor / solve many
  - free explicitly
- Sprint 50 already fixed the implementation order:
  1. public headers / API surface
  2. implementation and wrapper integration
  3. high-signal example / benchmark adoption
  4. compatibility sweep
  5. final validation
- Sprint 50 already fixed the non-goal and compatibility fence:
  - no broad direct-solver API redesign
  - no generic public direct-handle introduction as the main landing
  - no removal or demotion of one-shot direct APIs
  - no raw CSC/native storage exposure

Interpretation:

- Sprint 51 is not a second design sprint
- Sprint 51 is the first implementation sprint for the already-bounded public
  direct lifecycle contract

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible on all substantial public API batches

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 51 should keep using the exact “strongest local reviewed baseline”
  phrasing
- substantial public direct-lifecycle batches should treat the reviewed CMake
  count and parity contract as truthfulness anchors, not soft guidance

#### 3. The live direct-solver asymmetry is now implementation-shaped rather than conceptual

The public repeated-run direct path is already real in:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `examples/example_analysis.c`

But the one-shot family-local surfaces still dominate the simpler caller story:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Interpretation:

- Sprint 51 does not need to decide whether the analysis/factor/refactor path
  should exist
- it needs to make that path more concretely present in the public headers and
  route the touched one-shot direct entries through it where appropriate

#### 4. The highest-risk implementation seam is still the compatibility-facing mutable matrix model on the LU / Cholesky side

The key carried-forward tradeoff remains explicit:

- one-shot LU and Cholesky are still the simple/default path for one-off solves
- both still rely on the copied-matrix / in-place factorization teaching model
- Sprint 50 explicitly preserved that behavior as an accepted compatibility
  boundary rather than promising to eliminate it

Interpretation:

- Sprint 51 must preserve one-shot LU / Cholesky behavior and caller guidance
  while making the analysis/factor/refactor path more concrete
- the implementation risk is not only correctness but also overstatement of
  what “reuse” means on the direct side

#### 5. The Sprint 51 hotspot map is already concentrated enough to name directly

The live file sizes make the first likely implementation surfaces clear:

- public headers:
  - `include/sparse_analysis.h` = `334`
  - `include/sparse_lu.h` = `327`
  - `include/sparse_cholesky.h` = `191`
  - `include/sparse_ldlt.h` = `310`
- implementation seams:
  - `src/sparse_analysis.c` = `614`
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_ldlt_csc.c` = `2723`
- strongest repeated-run support surfaces:
  - `examples/example_analysis.c` = `191`
  - `benchmarks/bench_refactor.c` = `159`
  - `benchmarks/bench_refactor_csc.c` = `388`
- strongest direct regression concentrations:
  - `tests/test_cholesky.c` = `535`
  - `tests/test_ldlt.c` = `2774`
  - `tests/test_etree.c` = `2962`
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_ldlt_csc.c` = `3637`

Interpretation:

- Sprint 51 is correctly focused on a narrow first API batch, not on broad
  direct-solver decomposition
- the main header and source hotspots are already clear before Day 3’s header
  mapping and Day 4’s landing begin

#### 6. The strongest direct repeated-run teaching surface is already shipped, and Sprint 51 should treat it as a primary adoption target rather than an afterthought

`example_analysis.c` already teaches:

- zeroed `sparse_analysis_t`
- zeroed `sparse_factors_t`
- analyze once
- factor
- solve
- refactor
- solve again
- explicit free

Interpretation:

- Sprint 51’s later example/benchmark adoption should stay centered on
  `example_analysis.c` and `bench_refactor.c`
- broad conversion of small one-shot examples is still lower-value than keeping
  the strongest repeated-run direct example aligned with the final public API

#### 7. Sprint 51’s implementation workstreams are now explicit before code changes begin

The Day 1 implementation workstreams are:

1. public header surface
2. LU lifecycle integration
3. Cholesky lifecycle integration
4. LDL^T lifecycle integration
5. wrapper preservation
6. focused regression expansion
7. validation and closeout

Interpretation:

- the Sprint 51 queue is already narrowed to implementation slices, not broad
  direct-solver research
- the correct Day 1 close is a clean implementation baseline and
  authoritative-input package
