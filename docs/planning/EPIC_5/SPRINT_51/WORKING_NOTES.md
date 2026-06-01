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

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 51 must preserve, then define the smallest authoritative
validation boundary for the later header/source integration days and the
high-signal direct-solver rerun set those code-touch batches should use.

### Commands Run

1. Re-read the Sprint 51 Day 2 plan item and the current sprint notes:
   - `sed -n '70,150p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_51/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode-check|strongest local reviewed baseline" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the direct-solver example, benchmark, and regression binaries
   most likely to matter once Sprint 51 `*.c` / `*.h` edits begin:
   - `rg -n "example_analysis|bench_refactor|bench_refactor_csc|test_cholesky|test_ldlt|test_etree|test_chol_csc|test_ldlt_csc" Makefile CMakeLists.txt tests benchmarks examples`

### Day 2 Findings

#### 1. The strongest local reviewed baseline remains exact and should stay visible on all substantial Sprint 51 public API batches

The maintained wrapper surface still says exactly:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

The README and maintainer guide remain aligned with that same language:

- `README.md` still calls `make quality-review-full` the strongest local
  reviewed baseline
- `docs/maintainer_guide.md` still treats that phrasing as the authoritative
  maintainer close state

Interpretation:

- Sprint 51 should preserve the exact “strongest local reviewed baseline”
  wording
- public direct-lifecycle batches should not introduce narrower or looser
  baseline language

#### 2. The reviewed CMake parity anchor remains exact and is still the main truthfulness backstop for the phase-1 direct lifecycle landing

The maintained reviewed CMake path still resolves to:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

And that suite still includes the strongest direct-solver structural and family
tests Sprint 51 is likely to rely on:

- `test_cholesky`
- `test_ldlt`
- `test_etree`
- `test_chol_csc`
- `test_ldlt_csc`

Interpretation:

- Sprint 51 should continue to treat the exact `53` count as a truthfulness
  anchor rather than an approximate expectation
- the direct-lifecycle landing must preserve both the count and the
  Makefile/CMake parity contract

#### 3. The quality contract remains layered, and Day 2 fixes the authority split Sprint 51 should use

The live repo still divides authority cleanly:

- `make quality-review-full`:
  - strongest local reviewed baseline
- `make quality-review`:
  - reviewed Makefile local path
  - `format-check + lint + test + deadcode-check`
- `make quality-review-cmake`:
  - reviewed CMake parity path with full suite execution
- `make deadcode-check`:
  - report-completeness gate, not a zero-findings or removal-ready gate

Interpretation:

- Sprint 51 should use this same split rather than inventing a sprint-local
  quality contract
- direct public lifecycle code days should distinguish between the mandatory
  gate and the stronger reviewed baseline rerun clearly

#### 4. The later Sprint 51 code-day validation boundary is now explicit before any header/source landing begins

For any later Sprint 51 `*.c` / `*.h` batch, the mandatory gate remains:

- `make format`
- `make lint`
- `make test`

For substantial public API batches, the stronger default remains:

- `make quality-review-full`

Interpretation:

- Day 2 fixes the exact code-day validation boundary before Day 3’s header map
  and Day 4’s implementation landing
- Sprint 51 should not blur docs-only notes with code-touch validation claims

#### 5. The high-signal touched-surface rerun list is already clear enough to freeze before code changes begin

The most relevant later direct-lifecycle follow-ons remain:

- examples:
  - `./build/example_analysis`
- benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
- regression tests:
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`

Interpretation:

- Sprint 51 already knows the highest-signal rerun set before header/source
  edits land
- the touched-surface story is centered on the explicit repeated-run direct
  example, the direct refactor benchmarks, and the strongest family/structural
  regression binaries

#### 6. The current docs already expose one concrete adoption-adjacent drift Sprint 51 may naturally fix later, but it should not distort the validation plan

The live benchmark README still says:

- `bench_refactor` = “LDL^T re-factor with cached symbolic”

while the live driver is still a Cholesky analyze-once / factor-many benchmark.

Interpretation:

- this is a real later docs drift
- but the existence of that drift should not change the authoritative
  validation boundary for the actual public API landing days

#### 7. Day 2 leaves Sprint 51 with a clean operational starting point for Day 3

By the end of Day 2, Sprint 51 now has:

- exact baseline wording
- exact reviewed CMake truthfulness count
- fixed mandatory code-day gate
- fixed stronger reviewed default for substantial public API batches
- fixed high-signal direct rerun set

Interpretation:

- Day 3 can now focus on the header implementation map instead of re-arguing
  the quality contract
- the remaining Sprint 51 work is concrete header/source integration planning
  and implementation, not validation-policy discovery
