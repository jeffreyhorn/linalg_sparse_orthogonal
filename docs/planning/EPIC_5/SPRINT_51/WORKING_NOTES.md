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

## Day 3

**Objective:** Turn the Sprint 50 direct-lifecycle contract and the Sprint 51
validation fence into a concrete public-header edit map across
`sparse_analysis.h`, `sparse_lu.h`, `sparse_cholesky.h`, and
`sparse_ldlt.h`, while fixing the boundary between shared repeated-run
vocabulary and family-local one-shot wording before Day 4 header edits begin.

### Commands Run

1. Re-read the Sprint 51 Day 3 plan item and the current sprint notes:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_5/SPRINT_51/WORKING_NOTES.md`
2. Reconfirm the direct repeated-run public anchor and the family-local direct
   headers:
   - `rg -n "sparse_analysis_t|sparse_factors_t|sparse_analyze\\(|sparse_factor_numeric\\(|sparse_refactor_numeric\\(|sparse_factor_solve\\(|sparse_solve_.*lu|sparse_solve_.*chol|sparse_ldlt" include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h`
3. Re-read the live header contracts in full:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,240p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`

### Day 3 Findings

#### 1. `sparse_analysis.h` is already the shared repeated-run direct anchor, so Day 4 should extend clarity there rather than invent new public vocabulary elsewhere

The live header already carries the strongest repeated-run direct workflow:

- zeroed `sparse_analysis_t`
- zeroed `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`
- explicit `sparse_factor_free(...)`
- explicit `sparse_analysis_free(...)`

And the header-level prose already says:

- symbolic analysis is reusable for multiple numeric factorizations
- the analysis object does not own or retain the source matrix
- refactor requires the same sparsity pattern

Interpretation:

- Day 4 should keep `sparse_analysis.h` as the shared repeated-run vocabulary
  home
- Sprint 51 should not scatter equivalent repeated-run lifecycle wording across
  all direct family headers as though each family needs its own generic handle
  model

#### 2. `sparse_lu.h` is still strongly one-shot-first, and that is the correct compatibility posture to preserve

The live LU header still centers:

- in-place factorization of a copied matrix
- `sparse_lu_factor(...)`
- `sparse_lu_factor_opts(...)`
- `sparse_lu_solve(...)`
- iterative refinement / condition-estimation helpers
- explicit “use `sparse_copy()` first to preserve the original” guidance

What it does not yet do clearly enough for the Sprint 51 phase-1 story:

- point callers to the existing analyze/factor/refactor repeated-run path when
  the matrix pattern is stable across many solves
- distinguish the one-shot convenience path from the repeated-run lifecycle
  path without demoting the one-shot API

Interpretation:

- Day 4 LU header edits should stay additive and explanatory
- the LU header should remain family-local and one-shot-first, but it should
  acknowledge the shared repeated-run direct path in `sparse_analysis.h`

#### 3. `sparse_cholesky.h` has the same one-shot-first shape as LU, but its wording should stay even more explicit about matrix mutation

The live Cholesky header still centers:

- in-place SPD factorization
- overwrite of the lower triangle with `L`
- removal of upper-triangle entries
- reordered one-shot factorization via `sparse_cholesky_factor_opts(...)`
- one-shot solve via `sparse_cholesky_solve(...)`

Its compatibility-sensitive truths are already explicit:

- caller should copy the matrix first if the original is needed later
- one-shot factorization mutates the matrix structure and values
- the factorization path is still the simple/default path for one-off SPD
  solves

Interpretation:

- Day 4 Cholesky header edits should preserve this visible mutable-matrix truth
- any cross-reference to the repeated-run path must not obscure that the
  one-shot Cholesky surface remains intentionally simple and compatibility-first

#### 4. `sparse_ldlt.h` already exposes an owned factor object, so its Sprint 51 header work is mainly relationship wording rather than lifecycle invention

The live LDL^T header already has:

- explicit owned factor state in `sparse_ldlt_t`
- explicit free discipline via `sparse_ldlt_free(...)`
- one-shot factor / solve separation
- family-local options / backend / telemetry discussion

But relative to the Sprint 50 repeated-run direct contract, the LDL^T header
still leaves one relationship under-centered:

- how the family-local `sparse_ldlt_t` surface relates to the shared
  `sparse_analysis_t` / `sparse_factors_t` repeated-run path

Interpretation:

- Sprint 51 should not try to replace the existing `sparse_ldlt_t` public shape
- the useful header work is to clarify that the analysis/factor/refactor path is
  the shared repeated-run direct story, while `sparse_ldlt_t` remains the
  family-local one-shot / owned-factor surface

#### 5. The shared-vs-family-local vocabulary split is now explicit enough to guide Day 4 edits

Shared repeated-run wording should stay centered in `sparse_analysis.h`:

- analyze once
- factor / solve
- refactor / solve many
- same-pattern reuse
- analysis owns symbolic/permutation setup
- factors own numeric factor state
- neither object owns the source matrix
- explicit free on zeroed/init state

Family-local wording should stay with the direct family headers:

- matrix mutation and copy-before-factor guidance for LU / Cholesky
- backend / telemetry and pivoting details
- family-specific factor object semantics (`sparse_ldlt_t`)
- one-shot convenience path as the simple/default story
- refinement / condest / inertia helpers

Interpretation:

- Day 4 should update shared lifecycle truth once in `sparse_analysis.h`
- Day 4 should use cross-references plus short family-local wording in the
  LU / Cholesky / LDL^T headers rather than copy the whole repeated-run story

#### 6. The true phase-1 header batch is now small enough to name directly

The first real header batch should be limited to:

- `include/sparse_analysis.h`
  - sharpen the shared repeated-run contract wording
  - keep zero/init, analyze, factor, solve, refactor, and free semantics
    explicit
- `include/sparse_lu.h`
  - add a bounded cross-reference to the shared repeated-run direct path
  - preserve the one-shot and copied-matrix guidance
- `include/sparse_cholesky.h`
  - add a bounded cross-reference to the shared repeated-run direct path
  - preserve the one-shot SPD and in-place-mutation guidance
- `include/sparse_ldlt.h`
  - add a bounded relationship note between the owned LDL^T factor object and
    the shared analysis/factor/refactor path

Later documentation-only follow-ons should stay out of Day 4:

- README repeated-run direct wording
- `examples/README.md`
- `benchmarks/README.md`
- tutorial updates

Interpretation:

- the header phase is now reduced to named additive edits instead of a broad
  “touch all direct docs” instruction
- Sprint 51 can begin header changes without reopening wider documentation scope

#### 7. Day 3 leaves Sprint 51 ready for the first public header/API landing

By the end of Day 3, Sprint 51 now has:

- a concrete shared repeated-run header anchor
- explicit family-local one-shot wording boundaries
- a small named first header batch
- a clear “later docs-only” exclusion set

Interpretation:

- Day 4 can proceed directly to header edits plus required validation
- the remaining uncertainty is implementation detail, not public-contract shape

## Day 4

**Objective:** Land the bounded first public direct-lifecycle header/API batch
across `sparse_analysis.h`, `sparse_lu.h`, `sparse_cholesky.h`, and
`sparse_ldlt.h`, while preserving one-shot direct family compatibility wording
and validating the touched `*.h` surface with the full required gate plus the
stronger reviewed baseline.

### Commands Run

1. Re-read the Sprint 51 Day 4 plan item and the Day 3 header-map artifact:
   - `sed -n '120,260p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_51/artifacts/day3-public-header-surface-design-to-code-map.md`
2. Re-read the live direct header surfaces to patch the bounded batch exactly:
   - `sed -n '1,360p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
3. Reconfirm existing free/zero-init semantics before tightening comments:
   - `rg -n "sparse_analysis_free|sparse_factor_free|sparse_ldlt_free|safe on a zeroed|zeroed struct|no-op" src include`
4. Land the bounded header batch:
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
5. Run the mandatory code-day gate:
   - `make format`
   - `make lint`
   - `make test`
6. Run the stronger reviewed baseline:
   - `make quality-review-full`

### Day 4 Findings

#### 1. The shared repeated-run direct contract is now explicit in `sparse_analysis.h` instead of relying mainly on inference

The Day 4 header batch made the shared repeated-run direct story explicit in
the public analysis/factor header:

- `sparse_analysis.h` now states directly that it is the explicit public
  repeated-run direct-solver path
- the zero/init → analyze once → factor / solve → refactor / solve many →
  free lifecycle is now named in the file-level contract
- the object-ownership boundary is now clearer:
  - `sparse_analysis_t` owns symbolic/permutation setup only
  - `sparse_factors_t` owns numeric factor state only
  - neither owns the source matrix
- reuse wording is now more exact:
  - same-pattern reuse preserves symbolic/permutation setup
  - refactor rebuilds numeric factor contents instead of preserving old numeric
    state

Interpretation:

- the shared direct repeated-run contract is now visible where Sprint 50 said
  it should live
- Sprint 51 no longer depends on later README/example work just to explain the
  public direct lifecycle truth

#### 2. LU remains one-shot-first, but it now points callers to the shared repeated-run path without demotion

The Day 4 LU header edits stayed bounded:

- the file-level LU surface now points stable-pattern repeated runs to
  `sparse_analysis.h`
- `sparse_lu_factor(...)` now explicitly steers repeated same-pattern solves to
  the shared analyze/factor/refactor path
- the one-shot LU surface remained intact:
  - copied-matrix guidance stayed explicit
  - one-shot factor / solve remains the simple/default path

Interpretation:

- Day 4 made the repeated-run relationship visible without turning the LU
  header into a second copy of `sparse_analysis.h`
- the compatibility-facing LU teaching remains honest and unchanged

#### 3. Cholesky now exposes the same relationship boundary while preserving visible mutable-matrix truth

The Day 4 Cholesky header edits also stayed additive:

- the file-level header now identifies the one-shot Cholesky role explicitly
- the shared repeated-run direct path in `sparse_analysis.h` is now named as
  the stable-pattern alternative
- the one-shot SPD / in-place mutation truths remained explicit:
  - copy first if the original matrix is needed later
  - lower triangle overwritten with `L`
  - upper triangle removed

Interpretation:

- Day 4 improved the caller-facing relationship wording without softening the
  mutable-matrix compatibility story
- the header still reads as a one-shot-first SPD surface rather than a generic
  lifecycle wrapper

#### 4. LDL^T now names its relationship to the shared repeated-run direct path while keeping the owned-factor model intact

The Day 4 LDL^T header edits clarified the intended split:

- `sparse_ldlt.h` now says directly that it is the family-local owned-factor
  LDL^T surface
- it now cross-references the shared `sparse_analysis.h` path as the common
  repeated-run direct contract across LU / Cholesky / LDL^T
- the owned `sparse_ldlt_t` semantics remained intact and were not generalized
  into a new direct-handle abstraction

Interpretation:

- Sprint 51 preserved the existing LDL^T factor-object surface
- the new wording clarifies relationship and scope rather than introducing a
  second public lifecycle model

#### 5. The Day 4 batch stayed inside the Sprint 50/51 scope fence

The header landing did not reopen any of the preserved non-goals:

- no raw internal CSC/native storage exposure
- no generic direct-handle redesign
- no demotion or removal of one-shot direct APIs
- no promise that repeated-run direct reuse preserves old numeric factor state
- no broader README/tutorial/benchmark-doc expansion in this batch

Interpretation:

- Day 4 landed the first public direct lifecycle header/API batch without
  expanding the sprint scope
- the repo now has a cleaner header contract while preserving the planned
  compatibility fence

#### 6. The mandatory gate and stronger reviewed baseline both passed after the header batch

The required code-day gate passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

The maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 440.65 sec`

Interpretation:

- the public header batch is validated from both the normal code-day gate and
  the stronger reviewed close state
- Sprint 51 can move to LU integration from a green public-header baseline

#### 7. Day 4 leaves the next seam concrete: source integration, not more public contract design

With the header batch landed:

- the shared repeated-run direct contract is now public and explicit
- the family headers now point to that shared path cleanly
- the next real work is implementation routing in the LU / Cholesky / LDL^T
  source paths

Interpretation:

- Day 5 should focus on LU lifecycle integration rather than further header
  architecture
- the remaining Sprint 51 uncertainty is now implementation behavior, not
  public header wording

## Day 5

**Objective:** Route the bounded default LU options path through the shared
`sparse_analysis` / `sparse_factor_numeric` lifecycle seam while preserving the
existing one-shot LU caller story for the simple/default entry point and for
legacy custom-pivot / callback cases.

### Commands Run

1. Re-read the Sprint 51 Day 5 plan item and the Day 4 header artifact:
   - `sed -n '192,240p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_51/artifacts/day4-public-direct-lifecycle-header-batch.md`
2. Audit the live LU and shared lifecycle implementation seams:
   - `sed -n '1,420p' src/sparse_lu.c`
   - `sed -n '240,520p' src/sparse_analysis.c`
   - `sed -n '1,180p' src/sparse_factor_state_internal.c`
   - `sed -n '1,240p' src/sparse_matrix_state_internal.h`
   - `sed -n '430,700p' tests/test_integration.c`
3. Land the bounded LU lifecycle routing batch:
   - `src/sparse_lu.c`
   - `tests/test_integration.c`
4. Run the mandatory code-day gate:
   - `make format`
   - `make lint`
   - `make test`
5. Run the stronger reviewed baseline:
   - `make quality-review-full`
6. Run the high-signal direct-lifecycle follow-ons:
   - `./build/example_analysis`
   - `./build/bench_refactor`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`

### Day 5 Findings

#### 1. The narrowest credible LU lifecycle seam was `sparse_lu_factor_opts(...)`, not `sparse_lu_factor(...)`

The live implementation audit showed:

- the shared repeated-run direct seam already exists in `sparse_analysis.c`
- the shared LU numeric path already delegates to:
  - `sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12)`
- the simple `sparse_lu_factor(...)` entry point still exposes caller-chosen:
  - pivot strategy
  - tolerance

Interpretation:

- routing the simple one-shot LU entry point through the shared lifecycle path
  directly would have reopened the Sprint 50/51 contract, because the shared
  analysis path does not yet expose arbitrary LU pivot/tolerance controls
- `sparse_lu_factor_opts(...)` was the right Phase-1 seam because it already
  represents the more structured LU surface without demoting the simple
  one-shot API

#### 2. The new routing is intentionally bounded to the default lifecycle-compatible LU option set

The landed gate for the shared lifecycle route is explicit:

- partial pivoting
- `tol == 1e-12`
- `progress_cb == NULL`
- matrix still in original row/column state

When those conditions hold, `sparse_lu_factor_opts(...)` now:

- analyzes with `SPARSE_FACTOR_LU`
- factors through `sparse_factor_numeric(...)`
- republishes the resulting factored matrix back onto the caller-owned
  one-shot LU matrix

Interpretation:

- this matches the Sprint 50 design exactly: Phase 1 centers the shared
  analysis/factor/refactor contract without pretending the lifecycle layer
  already covers every LU-specific option
- the route is real for the bounded default LU options surface, not merely a
  docs-only cross-reference

#### 3. Legacy LU option surfaces remain intact where the shared lifecycle contract is not yet expressive enough

The pre-existing direct LU path still handles:

- custom pivot/tolerance combinations
- progress / cancellation callback routing
- non-original or already-mutated matrix state

Interpretation:

- Day 5 preserved the one-shot LU compatibility story instead of forcing the
  lifecycle path into cases it cannot yet represent cleanly
- the sprint advanced the public repeated-run direction without breaking the
  already-shipped cancellation and custom-LU semantics

#### 4. The implementation had to republish lifecycle-owned LU state back onto the matrix to preserve `sparse_lu_solve(...)`

The shared lifecycle factor object keeps permutation state outside the matrix,
but `sparse_lu_solve(...)` expects:

- a factored matrix payload
- matrix-local `reorder_perm`

The Day 5 patch therefore:

- steals the factorized working-copy payload from the lifecycle LU matrix
- transfers the analysis permutation back onto the caller-owned LU matrix
- republishes the factor-state compatibility mirrors through
  `sparse_factor_state_publish_factored(...)`

Interpretation:

- the implementation now uses the shared lifecycle seam internally while
  preserving the public one-shot LU solve contract unchanged
- this is the key compatibility bridge that lets Sprint 51 advance without a
  larger public solver-handle redesign

#### 5. Direct regression coverage now proves the bounded lifecycle route matches the explicit analysis API

Day 5 added a focused integration test that compares:

- `sparse_lu_factor_opts(...)` on the bounded default LU+AMD path
- explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` +
  `sparse_factor_solve(...)`

The new test verifies solution parity on the same matrix/right-hand side pair.

Interpretation:

- the new source routing is not just structurally plausible; it is covered by
  a direct public-surface parity check
- Sprint 51 now has an explicit regression that will fail if the one-shot LU
  options path drifts away from the shared lifecycle route again

#### 6. The LU lifecycle batch stayed inside the Sprint 50/51 scope fence

The Day 5 patch did not:

- expose raw internal CSC/native storage layout
- introduce a new generic direct handle
- demote or remove one-shot LU APIs
- broaden the shared lifecycle path to arbitrary LU pivot/tolerance options
- reopen README/tutorial/example teaching work before the source path settled

Interpretation:

- the patch is a true Phase-1 implementation batch rather than the start of a
  larger direct-solver redesign
- the Sprint 50 compatibility and non-goal fence held through the first source
  routing step

#### 7. Validation stayed green through both the required gate and the stronger reviewed baseline

The required code-day gate passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

The maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 357.29 sec`

Interpretation:

- the first LU source-routing batch is validated from both the normal code-day
  gate and the stronger reviewed close state
- Sprint 51 can move to Cholesky routing from a green LU lifecycle baseline

#### 8. The direct-lifecycle follow-ons that completed stayed green after the LU routing batch

The targeted follow-ons that completed cleanly were:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` still showed the analyze-once / refactor-many path with
  residuals at `4.44e-16`
- `bench_refactor` still completed all listed fixtures, with the analyze-once
  path beating one-shot on `bcsstk04`
- the direct structural regression binaries remained green after the LU batch

Interpretation:

- the LU routing change did not destabilize the surrounding direct-solver
  lifecycle surfaces
- Day 6 can focus on Cholesky integration rather than re-debugging the LU
  bridge
