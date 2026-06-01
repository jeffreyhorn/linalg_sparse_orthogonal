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

## Day 6

### Objective

Wire the shared repeated-run direct lifecycle path through the bounded
Cholesky options route so the analyze/factor/refactor seam inherits the same
linked-list vs CSC backend dispatch behavior as the public one-shot Cholesky
surface, while preserving the existing one-shot API contract.

### Commands Run

- `git status --short --branch`
- `sed -n '240,300p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
- `rg -n "Cholesky|analysis|factor_numeric|bench_refactor|example_analysis|used_csc_path" include src tests benchmarks examples`
- `sed -n '1,260p' src/sparse_cholesky.c`
- `sed -n '1,220p' include/sparse_cholesky.h`
- `sed -n '360,520p' src/sparse_analysis.c`
- `rg -n "factor_opts|analyze|factor_numeric|Cholesky" tests/test_integration.c tests/test_cholesky.c tests/test_chol_csc.c tests/test_etree.c`
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

### Files Changed

- `src/sparse_analysis.c`
- `tests/test_integration.c`

### Key Findings

#### 1. The shared Cholesky repeated-run path had still been bypassing the normal backend-selection seam

Before the Day 6 patch, the `SPARSE_FACTOR_CHOLESKY` branch inside
`sparse_factor_numeric(...)` built the already-permuted working copy and then
called `sparse_cholesky_factor(L)` directly.

Interpretation:

- the shared repeated-run direct path was functionally correct
- but it was not inheriting the public one-shot Cholesky routing logic that
  chooses between linked-list and CSC backends
- that left a real Phase-1 drift between the explicit analysis API and the
  one-shot options surface

#### 2. The right Phase-1 seam was to route shared Cholesky factoring through `sparse_cholesky_factor_opts(...)` with `REORDER_NONE`

Day 6 changed the `SPARSE_FACTOR_CHOLESKY` branch in
`sparse_factor_numeric(...)` to:

- build the already-permuted working copy exactly as before
- call `sparse_cholesky_factor_opts(...)`
- force `.reorder = SPARSE_REORDER_NONE`

Interpretation:

- the shared repeated-run path now inherits the same linked-list vs CSC
  backend dispatch and writeback behavior as the public one-shot Cholesky path
- the patch did not double-apply reordering, because the explicit analysis
  object already owns the symbolic permutation choice
- this is the narrowest source change that reconciles the lifecycle path with
  the shipped one-shot options surface

#### 3. One-shot Cholesky semantics stayed intact

The Day 6 patch did not alter the caller-facing one-shot Cholesky APIs:

- `sparse_cholesky_factor(...)`
- `sparse_cholesky_factor_opts(...)`
- `sparse_cholesky_solve(...)`

It only changed how the shared analyze/factor path internally realizes the
Cholesky numeric phase.

Interpretation:

- Sprint 51 advanced the repeated-run direct contract without demoting the
  simple/default one-shot Cholesky story
- the Sprint 50 compatibility fence remained intact through the second source
  routing batch

#### 4. The shared lifecycle path still intentionally does not publish backend telemetry

The public one-shot Cholesky options path can surface backend-routing details
through its existing result/telemetry surface. The shared
`sparse_analysis_t` / `sparse_factors_t` lifecycle contract still does not
publish whether a given Cholesky factorization used the CSC route.

Interpretation:

- Day 6 correctly reused the backend-selection seam without broadening the
  public direct repeated-run API
- backend telemetry remains a later possible refinement rather than accidental
  Sprint 51 scope creep

#### 5. Direct parity coverage now proves the bounded Cholesky route matches the explicit analysis API even on the CSC-threshold side

Day 6 added a focused integration test that compares:

- `sparse_cholesky_factor_opts(...)` with AMD reordering on a `200x200`
  tridiagonal SPD matrix
- explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` +
  `sparse_factor_solve(...)`

The chosen matrix size is intentionally above the default CSC auto-routing
threshold, so the parity test exercises the backend-selection seam that Day 6
was meant to reconcile.

Interpretation:

- the repeated-run Cholesky route is now covered by a direct public-surface
  regression, not just by indirect suite fallout
- future drift between the one-shot options path and the explicit lifecycle
  path will now fail visibly

#### 6. The Cholesky lifecycle batch stayed inside the Sprint 50/51 scope fence

The Day 6 patch did not:

- expose raw internal CSC/native storage layout
- introduce a new generic direct handle
- demote/remove one-shot Cholesky APIs
- broaden the public lifecycle contract to promise backend telemetry
- reopen documentation/example conversion before the main source seam settled

Interpretation:

- Day 6 is a real Phase-1 implementation batch, not the start of a larger
  direct-solver redesign
- the bounded analysis-centric lifecycle plan from Sprint 50 still governs the
  source work

#### 7. Validation stayed green through the required gate and the stronger reviewed baseline

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
- `Total Test time (real) = 347.18 sec`

Interpretation:

- the shared Cholesky lifecycle routing landed from a fully green reviewed
  state
- Sprint 51 can proceed without reopening validation-baseline repair work

#### 8. The targeted direct-lifecycle follow-ons stayed green after the Cholesky routing batch

The targeted follow-ons that completed cleanly were:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` retained residuals at `4.44e-16` through the
  analyze-once / refactor-many path
- `bench_refactor_csc` continued to show the CSC refactor path beating the
  linked-list refactor path on larger SuiteSparse cases, including:
  - `Pres_Poisson`: `speedup_refactor=13.11`
  - `Kuu`: `speedup_refactor=6.30`
  - `s3rmt3m3`: `speedup_refactor=8.13`
- the structural direct-solver regression binaries remained green after the
  Cholesky lifecycle reroute

Interpretation:

- the Day 6 source change did not destabilize the surrounding repeated-run
  direct surfaces
- Sprint 51 now has both LU and Cholesky bounded lifecycle routing in place on
  a green validation baseline

## Day 7

### Objective

Extend the same bounded repeated-run direct lifecycle routing through LDL^T so
the shared analysis/factor/refactor path inherits the same linked-list vs CSC
backend dispatch behavior as the public one-shot LDL^T options surface, while
preserving the family-local owned-factor contract.

### Commands Run

- `git status --short --branch`
- `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
- `sed -n '260,420p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
- `rg -n "SPARSE_FACTOR_LDLT|sparse_ldlt_factor_opts|sparse_ldlt_factor\\(|sparse_factor_numeric|sparse_refactor_numeric|used_csc_path|reorder_perm" src include tests`
- `sed -n '1,260p' src/sparse_ldlt.c`
- `sed -n '420,560p' src/sparse_analysis.c`
- `sed -n '1,260p' include/sparse_ldlt.h`
- `sed -n '969,1125p' src/sparse_ldlt.c`
- `sed -n '1125,1265p' src/sparse_ldlt.c`
- `sed -n '640,840p' tests/test_integration.c`
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

### Files Changed

- `src/sparse_analysis.c`
- `tests/test_integration.c`

### Key Findings

#### 1. The shared LDL^T repeated-run path had still been bypassing the normal backend-selection seam

Before the Day 7 patch, the `SPARSE_FACTOR_LDLT` branch inside
`sparse_factor_numeric(...)` built the already-permuted working copy and then
called `sparse_ldlt_factor(...)` directly.

Interpretation:

- the shared repeated-run LDL^T path was functionally correct
- but it was still bypassing the public one-shot LDL^T options seam that owns
  backend dispatch and CSC-path selection
- that left a real remaining drift between the explicit analysis API and the
  one-shot LDL^T options surface

#### 2. The right Phase-1 seam was to route shared LDL^T factoring through `sparse_ldlt_factor_opts(...)` with `REORDER_NONE`

Day 7 changed the `SPARSE_FACTOR_LDLT` branch in
`sparse_factor_numeric(...)` to:

- build the already-permuted working copy exactly as before
- call `sparse_ldlt_factor_opts(...)`
- force `.reorder = SPARSE_REORDER_NONE`

Interpretation:

- the shared repeated-run direct path now inherits the same linked-list vs CSC
  backend dispatch and writeback behavior as the public one-shot LDL^T path
- the patch does not double-apply reordering, because the explicit analysis
  object still owns the symbolic permutation choice
- this is the narrowest LDL^T source change that reconciles the lifecycle path
  with the shipped one-shot options surface

#### 3. The family-local owned-factor LDL^T contract stayed intact

The Day 7 patch did not alter the caller-facing LDL^T owned-factor surface:

- `sparse_ldlt_factor(...)`
- `sparse_ldlt_factor_opts(...)`
- `sparse_ldlt_solve(...)`
- `sparse_ldlt_free(...)`

It only changed how the shared `sparse_analysis_t` / `sparse_factors_t`
direct path internally realizes the numeric LDL^T phase.

Interpretation:

- Sprint 51 advanced the repeated-run direct contract without flattening the
  distinct LDL^T factor-object story
- the Sprint 50 compatibility fence remained intact through the third source
  routing batch

#### 4. The shared lifecycle path still intentionally does not publish LDL^T backend telemetry

The public one-shot LDL^T options surface can expose backend-routing details
through:

- `sparse_ldlt_opts_t::backend`
- `sparse_ldlt_opts_t::used_csc_path`

The shared repeated-run direct lifecycle contract still does not publish
whether a given LDL^T factorization used the CSC path.

Interpretation:

- Day 7 correctly reused the backend-selection seam without broadening the
  public repeated-run direct API
- LDL^T backend telemetry remains a possible later refinement rather than
  accidental Sprint 51 scope creep

#### 5. Direct parity coverage now proves the bounded LDL^T route matches the explicit analysis API on the CSC-threshold side

Day 7 added a focused integration test that compares:

- `sparse_ldlt_factor_opts(...)` with AMD reordering on a `200x200`
  tridiagonal SPD matrix
- explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` +
  `sparse_factor_solve(...)`

The chosen matrix size is intentionally above the default CSC auto-routing
threshold, so the parity test exercises the backend-selection seam that Day 7
was meant to reconcile.

Interpretation:

- the repeated-run LDL^T route is now covered by a direct public-surface
  regression, not just by indirect suite fallout
- future drift between the one-shot options path and the explicit lifecycle
  path will now fail visibly

#### 6. The LDL^T lifecycle batch stayed inside the Sprint 50/51 scope fence

The Day 7 patch did not:

- expose raw internal CSC/native storage layout
- introduce a new generic direct handle
- demote/remove one-shot LDL^T APIs
- broaden the public lifecycle contract to promise backend telemetry
- reopen documentation/example conversion before the source seam stabilized

Interpretation:

- Day 7 is a true Phase-1 implementation batch rather than the start of a
  larger direct-solver redesign
- the bounded analysis-centric lifecycle plan from Sprint 50 still governs the
  source work

#### 7. Validation stayed green through the required gate and the stronger reviewed baseline

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
- `Total Test time (real) = 390.43 sec`

Interpretation:

- the shared LDL^T lifecycle routing landed from a fully green reviewed state
- Sprint 51 can move to wrapper-preservation work from a validated LU +
  Cholesky + LDL^T baseline

#### 8. The targeted direct-lifecycle follow-ons stayed green after the LDL^T routing batch

The targeted follow-ons that completed cleanly were:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` retained residuals at `4.44e-16` through the
  analyze-once / refactor-many path
- `bench_refactor_csc` continued to show strong CSC refactor wins on the
  larger SuiteSparse cases, including:
  - `bcsstk14`: `speedup_refactor=5.49`
  - `Kuu`: `speedup_refactor=6.17`
  - `Pres_Poisson`: `speedup_refactor=11.31`
- the structural direct-solver regression binaries remained green after the
  LDL^T lifecycle reroute

Interpretation:

- the Day 7 source change did not destabilize the surrounding repeated-run
  direct surfaces
- Sprint 51 now has bounded lifecycle routing in place across LU, Cholesky,
  and LDL^T on a green validation baseline

# Day 8 - Wrapper Preservation Batch

Date: 2026-06-01

## Goal

Preserve the one-shot direct entry points while routing the safe default
wrappers through the Phase-1 lifecycle-aware options seams where appropriate,
without reopening the broader Sprint 50 non-goal fence.

## Files touched

- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `tests/test_integration.c`

## What changed

### 1. The default one-shot Cholesky wrapper now delegates through the options seam

`sparse_cholesky_factor(...)` no longer calls the linked-list inner routine
directly.

It now builds the bounded default option set:

- `.reorder = SPARSE_REORDER_NONE`
- `.backend = SPARSE_CHOL_BACKEND_AUTO`
- no telemetry output
- no progress callback

and delegates to `sparse_cholesky_factor_opts(...)`.

Interpretation:

- the simple/default Cholesky wrapper now inherits the same linked-list vs CSC
  backend dispatch and validation behavior as the explicit options surface
- the one-shot caller contract stays intact

### 2. The default one-shot LDL^T wrapper now delegates through the options seam

`sparse_ldlt_factor(...)` no longer calls the linked-list internal factor
routine directly.

It now builds the bounded default option set:

- `.reorder = SPARSE_REORDER_NONE`
- `.tol = 0.0`
- `.backend = SPARSE_LDLT_BACKEND_AUTO`
- no telemetry output
- no progress callback

and delegates to `sparse_ldlt_factor_opts(...)`.

Interpretation:

- the simple/default LDL^T wrapper now inherits the same linked-list vs CSC
  backend dispatch behavior as the explicit options surface
- the family-local owned-factor contract remains unchanged for callers

### 3. LU intentionally stayed on the family-local one-shot entry path

The first Day 8 attempt also routed `sparse_lu_factor(...)` through
`sparse_lu_factor_opts(...)`.

That surfaced a real recursion seam:

- `sparse_lu_factor_opts(...)`
- bounded shared-lifecycle route
- `sparse_factor_numeric(..., SPARSE_FACTOR_LU)`
- `sparse_lu_factor(...)`

So the LU wrapper change was explicitly backed out in the final Day 8 landing.

Interpretation:

- Day 8 still preserved the simple/default LU wrapper contract
- the bounded Phase-1 result is now clearer:
  - Cholesky and LDL^T default wrappers safely reuse their options seams
  - LU still needs a later dedicated routing refactor before its one-shot
    wrapper can safely delegate the same way
- this stays inside the Sprint 50 scope fence because it avoids inventing a
  new generic direct-handle layer or broad LU redesign

### 4. Focused wrapper-parity regression coverage now exists for Cholesky and LDL^T

The integration suite gained two direct public-surface regressions:

- `test_cholesky_default_wrapper_matches_default_opts`
- `test_ldlt_default_wrapper_matches_default_opts`

Each compares:

- the simple/default one-shot wrapper
- the explicit default options form

on the same tridiagonal SPD case and checks that the solved outputs are
bit-identical.

Interpretation:

- future drift between the wrapper and default options entry for Cholesky or
  LDL^T now fails visibly
- LU already kept its existing default-wrapper parity check from earlier in
  Sprint 51, so Day 8 did not need a third new wrapper test there

### 5. The batch stayed inside the Sprint 50/51 compatibility fence

Day 8 did not:

- expose raw internal CSC/native storage layout
- introduce a new generic direct handle
- demote/remove one-shot direct APIs
- claim that reuse preserves old numeric factor state
- reopen broad docs/example conversion before the routing seam stabilized

Interpretation:

- this is a bounded wrapper-preservation batch, not a direct-solver redesign
- Sprint 51 can continue from a clearer "where appropriate" wrapper-routing
  rule instead of an over-broadened all-families rewrite

## Validation

### Required code-day gate

The required gate passed:

- `make format`
- `make lint`
- `make test`

### Stronger reviewed baseline

The stronger reviewed baseline also passed:

- `make quality-review-full`

Maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 427.72 sec`

### Targeted direct-lifecycle follow-ons

The targeted follow-ons that completed cleanly were:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` kept solve residuals at `4.44e-16`
- `bench_refactor` remained behavior-stable, including:
  - `tridiag-50`: `speedup=1.14x`
  - `bcsstk04`: `speedup=1.02x`
- `bench_refactor_csc` preserved the larger CSC refactor wins:
  - `bcsstk04`: `speedup_refactor=5.78`
  - `bcsstk14`: `speedup_refactor=5.48`
  - `s3rmt3m3`: `speedup_refactor=7.97`
  - `Kuu`: `speedup_refactor=6.96`
  - `Pres_Poisson`: `speedup_refactor=12.14`
- all touched direct structural regression binaries stayed green

## Day 8 outcome

Sprint 51 Day 8 closed the wrapper-preservation seam in the bounded safe
places:

- Cholesky default one-shot entry now reuses the normal options seam
- LDL^T default one-shot entry now reuses the normal options seam
- LU deliberately remains on the family-local one-shot path until a later
  dedicated routing refactor removes the recursion seam
- direct regression coverage now proves wrapper-vs-default-options parity for
  Cholesky and LDL^T

# Day 9 - Focused Regression Expansion Design & Inventory

Date: 2026-06-01

## Goal

Re-audit the live direct-solver lifecycle regression and adoption surfaces
after the Day 4-8 implementation work so Day 10 only lands the smallest
remaining high-signal additions.

## Surfaces reviewed

Primary regression and lifecycle surfaces:

- `tests/test_integration.c`
- `tests/test_etree.c`
- `tests/test_cholesky.c`
- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`

Strongest adoption / caller-story surfaces:

- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/README.md`

## Findings

### 1. The current direct lifecycle regression surface is already deeper than the original Day 9 placeholder assumed

The live tree already has substantial direct lifecycle coverage spread across
two main surfaces:

- `tests/test_etree.c`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
  - family-specific compatibility checks for Cholesky, LU, and LDL^T
- `tests/test_integration.c`
  - default-wrapper parity for LU
  - Day 8 default-wrapper parity for Cholesky and LDL^T
  - explicit analysis-path parity for LU / Cholesky / LDL^T

Interpretation:

- the public lifecycle path is no longer under-tested in a generic sense
- Day 10 should not broaden into a large new test campaign just because the
  original sprint plan reserved room for regression work

### 2. The strongest remaining direct-test gap is sequencing/ownership truth, not raw solve parity

The live tests now cover:

- one-shot vs explicit-analysis parity
- wrapper vs default-options parity for the safe wrapped families
- analyze/refactor solve loops
- null / shape / some invalid-state rejection

The main remaining phase-1 test gap is narrower:

- making the public lifecycle ownership/sequence rules more directly visible in
  small focused tests instead of only through larger `test_etree.c` end-to-end
  coverage

Likely highest-value Day 10 targets:

- small focused sequencing coverage around:
  - zero-init `sparse_analysis_t`
  - zero-init `sparse_factors_t`
  - analyze → factor → solve → refactor → solve flow
- direct invalid-sequence rejection where already supported by the public
  contract, without inventing new behavior requirements

Interpretation:

- the next test batch should add clarity, not bulk
- broad family-by-family parity expansion is no longer the best use of the day

### 3. LU is no longer a wrapper-routing target for Day 10

Day 8 confirmed a real recursion seam if the default LU wrapper is pushed
through the options/lifecycle route today:

- `sparse_lu_factor_opts(...)`
- shared lifecycle route
- `sparse_factor_numeric(..., SPARSE_FACTOR_LU)`
- `sparse_lu_factor(...)`

Interpretation:

- Day 10 must not try to “finish” LU wrapper routing under the banner of test
  completion
- LU’s remaining work is a later routing refactor problem, not a missing
  Sprint 51 regression-addition problem

### 4. The strongest adoption surfaces are already identified and still narrow

The strongest repeated-run direct example remains:

- `examples/example_analysis.c`

The strongest benchmark adoption surfaces remain:

- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`

Those are still the right next caller-story surfaces because they already sit
closest to:

- analyze-once / factor-many
- refactor-many
- direct timing and residual reporting

Interpretation:

- Sprint 51 should continue to center the repeated-run direct story where the
  repo already has natural ownership
- the smaller one-shot examples and the tutorial can remain out of scope

### 5. The two carried-forward docs drifts are still the most concrete adoption-side fixes

The carried-forward documentation drift from Sprint 50 remains live:

- `benchmarks/README.md`
  - still labels `bench_refactor` as LDL^T re-factor with cached symbolic
  - the live driver is the Cholesky analyze-once / factor-many benchmark
- `examples/README.md`
  - still omits `example_analysis`

Interpretation:

- Day 11 can fix these naturally if it touches the surrounding adoption files
- they do not justify an earlier standalone docs batch

## Day 10 boundary

### Mandatory targets

- small focused lifecycle sequencing/ownership regressions
- no new broad family-by-family parity matrix

### Likely best landing surface

- `tests/test_integration.c`

Rationale:

- it already hosts the small public-surface parity checks added in Sprint 51
- it is a better fit for bounded lifecycle-sequencing tests than expanding the
  already-large `tests/test_etree.c` sweep further without need

### Explicit non-goals

- no new LU wrapper-routing attempt
- no broad rework of `tests/test_etree.c`
- no benchmark/example adoption yet
- no tutorial churn

## Day 9 outcome

Sprint 51’s remaining queue is now smaller than the original plan placeholder
implied:

- the lifecycle core is already well-covered across `test_etree.c` and
  `test_integration.c`
- the strongest remaining Day 10 work is small public-surface
  sequencing/ownership coverage
- the strongest later Day 11 work remains `example_analysis`,
  `bench_refactor*`, and the two carried-forward README drifts

## Day 10

### Objective

Land the smallest high-signal direct-lifecycle regression expansion left after
the Day 9 audit, centered on public sequencing and ownership truth rather than
new routing or broad parity churn.

### Commands Run

- `git status --short --branch`
- `rg -n "test_lu_opts_match_explicit_analysis_path|test_cholesky_opts_match_explicit_analysis_path|test_ldlt_opts_match_explicit_analysis_path|test_cholesky_default_wrapper_matches_default_opts|test_ldlt_default_wrapper_matches_default_opts|build_tridiag_spd|int main\\(" tests/test_integration.c`
- `rg -n "sparse_factor_solve\\(|sparse_refactor_numeric\\(|sparse_analysis_free\\(|sparse_factor_free\\(" src/sparse_analysis.c include/sparse_analysis.h tests/test_etree.c`
- `sed -n '1,120p' tests/test_integration.c`
- `sed -n '410,460p' tests/test_integration.c`
- `sed -n '640,940p' tests/test_integration.c`
- `sed -n '1180,1235p' tests/test_integration.c`
- `sed -n '2330,2415p' tests/test_etree.c`
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `./build/test_integration`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`

### Files Changed

- `tests/test_integration.c`

### Key Findings

#### 1. The highest-value Day 10 coverage gap was sequencing clarity, not family parity volume

The Day 9 audit held up once the live tree was rechecked:

- `tests/test_etree.c` already carries broad analyze/factor/solve/refactor
  coverage
- `tests/test_integration.c` already carries the bounded public-surface parity
  checks added in Sprint 51

What was still worth adding was smaller:

- direct rejection for solve-before-factor on a zeroed `sparse_factors_t`
- direct acceptance of `sparse_refactor_numeric(...)` when `factors` is zeroed

Interpretation:

- the best Day 10 work was to make the public lifecycle sequence easier to see
  and harder to drift
- broad family-by-family parity expansion would have been lower-signal churn

#### 2. Day 10 added a direct public regression for solve-before-factor rejection

The first new integration test now:

- analyzes a Cholesky repeated-run path
- leaves `sparse_factors_t factors = {0}`
- verifies `sparse_factor_solve(&factors, &analysis, b, x)` returns
  `SPARSE_ERR_BADARG`

Interpretation:

- the public direct lifecycle now has an explicit small regression that proves
  callers cannot skip the numeric-factor step
- this was already the documented contract in `sparse_analysis.h`, and the
  test makes that contract locally visible in the bounded integration surface

#### 3. Day 10 also proved that `sparse_refactor_numeric(...)` can act as the first numeric factorization on zeroed factors

The second new integration test now:

- analyzes a Cholesky repeated-run path once
- calls `sparse_refactor_numeric(...)` with zeroed `sparse_factors_t`
- solves successfully against the original matrix
- mutates only diagonal values on a same-pattern matrix
- calls `sparse_refactor_numeric(...)` again and solves successfully again

Interpretation:

- the public “analyze once / refactor-solve many” story is now explicit in a
  small test rather than only inferable from larger `test_etree.c` coverage
- Sprint 51 now has direct regression evidence that zeroed factor ownership is
  a supported starting state for the repeated-run direct path

#### 4. The batch stayed inside the Day 9 and Sprint 50/51 fences

Day 10 did not:

- reopen LU wrapper-routing work
- touch `tests/test_etree.c`
- broaden into example or benchmark adoption
- introduce new public behavior beyond already-documented header truth

Interpretation:

- this was a true focused regression expansion batch, not a disguised routing
  or API redesign step
- the sprint is still tracking the bounded Phase-1 direct lifecycle plan

#### 5. Validation stayed green through both the required code-day gate and the stronger reviewed baseline

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

Interpretation:

- the new regression coverage did not destabilize the direct-lifecycle path
- Sprint 51 can move to the adoption/documentation surfaces from a green
  validated baseline

#### 6. The targeted direct-lifecycle follow-ons stayed green after the test expansion

The touched-surface follow-ons that completed cleanly were:

- `./build/test_integration`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`

Representative direct results:

- `test_integration` remained fully green with the two new lifecycle cases
- `example_analysis` still showed analyze-once / refactor-many residuals at
  `4.44e-16`
- the refactor benchmarks still preserved the repeated-run direct story after
  the Day 10 regression expansion

Interpretation:

- the added tests are aligned with the real shipped repeated-run direct
  surfaces
- Day 11 can focus on adoption and documentation rather than re-debugging the
  lifecycle core

## Day 10 outcome

Sprint 51’s lifecycle/test queue is now narrower again:

- the remaining implementation/value is no longer in public lifecycle core
  routing or regression basics
- the strongest next surfaces are adoption and documentation:
  - `examples/example_analysis.c`
  - `benchmarks/bench_refactor.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/README.md`
  - `examples/README.md`

## Day 11

### Objective

Land the smallest high-signal adoption/docs cleanup left after the Day 10
regression batch by aligning the strongest shipped repeated-run direct example
and benchmark documentation surfaces with the now-live lifecycle path.

### Commands Run

- `git status --short --branch`
- `rg -n "Day 11|Day 12|example_analysis|bench_refactor|README" docs/planning/EPIC_5/SPRINT_51/PLAN.md docs/planning/EPIC_5/SPRINT_51/WORKING_NOTES.md docs/planning/EPIC_5/SPRINT_51/artifacts/day9-focused-regression-expansion-design-and-inventory.md`
- `rg -n "example_analysis|analyze once|refactor|bench_refactor|LDL\\^T|Cholesky" examples/README.md benchmarks/README.md examples/example_analysis.c benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c`
- `sed -n '392,427p' docs/planning/EPIC_5/SPRINT_51/PLAN.md`
- `sed -n '1,220p' examples/README.md`
- `sed -n '1,160p' benchmarks/README.md`
- `sed -n '1,240p' examples/example_analysis.c`
- `sed -n '1,220p' benchmarks/bench_refactor.c`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `rg -n "example_analysis|analyze-once|refactor-many|bench_refactor|CSC supernodal" examples/README.md benchmarks/README.md`

### Files Changed

- `examples/README.md`
- `benchmarks/README.md`

### Key Findings

#### 1. The strongest source surfaces were already aligned enough to leave alone

The live Day 11 check confirmed that:

- `examples/example_analysis.c` already teaches the explicit repeated-run
  direct path clearly
- `benchmarks/bench_refactor.c` already frames the benchmark around one-shot
  vs analyze-once / refactor-many Cholesky usage
- `benchmarks/bench_refactor_csc.c` already documents the CSC comparison as a
  repeated-run direct benchmark rather than a generic factor benchmark

Interpretation:

- the main Day 11 drift was no longer in the source files themselves
- touching those `.c` files would have added churn without improving the
  caller story materially

#### 2. `examples/README.md` had one real omission: `example_analysis`

Before the Day 11 patch, the examples README:

- explained the one-shot examples
- explained iterative and eigensolver repeated-run context
- omitted the strongest shipped repeated-run direct example entirely

Day 11 added an explicit `example_analysis` entry that now calls out:

- zero-init `sparse_analysis_t` / `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve many

Interpretation:

- the examples index now includes the repo’s strongest direct repeated-run
  public example instead of leaving it discoverable only from filenames
- the one-shot-vs-repeated-run split is now clearer for callers browsing the
  examples surface

#### 3. `benchmarks/README.md` had one real labeling drift: `bench_refactor`

Before the Day 11 patch, the benchmark table still described:

- `bench_refactor` as LDL^T re-factor with cached symbolic

That no longer matched the live driver. Day 11 corrected the benchmark docs
to say:

- `bench_refactor` = Cholesky analyze-once / refactor-many path
- `bench_refactor_csc` = the same repeated-run caller story plus CSC
  supernodal comparison

Interpretation:

- the benchmark README now matches the real repeated-run direct benchmark
  ownership instead of carrying a stale LDL^T description forward
- the benchmark-side public lifecycle story is now consistent with Sprint 50
  and Sprint 51

#### 4. The Day 11 batch stayed bounded to the strongest adoption surfaces

Day 11 did not:

- broaden into tutorial work
- reopen README-wide repeated-run restructuring
- touch the lifecycle source code
- change the public contracts themselves

Interpretation:

- this was a true adoption/docs batch, not a disguised implementation change
- Sprint 51 kept the Day 9-10 promise to limit adoption work to the strongest
  repeated-run direct example/benchmark surfaces

#### 5. Targeted runtime sanity checks stayed green on the touched caller story

Because Day 11 was documentation-only, the full C-file validation gate was not
required. Targeted touched-surface runtime checks still completed cleanly:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative direct results:

- `example_analysis` still reported residuals at `4.44e-16`
- `bench_refactor` still completed its one-shot vs analyze-once comparison
- `bench_refactor_csc` still completed the repeated-run linked-list vs CSC
  comparison on `nos4`

Interpretation:

- the touched docs now match live shipped behavior on the exact adoption
  surfaces they describe
- Day 12 can move to compatibility audit work from a cleaner caller-story
  baseline

## Day 11 outcome

Sprint 51’s remaining queue is narrower again:

- the strongest repeated-run direct example and benchmark docs now match the
  live lifecycle path
- the remaining work is no longer basic adoption discoverability; it is later
  compatibility/audit/closeout work
