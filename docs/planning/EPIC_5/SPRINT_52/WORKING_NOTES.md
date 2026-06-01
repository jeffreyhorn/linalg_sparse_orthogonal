# Sprint 52 Working Notes

# Day 1 - baseline and scope

Date: 2026-06-01

## Goal

Start Sprint 52 from the real Sprint 51 Phase 1 lifecycle implementation
state, not from a reopened design discussion.

## Inputs reviewed

- `docs/planning/EPIC_5/PROJECT_PLAN.md`
  - Sprint 52 section
- `docs/planning/EPIC_5/SPRINT_52/PLAN.md`
- `docs/planning/EPIC_5/SPRINT_51/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_5/SPRINT_51/RETROSPECTIVE.md`

## Baseline conclusions

### 1. Sprint 52 starts from a preserved reviewed baseline, not from repair work

The strongest local reviewed baseline is still:

- `make quality-review-full`

The maintained truthfulness anchor remains explicit and measurable:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The reviewed wrapper surface remains the authoritative local closeout path for
substantial lifecycle batches.

### 2. Sprint 52 inherits a real implemented Phase 1 lifecycle surface

Sprint 51 already landed the first implemented public direct-lifecycle package:

- shared/family public header contract refresh in:
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- bounded LU lifecycle integration through the shared path where the default
  option surface fit cleanly
- shared Cholesky lifecycle routing through the public
  `sparse_analyze(...)` + `sparse_factor_numeric(...)` path
- shared LDL^T lifecycle routing through the same public lifecycle path
- focused lifecycle regression coverage in `tests/test_integration.c`
- aligned repeated-run caller-surface docs in:
  - `examples/README.md`
  - `benchmarks/README.md`

Interpretation:

- Sprint 52 is not about exposing the first public lifecycle path
- Sprint 52 is about deepening the already-landed analysis/factor/refactor
  workflow so it behaves more like the first-class direct repeated-run path

### 3. The main Phase 2 work is deeper integration, not new abstraction

The Sprint 52 project-plan items reduce cleanly to:

- audit the live `sparse_analysis_t` / `sparse_factors_t` path
- reduce avoidable fallback into one-shot symbolic work
- tighten refactor semantics and result reuse
- refresh factor-many benchmark proof
- update the strongest repeated-run docs/example surfaces as needed
- expand regression proof around the deeper lifecycle path

Interpretation:

- the sprint should stay analysis/factors-centric
- this is not the place to invent a new generic direct handle
- this is not the place to expose raw CSC/native storage or reopen the Sprint
  50 non-goal fence

### 4. The strongest code hotspots are now explicit

Current high-value lifecycle hotspots by file size and role:

- public contract:
  - `include/sparse_analysis.h` = `355`
  - `include/sparse_lu.h` = `337`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `320`
- implementation:
  - `src/sparse_analysis.c` = `626`
  - `src/sparse_lu.c` = `1040`
  - `src/sparse_cholesky.c` = `514`
  - `src/sparse_ldlt.c` = `1494`
- proof / adoption:
  - `tests/test_integration.c` = `1314`
  - `benchmarks/bench_refactor.c` = `159`
  - `benchmarks/bench_refactor_csc.c` = `388`
  - `examples/example_analysis.c` = `191`

Interpretation:

- the highest-risk implementation seams are still concentrated in
  `src/sparse_analysis.c`, `src/sparse_lu.c`, and `src/sparse_ldlt.c`
- the strongest proof and adoption surfaces are still
  `tests/test_integration.c`, `bench_refactor*`, and `example_analysis`

### 5. The preserved compatibility fence remains the key control point

Sprint 52 still inherits the Sprint 50-51 fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- one-shot usage remains the simple/default path for one-off solves
- repeated direct runs remain centered on:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- reuse preserves symbolic/permutation/setup state, not old numeric factor
  contents
- no raw internal CSC/native storage exposure
- no broad generic direct-handle redesign
- no structural-pattern verifier redesign

Interpretation:

- the Phase 2 integration work must strengthen the real repeated-run path
  without changing the basic public compatibility story

## Day 1 outcome

Sprint 52 now starts from a concrete Phase 2 baseline:

- reviewed baseline and parity anchor rechecked
- Sprint 51 implementation handoff reduced to a bounded Phase 2 queue
- strongest direct-lifecycle hotspots named explicitly
- preserved compatibility and non-goal fence restated before deeper code work

That is enough to move into the Day 2 validation baseline and touched-surface
recheck without reopening Sprint 50 or Sprint 51 design decisions.

# Day 2 - validation baseline

Date: 2026-06-01

## Goal

Reconfirm the reviewed local baseline and fix the exact Sprint 52 rerun set
before any deeper analysis/refactor code work begins.

## Validation baseline conclusions

### 1. The strongest reviewed local baseline remains unchanged

The authoritative local reviewed closeout command is still:

- `make quality-review-full`

The wrapper wording remains aligned with the current repo state:

- `quality-review-full: strongest local reviewed baseline`

Interpretation:

- Sprint 52 should continue to treat `make quality-review-full` as the
  strongest local reviewed baseline on substantial lifecycle batches
- there is no need to invent a new Sprint 52-specific validation authority

### 2. The main truthfulness anchor remains exact

The maintained reviewed CMake parity anchor is still:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- the Phase 2 lifecycle work starts from the same parity/truthfulness anchor
  Sprint 51 closed on
- Day 13 validation later in the sprint should continue to use that exact count
  as the key anchor

### 3. The code-day gate remains fixed

For later `*.c` / `*.h` lifecycle batches, the mandatory code-day gate remains:

- `make format`
- `make lint`
- `make test`

The stronger default for substantial public direct-lifecycle batches remains:

- `make quality-review-full`

Interpretation:

- Sprint 52 should not weaken its code-day validation just because the
  public lifecycle surface already exists
- deeper integration work still needs the full baseline gate

### 4. The targeted Sprint 52 rerun set is now explicit

The highest-value direct-lifecycle follow-ons remain present in the current
`build/` tree:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_integration`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Interpretation:

- Sprint 52 already knows the strongest repeated-run direct example surface
- Sprint 52 already knows the strongest factor-many benchmark surfaces
- Sprint 52 already knows the strongest direct lifecycle and factor-family
  regression surfaces

### 5. The docs-only vs code-day boundary is clean

Docs-only audit/design/narrowing days:

- preserve wording/truthfulness anchors
- run only targeted sanity checks when needed

Code-touch analysis/refactor lifecycle days:

- run `make format`
- run `make lint`
- run `make test`
- default to `make quality-review-full` for substantial public/direct-lifecycle
  batches

Interpretation:

- the validation contract is fixed before Sprint 52 begins changing lifecycle
  behavior
- later implementation days do not need to renegotiate their gate

## Day 2 outcome

Sprint 52 now has an explicit validation contract before deeper integration
begins:

- strongest reviewed local baseline rechecked
- reviewed CMake parity anchor rechecked at `53`
- mandatory code-day gate restated
- stronger reviewed default restated
- targeted direct-lifecycle rerun set fixed

That is enough to move into the Day 3 analysis/factors contract audit without
validation ambiguity.

# Day 3 - analysis/factors contract audit

Date: 2026-06-01

## Goal

Audit the live `sparse_analysis_t` / `sparse_factors_t` repeated-run direct
path against the Sprint 52 Phase 2 goal and reduce the remaining work to named
fallback and reuse seams.

## Main audit findings

### 1. The public lifecycle contract is analysis/factors-centric and stable enough to keep

The live public header still describes the direct repeated-run story around:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

The header wording also stays consistent with the Sprint 50-51 fence:

- one-shot LU / Cholesky / LDL^T remain first-class peer entry points
- reuse preserves symbolic/permutation setup, not old numeric factor state
- refactor assumes the same sparsity pattern and treats structural
  compatibility as a caller precondition

Interpretation:

- Sprint 52 does not need a new public abstraction
- the public contract is already coherent enough to deepen rather than replace

### 2. The strongest remaining gap is still internal fallback under the shared path

The live `sparse_factor_numeric(...)` path still:

- builds a fresh working copy for every factorization/refactor call
- reapplies the analysis permutation by materializing a permuted matrix copy
- delegates into the family one-shot implementations

The public header says this explicitly:

- symbolic structures are "available for future optimizations"
- they are "not currently used to bypass internal symbolic work"

Interpretation:

- this is the core Sprint 52 integration problem
- the repeated-run direct path is public and correct, but it is still more of
  a shared orchestration layer than a deeply integrated factor-many path

### 3. Cholesky and LDL^T are the cleanest shared-path candidates for deeper integration

In `src/sparse_analysis.c`, the shared repeated-run path for:

- `SPARSE_FACTOR_CHOLESKY`
- `SPARSE_FACTOR_LDLT`

already routes through the corresponding public one-shot options entry with:

- `.reorder = SPARSE_REORDER_NONE`

after the matrix has already been permuted by the analysis path.

Interpretation:

- these families already have the cleanest shared-path relationship
- Sprint 52 should look first at reducing avoidable repeated setup/fallback
  here before trying to push LU into uniformity

### 4. LU remains the strongest family-specific seam

The shared LU path in `sparse_factor_numeric(...)` still:

- builds a fresh permuted/copy working matrix
- calls `sparse_lu_factor(...)`
- uses fixed parameters:
  - `SPARSE_PIVOT_PARTIAL`
  - tolerance `1e-12`

This matches the public header note that LU currently uses fixed parameters and
may expose more through analysis options later.

Interpretation:

- LU is still the highest-risk family for deeper integration
- Sprint 52 should treat LU as a bounded Phase 2 seam, not as a forced
  symmetry target with Cholesky/LDL^T

### 5. Refactor is still a convenience full numeric rebuild, not a tighter reuse path

The live `sparse_refactor_numeric(...)` path still:

- allocates a fresh temporary `sparse_factors_t`
- calls `sparse_factor_numeric(...)`
- replaces the old factors only on success

The implementation explicitly says it:

- performs a full numeric refactorization
- does not validate or reuse previous numeric structure

Interpretation:

- Sprint 52’s "Refactor Path Tightening" item is real
- the biggest open gap is not correctness but the shallowness of the current
  refactor implementation relative to the public repeated-run story

### 6. Solve-path workspace churn is real but secondary to the main Sprint 52 queue

The live `sparse_factor_solve(...)` path still allocates fresh temporary
buffers for:

- permuted RHS storage
- temporary solution storage

on each call.

Interpretation:

- this is a real repeated-run efficiency seam
- but it is secondary to the stronger Sprint 52 mandate around analysis/factor
  fallback and refactor tightening
- it should not displace the main Phase 2 queue unless later integration work
  depends on it

### 7. The strongest proof and adoption surfaces are already the right ones

The live strongest proof/adoption surfaces still are:

- `tests/test_integration.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `examples/example_analysis.c`

The benchmark comments already frame:

- analyze once
- factor/prime once
- refactor many
- solve after refactor

as the intended repeated-run direct story.

Interpretation:

- Sprint 52 should concentrate proof and adoption work here
- it does not need to broaden into scattered tutorial/example churn

## Ranked Phase 2 target list

1. reduce avoidable family-local fallback inside `sparse_factor_numeric(...)`
2. tighten `sparse_refactor_numeric(...)` so the public refactor story is less
   shallow
3. preserve Cholesky/LDL^T as the cleanest shared-path deepening targets
4. treat LU as a bounded special-case seam instead of forcing full symmetry
5. refresh factor-many benchmark proof in `bench_refactor*`
6. expand regression proof in `tests/test_integration.c`
7. keep solve-path workspace churn as a secondary seam, not the main Sprint 52
   target

## Explicit non-targets for Sprint 52

This audit also fixes what Sprint 52 should *not* broaden into:

- new public generic direct-handle abstraction
- raw CSC/native storage exposure
- broad factor-container redesign
- structural-pattern verifier redesign
- sweeping tutorial rewrite
- broad example conversion outside the strongest repeated-run surfaces

## Day 3 outcome

Sprint 52’s main problem is now concrete instead of generic:

- the public direct repeated-run contract is stable enough to keep
- the main remaining gap is internal fallback and shallow refactor behavior
- Cholesky/LDL^T are the strongest shared-path deepening candidates
- LU is the strongest bounded family-specific seam
- benchmarks/tests/example surfaces for later proof are already clear

That is enough to start the Day 4 numeric-reuse integration batch without
guessing at the real Phase 2 target.

## Day 4 outcome

Sprint 52 now has a real first shared-path deepening batch rather than only a
Phase 2 audit:

- the shared Cholesky path in `sparse_factor_numeric(...)` now reuses the
  caller's `sparse_analysis_t` directly on CSC-sized matrices
- the old hidden second `sparse_analyze(...)` inside the one-shot Cholesky CSC
  wrapper is no longer on that repeated-run path
- the linked-list Cholesky path stays unchanged for smaller matrices
- LDL^T and LU remain explicitly deferred:
  - LDL^T still has the harder BK/symmetric-permutation seam
  - LU still remains the bounded family-specific seam

### What changed

The Day 4 code batch landed in:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc_internal.h`
- `src/sparse_cholesky.c`
- `include/sparse_analysis.h`

The key implementation move is narrow and deliberate:

- add a shared-path Cholesky CSC helper in `src/sparse_analysis.c`
- feed `chol_csc_from_sparse_with_analysis(...)` directly from the caller's
  `analysis`
- run `chol_csc_eliminate_supernodal(...)` with the same
  `SPARSE_CSC_SUPERNODE_MIN_SIZE` cutoff used by the one-shot Cholesky surface
- write the factor back into a fresh factor-owned `SparseMatrix`
- keep the factors matrix in analysis coordinate space with `reorder_perm ==
  NULL`, so `analysis->perm` remains the single published symmetric
  permutation for the repeated-run direct path

### What stayed intentionally unchanged

Day 4 stayed inside the Sprint 52 scope fence:

- no new public generic direct handle
- no raw CSC/native storage exposure
- no redesign of `sparse_factors_t`
- no change to one-shot LU / Cholesky / LDL^T public posture
- no overclaim that reuse preserves old numeric factor state
- no attempt yet to deepen the LDL^T BK prepass path
- no LU parameterization or wrapper redesign

### Validation

Because `*.c` / `*.h` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

The maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 354.23 sec`

### Focused follow-ons

The highest-value repeated-run direct follow-ons also stayed clean:

- `./build/example_analysis`
  - residuals remained `4.44e-16`
- `./build/bench_refactor`
  - analyze-once stayed ahead on the main repeated-run cases:
    - `tridiag-200` = `1.46x`
    - `bcsstk04` = `1.66x`
    - `nos4` = `1.57x`

### Day 4 conclusion

Sprint 52 has now converted the Day 3 audit into one live repeated-run
integration improvement:

- Cholesky is no longer purely "analysis outside, one-shot symbolic work
  inside" on the CSC repeated-run path
- the repeated-run direct story remains analysis/factors-centric
- the next bounded work should target either:
  - deeper LDL^T integration
  - refactor-path tightening
  - both, if the next batch can keep LU excluded and the scope narrow

## Day 5 outcome

Sprint 52 now has a second real Phase 2 shared-path deepening batch rather
than only a Cholesky-only result:

- the shared LDL^T path in `sparse_factor_numeric(...)` now reuses the
  caller's `sparse_analysis_t` directly on CSC-sized repeated-run problems
  when the scalar BK pre-pass does not introduce extra symmetric swaps beyond
  the caller's reorder
- if BK *does* introduce extra swaps, the path now rebuilds symbolic analysis
  only on the resulting pre-permuted matrix rather than pretending the caller's
  analysis still matches
- the smaller linked-list LDL^T route stays unchanged
- LU remains intentionally out of scope

### What changed

The Day 5 code batch landed in:

- `src/sparse_analysis.c`
- `src/sparse_ldlt_csc_internal.h`
- `include/sparse_analysis.h`
- `tests/test_integration.c`

The key implementation move is narrow and deliberate:

- add a shared-path LDL^T CSC helper in `src/sparse_analysis.c`
- run the existing scalar BK CSC pre-pass first to determine whether the final
  symmetric permutation still matches the caller's reorder
- reuse `ldlt_csc_from_sparse_with_analysis(...)` directly from the caller's
  `analysis` when that permutation matches
- rebuild analysis only on the BK-pre-permuted matrix when the pre-pass adds
  extra swaps
- seed the batched CSC factor with the scalar pre-pass pivot-size choices
- keep the existing scalar-pre-pass factor as the fallback source if the
  supernodal CSC elimination path does not complete cleanly

### What stayed intentionally unchanged

Day 5 stayed inside the Sprint 52 scope fence:

- no new public direct-handle abstraction
- no raw CSC/native storage exposure
- no redesign of `sparse_factors_t`
- no one-shot LU / Cholesky / LDL^T public demotion
- no overclaim that reuse preserves old numeric factor contents
- no LU routing change
- no refactor-path redesign yet

### Validation

Because `*.c` / `*.h` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

The maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 220.30 sec`

### Focused follow-ons

The highest-value repeated-run direct follow-ons also stayed clean:

- `./build/example_analysis`
  - residuals remained `4.44e-16`
- `./build/test_integration`
  - `29 / 29` passed
- `./build/test_ldlt`
  - `83 / 83` passed
- `./build/test_ldlt_csc`
  - `95 / 95` passed
- `./build/bench_refactor`
  - analyze-once stayed ahead on repeated-run cases including:
    - `tridiag-200` = `1.69x`
    - `tridiag-500` = `1.40x`
    - `bcsstk04` = `1.73x`
    - `nos4` = `1.48x`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - CSC repeated-run path stayed ahead:
    - `speedup_refactor = 1.70x`
    - `res_ll = 8.24e-16`
    - `res_csc = 7.06e-16`

### Day 5 conclusion

Sprint 52 now has two live repeated-run direct deepening batches in code:

- Cholesky no longer hides an avoidable second symbolic-analysis pass on the
  CSC repeated-run path
- LDL^T now reuses the caller's analysis directly whenever BK does not force a
  deeper symmetric-permutation change, and rebuilds analysis only when it has
  to
- the next bounded work should focus on:
  - tightening `sparse_refactor_numeric(...)`
  - additional factor-many proof surfaces
  - keeping LU as the intentionally bounded special-case seam

## Day 6 outcome

Sprint 52 now has a first real refactor-path tightening batch rather than only
deeper factor-many routing:

- `sparse_refactor_numeric(...)` is no longer just a second spelling of
  `sparse_factor_numeric(...)`
- the Sprint 51 zero-init first-factorization path is still preserved
- non-zeroed factor objects now have to match the analysis family and
  dimension, and they must carry the expected family-specific payload
- failed refactor attempts still preserve the last good factorization

### What changed

The Day 6 code batch landed in:

- `src/sparse_analysis.c`
- `include/sparse_analysis.h`
- `tests/test_integration.c`

The key implementation move is narrow and deliberate:

- add a shared validator in `src/sparse_analysis.c` for the incoming
  `sparse_factors_t` object used by `sparse_refactor_numeric(...)`
- keep accepting the all-zero initial state so the public repeated-run path
  still supports "analyze once, first factor via refactor"
- require existing factors to match the analysis family and dimension before
  attempting a replacement numeric factorization
- require LDL^T factors to carry their `D`, `D_offdiag`, `pivot_size`, and
  `ldlt_perm` payload, while non-LDL^T factors must not carry those LDL^T-only
  fields
- continue factoring into a temporary object first, so old factors survive any
  later refactor failure unchanged

### What stayed intentionally unchanged

Day 6 stayed inside the Sprint 52 scope fence:

- no public API redesign
- no new direct-handle abstraction
- no raw CSC/native storage exposure
- no change to one-shot LU / Cholesky / LDL^T public posture
- no incremental numeric-update claim for `sparse_refactor_numeric(...)`
- no LU routing expansion

### Validation

Because `*.c` / `*.h` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this was a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

The maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 229.89 sec`

### Focused follow-ons

The highest-value repeated-run direct proof also stayed clean:

- `./build/test_integration`
  - `31 / 31` passed
  - new coverage now proves:
    - zeroed factors remain valid for first-factorization through
      `sparse_refactor_numeric(...)`
    - mismatched preexisting factors are rejected before replacement
    - failed refactor attempts preserve the last good factors

### Day 6 conclusion

Sprint 52 now has three live Phase 2 direct-lifecycle improvements in code:

- Cholesky avoids an unnecessary second symbolic-analysis pass on the CSC
  repeated-run path
- LDL^T reuses caller analysis directly whenever BK does not force a deeper
  symmetric-permutation change
- `sparse_refactor_numeric(...)` now has a tighter and more truthful
  replacement contract

The next bounded work should focus on:

- deeper factor-many benchmark proof
- additional direct-lifecycle sequencing/ownership proof if needed
- keeping LU as the intentionally bounded special-case seam

## Day 7 outcome

Sprint 52 now has a second refactor-tightening batch rather than only the Day
6 factors-object validation layer:

- the shared analysis path now rejects obvious gross structure drift before
  later numeric factorization/refactor work begins
- the repeated-run direct contract is tighter without claiming a full
  structural-pattern verifier
- zero-init first-factorization, repeat refactor/solve, and old-factor
  preservation behavior all remain intact

### What changed

The Day 7 code batch landed in:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `tests/test_integration.c`

The key implementation move is narrow and deliberate:

- cache the analyzed matrix nonzero count in `sparse_analysis_t`
- add a shared input validator for the analysis/factor path
- reject matrices whose current `sparse_nnz(...)` no longer matches the matrix
  that produced the analysis
- use that shared validator in both `sparse_factor_numeric(...)` and
  `sparse_refactor_numeric(...)`
- prove that rejected NNZ-drift refactor attempts still leave the prior good
  factors usable

### What stayed intentionally unchanged

Day 7 stayed inside the Sprint 52 scope fence:

- no public direct-handle redesign
- no raw CSC/native storage exposure
- no full structural-pattern verifier
- no one-shot LU / Cholesky / LDL^T posture change
- no incremental numeric-update claim
- no LU routing expansion

### Validation

Because `*.c` / `*.h` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this remained a substantial shared direct-lifecycle batch, the stronger
reviewed baseline was also run:

- `make quality-review-full`

That also passed.

The maintained truthfulness anchors stayed exact:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 242.05 sec`

### Focused follow-ons

The highest-value repeated-run direct proof also stayed clean:

- `./build/test_integration`
  - `32 / 32` passed
  - new coverage now proves:
    - NNZ-drift refactor attempts are rejected as obvious gross structure
      mismatch
    - the prior good factors still solve the original system afterward
- `./build/example_analysis`
  - residuals stayed at `4.44e-16`
- `./build/bench_refactor`
  - analyze-once stayed ahead on the repeated-run cases:
    - `tridiag-200` = `1.86x`
    - `tridiag-500` = `1.37x`
    - `bcsstk04` = `1.84x`
    - `nos4` = `1.72x`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - CSC repeated-run path stayed ahead:
    - `speedup_refactor = 1.72x`
    - `res_ll = 8.24e-16`
    - `res_csc = 7.06e-16`

### Day 7 conclusion

Sprint 52 now has a materially more complete refactor boundary on the shared
analysis path:

- zeroed-state first factorization still works
- mismatched preexisting factors are rejected
- obvious gross structure drift is rejected
- failed refactor attempts still preserve the prior factors

The remaining bounded work should now focus on:

- factor-many benchmark proof closeout
- later docs/example adoption
- keeping LU as the intentionally bounded special-case seam

## Day 8 outcome

Sprint 52 now has a real factor-many benchmark-proof batch rather than only
the earlier integration and contract-tightening work:

- `bench_refactor` now measures same-pattern numeric value changes across
  iterations instead of repeatedly refactoring the identical matrix
- the repeated-run public direct path is broken out more truthfully:
  - one-shot average
  - analyze-once cost
  - initial numeric factorization cost
  - later refactor average
  - repeated-run average
  - final solve residual on the last perturbed matrix
- the benchmark ownership docs now match the live benchmark behavior
- a real reviewed CMake portability seam was caught and fixed before closeout

### What changed

The Day 8 batch landed in:

- `benchmarks/bench_refactor.c`
- `benchmarks/README.md`

The main implementation move is that `bench_refactor` now behaves like an
honest same-pattern repeated-run proof:

- keep the one-shot path as:
  - `sparse_copy(...)`
  - perturb values
  - `sparse_cholesky_factor(...)`
- keep the repeated-run path as:
  - `sparse_analyze(...)` once
  - `sparse_factor_numeric(...)` once
  - perturb later copies with the same pattern
  - `sparse_refactor_numeric(...)` on later iterations
  - `sparse_factor_solve(...)` on the final perturbed matrix
- report the timing breakdown in a form that makes the repeated-run story
  auditable instead of implied

The small but important review-quality correction is also explicit:

- the first Day 8 draft perturbed values by walking `SparseMatrix` internals
  through a private header
- `make quality-review-full` caught that on the reviewed CMake parity path
- the final benchmark uses only the public matrix API for perturbation, which
  makes the benchmark both portable and more faithful to the public proof
  boundary

### What stayed intentionally unchanged

Day 8 stayed inside the Sprint 52 scope fence:

- no direct-solver API redesign
- no changes to `sparse_analysis_t` / `sparse_factors_t` ownership
- no LU routing expansion
- no benchmark-framework redesign
- no docs/tutorial sweep beyond the benchmark ownership surface

### Validation

Because `bench_refactor.c` changed, the full required code-day gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this remained a substantial repeated-run proof patch, the stronger
reviewed baseline was also rerun:

- `make quality-review-full`

That also passed after the public-API portability fix.

### Focused follow-ons

The main benchmark proof is now measured rather than inferred:

- `./build/bench_refactor`
  - `tridiag-50` repeated-run speedup = `2.83x`
  - `tridiag-200` repeated-run speedup = `4.80x`
  - `tridiag-500` repeated-run speedup = `5.19x`
  - `bcsstk04` repeated-run speedup = `2.42x`
  - `nos4` repeated-run speedup = `2.49x`
  - final residuals stayed in the `1e-15` to `1e-16` range
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `analyze_ms = 1.224`
  - `refactor_ll_ms = 0.426`
  - `refactor_csc_ms = 0.186`
  - `speedup_refactor = 2.29x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`

### Day 8 conclusion

Sprint 52 now has measured factor-many evidence that matches the strengthened
shared lifecycle story:

- the repeated-run direct path is still ahead on the moderate and corpus cases
- the benchmark now proves same-pattern value-changing work instead of a static
  no-op refactor story
- the proof surface stays inside public API boundaries

The next bounded work should now focus on:

- any remaining direct-lifecycle regression proof
- later docs/example adoption
- keeping LU as the intentionally bounded special-case seam

## Day 9 outcome

Sprint 52 now has a post-benchmark caller-surface audit instead of a generic
“update the docs/examples” placeholder:

- the strongest remaining adoption work is narrower than the original Day 9
  placeholder implied
- `README.md` and `examples/example_analysis.c` are the only clearly
  high-value Day 10 adoption targets
- `examples/README.md` and `benchmarks/README.md` are already aligned enough
  to leave alone unless a very small supporting cross-reference becomes useful
- tutorial-scale rewrite and broad example conversion remain out of scope

### What was audited

The Day 9 audit focused on the strongest caller-facing repeated-run direct
surfaces after the Day 4-8 code and benchmark work:

- `README.md`
- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/README.md`

The audit also cross-checked the public contract home and the family-local
headers to make sure the caller-facing docs still match the real lifecycle
boundary:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

### Main findings

The remaining queue is now concrete:

- strongest Day 10 target:
  - `README.md`
    - it already names the analyze/factor/refactor path, but the high-level
      user-facing repeated-run direct story is still more compact than the now
      stronger Sprint 52 behavior
    - the best follow-on is a bounded clarification of:
      - analyze once
      - factor / solve
      - refactor / solve many
      - reuse preserves symbolic/permutation setup, not old numeric factor
        contents
      - obvious gross-structure drift is rejected cheaply, not fully proven
- strongest shipped example target:
  - `examples/example_analysis.c`
    - it is already the strongest repeated-run direct example
    - it now lags the Phase 2 story mainly in explanation, not in mechanics
    - the best follow-on is a bounded clarification of:
      - same-pattern value changes as the governing contract
      - what is being reused
      - why rebuilding a fresh matrix with the same pattern is still the right
        example-level discipline

Surfaces that are aligned enough to leave alone for now:

- `examples/README.md`
  - already names `example_analysis` as the strongest repeated-run direct
    example
  - already keeps the one-shot examples as first-class and simpler defaults
- `benchmarks/README.md`
  - Day 8 already brought the factor-many benchmark ownership story up to date
  - no additional widening is justified on an adoption day

### Day 10 boundary

Day 10 should stay tightly bounded:

- primary targets:
  - `README.md`
  - `examples/example_analysis.c`
- optional tiny supporting touch only if truly needed:
  - `examples/README.md`

Day 10 should explicitly avoid:

- broad tutorial rewrite
- sweeping conversion of one-shot examples into repeated-run examples
- broad benchmark README expansion
- reopening LU as anything other than the intentionally bounded special-case
  seam
- changing the public lifecycle contract instead of just reflecting it

### Day 9 conclusion

Sprint 52’s remaining adoption queue is now smaller and clearer:

- the benchmark proof and shared header contract are already in good shape
- the highest-value remaining public adoption work is concentrated in:
  - `README.md`
  - `examples/example_analysis.c`
- everything else can remain secondary unless a small supporting edit proves
  necessary

## Day 10 outcome

Sprint 52 now has a bounded public adoption batch instead of only the Day 9
audit target list:

- `README.md` now states the repeated-run direct workflow more explicitly as a
  compact public contract
- `examples/example_analysis.c` now explains the Phase 2 boundary more clearly
  in both comments and runtime output
- the one-shot direct APIs remain visible and first-class
- the batch stayed inside the Day 9 fence and did not widen into tutorial or
  broad example churn

### What changed

The Day 10 adoption batch landed in:

- `README.md`
- `examples/example_analysis.c`

The README now carries a more explicit direct repeated-run summary:

- public objects:
  - `sparse_analysis_t`
  - `sparse_factors_t`
- lifecycle:
  - analyze once
  - factor / solve
  - refactor / solve many
  - free explicitly
- key Phase 2 boundaries:
  - one-shot LU / Cholesky / LDL^T remain first-class peer entries
  - reuse preserves symbolic/permutation setup, not stale numeric factor
    contents
  - `sparse_refactor_numeric(...)` is the same-pattern numeric refresh path
  - obvious gross-structure drift is rejected cheaply, not fully proven

The example now teaches the same boundary more directly:

- file-level comments now state what reuse means and does not mean
- the refactor-loop comments now explain why rebuilding a fresh same-pattern
  matrix is still the safest high-signal example discipline
- runtime output now prints the repeated-run contract and the reused-vs-not-
  reused split

### What stayed intentionally unchanged

Day 10 stayed inside the adoption fence:

- no public lifecycle contract redesign
- no library implementation changes
- no broad tutorial rewrite
- no conversion of smaller one-shot examples into repeated-run examples
- no benchmark README widening beyond the already-finished Day 8 work
- no LU posture change

### Validation

Because `examples/example_analysis.c` changed, the full required code-day gate
was run:

- `make format`
- `make lint`
- `make test`

All passed.

### Focused follow-ons

The main shipped repeated-run direct example also stayed clean:

- `./build/example_analysis`
  - the new explanatory output now states:
    - reuse preserves symbolic/permutation setup
    - refactor expects the same sparsity pattern
    - fresh same-pattern matrices keep the contract explicit
    - stale numeric factor contents are not what gets reused
  - residuals remained `4.44e-16`

### Day 10 conclusion

Sprint 52 now has the highest-value caller-facing adoption surfaces aligned
with the stronger Phase 2 lifecycle story:

- the README now says the repeated-run direct contract compactly and
  truthfully
- the main shipped example teaches the same boundary in code comments and live
  output
- the batch stayed narrow enough that later work can focus on regression proof
  and compatibility review instead of reopening adoption drift

## Day 11 outcome

Sprint 52 now has a tighter public-lifecycle regression proof instead of only
the Day 10 caller-facing adoption alignment:

- the integration suite now proves that `sparse_factor_solve(...)` rejects a
  mismatched `sparse_analysis_t` / `sparse_factors_t` pairing
- the solve path now has direct public proof for both main mismatch classes:
  - wrong factor family -> `SPARSE_ERR_BADARG`
  - wrong dimension -> `SPARSE_ERR_SHAPE`
- the same regression also proves that a rejected mismatched solve does not
  damage the already-factored good state
- the batch stayed deliberately narrow:
  - no library implementation changes
  - no public contract changes
  - no benchmark or example churn

### What changed

The Day 11 regression batch landed in:

- `tests/test_integration.c`

The new coverage adds a direct public-lifecycle solve-time ownership check:

- build a valid Cholesky analysis/factors pair on a 4x4 SPD matrix
- build a mismatched LU analysis on a same-size unsymmetric matrix
- build a mismatched Cholesky analysis on a different-size SPD matrix
- verify:
  - `sparse_factor_solve(&factors, &lu_analysis, ...)` returns
    `SPARSE_ERR_BADARG`
  - `sparse_factor_solve(&factors, &other_n_analysis, ...)` returns
    `SPARSE_ERR_SHAPE`
  - the original good `factors` still solve correctly with the matching
    `good_analysis`

This closes the most obvious remaining Day 9/10 public-lifecycle regression
gap without widening the Sprint 52 scope.

### What stayed intentionally unchanged

Day 11 stayed within the regression-expansion fence:

- no `src/` library code changed
- no new repeated-run contract wording was introduced
- no LU routing or wrapper posture was reopened
- no broad parity sweep was added beyond the one missing high-signal public
  solve-time seam
- no README / example / benchmark surfaces were retouched

### Validation

Because `tests/test_integration.c` changed, the full required code-day gate
was run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` remained `53`
- Makefile/CMake parity remained `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 156.92 sec`

### Focused follow-ons

The targeted Day 11 public repeated-run follow-ons also stayed clean:

- `./build/test_integration`
  - `33 / 33` passed
  - the new regression passed directly:
    - `test_public_lifecycle_solve_rejects_mismatched_analysis_and_preserves_factors`
- `./build/example_analysis`
  - solve residual remained `4.44e-16`
- `./build/bench_refactor`
  - the repeated-run Cholesky proof remained ahead on all shipped fixtures:
    - `tridiag-200 4.78x`
    - `tridiag-500 5.24x`
    - `bcsstk04 2.48x`
    - `nos4 2.81x`

### Day 11 conclusion

Sprint 52 now has a more complete public-lifecycle regression floor:

- zeroed/unfactored solve rejection was already covered
- refactor acceptance, mismatch rejection, and old-factor preservation were
  already covered
- solve-time analysis/factors mismatch rejection and post-failure state
  preservation are now covered too

That keeps Day 12 focused on compatibility review rather than reopening the
public repeated-run proof surface.

## Day 12 outcome

Sprint 52’s landed Phase 2 branch still matches the Sprint 50-51
compatibility fence instead of only looking green in tests:

- one-shot LU / Cholesky / LDL^T entries still read as first-class peer entry
  points
- repeated direct runs still read as analysis/factors-centric rather than as a
  new generic direct-handle redesign
- reuse/refactor semantics remain honestly bounded in both code and docs
- benchmark and README/example claims still map to measured or explicitly
  bounded behavior

### What was audited

The Day 12 compatibility audit rechecked the live touched surfaces after the
Day 4-11 implementation and proof batches:

- shared repeated-run direct contract:
  - `include/sparse_analysis.h`
  - `src/sparse_analysis.c`
- family-local one-shot direct headers:
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- strongest public caller-facing adoption surfaces:
  - `README.md`
  - `examples/example_analysis.c`
  - `examples/README.md`
  - `benchmarks/README.md`
- strongest public proof surface:
  - `tests/test_integration.c`

### Compatibility conclusions

The live branch still matches the intended compatibility fence:

- one-shot APIs remain first-class
  - `include/sparse_lu.h` still presents LU as the simple/default copied-
    matrix path
  - `include/sparse_cholesky.h` still presents Cholesky as the one-shot SPD
    path
  - `include/sparse_ldlt.h` still preserves the family-local owned-factor
    LDL^T surface
  - `README.md` still calls the one-shot direct APIs first-class peer entry
    points
- repeated direct runs remain analysis/factors-centric
  - `include/sparse_analysis.h` still centers:
    - `sparse_analysis_t`
    - `sparse_factors_t`
    - analyze once
    - factor / solve
    - refactor / solve many
  - `examples/example_analysis.c` and `README.md` teach the same lifecycle
    shape rather than a competing one
  - `benchmarks/README.md` still treats `bench_refactor*` as proof of the
    same caller story, not a separate benchmark-only abstraction
- reuse/refactor semantics remain bounded
  - `include/sparse_analysis.h` still says reuse preserves
    symbolic/permutation setup, not stale numeric factors
  - `src/sparse_analysis.c` still enforces cheap dimension / original-matrix /
    `nnz` boundary checks, not a full structural-pattern verifier
  - `tests/test_integration.c` still proves:
    - zeroed first-factorization support
    - mismatch rejection
    - old-factor preservation on failure
    - solve-time analysis/factors mismatch rejection
- benchmark and docs claims remain honest
  - `benchmarks/README.md` now describes the measured outputs actually printed
    by `bench_refactor` and `bench_refactor_csc`
  - `README.md` uses bounded language such as measured speedups and cheap
    gross-structure rejection rather than universal guarantees

### Residual-risk classification

No blocker-level residual drift surfaced before Day 13.

The remaining residual risks are the expected bounded ones, not closeout
defects:

- LU is still the strongest intentionally family-local special-case seam
- repeated-run structure validation is still cheap `nnz` drift rejection, not
  a full sparsity-pattern verifier
- benchmark evidence remains representative measured proof, not a promise that
  every matrix family gets the same speedup

### Day 13 pre-validation checklist

The final validation checklist is now explicit from the landed state:

- required full gate:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- truthfulness anchors:
  - `ctest -N --test-dir build/quality-review-cmake`
  - Makefile/CMake parity check
  - full reviewed CMake `ctest`
- targeted Sprint 52 follow-ons:
  - `./build/example_analysis`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/test_integration`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`

### Day 12 conclusion

Sprint 52’s live branch still reads like the intended bounded Phase 2 package:

- stronger shared analysis/refactor integration
- preserved first-class one-shot family entries
- honestly bounded reuse/refactor semantics
- caller-facing docs/examples aligned with the implementation
- measured benchmark claims that still track the live binaries

That leaves Day 13 with a clean validation task rather than a hidden
compatibility repair queue.

## Day 13 outcome

Sprint 52 now has a full measured validation close state rather than only the
Day 12 compatibility audit and checklist:

- the full required gate passed:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the maintained reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - full reviewed CMake `ctest` = `53 / 53`
- the targeted Sprint 52 follow-ons also all passed

### Required gate results

Day 13 ran the full required closeout gate from the live Sprint 52 branch:

- `make format`
  - passed
- `make lint`
  - passed
- `make test`
  - passed
- `make quality-review-full`
  - passed

The strongest reviewed truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake`
  - `53`
- Makefile/CMake parity
  - `53 vs 53`
- full reviewed CMake `ctest`
  - `53 / 53`
- `Total Test time (real)`
  - `200.43 sec`

### Targeted follow-ons

The targeted Sprint 52 follow-ons also all ran cleanly:

- `./build/test_integration`
  - `33 / 33`
- `./build/example_analysis`
  - solve residual stayed `4.44e-16`
  - repeated-run output still states:
    - reused state = symbolic/permutation setup only
    - not reused = stale numeric factor contents
- `./build/bench_refactor`
  - repeated-run direct path stayed ahead on all shipped fixtures:
    - `tridiag-50 2.73x`
    - `tridiag-200 4.81x`
    - `tridiag-500 5.28x`
    - `bcsstk04 2.45x`
    - `nos4 2.72x`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - repeated-run CSC path stayed ahead:
    - `speedup_refactor = 1.52x`
    - `res_ll = 8.24e-16`
    - `res_csc = 7.06e-16`
- direct family/regression reruns:
  - `./build/test_cholesky`
    - `21 / 21`
  - `./build/test_ldlt`
    - `83 / 83`
  - `./build/test_etree`
    - `97 / 97`
  - `./build/test_chol_csc`
    - `137 / 137`
  - `./build/test_ldlt_csc`
    - `95 / 95`

### Validation conclusion

The final measured close state matches the intended Sprint 52 package:

- the stronger shared analysis/factor/refactor path still validates cleanly
- the preserved one-shot family paths still validate cleanly
- the factor-many benchmark story remains measured and positive
- the reviewed Makefile/CMake truthfulness baseline stayed exact through the
  full closeout gate

No new reconciliation queue surfaced during validation. That leaves Day 14 as
true closeout and handoff work, not post-validation repair.

## Day 14 outcome

Sprint 52 now closes from the Day 13 validated baseline with one coherent
Phase 2 direct-lifecycle package instead of only a sequence of bounded
implementation and audit batches.

### Sprint 52 delivered package

Sprint 52 leaves behind one coherent Phase 2 package:

- stronger shared analysis/factor integration on the highest-value repeated-run
  paths:
  - shared Cholesky CSC path now reuses the caller's `sparse_analysis_t`
    directly on larger repeated-run problems
  - shared LDL^T CSC path now reuses the caller's `sparse_analysis_t`
    directly when the scalar pivot pre-pass does not introduce extra swaps
- tighter shared refactor boundary:
  - zero-init first-factorization support preserved
  - family/dimension/payload mismatch rejection made more explicit
  - cheap gross-structure drift rejection added via analyzed `nnz` tracking
- refreshed factor-many benchmark proof:
  - `bench_refactor` now measures real same-pattern value changes
  - `bench_refactor_csc` still proves the heavier CSC repeated-run path
- aligned high-signal caller-facing adoption:
  - `README.md`
  - `examples/example_analysis.c`
- expanded public repeated-run regression proof in:
  - `tests/test_integration.c`

### Preserved compatibility fence

Sprint 52 closes with the Sprint 50-51 compatibility rules still intact:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- repeated direct runs remain analysis/factors-centric around:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- reuse preserves symbolic/permutation setup, not stale numeric factor
  contents
- repeated-run structure validation remains a cheap boundary check, not a full
  structural-pattern verifier
- LU remains the strongest intentionally family-local special-case seam
- no raw CSC/native storage layout was exposed
- no generic direct-handle redesign was introduced

### Validation close state

Sprint 52 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained truthfulness anchors:

- reviewed CMake parity = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 200.43 sec`

Representative measured Sprint 52 follow-on results:

- `example_analysis` residual remained `4.44e-16`
- `bench_refactor` kept the repeated-run direct path ahead:
  - `tridiag-200 4.81x`
  - `tridiag-500 5.28x`
  - `bcsstk04 2.45x`
  - `nos4 2.72x`
- `bench_refactor_csc nos4` kept the CSC repeated-run path ahead:
  - `speedup_refactor = 1.52x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`

### Handoff to Sprint 53

Sprint 53 no longer needs to prove that the shared public direct lifecycle is
real or validated for the main Phase 2 paths.

The next queue can therefore stay bounded to real post-Sprint-52 work such as:

- later direct-solver lifecycle depth beyond the Sprint 52 fence
- stronger or broader same-pattern structure validation if a later sprint
  chooses to pay that cost
- any later LU-specific follow-on that should remain family-local rather than
  reopening the shared direct contract
- future caller-surface or benchmark expansion that builds on the now-validated
  Phase 2 package

### Project-plan impact

Sprint 52 does not require a `PROJECT_PLAN.md` update.

Reason:

- the sprint closed from the planned Day 13 validation baseline
- the delivered package still matches the Epic 5 Sprint 52 intent
- no blocker or replanning queue surfaced during closeout

### Day 14 conclusion

Sprint 52 is complete. It hands off a validated Phase 2 direct-solver
lifecycle package with:

- stronger shared analysis/refactor integration
- preserved first-class one-shot family entries
- honestly bounded reuse/refactor semantics
- measured factor-many benchmark proof
- aligned high-signal docs/example surfaces
- stable reviewed-baseline truthfulness anchors

The next queue is now bounded to real follow-on work instead of unresolved
Phase 2 closeout defects.
