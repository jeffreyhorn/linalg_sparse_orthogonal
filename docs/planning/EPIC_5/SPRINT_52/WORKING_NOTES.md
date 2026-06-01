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
