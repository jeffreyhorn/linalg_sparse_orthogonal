# Sprint 56 Day 1 - scope and Phase 2 decomposition baseline

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Start Sprint 56 from the actual Sprint 55 large-source decomposition close
state and the Epic 5 remaining hotspot queue, then reduce the next work to a
bounded CSC direct-solver plus SVD decomposition package centered on the
highest-value remaining production implementation files.

## Authoritative baseline

Sprint 56 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 56 is not a validation-recovery sprint. It is a
maintainability and ownership sprint.

## What Sprint 55 already proved

The following is already real before Sprint 56 begins:

- bounded Phase 1 large-source decomposition already landed
- the main eigensolver hotspot was materially reduced:
  - `src/sparse_eigs.c`: `3233` -> `1534`
- the first iterative extraction already landed:
  - `src/sparse_iterative.c`: `2377` -> `1985`
  - `src/sparse_iterative_minres.c` exists as an owned implementation file
- one-shot public APIs remain first-class supported entry points
- the repeated direct-run analysis/factors lifecycle remains the validated
  direct-solver contract
- the repeated-run iterative/eigensolver support boundary remains the
  validated Sprint 54 shape
- build-surface parity was already preserved across:
  - `Makefile`
  - `CMakeLists.txt`

Interpretation:

- Sprint 56 does not need to re-decide the public solver or lifecycle surface
- Sprint 56 needs to improve internal implementation ownership while
  preserving that already-validated public contract

## What the Epic 5 review and todo list already fixed as the next queue

The Epic 5 review and todo notes already point to the same bounded
maintainability problem:

- `src/sparse_ldlt_csc.c` remains a top-tier large-source hotspot
- `src/sparse_chol_csc.c` remains a top-tier large-source hotspot
- `src/sparse_svd.c` remains a top-tier dense-algorithm maintainability
  hotspot
- the right improvement shape is:
  - split by stable ownership seams
  - separate helper logic from orchestration
  - reduce stale sprint-history narrative in permanent implementation files

The live repo state now confirms that the review queue is still current:

- `src/sparse_ldlt_csc.c` = `2723` lines
- `src/sparse_chol_csc.c` = `2194` lines
- `src/sparse_svd.c` = `1728` lines

Interpretation:

- Sprint 56 should treat the Epic 5 review as still live, not historical
- the remaining large-file queue is now concentrated enough to work directly
  from the production files instead of from generic review categories

## Actual Sprint 56 queue

The Sprint 56 project-plan items reduce to seven bounded work classes:

1. `sparse_ldlt_csc.c` residual audit
2. LDLT CSC decomposition batch
3. `sparse_chol_csc.c` residual audit
4. Cholesky CSC decomposition batch
5. `sparse_svd.c` maintainability batch
6. touched-doc and comment reconciliation on touched permanent implementation
   files
7. validation and closeout

The strongest architectural narrowing is:

- keep the work centered on the remaining large CSC direct-solver and SVD
  translation units
- prefer helper-vs-orchestration ownership splits over generic mechanical file
  splits
- preserve the Sprint 50-55 direct-solver and solver support boundary exactly
- do not broaden into public API redesign, new direct/solver-family exposure,
  or large documentation rewrites

## Main hotspots

Highest-value touched surfaces at sprint start:

- public headers:
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_svd.h` = `257`
- main implementations:
  - `src/sparse_ldlt_csc.c` = `2723`
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_svd.c` = `1728`
  - `src/sparse_ldlt_csc_internal.h` = `877`
  - `src/sparse_chol_csc_internal.h` = `994`
  - `src/sparse_svd_internal.h` = `21`
- proof surfaces:
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_svd.c` = `3746`
  - `tests/test_integration.c` = `1803`
  - `benchmarks/bench_refactor_csc.c` = `611`
- caller-facing adoption:
  - `examples/example_analysis.c` = `210`
  - `README.md` = `987`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest implementation risk seams now sit in the two CSC direct-solver
  implementation files, with `src/sparse_svd.c` as the strongest non-CSC
  maintainability follow-on
- the proof surfaces are large enough that extraction work must preserve
  benchmark/test parity deliberately

## Preserved fence

Sprint 56 still inherits the controlling compatibility and non-goal boundary:

- one-shot APIs remain first-class peer entry points
- the analysis/factors repeated direct-run path remains the validated direct
  lifecycle shape
- the repeated-run iterative/eigensolver support boundary remains unchanged
- no broad public API redesign
- no raw CSC/native storage exposure
- no generic public direct-handle redesign
- no unrelated dense-algorithm or solver-family expansion

## Conclusion

Day 1 fixes Sprint 56's real starting point:

- preserved reviewed baseline
- inherited validated decomposition and public-contract fence
- bounded remaining CSC/SVD decomposition queue
- named implementation and proof hotspots
- explicit non-goal fence against public API expansion

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 50-55 public contract decisions.
