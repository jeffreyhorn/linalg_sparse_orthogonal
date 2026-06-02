# Sprint 53 Day 1 - scope and CSC completion baseline

Date: 2026-06-01
Branch: `sprint-53`

## Scope

Start Sprint 53 from the actual Sprint 52 validated Phase 2 close state and
reduce the next work to a bounded CSC direct-solver follow-through queue.

## Authoritative baseline

Sprint 53 starts from a preserved reviewed validation baseline:

- strongest local reviewed baseline: `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

This means Sprint 53 is not a validation-recovery sprint. It is a CSC
follow-through sprint.

## What Sprint 52 already proved

The following is already real before Sprint 53 begins:

- stronger shared analysis/factor/refactor direct integration
- shared Cholesky CSC repeated-run path reuses caller analysis directly on the
  highest-value larger-problem path
- shared LDL^T CSC repeated-run path reuses caller analysis directly when the
  scalar BK pre-pass does not introduce extra swaps
- tighter bounded refactor contract with cheap gross-structure drift rejection
- refreshed repeated-run benchmark evidence in:
  - `benchmarks/bench_refactor.c`
  - `benchmarks/bench_refactor_csc.c`
- aligned caller-facing repeated-run adoption in:
  - `README.md`
  - `examples/example_analysis.c`
- expanded repeated-run regression proof in:
  - `tests/test_integration.c`

Interpretation:

- Sprint 53 does not need to prove the public repeated-run direct workflow is
  real
- Sprint 53 needs to close the most important CSC-specific completion gaps
  that still make the dispatch and indefinite story harder to reason about

## Actual Sprint 53 queue

The Sprint 53 project-plan items reduce to six bounded work classes:

1. LDL^T analysis-aware indefinite path audit and completion
2. transparent LDL^T dispatch follow-through
3. indefinite CSC factor-many proof
4. Cholesky / LDL^T dispatch reconciliation
5. targeted benchmark and regression refresh
6. validation and closeout

The strongest architectural narrowing is:

- keep the work centered on the existing analysis/factors direct lifecycle
- complete or tighten CSC-specific preparation and dispatch seams
- strengthen measured indefinite CSC evidence
- do not broaden into a new public abstraction or raw CSC/native storage
  exposure

## Main hotspots

Highest-value touched surfaces at sprint start:

- public/shared contract:
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `320`
- shared / family implementation:
  - `src/sparse_analysis.c` = `818`
  - `src/sparse_cholesky.c` = `494`
  - `src/sparse_ldlt.c` = `1494`
- CSC implementation:
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_ldlt_csc.c` = `2723`
  - `src/sparse_chol_csc_internal.h` = `994`
  - `src/sparse_ldlt_csc_internal.h` = `805`
- proof/adoption:
  - `benchmarks/bench_refactor_csc.c` = `388`
  - `tests/test_integration.c` = `1529`
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_ldlt_csc.c` = `3637`
  - `README.md` = `930`
  - `benchmarks/README.md` = `191`
  - `examples/example_analysis.c` = `210`

Interpretation:

- the strongest risk seams cluster in LDL^T CSC implementation and the
  dispatch/proof surfaces around it
- the strongest proof surfaces remain the CSC tests, the shared integration
  test file, and the CSC refactor benchmark

## Preserved fence

Sprint 53 still inherits the controlling compatibility boundary:

- one-shot LU / Cholesky / LDL^T remain first-class peer entry points
- repeated direct runs remain analysis/factors-centric
- reuse preserves symbolic/permutation setup, not stale numeric factor
  contents
- repeated-run structure validation remains a cheap boundary check rather than
  a full structural-pattern verifier
- no raw internal CSC/native storage exposure
- no generic direct-handle redesign

## Conclusion

Day 1 fixes Sprint 53's real starting point:

- preserved reviewed baseline
- validated Sprint 52 Phase 2 handoff
- bounded CSC completion queue
- named CSC code/test/benchmark/doc hotspots
- preserved compatibility and non-goal fence

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening earlier Sprint 50-52 direct-lifecycle design decisions.
