# Sprint 82 Day 1: Scope and Dense-Backend Baseline

## Purpose

Turn the Sprint 82 project-plan section and the Sprint 81 validated closeout
into one bounded dense-backend execution package before any backend-aware code
lands.

## Starting Truth

Sprint 82 begins from a validated Sprint 81 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 81 already moved the strongest prior contradiction:

- high-value linked-list-first construction/import costs were reduced
- repeated-run Cholesky and LDL^T now stay on the analysis-backed CSC-aware
  path for all problem sizes

That means Sprint 82 can start from the next real Epic 8 contradiction center:

- builtin scalar dense/backend performance ceiling

## Sprint 82 Workstreams

The highest-value Sprint 82 package is now fixed explicitly around:

- dense hotspot profiling
- backend ABI design
- first optional accelerated dense-kernel integration
- solver adoption follow-through
- focused benchmark and differential proof
- packaging/runtime alignment only where implementation truly moves the
  contract
- validation and closeout

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 82 surfaces:

- public/direct-family contract surfaces:
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
- dense-helper and solver-family implementation seams:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
- strongest proof and reporting surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `benchmarks/README.md`
  - `README.md`
  - `docs/maintainer_guide.md`

## Preserved Fence

Sprint 82 is explicitly bounded against:

- capability-surface widening
- broad package/platform reopening
- fake platform or shared-library maturity claims
- benchmark-threshold inflation
- mandatory heavyweight optional-backend dependency for the default build
- generic whole-library performance churn

## Day 1 Result

Sprint 82 now starts from one precise dense/backend execution package rather
than from a generic “performance improvement” bucket. The strongest likely
touch surfaces, preserved non-goals, and validated baseline are fixed in
writing before the validation/proof recheck begins.
