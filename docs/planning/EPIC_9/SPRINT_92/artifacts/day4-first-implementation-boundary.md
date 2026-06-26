# Sprint 92 Day 4: First Implementation Boundary

## Purpose

Fix one bounded first implementation fence so Sprint 92 starts with the
highest-value backend seam instead of generic dense or direct-family churn.

## Main Result

Sprint 92 now has one explicit first implementation fence:

- required first landing:
  - `src/sparse_dense.c`
  - the matching shared dense-kernel descriptor and optional-backend seam
    consumed by the strongest existing direct-family owner

- directly forced support surfaces only if the first landing truly needs them:
  - `src/sparse_chol_csc.c`
  - `include/sparse_chol_csc_internal.h`
  - `tests/test_dense.c`
  - `tests/test_chol_csc.c`
  - `benchmarks/bench_chol_csc.c`

- explicitly later unless the first landing truly forces movement:
  - `src/sparse_ldlt_csc.c`
  - `include/sparse_ldlt.h`
  - `tests/test_ldlt_csc.c`
  - `src/sparse_qr.c`
  - `tests/test_qr.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - install/export and workflow surfaces

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- Sprint 92 should start by improving the shared dense-kernel seam
- it should not begin by widening every dense consumer at once
- it should not reopen QR, package wording, runtime/threading, or fake
  cross-platform symmetry in the first batch unless the shared seam itself
  truly forces it

## Deferred From The First Landing

The first batch now explicitly defers:

- broad dense rewrite
- family-wide direct-solver backend convergence as a first-batch center
- QR/backend adoption as a first-batch center
- benchmark/reporting widening detached from a real backend seam
- build/package/workflow wording churn detached from the first code landing
- runtime/threading or capability-surface widening

## Exit State

- Sprint 92 has one explicit first implementation boundary.
- The first code landing is fixed to the shared dense-kernel owner and the
  strongest immediate Cholesky-side adoption seam.
- Day 5 can define the backend ABI/runtime contract without reopening the
  ranked first-center choice.
