# Sprint 92 Day 5: Portable Backend ABI and Runtime Contract Design

## Purpose

Define the bounded builtin-vs-portable backend contract so Day 6 can widen the
shared dense-kernel seam without breaking builtin-default truth or reopening
the Day 4 boundary.

## Main Result

Sprint 92 now has one explicit builtin-vs-portable backend contract:

- builtin dense kernels:
  - remain the default, self-contained, always-available product truth
  - continue to define correctness and fallback semantics for every caller
  - must stay usable even when no optional backend is present

- optional portable backend lane:
  - should widen the shared dense-kernel descriptor/runtime-selection seam
    rather than creating another family-local acceleration pocket
  - should remain optional and capability-gated
  - should fail closed to builtin kernels when unavailable or unsupported

- runtime or compile-time selection:
  - stays bounded to the shared dense owner
  - should present one explicit backend name / descriptor contract to direct
    consumers and proof surfaces
  - should not turn Sprint 92 into a broad public configuration-product
    rewrite

## Strongest Clarification

The useful Day 5 clarification is now explicit:

- Sprint 92 should not try to solve every backend problem at once
- the first landing should widen the shared descriptor/runtime-selection seam
  around the Cholesky dense kernel path
- LDL^T, QR, broader benchmark/reporting follow-through, and package wording
  stay later unless the shared seam truly forces them

## Exact Day 6 Center

The exact Day 6 implementation center is now fixed to:

- `src/sparse_dense.c`

Directly forced follow-through is limited to:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_internal.h`
- `tests/test_dense.c`
- `tests/test_chol_csc.c`
- `benchmarks/bench_chol_csc.c`

## Deferred From The First Backend Landing

The first batch now explicitly defers:

- a broad public API redesign in `include/sparse_dense.h`
- fake platform-symmetry claims
- LDL^T backend convergence as a first-batch center
- QR/backend adoption as a first-batch center
- build/package/workflow wording churn detached from the shared dense seam
- runtime/threading or capability-surface widening

## Exit State

- Sprint 92 has one explicit builtin-vs-portable backend contract before code
  movement.
- Day 6 is fixed to the shared dense-kernel seam with only the strongest
  Cholesky-side adopter as directly forced follow-through.
- Later LDL^T, QR, benchmark, and package work remains sequenced behind the
  first backend landing.
