# Sprint 82 Day 5: Dense-Kernel ABI Design

## Purpose

Define the bounded dense-kernel descriptor and runtime-selection contract that
Sprint 82 will actually land on the Cholesky CSC supernodal lane.

## Main Result

Sprint 82 now has one explicit first implementation contract:

- required implementation center:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- support only if the first batch truly forces it:
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Ownership Split

The Day 5 ownership split is now fixed:

- dense-kernel descriptor and builtin-default owner:
  - `src/sparse_dense.c`
- supernodal batch-time consumer and local backend-contract boundary owner:
  - `src/sparse_chol_csc_supernodal.c`
- family-level orchestration and caller-facing publication owner:
  - `src/sparse_chol_csc.c`

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- the first landing should preserve the builtin self-contained backend as the
  default product path
- it should widen the dense-kernel seam with one bounded optional runtime
  selection contract rather than a broad backend framework
- it should keep backend observability local to the touched Cholesky lane
  rather than widening into repo-wide runtime policy churn
- it should not reopen LDL^T, QR, SVD, package/platform convergence, or
  broader capability work in the same batch

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- self-contained default build remains the main product path
- optional acceleration remains bounded and proof-backed
- benchmark reporting remains threshold-free
- no fake platform/shared-library maturity or generic BLAS-everywhere claim

## Exit State

- Sprint 82 now has one bounded dense-kernel ABI/runtime contract.
- Ownership between descriptor, consumer, and caller-facing publication is
  fixed before Day 6 implementation begins.
