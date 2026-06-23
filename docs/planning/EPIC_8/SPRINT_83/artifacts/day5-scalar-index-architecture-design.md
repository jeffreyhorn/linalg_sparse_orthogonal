# Sprint 83 Day 5: Scalar / Index Architecture Design

## Purpose

Define the bounded scalar/index contract that Sprint 83 will actually land on
the shared matrix-shell and public-owner lane.

## Main Result

Sprint 83 now has one explicit first implementation contract:

- required implementation center:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- support only if the first batch truly forces it:
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Ownership Split

The Day 5 ownership split is now fixed:

- shared scalar and width vocabulary owner:
  - `include/sparse_types.h`
- public matrix-shell exposure owner:
  - `include/sparse_matrix.h`
- compatibility-preserving implementation and publication owner:
  - `src/sparse_matrix.c`
- family-level adoption follow-through owners, but not in the first batch:
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- the first landing should preserve the shipped scalar contract as real-only
  `double` even while widening ownership onto the shared public seams
- it should widen the shared matrix-shell/public-owner reading to use the
  already-real `sparse_scalar_t` / `idx_t` vocabulary where that can be done
  without implying broad numeric genericity
- it should keep compatibility-preserving internal representation and
  publication behavior centered in `src/sparse_matrix.c` rather than widening
  immediately into family-local algorithm code
- it should not reopen QR, SVD, Cholesky, LDL^T, true complex support, broad
  mixed precision, or generic package/platform maturity in the same batch

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- current callers should keep a truthful real-only reading
- width remains a compile-time contract, not a runtime-generic claim
- no repo-wide scalar genericity claim
- no benchmark, install/export, or package wording drift unless the touched
  public contract truly forces it

## Exit State

- Sprint 83 now has one bounded scalar/index architecture contract.
- Ownership between shared vocabulary, public matrix exposure, and
  compatibility-preserving implementation is fixed before Day 6 begins.
- Family-local capability widening remains explicitly outside the first batch.
