# Sprint 83 Day 4: First Capability Boundary

## Purpose

Fix the first bounded capability implementation fence for Sprint 83 so the
next design pass can define one real scalar/index contract instead of another
broad capability rewrite.

## Main Result

Sprint 83 now has one explicit first implementation fence:

- required first landing:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- support only if the first landing truly forces it:
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
- explicitly deferred from the first landing:
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - broad algorithm-family widening as a first-batch center
  - true complex-scalar support
  - broad mixed-precision support
  - generic package/platform maturity widening

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 83 move is the shared public scalar/index owner on the
  matrix shell and its highest-value compatibility seams
- touched-path wider-index and ABI maturity remains the strongest second seam,
  not the first implementation center
- QR / SVD family-local capability widening remains real, but it is explicitly
  later than the first shared-contract landing
- proof and support surfaces stay support-only unless the first landing truly
  changes behavior there

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no repo-wide complex-number promise
- no broad mixed-precision framework
- no ABI churn detached from touched public seams
- no algorithm-family widening before the shared contract is explicit
- no benchmark-governance drift
- no support-surface churn detached from a real landed capability seam

## Exit State

- Sprint 83 now has one bounded first capability landing center.
- Day 5 can design one scalar/index architecture contract inside that fence.
- Lower-value QR / SVD family breadth, later complex/mixed-precision work, and
  broader support/package spillover are held back until later lanes.
