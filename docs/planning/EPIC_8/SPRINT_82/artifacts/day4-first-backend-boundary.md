# Sprint 82 Day 4: First Backend Boundary

## Purpose

Fix the first bounded backend implementation fence for Sprint 82 so the next
design pass can define one real dense-kernel contract instead of another broad
performance rewrite.

## Main Result

Sprint 82 now has one explicit first implementation fence:

- required first landing:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- support only if the first landing truly forces it:
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- explicitly deferred from the first landing:
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `benchmarks/bench_svd.c`
  - broad package/platform convergence
  - broad state-of-the-art comparison work

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 82 move is the dense-kernel descriptor and Cholesky
  CSC supernodal consumer lane
- LDL^T backend/runtime parity remains the strongest second seam, not the first
  implementation center
- QR/SVD dense-workspace work remains real, but it is explicitly later than the
  first backend landing
- proof and benchmark surfaces stay support-only unless the first landing truly
  changes behavior there

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no mandatory heavyweight optional-backend dependency for the default build
- no fake platform parity or shared-library maturity claim
- no benchmark timing-gate conversion
- no broad direct-family or whole-library backend rewrite

## Exit State

- Sprint 82 now has one bounded first backend landing center.
- Day 5 can design one dense-kernel ABI/runtime contract inside that fence.
- Lower-value QR/SVD and broader packaging/platform spillover are held back
  until later lanes.
