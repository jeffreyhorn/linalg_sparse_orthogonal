# Sprint 82 Day 6 - Optional Dense-Backend Integration Batch

Date: 2026-06-19  
Branch: sprint-82

## Purpose

Land one bounded optional accelerated dense-backend slice on the Cholesky CSC
supernodal lane while preserving the builtin backend as the default product
path and keeping proof local to the family-level Cholesky surface.

## Main Result

The bounded Day 6 backend batch landed in:

- `src/sparse_dense.c`
- `tests/test_chol_csc.c`

The main implementation result is now explicit:

- the shipped builtin Cholesky dense-kernel descriptor remains the default path
- the dense-kernel owner now recognizes one bounded runtime selection knob:
  - `SPARSE_CHOL_DENSE_BACKEND=accelerate`
- on Darwin only, that knob can activate an optional Accelerate-backed dense
  descriptor for:
  - dense diagonal factor
  - dense lower solve
  - dense panel solve
- if the optional runtime path is unavailable or not requested, the builtin
  descriptor remains the active product path

## Landed Ownership

### Dense-kernel owner

- `src/sparse_dense.c`
  - still owns the shipped builtin dense-kernel descriptor
  - now owns the bounded runtime selector
  - now publishes an optional Darwin-only Accelerate-backed descriptor when the
    runtime probe succeeds

### Proof owner

- `tests/test_chol_csc.c`
  - owns builtin env-selection proof
  - owns accelerate env-selection proof
  - owns the callback-completeness checks under the selected descriptor
  - owns the small dense correctness checks for the accelerated callbacks when
    the runtime backend is actually active

## Preserved Fence

The Day 5 fence held:

- no mandatory external dependency was added to the default build
- no LDL^T backend/runtime widening occurred
- no QR or SVD widening occurred
- no package/platform maturity claim widened beyond the bounded Darwin runtime
  seam
- no benchmark/reporting or docs spill was needed

## Validation

Because `*.c` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 438.98 sec`

## Exit State

- Sprint 82 now has one real optional accelerated dense-kernel slice on the
  Cholesky CSC supernodal lane.
- The builtin backend remains the default shipped path.
- Optional runtime selection is now proof-backed and bounded instead of being
  only a design intent.
