# Sprint 64 Day 10: Backend-Contract Error and Fallback-Truthfulness Batch

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Land the bounded Day 10 fallback/error-path truthfulness slice for the first
Sprint 64 backend-aware integration seam:

- introduce a real public error-taxonomy answer for backend-contract failure
- wire the Cholesky CSC supernodal dense-kernel seam to use that answer
- prove the missing-descriptor and missing-function-pointer paths directly in
  the family-local test surface

## Landed Surfaces

Public error taxonomy:

- `include/sparse_types.h`
- `src/sparse_types.c`

Implementation:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`

Proof:

- `tests/test_chol_csc.c`

## Main Result

Sprint 64 now has a real public error-taxonomy answer for the new
backend-aware dense-kernel seam:

- `SPARSE_ERR_BACKEND_CONTRACT`

That new public enum value is used when:

- the caller contract was valid
- but the selected backend-owned implementation path could not resolve a
  required internal helper/callback

For the Day 10 hot path, that means the Cholesky CSC supernodal lane now
returns `SPARSE_ERR_BACKEND_CONTRACT` when it cannot resolve:

- the active dense-kernel descriptor
- the descriptor's `factor` callback
- the descriptor's `solve_lower` callback

The batch stayed inside the Day 9 fence:

- no benchmark widening
- no integration-test widening
- no build-option work
- no public Cholesky API/header widening
- no LDL^T / QR / SVD spillover

## Implementation Follow-Through

The Day 10 implementation has two parts:

### 1. Public error-taxonomy completion

Added:

- `SPARSE_ERR_BACKEND_CONTRACT` to `include/sparse_types.h`
- matching `sparse_strerror(...)` support in `src/sparse_types.c`

This keeps the backend-aware seam from overloading `SPARSE_ERR_BADARG` for an
internal implementation-contract violation.

### 2. Family-local backend seam proofability

The bounded test seam introduced in Day 10 lets `tests/test_chol_csc.c`
override the active dense-kernel descriptor temporarily:

- `chol_csc_supernodal_set_dense_kernels_override_for_test(...)`
- `chol_csc_supernodal_clear_dense_kernels_override_for_test(...)`

That made it possible to prove the exact new failure contract without widening
the public surface or relying on linker tricks or build-mode changes.

## New Proof

`tests/test_chol_csc.c` now proves all three Day 10 fallback/error-path cases
directly:

- `test_supernode_eliminate_diag_missing_dense_kernel_descriptor_is_backend_contract_error`
- `test_supernode_eliminate_diag_missing_factor_kernel_is_backend_contract_error`
- `test_supernode_eliminate_panel_missing_solve_kernel_is_backend_contract_error`

This means the first Sprint 64 backend-aware seam no longer depends on the
implicit assumption that the builtin descriptor is always present and always
complete.

## Validation

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 365.95 sec`

## Non-Blocking Note

The Day 10 ordinary `make test` path again spent most of its wall time in the
usual `test_reorder_nd` tail. The reviewed CMake path again re-emitted the
existing `bench_eigs_reuse.c` double-promotion warnings while rebuilding
`bench_eigs_reuse`, but still completed cleanly and passed all parity gates.

## Exit State

Sprint 64 Day 10 now hands off a materially smaller remaining queue:

- the backend-aware Cholesky CSC seam has a real public error-code contract
- the supernodal path now reports backend-contract failure explicitly
- family-local proof now exercises the missing-descriptor and missing-function
  pointer paths directly
- later benchmark or docs follow-through can be judged from the actual landed
  semantics instead of from pre-landing assumptions
