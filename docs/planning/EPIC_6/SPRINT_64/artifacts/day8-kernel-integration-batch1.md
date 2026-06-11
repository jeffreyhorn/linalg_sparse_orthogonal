# Sprint 64 Day 8: Kernel Integration Batch 1

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Land the first bounded backend-aware kernel integration slice for the selected
Sprint 64 hot path:

- Cholesky CSC supernodal dense-kernel integration

without widening the public API, the build/option surface, or the sprint
fence.

## Landed Surfaces

Implementation:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`

Proof:

- `tests/test_chol_csc.c`

## Main Result

Sprint 64 now has its first bounded backend-aware integration seam.

The landed batch introduces one internal dense-kernel descriptor for the
selected Cholesky CSC supernodal lane:

- `chol_dense_kernels_t`
- `chol_csc_supernodal_dense_kernels()`

That descriptor is:

- declared in `src/sparse_chol_csc_internal.h`
- backed by the builtin self-contained dense helpers in `src/sparse_dense.c`
- consumed by `src/sparse_chol_csc_supernodal.c`

The batch stayed inside the Day 7 fence:

- no public header widening
- no `CMakeLists.txt` or `Makefile` changes
- no LDL^T, QR, or SVD spillover
- no benchmark or docs widening in the first code batch

## Implementation Follow-Through

The core Day 8 architecture change is that the selected hot path no longer
owns its dense helpers as file-local hardwired implementations.

Instead:

- the builtin dense factor helper and lower-triangular solve helper live in
  `src/sparse_dense.c`
- the supernodal Cholesky path resolves them through the bounded internal
  descriptor
- the supernodal call sites now guard against a missing descriptor or missing
  function pointers and return `SPARSE_ERR_BACKEND_CONTRACT` explicitly rather than
  assuming an always-inlined local helper

This keeps the first Sprint 64 abstraction:

- internal-first
- default-safe
- local to the selected Cholesky CSC supernodal lane

It does not attempt to create a repository-wide backend hub yet.

## New Proof

`tests/test_chol_csc.c` now includes:

- `test_supernodal_dense_backend_default_contract`

That proof pins the minimum viable Day 8 contract:

- `chol_csc_supernodal_dense_kernels()` is non-null
- the builtin backend name is present
- the factor and solve function pointers are present
- the builtin kernel pair still factors and solves a simple dense `2 x 2`
  contract case correctly

This keeps the first backend-aware proof family-local and avoids widening into
the public integration surface before the semantics actually require it.

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
- `Total Test time (real) = 355.19 sec`

## Non-Blocking Note

The reviewed CMake path again completed cleanly, but `test_reorder_nd` ran
slower than the ordinary local path in the rebuilt reviewed tree and took
`229.54 sec`. That affected total reviewed runtime, not correctness, parity,
or pass/fail status.

## Exit State

Sprint 64 Day 8 now hands off a smaller and more concrete follow-through
queue:

- the first backend-aware integration seam is real
- the selected hot path still preserves the self-contained builtin default
- no public or build-surface widening was required for the first landing
- Day 9 can now audit the live branch for remaining fallback, selection, and
  benchmark-proof follow-through from the actual landed code
