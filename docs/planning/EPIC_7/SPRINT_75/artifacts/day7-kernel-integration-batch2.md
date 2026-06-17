# Sprint 75 Day 7 Artifact: Kernel Integration Batch 2

Date: 2026-06-17
Branch: sprint-75

## Purpose

Land the first real backend-aware dense-kernel follow-through inside the CSC
supernodal Cholesky lane while staying inside the bounded Day 6 ownership and
proof fence.

## Main Result

Sprint 75 Day 7 landed one bounded kernel-integration batch across:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`

The main backend-aware result is:

- the internal dense-kernel descriptor now owns a batched panel-solve callback
- the supernodal CSC panel-elimination path now consumes that callback
  directly
- missing batched-panel support now fails through the narrow
  `SPARSE_ERR_BACKEND_CONTRACT` boundary
- the proof stayed family-local in `tests/test_chol_csc.c`

## Landed Ownership

### Dense-kernel owner

- `src/sparse_dense.c`
  - owns `chol_dense_solve_panel(...)`
  - publishes the builtin `solve_panel` callback in the shipped dense-kernel
    descriptor

### Supernodal consumer

- `src/sparse_chol_csc_supernodal.c`
  - consumes `kernels->solve_panel`
  - uses one batched panel solve instead of repeated single-RHS lower solves
  - returns `SPARSE_ERR_BACKEND_CONTRACT` if the required callback is missing

### Internal contract seam

- `src/sparse_chol_csc_internal.h`
  - now names the batched panel-solve callback explicitly in the internal
    kernel descriptor

## Proof

The first proof owner stayed bounded to:

- `tests/test_chol_csc.c`

The landed proof covers:

- direct batched panel-solve correctness on a small dense fixture
- default dense-kernel descriptor completeness, including `solve_panel`
- backend-contract rejection when `solve_panel` is missing

## Explicit Non-Touches

The Day 7 batch did not need follow-through in:

- `include/sparse_cholesky.h`
- `benchmarks/bench_chol_csc.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

That keeps the batch truthful:

- no public header contract widened
- no benchmark/reporting wording widened
- no public-path lifecycle proof moved
- no broader backend-governance or policy spill occurred

## Validation

Because `*.c` and internal `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 334.93 sec`

## Exit State

Day 7 closes with:

- one real dense-kernel batching seam in the shipped builtin descriptor
- one supernodal CSC consumer path aligned to that seam
- one narrow backend-contract failure boundary tied to the actual missing
  required callback
- one bounded family-local proof expansion
