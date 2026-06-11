# Sprint 63 Day 6: LU Lifecycle Follow-Through Batch 1

Date: 2026-06-10
Branch: sprint-63

## Purpose

Land the first bounded LU lifecycle follow-through slice from the Sprint 63
Day 5 fence, keeping the work local to the LU one-shot wrapper boundary and the
public lifecycle proof surface.

## Landed Files

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`

## Problem Reduced

After Sprint 62, reordered LU publication and caller-matrix preservation were
already in better shape. The strongest remaining LU lifecycle seam was narrower:

- invalid `reorder` values were rejected explicitly
- invalid `pivot` values were not rejected explicitly at the same wrapper
  boundary
- that left invalid pivot handling less deterministic than the surrounding LU
  wrapper contract

For Sprint 63 Day 6, the right question was:

- can invalid LU pivot input be rejected early enough to preserve matrix state
  and keep retry behavior coherent?

## Change

### 1. Explicit invalid-pivot rejection in the LU one-shot path

`src/sparse_lu.c` now validates LU pivot values before the one-shot factor path
can mutate matrix state.

That rejection is applied in both wrapper shapes:

- `sparse_lu_factor_inner(...)`
- `sparse_lu_factor_opts(...)`

Shipped result:

- invalid pivot returns `SPARSE_ERR_BADARG`
- rejection happens before reorder/factor work begins
- valid behavior is unchanged

### 2. Public header truthfulness follow-through

`include/sparse_lu.h` now states the shipped invalid-input contract directly:

- invalid `opts->pivot` is a `SPARSE_ERR_BADARG` case for
  `sparse_lu_factor_opts(...)`
- invalid `pivot` is a `SPARSE_ERR_BADARG` case for `sparse_lu_factor(...)`

This keeps the public truth surface aligned to the live wrapper behavior without
widening into a broad direct-family docs pass.

### 3. Integration proof for preservation and retry

`tests/test_integration.c` now includes:

- `test_lu_invalid_pivot_opts_preserve_original_matrix_and_allow_retry`

That proof checks:

- invalid pivot through `sparse_lu_factor_opts(...)` returns `SPARSE_ERR_BADARG`
- the matrix row/column permutation state stays at identity
- representative matrix contents stay unchanged
- no usable factor is published by the failed call
- a later valid LU factor/solve retry still succeeds

## What Did Not Move

This batch intentionally did not widen into:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_internal.h`
- `src/sparse_matrix_state_internal.h`
- `tests/test_sparse_lu.c`
- `src/sparse_analysis.c`
- Cholesky, LDL^T, or QR files
- broad docs/examples/benchmark work

That preserves the Sprint 63 Day 5 implementation fence.

## Validation

Because `*.c` / `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 299.14 sec`

Non-blocking note:

- the reviewed CMake rebuild again emitted the existing
  `bench_eigs_reuse.c` double-promotion warnings
- the reviewed path still completed cleanly and passed all parity gates

## Exit State

Sprint 63 Day 6 closes one real LU lifecycle gap without widening the sprint:

- invalid LU pivot input is now rejected deterministically
- matrix preservation on that failure path is explicit and tested
- later valid retry behavior remains intact
- the branch is ready for the next bounded CSC / direct-lifecycle follow-through
  slice
