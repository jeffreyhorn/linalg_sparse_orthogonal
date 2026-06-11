# Sprint 63 Day 7: Cholesky CSC Lifecycle Follow-Through Batch 1

Date: 2026-06-10
Branch: sprint-63

## Purpose

Land the first bounded Cholesky CSC repeated-run uniformity slice from the
Sprint 63 Day 5 fence, keeping the work local to CSC dispatch coherence,
header truthfulness, and the highest-signal direct proof homes.

## Landed Files

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`

## Problem Reduced

After the Sprint 62 Cholesky preservation work, the strongest remaining CSC
follow-through seam was narrower than a broad “direct family asymmetry” label
suggested:

- invalid Cholesky backend enum values were not rejected explicitly
- `used_csc_path` was only published after later failure points
- that made Cholesky CSC dispatch/result semantics looser than the adjacent
  LDLT CSC-backed lane

For Sprint 63 Day 7, the right question was:

- can the Cholesky CSC dispatch boundary reject invalid backend input earlier
  and publish CSC-path telemetry sooner without widening into a larger
  lifecycle redesign?

## Change

### 1. Early backend validation and single dispatch selection

`src/sparse_cholesky.c` now resolves the Cholesky backend once at the wrapper
boundary through a small local dispatch helper.

Shipped result:

- invalid backend returns `SPARSE_ERR_BADARG`
- valid backend selection happens once
- the selected `use_csc` decision is threaded into both:
  - the no-reorder path
  - the reordered working-copy path

This removes late duplicate dispatch logic from the no-reorder path and makes
the CSC selection semantics tighter.

### 2. Earlier `used_csc_path` publication

`used_csc_path` is now written immediately after backend selection, before
later reorder or factor failures can return.

That gives Cholesky CSC telemetry the same stronger shape already present on
the adjacent LDLT CSC path:

- dispatch decision first
- publish telemetry
- then proceed into later reorder/factor work

### 3. Public header truthfulness follow-through

`include/sparse_cholesky.h` was updated only enough to keep the public truth
surface exact.

`sparse_cholesky_factor_opts(...)` now explicitly documents:

- invalid `opts->backend` as a `SPARSE_ERR_BADARG` case
- the surrounding invalid reorder/state cases in the same contract block

This keeps the header aligned to the shipped wrapper behavior without widening
into a larger direct-family docs pass.

### 4. Proof expansion

`tests/test_integration.c` now includes:

- `test_cholesky_invalid_backend_preserves_original_matrix_and_allows_retry`

That proof checks:

- invalid backend through `sparse_cholesky_factor_opts(...)` returns
  `SPARSE_ERR_BADARG`
- the caller matrix remains in original identity-permutation state
- representative matrix entries stay unchanged
- no usable factor is published by the failed call
- a later valid reordered CSC retry still succeeds

`tests/test_chol_csc.c` now includes:

- `test_dispatch_invalid_backend_rejected`
- `test_dispatch_csc_reports_selected_path_before_reorder_error`

Those prove:

- invalid backend is rejected explicitly
- `used_csc_path` is reported as `1` on the selected CSC path even when a
  later invalid reorder argument fails

## What Did Not Move

This batch intentionally did not widen into:

- `src/sparse_analysis.c`
- `include/sparse_analysis.h`
- LDLT files
- QR files
- broad docs/examples/benchmark work
- cancellation-model redesign

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
- `Total Test time (real) = 311.10 sec`

Non-blocking note:

- the reviewed CMake rebuild again emitted the existing
  `bench_eigs_reuse.c` double-promotion warnings
- the reviewed path still completed cleanly and passed all parity gates

## Exit State

Sprint 63 Day 7 closes one real Cholesky CSC lifecycle gap without widening
the sprint:

- invalid backend input is now rejected deterministically
- CSC-path telemetry is published earlier and more uniformly
- invalid-backend failure preserves the caller matrix and allows a later valid
  retry
- the branch is ready for the next bounded follow-through audit and residual
  re-rank
