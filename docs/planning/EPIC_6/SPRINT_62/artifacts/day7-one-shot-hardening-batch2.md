# Sprint 62 Day 7: One-Shot Hardening Batch II

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Finish the bounded LU one-shot hardening follow-through by making reordered
one-shot LU factorization preserve the caller-owned matrix unless the numeric
factorization actually succeeds.

## Landed Scope

### Touched files

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`

### Untouched by design

- `src/sparse_analysis.c`
- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `tests/test_sparse_lu.c`
- broad docs/examples/benchmark surfaces

## Main Implementation Result

### 1. Reordered LU one-shot calls now factor on a temporary working copy

The Day 7 batch introduces:

- `s62_lu_factor_reordered_working_copy(...)`

inside `src/sparse_lu.c`.

That helper:

- computes the reordered working matrix with `sparse_permute(...)`
- runs numeric LU factorization on that temporary matrix
- publishes the reordered/factored payload back into the caller matrix only
  after success

If factorization fails or is cancelled:

- the temporary matrix is freed
- the reorder permutation is freed
- the original caller matrix remains untouched

### 2. Cancelled reordered LU one-shot attempts no longer strand caller state

This closes the strongest remaining Day 6 seam.

Before Day 7, reordered LU one-shot factorization could mutate the
caller-owned matrix into reordered form before numeric LU had actually
succeeded.

After Day 7:

- reordered LU one-shot failure or cancellation leaves the caller matrix in
  its original row/column layout
- row/column permutation arrays remain identity
- no partially published reordered payload is left behind

### 3. The public LU header now states the reordered preservation rule directly

`include/sparse_lu.h` now documents that reordered one-shot LU calls outside
the default-compatible fast path:

- factor a temporary reordered working copy
- publish back to `mat` only on success

That keeps the header truthful about the strengthened one-shot safety
contract without widening the explicit public lifecycle.

## Regression Proof

### New integration proof

`tests/test_integration.c` now includes:

- `test_progress_cb_lu_cancel_after_reorder_preserves_original_matrix`

It proves that a cancelled reordered LU one-shot attempt:

- returns `SPARSE_ERR_CANCELLED`
- preserves row/column permutation identity
- preserves original tridiagonal matrix entries
- leaves the matrix unfactored for `sparse_lu_solve(...)`
- allows a later successful reordered LU one-shot retry

### Day 7 debugging note

The first cut of the new regression test accidentally retried LU with the same
cancelling options object, so the “success” retry also cancelled.

The landed fix uses a separate retry options object with:

- the same reorder and pivot selections
- no progress callback

The implementation did not need correction for that issue.

## Compatibility and Safety Notes

### Preserved compatibility

- one-shot LU wrappers remain first-class/default entry points
- explicit repeated-run direct solves remain the canonical:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
  lifecycle
- no hidden always-copy behavior was introduced

### Strengthened one-shot rule

The strengthened Day 7 rule is narrower and more truthful:

- fresh-matrix requirements still apply to LU one-shot wrappers
- reordered one-shot attempts now preserve caller state until success
- callers get a safer failure/cancel path without losing the explicit
  lifecycle distinction

## Validation

### Required code-day gate

- `make format`
- `make lint`
- `make test`

All passed.

### Stronger reviewed baseline

- `make quality-review-full`

Passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 360.27 sec`

### Non-blocking note

The reviewed CMake rebuild again emitted ordinary compiler warnings while
rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
cleanly and passed all parity gates.

## Day 7 Exit State

Sprint 62 now has one coherent first LU usability package:

- LU one-shot wrappers reject reused matrix state explicitly
- reordered LU one-shot attempts preserve caller-owned matrix state until
  success
- the public LU header explains that strengthened reordered behavior
  directly
- integration proof now covers both rejected reuse and cancelled reordered
  preservation semantics
- the batch stayed inside the Day 5 touched-file and proof fence
