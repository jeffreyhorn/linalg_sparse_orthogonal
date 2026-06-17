# Sprint 75 Day 10 Artifact: Callback / Runtime Follow-Through Batch

Date: 2026-06-17
Branch: sprint-75

## Purpose

Land the bounded CSC Cholesky callback/runtime follow-through designed on Day
9 while preserving the truthful difference between linked-list per-column
progress and CSC orchestration-level progress.

## Main Result

Sprint 75 Day 10 landed one bounded public runtime batch across:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`

The main result is:

- linked-list callback semantics stayed unchanged
- the CSC lane now emits one explicit public wrapper-owned phase
  `cholesky_factor_csc`
- the CSC phase uses `4` bounded orchestration checkpoints instead of fake
  per-column parity
- CSC cancellation before publish-back now has public-path proof that the
  caller-owned matrix shell stays in the original coordinate space

## Landed Runtime Contract

### Linked-list lane

- still emits `phase = "cholesky_factor"`
- still reports per-column elimination progress
- still uses the existing linked-list cancellation contract

### CSC lane

- now emits `phase = "cholesky_factor_csc"`
- uses `total = 4`
- reports bounded orchestration checkpoints at:
  - analysis entry
  - CSC conversion/materialization entry
  - supernodal factorization entry
  - pre-writeback publish entry

This is intentionally not a claim of:

- per-column CSC callback parity
- mid-supernode rollback
- broader repo-wide backend runtime uniformity

## Cancellation Truth

The CSC cancellation fence is now explicit:

- cancellation can happen through the wrapper-owned checkpoint path
- cancellation before the final pre-writeback checkpoint leaves the caller
  matrix in the original coordinate space
- no solve-ready factored shell is published on that cancellation path
- `SPARSE_ERR_CANCELLED` remains distinct from
  `SPARSE_ERR_BACKEND_CONTRACT`

## Proof

The public-path proof owner is:

- `tests/test_integration.c`

The landed proof covers:

- `test_progress_cb_cholesky_csc_emits`
- `test_progress_cb_cholesky_csc_cancel_before_writeback_preserves_original_matrix`

Those tests prove:

- the CSC wrapper emits the expected `4` checkpoints
- the emitted public phase is `cholesky_factor_csc`
- cancellation at the pre-writeback checkpoint preserves the original matrix
  shell
- retry succeeds and the matrix can still become solve-ready afterward

## Explicit Non-Touches

The Day 10 batch did not need follow-through in:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`
- `benchmarks/bench_chol_csc.c`
- `docs/maintainer_guide.md`

That keeps the batch bounded and truthful:

- no family-local kernel contract had to move
- no benchmark/report wording was required for correctness
- no maintainer-policy widening was needed

## Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 335.03 sec`

## Exit State

Day 10 closes with:

- one truthful public CSC runtime phase
- one explicit CSC cancel-before-writeback safety fence
- one public-path proof expansion in `tests/test_integration.c`
- one validated reviewed close without widening into support-only surfaces
