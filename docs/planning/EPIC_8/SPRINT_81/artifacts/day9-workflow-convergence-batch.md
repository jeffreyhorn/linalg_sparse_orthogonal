# Sprint 81 Day 9 - Workflow Convergence Batch

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Land one bounded repeated-run workflow convergence batch so Sprint 81 stops
treating smaller Cholesky and LDL^T repeated-run numeric factoring as a
linked-list-first fallback path while preserving the existing public failure
semantics.

## Main Result

The bounded Day 9 workflow-convergence batch landed in:

- `src/sparse_analysis.c`
- `tests/test_integration.c`
- `benchmarks/bench_refactor_csc.c`

The main implementation result is now explicit:

- repeated-run public Cholesky factoring no longer drops through the
  smaller-problem linked-list `build_permuted_copy(...)` fallback
- repeated-run public LDL^T factoring no longer drops through the same
  smaller-problem linked-list fallback
- the shared repeated-run owner now keeps those lanes on the
  analysis-backed CSC-aware path for all problem sizes

## Day 9 Safeguard

The important Day 9 safeguard was preserved too:

- symmetric direct repeated-run inputs still reject non-symmetric matrices
  before old factors are replaced
- that keeps the public failure-preserves-old-factors reading intact for the
  Cholesky / LDL^T analysis path

## Focused Proof

Focused public proof landed exactly where Day 8 said it should:

- `test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_cholesky`
- `test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_ldlt`
- `./build/quality-review-cmake/test_integration` retained `53 / 53`

The benchmark surface stayed support-only:

- `benchmarks/bench_refactor_csc.c` only needed wording follow-through so the
  comment no longer describes the old linked-list-side cost structure as the
  shared repeated-run path

## Preserved Fence

The Day 8 preserved fence held:

- no LU widening
- no `src/sparse_matrix.c` reopening
- no wrapper-family cleanup in `src/sparse_cholesky.c`, `src/sparse_ldlt.c`,
  or `src/sparse_qr.c`
- no support-surface churn in headers or docs

## Validation

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed
- reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 424.67 sec`

## Exit State

- Sprint 81 has now closed its strongest remaining repeated-run convergence
  contradiction.
- The public matrix-shell and repeated-run direct lanes now read more
  consistently as a bounded compressed-first modernization path.
- The next rerank can now judge whether follow-through pressure shifts to
  proof, benchmark measurability, or residual support-surface drift.
