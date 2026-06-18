# Sprint 78 Day 10 - Giant-Test Architecture Batch

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Land one bounded proof-architecture cleanup inside the strongest remaining giant-test surface without widening into shared harness or multi-family test churn.

## Main Result
Sprint 78 Day 10 landed exactly inside the Day 9 fence:

- touched center:
  - `tests/test_chol_csc.c`
- untouched support seam:
  - `tests/test_chol_csc_supernodal_helpers.h`
- untouched support-only proof and policy surfaces:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`
  - `docs/maintainer_guide.md`

The batch did not change test logic. It changed the giant-test architecture of the densest Cholesky CSC proof cluster.

## Landed Architecture Change
`tests/test_chol_csc.c` now groups the strongest supernodal/writeback/dispatch registration wall behind explicit family-local runner functions:

- `run_supernode_detection_tests()`
- `run_supernodal_postorder_tests()`
- `run_supernodal_dense_tests()`
- `run_supernode_extract_writeback_tests()`
- `run_supernode_diag_factor_tests()`
- `run_supernode_panel_tests()`
- `run_supernodal_parametrised_tests()`
- `run_writeback_tests()`
- `run_dispatch_tests()`

This keeps the same proof-owner file and the same execution order, but removes the most review-hostile part of the permanent `main()` registration block.

## Why This Was the Right Batch
The strongest bounded payoff was not another helper-header extraction and not a shared test-framework rewrite.

It was reducing the densest mixed proof-role cluster inside `tests/test_chol_csc.c` itself:

- supernodal detection and postorder proof
- dense and backend-contract proof
- panel solve and parametrised cross-check proof
- writeback and dispatch proof

Those roles are still owned by the same family-local giant test, but they now read as explicit local sections instead of one uninterrupted chronology wall.

## Preserved Fence
The batch preserved:

- current behavior coverage
- current proof ownership
- current family-local helper boundary
- current validation contract for touched code/test batches

The batch explicitly did not widen into:

- shared harness redesign
- unrelated giant tests
- public API or header edits
- workflow, platform, benchmark, or packaging follow-through

## Validation
Passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 330.90 sec`

## Exit State
- `tests/test_chol_csc.c` remains the Cholesky CSC family-local proof owner.
- The strongest permanent giant-test review seam is now materially cleaner.
- Sprint 78 can rerank the next remaining giant-test architecture or chronology seam from this landed state.
