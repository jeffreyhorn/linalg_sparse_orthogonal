# Sprint 78 Day 13 - Full Validation Sweep

Date: 2026-06-18  
Branch: sprint-78

## Purpose
Validate the full Sprint 78 maintainability package from the aligned Day 12 state and retain the highest-signal outputs from the touched source and proof owners.

## Main Result
Day 13 validation completed cleanly:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 310.71 sec`

## Focused Follow-Ons
The Day 12 explicit follow-on queue also passed:

- `./build/quality-review-cmake/test_ldlt_csc` -> `96 / 96`
- `./build/quality-review-cmake/test_chol_csc` -> `147 / 147`
- `./build/quality-review-cmake/test_ldlt` -> `84 / 84`
- `./build/quality-review-cmake/test_integration` -> `50 / 50`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

## Representative Retained Outputs
Representative retained outputs from the touched Sprint 78 owners:

- `test_ldlt_csc`
  - `tridiag indefinite n=10: rel_res = 0.000e+00`
  - `arrow 6x6 indefinite (AMD): rel_res = 9.869e-17`
- `test_chol_csc`
  - `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
  - retained `test_writeback_publishes_solve_ready_factored_shell`
- `test_ldlt`
  - `KKT 500x500: relres=4.465e-17, nnz(L)=1298`
  - retained `test_ldlt_backend_csc_forced_factors`
- `test_integration`
  - retained `test_progress_cb_cholesky_csc_cancel_before_writeback_preserves_original_matrix`
  - retained `test_public_lifecycle_cholesky_csc_refactor_preserves_old_factors_on_failure`
  - retained `test_public_lifecycle_ldlt_refactor_rejects_nnz_drift_and_preserves_old_factors_amd`
- `example_analysis`
  - solve residual `4.44e-16`
  - repeated-run refactor average `0.000494 s`
- `example_basic_solve`
  - residual `0.00e+00`

## Runtime Note
The non-blocking runtime concentration stayed where recent reviewed baselines already put it:

- reviewed CMake `test_reorder_nd` consumed `218.14 sec`
- total reviewed CMake time was `310.71 sec`

That remains a retained runtime note, not a Sprint 78 validation blocker.

## Exit State
- Sprint 78 now has one explicit validated close baseline.
- The touched source and proof owners all reconfirmed cleanly.
- Day 14 can close the sprint from a fresh reviewed state instead of from inference.
