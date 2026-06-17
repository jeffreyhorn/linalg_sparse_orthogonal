# Sprint 75 Day 13: Full Validation Sweep

## Objective

Validate the landed Sprint 75 branch from the strongest reviewed baseline and
the exact Day 12 follow-on queue.

## Validation Gate

All required Day 13 validation passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Reviewed Anchors

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 346.76 sec`

## Focused Sprint 75 Follow-Ons

All Day 12 follow-ons also passed:

- `./build/quality-review-cmake/test_chol_csc` -> `147 / 147`
- `./build/quality-review-cmake/test_integration` -> `50 / 50`
- `make examples-build`
- `make bench-build`
- `./build/example_analysis`
- `./build/example_basic_solve`
- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs

- `example_analysis` residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `test_chol_csc` retained:
  - `test_chol_dense_solve_panel_2x2_two_rhs`
  - `test_supernodal_dense_backend_default_contract`
  - `test_supernode_eliminate_panel_missing_solve_panel_is_backend_contract_error`
- `test_integration` retained:
  - `test_progress_cb_cholesky_csc_emits`
  - `test_progress_cb_cholesky_csc_cancel_before_writeback_preserves_original_matrix`
- `bench_refactor_csc nos4` retained `speedup_refactor = 1.11`, residuals
  `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4` retained:
  - `csc_scalar_path = scalar`
  - `csc_supernodal_path = supernodal`
  - `csc_supernodal_dense_kernel = builtin`
  - `csc_supernodal_panel_solver = batched_panel`
  - `speedup_csc = 0.48`
  - `speedup_csc_sn = 0.81`
  - residuals `7.06e-16`, `5.89e-16`, `5.89e-16`
- both install regressions retained installed `pkg-config` version `2.2.0`

## Non-Blocking Note

Reviewed CMake `test_reorder_nd` still dominated runtime at `234.86 sec` out
of the `346.76 sec` total, but the full reviewed path completed cleanly and
all parity anchors stayed exact.

## Bottom Line

Sprint 75 Day 13 closes with a fully validated second backend/performance
package:

- reviewed baseline still passes
- touched family-local and public runtime proof owners still pass
- maintained benchmark measurement still reports `batched_panel`
- example and install/package follow-ons still pass
