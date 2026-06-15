# Sprint 69 Day 12: Full Validation Sweep

Date: 2026-06-15
Branch: `sprint-69`

## Purpose

Run the final maintained quality gates, reconfirm the reviewed truthfulness
anchors, and freeze the final Sprint 69 validated baseline from the integrated
Epic 6 branch state.

## Maintained Gates

The full maintained validation stack passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Reviewed Truthfulness Anchors

The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 797.77 sec`

## Targeted Follow-Ons

All targeted Sprint 69 follow-ons passed:

- `./build/test_integration` -> `47 / 47`
- `./build/test_chol_csc` -> `145 / 145`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_reorder_nd` -> `34 / 34`
- `./build/test_fuzz` -> `25 / 25`
- `./build/test_framework_optin` -> `8` run, `3` skipped, `0` failed
- `./build/test_iterative` -> `79 / 79`
- `./build/test_eigs` -> `30 / 30`
- `./build/example_analysis`
- `./build/example_basic_solve`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Representative Retained Outputs

- `example_analysis` residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `test_fuzz`: `large-n CSC lifecycle property: 3/3 passed`
- `test_reorder_nd`:
  - `Pres_Poisson ND/AMD = 0.923`
  - `bcsstk14 ND/AMD = 1.124`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.73`
  - residuals `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4`:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
  - `speedup_csc = 1.04x`
  - `speedup_csc_sn = 1.07x`
- `bench_iterative_reuse`:
  - `cg 0.96x`
  - `gmres 1.01x`
  - `minres 0.77x`
- `bench_eigs_reuse`:
  - `growm 1.00x`
  - `thick_restart 0.92x`
  - `lobpcg 0.98x`
  - `lambda_max_diff = 0.000e+00`
- install/package regressions:
  - both reported installed `pkg-config` version `2.2.0`
- canonical report regenerated:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`
  - `build/bench-reports/canonical/bench_chol_csc.csv`
  - `build/bench-reports/canonical/bench_iterative_reuse.csv`
  - `build/bench-reports/canonical/bench_eigs_reuse.csv`
  - `build/bench-reports/canonical/manifest.txt`

## Non-Blocking Note

The reviewed CMake path was still dominated by `test_reorder_nd`:

- `test_reorder_nd = 525.85 sec`
- total reviewed CMake time = `797.77 sec`

This remains a runtime concentration point, but it did not affect validation
success, reviewed parity, or final truthfulness.

## Exit State

Sprint 69 now closes from one measured final validation baseline:

- all maintained gates passed
- all targeted final proof surfaces passed
- reviewed truthfulness anchors stayed exact
