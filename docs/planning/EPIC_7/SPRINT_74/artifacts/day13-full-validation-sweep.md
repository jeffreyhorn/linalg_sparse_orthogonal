# Sprint 74 Day 13: Full Validation Sweep

## Objective

Validate the landed Sprint 74 branch from the strongest reviewed baseline and
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
- `Total Test time (real) = 372.56 sec`

## Focused Sprint 74 Follow-Ons

All Day 12 follow-ons also passed:

- `./build/quality-review-cmake/test_sparse_matrix` -> `57 / 57`
- `./build/quality-review-cmake/test_iterative` -> `80 / 80`
- `./build/quality-review-cmake/test_eigs` -> `31 / 31`
- `./build/quality-review-cmake/test_integration` -> `48 / 48`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/quality-review-cmake/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/quality-review-cmake/bench_iterative_reuse`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs

- `example_analysis` residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `test_sparse_matrix` retained `test_idx_width_contract`
- `test_iterative` retained `test_iterative_public_scalar_alias`
- `test_eigs` retained `test_eigs_public_scalar_alias`
- `bench_refactor_csc nos4` retained `speedup_refactor = 1.41`, residuals
  `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4` retained `scalar`, `supernodal`, `builtin`, with
  `speedup_csc = 0.65`, `speedup_csc_sn = 0.69`, and residuals `7.06e-16`,
  `5.89e-16`, `5.89e-16`
- `bench_iterative_reuse` retained `cg 1.01x`, `gmres 1.01x`, `minres 1.01x`
- `bench_eigs_reuse` retained `growm 1.01x`, `thick_restart 1.00x`,
  `lobpcg 1.01x`, `lambda_max_diff = 0.000e+00`
- both install regressions retained installed `pkg-config` version `2.2.0`

## Non-Blocking Note

Reviewed CMake `test_reorder_nd` still dominated runtime at `259.98 sec` out
of the `372.56 sec` total, but the full reviewed path completed cleanly and
all parity anchors stayed exact.

## Bottom Line

Sprint 74 Day 13 closes with a fully validated first-phase capability package:

- reviewed baseline still passes
- touched width/scalar proof owners still pass
- adoption, benchmark, and install/package follow-ons still pass
