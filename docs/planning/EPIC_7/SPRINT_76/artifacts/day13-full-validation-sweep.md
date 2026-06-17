# Sprint 76 Day 13 Artifact: Full Validation Sweep

Date: 2026-06-17
Branch: sprint-76

## Purpose

Validate the full Sprint 76 benchmark-governance package from the landed
Day 12 state and retain one explicit reviewed proof, report, and install
baseline before Sprint 76 closeout.

## Main Result

Day 13 validation was complete.

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

## Reviewed Anchors

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 346.44 sec`

## Focused Follow-On Results

The Day 12 focused follow-on queue also all passed:

- `./build/quality-review-cmake/test_chol_csc` -> `147 / 147`
- `./build/quality-review-cmake/test_integration` -> `50 / 50`
- `./build/quality-review-cmake/test_eigs` -> `31 / 31`
- `./build/quality-review-cmake/test_qr` -> `72 / 72`
- `./build/quality-review-cmake/test_svd` -> `97 / 97`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/quality-review-cmake/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/quality-review-cmake/bench_iterative_reuse`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `make bench-canonical-report`
- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs

- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`
- `bench_refactor_csc nos4` retained `speedup_refactor = 1.37`, residuals
  `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4` retained
  `csc_supernodal_panel_solver = batched_panel`, with residuals
  `7.06e-16`, `5.89e-16`, `5.89e-16`
- `bench_iterative_reuse` retained:
  - `cg 1.01x`
  - `gmres 1.01x`
  - `minres 0.99x`
- `bench_eigs_reuse` retained:
  - `growm 1.01x`
  - `thick_restart 1.01x`
  - `lobpcg 1.01x`
  - `lambda_max_diff = 0.000e+00`
- `make bench-canonical-report` retained the stronger canonical bundle:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`
- both install regressions retained installed `pkg-config` version `2.2.0`

## Non-Blocking Note

Reviewed CMake `test_reorder_nd` still dominated runtime at `244.00 sec` out
of the `346.44 sec` total, but the full reviewed path completed cleanly and
all parity anchors stayed exact.

## Exit State

Sprint 76 now has one explicit validated proof, benchmark, report, and
install baseline, and Day 14 can close from that state without reopening the
validation queue.
