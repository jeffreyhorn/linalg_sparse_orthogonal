# Sprint 77 Day 13 Artifact: Full Validation Sweep

Date: 2026-06-17
Branch: sprint-77

## Purpose

Validate the full Sprint 77 release/install/platform package from the landed
Day 12 state and retain one explicit reviewed proof, install, and workflow
baseline before Sprint 77 closeout.

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
- `Total Test time (real) = 384.11 sec`

## Focused Follow-On Results

The Day 12 focused follow-on queue also all passed:

- `./build/quality-review-cmake/test_integration` -> `50 / 50`
- `./build/quality-review-cmake/test_chol_csc` -> `147 / 147`
- `./build/quality-review-cmake/test_qr` -> `72 / 72`
- `./build/quality-review-cmake/test_svd` -> `97 / 97`
- `./build/quality-review-cmake/test_eigs` -> `31 / 31`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs

- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`
- `test_chol_csc` retained the Day 7 backend seam and Day 10 public-path
  contract proof
- both install regressions retained installed `pkg-config` version `2.2.0`

## Non-Blocking Note

Reviewed CMake `test_reorder_nd` still dominated runtime at `246.25 sec` out
of the `384.11 sec` total, but the full reviewed path completed cleanly and
all parity anchors stayed exact.

## Exit State

Sprint 77 now has one explicit validated proof, install, and reviewed-platform
baseline, and Day 14 can close from that state without reopening the
validation queue.
