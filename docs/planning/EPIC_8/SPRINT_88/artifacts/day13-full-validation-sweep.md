# Sprint 88 Day 13: Full Validation Sweep

## Purpose

Run the complete Sprint 88 validation queue and capture the refreshed
usability-close baseline.

## Validation Queue

The full Day 13 queue passed cleanly:

- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `make bench-canonical-report`

## Reviewed Baseline

The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `408.39 sec`

Non-blocking runtime note:

- reviewed `test_reorder_nd` = `222.30 sec`
- reviewed total = `408.39 sec`

## Focused Follow-Ons

The focused reruns and follow-ons also all passed:

- `example_analysis`
- `example_basic_solve`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `make bench-canonical-report`

Representative measured outputs:

- `example_analysis` solve residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `tests/test_install.sh` = `13` passed, `0` failed
- `tests/test_cmake_install.sh` = `15` passed, `0` failed, `0` skipped

## Canonical Benchmark Report Surface

`make bench-canonical-report` wrote:

- `build/bench-reports/canonical/bench_refactor_csc.csv`
- `build/bench-reports/canonical/bench_chol_csc.csv`
- `build/bench-reports/canonical/bench_iterative_reuse.csv`
- `build/bench-reports/canonical/bench_eigs_reuse.csv`
- `build/bench-reports/canonical/index.tsv`
- `build/bench-reports/canonical/manifest.txt`

## Exit State

- Sprint 88's full validation queue now passes from the live branch state.
- Reviewed, example, package, and canonical benchmark-report anchors are all
  explicit in writing.
- Only non-blocking runtime debt remains going into Sprint 88 closeout.
