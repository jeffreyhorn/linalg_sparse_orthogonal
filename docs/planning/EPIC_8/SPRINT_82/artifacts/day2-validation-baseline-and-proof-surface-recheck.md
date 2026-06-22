# Sprint 82 Day 2: Validation Baseline and Proof-Surface Recheck

## Purpose

Refresh the strongest local validation contract and the live proof split across
reviewed tests/examples, backend-facing benchmark surfaces, canonical report
commands, and install/export proof owners before any Sprint 82 backend code
lands.

## Strongest Validation Baseline

Sprint 82 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

Reviewed CMake parity remains the main truthfulness anchor:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The implementation-day authority split stays fixed:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial backend, solver-adoption, or package/runtime batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split

The reviewed CMake tree currently owns the strongest early-Sprint-82 proof
surfaces:

- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_svd`

The canonical report-generation workflow remains command- and script-owned:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- root `build/` canonical emitters consumed by that script:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

Maintained install/package proof remains script-owned:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Highest-Signal Sprint 82 Rerun Set

The strongest likely Sprint 82 rerun queue is now fixed around:

- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_svd`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Day 2 Result

Sprint 82 now has one explicit implementation-day validation contract and one
explicit proof-owner split before backend work begins:

- reviewed tests/examples own executable regression truth
- reviewed benchmark binaries own backend-side measurability
- canonical reporting remains command/script owned
- install/export proof remains script owned
