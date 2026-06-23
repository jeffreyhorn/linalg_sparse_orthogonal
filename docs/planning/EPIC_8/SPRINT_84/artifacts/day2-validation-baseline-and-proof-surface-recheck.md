# Sprint 84 Day 2: Validation Baseline and Proof-Surface Recheck

## Purpose

Refresh the implementation-day validation contract and the live proof-owner
split before Sprint 84 widens any external differential, seeded-property, or
failure-path assurance seam.

## Reviewed Validation Contract

Sprint 84 continues to inherit the strongest local reviewed baseline:

- `make quality-review-full`

The code-day and docs-day split is now fixed explicitly:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial differential, property, failure-path, or support-policy
  batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

Reviewed CMake parity remains the primary truthfulness anchor:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Reviewed Proof Owners

The reviewed CMake tree currently owns the strongest early-Sprint-84 proof
surfaces:

- public/shared, direct-family, and workflow-sensitive tests:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_integration`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- reviewed benchmark follow-on binaries:
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`

## Canonical Reporting and Install Ownership

Canonical benchmark reporting remains command- and script-owned rather than
reviewed-binary-owned:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- root `build/` canonical emitters:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

Maintained install/package proof remains script-owned:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Day 2 Result

Sprint 84 now has one explicit validation and proof-owner contract before the
assurance rerank begins:

- reviewed CMake proof-owner tests and representative examples remain the main
  executable truth surfaces
- reviewed benchmark binaries remain benchmark-side measurability surfaces
- canonical benchmark reporting remains command/script owned
- install/export proof remains script owned

The highest-signal rerun set is now fixed around:

- `./build/quality-review-cmake/test_sparse_matrix`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_svd`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
