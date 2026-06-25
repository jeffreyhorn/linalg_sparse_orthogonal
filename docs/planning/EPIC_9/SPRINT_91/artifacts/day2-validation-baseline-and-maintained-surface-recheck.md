# Sprint 91 Day 2: Validation Baseline and Maintained Surface Recheck

## Purpose

Refresh the implementation-day validation contract and the live maintained
reviewed, install/export, reporting, example, and workflow truth split before
Sprint 91 begins compressed-first implementation work on the direct workflow
surface.

## Reviewed Validation Contract

Sprint 91 continues to inherit the strongest local reviewed baseline:

- `make quality-review-full`

The code-day and docs-day split is now fixed explicitly:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial product-contract, proof-owner, or support-surface batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

Reviewed CMake parity remains the primary truthfulness anchor:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Reviewed Executable Truth Owners

The reviewed CMake tree currently remains the strongest shared executable truth
surface entering Sprint 91’s direct-product lane:

- reviewed representative proof owners:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`

## Canonical Reporting, Install/Export, and Consumer Ownership

Canonical benchmark reporting remains command- and script-owned rather than
reviewed-binary-owned:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- root `build/` canonical emitters:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

Maintained install/export proof remains script- and fixture-owned:

- `bash tests/test_install.sh` proves the local Unix-side Make
  install/uninstall + `pkg-config` path
- `bash tests/test_cmake_install.sh` proves the local Unix-side CMake
  install/export + `find_package(Sparse)` path
- `examples/cmake_example/CMakeLists.txt` remains the representative
  downstream CMake consumer surface used by the CMake install/export proof

## Workflow-Side Reviewed and Support Truth

Workflow evidence remains intentionally layered rather than flattened into one
broad parity claim:

- Linux remains the strongest reviewed source of truth through the enforced
  reviewed Makefile compile-quality, reviewed CMake parity, and dead-code
  lanes
- macOS carries a narrower reviewed Apple Clang lane plus a supplemental
  static-first Make install/`pkg-config` confidence lane
- Windows remains a reviewed CMake-first consumer subset and does not claim a
  reviewed Makefile or separate reviewed install-validation lane

## Day 2 Result

Sprint 91 now has one explicit validation and maintained-surface contract
before compressed-first implementation begins:

- reviewed direct-workflow binaries remain the main executable truth anchor
- canonical benchmark reporting remains command/script owned
- install/export proof remains script owned
- downstream consumer proof remains local and bounded
- workflow lanes remain layered support evidence rather than broad
  cross-platform parity claims

The highest-signal rerun set is now fixed around:

- `./build/quality-review-cmake/test_sparse_matrix`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt_csc`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
