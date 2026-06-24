# Sprint 88 Day 2: Validation Baseline and Maintained Support-Surface Recheck

## Purpose

Refresh the implementation-day validation contract and the live maintained
install/export, example, workflow, benchmark-reporting, and reviewed-surface
split before Sprint 88 changes any front-door, support, or public-narrative
surface.

## Reviewed Validation Contract

Sprint 88 continues to inherit the strongest local reviewed baseline:

- `make quality-review-full`

The code-day and docs-day split is now fixed explicitly:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial front-door, support-surface, or public-header narrative
  batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

Reviewed CMake parity remains the primary truthfulness anchor:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Reviewed Executable Truth Owners

The reviewed CMake tree currently remains the strongest shared executable
truth surface entering Sprint 88:

- reviewed representative proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`

## Canonical Reporting, Install/Export, Example, and Support Ownership

Canonical benchmark reporting remains command- and script-owned rather than
front-door-owned:

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

## Workflow-Side Support and Platform Truth

Workflow evidence remains intentionally narrower than a broad cross-platform
adoption or install/export parity claim:

- Linux remains the strongest reviewed source of truth through the maintained
  reviewed paths
- macOS carries a supplemental static-first Make install/`pkg-config`
  confidence lane only
- Windows remains the reviewed CMake-first consumer subset and does not claim
  a separate reviewed install-validation lane

## Day 2 Result

Sprint 88 now has one explicit validation and maintained-support-surface
contract before the user-journey audit begins:

- reviewed CMake binaries remain the main executable truth anchor
- canonical benchmark reporting remains command/script owned
- install/export proof remains script owned
- downstream example proof remains local and bounded
- workflow lanes remain support evidence rather than broad adoption or package
  parity claims

The highest-signal rerun set is now fixed around:

- `./build/quality-review-cmake/test_reorder_nd`
- `./build/quality-review-cmake/test_reorder`
- `./build/quality-review-cmake/test_reorder_amd_qg`
- `./build/quality-review-cmake/test_graph`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
