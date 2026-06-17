# Sprint 77 Day 2 - Validation Baseline and Truth-Surface Recheck

Date: 2026-06-17  
Branch: sprint-77

## Purpose
Reconfirm the Sprint 77 implementation-day validation contract and the live proof split across reviewed CMake proof owners, canonical benchmark/report commands, install/package scripts, and workflow-facing packaging/platform commands.

## Main Result
Sprint 77 now has one explicit implementation-day validation contract before any release, install, export, or platform-quality batch lands.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity remains the main truthfulness anchor:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The Sprint 77 authority split is now fixed:
- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial packaging, platform, workflow, or export batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split
Reviewed CMake proof owners currently carry the strongest Sprint 77 executable proof:
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

Canonical benchmark/report workflow remains command and source owned rather than reviewed-binary owned:
- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`

The root `build/` tree is currently carrying the canonical maintained benchmark emitters consumed by that report path:
- `build/bench_refactor_csc`
- `build/bench_chol_csc`
- `build/bench_iterative_reuse`
- `build/bench_eigs_reuse`

Maintained install/package proof remains script-owned:
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Workflow-Facing Command Owners
The strongest package/platform command owners remain explicit:
- `make quality-review-full`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`
- `make install`
- `make uninstall`
- `make bench-canonical-report`

## Interpretation
The useful Day 2 clarification is now explicit:
- the reviewed CMake tree remains the strongest executable truth surface for caller-facing proof-owner tests and representative examples
- the root `build/` tree still matters because the canonical maintained reporting surface consumes those root benchmark emitters directly
- install and exported-package truth still depends on script-owned proof rather than on a reviewed CI lane that fully duplicates local install validation
- Sprint 77 should therefore preserve all three proof layers together rather than treating any single one as the whole package/platform truth

## Exit State
- the code-day validation contract is explicit
- the reviewed CMake parity anchor is current
- the live proof split across reviewed binaries, root canonical emitters, and install/package scripts is fixed in writing
- Sprint 77 can now move into release-surface and platform-gap audit work without rerun ambiguity
