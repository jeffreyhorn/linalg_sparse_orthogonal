# Sprint 77 Day 12 Artifact: Regression Coverage and Proof Alignment

Date: 2026-06-17
Branch: sprint-77

## Purpose

Confirm that the touched Sprint 77 release/install/platform proofs already sit
in the right owners after the Day 6 and Day 9 landings, and fix the exact Day
13 validation queue in writing.

## Main Result

No new focused regression code is actually needed.

## Current Proof Owners

The touched Sprint 77 proof already sits in the right owners:

- `tests/test_install.sh`
  - local Make install/uninstall plus `pkg-config` proof
- `tests/test_cmake_install.sh`
  - local CMake install/export plus `find_package(Sparse)` proof
- reviewed CMake parity and representative executable proof:
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - representative reviewed tests and examples in `build/quality-review-cmake`
- `.github/workflows/macos-ci.yml`
  - explicit macOS supplemental install-confidence reading
- `.github/workflows/windows-ci.yml`
  - explicit Windows reviewed CMake consumer-scope reading

## Why No New Regression Code Was Needed

The Day 6 and Day 9 landings changed:

- operator-facing install wording
- workflow-level proof reading

They did not change:

- install mechanics
- export mechanics
- local install-proof ownership
- reviewed CMake executable proof ownership

That means the real remaining gap is not missing test code. It is simply making
the final validation queue explicit from the landed state.

## Day 13 Validation Queue

The exact Day 13 validation queue is now fixed around:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- representative reviewed executable proof:
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`

## Exit State

Sprint 77 now has one explicit Day 12 proof-alignment outcome:

- no new regression code required
- proof owners already match the landed contract
- Day 13 validation queue fixed explicitly before the full sweep
