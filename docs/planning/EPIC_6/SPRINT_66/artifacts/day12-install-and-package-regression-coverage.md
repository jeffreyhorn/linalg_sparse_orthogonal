# Sprint 66 Day 12: Install and Package Regression Coverage

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Close the exact Day 11 proof gap by tightening the Unix-side Make install
regression so it uses the same version-source-of-truth contract as the focused
CMake install regression, without widening Sprint 66 into broader assurance
expansion.

## Landed Proof Tightening

The Day 12 batch landed on:

- `tests/test_install.sh`

`tests/test_install.sh` now:

- reads the expected installed package version from the repo `VERSION` file
- compares `pkg-config --modversion sparse` against that exact expected value
- no longer treats a merely non-empty installed version string as sufficient

That brings the Unix-side Make install proof into line with the already-tight
version check in `tests/test_cmake_install.sh`.

## Validation

Focused proof surfaces rerun:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

Retained proof points:

- Make install/uninstall path passed
- CMake install/export/find-package path passed
- both scripts reported installed `pkg-config` version `2.2.0`

This was a focused proof-surface batch only, so no broader reviewed baseline
rerun was required on Day 12.

## Exit State

Sprint 66 Day 12 closes with:

- one closed Unix-side Make install proof gap
- one uniform version-source-of-truth rule across the two maintained local
  install/package regressions
- one explicit Day 13 rerun set for the touched productization surface
