# Sprint 77 Day 1 - Scope and Packaging/Platform Baseline

Date: 2026-06-17  
Branch: sprint-77

## Purpose
Establish a precise Sprint 77 starting baseline for packaging, ABI, install, and cross-platform quality convergence using the live tree rather than broad Epic 7 review language.

## Main Result
Sprint 77 now starts from a precise packaging, install, and cross-platform quality queue, not from another planning reset and not from another benchmark or capability landing.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity was re-materialized live and remains explicit:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The highest-value Sprint 77 pressure is now clearly narrowed to:
- release-surface re-audit
- platform gap rerank
- packaging and productization follow-through
- Windows and macOS proof follow-through
- workflow and contract reconciliation
- install, package, and platform regression coverage
- validation and closeout

## Preserved Fence
Sprint 77 must preserve the current packaging and platform truth:
- Linux remains the strongest reviewed source of truth across reviewed Makefile quality, reviewed CMake parity, and dead-code ownership.
- macOS keeps the reviewed Apple Clang lane plus narrower supplemental static-first Make install and `pkg-config` verification.
- Windows remains the reviewed CMake subset and CMake-first consumer story rather than a separate reviewed install-validation lane.
- `tests/test_install.sh` and `tests/test_cmake_install.sh` remain the local Unix-side install/package proof owners.
- Installed package truth must stay coherent across:
  - installed headers
  - static library export
  - `pkg-config`
  - exported CMake package files
  - install and uninstall workflows
- Sprint 77 must not widen current evidence into a fake shared-library, Windows Makefile, or broader reviewed install-validation claim.

## Strongest Likely Sprint 77 Touch Surfaces
Maintained packaging, install, and product-contract surfaces:
- `README.md` = `1045`
- `INSTALL.md` = `237`
- `docs/maintainer_guide.md` = `685`
- `Makefile` = `899`
- `CMakeLists.txt` = `413`
- `cmake/SparseConfig.cmake.in` = `5`

Install/package proof owners:
- `tests/test_install.sh` = `172`
- `tests/test_cmake_install.sh` = `146`

Reviewed CI and platform-truth surfaces:
- `.github/workflows/ci.yml` = `223`
- `.github/workflows/macos-ci.yml` = `113`
- `.github/workflows/windows-ci.yml` = `61`

## High-Signal Live Reading
The live tree already fixes several important Sprint 77 truths:
- `README.md` carries the compact caller-facing platform and install summary.
- `INSTALL.md` is the strongest operator-facing install and packaging contract surface.
- `docs/maintainer_guide.md` is the deepest policy authority for reviewed platform scope, install proof ownership, and product truthfulness.
- `Makefile` still defines the strongest local reviewed wrapper path and the main static-first install and quality-review surface.
- `CMakeLists.txt` plus `cmake/SparseConfig.cmake.in` remain the strongest CMake-first consumer and exported-package truth surfaces.
- `tests/test_install.sh` owns local Make install, uninstall, and `pkg-config` proof.
- `tests/test_cmake_install.sh` owns local CMake install, exported package, and downstream consumer proof.
- `.github/workflows/ci.yml` remains the strongest reviewed Linux source of truth.
- `.github/workflows/macos-ci.yml` preserves the narrower reviewed Apple Clang lane plus supplemental install verification.
- `.github/workflows/windows-ci.yml` preserves the reviewed CMake subset and CMake-first consumer story without claiming broader reviewed install parity.

## Interpretation
The strongest Day 1 narrowing is now explicit:
- Sprint 77 is not primarily about new build-system features.
- It is primarily about clarifying and tightening the release, install, export, and reviewed platform contract that already exists in bounded form.
- The highest-value early work is therefore packaging and cross-platform truthfulness convergence, not broad ABI expansion or new consumer surfaces.

## Exit State
- The maintained reviewed baseline is rechecked.
- The reviewed CMake parity anchor is explicit and current.
- The live packaging/platform owners, proof surfaces, and hotspot map are fixed in writing.
- Sprint 77 can now move into a ranked release-surface and platform-gap audit from a precise Day 1 baseline.
