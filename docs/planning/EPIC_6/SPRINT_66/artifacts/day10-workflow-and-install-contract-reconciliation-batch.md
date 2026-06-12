# Sprint 66 Day 10: Workflow and Install-Contract Reconciliation Batch

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Land the bounded Day 10 follow-through from the Day 9 rerank: reconcile the
shipped static-first package story with the cross-platform workflow and proof
ownership surfaces, without reopening build/install mechanics or widening the
repo's ABI/platform claims.

## Landed Batch

The Day 10 batch landed on:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### README

`README.md` now states the proof ownership split directly:

- the Unix-side local install proof surfaces are:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- macOS CI is only a narrower supplemental Make install/`pkg-config`
  verification lane
- Windows remains the reviewed CMake subset rather than a separate reviewed
  install-validation lane

### INSTALL

`INSTALL.md` now treats the focused regression scripts explicitly as Unix-side
local proof for the maintained static-first contract:

- `tests/test_install.sh` covers Make install/uninstall plus `pkg-config`
- `tests/test_cmake_install.sh` covers CMake install/export plus
  `find_package(Sparse)`

It also states directly that those scripts complement, rather than replace, the
narrower reviewed platform lanes.

### Maintainer guide

`docs/maintainer_guide.md` now owns install/package regression ownership
explicitly:

- local Unix-side Make install/`pkg-config` proof
- local Unix-side CMake install/export proof
- narrower macOS supplemental verification
- Windows reviewed CMake subset and CMake-first consumer story

### Workflow comments and job truth

Workflow commentary now matches the shipped contract more directly:

- Linux remains the strongest reviewed source of truth without implying a
  separate reviewed install-validation lane
- macOS now describes its install job as supplemental static-first package
  verification
- Windows now states directly that its reviewed CMake subset supports the
  CMake-first consumer story but is not a separate reviewed install-validation
  lane

## Validation

Because this was substantial packaging/platform/workflow contract work, the
stronger reviewed baseline was used:

- `make quality-review-full`

Retained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 523.37 sec`

Because the install/package contract wording moved materially, the focused
proof surfaces were also rerun:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

Retained focused proof points:

- Make install/uninstall path passed
- CMake install/export/find-package path passed
- installed `pkg-config` version stayed `2.2.0`

One non-blocking note remains unchanged from the recent reviewed baselines:

- `test_reorder_nd` still dominated the reviewed CMake path at `369.10 sec`
  out of `523.37 sec`, but the full reviewed path completed cleanly and all
  parity anchors stayed exact

## Exit State

Sprint 66 Day 10 closes with:

- one reconciled workflow/install-contract ownership story
- one validated proof chain for the shipped static-first package surface
- one much smaller remaining queue centered on residual tightening and closeout
