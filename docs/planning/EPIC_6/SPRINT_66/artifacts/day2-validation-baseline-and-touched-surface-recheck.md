# Sprint 66 Day 2: Validation Baseline and Touched-Surface Recheck

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Reconfirm the reviewed baseline and the targeted rerun set that Sprint 66
packaging, ABI, install, workflow, and platform-quality changes must preserve
before any implementation work lands.

## Reviewed Validation Contract

The strongest local reviewed baseline remains:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The authoritative Sprint 66 split is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial packaging, install/export, workflow, or
  platform-quality work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

## Targeted Sprint 66 Rerun Set

The high-signal Sprint 66 rerun set is:

- direct lifecycle and CSC proof surfaces:
  - `./build/test_integration`
  - `./build/test_sparse_lu`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- adjacent numerical sentinels:
  - `./build/test_qr`
  - `./build/test_svd`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_svd_lowrank`
- canonical maintained benchmark surfaces:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_ldlt_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

All of those surfaces were present in the current `build/` tree at Day 2.

## Touched-Surface Recheck

The highest-signal likely Sprint 66 touch surfaces at Day 2 are:

- packaging/install/build:
  - `CMakeLists.txt`
  - `Makefile`
  - `INSTALL.md`
- workflow/platform truth surfaces:
  - `.github/workflows/ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `README.md`
  - `docs/maintainer_guide.md`
- likely narrow header truth surfaces only if the audit proves they need moving:
  - `include/sparse_types.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

Measured hotspot sizes at Sprint 66 start:

- `README.md` = `1000`
- `INSTALL.md` = `206`
- `docs/maintainer_guide.md` = `511`
- `CMakeLists.txt` = `397`
- `Makefile` = `897`
- `.github/workflows/ci.yml` = `221`
- `.github/workflows/windows-ci.yml` = `57`
- `.github/workflows/macos-ci.yml` = `111`
- `include/sparse_types.h` = `233`
- `include/sparse_cholesky.h` = `232`
- `include/sparse_ldlt.h` = `334`

## Exit State

Sprint 66 Day 2 closes with:

- the same reviewed truthfulness baseline as the Sprint 65 close
- one explicit validation split for docs-only versus bounded code-touching
  versus substantial packaging/platform work
- one fixed rerun set centered on the actual productization-sensitive proof
  surface
- one clear starting point for the Day 3 packaging and ABI audit
