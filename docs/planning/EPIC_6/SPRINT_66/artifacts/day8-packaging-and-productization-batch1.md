# Sprint 66 Day 8: Packaging and Productization Batch 1

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Land the first bounded Sprint 66 packaging/productization slice: make the
maintained static-first install/export contract explicit across build and docs
surfaces, and align the focused CMake install regression with the repo's single
`VERSION` source of truth.

## Landed Batch

The Day 8 batch landed on:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `tests/test_cmake_install.sh`

### Build/install truth

`CMakeLists.txt` now states the maintained release shape more directly:

- `BUILD_SHARED_LIBS=ON` emits an explicit status note that the shipped package
  surface still remains the static archive output
- the maintained target now has explicit export/output naming on the package
  surface

### User-facing install contract

`INSTALL.md` now makes the current package shape explicit:

- the maintained install/export story is static-first
- `make install` and `cmake --install` both resolve to the same maintained
  static archive surface
- `pkg-config` and `find_package(Sparse)` are described as two consumer fronts
  onto that same installed package
- the version metadata propagation chain is stated directly

### Top-level productization story

`README.md` now says plainly that:

- the installed library is `libsparse_lu_ortho.a`
- exported CMake and `pkg-config` metadata both describe that same static
  archive surface
- version metadata is single-sourced from `VERSION`
- the current package surface is real but not a broad shared-library or
  dynamic-ABI guarantee

### Maintainer policy

`docs/maintainer_guide.md` now owns the maintained interpretation directly:

- real install/export surface
- static-first release shape
- narrow ABI promise
- reviewed platform fence for that package story

### Focused regression support

`tests/test_cmake_install.sh` now reads the expected installed package version
from the repo `VERSION` file instead of hardcoding `1.0.0`.

That removes a real packaging/versioning contradiction from the focused CMake
install regression surface.

## Validation

Because this was substantial packaging/productization work, the stronger
reviewed baseline was used:

- `make quality-review-full`

Retained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 820.96 sec`

Because the package/install contract moved materially, the focused install
regressions were also run:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

Retained focused proof points:

- Make install/uninstall path passed
- `pkg-config --modversion sparse` reported `2.2.0`
- CMake install/export/find-package path passed
- the CMake install regression now verified the installed `pkg-config` version
  against the repo `VERSION` value

## Exit State

Sprint 66 Day 8 closes with:

- one landed static-first packaging/productization batch
- one resolved version-source-of-truth regression contradiction
- one explicit maintained packaging contract shared across build, install,
  top-level, and maintainer surfaces
- one clean starting point for the Day 9 post-landing rerank
