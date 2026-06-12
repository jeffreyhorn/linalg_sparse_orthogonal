# Sprint 66 Day 3: Packaging and ABI Surface Audit

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Audit the current static-first packaging surface, versioning and ABI story, and
install/platform claims from the live repo state before any Sprint 66
productization changes land.

## Current Packaging Surface

The repo already ships a real install/export surface:

- `CMakeLists.txt` installs:
  - `sparse_lu_ortho`
  - public headers under `include/sparse/`
  - generated `sparse_version.h`
  - exported `SparseTargets.cmake`
  - `SparseConfig.cmake`
  - `SparseConfigVersion.cmake`
  - generated `sparse.pc`
- `INSTALL.md` documents:
  - `make install`
  - `cmake --install`
  - downstream `pkg-config` consumption
  - downstream `find_package(Sparse)` consumption
- `README.md` repeats the same downstream consumption story at the top-level
- macOS CI already runs a supplemental install + `pkg-config` verification path

This means Sprint 66 is not starting from "no packaging story." It is starting
from a real but narrow one.

## Ranked Gaps

### 1. Strongest packaging/productization gap: static-first release shape

The current library target is still declared as:

- `add_library(sparse_lu_ortho STATIC ...)`

That is the strongest current limitation because it means:

- archive install is first-class
- exported CMake target is first-class
- `pkg-config` integration is first-class
- broader shared-library distribution and ABI expectations are still outside
  the shipped promise

The strongest Day 3 conclusion is therefore not "packaging is missing." It is
"the release shape is still intentionally narrow."

### 2. Versioning is coherent, but the broader ABI story is still narrow

The version chain is already healthier than the sprint headline alone suggests:

- root `VERSION` file is the single source of truth
- CMake `project(... VERSION ...)` reads from that file
- generated `sparse_version.h` is installed
- `SparseConfigVersion.cmake` uses `SameMajorVersion`
- `sparse.pc` is generated from the same project version

That means the weak point is not the existence of version metadata. The weak
point is that the repo still does not present a broader ABI-distribution
contract that would make those version signals carry more product weight.

### 3. Downstream consumption is already stronger than the shared/ABI story

The repo already supports two real downstream consumption paths:

- Makefile install + `pkg-config`
- CMake install + `find_package(Sparse)` + `Sparse::sparse_lu_ortho`

That downstream story is stronger than the repo's broader release-shape story.
Sprint 66 should preserve that distinction instead of flattening everything
into one generic packaging complaint.

### 4. Platform truth is already partially converged, but intentionally asymmetric

The current platform claims are explicit:

- Windows stays on the reviewed CMake subset only
- `INSTALL.md` already routes Windows callers to the CMake workflow
- macOS already carries a supplemental install + `pkg-config` verification lane

So the platform problem is not "nothing is reviewed." The stronger problem is
that the reviewed packaging/install story remains narrower and more asymmetric
than a more product-like release story would eventually want.

## Highest-Value First Target

From the live repo state, the strongest first Sprint 66 implementation target
is:

- packaging/productization convergence around the existing static-first
  install/export surface

That points to the highest-value first-touch surfaces:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- workflow files only where platform truth must move with the packaging
  contract

## Measured Hotspots

Measured Day 3 hotspot sizes for the main packaging/ABI truth surfaces:

- `README.md` = `1000`
- `INSTALL.md` = `206`
- `docs/maintainer_guide.md` = `511`
- `CMakeLists.txt` = `397`
- `.github/workflows/windows-ci.yml` = `57`
- `.github/workflows/macos-ci.yml` = `111`

## Exit State

Sprint 66 Day 3 closes with:

- one explicit packaging/ABI baseline grounded in the live install/export
  surface
- one ranked gap map that separates "real install support exists" from
  "release shape is still narrow"
- one fixed interpretation that the repo already has credible developer-install
  quality even though it does not yet ship a broader shared/ABI promise
- one clear Day 4 starting point for the platform-residual recheck
