# Sprint 66 Day 5: Packaging and Productization Design

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Define the maintained Sprint 66 packaging, install, export, and release-shape
contract before implementation begins, so later productization work tightens
the shipped story without overstating ABI or platform guarantees.

## Maintained Packaging Contract

The maintained Sprint 66 packaging contract is:

- keep the shipped release shape static-first
- keep install/export support first-class
- keep downstream `pkg-config` and `find_package(Sparse)` consumption
  first-class
- tighten wording and ownership around what that surface really promises
- do not imply broader shared-library or ABI guarantees by accident

This fixes Sprint 66 around convergence of the current install/export story,
not around broad release-shape expansion.

## Release-Shape Contract

The current static-first release shape stays authoritative:

- installed archive remains first-class
- exported CMake package remains first-class
- generated `sparse.pc` remains first-class
- generated version metadata remains first-class

What Sprint 66 should not imply:

- no default shared-library promise
- no SONAME-style or dynamic-ABI compatibility promise
- no platform-universal binary packaging claim beyond the reviewed install
  surfaces

## ABI and Versioning Contract

The authoritative version metadata chain remains:

- root `VERSION`
- CMake `project(... VERSION ...)`
- generated `sparse_version.h`
- generated `SparseConfigVersion.cmake`
- generated `sparse.pc`

The maintained interpretation is intentionally narrow:

- version metadata exists and is coherent
- package-version metadata exists and is coherent
- this does not automatically imply a broad, validated shared-library ABI
  promise

Sprint 66 should therefore clarify the ABI claim boundary rather than widen it
silently.

## Consumer Story

The maintained downstream-consumption story is:

- Unix-like Makefile install remains supported
- downstream `pkg-config` remains supported
- downstream CMake `find_package(Sparse)` remains supported
- Windows consumption remains CMake-first

The design goal is to make those paths read as one intentional story instead of
as scattered install conveniences with different strengths of claim.

## Platform Truth Fence

The converged packaging story must still respect the reviewed platform split:

- Linux remains the strongest reviewed source of truth
- macOS remains reviewed but narrower, with supplemental install + `pkg-config`
  validation
- Windows remains the reviewed CMake subset and CMake install-consumer lane
- dead-code asymmetries remain staged and explicit

Sprint 66 packaging work may improve clarity and product maturity, but it must
not imply stronger platform closure than the workflows actually enforce.

## Ownership Split

The converged ownership model is:

- `CMakeLists.txt`
  - release shape
  - install/export topology
  - package-config generation truth
- `INSTALL.md`
  - operator-facing install instructions
  - downstream-consumption instructions
  - platform-specific install-path caveats
- `README.md`
  - compact top-level packaging/productization story
  - compact downstream-consumption summary
- `docs/maintainer_guide.md`
  - authoritative interpretation of the narrow ABI/platform contract
  - staged/deferred residual queue
- workflows and regression checks
  - reviewed evidence only for the install/platform lanes that are actually
    claimed

## First Implementation Fence

The highest-value first implementation set is:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`

Likely support or reconciliation surfaces only if the landing proves they are
needed:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Explicitly not part of the first packaging batch:

- broad shared-library enablement
- broad ABI guarantee widening
- Windows Makefile reviewed-wrapper parity
- macOS dead-code enablement
- Windows dead-code enablement
- dead-code topology redesign

## Exit State

Sprint 66 Day 5 closes with:

- one explicit packaging/productization contract
- one fixed static-first safety fence
- one explicit ownership split across build, docs, maintainer policy, and
  workflows
- one bounded Day 6-10 implementation fence for the first landing
