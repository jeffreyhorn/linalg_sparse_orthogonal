# Sprint 77 Day 3 Artifact: Release-Surface Re-audit

Date: 2026-06-17
Branch: sprint-77

## Purpose

Re-rank the live release, install, export, and platform-quality surface by
downstream value, maintenance clarity, and truthfulness risk so Sprint 77
starts from the strongest bounded packaging/platform seam rather than from a
generic release wishlist.

## Inputs Reviewed

- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `CMakeLists.txt`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Main Result

Sprint 77's release/install/platform pressure is no longer a generic
"more packaging, more CI, more parity" problem. It is now one ranked
contradiction map:

- strongest first target:
  - operator-facing install/export contract and release-shape reading
- strongest second target:
  - authoritative reviewed-platform and packaging-policy surface
- strongest third target:
  - exported-package metadata and local install-proof ownership seam
- strongest support-surface contradiction:
  - compact front-door package/platform summary
- strongest adjacent but not first-batch lane:
  - reviewed-platform asymmetry across macOS supplemental install proof and
    Windows CMake-only consumer proof

## Ranked Findings

### 1. The strongest first landing is the operator-facing install/export contract

The strongest current release seam is concentrated in:

- `INSTALL.md`
- the shipped static-first install/export story
- the local install-proof scripts that contract points to

This lane ranks first because it is where downstream readers most directly
interpret:

- what actually gets installed
- what `pkg-config` and `find_package(Sparse)` really promise
- whether the package surface is broad or intentionally narrow
- which platform/install claims are reviewed versus merely locally provable

The package story is already mostly coherent:

- the release shape is explicitly static-first
- the installed surface is real and maintained
- the generated version and export metadata are real
- Windows is already clearly bounded to the CMake-first consumer story

But the strongest remaining gap is also clear:

- the operator-facing install surface still carries the densest mix of
  release-shape explanation, proof interpretation, and platform-asymmetry
  reading
- that makes it the easiest place for truthful but narrow evidence to start
  sounding broader than it is

### 2. `docs/maintainer_guide.md` is the strongest second contradiction center

The maintainer guide already owns the authoritative packaging/platform policy:

- static-first packaging interpretation
- bounded ABI/version reading
- Linux as the strongest reviewed truth
- macOS as reviewed plus narrower supplemental install verification
- Windows as reviewed CMake subset and consumer lane only

It ranks second rather than first because:

- it is already policy-coherent
- it is support-first rather than the best first edit center
- the stronger immediate Sprint 77 gap is not the existence of policy, but
  making the operator-facing package contract easier to read against that
  policy

### 3. Export metadata and local install-proof ownership are the strongest third lane

The concrete package surface becomes real in:

- `CMakeLists.txt`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

This lane ranks third because it is where the maintained package surface is
materialized:

- installed static archive
- installed headers
- exported `Sparse::sparse_lu_ortho`
- generated `SparseConfig.cmake` and `SparseConfigVersion.cmake`
- generated `sparse.pc`

It does not rank first because the strongest current problem is not "the
export surface is absent." It is:

- the export surface exists
- the proof exists
- but the surrounding wording and platform interpretation still carry the
  most downstream-reading risk

### 4. `README.md` is support-only, not the first design center

The top-level README still matters because it owns the compact package and
platform story:

- strongest local reviewed baseline commands
- static-first install summary
- bounded Windows/macOS package proof summary
- compact CI contract summary

But it is not the strongest first landing because it is already intentionally
compact and deliberately avoids owning the full install/export/platform
contract.

### 5. Platform asymmetry is important, but not the first batch center

The live platform asymmetry is real:

- macOS carries supplemental Make install and `pkg-config` verification
- Windows carries only the reviewed CMake subset and consumer story
- Linux remains the strongest reviewed truth

This lane matters because it is where overclaim risk is highest if wording
drifts.

It does not rank first because the strongest current Sprint 77 problem is not
"add more platform review first." It is:

- keep the package story small and truthful
- make reviewed versus supplemental proof easier to read
- avoid letting narrower local install proof read like broader reviewed
  cross-platform parity

## Day 4 Implication

The next boundary pass should treat Sprint 77's first landing as:

- required first center:
  - `INSTALL.md`
- strongest second center:
  - `docs/maintainer_guide.md`
- strongest third/support center:
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- support-only front-door follow-through:
  - `README.md`
- adjacent but not first-batch context:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

## Exit State

Sprint 77 now has one explicit Day 3 release/platform rerank:

- start from the install/export contract lane
- treat maintainer policy as the strongest second seam
- keep export metadata and local proof ownership as the strongest third seam
- keep platform asymmetry explicit without forcing CI-lane expansion into the
  first batch
