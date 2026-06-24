# Sprint 87 Day 5: Product-Matrix Design

## Purpose

Define the bounded static/shared, ABI, and downstream-consumer contract that
Sprint 87 will actually support on its first packaging lane.

## Main Result

Sprint 87 now has one explicit first implementation contract:

- required implementation center:
  - `CMakeLists.txt`
- directly forced support surfaces only if the first batch truly needs them:
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- consumer-proof and workflow surfaces remain later owners unless the first
  batch truly changes their obligations:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

## Ownership Split

The Day 5 ownership split is now fixed:

- product-matrix contract owner:
  - `CMakeLists.txt`
- retained CMake export/config owner if the product contract truly changes
  installed package semantics:
  - `cmake/SparseConfig.cmake.in`
- retained pkg-config contract owner if the product contract truly changes
  Make-installed consumer semantics:
  - `sparse.pc.in`
- retained local install/export proof owners after the first landing:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- retained downstream-consumer proof owner after the first landing:
  - `examples/cmake_example/CMakeLists.txt`
- retained workflow/platform evidence owners after the first landing:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- support-surface wording owners only if implementation truly changes the
  package contract reading:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

## Product Decision

The strongest product decision is now explicit:

- Sprint 87 remains static-first only
- it does not open a bounded shared-library product lane in this sprint
- the maintained contract should instead become sharper about:
  - static archive output as the only shipped product shape
  - `pkg-config` and `find_package(Sparse)` describing that same static
    package surface
  - version metadata being real package metadata rather than a broad
    dynamic-ABI guarantee
  - platform truth remaining narrower on macOS and Windows than on Linux

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- Day 6 should not try to "add shared"
- it should tighten the live build/export semantics so they read exactly like
  the maintained static-first product contract
- it should preserve downstream proof ownership and workflow evidence as later
  lanes rather than folding them into the first packaging batch
- it should treat package-version/export metadata as part of the contract
  language problem, not as evidence for a broader ABI promise

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- no shared-library product claim without bounded proof
- no dynamic-ABI promise detached from explicit validation ownership
- no workflow widening folded into the first batch unless the product contract
  truly forces it
- no support-surface churn detached from the landed package seam
- no generic build-system rewrite detached from the chosen product contract

## Exit State

- Sprint 87 now has one bounded static-first product-matrix contract.
- Ownership between the first package/build lane, retained consumer-proof
  owners, retained workflow evidence, and later support-surface alignment is
  fixed before Day 6 begins.
- Consumer-proof expansion, workflow/platform follow-through, and broader docs
  alignment remain explicitly outside the first packaging batch.
