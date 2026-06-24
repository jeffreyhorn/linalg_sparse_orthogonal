# Sprint 87 Day 4: First Packaging and ABI Boundary

## Purpose

Fix the first bounded Sprint 87 packaging / ABI implementation fence so the
next design pass can define one real product-matrix contract instead of
another broad release rewrite.

## Main Result

Sprint 87 now has one explicit first implementation fence:

- required first landing:
  - `CMakeLists.txt`
- directly forced support surfaces only if the first landing truly needs them:
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- support-only proof and workflow surfaces that stay later unless the first
  landing truly forces movement:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- explicitly deferred from the first landing:
  - consumer-proof expansion as a first-batch center
  - workflow/platform follow-through as a first-batch center
  - broad docs alignment detached from a real package-contract change
  - immediate shared-library product widening
  - broad ABI-compatibility promise widening
  - generic build-system rewrite detached from the chosen product contract

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 87 move is one bounded product-matrix and build/export
  contract pass centered on `CMakeLists.txt`
- the first landing should decide how the repo wants its static/shared and
  package-version/export semantics to read before proof scripts or workflow
  widening move
- `cmake/SparseConfig.cmake.in` and `sparse.pc.in` remain directly allowed
  support surfaces only if that contract truly forces them to move
- install/export proof, downstream-consumer proof, and workflow surfaces stay
  later unless the product-contract landing truly changes their obligations

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no platform claims without maintained proof
- no broad shared-library product claim without bounded evidence
- no dynamic-ABI promise detached from explicit validation ownership
- no generic build-system rewrite detached from the chosen product contract
- no support-surface churn detached from a real landed packaging seam
- no workflow widening that outruns maintained local proof

## Exit State

- Sprint 87 now has one bounded first packaging/ABI landing center.
- Day 5 can design one explicit product-matrix contract inside that fence.
- Later consumer-proof expansion, workflow/platform follow-through, and broad
  support-surface alignment are held back until later lanes.
