# Sprint 88 Day 6: README / Tutorial Batch

## Purpose

Land one bounded README/front-door simplification batch that makes the live
first-user path read like Sprint 88's adoption-first contract.

## Main Result

Sprint 88's first implementation landing stayed inside the Day 5 fence:

- required implementation center:
  - `README.md`
- directly forced support follow-through actually needed:
  - none
- not needed in the batch:
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Landed Surface

The kept front-door usability win is explicit:

- the README now opens with one compact `Start Here` section
- that section routes first-time users to:
  - the shortest local build path
  - workflow choice
  - install/downstream-consumer setup when needed
  - deeper example, benchmark, and maintainer surfaces only when needed
- `Choose a Workflow` now emphasizes the smallest real workflow choice first:
  - one-shot direct
  - repeated-run direct
  - repeated-run iterative handles
  - repeated-run eigensolver handles
- the build section now starts with the shortest realistic first-adoption path
  before widening into the full Make/CMake command reference
- the quick start now has explicit next-step routing into repeated-run direct,
  iterative, and install surfaces

## Strongest Clarification

The useful Day 6 clarification is now explicit:

- the first Sprint 88 usability win does not require touching install,
  benchmark, workflow-proof, or public-header surfaces
- it comes from making the README behave like an adoption-first front door
  instead of a mixed operator/reference surface
- examples/workflow simplification, support-surface consolidation, and later
  public narrative cleanup remain separate follow-through lanes

## Validation

The landed batch passed:

- `make quality-review-full`

Reviewed parity remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `412.54 sec`

## Exit State

- Sprint 88 now has one landed bounded README/front-door batch.
- The live repo now gives first-time users a clearer path from build to
  workflow choice to quick-start follow-on.
- Later Sprint 88 work remains centered on example/workflow simplification,
  support-surface consolidation, and public-header narrative cleanup.
