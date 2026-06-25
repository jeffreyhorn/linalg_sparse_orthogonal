# Sprint 88 Day 9: Examples / Workflow Simplification Batch

## Purpose

Land one bounded examples/workflow simplification batch that makes the live
post-README adoption path read like Sprint 88's second usability contract.

## Main Result

Sprint 88's second implementation landing stayed inside the Day 8 fence:

- required implementation center:
  - `examples/README.md`
- directly forced support follow-through actually needed:
  - none
- not needed in the batch:
  - `docs/tutorial.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `README.md`
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
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

The kept example/workflow usability win is explicit:

- `examples/README.md` now opens with a compact `Start Here` section
- the strongest next-step examples are ordered by workflow/adoption intent
- the direct, repeated-run direct, iterative, eigensolver, and
  installed-consumer example lanes now read as distinct choices
- the strongest example entries now include clearer follow-on routing instead
  of leaving the next learning step implicit
- the file now preserves the support split more clearly:
  - examples = adoption and workflow teaching
  - tutorial = fuller repeated-run and API walkthrough
  - benchmarks = retained workflow/performance proof
  - tests = regression, oracle, and property guarantees

## Strongest Clarification

The useful Day 9 clarification is now explicit:

- the second Sprint 88 usability win does not require tutorial expansion or
  support-surface cleanup
- it comes from making the example surface behave like a compact post-README
  adoption map instead of a flatter inventory
- support-surface consolidation and later public narrative cleanup remain
  separate follow-through lanes

## Validation

The landed batch passed:

- `make quality-review-full`

Reviewed parity remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `370.46 sec`

## Exit State

- Sprint 88 now has one landed bounded examples/workflow batch.
- The live repo now gives users a clearer path from the README front door into
  the shipped example surfaces.
- The next later lane remains support-surface consolidation rather than more
  example churn.
