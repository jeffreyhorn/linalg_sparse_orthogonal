# Sprint 34 Day 7: CMake parity design

## Goal

Define a truthful Sprint 34 CMake-path reviewed-quality contract that complements the Day 5-6 Makefile wrappers without pretending the CMake path already replaces every local quality check.

## Current audited state

The inherited `build/sprint33-day1-cmake` tree still proves the active-suite surface and a meaningful compile-quality cross-check:

- `ctest -N` reports `53` registered tests
- the CMake tree builds:
  - `25` `src` translation units
  - `53` `tests`
  - `13` `benchmarks`
  - `6` `examples`

The build tree therefore already proves more than just library compilation, but it is not a full mirror of the Makefile-reviewed surface.

## What CMake parity must cover directly

Sprint 34 phase 1 should require a named CMake-path flow for:

1. configure a reviewed build tree
2. perform a clean serialized rebuild
3. show the active registered suite via `ctest -N`
4. run the active suite via full `ctest`

These are the parts that are genuinely CMake-native and should not be left as tribal knowledge.

## What remains Makefile-authoritative in phase 1

The following checks should stay outside the Day 8 CMake wrapper contract:

- `make format-check`
- `make lint` static-analysis phases (`clang-tidy`, `cppcheck`)
- `make deadcode-check`
- the full Makefile compile-only bench/example surface

Reason: these are either formatter/static-analysis workflows that are already intentionally Makefile-centered, or they still depend on the Sprint 33 dead-code reporting pipeline and its explicit compile-db limitation.

## Current compile-db coverage gap

The current CMake/dead-code compilation database still excludes:

- benchmark:
  - `bench_svd`
- examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

This gap must remain explicit in reporting and documentation until it is actually closed. Scanner silence on those files is not evidence that they are covered.

## Day 8 implementation contract

Add dedicated reviewed CMake wrapper targets with a neutral build tree:

- variable:
  - `QUALITY_REVIEW_CMAKE_DIR ?= build/quality-review-cmake`
- target:
  - `quality-review-cmake-compile`
  - steps:
    - `cmake -S . -B $(QUALITY_REVIEW_CMAKE_DIR) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
    - `cmake --build $(QUALITY_REVIEW_CMAKE_DIR) --parallel 1 --clean-first`
    - `ctest -N --test-dir $(QUALITY_REVIEW_CMAKE_DIR)`
- target:
  - `quality-review-cmake`
  - steps:
    - run `quality-review-cmake-compile`
    - then `ctest --test-dir $(QUALITY_REVIEW_CMAKE_DIR) --output-on-failure`

## Why this shape

- It mirrors the existing Day 5-6 reviewed wrapper pattern:
  - compile-oriented path
  - full reviewed path
- It keeps `ctest -N` visible as an operator-facing active-suite audit step.
- It preserves readable failure attribution.
- It avoids reusing `build/sprint33-day1-cmake`, which is a historical sprint artifact tree rather than a maintained reviewed-quality path.

## Explicit non-goals for Day 8

- Do not redefine `check`, `lint`, `test`, `quality-review`, or `deadcode-check`.
- Do not claim dead-code compile-db parity is complete unless the excluded benchmark/example set is actually added.
- Do not turn benchmark/example runtime execution into part of the reviewed compile-quality contract.

## Expected outcome

After Day 8, Sprint 34 should have:

- a named reviewed Makefile path
- a named reviewed CMake parity path
- explicit documentation of the dead-code compile-db exclusion list

That is enough for phase-1 enforcement without overstating what the CMake path currently guarantees.
