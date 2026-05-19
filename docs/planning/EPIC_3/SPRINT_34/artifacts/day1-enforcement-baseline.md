# Sprint 34 Day 1 Enforcement Baseline

**Date:** 2026-05-19  
**Branch:** `sprint-34`

## Objective

Turn the Sprint 33 closeout into a concrete Sprint 34 starting inventory by
confirming the inherited validated-state guarantees, auditing the current
Makefile/CMake/CI quality entry points, and naming the first implementation
surfaces for build-quality enforcement work.

## Baseline Summary

Sprint 34 starts from the Sprint 33 closeout exactly as intended:

- no inherited warning debt
- no inherited dormant-scaffold debt
- no inherited definitely-unused internal cleanup queue
- active `ctest` registry remains `53`
- active full `ctest` closeout baseline remains `53 / 53` passing
- dead-code workflow already exists and passes locally:
  - `make deadcode-report`
  - `make deadcode-check`

Current branch head during the Day 1 baseline capture:

- `f1fe556`

This means Sprint 34 is not a cleanup sprint. It is an enforcement sprint:
warning cleanliness, dead-code reporting, CMake parity, and CI wiring must now
be turned into repeatable rules instead of remaining mostly maintainer-driven
habits.

## Current Quality Surface

### Makefile

Existing maintained quality targets:

- `format`
- `format-check`
- `lint`
- `test`
- `check`
- `bench-build`
- `examples-build`
- `tooling-build`
- `deadcode-compile-db`
- `deadcode`
- `deadcode-report`
- `deadcode-check`
- `wall-check`

Important current behavior:

- `lint` already depends on `tooling-build`
- the strict `-Werror` compile step inside `lint` is still library-only
- benchmarks/examples are compile-checked via `tooling-build`, but not yet under
  an explicit reviewed-target warning-gate contract
- `deadcode*` is implemented, but not yet part of the normal quality path

### CMake

Current authoritative CMake validation view:

- `ctest -N --test-dir build/sprint33-day1-cmake`: `53` tests
- full `ctest`: `53 / 53` passing at Sprint 33 close

Current compilation-database visibility from the dead-code report:

- `src=25`
- `tests=53`
- `benchmarks=13`
- `examples=6`

Interpretation:

- CMake remains the auditable active-suite surface
- CMake still under-covers the full benchmark/example Makefile surface for
  dead-code purposes

### CI

Current workflow coverage:

- Ubuntu CI:
  - `make test`
  - `make sanitize`
  - `make asan`
  - `make bench-build`
  - `make bench-fast`
  - `make format-check`
  - `make lint`
  - configure/build/`ctest`
  - CMake vs Makefile test-count parity
- macOS CI:
  - build
  - `make test`
  - `make wall-check`
  - Apple Clang `make sanitize`
- Windows CI:
  - configure
  - build
  - `ctest`

What does **not** exist in CI yet:

- `make deadcode`
- `make deadcode-report`
- `make deadcode-check`

## Inherited Constraints From Sprint 33

### 1. Dead-code coverage gap remains real

The current compilation-database exclusion list still contains:

- benchmark:
  - `bench_svd`
- examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Implication:

- stronger dead-code enforcement must either broaden that coverage or preserve
  the exclusion list explicitly and truthfully in generated reporting

### 2. Dead-code execution model is still serial-first

Current dead-code targets still share:

- `build/deadcode-cmake`
- `build/deadcode/`

Implication:

- Sprint 34 should either preserve serialized invocation or isolate the
  build/artifact paths before adding CI-backed dead-code enforcement

### 3. Public-surface findings remain audited keeps, not cleanup debt

Sprint 33 already audited and kept:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

Implication:

- Sprint 34 should not reopen public-surface cleanup without a new API decision

## Local Tooling Baseline

Day 1 local tool availability:

- `cppcheck`: present
- `xunused`: present
- `clang-tidy`: present

That closes the Sprint 33 Day 1 local-prerequisite gap and means Sprint 34 can
focus on workflow hardening instead of first-install setup.

## First Implementation Surfaces

### 1. Makefile enforcement wiring

Primary file:

- `Makefile`

High-value target region:

- `lint`
- `check`
- `tooling-build`
- `deadcode*`

### 2. Dead-code helper behavior

Primary files:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`

### 3. CMake parity and coverage scope

Primary file:

- `CMakeLists.txt`

### 4. CI enforcement insertion points

Primary files:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Day 1 Conclusion

Sprint 34 starts from a strong technical baseline but an only partially unified
enforcement model:

- validated quality paths already exist
- dead-code targets already exist
- CI already enforces several quality checks
- but compile-quality and dead-code gates are not yet aligned into one reviewed
  local/CMake/CI enforcement story

That makes the next step clear: Day 2 should turn the existing surfaces into an
explicit toolchain/target enforcement matrix before Sprint 34 edits the target
graph.
