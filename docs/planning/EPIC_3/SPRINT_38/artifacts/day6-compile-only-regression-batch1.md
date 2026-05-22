# Sprint 38 Day 6 Compile-Only Regression Protection Batch I

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Close the named Sprint 34 dead-code compile-db exclusion list by expanding the
dead-code CMake registration surface to include the omitted benchmark/example
entry points, while keeping the maintained Makefile compile-only contract
unchanged.

## Changes Made

### 1. Added the missing benchmark target to the dead-code CMake surface

Added:

- `bench_svd`

to the non-Windows benchmark registration block in `CMakeLists.txt`.

Additional target detail:

- `bench_svd` now also gets private `src/` include dirs because it includes the
  internal header `sparse_svd_internal.h`

### 2. Added the missing example targets to the dead-code CMake surface

Added:

- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

to the example registration block in `CMakeLists.txt`.

### 3. Updated dead-code report wording for the zero-gap state

Updated `scripts/deadcode_report.py` so the `## Coverage Gaps` section now says:

- gap-present wording when coverage-gap rows exist
- `No current benchmark/example compile-db coverage gaps are recorded in this run.`
  when the bucket is empty

This keeps the report truthful after the compile-db expansion closes the named
list.

### 4. Removed the stale README dead-code limitation claim

Removed the old README statement that still said the dead-code compilation
database missed `bench_svd` and six examples, since that is no longer true after
this batch.

## Validation

Authoritative serial validation:

- `make deadcode-report && make deadcode-check`
- `python3 -m py_compile scripts/deadcode_report.py`

Observed end state:

- `build/deadcode/coverage-notes.txt`
  - `benchmarks 14`
  - `examples 12`
  - empty `missing_benchmarks`
  - empty `missing_examples`
- `build/deadcode/report.md`
  - `src=25`
  - `tests=53`
  - `benchmarks=14`
  - `examples=12`
  - coverage-gap section now reports no current benchmark/example compile-db gaps
- `build/deadcode/report.tsv` bucket counts:
  - `public-surface-review` = `4`
  - `secondary-candidate-signal` = `35`
  - `non-deadcode-static-analysis-noise` = `6`
  - `coverage-gap` = `0`

## What This Batch Did Not Change

- no benchmark/example runtime execution behavior
- no reviewed wrapper behavior
- no shared-path dead-code serialization model
- no content-based strengthening of `deadcode-check`

## Residual Queue After Day 6

Closed:

- the named Sprint 34 dead-code compile-db exclusion list

Still remaining for later Sprint 38 work:

- dead-code report/check maturity beyond completeness
- shared-path execution-model limits
- residual `cppcheck` supporting/noise bucket interpretation improvements

That leaves the dead-code maturity queue narrower and cleaner for Days 7-8.
