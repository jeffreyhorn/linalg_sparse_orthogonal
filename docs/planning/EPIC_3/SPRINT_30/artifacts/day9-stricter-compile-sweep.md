# Sprint 30 Day 9 Stricter Compile Sweep

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Run a stricter compile-only or warning-focused quality pass beyond the default Sprint 30 baseline, identify any new findings, triage which are Sprint 30-actionable, and confirm that the core-library cleanup still holds under the stricter settings.

## Stricter Passes Run

### Pass 1: clean CMake build with extra prototype warnings

Configured and built a fresh tree with:

- `-Wstrict-prototypes`
- `-Wmissing-prototypes`

Commands:

1. `cmake -S . -B build/day9-strict-final-cmake -DCMAKE_C_FLAGS='-Wstrict-prototypes -Wmissing-prototypes'`
2. `cmake --build build/day9-strict-final-cmake --parallel 1`

The serialized build was intentional so warning lines stayed non-interleaved and parse-stable.

### Pass 2: warning-focused quality invocation

Ran:

- `make lint`

Result:

- completed successfully

### End-of-day validation

Ran:

- `make format`
- `make lint`
- `make test`

Result:

- all three completed successfully

## New Findings Beyond The Day 8 Default Baseline

The first strict CMake sweep surfaced exactly one new compiler warning not present in the Day 8 baseline:

- `src/sparse_types.c`
  - `-Wmissing-prototypes`
  - function: `sparse_set_errno_`

Why this mattered:

- it originated in `src/`
- Day 6’s playbook treats any `src/` warning on a supported build surface as Sprint-blocking
- it was a narrow internal-declaration issue, not a public-API or algorithm problem

## Same-Day Fix

Applied a minimal fix in `src/sparse_types.c:5`:

- added a prior declaration for `sparse_set_errno_` in its own translation unit before the definition

This preserved the helper’s internal-only status and avoided leaking it into the public header surface.

## Final Strict-Sweep Results After The Fix

### Strict CMake build

- warnings: `112`
- warning delta versus Day 8 baseline: `0`

By area:

- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`
- `src`: `0`

By warning class:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `34`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

No residual `-Wmissing-prototypes` warning remained after the fix.

### Targeted strict-tree regression slice

Ran:

- `ctest --test-dir build/day9-strict-final-cmake --output-on-failure -R 'test_sparse_matrix|test_sparse_io|test_sparse_lu|test_ldlt|test_qr|test_svd'`

Observed result:

- `7/7` passed
- total time: `6.91 sec`

Tests exercised:

- `test_sparse_matrix`
- `test_sparse_lu`
- `test_sparse_io`
- `test_qr`
- `test_svd`
- `test_ldlt`
- `test_ldlt_csc`

## Actionable vs Deferred Split

### Actionable in Sprint 30

- `src/sparse_types.c` `-Wmissing-prototypes`
  - fixed immediately on Day 9
  - reason: core-library warning under stricter supported compile settings

### Deferred / inventory only

- none newly introduced by the stricter compiler warning pass after the `sparse_set_errno_` fix
- the remaining strict-tree warnings are the same auxiliary-code classes already present in the Day 8 baseline

## Day 9 Conclusion

Day 9 succeeded as a controlled stricter-pass exercise:

- the stricter compile sweep did find a real new `src/` issue
- the issue was small, fixed the same day, and revalidated
- after the fix, the strict-tree warning profile matched the Day 8 baseline classes exactly
- the Sprint 30 core-library cleanup still holds under the stricter prototype-oriented compile settings
