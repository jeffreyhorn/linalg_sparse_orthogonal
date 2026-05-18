# Sprint 31 Day 5 Portability Audit

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Confirm the exact cause of the Sprint 31 benchmark portability warnings and choose the standard feature-test pattern for the Day 6 implementation batch.

## Current Warning Sites

The active Sprint 31 portability warnings still live in:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

From the current warning inventory:

- `bench_main.c`
  - `1` `-Wimplicit-function-declaration`
  - site is `snprintf`
- `bench_convergence.c`
  - `1` `-Wimplicit-function-declaration`
  - site is `snprintf`

## Root Cause Analysis

Both files share the same structure:

- `#define _POSIX_C_SOURCE 199309L`
- `#include <stdio.h>`
- `#include <time.h>`
- use `clock_gettime(CLOCK_MONOTONIC, ...)`
- later call `snprintf(...)`

This matters because:

- the files are not missing `<stdio.h>`
- the implicit-declaration warning is therefore not a simple include omission
- the declaration is hidden by the chosen feature-test surface

Conclusion:

- this is a feature-test macro mismatch
- not a missing-header bug
- not a benchmark-local typo

## Nearby Pattern Audit

Benchmark entry points using the same old POSIX surface:

- `benchmarks/bench_scaling.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_main.c`
- `benchmarks/bench_ldlt_csc.c`
- `benchmarks/bench_convergence.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_svd.c`

Important distinction:

- only `bench_main.c` and `bench_convergence.c` currently combine the old `_POSIX_C_SOURCE 199309L` baseline with `snprintf`
- the other files still use the same timer/macro pattern, so they are relevant when choosing a consistent benchmark-wide portability convention

Examples check:

- no Sprint 31 example target currently shows this `_POSIX_C_SOURCE` / `snprintf` combination
- the example queue remains an initializer-style issue, not a timer-feature-surface issue

## Chosen Day 6 Fix Pattern

Chosen standard for benchmark entry points:

- raise `_POSIX_C_SOURCE` from `199309L` to `200809L`

Why:

- keeps `clock_gettime` available
- exposes the declarations the files already rely on
- removes the current Apple Clang `snprintf` warning at the right abstraction layer
- gives the benchmark directory one clearer modern POSIX baseline instead of a split convention

## Rejected Alternatives

### Remove `_POSIX_C_SOURCE`

Rejected because:

- the benchmark timer helpers rely on `clock_gettime`
- removing the macro changes the feature-surface contract more broadly and less explicitly

### Add manual declarations for `snprintf`

Rejected because:

- it patches around the symptom instead of the root cause
- it is not an appropriate portability pattern for project source files that already include `<stdio.h>`

### Change only the two warninging files and leave the rest on `199309L`

Rejected as the preferred standard because:

- it would fix the immediate warnings
- but it leaves the benchmark directory with avoidable split portability conventions

## Day 6 Implementation Scope

Primary files:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

Adjacent benchmark entry points to bring onto the same macro convention while the portability batch is open:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_scaling.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_svd.c`

## Day 5 Conclusion

Sprint 31’s portability debt is now precisely scoped:

- the live warning is in two files
- the root cause is the old feature-test baseline, not missing includes
- the Day 6 fix pattern is to move benchmark entry points to `_POSIX_C_SOURCE 200809L`
