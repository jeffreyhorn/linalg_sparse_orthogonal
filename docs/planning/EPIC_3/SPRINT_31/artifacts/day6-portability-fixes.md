# Sprint 31 Day 6 Portability Fixes

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Apply the Day 5 benchmark portability pattern by moving the timer-based benchmark entry points from `_POSIX_C_SOURCE 199309L` to `_POSIX_C_SOURCE 200809L`, then confirm the warning-class impact with a clean serialized CMake rebuild.

## Files Updated

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

Every change in this batch was the same:

- `_POSIX_C_SOURCE 199309L -> 200809L`

No benchmark logic, data paths, or CLI behavior changed in this Day 6 batch.

## Why This Fix

From the Day 5 audit:

- `bench_main.c` and `bench_convergence.c` already included `<stdio.h>`
- both files used `clock_gettime(...)`
- both files later called `snprintf(...)`
- both files still requested `_POSIX_C_SOURCE 199309L`

That meant the active warning was a feature-test mismatch, not a missing-include bug.

Raising the macro to `200809L` was chosen because it:

- keeps the POSIX timer surface explicit
- preserves `clock_gettime` availability
- exposes the declarations the files already rely on
- avoids a split portability convention across benchmark entry points

## Validation

Validation commands:

- `make format`
- `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`

Derived from the captured Day 6 rebuild logs:

- full-tree warnings: `109`
- benchmark/example warnings: `11`

Relative warning deltas:

- full-tree warnings: `111 -> 109` versus Day 3
- benchmark/example warnings: `13 -> 11` versus Day 3
- full-tree warnings: `112 -> 109` versus Day 1 baseline
- benchmark/example warnings: `14 -> 11` versus Day 1 baseline

Per-file benchmark/example deltas most directly affected by the portability change:

- `bench_main.c`: `4 -> 3`
- `bench_convergence.c`: `2 -> 1`

Warning-class deltas:

- `-Wimplicit-function-declaration`: `2 -> 0`
- `-Wmissing-field-initializers`: unchanged at `10`
- `-Wdouble-promotion`: unchanged at `1`

## Remaining Benchmark/Example Warning Queue

After Day 6, the benchmark/example warnings are now:

- `benchmarks/bench_chol_csc.c`
  - `1` `-Wmissing-field-initializers`
- `benchmarks/bench_colamd.c`
  - `3` `-Wmissing-field-initializers`
- `benchmarks/bench_convergence.c`
  - `1` `-Wdouble-promotion`
- `benchmarks/bench_ldlt_csc.c`
  - `2` `-Wmissing-field-initializers`
- `benchmarks/bench_main.c`
  - `3` `-Wmissing-field-initializers`
- `examples/example_colamd.c`
  - `1` `-Wmissing-field-initializers`

Interpretation:

- the active portability warning class is closed
- the remaining queue is now limited to designated-initializer cleanup plus one numeric-promotion warning in `bench_convergence.c`

## Day 6 Conclusion

Day 6 completed the Sprint 31 portability batch cleanly:

- the benchmark entry points now share one `_POSIX_C_SOURCE 200809L` convention
- the Apple Clang `snprintf` implicit-declaration warnings are gone
- no new warning class was introduced
- the next sprint day can focus on the remaining mechanical warning queue
