# Sprint 30 Day 5 Core Fixes — Batch 2

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Apply the remaining narrow source-edit batch from the Day 3 audit:

- fix the library-proper warning sites in `src/sparse_qr.c` and `src/sparse_svd.c`,
- keep runtime behavior unchanged,
- and verify the final Sprint 30 core-library warning reduction on both primary local build paths.

## Source Changes

Edited files:

- `src/sparse_qr.c`
- `src/sparse_svd.c`

Change applied:

- `INFINITY` was replaced with `HUGE_VAL` at the remaining implementation-side `double` sentinel, condition-estimate, and infinity-return sites selected for Day 5.

Exact transformations:

- `src/sparse_qr.c`
  - `double prev_rnorm = INFINITY;` -> `double prev_rnorm = HUGE_VAL;` at two refinement sites
  - `info->condest = INFINITY;` -> `info->condest = HUGE_VAL;`
  - `return INFINITY;` -> `return HUGE_VAL;`
- `src/sparse_svd.c`
  - five `return INFINITY;` sites -> `return HUGE_VAL;`

No public headers, contracts, algorithms, or test expectations were changed.

## Validation Performed

### Fresh bounded CMake build

Build tree:

- `build/sprint30-day5-cmake`

Captured artifacts:

- `day5-cmake-configure.stdout.txt`
- `day5-cmake-configure.stderr.txt`
- `day5-cmake-build.stdout.txt`
- `day5-cmake-build.stderr.txt`

Result:

- configure: success
- full build: success

### Makefile build path

Build tree:

- `build/sprint30-day5-make`

Captured artifacts:

- `day5-make-build.stdout.txt`
- `day5-make-build.stderr.txt`

Result:

- build: success
- warning count: `0`

### Warning delta

Compared to Day 4:

- total warnings: `121` -> `112`
- `src` warnings: `9` -> `0`

Per edited file:

- `src/sparse_qr.c`: `4` -> `0`
- `src/sparse_svd.c`: `5` -> `0`

Compared to Day 1 baseline:

- total warnings: `123` -> `112`
- library-proper warnings: `11` -> `0`

Remaining full-build warnings after Day 5 are all auxiliary:

- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

### Targeted regression slice

Ran:

- `ctest --test-dir build/sprint30-day5-cmake --output-on-failure -R 'test_qr|test_svd|test_bidiag|test_dense'`

Observed executed tests:

- `test_qr`
- `test_dense`
- `test_bidiag`
- `test_svd`

Result:

- `4/4` passed
- total time: `5.32 sec`

## Day 5 Conclusion

Day 5 completed the Sprint 30 core-library warning cleanup batch:

- the remaining `src/` warning sites in `sparse_qr` and `sparse_svd` are clean,
- the full-tree CMake warning count dropped by the exact remaining library-proper amount,
- and both the Makefile path and a QR/SVD-focused regression slice stayed green.

Sprint 30’s targeted core warning cluster is now fully closed. The remaining warning debt is outside the library proper and can move to the later auxiliary-code cleanup work already identified in the Day 2 taxonomy.
