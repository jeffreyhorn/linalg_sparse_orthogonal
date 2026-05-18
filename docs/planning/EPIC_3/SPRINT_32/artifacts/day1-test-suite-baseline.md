# Sprint 32 Day 1 Test-Suite Baseline

**Date:** 2026-05-17  
**Branch:** `sprint-32`

## Objective

Re-run the authoritative Apple Clang CMake baseline for the current branch and turn the Sprint 31 handoff into a concrete Sprint 32 inventory covering remaining test warnings, dormant-scaffold truthfulness in `tests/test_reorder_nd.c`, and the file-level cleanup themes that now define the sprint.

## Baseline Summary

Current authoritative Apple Clang CMake state:

- full-tree warnings: `98`
- `src` warnings: `0`
- `tests` warnings: `98`
- `benchmarks` warnings: `0`
- `examples` warnings: `0`
- warning-bearing test files: `20`

Sprint 32 starts from the validated Sprint 31 handoff exactly as intended: the compile-warning queue is now test-only, and the first structural target is `tests/test_reorder_nd.c`.

Derived support files:

- `day1-warning-counts-by-area.txt`
- `day1-warning-counts-by-class.txt`
- `day1-warning-counts-by-file.txt`
- `day1-warning-counts-by-file-and-class.txt`
- `day1-test-reorder-nd-structure.txt`

## Warning Breakdown

By warning class:

- `-Wmissing-field-initializers`: `62`
- `-Wdouble-promotion`: `33`
- `-Wunused-function`: `3`

By file:

1. `tests/test_ldlt.c`: `18`
2. `tests/test_sprint20_integration.c`: `9`
3. `tests/test_colamd.c`: `8`
4. `tests/test_chol_csc.c`: `8`
5. `tests/test_reorder_nd.c`: `7`
6. `tests/test_svd.c`: `6`
7. `tests/test_sprint6_integration.c`: `6`
8. `tests/test_sprint18_integration.c`: `6`
9. `tests/test_sprint12_integration.c`: `5`
10. `tests/test_cholesky.c`: `5`
11. `tests/test_reorder.c`: `4`
12. `tests/test_sprint19_integration.c`: `3`
13. `tests/test_bidiag.c`: `3`
14. `tests/test_sprint10_integration.c`: `2`
15. `tests/test_qr.c`: `2`
16. `tests/test_sprint5_integration.c`: `1`
17. `tests/test_lu_csr.c`: `1`
18. `tests/test_ilu.c`: `1`
19. `tests/test_etree.c`: `1`
20. `tests/test_block_solvers.c`: `1`

By file and warning class:

- `tests/test_ldlt.c`
  - `-Wmissing-field-initializers`: `18`
- `tests/test_sprint20_integration.c`
  - `-Wmissing-field-initializers`: `5`
  - `-Wdouble-promotion`: `4`
- `tests/test_colamd.c`
  - `-Wmissing-field-initializers`: `7`
  - `-Wdouble-promotion`: `1`
- `tests/test_chol_csc.c`
  - `-Wmissing-field-initializers`: `8`
- `tests/test_reorder_nd.c`
  - `-Wmissing-field-initializers`: `4`
  - `-Wunused-function`: `3`
- `tests/test_svd.c`
  - `-Wdouble-promotion`: `6`
- `tests/test_sprint6_integration.c`
  - `-Wdouble-promotion`: `6`
- `tests/test_sprint18_integration.c`
  - `-Wmissing-field-initializers`: `3`
  - `-Wdouble-promotion`: `3`
- `tests/test_sprint12_integration.c`
  - `-Wmissing-field-initializers`: `5`
- `tests/test_cholesky.c`
  - `-Wmissing-field-initializers`: `5`
- `tests/test_reorder.c`
  - `-Wmissing-field-initializers`: `4`
- remaining smaller files
  - `tests/test_sprint19_integration.c`: `2` initializer, `1` double-promotion
  - `tests/test_bidiag.c`: `3` double-promotion
  - `tests/test_sprint10_integration.c`: `2` double-promotion
  - `tests/test_qr.c`: `2` double-promotion
  - `tests/test_sprint5_integration.c`: `1` double-promotion
  - `tests/test_lu_csr.c`: `1` double-promotion
  - `tests/test_ilu.c`: `1` double-promotion
  - `tests/test_etree.c`: `1` initializer
  - `tests/test_block_solvers.c`: `1` double-promotion

## Current `test_reorder_nd.c` Truthfulness State

Observed from the current source:

- static test functions in file: `26`
- active `RUN_TEST(...)` sites: `23`
- commented-out `RUN_TEST(...)` sites: `3`

The dormant helpers still compiled into the file are:

1. `test_finest_fm_annealing_pres_poisson_close_to_target`
2. `test_nd_root_spectral_pres_poisson_close_to_target`
3. `test_non_pipeline_pres_poisson_close_to_target`

Interpretation:

- these functions are not part of the active suite today, but they still compile and therefore still generate the tree’s only `-Wunused-function` warnings
- the file is not merely carrying mechanical cleanup debt; it also misstates the real executed protection surface by retaining explicit but inactive historical scaffold
- the `4` positional initializer warnings in the same file mean Sprint 32 should clean truthfulness drift and warning drift together rather than treating them as unrelated

## File-Level Cleanup Themes

### Priority 1: truthfulness and dormant scaffold

Primary file:

- `tests/test_reorder_nd.c`

Why it leads:

- it is the only file with dormant compiled test bodies and the only source of `-Wunused-function` warnings
- it directly determines whether the suite honestly represents what CI protects today

### Priority 2: high-volume designated-initializer cleanup

Primary files:

- `tests/test_ldlt.c`
- `tests/test_colamd.c`
- `tests/test_chol_csc.c`
- `tests/test_cholesky.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_reorder.c`
- `tests/test_etree.c`

Why this is bounded:

- the warning class matches the already-solved Sprint 31 benchmark/example pattern
- the remaining queue is large in count but mechanically uniform

### Priority 3: mechanical double-promotion cleanup

Primary files:

- `tests/test_sprint20_integration.c`
- `tests/test_svd.c`
- `tests/test_sprint6_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_bidiag.c`
- `tests/test_sprint19_integration.c`
- `tests/test_qr.c`
- `tests/test_sprint10_integration.c`
- `tests/test_block_solvers.c`
- `tests/test_ilu.c`
- `tests/test_lu_csr.c`
- `tests/test_sprint5_integration.c`

Why it is secondary:

- it is real warning debt, but it does not distort the executed suite the way the dormant ND scaffold does
- the fixes should be narrow and localized once the structural truthfulness work is settled

## Day 1 Conclusion

Sprint 32 begins from a stable and sharply bounded queue:

- all remaining compile warnings are in `tests/`
- `tests/test_reorder_nd.c` is the leading structural target because it combines dormant-scaffold honesty drift with the tree’s only `-Wunused-function` warnings
- designated-initializer cleanup remains the dominant raw warning class and can likely reuse the Sprint 31 option-struct cleanup pattern directly
- double-promotion cleanup is smaller and more isolated, making it a later mechanical batch once the truthfulness and initializer passes are complete
