# Sprint 31 Day 1 Benchmark Tooling Baseline

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Re-run the authoritative Apple Clang CMake baseline for the current branch and turn the Sprint 30 benchmark/example handoff into a concrete Sprint 31 inventory covering warning sites, benchmark CLI/help drift, and file-level cleanup themes.

## Baseline Summary

Current authoritative Apple Clang CMake state:

- full-tree warnings: `112`
- benchmark warnings: `13`
- example warnings: `1`
- benchmark/example warnings combined: `14`
- warning-bearing benchmark/example files: `6`

The Sprint 31 tooling queue is unchanged from the Sprint 30 handoff:

1. `benchmarks/bench_main.c`
2. `benchmarks/bench_convergence.c`
3. `benchmarks/bench_colamd.c`
4. `benchmarks/bench_chol_csc.c`
5. `benchmarks/bench_ldlt_csc.c`
6. `examples/example_colamd.c`

Derived support files:

- `day1-tooling-warning-counts-by-file.txt`
- `day1-tooling-warning-counts-by-class.txt`
- `day1-tooling-warning-counts-by-file-and-class.txt`

## Warning Breakdown

By warning class:

- `-Wmissing-field-initializers`: `10`
- `-Wimplicit-function-declaration`: `2`
- `-Wdouble-promotion`: `1`
- `-Wswitch`: `1`

By file:

1. `benchmarks/bench_main.c`: `5`
2. `benchmarks/bench_colamd.c`: `3`
3. `benchmarks/bench_convergence.c`: `2`
4. `benchmarks/bench_ldlt_csc.c`: `2`
5. `benchmarks/bench_chol_csc.c`: `1`
6. `examples/example_colamd.c`: `1`

By file and warning class:

- `benchmarks/bench_main.c`
  - `-Wmissing-field-initializers`: `3`
  - `-Wimplicit-function-declaration`: `1`
  - `-Wswitch`: `1`
- `benchmarks/bench_convergence.c`
  - `-Wimplicit-function-declaration`: `1`
  - `-Wdouble-promotion`: `1`
- `benchmarks/bench_colamd.c`
  - `-Wmissing-field-initializers`: `3`
- `benchmarks/bench_chol_csc.c`
  - `-Wmissing-field-initializers`: `1`
- `benchmarks/bench_ldlt_csc.c`
  - `-Wmissing-field-initializers`: `2`
- `examples/example_colamd.c`
  - `-Wmissing-field-initializers`: `1`

## Current CLI/Help Behavior

### `bench_main`

Observed from the current source:

- usage text still advertises `--reorder rcm|amd|none`
- parser accepts only `none`, `rcm`, and `amd`
- unknown-reorder error text still says to use only `none`, `rcm`, or `amd`
- `reorder_name()` handles `SPARSE_REORDER_NONE`, `SPARSE_REORDER_RCM`, and `SPARSE_REORDER_AMD`, but omits `SPARSE_REORDER_COLAMD` and `SPARSE_REORDER_ND`

Interpretation:

- this remains the highest-signal Sprint 31 correctness issue
- the compile warning is small, but the bigger problem is that the main benchmark harness misrepresents the reorder modes the library already supports

### `bench_reorder`

Observed from the current source:

- benchmark set includes `NONE`, `RCM`, `AMD`, `COLAMD`, and `ND`
- CLI supports `--nd-threshold`, `--skip-factor`, `--reorder-via-analyze`, and `--only`
- output format is a fixed CSV header: `matrix,n,reorder,nnz_L,reorder_ms,factor_ms`

Interpretation:

- `bench_reorder` is already aligned to the broader reorder API surface
- Sprint 31 should treat it as a consistency reference when fixing `bench_main`

### Specialized benchmark/example programs

Observed from the current source:

- `bench_colamd` and `example_colamd` both assume `COLAMD` is a standard supported reorder path
- `bench_chol_csc` and `bench_ldlt_csc` are backend-comparison tools and do not carry the same general reorder CLI surface
- the public-facing `example_colamd.c` still demonstrates positional QR options initialization

Interpretation:

- Sprint 31 does not need to expand these programs into general reorder harnesses
- Sprint 31 does need to ensure their public-facing option initialization style is no longer brittle

## File-Level Cleanup Themes

### Priority 1: benchmark tool correctness

Primary file:

- `benchmarks/bench_main.c`

Why it leads:

- it combines stale usage text, rejected supported reorder values, incomplete reorder-name coverage, and the only benchmark-side `-Wswitch` warning

### Priority 2: portability cleanup

Primary files:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

Why this is bounded:

- the current branch still shows exactly two `snprintf` visibility warnings
- both live under the same `_POSIX_C_SOURCE 199309L` pattern identified by Sprint 30

### Priority 3: designated-initializer cleanup

Target files:

- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `examples/example_colamd.c`
- plus the touched options sites in `benchmarks/bench_main.c`

Why this is bounded:

- all current initializer warnings are the same maintenance-drift class
- the fix style is straightforward and consistent across the affected files

### Priority 4: residual mechanical numeric cleanup

Target file:

- `benchmarks/bench_convergence.c`

Why it is lower priority:

- it is real warning debt, but it is mechanical follow-through rather than API/help or portability drift

## Day 1 Conclusion

Sprint 31 starts from a stable, well-bounded tooling queue:

- no new benchmark/example warning sites appeared after Sprint 30
- the `bench_main` correctness drift remains the most important user-facing issue
- portability debt is still tightly scoped to the two benchmark entry points identified in the handoff
- initializer drift is still the dominant raw warning cluster and remains a clean mechanical batch for later Sprint 31 days
