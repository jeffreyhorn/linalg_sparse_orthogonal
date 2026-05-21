# Sprint 37 Day 6 Benchmark-Helper Consolidation Batch I

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Convert the Day 3 benchmark-helper audit into the safest first consolidation
slice by extracting the repeated backend-comparison support logic shared by
`bench_chol_csc.c` and `bench_ldlt_csc.c`, while preserving benchmark semantics
and the Sprint 31 benchmark behavior contract.

## Executive Summary

Day 6 shipped a narrow shared helper layer for the backend-comparison
benchmark pair:

- `benchmarks/bench_backend_compare_helpers.h`

It consolidates the repeated timer, residual, path-load, and unit-RHS setup
logic shared by:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`

This removes real duplicated support code without introducing a new linked
benchmark-common module or widening the refactor into the larger benchmark
behavior-owner files.

## What Landed

### New shared helper surface

- `bench_backend_result_t`
- `bench_backend_wall_time(...)`
- `bench_backend_rel_residual_max(...)`
- `bench_backend_load_matrix(...)`
- `bench_backend_make_unit_rhs(...)`

These helpers cover the exact low-risk overlap Day 3 identified:

- wall-clock timing
- max-relative-residual calculation
- file-backed matrix load + display-label extraction
- `b = A * [1, ..., 1]` setup plus solve workspace allocation
- common factor/solve/residual result shape

## Why This Landing Shape Was Chosen

The Day 3 audit already established the key constraint:

- benchmarks still build one executable per source file in both Makefile and
  CMake

That makes a shared `.c` benchmark layer a worse first move because it would
add link-time ownership and broaden the review surface.

So Day 6 deliberately chose:

- a header-only helper layer
- pair-scoped support functions
- no CLI rewrites
- no CSV schema rewrites
- no cross-benchmark framework claims

This keeps the batch low risk and easy to review.

## What Was Reduced

Before Day 6, both benchmark files carried their own copies of:

- `wall_time()`
- `rel_residual(...)`
- result struct definitions
- matrix-load + basename extraction
- `b = A * 1` workspace setup

After Day 6:

- those support pieces live in one narrow helper header
- each benchmark file keeps only the behavior-owner logic that is genuinely
  local to its benchmark contract

## What Stayed Local By Design

Day 6 did **not** try to merge the parts that still own distinct semantics.

Still local in `bench_chol_csc.c`:

- scalar vs supernodal Cholesky path definitions
- synthetic small-corpus builders
- Cholesky-specific CSV columns

Still local in `bench_ldlt_csc.c`:

- native vs wrapper vs supernodal LDLT path definitions
- `--dispatch` mode
- dispatch-only CSV/reporting path

Also intentionally untouched:

- `bench_main.c`
- `bench_reorder.c`
- `bench_eigs.c`

Those files still own larger CLI/reporting/behavior contracts and did not fit
the first narrow batch.

## Validation

Focused benchmark validation:

- `make build/bench_chol_csc build/bench_ldlt_csc`
- `./build/bench_chol_csc --small-corpus --repeat 1`
- `./build/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_ldlt_csc --dispatch --repeat 1`

These direct runs confirmed the pair still behaves correctly in:

- normal file-backed backend-comparison mode
- Cholesky synthetic-corpus mode
- LDLT dispatch mode

Because benchmark `*.c` and helper `*.h` files changed, the required full gate
was:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Residual Queue After Day 6

Still deferred from the broader Day 3 benchmark-helper audit:

- timer/residual drift outside this pair:
  - `bench_refactor_csc.c`
  - `bench_bicgstab.c`
  - `bench_convergence.c`
- larger behavior-owner benchmark files:
  - `bench_main.c`
  - `bench_reorder.c`
  - `bench_eigs.c`
- benchmark-specific reporting / CLI helpers

These remain deferred because they either:

- diverge structurally from the backend-comparison pair
- own broader user-facing CLI/reporting behavior
- or would have widened the first batch beyond the intended low-risk scope

## Day 6 Conclusion

Day 6 reduced the safest high-value benchmark-helper duplication without
inventing a broad benchmark framework.

The backend-comparison pair now shares a small explicit helper layer, and the
remaining Sprint 37 benchmark-helper queue is narrower and more clearly
partitioned for later work.
