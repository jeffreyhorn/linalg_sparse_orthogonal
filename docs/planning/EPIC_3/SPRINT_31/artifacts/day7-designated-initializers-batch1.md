# Sprint 31 Day 7 Designated Initializers Batch I

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Replace the first benchmark batch of positional public option-struct initialization with designated initializers, using a style that is explicit about non-default overrides and resilient to future struct growth.

## Files Updated

- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`

## Changes

### `bench_colamd.c`

Converted the three QR option initializers:

- `opts_none`
- `opts_amd`
- `opts_colamd`

from positional form:

- `{SPARSE_REORDER_..., 0, 0}`

to designated form:

- `{.reorder = SPARSE_REORDER_...}`

Reason:

- the benchmark only intends to override the reorder mode
- the remaining QR option fields should stay at their default zero / `NULL` values
- designated initialization avoids binding the benchmark to the current trailing field layout of `sparse_qr_opts_t`

### `bench_chol_csc.c`

Converted the forced linked-list Cholesky option initializer from positional form to:

- `.reorder = SPARSE_REORDER_AMD`
- `.backend = SPARSE_CHOL_BACKEND_LINKED_LIST`

Reason:

- those are the only two non-default behaviors the benchmark needs to force on that path
- leaving trailing fields implicit avoids stale coupling to callback / context / telemetry additions in `sparse_cholesky_opts_t`

## Chosen Initialization Style

Day 7 standard for public-facing benchmark/example option structs:

- name only the fields the benchmark is intentionally overriding
- rely on zero-initialization for default-valued trailing fields
- do not mirror the full struct layout positionally in benchmark code

This matters because Sprint 29 already added trailing callback fields to several option structs, and positional initialization now creates avoidable warning churn whenever public structs grow.

## Validation

Validation commands:

- `make format`
- `cmake --build build/sprint31-day1-cmake --parallel 1 --target bench_colamd bench_chol_csc`
- `./build/sprint31-day1-cmake/bench_colamd`
- `./build/sprint31-day1-cmake/bench_chol_csc --small-corpus --repeat 1`
- `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`

Observed results:

- targeted rebuild stderr was clean for the Day 7 benchmark pair
- `bench_colamd` still ran and printed the expected reorder comparison table
- `bench_chol_csc --small-corpus --repeat 1` still ran and printed the expected CSV rows

Warning deltas from the clean comparable rebuild:

- full-tree warnings: `109 -> 105`
- benchmark/example warnings: `11 -> 7`
- `-Wmissing-field-initializers`: `10 -> 6`
- `-Wdouble-promotion`: unchanged at `1`

Per-file Day 7 reduction:

- `benchmarks/bench_colamd.c`: `3 -> 0`
- `benchmarks/bench_chol_csc.c`: `1 -> 0`

## Remaining Sprint 31 Initializer Queue

After Day 7, the remaining benchmark/example initializer-cleanup sites are:

- `benchmarks/bench_ldlt_csc.c`
  - `2` `-Wmissing-field-initializers`
- `benchmarks/bench_main.c`
  - `3` `-Wmissing-field-initializers`
- `examples/example_colamd.c`
  - `1` `-Wmissing-field-initializers`

Non-initializer warning still remaining in the Sprint 31 benchmark/example queue:

- `benchmarks/bench_convergence.c`
  - `1` `-Wdouble-promotion`

## Day 7 Conclusion

Day 7 completed the first initializer batch cleanly:

- the two targeted benchmarks now use explicit designated initialization
- the first four missing-field warnings in the Sprint 31 queue are gone
- the remaining initializer work is reduced to `bench_ldlt_csc.c`, `bench_main.c`, and `example_colamd.c`
