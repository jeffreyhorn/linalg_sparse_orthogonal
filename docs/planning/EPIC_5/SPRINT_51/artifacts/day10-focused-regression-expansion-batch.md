# Sprint 51 Day 10: Focused Regression Expansion Batch

## Objective

Land the smallest remaining high-signal direct-lifecycle regression additions
after the Day 9 audit, centered on public sequencing and ownership behavior
rather than broad parity churn.

## Files Changed

- `tests/test_integration.c`

## What Landed

### 1. Direct solve-before-factor rejection coverage

Added a bounded integration test that:

- analyzes a Cholesky repeated-run path
- keeps `sparse_factors_t` zeroed
- verifies `sparse_factor_solve(...)` returns `SPARSE_ERR_BADARG`

Why this mattered:

- the header contract already says callers need a valid numeric factorization
  before solve
- this makes that sequence rule locally visible in the smaller public-surface
  integration suite instead of only implicitly covered in larger lifecycle
  tests

### 2. Direct zeroed-factor refactor acceptance coverage

Added a bounded integration test that:

- analyzes once through `sparse_analyze(...)`
- uses `sparse_refactor_numeric(...)` as the first numeric factorization into a
  zeroed `sparse_factors_t`
- solves successfully
- updates only diagonal values on a same-pattern matrix
- refactors and solves successfully again

Why this mattered:

- the public repeated-run direct contract explicitly allows zeroed
  `sparse_factors_t` as the starting state for refactor
- this now has small direct regression coverage in the same suite that carries
  the Sprint 51 parity checks

## What Did Not Change

The batch intentionally did not:

- retry LU wrapper routing
- broaden `tests/test_etree.c`
- touch examples, benchmarks, or docs
- introduce any new public lifecycle semantics beyond existing header truth

## Validation

Because `tests/test_integration.c` changed, the full required code-day gate
ran and passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also ran and passed:

- `make quality-review-full`

Truthfulness anchors remained exact:

- reviewed CMake parity: `53`
- Makefile/CMake parity: `53 vs 53`
- full reviewed CMake `ctest`: `53 / 53`

Targeted follow-ons also passed:

- `./build/test_integration`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`

## Bottom Line

Sprint 51’s remaining regression work is now materially smaller:

- the public direct lifecycle core is explicitly covered for the main
  sequencing/ownership edges that were still under-centered
- the strongest next work is on adoption/documentation surfaces, not on new
  lifecycle routing or large regression expansion
