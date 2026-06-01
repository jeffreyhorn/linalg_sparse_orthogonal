# Sprint 52 Day 8: Factor-Many Benchmark Proof

## Purpose

Day 8 turns Sprint 52's factor-many story into measured evidence rather than
leaving it implied by the earlier integration patches. The goal is not to
redesign the benchmark framework; it is to ensure the strongest repeated-run
direct benchmark actually exercises the public same-pattern refactor contract.

## Main Day 8 Conclusion

Sprint 52 now has a materially stronger repeated-run benchmark proof:

- `bench_refactor` no longer refactors the exact same unchanged matrix each
  iteration
- it now perturbs numeric values while preserving sparsity pattern
- it times the public repeated-run lifecycle in auditable pieces:
  - one-shot average
  - analyze-once cost
  - first numeric factorization cost
  - later refactor average
  - total repeated-run average
  - final solve residual
- the repeated-run path remains faster on every shipped Day 8 case

This stays inside the Sprint 52 scope fence:

- no benchmark-framework redesign
- no direct-solver API redesign
- no LU lifecycle expansion
- no broad docs/tutorial sweep

## Touched Code

### `benchmarks/bench_refactor.c`

Day 8 rewrites the benchmark's core proof surface so it measures a real
same-pattern repeated-run scenario.

The one-shot path now does this on every iteration:

1. copy the base matrix
2. perturb the copied matrix values
3. factor with `sparse_cholesky_factor(...)`

The repeated-run path now does this:

1. `sparse_analyze(...)` once
2. `sparse_factor_numeric(...)` once on the base matrix
3. for later iterations:
   - copy the base matrix
   - perturb values while preserving sparsity pattern
   - `sparse_refactor_numeric(...)`
4. on the final perturbed matrix:
   - build a unit-solution right-hand side
   - `sparse_factor_solve(...)`
   - report the final relative residual

The benchmark output now makes the lifecycle story explicit instead of hiding
everything in one aggregate timing:

- `oneshot`
- `analyze_once`
- `initial`
- `refactor_avg`
- `repeated_avg`
- `speedup`
- `residual`

### `benchmarks/README.md`

Day 8 also aligns the benchmark docs with the live measured behavior:

- `bench_refactor` is now described as a same-pattern numeric-value-change
  benchmark, not just a generic analyze-once/refactor-many claim
- the README now states the concrete timing breakdown the benchmark prints
- `bench_refactor_csc` remains the CSC-vs-linked-list repeated-run comparison
  companion

## Important Mid-Batch Catch

The first Day 8 draft used a private internal header to mutate matrix values by
walking `SparseMatrix` internals directly.

That was a real mistake for two reasons:

1. it violated the intended public-proof boundary for this benchmark
2. it was not portable across the reviewed CMake path

`make quality-review-full` caught the issue when the CMake parity build failed
to find the private header from the benchmark target.

The final Day 8 landing fixes that properly:

- remove the private header
- perturb values through the public matrix API only

That leaves the benchmark both more portable and more honest as a public
lifecycle proof surface.

## Measured Results

### `./build/bench_refactor`

Representative Day 8 results:

- `tridiag-50`
  - `oneshot = 0.0001s`
  - `repeated_avg = 0.0000s`
  - `speedup = 2.83x`
  - `residual = 3.70e-16`
- `tridiag-200`
  - `oneshot = 0.0009s`
  - `repeated_avg = 0.0002s`
  - `speedup = 4.80x`
  - `residual = 5.18e-16`
- `tridiag-500`
  - `oneshot = 0.0046s`
  - `repeated_avg = 0.0009s`
  - `speedup = 5.19x`
  - `residual = 4.44e-16`
- `bcsstk04`
  - `oneshot = 0.0038s`
  - `repeated_avg = 0.0016s`
  - `speedup = 2.42x`
  - `residual = 1.15e-15`
- `nos4`
  - `oneshot = 0.0005s`
  - `repeated_avg = 0.0002s`
  - `speedup = 2.49x`
  - `residual = 1.18e-15`

### `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Measured Day 8 result:

- `analyze_ms = 1.224`
- `refactor_ll_ms = 0.426`
- `refactor_csc_ms = 0.186`
- `solve_ll_ms = 0.025`
- `solve_csc_ms = 0.007`
- `speedup_refactor = 2.29x`
- `res_ll = 8.24e-16`
- `res_csc = 7.06e-16`

## Validation

Because `bench_refactor.c` changed, the full required code-day gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Because this remained a substantial repeated-run proof patch, the stronger
reviewed baseline was also rerun:

- `make quality-review-full`

That also passed after the public-API portability correction.

## Day 8 Operational Result

Sprint 52 now has real measured evidence for the repeated-run direct story:

1. analyze-once / factor-many still beats repeated one-shot factorization on
   all shipped Day 8 cases
2. the benchmark now proves same-pattern value-changing work, not a static
   unchanged-matrix loop
3. the benchmark no longer depends on private matrix internals
4. the README now matches the live benchmark contract

That closes the benchmark-proof gap cleanly enough for the next day to focus
on any remaining regression-proof or caller-surface work rather than reopening
whether Sprint 52's factor-many claims are actually measured.
