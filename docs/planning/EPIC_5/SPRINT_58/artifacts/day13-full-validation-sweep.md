# Sprint 58 Day 13 - full validation sweep

Date: 2026-06-07
Branch: `sprint-58`

## Scope

Reconfirm the full reviewed baseline and the targeted Sprint 58 public-surface
follow-ons from the final landed tree so Day 14 can close from one explicit
validated baseline.

## Required baseline

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

## Reviewed parity anchors

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 481.74 sec`

Recorded nuance:

- the reviewed CMake rebuild emitted ordinary compiler warnings while building
  some benchmark/example binaries
- the reviewed path still completed cleanly and passed all parity gates

## Targeted Sprint 58 public-surface follow-ons

Passed:

- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Representative retained outputs

### Examples

- `example_analysis`
  - solve residual = `4.44e-16`
  - repeated-run messaging still distinguishes reused symbolic/permutation
    setup from non-reused stale numeric factors
- `example_iterative`
  - unpreconditioned GMRES: `25` iterations, residual `9.56e-11`
  - ILU(0)-GMRES: `9` iterations, residual `3.14e-11`
- `example_ic_minres`
  - `MINRES` on KKT `42x42`: `39` iterations, residual `3.87e-11`
  - `Jacobi-MINRES`: `26` iterations, residual `4.16e-11`
- `example_eigs`
  - nos4 largest-eigenvalue demo: `5 / 5` converged pairs in `115`
    Lanczos iterations
  - KKT nearest-sigma demo: `3 / 3` converged pairs in `6` Lanczos iterations
  - explicit `LOBPCG` on `bcsstk04`: `3 / 3` converged pairs in `62`
    outer iterations, residual `8.808e-09`
- `example_svd_lowrank`
  - sparse low-rank `k=2`: `22 -> 6` nnz
  - compression = `3.7x`

### Benchmarks

- `bench_refactor`
  - `tridiag-200 2.46x`
  - `tridiag-500 1.25x`
  - `bcsstk04 2.01x`
  - `nos4 0.72x`
- `bench_refactor_csc nos4 --repeat 1`
  - `speedup_refactor = 2.66x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `bench_iterative_reuse`
  - `cg-tridiag-300 2.55x`
  - `gmres-unsym-220 1.26x`
  - `minres-kkt-42 1.97x`
- `bench_eigs_reuse`
  - `growm-nos4-k5 1.31x`
  - `thick-bcsstk14-k5 1.07x`
  - `lobpcg-diag40-k3 1.00x`
  - `|lambda|max diff = 0.000e+00`

## Conclusion

Sprint 58 Day 13 closes with one explicit validated baseline:

- all required validation passed
- reviewed parity remained exact
- the simplified public docs/example/benchmark story still matches the live
  outputs from the final tree
- no blocker-level drift remains before Day 14 closeout
