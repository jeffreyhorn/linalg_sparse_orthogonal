# Sprint 60 Day 13: Full Validation Sweep

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Validate the full Sprint 60 baseline package from the frozen Day 12 checklist
so the sprint can close from one explicit reviewed baseline instead of from a
mix of intermediate checks.

## Full Validation Gate

The full required gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- reviewed CMake total time from `make quality-review-full`:
  - `Total Test time (real) = 199.78 sec`

## Targeted Sprint 60 Follow-Ons

The fixed Day 12 follow-ons also all passed:

- `./build/test_integration` -> `39 / 39`
- `./build/test_iterative` -> `79 / 79`
- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Representative Retained Outputs

- `example_analysis`
  - residual stayed `4.44e-16`
- `example_iterative`
  - GMRES: `25` iterations unpreconditioned
  - ILU(0)-GMRES: `9` iterations
- `example_ic_minres`
  - MINRES on KKT `42x42`: `39` iterations
  - Jacobi-MINRES: `26` iterations
- `example_eigs`
  - `nos4`: `5 / 5` pairs in `115` Lanczos iterations
  - KKT nearest-sigma case: `3 / 3` pairs in `6` Lanczos iterations
  - explicit `LOBPCG` on `bcsstk04`: `3 / 3` pairs in `62` outer iterations
    with reported residual `8.808e-09`
- `example_svd_lowrank`
  - sparse low-rank `k=2` kept `22 -> 6` nnz and `3.7x` compression
- `bench_refactor`
  - `tridiag-200 1.53x`
  - `tridiag-500 1.23x`
  - `bcsstk04 1.29x`
  - `nos4 1.43x`
- `bench_refactor_csc nos4`
  - `speedup_refactor = 2.37x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `bench_iterative_reuse`
  - `cg-tridiag-300 1.06x`
  - `gmres-unsym-220 1.11x`
  - `minres-kkt-42 1.00x`
- `bench_eigs_reuse`
  - `growm-nos4-k5 1.03x`
  - `thick-bcsstk14-k5 0.99x`
  - `lobpcg-diag40-k3 0.97x`
  - `|lambda|max diff = 0.000e+00`

## Day 13 Note

The reviewed CMake rebuild emitted ordinary compiler warnings while rebuilding
`bench_eigs_reuse`, but the reviewed path still completed cleanly and passed
all parity gates. No blocker-level validation drift surfaced.

## Day 13 Exit State

Sprint 60 now has a fully validated baseline package:

- full reviewed local gate passed
- reviewed CMake parity stayed exact
- targeted workflow-proof tests/examples/benchmarks all passed
- no new reconciliation queue surfaced during validation
