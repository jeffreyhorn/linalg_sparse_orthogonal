# Sprint 59 Day 13 - full validation sweep

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Run the full maintained quality gates, preserve the reviewed parity anchors,
and rerun the targeted Sprint 59 follow-ons from the final landed tree.

## Required maintained gate

The full required gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Reviewed truthfulness anchors

The maintained anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 143.38 sec`

## Targeted Sprint 59 follow-ons

The explicit final-sprint follow-on set also passed:

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

## Representative retained outputs

### Direct repeated-run lifecycle

- `example_analysis` residual stayed `4.44e-16`
- `bench_refactor`:
  - `tridiag-200 1.92x`
  - `tridiag-500 1.36x`
  - `bcsstk04 1.37x`
  - `nos4 1.43x`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 2.25x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

### Iterative one-shot and handle surfaces

- `example_iterative`:
  - GMRES unpreconditioned = `25` iterations
  - ILU(0)-GMRES = `9` iterations
- `example_ic_minres`:
  - MINRES on KKT `42x42` = `39` iterations
  - Jacobi-MINRES = `26` iterations
- `bench_iterative_reuse`:
  - `cg-tridiag-300 1.23x`
  - `gmres-unsym-220 1.16x`
  - `minres-kkt-42 1.02x`

### Eigensolver one-shot and handle surfaces

- `example_eigs`:
  - nos4 = `5 / 5` pairs in `115` Lanczos iterations
  - KKT nearest-sigma = `3 / 3` pairs in `6` Lanczos iterations
  - explicit `LOBPCG` on `bcsstk04` = `3 / 3` pairs in `62` outer iterations
    with reported residual `8.808e-09`
- `bench_eigs_reuse`:
  - `growm-nos4-k5 1.09x`
  - `thick-bcsstk14-k5 1.07x`
  - `lobpcg-diag40-k3 1.00x`
  - `|lambda|max diff = 0.000e+00`

### SVD example surface

- `example_svd_lowrank`:
  - sparse low-rank `k=2` kept `22 -> 6` nnz
  - compression stayed `3.7x`

## Warning note

The reviewed CMake rebuild emitted ordinary compiler warnings while rebuilding
`bench_eigs_reuse`:

- `implicit conversion increases floating-point precision`
- triggered by `NAN` macro use in the benchmark source

This did not fail the reviewed path and is recorded here as ordinary
reviewed-build noise rather than blocker-level Sprint 59 drift.

## Conclusion

Sprint 59 now has a full measured validation baseline for final closeout:

- the full maintained gate passed
- reviewed parity remained exact
- the targeted final-sprint follow-ons passed
- representative retained outputs still support the final Epic 5 caller story
