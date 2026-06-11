# Sprint 62 Day 13: Full Validation Sweep

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Revalidate the full landed Sprint 62 state from the strongest reviewed
baseline and rerun the direct-family, example, and benchmark follow-ons that
carry the highest signal for the Sprint 62 usability package.

## Full Gate

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 377.56 sec`

## Targeted Follow-Ons

### Direct and lifecycle proof surfaces

Passed:

- `./build/test_integration` -> `43 / 43`
- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_cholesky` -> `21 / 21`
- `./build/test_ldlt` -> `84 / 84`
- `./build/test_sparse_lu` -> `37 / 37`

### Adjacent repeated-run solver proof surfaces

Passed:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_lobpcg` -> `26 / 26`

### Representative examples

Passed:

- `./build/example_analysis`
- `./build/example_basic_solve`
- `./build/example_ldlt`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`

Representative retained outputs:

- `example_analysis` residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `example_ldlt` relative residual = `1.555e-16`
- `example_iterative`: GMRES `25` iterations unpreconditioned, `9` with ILU(0)
- `example_ic_minres`: MINRES on KKT `42x42` at `39` iterations,
  Jacobi-MINRES at `26`
- `example_eigs`: `nos4` `5 / 5` pairs in `115` Lanczos iterations; KKT
  nearest-sigma `3 / 3` in `6`; explicit `LOBPCG` on `bcsstk04` `3 / 3` in
  `62` outer iterations with reported residual `8.808e-09`
- `example_svd_lowrank`: sparse low-rank `k=2` kept `22 -> 6` nnz and `3.7x`
  compression

### Representative workflow benchmarks

Passed:

- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative retained outputs:

- `bench_refactor`: `tridiag-200 1.47x`, `tridiag-500 1.25x`,
  `bcsstk04 1.30x`, `nos4 1.41x`
- `bench_refactor_csc nos4`: `speedup_refactor = 0.61x`, residuals
  `8.24e-16` / `7.06e-16`
- `bench_iterative_reuse`: `cg-tridiag-300 1.01x`,
  `gmres-unsym-220 1.02x`, `minres-kkt-42 1.01x`
- `bench_eigs_reuse`: `growm-nos4-k5 1.01x`,
  `thick-bcsstk14-k5 0.95x`, `lobpcg-diag40-k3 1.00x`, with
  `|lambda|max diff = 0.000e+00`

## Non-Blocking Note

The reviewed CMake rebuild again emitted the ordinary
`bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
binary, but the reviewed path still completed cleanly and passed all parity
gates.

## Exit State

Sprint 62 Day 13 closes from a fully validated branch state:

- the strongest local reviewed baseline passed
- the direct-family lifecycle proof surfaces remained green
- the adjacent repeated-run solver surfaces remained green
- the representative examples and workflow benchmarks still support the landed
  Sprint 62 direct-usability story
