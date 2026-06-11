# Sprint 64 Day 13: Full Validation Sweep

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Run the full Sprint 64 validation gate from the landed backend-aware branch
state, then rerun the targeted direct/CSC/example/benchmark proof surfaces and
capture the retained evidence.

## Validation Gate

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors retained exactly:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 574.42 sec`

## Targeted Proof Reruns

Targeted proof binaries passed:

- `test_integration` = `47 / 47`
- `test_chol_csc` = `144 / 144`
- `test_ldlt_csc` = `96 / 96`
- `test_cholesky` = `21 / 21`
- `test_ldlt` = `84 / 84`
- `test_sparse_lu` = `37 / 37`
- `test_qr` = `72 / 72`
- `test_svd` = `97 / 97`

Representative retained example outputs:

- `example_analysis` residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `example_ldlt` refinement residual after solve = `0.000e+00`
- `example_svd_lowrank` sparse low-rank `k=2` kept `22 -> 6` nnz for `3.7x`
  compression

Representative retained benchmark outputs:

- `bench_refactor`:
  - `tridiag-200 1.78x`
  - `tridiag-500 1.34x`
  - `bcsstk04 1.34x`
  - `nos4 1.66x`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.63x`
  - residuals `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4`:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
- `bench_chol_csc bcsstk04`:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
  - `speedup_csc = 1.20x`
  - `speedup_csc_sn = 1.17x`
- `bench_ldlt_csc nos4`:
  - `speedup_csc_native = 1.60x`
- `bench_iterative_reuse`:
  - `cg-tridiag-300 1.07x`
  - `gmres-unsym-220 1.03x`
  - `minres-kkt-42 1.00x`
- `bench_eigs_reuse`:
  - `growm-nos4-k5 1.05x`
  - `thick-bcsstk14-k5 1.00x`
  - `lobpcg-diag40-k3 1.04x`
  - `|lambda|max diff = 0.000e+00`

## Non-Blocking Note

The reviewed CMake rebuild again emitted the existing
`bench_eigs_reuse.c` double-promotion warnings while rebuilding
`bench_eigs_reuse`.

That warning remains non-blocking:

- the full reviewed path completed cleanly
- the Makefile/CMake parity anchors stayed exact
- all targeted Sprint 64 proof reruns passed

## Exit State

Sprint 64 Day 13 closes with a validated backend-aware baseline:

- the full required validation gate passed
- the reviewed parity anchors remained exact
- the targeted direct/CSC/example/benchmark proof surfaces still match the
  landed Sprint 64 story
- the branch is ready for Day 14 closeout from a fully validated state
