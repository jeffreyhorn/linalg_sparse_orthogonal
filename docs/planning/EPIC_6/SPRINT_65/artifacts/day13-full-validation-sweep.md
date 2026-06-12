# Sprint 65 Day 13: Full Validation Sweep

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Run the full Sprint 65 validation gate from the landed benchmark-governance and
solver-efficiency branch state, then rerun the targeted proof surfaces and
capture the retained benchmark and example evidence.

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
- `Total Test time (real) = 784.97 sec`

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
- `example_ldlt` relative residual = `1.555e-16`
- `example_svd_lowrank` sparse low-rank `k=2` kept `22 -> 6` nnz for `3.7x`
  compression

Representative retained benchmark outputs:

- `bench_refactor`:
  - `tridiag-200 1.63x`
  - `tridiag-500 1.68x`
  - `bcsstk04 1.44x`
  - `nos4 1.58x`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 0.85x`
  - residuals `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4`:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
  - `speedup_csc = 1.54x`
  - `speedup_csc_sn = 0.24x`
- `bench_ldlt_csc nos4`:
  - `speedup_csc_native = 1.45x`
- `bench_iterative_reuse`:
  - `cg-tridiag-300 1.17x`
  - `gmres-unsym-220 1.02x`
  - `minres-kkt-42 1.40x`
- `bench_eigs_reuse`:
  - `growm-nos4-k5 1.08x`
  - `thick-bcsstk14-k5 1.07x`
  - `lobpcg-diag40-k3 1.01x`
  - `|lambda|max diff = 0.000e+00`

## Non-Blocking Note

The reviewed CMake path was still dominated by the existing reorder stress
tail:

- `test_reorder_nd` consumed `574.47 sec` of the `784.97 sec` reviewed CMake
  `ctest` wall time

That remains non-blocking:

- the full reviewed path completed cleanly
- the Makefile/CMake parity anchors stayed exact
- all targeted Sprint 65 proof reruns passed

## Exit State

Sprint 65 Day 13 closes with a validated benchmark-governance and
solver-efficiency baseline:

- the full required validation gate passed
- the reviewed parity anchors remained exact
- the normalized canonical benchmark surface still emits coherent retained rows
- the targeted direct/CSC/example/benchmark proof surfaces still match the
  landed Sprint 65 story
- the branch is ready for Day 14 closeout from a fully validated state
