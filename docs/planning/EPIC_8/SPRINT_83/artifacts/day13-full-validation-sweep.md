# Sprint 83 Day 13 - Full Validation Sweep

Date: 2026-06-22  
Branch: sprint-83

## Purpose

Run the full Sprint 83 validation queue fixed on Day 12 and capture the
retained closeout baseline from measured evidence rather than partial
implementation state.

## Validation Summary

The full Sprint 83 implementation-day gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 446.47 sec`

## Focused Proof-Owner Follow-Ons

The Day 12 focused reruns also all passed:

- `./build/quality-review-cmake/test_sparse_matrix` -> `59 / 59`
- `./build/quality-review-cmake/test_qr` -> `73 / 73`
- `./build/quality-review-cmake/test_svd` -> `97 / 97`
- `./build/quality-review-cmake/test_chol_csc` -> `149 / 149`
- `./build/quality-review-cmake/test_ldlt` -> `87 / 87`
- `./build/quality-review-cmake/test_integration` -> `53 / 53`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `make bench-canonical-report`

Representative retained outputs stayed clean:

- `test_sparse_matrix` retained:
  - `test_matrix_public_scalar_alias`
  - `test_idx_width_contract`
- `test_qr` retained:
  - `test_qr_public_scalar_alias`
  - `nos4 QR solve: rank=100`
  - `nos4 QR solve: res_norm=0.000e+00, true_res=9.415e-15`
- `test_svd` retained:
  - `outer-product vs dense: ||A_off - A_on||_F / ||A_off||_F = 0.000e+00`
  - `full-mode recon: ||A - U Sigma Vt||_F / ||A||_F = 9.648e-16`
- `test_chol_csc` retained:
  - `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
- `test_ldlt` retained:
  - `test_ldlt_dense_backend_accelerate_accepts_noperm_2x2`
  - `KKT 500x500: relres=4.465e-17, nnz(L)=1298`
- `test_integration` retained `53 / 53`
- `example_analysis` retained solve residual `4.44e-16`
- `example_basic_solve` retained residual `0.00e+00`
- `bench_svd nos4` retained:
  - `Full SVD (σ only): 6.612 ms`
  - `Partial SVD (k=5, σ): 2.170 ms`
  - `Partial/Full: 3.0x speedup`
- `bench_refactor_csc nos4` retained:
  - `speedup_refactor = 1.40`
  - residuals `8.24e-16` / `7.06e-16`
- `make bench-canonical-report` retained the canonical bundle write:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`

## Explicit Non-Rerun Surface

Install/export proof was intentionally not rerun on Day 13:

- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

That remains the correct Sprint 83 reading because this sprint did not move
package, install, export, or runtime-package mechanics.

## Runtime Note

One non-blocking runtime note is explicit in the measured baseline:

- reviewed CMake `test_reorder_nd` still dominated runtime at `314.43 sec` out
  of the `446.47 sec` total

The full reviewed path still completed cleanly, and all parity anchors stayed
exact.

## Exit State

- Sprint 83 now has one measured Day 13 close baseline.
- The widened shared-owner and bounded QR public-header surfaces passed
  together with retained deferred-family and direct-family proof.
- Day 14 can close from validated evidence rather than from intermediate
  implementation state.
