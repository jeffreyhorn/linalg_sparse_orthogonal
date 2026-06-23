# Sprint 84 Day 13 - Full Validation Sweep

Date: 2026-06-23  
Branch: sprint-84

## Purpose

Run the full Sprint 84 validation queue fixed on Day 12 and capture the
retained closeout baseline from measured evidence rather than partial
implementation state.

## Validation Summary

The full Sprint 84 implementation-day gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 477.50 sec`

## Focused Proof-Owner Follow-Ons

The Day 12 focused reruns also all passed:

- `./build/quality-review-cmake/test_chol_csc` -> `151 / 151`
- `./build/quality-review-cmake/test_ldlt` -> `87 / 87`
- `./build/quality-review-cmake/test_fuzz` -> `28 / 28`
- `./build/quality-review-cmake/test_integration` -> `56 / 56`
- `./build/quality-review-cmake/test_iterative` -> `80 / 80`
- `./build/quality-review-cmake/test_eigs` -> `31 / 31`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `make bench-canonical-report`

Representative retained outputs stayed clean:

- `test_chol_csc` retained:
  - `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
  - `external dense ref tests/data/suitesparse/nos4.mtx: n=100, max|x-x_ref|=4.690e-13, rel_residual=3.907e-15`
  - `external dense ref tests/data/suitesparse/bcsstk04.mtx: n=132, max|x-x_ref|=3.224e-11, rel_residual=3.010e-16`
- `test_ldlt` retained:
  - `KKT 500x500: relres=4.465e-17, nnz(L)=1298`
- `test_fuzz` retained:
  - `28 / 28`
  - `20544` assertions
  - `large-n CSC lifecycle property: 3/3 passed`
  - `large-n LDLT reorder/repeat property: 3/3 passed`
- `test_integration` retained `56 / 56`
- `test_iterative` retained:
  - `nos4: CG iters=92, rel_res=4.830e-11`
  - `west0067: GMRES(67) iters=67, rel_res=4.036e-16`
- `test_eigs` retained:
  - `refined max ||A v - lambda v|| / |lambda| = 9.861e-16`
  - `LOBPCG refined max ||A v - lambda v|| / |lambda| = 3.850e-24`
- `example_analysis` retained solve residual `4.44e-16`
- `example_basic_solve` retained residual `0.00e+00`
- `bench_svd nos4` retained:
  - `Full SVD (σ only): 9.224 ms`
  - `Partial SVD (k=5, σ): 3.277 ms`
  - `Partial/Full: 2.8x speedup`
- `bench_refactor_csc nos4` retained:
  - `speedup_refactor = 0.99`
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

That remains the correct Sprint 84 reading because this sprint did not move
package, install, export, or runtime-package mechanics.

## Runtime Note

One non-blocking runtime note is explicit in the measured baseline:

- reviewed CMake `test_reorder_nd` still dominated runtime at `344.21 sec` out
  of the `477.50 sec` total

The full reviewed path still completed cleanly, and all parity anchors stayed
exact.

## Exit State

- Sprint 84 now has one measured Day 13 close baseline.
- The bounded direct-family external differential, seeded-property, and
  failure-path lifecycle assurance lanes passed together with retained
  iterative/eigensolver proof owners.
- Day 14 can close from validated evidence rather than from intermediate
  implementation state.
