# Sprint 85 Day 13: Full Validation Sweep

## Purpose

Run the full Sprint 85 validation queue fixed on Day 12 and capture the
measured close baseline from actual execution.

## Main Result

The full Day 13 queue passed cleanly:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `404.15 sec`

## Focused Reviewed Reruns

The focused reviewed proof owners all passed:

- `test_iterative` = `80 / 80`
- `test_chol_csc` = `151 / 151`
- `test_integration` = `56 / 56`
- `test_ldlt` = `87 / 87`
- `test_qr` = `73 / 73`

Representative retained proof outputs:

- `test_iterative`:
  - `nos4`: CG `92` iterations, relres = `4.830e-11`
  - `bcsstk04`: CG `556` iterations, relres = `7.200e-11`
- `test_chol_csc`:
  - `bcsstk14`: rel_residual = `1.080e-15`
  - external dense ref `nos4`: max `|x-x_ref| = 4.690e-13`
- `test_integration`:
  - all `56 / 56` shared public lifecycle tests passed
- `test_ldlt`:
  - all `87 / 87` passed with retained KKT, refinement, and backend coverage
- `test_qr`:
  - `nos4` QR solve true residual = `9.415e-15`

## Examples and Benchmark Follow-Ons

Representative examples passed:

- `example_analysis`:
  - solve residual = `4.44e-16`
- `example_basic_solve`:
  - residual `||b - Ax|| = 0.00e+00`

Representative benchmark/reporting follow-ons passed:

- `bench_svd tests/data/suitesparse/nos4.mtx`:
  - `Partial/Full = 3.4x speedup`
- `bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`:
  - `speedup_refactor = 1.45`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `make bench-canonical-report` wrote:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`

## Runtime Note

One non-blocking runtime note remains explicit:

- reviewed CMake `test_reorder_nd` remained the long tail at `283.53 sec`
  out of `404.15 sec`

The full reviewed path still completed cleanly, so this remains a closeout
runtime note rather than a Sprint 85 blocker.

## Exit State

- Sprint 85 now has a measured validated close baseline.
- The reviewed anchors stayed exact across the full sweep.
- Day 14 can close from execution evidence rather than implementation state.
