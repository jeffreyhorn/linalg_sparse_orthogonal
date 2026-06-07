# Sprint 57 Day 13 - full validation sweep

Date: 2026-06-06  
Branch: `sprint-57`

## Scope

Run the full Sprint 57 validation contract from the landed giant-test
refactor and lifecycle-regression state, then rerun the targeted proof and
benchmark surfaces locked by the Day 12 audit.

## Required validation gate

All required Day 13 gates passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Reviewed truthfulness anchors

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 202.24 sec`

Interpretation:

- the reviewed local path and reviewed CMake parity path still agree exactly
- Sprint 57 did not introduce hidden Makefile-only or CMake-only drift

## Targeted Sprint 57 follow-ons

All targeted follow-ons passed:

- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_svd` -> `97 / 97`
- `./build/test_iterative` -> `79 / 79`
- `./build/test_integration` -> `39 / 39`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Representative retained behavior

Representative direct results remained healthy:

- `example_analysis`
  - solve residual = `4.44e-16`
- `bench_refactor`
  - `tridiag-200` = `1.31x`
  - `tridiag-500` = `1.30x`
  - `bcsstk04` = `1.16x`
  - `nos4` = `1.40x`
- `bench_refactor_csc` on `nos4`
  - `speedup_refactor = 2.20x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `bench_iterative_reuse`
  - `cg-tridiag-300` = `0.51x`
  - `gmres-unsym-220` = `0.99x`
  - `minres-kkt-42` = `1.04x`
- `bench_eigs_reuse`
  - `growm-nos4-k5` = `1.04x`
  - `thick-bcsstk14-k5` = `0.94x`
  - `lobpcg-diag40-k3` = `1.00x`
  - `|lambda|max diff = 0.000e+00` across all repeated-run cases

## Conclusion

Sprint 57 Day 13 completed the full validation contract successfully:

- local format, lint, test, and reviewed-baseline gates all passed
- reviewed Makefile/CMake parity stayed exact at `53 vs 53`
- the focused direct, iterative, eigensolver, and repeated-run rerun surfaces
  remained green
- no new blocker-level reconciliation queue surfaced during validation

Sprint 57 is ready for Day 14 closeout from a fully validated landed state.
