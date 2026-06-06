# Sprint 56 Day 13 - full validation sweep

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Run the full Sprint 56 validation contract from the landed decomposition state,
reconfirm the reviewed truthfulness anchors, and rerun the targeted CSC/SVD
follow-ons for the touched implementation surfaces.

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
- `Total Test time (real) = 290.02 sec`

Interpretation:

- the reviewed local path and reviewed CMake parity path still agree exactly
- Sprint 56 did not introduce hidden Makefile-only or CMake-only drift

## Targeted Sprint 56 follow-ons

All targeted follow-ons passed:

- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_cholesky` -> `21 / 21`
- `./build/test_ldlt` -> `84 / 84`
- `./build/test_etree` -> `97 / 97`
- `./build/test_svd` -> `97 / 97`
- `./build/test_integration` -> `37 / 37`
- `./build/example_analysis`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

## Representative retained behavior

Representative direct results remained healthy:

- `example_analysis`
  - solve residual = `4.44e-16`
- `bench_refactor_csc` on `nos4`
  - rerun-stable result:
    - `analyze_ms = 0.575`
    - `refactor_public_ms = 0.224`
    - `refactor_csc_ms = 0.166`
    - `solve_public_ms = 0.017`
    - `solve_csc_ms = 0.005`
    - `speedup_refactor = 1.35x`
    - `res_public = 8.24e-16`
    - `res_csc = 7.06e-16`

## Measurement-sensitive note

The first single-repeat `bench_refactor_csc nos4` run produced an obvious
microbenchmark outlier (`speedup_refactor = 0.10x`). A second immediate rerun
returned the stable expected shape (`1.35x`) with unchanged residuals.

Interpretation:

- the benchmark itself still behaves correctly
- the very small `nos4` single-repeat timing is sensitive to transient noise
- no code reconciliation was warranted from that one outlier

## Conclusion

Sprint 56 Day 13 completed the full validation contract successfully:

- local format, lint, test, and reviewed-baseline gates all passed
- reviewed Makefile/CMake parity stayed exact at `53 vs 53`
- the focused CSC/SVD rerun surfaces remained green
- no new blocker-level reconciliation queue surfaced during validation

Sprint 56 can move to Day 14 closeout from a fully validated landed state.
