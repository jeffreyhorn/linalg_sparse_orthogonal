# Sprint 92 Day 13: Full Validation Sweep

## Purpose

Run the full frozen Sprint 92 validation queue from the live post-Day-12
branch and record the exact reviewed, proof-owner, example, and reporting
results.

## Main Result

The full Day 13 queue passed cleanly:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `./build/quality-review-cmake/test_dense`
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_ldlt_csc`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
- `SPARSE_LDLT_DENSE_BACKEND=external ./build/bench_refactor_csc --indefinite-kkt --repeat 1`
- `make bench-canonical-report`

## Reviewed Anchors

The reviewed baseline stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `326.70 sec`

## Focused Reruns

The focused proof-owner reruns also all passed:

- `test_dense` = `34 / 34`
- `test_chol_csc` = `152 / 152`
- `test_ldlt` = `88 / 88`
- `test_ldlt_csc` = `96 / 96`
- `test_qr` = `73 / 73`

The representative examples also passed:

- `example_analysis`:
  - solve residual = `4.44e-16`
- `example_basic_solve`:
  - residual `||b - Ax|| = 0.00e+00`

## Backend Observability Follow-Through

The retained repeated-run LDLT benchmark proof also stayed clean:

- default request:
  - `ldlt_dense_backend_request=builtin`
  - `ldlt_dense_backend_selected=builtin`
  - `ldlt_dense_backend_fallback=no`
  - `speedup_refactor=0.99`
- explicit external request:
  - `ldlt_dense_backend_request=external`
  - `ldlt_dense_backend_selected=accelerate`
  - `ldlt_dense_backend_fallback=no`
  - `speedup_refactor=1.59`

Both runs kept:

- `res_public = 2.96e-16`
- `res_csc = 2.96e-16`

## Reporting Follow-Through

Canonical reporting completed cleanly:

- `make bench-canonical-report` wrote:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`
  - `build/bench-reports/canonical/bench_chol_csc.csv`
  - `build/bench-reports/canonical/bench_iterative_reuse.csv`
  - `build/bench-reports/canonical/bench_eigs_reuse.csv`
  - `build/bench-reports/canonical/index.tsv`
  - `build/bench-reports/canonical/manifest.txt`

## Non-Blocking Runtime Note

The reviewed long pole remains:

- `test_reorder_nd` = `183.22 sec`
- reviewed total = `326.70 sec`

That remains a real later Epic 9 runtime/threading concern, but it is not a
Sprint 92 blocker because the Sprint 92 backend package stayed bounded to the
shared dense-kernel seam, LDLT backend convergence, and benchmark-side
observability.

## Exit State

- Sprint 92 now has one validated close baseline.
- The bounded portable backend package, LDLT adoption follow-through, and
  benchmark observability surfaces all passed from the same live branch state.
- Day 14 can now close Sprint 92 from this validated baseline.
