# Sprint 73 Day 13: Full Validation Sweep

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Validate the landed Sprint 73 branch from the strongest reviewed baseline and
the touched configuration surfaces before closeout.

## Standard Gate

The standard validation gate passed:

- `make format`
- `make lint`
- `make test`

The strongest reviewed baseline also passed:

- `make quality-review-full`

## Reviewed Anchors

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 421.78 sec`

## Focused Sprint 73 Follow-Ons

The touched Sprint 73 follow-ons also passed:

- `./build/quality-review-cmake/test_graph` -> `61 / 61`
- `./build/quality-review-cmake/test_graph_fm_buckets` -> `10 / 10`
- `./build/quality-review-cmake/test_reorder_nd` -> `35 / 35`
- `./build/quality-review-cmake/test_integration` -> `48 / 48`
- `./build/quality-review-cmake/test_fuzz` -> `25 / 25`
- `make examples-build`
- `./build/example_analysis`
- `./build/example_basic_solve`
- `./build/quality-review-cmake/bench_reorder`
- `./build/quality-review-cmake/bench_amd_qg`
- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

## Representative Retained Outputs

- `example_analysis`
  - solve residual `4.44e-16`
- `example_basic_solve`
  - residual `0.00e+00`
- `test_fuzz`
  - `large-n CSC lifecycle property: 3/3 passed`
- `test_graph`
  - `bcsstk14 under SPARSE_ND_COARSENING=hcc: sep=97`
  - `Pres_Poisson.mtx (n=14822): sep=216`
- `test_reorder_nd`
  - `Pres_Poisson ND/AMD = 0.923`
  - `bcsstk14 ND/AMD = 1.124`
- `bench_reorder`
  - `nos4 nd nnz(L) = 637`
  - `bcsstk04 nd nnz(L) = 3722`
  - `bcsstk14 nd nnz(L) = 130422`
  - `s3rmt3m3 nd nnz(L) = 487832`
- `bench_amd_qg`
  - `Kuu: qg 1252.7 ms vs bitset 3195.9 ms`
  - `Pres_Poisson: qg 22084.6 ms vs bitset 30078.4 ms`
  - `banded_10000: qg 557.0 ms vs bitset 4753.8 ms`
- install/package regressions
  - installed `pkg-config` version `2.2.0`

## Non-Blocking Runtime Note

Reviewed CMake `test_reorder_nd` still dominated the total runtime at
`294.22 sec` out of the `421.78 sec` total, but the full reviewed path
completed cleanly and all parity anchors stayed exact.

## Exit State

Sprint 73 Day 13 closes with:

1. the standard gate passing
2. the strongest reviewed baseline passing
3. touched graph/reorder proof owners revalidated explicitly
4. retained example, benchmark, and install/package anchors captured for
   closeout
