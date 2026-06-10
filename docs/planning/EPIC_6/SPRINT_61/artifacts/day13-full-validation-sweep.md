# Sprint 61 Day 13: Full Validation Sweep

Date: 2026-06-09
Branch: sprint-61

## Purpose

Validate the full landed Sprint 61 Phase 1 configuration-modernization package
from the frozen Day 12 checklist so the sprint can close from one explicit
reviewed baseline instead of from a mix of intermediate validation points.

## Full Validation Gate

The full required gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- reviewed CMake total time from `make quality-review-full`:
  - `Total Test time (real) = 368.17 sec`

## Targeted Sprint 61 Follow-Ons

The fixed Day 12 follow-ons also all passed:

- `./build/test_integration` -> `39 / 39`
- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_graph` -> `60 / 60`
- `./build/test_graph_fm_buckets` -> `10 / 10`
- `./build/test_reorder_nd` -> `34 / 34`
- `./build/test_reorder_amd_qg` -> `7 / 7`
- `./build/test_iterative` -> `79 / 79`
- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

## Representative Retained Outputs

- `test_reorder_nd`
  - typed precedence/default cluster stayed fully green for:
    - supernodal postorder
    - root bisect
    - root-bisect max-n
    - coarsening
    - coarsen floor ratio
    - coarsest bisection
    - separator-lift strategy
    - separator-lift weight
    - internal CV-fallthrough default/compat behavior
- `test_graph`
  - multilevel graph/coarsen/bisect proof home stayed clean at `60 / 60`
- `test_chol_csc`
  - supernodal-postorder invariance checks stayed clean across `nos4`,
    `bcsstk04`, `bcsstk14`, and `s3rmt3m3`
- `example_analysis`
  - residual stayed `4.44e-16`
- `example_iterative`
  - GMRES: `25` iterations unpreconditioned
  - ILU(0)-GMRES: `9` iterations
- `example_ic_minres`
  - MINRES on KKT `42x42`: `39` iterations
  - Jacobi-MINRES: `26` iterations
- `example_eigs`
  - `nos4`: `5 / 5` pairs in `115` Lanczos iterations
  - KKT nearest-sigma case: `3 / 3` pairs in `6` Lanczos iterations
  - explicit `LOBPCG` on `bcsstk04`: `3 / 3` pairs in `62` outer iterations
    with reported residual `8.808e-09`
- `example_svd_lowrank`
  - sparse low-rank `k=2` kept `22 -> 6` nnz and `3.7x` compression
- `bench_refactor`
  - `tridiag-200 1.78x`
  - `tridiag-500 1.66x`
  - `bcsstk04 1.33x`
  - `nos4 1.53x`
- `bench_refactor_csc nos4`
  - `speedup_refactor = 1.52x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `bench_iterative_reuse`
  - `cg-tridiag-300 0.95x`
  - `gmres-unsym-220 1.12x`
  - `minres-kkt-42 0.94x`
- `bench_eigs_reuse`
  - `growm-nos4-k5 1.04x`
  - `thick-bcsstk14-k5 1.01x`
  - `lobpcg-diag40-k3 1.02x`
  - `|lambda|max diff = 0.000e+00`

## Day 13 Note

The reviewed CMake rebuild emitted ordinary compiler warnings while rebuilding
`bench_eigs_reuse`, but the reviewed path still completed cleanly and passed
all parity gates. No blocker-level validation drift surfaced.

## Day 13 Exit State

Sprint 61 now has a fully validated Phase 1 configuration-modernization
baseline:

- full reviewed local gate passed
- reviewed CMake parity stayed exact
- graph/reorder-sensitive proof surfaces all passed
- representative workflow examples and benchmark drivers all passed
- no new reconciliation queue surfaced during validation
