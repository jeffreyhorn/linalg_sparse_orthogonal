# Sprint 65 Day 9: Canonical Baseline Consolidation Batch

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Complete the canonical maintained performance surface by normalizing the
iterative and eigensolver reuse benchmarks and tightening the repo-level
classification story around the smaller Sprint 65 benchmark surface.

## Landed Scope

This batch intentionally stays limited to:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/README.md`
- `README.md`
- `docs/maintainer_guide.md`

It intentionally does not widen into:

- direct benchmark binaries already normalized on Day 8
- solver implementation files
- public headers
- build wiring
- CI runtime-lane changes
- the Day 10 efficiency target

## Output Consolidation Landed

The remaining two canonical maintained surfaces now also begin with:

- `benchmark`
- `category`
- `matrix`
- `scenario`

### `bench_iterative_reuse`

Stable interpretation:

- `benchmark = bench_iterative_reuse`
- `category = proof`
- `scenario = iter_handle_reuse`

The emitted rows now include stable repeated-run proof fields for:

- `solver`
- `n`
- `repeats`
- `one_shot_total_ms`
- `reuse_total_ms`
- `speedup`
- `one_shot_iters`
- `reuse_iters`
- `one_shot_relres`
- `reuse_relres`
- `one_shot_converged`
- `reuse_converged`
- `one_shot_status`
- `reuse_status`

### `bench_eigs_reuse`

Stable interpretation:

- `benchmark = bench_eigs_reuse`
- `category = proof`
- `scenario = eigs_handle_reuse`

The emitted rows now include stable repeated-run proof fields for:

- `backend`
- `n`
- `k`
- `repeats`
- `one_shot_median_ms`
- `reuse_median_ms`
- `speedup`
- `one_shot_iters`
- `reuse_iters`
- `one_shot_nconv`
- `reuse_nconv`
- `one_shot_relres`
- `reuse_relres`
- `one_shot_peak_basis`
- `reuse_peak_basis`
- `lambda_max_diff`
- `residual_diff`
- `backend_used`
- `one_shot_status`
- `reuse_status`

## Classification Consolidation

The benchmark story is now coherent across the maintained surfaces:

- canonical maintained performance surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- regression-sensitive runtime lane:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_reorder --skip-factor`
  - bounded adjacent lane:
    - `bench_amd_qg`
- exploratory or broader comparison lane:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - broader `bench_reorder`

## Ownership Split

The landed ownership model is now explicit:

- benchmark binaries own emitted fields and semantics
- `benchmarks/README.md` owns the benchmark-local schema explanation
- `README.md` owns only the compact top-level canonical-surface summary
- `docs/maintainer_guide.md` owns the authoritative category policy

## Validation

Because benchmark `*.c` files changed, the Day 9 validation gate was:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed. The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 571.75 sec`

## Retained Output Checks

The retained benchmark-proof spot checks were:

- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative normalized rows are now:

- `bench_iterative_reuse,proof,cg-tridiag-300,iter_handle_reuse,cg,300,400,112.1140,91.2320,1.23,17,17,5.192e-11,5.192e-11,1,1,OK,OK`
- `bench_iterative_reuse,proof,gmres-unsym-220,iter_handle_reuse,gmres,220,300,64.2980,60.8370,1.06,12,12,7.364e-11,7.364e-11,1,1,OK,OK`
- `bench_iterative_reuse,proof,minres-kkt-42,iter_handle_reuse,minres,42,250,20.7880,30.5190,0.68,39,39,3.870e-11,3.870e-11,1,1,OK,OK`
- `bench_eigs_reuse,proof,growm-nos4-k5,eigs_handle_reuse,lanczos_growm,100,5,40,3.9310,5.6480,0.70,115,115,5,5,4.326e-14,4.326e-14,100,100,0.000e+00,0.000e+00,lanczos_growm,OK,OK`
- `bench_eigs_reuse,proof,thick-bcsstk14-k5,eigs_handle_reuse,lanczos_thick_restart,1806,5,8,175.3030,153.8770,1.14,105,105,5,5,7.864e-14,7.864e-14,40,40,0.000e+00,0.000e+00,lanczos_thick_restart,OK,OK`
- `bench_eigs_reuse,proof,lobpcg-diag40-k3,eigs_handle_reuse,lobpcg,40,3,40,2.6230,2.4450,1.07,45,45,3,3,6.696e-11,6.696e-11,30,30,0.000e+00,0.000e+00,lobpcg,OK,OK`

## Day 9 Exit State

Sprint 65 now has:

- one fully normalized four-binary canonical maintained performance surface
- one explicit canonical / runtime / exploratory split across outputs and docs
- one compact top-level performance-governance story instead of a broad mixed
  benchmark catalog
- one fixed direct CSC/Cholesky efficiency target carried into Day 10
