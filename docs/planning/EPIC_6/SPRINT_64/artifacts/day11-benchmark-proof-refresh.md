# Sprint 64 Day 11: Benchmark Proof Refresh

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Refresh the maintained benchmark proof surface for the landed Sprint 64
backend-aware Cholesky CSC lane so the benchmark output identifies:

- the maintained fallback CSC lane
- the accelerated supernodal lane
- the active dense-kernel descriptor backing the supernodal lane

without widening into broad benchmark-governance work.

## Landed Surfaces

Implementation / proof surface:

- `benchmarks/bench_chol_csc.c`

Benchmark-local docs:

- `benchmarks/README.md`

## Main Result

`bench_chol_csc` now reports three new path-identification CSV fields:

- `csc_scalar_path`
- `csc_supernodal_path`
- `csc_supernodal_dense_kernel`

On the current default build the maintained values are:

- `scalar`
- `supernodal`
- `builtin`

This means the benchmark surface now proves not just that the CSC supernodal
lane ran, but which dense-kernel descriptor backed that lane for the reported
numbers.

## Why This Was the Right Day 11 Slice

Before this batch, the benchmark already measured:

- linked-list timing
- CSC scalar timing
- CSC supernodal timing
- residuals for all three

The missing truthfulness signal was not another timing column. It was the
identity of the active dense-kernel descriptor behind the new Sprint 64
backend-aware seam.

So the Day 11 batch stayed bounded to output measurability:

- no kernel implementation changes
- no build-option changes
- no integration-test widening
- no benchmark-governance rewrite

## Representative Output

`./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`

```text
matrix,n,nnz,csc_scalar_path,csc_supernodal_path,csc_supernodal_dense_kernel,factor_ll_ms,factor_csc_ms,factor_csc_sn_ms,solve_ll_ms,solve_csc_ms,solve_csc_sn_ms,speedup_csc,speedup_csc_sn,res_ll,res_csc,res_csc_sn
nos4.mtx,100,594,scalar,supernodal,builtin,0.800,1.024,0.715,0.010,0.010,0.005,0.78,1.12,7.06e-16,5.89e-16,5.89e-16
```

`./build/bench_chol_csc tests/data/suitesparse/bcsstk04.mtx --repeat 1`

```text
matrix,n,nnz,csc_scalar_path,csc_supernodal_path,csc_supernodal_dense_kernel,factor_ll_ms,factor_csc_ms,factor_csc_sn_ms,solve_ll_ms,solve_csc_ms,solve_csc_sn_ms,speedup_csc,speedup_csc_sn,res_ll,res_csc,res_csc_sn
bcsstk04.mtx,132,3648,scalar,supernodal,builtin,4.375,4.144,4.347,0.047,0.023,0.018,1.06,1.01,6.05e-16,1.06e-15,9.08e-16
```

`./build/bench_chol_csc --small-corpus --repeat 1 | head -n 6`

```text
matrix,n,nnz,csc_scalar_path,csc_supernodal_path,csc_supernodal_dense_kernel,factor_ll_ms,factor_csc_ms,factor_csc_sn_ms,solve_ll_ms,solve_csc_ms,solve_csc_sn_ms,speedup_csc,speedup_csc_sn,res_ll,res_csc,res_csc_sn
tridiag-20,20,58,scalar,supernodal,builtin,0.077,0.070,0.039,0.003,0.001,0.001,1.10,1.97,2.96e-16,2.96e-16,2.96e-16
tridiag-40,40,118,scalar,supernodal,builtin,0.038,0.058,0.054,0.003,0.002,0.002,0.66,0.70,2.96e-16,2.96e-16,2.96e-16
tridiag-60,60,178,scalar,supernodal,builtin,0.059,0.080,0.079,0.003,0.003,0.003,0.74,0.75,2.96e-16,2.96e-16,2.96e-16
tridiag-80,80,238,scalar,supernodal,builtin,0.071,0.116,0.112,0.005,0.003,0.004,0.61,0.63,2.96e-16,2.96e-16,2.96e-16
banded-20,20,160,scalar,supernodal,builtin,0.051,0.063,0.071,0.002,0.001,0.001,0.81,0.72,4.08e-16,4.08e-16,4.08e-16
```

## Validation

Ran:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Exit State

Sprint 64 Day 11 now leaves a benchmark proof surface that:

- keeps fallback and accelerated CSC lanes in one comparable CSV row
- identifies the active dense-kernel descriptor directly
- stays inside the bounded Sprint 64 benchmark-refresh fence
