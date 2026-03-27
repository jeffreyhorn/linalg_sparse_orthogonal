# Sprint 5 Benchmark Results

## SpMV Throughput (Serial)

| Matrix | n | nnz | Time/iter (s) | MFLOP/s |
|--------|--:|----:|-------------:|--------:|
| west0067 | 67 | 294 | 0.000001 | 752 |
| nos4 | 100 | 594 | 0.000002 | 773 |
| bcsstk04 | 132 | 3648 | 0.000020 | 363 |
| steam1 | 240 | 2248 | 0.000009 | 488 |
| fs_541_1 | 541 | 4282 | 0.000019 | 442 |
| orsirr_1 | 1030 | 6858 | 0.000028 | 488 |

Small matrices (west0067, nos4) show higher MFLOP/s due to cache effects.
Larger matrices (bcsstk04) show lower throughput due to irregular memory access
patterns in the orthogonal linked-list representation.

## Comprehensive Solver Comparison

### SPD Matrices

| Matrix | Solver | Iters | Time (s) | Residual | Conv |
|--------|--------|------:|---------:|---------:|------|
| nos4 (100×100) | CG | 92 | 0.000290 | 4.8e-11 | yes |
| | ILU-CG | 25 | 0.000159 | 3.7e-11 | yes |
| | Cholesky-CG | 1 | 0.000027 | 3.3e-15 | yes |
| | GMRES(50) | 233 | 0.002622 | 1.0e-10 | yes |
| | ILU-GMRES(50) | 26 | 0.000342 | 7.1e-12 | yes |
| | LU direct | — | 0.000563 | 2.5e-15 | yes |
| | Cholesky direct | — | 0.000161 | 3.3e-15 | yes |
| bcsstk04 (132×132) | CG | 556 | 0.014778 | 7.2e-11 | yes |
| | ILU-CG | 35 | 0.001998 | 3.4e-11 | yes |
| | Cholesky-CG | 1 | 0.000194 | 2.7e-16 | yes |
| | GMRES(50) | 2000 | 0.082462 | 2.9e-07 | no |
| | ILU-GMRES(50) | 26 | 0.001483 | 8.8e-06 | yes |
| | LU direct | — | 0.047520 | 2.5e-16 | yes |
| | Cholesky direct | — | 0.001451 | 2.0e-16 | yes |

### Unsymmetric Matrices

| Matrix | Solver | Iters | Time (s) | Residual | Conv |
|--------|--------|------:|---------:|---------:|------|
| west0067 (67×67) | GMRES(50) | 2000 | 0.015807 | 2.4e-01 | no |
| | ILU-GMRES(50) | — | — | (ILU fails: zero pivots) | — |
| | LU direct | — | 0.000664 | 2.1e-16 | yes |
| steam1 (240×240) | GMRES(50) | 2000 | 0.072551 | 5.7e-07 | no |
| | ILU-GMRES(50) | 2 | 0.000279 | 3.5e-06 | yes |
| | LU direct | — | 0.527226 | 2.5e-14 | yes |
| fs_541_1 (541×541) | GMRES(50) | 13 | 0.000780 | 1.9e-11 | yes |
| | ILU-GMRES(50) | 2 | 0.000225 | 1.1e-12 | yes |
| | LU direct | — | 0.004023 | 2.8e-15 | yes |
| orsirr_1 (1030×1030) | GMRES(50) | 2000 | 0.264920 | 2.7e-09 | no |
| | ILU-GMRES(50) | 38 | 0.006105 | 8.9e-08 | yes |
| | LU direct | — | 1.690728 | 1.4e-12 | yes |

## Convergence History

### nos4 (100×100 SPD) — CG

| Max Iters | CG Residual | ILU-CG Residual |
|----------:|------------:|----------------:|
| 1 | 4.889e-01 | 3.160e-01 |
| 2 | 3.988e-01 | 3.658e-01 |
| 5 | 4.017e-01 | 3.320e-02 |
| 10 | 3.236e-01 | 7.168e-03 |
| 20 | 4.701e-02 | 4.981e-07 |
| 50 | 6.261e-03 | 5.705e-15 |
| 100 | 4.883e-14 | 5.705e-15 |

ILU-CG reaches machine precision by 50 iterations; plain CG needs ~100.

### bcsstk04 (132×132 SPD) — CG

| Max Iters | CG Residual | ILU-CG Residual |
|----------:|------------:|----------------:|
| 1 | 2.263e-01 | 2.471e-01 |
| 5 | 1.691e-02 | 1.962e-03 |
| 10 | 9.792e-04 | 6.258e-04 |
| 20 | 3.771e-04 | 6.263e-05 |
| 50 | 6.322e-05 | 7.787e-16 |
| 100 | 4.491e-05 | — (converged) |
| 200 | 2.286e-06 | — |

ILU-CG reaches machine precision by 50 iterations. Plain CG is still at 2.3e-06
after 200 iterations due to bcsstk04's high condition number (condest ~5.6e6).

### orsirr_1 (1030×1030 unsymmetric) — GMRES

| Max Iters | GMRES(50) Residual | ILU-GMRES(50) Residual |
|----------:|-------------------:|-----------------------:|
| 1 | 6.161e-01 | 1.096e-01 |
| 5 | 1.583e-01 | 1.178e-02 |
| 10 | 6.492e-02 | 4.400e-03 |
| 20 | 9.368e-03 | 2.859e-05 |
| 50 | 3.895e-03 | 3.810e-10 |
| 100 | 1.968e-03 | 8.006e-13 |
| 500 | 8.951e-05 | — (converged) |
| 1000 | 7.137e-07 | — |

ILU-GMRES reaches 3.8e-10 by 50 iterations. Unpreconditioned GMRES(50) stalls
due to restart information loss, reaching only 7.1e-07 after 1000 iterations.

## Key Findings

### ILU(0) Preconditioning Effectiveness

| Matrix | Unpreconditioned | ILU-preconditioned | Speedup |
|--------|----------------:|------------------:|--------:|
| nos4 (CG) | 92 iters | 25 iters | 3.7× |
| bcsstk04 (CG) | 556 iters | 35 iters | 15.9× |
| steam1 (GMRES) | 2000 (no conv) | 2 iters | >1000× |
| fs_541_1 (GMRES) | 13 iters | 2 iters | 6.5× |
| orsirr_1 (GMRES) | 2000 (no conv) | 38 iters | >50× |
| west0067 (GMRES) | 2000 (no conv) | N/A (zero pivots) | — |

### Cholesky vs ILU as Preconditioner (SPD matrices)

| Matrix | ILU-CG iters | Cholesky-CG iters | Cholesky-CG residual |
|--------|-------------:|------------------:|--------------------:|
| nos4 | 25 | 1 | 3.3e-15 |
| bcsstk04 | 35 | 1 | 2.7e-16 |

Cholesky as exact preconditioner converges in 1 iteration (trivially), but requires
the full Cholesky factorization cost. ILU(0) is much cheaper to compute while still
providing significant iteration reduction.

### When to Use Each Solver

| Scenario | Recommended Solver |
|----------|--------------------|
| SPD, n < 200 | Cholesky direct (fastest overall) |
| SPD, n > 200 | ILU-CG (fast setup + few iterations) |
| SPD, exact solution needed | Cholesky direct |
| Unsymmetric, n < 100 | LU direct |
| Unsymmetric, n > 100, ILU works | ILU-GMRES (dramatic speedup over LU) |
| Unsymmetric, zero diagonal | LU direct (ILU(0) requires nonzero diagonal) |
| Multiple RHS, same matrix | Direct factorization (amortize factor cost) |
| Approximate solution OK | Iterative with loose tolerance |

### Parallel SpMV (OpenMP)

SpMV parallelization via `#pragma omp parallel for schedule(dynamic, 64)` is
available with `-DSPARSE_OPENMP`. Row-wise partitioning requires no synchronization
since each row writes to its own output element.

Expected behavior:
- Small matrices (n < 200): overhead dominates, serial is faster
- Large matrices (n > 1000): expect near-linear speedup for well-balanced rows
- The linked-list row traversal is inherently less cache-friendly than CSR,
  which limits absolute throughput compared to CSR-based SpMV implementations
