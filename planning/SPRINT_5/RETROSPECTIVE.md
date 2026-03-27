# Sprint 5 Retrospective

**Sprint Duration:** 14 days
**Goal:** Implement Krylov subspace iterative solvers (CG, GMRES) with ILU and Cholesky preconditioning, and add parallel SpMV.

---

## Definition of Done — Checklist

- [x] `sparse_solve_cg()` for SPD systems with optional preconditioner callback
- [x] `sparse_solve_gmres()` for general systems with restart and left preconditioning
- [x] `sparse_ilu_factor()` / `sparse_ilu_solve()` ILU(0) preconditioner
- [x] `sparse_ilu_precond()` callback compatible with `sparse_precond_fn`
- [x] Parallel SpMV with OpenMP (compile with `-DSPARSE_OPENMP`)
- [x] Convergence benchmarks: CG vs direct, GMRES vs LU, preconditioned vs unpreconditioned
- [x] All existing direct solver tests remain passing (backward compatible)
- [x] Updated algorithm documentation and README

**All items complete.**

---

## Sprint Items — Status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Conjugate Gradient (CG) solver | **DONE** | Standard preconditioned CG with relative residual convergence. Validated on nos4 (92 iters), bcsstk04 (556 iters). 34 tests. |
| 2 | GMRES solver | **DONE** | Restarted GMRES(k) with Arnoldi/Givens and left preconditioning. Lucky breakdown detection. Validated on all 6 SuiteSparse matrices. 28 tests. |
| 3 | Incomplete LU (ILU) preconditioner | **DONE** | IKJ variant ILU(0) with post-elimination diagonal check. 3.7-1000× iteration reduction. 18 tests. |
| 4 | Parallel SpMV | **DONE** | OpenMP row-wise parallelization with dynamic scheduling. Compile-time flag, zero overhead when disabled. 12 tests. |

---

## New API Surface

| Function | Header | Purpose |
|----------|--------|---------|
| `sparse_solve_cg()` | `sparse_iterative.h` | Preconditioned Conjugate Gradient for SPD systems |
| `sparse_solve_gmres()` | `sparse_iterative.h` | Restarted GMRES(k) for general systems |
| `sparse_ilu_factor()` | `sparse_ilu.h` | ILU(0) incomplete factorization |
| `sparse_ilu_solve()` | `sparse_ilu.h` | Apply ILU preconditioner (forward/backward sub) |
| `sparse_ilu_free()` | `sparse_ilu.h` | Free ILU factors |
| `sparse_ilu_precond()` | `sparse_ilu.h` | Preconditioner callback for iterative solvers |

New types: `sparse_iter_opts_t`, `sparse_gmres_opts_t`, `sparse_iter_result_t`, `sparse_precond_fn`, `sparse_ilu_t`.
New error code: `SPARSE_ERR_NOT_CONVERGED` (value 13).

---

## Iterative vs Direct Solver Comparison

### SPD Matrices

| Matrix | CG | ILU-CG | Cholesky-CG | Cholesky direct |
|--------|----|--------|-------------|----------------|
| nos4 (100×100) | 92 iters, 0.3 ms | 25 iters, 0.2 ms | 1 iter, 0.03 ms | 0.2 ms |
| bcsstk04 (132×132) | 556 iters, 15 ms | 35 iters, 2 ms | 1 iter, 0.2 ms | 1.5 ms |

### Unsymmetric Matrices

| Matrix | GMRES(50) | ILU-GMRES(50) | LU direct |
|--------|-----------|---------------|-----------|
| west0067 (67×67) | 2000 (no conv) | N/A (zero pivots) | 0.7 ms |
| steam1 (240×240) | 2000 (no conv) | 2 iters, 0.3 ms | 527 ms |
| fs_541_1 (541×541) | 13 iters, 0.8 ms | 2 iters, 0.2 ms | 4 ms |
| orsirr_1 (1030×1030) | 2000 (no conv) | 38 iters, 6 ms | 1,691 ms |

Key result: ILU-GMRES on orsirr_1 is **278× faster** than LU direct (6 ms vs 1.7 s).

---

## ILU(0) Preconditioning Effectiveness

| Matrix | Unpreconditioned | ILU-preconditioned | Speedup |
|--------|----------------:|------------------:|--------:|
| nos4 (CG) | 92 iters | 25 iters | 3.7× |
| bcsstk04 (CG) | 556 iters | 35 iters | 15.9× |
| steam1 (GMRES) | 2000 (no conv) | 2 iters | >1000× |
| fs_541_1 (GMRES) | 13 iters | 2 iters | 6.5× |
| orsirr_1 (GMRES) | 2000 (no conv) | 38 iters | >50× |

---

## Parallel SpMV

OpenMP parallelization with `#pragma omp parallel for schedule(dynamic, 64)`.
Row-wise partitioning — no synchronization needed.

- Compile-time flag: `-DSPARSE_OPENMP`
- macOS: Apple Clang + Homebrew libomp supported via `make omp`
- Linux: `make omp` works with default GCC
- Tested with 16 threads on all SuiteSparse matrices
- Serial throughput: 360-770 MFLOP/s depending on matrix structure

---

## Bugs Found During Sprint

| Bug | When Found | Fix |
|-----|-----------|-----|
| GMRES lucky breakdown: H(j+1,j) checked after Givens rotation zeroed it | Day 4 | Save breakdown flag before Givens rotation, check saved flag after |
| ILU(0) missing diagonal check: rows with no lower-triangular entries skip pivot check | Day 8 | Added post-elimination diagonal scan for all rows |
| west0067 has 65/67 zero diagonal entries: ILU(0) silently produced degenerate factors | Day 8 | Post-elimination diagonal check catches this; test updated to expect SPARSE_ERR_SINGULAR |

---

## Final Metrics

| Metric | Sprint 4 | Sprint 5 | Delta |
|--------|----------|----------|-------|
| Library source files | 7 (.c) + 1 internal header | 9 (.c) + 1 internal header | +2 |
| Public headers | 7 | 9 | +2 |
| Public API functions | 56 | 62 | +6 |
| Error codes | 13 | 14 | +1 |
| Test suites | 15 | 19 | +4 |
| Total unit tests | 305 | 406 | +101 |
| Total assertions | 2212 | 3841 | +1629 |
| Reference test matrices | 14 | 14 | — |
| Benchmark programs | 3 | 4 | +1 |
| Compiler warnings | 0 | 0 | — |
| UBSan violations | 0 | 0 | — |

---

## Lessons Learned

1. **GMRES requires careful Givens rotation bookkeeping.** The lucky breakdown detection must happen before the rotation zeroes H(j+1,j). This is a subtle ordering bug that only manifests when GMRES would otherwise take multiple steps — small test cases (where GMRES converges in 1-2 steps via breakdown) don't expose it.

2. **ILU(0) needs explicit diagonal checks.** The IKJ elimination loop only checks pivots when a row has lower-triangular entries. Rows without such entries (common in matrices with structurally zero diagonals) bypass the check entirely. A post-elimination diagonal scan is essential.

3. **Left preconditioning creates a residual gap.** The preconditioned residual ||M^{-1}(b-Ax)|| may converge while the true residual ||b-Ax|| remains relatively large, especially for ill-conditioned systems like steam1 (condest ~3e7). Tests must use relaxed tolerances on the true residual.

4. **West0067 is pathological for ILU(0).** With 65/67 zero diagonal entries, no ILU without pivoting can work. This highlights that ILU(0) is not universal — matrices requiring pivoting need either direct solvers or more sophisticated preconditioners (ILU with pivoting, ILUT).

5. **OpenMP on macOS requires special handling.** Apple Clang lacks `-fopenmp` but supports `-Xpreprocessor -fopenmp` with Homebrew's libomp. The `#pragma GCC diagnostic` suppression for libomp's `-Wpedantic` warnings keeps our build clean. Platform-conditional Makefile rules handle the differences transparently.

6. **Iterative solvers dominate for larger matrices.** The crossover point where ILU-GMRES beats LU direct is surprisingly low — around n=200 for steam1. For orsirr_1 (n=1030), the speedup is 278×. This validates the sprint's investment in iterative infrastructure.

---

## Deferred / Carried to Sprint 6

All Sprint 5 items completed. Potential Sprint 6 work:

1. **Sparse QR factorization** — Householder-based for least-squares problems
2. **Sparse SVD** — iterative (Lanczos/Arnoldi-based) for eigenvalue problems
3. **ILUT preconditioner** — ILU with threshold dropping (handles matrices like west0067 that need pivoting)
4. **Right preconditioning for GMRES** — alternative to left preconditioning where true residual is directly available
5. **Block solvers** — handle multiple RHS vectors simultaneously
