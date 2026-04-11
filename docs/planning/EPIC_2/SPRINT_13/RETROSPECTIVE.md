# Sprint 13 Retrospective: Incomplete Cholesky Preconditioner & MINRES Solver

**Sprint Duration:** 14 days
**Goal:** Add IC(0) preconditioning for SPD systems and MINRES solver for symmetric indefinite systems.

---

## Definition of Done Checklist

| Item | Status |
|------|--------|
| `sparse_ic_factor()` — IC(0) factorization (no-fill, left-looking) | Done |
| `sparse_ic_solve()` — forward/backward substitution with L and L^T | Done |
| `sparse_ic_precond()` — callback compatible with `sparse_precond_fn` | Done |
| `sparse_ic_free()` — safe cleanup (delegates to `sparse_ilu_free`) | Done |
| IC(0) validation: NULL, shape, symmetry, SPD checks | Done |
| IC(0) reuses `sparse_ilu_t` struct (L in L field, L^T in U field) | Done |
| `sparse_solve_minres()` — Lanczos-based MINRES with Givens QR | Done |
| `sparse_minres_solve_block()` — per-column block MINRES | Done |
| Preconditioned MINRES (M-inner product, SPD preconditioner) | Done |
| MINRES convergence control: tol, max_iter, zero RHS, lucky breakdown | Done |
| MINRES verbose mode | Done |
| IC(0) tested as CG preconditioner on SPD systems | Done |
| IC(0) vs ILU(0) comparison on SuiteSparse matrices | Done |
| MINRES tested on symmetric indefinite KKT systems | Done |
| MINRES vs GMRES equivalence on symmetric systems | Done |
| MINRES vs LDL^T direct solve comparison | Done |
| Block MINRES with per-column convergence tracking | Done |
| Extreme-scale tests (IC(0) and MINRES) | Done |
| Ill-conditioned and edge case tests | Done |
| README updated (features, API, test counts, project structure) | Done |
| Algorithm documentation updated (IC(0) and MINRES sections) | Done |
| Example program (`example_ic_minres.c`) | Done |
| Sprint 13 integration test suite | Done |
| ASan/UBSan clean | Done |
| CMake `ctest` all 35 tests pass | Done |
| `make format && make lint && make test` clean | Done |

---

## Final Metrics

### Test Counts

| Suite | Tests | Assertions |
|-------|------:|----------:|
| test_ic.c | 27 | 442 |
| test_minres.c | 43 | 700 |
| test_sprint13_integration.c | 14 | 1,068 |
| **Sprint 13 total** | **84** | **2,210** |
| **Full library** | **35 suites, 976 tests** | — |

### Source Lines

| File | Lines |
|------|------:|
| src/sparse_ic.c | 248 |
| include/sparse_ic.h | 114 |
| src/sparse_iterative.c (MINRES additions) | ~327 |
| include/sparse_iterative.h (MINRES declarations) | ~90 |
| tests/test_ic.c | 918 |
| tests/test_minres.c | 1,598 |
| tests/test_sprint13_integration.c | 909 |
| examples/example_ic_minres.c | 203 |
| **Sprint 13 total** | **~4,407** |

### IC(0) vs ILU(0) Benchmark (CG preconditioner on SPD matrices)

| Matrix | n | Unprec CG | IC(0)-CG | ILU(0)-CG | nnz(IC L) | nnz(ILU L+U) |
|--------|--:|----------:|---------:|----------:|----------:|--------------:|
| bcsstk04 | 132 | 653 iters | 39 iters | 39 iters | 1,890 | 3,780 |
| nos4 | 100 | 92 iters | 25 iters | 26 iters | — | — |
| banded (bw=3) | 50 | 10 iters | 1 iter | 1 iter | — | — |
| tridiag | 20 | 18 iters | 1 iter | — | — | — |

IC(0) produces identical or slightly better iteration counts compared to ILU(0) on SPD systems, with approximately half the storage (only L, not L+U).

### MINRES vs GMRES Benchmark (symmetric indefinite KKT systems)

| System | n | MINRES | GMRES | Jacobi-MINRES |
|--------|--:|-------:|------:|--------------:|
| KKT (15+6) | 21 | 21 iters | 21 iters | 17 iters |
| KKT (20+8) | 28 | 28 iters | 28 iters | — |
| KKT (35+15) | 50 | 39 iters | 39 iters | — |
| KKT (70+30) | 100 | 43 iters | 43 iters | 29 iters |
| KKT (200+80) | 280 | 45 iters | 45 iters | 37 iters |

MINRES matches GMRES iteration-for-iteration on symmetric systems, with O(n) storage per iteration vs O(n*k) for unrestarted GMRES.

### Cross-Solver Consistency

All tested configurations produce matching solutions (within 1e-6):
- SPD: CG = MINRES = GMRES (identical iteration counts)
- Indefinite: MINRES = GMRES = LDL^T direct
- Preconditioned: IC(0)-CG = IC(0)-MINRES on SPD

---

## What Went Well

1. **IC(0) implementation was straightforward.** The left-looking column-by-column algorithm worked correctly on first attempt. Reusing `sparse_ilu_t` for storage avoided defining a new struct.

2. **MINRES Givens rotation scheme worked after one sign fix.** The Lanczos + implicit QR algorithm is mathematically elegant — only 6-9 vectors of storage regardless of iteration count. One sign error in the phi_bar update was caught immediately by tests.

3. **Preconditioned MINRES integrated cleanly.** The M-inner product variant required only adding 3 extra workspace vectors and modifying the inner products. The `sparse_precond_fn` callback interface worked without changes.

4. **Excellent test coverage.** 84 tests with 2,210 assertions cover: entry validation, pattern correctness, solve accuracy, CG/MINRES preconditioning comparison, SuiteSparse validation, extreme scales, edge cases, block solve, and cross-solver consistency.

5. **Sprint 11-12 infrastructure paid dividends.** Norm-relative tolerance (`sparse_rel_tol`), the test framework (`REQUIRE_OK`), and LDL^T direct solve all composed naturally with the new features.

---

## Bugs Found and Fixed

1. **MINRES phi_bar sign error (Day 5):** The residual norm update `phi_bar = sn_new * phi_bar` should have been `phi_bar = -sn_new * phi_bar`. The positive sign caused the MINRES residual estimate to grow instead of decrease, leading to non-convergence. Fixed immediately — all subsequent tests passed.

2. **clang-tidy false positive on pattern array bounds (Day 2):** The static analyzer couldn't prove `pat_len < n` in the IC(0) column gather loop. Added an explicit `pat_len < n` guard to silence the warning without changing behavior.

3. **0x0 matrix handling (Day 2):** `sparse_create(0, 0)` returns NULL in this library, so the IC(0) factor function's n=0 early return needed to come before the symmetry check (which calls `sparse_is_symmetric` on a NULL pointer).

---

## Items Deferred

None. All 6 planned Sprint 13 items were completed:

1. IC(0) factorization — Done (Day 1-3)
2. IC(0) preconditioner callback — Done (Day 3-4)
3. MINRES solver — Done (Day 5-7)
4. Block MINRES — Done (Day 8-9)
5. IC(0) + MINRES integration — Done (Day 10-11)
6. Documentation and benchmarks — Done (Day 12)

---

## API Summary

```c
// IC(0) factorization and solve
sparse_ic_factor(A, &ic);            // IC(0): L*L^T ≈ A (SPD only)
sparse_ic_solve(&ic, r, z);          // solve L*L^T*z = r
sparse_ic_precond(&ic, n, r, z);     // preconditioner callback
sparse_ic_free(&ic);                 // cleanup

// MINRES solver
sparse_solve_minres(A, b, x, &opts, precond, ctx, &result);
sparse_minres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result);
```
