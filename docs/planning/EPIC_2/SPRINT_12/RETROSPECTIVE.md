# Sprint 12 Retrospective: Sparse LDL^T Factorization

**Sprint Duration:** 14 days
**Goal:** Add sparse LDL^T factorization with Bunch-Kaufman symmetric pivoting for symmetric indefinite systems.

---

## Definition of Done Checklist

| Item | Status |
|------|--------|
| `sparse_ldlt_factor()` with Bunch-Kaufman 1x1/2x2 pivoting | Done |
| `sparse_ldlt_factor_opts()` with AMD/RCM reordering | Done |
| `sparse_ldlt_solve()` with 5-phase permuted solve | Done |
| `sparse_ldlt_inertia()` eigenvalue sign counting | Done |
| `sparse_ldlt_refine()` iterative refinement | Done |
| `sparse_ldlt_condest()` Hager/Higham condition estimation | Done |
| `sparse_ldlt_free()` safe cleanup | Done |
| Bunch-Kaufman 2x2 pivot selection (4 criteria) | Done |
| Symmetric row/column interchange | Done |
| Block-relative singularity detection | Done |
| Element growth control | Done |
| KKT/saddle-point system tests | Done |
| SuiteSparse validation (bcsstk04, nos4) | Done |
| LDL^T vs LU fill-in comparison | Done |
| LDL^T vs Cholesky equivalence on SPD | Done |
| Extreme-scale tolerance tests (1e-35 to 1e+35) | Done |
| README updated | Done |
| Example program (`example_ldlt.c`) | Done |
| Sprint 12 integration test suite | Done |
| ASan clean | Done |
| UBSan clean | Done |
| `make format && make test` clean | Done |

---

## Final Metrics

### Test Counts

| Suite | Tests | Assertions |
|-------|------:|----------:|
| test_ldlt.c | 72 | 708 |
| test_sprint12_integration.c | 8 | 121 |
| **Sprint 12 total** | **80** | **829** |
| **Full library** | **32 suites** | — |

### Source Lines

| File | Lines |
|------|------:|
| src/sparse_ldlt.c | 1,084 |
| include/sparse_ldlt.h | 215 |
| tests/test_ldlt.c | 2,372 |
| tests/test_sprint12_integration.c | 410 |
| examples/example_ldlt.c | 186 |
| **Sprint 12 total** | **4,267** |

### Fill-In Comparison (LDL^T vs LU)

| Matrix | n | nnz(L_LDL^T) | nnz(LU) | Ratio |
|--------|--:|-------------:|--------:|------:|
| nos4 | 100 | 805 | 1,510 | 0.53x |
| bcsstk04 | 132 | 3,665 | 8,581 | 0.43x |
| arrow 30x30 | 30 | 59 (AMD) | — | 7.9x reduction vs no-AMD |

### SuiteSparse Validation

| Matrix | n | Relative Residual |
|--------|--:|------------------:|
| nos4 (no reorder) | 100 | 1.0e-15 |
| nos4 (AMD) | 100 | 8.4e-16 |
| bcsstk04 (no reorder) | 132 | 6.0e-09 |
| bcsstk04 (AMD) | 132 | within 1e-4 of Cholesky |

### KKT System Performance

| Size | Inertia | Relative Residual | nnz(L) |
|-----:|--------:|------------------:|-------:|
| 6x6 | (4,2,0) | 1.6e-16 | 17 |
| 20x20 | (14,6,0) | < 1e-10 | — |
| 50x50 | (35,15,0) | 1.2e-16 | — |
| 100x100 | (70,30,0) | 9.5e-17 | 258 |
| 280x280 | (200,80,0) | 6.2e-17 | 718 |
| 500x500 | (350,150,0) | 4.6e-17 | 1,298 |

---

## What Went Well

1. **Bunch-Kaufman pivoting worked on first implementation.** The 4-criterion pivot selection with symmetric row/column interchange and 2x2 block elimination produced correct results immediately on the standard test cases.

2. **Tolerance infrastructure from Sprint 11 paid off.** The `sparse_rel_tol()` function and norm-relative tolerance strategy meant extreme-scale matrices (1e-35 to 1e+35) worked without any special-casing.

3. **Test-driven development was effective.** Writing tests for each day's work caught a drop-tolerance bug (Day 9) before it could propagate.

4. **Fill-in reduction matches theory.** LDL^T uses ~50% less fill than LU on symmetric matrices (0.43x-0.53x), confirming the symmetry exploitation works correctly.

5. **Clean integration with existing infrastructure.** AMD/RCM reordering, Matrix Market I/O, and the test framework all composed naturally with the new factorization.

---

## Bugs Found and Fixed

1. **L-entry drop tolerance (Day 9):** The threshold `DROP_TOL * |dk|` for dropping L entries was incorrect — it effectively used `DROP_TOL * dk^2` as the threshold for `col_acc[i]`, which dropped all L entries for large-scale matrices. Fixed to use `DROP_TOL` directly since L entries are already divided by dk.

---

## Items Deferred

None. All planned Sprint 12 items were completed.

---

## API Summary

```c
// Factor
sparse_ldlt_factor(A, &ldlt);
sparse_ldlt_factor_opts(A, &opts, &ldlt);  // with AMD/RCM

// Solve
sparse_ldlt_solve(&ldlt, b, x);

// Analysis
sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero);
sparse_ldlt_condest(A, &ldlt, &cond);

// Refinement
sparse_ldlt_refine(A, &ldlt, b, x, max_iters, tol);

// Cleanup
sparse_ldlt_free(&ldlt);
```
