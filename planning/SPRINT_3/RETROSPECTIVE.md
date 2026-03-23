# Sprint 3 Retrospective

**Sprint Duration:** 14 days
**Goal:** Add condition number estimation for diagnostics and implement fill-reducing reordering (AMD/RCM) to improve factorization performance on larger matrices.

---

## Definition of Done — Checklist

- [x] `sparse_lu_condest()` returns 1-norm condition estimate from existing LU factors
- [x] Transpose solve (`sparse_lu_solve_transpose()`) for Hager's algorithm
- [x] Solver can warn on ill-conditioned systems (condest in benchmark output, usage example in header)
- [x] `sparse_reorder_rcm()` — Reverse Cuthill-McKee ordering
- [x] `sparse_reorder_amd()` — Approximate Minimum Degree ordering
- [x] `sparse_permute()` — apply row/column permutation
- [x] `sparse_bandwidth()` — compute matrix bandwidth
- [x] `sparse_lu_opts_t` options struct with `sparse_lu_factor_opts()`
- [x] Factorization pipeline: reorder → factor → solve → auto-unpermute
- [x] Benchmark data showing fill-in reduction on SuiteSparse matrices
- [x] All existing tests continue to pass; new tests for all new functionality
- [x] Updated algorithm documentation and README

**All items complete.**

---

## Sprint Items — Status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Condition number estimation | **DONE** | Hager/Higham 1-norm estimator, transpose solve, 11 tests |
| 2 | Fill-reducing reordering | **DONE** | RCM (BFS + pseudo-peripheral), AMD (bitset adjacency), permute, bandwidth, opts integration, 27 tests |

---

## New API Surface

| Function | Header | Purpose |
|----------|--------|---------|
| `sparse_lu_condest()` | `sparse_lu.h` | 1-norm condition number estimate |
| `sparse_lu_solve_transpose()` | `sparse_lu.h` | Solve A^T*x = b using LU factors |
| `sparse_lu_factor_opts()` | `sparse_lu.h` | Factor with reordering options |
| `sparse_reorder_rcm()` | `sparse_reorder.h` | Reverse Cuthill-McKee ordering |
| `sparse_reorder_amd()` | `sparse_reorder.h` | Approximate Minimum Degree ordering |
| `sparse_permute()` | `sparse_reorder.h` | Apply row/column permutation |
| `sparse_bandwidth()` | `sparse_reorder.h` | Compute matrix bandwidth |

New types: `sparse_lu_opts_t` (options struct), `sparse_reorder_t` (enum: NONE/RCM/AMD).
New error code: `SPARSE_ERR_BADARG` (value 11).

---

## Condition Number Results (SuiteSparse, partial pivoting)

| Matrix | n | condest | Residual | Assessment |
|--------|--:|--------:|---------:|------------|
| west0067 | 67 | 3.0e+02 | 3.3e-15 | Well-conditioned |
| nos4 | 100 | 7.9e+02 | 1.6e-16 | Well-conditioned |
| bcsstk04 | 132 | 5.6e+06 | 5.6e-09 | Moderately ill-conditioned |
| steam1 | 240 | 3.0e+07 | 6.1e-07 | Ill-conditioned |

Condest correlates with observed residual norms: higher condition numbers produce larger residuals, validating the estimator.

---

## Reordering Effectiveness (partial pivoting)

### Fill-in Comparison

| Matrix | Natural | RCM | AMD | Best |
|--------|--------:|----:|----:|------|
| west0067 | 928 | 982 (+6%) | **819 (-12%)** | AMD |
| nos4 | 1510 | 1676 (+11%) | **1174 (-22%)** | AMD |
| bcsstk04 | **8581** | 8701 (+1%) | 9153 (+7%) | None |
| steam1 | 22956 | **15340 (-33%)** | 15402 (-33%) | RCM ≈ AMD |
| fs_541_1 | **7401** | 8875 (+20%) | 7466 (+1%) | None |
| orsirr_1 | 78069 | 96159 (+23%) | **55212 (-29%)** | AMD |

### Factorization Speedup

| Matrix | None (ms) | RCM (ms) | AMD (ms) | Best speedup |
|--------|----------:|---------:|---------:|-------------|
| steam1 | 385.6 | **70.2** | 152.0 | RCM: 5.5x |
| orsirr_1 | 1153.6 | 1240.4 | **886.1** | AMD: 1.3x |

---

## Bugs Found During Sprint

| Bug | When Found | Fix |
|-----|-----------|-----|
| RCM pseudo-peripheral heuristic could leave stale visited markers across components | Day 7 | Reset visited array between components, use explicit placed-node tracking |
| `sparse_create(0, 0)` returns NULL, causing reorder edge-case test to fail | Day 13 | Updated test to expect `SPARSE_ERR_NULL` (correct library behavior) |

---

## Final Metrics

| Metric | Sprint 2 | Sprint 3 | Delta |
|--------|----------|----------|-------|
| Library source files | 4 (.c) + 1 internal header | 5 (.c) + 1 internal header | +1 |
| Public headers | 4 | 5 | +1 |
| Public API functions | 30 | 38 | +8 |
| Error codes | 11 | 12 | +1 |
| Test suites | 9 | 10 | +1 |
| Total unit tests | 192 | 242 | +50 |
| Total assertions | 962 | 1255 | +293 |
| Reference test matrices | 14 | 14 | — |
| Benchmark programs | 3 | 3 (enhanced) | — |
| Compiler warnings | 0 | 0 | — |
| UBSan violations | 0 | 0 | — |

---

## Lessons Learned

1. **Bitset adjacency is practical for AMD at our scale.** For n ≤ 1030, the bitset approach (n² / 64 bytes ≈ 130KB) is simple and fast. A quotient graph would be needed for n > 10K, but that's beyond our current target.

2. **Neither reordering universally dominates.** AMD is best for unstructured matrices (orsirr_1, nos4), RCM is best for banded/thermal matrices (steam1), and some matrices (bcsstk04, fs_541_1) are best with no reordering. The recommendation must be matrix-dependent.

3. **Transparent reorder-solve integration is worth the complexity.** Storing `reorder_perm` in the matrix and auto-permuting in `sparse_lu_solve()` means callers don't need to manually manage permutation vectors. This required changes to the struct and solve path, but the API is much cleaner.

4. **Condition estimation validates residual expectations.** The condest values for SuiteSparse matrices directly explain the residual magnitudes: steam1's condition number of 3e7 explains its 6e-7 residual (roughly κ × machine epsilon).

5. **Transpose solve via column lists works cleanly.** The orthogonal linked-list structure makes transpose operations natural — forward-sub with U^T just walks column lists instead of row lists. No transposition of the matrix is needed.

---

## Deferred / Carried to Sprint 4

All Sprint 3 items completed. Potential Sprint 4 topics:
1. **Sparse matrix-matrix multiply** — C = A*B for forming Schur complements or preconditioning
2. **Block LU factorization** — exploit dense subblocks for better cache performance
3. **Parallel SpMV** — OpenMP parallelization of matrix-vector product
4. **Iterative solvers** — CG/GMRES using the sparse infrastructure
5. **CSR/CSC export** — convert to compressed formats for interoperability
