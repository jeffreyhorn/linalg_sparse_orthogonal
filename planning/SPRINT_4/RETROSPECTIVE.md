# Sprint 4 Retrospective

**Sprint Duration:** 14 days
**Goal:** Add Cholesky factorization for SPD matrices, make the library safe for concurrent use, implement sparse matrix-matrix multiply, and add CSR/CSC export/import.

---

## Definition of Done — Checklist

- [x] `sparse_cholesky_factor()` and `sparse_cholesky_solve()` in public API
- [x] Cholesky exploits symmetry: stores only lower triangle
- [x] Non-SPD detection with `SPARSE_ERR_NOT_SPD`
- [x] Cholesky integrates with AMD/RCM reordering (`sparse_cholesky_factor_opts()`)
- [x] Thread-safety contract documented and verified
- [x] Concurrent solve stress tests pass (8 threads × 1000 iterations)
- [x] Optional mutex via `-DSPARSE_MUTEX` compile flag
- [x] `sparse_to_csr()`, `sparse_to_csc()`, `sparse_from_csr()`, `sparse_from_csc()` in public API
- [x] `sparse_matmul(A, B, &C)` in public API
- [x] `sparse_is_symmetric(mat, tol)` in public API
- [x] All existing tests remain passing (backward compatible)
- [x] Updated algorithm documentation and README

**All items complete.**

---

## Sprint Items — Status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Cholesky factorization | **DONE** | Left-looking column-by-column with dense accumulator. SPD detection, AMD/RCM integration, solve with auto-unpermute. 21 tests. |
| 2 | Thread safety | **DONE** | Audit: only global is `_Thread_local` errno. Solve verified read-only (const). Optional mutex. 7 thread tests + 5 integration. |
| 3 | CSR/CSC export | **DONE** | Bidirectional conversion with validation. Round-trip verified on SuiteSparse matrices. 11 tests. |
| 4 | SpMM | **DONE** | Gustavson's algorithm with dense accumulator. Associativity verified on nos4. L*L^T reconstruction. 14 tests. |

---

## New API Surface

| Function | Header | Purpose |
|----------|--------|---------|
| `sparse_cholesky_factor()` | `sparse_cholesky.h` | In-place Cholesky A = L*L^T |
| `sparse_cholesky_factor_opts()` | `sparse_cholesky.h` | Cholesky with AMD/RCM reordering |
| `sparse_cholesky_solve()` | `sparse_cholesky.h` | Solve using Cholesky factors |
| `sparse_is_symmetric()` | `sparse_matrix.h` | Check matrix symmetry |
| `sparse_matmul()` | `sparse_matrix.h` | Sparse matrix-matrix multiply |
| `sparse_to_csr()` | `sparse_csr.h` | Convert to CSR format |
| `sparse_from_csr()` | `sparse_csr.h` | Convert from CSR format |
| `sparse_to_csc()` | `sparse_csr.h` | Convert to CSC format |
| `sparse_from_csc()` | `sparse_csr.h` | Convert from CSC format |
| `sparse_csr_free()` | `sparse_csr.h` | Free CSR structure |
| `sparse_csc_free()` | `sparse_csr.h` | Free CSC structure |

New types: `sparse_cholesky_opts_t`, `SparseCsr`, `SparseCsc`.
New error code: `SPARSE_ERR_NOT_SPD` (value 12).

---

## Cholesky vs LU Comparison

| Matrix | LU nnz | Cholesky nnz | Savings | LU residual | Cholesky residual |
|--------|-------:|-------------:|--------:|------------:|------------------:|
| nos4 (100×100) | 1510 | 805 | 47% | 1.63e-16 | 1.67e-16 |
| bcsstk04 (132×132) | 8581 | 3664 | 57% | 5.59e-09 | 7.45e-09 |

With AMD reordering:
| Matrix | Cholesky nnz (none) | Cholesky nnz (AMD) | Reduction |
|--------|--------------------:|-------------------:|----------:|
| nos4 | 805 | 637 | 21% |
| bcsstk04 | 3664 | 3575 | 2% |

---

## Thread-Safety Verification

| Test | Threads | Iterations | Max Error | Result |
|------|--------:|-----------:|----------:|--------|
| Independent LU factor+solve | 4 | 1 | 2.22e-16 | PASS |
| Concurrent LU solve (shared) | 4 | 100 | 5.68e-14 | PASS |
| Concurrent Cholesky solve (shared) | 4 | 100 | 5.68e-14 | PASS |
| LU solve stress | 8 | 1000 | ~1e-13 | PASS |
| Cholesky solve stress | 8 | 1000 | ~1e-13 | PASS |
| Independent stress | 8 | 1 | ~1e-16 | PASS |
| Concurrent insert (non-overlapping) | 4 | 1 | exact | PASS |

Key finding: all solve functions are verified pure read-only on the matrix struct (`const SparseMatrix *`). All workspace is stack/heap allocated — no pool allocation during solve. This makes concurrent solves on shared factored matrices inherently safe without any synchronization.

---

## Bugs Found During Sprint

| Bug | When Found | Fix |
|-----|-----------|-----|
| Cholesky 3x3 tridiag test had wrong expected L(1,0) value | Day 2 | Fixed: L(1,0) = -1/2, not -1/4. Arithmetic error in manual calculation. |
| Concurrent insert test: row 0 has col 0 as both "column 0" entry and diagonal | Day 11 | Fixed: adjusted expected values to account for last-write-wins on overlapping position. |

---

## Final Metrics

| Metric | Sprint 3 | Sprint 4 | Delta |
|--------|----------|----------|-------|
| Library source files | 5 (.c) + 1 internal header | 7 (.c) + 1 internal header | +2 |
| Public headers | 5 | 7 | +2 |
| Public API functions | 38 | 56 | +18 |
| Error codes | 12 | 13 | +1 |
| Test suites | 10 | 15 | +5 |
| Total unit tests | 242 | 305 | +63 |
| Total assertions | 1255 | 2212 | +957 |
| Reference test matrices | 14 | 14 | — |
| Benchmark programs | 3 (enhanced) | 3 (enhanced) | — |
| Compiler warnings | 0 | 0 | — |
| UBSan violations | 0 | 0 | — |

---

## Lessons Learned

1. **Cholesky stores ~50% less than LU on SPD matrices.** The lower-triangle-only storage consistently halves the fill-in compared to full LU, confirming the theoretical advantage.

2. **Thread safety was easier than expected.** The per-matrix pool design and `const SparseMatrix *` on solve functions meant the library was already effectively thread-safe for the most important use case (concurrent solves). The audit was the main work, not the code changes.

3. **Gustavson's algorithm is clean and efficient.** The dense accumulator approach for SpMM handles cancellation naturally (zeros are simply not flushed) and avoids the complexity of hash-based or sort-based merging.

4. **CSR/CSC round-trips validate data structure integrity.** Converting to a completely different format and back is an excellent structural test — it caught no bugs (everything worked first try), which validates the orthogonal linked-list implementation quality.

5. **Optional mutex via compile flag is the right pattern.** Zero overhead for single-threaded code, available when needed. Most users will never enable it since separate matrices per thread is the natural pattern.

---

## Deferred / Carried to Sprint 5

All Sprint 4 items completed. Sprint 5 begins with:
1. **Conjugate Gradient (CG) solver** for SPD systems
2. **GMRES solver** for general (unsymmetric) systems
3. **ILU(0) preconditioner** for accelerating iterative solvers
4. **Parallel SpMV** via OpenMP
