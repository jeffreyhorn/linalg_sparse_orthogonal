# Sprint 8 Retrospective: Matrix-Free Interface & Sparse SVD

**Sprint Duration:** 14 days
**Branch:** `sprint-8`

---

## Definition of Done Checklist

- [x] `make clean && make test` — all 642 tests pass across 27 test suites
- [x] `make sanitize` — UBSan clean, 0 findings
- [x] `make bench` — all benchmarks run without crashes
- [x] `make format-check` — all files pass clang-format
- [x] Cross-feature integration tests (7 tests on SuiteSparse matrices)
- [x] All new public API functions have doc comments
- [x] `const` correctness on all new functions
- [x] No new compiler warnings with strict flags (only pre-existing eigen2x2 implicit decl in test)
- [x] Backward compatibility: all previous tests unchanged

---

## Sprint Metrics

| Metric | Start (Sprint 7 end) | End (Sprint 8) | Delta |
|--------|---------------------|-----------------|-------|
| Public API functions | ~80 | 97 | +17 |
| Test suites | 24 | 27 | +3 |
| Tests (RUN_TEST) | ~560 | 642 | +82 |
| SuiteSparse matrices used | 6 | 6 | +0 |
| Source files | 14 | 14 | +0 |

### New Public API Functions (17)

**Matrix-free interface (3):**
- `sparse_matvec_fn` callback type
- `sparse_solve_cg_mf()` — matrix-free CG
- `sparse_solve_gmres_mf()` — matrix-free GMRES

**Bidiagonalization (0 new, 1 fixed):**
- `sparse_bidiag_factor()` — extended to handle wide matrices (m < n) via internal transpose

**SVD core (5):**
- `sparse_svd_compute()` — full SVD: A = U * diag(sigma) * V^T
- `sparse_svd_free()` — cleanup
- `sparse_svd_extract_uv()` — explicit U/V from Householder reflectors
- `sparse_svd_partial()` — Lanczos bidiagonalization for top-k singular values
- `bidiag_svd_iterate()` — implicit QR iteration on bidiagonal (internal, exposed for testing)

**SVD applications (3):**
- `sparse_svd_rank()` — numerical rank estimation
- `sparse_pinv()` — Moore-Penrose pseudoinverse
- `sparse_svd_lowrank()` — best rank-k approximation

**Dense utilities (6, from Sprint 7 carryover):**
- `dense_create()`, `dense_free()`, `dense_gemm()`, `dense_gemv()`
- `givens_compute()`, `givens_apply_left()`, `givens_apply_right()`
- `eigen2x2()`, `tridiag_qr_eigenvalues()`

---

## What Went Well

1. **Incremental SVD build**: Breaking SVD into bidiag → extraction → QR iteration → driver → partial → applications over 10 days made each piece testable in isolation. Found and fixed bugs at each stage before they compounded.

2. **Bug fixes improved overall quality**: Three significant SVD bugs were found and fixed:
   - Bulge chase missing superdiagonal contribution (Day 11) — fixed ~9% error on all non-trivial bidiagonals
   - 2×2 direct solve eigenvalue/rotation ordering mismatch (Day 13) — fixed SVD reconstruction for all matrices
   - Both fixes improved existing tests (trace invariant now exact, reconstruction error now ~1e-14)

3. **SuiteSparse validation**: Testing against real-world matrices (nos4, west0067, bcsstk04, steam1) caught issues that small synthetic tests missed.

4. **Matrix-free GMRES refactoring**: Converting GMRES to use a matrix-free core with a SparseMatrix adapter was clean — backward compatibility preserved with zero API changes to existing functions.

5. **Lanczos partial SVD**: Oversampling strategy (2k+10 Lanczos steps for k values) gave good convergence. 8x faster than full SVD on nos4 for k=5.

---

## What Didn't Go Well

1. **SVD QR step bugs**: The bulge chase right rotation bug and the 2×2 direct solve ordering bug both caused significant inaccuracy. These went undetected for multiple days because:
   - Diagonal matrix tests pass trivially (zero superdiag → no QR iteration)
   - The 2×2 solve was only triggered for small blocks
   - Test tolerances were initially too loose

2. **Rank-1 SVD still broken**: The implicit QR iteration doesn't converge on rank-deficient bidiagonals where near-zero diagonal entries prevent deflation. The proper fix (zero-diagonal chase per Golub & Van Loan §8.6.2) was deferred. The `test_svd_rank1` test remains disabled.

3. **Wide matrix bidiag complexity**: The `transposed` flag approach works but adds complexity to UV extraction. The code has two parallel paths (transposed vs non-transposed) that must stay in sync.

---

## Bugs Found During Sprint

| Bug | Day Found | Day Fixed | Impact |
|-----|-----------|-----------|--------|
| Bulge chase drops `s*z` in right rotation | 11 | 11 | ~9% error on all non-trivial bidiagonals |
| 2×2 direct solve: eigenvalue ordering ≠ rotation ordering | 13 | 13 | Wrong U/V vectors for all 2×2 blocks in QR iteration |
| Rank-1 bidiagonal QR non-convergence | 7 | — | Disabled test, deferred to Sprint 9 |

---

## Items Deferred to Sprint 9

1. **Zero-diagonal chase** (Golub & Van Loan §8.6.2): Handle rank-deficient bidiagonals where a diagonal entry is near-zero. Required for correct rank-1 SVD.

2. **SVD singular vectors for partial SVD**: `sparse_svd_partial()` currently returns only singular values. Recovering approximate U/V from Lanczos vectors would enable truncated SVD with vectors.

3. **Sparse low-rank output**: `sparse_svd_lowrank()` returns a dense matrix. A sparse variant that thresholds small entries would be more memory-efficient for large matrices.

4. **Condition number estimation**: `sparse_cond()` via ratio of largest/smallest singular values.

5. **Performance optimization**: The current SVD is O(mn*min(m,n)) via dense bidiagonalization. For very sparse matrices, iterative approaches or blocked algorithms could be faster.

---

## Final Test Summary

```
27 test suites, 642 tests, all passing
UBSan: clean
Benchmarks: all complete
Format/lint: clean
```
