# Sprint 15 Retrospective: COLAMD Ordering & QR Minimum-Norm Least Squares

**Sprint Duration:** 14 days
**Goal:** Upgrade the ordering stack with COLAMD for unsymmetric/QR problems and add minimum-norm least-squares solves for underdetermined systems.

---

## Definition of Done Checklist

| Item | Status |
|------|--------|
| `colamd_build_graph()` — column adjacency graph without forming A^T*A | Done |
| `colamd_order()` — minimum-degree elimination on column graph | Done |
| Dense row detection and skipping (threshold: 10·sqrt(n)) | Done |
| Overflow guards and SIZE_MAX validation | Done |
| `SPARSE_REORDER_COLAMD = 3` enum value | Done |
| `sparse_reorder_colamd()` public API | Done |
| COLAMD integrated with `sparse_qr_factor_opts()` (dense + sparse modes) | Done |
| COLAMD integrated with `sparse_analyze()` | Done |
| `sparse_qr_solve_minnorm()` — minimum 2-norm for underdetermined systems | Done |
| `sparse_qr_refine_minnorm()` — iterative refinement for minimum-norm | Done |
| `sparse_qr_diag_r()` — R diagonal extraction | Done |
| `sparse_qr_rank_info()` — rank diagnostics with condition estimate | Done |
| `sparse_qr_condest()` — quick condition estimate from R diagonal | Done |
| QR solve/rank documentation improved (overdetermined/square/underdetermined) | Done |
| SuiteSparse integration tests (west0067, steam1, fs_541_1, orsirr_1) | Done |
| Minimum-norm vs SVD pseudoinverse comparison | Done |
| Fill-reduction benchmark (`bench_colamd.c`) | Done |
| Example programs (`example_minnorm.c`, `example_colamd.c`) | Done |
| README and algorithm docs updated | Done |
| `make format && make lint && make test` clean | Done |

---

## Final Metrics

| Metric | Value |
|--------|-------|
| Total tests (all suites) | 1,140 |
| Sprint-specific tests (test_colamd) | 70 |
| Sprint-specific assertions | 249 |
| Public headers | 17 (unchanged) |
| Source files | 23 (was 21, +sparse_colamd.c, +sparse_colamd_internal.h) |
| New public API functions | 7 (reorder_colamd, qr_solve_minnorm, qr_refine_minnorm, qr_diag_r, qr_rank_info, qr_condest, COLAMD enum) |
| New lines of code | ~2,604 (src + tests + bench + examples) |
| Test suites | 37 (was 36, +test_colamd) |

### COLAMD Fill Reduction (LU)

| Matrix | n | Natural nnz(LU) | COLAMD nnz(LU) | Reduction |
|--------|---|-----------------|----------------|-----------|
| west0067 | 67 | 928 | 723 | 22% |
| steam1 | 240 | 22,956 | 9,344 | 59% |
| fs_541_1 | 541 | 7,401 | 7,053 | 5% |
| orsirr_1 | 1,030 | 78,069 | 79,006 | -1% |

### QR Condition Estimation

| Matrix | QR condest | SVD true cond | Ratio |
|--------|-----------|---------------|-------|
| Upper triangular 4x4 | 10.00 | 11.58 | 0.86 |

### Minimum-Norm Accuracy

| System | ||x_min||_2 | ||x_alt||_2 | A*x = b error |
|--------|-----------|-------------|---------------|
| 2×4 block diagonal | 1.000 | 1.414 | 0.00e+00 |
| 3×6 diagonal+offdiag | 2.898 | 5.388 | 0.00e+00 |
| 30×67 west0067 submatrix | 4.30 | 8.19 (ones) | 1.78e-15 |

---

## What Went Well

1. **COLAMD column adjacency graph is clean.** The per-column marker approach (`marker[k] = j` for dedup) correctly handles all cases without forming A^T*A. Dense row skipping with sqrt(n) threshold prevents blowup on matrices with very dense rows.

2. **Minimum-norm algorithm is elegant.** The QR-of-A^T approach reuses the entire existing QR infrastructure — factorization, Householder application, column pivoting. No new data structures needed.

3. **SVD pseudoinverse comparison validates correctness.** The `sparse_pinv`-based test confirms QR minnorm matches the SVD pseudoinverse to machine precision on the 2×4 test case.

4. **Steam1 shows dramatic fill reduction.** 59% LU fill reduction on steam1 (240×240) demonstrates COLAMD's value on real unsymmetric matrices with structured column interactions.

5. **Iterative refinement for minnorm works.** The refine_minnorm test shows residual improving from 8.88e-16 to 0.00e+00 in 3 iterations.

## What Didn't Go Well

1. **QR R-factor fill is unchanged by COLAMD.** The QR benchmark shows 0% fill reduction on all test matrices because Householder QR with column pivoting already reorders columns internally during factorization. COLAMD is most useful as a pre-ordering for LU or for non-pivoted QR variants. The benefit is in the column adjacency structure, not the R-factor fill after full column pivoting.

2. **COLAMD uses O(n^2) bitset memory.** Like AMD, the elimination loop uses n-bit adjacency rows. This limits scalability to matrices where n < ~50,000 (where the bitset fits in memory). For larger matrices, a compressed sparse representation would be needed.

3. **SVD API complexity.** The test comparing minnorm with direct SVD pseudoinverse computation initially failed due to column-major storage order confusion in the SVD API. Switching to `sparse_pinv` was the pragmatic fix.

---

## Bugs Found During Sprint

1. **Day 1:** Initial column adjacency graph used a row-pair marker approach (`marker[cb] = ca`) that failed to deduplicate across rows. Fixed by switching to per-column traversal with `marker[k] = j`.

2. **Day 9:** SVD comparison test used incorrect indexing for economy SVD's U matrix (m×k vs m×m storage). Fixed by using `sparse_pinv` instead of manual SVD pseudoinverse computation.

---

## Items Deferred

None. All 6 project plan items are complete.

---

## Architecture Notes for Future Sprints

- COLAMD's O(n²) bitset representation matches AMD's approach and is adequate for the library's current target matrix sizes. For much larger matrices, a compressed-representation COLAMD (like SuiteSparse's reference implementation) would be needed.

- The minimum-norm solve creates a transposed copy and a full QR factorization each time. For repeated solves with the same A but different b, caching the A^T QR factorization would avoid redundant work.

- The `sparse_qr_condest` from R diagonals is a 2-norm estimate. It could be combined with the Hager/Higham 1-norm estimator for a more robust condition estimate.
