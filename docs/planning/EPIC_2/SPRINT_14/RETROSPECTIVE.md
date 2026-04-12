# Sprint 14 Retrospective: Symbolic Analysis / Numeric Factorization Split

**Sprint Duration:** 14 days
**Goal:** Separate symbolic analysis from numeric factorization for LU, Cholesky, and LDL^T

---

## Definition of Done Checklist

| Item | Status |
|------|--------|
| `sparse_etree_compute()` — Liu's algorithm with path compression | Done |
| `sparse_etree_postorder()` — iterative DFS postorder | Done |
| `sparse_colcount()` — exact column counts from etree | Done |
| `sparse_symbolic_cholesky()` — exact symbolic L structure | Done |
| `sparse_symbolic_lu()` — upper-bound L/U via column etree of A^T*A | Done |
| `sparse_symbolic_free()` — cleanup | Done |
| `sparse_analysis_t` public struct | Done |
| `sparse_analyze()` — Cholesky, LU, LDL^T paths with AMD/RCM | Done |
| `sparse_analysis_free()` — cleanup | Done |
| `sparse_factors_t` struct (Cholesky, LU, LDL^T) | Done |
| `sparse_factor_numeric()` — all three factorization types | Done |
| `sparse_factor_solve()` — with auto-permutation | Done |
| `sparse_factor_free()` — cleanup | Done |
| `sparse_refactor_numeric()` — repeated factorization | Done |
| Backward compatibility tests (one-shot == analyze+factor) | Done |
| Documentation (README, algorithm.md) | Done |
| Example program (`example_analysis.c`) | Done |
| Refactorization benchmark (`bench_refactor.c`) | Done |
| `make format && make lint && make test` clean | Done |
| UBSan clean | Done |

---

## Final Metrics

| Metric | Value |
|--------|-------|
| Total tests (all suites) | 1,069 |
| Analysis-specific tests (test_etree) | 93 |
| Analysis-specific assertions | 638 |
| Public headers | 17 (was 16, +sparse_analysis.h) |
| Source files | 21 (was 19, +sparse_analysis.c, +sparse_etree.c) |
| New public API functions | 6 (analyze, factor_numeric, refactor_numeric, factor_solve, analysis_free, factor_free) |
| New internal functions | 7 (etree_compute, etree_postorder, colcount, symbolic_cholesky, symbolic_lu, symbolic_free, isort) |
| New lines of code | ~4,474 (src + tests + bench + example + header) |
| Test suites | 36 (unchanged) |

### Benchmark: Analyze-Once vs One-Shot (100 reps, Cholesky)

| Matrix | n | nnz | One-shot | Analyze-once | Speedup |
|--------|---|-----|----------|--------------|---------|
| tridiag-50 | 50 | 148 | 0.0020s | 0.0013s | 1.48x |
| tridiag-200 | 200 | 598 | 0.0122s | 0.0072s | 1.70x |
| tridiag-500 | 500 | 1498 | 0.0231s | 0.0187s | 1.24x |
| bcsstk04 | 132 | 3648 | 0.3350s | 0.3282s | 1.02x |
| nos4 | 100 | 594 | 0.0146s | 0.0201s | 0.73x |

### Symbolic Analysis Accuracy

| Matrix | Symbolic nnz(L) | Numeric nnz(L) | Match |
|--------|----------------|-----------------|-------|
| tridiag (n=8) | 15 | 15 | Exact |
| arrow (n=6) | 11 | 11 | Exact |
| dense (n=4) | 10 | 10 | Exact |
| bcsstk04 (132x132) | 3,763 | 3,664 | Superset (2.7% over) |
| nos4 (100x100) | 805 | 805 | Exact |

---

## What Went Well

1. **Clean layered design.** Building from etree → postorder → colcounts → symbolic structure → public API was natural and each layer was testable independently.

2. **Test-first for each day.** Every day ended with comprehensive tests, making each subsequent day's work reliable. The test count grew from 16 (Day 1) to 93 (Day 14) in the analysis suite alone.

3. **Backward compatibility verified.** The one-shot APIs produce bit-identical residuals to the analyze+factor path, confirmed on all SuiteSparse test matrices.

4. **Three factorization types from Day 1.** By designing the `sparse_factor_type_t` enum and the switch-based dispatch early, adding LU and LDL^T paths was straightforward after Cholesky.

5. **SuiteSparse integration.** Testing against real matrices (bcsstk04, nos4, west0067, steam1) caught the DROP_TOL issue in symbolic Cholesky on Day 3 early.

## What Didn't Go Well

1. **Benchmark speedup is modest.** The linked-list representation means `sparse_copy()` dominates refactorization cost. The analyze-once approach would show much larger gains with a CSR/CSC internal representation where pre-allocated arrays avoid re-allocation. This is a known architectural limitation.

2. **clang-tidy false positives.** The static analyzer repeatedly flagged valid array accesses as out-of-bounds, requiring ~15 NOLINT annotations across the codebase. These are all false positives from the analyzer not tracking array-length relationships across function boundaries.

3. **`sparse_symbolic_lu()` uses A^T*A approach.** The initial implementation using A+A^T was insufficient for unsymmetric matrices (failed on west0067 and steam1). Switching to the column interaction graph (A^T*A) was correct but is O(nnz^2/n) in the worst case for dense rows. For matrices with very dense rows, this could be expensive.

---

## Bugs Found During Sprint

1. **Day 3:** Symbolic nnz(L) for bcsstk04 was 3,763 while numeric nnz was 3,664. Root cause: numeric Cholesky drops tiny fill-in entries below DROP_TOL. Fix: symbolic tests for real matrices use containment (>=) rather than exact equality.

2. **Day 4:** `A + A^T` symmetrization for symbolic LU was insufficient — it doesn't capture column interactions through shared nonzero rows. Fix: replaced with `A^T * A` column interaction graph.

3. **Day 5:** `sparse_norminf()` requires a non-const SparseMatrix (for caching). The analysis function uses a `(uintptr_t)` cast to call it on a const input. This is safe because norminf only writes an atomic cache field, but it's not ideal.

---

## Items Deferred

None. All 14 days' deliverables are complete.

---

## Architecture Notes for Future Sprints

- The refactorization path currently frees and re-allocates factors. A future optimization would reuse the pre-allocated compressed-column storage from the symbolic structure, writing numeric values directly into it. This would eliminate `sparse_copy()` overhead and show the full benefit of symbolic reuse.

- The `sparse_symbolic_pub_t` type in the public header duplicates `sparse_symbolic_t` from the internal header. These could be unified if the internal header is ever exposed or if the types are made identical via a typedef.

- The LDL^T solve path reconstructs a temporary `sparse_ldlt_t` from the factors struct on each call. This could be cached.
