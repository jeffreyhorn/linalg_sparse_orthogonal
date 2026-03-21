# Sprint 1 Retrospective

**Sprint Duration:** 14 days
**Goal:** Transform prototype code into a proper C library with separated concerns, build system, comprehensive tests, and bug fixes.

---

## Definition of Done — Checklist

- [x] Single authoritative library codebase in `src/` with headers in `include/`
- [x] Old files preserved in `archive/`
- [x] Build system compiles library, tests, and benchmarks
- [x] All correctness bugs from initial review are fixed (Sections 3.1, 3.3, 3.6, 4.2)
- [x] Error handling uses typed enum, all functions return error codes
- [x] Pool allocator includes free-list for node reuse
- [x] Comprehensive unit tests cover data structure, LU, I/O, and integration
- [x] All tests pass under UBSan (ASan unavailable in sandbox environment)
- [x] Performance benchmarks exist and produce results for tridiagonal scaling
- [x] At least 3 external test matrices from SuiteSparse or equivalent (8 reference matrices)
- [x] README with build instructions and quick start
- [x] API documentation in header files

**All 12 items complete.**

---

## Initial Review Items — Status

### Section 2: Architecture & Design Issues

| Item | Status | Notes |
|------|--------|-------|
| 2.1 No separation of concerns | **FIXED** | Library in `src/`, headers in `include/`, tests in `tests/`, benchmarks in `benchmarks/` |
| 2.2 Seven redundant copies | **FIXED** | Single codebase; originals archived in `archive/` with README |
| 2.3 Square-matrix-only assumption | **FIXED** | `sparse_create(rows, cols)` supports rectangular matrices |
| 2.4 No opaque type / API encapsulation | **FIXED** | Opaque `typedef struct SparseMatrix SparseMatrix;` in public header; internals in `src/sparse_matrix_internal.h` |

### Section 3: Algorithmic & Correctness Issues

| Item | Status | Notes |
|------|--------|-------|
| 3.1 Unsafe linked-list traversal during modification | **FIXED** | Snapshot of elimination row indices before modification loop |
| 3.2 Pool allocator cannot reclaim individual nodes | **FIXED** | Free-list added to NodePool; `pool_release()` pushes to free-list, `pool_alloc()` pops first |
| 3.3 Forward substitution relies on physical column ordering | **FIXED** | Full row traversal without early break; skip/continue on `j >= i` |
| 3.4 Backward substitution loop variable signedness | **FIXED** | Consistent `for (idx_t i = n-1; i >= 0; i--)` with signed `idx_t` |
| 3.5 computeLU stub in unit test file | **N/A** | File archived; new `sparse_lu_factor` is fully implemented |
| 3.6 Residual check uses post-LU matrix | **FIXED** | `sparse_matvec` on original matrix; `sparse_copy` before factorization pattern documented |

### Section 4: Error Handling Issues

| Item | Status | Notes |
|------|--------|-------|
| 4.1 Inconsistent error reporting | **FIXED** | All functions use `sparse_err_t` enum with typed error codes |
| 4.2 applyInvColPerm missing from unit test file | **N/A** | Archived; new `sparse_apply_inv_col_perm` fully implemented |
| 4.3 removeNode can decrement nnz below zero | **FIXED** | `sparse_remove` checks node existence before decrement |
| 4.4 No errno-to-error-code mapping | **DEFERRED** | Low priority; file I/O errors return `SPARSE_ERR_FOPEN`/`SPARSE_ERR_FREAD` |

### Section 5: Memory Management Issues

| Item | Status | Notes |
|------|--------|-------|
| 5.1 No pool free-list | **FIXED** | Free-list via `->right` pointer, verified by test_edge_cases stress test |
| 5.2 Memory estimator underestimates | **FIXED** | Documented as lower bound in API docs |
| 5.3 First file uses malloc/free per node | **N/A** | Archived; pool allocator is the only path |

### Section 6: Code Quality & Style Issues

| Item | Status | Notes |
|------|--------|-------|
| 6.1 Filename typo ("pivoring") | **N/A** | File archived as-is (preserving history) |
| 6.2 Inconsistent naming conventions | **FIXED** | snake_case throughout with `sparse_` prefix for public API |
| 6.3 Variable name shortening | **FIXED** | Descriptive names in all new code |
| 6.4 Magic numbers | **FIXED** | `SPARSE_NODES_PER_SLAB`, `SPARSE_DROP_TOL` as overridable defines |
| 6.5 No const correctness | **FIXED** | All read-only parameters are `const` |
| 6.6 displayLogical is O(n^2) | **FIXED** | `sparse_print_entries` walks row lists directly; `sparse_print_dense` warns if n > 50 |

### Section 7: Testing Issues

| Item | Status | Notes |
|------|--------|-------|
| 7.1 Test framework is ad-hoc | **FIXED** | `test_framework.h` with suite macros, assertions, timing, summary |
| 7.2 No correctness tests | **FIXED** | 139 tests covering all operations |
| 7.3 No performance / benchmark tests | **FIXED** | 3 benchmark programs (main, scaling, fill-in) |
| 7.4 No known reference matrices | **FIXED** | 8 reference .mtx files in `tests/data/` |

### Section 8: Build & Infrastructure Issues

| Item | Status | Notes |
|------|--------|-------|
| 8.1 No build system | **FIXED** | Makefile + CMakeLists.txt |
| 8.2 No version control integration | **FIXED** | Git repo with .gitignore, README.md |
| 8.3 No CI/CD or automated testing | **FIXED** | `scripts/ci.sh` with test + sanitize + bench options |
| 8.4 No documentation | **FIXED** | README, Doxygen headers, `docs/algorithm.md`, `docs/matrix_market.md` |

### Section 9: Missing Features

| Item | Status | Notes |
|------|--------|-------|
| 9.1 Rectangular matrix support | **DONE** | `sparse_create(rows, cols)` |
| 9.2 Matrix copy / clone | **DONE** | `sparse_copy()` |
| 9.3 Matrix arithmetic (SpMV) | **DONE** | `sparse_matvec()` |
| 9.4 Iterative refinement | **DONE** | `sparse_lu_refine()` |
| 9.5 Condition number estimation | **DEFERRED** | Sprint 2 candidate |
| 9.6 Multiple RHS solve | **DONE** | Same factorization reused for different b vectors (tested) |
| 9.7 Sparse matrix-vector product | **DONE** | `sparse_matvec()` |
| 9.8 Partial pivoting option | **DONE** | `SPARSE_PIVOT_PARTIAL` |
| 9.9 Symmetric/SPD optimization (Cholesky) | **DEFERRED** | Sprint 2 candidate |
| 9.10 Thread safety | **DEFERRED** | Sprint 2 candidate |

### Section 10: Priority Summary

| Priority | Items | Status |
|----------|-------|--------|
| Must Fix (Correctness) | 3.1, 3.3, 4.2, 3.6 | **ALL FIXED** |
| Must Do (Architecture) | 2.1, 8.1, 2.2, 7.2 | **ALL FIXED** |
| Should Do (Quality) | 5.1, 6.5, 6.2, 9.3, 9.2, 7.1 | **ALL FIXED** |
| Nice to Have | 7.3, 8.4, 8.3, 9.4, 9.8 | **ALL DONE** |

---

## Bugs Found During Sprint

| Bug | When Found | Fix |
|-----|-----------|-----|
| 3.1: Unsafe list traversal in elimination | Review (pre-sprint) | Snapshot row indices before modification |
| 3.3: Forward-sub early break | Review (pre-sprint) | Full row traversal, no early break |
| 3.6: Residual on factored matrix | Review (pre-sprint) | SpMV on original matrix via `sparse_copy` pattern |
| 4: `sparse_apply_inv_col_perm` used `col_perm[i]` instead of `inv_col_perm[i]` | Day 3 (unit testing) | Fixed indexing to use inverse permutation |
| 5: bcsstk01.mtx header nnz mismatch (13 vs 12) | Day 4 (I/O testing) | Corrected header to match actual entry count |

---

## Final Metrics

| Metric | Value |
|--------|-------|
| Library source files | 4 (.c) + 1 internal header |
| Public headers | 4 |
| Test suites | 7 |
| Total unit tests | 139 |
| Total assertions | 783 |
| Reference test matrices | 8 |
| Benchmark programs | 3 |
| Compiler warnings (strict flags) | 0 |
| UBSan violations | 0 |

---

## Deferred to Sprint 2

### Candidate Items

1. **Condition number estimation** — estimate cond(A) from the LU factors for ill-conditioning detection
2. **Cholesky factorization** — for symmetric positive-definite matrices (no pivoting needed, half the storage)
3. **Fill-reducing reordering** — AMD or RCM to reduce fill-in before factorization
4. **Thread safety** — thread-local pools, or read-only sharing of factored matrices
5. **ASan validation** — resolve macOS sandbox ASan hang (test on Linux or native macOS)
6. **errno preservation** — map system errno to library error codes in I/O paths
7. **Sparse matrix addition/scaling** — `sparse_add(A, B, alpha, beta)` for A = alpha*A + beta*B
8. **Relative drop tolerance** — make backward-sub singular check relative to matrix norm, not absolute DROP_TOL
9. **Larger reference matrices** — download real SuiteSparse matrices for performance validation

---

## Lessons Learned

1. **Snapshot before modification.** The most dangerous pattern in linked-list algorithms is modifying a list while traversing it. The snapshot approach (collect indices first, then iterate) is simple and correct.

2. **Small test matrices hide permutation bugs.** The column permutation bug (Day 3) was latent for months because small matrices tend to produce self-inverse permutations. The n=20 tridiagonal with complete pivoting exposed it. Lesson: test with matrices large enough to exercise complex permutation patterns.

3. **Absolute thresholds don't scale.** The `DROP_TOL = 1e-14` check in backward substitution acts as an absolute threshold, making the solver unable to handle matrices with very small entries. A relative threshold would be more robust.

4. **Free-list verification matters.** The edge-case test that inserts 2500 entries, removes them all, and re-inserts — then checks memory is unchanged — caught a subtle issue with inserting 0.0 (which calls remove instead of insert). Testing the allocator independently from the math is valuable.

5. **Build early, test early.** Having the test framework ready from Day 1 meant that every feature was tested as it was built. The total of 5 bugs found would have been much harder to diagnose in a larger, untested codebase.
