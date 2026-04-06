# Sprint 4 Plan: Cholesky Factorization, Thread Safety, SpMM & CSR/CSC

**Sprint Duration:** 14 days
**Goal:** Add Cholesky factorization for SPD matrices, make the library safe for concurrent use, implement sparse matrix-matrix multiply, and add CSR/CSC export/import for interoperability.

**Starting Point:** A sparse LU library with 38 public API functions, 242 tests across 10 test suites, 6 SuiteSparse reference matrices, condition estimation, AMD/RCM fill-reducing reordering, and a `sparse_lu_opts_t` options struct. The library is single-threaded with a global slab pool allocator.

**End State:** `sparse_cholesky_factor()` and `sparse_cholesky_solve()` for SPD matrices with AMD/RCM integration, thread-safe pool allocator with documented concurrency guarantees, `sparse_matmul()` for C = A*B, and `sparse_to_csr()`/`sparse_from_csr()` for compressed format interop.

---

## Day 1: Cholesky — API Design & SPD Detection

**Theme:** Design the Cholesky interface and implement symmetry/SPD checking

**Time estimate:** 7 hours

### Tasks
1. Add `SPARSE_ERR_NOT_SPD` error code to `sparse_err_t` enum in `include/sparse_types.h`
2. Update `sparse_strerror()` to handle the new error code
3. Design the Cholesky API in a new `include/sparse_cholesky.h`:
   - `sparse_err_t sparse_cholesky_factor(SparseMatrix *mat)` — in-place Cholesky: stores L in the lower triangle. Detects non-SPD via negative/zero pivot
   - `sparse_err_t sparse_cholesky_solve(const SparseMatrix *mat, const double *b, double *x)` — solve A*x = b using factored L: forward-sub with L, back-sub with L^T
   - `sparse_err_t sparse_cholesky_factor_opts(SparseMatrix *mat, const sparse_cholesky_opts_t *opts)` — with optional AMD/RCM reordering
   - `typedef struct { sparse_reorder_t reorder; } sparse_cholesky_opts_t`
4. Implement `sparse_is_symmetric(A, tol)` helper that checks whether A equals A^T within tolerance (walk rows and verify matching column entries). Add to `sparse_matrix.h`
5. Add stub implementations in `src/sparse_cholesky.c` that compile cleanly
6. Add `sparse_cholesky.c` to Makefile and CMakeLists.txt

### Deliverables
- `SPARSE_ERR_NOT_SPD` error code
- `sparse_cholesky.h` with full API declarations and doc comments
- `sparse_is_symmetric()` implemented and tested
- Stub implementations that compile cleanly
- Build system updated

### Completion Criteria
- Code compiles with zero warnings
- Existing `make test` passes

---

## Day 2: Cholesky — Core Factorization

**Theme:** Implement the sparse Cholesky factorization algorithm

**Time estimate:** 9 hours

### Tasks
1. Implement `sparse_cholesky_factor()` in `src/sparse_cholesky.c`:
   - Verify matrix is square, return `SPARSE_ERR_SHAPE` if not
   - For k = 0 to n-1:
     - Compute L(k,k) = sqrt(A(k,k) - sum_{j<k} L(k,j)^2)
     - If argument to sqrt is negative/zero, return `SPARSE_ERR_NOT_SPD`
     - For i = k+1 to n-1:
       - L(i,k) = (A(i,k) - sum_{j<k} L(i,j)*L(k,j)) / L(k,k)
       - Apply drop tolerance: skip if |L(i,k)| < DROP_TOL * L(k,k)
   - Store L in the lower triangle of the matrix (overwrite in-place)
   - Use the physical storage directly (no permutation arrays needed since Cholesky has no pivoting)
2. Handle the sparse access pattern:
   - For each column k, walk column k's entries to find rows i > k that have nonzeros
   - For the inner sum, walk row k and row i to find common columns j < k
   - Insert new fill-in entries via `sparse_insert()`
3. Write basic tests:
   - 2x2 SPD matrix: [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
   - 3x3 SPD tridiagonal: verify L values
   - Non-SPD matrix → `SPARSE_ERR_NOT_SPD`
   - Non-square → `SPARSE_ERR_SHAPE`

### Deliverables
- Working `sparse_cholesky_factor()` for small matrices
- ≥4 basic Cholesky tests

### Completion Criteria
- Cholesky produces correct L for known 2x2 and 3x3 SPD matrices
- Non-SPD and non-square matrices correctly rejected
- `make test` passes

---

## Day 3: Cholesky — Solve & Integration

**Theme:** Implement Cholesky solve and integrate with reordering

**Time estimate:** 8 hours

### Tasks
1. Implement `sparse_cholesky_solve()`:
   - Forward substitution: solve L*y = b (L is lower triangular, stored in lower triangle)
   - Backward substitution: solve L^T*x = y (using the same L entries, traversed by columns)
   - Handle the implicit upper triangle (L^T) by walking column lists
2. Implement `sparse_cholesky_factor_opts()`:
   - If reorder != NONE, compute AMD/RCM permutation, apply symmetric permutation, then factor
   - Store reorder_perm in matrix (same mechanism as `sparse_lu_factor_opts()`)
   - `sparse_cholesky_solve()` auto-permutes/unpermutes when reorder_perm is present
3. Write solve tests:
   - Factor and solve 3x3 SPD → verify x matches known solution
   - Factor and solve 5x5 SPD tridiagonal → verify residual < tol
   - Factor with AMD reordering → solve → verify same result as without reordering
   - Factor with RCM reordering → solve → verify same result
4. Run `make test` — all pass

### Deliverables
- Working `sparse_cholesky_solve()` with forward/backward substitution
- `sparse_cholesky_factor_opts()` with AMD/RCM integration
- ≥4 solve and reordering integration tests

### Completion Criteria
- Cholesky solve produces correct results on test cases
- Reordered solutions match non-reordered solutions
- `make test` clean

---

## Day 4: Cholesky — SuiteSparse Validation & Polish

**Theme:** Validate Cholesky on real SPD matrices, edge cases, benchmarks

**Time estimate:** 8 hours

### Tasks
1. Test Cholesky on SPD SuiteSparse matrices:
   - nos4 (100×100, symmetric structural) — factor and solve, check residual
   - bcsstk04 (132×132, symmetric stiffness) — factor and solve, check residual
   - Compare Cholesky fill-in vs LU fill-in on same matrices
   - Test with AMD and RCM reordering
2. Add edge-case tests:
   - 1×1 matrix → trivial Cholesky
   - Identity matrix → L = I
   - Diagonal SPD matrix → L = diag(sqrt(d_ii))
   - Nearly singular SPD (condition number ~1e12) → should factor but condest is large
   - Unsymmetric matrix → error (check symmetry before factoring? or detect via pivot failure?)
3. Add Cholesky benchmarks to `bench_main.c`:
   - Add `--cholesky` flag to benchmark Cholesky instead of LU
   - Report factor time, solve time, fill-in, residual
4. Update documentation:
   - Add Cholesky section to `docs/algorithm.md`
   - Update README with Cholesky in feature list and API overview

### Deliverables
- Cholesky validated on nos4 and bcsstk04 with correct residuals
- Edge-case tests for 1×1, identity, diagonal, nearly singular
- Cholesky benchmark support in bench_main
- Updated algorithm docs and README

### Completion Criteria
- Cholesky residual < 1e-10 on nos4, < 1e-4 on bcsstk04
- Cholesky fill-in ≤ LU fill-in on SPD matrices
- All tests pass, `make sanitize` clean

---

## Day 5: CSR/CSC Export — Design & CSR Implementation

**Theme:** Implement conversion from orthogonal linked-list to CSR format

**Time estimate:** 8 hours

### Tasks
1. Define CSR/CSC data structures in a new `include/sparse_csr.h`:
   - `typedef struct { idx_t rows, cols, nnz; idx_t *row_ptr; idx_t *col_idx; double *values; } SparseCsr;`
   - `typedef struct { idx_t rows, cols, nnz; idx_t *col_ptr; idx_t *row_idx; double *values; } SparseCsc;`
   - `void sparse_csr_free(SparseCsr *csr)` and `void sparse_csc_free(SparseCsc *csc)`
2. Implement `sparse_to_csr()`:
   - Walk each row's linked list, count nnz per row for row_ptr
   - Allocate col_idx and values arrays
   - Walk rows again, fill col_idx and values in sorted column order
   - Entries are already sorted by column in the row list — direct copy
3. Implement `sparse_from_csr()`:
   - Create a SparseMatrix, iterate CSR arrays, call sparse_insert for each entry
   - Validate CSR structure (row_ptr monotonic, col_idx in range)
4. Write tests:
   - Round-trip: create matrix → to_csr → from_csr → verify entries match
   - Empty matrix → CSR with nnz=0
   - Dense matrix → all entries present
   - Known 3×3 matrix → verify CSR arrays have correct values
   - NULL inputs → proper error codes

### Deliverables
- `SparseCsr` struct and `sparse_to_csr()` / `sparse_from_csr()`
- `sparse_csr_free()`
- ≥5 CSR tests including round-trip

### Completion Criteria
- Round-trip preserves all entries exactly
- CSR arrays are correctly formed (verified against known matrices)
- `make test` clean

---

## Day 6: CSR/CSC Export — CSC Implementation & Validation

**Theme:** Implement CSC conversion and validate both formats on real matrices

**Time estimate:** 7 hours

### Tasks
1. Implement `sparse_to_csc()`:
   - Walk each column's linked list (using col_headers), count nnz per column
   - Fill row_idx and values in sorted row order
2. Implement `sparse_from_csc()`:
   - Create SparseMatrix from CSC arrays
   - Validate CSC structure
3. Write CSC tests:
   - Round-trip: matrix → to_csc → from_csc → verify
   - Known matrix → verify CSC arrays
   - Transpose relationship: CSR of A = CSC of A^T (verify structure matches)
4. Test CSR/CSC on SuiteSparse matrices:
   - Load west0067.mtx → to_csr → from_csr → verify nnz and sample entries
   - Load nos4.mtx → to_csc → verify column structure
5. Add CSR/CSC source file to build system, update CMakeLists.txt

### Deliverables
- `SparseCsc` struct and `sparse_to_csc()` / `sparse_from_csc()`
- `sparse_csc_free()`
- ≥4 CSC tests plus SuiteSparse validation
- Build system updated

### Completion Criteria
- CSC round-trip preserves all entries
- CSR/CSC validated on real SuiteSparse matrices
- `make test` clean

---

## Day 7: Sparse Matrix-Matrix Multiply — Implementation

**Theme:** Implement C = A*B for sparse matrices

**Time estimate:** 9 hours

### Tasks
1. Add `sparse_matmul()` declaration to `include/sparse_matrix.h`:
   - `sparse_err_t sparse_matmul(const SparseMatrix *A, const SparseMatrix *B, SparseMatrix **C)`
   - A is m×k, B is k×n → C is m×n
   - Return `SPARSE_ERR_SHAPE` if inner dimensions mismatch
2. Implement `sparse_matmul()` in `src/sparse_matrix.c`:
   - Algorithm: for each row i of A, compute row i of C as a linear combination of rows of B
   - For each nonzero A(i,j), add A(i,j) * row_j(B) to the accumulator for row i of C
   - Use a dense accumulator array of length n (zeroed per row, sparse writeback)
   - This is the "row-wise SpMM" or "Gustavson's algorithm"
   - Drop entries below DROP_TOL threshold
3. Write tests:
   - I * A = A
   - A * I = A
   - Diagonal * A = scaled rows
   - A * Diagonal = scaled columns
   - Known 2×2: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
   - Dimension mismatch → `SPARSE_ERR_SHAPE`
   - NULL inputs → `SPARSE_ERR_NULL`
   - Verify C nnz is correct (no spurious zeros)

### Deliverables
- `sparse_matmul()` in public API
- Gustavson's row-wise algorithm with dense accumulator
- ≥7 SpMM tests

### Completion Criteria
- All SpMM tests pass with exact results
- I*A = A*I = A verified
- `make test` clean

---

## Day 8: SpMM — Validation & Performance

**Theme:** Validate SpMM on real matrices, benchmark, test edge cases

**Time estimate:** 7 hours

### Tasks
1. Test SpMM on SuiteSparse matrices:
   - Compute A * A^T (via sparse_transpose if available, or manual construction) for small matrices
   - Verify (A*B)*x = A*(B*x) using matvec (associativity check)
   - Benchmark SpMM time on west0067 and nos4
2. Edge-case tests:
   - A * empty = empty
   - Rectangular: (3×5) * (5×2) → (3×2)
   - Single-row * single-column → 1×1
   - Very sparse (1 nnz each) → verify single entry in product
3. Add SpMM to integration tests:
   - Form A = L * U from Cholesky L → verify A matches original (within tolerance)
   - This validates both SpMM and Cholesky correctness
4. Run `make sanitize` — all clean

### Deliverables
- SpMM validated on real matrices
- Edge-case tests
- SpMM + Cholesky integration test
- UBSan clean

### Completion Criteria
- Associativity check passes on SuiteSparse matrices
- All edge cases handled correctly
- `make test` and `make sanitize` clean

---

## Day 9: Thread Safety — Audit & Pool Allocator

**Theme:** Audit global state and make pool allocator thread-safe

**Time estimate:** 9 hours

### Tasks
1. Audit all global/static mutable state in the library:
   - `sparse_errno_` in `sparse_types.c` — already `_Thread_local`, safe
   - Pool allocator in `SparseMatrix.pool` — per-matrix, so concurrent access to different matrices is safe; concurrent mutation of the same matrix is not
   - Identify any other static/global state (there should be none)
2. Document the thread-safety contract:
   - **Safe:** concurrent read-only access to different matrices (including factored matrices for solve), concurrent solves on the same factored matrix (solve only reads)
   - **Safe:** concurrent factorization of different matrices (each has its own pool)
   - **Unsafe:** concurrent mutation of the same matrix (insert/remove/factor)
   - **Safe:** `sparse_errno()` is thread-local
3. Add thread-safety documentation to header files:
   - Add `@threadsafety` sections to key function doc comments
   - Add a thread-safety summary to README
4. Verify: can two threads independently create, populate, factor, and solve different matrices without interference?
   - Write a test that spawns 2-4 pthreads, each creating and solving an independent system
   - Compile with `-pthread`
5. Add `-pthread` flag support to Makefile (conditional, for thread tests)

### Deliverables
- Thread-safety audit complete (all global state identified)
- Thread-safety contract documented in headers and README
- Basic multi-threaded test (independent matrices)
- Build system supports pthread

### Completion Criteria
- Multi-threaded test passes without data races
- Thread-safety contract is clear and documented
- `make test` clean

---

## Day 10: Thread Safety — Concurrent Solve Stress Test

**Theme:** Stress-test concurrent read-only solve on shared factored matrix

**Time estimate:** 8 hours

### Tasks
1. Write concurrent solve stress test:
   - Create a matrix, factor it once
   - Spawn N threads (e.g., 4-8), each solving A*x = b_i with a different RHS
   - All threads share the same factored matrix (read-only)
   - Verify all solutions are correct
   - Run 1000+ iterations to flush out races
2. Run under Thread Sanitizer (TSan):
   - Add `tsan` target to Makefile: compile with `-fsanitize=thread`
   - Run the concurrent test under TSan
   - Fix any reported data races
3. Test concurrent Cholesky:
   - Same pattern: factor once, concurrent solves on shared factored matrix
4. Document any limitations found:
   - If pool_alloc is not called during solve (only during factor), concurrent solve should be safe
   - If solve allocates workspace, verify those allocations are stack/heap (not pool)
5. Update thread-safety docs based on findings

### Deliverables
- Concurrent solve stress test (LU and Cholesky)
- TSan target in Makefile
- TSan-clean run on concurrent tests
- Updated thread-safety documentation

### Completion Criteria
- Concurrent solve produces correct results for all threads
- TSan reports zero data races
- Thread-safety documentation updated with verified guarantees

---

## Day 11: Thread Safety — Edge Cases & Mutex Protection

**Theme:** Handle edge cases and add mutex for unsafe operations

**Time estimate:** 8 hours

### Tasks
1. Add optional mutex protection for matrix mutation:
   - Add a `pthread_mutex_t` (or platform abstraction) to SparseMatrix struct
   - Optionally lock during `sparse_insert()`, `sparse_remove()`, `sparse_lu_factor()`
   - Gate behind a compile-time flag (`-DSPARSE_MUTEX`) so it's opt-in (zero overhead by default)
   - Document: "Enable `-DSPARSE_MUTEX` if you need concurrent mutation of the same matrix (not recommended; prefer separate matrices per thread)"
2. Test the mutex path:
   - Concurrent inserts to the same matrix with mutex enabled → no crashes
   - Verify performance impact is minimal for single-threaded use (mutex not compiled in)
3. Review `sparse_lu_solve()` for hidden mutation:
   - Solve should be pure read of the factored matrix + stack/heap workspace
   - Verify no writes to the SparseMatrix struct during solve
   - If any found, fix to use local workspace instead
4. Add TSan run for non-concurrent tests (verify no false positives in single-threaded code)

### Deliverables
- Optional mutex protection (compile-time flag)
- Verification that solve is pure read-only on the matrix
- TSan clean on both concurrent and single-threaded tests

### Completion Criteria
- Mutex-protected concurrent mutation test passes
- `make test` clean (with and without `-DSPARSE_MUTEX`)
- TSan clean
- No performance regression in single-threaded mode

---

## Day 12: Integration Testing & Cross-Feature Validation

**Theme:** Test interactions between all Sprint 4 features

**Time estimate:** 8 hours

### Tasks
1. Cross-feature integration tests:
   - Cholesky factor → to_csr → from_csr → solve → correct result
   - SpMM: compute L * L^T from Cholesky factors → compare with original A
   - CSR export of Cholesky factor → verify triangular structure in CSR arrays
   - Concurrent threads: each does load_mm → Cholesky → solve → verify
2. Test Cholesky with condest:
   - Factor SPD matrix → compute condition estimate → verify reasonable
   - Note: condest uses LU internally, so test that Cholesky-factored matrices can be condest'd (may need separate LU factorization for condest, or adapt condest for Cholesky)
3. Run full regression:
   - `make test` — all suites pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run
4. Verify backward compatibility:
   - All Sprint 1-3 tests still pass unchanged
   - Existing API unchanged (no breaking changes)

### Deliverables
- ≥4 cross-feature integration tests
- Full regression pass
- Backward compatibility verified

### Completion Criteria
- All integration tests pass
- All 242+ existing tests still pass
- `make sanitize` clean

---

## Day 13: Documentation, Benchmarks & Hardening

**Theme:** Update all documentation, run comprehensive benchmarks

**Time estimate:** 8 hours

### Tasks
1. Update `docs/algorithm.md`:
   - Add Cholesky factorization section (algorithm, complexity, SPD requirement)
   - Add CSR/CSC section (data layout, conversion complexity)
   - Add SpMM section (Gustavson's algorithm, complexity)
   - Add thread-safety section
2. Update `README.md`:
   - Add Cholesky, SpMM, CSR/CSC, thread safety to feature list
   - Update API overview table (add `sparse_cholesky.h`, `sparse_csr.h`)
   - Update project structure
   - Update test counts
   - Add thread-safety section to Known Limitations (replace "Not thread-safe")
3. Run comprehensive benchmarks:
   - Cholesky vs LU on SPD matrices (nos4, bcsstk04): timing, fill-in, residual
   - Cholesky with AMD vs RCM vs no reordering
   - SpMM timing on small/medium matrices
   - CSR/CSC conversion timing
4. Run `make bench` and `make bench-suitesparse` — all clean

### Deliverables
- Updated algorithm documentation
- Updated README
- Benchmark results for Cholesky, SpMM, CSR/CSC
- Clean benchmark runs

### Completion Criteria
- Documentation covers all new features
- Benchmark data captured
- `make bench` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final validation, cleanup, and retrospective

**Time estimate:** 6 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run, no crashes
2. Code review pass:
   - All new public API functions have doc comments
   - All new error codes handled in `sparse_strerror()`
   - `const` correctness on all new functions
   - No compiler warnings with strict flags
3. Verify backward compatibility:
   - Existing code using LU factor/solve works unchanged
   - No breaking API changes
4. Write `docs/planning/EPIC_1/SPRINT_4/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, assertion count, API functions, etc.)
   - Cholesky vs LU comparison data
   - Thread-safety verification results
   - Items deferred to Sprint 5

### Deliverables
- All tests pass under all sanitizers
- Updated README with new API surface
- Sprint retrospective document
- Clean git history with meaningful commits

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 UBSan findings
- `make bench` completes without error
- README reflects current API
- Retrospective written with honest assessment

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | Cholesky — API design & SPD detection | 7 | API declarations, `sparse_is_symmetric()`, stubs |
| 2 | Cholesky — core factorization | 9 | `sparse_cholesky_factor()`, basic tests |
| 3 | Cholesky — solve & reordering | 8 | `sparse_cholesky_solve()`, opts integration, solve tests |
| 4 | Cholesky — SuiteSparse & polish | 8 | Real-matrix validation, edge cases, benchmarks, docs |
| 5 | CSR — design & CSR implementation | 8 | `sparse_to_csr()`, `sparse_from_csr()`, round-trip tests |
| 6 | CSC — implementation & validation | 7 | `sparse_to_csc()`, `sparse_from_csc()`, SuiteSparse validation |
| 7 | SpMM — implementation | 9 | `sparse_matmul()`, Gustavson's algorithm, ≥7 tests |
| 8 | SpMM — validation & performance | 7 | Real-matrix validation, edge cases, integration tests |
| 9 | Thread safety — audit & pool | 9 | Global state audit, thread-safety docs, basic pthread test |
| 10 | Thread safety — concurrent stress | 8 | Concurrent solve stress test, TSan target, TSan clean |
| 11 | Thread safety — edge cases & mutex | 8 | Optional mutex, solve read-only verification, TSan clean |
| 12 | Integration testing | 8 | Cross-feature tests, full regression |
| 13 | Documentation & benchmarks | 8 | Updated docs, comprehensive benchmarks |
| 14 | Sprint review & retrospective | 6 | Retrospective, final validation, cleanup |

**Total estimate:** 110 hours (avg ~7.9 hrs/day, max 9 hrs/day)
