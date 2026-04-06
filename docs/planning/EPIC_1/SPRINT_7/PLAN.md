# Sprint 7 Plan: QR Applications, Preconditioner Hardening & Eigenvalue Infrastructure

**Sprint Duration:** 14 days
**Goal:** Harden the ILUT preconditioner with true partial pivoting, improve QR scalability by eliminating the dense workspace, build QR-based applications (iterative refinement, economy QR), and lay the eigenvalue computation groundwork needed for SVD. Implement the QR algorithm for eigenvalues of symmetric tridiagonal matrices, which is the core iteration inside SVD.

**Starting Point:** A sparse linear algebra library with 73 public API functions, 467 tests across 22 test suites, 14 SuiteSparse reference matrices, LU/Cholesky/QR factorization with AMD/RCM reordering, CG/GMRES iterative solvers with ILU(0)/ILUT preconditioning (left and right), parallel SpMV via OpenMP, CI/CD with GitHub Actions, and code quality tooling (clang-format, clang-tidy, cppcheck).

**End State:** ILUT with true partial pivoting (`sparse_ilut_opts_t.pivot` option), `sparse_transpose()`, dense matrix utilities (Givens rotations, 2×2 eigensolvers), economy QR, sparse QR without dense workspace, `sparse_qr_refine()` for iterative refinement, symmetric tridiagonal QR eigenvalue solver, and Householder bidiagonalization for SVD.

---

## Day 1: ILUT Partial Pivoting — API & Core Algorithm

**Theme:** Add row partial pivoting as an alternative to diagonal modification in ILUT

**Time estimate:** 12 hours

### Tasks
1. Extend `sparse_ilut_opts_t` with a `pivot` flag (default: 0 for diagonal modification, 1 for partial pivoting):
   - When `pivot = 1`, ILUT selects the largest candidate in the pivot column and swaps rows
   - Track row permutation in `sparse_ilu_t.perm` (allocate when pivoting is enabled)
2. Implement row partial pivoting in `sparse_ilut_factor()`:
   - Before processing row `i`, scan the uneliminated column `i` entries for the largest magnitude
   - If the best candidate is in row `j > i`, swap the dense workspace rows and record `perm[i] = j`
   - Apply the row swap to L entries already computed for the swapped row
   - Fall back to diagonal modification when pivoting still produces a small pivot
3. Update `sparse_ilu_solve()` to apply row permutation when `ilu->perm != NULL`:
   - Forward substitution: permute the RHS vector before solving L*y = P*r
   - Back substitution: unchanged (U is already in permuted order)
4. Handle edge cases:
   - `pivot = 0` (default): existing diagonal modification behavior unchanged
   - `pivot = 1` on a matrix where pivoting doesn't help: fall back gracefully
   - NULL perm in solve path: no permutation applied (backward compatible)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_ilut_opts_t.pivot` flag for enabling partial pivoting
- Row partial pivoting in ILUT factorization with permutation tracking
- Updated ILUT solve with row permutation support
- All existing ILUT tests still pass (default behavior unchanged)

### Completion Criteria
- Code compiles with zero warnings
- Existing `make test` passes (no regression)
- `make format && make lint` clean

---

## Day 2: ILUT Partial Pivoting — Testing & Validation

**Theme:** Validate ILUT partial pivoting on synthetic and real matrices

**Time estimate:** 10 hours

### Tasks
1. Write ILUT pivoting tests:
   - Dense 3×3 with pivoting enabled: should match exact LU
   - west0067 with pivoting: compare convergence vs diagonal modification
   - Matrix requiring pivoting: construct a matrix where diagonal modification fails but pivoting succeeds
   - Verify `ilu->perm` is a valid permutation when pivoting is enabled
   - Verify `ilu->perm` is NULL when pivoting is disabled (default)
2. Test ILUT-pivoted preconditioner with GMRES:
   - west0067 with right-preconditioned GMRES: compare iteration count
   - steam1: pivoted vs non-pivoted ILUT iteration comparison
3. Edge-case tests:
   - pivoting on identity matrix: perm should be identity
   - pivoting on already-diagonally-dominant matrix: should not change behavior
   - NULL opts (defaults): pivoting disabled, backward compatible
4. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥8 ILUT pivoting tests
- ILUT pivoting validated on west0067
- Comparison data: pivoting vs diagonal modification

### Completion Criteria
- ILUT with pivoting produces valid L/U factors and permutation
- All existing and new tests pass
- `make format && make lint && make test` clean

---

## Day 3: Sparse Transpose

**Theme:** Implement `sparse_transpose()` for computing A^T

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_transpose()` in `src/sparse_matrix.c`:
   - Create a new matrix B with dimensions (cols_A × rows_A)
   - For each nonzero A(i,j) = v, insert B(j,i) = v
   - Walk row headers of A to collect all entries efficiently
   - Handle rectangular matrices (m ≠ n)
2. Add declaration to `include/sparse_matrix.h`:
   - `SparseMatrix *sparse_transpose(const SparseMatrix *A);`
   - Returns NULL on allocation failure or NULL input
3. Write tests in `tests/test_sparse_matrix.c` (or new test file):
   - Transpose of identity = identity
   - Transpose of transpose = original (A^T^T = A)
   - Rectangular matrix: 3×5 → 5×3
   - Symmetric matrix: A^T = A
   - Single row/column vectors
   - Empty matrix (0×0)
   - Verify A^T has same nnz as A
4. Test with SuiteSparse matrices:
   - nos4 (symmetric): A^T should match A
   - west0067 (unsymmetric): verify (A^T)^T = A
5. Update Makefile/CMakeLists.txt if new source file needed
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_transpose()` in public API
- ≥8 transpose tests including rectangular and SuiteSparse matrices
- Build system updated

### Completion Criteria
- `sparse_transpose()` correct on all test cases
- `make format && make lint && make test` clean

---

## Day 4: Dense Utilities — Core Operations

**Theme:** Implement dense matrix create/free/multiply for eigenvalue/SVD inner computations

**Time estimate:** 10 hours

### Tasks
1. Create `include/sparse_dense.h` and `src/sparse_dense.c`:
   - `typedef struct { double *data; idx_t rows, cols; } dense_matrix_t;`
   - `dense_matrix_t *dense_create(idx_t rows, idx_t cols);` — allocate zeroed
   - `void dense_free(dense_matrix_t *M);`
   - `sparse_err_t dense_gemm(const dense_matrix_t *A, const dense_matrix_t *B, dense_matrix_t *C);` — C = A*B
   - `sparse_err_t dense_gemv(const dense_matrix_t *A, const double *x, double *y);` — y = A*x
   - Column-major storage for compatibility with LAPACK conventions
2. Implement dense matrix-matrix multiply:
   - Triple loop: C(i,j) = sum_k A(i,k)*B(k,j)
   - Dimension checks: A.cols == B.rows, C.rows == A.rows, C.cols == B.cols
3. Implement dense matrix-vector multiply:
   - y(i) = sum_j A(i,j)*x(j)
4. Write basic tests:
   - Identity multiply: I*A = A
   - Dimension mismatch: proper error code
   - Small known products (2×3 * 3×2 = 2×2)
   - NULL inputs
5. Add to Makefile and CMakeLists.txt
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_dense.h` / `sparse_dense.c` with create/free/gemm/gemv
- ≥6 dense matrix operation tests
- Build system updated

### Completion Criteria
- Dense operations correct on all test cases
- `make format && make lint && make test` clean

---

## Day 5: Dense Utilities — Givens Rotations & 2×2 Eigensolvers

**Theme:** Implement Givens rotations and small eigenvalue solvers for tridiagonal QR

**Time estimate:** 10 hours

### Tasks
1. Implement Givens rotation computation:
   - `void givens_compute(double a, double b, double *c, double *s);` — compute c, s such that [c s; -s c]^T * [a; b] = [r; 0]
   - Handle special cases: b=0 (no rotation), a=0 (swap), both zero
   - Use `hypot()` for numerical stability
2. Implement Givens rotation application:
   - `void givens_apply_left(double c, double s, double *x, double *y, idx_t n);` — apply [c s; -s c]^T to rows x, y
   - `void givens_apply_right(double c, double s, double *x, double *y, idx_t n);` — apply [c s; -s c] to columns x, y
3. Implement 2×2 symmetric eigenvalue solver:
   - `void eigen2x2(double a, double b, double d, double *lambda1, double *lambda2);`
   - Solve eigenvalues of [[a, b], [b, d]] using the quadratic formula with careful numerics
   - Optional: compute eigenvectors
4. Write comprehensive tests:
   - Givens: rotation zeroes target entry, preserves vector norm
   - Givens: composition of rotations = product of rotation matrices
   - 2×2 eigensolver: known symmetric matrices, diagonal matrices, zero off-diagonal
   - Numerical edge cases: nearly equal eigenvalues, very small/large values
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Givens rotation compute/apply functions
- 2×2 symmetric eigenvalue solver
- ≥8 tests for Givens and eigensolver

### Completion Criteria
- Givens rotations numerically stable (norm-preserving)
- 2×2 eigensolver matches analytical solutions
- `make format && make lint && make test` clean

---

## Day 6: Economy QR — Implementation

**Theme:** Implement economy (thin) QR factorization for tall-skinny matrices

**Time estimate:** 10 hours

### Tasks
1. Add `economy` flag to `sparse_qr_opts_t`:
   - `int economy;` — when nonzero, compute only the first n Householder reflectors and n×n R
   - Default: 0 (full QR as before)
2. Modify `sparse_qr_factor_opts()` to support economy mode:
   - When economy=1 and m > n: only iterate k = min(m,n) = n steps (same as now)
   - Store only n Householder vectors (length m-k for k=0..n-1)
   - R is n×n upper triangular (not min(m,n)×n)
   - Column permutation is n-length (unchanged)
3. Modify `sparse_qr_apply_q()` for economy Q:
   - Q*x where Q is m×n (thin): apply n Householder reflectors to x (length m)
   - Q^T*x: apply reflectors in reverse order, result is length m but only first n entries meaningful
4. Modify `sparse_qr_solve()` for economy mode:
   - Q^T*b gives m-vector, but only first n entries used for R back-substitution
   - Residual = ||Q^T*b[n:]||
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Economy QR via `sparse_qr_opts_t.economy` flag
- Updated Q application and solve for economy mode
- Existing full-QR tests still pass

### Completion Criteria
- Economy flag accepted without regression
- `make format && make lint && make test` clean

---

## Day 7: Economy QR — Testing & Validation

**Theme:** Validate economy QR on tall-skinny matrices and compare with full QR

**Time estimate:** 10 hours

### Tasks
1. Write economy QR tests:
   - Tall-skinny 100×10: economy QR solve matches full QR solve
   - Overdetermined least-squares: economy and full produce same solution and residual
   - Rank-deficient tall matrix: economy handles rank < n correctly
   - Economy Q orthogonality: Q^T*Q = I_n (thin Q is m×n with orthonormal columns)
   - Economy R: verify R is n×n upper triangular
2. Memory comparison:
   - Print memory usage: economy vs full for 200×20 matrix
   - Verify economy uses fewer Householder vectors (n vs min(m,n))
3. Test on SuiteSparse matrices:
   - nos4 (square): economy should behave identically to full QR
   - Synthetic tall-skinny: verify least-squares residual
4. Edge cases:
   - Square matrix with economy=1: should behave same as economy=0
   - Wide matrix (m < n) with economy=1: n Householder reflectors = m (same as full)
   - 1×1 matrix
5. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥8 economy QR tests
- Economy vs full comparison data
- Edge-case coverage

### Completion Criteria
- Economy QR produces identical solutions to full QR
- Economy QR uses less memory for tall-skinny matrices
- `make format && make lint && make test` clean

---

## Day 8: Sparse Householder QR — Design & Column Operations

**Theme:** Design the sparse Householder QR and implement sparse column operations

**Time estimate:** 12 hours

### Tasks
1. Design the sparse Householder QR approach:
   - Instead of materializing a full m×n dense workspace, operate directly on columns of A
   - Maintain a dense working vector (length m) for each column being processed
   - After Householder application, write back nonzeros to a sparse R column
   - Q is still stored as Householder vectors (dense, length m-k each) — same as before
2. Implement sparse column extraction:
   - `static void extract_column(const SparseMatrix *A, idx_t col, double *dense);` — scatter column col into dense vector
   - Handle column traversal efficiently via col_headers (or row-scan if no col headers)
3. Implement sparse column write-back:
   - After Householder application, scan the dense result vector and insert nonzeros into R
   - Drop entries below a threshold (e.g., `1e-15`) to maintain sparsity
4. Implement Householder application to sparse column:
   - Given Householder vector v and scalar beta, apply (I - beta*v*v^T) to a dense column
   - This is the same dense operation as before, but only on one column at a time
5. Test column extraction and write-back on small matrices
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Sparse column extraction and write-back utilities
- Householder application to individual dense columns
- Design document / comments for sparse QR approach

### Completion Criteria
- Column operations correct on small test cases
- `make format && make lint && make test` clean

---

## Day 9: Sparse Householder QR — Factorization Loop

**Theme:** Implement the full sparse QR factorization without dense workspace

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_qr_factor_sparse()` (or update `sparse_qr_factor_opts()` with a sparse path):
   - For each step k = 0..min(m,n)-1:
     - Extract column k from the working matrix into dense vector
     - Compute Householder vector v_k and beta_k from entries k..m-1
     - Apply Householder to column k → produces R(k,k) and zeros below
     - Store R column k entries (row k..min(m,n)-1 above diagonal, value at diagonal)
     - For each remaining column j = k+1..n-1:
       - Extract column j into dense vector
       - Apply Householder (I - beta_k * v_k * v_k^T) to dense vector
       - Write back column j (now with entry at row k zeroed and fill above)
   - Column pivoting: track column norms, select largest-norm column at each step
2. Handle the working matrix:
   - Option A: Copy A into a mutable SparseMatrix, modify in-place
   - Option B: Maintain an array of dense column vectors (only the active submatrix)
   - Choose based on memory/performance trade-offs
3. Compare results with existing dense-workspace QR on small matrices
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Sparse QR factorization without O(m*n) dense workspace
- Produces same R, Q, permutation as dense-workspace version
- Memory usage proportional to nnz(A) + O(m) per column operation

### Completion Criteria
- Sparse QR matches dense QR on all existing test cases
- `make format && make lint && make test` clean

---

## Day 10: Sparse Householder QR — Testing & Validation

**Theme:** Validate sparse QR against dense QR, benchmark memory savings

**Time estimate:** 10 hours

### Tasks
1. Run all existing QR tests against the sparse Householder path:
   - Verify identical rank, R factor, Q orthogonality, solve results
   - Compare reconstruction error ||A - Q*R*P^T|| for both paths
2. Benchmark memory usage:
   - Compare peak allocation: dense path (m*n doubles) vs sparse path
   - Test on synthetic large matrices (e.g., 1000×100 sparse) where dense path would be expensive
3. Benchmark timing:
   - Compare factorization time on nos4, bcsstk04, west0067
   - Sparse path may be slower due to column extraction overhead — document trade-offs
4. Test sparse QR with column reordering (AMD):
   - Verify reordering still works correctly with sparse path
   - Compare R fill-in with and without reordering
5. Edge cases:
   - Very sparse matrix (few nonzeros per column)
   - Dense matrix (should still work, just slower)
   - Single-column and single-row matrices
6. Run `make format && make lint && make test` — all clean

### Deliverables
- All existing QR tests pass with sparse Householder path
- Memory and timing comparison data
- Sparse QR validated on SuiteSparse matrices

### Completion Criteria
- Sparse QR produces results matching dense QR within tolerance
- Memory savings demonstrated on large sparse matrices
- `make format && make lint && make test` clean

---

## Day 11: QR Iterative Refinement

**Theme:** Implement `sparse_qr_refine()` for improving least-squares solutions

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_qr_refine()` in `src/sparse_qr.c`:
   - Input: factored QR (sparse_qr_t), original A, b, and initial solution x
   - Algorithm:
     - Compute residual r = b - A*x
     - Solve correction: QR solve for delta_x using the existing factorization
     - Update: x = x + delta_x
     - Repeat for a fixed number of iterations (or until ||r|| stops decreasing)
   - Return: updated x, final residual norm, number of refinement iterations
2. Add declaration to `include/sparse_qr.h`:
   - `sparse_err_t sparse_qr_refine(const sparse_qr_t *qr, const SparseMatrix *A, const double *b, double *x, idx_t max_refine, double *residual);`
3. Write tests:
   - Well-conditioned system: refinement should not change solution much
   - Ill-conditioned system: refinement should improve residual by ≥1 order of magnitude
   - Overdetermined least-squares: refinement improves least-squares residual
   - Compare refined vs unrefined solution on nos4
   - max_refine = 0: should return immediately with current residual
4. Test on SuiteSparse matrices:
   - nos4: refinement on QR solution
   - Compare QR+refine residual with LU solve residual
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_refine()` in public API
- ≥6 refinement tests
- Refinement effectiveness data on SuiteSparse matrices

### Completion Criteria
- Refinement reduces residual on ill-conditioned systems
- No regression on well-conditioned systems
- `make format && make lint && make test` clean

---

## Day 12: Symmetric Tridiagonal QR Algorithm

**Theme:** Implement the implicit QR algorithm with Wilkinson shifts for tridiagonal eigenvalues

**Time estimate:** 12 hours

### Tasks
1. Implement tridiagonal data structure:
   - Store as two arrays: `diag[0..n-1]` (diagonal) and `subdiag[0..n-2]` (subdiagonal)
   - Symmetric: superdiagonal = subdiagonal
2. Implement the implicit QR step with Wilkinson shift:
   - Compute shift from trailing 2×2 submatrix using `eigen2x2()` from Day 5
   - Choose the eigenvalue of the 2×2 closer to the last diagonal entry
   - Compute initial Givens rotation to introduce a bulge
   - Chase the bulge down the tridiagonal using Givens rotations
   - Each rotation zeros the bulge entry and creates a new one in the next position
3. Implement deflation:
   - After each QR step, check subdiagonal entries for convergence to zero
   - When `|subdiag[k]| < tol * (|diag[k]| + |diag[k+1]|)`, deflate: split into two subproblems
   - Recursively solve smaller subproblems
4. Implement the top-level driver:
   - `sparse_err_t tridiag_qr_eigenvalues(const double *diag, const double *subdiag, idx_t n, double *eigenvalues, idx_t max_iter);`
   - Iterate QR steps until all subdiagonal entries converge
   - Return eigenvalues in `diag` (modified in-place) sorted in ascending order
5. Write tests:
   - Diagonal matrix: eigenvalues = diagonal entries
   - 2×2 tridiagonal: compare with `eigen2x2()`
   - Known tridiagonal (e.g., -1, 2, -1 tridiag of size n): eigenvalues are 2 - 2*cos(k*pi/(n+1))
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Implicit QR algorithm with Wilkinson shifts for symmetric tridiagonal matrices
- Deflation for converged subdiagonal entries
- `tridiag_qr_eigenvalues()` driver function
- ≥6 eigenvalue tests

### Completion Criteria
- Eigenvalues correct to machine precision on test cases
- Convergence within reasonable iteration count
- `make format && make lint && make test` clean

---

## Day 13: Bidiagonal Reduction & Tridiagonal QR Hardening

**Theme:** Implement Householder bidiagonalization and harden the tridiagonal QR solver

**Time estimate:** 12 hours

### Tasks
1. Implement Householder bidiagonalization:
   - Input: m×n matrix A (m ≥ n assumed; transpose first if m < n)
   - Algorithm: alternating left and right Householder reflections
     - Left reflection on column k: zero entries below diagonal in column k
     - Right reflection on row k: zero entries right of superdiagonal in row k
   - Output: bidiagonal B (diagonal + superdiagonal arrays), left Householder vectors U, right Householder vectors V
   - Store U and V implicitly as Householder sequences (same format as QR)
2. Add API:
   - `typedef struct { double *diag; double *superdiag; double **u_vecs; double *u_betas; double **v_vecs; double *v_betas; idx_t m, n; } sparse_bidiag_t;`
   - `sparse_err_t sparse_bidiag_factor(const SparseMatrix *A, sparse_bidiag_t *bidiag);`
   - `void sparse_bidiag_free(sparse_bidiag_t *bidiag);`
3. Harden the tridiagonal QR solver:
   - Test with larger matrices (n=50, n=100)
   - Test with clustered eigenvalues
   - Test with graded matrices (eigenvalues spanning many orders of magnitude)
   - Verify convergence rate: should be cubic for Wilkinson shifts
4. Write bidiagonal reduction tests:
   - Verify B is bidiagonal: only diagonal and superdiagonal nonzero
   - Verify ||A - U*B*V^T|| < tol (reconstruction)
   - Rectangular matrices: 10×5, 5×10
   - Square matrix: reconstruction and orthogonality of U, V
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_bidiag_factor()` and `sparse_bidiag_free()` in API
- Bidiagonal reduction with implicit U, V storage
- Hardened tridiagonal QR with larger test cases
- ≥8 tests (bidiagonal + tridiagonal hardening)

### Completion Criteria
- Bidiagonal reconstruction error < 1e-10
- Tridiagonal QR converges on all test cases including clustered/graded
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final validation, cross-feature integration, cleanup, and retrospective

**Time estimate:** 8 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run, no crashes
   - `make format && make lint` — all clean
2. Cross-feature integration tests:
   - ILUT pivoting + right-preconditioned GMRES on west0067
   - Economy QR solve matches full QR solve on nos4
   - Sparse Householder QR matches dense workspace QR on all SuiteSparse matrices
   - QR refine improves residual on ill-conditioned system
   - Bidiagonal reduction → tridiagonal eigenvalues → verify singular values match QR rank
3. Code review pass:
   - All new public API functions have doc comments
   - `const` correctness on all new functions
   - No compiler warnings with strict flags
4. Verify backward compatibility:
   - Existing code using LU/Cholesky/CG/GMRES/ILU/QR works unchanged
   - No breaking API changes
5. Update documentation:
   - Update `docs/algorithm.md` with new algorithms (tridiagonal QR, bidiagonalization)
   - Update `README.md` with new API functions and test counts
6. Write `docs/planning/EPIC_1/SPRINT_7/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, API functions, etc.)
   - Items deferred to Sprint 8

### Deliverables
- All tests pass under all sanitizers
- Cross-feature integration tests
- Updated documentation
- Sprint retrospective document

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 UBSan findings
- `make bench` completes without error
- `make format && make lint` clean
- README reflects current API
- Retrospective written with honest assessment

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | ILUT partial pivoting — API & core | 12 | Row pivoting in ILUT, perm tracking, updated solve |
| 2 | ILUT partial pivoting — testing | 10 | ≥8 pivoting tests, west0067 validation, comparison data |
| 3 | Sparse transpose | 12 | `sparse_transpose()`, ≥8 tests, SuiteSparse validation |
| 4 | Dense utilities — core | 10 | `sparse_dense.h`, create/free/gemm/gemv, ≥6 tests |
| 5 | Dense utilities — Givens & eigensolvers | 10 | Givens rotations, 2×2 eigensolver, ≥8 tests |
| 6 | Economy QR — implementation | 10 | `economy` flag, thin Q/R, updated solve |
| 7 | Economy QR — testing | 10 | ≥8 economy tests, memory comparison, edge cases |
| 8 | Sparse Householder QR — design & column ops | 12 | Column extract/writeback, sparse Householder design |
| 9 | Sparse Householder QR — factorization loop | 12 | Full sparse QR factorization without dense workspace |
| 10 | Sparse Householder QR — testing | 10 | All QR tests on sparse path, memory/timing benchmarks |
| 11 | QR iterative refinement | 12 | `sparse_qr_refine()`, ≥6 tests, SuiteSparse validation |
| 12 | Symmetric tridiagonal QR algorithm | 12 | Implicit QR with Wilkinson shifts, deflation, ≥6 tests |
| 13 | Bidiagonal reduction & tridiag hardening | 12 | `sparse_bidiag_factor()`, hardened tridiag QR, ≥8 tests |
| 14 | Sprint review & retrospective | 8 | Integration tests, docs, retrospective, final regression |

**Total estimate:** 152 hours (avg ~10.9 hrs/day, max 12 hrs/day)
